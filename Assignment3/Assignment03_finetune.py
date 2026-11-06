"""Assignment 3, part 3: LoRA fine-tuning with hyperparameter tuning for the pretrained LLMs.

Assignment03_KshaunishSoni.py evaluates each pretrained model few-shot, with no
training. This file fine-tunes each of them on MBPP and tunes the fine-tuning
hyperparameters (learning rate, batch size, epochs), then scores the tuned model on
the same test problems with the same metrics (corpus / sentence BLEU, syntax validity).

Why LoRA: full fine-tuning a 3B model with AdamW needs ~50 GB (weights, gradients and
two optimizer states), far beyond a 16 GB Mac. LoRA (Hu et al., 2021) freezes the
pretrained weights and trains small low-rank adapters on every linear layer
(LORA_CONFIG: rank 16, ~0.5-1% of the parameters), so a 3B model trains in ~7 GB.
Gradient checkpointing keeps activation memory low; weights are bf16 on GPU/MPS.

Training data: MBPP train (374) + prompt (10). Each example is the zero-shot MBPP prompt
(task + tests + [BEGIN]) followed by the reference code and [DONE]; the loss is on the
code and [DONE] only. Fine-tuned models are evaluated zero-shot (they learned the
format, so the few-shot examples are no longer needed); everything else (greedy
decoding, stop at [DONE], scoring) is identical to the few-shot evaluation.

Hyperparameter tuning (TUNING_GRID): one training run per learning rate x batch size
(2 x 2 = 4 runs per model). Batch size is the effective batch size (micro-batches
with gradient accumulation, so memory does not grow with it). The learning rate is
constant after a short warmup, so the adapters after epoch k of a run are exactly what
a k-epoch run would produce: validation loss is measured after every epoch, and each
run therefore also scores every value in the epochs grid (1, 2, 3) at no extra cost.
The (learning rate, batch size, epochs) with the lowest validation loss (MBPP
validation split, 90 problems) is selected; the test split is never used for selection.

Caching: the selected adapters are saved to results/adapters/<model>/ with the tuning
table, so a rerun skips training (use --overwrite to retrain). Generations are cached
like the few-shot ones.

Outputs (results/):
    generations_<model>-LoRA_<split><n>.jsonl  per-problem predictions and scores
    finetuned_metrics_summary.csv              one row per model, incl. selected
                                               hyperparameters and timings
    finetuned_tuning.csv                       one row per (model, lr, batch size, epochs)
    finetuned_tuning_heatmap.png               validation loss over lr x batch size
    adapters/<model>/                          the selected LoRA adapters

Runtime on a 16 GB Apple Silicon Mac: roughly 30-40 min of tuning per 1.5B model and
60-80 min per 3B model, plus zero-shot test generation. Use --models to run a subset.

    python Assignment03_finetune.py                       # every model in MODELS
    python Assignment03_finetune.py --quick               # smoke test
    python Assignment03_finetune.py --models Qwen/Qwen2.5-Coder-1.5B
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Shared data, generation and scoring, so results match the few-shot experiments
from Assignment03_KshaunishSoni import (CONFIG, MODELS, RESULTS_DIR, clean_code,
                                        extract_code, format_problem, free_device_memory,
                                        generate, load_mbpp, model_family, pick_device,
                                        score_generations, short_name, summarise)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
TUNING_GRID = {
    "learning_rate": [1e-4, 3e-4],
    "batch_size": [8, 16],        # effective batch size (gradient accumulation)
    "epochs": [1, 2, 3],          # must be 1..N: scored after every epoch of one run
}
QUICK_GRID = {"learning_rate": [3e-4], "batch_size": [8], "epochs": [1]}
QUICK_LIMITS = {"train": 32, "validation": 16, "test": 20}

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": "all-linear",   # every attention and MLP projection
}

FT_CONFIG = {
    "micro_batch": {"cuda": 8, "mps": 2, "cpu": 2},   # sequences per forward pass
    "max_len": 512,          # prompt + code tokens (MBPP examples are ~150-350)
    "warmup_ratio": 0.1,     # of the first epoch's optimizer steps
    "weight_decay": 0.0,
    "eval_batch_size": {"cuda": 8, "mps": 4, "cpu": 2},
    "seed": 42,
}

ADAPTER_DIR = os.path.join(RESULTS_DIR, "adapters")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lora_name(model_id):
    return f"{model_id} +LoRA"


def train_dtype(device):
    """bf16 where the hardware has it (Apple Silicon, Ampere+ GPUs such as A100/L4).
    Older GPUs (Kaggle/Colab T4, P100) only emulate bf16, which is slow: use fp16 there
    (the frozen weights are fp16; PEFT keeps the trainable LoRA weights in fp32)."""
    if device == "mps":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.float32


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def encode_examples(tokenizer, split, max_len):
    """[(input ids, labels)]: prompt tokens are masked (-100), code + [DONE] are learned."""
    pairs = []
    for ex in split:
        prompt = tokenizer(format_problem(ex)).input_ids     # adds BOS where the model uses one
        target = tokenizer(f"{clean_code(ex['code'])}\n{CONFIG['stop_string']}",
                           add_special_tokens=False).input_ids
        if tokenizer.eos_token_id is not None:
            target.append(tokenizer.eos_token_id)
        ids = (prompt + target)[:max_len]
        labels = ([-100] * len(prompt) + target)[:max_len]
        pairs.append((ids, labels))
    return pairs


def collate(pairs, pad_id, device):
    width = max(len(ids) for ids, _ in pairs)
    ids = torch.full((len(pairs), width), pad_id)
    labels = torch.full((len(pairs), width), -100)
    mask = torch.zeros((len(pairs), width), dtype=torch.long)
    for i, (x, y) in enumerate(pairs):        # right padding for training
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    return ids.to(device), labels.to(device), mask.to(device)


def token_loss(model, ids, labels, mask):
    """Summed next-token cross-entropy over the labelled tokens, and their count."""
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    target = labels[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                           ignore_index=-100, reduction="sum")
    return loss, (target != -100).sum()


@torch.no_grad()
def validation_loss(model, pairs, pad_id, micro, device):
    model.eval()
    total, count = 0.0, 0
    for b in range(0, len(pairs), micro):
        loss, n = token_loss(model, *collate(pairs[b:b + micro], pad_id, device))
        total, count = total + loss.item(), count + n.item()
    return total / max(count, 1)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def lora_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}


def train_run(model, train_pairs, val_pairs, pad_id, lr, batch_size, max_epochs, device):
    """AdamW on the LoRA parameters; linear warmup, then constant learning rate.
    Returns the per-epoch history and the adapter weights after each epoch's eval
    (only the best epoch's are kept)."""
    micro = min(FT_CONFIG["micro_batch"][device], batch_size)
    accum = math.ceil(batch_size / micro)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=FT_CONFIG["weight_decay"])
    steps_per_epoch = math.ceil(len(train_pairs) / batch_size)
    warmup = max(1, int(FT_CONFIG["warmup_ratio"] * steps_per_epoch))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup))

    history, best = [], None
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        model.train()
        order = list(range(len(train_pairs)))
        random.shuffle(order)
        train_loss, train_tokens = 0.0, 0
        for step, b in enumerate(range(0, len(order), batch_size)):
            batch = [train_pairs[i] for i in order[b:b + batch_size]]
            n_tokens = sum(sum(l != -100 for l in y[1:]) for _, y in batch)
            optimizer.zero_grad(set_to_none=True)
            for m in range(0, len(batch), micro):
                loss, n = token_loss(model, *collate(batch[m:m + micro], pad_id, device))
                (loss / max(n_tokens, 1)).backward()     # token-weighted mean over the batch
                train_loss, train_tokens = train_loss + loss.item(), train_tokens + n.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            if step % 10 == 0:
                print(f"    epoch {epoch} step {step + 1}/{steps_per_epoch}  "
                      f"loss {train_loss / max(train_tokens, 1):.3f}  ({time.time() - start:.0f}s)",
                      flush=True)
        val = validation_loss(model, val_pairs, pad_id, FT_CONFIG["eval_batch_size"][device], device)
        history.append({"epoch": epoch, "train_loss": train_loss / max(train_tokens, 1),
                        "val_loss": val, "seconds": time.time() - start})
        print(f"  epoch {epoch}/{max_epochs}  train {history[-1]['train_loss']:.3f}  "
              f"val {val:.3f}  ({time.time() - start:.0f}s)", flush=True)
        if best is None or val < best[0]:
            best = (val, epoch, lora_state(model))
    del optimizer
    return history, best


def load_for_training(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # One device: a <=3B model fits a 16 GB GPU, and splitting it across Kaggle's two
    # T4s would run them one after the other (no speed-up, extra transfers).
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=train_dtype(device))
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()     # so checkpointed layers get gradients with frozen embeddings
    return tokenizer, model


def tune_model(model_id, data, grid, device):
    """Grid search for one model. Returns (tokenizer, base model, best adapter weights,
    selected config, tuning table, tuning seconds)."""
    from peft import LoraConfig, get_peft_model

    tokenizer, base = load_for_training(model_id, device)
    lora = LoraConfig(task_type="CAUSAL_LM", **LORA_CONFIG)
    train_pairs = encode_examples(tokenizer, data["train"], FT_CONFIG["max_len"])
    val_pairs = encode_examples(tokenizer, data["validation"], FT_CONFIG["max_len"])
    max_epochs = max(grid["epochs"])

    rows, best, start = [], None, time.time()
    runs = [(lr, bs) for lr in grid["learning_rate"] for bs in grid["batch_size"]]
    for i, (lr, bs) in enumerate(runs, 1):
        print(f"\n  [{short_name(model_id)}] run {i}/{len(runs)}: lr {lr:g}, batch {bs}, "
              f"up to {max_epochs} epochs", flush=True)
        set_seed(FT_CONFIG["seed"])
        model = get_peft_model(base, lora)     # fresh adapters on the frozen base
        if i == 1:
            model.print_trainable_parameters()
        run_start = time.time()
        history, (val, epoch, state) = train_run(model, train_pairs, val_pairs,
                                                 tokenizer.pad_token_id, lr, bs, max_epochs, device)
        for h in history:
            if h["epoch"] in grid["epochs"]:
                rows.append({"model": model_id, "learning_rate": lr, "batch_size": bs,
                             "epochs": h["epoch"], "train_loss": round(h["train_loss"], 4),
                             "val_loss": round(h["val_loss"], 4),
                             "run_time_s": round(time.time() - run_start, 1)})
        # epochs grid is 1..max, so the run's best epoch is always a grid value
        if best is None or val < best[0]:
            best = (val, {"learning_rate": lr, "batch_size": bs, "epochs": epoch}, state)
        base = model.unload()                  # remove this run's adapters
        del model
        free_device_memory()

    val, cfg, state = best
    table = pd.DataFrame(rows)
    table["best_epoch"] = table["epochs"]      # for the shared heatmap: epochs of each cell
    table["best_val_loss"] = table["val_loss"]
    table["selected"] = ((table["learning_rate"] == cfg["learning_rate"])
                         & (table["batch_size"] == cfg["batch_size"])
                         & (table["epochs"] == cfg["epochs"]))
    print(f"\n  [{short_name(model_id)}] selected lr {cfg['learning_rate']:g}, "
          f"batch {cfg['batch_size']}, epochs {cfg['epochs']} (val loss {val:.3f})")
    return tokenizer, base, state, cfg, table, time.time() - start


# ----------------------------------------------------------------------------
# Per-model pipeline: tune (or load cached adapters), then evaluate zero-shot
# ----------------------------------------------------------------------------
def generations_path(model_id, n_problems, split):
    return os.path.join(RESULTS_DIR,
                        f"generations_{short_name(model_id)}-LoRA_{split}{n_problems}.jsonl")


def run_model(model_id, data, eval_split, grid, args, device):
    from peft import LoraConfig, PeftModel, get_peft_model

    gen_path = generations_path(model_id, len(eval_split), args.split)
    info_path = gen_path.replace(".jsonl", ".info.json")
    adapter_dir = os.path.join(ADAPTER_DIR, short_name(model_id) + ("_quick" if args.quick else ""))
    meta_path = os.path.join(adapter_dir, "tuning.json")

    if os.path.exists(gen_path) and os.path.exists(info_path) and not args.overwrite:
        print(f"[{model_id} +LoRA] using cached generations: {gen_path}")
        with open(info_path) as f:
            info = json.load(f)
        tuning = (pd.read_csv(os.path.join(adapter_dir, "tuning.csv"))
                  if os.path.exists(os.path.join(adapter_dir, "tuning.csv")) else None)
        return score_generations(pd.read_json(gen_path, lines=True)), info, tuning

    if os.path.exists(meta_path) and not args.overwrite:
        print(f"[{model_id} +LoRA] loading tuned adapters from {adapter_dir}")
        with open(meta_path) as f:
            meta = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        kwargs = {"torch_dtype": train_dtype(device)}
        base = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device)
        model = PeftModel.from_pretrained(base, adapter_dir)
        cfg, tune_s = meta["selected"], meta["tuning_time_s"]
        tuning = pd.read_csv(os.path.join(adapter_dir, "tuning.csv"))
    else:
        print(f"[{model_id} +LoRA] tuning on {device}: grid {grid}")
        tokenizer, base, state, cfg, tuning, tune_s = tune_model(model_id, data, grid, device)
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **LORA_CONFIG))
        missing = model.load_state_dict(state, strict=False).unexpected_keys
        assert not missing, f"adapter keys did not match: {missing[:3]}"
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tuning.to_csv(os.path.join(adapter_dir, "tuning.csv"), index=False)
        with open(meta_path, "w") as f:
            json.dump({"selected": cfg, "tuning_time_s": round(tune_s, 1),
                       "lora": LORA_CONFIG, "grid": grid}, f, indent=2)

    n_trainable = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    model = model.merge_and_unload()           # plain model again: fast generation
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    n_params = model.num_parameters()

    tokenizer.padding_side = "left"
    prompts = [format_problem(ex) for ex in eval_split]       # zero-shot
    start = time.time()
    batch_size = args.batch_size or CONFIG["batch_size"][device]
    raw = generate(tokenizer, model, prompts, batch_size, CONFIG["max_new_tokens"])
    info = {"parameters_m": round(n_params / 1e6, 1),
            "lora_parameters_m": round(n_trainable / 1e6, 2),
            **cfg, "best_val_loss": float(tuning.loc[tuning["selected"], "val_loss"].iloc[0]),
            "tuning_time_s": round(tune_s, 1), "inference_time_s": round(time.time() - start, 1)}
    with open(info_path, "w") as f:
        json.dump(info, f)
    del model, base
    free_device_memory()

    df = score_generations(pd.DataFrame([{
        "task_id": ex["task_id"], "text": ex["text"], "reference": clean_code(ex["code"]),
        "prediction": extract_code(gen), "raw_generation": gen,
    } for ex, gen in zip(eval_split, raw)]))
    df.to_json(gen_path, orient="records", lines=True)
    print(f"[{model_id} +LoRA] saved {len(df)} generations -> {gen_path}")
    return df, info, tuning


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", nargs="+", default=MODELS, help="pretrained models to fine-tune")
    p.add_argument("--quick", action="store_true",
                   help=f"smoke test: {QUICK_LIMITS} examples, grid {QUICK_GRID}")
    p.add_argument("--split", default="test", choices=["test", "validation"])
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N problems")
    p.add_argument("--batch-size", type=int, default=None, help="generation batch size")
    p.add_argument("--overwrite", action="store_true",
                   help="retune and regenerate even if cached adapters / generations exist")
    return p.parse_known_args(argv)[0]


def main(argv=None):
    from Assignment03_transformer_models import plot_tuning_heatmap

    args = parse_args(argv)
    device = pick_device()
    grid = QUICK_GRID if args.quick else TUNING_GRID
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ds = load_mbpp()
    data = {"train": [*ds["train"], *ds["prompt"]], "validation": list(ds["validation"])}
    eval_split = ds[args.split]
    limit = args.limit or (QUICK_LIMITS["test"] if args.quick else None)
    if args.quick:
        data = {k: v[:QUICK_LIMITS[k]] for k, v in data.items()}
    if limit:
        eval_split = eval_split.select(range(min(limit, len(eval_split))))

    rows, tunings = [], []
    for model_id in args.models:
        df, info, tuning = run_model(model_id, data, eval_split, grid, args, device)
        if tuning is not None:
            tunings.append(tuning)
        rows.append({
            **summarise(lora_name(model_id), df),
            "family": model_family(model_id).replace("pretrained", "pretrained + LoRA"),
            "parameters_m": info.get("parameters_m"),
            "lora_parameters_m": info.get("lora_parameters_m"),
            "learning_rate": info.get("learning_rate"),
            "batch_size": info.get("batch_size"),
            "epochs": info.get("epochs"),
            "best_val_loss": info.get("best_val_loss"),
            "training_time_s": info.get("tuning_time_s"),
            "tuning_time_s": info.get("tuning_time_s"),
            "inference_time_s": info.get("inference_time_s"),
            "total_time_s": round(info.get("tuning_time_s", 0) + info.get("inference_time_s", 0), 1),
        })

    summary = pd.DataFrame(rows).sort_values("corpus_bleu", ascending=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "finetuned_metrics_summary.csv"), index=False)
    if tunings:
        tuning = pd.concat(tunings, ignore_index=True)
        tuning.to_csv(os.path.join(RESULTS_DIR, "finetuned_tuning.csv"), index=False)
        plot_tuning_heatmap(tuning, "model",
                            os.path.join(RESULTS_DIR, "finetuned_tuning_heatmap.png"),
                            "LoRA fine-tuning: best validation loss by learning rate and batch "
                            "size (ep = best epochs; outlined = selected)")
    cols = ["model", "learning_rate", "batch_size", "epochs", "corpus_bleu",
            "syntax_validity_pct", "tuning_time_s", "inference_time_s"]
    print("\nLoRA fine-tuned models:\n", summary[cols].to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
