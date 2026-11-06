"""Assignment 3: code generation on MBPP with pretrained code LLMs, scored with BLEU.

Data: MBPP ("Mostly Basic Python Problems", google-research-datasets/mbpp, config
"full") from the Hugging Face Hub. Each problem has a natural-language task
(`text`), a reference solution (`code`) and three assert tests (`test_list`).
Splits: prompt (10), train (374), validation (90), test (500).

Pretrained models (MODELS), evaluated with no training. Small (1.3-3B) models so
the whole workflow runs on a 16 GB Apple Silicon Mac:
    base        bigcode/starcoder2-3b                  (3B)
                deepseek-ai/deepseek-coder-1.3b-base   (1.3B)
                Qwen/Qwen2.5-Coder-1.5B                (1.5B)
                Qwen/Qwen2.5-Coder-3B                  (3B)
    instruct    deepseek-ai/deepseek-coder-1.3b-instruct
                Qwen/Qwen2.5-Coder-1.5B-Instruct
                Qwen/Qwen2.5-Coder-3B-Instruct         (3B)
The four Qwen2.5-Coder models form a 2 x 2 design (1.5B vs 3B, base vs instruct), so
the effects of size and of instruction tuning can be read separately.
(CodeLlama is not included: its smallest version is 7B. Phi-3-mini (3.8B) stalled on a
16 GB Mac. LARGE_MODELS lists these larger models; they need a CUDA GPU: pass them with
--models on Colab/Kaggle.)
From-scratch models: the three small transformers in Assignment03_transformer_models.py
(encoder-decoder, decoder-only, autoregressive decoder-only) are trained on MBPP and
scored the same way, after a grid search over learning rate, batch size and epochs.
Fine-tuned models: every pretrained model is also LoRA fine-tuned on MBPP with the same
kind of grid search (Assignment03_finetune.py), and its tuned version scored zero-shot.
One call to this file runs all of them: 7 few-shot, 7 fine-tuned, 3 from scratch.

Hyperparameter tuning (learning rate x batch size x epochs) is selected on the MBPP
validation split (90 problems) by validation loss; the test split is only scored.
Few-shot evaluation has no trainable hyperparameters (its batch size only changes
speed), so for the pretrained models tuning means tuning their fine-tuning.

Prompting: the standard MBPP few-shot format from the original paper
(Austin et al., 2021). The task text and its tests are shown, and the model writes
code between [BEGIN] and [DONE]. N_SHOT worked examples from the "prompt" split
come first so the base (non-instruct) models learn the format. Instruct models get the
exact same plain-text prompt (no chat template), so every pretrained model sees
identical input. Decoding is greedy and stops at "[DONE]".

Evaluation: BLEU (Bilingual Evaluation Understudy, Papineni et al., 2002) between
each generated solution and the MBPP reference solution, via sacrebleu:
    corpus BLEU  - one score per model over the whole split (the headline number)
    sentence BLEU - one score per problem (distribution / per-problem analysis)
BLEU measures n-gram overlap with the reference, not whether the code runs, so a
correct solution written differently from the reference can still score low.

Syntax validity score: the share of generated solutions that are valid Python.
Each solution is compiled with compile(code, ..., "exec"), which parses it and
builds its bytecode without running it, so it catches syntax errors (unbalanced
brackets, bad indentation, truncated code, `return` outside a function) safely.
An empty generation counts as invalid. This complements BLEU: it says whether the
output is usable code at all, independent of how closely it matches the reference.

Outputs (results/):
    generations_<model>_<split><n>.jsonl  task, reference, prediction, sentence BLEU,
                               syntax validity per problem
    pretrained_metrics_summary.csv  all BLEU / syntax metrics + timing, pretrained models
    finetuned_metrics_summary.csv   the same for the LoRA fine-tuned models
    scratch_metrics_summary.csv     the same for the from-scratch models
    finetuned_tuning.csv / scratch_tuning.csv   every hyperparameter combination tried
    *_tuning_heatmap.png            validation loss over learning rate x batch size
    model_comparison_summary.csv    every model: family, parameters, selected learning
                                    rate / batch size / epochs, corpus BLEU, mean
                                    sentence BLEU, syntax validity, training, tuning,
                                    inference and total time (seconds)
    model_comparison.png            corpus BLEU and syntax validity, every model
    sentence_bleu_dist.png          distribution of per-problem BLEU, pretrained models
Timing: pretrained models have no training (training_time_s = 0); their total time is
model loading + generation. Tuned models: total = the whole grid search + generation
(training_time_s is the selected run alone for the from-scratch models).
Generations are cached: a pretrained model whose generations file already exists is
not re-run (use --overwrite to regenerate); its timing is read from the
<generations>.info.json file saved with it. When one family is skipped (--only), its
last saved summary CSV is used for the comparison, so the pretrained models can run on
a GPU and the from-scratch models on a Mac.

Hardware: models run in fp16 on a CUDA GPU or Apple Silicon (MPS). Without CUDA, a
model whose fp16 weights exceed MAC_MAX_MODEL_GIB is skipped (a 7B model needs ~14 GB
and would swap on a 16 GB Mac) unless --force-pretrained is given. On CUDA, a model
that does not fit the GPU(s) is loaded in 4-bit automatically, and a batch that runs
out of memory is retried with half the batch size.

    pip install -r requirements.txt     # or ./run_workflow.sh, which also makes a venv

    python Assignment03_KshaunishSoni.py                  # everything, full test split
    python Assignment03_KshaunishSoni.py --quick          # smoke test (20 problems, tiny grids)
    python Assignment03_KshaunishSoni.py --only scratch   # just the from-scratch models
    python Assignment03_KshaunishSoni.py --only finetuned # just LoRA fine-tuning + tuning
    python Assignment03_KshaunishSoni.py --no-tune        # no grid search (defaults only;
                                                          # skips LoRA fine-tuning)
    python Assignment03_KshaunishSoni.py --only pretrained --models Qwen/Qwen2.5-Coder-3B-Instruct
"""

import argparse
import gc
import json
import os

# Let PyTorch grow memory segments instead of fragmenting (fewer CUDA OOMs during generation)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import time

import pandas as pd
import sacrebleu
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
if "ipykernel" not in sys.modules:   # headless when run as a script; inline in Colab/Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MODELS = [
    "bigcode/starcoder2-3b",
    "deepseek-ai/deepseek-coder-1.3b-base",
    "Qwen/Qwen2.5-Coder-1.5B",
    "Qwen/Qwen2.5-Coder-3B",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
]

# 7B versions (CUDA GPU only), e.g. --models bigcode/starcoder2-7b meta-llama/CodeLlama-7b-hf
LARGE_MODELS = [
    "bigcode/starcoder2-7b",
    "deepseek-ai/deepseek-coder-6.7b-base",
    "meta-llama/CodeLlama-7b-hf",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "meta-llama/CodeLlama-7b-Instruct-hf",
    "microsoft/Phi-3-mini-4k-instruct",    # 3.8B: fits the 8 GiB limit but stalls on a 16 GB Mac
]

MAC_MAX_MODEL_GIB = 8.0   # largest fp16 model run without CUDA (16 GB unified memory)


def model_family(model_id):
    return "pretrained (instruct)" if "instruct" in model_id.lower() else "pretrained (base)"

CONFIG = {
    "dataset": "google-research-datasets/mbpp",
    "dataset_config": "full",
    "eval_split": "test",
    "n_shot": 3,                # few-shot examples taken from the "prompt" split
    "max_new_tokens": 384,
    "batch_size": {"cuda": 8, "mps": 4, "cpu": 2},   # default per device
    "stop_string": "[DONE]",
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_mbpp():
    """Load MBPP (full config) from the Hugging Face Hub as a DatasetDict."""
    ds = load_dataset(CONFIG["dataset"], CONFIG["dataset_config"])
    print("MBPP splits:", {name: len(split) for name, split in ds.items()})
    return ds


def clean_code(code):
    """MBPP reference code uses \\r\\n line endings and tabs; normalise them."""
    code = code.replace("\r\n", "\n").replace("\t", "    ")
    return "\n".join(line.rstrip() for line in code.split("\n")).strip()


def format_problem(example):
    """The MBPP prompt for one problem, up to and including [BEGIN]."""
    tests = "\n".join(example["test_list"])
    return (
        f"You are an expert Python programmer, and here is your task: {example['text']} "
        f"Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
    )


def build_few_shot_prefix(shots):
    """Worked examples (problem + reference solution) shown before every query."""
    return "".join(
        f"{format_problem(ex)}{clean_code(ex['code'])}\n{CONFIG['stop_string']}\n\n"
        for ex in shots
    )


def build_prompts(eval_split, shots):
    prefix = build_few_shot_prefix(shots)
    return [prefix + format_problem(ex) for ex in eval_split]


def extract_code(generation):
    """Keep only the code before [DONE] (or before a new task if the model ran on)."""
    for marker in (CONFIG["stop_string"], "You are an expert Python programmer"):
        generation = generation.split(marker)[0]
    return generation.strip()


# ----------------------------------------------------------------------------
# Models and generation
# ----------------------------------------------------------------------------
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


GPU_HEADROOM_GIB = 4.0    # per-GPU memory kept free for activations and the KV cache


def gpu_memory_gib():
    return [torch.cuda.get_device_properties(i).total_memory / 2**30
            for i in range(torch.cuda.device_count())]


def model_size_gib(model_id):
    """fp16 weight size from the Hub's safetensors metadata (None if unavailable)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_id)
        return info.safetensors.total * 2 / 2**30 if info.safetensors else None
    except Exception:
        return None


def load_model(model_id, load_4bit=False):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"            # decoder-only models pad on the left
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = pick_device()
    kwargs = {"torch_dtype": torch.float16 if device != "cpu" else torch.float32}
    if device == "cuda":
        gpus = gpu_memory_gib()
        usable = sum(max(g - GPU_HEADROOM_GIB, 1) for g in gpus)
        size = model_size_gib(model_id)
        print(f"  {len(gpus)} GPU(s): {', '.join(f'{g:.1f}' for g in gpus)} GiB; "
              f"fp16 weights ~{size:.1f} GiB" if size else f"  {len(gpus)} GPU(s)")
        if not load_4bit and size and size > usable:
            print(f"  fp16 weights don't fit in {usable:.1f} GiB (after {GPU_HEADROOM_GIB:.0f} GiB "
                  f"headroom per GPU): loading in 4-bit instead")
            load_4bit = True
        # cap each GPU so the KV cache has room; device_map="auto" balances across GPUs
        kwargs["max_memory"] = {i: f"{max(g - GPU_HEADROOM_GIB, 1):.0f}GiB"
                                for i, g in enumerate(gpus)}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4")
        kwargs["device_map"] = "auto"
    elif device == "cuda":
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if "device_map" not in kwargs:
        model.to(device)
    model.eval()
    return tokenizer, model


def free_device_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.no_grad()
def generate(tokenizer, model, prompts, batch_size, max_new_tokens):
    """Greedy decoding in batches; prompts are length-sorted to reduce padding."""
    """If a batch runs out of GPU memory, the batch size is halved and the batch retried."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs = [None] * len(prompts)
    start, b = time.time(), 0
    while b < len(order):
        idx = order[b:b + batch_size]
        try:
            enc = tokenizer([prompts[i] for i in idx], return_tensors="pt",
                            padding=True).to(model.device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                stop_strings=[CONFIG["stop_string"]],
                tokenizer=tokenizer,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if batch_size == 1 or "out of memory" not in str(err).lower():
                raise
            oom = True
        else:
            oom = False
        if oom:   # outside the except block, so the failed batch's tensors can be freed
            enc = out = None
            free_device_memory()
            batch_size = max(1, batch_size // 2)
            print(f"  out of GPU memory: retrying with batch size {batch_size}", flush=True)
            continue
        for i, text in zip(idx, texts):
            outputs[i] = text
        b += len(idx)
        print(f"  {b}/{len(order)} generated ({time.time() - start:.0f}s)", flush=True)
    return outputs


# ----------------------------------------------------------------------------
# Evaluation: BLEU and syntax validity
# ----------------------------------------------------------------------------
def corpus_bleu(predictions, references):
    return sacrebleu.corpus_bleu(predictions, [references])


def sentence_bleu(prediction, reference):
    return sacrebleu.sentence_bleu(prediction, [reference]).score


def syntax_error(code):
    """None if `code` is valid Python, otherwise a short description of the error.

    compile() only parses and builds bytecode; the generated code is never executed.
    """
    if not code.strip():
        return "empty generation"
    try:
        compile(code, "<generated>", "exec")
        return None
    except (SyntaxError, ValueError) as err:     # ValueError: e.g. null bytes in source
        return f"{type(err).__name__}: {getattr(err, 'msg', err)}"


def score_generations(df):
    """Per-problem metrics; recomputed on load so cached generations stay current."""
    df = df.copy()
    df["sentence_bleu"] = [sentence_bleu(p, r) for p, r in zip(df["prediction"], df["reference"])]
    df["syntax_error"] = df["prediction"].map(syntax_error)
    df["syntax_valid"] = df["syntax_error"].isna()
    return df


def short_name(model_id):
    return model_id.split("/")[-1]


def generations_path(model_id, n_problems):
    tag = f"{CONFIG['eval_split']}{n_problems}"     # e.g. test500, so smoke tests don't clash
    return os.path.join(RESULTS_DIR, f"generations_{short_name(model_id)}_{tag}.jsonl")


def run_model(model_id, eval_split, prompts, args):
    """Generate (or load cached) solutions for one model and score them.
    Returns (per-problem DataFrame, info dict with parameter count and timings)."""
    path = generations_path(model_id, len(prompts))
    info_path = path.replace(".jsonl", ".info.json")
    if os.path.exists(path) and not args.overwrite:
        print(f"[{model_id}] using cached generations: {path}")
        info = {}
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
        return score_generations(pd.read_json(path, lines=True)), info

    if pick_device() != "cuda" and not args.force_pretrained:
        size = model_size_gib(model_id)
        if size and size > MAC_MAX_MODEL_GIB:
            print(f"[{model_id}] skipped: fp16 weights ~{size:.1f} GiB > {MAC_MAX_MODEL_GIB:.0f} GiB "
                  f"limit without CUDA (run it on a GPU, or pass --force-pretrained)")
            return None, None

    print(f"[{model_id}] loading model")
    start = time.time()
    tokenizer, model = load_model(model_id, load_4bit=args.load_4bit)
    load_s = time.time() - start
    n_params = model.num_parameters()
    start = time.time()
    batch_size = args.batch_size or CONFIG["batch_size"][pick_device()]
    raw = generate(tokenizer, model, prompts, batch_size, CONFIG["max_new_tokens"])
    info = {"parameters_m": round(n_params / 1e6, 1), "load_time_s": round(load_s, 1),
            "inference_time_s": round(time.time() - start, 1)}
    with open(info_path, "w") as f:
        json.dump(info, f)

    del model
    free_device_memory()

    rows = []
    for ex, prompt, gen in zip(eval_split, prompts, raw):
        pred, ref = extract_code(gen), clean_code(ex["code"])
        rows.append({
            "task_id": ex["task_id"],
            "text": ex["text"],
            "reference": ref,
            "prediction": pred,
            "raw_generation": gen,
        })
    df = score_generations(pd.DataFrame(rows))
    df.to_json(path, orient="records", lines=True)
    print(f"[{model_id}] saved {len(df)} generations -> {path}")
    return df, info


def summarise(model_id, df):
    bleu = corpus_bleu(df["prediction"].tolist(), df["reference"].tolist())
    p1, p2, p3, p4 = bleu.precisions
    return {
        "model": model_id,
        "n_problems": len(df),
        "corpus_bleu": round(bleu.score, 2),
        "mean_sentence_bleu": round(df["sentence_bleu"].mean(), 2),
        "median_sentence_bleu": round(df["sentence_bleu"].median(), 2),
        "precision_1gram": round(p1, 2),
        "precision_2gram": round(p2, 2),
        "precision_3gram": round(p3, 2),
        "precision_4gram": round(p4, 2),
        "brevity_penalty": round(bleu.bp, 3),
        "hyp_ref_length_ratio": round(bleu.sys_len / max(bleu.ref_len, 1), 3),
        "syntax_validity_pct": round(100 * df["syntax_valid"].mean(), 2),
        "syntax_errors": int((~df["syntax_valid"]).sum()),
        "empty_predictions": int((df["prediction"] == "").sum()),
    }


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
BAR_COLOR = "#3b6fb6"
INK, MUTED = "#1f2328", "#6e7781"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.grid(axis="x", color="#d0d7de", linewidth=0.6)
    ax.set_axisbelow(True)


def _bar_panel(ax, s, column, xlabel, fmt, xmax=None):
    bars = ax.barh(s["model"].map(short_name), s[column], color=BAR_COLOR, height=0.55)
    top = xmax or max(s[column].max() * 1.15, 1)
    for bar, val in zip(bars, s[column]):
        ax.text(bar.get_width() + top * 0.01, bar.get_y() + bar.get_height() / 2,
                fmt.format(val), va="center", color=INK, fontsize=10)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_xlim(0, top * (1.12 if xmax else 1))
    _style(ax)


def plot_model_comparison(summary):
    """Corpus BLEU and syntax validity side by side (separate axes, different scales)."""
    s = summary.sort_values("corpus_bleu")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 0.8 * len(s) + 1.6), sharey=True)
    _bar_panel(ax1, s, "corpus_bleu",
               "Corpus BLEU (higher = closer to reference)", "{:.1f}")
    _bar_panel(ax2, s, "syntax_validity_pct",
               "Syntax validity (% of solutions that compile)", "{:.1f}%", xmax=100)
    ax1.set_title("Corpus BLEU", color=INK, loc="left")
    ax2.set_title("Syntax validity score", color=INK, loc="left")
    fig.suptitle(f"MBPP {CONFIG['eval_split']} split: pretrained LLMs "
                 f"({CONFIG['n_shot']}-shot), LoRA fine-tuned LLMs (0-shot) and small "
                 f"transformers trained from scratch; greedy decoding",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)


def plot_sentence_bleu(per_model):
    names = list(per_model)
    fig, ax = plt.subplots(figsize=(8, 0.8 * len(names) + 1.2))
    ax.boxplot([per_model[n]["sentence_bleu"] for n in names], vert=False,
               widths=0.5, patch_artist=True,
               boxprops={"facecolor": "#c9d8ee", "edgecolor": BAR_COLOR},
               medianprops={"color": INK, "linewidth": 2},
               whiskerprops={"color": MUTED}, capprops={"color": MUTED},
               flierprops={"markeredgecolor": MUTED, "markersize": 3})
    ax.set_yticks(range(1, len(names) + 1), [short_name(n) for n in names])
    ax.set_xlabel("Sentence BLEU per problem", color=INK)
    ax.set_title("Distribution of per-problem BLEU", color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sentence_bleu_dist.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--only", choices=["all", "pretrained", "finetuned", "scratch"],
                   default="all",
                   help="run only one family; the others' last saved summaries are reused")
    p.add_argument("--no-tune", action="store_true",
                   help="skip hyperparameter tuning: scratch models train once with defaults "
                        "and LoRA fine-tuning is skipped")
    p.add_argument("--models", nargs="+", default=MODELS, help="pretrained models to run")
    p.add_argument("--force-pretrained", action="store_true",
                   help=f"without CUDA, also run models over {MAC_MAX_MODEL_GIB:.0f} GiB (very slow)")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 20 problems for pretrained models, --quick for scratch models")
    p.add_argument("--split", default=CONFIG["eval_split"],
                   choices=["test", "validation", "train"])
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N problems (smoke test)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="generation batch size (default: 8 on CUDA, 4 on Mac MPS, 2 on CPU)")
    p.add_argument("--4bit", dest="load_4bit", action="store_true",
                   help="load models in 4-bit (bitsandbytes) to fit a 16 GB GPU")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate even if cached generations exist")
    # parse_known_args so the script also runs inside Jupyter/Colab
    return p.parse_known_args(argv)[0]


COMPARISON_COLUMNS = ["model", "family", "parameters_m", "learning_rate", "batch_size",
                      "epochs", "n_problems", "corpus_bleu", "mean_sentence_bleu",
                      "syntax_validity_pct", "training_time_s", "tuning_time_s",
                      "inference_time_s", "total_time_s"]


def run_pretrained(args):
    """Few-shot evaluation of every pretrained model; returns its summary table."""
    ds = load_mbpp()
    eval_split = ds[args.split]
    if args.limit:
        eval_split = eval_split.select(range(min(args.limit, len(eval_split))))
    shots = ds["prompt"].select(range(CONFIG["n_shot"]))
    prompts = build_prompts(eval_split, shots)
    print(f"Evaluating on {len(eval_split)} problems from '{args.split}'. Example prompt:\n")
    print(prompts[0])

    per_model, summary_rows = {}, []
    for model_id in args.models:
        df, info = run_model(model_id, eval_split, prompts, args)
        if df is None:
            continue
        per_model[model_id] = df
        load_s, infer_s = info.get("load_time_s"), info.get("inference_time_s")
        summary_rows.append({
            **summarise(model_id, df),
            "family": model_family(model_id),
            "parameters_m": info.get("parameters_m"),
            "training_time_s": 0.0,
            "load_time_s": load_s,
            "inference_time_s": infer_s,
            "total_time_s": round(load_s + infer_s, 1) if info else None,
        })

    summary = pd.DataFrame(summary_rows).sort_values("corpus_bleu", ascending=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "pretrained_metrics_summary.csv"), index=False)
    plot_sentence_bleu(per_model)
    return summary


def run_scratch(args):
    """Train and evaluate the from-scratch transformers (imported here, not at the top,
    because that file imports its data/scoring helpers from this one)."""
    from Assignment03_transformer_models import main as scratch_main
    flags = (["--quick"] if args.quick else []) + (["--no-tune"] if args.no_tune else [])
    summary, _ = scratch_main(flags)
    return summary.assign(family="from scratch")


def run_finetuned(args):
    """LoRA fine-tune every pretrained model with a hyperparameter search
    (imported here for the same reason as run_scratch)."""
    from Assignment03_finetune import main as finetune_main
    flags = ["--models", *args.models, "--split", args.split]
    if args.quick:
        flags.append("--quick")
    if args.limit:
        flags += ["--limit", str(args.limit)]
    if args.batch_size:
        flags += ["--batch-size", str(args.batch_size)]
    if args.overwrite:
        flags.append("--overwrite")
    return finetune_main(flags)


def load_saved(filename, family_hint):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"(no saved {family_hint} results at {path}; leaving them out of the comparison)")
        return None
    print(f"(using saved {family_hint} results from {path})")
    return pd.read_csv(path)


def main(argv=None):
    args = parse_args(argv)
    CONFIG["eval_split"] = args.split
    if args.quick and not args.limit:
        args.limit = 20
    os.makedirs(RESULTS_DIR, exist_ok=True)
    start = time.time()

    pretrained = (run_pretrained(args) if args.only in ("all", "pretrained")
                  else load_saved("pretrained_metrics_summary.csv", "pretrained"))
    if args.no_tune and args.only in ("all", "finetuned"):
        print("(--no-tune: LoRA fine-tuning skipped)")
        finetuned = None
    else:
        finetuned = (run_finetuned(args) if args.only in ("all", "finetuned")
                     else load_saved("finetuned_metrics_summary.csv", "fine-tuned"))
    scratch = (run_scratch(args) if args.only in ("all", "scratch")
               else load_saved("scratch_metrics_summary.csv", "from-scratch"))
    if scratch is not None and "family" not in scratch:
        scratch = scratch.assign(family="from scratch")

    tables = [t for t in (pretrained, finetuned, scratch) if t is not None]
    comparison = pd.concat(tables, ignore_index=True).reindex(columns=COMPARISON_COLUMNS)
    comparison = comparison.sort_values("corpus_bleu", ascending=False)
    comparison.to_csv(os.path.join(RESULTS_DIR, "model_comparison_summary.csv"), index=False)
    plot_model_comparison(comparison)

    with pd.option_context("display.width", 200):
        print("\nModel comparison (times in seconds):\n", comparison.to_string(index=False))
    print(f"\nResults written to {RESULTS_DIR}  (run took {time.time() - start:.0f}s)")
    return comparison


if __name__ == "__main__":
    main()
