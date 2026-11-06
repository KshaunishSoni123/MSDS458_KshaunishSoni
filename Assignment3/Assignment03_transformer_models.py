"""Assignment 3, part 2: small transformers built from scratch for code generation on MBPP.

Assignment03_KshaunishSoni.py evaluates pretrained code LLMs (1.3-3B parameters).
This file builds three deliberately tiny transformers (~1-2M parameters) from scratch,
trains them on MBPP only, and scores them on the same 500 test problems with the same
metrics (corpus / sentence BLEU, syntax validity score), as a baseline for comparison.
Data loading and scoring are imported from that file. Runs on a Mac (MPS) or CPU.

Building blocks (modular, PyTorch nn.MultiheadAttention):
    EncoderBlock   bidirectional self-attention -> feed-forward
    DecoderBlock   self-attention (causal if autoregressive=True) -> optional
                   cross-attention to an encoder's output -> feed-forward
Both are post-norm residual blocks: x = LayerNorm(x + Dropout(sublayer(x))).

Experiments (EXPERIMENTS):
    encoder_decoder              encoder reads the prompt; an autoregressive decoder with
                                 cross-attention writes the code token by token
    decoder_only                 one decoder stack, NOT autoregressive: the input is
                                 prompt <sep> <mask> <mask> ... and every code token is
                                 predicted in a single parallel pass (no causal mask)
    decoder_only_autoregressive  GPT-style: one causal decoder stack over
                                 prompt <sep> code, generated one token at a time

Tokenizer: a byte-level BPE tokenizer (vocab TOKENIZER_VOCAB) trained from scratch on
the MBPP training split only, so the test set is never seen.

Data: MBPP train (374) + prompt (10) splits for training, validation (90) for
validation loss each epoch (the best epoch's weights are kept), test (500) for the
final scores. The prompt is the task description plus its test asserts, as for the
pretrained LLMs.

Hyperparameter tuning: every experiment is trained once per combination in TUNING_GRID
(learning rate x batch size x epochs; 3 x 3 x 2 = 18 runs, ~1 min each on a Mac).
Each run uses its own warmup + cosine schedule over its own number of epochs, and is
scored by its best validation loss. The run with the lowest validation loss is the
one evaluated on the test split. The test split is never used for selection.
--no-tune skips the search and trains once with TRAIN_CONFIG.

Outputs (results/):
    generations_<experiment>_test500.jsonl   per-problem predictions and scores
    scratch_metrics_summary.csv              one row per experiment, incl. training,
                                             inference and total time (seconds)
    scratch_training_history.csv             train / validation loss per epoch, every run
    scratch_tuning.csv                       one row per hyperparameter combination:
                                             best validation loss, best epoch, time
    scratch_tuning_heatmap.png               validation loss over learning rate x batch size
    scratch_loss_curves.png                  loss curves of the selected configuration
Running Assignment03_KshaunishSoni.py runs these experiments too and adds them to the
all-model comparison; this file can also be run on its own:

    python Assignment03_transformer_models.py            # all three experiments
    python Assignment03_transformer_models.py --quick    # smoke test: 2 epochs, 20 test problems
    python Assignment03_transformer_models.py --no-tune  # one run per experiment, TRAIN_CONFIG
"""

import argparse
import copy
import itertools
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Shared data loading and evaluation, so results match the pretrained-LLM experiments
from Assignment03_KshaunishSoni import (RESULTS_DIR, INK, _style, clean_code,
                                        load_mbpp, pick_device, score_generations,
                                        summarise)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
EXPERIMENTS = {
    "encoder_decoder":             {"kind": "encoder_decoder", "encoder_layers": 2, "decoder_layers": 2},
    "decoder_only":                {"kind": "decoder_only", "layers": 4, "autoregressive": False},
    "decoder_only_autoregressive": {"kind": "decoder_only", "layers": 4, "autoregressive": True},
}

MODEL_CONFIG = {
    "d_model": 128,         # embedding size
    "heads": 4,
    "ff_mult": 4,           # feed-forward hidden size = ff_mult * d_model
    "dropout": 0.1,
}

TRAIN_CONFIG = {
    "epochs": 40,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "warmup_ratio": 0.1,
    "max_src_len": 192,     # prompt tokens (task + tests)
    "max_tgt_len": 192,     # code tokens, including <eos>
    "seed": 42,
}

# Grid searched for every experiment; the selected values replace those in TRAIN_CONFIG
TUNING_GRID = {
    "learning_rate": [3e-4, 1e-3, 3e-3],
    "batch_size": [8, 16, 32],
    "epochs": [20, 40],
}

QUICK_CONFIG = {"epochs": 2, "test_limit": 20}
QUICK_GRID = {"learning_rate": [1e-3, 3e-3], "batch_size": [16], "epochs": [2]}

TOKENIZER_VOCAB = 2000
SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>", "<mask>"]
PAD, BOS, EOS, SEP, MASK = range(len(SPECIALS))     # ids of the special tokens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------
# Tokenizer and data
# ----------------------------------------------------------------------------
def format_source(example):
    """Prompt: the task and its tests (the tests give the function name/signature)."""
    tests = "\n".join(example["test_list"])
    return f"{example['text']}\nYour code should pass these tests:\n{tests}"


def train_tokenizer(texts):
    """Byte-level BPE from scratch: any text can be encoded, whitespace is preserved."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=TOKENIZER_VOCAB, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    return tok


def encode_split(tok, split, cfg):
    """[(prompt ids, code ids + <eos>)] truncated to the configured lengths."""
    pairs = []
    for ex in split:
        src = tok.encode(format_source(ex)).ids[:cfg["max_src_len"]]
        tgt = tok.encode(clean_code(ex["code"])).ids[:cfg["max_tgt_len"] - 1] + [EOS]
        pairs.append((src, tgt))
    return pairs


def pad(seqs, value):
    width = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (width - len(s)) for s in seqs])


# Each model kind turns (prompt, code) pairs into (model inputs, labels).
# Labels of -100 are ignored by the loss.
def batch_encoder_decoder(pairs, cfg):
    src = pad([s for s, _ in pairs], PAD)
    tgt_in = pad([[BOS] + t[:-1] for _, t in pairs], PAD)     # decoder input, shifted right
    labels = pad([t for _, t in pairs], -100)
    return (src, tgt_in), labels


def batch_decoder_autoregressive(pairs, cfg):
    # sequence: prompt <sep> code <eos>; predict the next token, scored on the code part only
    ids = pad([s + [SEP] + t[:-1] for s, t in pairs], PAD)
    labels = pad([[-100] * len(s) + t for s, t in pairs], -100)
    return (ids,), labels


def batch_decoder_parallel(pairs, cfg):
    # sequence: prompt <sep> <mask> x max_tgt_len; each <mask> predicts its code token
    T = cfg["max_tgt_len"]
    ids = pad([s + [SEP] + [MASK] * T for s, _ in pairs], PAD)
    labels = pad([[-100] * (len(s) + 1) + t + [-100] * (T - len(t)) for s, t in pairs], -100)
    return (ids,), labels


# ----------------------------------------------------------------------------
# Transformer blocks
# ----------------------------------------------------------------------------
class Embeddings(nn.Module):
    """Token embedding + learned positional embedding (as in GPT-2)."""
    def __init__(self, vocab_size, d_model, max_len, dropout):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.position = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids):
        positions = torch.arange(ids.size(1), device=ids.device)
        return self.dropout(self.token(ids) + self.position(positions))


def feed_forward(d_model, ff_mult):
    return nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
                         nn.Linear(ff_mult * d_model, d_model))


class EncoderBlock(nn.Module):
    """Bidirectional self-attention + feed-forward: every token sees every other token."""
    def __init__(self, d_model, heads, ff_mult, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ff = feed_forward(d_model, ff_mult)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        a, _ = self.self_attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = self.norm1(x + self.dropout(a))
        return self.norm2(x + self.dropout(self.ff(x)))


class DecoderBlock(nn.Module):
    """Self-attention (causal when autoregressive) -> optional cross-attention -> feed-forward.

    autoregressive=True   each position attends only to itself and earlier positions,
                          so the model can be trained to predict the next token
    cross_attention=True  also attends to an encoder's output (encoder-decoder model)
    """
    def __init__(self, d_model, heads, ff_mult, dropout, autoregressive=True, cross_attention=False):
        super().__init__()
        self.autoregressive = autoregressive
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.cross_attn = (nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
                           if cross_attention else None)
        self.ff = feed_forward(d_model, ff_mult)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None, memory=None, memory_pad_mask=None):
        causal = None
        if self.autoregressive:   # True above the diagonal = "may not attend to the future"
            L = x.size(1)
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        a, _ = self.self_attn(x, x, x, attn_mask=causal, key_padding_mask=pad_mask,
                              need_weights=False)
        x = self.norm1(x + self.dropout(a))
        if self.cross_attn is not None:
            c, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_pad_mask,
                                   need_weights=False)
            x = self.norm2(x + self.dropout(c))
        return self.norm3(x + self.dropout(self.ff(x)))


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, encoder_layers, decoder_layers,
                 d_model, heads, ff_mult, dropout):
        super().__init__()
        self.embed = Embeddings(vocab_size, d_model, max_len, dropout)   # shared by both stacks
        self.encoder = nn.ModuleList(
            EncoderBlock(d_model, heads, ff_mult, dropout) for _ in range(encoder_layers))
        self.decoder = nn.ModuleList(
            DecoderBlock(d_model, heads, ff_mult, dropout, autoregressive=True, cross_attention=True)
            for _ in range(decoder_layers))
        self.out = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        src_pad = src == PAD
        x = self.embed(src)
        for block in self.encoder:
            x = block(x, pad_mask=src_pad)
        return x, src_pad

    def decode(self, tgt_in, memory, src_pad):
        x = self.embed(tgt_in)
        for block in self.decoder:
            x = block(x, memory=memory, memory_pad_mask=src_pad)
        return self.out(x)

    def forward(self, src, tgt_in):
        memory, src_pad = self.encode(src)
        return self.decode(tgt_in, memory, src_pad)


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, layers, autoregressive,
                 d_model, heads, ff_mult, dropout):
        super().__init__()
        self.embed = Embeddings(vocab_size, d_model, max_len, dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(d_model, heads, ff_mult, dropout, autoregressive=autoregressive)
            for _ in range(layers))
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        pad_mask = ids == PAD
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)
        return self.out(x)


def build_model(spec, vocab_size, cfg):
    max_len = cfg["max_src_len"] + 1 + cfg["max_tgt_len"]
    if spec["kind"] == "encoder_decoder":
        return EncoderDecoderTransformer(vocab_size, max_len, spec["encoder_layers"],
                                         spec["decoder_layers"], **MODEL_CONFIG)
    return DecoderOnlyTransformer(vocab_size, max_len, spec["layers"], spec["autoregressive"],
                                  **MODEL_CONFIG)


def batch_fn(spec):
    if spec["kind"] == "encoder_decoder":
        return batch_encoder_decoder
    return batch_decoder_autoregressive if spec["autoregressive"] else batch_decoder_parallel


# ----------------------------------------------------------------------------
# Generation (greedy)
# ----------------------------------------------------------------------------
def until_eos(ids):
    return ids[:ids.index(EOS)] if EOS in ids else ids


@torch.no_grad()
def generate_encoder_decoder(model, srcs, cfg, device):
    """Encode the prompts once, then decode one token per step for the whole batch."""
    src = pad(srcs, PAD).to(device)
    memory, src_pad = model.encode(src)
    out = torch.full((len(srcs), 1), BOS, device=device)
    done = torch.zeros(len(srcs), dtype=torch.bool, device=device)
    for _ in range(cfg["max_tgt_len"]):
        nxt = model.decode(out, memory, src_pad)[:, -1].argmax(-1)
        nxt[done] = PAD
        out = torch.cat([out, nxt[:, None]], dim=1)
        done |= nxt == EOS
        if done.all():
            break
    return [until_eos(row[1:]) for row in out.tolist()]


@torch.no_grad()
def generate_decoder_autoregressive(model, srcs, cfg, device):
    """One prompt at a time: append the most likely next token until <eos>."""
    results = []
    for s in srcs:
        ids = torch.tensor([s + [SEP]], device=device)
        for _ in range(cfg["max_tgt_len"]):
            nxt = model(ids)[0, -1].argmax()
            if nxt.item() == EOS:
                break
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
        results.append(ids[0, len(s) + 1:].tolist())
    return results


@torch.no_grad()
def generate_decoder_parallel(model, srcs, cfg, device):
    """A single forward pass predicts every <mask> slot at once."""
    T = cfg["max_tgt_len"]
    ids = pad([s + [SEP] + [MASK] * T for s in srcs], PAD).to(device)
    preds = model(ids).argmax(-1).tolist()
    return [until_eos(p[len(s) + 1:len(s) + 1 + T]) for p, s in zip(preds, srcs)]


def generate(spec, model, srcs, cfg, device, batch_size=32):
    model.eval()
    if spec["kind"] == "encoder_decoder":
        fn = generate_encoder_decoder
    else:
        fn = generate_decoder_autoregressive if spec["autoregressive"] else generate_decoder_parallel
    out = []
    for b in range(0, len(srcs), batch_size):
        out += fn(model, srcs[b:b + batch_size], cfg, device)
    return out


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def compute_loss(model, inputs, labels, device):
    logits = model(*(t.to(device) for t in inputs))
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.to(device).reshape(-1),
                           ignore_index=-100)


def batches(pairs, make_batch, cfg, shuffle):
    order = list(range(len(pairs)))
    if shuffle:
        random.shuffle(order)
    for b in range(0, len(order), cfg["batch_size"]):
        yield make_batch([pairs[i] for i in order[b:b + cfg["batch_size"]]], cfg)


def train(model, spec, train_pairs, val_pairs, cfg, device, name):
    """AdamW, linear warmup then cosine decay; keeps the best validation-loss weights."""
    make_batch = batch_fn(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.01)
    steps = cfg["epochs"] * math.ceil(len(train_pairs) / cfg["batch_size"])
    warmup = max(1, int(cfg["warmup_ratio"] * steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(
        (s + 1) / warmup, 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))))

    history, best_val, best_state = [], math.inf, None
    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        model.train()
        train_losses = []
        for inputs, labels in batches(train_pairs, make_batch, cfg, shuffle=True):
            loss = compute_loss(model, inputs, labels, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([compute_loss(model, i, l, device).item()
                                      for i, l in batches(val_pairs, make_batch, cfg, False)]))
        train_loss = float(np.mean(train_losses))
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        history.append({"experiment": name, "learning_rate": cfg["learning_rate"],
                        "batch_size": cfg["batch_size"], "epochs": cfg["epochs"],
                        "epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "seconds": time.time() - start})
        print(f"  epoch {epoch:2d}/{cfg['epochs']}  train {train_loss:.3f}  val {val_loss:.3f}"
              f"{'  *best' if best_epoch == epoch else ''}  ({time.time() - start:.1f}s)", flush=True)

    model.load_state_dict(best_state)
    print(f"  restored weights from epoch {best_epoch} (val loss {best_val:.3f})")
    return history, best_val, best_epoch


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def grid_configs(base_cfg, grid):
    """One training config per combination of the grid's values."""
    keys = list(grid)
    return [{**base_cfg, **dict(zip(keys, values))}
            for values in itertools.product(*(grid[k] for k in keys))]


def tune(name, spec, tok, data, base_cfg, grid, device):
    """Train once per grid combination; keep the model with the lowest validation loss.
    Returns (best model, its config, its history, all histories, tuning table)."""
    configs = grid_configs(base_cfg, grid)
    best, histories, rows = None, [], []
    tune_start = time.time()
    for i, cfg in enumerate(configs, 1):
        print(f"\n  [{name}] run {i}/{len(configs)}: lr {cfg['learning_rate']:g}, "
              f"batch {cfg['batch_size']}, epochs {cfg['epochs']}")
        set_seed(cfg["seed"])
        model = build_model(spec, tok.get_vocab_size(), cfg).to(device)
        start = time.time()
        history, val, epoch = train(model, spec, data["train"], data["validation"], cfg, device, name)
        seconds = time.time() - start
        histories += history
        rows.append({"experiment": name, "learning_rate": cfg["learning_rate"],
                     "batch_size": cfg["batch_size"], "epochs": cfg["epochs"],
                     "best_val_loss": round(val, 4), "best_epoch": epoch,
                     "train_time_s": round(seconds, 1)})
        if best is None or val < best[2]:
            best = (model, cfg, val, history, seconds)
        else:
            del model
    model, cfg, val, history, train_s = best
    table = pd.DataFrame(rows)
    table["selected"] = ((table["learning_rate"] == cfg["learning_rate"])
                         & (table["batch_size"] == cfg["batch_size"])
                         & (table["epochs"] == cfg["epochs"]))
    print(f"\n  [{name}] selected lr {cfg['learning_rate']:g}, batch {cfg['batch_size']}, "
          f"epochs {cfg['epochs']} (val loss {val:.3f}; {len(configs)} runs, "
          f"{time.time() - tune_start:.0f}s)")
    return model, cfg, history, histories, table, train_s, time.time() - tune_start


def run_experiment(name, spec, tok, data, test_split, base_cfg, grid, device):
    print(f"\n=== {name} ===")
    model, cfg, history, all_history, tuning, train_s, tune_s = tune(
        name, spec, tok, data, base_cfg, grid, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e6:.2f}M parameters")

    start = time.time()
    preds = generate(spec, model, [s for s, _ in data["test"]], cfg, device)
    df = score_generations(pd.DataFrame({
        "task_id": test_split["task_id"],
        "text": test_split["text"],
        "reference": [clean_code(c) for c in test_split["code"]],
        "prediction": [tok.decode(p).strip() for p in preds],
    }))
    df.to_json(os.path.join(RESULTS_DIR, f"generations_{name}_test{len(df)}.jsonl"),
               orient="records", lines=True)
    infer_s = time.time() - start
    row = {**summarise(name, df), "family": "from scratch",
           "parameters_m": round(n_params / 1e6, 2),
           "learning_rate": cfg["learning_rate"], "batch_size": cfg["batch_size"],
           "epochs": cfg["epochs"], "best_val_loss": round(tuning["best_val_loss"].min(), 4),
           "training_time_s": round(train_s, 1), "tuning_time_s": round(tune_s, 1),
           "inference_time_s": round(infer_s, 1), "total_time_s": round(tune_s + infer_s, 1)}
    print(f"  corpus BLEU {row['corpus_bleu']:.2f}  syntax validity "
          f"{row['syntax_validity_pct']:.1f}%  (tuning {tune_s:.0f}s, generation {infer_s:.0f}s)")
    return row, history, all_history, tuning


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_loss_curves(history):
    names = list(dict.fromkeys(history["experiment"]))
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 3.8), squeeze=False)
    for ax, name in zip(axes[0], names):
        h = history[history["experiment"] == name]
        ax.plot(h["epoch"], h["train_loss"], color="#3b6fb6", linewidth=2, label="train")
        ax.plot(h["epoch"], h["val_loss"], color="#d9822b", linewidth=2, label="validation")
        best = h.loc[h["val_loss"].idxmin()]
        ax.scatter([best["epoch"]], [best["val_loss"]], color="#d9822b", s=40, zorder=3)
        ax.annotate(f"best epoch {int(best['epoch'])}", (best["epoch"], best["val_loss"]),
                    textcoords="offset points", xytext=(4, 8), color=INK, fontsize=9)
        ax.set_title(name, color=INK, loc="left")
        ax.set_xlabel("Epoch", color=INK)
        ax.set_ylabel("Cross-entropy loss", color=INK)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(frameon=False, labelcolor=INK)
        _style(ax)
        ax.grid(axis="y", color="#d0d7de", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "scratch_loss_curves.png"), dpi=150)
    plt.close(fig)


def plot_tuning_heatmap(tuning, group_col, path, title):
    """One panel per model: best validation loss over learning rate x batch size.
    Each cell is the best value over the epoch settings; the selected run is outlined."""
    names = list(dict.fromkeys(tuning[group_col]))
    cols = min(3, len(names))
    rows_n = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.3 * cols, 3.4 * rows_n), squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax, name in zip(axes.flat, names):
        t = tuning[tuning[group_col] == name]
        best = t.loc[t.groupby(["learning_rate", "batch_size"])["best_val_loss"].idxmin()]
        grid = best.pivot(index="learning_rate", columns="batch_size", values="best_val_loss")
        epochs = best.pivot(index="learning_rate", columns="batch_size", values="best_epoch")
        # one hue, light -> dark; darker = lower (better) validation loss
        im = ax.imshow(grid.values, cmap="Blues_r", aspect="auto")
        lo, hi = np.nanmin(grid.values), np.nanmax(grid.values)
        sel = t[t["selected"]].iloc[0]
        for r, lr in enumerate(grid.index):
            for c, bs in enumerate(grid.columns):
                v = grid.loc[lr, bs]
                if np.isnan(v):
                    continue
                dark = hi > lo and (v - lo) / (hi - lo) < 0.5
                ax.text(c, r, f"{v:.3f}\nep {int(epochs.loc[lr, bs])}", ha="center",
                        va="center", fontsize=8.5, color="white" if dark else INK)
                if lr == sel["learning_rate"] and bs == sel["batch_size"]:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                               edgecolor="#d9822b", linewidth=2.5))
        ax.set_xticks(range(len(grid.columns)), [str(b) for b in grid.columns])
        ax.set_yticks(range(len(grid.index)), [f"{lr:g}" for lr in grid.index])
        ax.set_xlabel("Batch size", color=INK)
        ax.set_ylabel("Learning rate", color=INK)
        ax.set_title(str(name).split("/")[-1], color=INK, loc="left", fontsize=10)
        ax.tick_params(colors=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(title, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS),
                   choices=list(EXPERIMENTS))
    p.add_argument("--epochs", type=int, default=None,
                   help="train for exactly this many epochs (replaces the epochs grid)")
    p.add_argument("--no-tune", action="store_true",
                   help="skip the hyperparameter search: one run per experiment with TRAIN_CONFIG")
    p.add_argument("--quick", action="store_true",
                   help=f"smoke test: {QUICK_CONFIG['epochs']} epochs, "
                        f"{QUICK_CONFIG['test_limit']} test problems")
    return p.parse_known_args(argv)[0]   # parse_known_args so it also runs in Jupyter


def main(argv=None):
    args = parse_args(argv)
    cfg = dict(TRAIN_CONFIG)
    grid = dict(QUICK_GRID if args.quick else TUNING_GRID)
    if args.no_tune:
        grid = {k: [cfg[k]] for k in grid}
        if args.quick:
            grid["epochs"] = [QUICK_CONFIG["epochs"]]
    if args.epochs:
        grid["epochs"] = [args.epochs]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = pick_device()
    print(f"Device: {device}\nModel: {MODEL_CONFIG}\nTraining: {cfg}\nTuning grid: {grid}")

    ds = load_mbpp()
    train_split = [*ds["train"], *ds["prompt"]]
    test_split = ds["test"]
    if args.quick:
        test_split = test_split.select(range(QUICK_CONFIG["test_limit"]))

    tok = train_tokenizer([format_source(ex) for ex in train_split]
                          + [clean_code(ex["code"]) for ex in train_split])
    data = {"train": encode_split(tok, train_split, cfg),
            "validation": encode_split(tok, ds["validation"], cfg),
            "test": encode_split(tok, test_split, cfg)}
    src_lens = [len(tok.encode(format_source(ex)).ids) for ex in train_split]
    tgt_lens = [len(tok.encode(clean_code(ex["code"])).ids) + 1 for ex in train_split]
    print(f"Tokenizer: {tok.get_vocab_size()} tokens. Train prompts truncated: "
          f"{np.mean(np.array(src_lens) > cfg['max_src_len']):.1%}, "
          f"code truncated: {np.mean(np.array(tgt_lens) > cfg['max_tgt_len']):.1%}")

    rows, selected, histories, tunings = [], [], [], []
    for name in args.experiments:
        row, history, all_history, tuning = run_experiment(
            name, EXPERIMENTS[name], tok, data, test_split, cfg, grid, device)
        rows.append(row)
        selected += history
        histories += all_history
        tunings.append(tuning)

    summary = pd.DataFrame(rows)
    tuning = pd.concat(tunings, ignore_index=True)
    summary.to_csv(os.path.join(RESULTS_DIR, "scratch_metrics_summary.csv"), index=False)
    pd.DataFrame(histories).to_csv(os.path.join(RESULTS_DIR, "scratch_training_history.csv"),
                                   index=False)
    tuning.to_csv(os.path.join(RESULTS_DIR, "scratch_tuning.csv"), index=False)
    plot_loss_curves(pd.DataFrame(selected))
    plot_tuning_heatmap(tuning, "experiment",
                        os.path.join(RESULTS_DIR, "scratch_tuning_heatmap.png"),
                        "From-scratch transformers: best validation loss by learning rate and "
                        "batch size (outlined = selected)")
    print("\nHyperparameter search (best validation loss per run):\n",
          tuning.to_string(index=False))
    cols = ["model", "parameters_m", "learning_rate", "batch_size", "epochs",
            "corpus_bleu", "mean_sentence_bleu", "syntax_validity_pct",
            "tuning_time_s", "inference_time_s", "total_time_s"]
    print("\nFrom-scratch transformers:\n", summary[cols].to_string(index=False))

    print(f"\nResults written to {RESULTS_DIR}")
    return summary, history


if __name__ == "__main__":
    main()
