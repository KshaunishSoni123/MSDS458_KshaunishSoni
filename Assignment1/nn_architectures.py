"""Architecture sweep for the diamond price regression networks.

Every network shares the same training setup (CONFIG): optimizer, learning rate,
batch size, epochs, early stopping, batch normalization, ReLU activation and seed.
Only the hidden-layer architecture changes, so differences in the results come
from depth and width alone. The output node predicts price directly, in USD.

An architecture is a list of hidden-layer widths, e.g. [128, 64] is two hidden
layers with 128 and 64 nodes. Pass a dict {name: widths} (or a list of width
lists) to run_architecture_sweep().

Run the full 10-architecture sweep with:

    SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())") python3 nn_architectures.py
"""

import os
import time
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from Assignment1.diamonds_utils import (prepare_data, fit_regression, validate_model, OutputScale,
                                        hidden_activation, init_he_weights, fit_full,
                                        lr_schedule_to_best_epoch)
from feature_engineering import (build_features, NUMERICAL_FEATURES,
                                 CATEGORICAL_FEATURES, TARGET)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Shared training setup: identical for every architecture
CONFIG = {
    "lr": 1e-3,            # Adam learning rate
    "batch_size": 64,
    "epochs": 100,         # maximum; early stopping usually ends sooner
    "patience": 15,        # epochs without val_loss improvement before stopping
    "batch_norm": True,    # Linear -> BatchNorm1d -> activation in every hidden layer
    "activation": "leaky_relu",  # hidden activation: 'leaky_relu' or 'relu'
    "negative_slope": 0.01,      # LeakyReLU slope for x < 0
    "init": "he",          # He (Kaiming) normal init matched to the activation; None = PyTorch default
    "lr_factor": 0.5,      # performance scheduling: halve the lr when val_loss plateaus...
    "lr_patience": 4,      # ...for this many epochs (kept below the early-stopping patience)
    "min_lr": 1e-6,
    "ensemble_size": 5,    # refit the selected architecture on all training data with this many seeds
    "val_size": 0.2,       # fraction of the training set held out for validation
    "seed": 73,
}

# Architecture matrix: name -> hidden-layer widths (depth = len, width = values)
ARCHITECTURES = {
    "A01_1x32":        [32],
    "A02_1x64":        [64],
    "A03_1x128":       [128],
    "A04_2_64-32":     [64, 32],
    "A05_2_128-64":    [128, 64],
    "A06_3_64-64-64":  [64, 64, 64],
    "A07_3_128-64-32": [128, 64, 32],
    "A08_4_128-64-32-16":     [128, 64, 32, 16],
    "A09_4_256-128-64-32":    [256, 128, 64, 32],
    "A10_5_256-128-64-32-16": [256, 128, 64, 32, 16],
}

# Accuracy requirements: mean APE (MAPE) at or below the target, and 95% of
# individual predictions within the limit (95th percentile of APE)
MAPE_TARGET = 5.0
APE_P95_LIMIT = 20.0

# Plot palette (validated categorical slots 1-2, status pair and text tokens)
GOOD_COLOR = "#0ca30c"
CRITICAL_COLOR = "#d03b3b"
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID_GRAY = "#e1e0d9"


def set_seed(seed):
    """Seed Python, NumPy and PyTorch so every architecture starts from the same state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_mlp(input_dim, hidden_layers, batch_norm=True, output_mean=0.0, output_scale=1.0,
              n_outputs=1, activation="relu", negative_slope=0.01, init=None):
    """Build a fully connected network: a price regressor or a tier classifier.

    Args:
        input_dim (int): Number of input features
        hidden_layers (list[int]): Width of each hidden layer, in order
        batch_norm (bool): Insert BatchNorm1d between each Linear and its activation
        output_mean, output_scale (float): Fixed rescaling of the regression output,
            normally the training-set mean and std of price
        n_outputs (int): 1 for regression (price, rescaled by OutputScale); >1 for a
            classifier emitting raw logits (CrossEntropyLoss applies the softmax)
        activation (str): Hidden activation, 'relu' or 'leaky_relu'
        negative_slope (float): LeakyReLU slope
        init (str): 'he' for He (Kaiming) normal initialization, None for PyTorch default

    Returns:
        torch.nn.Sequential: The network (not yet moved to a device)
    """
    if not hidden_layers:
        raise ValueError("build_mlp(): hidden_layers must contain at least one layer width")
    layers, in_dim = [], input_dim
    for width in hidden_layers:
        layers.append(nn.Linear(in_dim, width))
        if batch_norm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(hidden_activation(activation, negative_slope))
        in_dim = width
    layers.append(nn.Linear(in_dim, n_outputs))
    if n_outputs == 1:
        layers.append(OutputScale(output_mean, output_scale))
    model = nn.Sequential(*layers)
    if init == "he":
        init_he_weights(model, activation, negative_slope)
    return model


class EnsembleRegressor(nn.Module):
    """Averages the price predictions of several independently trained networks."""
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members], dim=0).mean(dim=0)


def refit_ensemble(hidden_layers, X_train, y_train, history, config=CONFIG, n_members=None):
    """Refit an architecture on ALL training rows with several seeds and average them.

    There is no validation data left, so each member trains for the same number of
    epochs, with the same per-epoch learning rates, as the selection run did up to
    its best epoch (du.lr_schedule_to_best_epoch). Member i uses seed config['seed'] + i.

    Args:
        hidden_layers (list[int]): Architecture chosen by the sweep
        X_train, y_train: Full preprocessed training set (no validation hold-out)
        history (dict | HistoryObject): The selection run's training history
        config (dict): Shared training setup (see CONFIG)
        n_members (int): Ensemble size, defaults to config['ensemble_size']

    Returns:
        EnsembleRegressor: Averaging ensemble of the refit networks
    """
    n_members = n_members or config["ensemble_size"]
    schedule = lr_schedule_to_best_epoch(history)
    y_np = np.asarray(y_train, dtype=np.float32)
    members = []
    for i in range(n_members):
        set_seed(config["seed"] + i)
        model = build_mlp(X_train.shape[1], hidden_layers, batch_norm=config["batch_norm"],
                          output_mean=float(y_np.mean()), output_scale=float(y_np.std()),
                          activation=config["activation"],
                          negative_slope=config["negative_slope"], init=config["init"])
        print(f"  refit member {i + 1}/{n_members}: {len(schedule)} epochs on "
              f"{len(X_train):,} rows (seed {config['seed'] + i})")
        fit_full(model, X_train, y_np, schedule, batch_size=config["batch_size"],
                 task="regression")
        members.append(model)
    return EnsembleRegressor(members)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _as_named(architectures):
    """Accept {name: widths} or [widths, ...] and return {name: widths}."""
    if isinstance(architectures, dict):
        return dict(architectures)
    return {f"A{i:02d}_" + "-".join(map(str, w)): list(w)
            for i, w in enumerate(architectures, start=1)}


def run_architecture_sweep(architectures, X_train, y_train, config=CONFIG,
                           inverse_transform=None, results_dir=RESULTS_DIR,
                           verbose=False):
    """Train one network per architecture under an identical setup and compare them.

    The training set is split once into train/validation (same split for every
    architecture). Each model is scored with validate_model() on both splits and
    its row is written to
    results/metrics.csv. The test set is not touched here; keep it for the final
    chosen model.

    Args:
        architectures (dict | list): {name: [widths]} or a list of width lists
        X_train (numpy.ndarray): Preprocessed training features
        y_train (array-like): Training targets (price in USD)
        config (dict): Shared training setup (see CONFIG)
        inverse_transform (callable, optional): Maps targets to dollars before scoring;
            None when the target is already price
        results_dir (str): Where the comparison table, metrics and plots are written
        verbose (bool): Print per-epoch training metrics

    Returns:
        tuple: (comparison DataFrame sorted by val MAPE, {name: history dict}, {name: model})
    """
    architectures = _as_named(architectures)
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.csv")

    y_np = np.asarray(y_train, dtype=np.float32)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_np, test_size=config["val_size"], random_state=config["seed"])

    # BatchNorm cannot normalize a training batch of one sample
    if config["batch_norm"] and len(X_tr) % config["batch_size"] == 1:
        raise ValueError(f"Training set size {len(X_tr)} leaves a final batch of 1 with "
                         f"batch_size={config['batch_size']}; pick another batch size.")

    # Output rescaling comes from the training split only (no validation leakage)
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())

    rows, histories, models = [], {}, {}
    for name, widths in architectures.items():
        set_seed(config["seed"])
        model = build_mlp(X_tr.shape[1], widths, batch_norm=config["batch_norm"],
                          output_mean=y_mean, output_scale=y_std,
                          activation=config["activation"],
                          negative_slope=config["negative_slope"], init=config["init"])
        n_params = count_parameters(model)
        print(f"\n--- {name}: hidden layers {widths} ({n_params:,} parameters) ---")

        start = time.time()
        history = fit_regression(model, X_tr, y_tr, X_val, y_val,
                                 lr=config["lr"], epochs=config["epochs"],
                                 batch_size=config["batch_size"],
                                 patience=config["patience"], lr_factor=config["lr_factor"],
                                 lr_patience=config["lr_patience"], min_lr=config["min_lr"],
                                 verbose=verbose)
        train_time = time.time() - start
        h = history.history
        best_epoch = int(np.argmin(h["val_loss"]))

        notes = (f"layers={widths}; bn={config['batch_norm']}; act={config['activation']}; "
                 f"init={config['init']}; lr={config['lr']}; bs={config['batch_size']}")
        train_table = validate_model(model, X_tr, y_tr, name, split="train",
                                     inverse_transform=inverse_transform,
                                     table_path=metrics_path, notes=notes)
        val_table = validate_model(model, X_val, y_val, name, split="val",
                                   inverse_transform=inverse_transform,
                                   table_path=metrics_path, notes=notes)
        tr = train_table[(train_table.model == name) & (train_table.split == "train")].iloc[-1]
        va = val_table[(val_table.model == name) & (val_table.split == "val")].iloc[-1]

        rows.append({
            "model": name,
            "hidden_layers": str(widths),
            "depth": len(widths),
            "params": n_params,
            "epochs_run": len(h["loss"]),
            "best_epoch": best_epoch + 1,
            "train_time_s": round(train_time, 1),
            "train_MAE": tr["MAE"], "train_RMSE": tr["RMSE"], "train_MAPE": tr["MAPE"],
            "val_MAE": va["MAE"], "val_RMSE": va["RMSE"], "val_MAPE": va["MAPE"],
            "val_APE_P95": va["APE_P95"], "val_R2": va["R2"],
        })
        histories[name] = h
        models[name] = model

    comparison = (pd.DataFrame(rows).sort_values("val_MAPE").reset_index(drop=True))
    comparison.to_csv(os.path.join(results_dir, "architecture_comparison.csv"), index=False)
    pd.DataFrame([{"model": n, "epoch": e + 1, **{k: v[e] for k, v in h.items()}}
                  for n, h in histories.items() for e in range(len(h["loss"]))]
                 ).to_csv(os.path.join(results_dir, "architecture_histories.csv"), index=False)
    return comparison, histories, models


def _style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID_GRAY, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_GRAY)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_learning_curves(histories, path, metric="mape", ylabel="MAPE (%)", dollars=False,
                         log_scale=True, title="Learning curves by architecture"):
    """Small multiples: train vs validation `metric` (a history key) per architecture."""
    names = list(histories)
    ncols = 5
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows),
                             sharey=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        h = histories[name]
        epochs = np.arange(1, len(h[metric]) + 1)
        ax.plot(epochs, h[metric], color=TRAIN_COLOR, linewidth=2, label="Train")
        ax.plot(epochs, h[f"val_{metric}"], color=VAL_COLOR, linewidth=2, label="Validation")
        ax.set_title(name, fontsize=10, color=INK)
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        if dollars:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _style_axis(ax)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch", color=INK, fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)
    fig.suptitle(title, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


def plot_comparison(comparison, path, title="Validation MAPE by architecture"):
    """Horizontal bars of validation MAPE, best (lowest) architecture on top."""
    df = comparison.sort_values("val_MAPE", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(df) + 1.2))
    bars = ax.barh(df["model"], df["val_MAPE"], color=TRAIN_COLOR, height=0.6)
    for bar, mape, rmse in zip(bars, df["val_MAPE"], df["val_RMSE"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {mape:.2f}%  (RMSE ${rmse:,.0f})", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("Validation MAPE (%)", color=INK)
    ax.set_title(title, fontweight="bold", color=INK, loc="left")
    ax.set_xlim(0, df["val_MAPE"].max() * 1.3)
    _style_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9, colors=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    df = build_features(verbose=False)
    X_train, X_test, y_train, y_test, _ = prepare_data(
        df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, target=TARGET)

    print("Shared config:", CONFIG)
    comparison, histories, _ = run_architecture_sweep(ARCHITECTURES, X_train, y_train)

    plot_learning_curves(histories, os.path.join(RESULTS_DIR, "architecture_learning_curves.png"))
    plot_comparison(comparison, os.path.join(RESULTS_DIR, "architecture_val_mape.png"))

    pd.set_option("display.width", 200)
    cols = ["model", "params", "epochs_run", "best_epoch", "train_time_s",
            "train_RMSE", "val_MAE", "val_RMSE", "val_MAPE", "val_R2"]
    print("\nArchitecture comparison (sorted by validation MAPE):")
    print(comparison[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()


def plot_target_comparison(df, path, title, label_col="model", prefix="val_",
                           mape_target=MAPE_TARGET, p95_limit=APE_P95_LIMIT):
    """Three panels per model: RMSE, MAPE vs the MAPE target, and APE P95 vs the limit.

    Bars in the MAPE / APE P95 panels are green when the model meets the requirement
    and red when it misses, with PASS / MISS in the value label so pass/fail
    does not rely on color alone. Models keep the same order (best MAPE on top)
    across panels.

    Args:
        df (pandas.DataFrame): One row per model with {prefix}RMSE, {prefix}MAPE and
            {prefix}APE_P95 columns
        path (str): Output PNG path
        title (str): Figure title
        label_col (str): Column holding the model name
        prefix (str): Metric column prefix ('val_' for sweep tables, '' for test tables)
    """
    d = df.sort_values(f"{prefix}MAPE", ascending=False).reset_index(drop=True)
    labels = d[label_col].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.5 * len(d) + 2.2), sharey=True)

    panels = [
        ("RMSE", "RMSE (USD)", None, lambda v: f"${v:,.0f}"),
        ("MAPE", "MAPE (%)", (mape_target, f"Target {mape_target:g}%"), lambda v: f"{v:.1f}%"),
        ("APE_P95", "95th-percentile APE (%)",
         (p95_limit, f"Limit {p95_limit:g}% (95% of diamonds)"), lambda v: f"{v:.1f}%"),
    ]
    for ax, (metric, xlabel, line, fmt) in zip(axes, panels):
        values = d[f"{prefix}{metric}"].astype(float).to_numpy()
        if line is None:
            colors = [TRAIN_COLOR] * len(values)
            marks = [""] * len(values)
        else:
            passed = values <= line[0]
            colors = [GOOD_COLOR if ok else CRITICAL_COLOR for ok in passed]
            marks = ["  PASS" if ok else "  MISS" for ok in passed]
        bars = ax.barh(labels, values, color=colors, height=0.6)
        for bar, v, mark in zip(bars, values, marks):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {fmt(v)}{mark}",
                    va="center", fontsize=8, color=MUTED)
        xmax = max(values.max(), line[0] if line else 0) * 1.3
        if line is not None:
            ax.axvline(line[0], color=INK, linestyle="--", linewidth=1.2)
            ax.text(line[0], -0.75, f" {line[1]}", color=INK, fontsize=8, va="bottom")
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel, color=INK)
        _style_axis(ax)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=9, colors=INK)
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    axes[0].set_title("RMSE", fontweight="bold", color=INK, loc="left", fontsize=11)
    axes[1].set_title(f"MAPE (target \u2264 {mape_target:g}%)", fontweight="bold", color=INK,
                      loc="left", fontsize=11)
    axes[2].set_title(f"95% of errors (limit \u2264 {p95_limit:g}%)", fontweight="bold",
                      color=INK, loc="left", fontsize=11)
    axes[0].set_ylim(-0.9, len(d) - 0.4)

    handles = [plt.Rectangle((0, 0), 1, 1, color=GOOD_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=CRITICAL_COLOR)]
    fig.legend(handles, ["PASS: meets requirement", "MISS: misses requirement"],
               loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")
