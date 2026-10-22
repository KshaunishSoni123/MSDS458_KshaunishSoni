"""Assignment 2: DNN vs CNN architecture sweep on CIFAR-10 image classification (Keras).

Follows the Assignment 1 approach (Assignment1/nn_architectures.py): every network
shares the same training setup (CONFIG) and only the architecture changes, so the
differences in the results come from the architecture alone.

Ten experiments, defined in ARCHITECTURES:
    D01-D05  Dense networks (DNN): the 32x32x3 image is flattened to 3,072 inputs
             and passed through fully connected hidden layers.
    C01-C05  Convolutional networks (CNN): a stack of convolutional layers
             (Conv2D -> BatchNormalization -> ReLU -> MaxPooling2D), each with its
             own kernel size, number of filters, stride and pool size, followed by
             a fully connected head.

An architecture is a spec dict:
    {"type": "dnn", "hidden": [512, 256]}
    {"type": "cnn", "conv": [conv(64, kernel=3), conv(128, kernel=5, stride=2)],
     "dense": [128]}
conv(filters, kernel, stride, pool) describes one convolutional layer; pool=None
skips the max pooling after it. build_model(spec) turns any spec into a Keras
Sequential model, so new experiments only need a new entry in ARCHITECTURES.
Keras infers every layer's output shape from Input(shape=...), and the sweep prints
model.summary() for each network.

Hyperparameter tuning (TUNING): before its final training run, every architecture
runs the same random-search trials over learning rate, batch size, dropout, L2 and
activation, and trains with the combination that scores best on validation
accuracy. Trials are logged to results/tuning_results.csv.

Run the full 10-experiment sweep with:

    conda activate f1tf312
    cd "Assignment 2"
    python Assignment02_kshaunish_soni.py

add --no-tune to skip tuning and train every model with CONFIG as is, or run a
fast smoke test (subset of the data, 2 epochs, 2 short trials) with --quick.
"""

import gc
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

import matplotlib
if "ipykernel" not in sys.modules:   # headless when run as a script; inline in Colab/Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                                     Input, LeakyReLU, MaxPooling2D, ReLU)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.set_printoptions(precision=3, suppress=True)
print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU") or "none")

SEED = 42
QUICK = "--quick" in sys.argv
TUNE = "--no-tune" not in sys.argv
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                           *(["quick"] if QUICK else []))

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Shared training setup: identical for every architecture
CONFIG = {
    "lr": 1e-3,            # Adam learning rate
    "batch_size": 128,
    "epochs": 2 if QUICK else 30,   # maximum; early stopping usually ends sooner
    "patience": 5,         # epochs without val_loss improvement before stopping
    "batch_norm": True,    # BatchNormalization after every Dense / Conv2D, before the activation
    "activation": "relu",  # hidden activation: 'relu' or 'leaky_relu'
    "negative_slope": 0.01,  # LeakyReLU slope for x < 0
    "init": "he",          # He (Kaiming) normal init; None = Keras default (Glorot uniform)
    "dropout": 0.3,        # dropout after each dense hidden layer (0 disables)
    "l2": 0.0,             # L2 weight penalty on every Conv2D / hidden Dense kernel (0 disables)
    "lr_factor": 0.5,      # performance scheduling: halve the lr when val_loss plateaus...
    "lr_patience": 2,      # ...for this many epochs (kept below the early-stopping patience)
    "min_lr": 1e-6,
    "val_size": 0.1,       # fraction of the training set held out for validation
    "seed": SEED,
}

# Hyperparameter tuning: random search, run separately for every architecture.
# The same sampled trials are used for every model so each gets an equal budget.
# The best trial (highest validation accuracy) overrides CONFIG for that model's
# final training run. Disable with --no-tune.
TUNING = {
    "n_trials": 2 if QUICK else 8,   # combinations tried per architecture
    "epochs": 1 if QUICK else 10,    # shorter runs while searching
    "patience": 3,
    "space": {                       # any CONFIG key can be added here
        "lr": [3e-4, 1e-3, 3e-3],
        "batch_size": [64, 128, 256],
        "dropout": [0.0, 0.2, 0.3, 0.5],
        "l2": [0.0, 1e-4, 1e-3],
        "activation": ["relu", "leaky_relu"],
    },
}


def conv(filters, kernel=3, stride=1, pool=2):
    """One convolutional layer of a CNN spec.

    Args:
        filters (int): Number of filters (output channels)
        kernel (int): Square kernel size, e.g. 3 for 3x3 or 5 for 5x5
        stride (int): Convolution stride; padding is 'same', so stride 1 keeps the
            spatial size and stride 2 halves it
        pool (int | None): Square max-pooling window after the layer (2 = 2x2),
            None for no pooling
    """
    return {"filters": filters, "kernel": kernel, "stride": stride, "pool": pool}


# Architecture matrix: name -> spec
ARCHITECTURES = {
    # Dense networks: depth and width sweep on the flattened 3,072-pixel input
    "D01_1x128":                {"type": "dnn", "hidden": [128]},
    "D02_1x512":                {"type": "dnn", "hidden": [512]},
    "D03_2_512-256":            {"type": "dnn", "hidden": [512, 256]},
    "D04_3_1024-512-256":       {"type": "dnn", "hidden": [1024, 512, 256]},
    "D05_4_1024-512-256-128":   {"type": "dnn", "hidden": [1024, 512, 256, 128]},
    # Convolutional networks: number of conv layers, kernel size, filters and stride.
    # Every conv layer is followed by 2x2 max pooling; the dense head is the same for all.
    "C01_2conv_3x3_64":         {"type": "cnn", "conv": [conv(64, kernel=3)] * 2,
                                 "dense": [128]},
    "C02_3conv_3x3_64":         {"type": "cnn", "conv": [conv(64, kernel=3)] * 3,
                                 "dense": [128]},
    "C03_2conv_5x5_128":        {"type": "cnn", "conv": [conv(128, kernel=5)] * 2,
                                 "dense": [128]},
    "C04_3conv_5x5_128":        {"type": "cnn", "conv": [conv(128, kernel=5)] * 3,
                                 "dense": [128]},
    "C05_2conv_3x3_128_s2":     {"type": "cnn", "conv": [conv(128, kernel=3, stride=2)] * 2,
                                 "dense": [128]},
}

# Plot palette (same tokens as Assignment 1)
DNN_COLOR = "#2a78d6"
CNN_COLOR = "#eb6834"
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID_GRAY = "#e1e0d9"


def set_seed(seed=SEED):
    """Seed Python, NumPy and TensorFlow so that runs are repeatable."""
    keras.utils.set_random_seed(seed)


set_seed()
print(f"Random seed set to {SEED}")

# helper functions for plotting training history

def plot_history(history):
  h = history.history if hasattr(history, 'history') else history
  losses = h['loss']
  accs = h['accuracy']
  val_losses = h['val_loss']
  val_accs = h['val_accuracy']
  epochs = len(losses)

  plt.figure(figsize=(16, 4))
  for i, metrics in enumerate(zip([losses, accs], [val_losses, val_accs], ['Loss', 'Accuracy'])):
    plt.subplot(1, 2, i + 1)
    plt.plot(range(epochs), metrics[0], label='Training {}'.format(metrics[2]))
    plt.plot(range(epochs), metrics[1], label='Validation {}'.format(metrics[2]))
    plt.legend()
  plt.show()

def display_training_curves(training, validation, title, subplot):
  ax = plt.subplot(subplot)
  ax.plot(training)
  ax.plot(validation)
  ax.set_title('model '+ title)
  ax.set_ylabel(title)
  ax.set_xlabel('epoch')
  ax.legend(['training', 'validation'])

def print_validation_report(y_test, predictions):
    print("Classification Report")
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES))
    print('Accuracy Score: {}'.format(accuracy_score(y_test, predictions)))

def plot_confusion_matrix(y_true, y_pred, path=None, title=None):
    mtx = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.75, cbar=False, ax=ax, cmap='Blues',
                linecolor='white', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    if title:
        ax.set_title(title, fontweight="bold", color=INK, loc="left")
    if path:
        fig.tight_layout()
        fig.savefig(path, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"  saved {path}")


def predict_proba(model, x, batch_size=512):
    """Return class probabilities for images x (N, 32, 32, 3 numpy, scaled to [0, 1])."""
    return model.predict(x, batch_size=batch_size, verbose=0)


#load data

def load_cifar10():
    """Load CIFAR-10 as numpy arrays from the Hugging Face Hub; fall back to Keras.

    The Hub copy is CDN-backed and downloads in a few seconds. The Keras download
    (keras.datasets.cifar10) is a single host that can be slow from Colab, so it is
    kept as the fallback for when the Hub is unreachable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("uoft-cs/cifar10")

        def to_arrays(split):
            x = np.stack([np.array(img) for img in split["img"]]).astype(np.uint8)
            y = np.array(split["label"], dtype=np.uint8).reshape(-1, 1)
            return x, y

        return to_arrays(ds["train"]), to_arrays(ds["test"])
    except Exception as e:
        print(f"⚠️  Hugging Face Hub load failed ({type(e).__name__}: {e})")
        print("    Falling back to keras.datasets.cifar10 (this download can be slow)...")
        return keras.datasets.cifar10.load_data()   # (N, 32, 32, 3) uint8, (N, 1) labels


def to_model_input(x):
    """uint8 images (N, 32, 32, 3) -> float32 scaled to [0, 1] (Keras is channels-last)."""
    return x.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Generic network builder
# ---------------------------------------------------------------------------

def hidden_activation(name='relu', negative_slope=0.01, layer_name=None):
    """Hidden-layer activation layer: 'relu' or 'leaky_relu'."""
    if name == 'leaky_relu':
        return LeakyReLU(negative_slope=negative_slope, name=layer_name)
    if name == 'relu':
        return ReLU(name=layer_name)
    raise ValueError(f"hidden_activation(): unknown activation '{name}'")


def build_model(spec, name=None, input_shape=(32, 32, 3), n_classes=10, batch_norm=True,
                activation="relu", negative_slope=0.01, dropout=0.0, l2=0.0, init=None):
    """Build a DNN or CNN classifier from an architecture spec.

    Args:
        spec (dict): {"type": "dnn", "hidden": [widths]} or
            {"type": "cnn", "conv": [conv(...), ...], "dense": [widths]}
        name (str): Model name shown in model.summary()
        input_shape (tuple): (height, width, channels) of one image
        n_classes (int): Number of output classes (softmax output)
        batch_norm (bool): BatchNormalization between each Dense/Conv2D and its activation
        activation (str): Hidden activation, 'relu' or 'leaky_relu'
        negative_slope (float): LeakyReLU slope
        dropout (float): Dropout rate after each dense hidden layer
        l2 (float): L2 weight penalty on every Conv2D / hidden Dense kernel (0 disables)
        init (str): 'he' for He (Kaiming) normal initialization, None for Keras default

    Returns:
        keras.Sequential: The (uncompiled) model
    """
    kernel_init = "he_normal" if init == "he" else "glorot_uniform"
    kernel_reg = keras.regularizers.l2(l2) if l2 else None
    model = Sequential(name=name)
    model.add(Input(shape=input_shape))

    #First create the convolutional layers if the spec is for a CNN, otherwise create the dense layers for a DNN
    # dense layers and hidden DNN layers are handled in the same way after the convolutional layers, so we can just create a list of hidden layer widths to iterate over later
    if spec["type"] == "cnn":
        if not spec.get("conv"):
            raise ValueError("build_model(): a CNN spec needs at least one conv layer")
        for i, layer in enumerate(spec["conv"], start=1):
            model.add(Conv2D(layer["filters"], layer["kernel"], strides=layer["stride"],
                             padding="same", use_bias=not batch_norm,
                             kernel_initializer=kernel_init, kernel_regularizer=kernel_reg,
                             name=f"conv{i}"))
            if batch_norm:
                model.add(BatchNormalization(name=f"conv{i}_bn"))
            model.add(hidden_activation(activation, negative_slope, f"conv{i}_{activation}"))
            if layer.get("pool"):
                model.add(MaxPooling2D(layer["pool"], name=f"pool{i}"))
        hidden = spec.get("dense", [])
    elif spec["type"] == "dnn":
        if not spec.get("hidden"):
            raise ValueError("build_model(): a DNN spec needs at least one hidden layer")
        hidden = spec["hidden"]
    else:
        raise ValueError(f"build_model(): unknown architecture type '{spec['type']}'")

    model.add(Flatten(name="flatten"))
    for i, width in enumerate(hidden, start=1):
        model.add(Dense(width, use_bias=not batch_norm, kernel_initializer=kernel_init,
                        kernel_regularizer=kernel_reg, name=f"dense{i}"))
        if batch_norm:
            model.add(BatchNormalization(name=f"dense{i}_bn"))
        model.add(hidden_activation(activation, negative_slope, f"dense{i}_{activation}"))
        if dropout:
            model.add(Dropout(dropout, name=f"dense{i}_dropout"))
    model.add(Dense(n_classes, activation="softmax", name="output"))
    return model


def describe(spec):
    """Short human-readable summary of a spec for tables and logs."""
    if spec["type"] == "dnn":
        return f"dense {spec['hidden']}"
    convs = [f"conv {l['kernel']}x{l['kernel']}/{l['filters']}/s{l['stride']}"
             + (f" + pool {l['pool']}x{l['pool']}" if l.get("pool") else "")
             for l in spec["conv"]]
    return " -> ".join(convs + [f"dense {spec.get('dense', [])}"])


def layer_summary(model):
    """Plain-text version of model.summary(): every layer with its output shape and
    parameter count (Keras clips its own table when the output is captured to a file)."""
    lines = [f"Model: {model.name}", f"{'Layer (type)':<42}{'Output Shape':<24}{'Param #':>12}",
             "-" * 78]
    for layer in model.layers:
        label = f"{layer.name} ({type(layer).__name__})"
        lines.append(f"{label:<42}{str(tuple(layer.output.shape)):<24}"
                     f"{layer.count_params():>12,}")
    n_trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    lines += ["-" * 78, f"Total params: {model.count_params():,}",
              f"Trainable params: {n_trainable:,}",
              f"Non-trainable params: {model.count_params() - n_trainable:,}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fit_classifier(model, X_train, y_train, X_val, y_val, lr=1e-3, epochs=30, batch_size=128,
                   patience=5, lr_factor=None, lr_patience=2, min_lr=1e-6, verbose=True):
    """Compile and fit with Adam + sparse categorical cross-entropy, early stopping with
    best-weight restore and optional ReduceLROnPlateau performance scheduling
    (same setup as Assignment 1)."""
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [EarlyStopping(monitor="val_loss", patience=patience,
                               restore_best_weights=True, verbose=int(verbose))]
    if lr_factor is not None:
        callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=lr_factor,
                                           patience=lr_patience, min_lr=min_lr,
                                           verbose=int(verbose)))
    return model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
                     batch_size=batch_size, callbacks=callbacks, verbose=2 if verbose else 0)


def predict_classes(model, x):
    return predict_proba(model, x).argmax(axis=1)


def evaluate_split(model, x, y):
    """Accuracy and cross-entropy loss on one data split.

    The loss is plain cross-entropy, so it is comparable across models; Keras's own
    training loss also adds the L2 penalty, which differs per model after tuning.
    """
    proba = predict_proba(model, x)
    return (accuracy_score(y, proba.argmax(axis=1)),
            log_loss(y, proba, labels=list(range(len(CLASS_NAMES)))))


def train_model(spec, name, config, X_train, y_train, X_val, y_val, epochs=None,
                patience=None, verbose=True):
    """Build `spec` with the settings in `config` and fit it. Returns (model, history dict)."""
    set_seed(config["seed"])
    model = build_model(spec, name=name, batch_norm=config["batch_norm"],
                        activation=config["activation"],
                        negative_slope=config["negative_slope"],
                        dropout=config["dropout"], l2=config["l2"], init=config["init"])
    history = fit_classifier(model, X_train, y_train, X_val, y_val,
                             lr=config["lr"], epochs=epochs or config["epochs"],
                             batch_size=config["batch_size"],
                             patience=patience or config["patience"],
                             lr_factor=config["lr_factor"],
                             lr_patience=config["lr_patience"], min_lr=config["min_lr"],
                             verbose=verbose)
    return model, history.history


def sample_trials(space, n_trials, seed=SEED):
    """Draw n_trials distinct combinations from the search space (random search)."""
    keys = list(space)
    combos = list(itertools.product(*(space[k] for k in keys)))
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(combos), size=min(n_trials, len(combos)), replace=False)
    return [dict(zip(keys, combos[i])) for i in picks]


def tune_architecture(name, spec, trials, X_train, y_train, X_val, y_val, config=CONFIG,
                      tuning=TUNING):
    """Random search for one architecture: train each trial briefly and score it on the
    validation set. Returns (best trial params, list of per-trial result rows)."""
    rows = []
    for t, params in enumerate(trials, start=1):
        start = time.time()
        model, h = train_model(spec, name, {**config, **params}, X_train, y_train,
                               X_val, y_val, epochs=tuning["epochs"],
                               patience=tuning["patience"], verbose=False)
        val_acc = accuracy_score(y_val, predict_classes(model, X_val))
        rows.append({"model": name, "trial": t, **params, "val_acc": val_acc,
                     "val_loss": min(h["val_loss"]), "epochs_run": len(h["loss"]),
                     "time_s": round(time.time() - start, 1)})
        print(f"  trial {t}/{len(trials)} {params} -> val_acc {val_acc:.4f}")
        del model
        gc.collect()
    best = max(rows, key=lambda r: r["val_acc"])
    return {k: best[k] for k in trials[0]}, rows


def run_architecture_sweep(architectures, X_train, y_train, X_val, y_val, X_test, y_test,
                           config=CONFIG, tuning=TUNING, tune=TUNE, results_dir=None,
                           verbose=True):
    """Tune, train and compare one network per architecture.

    With tune=True each architecture first runs the random search in `tuning` (the
    same sampled trials for every model); its best hyperparameters then override
    `config` for a full-length training run. With tune=False every architecture
    trains with `config` as is. All models see the same train/validation split and
    seed. Selection is by validation accuracy; test accuracy is recorded for the report.

    Returns:
        tuple: (comparison DataFrame sorted by val accuracy, {name: history dict},
                {name: model}, tuning DataFrame of every trial (empty if tune=False))
    """
    results_dir = results_dir or RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    trials = sample_trials(tuning["space"], tuning["n_trials"], config["seed"]) if tune else []
    if tune:
        print(f"Tuning {len(trials)} trials per architecture, {tuning['epochs']} epochs each")
    rows, histories, models, summaries, tuning_rows = [], {}, {}, [], []
    for name, spec in architectures.items():
        print(f"\n--- {name}: {describe(spec)} ---")
        best_params, tuning_time = {}, 0.0
        if tune:
            best_params, trial_rows = tune_architecture(name, spec, trials, X_train, y_train,
                                                        X_val, y_val, config, tuning)
            tuning_rows += trial_rows
            tuning_time = sum(r["time_s"] for r in trial_rows)
            print(f"  best hyperparameters: {best_params}")
        cfg = {**config, **best_params}

        start = time.time()
        model, h = train_model(spec, name, cfg, X_train, y_train, X_val, y_val,
                               verbose=verbose)
        train_time = time.time() - start
        n_params = model.count_params()
        model.summary()
        summaries.append(layer_summary(model) + "\n")

        rows.append({
            "model": name,
            "type": spec["type"].upper(),
            "architecture": describe(spec),
            "params": n_params,
            **{k: cfg[k] for k in tuning["space"]},
            "epochs_run": len(h["loss"]),
            "best_epoch": int(np.argmin(h["val_loss"])) + 1,
            "tuning_time_s": round(tuning_time, 1),
            "train_time_s": round(train_time, 1),
            "total_time_s": round(tuning_time + train_time, 1),
        })
        for split, (x, y) in {"train": (X_train, y_train), "val": (X_val, y_val),
                              "test": (X_test, y_test)}.items():
            rows[-1][f"{split}_acc"], rows[-1][f"{split}_loss"] = evaluate_split(model, x, y)
        print(f"  train_acc {rows[-1]['train_acc']:.4f}  val_acc {rows[-1]['val_acc']:.4f}"
              f"  test_acc {rows[-1]['test_acc']:.4f}  ({train_time:.0f}s)")
        histories[name] = h
        models[name] = model

    comparison = pd.DataFrame(rows).sort_values("val_acc", ascending=False).reset_index(drop=True)
    comparison.to_csv(os.path.join(results_dir, "architecture_comparison.csv"), index=False)
    with open(os.path.join(results_dir, "architecture_summaries.txt"), "w") as f:
        f.write("\n".join(summaries))
    pd.DataFrame([{"model": n, "epoch": e + 1, **{k: v[e] for k, v in h.items()}}
                  for n, h in histories.items() for e in range(len(h["loss"]))]
                 ).to_csv(os.path.join(results_dir, "architecture_histories.csv"), index=False)
    tuning_results = pd.DataFrame(tuning_rows)
    if tune:
        tuning_results.to_csv(os.path.join(results_dir, "tuning_results.csv"), index=False)
    return comparison, histories, models, tuning_results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID_GRAY, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_GRAY)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_learning_curves(histories, path, metric="accuracy", ylabel="Accuracy",
                         title="Learning curves by architecture"):
    """Small multiples: train vs validation `metric` per architecture (DNNs top, CNNs bottom)."""
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
        ax.set_title(name, fontsize=9, color=INK)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if metric == "accuracy":
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
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


def plot_comparison(comparison, path, title="Accuracy by architecture: DNN vs CNN"):
    """Horizontal bars of validation accuracy, colored by network type, best on top."""
    df = comparison.sort_values("val_acc", ascending=True)
    colors = [DNN_COLOR if t == "DNN" else CNN_COLOR for t in df["type"]]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(df) + 1.4))
    bars = ax.barh(df["model"], df["val_acc"], color=colors, height=0.6)
    for bar, va, te, p in zip(bars, df["val_acc"], df["test_acc"], df["params"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  val {va:.1%} | test {te:.1%} | {p:,} params", va="center",
                fontsize=8, color=MUTED)
    ax.set_xlabel("Validation accuracy", color=INK)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, min(1.0, df["val_acc"].max() * 1.45))
    ax.set_title(title, fontweight="bold", color=INK, loc="left")
    _style_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9, colors=INK)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=DNN_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=CNN_COLOR)], ["DNN", "CNN"],
              loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


def plot_tuning(tuning_results, path, title="Hyperparameter search: validation accuracy per trial"):
    """One row per architecture: every trial as a gray dot, the selected trial colored."""
    names = list(dict.fromkeys(tuning_results["model"]))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(names) + 1.4))
    for y, name in enumerate(names):
        d = tuning_results[tuning_results["model"] == name]
        best = d.loc[d["val_acc"].idxmax()]
        color = CNN_COLOR if name.startswith("C") else DNN_COLOR
        ax.scatter(d["val_acc"], [y] * len(d), color=GRID_GRAY, edgecolor=MUTED,
                   linewidth=0.5, s=36, zorder=2)
        ax.scatter([best["val_acc"]], [y], color=color, s=70, zorder=3)
        ax.text(d["val_acc"].max(), y, f"  {best['val_acc']:.1%}", va="center",
                fontsize=8, color=MUTED)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("Validation accuracy (short tuning runs)", color=INK)
    ax.set_title(title, fontweight="bold", color=INK, loc="left")
    _style_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9, colors=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Final deliverables
# ---------------------------------------------------------------------------

RESULTS_TABLE_COLUMNS = {
    "model": "Model", "type": "Type", "params": "Params",
    "train_acc": "Train acc", "train_loss": "Train loss",
    "val_acc": "Val acc", "val_loss": "Val loss",
    "test_acc": "Test acc", "test_loss": "Test loss",
    "tuning_time_s": "Tuning (s)", "train_time_s": "Training (s)", "total_time_s": "Total (s)",
}


def save_results_table(comparison, path_stem, order=None):
    """Write the final results table (accuracy and loss on train / validation / test plus
    processing time for every model) as CSV and as a PNG image.

    The best value in each accuracy column (highest) and loss column (lowest) is bold.
    """
    df = comparison.copy()
    if order is not None:
        df = df.set_index("model").loc[[m for m in order if m in set(df["model"])]].reset_index()
    table = df[list(RESULTS_TABLE_COLUMNS)].rename(columns=RESULTS_TABLE_COLUMNS)
    table.round(4).to_csv(f"{path_stem}.csv", index=False)

    cells = table.copy()
    for col in cells:
        if col.endswith("acc"):
            cells[col] = table[col].map(lambda v: f"{v:.2%}")
        elif col.endswith("loss"):
            cells[col] = table[col].map(lambda v: f"{v:.4f}")
        elif col == "Params":
            cells[col] = table[col].map(lambda v: f"{v:,}")
        elif col.endswith("(s)"):
            cells[col] = table[col].map(lambda v: f"{v:,.1f}")

    fig, ax = plt.subplots(figsize=(11, 0.3 * len(cells) + 0.6))
    ax.axis("off")
    tbl = ax.table(cellText=cells.values, colLabels=cells.columns, loc="upper left",
                   cellLoc="right", colLoc="right")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    tbl.auto_set_column_width(list(range(len(cells.columns))))
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_GRAY)
        if r == 0:
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_facecolor("#f4f3ee")
        if c <= 1:
            cell.set_text_props(ha="left")
            cell._loc = "left"
    for c, col in enumerate(table.columns):
        if col.endswith("acc") or col.endswith("loss"):
            best = table[col].idxmax() if col.endswith("acc") else table[col].idxmin()
            tbl[best + 1, c].set_text_props(fontweight="bold")
    ax.set_title("Final results: accuracy, loss and processing time by model "
                 "(best per column in bold)", fontweight="bold", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(f"{path_stem}.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path_stem}.csv and {path_stem}.png")


def plot_feature_maps(model, images, labels, path, n_pools=2, n_filters=2):
    """Grid of feature maps: the outputs of `n_filters` filters from each of the first
    `n_pools` MaxPooling2D layers, one row per input image.

    For each pooling layer the filters shown are the ones with the highest mean
    activation over the sample images (the most active channels), so the panels are
    not blank channels that the ReLU switched off.
    """
    pools = [layer for layer in model.layers if isinstance(layer, MaxPooling2D)][:n_pools]
    if len(pools) < n_pools:
        print(f"  {model.name}: only {len(pools)} max-pooling layer(s), skipping feature maps")
        return
    extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in pools])
    maps = extractor.predict(images, verbose=0)
    if not isinstance(maps, (list, tuple)):
        maps = [maps]
    chosen = [np.argsort(m.mean(axis=(0, 1, 2)))[::-1][:n_filters] for m in maps]

    ncols = 1 + n_pools * n_filters
    fig, axes = plt.subplots(len(images), ncols, figsize=(2.1 * ncols, 2.2 * len(images) + 0.6),
                             squeeze=False)
    for i, (image, label) in enumerate(zip(images, labels)):
        axes[i, 0].imshow(image)
        axes[i, 0].set_ylabel(CLASS_NAMES[label], color=INK, fontsize=10)
        col = 1
        for layer, m, filters in zip(pools, maps, chosen):
            for f in filters:
                axes[i, col].imshow(m[i, :, :, f], cmap="viridis")
                if i == 0:
                    h, w = m.shape[1:3]
                    axes[i, col].set_title(f"{layer.name}, filter {f}\n({h}x{w})",
                                           fontsize=9, color=INK)
                col += 1
    axes[0, 0].set_title("Input", fontsize=9, color=INK)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID_GRAY)
    fig.suptitle(f"{model.name}: max-pooling feature maps", fontweight="bold", color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print('train_images:\t{}'.format(x_train.shape))
    print('train_labels:\t{}'.format(y_train.shape))
    print('test_images:\t\t{}'.format(x_test.shape))
    print('test_labels:\t\t{}'.format(y_test.shape))

    #data preprocessing
    y_train, y_test = y_train.ravel().astype(np.int64), y_test.ravel().astype(np.int64)
    if QUICK:   # small stratified subset for a fast end-to-end check
        x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size=5000,
                                                  stratify=y_train, random_state=SEED)
        x_test, y_test = x_test[:2000], y_test[:2000]

    x_train_split, x_valid_split, y_train_split, y_valid_split = train_test_split(
        x_train, y_train, test_size=CONFIG["val_size"], random_state=SEED,
        shuffle=True, stratify=y_train)

    x_train_norm = to_model_input(x_train_split)
    x_valid_norm = to_model_input(x_valid_split)
    x_test_norm = to_model_input(x_test)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"train {x_train_norm.shape} | valid {x_valid_norm.shape} | test {x_test_norm.shape}")

    print("Shared config:", CONFIG)
    if TUNE:
        print("Tuning:", TUNING)
    comparison, histories, models, tuning_results = run_architecture_sweep(
        ARCHITECTURES, x_train_norm, y_train_split, x_valid_norm, y_valid_split,
        x_test_norm, y_test)

    plot_learning_curves(histories, os.path.join(RESULTS_DIR, "architecture_learning_curves.png"))
    plot_learning_curves(histories, os.path.join(RESULTS_DIR, "architecture_loss_curves.png"),
                         metric="loss", ylabel="Cross-entropy loss",
                         title="Loss curves by architecture")
    plot_comparison(comparison, os.path.join(RESULTS_DIR, "architecture_val_accuracy.png"))
    if TUNE:
        plot_tuning(tuning_results, os.path.join(RESULTS_DIR, "tuning_val_accuracy.png"))

    pd.set_option("display.width", 200)
    cols = ["model", "type", "params", *TUNING["space"], "epochs_run", "best_epoch",
            "total_time_s", "train_acc", "val_acc", "test_acc"]
    print("\nArchitecture comparison (sorted by validation accuracy):")
    print(comparison[cols].round(4).to_string(index=False))

    # Deliverable 1: accuracy / loss on every split plus processing time, in experiment order
    save_results_table(comparison, os.path.join(RESULTS_DIR, "final_results_table"),
                       order=list(ARCHITECTURES))

    # Deliverable 2: feature maps of 2 filters from the 2 max-pooling layers of every CNN,
    # on the first test image of each of a few classes
    sample_idx = [int(np.flatnonzero(y_test == CLASS_NAMES.index(c))[0])
                  for c in ("airplane", "cat", "horse", "ship")]
    fmap_dir = os.path.join(RESULTS_DIR, "feature_maps")
    os.makedirs(fmap_dir, exist_ok=True)
    for name, spec in ARCHITECTURES.items():
        if spec["type"] == "cnn":
            plot_feature_maps(models[name], x_test_norm[sample_idx], y_test[sample_idx],
                              os.path.join(fmap_dir, f"{name}_feature_maps.png"))

    # Best DNN and best CNN on the test set
    for net_type in ("DNN", "CNN"):
        best = comparison[comparison["type"] == net_type].iloc[0]["model"]
        y_pred = predict_classes(models[best], x_test_norm)
        print(f"\nBest {net_type}: {best}")
        print_validation_report(y_test, y_pred)
        plot_confusion_matrix(y_test, y_pred,
                              path=os.path.join(RESULTS_DIR, f"best_{net_type.lower()}_confusion_matrix.png"),
                              title=f"Test confusion matrix: {best}")


if __name__ == "__main__":
    main()
