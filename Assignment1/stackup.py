"""Stackup architecture sweep built with the stackup_utils functions.

Each stackup = 1 tier classifier + 3 tier regressors (low / medium / high), with
its own architectures defined in STACKUP_ARCHITECTURES (mirroring the 10 single
networks in nn_architectures.ARCHITECTURES). For every stackup:

  1. Tier regressors: sf.train_tier_regression_models(...)
  2. Tier classifier: sf.train_tier_classifier(...)
  3. Stackup:         sf.create_stackup_from_trained_models(...)
  4. Scored on a validation split of diamonds_train, both hard routing
     (sf predict_batch) and the probability-weighted price
     sum_k P(tier k | x) * regressor_k(x).

After the sweep, the best stackup is refit on all of diamonds_train with several
seeds (refit_stackup_ensemble) and the ensemble's averaged prices are scored too.

Training setup matches the step-5 single networks (nn_architectures.CONFIG): same
Adam learning rate and performance scheduling, batch size, epoch limit, early-stopping
patience, BatchNorm, LeakyReLU activation, He initialization, and the same fixed
output rescaling on the regressors. Like step 5, stackups are ranked by
validation MAPE and only the best one is scored on diamonds_test.
"""

import math
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split

import diamonds_utils as du
import stackup_utils as sf
from nn_architectures import CONFIG, RESULTS_DIR, set_seed, count_parameters, \
    TRAIN_COLOR, INK, MUTED
from price_tiers import TIER_NAMES

# --- EXPERIMENT HERE: classifier and tier-regressor architectures for each stackup ---
# Mirrors A01-A10 in nn_architectures.ARCHITECTURES. The same regressor layers are
# used for all three tiers; give a stackup different classifier/regressor layers to
# experiment.
STACKUP_ARCHITECTURES = {
    "S01_1x32":        {"classifier": [32],          "regressor": [32]},
    "S02_1x64":        {"classifier": [64],          "regressor": [64]},
    "S03_1x128":       {"classifier": [128],         "regressor": [128]},
    "S04_2_64-32":     {"classifier": [64, 32],      "regressor": [64, 32]},
    "S05_2_128-64":    {"classifier": [128, 64],     "regressor": [128, 64]},
    "S06_3_64-64-64":  {"classifier": [64, 64, 64],  "regressor": [64, 64, 64]},
    "S07_3_128-64-32": {"classifier": [128, 64, 32], "regressor": [128, 64, 32]},
    "S08_4_128-64-32-16":     {"classifier": [128, 64, 32, 16],
                               "regressor":  [128, 64, 32, 16]},
    "S09_4_256-128-64-32":    {"classifier": [256, 128, 64, 32],
                               "regressor":  [256, 128, 64, 32]},
    "S10_5_256-128-64-32-16": {"classifier": [256, 128, 64, 32, 16],
                               "regressor":  [256, 128, 64, 32, 16]},
}

# Classifier-only regularization (the step-5 regressors have no dropout)
CLASSIFIER_DROPOUT = 0.2


def stackup_configs(arch, config=CONFIG):
    """Build the sf tier_configs and classifier_config for one stackup.

    The shared training settings come from the step-5 CONFIG so the stackups and
    the single networks train identically.
    """
    shared = {k: config[k] for k in ('lr', 'epochs', 'batch_size', 'patience', 'batch_norm',
                                     'activation', 'negative_slope', 'init',
                                     'lr_factor', 'lr_patience', 'min_lr')}
    tier_configs = {tier: {'layers': list(arch['regressor']), 'output_scale': True, **shared}
                    for tier in TIER_NAMES}
    classifier_config = {'layers': list(arch['classifier']), 'dropout': CLASSIFIER_DROPOUT,
                         **shared}
    return tier_configs, classifier_config


def _train_size(n, val_size=0.2):
    """Rows left for training after the internal 80/20 split train_test_split makes."""
    return n - math.ceil(val_size * n)


def _check_batchnorm_batches(fit_df, thresholds, config):
    """BatchNorm cannot normalize a training batch of one sample; fail early with a clear message."""
    if not config['batch_norm']:
        return
    low_t, high_t = thresholds
    sizes = {"classifier": _train_size(len(fit_df))}
    for tier, d in zip(TIER_NAMES, du.split_data(fit_df, low_t, high_t)):
        sizes[f"{tier} regressor"] = _train_size(len(d))
    bad = {k: n for k, n in sizes.items() if n % config['batch_size'] == 1}
    if bad:
        raise ValueError(f"Training sizes {bad} leave a final batch of 1 with "
                         f"batch_size={config['batch_size']}; pick another batch size.")


class _PricePredictor:
    """Adapter giving a stackup prediction function the .predict(X) interface that
    du.predict / du.validate_model use for non-torch models."""
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn

    def predict(self, X):
        return self.predict_fn(X)


def predict_price_weighted(stackup, X):
    """Probability-weighted price: sum over tiers of P(tier | x) * that tier's regressor(x).

    Uses the stackup's own classifier, per-tier preprocessors and regressors. The
    probability columns follow stackup.tier_encoder.classes_ (the same LabelEncoder
    ordering sf.train_tier_classifier trained with).

    Args:
        stackup (sf.DiamondPriceStackup): Stackup from sf.create_stackup_from_trained_models
        X (pandas.DataFrame): Diamond feature rows

    Returns:
        numpy.ndarray: Predicted prices
    """
    _, tier_probs = stackup.predict_tier(X)
    price = np.zeros(len(X))
    for j, tier in enumerate(stackup.tier_encoder.classes_):
        X_tier = stackup.preprocessors[tier].transform(X)
        price += tier_probs[:, j] * du.predict(stackup.regression_models[tier], X_tier).ravel()
    return price


def _as_list(stacks):
    return stacks if isinstance(stacks, (list, tuple)) else [stacks]


def stackup_predictors(stacks):
    """{'weighted': ..., 'hard': ...} price predictors usable with du.validate_model.

    `stacks` is one stackup or a list of them (an ensemble); with several, each
    variant's price is the average of the members' prices.
    """
    stacks = _as_list(stacks)
    return {"weighted": _PricePredictor(
                lambda X: np.mean([predict_price_weighted(s, X) for s in stacks], axis=0)),
            "hard": _PricePredictor(
                lambda X: np.mean([s.predict_batch(X)[0] for s in stacks], axis=0))}


def predict_tier_ensemble(stacks, X):
    """Tier names and probabilities, averaging the classifier probabilities of all stacks."""
    stacks = _as_list(stacks)
    probs = np.mean([s.predict_tier(X)[1] for s in stacks], axis=0)
    return stacks[0].tier_encoder.inverse_transform(probs.argmax(axis=1)), probs


def evaluate_tier_classifier(stacks, X, tiers_true):
    """Accuracy, macro F1 and log loss of the stackup's (or ensemble's) classifier."""
    tier_pred, tier_probs = predict_tier_ensemble(stacks, X)
    classes = list(_as_list(stacks)[0].tier_encoder.classes_)
    return {"accuracy": accuracy_score(tiers_true, tier_pred),
            "macro_F1": f1_score(tiers_true, tier_pred, average="macro"),
            "log_loss": log_loss(tiers_true, tier_probs, labels=classes)}, tier_pred


def run_stackup_sweep(diamonds_train, stackups, numerical_features, categorical_features,
                      thresholds, config=CONFIG, results_dir=RESULTS_DIR):
    """Train and compare one full stackup per entry in `stackups`.

    diamonds_train is split once (config val_size / seed, as in step 5) into a fit
    set and a validation set shared by every stackup. Each stackup is trained on the
    fit set with the sf functions (which hold out their own 20% for early stopping),
    then scored on the validation set. diamonds_test is never used here.

    Returns:
        tuple: (comparison DataFrame sorted by weighted val MAPE,
                {name: stackup}, {name: classifier history dict},
                {name: {'classifier' | tier: per-epoch lr schedule up to the best epoch}})
    """
    low_t, high_t = thresholds
    fit_df, val_df = train_test_split(diamonds_train, test_size=config['val_size'],
                                      random_state=config['seed'])
    _check_batchnorm_batches(fit_df, thresholds, config)
    y_val = val_df['price'].to_numpy(dtype=float)
    metrics_path = os.path.join(results_dir, "metrics.csv")
    os.makedirs(results_dir, exist_ok=True)

    rows, stacks, clf_histories, schedules = [], {}, {}, {}
    for name, arch in stackups.items():
        tier_configs, classifier_config = stackup_configs(arch, config)
        print(f"\n{'=' * 70}\nStackup {name}: classifier {arch['classifier']}, "
              f"tier regressors {arch['regressor']}\n{'=' * 70}")
        set_seed(config['seed'])
        start = time.time()

        models, preprocessors, reg_histories = sf.train_tier_regression_models(
            fit_df, tier_configs, numerical_features, categorical_features,
            low_threshold=low_t, high_threshold=high_t,
        )
        classifier_model, classifier_preprocessor, clf_history = sf.train_tier_classifier(
            fit_df, classifier_config, numerical_features, categorical_features,
            low_threshold=low_t, high_threshold=high_t,
        )
        plt.close("all")  # figures drawn by du.evaluate_model / plot_training_history
        stack = sf.create_stackup_from_trained_models(
            classifier_model=classifier_model,
            regression_models=models,
            preprocessors=preprocessors,
            classifier_preprocessor=classifier_preprocessor,
            low_threshold=low_t,
            high_threshold=high_t,
        )
        train_time = time.time() - start

        notes = (f"clf={arch['classifier']}; reg={arch['regressor']}; "
                 f"bn={config['batch_norm']}; bs={config['batch_size']}")
        val = {}
        for mode, predictor in stackup_predictors(stack).items():
            table = du.validate_model(predictor, val_df, y_val, f"{name}_{mode}", split="val",
                                      table_path=metrics_path, notes=notes)
            val[mode] = table[(table.model == f"{name}_{mode}") & (table.split == "val")].iloc[-1]
        clf_val, _ = evaluate_tier_classifier(stack, val_df, val_df['price_tier'])

        h = clf_history.history
        rows.append({
            "model": name,
            "classifier_layers": str(arch['classifier']),
            "regressor_layers": str(arch['regressor']),
            "params": count_parameters(classifier_model)
                      + sum(count_parameters(m) for m in models.values()),
            "clf_epochs": len(h['loss']),
            "reg_epochs": "/".join(str(len(reg_histories[t].history['loss'])) for t in TIER_NAMES),
            "train_time_s": round(train_time, 1),
            "clf_val_accuracy": clf_val["accuracy"],
            "clf_val_log_loss": clf_val["log_loss"],
            "val_MAE": val["weighted"]["MAE"], "val_RMSE": val["weighted"]["RMSE"],
            "val_MAPE": val["weighted"]["MAPE"], "val_APE_P95": val["weighted"]["APE_P95"],
            "val_R2": val["weighted"]["R2"],
            "hard_val_RMSE": val["hard"]["RMSE"], "hard_val_MAPE": val["hard"]["MAPE"],
        })
        stacks[name], clf_histories[name] = stack, h
        schedules[name] = {"classifier": du.lr_schedule_to_best_epoch(clf_history),
                           **{t: du.lr_schedule_to_best_epoch(reg_histories[t])
                              for t in TIER_NAMES}}

    comparison = pd.DataFrame(rows).sort_values("val_MAPE").reset_index(drop=True)
    comparison.to_csv(os.path.join(results_dir, "stackup_comparison.csv"), index=False)
    return comparison, stacks, clf_histories, schedules


def refit_stackup_ensemble(arch, diamonds_train, schedules, numerical_features,
                           categorical_features, thresholds, config=CONFIG, n_members=None):
    """Refit a stackup on ALL of diamonds_train with several seeds (an ensemble).

    Each member retrains the classifier and the three tier regressors with the sf
    functions on every training row (no validation split). With no data left for
    early stopping, each network replays the per-epoch learning rates its selection
    run used up to its best epoch (`schedules`, from run_stackup_sweep). Member i
    uses seed config['seed'] + i.

    Returns:
        list[sf.DiamondPriceStackup]: The refit stackups (average them with
            stackup_predictors / predict_tier_ensemble)
    """
    n_members = n_members or config['ensemble_size']
    low_t, high_t = thresholds
    tier_configs, classifier_config = stackup_configs(arch, config)
    tier_configs = {t: {**c, 'lr_schedule': schedules[t]} for t, c in tier_configs.items()}
    classifier_config = {**classifier_config, 'lr_schedule': schedules['classifier']}

    stacks = []
    for i in range(n_members):
        print(f"\n--- Stackup refit member {i + 1}/{n_members} (seed {config['seed'] + i}): "
              f"classifier {len(schedules['classifier'])} epochs, regressors "
              f"{'/'.join(str(len(schedules[t])) for t in TIER_NAMES)} epochs ---")
        set_seed(config['seed'] + i)
        models, preprocessors, _ = sf.train_tier_regression_models(
            diamonds_train, tier_configs, numerical_features, categorical_features,
            low_threshold=low_t, high_threshold=high_t,
        )
        classifier_model, classifier_preprocessor, _ = sf.train_tier_classifier(
            diamonds_train, classifier_config, numerical_features, categorical_features,
            low_threshold=low_t, high_threshold=high_t,
        )
        stacks.append(sf.create_stackup_from_trained_models(
            classifier_model=classifier_model,
            regression_models=models,
            preprocessors=preprocessors,
            classifier_preprocessor=classifier_preprocessor,
            low_threshold=low_t,
            high_threshold=high_t,
        ))
    return stacks


def evaluate_stackup(name, stacks, diamonds_test, thresholds, results_dir=RESULTS_DIR):
    """Score a stackup (or an ensemble of stackups) on diamonds_test.

    Writes the weighted and hard price rows to results/metrics.csv. For a single
    stackup the sf evaluate_stackup report is printed as well.

    Returns:
        pandas.DataFrame: test MAE / RMSE / MAPE / APE_P95 / R2 per variant
    """
    stacks = _as_list(stacks)
    low_t, high_t = thresholds
    y_test = diamonds_test['price'].to_numpy(dtype=float)
    if len(stacks) == 1:
        stacks[0].evaluate_stackup(diamonds_test, y_test, low_threshold=low_t,
                                   high_threshold=high_t)
    tier_pred, _ = predict_tier_ensemble(stacks, diamonds_test)
    tier_accuracy = float(np.mean(tier_pred == diamonds_test['price_tier'].to_numpy()))

    rows = []
    for mode, predictor in stackup_predictors(stacks).items():
        label = f"{name}_{mode}"
        table = du.validate_model(predictor, diamonds_test, y_test, label, split="test",
                                  table_path=os.path.join(results_dir, "metrics.csv"),
                                  notes=f"best stackup by val MAPE ({mode}); "
                                        f"{len(stacks)} member(s)")
        row = table[(table.model == label) & (table.split == "test")].iloc[-1]
        rows.append({"variant": label,
                     **{k: row[k] for k in ("MAE", "RMSE", "MAPE", "APE_P95", "R2")}})
    summary = pd.DataFrame(rows)
    summary["tier_accuracy"] = tier_accuracy
    return summary


def plot_confusion_matrix(tiers_true, tiers_pred, path,
                          title="Tier classifier confusion matrix (test)"):
    """Row-normalized confusion matrix (share of each true tier), single-hue sequential."""
    cm = confusion_matrix(tiers_true, tiers_pred, labels=TIER_NAMES)
    share = cm / cm.sum(axis=1, keepdims=True)
    labels = [t.capitalize() for t in TIER_NAMES]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{100 * share[i, j]:.1f}%\n({cm[i, j]:,})", ha="center", va="center",
                    fontsize=9, color="white" if share[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted tier", color=INK)
    ax.set_ylabel("True tier", color=INK)
    ax.set_title(title, fontweight="bold", color=INK, loc="left", fontsize=11)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")
