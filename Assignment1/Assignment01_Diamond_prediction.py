"""MSDS 458 Assignment 1: diamond price prediction with regression neural networks.

Pipeline:
  1. Environment checks, device and seed
  2. EDA on the raw dataset                      (eda_diamonds.run_eda)
  3. Feature engineering                         (feature_engineering.build_features)
  4. Train/test split and scaling                (diamonds_utils.prepare_data)
  5. Architecture sweep, identical setup per net (nn_architectures.run_architecture_sweep)
  6. Best architecture refit on all training data as a seed ensemble, scored on the test set
  7. Balanced low/medium/high price tiers        (price_tiers.build_price_tiers) for the stackup
  8. Stackup architecture sweep, same setup as step 5 (stackup.run_stackup_sweep, stackup_utils)
  9. Best stackup refit on all training data as a seed ensemble, scored on the test set and
     compared with the best single network

Run with:

    SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())") python3 Assignment01_Diamond_prediction.py

Options: --skip-eda to skip step 2, --skip-sweep to skip steps 5-6, --quick for a
fast smoke test (2 architectures, 3 epochs, written to results/quick/).
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from packaging import version

# stackup_utils imports `Assignment1.diamonds_utils`, so the folder above this one must be
# importable too
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diamonds_utils as du
import stackup_utils as sf
from eda_diamonds import run_eda
from feature_engineering import (build_features, summarize_features, NUMERICAL_FEATURES,
                                 CATEGORICAL_FEATURES, TARGET)
from price_tiers import build_price_tiers, plot_tier_distribution
import stackup
from nn_architectures import (CONFIG, ARCHITECTURES, RESULTS_DIR, set_seed,
                              run_architecture_sweep, plot_learning_curves, plot_comparison,
                              plot_target_comparison, refit_ensemble)

# diamonds_utils plotting helpers call plt.show(); under Agg the figures are saved instead
warnings.filterwarnings("ignore", message=".*non-interactive.*")

MIN_VERSIONS = {"Python": "3.10", "PyTorch": "2.0"}


def section(title):
    print("\n" + "#" * 70)
    print(f"# {title}")
    print("#" * 70)


def check_version(name, current, minimum):
    if version.parse(current) < version.parse(minimum):
        print(f"⚠️ {name} version {current} < required {minimum}. Please upgrade.")


def save_current_figure(path):
    """Save the figure a diamonds_utils plotting helper just drew, then close it."""
    plt.gcf().savefig(path, dpi=150, facecolor="white")
    plt.close("all")
    print(f"  saved {path}")


def setup_environment(seed):
    python_version = sys.version.split()[0]
    torch_version = torch.__version__.split("+")[0]
    print("Python version:", python_version)
    print("PyTorch version:", torch_version)
    check_version("Python", python_version, MIN_VERSIONS["Python"])
    check_version("PyTorch", torch_version, MIN_VERSIONS["PyTorch"])
    print("Using device:", du.get_device())
    set_seed(seed)
    print(f"Random seed set to {seed}")


def score_on_test(name, model, X_test, y_test, results_dir, notes):
    """Score a model on the test set (row written to metrics.csv) and return its metrics."""
    table = du.validate_model(model, X_test, np.asarray(y_test, dtype=float), name, split="test",
                              table_path=os.path.join(results_dir, "metrics.csv"), notes=notes)
    row = table[(table.model == name) & (table.split == "test")].iloc[-1]
    return {"variant": name, **{k: row[k] for k in ("MAE", "RMSE", "MAPE", "APE_P95", "R2")}}


def evaluate_best_model(name, model, history, X_test, y_test, results_dir):
    """Score the final model on the held-out test set and save its diagnostic plots.

    `history` is the selection run's training history (plotted for reference).

    Returns:
        dict: variant name and test MAE / RMSE / MAPE / APE_P95 / R2
    """
    y_true = np.asarray(y_test, dtype=float)
    result = score_on_test(name, model, X_test, y_true, results_dir,
                           notes="best architecture by val MAPE, refit ensemble")

    du.plot_training_history(history)
    save_current_figure(os.path.join(results_dir, "best_training_history.png"))

    y_pred = du.evaluate_model(model, X_test, y_true)
    save_current_figure(os.path.join(results_dir, "best_actual_vs_predicted.png"))

    du.residual_plot(y_true, y_pred)
    save_current_figure(os.path.join(results_dir, "best_residuals.png"))

    du.plot_percentage_error_by_price(y_true, y_pred)
    save_current_figure(os.path.join(results_dir, "best_percentage_error_by_price.png"))
    return result


def main(skip_eda=False, skip_sweep=False, quick=False):
    config, architectures, results_dir = CONFIG, ARCHITECTURES, RESULTS_DIR
    stackups = stackup.STACKUP_ARCHITECTURES
    if quick:
        config = {**CONFIG, "epochs": 3, "ensemble_size": 2}
        names = list(ARCHITECTURES)
        architectures = {k: ARCHITECTURES[k] for k in (names[0], names[-1])}
        names = list(stackups)
        stackups = {k: stackups[k] for k in (names[0], names[-1])}
        results_dir = os.path.join(RESULTS_DIR, "quick")
    os.makedirs(results_dir, exist_ok=True)

    section("1. Environment")
    setup_environment(config["seed"])

    diamonds = du.load_data()
    if not skip_eda:
        section("2. Exploratory data analysis")
        run_eda(diamonds)

    section("3. Feature engineering")
    df = build_features(diamonds)
    summarize_features(df)

    section("4. Train/test split and scaling")
    X_train, X_test, y_train, y_test, _ = du.prepare_data(
        df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, target=TARGET)

    single_tests = []
    if not skip_sweep:
        section("5. Architecture sweep")
        print("Shared config:", config)
        comparison, histories, models = run_architecture_sweep(
            architectures, X_train, y_train, config=config, results_dir=results_dir)
        plot_learning_curves(histories, os.path.join(results_dir, "architecture_learning_curves.png"))
        plot_comparison(comparison, os.path.join(results_dir, "architecture_val_mape.png"))
        plot_target_comparison(comparison, os.path.join(results_dir, "regression_targets.png"),
                               title="Single regression networks vs accuracy targets (validation)")

        pd.set_option("display.width", 200)
        cols = ["model", "params", "epochs_run", "best_epoch", "train_time_s",
                "train_RMSE", "val_MAE", "val_RMSE", "val_MAPE", "val_APE_P95", "val_R2"]
        print("\nArchitecture comparison (sorted by validation MAPE):")
        print(comparison[cols].round(3).to_string(index=False))

        best = comparison.iloc[0]["model"]
        section(f"6. Best architecture {best}: refit on all training data + "
                f"{config['ensemble_size']}-seed ensemble, test-set evaluation")
        single_tests.append(score_on_test(best, models[best], X_test, y_test, results_dir,
                                          notes="best architecture by val MAPE, sweep model"))
        ensemble = refit_ensemble(architectures[best], X_train, y_train, histories[best],
                                  config=config)
        ens_name = f"{best}_refit_ens{config['ensemble_size']}"
        single_tests.append(evaluate_best_model(ens_name, ensemble, histories[best],
                                                X_test, y_test, results_dir))

    section("7. Price tiers for the stackup models")
    train_df, test_df, thresholds = build_price_tiers(df)
    plot_tier_distribution(train_df, thresholds, os.path.join(results_dir, "price_tiers.png"))

    section("8. Stackup architecture sweep (stackup_utils)")
    # 80/20 split from step 7; diamonds_test is kept completely separate until the best
    # stackup (chosen on a validation split of diamonds_train) is evaluated
    diamonds_train, diamonds_test = train_df, test_df
    print(f"diamonds_train: {len(diamonds_train):,} rows | diamonds_test: {len(diamonds_test):,} rows "
          f"({100 * len(diamonds_test) / len(df):.0f}% test)")
    print("Shared config (same as step 5):", config)

    stack_comparison, stacks, clf_histories, schedules = stackup.run_stackup_sweep(
        diamonds_train, stackups, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, thresholds,
        config=config, results_dir=results_dir)
    print("\n✅ IMPORTANT: diamonds_test is kept completely separate for stackup evaluation")
    plot_comparison(stack_comparison, os.path.join(results_dir, "stackup_val_mape.png"),
                    title="Stackup validation MAPE (probability-weighted price)")
    plot_target_comparison(stack_comparison, os.path.join(results_dir, "stackup_targets.png"),
                           title="Stackup models vs accuracy targets "
                                 "(validation, probability-weighted price)")
    plot_learning_curves(clf_histories,
                         os.path.join(results_dir, "stackup_classifier_learning_curves.png"),
                         metric="loss", ylabel="Cross-entropy loss", dollars=False,
                         log_scale=False, title="Stackup tier classifier learning curves")
    stack_cols = ["model", "params", "clf_epochs", "reg_epochs", "train_time_s",
                  "clf_val_accuracy", "val_MAE", "val_RMSE", "val_MAPE", "val_APE_P95", "val_R2",
                  "hard_val_RMSE", "hard_val_MAPE"]
    print("\nStackup comparison (sorted by validation MAPE of the weighted price):")
    print(stack_comparison[stack_cols].round(3).to_string(index=False))

    best_stack = stack_comparison.iloc[0]["model"]
    section(f"9. Best stackup {best_stack}: refit on all training data + "
            f"{config['ensemble_size']}-seed ensemble, test-set evaluation")
    sweep_stack_test = stackup.evaluate_stackup(best_stack, stacks[best_stack], diamonds_test,
                                                thresholds, results_dir=results_dir)
    stack_ensemble = stackup.refit_stackup_ensemble(
        stackups[best_stack], diamonds_train, schedules[best_stack],
        NUMERICAL_FEATURES, CATEGORICAL_FEATURES, thresholds, config=config)
    plt.close("all")
    stack_ens_name = f"{best_stack}_refit_ens{config['ensemble_size']}"
    clf_test, tier_pred = stackup.evaluate_tier_classifier(
        stack_ensemble, diamonds_test, diamonds_test["price_tier"])
    print(f"\nEnsemble tier classifier on diamonds_test: accuracy {clf_test['accuracy']:.4f}, "
          f"macro F1 {clf_test['macro_F1']:.4f}, log loss {clf_test['log_loss']:.4f}")
    stackup.plot_confusion_matrix(diamonds_test["price_tier"], tier_pred,
                                  os.path.join(results_dir, "stackup_confusion_matrix.png"),
                                  title="Stackup ensemble tier classifier (test)")
    ens_stack_test = stackup.evaluate_stackup(stack_ens_name, stack_ensemble, diamonds_test,
                                              thresholds, results_dir=results_dir)
    stack_test = pd.concat([sweep_stack_test, ens_stack_test], ignore_index=True)
    stack_test.to_csv(os.path.join(results_dir, "stackup_test_summary.csv"), index=False)

    final = stack_test.drop(columns="tier_accuracy")
    if single_tests:
        final = pd.concat([pd.DataFrame(single_tests), final], ignore_index=True)
    final.to_csv(os.path.join(results_dir, "final_test_comparison.csv"), index=False)
    plot_target_comparison(final, os.path.join(results_dir, "final_test_targets.png"),
                           title="Best single network vs best stackup: accuracy targets (test)",
                           label_col="variant", prefix="")
    print("\nFinal comparison on the same diamonds_test rows (USD):")
    print(final.round(3).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-eda", action="store_true", help="skip the EDA step")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="skip the architecture sweep and its test evaluation")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 2 architectures, 3 epochs, results/quick/")
    args = parser.parse_args()
    main(skip_eda=args.skip_eda, skip_sweep=args.skip_sweep, quick=args.quick)
