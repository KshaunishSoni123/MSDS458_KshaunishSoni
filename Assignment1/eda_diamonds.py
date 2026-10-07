"""Initial EDA on the diamonds dataset.

Prints summary statistics for the raw dataset and saves a set of exploratory
plots to eda_output/. Called from Assignment01_Diamond_prediction.py via run_eda().
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from diamonds_utils import load_data, remove_specific_outliers
from feature_engineering import CUT_ORDER, COLOR_ORDER, CLARITY_ORDER

OUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")
os.makedirs(OUT_DIR, exist_ok=True)

# --- palette (validated categorical/sequential/diverging slots) ---
BLUE = "#2a78d6"
RED = "#e34948"
NEUTRAL_GRAY = "#f0efec"
GRID_GRAY = "#e1e0d9"
INK = "#0b0b0b"
MUTED = "#898781"

sns.set_style("whitegrid", {"grid.color": GRID_GRAY, "axes.edgecolor": MUTED})
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red", [BLUE, NEUTRAL_GRAY, RED]
)


def sequential_blues(n, lo=0.35, hi=0.9):
    """n evenly spaced blues, light (low rank) -> dark (high rank)."""
    cmap = plt.get_cmap("Blues")
    return [cmap(x) for x in np.linspace(lo, hi, n)]


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor="white")
    plt.close()
    print(f"  saved {path}")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_eda(diamonds=None):
    """Run the EDA on the raw dataset (loaded if not given) and save figures to eda_output/."""
    if diamonds is None:
        diamonds = load_data()

    # ---------------------------------------------------------------
    section("1. Dataset overview")
    print(f"Shape: {diamonds.shape[0]:,} rows x {diamonds.shape[1]} columns")
    print("\nDtypes:")
    print(diamonds.dtypes)
    print("\nMissing values per column:")
    print(diamonds.isna().sum())
    print("\nFirst 5 rows:")
    print(diamonds.head())
    print("\nSummary statistics (numeric columns):")
    print(diamonds.describe().round(2))
    print(f"\nDuplicate rows: {diamonds.duplicated().sum():,}")

    # ---------------------------------------------------------------
    section("2. Categorical variables: cut, color, clarity")
    cut_order, color_order, clarity_order = CUT_ORDER, COLOR_ORDER, CLARITY_ORDER  # worst -> best

    cat_specs = [("cut", cut_order), ("color", color_order), ("clarity", clarity_order)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, order) in zip(axes, cat_specs):
        counts = diamonds[col].value_counts().reindex(order)
        colors = sequential_blues(len(order))
        ax.bar(order, counts.values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{col.capitalize()} distribution")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        print(f"\n{col.upper()} value counts:")
        print(counts)
    savefig("01_categorical_distributions.png")

    # ---------------------------------------------------------------
    section("3. Price distribution")
    print(diamonds["price"].describe().round(2))
    plt.figure(figsize=(10, 5))
    sns.histplot(data=diamonds, x="price", bins=50, kde=True, color=BLUE, edgecolor="white")
    plt.title("Distribution of diamond prices")
    plt.xlabel("Price (USD)")
    plt.ylabel("Count")
    savefig("02_price_distribution.png")

    # ---------------------------------------------------------------
    section("4. Carat vs. price (by cut)")
    corr_carat_price = diamonds["carat"].corr(diamonds["price"])
    print(f"Correlation (carat, price): {corr_carat_price:.3f}")

    plt.figure(figsize=(9, 6))
    colors = sequential_blues(len(cut_order))
    for color, cut in zip(colors, cut_order):
        subset = diamonds[diamonds["cut"] == cut]
        plt.scatter(subset["carat"], subset["price"], s=10, alpha=0.4,
                    color=color, label=cut, linewidths=0)
    plt.title("Carat vs. price, by cut quality")
    plt.xlabel("Carat")
    plt.ylabel("Price (USD)")
    plt.legend(title="Cut", loc="upper left", framealpha=0.9)
    savefig("03_carat_vs_price.png")

    # ---------------------------------------------------------------
    section("5. Price by cut quality")
    avg_by_cut = diamonds.groupby("cut", observed=True)["price"].mean().reindex(cut_order)
    print("Average price by cut:")
    print(avg_by_cut.round(2))

    plt.figure(figsize=(9, 6))
    data_by_cut = [diamonds.loc[diamonds["cut"] == c, "price"] for c in cut_order]
    bp = plt.boxplot(data_by_cut, tick_labels=cut_order, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], sequential_blues(len(cut_order))):
        patch.set_facecolor(color)
        patch.set_edgecolor(INK)
    for median in bp["medians"]:
        median.set_color(INK)
    plt.title("Price distribution by cut quality")
    plt.xlabel("Cut")
    plt.ylabel("Price (USD)")
    savefig("04_price_by_cut.png")

    # ---------------------------------------------------------------
    section("6. Correlation among numerical variables")
    numerical_cols = diamonds.select_dtypes(include=[np.number]).columns
    corr = diamonds[numerical_cols].corr()
    print(corr.round(2))

    plt.figure(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=DIVERGING_CMAP, center=0,
                vmin=-1, vmax=1, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Pearson r"})
    plt.title("Correlation matrix (numerical variables)")
    savefig("05_correlation_heatmap.png")

    # ---------------------------------------------------------------
    section("7. Candidate outliers (domain-specific checks)")
    remove_specific_outliers(diamonds, verbose=True)

    print("\nEDA complete. Figures saved to:", OUT_DIR)
