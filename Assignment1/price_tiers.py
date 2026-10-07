"""Price-tier separation for the stackup (classifier -> tier-specific regression) models.

Price is right-skewed, so fixed round-number cut points leave most diamonds in
the low tier. Instead the tier thresholds are the training-set tertiles of price,
giving three classes of (nearly) equal size. Thresholds are computed on the
training split only and then applied unchanged to the test split, so no test
information leaks into the class definitions.

Tier boundaries follow diamonds_utils.create_price_tiers / split_data:
    low:    price <  low_threshold
    medium: low_threshold <= price <= high_threshold
    high:   price >  high_threshold
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split

import diamonds_utils as du

TIER_NAMES = ["low", "medium", "high"]

# Sequential blue steps (light -> dark = low -> high price) and text tokens
TIER_COLORS = {"low": "#86b6ef", "medium": "#2a78d6", "high": "#104281"}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID_GRAY = "#e1e0d9"


def compute_balanced_thresholds(prices):
    """Find (low_threshold, high_threshold) that split prices into three equal-sized tiers.

    Picks thresholds among the observed prices so that the share of rows with
    price < low_threshold is as close as possible to 1/3 and the share with
    price <= high_threshold is as close as possible to 2/3, matching the
    boundary rules of du.create_price_tiers. Searching observed values (rather
    than taking a raw quantile) keeps tied prices from all landing on one side.

    Args:
        prices (array-like): Training-set prices

    Returns:
        tuple: (low_threshold, high_threshold)
    """
    p = np.sort(np.asarray(prices, dtype=float))
    n = len(p)
    candidates = np.unique(p)
    share_below = np.searchsorted(p, candidates, side="left") / n      # P(price <  v)
    share_at_or_below = np.searchsorted(p, candidates, side="right") / n  # P(price <= v)
    low_threshold = candidates[np.argmin(np.abs(share_below - 1 / 3))]
    high_threshold = candidates[np.argmin(np.abs(share_at_or_below - 2 / 3))]
    return float(low_threshold), float(high_threshold)


def split_train_test(df, test_size=0.2, random_state=42):
    """Row-level train/test split of the model dataframe.

    Uses the same test_size and random_state as du.prepare_data, so for the same
    dataframe the test rows match the ones the single-network sweep was scored on.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df


def build_price_tiers(df, test_size=0.2, random_state=42, verbose=True):
    """Split into train/test, fit balanced thresholds on train, and label both splits.

    Args:
        df (pandas.DataFrame): Model dataframe (from feature_engineering.build_features)
        test_size, random_state: Passed to split_train_test
        verbose (bool): Print the thresholds and the per-tier summary

    Returns:
        tuple: (train_df, test_df, (low_threshold, high_threshold)), where both
            dataframes carry a 'price_tier' column
    """
    train_df, test_df = split_train_test(df, test_size=test_size, random_state=random_state)
    low_t, high_t = compute_balanced_thresholds(train_df["price"])
    train_df = du.create_price_tiers(train_df, low_threshold=low_t, high_threshold=high_t)
    test_df = du.create_price_tiers(test_df, low_threshold=low_t, high_threshold=high_t)
    if verbose:
        print(f"Tier thresholds (fit on train): low < ${low_t:,.0f} <= medium <= "
              f"${high_t:,.0f} < high")
        print("\nTier summary:")
        print(tier_summary(train_df, test_df).to_string())
    return train_df, test_df, (low_t, high_t)


def tier_summary(train_df, test_df=None):
    """Per-tier row counts, class shares and price statistics."""
    frames = [("train", train_df)] + ([("test", test_df)] if test_df is not None else [])
    rows = []
    for split, d in frames:
        for tier in TIER_NAMES:
            prices = d.loc[d["price_tier"] == tier, "price"]
            rows.append({
                "split": split, "tier": tier, "n": len(prices),
                "share_%": 100 * len(prices) / len(d),
                "price_min": prices.min(), "price_median": prices.median(),
                "price_max": prices.max(), "price_mean": prices.mean(),
                "price_std": prices.std(), "price_skew": prices.skew(),
            })
    return pd.DataFrame(rows).set_index(["split", "tier"]).round(2)


def plot_tier_distribution(train_df, thresholds, path):
    """Training price histogram with the tier thresholds, beside per-tier class counts."""
    low_t, high_t = thresholds
    fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(13, 4.8),
                                          gridspec_kw={"width_ratios": [2.2, 1]})

    bins = np.linspace(train_df["price"].min(), train_df["price"].max(), 80)
    for tier in TIER_NAMES:
        ax_hist.hist(train_df.loc[train_df["price_tier"] == tier, "price"], bins=bins,
                     color=TIER_COLORS[tier], edgecolor="white", linewidth=0.5,
                     label=tier.capitalize())
    for t in (low_t, high_t):
        ax_hist.axvline(t, color=INK, linestyle="--", linewidth=1)
        ax_hist.text(t, ax_hist.get_ylim()[1] * 0.97, f" ${t:,.0f}", color=INK,
                     fontsize=9, va="top")
    ax_hist.set_title("Training price distribution by tier", fontweight="bold",
                      color=INK, loc="left")
    ax_hist.set_xlabel("Price (USD)", color=INK)
    ax_hist.set_ylabel("Count", color=INK)
    ax_hist.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax_hist.legend(title="Tier", frameon=False)

    counts = train_df["price_tier"].value_counts().reindex(TIER_NAMES)
    bars = ax_bar.bar([t.capitalize() for t in TIER_NAMES], counts.values,
                      color=[TIER_COLORS[t] for t in TIER_NAMES], width=0.6)
    for bar, n in zip(bars, counts.values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{n:,}\n({100 * n / counts.sum():.1f}%)", ha="center", va="bottom",
                    fontsize=9, color=MUTED)
    ax_bar.set_title("Training rows per tier", fontweight="bold", color=INK, loc="left")
    ax_bar.set_ylabel("Count", color=INK)
    ax_bar.set_ylim(0, counts.max() * 1.2)

    for ax in (ax_hist, ax_bar):
        ax.set_facecolor("white")
        ax.grid(axis="y", color=GRID_GRAY, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_GRAY)
        ax.tick_params(colors=MUTED)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")
