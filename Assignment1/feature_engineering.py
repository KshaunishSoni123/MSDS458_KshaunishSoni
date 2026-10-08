"""Feature engineering for the diamond price regression networks.

Builds the model-ready dataframe from the raw seaborn diamonds dataset:
  - removes domain-specific outliers (zero / extreme dimensions)
  - removes IQR outliers (outside Q1 - 3*IQR .. Q3 + 3*IQR) on the raw numeric columns
  - ordinal-encodes cut, color and clarity (worst -> best, starting at 1)
  - adds engineered features: log_carat, volume, density (clipped), cut/wt, color/wt, clarity/wt
  - keeps the raw measurements x, y, z, depth and table as model features
  - keeps price (USD) as the regression target

Typical use:

    from feature_engineering import build_features, NUMERICAL_FEATURES, \
        CATEGORICAL_FEATURES, TARGET
    df = build_features()
    X_train, X_test, y_train, y_test, pre = prepare_data(
        df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, target=TARGET)

Run directly to print a feature summary and correlations with the target:

    SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())") python3 feature_engineering.py
"""

import numpy as np
import pandas as pd

from Assignment1.diamonds_utils import load_data, remove_specific_outliers

# Ordinal grade orders, worst -> best (encoded as 1..n)
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
GRADE_ORDERS = {"cut": CUT_ORDER, "color": COLOR_ORDER, "clarity": CLARITY_ORDER}

# carat / (x*y*z) sits in 0.0057-0.0070 for 99.9% of stones; values outside come from
# mis-measured dimensions and reach 90 std devs after scaling, which makes the networks
# extrapolate wildly. Fixed bounds (not data quantiles) so no train/test leakage.
DENSITY_BOUNDS = (0.0055, 0.0070)

# IQR outlier filter, applied to the whole dataset before the train/test split
IQR_COLUMNS = ["carat", "depth", "table", "x", "y", "z", "price"]
IQR_K = 3.0

TARGET = "price"

# Grades are already ordinal integers, so they go in as numerical features
# (prepare_data would otherwise one-hot encode them).
NUMERICAL_FEATURES = [
    "log_carat",
    "volume",
    "density",
    "cut",
    "color",
    "clarity",
    "cut/wt",
    "color/wt",
    "clarity/wt",
    # Raw measurements: volume and density compress x, y, z and lose shape
    # information (a gradient-boosting check dropped MAPE ~7.7% -> ~6.1% with these)
    "x",
    "y",
    "z",
    "depth",
    "table",
]
CATEGORICAL_FEATURES = []


def encode_grades(df):
    """Map cut, color and clarity to ordinal integers (1 = worst grade)."""
    df = df.copy()
    for col, order in GRADE_ORDERS.items():
        mapping = {grade: i + 1 for i, grade in enumerate(order)}
        encoded = df[col].astype(str).map(mapping)
        if encoded.isna().any():
            unknown = sorted(df.loc[encoded.isna(), col].astype(str).unique())
            raise ValueError(f"encode_grades(): unknown {col} value(s): {unknown}")
        df[col] = encoded.astype(int)
    return df


def add_engineered_features(df):
    """Add size, density and grade-per-weight features.

    Expects grades already ordinal-encoded and zero dimensions already removed.
    """
    df = df.copy()
    df["log_carat"] = np.log(df["carat"])
    df["volume"] = df["x"] * df["y"] * df["z"]
    df["density"] = (df["carat"] / df["volume"]).clip(*DENSITY_BOUNDS)
    df["cut/wt"] = df["cut"] / df["carat"]
    df["color/wt"] = df["color"] / df["carat"]
    df["clarity/wt"] = df["clarity"] / df["carat"]
    return df


def remove_iqr_outliers(df, columns=IQR_COLUMNS, k=IQR_K, verbose=True):
    """Drop rows with any column outside [Q1 - k*IQR, Q3 + k*IQR].

    Fences are computed once on the input dataframe (before filtering) so every
    column is judged against the same population.
    """
    q1, q3 = df[columns].quantile(0.25), df[columns].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    outside = (df[columns] < lower) | (df[columns] > upper)
    keep = ~outside.any(axis=1)
    if verbose:
        print(f"\nIQR outlier filter (k={k}): removed {(~keep).sum():,} of {len(df):,} rows "
              f"({100 * (~keep).mean():.2f}%)")
        print(pd.DataFrame({"lower": lower, "upper": upper,
                            "rows_outside": outside.sum()}).round(3).to_string())
    return df[keep]


def build_features(diamonds=None, verbose=True):
    """Full pipeline: load -> remove outliers (specific + IQR) -> encode grades -> engineer features.

    Args:
        diamonds (pandas.DataFrame, optional): raw dataset; loaded if not given
        verbose (bool): print outlier-removal statistics

    Returns:
        pandas.DataFrame: dataset with original columns, engineered features and target
    """
    if diamonds is None:
        diamonds = load_data()
    df = remove_specific_outliers(diamonds, verbose=verbose)
    df = remove_iqr_outliers(df, verbose=verbose)
    df = encode_grades(df)
    df = add_engineered_features(df)
    return df.reset_index(drop=True)


def main():
    summarize_features(build_features())


def summarize_features(df):
    """Print the model feature set, a summary of each feature and its correlation with the target."""
    print("\nModel features:", NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    print("Target:", TARGET)
    print(f"Rows: {len(df):,}")

    print("\nFeature summary:")
    print(df[NUMERICAL_FEATURES + [TARGET]].describe().T.round(3))

    print("\nCorrelation with target:")
    corr = pd.DataFrame({
        "pearson_price": df[NUMERICAL_FEATURES].corrwith(df[TARGET]),
        "spearman_price": df[NUMERICAL_FEATURES].corrwith(df[TARGET], method="spearman"),
    })
    print(corr.sort_values("pearson_price", key=abs, ascending=False).round(3))

    n_bad = (~np.isfinite(df[NUMERICAL_FEATURES + [TARGET]])).sum().sum()
    print(f"\nNon-finite values in features/target: {n_bad}")


if __name__ == "__main__":
    main()
