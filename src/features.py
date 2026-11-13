"""Feature definitions and preprocessing shared by training and the API."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# --- Late delivery model -------------------------------------------------
# Only information that is known at the moment the order is placed.
DELAY_NUMERIC = [
    "Customer Zipcode",
    "order_year",
    "order_month",
    "order_week_day",
    "order_hour",
]
DELAY_CATEGORICAL = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Category Name",
    "Department Name",
    "Customer Segment",
    "Type",
]
DELAY_FEATURES = DELAY_NUMERIC + DELAY_CATEGORICAL
DELAY_TARGET = "Late_delivery_risk"


def delay_preprocessor() -> ColumnTransformer:
    """One-hot encode the categorical columns, pass numeric columns through.

    handle_unknown="ignore": a category never seen in training (e.g. a new
    region sent to the API) becomes all zeros instead of crashing.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             DELAY_CATEGORICAL),
            ("num", "passthrough", DELAY_NUMERIC),
        ]
    )



# --- Demand model ----------------------------------------------------------
DEMAND_START = "2015-01"   # first month in the data
DEMAND_END = "2017-09"     # last complete month (after this the data is broken)
DEMAND_LAGS = [1, 2, 3, 12]


def monthly_demand_panel(df, start=DEMAND_START, end=DEMAND_END, complete_only=True):
    """Monthly ordered quantity per product category.

    complete_only=True keeps only categories that have orders in EVERY month
    of the window, so that lag features are always available.
    """
    import pandas as pd

    period = df["order_date"].dt.to_period("M")
    mask = (period >= pd.Period(start, "M")) & (period <= pd.Period(end, "M"))
    panel = (
        df[mask]
        .assign(period=period[mask])
        .groupby(["Category Name", "period"])["Order Item Quantity"]
        .sum()
        .rename("demand")
        .reset_index()
    )
    if complete_only:
        n_periods = panel["period"].nunique()
        months_per_cat = panel.groupby("Category Name")["period"].nunique()
        keep = months_per_cat[months_per_cat == n_periods].index
        panel = panel[panel["Category Name"].isin(keep)]
    return panel.sort_values(["Category Name", "period"]).reset_index(drop=True)


def add_lag_features(panel, lags=DEMAND_LAGS):
    """Add features that only use PAST demand (no information from the target month)."""
    panel = panel.copy()
    by_cat = panel.groupby("Category Name")["demand"]
    for lag in lags:
        panel[f"lag_{lag}"] = by_cat.shift(lag)
    panel["roll_mean_3"] = by_cat.transform(lambda s: s.shift(1).rolling(3).mean())
    panel["month"] = panel["period"].dt.month
    return panel.dropna().reset_index(drop=True)



DEMAND_LAG_FEATURES = [f"lag_{lag}" for lag in DEMAND_LAGS] + ["roll_mean_3"]
DEMAND_FEATURES = DEMAND_LAG_FEATURES + ["month", "Category Name"]
DEMAND_TARGET = "demand"


def demand_tree_preprocessor() -> ColumnTransformer:
    """For Random Forest / XGBoost: one-hot category, numeric features as they are."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["Category Name"]),
            ("num", "passthrough", DEMAND_LAG_FEATURES + ["month"]),
        ]
    )


def demand_poisson_preprocessor() -> ColumnTransformer:
    """For Poisson regression (log link): log of past demand, one-hot category and month."""
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    return ColumnTransformer(
        transformers=[
            ("lags", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()),
             DEMAND_LAG_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["Category Name", "month"]),
        ]
    )