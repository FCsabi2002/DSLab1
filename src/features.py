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