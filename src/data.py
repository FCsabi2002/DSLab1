"""Loading and basic cleaning of the DataCo Smart Supply Chain dataset."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "DataCoSupplyChainDataset.csv"

# Columns that are empty, almost empty, or personal data not needed for modelling
DROP_COLUMNS = [
    "Product Description",   # all values missing
    "Order Zipcode",         # ~86% missing
    "Product Image",         # URL, no information
    "Customer Email",        # personal data
    "Customer Password",     # personal data
    "Customer Street",       # personal data
    "Customer Fname",        # personal data
    "Customer Lname",        # personal data
]


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Read the original CSV file without any changes."""
    return pd.read_csv(path, encoding="unicode_escape")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning shared by every notebook and by the API."""
    df = df.copy()
    df = df.drop(columns=DROP_COLUMNS)
    df["Customer Zipcode"] = df["Customer Zipcode"].fillna(0)

    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"])
    df["shipping_date"] = pd.to_datetime(df["shipping date (DateOrders)"])
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_week_day"] = df["order_date"].dt.weekday
    df["order_hour"] = df["order_date"].dt.hour

    # Chronological order is needed for time-based validation
    df = df.sort_values("order_date", kind="stable").reset_index(drop=True)
    return df


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load and clean the dataset in one step."""
    return clean(load_raw(path))