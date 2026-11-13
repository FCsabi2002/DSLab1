"""Train/test splits and evaluation metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)


def time_split(df: pd.DataFrame, test_size: float = 0.2):
    """Train on the oldest rows, test on the most recent ones.

    The DataFrame must already be sorted by date (load_data() does this).
    Returns the row positions of the training and the test set.
    """
    cut = int(len(df) * (1 - test_size))
    idx = np.arange(len(df))
    return idx[:cut], idx[cut:]


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Standard metrics for a binary classifier (1 = late)."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision_late": precision_score(y_true, pred),
        "recall_late": recall_score(y_true, pred),
        "f1_late": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }



def regression_metrics(y_true, pred) -> dict:
    """Error metrics for demand forecasts.

    WAPE = sum of absolute errors / sum of actual demand. It is scale-free,
    so categories with large and small demand can be compared.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y_true
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "WAPE": np.sum(np.abs(err)) / np.sum(y_true),
        "bias": np.mean(err),
    }