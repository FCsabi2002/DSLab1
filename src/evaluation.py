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



# --- Rolling-origin validation and uncertainty ------------------------------

def rolling_origin_months(panel, first_test, last_test, period_col="period"):
    """Yield (test_month, train_index, test_index) for one-month-ahead forecasts.

    For every test month the model is trained on ALL earlier months
    (expanding window) and tested on that single month.
    """
    for month in pd.period_range(first_test, last_test, freq="M"):
        train_idx = panel.index[panel[period_col] < month]
        test_idx = panel.index[panel[period_col] == month]
        yield month, train_idx, test_idx


def wape(y_true, pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return np.sum(np.abs(pred - y_true)) / np.sum(y_true)


def cluster_bootstrap(y_true, pred_a, clusters, metric, pred_b=None,
                      n_boot=2000, seed=42):
    """95% bootstrap confidence interval of a metric (or of a difference).

    Whole clusters (e.g. months or orders) are resampled with replacement,
    because rows inside one cluster are not independent.
    If pred_b is given, the interval is for metric(pred_a) - metric(pred_b).
    Returns (point_estimate, lower, upper).
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = None if pred_b is None else np.asarray(pred_b, dtype=float)
    clusters = np.asarray(clusters)

    def stat(idx):
        value = metric(y_true[idx], pred_a[idx])
        if pred_b is not None:
            value -= metric(y_true[idx], pred_b[idx])
        return value

    unique, codes = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(seed)
    all_rows = np.arange(len(y_true))
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(unique, size=len(unique), replace=True)
        # how many times each cluster was drawn -> repeat its rows that many times
        times = np.bincount(np.searchsorted(unique, sample), minlength=len(unique))
        boot.append(stat(np.repeat(all_rows, times[codes])))
    point = stat(np.arange(len(y_true)))
    lower, upper = np.percentile(boot, [2.5, 97.5])
    return point, lower, upper