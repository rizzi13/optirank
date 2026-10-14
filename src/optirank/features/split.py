"""Temporal dataset partitioning and prior shift correction for OptiRank."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from optirank.config import DATA_PROCESSED, SplitConfig


def get_split_days(config: SplitConfig | None = None) -> dict[str, list[int]]:
    """Return lists of day integers for train, val, and test partitions."""
    cfg = config or SplitConfig()
    train_end = cfg.train_end_day // 100
    val_end = cfg.val_end_day // 100

    all_days = [141021, 141022, 141023, 141024, 141025, 141026, 141027, 141028, 141029, 141030]
    return {
        "train": [d for d in all_days if d <= train_end],
        "val": [d for d in all_days if train_end < d <= val_end],
        "test": [d for d in all_days if d > val_end],
    }


def downsample_negatives(
    df: pd.DataFrame,
    negative_rate: float = 0.25,
    random_state: int = 42,
) -> pd.DataFrame:
    """Keep 100% of clicks and downsample non-clicks by negative_rate."""
    if negative_rate >= 1.0:
        return df

    positives = df[df["click"] == 1]
    negatives = df[df["click"] == 0]

    sampled_negatives = negatives.sample(frac=negative_rate, random_state=random_state)
    combined = pd.concat([positives, sampled_negatives], ignore_index=True)
    return combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def correct_prior(
    raw_p: float | np.ndarray,
    negative_rate: float = 0.25,
) -> float | np.ndarray:
    """Invert prior shift analytically using Bayes odds formula.

    q_hat = p_hat / (p_hat + (1 - p_hat) / w)
    """
    if negative_rate >= 1.0:
        return raw_p
    odds = (1.0 - raw_p) / negative_rate
    return raw_p / (raw_p + odds)