"""Unit tests for temporal splitting and Bayes prior correction."""
import numpy as np
import pandas as pd
import pytest

from optirank.config import SplitConfig
from optirank.features.split import correct_prior, downsample_negatives, get_split_days


def test_split_days_strictly_chronological():
    splits = get_split_days()
    assert splits["train"] == [141021, 141022, 141023, 141024, 141025, 141026, 141027, 141028]
    assert splits["val"] == [141029]
    assert splits["test"] == [141030]

    # Verify no day overlap between splits
    assert set(splits["train"]).isdisjoint(set(splits["val"]))
    assert set(splits["train"]).isdisjoint(set(splits["test"]))
    assert set(splits["val"]).isdisjoint(set(splits["test"]))


def test_downsample_negatives_preserves_clicks():
    # 20 clicks and 80 non-clicks (100 total, 20% CTR)
    df = pd.DataFrame({
        "click": np.array([1] * 20 + [0] * 80),
        "feature": np.arange(100),
    })

    downsampled = downsample_negatives(df, negative_rate=0.25, random_state=42)

    # All 20 clicks must be retained
    assert (downsampled["click"] == 1).sum() == 20
    # 80 negatives * 0.25 = 20 negatives retained
    assert (downsampled["click"] == 0).sum() == 20
    assert len(downsampled) == 40


def test_correct_prior_recovers_true_probability():
    # Suppose true CTR is 0.1742
    true_p = 0.1742
    w = 0.25

    # After downsampling negatives by 0.25, the shifted prior is:
    shifted_p = true_p / (true_p + w * (1.0 - true_p))

    # Our analytical correction must invert this back to 0.1742
    recovered = correct_prior(shifted_p, negative_rate=w)
    assert np.isclose(recovered, true_p, atol=1e-6)

    # Edge case: no downsampling (w = 1.0) must leave probability unchanged
    assert np.isclose(correct_prior(0.35, negative_rate=1.0), 0.35)