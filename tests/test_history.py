from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optirank.features.history import compute_point_in_time_user_features


@pytest.fixture
def synthetic_history_df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_proxy": ["u1", "u1", "u1", "u1", "u2", "u2"],
        "hour": [14102100, 14102101, 14102102, 14102103, 14102100, 14102101],
        "C14": ["camp_A", "camp_A", "camp_A", "camp_A", "camp_A", "camp_B"],
    })


def test_history_monotonically_increasing(synthetic_history_df: pd.DataFrame) -> None:
    df = compute_point_in_time_user_features(synthetic_history_df)

    u1_rows = df[df["user_proxy"] == "u1"].sort_values("hour")
    # Impression counts must be exactly [0, 1, 2, 3]
    assert list(u1_rows["user_historical_impressions"]) == [0, 1, 2, 3]

    u2_rows = df[df["user_proxy"] == "u2"].sort_values("hour")
    # u2 impression counts must start at 0 independently
    assert list(u2_rows["user_historical_impressions"]) == [0, 1]


def test_no_lookahead_leakage(synthetic_history_df: pd.DataFrame) -> None:
    # First 3 rows of u1
    df_short = synthetic_history_df.iloc[:3].copy()
    out_short = compute_point_in_time_user_features(df_short)

    # All 4 rows of u1
    df_long = synthetic_history_df.iloc[:4].copy()
    out_long = compute_point_in_time_user_features(df_long)

    # The features for the first 3 events must be 100% IDENTICAL regardless of whether row 4 exists
    pd.testing.assert_series_equal(
        out_short["user_historical_impressions"],
        out_long["user_historical_impressions"].iloc[:3],
    )
    pd.testing.assert_series_equal(
        out_short["user_campaign_historical_count"],
        out_long["user_campaign_historical_count"].iloc[:3],
    )


def test_ad_fatigue_threshold(synthetic_history_df: pd.DataFrame) -> None:
    df = compute_point_in_time_user_features(synthetic_history_df)

    u1_rows = df[df["user_proxy"] == "u1"].sort_values("hour")
    # Impressions 0, 1, 2 should NOT be fatigued; impression 3 (4th ad seen) MUST be fatigued
    assert list(u1_rows["is_user_ad_fatigued"]) == [0, 0, 0, 1]


def test_independent_campaign_counters(synthetic_history_df: pd.DataFrame) -> None:
    df = compute_point_in_time_user_features(synthetic_history_df)

    u2_rows = df[df["user_proxy"] == "u2"].sort_values("hour")
    # u2 saw camp_A at hour 0 (count=0), then camp_B at hour 1 (count=0)
    assert list(u2_rows["user_campaign_historical_count"]) == [0, 0]
