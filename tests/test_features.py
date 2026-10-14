"""Unit tests for feature engineering definitions."""
import numpy as np
import pandas as pd
import pytest

from optirank.config import DEFAULT_DEVICE_ID
from optirank.data.load import make_synthetic_avazu_df
from optirank.features.definitions import build_context_features


# Test 1: Check that all 7 new columns exist after feature engineering
def test_build_context_features_adds_expected_columns():
    df = make_synthetic_avazu_df(n_rows=100)
    enriched = build_context_features(df)

    expected_cols = [
        "hour_sin",
        "hour_cos",
        "user_proxy",
        "is_mobile_web",
        "banner_area",
        "banner_aspect_ratio",
        "ad_app_key",
    ]
    for col in expected_cols:
        assert col in enriched.columns


# Test 2: Check that sine and cosine stay strictly between -1.0 and +1.0
def test_cyclical_time_bounds():
    df = make_synthetic_avazu_df(n_rows=50)
    enriched = build_context_features(df)

    assert enriched["hour_sin"].between(-1.0, 1.0).all()
    assert enriched["hour_cos"].between(-1.0, 1.0).all()
    assert enriched["hour_sin"].dtype == np.float32
    assert enriched["hour_cos"].dtype == np.float32


# Test 3: Check that masked "a99f214a" IDs get unmasked into IP_Model
def test_composite_user_proxy_unmasking():
    df = pd.DataFrame({
        "hour": [14102100, 14102100],
        "device_id": [DEFAULT_DEVICE_ID, "real_device_123"],
        "device_ip": ["192.168.1.1", "10.0.0.1"],
        "device_model": ["iPhone14", "Pixel7"],
        "app_id": ["ecad2386", "app_456"],
        "C14": [15706, 20362],
        "C15": [320, 300],
        "C16": [50, 250],
    })

    enriched = build_context_features(df)

    # Row 0: masked device_id -> should become IP_Model
    assert enriched.loc[0, "user_proxy"] == "192.168.1.1_iPhone14"
    assert enriched.loc[0, "is_mobile_web"] == 1
    assert enriched.loc[0, "banner_area"] == 16000
    assert np.isclose(enriched.loc[0, "banner_aspect_ratio"], 6.4)

    # Row 1: real device_id -> should keep original device_id
    assert enriched.loc[1, "user_proxy"] == "real_device_123"
    assert enriched.loc[1, "is_mobile_web"] == 0
    assert enriched.loc[1, "banner_area"] == 75000
    assert np.isclose(enriched.loc[1, "banner_aspect_ratio"], 1.2)


# Test 4: Check that height = 0 does NOT crash with ZeroDivisionError
def test_zero_height_safety():
    df = pd.DataFrame({
        "hour": [14102100],
        "device_id": ["dev_1"],
        "device_ip": ["1.1.1.1"],
        "device_model": ["model_1"],
        "app_id": ["app_1"],
        "C14": [100],
        "C15": [320],
        "C16": [0],  # Edge case: zero height!
    })

    enriched = build_context_features(df)
    assert not np.isinf(enriched.loc[0, "banner_aspect_ratio"])
    assert not np.isnan(enriched.loc[0, "banner_aspect_ratio"])