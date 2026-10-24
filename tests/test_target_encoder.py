from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optirank.features.target_encoder import OutOfFoldTargetEncoder


@pytest.fixture
def synthetic_encoder_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    apps = ["app_popular"] * 160 + ["app_medium"] * 35 + ["app_rare"] * 5
    # app_popular has 10% CTR, app_medium has 80% CTR, app_rare has 100% CTR
    clicks = []
    for app in apps:
        if app == "app_popular":
            clicks.append(1 if np.random.rand() < 0.10 else 0)
        elif app == "app_medium":
            clicks.append(1 if np.random.rand() < 0.80 else 0)
        else:
            clicks.append(1)  # 100% CTR for the 5 rare clicks

    return pd.DataFrame({
        "app_id": apps,
        "site_id": ["site_1"] * 100 + ["site_2"] * 100,
        "click": np.array(clicks, dtype=np.int8),
    })


def test_target_encoder_shape_and_types(synthetic_encoder_df: pd.DataFrame) -> None:
    df = synthetic_encoder_df.copy()
    encoder = OutOfFoldTargetEncoder(cols=["app_id", "site_id"], n_splits=5, m=10.0)
    out_df = encoder.fit_transform(df, target_col="click")

    assert "app_id_te" in out_df.columns
    assert "site_id_te" in out_df.columns
    assert out_df["app_id_te"].dtype == np.float32
    assert out_df["site_id_te"].dtype == np.float32
    assert len(out_df) == len(df)
    assert not out_df["app_id_te"].isna().any()
    assert not out_df["site_id_te"].isna().any()


def test_target_encoder_no_target_leakage(synthetic_encoder_df: pd.DataFrame) -> None:
    df = synthetic_encoder_df.copy()
    encoder = OutOfFoldTargetEncoder(cols=["app_id"], n_splits=5, m=10.0)
    out_df = encoder.fit_transform(df, target_col="click")

    # Even though app_rare has 100% click in the dataset,
    # because of OOF and shrinkage, its value MUST be strictly less than 1.0
    rare_encoded_vals = out_df[out_df["app_id"] == "app_rare"]["app_id_te"].values
    assert np.all(rare_encoded_vals < 0.95), f"Target leaked! Rare got {rare_encoded_vals}"


def test_empirical_bayes_shrinkage_bounds(synthetic_encoder_df: pd.DataFrame) -> None:
    df = synthetic_encoder_df.copy()
    encoder = OutOfFoldTargetEncoder(cols=["app_id"], n_splits=5, m=10.0)
    encoder.fit_transform(df, target_col="click")

    global_prior = encoder.global_prior
    mapping = encoder.encoding_maps["app_id"]

    # app_popular (160 samples) should be very close to its true empirical mean
    pop_empirical_mean = df[df["app_id"] == "app_popular"]["click"].mean()
    pop_smoothed = mapping["app_popular"]
    assert abs(pop_smoothed - pop_empirical_mean) < 0.05

    # app_rare (5 samples) with 100% clicks should be pulled strongly toward global_prior
    rare_smoothed = mapping["app_rare"]
    assert rare_smoothed < 0.70  # Pulled down from 1.0 toward prior (~0.25)
    assert rare_smoothed > global_prior


def test_inference_cold_start_fallback(synthetic_encoder_df: pd.DataFrame) -> None:
    train_df = synthetic_encoder_df.copy()
    encoder = OutOfFoldTargetEncoder(cols=["app_id"], n_splits=5, m=10.0)
    encoder.fit_transform(train_df, target_col="click")

    # Test set containing a brand new unseen app ID
    test_df = pd.DataFrame({
        "app_id": ["app_unseen_xyz", "app_popular"],
    })
    transformed = encoder.transform(test_df)

    assert "app_id_te" in transformed.columns
    assert not transformed["app_id_te"].isna().any()
    # Unseen app ID should fall back exactly to global_prior
    assert transformed.loc[0, "app_id_te"] == pytest.approx(encoder.global_prior, abs=1e-5)
