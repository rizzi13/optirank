from __future__ import annotations

import numpy as np
import pandas as pd


def compute_point_in_time_user_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.sort_values(by=["user_proxy", "hour"]).reset_index(drop=True)

    df["user_historical_impressions"] = (
        df.groupby("user_proxy").cumcount().astype(np.int32)
    )

    df["user_campaign_historical_count"] = (
        df.groupby(["user_proxy", "C14"]).cumcount().astype(np.int16)
    )

    df["is_user_ad_fatigued"] = (
        df["user_campaign_historical_count"] >= 3
    ).astype(np.int8)

    return df
