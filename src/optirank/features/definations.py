from_future_ import annotations
import pandas as pd
import numpy as np
from optirank.config import DEFAULT_DEVICE_ID
def build_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build context features for the given DataFrame."""
    df = df.copy()
    hour_of_day = df["hour"] % 100
    theta = 2 * np.pi * hour_of_day / 24
    df["hour_sin"] = np.sin(theta).astype(np.float32)
    df["hour_cos"] = np.cos(theta).astype(np.float32)

    df[user_proxy] = np.where(
        df["device_id"] != DEFAULT_DEVICE_ID,   
        df["device_id"],
        df["device_ip"] + "_" + df["device_model"],

 df["is_mobile_web"] = (df["app_id"] == "ecad2386").astype(np.int8)
    df["banner_area"] = (df["C15"].astype(np.int32) * df["C16"].astype(np.int32)).astype(np.int32)
    df["banner_aspect_ratio"] = (df["C15"].astype(np.float32) / np.maximum(df["C16"].astype(np.float32), 1.0)).astype(np.float32)
    df["ad_app_key"] = df["C14"].astype(str) + "_" + df["app_id"].astype(str)
    return df