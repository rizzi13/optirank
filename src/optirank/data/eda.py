"""Exploratory Data Analysis (EDA) for OptiRank."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optirank.config import DATA_PROCESSED, DATA_RAW, DEFAULT_DEVICE_ID, ROOT
from optirank.data.load import make_synthetic_avazu_df

DOCS_IMG = ROOT / "docs" / "img"


def load_eda_dataset() -> pd.DataFrame:
    """Load day_141021.parquet if available, otherwise a 50k realistic sample."""
    day1_path = DATA_PROCESSED / "daily" / "day_141021.parquet"
    if day1_path.exists():
        print(f"Loading real day 1 from {day1_path}...")
        return pd.read_parquet(day1_path)
    print("Real daily Parquet not found yet -> generating realistic 50,000-row sample for EDA...")
    return make_synthetic_avazu_df(n_rows=50_000)


def run_eda(df: pd.DataFrame | None = None) -> dict[str, float]:
    """Answer the 6 core EDA questions and save 5 visualization charts."""
    DOCS_IMG.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = load_eda_dataset()

    # 1. Overall Click Rate
    overall_ctr = float(df["click"].mean())
    print(f"\n[1] Overall Click-Through Rate (CTR): {overall_ctr * 100:.2f}% (Total rows: {len(df):,})")

    # 2. Hourly Pattern (hour % 100 extracts 0..23)
    df["hour_of_day"] = df["hour"] % 100
    hourly_stats = df.groupby("hour_of_day")["click"].agg(["count", "mean"])

    plt.figure(figsize=(9, 4))
    plt.plot(hourly_stats.index, hourly_stats["mean"] * 100, marker="o", color="#2563eb", linewidth=2)
    plt.title("Ad Click-Through Rate (CTR) by Hour of Day (0..23)")
    plt.xlabel("Hour of Day (24-hour clock)")
    plt.ylabel("CTR (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(DOCS_IMG / "eda_hourly_ctr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[2] Saved hourly CTR chart -> docs/img/eda_hourly_ctr.png")

    # 3. The device_id Mystery & User Proxy
    dummy_ratio = float((df["device_id"] == DEFAULT_DEVICE_ID).mean())
    print(f"[3] Dummy device_id ('{DEFAULT_DEVICE_ID}') ratio: {dummy_ratio * 100:.2f}% of all rows!")
    df["user_proxy"] = np.where(
        df["device_id"] != DEFAULT_DEVICE_ID,
        df["device_id"],
        df["device_ip"] + "_" + df["device_model"],
    )
    raw_unique = df["device_id"].nunique()
    proxy_unique = df["user_proxy"].nunique()
    print(f"    Raw device_id unique count: {raw_unique:,} -> After User Proxy: {proxy_unique:,} unique users!")

    # 4. Banner Position CTR
    banner_ctr = df.groupby("banner_pos")["click"].agg(["count", "mean"])
    plt.figure(figsize=(7, 4))
    plt.bar([str(x) for x in banner_ctr.index], banner_ctr["mean"] * 100, color="#10b981")
    plt.title("CTR by Ad Banner Position")
    plt.xlabel("Banner Position ID")
    plt.ylabel("CTR (%)")
    plt.savefig(DOCS_IMG / "eda_banner_pos.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[4] Saved banner position chart -> docs/img/eda_banner_pos.png")

    # 5. App vs Website Traffic
    df["traffic_type"] = np.where(df["app_id"] == "ecad2386", "Mobile Web", "In-App")
    app_vs_web = df.groupby("traffic_type")["click"].agg(["count", "mean"])
    print(f"[5] Traffic type breakdown:")
    for t_type, row in app_vs_web.iterrows():
        print(f"    {t_type}: {int(row['count']):,} impressions, CTR = {row['mean'] * 100:.2f}%")

    # 6. Ad Turnover (Cold-Start Rate)
    half = len(df) // 2
    first_half_ads = set(df.iloc[:half]["C14"].unique())
    second_half_ads = set(df.iloc[half:]["C14"].unique())
    new_ads = second_half_ads - first_half_ads
    cold_start_ratio = len(new_ads) / max(len(second_half_ads), 1)
    print(f"[6] Cold-Start Ad Rate: {cold_start_ratio * 100:.2f}% of ads in the second half were brand-new!")

    return {
        "overall_ctr": overall_ctr,
        "dummy_ratio": dummy_ratio,
        "cold_start_ratio": cold_start_ratio,
    }


if __name__ == "__main__":
    run_eda()