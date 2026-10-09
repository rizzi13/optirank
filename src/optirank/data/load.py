"""Stream Avazu train.gz in chunks and write daily Parquet partitions."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from optirank.config import DATA_PROCESSED, DATA_RAW, DEFAULT_DEVICE_ID, SEED

INT_COLS = {
    "click": "int8",
    "hour": "int32",
    "C1": "int16",
    "banner_pos": "int8",
    "device_type": "int8",
    "device_conn_type": "int8",
    "C14": "int32",
    "C15": "int16",
    "C16": "int16",
    "C17": "int16",
    "C18": "int8",
    "C19": "int16",
    "C20": "int32",
    "C21": "int16",
}

STR_COLS = [
    "id",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
]


def get_avazu_dtypes() -> dict[str, str]:
    """Return compact dtypes dictionary (~65% less RAM than pandas defaults)."""
    return {**INT_COLS, **{c: "string" for c in STR_COLS}}


def convert_to_daily_parquet(
    csv_path: Path = DATA_RAW / "train.gz",
    out_dir: Path = DATA_PROCESSED / "daily",
    chunksize: int = 1_000_000,
) -> dict[int, int]:
    """Stream Avazu CSV/GZ in chunks and write one Parquet file per day (YYMMDD)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dtypes = get_avazu_dtypes()

    day_buffers: dict[int, list[pd.DataFrame]] = {}
    day_counts: dict[int, int] = {}

    for chunk in pd.read_csv(csv_path, dtype=dtypes, chunksize=chunksize):
        # Extract YYMMDD from YYMMDDHH (e.g. 14102100 // 100 -> 141021)
        chunk["day"] = (chunk["hour"] // 100).astype("int32")
        for day, group in chunk.groupby("day"):
            day_int = int(day)
            day_buffers.setdefault(day_int, []).append(group)
            day_counts[day_int] = day_counts.get(day_int, 0) + len(group)

        # Flush completed days to disk to free RAM (Avazu is ordered by hour)
        max_day_in_chunk = int(chunk["day"].max())
        for buffered_day in list(day_buffers.keys()):
            if buffered_day < max_day_in_chunk:
                df_day = pd.concat(day_buffers.pop(buffered_day), ignore_index=True)
                df_day.to_parquet(out_dir / f"day_{buffered_day}.parquet", index=False)

    # Flush remaining days
    for buffered_day, parts in day_buffers.items():
        df_day = pd.concat(parts, ignore_index=True)
        df_day.to_parquet(out_dir / f"day_{buffered_day}.parquet", index=False)

    return day_counts


def make_synthetic_avazu_df(n_rows: int = 200, seed: int = SEED) -> pd.DataFrame:
    """Create a small Avazu-schema DataFrame for fast unit testing & CI."""
    rng = np.random.default_rng(seed)
    hours = np.repeat([14102100, 14102101, 14102200, 14102201], n_rows // 4)
    df = pd.DataFrame({
        "id": [f"ad_{i}" for i in range(len(hours))],
        "click": rng.binomial(1, 0.17, size=len(hours)).astype("int8"),
        "hour": hours.astype("int32"),
        "C1": np.full(len(hours), 1005, dtype="int16"),
        "banner_pos": rng.choice([0, 1], size=len(hours)).astype("int8"),
        "site_id": rng.choice(["85f751fd", "1fbe01fe"], size=len(hours)),
        "site_domain": rng.choice(["c4e18dd6", "f3845767"], size=len(hours)),
        "site_category": rng.choice(["50e219e0", "28905ebd"], size=len(hours)),
        "app_id": rng.choice(["ecad2386", "febd1138"], size=len(hours)),
        "app_domain": rng.choice(["7801e8d9", "2347f47a"], size=len(hours)),
        "app_category": rng.choice(["07d7df22", "cef3e649"], size=len(hours)),
        "device_id": rng.choice([DEFAULT_DEVICE_ID, "dev_1", "dev_2"], p=[0.8, 0.1, 0.1], size=len(hours)),
        "device_ip": rng.choice(["ip_1", "ip_2", "ip_3"], size=len(hours)),
        "device_model": rng.choice(["mod_a", "mod_b"], size=len(hours)),
        "device_type": np.ones(len(hours), dtype="int8"),
        "device_conn_type": np.zeros(len(hours), dtype="int8"),
        "C14": rng.choice([15706, 20362, 21611], size=len(hours)).astype("int32"),
        "C15": np.full(len(hours), 320, dtype="int16"),
        "C16": np.full(len(hours), 50, dtype="int16"),
        "C17": np.full(len(hours), 1722, dtype="int16"),
        "C18": np.zeros(len(hours), dtype="int8"),
        "C19": np.full(len(hours), 35, dtype="int16"),
        "C20": np.full(len(hours), -1, dtype="int32"),
        "C21": np.full(len(hours), 79, dtype="int16"),
    })
    return df.astype(get_avazu_dtypes())