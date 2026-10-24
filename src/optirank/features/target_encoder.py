from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class OutOfFoldTargetEncoder:
    def __init__(self, cols: list[str], n_splits: int = 5, m: float = 10.0) -> None:
        self.cols = cols
        self.n_splits = n_splits
        self.m = m
        self.global_prior: float = 0.0
        self.encoding_maps: dict[str, dict[Any, float]] = {}

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        self.global_prior = float(df[target_col].mean())
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for col in self.cols:
            encoded_col = f"{col}_te"
            df[encoded_col] = np.float32(self.global_prior)

            for train_idx, val_idx in kf.split(df):
                train_fold = df.iloc[train_idx]
                stats = train_fold.groupby(col)[target_col].agg(["count", "sum"])

                smoothed = (stats["sum"] + self.m * self.global_prior) / (stats["count"] + self.m)

                val_categories = df.iloc[val_idx][col]
                encoded_vals = val_categories.map(smoothed).fillna(self.global_prior).astype(np.float32)
                df.iloc[val_idx, df.columns.get_loc(encoded_col)] = encoded_vals

            full_stats = df.groupby(col)[target_col].agg(["count", "sum"])
            self.encoding_maps[col] = (
                (full_stats["sum"] + self.m * self.global_prior) / (full_stats["count"] + self.m)
            ).to_dict()

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.cols:
            encoded_col = f"{col}_te"
            mapping = self.encoding_maps.get(col, {})
            df[encoded_col] = (
                df[col].map(mapping).fillna(self.global_prior).astype(np.float32)
            )
        return df
