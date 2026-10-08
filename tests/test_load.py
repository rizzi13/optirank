import pandas as pd
from optirank.data.load import convert_to_daily_parquet, make_synthetic_avazu_df


def test_chunked_parquet_loader_and_memory_savings(tmp_path):
    df_synth = make_synthetic_avazu_df(n_rows=200)
    csv_path = tmp_path / "mini_train.csv"
    out_dir = tmp_path / "daily"
    df_synth.to_csv(csv_path, index=False)

    # Compare default pandas memory vs our compact dtypes
    df_default = pd.read_csv(csv_path)
    default_bytes = df_default.memory_usage(deep=True).sum()
    compact_bytes = df_synth.memory_usage(deep=True).sum()
    assert compact_bytes < default_bytes

    # Run chunked conversion (chunksize=50 forces 4 chunks)
    counts = convert_to_daily_parquet(csv_path=csv_path, out_dir=out_dir, chunksize=50)
    assert counts == {141021: 100, 141022: 100}
    assert (out_dir / "day_141021.parquet").exists()
    assert (out_dir / "day_141022.parquet").exists()

    df_loaded = pd.read_parquet(out_dir / "day_141021.parquet")
    assert len(df_loaded) == 100
    assert str(df_loaded["click"].dtype) == "int8"