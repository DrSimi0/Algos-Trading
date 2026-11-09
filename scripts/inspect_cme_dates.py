import pandas as pd

for sym in ["NQ", "ES"]:
    p = f"data/raw/CME_MINI_{sym}_2021_2023_5m.parquet"
    df = pd.read_parquet(p)
    print(f"[{sym}] shape: {df.shape}, min: {df.index.min()}, max: {df.index.max()}")
    print(f"[{sym}] years: {pd.Series(df.index.year).value_counts().to_dict()}")
