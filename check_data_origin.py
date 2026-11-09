import pandas as pd
import glob

print("=== CHECKING FILES ORIGIN AND STATS ===")
for yr in [2021, 2022, 2023]:
    f_es = f"data/raw/CME_MINI_ES_{yr}_5m.parquet"
    f_nq = f"data/raw/CME_MINI_NQ_{yr}_5m.parquet"
    es = pd.read_parquet(f_es)
    nq = pd.read_parquet(f_nq)
    print(f"\n--- AÑO {yr} ---")
    print(f"ES: {len(es)} filas | Fechas: {es.index[0]} a {es.index[-1]}")
    print(f"ES Close: Min={es['Close'].min():.2f}, Max={es['Close'].max():.2f}, Mean={es['Close'].mean():.2f}")
    print(f"NQ: {len(nq)} filas | Fechas: {nq.index[0]} a {nq.index[-1]}")
    print(f"NQ Close: Min={nq['Close'].min():.2f}, Max={nq['Close'].max():.2f}, Mean={nq['Close'].mean():.2f}")
