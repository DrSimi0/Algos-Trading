import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.core.vault_security import decrypt_vault_to_memory

key = "a33a292885fa3c3cc9389cabd2e812e3a0de18bbfe0bb4f3ccc8ac0a8512a159"
out_dir = Path("exports/strategyquant")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Cargar datos de los 3 años
# 2023 In-Sample
es_2023 = pd.read_parquet("data/raw/CME_MINI_ES_5m_3y.parquet")
nq_2023 = pd.read_parquet("data/raw/CME_MINI_NQ_5m_3y.parquet")
idx_2023 = es_2023.index.intersection(nq_2023.index)
es_2023, nq_2023 = es_2023.loc[idx_2023], nq_2023.loc[idx_2023]

# 2024 Out-of-Sample
es_2024 = decrypt_vault_to_memory("data/vault/oos_2024_5m.enc", key, symbol="ES")
nq_2024 = decrypt_vault_to_memory("data/vault/oos_2024_5m.enc", key, symbol="NQ")
idx_2024 = es_2024.index.intersection(nq_2024.index)
es_2024, nq_2024 = es_2024.loc[idx_2024], nq_2024.loc[idx_2024]

# 2025 Holdout
es_2025 = decrypt_vault_to_memory("data/vault/holdout_2025_5m.enc", key, symbol="ES")
nq_2025 = decrypt_vault_to_memory("data/vault/holdout_2025_5m.enc", key, symbol="NQ")
idx_2025 = es_2025.index.intersection(nq_2025.index)
es_2025, nq_2025 = es_2025.loc[idx_2025], nq_2025.loc[idx_2025]

def create_mt4_lines(df_nq, df_es):
    spread_open = (df_nq["Open"] / df_es["Open"]).round(5)
    spread_high = (df_nq["High"] / df_es["Low"]).round(5)
    spread_low = (df_nq["Low"] / df_es["High"]).round(5)
    spread_close = (df_nq["Close"] / df_es["Close"]).round(5)
    volume = df_nq.get("Volume", 100).astype(int)
    idx = df_nq.index

    lines = []
    for i in range(len(idx)):
        dt = idx[i]
        d_str = dt.strftime("%Y.%m.%d")
        t_str = dt.strftime("%H:%M")
        lines.append(
            f"{d_str},{t_str},{spread_open.iloc[i]:.5f},{spread_high.iloc[i]:.5f},{spread_low.iloc[i]:.5f},{spread_close.iloc[i]:.5f},{volume.iloc[i]}"
        )
    return lines

lines_2023 = create_mt4_lines(nq_2023, es_2023)
lines_2024 = create_mt4_lines(nq_2024, es_2024)
lines_2025 = create_mt4_lines(nq_2025, es_2025)
lines_full = lines_2023 + lines_2024 + lines_2025

# 1. Guardar archivos de Spread OHLCV (Formato MetaTrader4)
# El archivo principal SPREAD_NQ_ES_M5.csv ahora tiene los 3 años completos (61,074 barras)
with open(out_dir / "SPREAD_NQ_ES_M5.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(lines_full) + "\n")

with open(out_dir / "SPREAD_NQ_ES_M5_2024_OOS.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(lines_2024) + "\n")

with open(out_dir / "SPREAD_NQ_ES_M5_2025_Holdout.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(lines_2025) + "\n")

print(f"OHLCV Spread exportado en formato MetaTrader4:")
print(f"  SPREAD_NQ_ES_M5.csv (Full 3Y 2023-2025): {len(lines_full)} barras")
print(f"  SPREAD_NQ_ES_M5_2024_OOS.csv: {len(lines_2024)} barras")
print(f"  SPREAD_NQ_ES_M5_2025_Holdout.csv: {len(lines_2025)} barras")

# 2. Procesar Trades para QuantAnalyzer / SQX Report
df_trades_is = pd.read_csv("reports/f4_optimizacion_is_n2_trades.csv")
df_trades_oos = pd.read_csv("reports/vault/f4_optimizacion_n2_trades_oos.csv")
df_trades_hold = pd.read_csv("reports/vault/f4_optimizacion_n2_trades_holdout.csv")

def format_qa_trades(df_list, start_ticket=1):
    records = []
    ticket = start_ticket
    for df in df_list:
        for _, r in df.iterrows():
            open_time = f"{str(r['entry_date']).replace('-', '.')} {r['entry_time']}"
            close_time = f"{str(r['exit_date']).replace('-', '.')} {r['exit_time']}"
            t_type = "buy" if r["direction"] == 1 else "sell"
            size = int(r["contracts_nq"])
            item = "NQ_ES_SPREAD"
            open_price = round(float(r["entry_price_nq"]), 2)
            close_price = round(float(r["exit_price_nq"]), 2)
            comm = round(-abs(float(r["commissions_usd"])), 2)
            swap = 0.0
            profit = round(float(r["net_pnl_usd"]), 2)
            comment = f"{r['exit_reason']}_NQ{size}_ES{int(r['contracts_es'])}"

            records.append({
                "Ticket": ticket,
                "Open Time": open_time,
                "Type": t_type,
                "Size": size,
                "Item": item,
                "Open Price": open_price,
                "Close Time": close_time,
                "Close Price": close_price,
                "Commission": comm,
                "Swap": swap,
                "Profit": profit,
                "Comment": comment,
            })
            ticket += 1
    return pd.DataFrame(records)

df_all_trades = format_qa_trades([df_trades_is, df_trades_oos, df_trades_hold])
df_oos_trades = format_qa_trades([df_trades_oos])
df_hold_trades = format_qa_trades([df_trades_hold])

df_all_trades.to_csv(out_dir / "F4_Trades_QuantAnalyzer.csv", index=False)
df_oos_trades.to_csv(out_dir / "F4_Trades_2024_OOS.csv", index=False)
df_hold_trades.to_csv(out_dir / "F4_Trades_2025_Holdout.csv", index=False)

print(f"Trades exportados para QuantAnalyzer:")
print(f"  F4_Trades_QuantAnalyzer.csv (Full 3Y): {len(df_all_trades)} trades (Net PnL: ${df_all_trades['Profit'].sum():,.2f})")
print(f"  F4_Trades_2024_OOS.csv: {len(df_oos_trades)} trades (Net PnL: ${df_oos_trades['Profit'].sum():,.2f})")
print(f"  F4_Trades_2025_Holdout.csv: {len(df_hold_trades)} trades (Net PnL: ${df_hold_trades['Profit'].sum():,.2f})")
