import csv
from pathlib import Path
import numpy as np
import pandas as pd

# 1. Rutas
trades_src = Path("reports/f4_optimizacion_is_n2_trades.csv")
out_dir = Path("exports/strategyquant")
out_dir.mkdir(parents=True, exist_ok=True)

# 2. Procesar Trades para QuantAnalyzer / SQX
if trades_src.exists():
    df_trades = pd.read_csv(trades_src)
    qa_records = []
    for _, r in df_trades.iterrows():
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

        qa_records.append({
            "Ticket": int(r["trade_id"]),
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

    qa_df = pd.DataFrame(qa_records)

    # 1. Coma (CSV estándar)
    qa_df.to_csv(out_dir / "F4_Trades_QuantAnalyzer.csv", index=False, sep=",")
    qa_df.to_csv(out_dir / "F4_Trades_QuantAnalyzer_comma.csv", index=False, sep=",")

    # 2. Semicolon (Punto y coma)
    qa_df.to_csv(out_dir / "F4_Trades_QuantAnalyzer_semicolon.csv", index=False, sep=";")

    # 3. Tab (Tabulación .tsv y .txt)
    qa_df.to_csv(out_dir / "F4_Trades_QuantAnalyzer_tab.tsv", index=False, sep="\t")
    qa_df.to_csv(out_dir / "F4_Trades_QuantAnalyzer_tab.txt", index=False, sep="\t")

    # 4. Space (Espacio)
    qa_df.to_csv(out_dir / "F4_Trades_QuantAnalyzer_space.txt", index=False, sep=" ", quoting=csv.QUOTE_NONNUMERIC)
    print(f"Generated Trades with Comma, Semicolon, Tab, Space: {len(qa_df)} trades")

# 3. Procesar Serie de Precios OHLCV del Spread para Data Manager de SQX
# Usamos las series 2023 de NQ y ES de 20,280 barras sincronizadas
nq_path = Path("data/raw/CME_MINI_NQ_5m_3y.parquet")
es_path = Path("data/raw/CME_MINI_ES_5m_3y.parquet")

if not nq_path.exists() or not es_path.exists():
    nq_path = Path("data/raw/CME_MINI_NQ_5m_bars5000.parquet")
    es_path = Path("data/raw/CME_MINI_ES_5m_bars5000.parquet")

df_nq = pd.read_parquet(nq_path) if nq_path.suffix == ".parquet" else pd.read_csv(nq_path, index_col=0, parse_dates=True)
df_es = pd.read_parquet(es_path) if es_path.suffix == ".parquet" else pd.read_csv(es_path, index_col=0, parse_dates=True)

common_idx = df_nq.index.intersection(df_es.index)
df_nq = df_nq.loc[common_idx]
df_es = df_es.loc[common_idx]

# Beta medio de equilibrio ~1.00 para log-spread, o ratio de precio normalizado
# Para StrategyQuant X, es ideal que el activo sintético tenga precios positivos realistas (ej. Base 100 o Ratio NQ/ES)
# Creamos tanto el Spread en Ratio (NQ / ES) como el S-Score Spread
# El Ratio NQ/ES (alrededor de 3.8 a 4.2) se comporta exactamente como una acción o par forex en SQX
spread_open = df_nq["Open"] / df_es["Open"]
spread_high = df_nq["High"] / df_es["Low"]
spread_low = df_nq["Low"] / df_es["High"]
spread_close = df_nq["Close"] / df_es["Close"]
volume = df_nq.get("Volume", 100)

spread_df = pd.DataFrame({
    "Date": common_idx.strftime("%Y.%m.%d"),
    "Time": common_idx.strftime("%H:%M"),
    "Open": spread_open.round(5),
    "High": spread_high.round(5),
    "Low": spread_low.round(5),
    "Close": spread_close.round(5),
    "Volume": volume.astype(int),
})

# Guardar en los 4 formatos de delimitador para Datos Históricos de SQX Data Manager:
# 1. Coma
spread_df.to_csv(out_dir / "NQ_ES_Spread_5m_StrategyQuant.csv", index=False, sep=",")
spread_df.to_csv(out_dir / "NQ_ES_Spread_5m_comma.csv", index=False, sep=",")

# 2. Semicolon (Punto y coma)
spread_df.to_csv(out_dir / "NQ_ES_Spread_5m_semicolon.csv", index=False, sep=";")

# 3. Tab (Tabulación .txt y .tsv)
spread_df.to_csv(out_dir / "NQ_ES_Spread_5m_tab.txt", index=False, sep="\t")
spread_df.to_csv(out_dir / "NQ_ES_Spread_5m_tab.tsv", index=False, sep="\t")

# 4. Space (Espacio)
spread_df.to_csv(out_dir / "NQ_ES_Spread_5m_space.txt", index=False, sep=" ")

print(f"Generated Spread OHLCV with Comma, Semicolon, Tab, Space: {len(spread_df)} bars")
