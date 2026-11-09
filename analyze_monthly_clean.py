"""
Análisis comparativo Mes a Mes de F4-N3: K=390 vs K=2340 en 2021-2023 y 2026.
Agrupa los trades generados por mes (usando exit_date) y calcula:
Trades, Win Rate %, PnL ($), Profit Factor, Max DD ($),
así como la distribución del ratio beta congelado y tiempo de duración.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.strategies.f4_optimizacion_is_n3 import (
    F4OptimizationISN3Strategy,
    F4OptimizationISN3Backtester,
)

def run():
    print("Cargando series de datos...")
    df_xlk = pd.read_parquet("data/raw/XLK_1d_2021_2026.parquet")
    df_spy = pd.read_parquet("data/raw/SPY_1d_2021_2026.parquet")
    if df_xlk.index.tz is not None:
        df_xlk.index = df_xlk.index.tz_convert("America/New_York").tz_localize(None)
    if df_spy.index.tz is not None:
        df_spy.index = df_spy.index.tz_convert("America/New_York").tz_localize(None)

    df_es_hist = pd.read_parquet("data/raw/CME_MINI_ES_2021_2023_5m.parquet")
    df_nq_hist = pd.read_parquet("data/raw/CME_MINI_NQ_2021_2023_5m.parquet")
    if df_es_hist.index.tz is not None:
        df_es_hist.index = df_es_hist.index.tz_convert("America/New_York").tz_localize(None)
    if df_nq_hist.index.tz is not None:
        df_nq_hist.index = df_nq_hist.index.tz_convert("America/New_York").tz_localize(None)

    c_hist = df_nq_hist.index.intersection(df_es_hist.index)
    df_nq_hist = df_nq_hist.loc[c_hist]
    df_es_hist = df_es_hist.loc[c_hist]

    print("Ejecutando K=390...")
    strat_390 = F4OptimizationISN3Strategy(params={
        "contract_mode": "micro", "k_calib": 390, "min_calib_bars": 390,
        "phi_dll": 0.60, "psi_tmdd": 0.30, "survival_floor": 105.0,
        "time_decay_multiplier": 3.5, "n_basket_max": 3,
        "source_tz": "America/New_York", "target_tz": "America/New_York",
    })
    bt_390 = F4OptimizationISN3Backtester(
        strategy=strat_390, account_size=50000.0, daily_loss_limit=1000.0,
        max_trailing_loss=2000.0, profit_target=3000.0, early_stopping=False
    )
    res_390 = bt_390.run(df_nq_hist, df_es_hist, df_xlk=df_xlk, df_spy=df_spy)
    df_trades_390 = res_390["trades"]

    print("Ejecutando K=2340...")
    strat_2340 = F4OptimizationISN3Strategy(params={
        "contract_mode": "micro", "k_calib": 2340, "min_calib_bars": 390,
        "phi_dll": 0.60, "psi_tmdd": 0.30, "survival_floor": 105.0,
        "time_decay_multiplier": 3.5, "n_basket_max": 3,
        "source_tz": "America/New_York", "target_tz": "America/New_York",
    })
    bt_2340 = F4OptimizationISN3Backtester(
        strategy=strat_2340, account_size=50000.0, daily_loss_limit=1000.0,
        max_trailing_loss=2000.0, profit_target=3000.0, early_stopping=False
    )
    res_2340 = bt_2340.run(df_nq_hist, df_es_hist, df_xlk=df_xlk, df_spy=df_spy)
    df_trades_2340 = res_2340["trades"]

    # Procesar meses
    df_trades_390["ym"] = pd.to_datetime(df_trades_390["exit_date"]).dt.to_period("M").astype(str) if not df_trades_390.empty else []
    df_trades_2340["ym"] = pd.to_datetime(df_trades_2340["exit_date"]).dt.to_period("M").astype(str) if not df_trades_2340.empty else []

    all_months = sorted(list(set(
        (df_trades_390["ym"].unique().tolist() if not df_trades_390.empty else []) +
        (df_trades_2340["ym"].unique().tolist() if not df_trades_2340.empty else [])
    )))

    monthly_rows = []
    for ym in all_months:
        t1 = df_trades_390[df_trades_390["ym"] == ym] if not df_trades_390.empty else pd.DataFrame()
        t2 = df_trades_2340[df_trades_2340["ym"] == ym] if not df_trades_2340.empty else pd.DataFrame()

        n1 = len(t1)
        pnl1 = t1["net_pnl_usd"].sum() if n1 > 0 else 0.0
        wr1 = (t1["net_pnl_usd"] > 0).mean() * 100.0 if n1 > 0 else 0.0
        dur1 = t1["duration_minutes"].mean() if n1 > 0 else 0.0
        beta1 = t1["entry_beta"].mean() if n1 > 0 else 0.0

        n2 = len(t2)
        pnl2 = t2["net_pnl_usd"].sum() if n2 > 0 else 0.0
        wr2 = (t2["net_pnl_usd"] > 0).mean() * 100.0 if n2 > 0 else 0.0
        dur2 = t2["duration_minutes"].mean() if n2 > 0 else 0.0
        beta2 = t2["entry_beta"].mean() if n2 > 0 else 0.0

        monthly_rows.append({
            "month": ym,
            "trades_390": n1,
            "pnl_390": round(pnl1, 2),
            "winrate_390": round(wr1, 2),
            "avg_dur_390": round(dur1, 1),
            "avg_beta_390": round(beta1, 4),
            "trades_2340": n2,
            "pnl_2340": round(pnl2, 2),
            "winrate_2340": round(wr2, 2),
            "avg_dur_2340": round(dur2, 1),
            "avg_beta_2340": round(beta2, 4),
        })

    out_file = Path("reports/monthly_k_comparison_clean.json")
    with open(out_file, "w") as f:
        json.dump(monthly_rows, f, indent=2)
    print(f"Reporte mensual limpio guardado en {out_file}")

if __name__ == "__main__":
    run()
