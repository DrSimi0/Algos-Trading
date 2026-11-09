"""
Análisis Mes a Mes de F4-N3: K=390 vs K=2340 en 2021-2023 y 2026.
Compara la dinámica econométrica (beta rodante, half-life tau, sigma_eq, trades, PnL y Win Rate)
para explicar matemáticamente y microestructuralmente por qué K=390 es más consistente y reactivo.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.strategies.f4_optimizacion_is_n3 import (
    F4OptimizationISN3Strategy,
    F4OptimizationISN3Backtester,
)

def evaluate_monthly():
    # 1. Cargar datos
    print("Cargando datos...")
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

    # Ejecutar el backtester completo sobre 2021-2023 para ambas calibraciones
    # y luego agrupar los trades mes a mes
    print("Ejecutando backtest completo 2021-2023 para K=390...")
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
    trades_390 = pd.DataFrame(res_390["trades"])
    signals_390 = res_390["signals"]

    print("Ejecutando backtest completo 2021-2023 para K=2340...")
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
    trades_2340 = pd.DataFrame(res_2340["trades"])
    signals_2340 = res_2340["signals"]

    # Procesar trades por mes
    if not trades_390.empty and "exit_time" in trades_390.columns:
        trades_390["exit_time"] = pd.to_datetime(trades_390["exit_time"])
        trades_390["year_month"] = trades_390["exit_time"].dt.to_period("M").astype(str)
    else:
        trades_390["year_month"] = []

    if not trades_2340.empty and "exit_time" in trades_2340.columns:
        trades_2340["exit_time"] = pd.to_datetime(trades_2340["exit_time"])
        trades_2340["year_month"] = trades_2340["exit_time"].dt.to_period("M").astype(str)
    else:
        trades_2340["year_month"] = []

    # Lista de todos los meses de 2021 a 2023
    periods = df_nq_hist.index.to_period("M").unique().astype(str).tolist()

    monthly_summary = []
    for ym in periods:
        t_390 = trades_390[trades_390["year_month"] == ym] if not trades_390.empty else pd.DataFrame()
        t_2340 = trades_2340[trades_2340["year_month"] == ym] if not trades_2340.empty else pd.DataFrame()

        # Metricas 390
        tr_390 = len(t_390)
        pnl_390 = t_390["pnl"].sum() if tr_390 > 0 else 0.0
        wr_390 = (t_390["pnl"] > 0).mean() * 100.0 if tr_390 > 0 else 0.0

        # Metricas 2340
        tr_2340 = len(t_2340)
        pnl_2340 = t_2340["pnl"].sum() if tr_2340 > 0 else 0.0
        wr_2340 = (t_2340["pnl"] > 0).mean() * 100.0 if tr_2340 > 0 else 0.0

        # Estadisticas econometricas del mes en senales (beta, tau, spread std)
        sub_sig_390 = signals_390[signals_390.index.to_period("M").astype(str) == ym]
        sub_sig_2340 = signals_2340[signals_2340.index.to_period("M").astype(str) == ym]

        beta_390_mean = float(sub_sig_390["beta"].mean()) if "beta" in sub_sig_390.columns else 0.0
        beta_390_std = float(sub_sig_390["beta"].std()) if "beta" in sub_sig_390.columns else 0.0
        tau_390_mean = float(sub_sig_390["tau_half_min"].mean()) if "tau_half_min" in sub_sig_390.columns else 0.0
        sigma_390_mean = float(sub_sig_390["sigma_eq"].mean()) if "sigma_eq" in sub_sig_390.columns else 0.0

        beta_2340_mean = float(sub_sig_2340["beta"].mean()) if "beta" in sub_sig_2340.columns else 0.0
        beta_2340_std = float(sub_sig_2340["beta"].std()) if "beta" in sub_sig_2340.columns else 0.0
        tau_2340_mean = float(sub_sig_2340["tau_half_min"].mean()) if "tau_half_min" in sub_sig_2340.columns else 0.0
        sigma_2340_mean = float(sub_sig_2340["sigma_eq"].mean()) if "sigma_eq" in sub_sig_2340.columns else 0.0

        monthly_summary.append({
            "year_month": ym,
            "trades_390": tr_390,
            "pnl_390": round(pnl_390, 2),
            "wr_390": round(wr_390, 2),
            "beta_390_mean": round(beta_390_mean, 4),
            "beta_390_std": round(beta_390_std, 4),
            "tau_390_mean": round(tau_390_mean, 2),
            "sigma_390_mean": round(sigma_390_mean, 6),
            "trades_2340": tr_2340,
            "pnl_2340": round(pnl_2340, 2),
            "wr_2340": round(wr_2340, 2),
            "beta_2340_mean": round(beta_2340_mean, 4),
            "beta_2340_std": round(beta_2340_std, 4),
            "tau_2340_mean": round(tau_2340_mean, 2),
            "sigma_2340_mean": round(sigma_2340_mean, 6),
        })

    out_path = Path("reports/monthly_f4_n3_k390_vs_k2340.json")
    with open(out_path, "w") as f:
        json.dump(monthly_summary, f, indent=2)
    print(f"Reporte mensual guardado en {out_path}")

if __name__ == "__main__":
    evaluate_monthly()
