"""
Script de backtest para datos de 2026.
Evalúa F4-N2 Base, F4-N2 con Filtro Horario (< 15:40), y F3 en las series disponibles de 2026.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.strategies.f4_optimizacion_is_n2 import (
    F4OptimizationISN2Strategy,
    F4OptimizationISN2Backtester,
)

def run_f4_on_dataset(es_path: str, nq_path: str, time_filter: str = "none", label: str = "2026"):
    print(f"\n==========================================")
    print(f" Ejecutando F4-N2 [{time_filter.upper()}] en {label}")
    print(f"==========================================")
    df_es = pd.read_parquet(es_path)
    df_nq = pd.read_parquet(nq_path)

    # Convert tz if present
    if df_es.index.tz is not None:
        df_es.index = df_es.index.tz_convert("America/New_York").tz_localize(None)
    if df_nq.index.tz is not None:
        df_nq.index = df_nq.index.tz_convert("America/New_York").tz_localize(None)

    common_idx = df_nq.index.intersection(df_es.index)
    df_nq = df_nq.loc[common_idx]
    df_es = df_es.loc[common_idx]

    print(f"Rango de fechas: {common_idx[0]} a {common_idx[-1]}")
    print(f"Total barras sincronizadas: {len(common_idx):,}")

    strat_params = {
        "contract_mode": "micro",
        "use_kalman": False,
        "unify_beta_ols": True,
        "max_coprime_nq": 7,
        "allow_sub_baskets": True,
        "dynamic_basket_ratio": True,
        "s_open": 1.25,
        "s_close": 0.25,
        "s_stop": 3.25,
        "k_calib": 2340,
        "min_calib_bars": 390,
        "phi_dll": 0.60,
        "psi_tmdd": 0.30,
        "survival_floor": 105.0,
        "time_decay_multiplier": 3.5,
        "n_basket_max": 3,
        "source_tz": "America/New_York",
        "target_tz": "America/New_York",
    }
    strategy = F4OptimizationISN2Strategy(params=strat_params)
    backtester = F4OptimizationISN2Backtester(
        strategy=strategy,
        account_size=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_loss=2000.0,
        profit_target=3000.0,
        early_stopping=False
    )

    # Patch generate_signals if time filter active
    orig_generate_signals = strategy.generate_signals

    def filtered_generate_signals(df_nq_in, df_es_in):
        signals = orig_generate_signals(df_nq_in, df_es_in)
        if time_filter == "sqx_1540":
            # SQX filter: 09:30 to 15:40 NY
            valid_mask = [
                (ts.hour > 9 or (ts.hour == 9 and ts.minute >= 30)) and
                (ts.hour < 15 or (ts.hour == 15 and ts.minute < 40))
                for ts in signals.index
            ]
            valid_series = pd.Series(valid_mask, index=signals.index)
            signals = signals.where(valid_series, 0)
        elif time_filter == "rth_1515":
            valid_mask = [
                (ts.hour > 9 or (ts.hour == 9 and ts.minute >= 30)) and
                (ts.hour < 15 or (ts.hour == 15 and ts.minute <= 15))
                for ts in signals.index
            ]
            valid_series = pd.Series(valid_mask, index=signals.index)
            signals = signals.where(valid_series, 0)
        return signals

    if time_filter != "none":
        strategy.generate_signals = filtered_generate_signals

    res = backtester.run(df_nq=df_nq, df_es=df_es)
    m = res["metrics"]
    trades = res["trades"]

    print(f"Total Trades        : {len(trades)}")
    print(f"Net PnL             : ${m.get('net_pnl', 0):,.2f}")
    print(f"Return              : {m.get('return_pct', 0):.2f}%")
    print(f"Win Rate            : {m.get('win_rate', 0)*100:.2f}%")
    print(f"Profit Factor       : {m.get('profit_factor', 0):.2f}")
    print(f"Sharpe Ratio        : {m.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio       : {m.get('sortino_ratio', 0):.2f}")
    print(f"Max Drawdown        : ${m.get('max_drawdown', 0):,.2f} ({m.get('max_drawdown_pct', 0):.2f}%)")
    print(f"Target Reached      : {m.get('profit_target_reached')}")
    print(f"Violó TMDD ($2,000) : {m.get('breached_tmdd')}")
    print(f"Violó DLL ($1,000)  : {m.get('breached_dll')}")

    return {
        "label": label,
        "time_filter": time_filter,
        "start": str(common_idx[0]),
        "end": str(common_idx[-1]),
        "bars": len(common_idx),
        "trades_count": len(trades),
        "metrics": m,
    }


if __name__ == "__main__":
    results = []

    # 1. CME_EXT 5m (Aug 2026 - Nov 2026, 13,877 bars)
    print("\n>>> DATASET 1: CME_EXT (13,877 barras de 2026) <<<")
    res_ext_base = run_f4_on_dataset(
        "data/raw/CME_EXT_ES_5m.parquet",
        "data/raw/CME_EXT_NQ_5m.parquet",
        time_filter="none",
        label="CME_EXT_2026 (Base)"
    )
    results.append(res_ext_base)

    res_ext_sqx = run_f4_on_dataset(
        "data/raw/CME_EXT_ES_5m.parquet",
        "data/raw/CME_EXT_NQ_5m.parquet",
        time_filter="sqx_1540",
        label="CME_EXT_2026 (Filtro Horario < 15:40)"
    )
    results.append(res_ext_sqx)

    # 2. CME_MINI 2026 (Oct 2026 - Nov 2026, 5,000 bars)
    print("\n>>> DATASET 2: CME_MINI 2026 (5,000 barras recientes de 2026) <<<")
    res_mini_base = run_f4_on_dataset(
        "data/raw/CME_MINI_ES_2026_5m.parquet",
        "data/raw/CME_MINI_NQ_2026_5m.parquet",
        time_filter="none",
        label="CME_MINI_2026 (Base)"
    )
    results.append(res_mini_base)

    res_mini_sqx = run_f4_on_dataset(
        "data/raw/CME_MINI_ES_2026_5m.parquet",
        "data/raw/CME_MINI_NQ_2026_5m.parquet",
        time_filter="sqx_1540",
        label="CME_MINI_2026 (Filtro Horario < 15:40)"
    )
    results.append(res_mini_sqx)

    # Export results JSON
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/backtest_2026_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n[OK] Resumen exportado a reports/backtest_2026_summary.json")
