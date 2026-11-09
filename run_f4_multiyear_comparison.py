"""
Script de Testeo Multianual por Año (2021, 2022, 2023, 2026)
Compara la estrategia F4-N2 Base vs F4-N2 con Filtro Horario (< 15:40)
Temporalidad: 5 minutos (5m)
Cuenta Prop Trading: $50,000 USD (DLL $1,000, TMDD $2,000, PT $3,000)
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.strategies.f4_optimizacion_is_n2 import (
    F4OptimizationISN2Strategy,
    F4OptimizationISN2Backtester,
)

def run_f4_simulation(df_nq: pd.DataFrame, df_es: pd.DataFrame, time_filter: bool = False, k_calib: int = 2340):
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
        "k_calib": k_calib,
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
    
    if time_filter:
        orig_gen = strategy.generate_signals
        def filtered_gen(df_nq_in, df_es_in=None):
            sigs = orig_gen(df_nq_in) if df_es_in is None else orig_gen(df_nq_in, df_es_in)
            mask = [
                (ts.hour > 9 or (ts.hour == 9 and ts.minute >= 30)) and
                (ts.hour < 15 or (ts.hour == 15 and ts.minute < 40))
                for ts in sigs.index
            ]
            return sigs.where(pd.Series(mask, index=sigs.index), 0)
        strategy.generate_signals = filtered_gen

    backtester = F4OptimizationISN2Backtester(
        strategy=strategy,
        account_size=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_loss=2000.0,
        profit_target=3000.0,
        early_stopping=False
    )
    res = backtester.run(df_nq=df_nq, df_es=df_es)
    return res

def main():
    years = [
        {"year": 2021, "es": "data/raw/CME_MINI_ES_2021_5m.parquet", "nq": "data/raw/CME_MINI_NQ_2021_5m.parquet", "k": 2340},
        {"year": 2022, "es": "data/raw/CME_MINI_ES_2022_5m.parquet", "nq": "data/raw/CME_MINI_NQ_2022_5m.parquet", "k": 2340},
        {"year": 2023, "es": "data/raw/CME_MINI_ES_2023_5m.parquet", "nq": "data/raw/CME_MINI_NQ_2023_5m.parquet", "k": 2340},
        {"year": 2026, "es": "data/raw/CME_EXT_ES_5m.parquet", "nq": "data/raw/CME_EXT_NQ_5m.parquet", "k": 390},
    ]

    all_results = []

    print("=" * 80)
    print("BACKTESTING COMPARATIVO POR AÑO: F4-N2 BASE VS CON FILTRO DE TIEMPO (< 15:40)")
    print("=" * 80)

    for yinfo in years:
        yr = yinfo["year"]
        k_val = yinfo["k"]
        print(f"\n>>> PROCESANDO AÑO {yr} (k_calib={k_val}) <<<")
        
        df_es = pd.read_parquet(yinfo["es"])
        df_nq = pd.read_parquet(yinfo["nq"])

        if df_es.index.tz is not None:
            df_es.index = df_es.index.tz_convert("America/New_York").tz_localize(None)
        if df_nq.index.tz is not None:
            df_nq.index = df_nq.index.tz_convert("America/New_York").tz_localize(None)

        common_idx = df_nq.index.intersection(df_es.index)
        df_nq = df_nq.loc[common_idx]
        df_es = df_es.loc[common_idx]

        print(f"Barras sincronizadas 5m: {len(common_idx):,} | Rango: {common_idx[0]} a {common_idx[-1]}")

        # 1. Base
        res_base = run_f4_simulation(df_nq, df_es, time_filter=False, k_calib=k_val)
        m_base = res_base["metrics"]
        p_base = res_base["prop_evaluation"]

        # 2. Con Filtro de Tiempo
        res_tf = run_f4_simulation(df_nq, df_es, time_filter=True, k_calib=k_val)
        m_tf = res_tf["metrics"]
        p_tf = res_tf["prop_evaluation"]

        print(f"\n--- AÑO {yr} RESULTADOS ---")
        print(f"  BASE            : Trades={m_base['Total Trades']}, WinRate={m_base.get('Win Rate (%)',0):.2f}%, PnL=${m_base.get('Net Profit ($)',0):,.2f}, MaxDD=${m_base.get('Max Drawdown ($)',0):,.2f}, Sharpe={m_base.get('Sharpe Ratio (Daily 252d)',0):.2f}, PF={m_base.get('Profit Factor',0):.2f}, PropStatus={p_base.get('status')}")
        print(f"  FILTRO HORARIO  : Trades={m_tf['Total Trades']}, WinRate={m_tf.get('Win Rate (%)',0):.2f}%, PnL=${m_tf.get('Net Profit ($)',0):,.2f}, MaxDD=${m_tf.get('Max Drawdown ($)',0):,.2f}, Sharpe={m_tf.get('Sharpe Ratio (Daily 252d)',0):.2f}, PF={m_tf.get('Profit Factor',0):.2f}, PropStatus={p_tf.get('status')}")

        all_results.append({
            "year": yr,
            "k_calib": k_val,
            "bars": len(common_idx),
            "start": str(common_idx[0]),
            "end": str(common_idx[-1]),
            "base": {
                "trades": m_base["Total Trades"],
                "win_rate": m_base.get("Win Rate (%)", 0),
                "net_pnl": m_base.get("Net Profit ($)", 0),
                "return_pct": m_base.get("Total Return (%)", 0),
                "max_drawdown": m_base.get("Max Drawdown ($)", 0),
                "sharpe_daily": m_base.get("Sharpe Ratio (Daily 252d)", 0),
                "sortino_daily": m_base.get("Sortino Ratio (Daily 252d)", 0),
                "profit_factor": m_base.get("Profit Factor", 0),
                "prop_status": p_base.get("status"),
                "prop_reason": p_base.get("reason"),
            },
            "time_filtered": {
                "trades": m_tf["Total Trades"],
                "win_rate": m_tf.get("Win Rate (%)", 0),
                "net_pnl": m_tf.get("Net Profit ($)", 0),
                "return_pct": m_tf.get("Total Return (%)", 0),
                "max_drawdown": m_tf.get("Max Drawdown ($)", 0),
                "sharpe_daily": m_tf.get("Sharpe Ratio (Daily 252d)", 0),
                "sortino_daily": m_tf.get("Sortino Ratio (Daily 252d)", 0),
                "profit_factor": m_tf.get("Profit Factor", 0),
                "prop_status": p_tf.get("status"),
                "prop_reason": p_tf.get("reason"),
            }
        })

    # Guardar en reports/f4_multiyear_comparison.json
    out_path = Path("reports/f4_multiyear_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Resultados completos guardados en {out_path}")

if __name__ == "__main__":
    main()
