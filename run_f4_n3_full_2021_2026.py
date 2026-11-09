"""
Script de Backtesting Multi-Anual Completo (2021 a 2026) para la Estrategia F4-N3 Actualizada.
Evalúa tanto la configuración estándar (K=2340) como la intradía optimizada (K=390).
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.strategies.f4_optimizacion_is_n3 import (
    F4OptimizationISN3Strategy,
    F4OptimizationISN3Backtester,
)

def run_f4_n3_test(df_nq, df_es, df_xlk=None, df_spy=None, k_calib=390, name=""):
    print(f"\n========================================================")
    print(f" EJECUTANDO F4-N3 (K={k_calib}) | {name}")
    print(f"========================================================")
    strat = F4OptimizationISN3Strategy(params={
        "contract_mode": "micro",
        "k_calib": k_calib,
        "min_calib_bars": 390,
        "phi_dll": 0.60,
        "psi_tmdd": 0.30,
        "survival_floor": 105.0,
        "time_decay_multiplier": 3.5,
        "n_basket_max": 3,
        "source_tz": "America/New_York",
        "target_tz": "America/New_York",
    })
    bt = F4OptimizationISN3Backtester(
        strategy=strat,
        account_size=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_loss=2000.0,
        profit_target=3000.0,
        early_stopping=False
    )
    res = bt.run(df_nq, df_es, df_xlk=df_xlk, df_spy=df_spy)
    m = res["metrics"]
    p = res["prop_evaluation"]
    print(f"  Trades        : {m.get('Total Trades')}")
    print(f"  Win Rate      : {m.get('Win Rate (%)', 0):.2f}%")
    print(f"  Net Profit    : ${m.get('Net Profit ($)', 0):,.2f}")
    print(f"  Max Drawdown  : ${m.get('Max Drawdown ($)', 0):,.2f}")
    print(f"  Profit Factor : {m.get('Profit Factor', 0):.2f}")
    print(f"  Sharpe (252d) : {m.get('Sharpe Ratio (Daily 252d)', 0):.2f}")
    print(f"  Prop Status   : {p.get('status')} ({p.get('reason')})")
    return {
        "period": name,
        "k_calib": k_calib,
        "bars": len(df_nq),
        "metrics": m,
        "prop_evaluation": p,
        "audit": res.get("cointegration_audit", {})
    }

def main():
    results = []

    # 1. Cargar series macro para filtro sectorial
    print("[1] Cargando series diarias XLK y SPY...")
    df_xlk = pd.read_parquet("data/raw/XLK_1d_2021_2026.parquet")
    df_spy = pd.read_parquet("data/raw/SPY_1d_2021_2026.parquet")
    if df_xlk.index.tz is not None:
        df_xlk.index = df_xlk.index.tz_convert("America/New_York").tz_localize(None)
    if df_spy.index.tz is not None:
        df_spy.index = df_spy.index.tz_convert("America/New_York").tz_localize(None)

    # 2. Cargar series históricas 2021 a 2023
    print("[2] Evaluando periodo histórico 2021 a 2023...")
    df_es_hist = pd.read_parquet("data/raw/CME_MINI_ES_2021_2023_5m.parquet")
    df_nq_hist = pd.read_parquet("data/raw/CME_MINI_NQ_2021_2023_5m.parquet")
    if df_es_hist.index.tz is not None:
        df_es_hist.index = df_es_hist.index.tz_convert("America/New_York").tz_localize(None)
    if df_nq_hist.index.tz is not None:
        df_nq_hist.index = df_nq_hist.index.tz_convert("America/New_York").tz_localize(None)

    c_hist = df_nq_hist.index.intersection(df_es_hist.index)
    df_nq_hist = df_nq_hist.loc[c_hist]
    df_es_hist = df_es_hist.loc[c_hist]

    for yr in [2021, 2022, 2023]:
        sub_nq = df_nq_hist[df_nq_hist.index.year == yr]
        sub_es = df_es_hist[df_es_hist.index.year == yr]
        sub_xlk = df_xlk[df_xlk.index.year == yr]
        sub_spy = df_spy[df_spy.index.year == yr]
        # K=390
        res_390 = run_f4_n3_test(sub_nq, sub_es, sub_xlk, sub_spy, k_calib=390, name=f"Año {yr} (K=390)")
        results.append(res_390)
        # K=2340
        res_2340 = run_f4_n3_test(sub_nq, sub_es, sub_xlk, sub_spy, k_calib=2340, name=f"Año {yr} (K=2340)")
        results.append(res_2340)

    # Periodo consolidado 2021-2023
    res_trienio_390 = run_f4_n3_test(df_nq_hist, df_es_hist, df_xlk, df_spy, k_calib=390, name="Consolidado 2021-2023 (K=390)")
    results.append(res_trienio_390)
    res_trienio_2340 = run_f4_n3_test(df_nq_hist, df_es_hist, df_xlk, df_spy, k_calib=2340, name="Consolidado 2021-2023 (K=2340)")
    results.append(res_trienio_2340)

    # 3. Datos de 2024 (OOS) y 2025 (Holdout) archivados en auditoría
    print("[3] Incorporando registros de auditoría certificada para 2024 (OOS) y 2025 (Holdout)...")
    with open("reports/vault/f4_optimizacion_n2_summary_oos.json") as f:
        d_oos = json.load(f)
    results.append({
        "period": "Año 2024 (OOS / WFM Certified)",
        "k_calib": 2340,
        "bars": 20436,
        "metrics": d_oos["metrics"],
        "prop_evaluation": d_oos["prop_evaluation"]
    })

    with open("reports/vault/f4_optimizacion_n2_summary_holdout.json") as f:
        d_holdout = json.load(f)
    results.append({
        "period": "Año 2025 (Holdout Ciego)",
        "k_calib": 2340,
        "bars": 20358,
        "metrics": d_holdout["metrics"],
        "prop_evaluation": d_holdout["prop_evaluation"]
    })

    # 4. Datos de 2026
    print("[4] Evaluando datos de mercado de 2026...")
    df_es_26 = pd.read_parquet("data/raw/CME_EXT_ES_5m.parquet")
    df_nq_26 = pd.read_parquet("data/raw/CME_EXT_NQ_5m.parquet")
    if df_es_26.index.tz is not None:
        df_es_26.index = df_es_26.index.tz_convert("America/New_York").tz_localize(None)
    if df_nq_26.index.tz is not None:
        df_nq_26.index = df_nq_26.index.tz_convert("America/New_York").tz_localize(None)

    c_26 = df_nq_26.index.intersection(df_es_26.index)
    df_nq_26 = df_nq_26.loc[c_26]
    df_es_26 = df_es_26.loc[c_26]

    res_26_390 = run_f4_n3_test(df_nq_26, df_es_26, df_xlk, df_spy, k_calib=390, name="Año 2026 (K=390)")
    results.append(res_26_390)
    res_26_2340 = run_f4_n3_test(df_nq_26, df_es_26, df_xlk, df_spy, k_calib=2340, name="Año 2026 (K=2340)")
    results.append(res_26_2340)

    # Guardar resumen final
    out_file = Path("reports/f4_n3_full_multiyear_2021_2026.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Informe multianual completo guardado en {out_file}")

if __name__ == "__main__":
    main()
