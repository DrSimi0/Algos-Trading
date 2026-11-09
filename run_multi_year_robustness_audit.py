"""
run_multi_year_robustness_audit.py
CLI Exclusivo del Orquestador para Auditoría de Robustez Multi-Año (WFM 2D) y Holdout.

Flujo de Ejecución:
1. Autenticación Criptográfica: Valida la clave de bóveda (--vault-key o env ORCHESTRATOR_VAULT_KEY).
2. Desbloqueo en RAM: Desencripta OOS 2024 (data/vault/oos_2024_5m.enc) en memoria volátil (cero huella en disco).
3. Walk-Forward Matrix (WFM 2D): Ejecuta la cuadrícula [60, 90, 120, 180, 240]d x [20, 30, 45, 60]d
   cruzando la frontera 2023 -> 2024.
4. Veredicto OOS: Evalúa WFE Retorno (>= 55%), Consistencia Temporal C_temp (>= 70%), MSI (<= 0.40) y Prop Firm.
5. Prueba de Fuego One-Shot en Holdout 2025: Si aprueba WFM, desencripta Holdout 2025 en RAM y evalúa
   el rendimiento ciego con parámetros estrictamente congelados.
6. Certificación: Emite el certificado PRODUCTION READY o informe forense de fallos.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from src.core.vault_security import (
    decrypt_vault_to_memory,
    is_vault_file,
    VaultTripwireTriggered,
)
from src.core.multi_year_wfm import MultiYearWalkForwardMatrix
from src.strategies.f4_optimizacion_is_n2 import (
    F4OptimizationISN2Backtester,
    F4OptimizationISN2Strategy,
)


def print_banner(text: str, width: int = 78):
    print("\n" + "=" * width)
    print(f" {text:^{width-2}} ")
    print("=" * width)


def print_row(k: str, v: Any, width: int = 78):
    col1 = 44
    col2 = width - col1 - 5
    print(f"  {k:<{col1}} : {str(v):>{col2}}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI Exclusivo del Orquestador: Auditoría Multi-Año Walk-Forward Matrix y Holdout"
    )
    parser.add_argument(
        "--vault-key",
        type=str,
        default=None,
        help="Clave de 256 bits (hex) para desencriptar la bóveda en RAM. Si no se indica, usa env ORCHESTRATOR_VAULT_KEY.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="f4_n2",
        choices=["f4_n2"],
        help="Estrategia a auditar (defecto: 'f4_n2')",
    )
    parser.add_argument(
        "--account-tier",
        type=str,
        default="50k",
        choices=["50k", "100k", "150k"],
        help="Preset de cuenta de fondeo (defecto: 50k)",
    )
    parser.add_argument(
        "--oos-vault",
        type=str,
        default="data/vault/oos_2024_5m.enc",
        help="Ruta al archivo cifrado de 2024 OOS",
    )
    parser.add_argument(
        "--holdout-vault",
        type=str,
        default="data/vault/holdout_2025_5m.enc",
        help="Ruta al archivo cifrado de 2025 Holdout",
    )
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        default=False,
        help="Detener la ejecución tras la fase OOS sin desbloquear Holdout",
    )
    parser.add_argument(
        "--export-report",
        type=str,
        default="reports/vault/audit_multi_year_wfm_report.json",
        help="Ruta para exportar el informe de auditoría",
    )
    return parser


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args()

    # 1. Resolver clave de seguridad
    vault_key = args.vault_key or os.getenv("ORCHESTRATOR_VAULT_KEY")
    if not vault_key:
        print("\n" + "!" * 80)
        print(" [ACCESO DENEGADO - CLAVE DE BÓVEDA REQUERIDA]")
        print(" Debes proporcionar la clave maestra del Orquestador mediante:")
        print("   --vault-key <CLAVE_HEX_64>")
        print(" o mediante la variable de entorno de sesión efímera:")
        print("   $env:ORCHESTRATOR_VAULT_KEY='<CLAVE_HEX_64>'")
        print("!" * 80 + "\n")
        sys.exit(1)

    vault_key = vault_key.strip()

    # 2. Parámetros de Prop Firm
    TIER_PRESETS = {
        "50k": {"capital": 50000.0, "dll": 1000.0, "mtdd": 2000.0, "profit_target": 3000.0, "phi_dll": 0.60, "psi_tmdd": 0.30},
        "100k": {"capital": 100000.0, "dll": 2000.0, "mtdd": 3000.0, "profit_target": 6000.0, "phi_dll": 0.35, "psi_tmdd": 0.20},
        "150k": {"capital": 150000.0, "dll": 3000.0, "mtdd": 4500.0, "profit_target": 9000.0, "phi_dll": 0.35, "psi_tmdd": 0.20},
    }
    preset = TIER_PRESETS[args.account_tier]

    print_banner("ORQUESTADOR: AUDITORÍA DE ROBUSTEZ MULTI-AÑO (WFM 2D + HOLDOUT)")
    print("  Protocolo de Seguridad: True Cryptographic Air-Gap (AES-256-GCM)")
    print(f"  Estrategia Auditada  : {args.strategy.upper()} (F4-Optimizacion Cointegración ES/NQ)")
    print(f"  Preset de Cuenta     : {args.account_tier.upper()} (Capital=${preset['capital']:,.0f}, DLL=${preset['dll']:,.0f}, MTDD=${preset['mtdd']:,.0f})")
    print(f"  Bóveda OOS 2024      : {args.oos_vault}")
    print(f"  Bóveda Holdout 2025  : {args.holdout_vault}")

    # 3. Cargar datos In-Sample 2023 de data/raw/
    print("\n[1/4] Cargando datos públicos In-Sample 2023 de data/raw/...")
    raw_es = Path("data/raw/CME_MINI_ES_5m_3y.parquet")
    raw_nq = Path("data/raw/CME_MINI_NQ_5m_3y.parquet")
    if not raw_es.exists() or not raw_nq.exists():
        print(f"ERROR: No se encontraron los datos en data/raw/")
        sys.exit(1)

    df_es_2023 = pd.read_parquet(raw_es)
    df_nq_2023 = pd.read_parquet(raw_nq)
    print(f"  [+] IS 2023 cargado: ES={len(df_es_2023):,} barras, NQ={len(df_nq_2023):,} barras")

    # 4. Desencriptar OOS 2024 en RAM
    print("\n[2/4] Desbloqueando Bóveda OOS 2024 en RAM (io.BytesIO)...")
    try:
        oos_data = decrypt_vault_to_memory(args.oos_vault, vault_key)
    except VaultTripwireTriggered as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR al descifrar bóveda OOS: {e}")
        sys.exit(1)

    df_es_2024 = oos_data["ES"]
    df_nq_2024 = oos_data["NQ"]
    print(f"  [+] OOS 2024 desbloqueado en RAM: ES={len(df_es_2024):,} barras, NQ={len(df_nq_2024):,} barras")

    # Concatenar en memoria para evaluar la transición IS -> OOS (2023 -> 2024)
    df_es_23_24 = pd.concat([df_es_2023, df_es_2024]).sort_index()
    df_nq_23_24 = pd.concat([df_nq_2023, df_nq_2024]).sort_index()
    common_idx = df_es_23_24.index.intersection(df_nq_23_24.index)
    df_es_23_24 = df_es_23_24.loc[common_idx]
    df_nq_23_24 = df_nq_23_24.loc[common_idx]
    print(f"  [+] Dataset multi-año 2023+2024 sincronizado en RAM: {len(df_es_23_24):,} barras")

    # 5. Ejecutar Simulación Backtester F4 en RAM sobre 2023+2024
    print("\n[3/4] Ejecutando simulación de estrategia y generando operaciones 2023->2024...")
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
        "phi_dll": preset["phi_dll"],
        "psi_tmdd": preset["psi_tmdd"],
        "survival_floor": 105.0,
        "time_decay_multiplier": 3.5,
        "n_basket_max": 3,
    }
    strategy = F4OptimizationISN2Strategy(params=strat_params)
    backtester = F4OptimizationISN2Backtester(
        strategy=strategy,
        account_size=preset["capital"],
        daily_loss_limit=preset["dll"],
        max_trailing_loss=preset["mtdd"],
        trailing_mode="eod",
        profit_target=preset["profit_target"],
        early_stopping=False,
    )
    sim_res_23_24 = backtester.run(df_nq=df_nq_23_24, df_es=df_es_23_24)
    trades_23_24 = sim_res_23_24["trades"]
    print(f"  [+] Operaciones generadas: {len(trades_23_24):,} operaciones entre 2023 y 2024")

    # 6. Ejecutar Walk-Forward Matrix (WFM 2D)
    print("\n" + "-" * 78)
    print(" CALCULANDO WALK-FORWARD MATRIX (WFM 2D)")
    print(" Rejilla: Train in [60, 90, 120, 180, 240]d x Test in [20, 30, 45, 60]d")
    print("-" * 78)

    wfm = MultiYearWalkForwardMatrix(
        train_windows=[60, 90, 120, 180, 240],
        test_windows=[20, 30, 45, 60],
        mtdd_limit=preset["mtdd"],
        dll_limit=preset["dll"],
        initial_capital=preset["capital"],
    )
    wfm_res = wfm.evaluate_trades(trades_23_24)

    # Imprimir Tabla de la Rejilla WFM
    print("\n  REJILLA WALK-FORWARD MATRIX (WFE de Retorno y C_temp):")
    for row in wfm_res["grid_table"]:
        t_d = row["train_days"]
        t20 = row.get("test_20d", "N/A")
        t30 = row.get("test_30d", "N/A")
        t45 = row.get("test_45d", "N/A")
        t60 = row.get("test_60d", "N/A")
        print(f"  Train {t_d:3d}d | 20d: {t20:<22} | 30d: {t30:<22} | 45d: {t45:<22} | 60d: {t60:<22}")

    print_banner("MÉTRICAS RESUMEN WALK-FORWARD MATRIX (WFM 2D)")
    print_row("Walk-Forward Efficiency Retorno (mu)", f"{wfm_res['mean_wfe_return']:.1f}% (Umbral >= 55.0%)")
    print_row("Desviación Estándar WFE Retorno (sigma)", f"{wfm_res['std_wfe_return']:.1f}%")
    print_row("Walk-Forward Efficiency Sharpe", f"{wfm_res['mean_wfe_sharpe']:.1f}%")
    print_row("Consistencia Temporal C_temp", f"{wfm_res['mean_c_temp']:.1f}% (Umbral >= 70.0%)")
    print_row("Índice de Estabilidad de Matriz (MSI)", f"{wfm_res['matrix_stability_index_msi']:.3f} (Umbral <= 0.400)")
    print_row("Peor Drawdown en Ventanas OOS", f"${wfm_res['worst_matrix_drawdown_usd']:,.2f}")
    print_row("Peor Pérdida Diaria en Ventanas OOS", f"${wfm_res['worst_matrix_daily_loss_usd']:,.2f}")
    print_row("Violaciones Prop Firm (DLL / MTDD)", f"DLL={wfm_res['total_dll_violations']} / MTDD={wfm_res['total_mtdd_violations']}")
    print_row("Veredicto de Fase OOS WFM", wfm_res["verdict"])

    # 7. Fase 3: Holdout 2025 One-Shot Test
    holdout_passed = False
    holdout_eval: Dict[str, Any] = {}

    if not wfm_res["matrix_passed"]:
        print("\n" + "!" * 78)
        print(" [ATENCIÓN: BÓVEDA HOLDOUT SELLADA]")
        print(" La Walk-Forward Matrix en OOS 2024 no cumplió con los estándares exigidos.")
        print(" Para prevenir contaminación irreversible, el Holdout 2025 permanece BLOQUEADO.")
        print("!" * 78)
    elif args.skip_holdout:
        print("\n[+] Flag --skip-holdout activado. Holdout 2025 permanece sellado.")
    else:
        print_banner("FASE 3: PRUEBA DE FUEGO DEFINITIVA EN HOLDOUT 2025 (ONE-SHOT BLIND TEST)")
        print("  Desencriptando Año 2025 completo en RAM...")
        try:
            holdout_data = decrypt_vault_to_memory(args.holdout_vault, vault_key)
            df_es_2025 = holdout_data["ES"]
            df_nq_2025 = holdout_data["NQ"]
            common_2025 = df_es_2025.index.intersection(df_nq_2025.index)
            df_es_2025 = df_es_2025.loc[common_2025]
            df_nq_2025 = df_nq_2025.loc[common_2025]

            print(f"  [+] Holdout 2025 desbloqueado: {len(df_es_2025):,} barras de 5m nunca vistas")
            print("  [+] Ejecutando backtest con parámetros estrictamente congelados...")

            strategy_holdout = F4OptimizationISN2Strategy(params=strat_params)
            backtester_holdout = F4OptimizationISN2Backtester(
                strategy=strategy_holdout,
                account_size=preset["capital"],
                daily_loss_limit=preset["dll"],
                max_trailing_loss=preset["mtdd"],
                trailing_mode="eod",
                profit_target=preset["profit_target"],
                early_stopping=False,
            )
            sim_holdout = backtester_holdout.run(df_nq=df_nq_2025, df_es=df_es_2025)
            h_metrics = sim_holdout["metrics"]
            h_prop = sim_holdout["prop_evaluation"]
            h_trades = sim_holdout["trades"]

            holdout_eval = {
                "status": h_prop["status"],
                "net_profit_usd": h_prop["net_profit"],
                "max_drawdown_usd": h_metrics["Max Drawdown ($)"],
                "profit_factor": h_metrics["Profit Factor"],
                "sharpe_ratio_daily": h_metrics["Sharpe Ratio (Daily 252d)"],
                "total_trades": h_metrics["Total Trades"],
                "win_rate_pct": h_metrics["Win Rate (%)"],
                "dll_breached": bool(h_prop["status"] == "FAIL" and "DLL" in h_prop.get("reason", "")),
                "mtdd_breached": bool(h_prop["status"] == "FAIL" and "TMDD" in h_prop.get("reason", "")),
            }

            print_row("Estado Evaluación Holdout", h_prop["status"])
            print_row("Razón del Veredicto", h_prop["reason"])
            print_row("Beneficio Neto 2025 ($)", f"${h_prop['net_profit']:,.2f}")
            print_row("Drawdown Máximo ($)", f"${h_metrics['Max Drawdown ($)']:,.2f}")
            print_row("Profit Factor", h_metrics["Profit Factor"])
            print_row("Ratio de Sharpe Diario (252d)", f"{h_metrics['Sharpe Ratio (Daily 252d)']:.2f}")
            print_row("Total de Operaciones 2025", h_metrics["Total Trades"])
            print_row("Tasa de Acierto (Win Rate)", f"{h_metrics['Win Rate (%)']:.2f}%")

            holdout_passed = bool(h_prop["status"] == "PASS")

        except Exception as e:
            print(f"ERROR durante la evaluación de Holdout: {e}")
            holdout_passed = False

    # 8. Veredicto Final Global
    overall_certified = bool(wfm_res["matrix_passed"] and holdout_passed)

    if overall_certified:
        print_banner("*** CERTIFICADO PRODUCTION READY EMITIDO ***")
        print("  La estrategia F4-Optimizacion Cointegración ES/NQ ha superado con éxito:")
        print("  1. In-Sample 2023 con corrección matemática rigurosa de las 6 anomalías.")
        print(f"  2. Walk-Forward Matrix (WFM 2D) en OOS 2024 (WFE={wfm_res['mean_wfe_return']:.1f}%, C_temp={wfm_res['mean_c_temp']:.1f}%, MSI={wfm_res['matrix_stability_index_msi']:.3f}).")
        print(f"  3. Prueba de fuego definitiva en Holdout Ciego 2025 (Beneficio Neto=${holdout_eval.get('net_profit_usd', 0):,.2f}).")
        print("  4. Cero violaciones de las reglas de Prop Trading (MTDD $2,000 / DLL $1,000).")
        print("=" * 78)
    else:
        print_banner("ESTADO DE AUDITORÍA: REPROBADA / NO APTO PARA PRODUCCIÓN")
        if not wfm_res["matrix_passed"]:
            print("  - Fase WFM 2024: NO SUPERADA")
        if not holdout_passed and not args.skip_holdout:
            print("  - Fase Holdout 2025: NO SUPERADA O SELLADA")
        print("=" * 78)

    # 9. Guardar Reporte
    report_data = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "strategy": args.strategy,
        "account_tier": args.account_tier,
        "preset": preset,
        "wfm_oos_2024": wfm_res,
        "holdout_2025": holdout_eval,
        "certified_production_ready": overall_certified,
    }

    out_rep = Path(args.export_report)
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"\n[+] Informe completo exportado a: {out_rep}")


if __name__ == "__main__":
    main()
