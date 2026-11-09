"""
run_robustness_test.py
Batería Cuantitativa de Pruebas de Robustez y Sistema de "Último Veredicto"
para Trading de Futuros CME (ES, NQ, MES, MNQ) en Cuentas de Fondeo (Topstep / Apex).

Evalúa:
1. Simulación de Monte Carlo (Reshuffling, Bootstrapping y Session Block Bootstrap).
2. Racha Teórica de Erdős-Rényi Corregida.
3. Deflated Sharpe Ratio (DSR - Bailey & López de Prado).
4. Pruebas de Estrés Hostiles What-If (Slippage 2x/3x, Top-3 Drop, Win-Pruning, Gap en SL).
5. Walk-Forward Matrix (WFM - Robert Pardo) dentro del período In-Sample 2023.
6. Scorecard de 100 Puntos y 6 Compuertas Duras (Hard Gates) para emisión del Veredicto Final.

Uso:
    python run_robustness_test.py --trades reports/f3_optimization_n3_trades.csv --symbol ES --prop-firm topstep_50k
    python run_robustness_test.py --trades reports/f4_optimizacion_is_n2_trades.csv --symbol ES --prop-firm topstep_50k
    python run_robustness_test.py --strategy f3_n2 --symbol ES --simulations 2500
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

# Asegurar codificación UTF-8 en consola de Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.core.monte_carlo import MonteCarloSimulator, run_session_block_bootstrap, compute_erdos_renyi_expected_loss_streak
from src.core.robustness import (
    compute_deflated_sharpe_ratio,
    run_what_if_stress_suite,
    PropFirmRobustnessScorecard,
    evaluate_parameter_stability,
)
from src.core.walk_forward import WalkForwardMatrix


PROP_FIRM_PROFILES = {
    "topstep_50k": {
        "name": "Topstep $50k Standard",
        "capital": 50000.0,
        "mtdd": 2000.0,
        "dll": 1000.0,
        "target": 3000.0,
    },
    "apex_50k": {
        "name": "Apex Trader Funding $50k",
        "capital": 50000.0,
        "mtdd": 2500.0,
        "dll": 1000.0,
        "target": 3000.0,
    },
}

STRATEGY_TRADES_MAP = {
    "f1": "reports/f1_trades.csv",
    "f2": "reports/f2_optimizacion_is_n1_trades.csv",
    "orb": "reports/orb_trades.csv",
    "f3": "reports/f3_optimization_n3_trades.csv",
    "f3_n3": "reports/f3_optimization_n3_trades.csv",
    "f3_n2": "reports/f3_optimization_n2_trades.csv",
    "f3_n1": "reports/f3_optimization_n1_trades.csv",
    "f4": "reports/f4_optimizacion_is_n2_trades.csv",
    "f4_n2": "reports/f4_optimizacion_is_n2_trades.csv",
    "f4_n1": "reports/f4_optimizacion_is_n1_trades.csv",
    "f5": "reports/f5_opt_trades.csv",
    "f7": "reports/f7_meta_labeling_trades.csv",
    "f8": "reports/har_rv_j_opt_n1_trades.csv",
    "f9": "reports/f9_trades.csv",
}


def print_banner(text: str, width: int = 86):
    print("\n" + "=" * width)
    print(f" {text:^{width-2}} ")
    print("=" * width)


def print_row(label: str, val: str, width: int = 86):
    col1 = 48
    col2 = width - col1 - 5
    print(f"  {label:<{col1}} : {val:>{col2}}")


def load_trades_dataframe(path: Path) -> pd.DataFrame:
    """Carga y valida el archivo CSV de operaciones."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de trades en: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"El archivo {path} no contiene filas de datos.")

    # Restricción de Cuarentena: Prohibir archivos explícitos de OOS o Holdout
    filename_lower = path.name.lower()
    if "oos" in filename_lower or "holdout" in filename_lower:
        raise PermissionError(
            f"VIOLACIÓN DE CUARENTENA: El archivo '{path.name}' corresponde a Out-of-Sample o Holdout. "
            "Las pruebas de robustez deben ejecutarse estrictamente dentro de la partición In-Sample."
        )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Suite Cuantitativa de Pruebas de Robustez y Veredicto para Prop Trading"
    )
    parser.add_argument("--trades", type=str, default=None,
                        help="Ruta al archivo CSV con registro de operaciones (ej. reports/f3_optimization_n3_trades.csv)")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=list(STRATEGY_TRADES_MAP.keys()),
                        help="Estrategia preconfigurada (ej. f3_n3, f4_n2, f7)")
    parser.add_argument("--symbol", type=str, default="ES",
                        choices=["ES", "MES", "NQ", "MNQ"],
                        help="Símbolo principal del futuro CME (def: ES)")
    parser.add_argument("--prop-firm", type=str, default="topstep_50k",
                        choices=list(PROP_FIRM_PROFILES.keys()),
                        help="Perfil de reglas de la prop firm (def: topstep_50k)")
    parser.add_argument("--simulations", type=int, default=2500,
                        help="Número de iteraciones de Monte Carlo (def: 2,500)")
    parser.add_argument("--export-json", type=str, default=None,
                        help="Ruta para exportar el informe completo en formato JSON")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla estocástica para reproducibilidad (def: 42)")

    args = parser.parse_args()

    # 1. Resolver archivo de trades
    trades_path: Optional[Path] = None
    strategy_label = "Personalizada"

    if args.trades:
        trades_path = Path(args.trades)
        strategy_label = trades_path.stem.upper()
    elif args.strategy:
        strategy_key = args.strategy.lower()
        trades_path = Path(STRATEGY_TRADES_MAP[strategy_key])
        strategy_label = strategy_key.upper()
    else:
        # Fallback a F3 N3 si existe
        default_candidate = Path("reports/f3_optimization_n3_trades.csv")
        if default_candidate.exists():
            trades_path = default_candidate
            strategy_label = "F3_OPTIMIZATION_N3"
        else:
            print("Error: Debe especificar '--trades <ruta>' o '--strategy <nombre>'.")
            sys.exit(1)

    trades_df = load_trades_dataframe(trades_path)
    profile = PROP_FIRM_PROFILES[args.prop_firm]

    # Extraer columna de PnL
    pnl_col = None
    for col in ["net_pnl_usd", "net_pnl", "net_profit", "pnl", "PnL", "Net_PnL"]:
        if col in trades_df.columns:
            pnl_col = col
            break

    if pnl_col is None:
        print(f"Error: No se encontró columna de PnL en {trades_path}. Columnas: {list(trades_df.columns)}")
        sys.exit(1)

    pnl_series = trades_df[pnl_col].dropna().astype(float)
    total_trades = len(pnl_series)
    net_profit_total = float(pnl_series.sum())
    win_rate = float((pnl_series > 0).mean() * 100.0)
    loss_rate = float((pnl_series < 0).mean())

    print_banner(f"SUITE DE PRUEBAS DE ROBUSTEZ Y ÚLTIMO VEREDICTO: {strategy_label}")
    print_row("Archivo de Operaciones", str(trades_path))
    print_row("Perfil de Evaluación Prop Firm", f"{profile['name']} (${profile['capital']:,.0f})")
    print_row("Max Trailing Drawdown Permitido", f"${profile['mtdd']:,.2f}")
    print_row("Daily Loss Limit Permitido", f"${profile['dll']:,.2f}")
    print_row("Profit Target Requerido", f"${profile['target']:,.2f}")
    print_row("Total Operaciones Analizadas", f"{total_trades:,} trades")
    print_row("Beneficio Neto Histórico", f"${net_profit_total:+,.2f} USD")
    print_row("Win Rate Histórico", f"{win_rate:.1f}%")

    # -------------------------------------------------------------------------
    # PILAR 1: MONTE CARLO & ESTIMACIÓN ESTOCÁSTICA
    # -------------------------------------------------------------------------
    print_banner("PILAR 1: SIMULACIÓN DE MONTE CARLO & RIESGO DE RUINA")

    mc_sim = MonteCarloSimulator(
        trades_pnl=pnl_series,
        initial_capital=profile["capital"],
        daily_loss_limit=profile["dll"],
        max_trailing_drawdown=profile["mtdd"],
        profit_target=profile["target"],
        seed=args.seed,
    )

    res_reshuffle = mc_sim.run_reshuffle(n_simulations=args.simulations)
    res_bootstrap = mc_sim.run_bootstrap(n_simulations=args.simulations)
    res_session = run_session_block_bootstrap(
        trades_df=trades_df,
        initial_capital=profile["capital"],
        daily_loss_limit=profile["dll"],
        max_trailing_drawdown=profile["mtdd"],
        profit_target=profile["target"],
        n_simulations=args.simulations,
        seed=args.seed,
    )

    erdos_renyi_streak = compute_erdos_renyi_expected_loss_streak(total_trades, loss_rate)

    print("\n  [A. Reshuffling (Permutación sin Reemplazo - Riesgo de Secuencia)]")
    print_row("Riesgo de Ruina (MTDD)", f"{res_reshuffle['risk_of_ruin_pct']:.2f}%")
    print_row("Probabilidad de Target antes de Ruina", f"{res_reshuffle['target_before_ruin_pct']:.2f}%")
    print_row("P95 Drawdown Proyectado", f"${res_reshuffle['drawdown_stats']['p95_dd_usd']:,.2f}")
    print_row("Peor Drawdown Absoluto", f"${res_reshuffle['drawdown_stats']['worst_case_dd_usd']:,.2f}")

    print("\n  [B. Bootstrapping (Muestreo con Reemplazo - Rachas Adversas)]")
    print_row("Riesgo de Ruina (MTDD)", f"{res_bootstrap['risk_of_ruin_pct']:.2f}%")
    print_row("Probabilidad de Target antes de Ruina", f"{res_bootstrap['target_before_ruin_pct']:.2f}%")
    print_row("P95 Drawdown Proyectado", f"${res_bootstrap['drawdown_stats']['p95_dd_usd']:,.2f}")
    print_row("CVaR 95% Drawdown", f"${res_bootstrap['drawdown_stats']['cvar_95_dd_usd']:,.2f}")

    print("\n  [C. Session Block Bootstrap (Remuestreo por Sesión - Auditoría DLL)]")
    print_row("Total de Sesiones Detectadas", f"{res_session['total_sessions']} días")
    print_row("Violación Daily Loss Limit (DLL -$1,000)", f"{res_session['dll_violation_pct']:.2f}%")
    print_row("Riesgo de Ruina Total (DLL o MTDD)", f"{res_session['risk_of_ruin_pct']:.2f}%")
    print_row("Peor Pérdida Diaria P95", f"${res_session['daily_loss_stats']['p95_worst_day_usd']:,.2f}")
    print_row("Peor Día Absoluto Simulado", f"${res_session['daily_loss_stats']['worst_day_usd']:,.2f}")

    print("\n  [D. Teorema de Erdős-Rényi Corregido]")
    print_row("Racha Teórica Máxima Esperada (E[L_max])", f"{erdos_renyi_streak:.1f} pérdidas consecutivas")

    # -------------------------------------------------------------------------
    # PILAR 2: ESTABILIDAD DE PARÁMETROS & DEFLATED SHARPE RATIO (DSR)
    # -------------------------------------------------------------------------
    print_banner("PILAR 2: ESTABILIDAD DE PARÁMETROS & DEFLATED SHARPE RATIO (DSR)")

    # DSR penalizando 50 combinaciones de optimización típicas de la fase IS
    dsr_results = compute_deflated_sharpe_ratio(
        trades_or_returns=pnl_series,
        num_trials=50,
        trades_per_year=252.0,
    )

    print_row("Sharpe Observado por Trade", f"{dsr_results['observed_sr']:.4f}")
    print_row("Sharpe Anualizado Estimado", f"{dsr_results['observed_sr_annualized']:.2f}")
    print_row("Benchmark Sharpe Penalizado (K=50)", f"{dsr_results['benchmark_sr_annualized']:.2f}")
    print_row("Asimetría (Skewness)", f"{dsr_results['skewness']:+.3f}")
    print_row("Curtosis (Pearson normal=3.0)", f"{dsr_results['kurtosis']:.3f}")
    print_row("Deflated Sharpe Ratio (DSR)", f"{dsr_results['dsr']:.4f} (p-value: {dsr_results['p_value']:.4f})")
    print_row("Veredicto DSR", dsr_results["verdict"])

    # -------------------------------------------------------------------------
    # PILAR 3: WALK-FORWARD MATRIX (WFM) EN IN-SAMPLE 2023
    # -------------------------------------------------------------------------
    print_banner("PILAR 3: WALK-FORWARD MATRIX (WFM) - CONFINADO EN IN-SAMPLE 2023")

    wfm = WalkForwardMatrix(
        train_windows=[60, 90, 120],
        test_windows=[20, 30, 45],
        enforce_quarantine=True,
    )
    wfm_results = wfm.evaluate_trades(trades_df=trades_df)

    print("\n  Cuadrícula 2D (Entrenamiento IS x Prueba OOS Ciega):")
    print(f"  {'Ventana IS':<14} | {'Test 20 Días':<22} | {'Test 30 Días':<22} | {'Test 45 Días':<22}")
    print("  " + "-" * 82)
    for row in wfm_results["grid_table"]:
        t20 = row.get("test_20d", "N/A")
        t30 = row.get("test_30d", "N/A")
        t45 = row.get("test_45d", "N/A")
        print(f"  {row['train_days']} días      | {t20:<22} | {t30:<22} | {t45:<22}")

    print("\n  [MÉTRICAS CONSOLIDADAS WFM]")
    print_row("Eficiencia Walk-Forward Retorno (WFE)", f"{wfm_results['mean_wfe_return']:.1f}% (Robert Pardo >= 55%)")
    print_row("Eficiencia Walk-Forward Sharpe", f"{wfm_results['mean_wfe_sharpe']:.1f}%")
    print_row("Consistencia de Ventanas Positivas", f"{wfm_results['pct_positive_oos_windows']:.1f}% (Meta >= 70%)")
    print_row("Diagnóstico Temporal WFM", wfm_results["verdict"])

    # -------------------------------------------------------------------------
    # PILAR 4: PRUEBAS DE ESTRÉS HOSTILES 'WHAT-IF'
    # -------------------------------------------------------------------------
    print_banner("PILAR 4: PRUEBAS DE ESTRÉS HOSTILES 'WHAT-IF'")

    what_if_res = run_what_if_stress_suite(
        trades_df=trades_df,
        symbol=args.symbol,
        initial_capital=profile["capital"],
        mtdd=profile["mtdd"],
        dll=profile["dll"],
        n_prune_simulations=1000,
        seed=args.seed,
    )

    base = what_if_res["baseline"]
    s2x = what_if_res["slippage_2x"]
    s3x = what_if_res["slippage_3x"]
    top3 = what_if_res["top3_drop"]
    wp = what_if_res["win_pruning"]
    gap = what_if_res["gap_adverse_sl"]

    print(f"\n  {'Escenario de Estrés':<30} | {'Beneficio Neto':<16} | {'Profit Factor':<14} | {'Max DD':<12} | {'DLL Violated':<12}")
    print("  " + "-" * 90)
    print(f"  {'0. Línea Base (Real)':<30} | ${base['net_profit_usd']:>+12,.2f} | {base['profit_factor']:>13.2f} | ${base['max_drawdown_usd']:>9,.2f} | {'SÍ (FALLA)' if base['dll_violated'] else 'NO (OK)':>12}")
    print(f"  {'1. Slippage Shock 2x (+1 tick)':<30} | ${s2x['net_profit_usd']:>+12,.2f} | {s2x['profit_factor']:>13.2f} | ${s2x['max_drawdown_usd']:>9,.2f} | {'SÍ (FALLA)' if s2x['dll_violated'] else 'NO (OK)':>12}")
    print(f"  {'2. Slippage Shock 3x (+2 ticks)':<30} | ${s3x['net_profit_usd']:>+12,.2f} | {s3x['profit_factor']:>13.2f} | ${s3x['max_drawdown_usd']:>9,.2f} | {'SÍ (FALLA)' if s3x['dll_violated'] else 'NO (OK)':>12}")
    print(f"  {'3. Sin Top-3 Trades (Azar)':<30} | ${top3['net_profit_usd']:>+12,.2f} | {top3['profit_factor']:>13.2f} | ${top3['max_drawdown_usd']:>9,.2f} | {'SÍ (FALLA)' if top3['dll_violated'] else 'NO (OK)':>12}")
    print(f"  {'4. Gap Adverso en SL (+2 ticks)':<30} | ${gap['net_profit_usd']:>+12,.2f} | {gap['profit_factor']:>13.2f} | ${gap['max_drawdown_usd']:>9,.2f} | {'SÍ (FALLA)' if gap['dll_violated'] else 'NO (OK)':>12}")

    print("\n  [5. Random Win-Pruning (Eliminar 10% de ganadores al azar - 1,000 corridas)]")
    print_row("Ganadores eliminados por corrida", f"{wp['pruned_trades_per_iter']} operaciones")
    print_row("Beneficio Pesimista (Percentil 5%)", f"${wp['p5_profit_usd']:+,.2f}")
    print_row("Profit Factor Pesimista (P5)", f"{wp['p5_profit_factor']:.2f}")
    print_row("Peor Drawdown Simulado en Pruning", f"${wp['worst_max_dd_usd']:,.2f}")
    print_row("Iteraciones Rentables (PF >= 1.0)", f"{wp['profitable_iterations_pct']:.1f}%")

    # -------------------------------------------------------------------------
    # SCORECARD & ÚLTIMO VEREDICTO (100 PUNTOS & 6 COMPUERTAS DURAS)
    # -------------------------------------------------------------------------
    print_banner("SISTEMA DE CALIFICACIÓN DE 100 PUNTOS & COMPUERTAS DURAS")

    monte_carlo_res_pack = {
        "reshuffle": res_reshuffle,
        "bootstrap": res_bootstrap,
        "session_bootstrap": res_session,
    }

    scorecard = PropFirmRobustnessScorecard(
        capital=profile["capital"],
        mtdd=profile["mtdd"],
        dll=profile["dll"],
        target=profile["target"],
    )

    verdict_report = scorecard.evaluate(
        monte_carlo_res=monte_carlo_res_pack,
        what_if_res=what_if_res,
        parameter_res=None,
        walk_forward_res=wfm_results,
        dsr_res=dsr_results,
    )

    print("\n  [DESGLOSE DE PUNTUACIÓN POR PILAR]")
    for p_name, p_data in verdict_report["pillar_breakdown"].items():
        clean_name = p_name.replace("_", " ").title()
        print_row(clean_name, f"{p_data['earned']:.1f} / {p_data['max']:.1f} pts")

    print_row("PUNTUACIÓN TOTAL NORMALIZADA", f"{verdict_report['score']:.1f} / 100.0 PUNTOS")

    print("\n  [AUDITORÍA DE LAS 6 COMPUERTAS DURAS (HARD GATES)]")
    if verdict_report["hard_gates_passed"]:
        print("  🟢 TODAS LAS COMPUERTAS DURAS APROBADAS (Cero violaciones críticas)")
    else:
        print("  🔴 VIOLACIONES DETECTADAS EN COMPUERTAS DURAS:")
        for violation in verdict_report["hard_gates_violations"]:
            print(f"     ❌ {violation}")

    print("\n" + "#" * 86)
    print(f"#{'':^84}#")
    print(f"#{'ÚLTIMO VEREDICTO DE ROBUSTEZ CUANTITATIVA':^84}#")
    print(f"#{'':^84}#")
    print(f"#{verdict_report['verdict']:^84}#")
    print(f"#{'':^84}#")
    print("#" * 86 + "\n")

    # Exportar JSON si fue solicitado
    if args.export_json:
        export_data = {
            "strategy": strategy_label,
            "trades_file": str(trades_path),
            "profile": profile,
            "erdos_renyi_expected_loss_streak": round(erdos_renyi_streak, 2),
            "monte_carlo": {
                "reshuffle": {
                    "risk_of_ruin_pct": res_reshuffle["risk_of_ruin_pct"],
                    "target_before_ruin_pct": res_reshuffle["target_before_ruin_pct"],
                    "drawdown_stats": res_reshuffle["drawdown_stats"],
                },
                "bootstrap": {
                    "risk_of_ruin_pct": res_bootstrap["risk_of_ruin_pct"],
                    "target_before_ruin_pct": res_bootstrap["target_before_ruin_pct"],
                    "drawdown_stats": res_bootstrap["drawdown_stats"],
                },
                "session_bootstrap": {
                    "dll_violation_pct": res_session["dll_violation_pct"],
                    "risk_of_ruin_pct": res_session["risk_of_ruin_pct"],
                    "daily_loss_stats": res_session["daily_loss_stats"],
                    "drawdown_stats": res_session["drawdown_stats"],
                },
            },
            "deflated_sharpe_ratio": dsr_results,
            "walk_forward_matrix": wfm_results,
            "what_if_stress": what_if_res,
            "scorecard_verdict": verdict_report,
        }

        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"  [+] Informe JSON exportado con éxito a: {out_path}\n")


if __name__ == "__main__":
    main()
