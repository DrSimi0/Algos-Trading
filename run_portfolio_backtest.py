"""
run_portfolio_backtest.py
Script ejecutable para evaluar portafolios multi-estrategia y auditoría de cuentas Prop Trading.

Permite:
1. Combinar dos o más estrategias cuantitativas.
2. Sincronizar series temporales y alinear PnL diario.
3. Evaluar el coeficiente de correlación y la ganancia por diversificación (reducción del Max Drawdown).
4. Auditar las reglas de Prop Firm (DLL $1,000 y MTDD $2,000) sobre la cuenta conjunta consolidada.
5. Ejecutar Simulación de Monte Carlo directamente sobre el portafolio combinado.

Uso:
    python run_portfolio_backtest.py --strat1 f7 --strat2 f4
    python run_portfolio_backtest.py --strat1 f7 --strat2 f8 --method equal_weight --monte-carlo
    python run_portfolio_backtest.py --trades1 reports/f7_meta_labeling_trades.csv --trades2 reports/cointegration_ou_trades.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Asegurar codificación UTF-8 en consola de Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.portfolio.allocator import PortfolioAllocator
from src.core.monte_carlo import MonteCarloSimulator

STRATEGY_MAP = {
    "f1": ("F1_Intraday_Momentum", "reports/f1_intraday_momentum_trades.csv"),
    "f1_n1": ("F1_Optimizacion_IS_N1", "reports/f1_optimizacion_is_n1_trades.csv"),
    "f2": ("F2_Filtered_ORB", "reports/orb_trades.csv"),
    "orb": ("F2_Filtered_ORB", "reports/orb_trades.csv"),
    "f3": ("F3_Optimization_IS_N3", "reports/f3_optimization_n3_trades.csv"),
    "f3_n3": ("F3_Optimization_IS_N3", "reports/f3_optimization_n3_trades.csv"),
    "f4": ("F4_Optimization_IS_N2", "reports/f4_optimizacion_is_n2_trades.csv"),
    "f4_n2": ("F4_Optimization_IS_N2", "reports/f4_optimizacion_is_n2_trades.csv"),
    "f4_n3": ("F4_Optimization_IS_N3", "reports/f4_optimizacion_is_n3_trades.csv"),
    "ou": ("F4_Optimization_IS_N2", "reports/f4_optimizacion_is_n2_trades.csv"),
    "f5": ("F5_Order_Flow_Imbalance", "reports/ofi_trades.csv"),
    "ofi": ("F5_Order_Flow_Imbalance", "reports/ofi_trades.csv"),
    "f7": ("F7_Meta_Labeling", "reports/f7_meta_labeling_trades.csv"),
    "f8": ("F8_HAR_RV_J", "reports/har_rv_j_trades.csv"),
    "har": ("F8_HAR_RV_J", "reports/har_rv_j_trades.csv"),
    "f9": ("F9_Anomalias_VRP", "reports/f9_trades.csv"),
}


def print_banner(text: str, width: int = 80):
    print("\n" + "=" * width)
    print(f" {text:^{width-2}} ")
    print("=" * width)


def print_row(label: str, val: str, width: int = 80):
    col1 = 44
    col2 = width - col1 - 5
    print(f"  {label:<{col1}} : {val:>{col2}}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluador de Portafolios Multi-Estrategia y Auditoría Prop Firm"
    )
    parser.add_argument("--strat1", type=str, default="f7", choices=list(STRATEGY_MAP.keys()),
                        help="Nombre de la primera estrategia (def: f7)")
    parser.add_argument("--strat2", type=str, default="f4", choices=list(STRATEGY_MAP.keys()),
                        help="Nombre de la segunda estrategia (def: f4)")
    parser.add_argument("--trades1", type=str, default=None,
                        help="Ruta alternativa a trades CSV de la estrategia 1")
    parser.add_argument("--trades2", type=str, default=None,
                        help="Ruta alternativa a trades CSV de la estrategia 2")
    parser.add_argument("--method", type=str, default="equal_weight",
                        choices=["equal_weight", "inverse_volatility", "custom"],
                        help="Método de asignación de capital: equal_weight, inverse_volatility, custom (def: equal_weight)")
    parser.add_argument("--weights", type=float, nargs=2, default=None,
                        help="Ponderaciones si method=custom (ej. 0.6 0.4)")
    parser.add_argument("--capital", type=float, default=50000.0,
                        help="Capital inicial de la cuenta consolidada en USD (def: 50,000)")
    parser.add_argument("--dll", type=float, default=1000.0,
                        help="Límite contractual de pérdida diaria (DLL) en USD (def: 1,000)")
    parser.add_argument("--mtdd", type=float, default=2000.0,
                        help="Límite contractual de Max Trailing Drawdown (MTDD) en USD (def: 2,000)")
    parser.add_argument("--target", type=float, default=3000.0,
                        help="Profit Target contractual en USD (def: 3,000)")
    parser.add_argument("--monte-carlo", action="store_true", default=False,
                        help="Ejecutar análisis de Monte Carlo sobre el portafolio combinado")
    parser.add_argument("--simulations", type=int, default=2500,
                        help="Número de simulaciones de Monte Carlo para el portafolio (def: 2,500)")
    parser.add_argument("--export-trades", type=str, default="reports/portfolio_combined_trades.csv",
                        help="Ruta para exportar los trades unificados del portafolio")

    args = parser.parse_args()

    # 1. Resolver nombres y archivos de trades
    name1, default_p1 = STRATEGY_MAP.get(args.strat1.lower(), (args.strat1, ""))
    name2, default_p2 = STRATEGY_MAP.get(args.strat2.lower(), (args.strat2, ""))

    path1 = Path(args.trades1) if args.trades1 else Path(default_p1)
    path2 = Path(args.trades2) if args.trades2 else Path(default_p2)

    if not path1.exists():
        print(f"Error: No se encontró archivo de trades para '{name1}' en: {path1}")
        sys.exit(1)
    if not path2.exists():
        print(f"Error: No se encontró archivo de trades para '{name2}' en: {path2}")
        sys.exit(1)

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    custom_weights = {}
    if args.method == "custom" and args.weights:
        custom_weights = {name1: args.weights[0], name2: args.weights[1]}

    print_banner(f"EVALUACIÓN DE PORTAFOLIO MULTI-ESTRATEGIA: {name1} + {name2}")
    print(f"  Estrategia A             : {name1} ({len(df1):,} trades en {path1})")
    print(f"  Estrategia B             : {name2} ({len(df2):,} trades en {path2})")
    print(f"  Ponderación de Capital   : {args.method.upper()}")
    print(f"  Capital Inicial Cuenta   : ${args.capital:,.2f}")
    print(f"  Límite Pérdida Diaria    : ${args.dll:,.2f} USD")
    print(f"  Max Trailing Drawdown    : ${args.mtdd:,.2f} USD")
    print(f"  Profit Target            : ${args.target:,.2f} USD")

    # 2. Combinar Estrategias y Auditar Portafolio
    allocator = PortfolioAllocator(method=args.method, custom_weights=custom_weights)
    strategy_trades = {name1: df1, name2: df2}
    port_res = allocator.combine_trades_prop_firm(
        strategy_trades=strategy_trades,
        initial_capital=args.capital,
        daily_loss_limit=args.dll,
        max_trailing_drawdown=args.mtdd,
        profit_target=args.target,
    )

    # 3. Mostrar Fichas Individuales
    print_banner("1. DESGLOSE INDIVIDUAL DE ESTRATEGIAS")
    stats1 = port_res["strat_stats"][name1]
    stats2 = port_res["strat_stats"][name2]
    w1 = port_res["weights"].get(name1, 0.5)
    w2 = port_res["weights"].get(name2, 0.5)

    print_row(f"{name1} - Asignación de Capital", f"{w1*100:.1f}%")
    print_row(f"{name1} - PnL Neto Total", f"${stats1['net_pnl_usd']:+,.2f}")
    print_row(f"{name1} - Max Drawdown Individual", f"${stats1['max_drawdown_usd']:,.2f}")
    print_row(f"{name1} - Win Rate", f"{stats1['win_rate_pct']:.2f}%")
    print("  " + "-" * 76)
    print_row(f"{name2} - Asignación de Capital", f"{w2*100:.1f}%")
    print_row(f"{name2} - PnL Neto Total", f"${stats2['net_pnl_usd']:+,.2f}")
    print_row(f"{name2} - Max Drawdown Individual", f"${stats2['max_drawdown_usd']:,.2f}")
    print_row(f"{name2} - Win Rate", f"{stats2['win_rate_pct']:.2f}%")

    # 4. Análisis de Correlación y Diversificación
    print_banner("2. ANÁLISIS DE CORRELACIÓN Y BENEFICIO DE DIVERSIFICACIÓN")
    corr = port_res["pearson_correlation"]
    if corr < 0.10:
        corr_diag = "🟢 EXCELENTE (Descorrelación casi total: Máxima amortiguación de riesgo)"
    elif corr < 0.40:
        corr_diag = "🟢 BUENA (Baja correlación: Aporte positivo a la estabilidad)"
    elif corr < 0.70:
        corr_diag = "🟡 MODERADA (Correlación intermedia: Diversificación parcial)"
    else:
        corr_diag = "🔴 ALTA (Estrategias correlacionadas: Poco beneficio de diversificación)"

    print_row("Coeficiente de Correlación (Pearson)", f"{corr:+.4f}")
    print_row("Diagnóstico de Descorrelación", corr_diag)
    print_row("Max Drawdown del Portafolio Conjunto", f"${port_res['portfolio_metrics']['portfolio_max_dd_usd']:,.2f}")
    print_row("Reducción de Drawdown vs Promedio", f"{port_res['portfolio_metrics']['dd_reduction_pct']:.1f}%")

    # 5. Auditoría Prop Firm Consolidada
    print_banner(f"3. AUDITORÍA DE PRUEBA DE FONDEO (CUENTA CONJUNTA ${args.capital:,.0f})")
    pm = port_res["portfolio_metrics"]
    print_row("Capital Final Consolidado", f"${pm['final_equity']:,.2f}")
    print_row("Beneficio Neto del Portafolio", f"${pm['net_profit_usd']:+,.2f}")
    print_row(f"Días con Violación de DLL (-${args.dll:,.0f})", str(pm["dll_violations_count"]))
    print_row(f"Incumplimiento de Trailing DD (${args.mtdd:,.0f})", "SÍ (RESCISIÓN)" if pm["mtdd_breached"] else "NO (CONFORME)")
    print_row("Estado Contractual de la Cuenta", pm["compliance_status"])

    # Exportar trades unificados
    if args.export_trades and not port_res["unified_trades"].empty:
        exp_p = Path(args.export_trades)
        exp_p.parent.mkdir(parents=True, exist_ok=True)
        port_res["unified_trades"].to_csv(exp_p, index=False)
        print(f"\n[+] Registro cronológico de {len(port_res['unified_trades'])} operaciones unificadas exportado a: {exp_p.resolve()}")

    # 6. Monte Carlo opcional sobre el portafolio
    if args.monte_carlo:
        print_banner("4. SIMULACIÓN DE MONTE CARLO SOBRE EL PORTAFOLIO CONSOLIDADO")
        pnl_series = port_res["unified_trades"]["weighted_pnl_usd"]

        if len(pnl_series) > 0:
            sim = MonteCarloSimulator(
                trades_pnl=pnl_series,
                initial_capital=args.capital,
                daily_loss_limit=args.dll,
                max_trailing_drawdown=args.mtdd,
                profit_target=args.target,
                seed=42,
            )
            mc_res = sim.run_reshuffle(n_simulations=args.simulations)
            print_row("Simulaciones Ejecutadas", f"{mc_res['simulations']:,}")
            print_row("Riesgo de Ruina del Portafolio", f"{mc_res['risk_of_ruin_pct']:.2f}%")
            print_row("Probabilidad de Aprobación Limpia", f"{mc_res['target_before_ruin_pct']:.2f}%")
            print_row("Drawdown en Riesgo al 95% (Portafolio)", f"${mc_res['drawdown_stats']['p95_dd_usd']:,.2f}")
            print_row("Peor Drawdown Absoluto (Portafolio)", f"${mc_res['drawdown_stats']['worst_case_dd_usd']:,.2f}")
            print_row("Veredicto de Monte Carlo", mc_res["verdict"])
        else:
            print("  [!] No se pudieron extraer operaciones para la simulación de Monte Carlo.")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
