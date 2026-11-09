"""
run_monte_carlo.py
Script ejecutable para análisis de Monte Carlo sobre backtests de trading cuantitativo.

Evalúa:
1. Reshuffling (Riesgo de secuencia de operaciones).
2. Bootstrapping (Escenarios extremos con rachas adversas extendidas).
3. Probabilidad de quiebra / ruina en pruebas de fondeo (Apex / Topstep).
4. Peor Drawdown proyectado al 95% y 99% de confianza.

Uso:
    python run_monte_carlo.py --trades reports/f7_meta_labeling_trades.csv --simulations 2500
    python run_monte_carlo.py --strategy f7 --simulations 2500 --method both
    python run_monte_carlo.py --strategy f1 --capital 50000 --mtdd 2000 --dll 1000 --target 3000
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

# Asegurar codificación UTF-8 en consola de Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.core.monte_carlo import MonteCarloSimulator


def print_banner(text: str, width: int = 80):
    print("\n" + "=" * width)
    print(f" {text:^{width-2}} ")
    print("=" * width)


def print_row(label: str, val: str, width: int = 80):
    col1 = 44
    col2 = width - col1 - 5
    print(f"  {label:<{col1}} : {val:>{col2}}")


STRATEGY_TRADES_MAP = {
    "f1": "reports/f1_intraday_momentum_trades.csv",
    "f2": "reports/orb_trades.csv",
    "orb": "reports/orb_trades.csv",
    "f3": "reports/f3_optimization_n3_trades.csv" if Path("reports/f3_optimization_n3_trades.csv").exists() else "reports/f3_optimization_n2_trades.csv",
    "f3_n3": "reports/f3_optimization_n3_trades.csv",
    "f3_opt": "reports/f3_optimization_n3_trades.csv" if Path("reports/f3_optimization_n3_trades.csv").exists() else "reports/f3_optimization_n2_trades.csv",
    "f3_n2": "reports/f3_optimization_n2_trades.csv",
    "f3_n1": "reports/f3_optimization_n1_trades.csv",
    "f3_base": "reports/anchored_vwap_trades.csv",
    "f4": "reports/f4_optimizacion_is_n2_trades.csv" if Path("reports/f4_optimizacion_is_n2_trades.csv").exists() else "reports/f4_optimizacion_is_n1_trades.csv",
    "f4_n2": "reports/f4_optimizacion_is_n2_trades.csv",
    "f4_opt": "reports/f4_optimizacion_is_n2_trades.csv" if Path("reports/f4_optimizacion_is_n2_trades.csv").exists() else "reports/f4_optimizacion_is_n1_trades.csv",
    "f4_n1": "reports/f4_optimizacion_is_n1_trades.csv",
    "f4_base": "reports/cointegration_ou_trades.csv",
    "ou": "reports/f4_optimizacion_is_n2_trades.csv" if Path("reports/f4_optimizacion_is_n2_trades.csv").exists() else "reports/f4_optimizacion_is_n1_trades.csv",
    "f5": "reports/ofi_trades.csv",
    "ofi": "reports/ofi_trades.csv",
    "f7": "reports/f7_meta_labeling_trades.csv",
    "f8": "reports/har_rv_j_trades.csv",
    "har": "reports/har_rv_j_trades.csv",
    "f9": "reports/f9_trades.csv",
}


def load_pnl_series(trades_path: Path) -> pd.Series:
    """Carga y extrae la columna de PnL de un archivo de trades CSV."""
    if not trades_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de trades en: {trades_path}")

    df = pd.read_csv(trades_path)
    for col in ["net_pnl_usd", "net_pnl", "net_profit", "pnl", "PnL", "Net_PnL"]:
        if col in df.columns:
            return df[col].dropna().astype(float)

    raise ValueError(f"No se encontró una columna de PnL válida en {trades_path}. Columnas: {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser(
        description="Simulador Cuantitativo de Monte Carlo para Backtesting y Prop Trading"
    )
    parser.add_argument("--trades", type=str, default=None,
                        help="Ruta al archivo CSV con registro de operaciones")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=list(STRATEGY_TRADES_MAP.keys()),
                        help="Nombre de la estrategia (ej. f1, f2, f4, f7, f8)")
    parser.add_argument("--simulations", type=int, default=2500,
                        help="Número de trayectorias aleatorias a simular (por defecto: 2,500)")
    parser.add_argument("--method", type=str, choices=["reshuffle", "bootstrap", "both"], default="both",
                        help="Método de simulación: 'reshuffle', 'bootstrap' o 'both' (def: both)")
    parser.add_argument("--capital", type=float, default=50000.0,
                        help="Capital inicial de la cuenta en USD (def: 50,000)")
    parser.add_argument("--mtdd", type=float, default=2000.0,
                        help="Límite contractual de Max Trailing Drawdown en USD (def: 2,000)")
    parser.add_argument("--dll", type=float, default=1000.0,
                        help="Límite contractual de Daily Loss Limit en USD (def: 1,000)")
    parser.add_argument("--target", type=float, default=3000.0,
                        help="Profit Target contractual de la prueba en USD (def: 3,000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla estocástica para reproducibilidad (def: 42)")

    args = parser.parse_args()

    # Resolver archivo de trades
    trades_file = None
    strat_name = "Personalizada"
    if args.trades:
        trades_file = Path(args.trades)
        strat_name = trades_file.stem
    elif args.strategy:
        strat_key = args.strategy.lower()
        trades_file = Path(STRATEGY_TRADES_MAP[strat_key])
        strat_name = strat_key.upper()
    else:
        # Buscar el archivo más reciente disponible en reports/
        candidates = [
            "reports/f7_meta_labeling_trades.csv",
            "reports/cointegration_ou_trades.csv",
            "reports/har_rv_j_trades.csv",
            "reports/ofi_trades.csv",
            "reports/orb_trades.csv",
        ]
        for c in candidates:
            if Path(c).exists():
                trades_file = Path(c)
                strat_name = trades_file.stem
                break

    if trades_file is None or not trades_file.exists():
        print(f"Error: No se encontró archivo de trades para simular.")
        print(f"Ejecuta primero un backtest (ej. 'python run_meta_labeling_backtest.py') o especifica '--trades <ruta>'.")
        sys.exit(1)

    print_banner(f"SIMULACIÓN DE MONTE CARLO CUANTITATIVA: {strat_name}")
    print(f"  Archivo de Origen        : {trades_file}")
    print(f"  Capital Inicial          : ${args.capital:,.2f}")
    print(f"  Max Trailing DD (MTDD)   : ${args.mtdd:,.2f}")
    print(f"  Daily Loss Limit (DLL)   : ${args.dll:,.2f}")
    print(f"  Profit Target            : ${args.target:,.2f}")
    print(f"  Simulaciones Solicitadas : {args.simulations:,} iteraciones")
    print(f"  Método de Análisis       : {args.method.upper()}")

    # Cargar serie de PnL
    trades_pnl = load_pnl_series(trades_file)
    print(f"  Operaciones Cargadas     : {len(trades_pnl):,} trades reales del backtest")

    sim = MonteCarloSimulator(
        trades_pnl=trades_pnl,
        initial_capital=args.capital,
        daily_loss_limit=args.dll,
        max_trailing_drawdown=args.mtdd,
        profit_target=args.target,
        seed=args.seed,
    )

    methods_to_run = []
    if args.method in ["reshuffle", "both"]:
        methods_to_run.append("reshuffle")
    if args.method in ["bootstrap", "both"]:
        methods_to_run.append("bootstrap")

    for m in methods_to_run:
        if m == "reshuffle":
            res = sim.run_reshuffle(n_simulations=args.simulations)
            title = f"MONTE CARLO: PERMUTACIÓN (RESHUFFLING) - RIESGO DE SECUENCIA"
        else:
            res = sim.run_bootstrap(n_simulations=args.simulations)
            title = f"MONTE CARLO: BOOTSTRAPPING - RACHAS Y ESCENARIOS ADVERSOS"

        print_banner(title)
        print_row("Total de Simulaciones", f"{res['simulations']:,}")
        print_row("Riesgo de Ruina (Violación MTDD)", f"{res['risk_of_ruin_pct']:.2f}%")
        print_row("Probabilidad de Tocar Target ($3,000)", f"{res['target_reach_pct']:.2f}%")
        print_row("Aprobación Limpia (Target antes de Ruina)", f"{res['target_before_ruin_pct']:.2f}%")

        dd = res["drawdown_stats"]
        print("\n  [DISTRIBUCIÓN DE DRAWDOWN MÁXIMO PROYECTADO]")
        print_row("Drawdown Mediano (50% de probabilidad)", f"${dd['median_dd_usd']:,.2f}")
        print_row("Drawdown en Riesgo (Percentil 95%)", f"${dd['p95_dd_usd']:,.2f}")
        print_row("Drawdown Extremo (Percentil 99%)", f"${dd['p99_dd_usd']:,.2f}")
        print_row("Peor Drawdown Absoluto Simulado", f"${dd['worst_case_dd_usd']:,.2f}")
        print_row("CVaR Drawdown (Promedio Peor 5%)", f"${dd['cvar_95_dd_usd']:,.2f}")

        eq = res["equity_stats"]
        print("\n  [DISTRIBUCIÓN DE CAPITAL FINAL TRAS EL PERÍODO]")
        print_row("Capital Final Pesimista (Percentil 5%)", f"${eq['p5_equity_usd']:,.2f}")
        print_row("Capital Final Mediano (Percentil 50%)", f"${eq['p50_median_equity_usd']:,.2f}")
        print_row("Capital Final Optimista (Percentil 95%)", f"${eq['p95_equity_usd']:,.2f}")
        print_row("Beneficio Neto Promedio Simulado", f"${eq['mean_profit_usd']:+,.2f}")

        print("\n  [VEREDICTO PARA PRUEBAS DE FONDEO]")
        print(f"  >> {res['verdict']}\n")


if __name__ == "__main__":
    main()
