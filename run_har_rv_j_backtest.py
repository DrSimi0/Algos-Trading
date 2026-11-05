"""
run_har_rv_j_backtest.py
Script ejecutable para correr el backtest cuantitativo de la Estrategia F8:
Modelos Econométricos Heterogéneos HAR-RV-J y Dimensionamiento por Volatilidad Dinámica
(Corsi 2009 / Andersen, Bollerslev & Diebold 2007 / Moreira & Muir 2017).

Uso:
    python run_har_rv_j_backtest.py --symbol ES
    python run_har_rv_j_backtest.py --symbol NQ --data data/raw/CME_MINI_NQ_5m_bars5000.csv
    python run_har_rv_j_backtest.py --capital 50000 --dll 1000 --mtdd 2000
"""

import argparse
from datetime import time
import json
from pathlib import Path
import sys
from typing import Any
import pandas as pd

from src.core.data_split import add_data_split_args, partition_from_args
from src.strategies.har_rv_j import (
    EconometricHARRVJStrategy,
    HARRVJBacktester,
    PropAccount,
)


def parse_time_str(t_str: str) -> time:
    """Convierte cadena 'HH:MM' o 'HH:MM:SS' a datetime.time."""
    parts = [int(p) for p in t_str.split(":")]
    if len(parts) == 2:
        return time(parts[0], parts[1], 0)
    elif len(parts) == 3:
        return time(parts[0], parts[1], parts[2])
    raise ValueError(f"Formato de hora inválido: {t_str}. Use 'HH:MM' o 'HH:MM:SS'.")


def print_banner(text: str, width: int = 78):
    print("\n" + "=" * width)
    print(f" {text:^{width-2}} ")
    print("=" * width)


def print_table_row(k: str, v: Any, width: int = 78):
    col1 = 44
    col2 = width - col1 - 5
    print(f"  {k:<{col1}} : {str(v):>{col2}}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest de Estrategia F8: Modelos Econométricos HAR-RV-J (Corsi 2009 / Andersen 2007)"
    )
    parser.add_argument("--symbol", type=str, default="ES", choices=["ES", "MES", "NQ", "MNQ"],
                        help="Símbolo del contrato CME (por defecto: ES)")
    parser.add_argument("--data", type=str, default=None,
                        help="Ruta al archivo CSV o Parquet con barras intradía a 5 minutos")
    parser.add_argument("--capital", type=float, default=50000.0,
                        help="Capital inicial de la cuenta en USD (def: 50,000)")
    parser.add_argument("--dll", type=float, default=1000.0,
                        help="Límite de Pérdida Diaria (Daily Loss Limit) en USD (def: 1,000)")
    parser.add_argument("--mtdd", type=float, default=2000.0,
                        help="Límite de Trailing Drawdown Máximo en USD (def: 2,000)")
    parser.add_argument("--kappa-tp", type=float, default=1.80,
                        help="Factor de multiplicador Take-Profit sobre volatilidad esperada (def: 1.80)")
    parser.add_argument("--kappa-sl", type=float, default=1.00,
                        help="Factor de multiplicador Stop-Loss sobre volatilidad esperada (def: 1.00)")
    parser.add_argument("--theta-jump", type=float, default=0.35,
                        help="Umbral de compuerta de salto rho_J (def: 0.35)")
    parser.add_argument("--xi-circuit", type=float, default=2.50,
                        help="Multiplicador del disyuntor de volatilidad intradía (def: 2.50)")
    parser.add_argument("--max-contracts", type=int, default=5,
                        help="Máximo número de contratos estándar permitidos (def: 5)")
    parser.add_argument("--slippage-ticks", type=float, default=1.0,
                        help="Ticks de deslizamiento por orden round-turn (def: 1.0 tick)")
    parser.add_argument("--session-open", type=str, default="09:30",
                        help="Hora de apertura de sesión RTH (def: '09:30')")
    parser.add_argument("--session-close", type=str, default="15:55",
                        help="Hora de cierre determinista de sesión RTH (def: '15:55')")
    parser.add_argument("--export-trades", type=str, default="reports/har_rv_j_trades.csv",
                        help="Ruta para exportar el log de operaciones en CSV")
    parser.add_argument("--export-summary", type=str, default="reports/har_rv_j_summary.json",
                        help="Ruta para exportar el resumen de métricas en JSON")
    add_data_split_args(parser)

    args = parser.parse_args()

    # Mapeo de contratos estándar a microcontratos
    pair_map = {
        "ES": ("ES", "MES"),
        "MES": ("ES", "MES"),
        "NQ": ("NQ", "MNQ"),
        "MNQ": ("NQ", "MNQ"),
    }
    std_sym, micro_sym = pair_map[args.symbol]

    # Resolver rutas por defecto si no fueron suministradas (priorizando dataset 3 años)
    if args.data is None:
        p3y = Path(f"data/raw/CME_MINI_{std_sym}_5m_3y.parquet")
        c3y = Path(f"data/raw/CME_MINI_{std_sym}_5m_3y.csv")
        p5k = Path(f"data/raw/CME_MINI_{std_sym}_5m_bars5000.parquet")
        c5k = Path(f"data/raw/CME_MINI_{std_sym}_5m_bars5000.csv")
        if p3y.exists():
            args.data = str(p3y)
        elif c3y.exists():
            args.data = str(c3y)
        elif p5k.exists():
            args.data = str(p5k)
        elif c5k.exists():
            args.data = str(c5k)
        else:
            args.data = str(p3y)

    # 1. Cargar datos intradía
    print(f"\n[1/4] Cargando datos intradía desde '{args.data}'...")
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Archivo de datos no encontrado: {data_path}")
        sys.exit(1)

    if data_path.suffix == ".parquet":
        df_intra = pd.read_parquet(data_path)
    else:
        df_intra = pd.read_csv(data_path)

    if not isinstance(df_intra.index, pd.DatetimeIndex):
        if "datetime" in df_intra.columns:
            df_intra["datetime"] = pd.to_datetime(df_intra["datetime"])
            df_intra.set_index("datetime", inplace=True)
        else:
            df_intra.index = pd.to_datetime(df_intra.index)

    df_intra.sort_index(inplace=True)
    print(f"      Total de barras cargadas: {len(df_intra):,} | Rango: {df_intra.index[0]} -> {df_intra.index[-1]}")

    df_intra, part_label = partition_from_args(df_intra, args, f"F8_HAR_RV_J_{args.symbol}")

    # 2. Configuración de parámetros y tiempos
    open_t = parse_time_str(args.session_open)
    close_t = parse_time_str(args.session_close)

    print("\n[2/4] Inicializando motor de riesgo de Prop Trading y modelo HAR-RV-J...")
    account = PropAccount(
        initial_capital=args.capital,
        dll=args.dll,
        mtdd=args.mtdd,
        equity=args.capital,
        peak_equity=args.capital
    )

    strategy = EconometricHARRVJStrategy(
        symbol=args.symbol,
        kappa_tp=args.kappa_tp,
        kappa_sl=args.kappa_sl,
        theta_jump=args.theta_jump,
        xi_circuit=args.xi_circuit,
        max_standard_contracts=args.max_contracts,
        session_open_time=open_t,
        session_close_time=close_t,
        slippage_ticks=args.slippage_ticks
    )

    backtester = HARRVJBacktester(strategy=strategy, account=account)

    # 3. Ejecutar simulación
    print("\n[3/4] Ejecutando simulación cronológica día a día (Lookahead Bias = 0)...")
    report = backtester.run(intraday_data=df_intra)

    # 4. Generar reportes e imprimir resultados
    print("\n[4/4] Procesando resultados y métricas cuantitativas...")

    print_banner(f"INFORME FINAL DE BACKTEST: ESTRATEGIA F8 (HAR-RV-J) - {args.symbol}")
    print_table_row("Contrato Símbolo", args.symbol)
    print_table_row("Capital Inicial", f"${report['initial_capital']:,.2f}")
    print_table_row("Capital Final", f"${report['final_equity']:,.2f}")
    print_table_row("PnL Neto Total", f"${report['total_net_pnl_usd']:,.2f}")
    print_table_row("Retorno Total Acumulado", f"{report['total_return_pct']:.2f}%")
    print_table_row("Total de Operaciones", report["total_trades"])
    print_table_row("Operaciones Ganadoras", f"{report['winning_trades']} ({report['win_rate_pct']}%)")
    print_table_row("Operaciones Perdedoras", report["losing_trades"])
    print_table_row("Operaciones Neutras (Empate)", report["break_even_trades"])
    print_table_row("Profit Factor", report["profit_factor"])
    print_table_row("Ganancia Bruta", f"${report['gross_profit_usd']:,.2f}")
    print_table_row("Pérdida Bruta", f"${report['gross_loss_usd']:,.2f}")
    print_table_row("Comisiones CME Totales", f"${report['total_commissions_usd']:,.2f}")
    print_table_row("Slippage Estimado", f"${report['total_slippage_usd']:,.2f}")
    print_table_row("Drawdown Máximo Flotante", f"${report['max_drawdown_usd']:,.2f} ({report['max_drawdown_pct']}%)")
    print_table_row("Sharpe Ratio (Anualizado)", report["sharpe_ratio"])
    print_table_row("Sortino Ratio (Anualizado)", report["sortino_ratio"])
    print_table_row("Calmar Ratio", report["calmar_ratio"])
    print_table_row("Razones de Salida", report["exit_reasons"])
    print_table_row("Estado Prop Trading", "DESCALIFICADA" if report["is_prop_breached"] else "CUMPLIMIENTO EXITOSO")
    print_table_row("Motivo de Incumplimiento", report["breach_reason"])
    print("=" * 78)

    # Exportar operaciones
    trades_df = backtester.get_trades_dataframe()
    if not trades_df.empty and args.export_trades:
        export_p = Path(args.export_trades)
        export_p.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(export_p, index=False)
        print(f"\n[OK] Log de {len(trades_df)} operaciones exportado exitosamente a '{export_p}'.")

    # Exportar resumen JSON
    if args.export_summary:
        sum_p = Path(args.export_summary)
        sum_p.parent.mkdir(parents=True, exist_ok=True)
        with open(sum_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        print(f"[OK] Resumen analítico guardado en '{sum_p}'.\n")


if __name__ == "__main__":
    main()
