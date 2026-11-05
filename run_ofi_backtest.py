"""
run_ofi_backtest.py
Script ejecutable para correr el backtest cuantitativo de la estrategia F5:
Order Flow Imbalance (OFI) de Nivel 1 (Cont, Kukanov & Stoikov 2014) con control de Prop Trading.

Uso:
    python run_ofi_backtest.py --symbol ES
    python run_ofi_backtest.py --symbol NQ --data data/raw/CME_MINI_NQ_5m_bars5000.csv
    python run_ofi_backtest.py --capital 50000 --dll 1000 --tmdd 2000
    python run_ofi_backtest.py --synthetic-lob --events 5000
"""

import argparse
from datetime import time
import json
from pathlib import Path
import sys
from typing import Any, Dict
import pandas as pd

from src.core.data_split import add_data_split_args, partition_from_args
from src.strategies.order_flow_imbalance import (
    OrderFlowImbalanceStrategy,
    OFIBacktester,
    OFIPropAccount,
    generate_synthetic_lob_events,
)


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
        description="Backtest de Estrategia F5: Order Flow Imbalance de Nivel 1 (Cont et al., 2014)"
    )
    parser.add_argument("--symbol", type=str, default="ES", choices=["ES", "MES", "NQ", "MNQ"],
                        help="Símbolo del contrato CME (por defecto: ES)")
    parser.add_argument("--data", type=str, default=None,
                        help="Ruta al archivo CSV o Parquet con datos de mercado o LOB")
    parser.add_argument("--synthetic-lob", action="store_true",
                        help="Ejecutar simulación con generador de eventos sintéticos L1 de CME MDP 3.0")
    parser.add_argument("--events", type=int, default=15000,
                        help="Cantidad de eventos para generador sintético L1 (por defecto: 15000)")
    parser.add_argument("--flow-persistence", type=float, default=0.65,
                        help="Persistencia autorregresiva de flujo informado en LOB sintético (def: 0.65)")
    parser.add_argument("--capital", type=float, default=50000.0,
                        help="Capital inicial de la cuenta en USD (def: 50,000)")
    parser.add_argument("--dll", type=float, default=1000.0,
                        help="Límite de Pérdida Diaria (Daily Loss Limit) en USD (def: 1,000)")
    parser.add_argument("--tmdd", type=float, default=2000.0,
                        help="Límite de Trailing Drawdown Máximo en USD (def: 2,000)")
    parser.add_argument("--theta-enter", type=float, default=2.00,
                        help="Umbral de disparo de entrada Z_OFI (def: 2.00)")
    parser.add_argument("--theta-exhaust", type=float, default=0.50,
                        help="Umbral de agotamiento de flujo Z_OFI para Take-Profit (def: 0.50)")
    parser.add_argument("--theta-reversal", type=float, default=-1.25,
                        help="Umbral de inversión contraria Z_OFI para Stop-Loss (def: -1.25)")
    parser.add_argument("--window-w", type=int, default=30,
                        help="Ventana de normalización Z-score en barras (def: 30)")
    parser.add_argument("--h-max", type=int, default=10,
                        help="Límite máximo de permanencia por decaimiento en barras (def: 10)")
    parser.add_argument("--export-trades", type=str, default="reports/ofi_trades.csv",
                        help="Ruta para exportar el log de operaciones en CSV")
    parser.add_argument("--export-summary", type=str, default="reports/ofi_summary.json",
                        help="Ruta para exportar el resumen de métricas en JSON")
    add_data_split_args(parser)

    args = parser.parse_args()

    pair_map = {
        "ES": ("ES", "MES"),
        "MES": ("ES", "MES"),
        "NQ": ("NQ", "MNQ"),
        "MNQ": ("NQ", "MNQ"),
    }
    std_sym, micro_sym = pair_map[args.symbol]

    # 1. Obtener o generar datos de mercado / LOB
    feed_type = "LOB Sintético Calibrado (CME MDP 3.0 L1)"
    if args.synthetic_lob:
        print(f"\n[1/3] Generando {args.events:,} eventos sintéticos de LOB L1 calibrados para {std_sym}...")
        mean_depth = 250.0 if "ES" in std_sym else 30.0
        start_p = 5800.0 if "ES" in std_sym else 20000.0
        df = generate_synthetic_lob_events(
            start_price=start_p,
            n_events=args.events,
            tick_size=0.25,
            mean_depth=mean_depth,
            flow_persistence=args.flow_persistence,
            seed=42
        )
    else:
        if args.data is None:
            # Buscar archivo predeterminado en data/raw/ (priorizando dataset 3 años)
            candidates = [
                f"data/raw/CME_MINI_{std_sym}_5m_3y.parquet",
                f"data/raw/CME_MINI_{std_sym}_5m_3y.csv",
                f"data/raw/CME_MINI_{std_sym}_5m_bars5000.csv",
                f"data/raw/CME_MINI_{std_sym}_5m_bars5000.parquet",
                f"data/raw/CME_MINI_{std_sym}_15m_bars5000.csv",
            ]
            for c in candidates:
                if Path(c).exists():
                    args.data = c
                    break

        if args.data is None or not Path(args.data).exists():
            print(f"\n[1/3] No se encontró archivo específico para {std_sym}. Generando secuencia de eventos L1 calibrada...")
            mean_depth = 250.0 if "ES" in std_sym else 30.0
            start_p = 5800.0 if "ES" in std_sym else 20000.0
            df = generate_synthetic_lob_events(
                start_price=start_p,
                n_events=args.events,
                tick_size=0.25,
                mean_depth=mean_depth,
                flow_persistence=args.flow_persistence,
                seed=42
            )
        else:
            print(f"\n[1/3] Cargando datos de mercado desde '{args.data}'...")
            data_path = Path(args.data)
            if data_path.suffix == ".parquet":
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path)

            if not isinstance(df.index, pd.DatetimeIndex):
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df.set_index("datetime", inplace=True)
                else:
                    df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            df, part_label = partition_from_args(df, args, f"F5_OFI_{args.symbol}")
            df.sort_index(inplace=True)

            l1_cols = {"bid_price", "bid_size", "ask_price", "ask_size"}
            has_l1 = l1_cols.issubset({c.lower() for c in df.columns}) or "OFI" in df.columns
            if not has_l1:
                feed_type = "Barras OHLCV Agregadas (Proxy con retardo temporal)"
                print("\n" + "!" * 78)
                print(" [AVISO CRÍTICO DE INFRAESTRUCTURA DE DATOS]")
                print(" Se detectaron barras OHLCV agregadas (5m) sin libro de órdenes L1/L2.")
                print(" El cálculo hidrodinámico aproximado genera retardo temporal que degrada")
                print(" el Win Rate (~21.57%). En producción se requiere CME MDP 3.0 tick a tick.")
                print(" Para validar la eficacia teórica de la estrategia ejecute con: --synthetic-lob")
                print("!" * 78 + "\n")
            else:
                feed_type = "Feed Real CME Globex MDP 3.0 (L1)"

    # 2. Configurar Estrategia y Cuenta Prop Trading
    print(f"[2/3] Configurando Estrategia F5 y Cuenta Prop Trading (${args.capital:,.2f})...")
    strategy = OrderFlowImbalanceStrategy(
        params={
            "symbol": args.symbol,
            "std_symbol": std_sym,
            "micro_symbol": micro_sym,
            "window_w": args.window_w,
            "theta_enter": args.theta_enter,
            "theta_exhaust": args.theta_exhaust,
            "theta_reversal": args.theta_reversal,
            "h_max": args.h_max,
        }
    )

    account = OFIPropAccount(
        initial_capital=args.capital,
        dll=args.dll,
        tmdd=args.tmdd,
        equity=args.capital,
        peak_equity=args.capital,
        trailing_floor=args.capital - args.tmdd,
        lock_floor_at_start=True
    )

    # 3. Ejecutar Backtester
    print("[3/3] Ejecutando simulación microestructural y auditoría de riesgo...")
    backtester = OFIBacktester(strategy=strategy, account=account)
    report = backtester.run(df)
    report["data_feed_type"] = feed_type

    # 4. Mostrar Resultados Formateados
    print_banner("ESTRATEGIA F5: ORDER FLOW IMBALANCE DE NIVEL 1 (CONT, KUKANOV & STOIKOV 2014)")
    print_table_row("Tipo de Feed de Datos", feed_type)
    print_table_row("Activo Principal / Micro", f"{std_sym} / {micro_sym}")
    print_table_row("Capital Inicial", f"${report['initial_capital']:,.2f}")
    print_table_row("Capital Final (Equity)", f"${report['final_equity']:,.2f}")
    print_table_row("Retorno Total Neto (USD)", f"${report['total_net_pnl_usd']:,.2f}")
    print_table_row("Retorno Total (%)", f"{report['total_return_pct']:.2f}%")
    print_table_row("Total Operaciones Ejecutadas", report['total_trades'])
    print_table_row("Operaciones Ganadoras / Perdedoras", f"{report['winning_trades']} / {report['losing_trades']}")
    print_table_row("Tasa de Acierto (Win Rate)", f"{report['win_rate_pct']:.2f}%")
    print_table_row("Factor de Beneficio (Profit Factor)", report['profit_factor'])
    print_table_row("Beneficio Bruto Ganador", f"${report['gross_profit_usd']:,.2f}")
    print_table_row("Pérdida Bruta Acumulada", f"${report['gross_loss_usd']:,.2f}")
    print_table_row("Comisiones Totales CME", f"${report['total_commissions_usd']:,.2f}")
    print_table_row("Deslizamiento Estimado (Slippage)", f"${report['total_slippage_usd']:,.2f}")
    print_table_row("Drawdown Máximo Flotante", f"${report['max_drawdown_usd']:,.2f} ({report['max_drawdown_pct']:.2f}%)")
    print_table_row("Sharpe Ratio Anualizado", report['sharpe_ratio'])
    print_table_row("Sortino Ratio Anualizado", report['sortino_ratio'])

    print_banner("AUDITORÍA DE RIESGO PARA CUENTAS DE PROP TRADING")
    print_table_row("Límite de Pérdida Diaria (DLL)", f"${args.dll:,.2f}")
    print_table_row("Límite de Trailing Drawdown (TMDD)", f"${args.tmdd:,.2f}")
    status_str = "REPROBADA (BREACH)" if report["is_prop_breached"] else "APROBADA (COMPLIANT)"
    print_table_row("Estado de Evaluación Prop Firm", status_str)
    print_table_row("Detalle de Cumplimiento", report["breach_reason"])
    print("=" * 78 + "\n")

    # 5. Exportar Resultados
    trades_df = backtester.get_trades_dataframe()
    if not trades_df.empty:
        out_trades_path = Path(args.export_trades)
        out_trades_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(out_trades_path, index=False)
        print(f"-> Registro de {len(trades_df)} operaciones exportado a: {out_trades_path.resolve()}")

        print("\nMuestra de operaciones ejecutadas:")
        cols_to_show = ["date", "symbol", "direction", "contracts", "entry_price", "exit_price", "exit_reason", "net_pnl_usd"]
        cols_avail = [c for c in cols_to_show if c in trades_df.columns]
        print(trades_df[cols_avail].tail(10).to_string(index=False))

    out_summary_path = Path(args.export_summary)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"\n-> Resumen cuantitativo JSON exportado a: {out_summary_path.resolve()}\n")


if __name__ == "__main__":
    main()
