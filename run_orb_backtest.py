"""
run_orb_backtest.py
Script ejecutable para correr el backtest cuantitativo de la estrategia F2:
Filtered Quantitative Opening Range Breakout (ORB) con control estricto de Prop Trading.

Uso:
    python run_orb_backtest.py --symbol ES
    python run_orb_backtest.py --symbol NQ --data data/raw/CME_MINI_NQ_5m_bars5000.csv
    python run_orb_backtest.py --capital 50000 --dll 1000 --mtdd 2000
"""

import argparse
from datetime import time
import json
from pathlib import Path
import sys
from typing import Any
import pandas as pd


from src.strategies.filtered_orb import (
    FilteredQuantitativeORB,
    PropAccount,
    FilteredORBBacktester,
    normalize_to_eastern,
)
from src.core.data_split import add_data_split_args, partition_from_args, clip_to_partition


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
        description="Backtest de Estrategia F2: Filtered Quantitative ORB (Crabel 1990 / Zarattini 2024)"
    )
    parser.add_argument("--symbol", type=str, default="ES", choices=["ES", "MES", "NQ", "MNQ"],
                        help="Símbolo del contrato CME (por defecto: ES)")
    parser.add_argument("--data", type=str, default=None,
                        help="Ruta al archivo CSV o Parquet con velas intradía")
    parser.add_argument("--daily-data", type=str, default=None,
                        help="Ruta al archivo CSV o Parquet con velas diarias (opcional)")
    parser.add_argument("--capital", type=float, default=50000.0,
                        help="Capital inicial de la cuenta en USD (def: 50,000)")
    parser.add_argument("--dll", type=float, default=1000.0,
                        help="Límite de Pérdida Diaria (Daily Loss Limit) en USD (def: 1,000)")
    parser.add_argument("--mtdd", type=float, default=2000.0,
                        help="Límite de Trailing Drawdown Máximo en USD (def: 2,000)")
    parser.add_argument("--lambda-stretch", type=float, default=0.50,
                        help="Multiplicador del amortiguador de ruido Stretch (def: 0.50)")
    parser.add_argument("--rvol-min", type=float, default=1.80,
                        help="Umbral mínimo de volumen relativo RVOL (def: 1.80)")
    parser.add_argument("--max-micros", type=int, default=15,
                        help="Máximo número de microcontratos permitidos (def: 15)")
    parser.add_argument("--slippage-ticks", type=float, default=1.0,
                        help="Ticks de deslizamiento por orden (def: 1.0 tick)")
    parser.add_argument("--source-tz", type=str, default=None,
                        help="Zona horaria de origen de los datos brutos (ej. 'America/Mexico_City', def: auto-detect)")
    parser.add_argument("--session-open", type=str, default=None,
                        help="Hora de apertura de sesión (ej. '09:30' America/New_York)")
    parser.add_argument("--or-end", type=str, default=None,
                        help="Hora de cierre del Opening Range (ej. '09:45' America/New_York)")
    parser.add_argument("--entry-cutoff", type=str, default=None,
                        help="Hora límite para cancelación de órdenes pendientes (ej. '12:00' America/New_York)")
    parser.add_argument("--flat-time", type=str, default=None,
                        help="Hora forzosa de neutralización diaria (ej. '15:45' America/New_York)")
    parser.add_argument("--export-trades", type=str, default="reports/orb_trades.csv",
                        help="Ruta para exportar el log de operaciones en CSV")
    parser.add_argument("--export-summary", type=str, default="reports/orb_summary.json",
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

    # Resolver rutas por defecto (prioriza dataset canónico 2021-2023 CME Globex)
    if args.data is None:
        candidates_3y = [
            f"data/raw/CME_MINI_{std_sym}_2021_2023_5m.parquet",
            f"data/raw/CME_MINI_{std_sym}_2021_2023_5m.csv",
            f"data/raw/CME_MINI_{std_sym}_5m_3y.parquet",
            f"data/raw/CME_MINI_{std_sym}_5m_3y.csv",
            f"data/raw/CME_MINI_{std_sym}_5m_bars5000.parquet",
            f"data/raw/CME_MINI_{std_sym}_5m_bars5000.csv",
        ]
        for c in candidates_3y:
            if Path(c).exists():
                args.data = c
                break
        if args.data is None:
            args.data = candidates_3y[0]

    if args.daily_data is None:
        daily_candidates = [
            f"data/raw/CME_MINI_{std_sym}_2021_2023_1d.parquet",
            f"data/raw/CME_MINI_{std_sym}_2021_2023_1d.csv",
            f"data/raw/CME_MINI_{std_sym}_1d_3y.parquet",
            f"data/raw/CME_MINI_{std_sym}_1d_3y.csv",
            f"data/raw/CME_MINI_{std_sym}_1d_bars5000.parquet",
            f"data/raw/CME_MINI_{std_sym}_1d_bars5000.csv",
        ]
        for c in daily_candidates:
            if Path(c).exists():
                args.daily_data = c
                break

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

    # Normalizar a America/New_York según la especificación matemática formal
    df_intra = normalize_to_eastern(df_intra, source_tz=args.source_tz)
    df_intra.sort_index(inplace=True)
    df_intra, part_label = partition_from_args(df_intra, args, "F2_Filtered_ORB")
    print(f"  -> Rango intradía activo: {df_intra.index.min()} a {df_intra.index.max()}")

    # 2. Cargar datos diarios
    df_daily = None
    if args.daily_data and Path(args.daily_data).exists():
        print(f"[2/4] Cargando datos diarios desde '{args.daily_data}'...")
        dpath = Path(args.daily_data)
        if dpath.suffix == ".parquet":
            df_daily = pd.read_parquet(dpath)
        else:
            df_daily = pd.read_csv(dpath)
        if not isinstance(df_daily.index, pd.DatetimeIndex):
            col_name = "datetime" if "datetime" in df_daily.columns else ("Unnamed: 0" if "Unnamed: 0" in df_daily.columns else None)
            if col_name:
                raw_col = df_daily[col_name]
                df_daily.drop(columns=[col_name], inplace=True)
            else:
                raw_col = df_daily.index
            try:
                df_daily.index = pd.to_datetime(raw_col)
            except Exception:
                df_daily.index = pd.to_datetime(raw_col, utc=True).dt.tz_convert("America/New_York")
        df_daily.sort_index(inplace=True)
        df_daily = clip_to_partition(df_daily, df_intra)
    else:
        print("[2/4] Datos diarios no especificados. Agregando directamente desde datos intradía...")

    # 3. Resolver horarios de sesión en hora del Este (America/New_York)
    t_open = parse_time_str(args.session_open) if args.session_open else time(9, 30)
    t_or_end = parse_time_str(args.or_end) if args.or_end else time(9, 45)
    t_cutoff = parse_time_str(args.entry_cutoff) if args.entry_cutoff else time(12, 0)
    t_flat = parse_time_str(args.flat_time) if args.flat_time else time(15, 45)

    print(f"  -> Ventana de Opening Range : {t_open.strftime('%H:%M')} a {t_or_end.strftime('%H:%M')} EST")
    print(f"  -> Ventana de Disparo       : {t_or_end.strftime('%H:%M')} a {t_cutoff.strftime('%H:%M')} EST")
    print(f"  -> Cierre Forzoso Diario   : {t_flat.strftime('%H:%M')} EST")

    # 4. Instanciar Estrategia y Cuenta Prop Trading
    print(f"\n[3/4] Inicializando estrategia F2 y Cuenta Prop Trading (${args.capital:,.2f})...")
    strategy = FilteredQuantitativeORB(
        std_symbol=std_sym,
        micro_symbol=micro_sym,
        lambda_stretch=args.lambda_stretch,
        rvol_min=args.rvol_min,
        max_micros=args.max_micros,
        market_open=t_open,
        or_end=t_or_end,
        entry_cutoff=t_cutoff,
        flat_time=t_flat,
        slippage_ticks=args.slippage_ticks
    )

    account = PropAccount(
        initial_capital=args.capital,
        dll=args.dll,
        mtdd=args.mtdd,
        equity=args.capital,
        peak_equity=args.capital
    )

    # 5. Ejecutar el Backtester
    print("[4/4] Ejecutando simulación de mercado y auditoría de riesgo...")
    backtester = FilteredORBBacktester(strategy=strategy, account=account)
    report = backtester.run(intraday_data=df_intra, daily_data=df_daily, auto_normalize_tz=False)

    # 6. Mostrar Resultados Formateados
    print_banner("ESTRATEGIA F2: FILTERED QUANTITATIVE OPENING RANGE BREAKOUT (ORB)")
    print_table_row("Activo Primario / Micro", f"{std_sym} / {micro_sym}")
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
    print_table_row("Calmar Ratio", report['calmar_ratio'])

    print_banner("AUDITORÍA DE RIESGO PARA CUENTAS DE PROP TRADING")
    print_table_row("Límite de Pérdida Diaria (DLL)", f"${args.dll:,.2f}")
    print_table_row("Límite de Trailing Drawdown (MTDD)", f"${args.mtdd:,.2f}")
    status_str = "REPROBADA (BREACH)" if report["is_prop_breached"] else "APROBADA (COMPLIANT)"
    print_table_row("Estado de Evaluación Prop Firm", status_str)
    print_table_row("Detalle de Cumplimiento", report["breach_reason"])
    print("=" * 78 + "\n")

    # 7. Exportar Resultados
    trades_df = backtester.get_trades_dataframe()
    if not trades_df.empty:
        out_trades_path = Path(args.export_trades)
        out_trades_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(out_trades_path, index=False)
        print(f"-> Log de {len(trades_df)} operaciones exportado exitosamente a: {out_trades_path.resolve()}")

        # Mostrar tabla resumida de trades en consola
        print("\nÚltimas operaciones registradas:")
        cols_to_show = ["date", "symbol", "direction", "contracts", "entry_price", "exit_price", "exit_reason", "net_pnl_usd"]
        print(trades_df[cols_to_show].tail(10).to_string(index=False))

    out_summary_path = Path(args.export_summary)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"-> Reporte cuantitativo JSON exportado a: {out_summary_path.resolve()}\n")


if __name__ == "__main__":
    main()
