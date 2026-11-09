"""
run_f2_optimizacion_is_n1_backtest.py
Script ejecutable para correr el backtest de la Versión Alterna Optimizada:
Estrategia "F2-Optimizacion -IS- N1": Filtered Quantitative Opening Range Breakout (ORB).

Mejoras Matemáticas Integradas:
1. Supresión de Falsas Rupturas (Fakeouts) mediante Asimetría de Flujo de Órdenes (OFA / CVD)
   con umbral dinámico ajustado por volatilidad (theta_flow).
2. Umbral de Volumen Relativo Crítico Continuo RVOL_crit(R_comp) acotado en [1.20, 1.70].
3. Reemplazo del Stop Loss rígido en punto medio por una Barrera Elástica Proporcional a la anchura
   del rango inicial (k_elastic * Range_OR), acotada por la volatilidad diaria (0.85 * ATR_daily).
4. Breakeven Dinámico Protector al alcanzar +1.0R (mitad del objetivo Take-Profit 2.0R).
5. Cumplimiento estricto de reglas de Prop Trading (DLL $1,000, MTDD $2,000, atenuación por racha phi(K_loss)
   y conmutación automática estándar/micro CME).

Uso:
    python run_f2_optimizacion_is_n1_backtest.py --symbol ES
    python run_f2_optimizacion_is_n1_backtest.py --symbol NQ
    python run_f2_optimizacion_is_n1_backtest.py --symbol ES --data data/raw/CME_EXT_ES_5m.csv --daily-data data/raw/CME_EXT_ES_1d.csv
"""

import argparse
from datetime import time
import json
from pathlib import Path
import sys
from typing import Any
import pandas as pd

from src.strategies.f2_optimizacion_is_n1 import (
    F2OptimizationISN1,
    F2OptimizationISN1Backtester,
)
from src.strategies.filtered_orb import (
    PropAccount,
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


def print_banner(text: str, width: int = 82):
    print("\n" + "=" * width)
    print(f" {text:^{width-2}} ")
    print("=" * width)


def print_table_row(k: str, v: Any, width: int = 82):
    col1 = 46
    col2 = width - col1 - 5
    print(f"  {k:<{col1}} : {str(v):>{col2}}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest de Estrategia F2-Optimizacion -IS- N1: Filtered Quantitative ORB Optimizado"
    )
    parser.add_argument("--symbol", type=str, default="ES", choices=["ES", "MES", "NQ", "MNQ"],
                        help="Símbolo del contrato CME (por defecto: ES)")
    parser.add_argument("--data", type=str, default=None,
                        help="Ruta al archivo CSV o Parquet con velas intradía (5m)")
    parser.add_argument("--daily-data", type=str, default=None,
                        help="Ruta al archivo CSV o Parquet con velas diarias (1d)")
    parser.add_argument("--capital", type=float, default=50000.0,
                        help="Capital inicial de la cuenta en USD (def: 50,000)")
    parser.add_argument("--dll", type=float, default=1000.0,
                        help="Límite de Pérdida Diaria (Daily Loss Limit) en USD (def: 1,000)")
    parser.add_argument("--mtdd", type=float, default=2000.0,
                        help="Límite de Trailing Drawdown Máximo en USD (def: 2,000)")
    parser.add_argument("--lambda-stretch", type=float, default=0.40,
                        help="Multiplicador del amortiguador de ruido Stretch (def: 0.40)")
    parser.add_argument("--base-flow", type=float, default=0.05,
                        help="Umbral base de asimetría de flujo de órdenes OFA (def: 0.05)")
    parser.add_argument("--k-elastic", type=float, default=0.50,
                        help="Multiplicador proporcional del rango para la barrera elástica (def: 0.50)")
    parser.add_argument("--atr-cap-mult", type=float, default=0.85,
                        help="Tope máximo de riesgo acotado por ATR diario (def: 0.85)")
    parser.add_argument("--rvol-floor", type=float, default=1.20,
                        help="Umbral mínimo de RVOL en máxima compresión (def: 1.20)")
    parser.add_argument("--rvol-cap", type=float, default=1.70,
                        help="Umbral máximo de RVOL en compresión moderada (def: 1.70)")
    parser.add_argument("--no-breakeven", action="store_true", default=False,
                        help="Desactiva el mecanismo elástico de breakeven al alcanzar +1.0R")
    parser.add_argument("--max-micros", type=int, default=12,
                        help="Máximo número de microcontratos permitidos (def: 12)")
    parser.add_argument("--slippage-ticks", type=float, default=1.0,
                        help="Ticks de deslizamiento por orden (def: 1.0 tick)")
    parser.add_argument("--source-tz", type=str, default=None,
                        help="Zona horaria de origen de los datos brutos (def: auto-detect)")
    parser.add_argument("--session-open", type=str, default=None,
                        help="Hora de apertura de sesión (ej. '09:30' America/New_York)")
    parser.add_argument("--or-end", type=str, default=None,
                        help="Hora de cierre del Opening Range (ej. '09:45' America/New_York)")
    parser.add_argument("--entry-cutoff", type=str, default=None,
                        help="Hora límite para cancelación de órdenes pendientes (ej. '12:00' America/New_York)")
    parser.add_argument("--flat-time", type=str, default=None,
                        help="Hora forzosa de neutralización diaria (ej. '15:45' America/New_York)")
    parser.add_argument("--export-trades", type=str, default="reports/f2_optimizacion_is_n1_trades.csv",
                        help="Ruta para exportar el log de operaciones en CSV")
    parser.add_argument("--export-summary", type=str, default="reports/f2_optimizacion_is_n1_summary.json",
                        help="Ruta para exportar el resumen de métricas en JSON")
    parser.add_argument("--all-data", action="store_true", default=False,
                        help="Ejecuta sobre la totalidad de los datos sin partición estricta de 3 años")
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

    # Resolver rutas por defecto (prioriza datasets canónicos 3y si se solicita partición específica)
    is_partition_mode = getattr(args, "split_mode", None) in ["in_sample", "out_of_sample", "holdout"]
    if args.data is None:
        if is_partition_mode:
            candidates_data = [
                f"data/raw/CME_MINI_{std_sym}_2021_2023_5m.parquet",
                f"data/raw/CME_MINI_{std_sym}_2021_2023_5m.csv",
                f"data/raw/CME_MINI_{std_sym}_5m_3y.parquet",
                f"data/raw/CME_MINI_{std_sym}_5m_3y.csv",
                f"data/raw/CME_EXT_{std_sym}_5m.csv",
                f"data/raw/CME_EXT_{std_sym}_5m.parquet",
                f"data/raw/CME_MINI_{std_sym}_5m_bars5000.parquet",
                f"data/raw/CME_MINI_{std_sym}_5m_bars5000.csv",
            ]
        else:
            candidates_data = [
                f"data/raw/CME_MINI_{std_sym}_2021_2023_5m.parquet",
                f"data/raw/CME_MINI_{std_sym}_2021_2023_5m.csv",
                f"data/raw/CME_EXT_{std_sym}_5m.csv",
                f"data/raw/CME_EXT_{std_sym}_5m.parquet",
                f"data/raw/CME_MINI_{std_sym}_5m_3y.parquet",
                f"data/raw/CME_MINI_{std_sym}_5m_3y.csv",
                f"data/raw/CME_MINI_{std_sym}_5m_bars5000.parquet",
                f"data/raw/CME_MINI_{std_sym}_5m_bars5000.csv",
            ]
        for c in candidates_data:
            if Path(c).exists():
                args.data = c
                break
        if args.data is None:
            args.data = candidates_data[0]

    if args.daily_data is None:
        if is_partition_mode:
            daily_candidates = [
                f"data/raw/CME_MINI_{std_sym}_2021_2023_1d.parquet",
                f"data/raw/CME_MINI_{std_sym}_2021_2023_1d.csv",
                f"data/raw/CME_MINI_{std_sym}_1d_3y.parquet",
                f"data/raw/CME_MINI_{std_sym}_1d_3y.csv",
                f"data/raw/CME_EXT_{std_sym}_1d.csv",
                f"data/raw/CME_EXT_{std_sym}_1d.parquet",
                f"data/raw/CME_MINI_{std_sym}_1d_bars5000.parquet",
                f"data/raw/CME_MINI_{std_sym}_1d_bars5000.csv",
            ]
        else:
            daily_candidates = [
                f"data/raw/CME_MINI_{std_sym}_2021_2023_1d.parquet",
                f"data/raw/CME_MINI_{std_sym}_2021_2023_1d.csv",
                f"data/raw/CME_EXT_{std_sym}_1d.csv",
                f"data/raw/CME_EXT_{std_sym}_1d.parquet",
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
        print(f"Error: Archivo de datos intradía no encontrado: {data_path}")
        sys.exit(1)

    if data_path.suffix == ".parquet":
        df_intra = pd.read_parquet(data_path)
    else:
        df_intra = pd.read_csv(data_path)

    # Normalizar a America/New_York
    df_intra = normalize_to_eastern(df_intra, source_tz=args.source_tz)
    if args.all_data:
        part_label = f"TOTALIDAD DE DATOS (Sin partición) | {df_intra.index.min()} -> {df_intra.index.max()} | {len(df_intra):,} barras"
    else:
        try:
            df_intra, part_label = partition_from_args(df_intra, args, "F2_Optimizacion_IS_N1")
        except ValueError as e:
            print(f"  [AVISO] Partición canónica 3Y no aplicable ({e}). Usando totalidad de los datos disponibles.")
            part_label = f"EXTENDED_DATASET | {df_intra.index.min()} -> {df_intra.index.max()} | {len(df_intra):,} barras"
    print(f"  -> Rango activo ({part_label}): {df_intra.index.min()} a {df_intra.index.max()} ({len(df_intra)} barras)")

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
        print(f"  -> Rango diario alineado: {df_daily.index.min()} a {df_daily.index.max()} ({len(df_daily)} sesiones)")
    else:
        print("[2/4] Datos diarios no especificados. Agregando directamente desde datos intradía...")

    # 3. Resolver horarios de sesión en hora del Este (America/New_York)
    t_open = parse_time_str(args.session_open) if args.session_open else time(9, 30)
    t_or_end = parse_time_str(args.or_end) if args.or_end else time(9, 45)
    t_cutoff = parse_time_str(args.entry_cutoff) if args.entry_cutoff else time(12, 0)
    t_flat = parse_time_str(args.flat_time) if args.flat_time else time(15, 45)

    print(f"  -> Ventana de Opening Range   : {t_open.strftime('%H:%M')} a {t_or_end.strftime('%H:%M')} EST")
    print(f"  -> Ventana de Ruptura         : {t_or_end.strftime('%H:%M')} a {t_cutoff.strftime('%H:%M')} EST")
    print(f"  -> Cierre Forzoso Diario      : {t_flat.strftime('%H:%M')} EST")

    # 4. Instanciar Estrategia Optimizada y Cuenta Prop Trading
    print(f"\n[3/4] Inicializando estrategia F2-Optimizacion -IS- N1 y Cuenta Prop Trading (${args.capital:,.2f})...")
    strategy = F2OptimizationISN1(
        std_symbol=std_sym,
        micro_symbol=micro_sym,
        lambda_stretch=args.lambda_stretch,
        base_flow_threshold=args.base_flow,
        k_elastic=args.k_elastic,
        atr_cap_mult=args.atr_cap_mult,
        rvol_floor=args.rvol_floor,
        rvol_cap=args.rvol_cap,
        enable_breakeven=(not args.no_breakeven),
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

    # 5. Ejecutar el Backtester Optimizado
    print("[4/4] Ejecutando simulación de mercado y auditoría de riesgo...")
    backtester = F2OptimizationISN1Backtester(strategy=strategy, account=account)
    report = backtester.run(intraday_data=df_intra, daily_data=df_daily, auto_normalize_tz=False)

    # 6. Mostrar Resultados Formateados
    print_banner("ESTRATEGIA OPTIMIZADA: F2-OPTIMIZACION -IS- N1 (FILTERED QUANTITATIVE ORB)")
    print_table_row("Activo Primario / Micro", f"{std_sym} / {micro_sym}")
    print_table_row("Partición de Datos", part_label)
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

    print_banner("MÉTRICAS ESPECÍFICAS DE OPTIMIZACIÓN MICROESTRUCTURAL (F2-N1)")
    print_table_row("Falsas Rupturas Evitadas (Fakeouts)", f"{report['fakeouts_avoided_total']} (L: {report['fakeouts_avoided_long']}, S: {report['fakeouts_avoided_short']})")
    print_table_row("Activaciones de Breakeven Protector", report['breakeven_activations'])
    print_table_row("Operaciones Beneficiadas con Breakeven", report['trades_with_breakeven'])
    print_table_row("RVOL Dinámico Rango Permitido", f"[{args.rvol_floor:.2f}, {args.rvol_cap:.2f}]")
    print_table_row("Barrera Elástica Proporcional (k)", f"{args.k_elastic:.2f} * Range_OR (Cap: {args.atr_cap_mult:.2f} * ATR)")

    print_banner("AUDITORÍA DE RIESGO PARA CUENTAS DE PROP TRADING")
    print_table_row("Límite de Pérdida Diaria (DLL)", f"${args.dll:,.2f}")
    print_table_row("Límite de Trailing Drawdown (MTDD)", f"${args.mtdd:,.2f}")
    status_str = "REPROBADA (BREACH)" if report["is_prop_breached"] else "APROBADA (COMPLIANT)"
    print_table_row("Estado de Evaluación Prop Firm", status_str)
    print_table_row("Detalle de Cumplimiento", report["breach_reason"])
    print("=" * 82 + "\n")

    # 7. Exportar Resultados
    trades_df = backtester.get_trades_dataframe()
    if not trades_df.empty:
        out_trades_path = Path(args.export_trades)
        out_trades_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(out_trades_path, index=False)
        print(f"-> Log de {len(trades_df)} operaciones exportado exitosamente a: {out_trades_path.resolve()}")

        # Mostrar tabla resumida de trades en consola
        print("\nÚltimas operaciones registradas:")
        cols_to_show = ["date", "symbol", "direction", "contracts", "entry_price", "exit_price", "exit_reason", "breakeven_triggered", "net_pnl_usd"]
        print(trades_df[cols_to_show].tail(10).to_string(index=False))

    out_summary_path = Path(args.export_summary)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"-> Reporte cuantitativo JSON exportado a: {out_summary_path.resolve()}\n")


if __name__ == "__main__":
    main()
