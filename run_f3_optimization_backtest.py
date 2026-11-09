"""
run_f3_optimization_backtest.py
Script ejecutable para correr el backtest formal de la versión alterna optimizada:
F3-Optimizacion -IS- N1: Anchored VWAP & Rotational Value Area Reversion

Cumple con:
- Microestructura CME Globex (ES, MES, NQ, MNQ).
- Comisiones y deslizamiento realistas de futuros.
- Reglas estrictas de Proprietary Trading (Daily Loss Limit, Trailing Drawdown flotante, conmutación estándar/micro).
- Filtro de régimen mejorado: Pendiente normalizada del AVWAP (kappa_norm), compuerta de aceptación AMT y asimetría de volumen.
- Prevención de Lookahead Bias y salidas dinámicas intradía (Trailing Stop al 60% de camino al VWAP, disyuntor 2.75 sigma).

Uso:
    python run_f3_optimization_backtest.py --symbol ES --mode in_sample
    python run_f3_optimization_backtest.py --symbol NQ --mode in_sample
    python run_f3_optimization_backtest.py --symbol ES --mode in_sample --compare-base
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.strategies.mean_reversion.f3_optimization_is_n1 import (
    F3OptimizationISN1Strategy,
    PropRiskManagerOptimized
)
from src.strategies.mean_reversion.anchored_vwap_value_area import (
    AnchoredVWAPValueAreaStrategy
)
from src.core.futures_specs import get_spec
from src.core.data_split import add_data_split_args, partition_from_args


def load_dataset(symbol: str, data_path: str = None) -> pd.DataFrame:
    """Carga los datos de 5 minutos desde archivo local o ruta especificada."""
    sym = symbol.upper()
    if data_path:
        path = Path(data_path)
    else:
        candidates = [
            Path(f"data/raw/CME_MINI_{sym}_2021_2023_5m.parquet"),
            Path(f"data/raw/CME_MINI_{sym}_2021_2023_5m.csv"),
            Path(f"data/raw/CME_MINI_{sym}_5m_3y.parquet"),
            Path(f"data/raw/CME_MINI_{sym}_5m_3y.csv"),
            Path(f"data/raw/CME_MINI_{sym}_5m_bars5000.csv"),
            Path(f"data/raw/CME_MINI_{sym}_5m_bars5000.parquet"),
            Path(f"data/raw/CME_MINI_ES_2021_2023_5m.parquet") if "ES" in sym else Path(f"data/raw/CME_MINI_NQ_2021_2023_5m.parquet"),
            Path(f"data/raw/CME_MINI_ES_5m_3y.parquet") if "ES" in sym else Path(f"data/raw/CME_MINI_NQ_5m_3y.parquet"),
            Path(f"data/raw/CME_MINI_ES_5m_bars5000.csv") if "ES" in sym else Path(f"data/raw/CME_MINI_NQ_5m_bars5000.csv")
        ]
        path = None
        for cand in candidates:
            if cand.exists():
                path = cand
                break

    if path is None or not path.exists():
        raise FileNotFoundError(
            f"No se encontró archivo de datos para {sym}. Especifique --data-path explícitamente."
        )

    print(f"[*] Cargando dataset desde: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida: {col} en el dataset.")

    df = df[required].copy().sort_index()
    df.dropna(inplace=True)
    return df


def print_banner():
    banner = """
========================================================================================
         ESTRATEGIA F3-OPTIMIZACION -IS- N1: FILTRO DE REGIMEN Y PROP RISK
    Fundamento: Steidlmayer (1986), Madhavan (2002), Gatheral (2010)
    Mejoras: Pendiente Normalizada de VWAP, Aceptación de Área de Valor & Skewness
    Marco de Riesgo: Proprietary Trading (Apex / Topstep / Lucid / MyFundedFutures)
========================================================================================
"""
    print(banner)


def print_metrics_table(metrics: dict, title: str = "RESULTADOS DEL BACKTEST"):
    print("\n" + "=" * 68)
    print(f" {title:^66} ")
    print("=" * 68)
    for key, val in metrics.items():
        if key == "Desglose de Salidas":
            print(f"  {key:<34} :")
            if isinstance(val, dict) and val:
                for r_name, r_count in val.items():
                    print(f"    - {r_name:<30} : {r_count}")
            else:
                print("    - Ninguna")
        else:
            print(f"  {key:<34} : {val}")
    print("=" * 68 + "\n")


def print_comparison_table(metrics_base: dict, metrics_opt: dict, symbol: str):
    print("\n" + "=" * 80)
    print(f" {'COMPARATIVA CUANTITATIVA: F3 BASE vs F3-OPTIMIZACION -IS- N1 (' + symbol + ')':^78} ")
    print("=" * 80)
    header = f"  {'Métrica':<32} | {'F3 Base':<18} | {'F3-Optimizada':<18} | {'Delta':<10}"
    print(header)
    print("-" * 80)

    keys = [
        ("P&L Neto Total ($)", "$", True),
        ("Retorno Total (%)", "%", True),
        ("Max Trailing Drawdown ($)", "$", False),
        ("Max Trailing Drawdown (%)", "%", False),
        ("Total Trades", "", False),
        ("Win Rate (%)", "%", True),
        ("Profit Factor", "", True),
        ("Trade Promedio ($)", "$", True)
    ]

    for k, unit, higher_is_better in keys:
        v_base = metrics_base.get(k, "N/A")
        v_opt = metrics_opt.get(k, "N/A")
        if isinstance(v_base, (int, float)) and isinstance(v_opt, (int, float)):
            diff = v_opt - v_base
            diff_str = f"{diff:+.2f}{unit}"
            print(f"  {k:<32} | {v_base:>15.2f}{unit} | {v_opt:>15.2f}{unit} | {diff_str:>10}")
        else:
            print(f"  {k:<32} | {str(v_base):>16} | {str(v_opt):>16} | {'-':>10}")
    print("=" * 80 + "\n")


def print_trades_table(df_trades: pd.DataFrame, max_trades: int = 15):
    if df_trades.empty:
        print("[*] No se ejecutaron operaciones en el período.")
        return

    print("\n" + "=" * 115)
    print(f" {'REGISTRO DE OPERACIONES AUDITADAS (ÚLTIMAS ' + str(min(len(df_trades), max_trades)) + ')':^113} ")
    print("=" * 115)
    headers = f"  {'Entrada':<16} | {'Salida':<16} | {'Tipo':<5} | {'Cnt':<3} | {'Precio In':<10} | {'Precio Out':<10} | {'Net PnL ($)':<12} | {'Razón Salida':<22}"
    print(headers)
    print("-" * 115)

    tail_trades = df_trades.tail(max_trades)
    for _, tr in tail_trades.iterrows():
        t_in = str(tr['entry_time'])[:16]
        t_out = str(tr['exit_time'])[:16]
        d = tr['direction']
        cnt = f"{tr['contracts']} {tr['contract_type'][:3]}"
        p_in = f"{tr['entry_price']:.2f}"
        p_out = f"{tr['exit_price']:.2f}"
        net = f"${tr['net_pnl']:+,.2f}"
        reason = str(tr['exit_reason'])
        print(f"  {t_in:<16} | {t_out:<16} | {d:<5} | {cnt:<7} | {p_in:<10} | {p_out:<10} | {net:<12} | {reason:<22}")
    print("=" * 115 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Cuantitativo Formal - Estrategia F3-Optimizacion -IS- N1"
    )
    parser.add_argument("--symbol", type=str, default="ES", choices=["ES", "MES", "NQ", "MNQ"],
                        help="Símbolo del contrato CME (ES, MES, NQ, MNQ)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Ruta al archivo CSV o Parquet con velas de 5 minutos")
    parser.add_argument("--account-size", type=float, default=50000.0,
                        help="Tamaño de cuenta de Prop Trading en USD (ej. 50,000)")
    parser.add_argument("--dll", type=float, default=1000.0,
                        help="Daily Loss Limit (DLL) en USD")
    parser.add_argument("--mtdd", type=float, default=2000.0,
                        help="Maximum Trailing Drawdown (MTDD) en USD")
    parser.add_argument("--k-band", type=float, default=None,
                        help="Múltiplo de desviación estándar para entrada (2.20 para ES, 2.5 para NQ)")
    parser.add_argument("--kappa-crit", type=float, default=None,
                        help="Umbral crítico de pendiente normalizada de VWAP (2.2 por defecto)")
    parser.add_argument("--convergence-protect-ratio", type=float, default=0.50,
                        help="Ratio de convergencia hacia VWAP para proteger (0.50 calibrado [default], 0.60 base)")
    parser.add_argument("--anchor-time", type=str, default="09:30",
                        help="Hora de anclaje de la sesión regular (HH:MM)")
    parser.add_argument("--compare-base", action="store_true",
                        help="Ejecutar y contrastar contra la estrategia base F3")
    parser.add_argument("--print-trades", action="store_true",
                        help="Imprimir tabla detallada de operaciones ejecutadas")
    parser.add_argument("--output-csv", type=str, default="reports/f3_optimization_n1_trades.csv",
                        help="Ruta para exportar el log de operaciones a CSV (por defecto: reports/f3_optimization_n1_trades.csv)")
    add_data_split_args(parser)

    args = parser.parse_args()

    print_banner()

    # 1. Cargar datos
    data = load_dataset(symbol=args.symbol, data_path=args.data_path)
    data, part_label = partition_from_args(data, args, "F3_Optimization_IS_N1")
    print(f"[*] Rango de Datos: {data.index[0]} a {data.index[-1]} ({len(data):,} barras de 5m)")

    # 2. Configurar estrategia optimizada
    k_band = args.k_band
    if k_band is None:
        k_band = 2.5 if args.symbol.upper() in ["NQ", "MNQ"] else 2.20

    kappa_crit = args.kappa_crit
    if kappa_crit is None:
        kappa_crit = 2.2

    opt_params = {
        "symbol": args.symbol,
        "anchor_time": args.anchor_time,
        "window_init_end": "10:15",
        "window_revers_end": "15:00",
        "window_inhibit_end": "15:15",
        "k_band": k_band,
        "k_catastrophic": 2.75,
        "kappa_crit": kappa_crit,
        "slope_lookback_bars": 3,
        "max_vol_outside_ratio": 0.55,
        "min_kurtosis": 1.7,
        "min_dist_to_vwap_ratio": 0.80,
        "k_buffer_ticks": 3,
        "convergence_protect_ratio": args.convergence_protect_ratio,
        "account_size": args.account_size,
        "daily_loss_limit": args.dll,
        "max_trailing_drawdown": args.mtdd,
        "psi": 0.20,
        "max_std_contracts": 1,
        "max_micro_contracts": 10,
        "alpha_dll": 0.85,
        "max_intraday_consecutive_losses": 2,
        "bar_minutes": 5.0
    }

    opt_strategy = F3OptimizationISN1Strategy(params=opt_params)

    # 3. Ejecutar Backtest de Estrategia Optimizada
    print(f"[*] Ejecutando backtest de F3-Optimizacion -IS- N1 con Prop Risk Management...")
    results_opt = opt_strategy.run_backtest(data)
    metrics_opt = results_opt["metrics"]
    trades_opt_df = results_opt["trades"]

    print_metrics_table(metrics_opt, title=f"RESULTADOS F3-OPTIMIZACION -IS- N1 - {args.symbol.upper()} (5m)")

    # 4. Comparación con Estrategia Base si fue solicitada
    if args.compare_base:
        print("[*] Ejecutando backtest de Estrategia Base F3 para comparación...")
        base_params = {
            "symbol": args.symbol,
            "anchor_time": args.anchor_time,
            "k_band": k_band,
            "k_catastrophic": 3.0,
            "delta_max_slope": 0.45 if args.symbol.upper() in ["NQ", "MNQ"] else 0.15,
            "adx_threshold": 25.0,
            "account_size": args.account_size,
            "daily_loss_limit": args.dll,
            "max_trailing_drawdown": args.mtdd,
            "psi": 0.20,
            "k_buffer_ticks": 4,
            "bar_minutes": 5.0
        }
        base_strategy = AnchoredVWAPValueAreaStrategy(params=base_params)
        results_base = base_strategy.run_backtest(data)
        metrics_base = results_base["metrics"]
        print_comparison_table(metrics_base, metrics_opt, symbol=args.symbol.upper())

    if args.print_trades or len(trades_opt_df) > 0:
        print_trades_table(trades_opt_df)

    if args.output_csv:
        out_p = Path(args.output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        trades_opt_df.to_csv(out_p, index=False)
        print(f"[*] Registro de operaciones exportado exitosamente a: {out_p.resolve()}")

    trailing_dd = metrics_opt["Max Trailing Drawdown ($)"]
    if trailing_dd > args.mtdd:
        print(f"\n[!] ALERTA: Trailing Drawdown (${trailing_dd:,.2f}) excedió el MTDD (${args.mtdd:,.2f}).")
    else:
        print(f"\n[OK] Cumplimiento de Riesgo Prop Trading: Trailing DD dentro del límite admisible (${trailing_dd:,.2f} <= ${args.mtdd:,.2f}).")


if __name__ == "__main__":
    main()
