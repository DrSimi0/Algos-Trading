"""
run_f1_opt_backtest.py
Script ejecutable para correr el backtest de la versión optimizada:
Estrategia F1-Optimizacion -IS- N1: Market Intraday Momentum & MOC Seasonality.

Mejoras Cuantitativas Evaluadas:
1. Normalización del retorno de apertura [09:30, 10:00) EST contra la volatilidad realizada intradía.
2. Filtro bilateral de apertura:
   - Filtro de Puro Ruido: Descarta movimientos con señal/ruido insuficiente (Z_norm < Z_noise_min).
   - Filtro de Agotamiento de Rango: Descarta sesiones donde la apertura ya consumió el rango esperado del día
     (Z_norm > Z_exhaust_max o Range_open / ATR_daily > max_range_fraction).
3. Escalado dinámico del tamaño de posición según el colchón restante del Daily Loss Limit (DLL).
4. Filtro de coherencia tendencial de cierre con el VWAP acumulado a las 15:30 EST.
5. Stop Loss adaptativo al ATR intradía y Take Profit asimétrico (1.6R).
6. Salida anticipada por decaimiento de momento (Momentum Decay Exit) a partir de las 15:45 EST.
7. Liquidación forzosa MOC a las 15:58:00 EST.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import pandas as pd

from src.strategies.momentum.f1_optimizacion_is_n1 import (
    MarketIntradayMomentumOptN1,
    IntradayMomentumOptN1Backtester,
)
from src.portfolio.prop_firm_rules import PROFILES
from src.core.data_split import add_data_split_args, partition_from_args


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest cuantitativo ejecutable para Estrategia F1-Optimizacion -IS- N1"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="ES",
        choices=["ES", "MES", "NQ", "MNQ"],
        help="Símbolo principal de futuros CME (ES o NQ). Por defecto: ES"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        choices=["1m", "5m", "15m"],
        help="Intervalo intradía de las barras. Por defecto: 5m"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=list(PROFILES.keys()),
        help="Perfil preconfigurado de firma de fondeo (ej: topstep_50k, lucid_50k)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=50000.0,
        help="Capital inicial de la cuenta en USD. Por defecto: 50,000"
    )
    parser.add_argument(
        "--dll",
        type=float,
        default=1000.0,
        help="Límite de pérdida diaria (Daily Loss Limit) en USD. Por defecto: 1,000"
    )
    parser.add_argument(
        "--mtdd",
        type=float,
        default=2000.0,
        help="Límite de Max Trailing Drawdown en USD. Por defecto: 2,000"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=3000.0,
        help="Objetivo de beneficio (Profit Target) en USD. Por defecto: 3,000"
    )
    parser.add_argument(
        "--trailing-mode",
        type=str,
        default="eod",
        choices=["eod", "intraday"],
        help="Modo de cálculo del trailing drawdown: 'eod' o 'intraday'. Por defecto: eod"
    )
    parser.add_argument(
        "--z-noise-min",
        type=float,
        default=0.60,
        help="Umbral mínimo de retorno normalizado por ATR para filtrar puro ruido (def: 0.60)"
    )
    parser.add_argument(
        "--z-exhaust-max",
        type=float,
        default=2.50,
        help="Umbral máximo de retorno normalizado para filtrar agotamiento de rango (def: 2.50)"
    )
    parser.add_argument(
        "--max-range-fraction",
        type=float,
        default=0.65,
        help="Fracción máxima del ATR diario consumida por la apertura antes de considerarse agotada (def: 0.65)"
    )
    parser.add_argument(
        "--cushion-exponent",
        type=float,
        default=1.0,
        help="Exponente de elasticidad para el escalado con el colchón DLL (1.0 = lineal, def: 1.0)"
    )
    parser.add_argument(
        "--disable-vwap",
        action="store_true",
        help="Desactiva el filtro de coherencia VWAP vespertino"
    )
    parser.add_argument(
        "--disable-decay-exit",
        action="store_true",
        help="Desactiva la salida anticipada por decaimiento de momento a las 15:45 EST"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Ruta opcional para exportar el registro detallado de trades en CSV"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Ruta a archivo de datos personalizado (parquet o csv)"
    )
    add_data_split_args(parser)
    return parser.parse_args()


def load_market_data(symbol: str, interval: str, custom_path: str = None) -> pd.DataFrame:
    """Carga los datos intradía disponibles en data/raw (prioriza series reales CME 2021-2023)."""
    if custom_path:
        p = Path(custom_path)
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p, index_col=0, parse_dates=True)

    base_sym = symbol.upper()
    underlying = base_sym[1:] if base_sym.startswith("M") else base_sym

    data_dir = ROOT_DIR / "data" / "raw"
    candidates = [
        data_dir / f"CME_MINI_{underlying}_2021_2023_{interval}.parquet",
        data_dir / f"CME_MINI_{underlying}_2021_2023_{interval}.csv",
        data_dir / f"CME_MINI_{underlying}_{interval}_3y.parquet",
        data_dir / f"CME_MINI_{underlying}_{interval}_3y.csv",
        data_dir / f"CME_MINI_{underlying}_{interval}_bars5000.parquet",
        data_dir / f"CME_MINI_{underlying}_{interval}_bars5000.csv",
    ]

    for p in candidates:
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p, index_col=0, parse_dates=True)

    raise FileNotFoundError(
        f"No se encontraron datos históricos para {underlying} ({interval}) en {data_dir}.\n"
        f"Verifique la existencia de datos reales CME Globex (ej: CME_MINI_ES_2021_2023_5m.parquet)."
    )


def main():
    args = parse_arguments()

    account_size = args.capital
    daily_loss_limit = args.dll
    max_trailing_loss = args.mtdd
    profit_target = args.target
    trailing_mode = args.trailing_mode
    lock_floor_at_start = True

    if args.profile and args.profile in PROFILES:
        prof = PROFILES[args.profile]
        account_size = prof.account_size
        profit_target = prof.profit_target
        max_trailing_loss = prof.max_loss
        daily_loss_limit = prof.daily_loss_limit
        trailing_mode = prof.trailing
        lock_floor_at_start = prof.lock_floor_at_start

    std_symbol = "NQ" if "NQ" in args.symbol else "ES"
    micro_symbol = "MNQ" if "NQ" in args.symbol else "MES"
    dll_str = f"${daily_loss_limit:,.2f} USD" if daily_loss_limit is not None else "N/A"

    print("=" * 84)
    print(" MOTOR DE BACKTEST CUANTITATIVO: ESTRATEGIA F1-OPTIMIZACION -IS- N1")
    print(" Modelo: Market Intraday Momentum & MOC Seasonality con Filtrado Avanzado y Escalado DLL")
    print("=" * 84)
    print(f" Instrumento:                {std_symbol} (Estándar) / {micro_symbol} (Micro)")
    print(f" Intervalo:                  {args.interval}")
    print(f" Capital Inicial:            ${account_size:,.2f} USD")
    print(f" Daily Loss Limit (DLL):     {dll_str}")
    print(f" Max Trailing DD (MTDD):     ${max_trailing_loss:,.2f} USD ({trailing_mode.upper()})")
    print(f" Profit Target:              ${profit_target:,.2f} USD")
    print(f" Filtro Puro Ruido:          Z_norm >= {args.z_noise_min:.2f} (|r_first| / sigma_atr)")
    print(f" Filtro Agotamiento Rango:   Z_norm <= {args.z_exhaust_max:.2f} y Range_open/ATR_daily <= {args.max_range_fraction:.2f}")
    print(f" Escalado Colchón DLL:       B_DLL = phi * C_DLL * (C_DLL / DLL)^{args.cushion_exponent:.1f}")
    print(f" Coherencia VWAP a 15:30:    {'DESACTIVADO' if args.disable_vwap else 'ACTIVADO (Delta >= 2 ticks)'}")
    print(f" Decaimiento Momento 15:45:  {'DESACTIVADO' if args.disable_decay_exit else 'ACTIVADO (<40% TP y vel<=0)'}")
    print("=" * 84)

    # 1. Cargar datos
    print(f"\n [1/3] Cargando datos de mercado...")
    df = load_market_data(std_symbol, args.interval, custom_path=args.data)
    df, part_label = partition_from_args(df, args, "F1_Optimizacion_IS_N1")
    start_dt = df.index.min().strftime("%Y-%m-%d %H:%M")
    end_dt = df.index.max().strftime("%Y-%m-%d %H:%M")
    print(f"       -> {len(df):,} barras cargadas desde {start_dt} hasta {end_dt}.")

    # 2. Configurar estrategia y simulador
    print(f"\n [2/3] Inicializando F1-Optimizacion -IS- N1 y simulador institucional...")
    strategy = MarketIntradayMomentumOptN1(params={
        "symbol": std_symbol,
        "std_symbol": std_symbol,
        "micro_symbol": micro_symbol,
        "source_tz": "America/New_York",
        "target_tz": "America/New_York",
        "z_noise_min": args.z_noise_min,
        "z_exhaust_max": args.z_exhaust_max,
        "max_range_fraction": args.max_range_fraction,
        "cushion_exponent": args.cushion_exponent,
        "vwap_coherence": not args.disable_vwap,
        "enable_decay_exit": not args.disable_decay_exit,
    })

    backtester = IntradayMomentumOptN1Backtester(
        strategy=strategy,
        account_size=account_size,
        daily_loss_limit=daily_loss_limit,
        max_trailing_loss=max_trailing_loss,
        trailing_mode=trailing_mode,
        lock_floor_at_start=lock_floor_at_start,
        profit_target=profit_target,
    )

    # 3. Ejecutar simulación
    print(f"\n [3/3] Ejecutando simulación sesión a sesión con reglas institucionales CME...")
    results = backtester.run(df)

    trades = results["trades"]
    metrics = results["metrics"]
    prop_eval = results["prop_evaluation"]
    skipped = results["skipped_sessions"]

    # 4. Imprimir registro detallado de operaciones
    print("\n" + "=" * 84)
    print(" REGISTRO DE TRADES EJECUTADOS (F1-OPTIMIZACION -IS- N1)")
    print("=" * 84)
    if trades.empty:
        print(" [AVISO] Ninguna sesión cumplió conjuntamente los filtros de señal optimizada.")
    else:
        fmt = "{:<11} {:<6} {:<5} {:<4} {:>9} {:>9} {:<6} {:>9} {:>8} {:>10} {:>10} {:>7}"
        print(fmt.format("Fecha", "Señal", "Sym", "Qty", "Entrada", "Salida", "Razón", "P&L Bruto", "Fricción", "P&L Neto", "Saldo USD", "Colchón"))
        print("-" * 98)
        for _, t in trades.iterrows():
            date_str = str(t["date"])
            gross_str = f"${t['gross_pnl']:+,.2f}"
            fric_str = f"-${t['friction']:,.2f}"
            net_str = f"${t['net_pnl']:+,.2f}"
            bal_str = f"${t['balance']:,.2f}"
            cush_str = f"{t['cushion_ratio']*100:.0f}%"
            print(fmt.format(
                date_str,
                t["signal"],
                t["symbol"],
                t["contracts"],
                f"{t['entry_price']:.2f}",
                f"{t['exit_price']:.2f}",
                t["exit_reason"],
                gross_str,
                fric_str,
                net_str,
                bal_str,
                cush_str
            ))

    # 5. Métricas cuantitativas
    print("\n" + "=" * 84)
    print(" MÉTRICAS CUANTITATIVAS DE RENDIMIENTO (F1-OPTIMIZACION -IS- N1)")
    print("=" * 84)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f" {k:<34}: {v:>12,.2f}")
        else:
            print(f" {k:<34}: {v:>12}")

    print(f" {'Sesiones Descartadas por Filtro':<34}: {len(skipped):>12}")

    # 6. Evaluación de Prop Trading
    print("\n" + "=" * 84)
    print(" AUDITORÍA DE EVALUACIÓN DE PROP TRADING")
    print("=" * 84)
    status_icon = "[PASS] APROBADO" if prop_eval["status"] == "PASS" else (
        "[FAIL] REPROBADO" if prop_eval["status"] == "FAIL" else "[ACTIVE] EN CURSO"
    )
    print(f" Estado Final:                 {status_icon}")
    print(f" Causa / Detalle:              {prop_eval['reason']}")
    print(f" Saldo Final Realizado:        ${prop_eval['final_balance']:,.2f} USD")
    print(f" Beneficio Neto:               ${prop_eval['net_profit']:+,.2f} USD")
    print(f" Sesiones Evaluadas:           {prop_eval['total_days_evaluated']} días")
    print("=" * 84)

    if args.export_csv and not trades.empty:
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(export_path, index=False)
        print(f"\n [OK] Registro de trades exportado a: {export_path.resolve()}")


if __name__ == "__main__":
    main()
