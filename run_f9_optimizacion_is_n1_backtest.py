"""
run_f9_optimizacion_is_n1_backtest.py
Script ejecutable para correr el backtest de la Versión Alterna Optimizada:
Estrategia "F9-Optimizacion -IS- N1": Anomalías Temporales y Variance Risk Premium (VRP).

Mejoras Matemáticas Integradas:
- Condicionamiento estricto a la pendiente de la estructura de plazos de volatilidad (TSR).
- Bloqueo total de compras cuando la curva se invierte (TSR >= 1.00 o TSR_synth >= 1.05) para mitigar el riesgo de cola izquierda.
- Estimador sintético endógeno de VRP y TSR (Garman-Klass + RV multiescala) sin dependencia de feeds externos de pago.
- Dinamización de la caída previa RTH con Z_drop <= -1.25 sigma.
- Desacoplamiento modular con bandera --no-overnight para cuentas Prop Trading con regla RTH estricta.
- Take Profit nocturno de 0.80% (asimetría 1.6:1 frente a SL 0.50%).
- Salida determinista forzosa pre-macro en T-3m (60s antes de la ventana de embargo [T-2m, T+2m]).
- Segregación fraccionaria de riesgo diario (20% Noche / 80% RTH).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import pandas as pd
from typing import List, Tuple, Optional

from src.core.data_split import add_data_split_args, partition_from_args
from src.strategies.f9_optimizacion_is_n1 import (
    F9OptimizationISN1Strategy,
    F9OptimizationISN1Backtester,
)
from src.portfolio.prop_firm_rules import PROFILES


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest cuantitativo ejecutable para Estrategia F9-Optimizacion -IS- N1"
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
        "--no-overnight",
        action="store_true",
        default=False,
        help="Desactiva el submódulo nocturno (para firmas que prohíben operaciones fuera de RTH)"
    )
    parser.add_argument(
        "--no-synthetic-vrp",
        action="store_true",
        default=False,
        help="Desactiva el estimador sintético y busca columnas VIX/TSR externas"
    )
    parser.add_argument(
        "--max-tsr-long",
        type=float,
        default=1.00,
        help="TSR máximo para permitir compras (bloqueo en inversión de curva). Por defecto: 1.00"
    )
    parser.add_argument(
        "--z-drop-crit",
        type=float,
        default=-1.25,
        help="Umbral de caída intradía estandarizada para activar overnight. Por defecto: -1.25"
    )
    parser.add_argument(
        "--delta-tp-on",
        type=float,
        default=0.0080,
        help="Take profit porcentual nocturno (ej. 0.0080 = 0.80%%). Por defecto: 0.0080"
    )
    parser.add_argument(
        "--macro-events",
        type=str,
        default=None,
        help="Lista de timestamps de anuncios macro separados por coma (ej. '2026-10-15 08:30:00,2026-10-22 14:00:00')"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Ruta directa opcional al archivo de datos de mercado (.parquet o .csv)"
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        default=False,
        help="Prioriza datasets extendidos CME_EXT con 24h Globex para análisis overnight"
    )
    parser.add_argument(
        "--source-tz",
        type=str,
        default=None,
        help="Zona horaria origen si los datos son ingenuos (naive). Por defecto: None (auto-detectado)"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Ruta opcional para exportar el registro detallado de trades en CSV"
    )
    add_data_split_args(parser)
    return parser.parse_args()


def load_market_data(
    symbol: str,
    interval: str,
    custom_path: Optional[str] = None,
    extended: bool = False
) -> pd.DataFrame:
    """Carga los datos intradía disponibles en data/raw o desde custom_path."""
    if custom_path:
        p = Path(custom_path)
        if not p.exists():
            raise FileNotFoundError(f"Archivo de datos no encontrado en: {custom_path}")
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p, index_col=0, parse_dates=True)
        expected_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"Falta columna requerida '{col}' en el archivo de datos.")
        return df

    base_sym = symbol.upper()
    if base_sym.startswith("M"):
        underlying = base_sym[1:]
    else:
        underlying = base_sym

    data_dir = ROOT_DIR / "data" / "raw"
    ext_parquet = data_dir / f"CME_EXT_{underlying}_{interval}.parquet"
    ext_csv = data_dir / f"CME_EXT_{underlying}_{interval}.csv"
    p3y = data_dir / f"CME_MINI_{underlying}_{interval}_3y.parquet"
    c3y = data_dir / f"CME_MINI_{underlying}_{interval}_3y.csv"
    parquet_path = data_dir / f"CME_MINI_{underlying}_{interval}_bars5000.parquet"
    csv_path = data_dir / f"CME_MINI_{underlying}_{interval}_bars5000.csv"

    if extended and ext_parquet.exists():
        df = pd.read_parquet(ext_parquet)
    elif extended and ext_csv.exists():
        df = pd.read_csv(ext_csv, index_col=0, parse_dates=True)
    elif p3y.exists():
        df = pd.read_parquet(p3y)
    elif c3y.exists():
        df = pd.read_csv(c3y, index_col=0, parse_dates=True)
    elif ext_parquet.exists():
        df = pd.read_parquet(ext_parquet)
    elif ext_csv.exists():
        df = pd.read_csv(ext_csv, index_col=0, parse_dates=True)
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(
            f"No se encontraron datos históricos para {underlying} ({interval}) en {data_dir}."
        )

    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida '{col}' en el archivo de datos.")

    return df


def parse_macro_schedule(arg_macro: str, df_index: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, str]]:
    """Parsea o programa anuncios macro Tier-1 institucionales."""
    events: List[Tuple[pd.Timestamp, str]] = []
    if arg_macro:
        for item in arg_macro.split(","):
            item_clean = item.strip()
            if item_clean:
                events.append((pd.Timestamp(item_clean, tz="America/New_York"), "Tier-1 Scheduled"))
        return events

    unique_dates = sorted(list(set(df_index.date)))
    for d in unique_dates:
        if d.weekday() == 3:  # Jueves CPI / Claims
            ts_macro = pd.Timestamp(f"{d} 08:30:00", tz="America/New_York")
            events.append((ts_macro, "CPI / Initial Claims Tier-1"))
        elif d.weekday() == 2 and d.day in [14, 21, 28]:  # Miércoles FOMC
            ts_macro = pd.Timestamp(f"{d} 14:00:00", tz="America/New_York")
            events.append((ts_macro, "FOMC Rate Decision Tier-1"))

    return events


def main():
    args = parse_arguments()

    account_size = args.capital
    daily_loss_limit = args.dll
    max_trailing_loss = args.mtdd
    profit_target = args.target
    trailing_mode = args.trailing_mode

    if args.profile and args.profile in PROFILES:
        prof = PROFILES[args.profile]
        account_size = prof.account_size
        profit_target = prof.profit_target
        max_trailing_loss = prof.max_loss
        daily_loss_limit = prof.daily_loss_limit or daily_loss_limit
        trailing_mode = prof.trailing

    std_symbol = "NQ" if "NQ" in args.symbol else "ES"
    micro_symbol = "MNQ" if "NQ" in args.symbol else "MES"
    enable_on = not args.no_overnight
    use_synth = not args.no_synthetic_vrp

    print("=" * 95)
    print(" MOTOR DE BACKTEST CUANTITATIVO: ESTRATEGIA F9-OPTIMIZACION -IS- N1")
    print(" Modificaciones Matemáticas: Condicionamiento Estricto TSR | VRP Sintético | Veto Backwardation")
    print("=" * 95)
    print(f" Instrumento Principal:    {std_symbol} (Estándar) / {micro_symbol} (Micro)")
    print(f" Intervalo de Barras:      {args.interval}")
    print(f" Capital Inicial:          ${account_size:,.2f} USD")
    print(f" Daily Loss Limit (DLL):   ${daily_loss_limit:,.2f} USD (20% Noche / 80% RTH)")
    print(f" Max Trailing DD (MTDD):   ${max_trailing_loss:,.2f} USD ({trailing_mode.upper()})")
    print(f" Profit Target:            ${profit_target:,.2f} USD")
    print(f" Submódulo Overnight:      {'HABILITADO' if enable_on else 'DESACTIVADO (RTH-Only)'} (01:30 a 08:30 EST, TP {args.delta_tp_on*100:.2f}%, SL 0.50%)")
    print(f" Submódulo Pre-Macro:      T-15m a T-3m EST (Salida determinista T-3m, SL 8.0 pts)")
    print(f" VRP & TSR Sintético:      {'ENDÓGENO (Garman-Klass + RV 22d)' if use_synth else 'EXTERNO CBOE (VIX/VIX3M)'}")
    print(f" Límite TSR Compras:       TSR < {args.max_tsr_long:.2f} (Bloqueo Total ante Curva Invertida)")
    print(f" Umbral Caída Previa:      Z_drop <= {args.z_drop_crit:.2f} sigma")
    print(f" Embargo Normativo Prop:   [T-2m, T+2m] EST (Cero contratos absolutos)")
    print(f" Cierre CME Universal:     16:45:00 EST")
    print("=" * 95)

    # 1. Carga de datos
    print(f"\n [1/3] Cargando datos de mercado...")
    df = load_market_data(std_symbol, args.interval, custom_path=args.data, extended=args.extended)
    try:
        df, part_label = partition_from_args(df, args, f"F9_Optimizacion_IS_N1_{std_symbol}")
    except ValueError as e:
        print(f"  [AVISO] Partición canónica 3Y no aplicable ({e}). Usando totalidad de los datos disponibles.")
        part_label = f"CUSTOM_DATASET | {df.index.min()} -> {df.index.max()} | {len(df):,} barras"
    start_dt = df.index.min().strftime("%Y-%m-%d %H:%M")
    end_dt = df.index.max().strftime("%Y-%m-%d %H:%M")
    print(f"       -> {len(df):,} barras cargadas desde {start_dt} hasta {end_dt}.")

    # 2. Cronograma Macro
    macro_schedule = parse_macro_schedule(args.macro_events, df.index)
    print(f"       -> {len(macro_schedule)} eventos macro Tier-1 identificados en el período.")

    # 3. Inicializar Estrategia y Backtester
    print(f"\n [2/3] Inicializando estrategia F9-Optimizacion -IS- N1 y gestor de Prop Trading...")
    strategy = F9OptimizationISN1Strategy(params={
        "symbol": std_symbol,
        "std_symbol": std_symbol,
        "micro_symbol": micro_symbol,
        "enable_overnight": enable_on,
        "use_synthetic_vrp": use_synth,
        "max_tsr_long": args.max_tsr_long,
        "z_drop_crit": args.z_drop_crit,
        "delta_sl_on": 0.0050,
        "delta_tp_on": args.delta_tp_on,
        "delta_p_stop_macro": 8.0,
        "macro_schedule": macro_schedule,
        "source_tz": args.source_tz,
        "target_tz": "America/New_York",
    })

    backtester = F9OptimizationISN1Backtester(
        strategy=strategy,
        account_size=account_size,
        daily_loss_limit=daily_loss_limit,
        max_trailing_loss=max_trailing_loss,
        trailing_mode=trailing_mode,
        profit_target=profit_target,
    )

    # 4. Ejecución del Backtest
    print(f"\n [3/3] Ejecutando simulación barra a barra con microestructura institucional CME Globex...")
    results = backtester.run(df)

    trades = results["trades"]
    metrics = results["metrics"]
    prop_eval = results["prop_evaluation"]

    # 5. Registro de Operaciones
    print("\n" + "=" * 95)
    print(" REGISTRO DE TRADES EJECUTADOS (F9-OPTIMIZACION -IS- N1)")
    print("=" * 95)
    if trades.empty:
        print(" [AVISO] Ninguna sesión cumplió las condiciones de entrada de los sub-modelos optimizados.")
    else:
        fmt = "{:<11} {:<9} {:<5} {:<5} {:<4} {:>9} {:>9} {:<22} {:>10} {:>8} {:>10} {:>10}"
        print(fmt.format("Fecha", "Submodelo", "Señal", "Sym", "Qty", "Entrada", "Salida", "Razón de Salida", "P&L Bruto", "Fricción", "P&L Neto", "Saldo USD"))
        print("-" * 120)
        for _, t in trades.iterrows():
            date_str = str(t["date"])
            sig_str = "LONG" if t["direction"] == 1 else "SHORT"
            gross_str = f"${t['gross_pnl_usd']:+,.2f}"
            fric_str = f"-${t['friction_usd']:,.2f}"
            net_str = f"${t['net_pnl_usd']:+,.2f}"
            bal_str = f"${t['running_balance']:,.2f}"
            print(fmt.format(
                date_str,
                t["submodel"][:9],
                sig_str,
                t["symbol"],
                t["contracts"],
                f"{t['entry_price']:.2f}",
                f"{t['exit_price']:.2f}",
                str(t["exit_reason"])[:22],
                gross_str,
                fric_str,
                net_str,
                bal_str
            ))

    # 6. Métricas Cuantitativas
    print("\n" + "=" * 95)
    print(" MÉTRICAS CUANTITATIVAS DE RENDIMIENTO (F9-N1)")
    print("=" * 95)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f" {k:<30}: {v:>14,.2f}")
        else:
            print(f" {k:<30}: {v:>14}")

    # 7. Evaluación de Prop Trading
    print("\n" + "=" * 95)
    print(" AUDITORÍA DE EVALUACIÓN DE PROP TRADING")
    print("=" * 95)
    status_icon = "[PASS] APROBADO" if prop_eval["status"] == "PASS" else (
        "[FAIL] REPROBADO" if prop_eval["status"] == "FAIL" else "[ACTIVE] EN CURSO"
    )
    print(f" Estado Final:             {status_icon}")
    print(f" Causa / Detalle:          {prop_eval['reason']}")
    print(f" Saldo Final Realizado:    ${prop_eval['final_balance']:,.2f} USD")
    print(f" Beneficio Neto Total:     ${prop_eval['net_profit']:+,.2f} USD")
    print(f" Sesiones Evaluadas:       {prop_eval['total_days_evaluated']} días")
    print("=" * 95)

    if args.export_csv and not trades.empty:
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(export_path, index=False)
        print(f"\n [OK] Registro de trades exportado a: {export_path.resolve()}")


if __name__ == "__main__":
    main()
