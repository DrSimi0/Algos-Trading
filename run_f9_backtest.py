"""
run_f9_backtest.py
Script ejecutable para correr el backtest de la Estrategia F9:
Anomalías Temporales Multi-Sesión y Variance Risk Premium (VRP).

Fundamentos Académicos:
- Bollerslev, T., Tauchen, G., & Zhou, H. (2009). Expected Stock Returns and Variance Risk Premia. RFS.
- Boyarchenko, N., Larsen, J. C., & Whelan, P. (2023). The Overnight Drift. Federal Reserve Bank of NY.
- Kurov, A., Sancetta, A., Strasser, G., & Wolfe, M. H. (2019). Pre-Announcement Drift. JFQA.
- Simon, D. P., & Campasano, J. (2014). The VIX Futures Basis: Evidence and Trading Strategies. JOD.

Reglas Cuantitativas y de Prop Trading:
- Sub-modelo 1 (Overnight Drift): Entrada 01:30 EST tras caída RTH previa >= 0.75%, salida determinista 08:30 EST (SL 0.50%).
- Sub-modelo 2 (Pre-Macro Drift): Entrada T-15m con Z_OFI >= +2.0 (o <= -2.0) y precio vs VWAP, salida forzosa T-3m (SL 10 pts).
- Compuerta VRP: Z_VRP >= 1.50 y TSR <= 0.95 (alcista), Z_VRP <= -1.20 o TSR >= 1.00 (veto total de compras).
- Embargo Prop Firm [T-2m, T+2m]: Exposición cero absoluta (60s de margen).
- Cierre CME universal forzoso a las 16:45:00 EST.
- Control Prop Trading: DLL diario con segregación 25% noche / 75% RTH, 5% max de colchón MTDD, conmutación Estándar/Micro.
"""

import sys
from pathlib import Path

# Asegurar importación de módulos internos del proyecto
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import pandas as pd
from typing import List, Tuple

from src.core.data_split import add_data_split_args, partition_from_args
from src.strategies.anomalias_temporales_vrp import (
    TemporalAnomaliesVRPStrategy,
    TemporalAnomaliesVRPBacktester,
)
from src.portfolio.prop_firm_rules import PROFILES


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest cuantitativo ejecutable para Estrategia F9: Anomalías Temporales y VRP"
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
        "--theta-drop",
        type=float,
        default=0.0075,
        help="Umbral de caída intradía previa para habilitar overnight (ej. 0.0075 = 0.75%%). Por defecto: 0.0075"
    )
    parser.add_argument(
        "--macro-events",
        type=str,
        default=None,
        help="Lista de timestamps de anuncios macro separados por coma (ej. '2026-10-15 08:30:00,2026-10-22 14:00:00')"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Ruta opcional para exportar el registro detallado de trades en CSV"
    )
    add_data_split_args(parser)
    return parser.parse_args()


def load_market_data(symbol: str, interval: str) -> pd.DataFrame:
    """Carga los datos intradía disponibles en data/raw, priorizando dataset de 3 años."""
    base_sym = symbol.upper()
    if base_sym.startswith("M"):
        # MES -> ES para datos de subyacente
        underlying = base_sym[1:]
    else:
        underlying = base_sym

    data_dir = ROOT_DIR / "data" / "raw"
    p3y = data_dir / f"CME_MINI_{underlying}_{interval}_3y.parquet"
    c3y = data_dir / f"CME_MINI_{underlying}_{interval}_3y.csv"
    parquet_path = data_dir / f"CME_MINI_{underlying}_{interval}_bars5000.parquet"
    csv_path = data_dir / f"CME_MINI_{underlying}_{interval}_bars5000.csv"

    if p3y.exists():
        df = pd.read_parquet(p3y)
    elif c3y.exists():
        df = pd.read_csv(c3y, index_col=0, parse_dates=True)
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(
            f"No se encontraron datos históricos para {underlying} ({interval}) en {data_dir}.\n"
            f"Verifique la existencia de CME_MINI_{underlying}_{interval}_3y.parquet o bars5000"
        )

    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida '{col}' en el archivo de datos.")

    return df


def parse_macro_schedule(arg_macro: str, df_index: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, str]]:
    """Parsea o genera un cronograma de eventos macro Tier-1 en la muestra."""
    events: List[Tuple[pd.Timestamp, str]] = []
    if arg_macro:
        for item in arg_macro.split(","):
            item_clean = item.strip()
            if item_clean:
                events.append((pd.Timestamp(item_clean, tz="America/New_York"), "Tier-1 Scheduled"))
        return events

    # Si no se especifican eventos, se programa CPI matutino a las 08:30 EST
    # en los días jueves de la muestra como referencia típica institucional
    unique_dates = sorted(list(set(df_index.date)))
    for d in unique_dates:
        # Programar CPI/NFP en jueves a las 08:30:00 EST
        if d.weekday() == 3:  # Jueves
            ts_macro = pd.Timestamp(f"{d} 08:30:00", tz="America/New_York")
            events.append((ts_macro, "CPI / Initial Claims Tier-1"))
        # Programar FOMC en miércoles a las 14:00:00 EST
        elif d.weekday() == 2 and d.day in [14, 21, 28]:
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

    # Sobreescribir con perfil de prop firm si se especificó
    if args.profile and args.profile in PROFILES:
        prof = PROFILES[args.profile]
        account_size = prof.account_size
        profit_target = prof.profit_target
        max_trailing_loss = prof.max_loss
        daily_loss_limit = prof.daily_loss_limit or daily_loss_limit
        trailing_mode = prof.trailing

    std_symbol = "NQ" if "NQ" in args.symbol else "ES"
    micro_symbol = "MNQ" if "NQ" in args.symbol else "MES"

    print("=" * 90)
    print(" MOTOR DE BACKTEST CUANTITATIVO: ESTRATEGIA F9 - ANOMALÍAS TEMPORALES Y VRP")
    print(" Fundamento: Bollerslev et al. (2009) | Boyarchenko et al. (2023) | Kurov et al. (2019)")
    print("=" * 90)
    print(f" Instrumento Principal:    {std_symbol} (Estándar) / {micro_symbol} (Micro)")
    print(f" Intervalo de Barras:      {args.interval}")
    print(f" Capital Inicial:          ${account_size:,.2f} USD")
    print(f" Daily Loss Limit (DLL):   ${daily_loss_limit:,.2f} USD (25% Noche / 75% RTH)")
    print(f" Max Trailing DD (MTDD):   ${max_trailing_loss:,.2f} USD ({trailing_mode.upper()})")
    print(f" Profit Target:            ${profit_target:,.2f} USD")
    print(f" Ventana Overnight:        01:30:00 a 08:30:00 EST (Salida determinista 08:30, SL 0.50%)")
    print(f" Ventana Pre-Macro:        T-15m a T-3m EST (Salida determinista T-3m, SL 10 pts)")
    print(f" Embargo Normativo Prop:   [T-2m, T+2m] EST (Cero contratos absolutos)")
    print(f" Cierre CME Universal:     16:45:00 EST")
    print("=" * 90)

    # 1. Cargar datos
    print(f"\n [1/3] Cargando datos de mercado...")
    df = load_market_data(std_symbol, args.interval)
    df, part_label = partition_from_args(df, args, f"F9_Temporal_Anomalies_VRP_{std_symbol}")
    start_dt = df.index.min().strftime("%Y-%m-%d %H:%M")
    end_dt = df.index.max().strftime("%Y-%m-%d %H:%M")
    print(f"       -> {len(df):,} barras cargadas desde {start_dt} hasta {end_dt}.")

    # 2. Configurar cronograma macro
    macro_schedule = parse_macro_schedule(args.macro_events, df.index)
    print(f"       -> {len(macro_schedule)} eventos macro Tier-1 identificados en el período.")

    # 3. Inicializar estrategia y simulador
    print(f"\n [2/3] Inicializando estrategia F9 y simulador de Prop Trading...")
    strategy = TemporalAnomaliesVRPStrategy(params={
        "symbol": std_symbol,
        "std_symbol": std_symbol,
        "micro_symbol": micro_symbol,
        "theta_drop": args.theta_drop,
        "delta_sl_on": 0.0050,
        "delta_p_stop_macro": 10.0,
        "macro_schedule": macro_schedule,
        "source_tz": "America/Mexico_City",  # Convierte datos locales UTC-6 a America/New_York
        "target_tz": "America/New_York",
    })

    backtester = TemporalAnomaliesVRPBacktester(
        strategy=strategy,
        account_size=account_size,
        daily_loss_limit=daily_loss_limit,
        max_trailing_loss=max_trailing_loss,
        trailing_mode=trailing_mode,
        profit_target=profit_target,
    )

    # 4. Ejecutar simulación
    print(f"\n [3/3] Ejecutando simulación barra a barra con microestructura CME Globex...")
    results = backtester.run(df)

    trades = results["trades"]
    metrics = results["metrics"]
    prop_eval = results["prop_evaluation"]

    # 5. Imprimir registro detallado de operaciones
    print("\n" + "=" * 90)
    print(" REGISTRO DE TRADES EJECUTADOS")
    print("=" * 90)
    if trades.empty:
        print(" [AVISO] Ninguna sesión cumplió las condiciones de entrada de los sub-modelos.")
    else:
        fmt = "{:<11} {:<9} {:<5} {:<5} {:<4} {:>9} {:>9} {:<22} {:>10} {:>8} {:>10} {:>10}"
        print(fmt.format("Fecha", "Submodelo", "Señal", "Sym", "Qty", "Entrada", "Salida", "Razón de Salida", "P&L Bruto", "Fricción", "P&L Neto", "Saldo USD"))
        print("-" * 118)
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

    # 6. Métricas cuantitativas
    print("\n" + "=" * 90)
    print(" MÉTRICAS CUANTITATIVAS DE RENDIMIENTO")
    print("=" * 90)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f" {k:<30}: {v:>14,.2f}")
        else:
            print(f" {k:<30}: {v:>14}")

    # 7. Evaluación de Prop Trading
    print("\n" + "=" * 90)
    print(" AUDITORÍA DE EVALUACIÓN DE PROP TRADING")
    print("=" * 90)
    status_icon = "[PASS] APROBADO" if prop_eval["status"] == "PASS" else (
        "[FAIL] REPROBADO" if prop_eval["status"] == "FAIL" else "[ACTIVE] EN CURSO"
    )
    print(f" Estado Final:             {status_icon}")
    print(f" Causa / Detalle:          {prop_eval['reason']}")
    print(f" Saldo Final Realizado:    ${prop_eval['final_balance']:,.2f} USD")
    print(f" Beneficio Neto Total:     ${prop_eval['net_profit']:+,.2f} USD")
    print(f" Sesiones Evaluadas:       {prop_eval['total_days_evaluated']} días")
    print("=" * 90)

    # Exportación opcional a CSV
    if args.export_csv and not trades.empty:
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(export_path, index=False)
        print(f"\n [OK] Registro de trades exportado a: {export_path.resolve()}")


if __name__ == "__main__":
    main()
