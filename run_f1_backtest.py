"""
run_f1_backtest.py
Script ejecutable para correr el backtest de la Estrategia F1:
Market Intraday Momentum & MOC Seasonality.

Fundamento Académico:
- Gao, L., Han, Y., Li, S. Z., & Zhou, G. (2018). Market Intraday Momentum. Journal of Financial Economics.
- Bogousslavsky, V. (2021). The Cross-Section of Intraday and Overnight Returns. JFE.
"""

import sys
from pathlib import Path

# Asegurar importación de módulos internos del proyecto
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import pandas as pd

from src.strategies.momentum.intraday_momentum_moc import (
    MarketIntradayMomentumMOC,
    IntradayMomentumBacktester,
)
from src.portfolio.prop_firm_rules import PROFILES
from src.core.data_split import add_data_split_args, partition_from_args


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest cuantitativo ejecutable para Estrategia F1: Market Intraday Momentum & MOC"
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
    """Carga los datos intradía disponibles en data/raw (prioriza dataset de 3 años)."""
    if custom_path:
        p = Path(custom_path)
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p, index_col=0, parse_dates=True)

    base_sym = symbol.upper()
    if base_sym.startswith("M"):
        # MES -> ES para datos de subyacente
        underlying = base_sym[1:]
    else:
        underlying = base_sym

    data_dir = ROOT_DIR / "data" / "raw"
    candidates = [
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
        f"Ejecute previamente: python scripts/generate_synthetic_futures_3y.py"
    )

    # Validar formato de columnas
    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida '{col}' en el archivo de datos.")

    return df


def main():
    args = parse_arguments()

    account_size = args.capital
    daily_loss_limit = args.dll
    max_trailing_loss = args.mtdd
    profit_target = args.target
    trailing_mode = args.trailing_mode

    lock_floor_at_start = True

    # Sobreescribir con perfil oficial si se especifica
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

    dll_str = f"${daily_loss_limit:,.2f} USD" if daily_loss_limit is not None else "N/A (Sin límite diario)"

    print("=" * 80)
    print(" MOTOR DE BACKTEST CUANTITATIVO: ESTRATEGIA F1 - MARKET INTRADAY MOMENTUM & MOC")
    print(" Fundamento: Gao, Han, Li, & Zhou (2018, JFE) | Bogousslavsky (2021, JFE)")
    print("=" * 80)
    print(f" Instrumento:              {std_symbol} (Estándar) / {micro_symbol} (Micro)")
    print(f" Intervalo:                {args.interval}")
    print(f" Capital Inicial:          ${account_size:,.2f} USD")
    print(f" Daily Loss Limit (DLL):   {dll_str}")
    print(f" Max Trailing DD (MTDD):   ${max_trailing_loss:,.2f} USD ({trailing_mode.upper()})")
    print(f" Profit Target:            ${profit_target:,.2f} USD")
    print(f" Ventana de Observación:   09:30:00 a 10:00:00 America/New_York (Intervalo [09:30, 10:00) )")
    print(f" Ventana de Ejecución MOC: 15:30:00 a 15:58:00 America/New_York (Salida 15:58:00)")
    print("=" * 80)

    # 1. Cargar datos
    print(f"\n [1/3] Cargando datos de mercado...")
    df = load_market_data(std_symbol, args.interval, custom_path=args.data)
    df, part_label = partition_from_args(df, args, "F1_Momentum_MOC")
    start_dt = df.index.min().strftime("%Y-%m-%d %H:%M")
    end_dt = df.index.max().strftime("%Y-%m-%d %H:%M")
    print(f"       -> {len(df):,} barras cargadas desde {start_dt} hasta {end_dt}.")

    # 2. Configurar estrategia y simulador
    print(f"\n [2/3] Inicializando estrategia F1 y simulador de Prop Trading...")
    strategy = MarketIntradayMomentumMOC(params={
        "symbol": std_symbol,
        "std_symbol": std_symbol,
        "micro_symbol": micro_symbol,
        "source_tz": "America/Mexico_City",  # Convierte datos locales UTC-6 a America/New_York
        "target_tz": "America/New_York",
    })

    backtester = IntradayMomentumBacktester(
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

    # 4. Imprimir registro detallado de operaciones
    print("\n" + "=" * 80)
    print(" REGISTRO DE TRADES EJECUTADOS")
    print("=" * 80)
    if trades.empty:
        print(" [AVISO] Ninguna sesión cumplió conjuntamente el umbral de retorno matutino y volumen relativo.")
    else:
        fmt = "{:<11} {:<6} {:<5} {:<4} {:>9} {:>9} {:<5} {:>10} {:>8} {:>10} {:>10}"
        print(fmt.format("Fecha", "Señal", "Sym", "Qty", "Entrada", "Salida", "Razón", "P&L Bruto", "Fricción", "P&L Neto", "Saldo USD"))
        print("-" * 88)
        for _, t in trades.iterrows():
            date_str = str(t["date"])
            gross_str = f"${t['gross_pnl']:+,.2f}"
            fric_str = f"-${t['friction']:,.2f}"
            net_str = f"${t['net_pnl']:+,.2f}"
            bal_str = f"${t['balance']:,.2f}"
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
                bal_str
            ))

    # 5. Métricas cuantitativas
    print("\n" + "=" * 80)
    print(" MÉTRICAS CUANTITATIVAS DE RENDIMIENTO")
    print("=" * 80)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f" {k:<30}: {v:>12,.2f}")
        else:
            print(f" {k:<30}: {v:>12}")

    # 6. Evaluación de Prop Trading
    print("\n" + "=" * 80)
    print(" AUDITORÍA DE EVALUACIÓN DE PROP TRADING")
    print("=" * 80)
    status_icon = "[PASS] APROBADO" if prop_eval["status"] == "PASS" else (
        "[FAIL] REPROBADO" if prop_eval["status"] == "FAIL" else "[ACTIVE] EN CURSO"
    )
    print(f" Estado Final:             {status_icon}")
    print(f" Causa / Detalle:          {prop_eval['reason']}")
    print(f" Saldo Final Realizado:    ${prop_eval['final_balance']:,.2f} USD")
    print(f" Beneficio Neto:           ${prop_eval['net_profit']:+,.2f} USD")
    print(f" Sesiones Evaluadas:       {prop_eval['total_days_evaluated']} días")
    print("=" * 80)

    # Exportación opcional a CSV
    if args.export_csv and not trades.empty:
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(export_path, index=False)
        print(f"\n [OK] Registro de trades exportado a: {export_path.resolve()}")


if __name__ == "__main__":
    main()
