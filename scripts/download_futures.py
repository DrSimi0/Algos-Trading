"""
scripts/download_futures.py
Script para descargar y actualizar datos históricos de futuros (ES, NQ, MES, MNQ)
utilizando tvdatafeed / TradingView CME hasta el cierre más reciente.
"""

import sys
from pathlib import Path

# Configurar salida UTF-8 en consola de Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Agregar la raíz del proyecto al sys.path para importar src
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import pandas as pd
from src.core.data_loader import DataLoader


def main():
    parser = argparse.ArgumentParser(
        description="Descarga datos de futuros de ES y NQ desde TradingView / CME."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["ES", "NQ", "MES", "MNQ"],
        help="Símbolos a descargar (ej: ES NQ MES MNQ). Por defecto: ES NQ MES MNQ"
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=["1m", "5m", "15m", "1h", "1d"],
        help="Intervalos a descargar (1m, 5m, 15m, 1h, 1d). Por defecto: 1m 5m 15m 1h 1d"
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=5000,
        help="Número máximo de velas a descargar por intervalo (máx ~5000 en modo anónimo). Por defecto: 5000"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Año de inicio para filtrar y exportar datasets específicos (ej: 2021)."
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Año de fin para filtrar y exportar datasets específicos (ej: 2023 o 2026)."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=True,
        help="Forzar descarga y sobreescribir la caché local con datos frescos."
    )

    args = parser.parse_args()

    loader = DataLoader(cache_dir="data/raw")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" INICIO DE DESCARGA DE FUTUROS HASTA CIERRE ACTUAL (TradingView / CME)")
    print(f" Símbolos:    {args.symbols}")
    print(f" Intervalos:  {args.intervals}")
    print(f" Velas (max): {args.bars}")
    print(f" Directorio:  {raw_dir.resolve()}")
    print("=" * 75)

    summary = []

    for symbol in args.symbols:
        print(f"\n[+] Procesando contrato continuo: {symbol}...")
        for interval in args.intervals:
            try:
                print(f"    -> Descargando {symbol:<4} ({interval:<3})...", end=" ", flush=True)
                df = loader.fetch_tradingview(
                    symbol=symbol,
                    exchange="CME_MINI",
                    interval=interval,
                    n_bars=args.bars,
                    fut_contract=1,
                    use_cache=not args.no_cache
                )
                start_dt = df.index.min().strftime("%Y-%m-%d %H:%M")
                end_dt = df.index.max().strftime("%Y-%m-%d %H:%M")
                bars_count = len(df)

                # Generar archivo específico para año 2026 por compatibilidad
                df_2026 = df[df.index >= "2026-01-01"].copy()
                bars_2026 = len(df_2026)
                if bars_2026 > 0:
                    try:
                        df_2026.to_parquet(raw_dir / f"CME_MINI_{symbol}_2026_{interval}.parquet")
                        df_2026.to_csv(raw_dir / f"CME_MINI_{symbol}_2026_{interval}.csv")
                    except Exception:
                        pass

                # Generar rango personalizado de años si se especificó
                bars_custom = 0
                custom_label = ""
                if args.start_year is not None or args.end_year is not None:
                    sy = args.start_year or 2021
                    ey = args.end_year or sy
                    custom_label = f"{sy}_{ey}" if sy != ey else f"{sy}"
                    df_custom = df[(df.index >= f"{sy}-01-01") & (df.index <= f"{ey}-12-31 23:59:59")].copy()
                    bars_custom = len(df_custom)
                    if bars_custom > 0:
                        try:
                            df_custom.to_parquet(raw_dir / f"CME_MINI_{symbol}_{custom_label}_{interval}.parquet")
                            df_custom.to_csv(raw_dir / f"CME_MINI_{symbol}_{custom_label}_{interval}.csv")
                        except Exception:
                            pass

                msg_extra = f" | 2026: {bars_2026} velas"
                if custom_label:
                    msg_extra += f" | {custom_label}: {bars_custom} velas"
                print(f"OK ({bars_count} velas | {start_dt} a {end_dt}{msg_extra})")
                summary.append({
                    "symbol": symbol,
                    "interval": interval,
                    "bars": bars_count,
                    "bars_2026": bars_2026,
                    "custom_range": custom_label,
                    "bars_custom": bars_custom,
                    "start": start_dt,
                    "end": end_dt,
                    "latest_close": df["Close"].iloc[-1]
                })
            except Exception as e:
                print(f"ERROR: {e}")

    print("\n" + "=" * 80)
    print(" RESUMEN DE DATOS HISTÓRICOS Y 2026 ACTUALIZADOS EN data/raw/")
    print("=" * 80)
    print(f" {'Símbolo':<7} | {'Int':<4} | {'Total':<6} | {'Velas 2026':<10} | {'Inicio':<16} | {'Fin (Cierre)':<16} | {'Último Cierre':<12}")
    print("-" * 80)
    for item in summary:
        print(
            f" {item['symbol']:<7} | {item['interval']:<4} | {item['bars']:>5}  | "
            f"{item['bars_2026']:>8}   | {item['start']:<16} | {item['end']:<16} | "
            f"{item['latest_close']:>10,.2f}"
        )
    print("=" * 80)
    print("¡Descarga completada con éxito! Archivos actualizados en .parquet y .csv.")


if __name__ == "__main__":
    main()
