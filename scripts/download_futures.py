"""
scripts/download_futures.py
Script para descargar y actualizar datos históricos de futuros (ES y NQ)
utilizando tvdatafeed / TradingView CME.
"""

import sys
from pathlib import Path

# Agregar la raíz del proyecto al sys.path para importar src
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import argparse
from src.core.data_loader import DataLoader


def main():
    parser = argparse.ArgumentParser(
        description="Descarga datos de futuros de ES y NQ desde TradingView / CME."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["ES", "NQ"],
        help="Símbolos a descargar (ej: ES NQ MES MNQ). Por defecto: ES NQ"
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=["5m", "15m", "1h", "1d"],
        help="Intervalos a descargar (1m, 5m, 15m, 1h, 1d). Por defecto: 5m 15m 1h 1d"
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=5000,
        help="Número máximo de velas a descargar por intervalo (máx ~5000 en modo anónimo). Por defecto: 5000"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Forzar descarga y sobreescribir la caché local."
    )

    args = parser.parse_args()

    loader = DataLoader(cache_dir="data/raw")

    print("=" * 65)
    print(" INICIO DE DESCARGA DE FUTUROS (TradingView / CME)")
    print(f" Símbolos:    {args.symbols}")
    print(f" Intervalos:  {args.intervals}")
    print(f" Velas (max): {args.bars}")
    print("=" * 65)

    summary = []

    for symbol in args.symbols:
        print(f"\n Procesando {symbol}...")
        for interval in args.intervals:
            try:
                print(f"   -> Descargando {symbol} ({interval})...", end=" ", flush=True)
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
                print(f"OK ({len(df)} velas | {start_dt} a {end_dt})")
                summary.append({
                    "symbol": symbol,
                    "interval": interval,
                    "bars": len(df),
                    "start": start_dt,
                    "end": end_dt,
                    "latest_close": df["Close"].iloc[-1]
                })
            except Exception as e:
                print(f"ERROR: {e}")

    print("\n" + "=" * 65)
    print(" RESUMEN DE DATOS DISPONIBLES EN data/raw/")
    print("=" * 65)
    for item in summary:
        print(
            f" * {item['symbol']:<4} [{item['interval']:<3}]: "
            f"{item['bars']:>5} barras | "
            f"Del {item['start']} al {item['end']} | "
            f"Último Cierre: {item['latest_close']:,.2f}"
        )
    print("=" * 65)
    print("¡Listo! Los archivos se guardaron en 'data/raw/' en formatos .parquet y .csv.")


if __name__ == "__main__":
    main()
