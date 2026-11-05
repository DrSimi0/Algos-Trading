"""
scripts/download_extended_futures.py
Script para descargar, extender y consolidar datos históricos de futuros CME (ES, NQ, MES, MNQ)
utilizando Yahoo Finance como fuente directa y gratuita, con fusión acumulativa (stitching)
para preservar y expandir los datasets locales en 'data/raw/'.
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd
import yfinance as yf

# Agregar la raíz del proyecto al sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.futures_specs import get_spec, FUTURES_SPECS
from src.core.data_loader import DataLoader


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga y consolida datos históricos extendidos de futuros CME (ES, NQ, MES, MNQ)."
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
        default=["5m", "1d"],
        help="Intervalos a descargar (5m, 15m, 1h, 1d). Por defecto: 5m 1d"
    )
    parser.add_argument(
        "--stitch",
        action="store_true",
        default=True,
        help="Combinar con datos locales existentes en data/raw/ para extender el historial continuo."
    )
    parser.add_argument(
        "--no-stitch",
        dest="stitch",
        action="store_false",
        help="Desactivar la combinación con archivos locales existentes."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directorio de destino para los archivos Parquet y CSV. Por defecto: data/raw"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="CME_EXT_",
        help="Prefijo de los archivos guardados. Por defecto: CME_EXT_"
    )
    return parser.parse_args()


def download_symbol_interval(symbol: str, interval: str) -> pd.DataFrame:
    """Descarga el máximo periodo permitido por la API para el intervalo dado."""
    spec = get_spec(symbol)
    yf_sym = spec.get("yf", f"{symbol}=F")

    # Mapear periodos máximos permitidos por Yahoo Finance
    period_map = {
        "1m": "7d",
        "5m": "60d",
        "15m": "60d",
        "30m": "60d",
        "1h": "730d",
        "1d": "5y"
    }
    period = period_map.get(interval, "60d")

    df = yf.download(
        tickers=yf_sym,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        raise ValueError(f"No se obtuvieron datos para {yf_sym} en intervalo {interval}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    req_cols = ["Open", "High", "Low", "Close", "Volume"]
    available_cols = [c for c in req_cols if c in df.columns]
    df = df[available_cols].copy()
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    # Normalizar zona horaria a America/New_York
    if df.index.tz is not None:
        df = df.tz_convert("America/New_York")
    else:
        df = df.tz_localize("America/New_York")

    return df


def load_existing_local_data(output_dir: Path, symbol: str, interval: str) -> pd.DataFrame:
    """Busca y carga archivos existentes locales (CME_EXT_ o CME_MINI_) para combinar."""
    candidates = [
        output_dir / f"CME_EXT_{symbol}_{interval}.parquet",
        output_dir / f"CME_EXT_{symbol}_{interval}.csv",
        output_dir / f"CME_MINI_{symbol}_{interval}_bars5000.parquet",
        output_dir / f"CME_MINI_{symbol}_{interval}_bars5000.csv"
    ]
    for p in candidates:
        if p.exists():
            try:
                if p.suffix == ".parquet":
                    df_loc = pd.read_parquet(p)
                else:
                    df_loc = pd.read_csv(p, index_col=0, parse_dates=True)
                if not df_loc.empty:
                    # Asegurar timezone
                    if df_loc.index.tz is None:
                        df_loc = df_loc.tz_localize("America/New_York")
                    else:
                        df_loc = df_loc.tz_convert("America/New_York")
                    return df_loc
            except Exception:
                continue
    return pd.DataFrame()


def main():
    args = parse_arguments()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" DESCARGA Y EXTENSIÓN DE DATOS HISTÓRICOS DE FUTUROS CME")
    print(f" Símbolos:      {args.symbols}")
    print(f" Intervalos:    {args.intervals}")
    print(f" Fusión local:  {'Activada (Stitching)' if args.stitch else 'Desactivada'}")
    print(f" Destino:       {out_dir.resolve()}")
    print("=" * 75)

    summary = []

    for sym in args.symbols:
        upper_sym = sym.upper()
        print(f"\n[+] Procesando futuros {upper_sym}...")

        for interval in args.intervals:
            print(f"    -> Solicitando {upper_sym} ({interval})...", end=" ", flush=True)
            try:
                df_new = download_symbol_interval(upper_sym, interval)
                bars_downloaded = len(df_new)

                if args.stitch:
                    df_old = load_existing_local_data(out_dir, upper_sym, interval)
                    if not df_old.empty:
                        df_final = DataLoader.stitch_dataframes(df_primary=df_new, df_secondary=df_old)
                        bars_stitched = len(df_final) - bars_downloaded
                    else:
                        df_final = df_new
                        bars_stitched = 0
                else:
                    df_final = df_new
                    bars_stitched = 0

                # Validar consistencia física de las barras
                invalid_bars = (df_final["High"] < df_final["Low"]).sum()
                if invalid_bars > 0:
                    print(f"ALERTA: {invalid_bars} barras corruptas descartadas.")
                    df_final = df_final[df_final["High"] >= df_final["Low"]]

                # Guardar en parquet y csv
                base_name = f"{args.prefix}{upper_sym}_{interval}"
                parquet_path = out_dir / f"{base_name}.parquet"
                csv_path = out_dir / f"{base_name}.csv"

                df_final.index.name = "datetime"
                df_final.to_parquet(parquet_path)
                df_final.to_csv(csv_path)

                start_str = df_final.index.min().strftime("%Y-%m-%d %H:%M")
                end_str = df_final.index.max().strftime("%Y-%m-%d %H:%M")
                days_span = (df_final.index.max().date() - df_final.index.min().date()).days
                unique_dates = len(set(df_final.index.date))

                print(f"OK: {len(df_final):,} barras ({unique_dates} días hábiles | {days_span} días totales)")
                summary.append({
                    "symbol": upper_sym,
                    "interval": interval,
                    "bars": len(df_final),
                    "days": unique_dates,
                    "start": start_str,
                    "end": end_str,
                    "csv_file": str(csv_path.name)
                })

            except Exception as e:
                print(f"ERROR: {e}")

    print("\n" + "=" * 75)
    print(" RESUMEN DE DATASETS HISTÓRICOS GENERADOS")
    print("=" * 75)
    for s in summary:
        print(
            f" * {s['symbol']:<4} [{s['interval']:<3}]: {s['bars']:>7,} barras | "
            f"{s['days']:>3} días de mercado | Del {s['start']} al {s['end']}\n"
            f"   Archivo: {s['csv_file']}"
        )
    print("=" * 75)
    print("¡Listo! Los datasets extendidos están disponibles para correr el backtest con mayor significancia estadística.\n")


if __name__ == "__main__":
    main()
