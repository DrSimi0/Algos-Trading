"""
scripts/import_external_futures.py
Utilidad para importar y normalizar datos históricos de futuros CME (ES, NQ, MES, MNQ)
exportados desde plataformas profesionales de trading (NinjaTrader, Tradovate, TradingView,
Sierra Chart, QuantConnect) que cubran de 3 a 12+ meses.
"""

import sys
from pathlib import Path
import argparse
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.data_loader import DataLoader


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Importa y normaliza archivos CSV/TXT externos a la estructura estándar del proyecto."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Ruta al archivo CSV o TXT exportado por la plataforma."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Símbolo del activo (ej: ES, NQ, MES, MNQ)."
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        help="Intervalo temporal del archivo (ej: 1m, 5m, 15m, 1h, 1d). Por defecto: 5m"
    )
    parser.add_argument(
        "--source-tz",
        type=str,
        default="America/New_York",
        help="Zona horaria original de los datos (ej: America/Chicago, America/New_York, UTC, America/Mexico_City)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directorio destino. Por defecto: data/raw"
    )
    return parser.parse_args()


def detect_and_load_csv(file_path: Path) -> pd.DataFrame:
    """Detecta el separador y estructura de columnas del archivo."""
    # Probar separadores comunes: coma, punto y coma, tabulación
    for sep in [",", ";", "\t"]:
        try:
            sample = pd.read_csv(file_path, sep=sep, nrows=5)
            if len(sample.columns) >= 5:
                return pd.read_csv(file_path, sep=sep)
        except Exception:
            continue
    return pd.read_csv(file_path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mapea nombres heterogéneos de columnas a [datetime, Open, High, Low, Close, Volume]."""
    col_map = {}
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    # Detectar columna de fecha y hora
    dt_col = None
    if "datetime" in cols_lower:
        dt_col = cols_lower["datetime"]
    elif "date_time" in cols_lower:
        dt_col = cols_lower["date_time"]
    elif "timestamp" in cols_lower:
        dt_col = cols_lower["timestamp"]
    elif "time" in cols_lower and "date" not in cols_lower:
        dt_col = cols_lower["time"]

    # Caso en que Date y Time vienen separados (ej: NinjaTrader, Sierra Chart)
    if "date" in cols_lower and "time" in cols_lower:
        date_c = cols_lower["date"]
        time_c = cols_lower["time"]
        df["datetime"] = pd.to_datetime(df[date_c].astype(str) + " " + df[time_c].astype(str))
        dt_col = "datetime"
    elif dt_col is not None:
        df["datetime"] = pd.to_datetime(df[dt_col])
    else:
        # Si la primera columna contiene fechas válidas
        try:
            df["datetime"] = pd.to_datetime(df.iloc[:, 0])
            dt_col = "datetime"
        except Exception:
            raise ValueError(f"No se pudo identificar la columna de fecha/hora en: {list(df.columns)}")

    # Mapear OHLCV
    for key, targets in [
        ("Open", ["open", "apertura", "o"]),
        ("High", ["high", "maximo", "máximo", "h"]),
        ("Low", ["low", "minimo", "mínimo", "l"]),
        ("Close", ["close", "cierre", "c", "last"]),
        ("Volume", ["volume", "volumen", "vol", "v", "size"])
    ]:
        found = None
        for t in targets:
            if t in cols_lower:
                found = cols_lower[t]
                break
        if found:
            col_map[found] = key
        else:
            if key == "Volume":
                df["Volume"] = 1.0  # Asignar volumen unitario si falta
                col_map["Volume"] = "Volume"
            else:
                raise ValueError(f"No se encontró columna requerida para '{key}' en {list(df.columns)}")

    df = df.rename(columns=col_map)
    df.set_index("datetime", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df


def main():
    args = parse_arguments()
    src_file = Path(args.file)
    if not src_file.exists():
        print(f"Error: El archivo especificado no existe: {src_file.resolve()}")
        sys.exit(1)

    print("=" * 70)
    print(" IMPORTADOR DE DATOS HISTÓRICOS EXTERNOS DE FUTUROS")
    print(f" Archivo origen:  {src_file.name}")
    print(f" Símbolo:         {args.symbol.upper()}")
    print(f" Intervalo:       {args.interval}")
    print(f" Zona horaria:    {args.source_tz}")
    print("=" * 70)

    try:
        raw_df = detect_and_load_csv(src_file)
        df = standardize_columns(raw_df)

        # Normalizar zona horaria a America/New_York
        if df.index.tz is None:
            df = df.tz_localize(args.source_tz).tz_convert("America/New_York")
        else:
            df = df.tz_convert("America/New_York")

        # Filtrar posibles barras corruptas
        corrupt = (df["High"] < df["Low"]).sum()
        if corrupt > 0:
            print(f"  [!] {corrupt} barras con High < Low fueron descartadas por integridad.")
            df = df[df["High"] >= df["Low"]]

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        target_base = f"CME_EXT_{args.symbol.upper()}_{args.interval}"
        out_parquet = out_dir / f"{target_base}.parquet"
        out_csv = out_dir / f"{target_base}.csv"

        df.to_parquet(out_parquet)
        df.to_csv(out_csv)

        unique_days = len(set(df.index.date))
        span_days = (df.index.max().date() - df.index.min().date()).days

        print("\n[+] Importación completada con éxito:")
        print(f"    - Total de barras:     {len(df):,}")
        print(f"    - Sesiones de mercado: {unique_days}")
        print(f"    - Rango temporal:      {df.index.min()} a {df.index.max()} ({span_days} días)")
        print(f"    - Archivo CSV:         {out_csv.resolve()}")
        print(f"    - Archivo Parquet:     {out_parquet.resolve()}")
        print("=" * 70)
        print("Ahora puedes ejecutar el backtest pasando: --data " + str(out_csv) + "\n")

    except Exception as e:
        print(f"\nError al procesar el archivo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
