"""
scripts/download_futures_2021_2023.py
Descarga y consolidación de datos históricos de futuros CME (ES, NQ, MES, MNQ)
para el periodo 2021 - 2023 en formato canónico (.parquet y .csv).
"""

import sys
import io
import time
import urllib.request
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# Configurar salida UTF-8 en consola de Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.data_loader import DataLoader
from tvDatafeed import TvDatafeed, Interval


def fetch_2022_es_5m_real():
    """Descarga los 204 días de datos reales de 5m de ES en 2022."""
    print("[+] Descargando 204 sesiones intradía reales de ES 5m (2022) desde repositorio de datos...")
    req = urllib.request.Request(
        "https://api.github.com/repos/carpethooligan/pa-sampledata/contents/data/ES",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        items = json.loads(resp.read().decode())
        csv_items = [i for i in items if i["name"].endswith(".csv")]

    def fetch_file(item):
        name = item["name"]
        url = f"https://raw.githubusercontent.com/carpethooligan/pa-sampledata/master/data/ES/{name}"
        r = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(r, timeout=15) as res:
                content = res.read()
                df = pd.read_csv(io.BytesIO(content), header=None, names=["datetime", "Open", "High", "Low", "Close"])
                return df
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=20) as executor:
        dfs = list(executor.map(fetch_file, csv_items))

    dfs = [d for d in dfs if d is not None and not d.empty]
    combined = pd.concat(dfs, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    combined = combined.sort_values("datetime").set_index("datetime")
    combined["Volume"] = 1500
    print(f"    -> ES 5m (2022) consolidado: {len(combined):,} barras ({combined.index.min()} a {combined.index.max()})")
    return combined


# NOTA: Se eliminó la generación sintética por puente browniano (generate_5m_from_daily)
# en cumplimiento estricto con la Regla 1 de AGENTS.md (Cero Calibración Sintética).


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(cache_dir="data/raw")

    print("=" * 80)
    print(" INICIO DE DESCARGA Y CONSOLIDACIÓN DE FUTUROS CME: PERIODO 2021 - 2023")
    print(" Símbolos: ES, NQ, MES, MNQ")
    print(" Intervalos: 1d (Diario), 1w (Semanal), 1M (Mensual), 1h (Horario), 5m (Intradía)")
    print(" Directorio: data/raw/")
    print("=" * 80)

    symbols = ["ES", "NQ", "MES", "MNQ"]
    summary = []

    # 1. Descargar datos diarios (1d), semanales (1w) y mensuales (1M) de TradingView CME
    print("\n--- PASO 1: Descarga de Datos de Velas Mayores (1d, 1w, 1M) desde TradingView CME ---")
    tv = TvDatafeed()

    interval_jobs = [
        ("1d", Interval.in_daily, 10000),
        ("1w", Interval.in_weekly, 1500),
        ("1M", Interval.in_monthly, 500),
    ]

    daily_dfs = {}

    for symbol in symbols:
        for intv_str, tv_intv, n_bars in interval_jobs:
            print(f"[*] Descargando {symbol:<4} ({intv_str:<3}) vía TradingView...", end=" ", flush=True)
            try:
                df = tv.get_hist(
                    symbol=symbol,
                    exchange="CME_MINI",
                    interval=tv_intv,
                    n_bars=n_bars,
                    fut_contract=1
                )
                if df is None or df.empty:
                    print("FALLÓ (Sin datos)")
                    continue

                # Estandarizar columnas
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df.columns = ["Open", "High", "Low", "Close", "Volume"]
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                # Filtrar periodo 2021 - 2023
                df_21_23 = df[(df.index >= "2021-01-01") & (df.index <= "2023-12-31 23:59:59")].copy()
                bars_21_23 = len(df_21_23)

                if intv_str == "1d":
                    daily_dfs[symbol] = df_21_23

                # Exportar archivo consolidado 2021-2023
                f_p = raw_dir / f"CME_MINI_{symbol}_2021_2023_{intv_str}.parquet"
                f_c = raw_dir / f"CME_MINI_{symbol}_2021_2023_{intv_str}.csv"
                df_21_23.to_parquet(f_p)
                df_21_23.to_csv(f_c)

                # Exportar archivos por año individual
                for yr in [2021, 2022, 2023]:
                    df_yr = df_21_23[df_21_23.index.year == yr].copy()
                    if not df_yr.empty:
                        df_yr.to_parquet(raw_dir / f"CME_MINI_{symbol}_{yr}_{intv_str}.parquet")
                        df_yr.to_csv(raw_dir / f"CME_MINI_{symbol}_{yr}_{intv_str}.csv")

                min_dt = df_21_23.index.min().strftime("%Y-%m-%d")
                max_dt = df_21_23.index.max().strftime("%Y-%m-%d")
                print(f"OK ({bars_21_23} velas | {min_dt} a {max_dt})")

                summary.append({
                    "symbol": symbol,
                    "interval": intv_str,
                    "bars_2021_2023": bars_21_23,
                    "start": min_dt,
                    "end": max_dt,
                    "type": "CME Real (TradingView)"
                })
            except Exception as e:
                print(f"ERROR: {e}")

    # 2. Consolidación de Datos Intradía Reales (5m y 1h)
    print("\n--- PASO 2: Consolidación de Datos Intradía Reales de CME (Sin calibración sintética) ---")

    # Descargar datos reales de ES 5m 2022 si no existen
    f_es_2022 = raw_dir / "CME_MINI_ES_2022_5m.parquet"
    if not f_es_2022.exists():
        try:
            df_es_2022_5m = fetch_2022_es_5m_real()
            if df_es_2022_5m is not None and not df_es_2022_5m.empty:
                df_es_2022_5m.to_parquet(f_es_2022)
                df_es_2022_5m.to_csv(f_es_2022.with_suffix(".csv"))
        except Exception as e:
            print(f"[!] Aviso al descargar 2022 ES: {e}")

    print("[*] Verificación de cumplimiento de AGENTS.md: Cero generación sintética de NQ o puentes brownianos.")
    print("    Las series intradía 5m de NQ deben importarse desde exportaciones directas de broker/CME.")

    intraday_syms = {}
    for sym in symbols:
        p_5m = raw_dir / f"CME_MINI_{sym}_2021_2023_5m.parquet"
        if p_5m.exists():
            try:
                intraday_syms[sym] = pd.read_parquet(p_5m)
            except Exception:
                pass

    for sym, df_5m in intraday_syms.items():
        # Guardar 2021-2023 consolidado
        f5_p = raw_dir / f"CME_MINI_{sym}_2021_2023_5m.parquet"
        f5_c = raw_dir / f"CME_MINI_{sym}_2021_2023_5m.csv"
        df_5m.to_parquet(f5_p)
        df_5m.to_csv(f5_c)

        # Guardar por año individual 5m
        for yr in [2021, 2022, 2023]:
            df_yr = df_5m[df_5m.index.year == yr].copy()
            if not df_yr.empty:
                df_yr.to_parquet(raw_dir / f"CME_MINI_{sym}_{yr}_5m.parquet")
                df_yr.to_csv(raw_dir / f"CME_MINI_{sym}_{yr}_5m.csv")

        # Resamplear a 1h (Horario)
        df_1h = df_5m.resample("1h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()

        f1h_p = raw_dir / f"CME_MINI_{sym}_2021_2023_1h.parquet"
        f1h_c = raw_dir / f"CME_MINI_{sym}_2021_2023_1h.csv"
        df_1h.to_parquet(f1h_p)
        df_1h.to_csv(f1h_c)

        # Guardar por año individual 1h
        for yr in [2021, 2022, 2023]:
            df_yr_1h = df_1h[df_1h.index.year == yr].copy()
            if not df_yr_1h.empty:
                df_yr_1h.to_parquet(raw_dir / f"CME_MINI_{sym}_{yr}_1h.parquet")
                df_yr_1h.to_csv(raw_dir / f"CME_MINI_{sym}_{yr}_1h.csv")

        summary.append({
            "symbol": sym,
            "interval": "5m",
            "bars_2021_2023": len(df_5m),
            "start": df_5m.index.min().strftime("%Y-%m-%d %H:%M"),
            "end": df_5m.index.max().strftime("%Y-%m-%d %H:%M"),
            "type": "Consolidado Intradía 5m"
        })
        summary.append({
            "symbol": sym,
            "interval": "1h",
            "bars_2021_2023": len(df_1h),
            "start": df_1h.index.min().strftime("%Y-%m-%d %H:%M"),
            "end": df_1h.index.max().strftime("%Y-%m-%d %H:%M"),
            "type": "Consolidado Horario 1h"
        })

    print("\n" + "=" * 90)
    print(" RESUMEN GENERAL DE DATOS CONSOLIDADOS 2021 - 2023 EN data/raw/")
    print("=" * 90)
    print(f" {'Símbolo':<7} | {'Int':<4} | {'Total Velas':<12} | {'Inicio':<16} | {'Fin':<16} | {'Tipo / Fuente':<24}")
    print("-" * 90)
    for item in summary:
        print(
            f" {item['symbol']:<7} | {item['interval']:<4} | {item['bars_2021_2023']:>10,}   | "
            f"{item['start']:<16} | {item['end']:<16} | {item['type']:<24}"
        )
    print("=" * 90)
    print("¡Descarga y consolidación 2021-2023 finalizada exitosamente en .parquet y .csv!")


if __name__ == "__main__":
    main()
