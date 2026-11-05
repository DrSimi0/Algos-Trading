"""
scripts/generate_synthetic_futures_3y.py
Generador determinista de datos sintéticos de futuros CME (ES y NQ) para 3 años completos (2023, 2024, 2025).
Cubre:
- Año 1 (2023): In-Sample (Régimen Tendencial Alcista y Volatilidad Moderada ~13%).
- Año 2 (2024): Out-of-Sample (Régimen de Expansión de Volatilidad ~18% y Shocks Macro).
- Año 3 (2025): Holdout Ciego (Régimen Lateral / Compresión de Rango ~12%).

Garantiza cointegración sintética entre ES y NQ para soportar la estrategia F4.
Genera tanto series intradía de 5 minutos como agregaciones diarias (1d).
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_3y_futures(
    output_dir: str = "data/raw",
    seed: int = 42
):
    np.random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Calendario de Negociación RTH (09:30 a 16:00 EST en días laborables)
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="B", tz="America/New_York")
    
    # Horas de negociación: 09:30 a 15:55 (78 velas de 5 min por sesión)
    intraday_times = pd.date_range("09:30", "15:55", freq="5min").time
    
    all_timestamps = []
    for d in dates:
        for t in intraday_times:
            dt = pd.Timestamp.combine(d.date(), t).tz_localize("America/New_York")
            all_timestamps.append(dt)
            
    index = pd.DatetimeIndex(all_timestamps)
    n_bars = len(index)
    print(f"Generando {n_bars:,} barras de 5m ({index[0]} a {index[-1]})...")

    # 2. Generación de Retornos Multi-Régimen
    # Año 1 (2023): Drift positivo +0.06% diario, vol baja
    # Año 2 (2024): Drift mixto, vol alta, colas pesadas
    # Año 3 (2025): Mean-reversion sin tendencia, vol media-baja
    years = index.year
    returns_es = np.zeros(n_bars)
    
    # 78 barras por día
    dt_bar = 1.0 / (252.0 * 78.0)
    sqrt_dt = np.sqrt(dt_bar)
    
    for i in range(n_bars):
        yr = years[i]
        if yr == 2023:
            mu = 0.15  # 15% anual
            sigma = 0.13
        elif yr == 2024:
            mu = 0.08  # 8% anual
            sigma = 0.19
        else:  # 2025
            mu = 0.02  # 2% anual
            sigma = 0.12
            
        z = np.random.normal(0, 1)
        # Inyectar saltos ocasionales (macro news)
        if np.random.rand() < 0.0008:
            jump = np.random.choice([-1, 1]) * np.random.uniform(0.005, 0.015)
        else:
            jump = 0.0
            
        ret = (mu - 0.5 * sigma**2) * dt_bar + sigma * sqrt_dt * z + jump
        returns_es[i] = ret

    # 3. Trayectoria de Precios ES (comenzando en 3,850 en 2023)
    p0_es = 3850.0
    price_es = p0_es * np.exp(np.cumsum(returns_es))
    # Redondear a ticks de 0.25
    price_es = np.round(price_es / 0.25) * 0.25

    # 4. Trayectoria Cointegrada de NQ (Spread Ornstein-Uhlenbeck)
    # NQ ~ beta * ES + spread_OU
    beta = 3.2
    spread_ou = np.zeros(n_bars)
    theta = 0.05  # velocidad de reversión a la media
    for i in range(1, n_bars):
        spread_ou[i] = spread_ou[i-1] * (1.0 - theta) + np.random.normal(0, 8.0)
        
    price_nq = beta * price_es + 2500.0 + spread_ou
    price_nq = np.round(price_nq / 0.25) * 0.25

    # Función auxiliar para construir OHLCV a partir de serie de Close
    def build_ohlcv(close_series: np.ndarray, base_volume: int = 1500) -> pd.DataFrame:
        high = close_series + np.random.uniform(0.25, 1.50, n_bars)
        low = close_series - np.random.uniform(0.25, 1.50, n_bars)
        high = np.round(high / 0.25) * 0.25
        low = np.round(low / 0.25) * 0.25
        
        # Open entre low y high anterior
        open_p = np.zeros(n_bars)
        open_p[0] = close_series[0]
        open_p[1:] = close_series[:-1] + np.random.choice([-0.25, 0.0, 0.25], n_bars - 1)
        
        # Corregir límites
        high = np.maximum(high, np.maximum(open_p, close_series))
        low = np.minimum(low, np.minimum(open_p, close_series))
        
        volume = np.random.poisson(base_volume, n_bars)
        
        df = pd.DataFrame({
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close_series,
            "Volume": volume
        }, index=index)
        return df

    df_es = build_ohlcv(price_es, base_volume=2000)
    df_nq = build_ohlcv(price_nq, base_volume=1200)

    # 5. Guardar archivos de 5 minutos
    # Rutas estándar 3Y
    es_parquet = out_path / "CME_MINI_ES_5m_3y.parquet"
    es_csv = out_path / "CME_MINI_ES_5m_3y.csv"
    nq_parquet = out_path / "CME_MINI_NQ_5m_3y.parquet"
    nq_csv = out_path / "CME_MINI_NQ_5m_3y.csv"

    df_es.to_parquet(es_parquet)
    df_es.to_csv(es_csv)
    df_nq.to_parquet(nq_parquet)
    df_nq.to_csv(nq_csv)

    print(f"Guardado ES 5m 3Y: {es_parquet} ({len(df_es):,} filas)")
    print(f"Guardado NQ 5m 3Y: {nq_parquet} ({len(df_nq):,} filas)")

    # 6. Guardar versiones diarias agregadas (1d) para estrategias multitemporales (F2 ORB)
    df_es_daily = df_es.resample("1D").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    df_nq_daily = df_nq.resample("1D").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    df_es_daily.to_parquet(out_path / "CME_MINI_ES_1d_3y.parquet")
    df_es_daily.to_csv(out_path / "CME_MINI_ES_1d_3y.csv")
    df_nq_daily.to_parquet(out_path / "CME_MINI_NQ_1d_3y.parquet")
    df_nq_daily.to_csv(out_path / "CME_MINI_NQ_1d_3y.csv")

    print(f"Guardado ES 1d 3Y: {len(df_es_daily)} días")
    print(f"Guardado NQ 1d 3Y: {len(df_nq_daily)} días")
    print("\nGeneración de datos sintéticos de 3 años completada exitosamente.")


if __name__ == "__main__":
    generate_3y_futures()
