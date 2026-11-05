"""
src/core/data_loader.py
Carga y estandarización de datos de mercado con soporte de caché local (yfinance y TradingView / tvDatafeed).
"""

from pathlib import Path
from typing import Optional, Union
import os
import pandas as pd
import yfinance as yf

try:
    from src.core.futures_specs import FUTURES_SPECS
except ImportError:
    FUTURES_SPECS = {}

try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    TvDatafeed = None
    Interval = None


class DataLoader:
    # Mapeo de cadenas de texto a intervalos de TradingView
    TV_INTERVAL_MAP = {
        "1m": "in_1_minute",
        "3m": "in_3_minute",
        "5m": "in_5_minute",
        "15m": "in_15_minute",
        "30m": "in_30_minute",
        "45m": "in_45_minute",
        "1h": "in_1_hour",
        "2h": "in_2_hour",
        "3h": "in_3_hour",
        "4h": "in_4_hour",
        "1d": "in_daily",
        "1w": "in_weekly",
        "1M": "in_monthly",
    }

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tv = None

    def _get_tv_client(self) -> "TvDatafeed":
        """Inicializa el cliente de TvDatafeed bajo demanda."""
        if TvDatafeed is None:
            raise ImportError(
                "tvDatafeed no está instalado. Instálalo con: pip install git+https://github.com/rongardF/tvdatafeed.git"
            )
        if self._tv is None:
            username = os.getenv("TRADINGVIEW_USERNAME")
            password = os.getenv("TRADINGVIEW_PASSWORD")
            if username and password:
                self._tv = TvDatafeed(username=username, password=password)
            else:
                self._tv = TvDatafeed()
        return self._tv

    def fetch_tradingview(
        self,
        symbol: str,
        exchange: str = "CME_MINI",
        interval: str = "1h",
        n_bars: int = 5000,
        fut_contract: Optional[int] = 1,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Descarga datos históricos de TradingView (ej. futuros ES, NQ) o los lee de la caché local.
        
        Args:
            symbol (str): Ticker ('ES', 'NQ', 'MES', 'MNQ', 'AAPL', etc.)
            exchange (str): Bolsa ('CME_MINI' para futuros CME, 'NASDAQ', 'NYSE', etc.)
            interval (str): Intervalo ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            n_bars (int): Cantidad máxima de velas a solicitar (hasta ~5,000 en modo público).
            fut_contract (int, optional): 1 para contrato continuo del mes frontal (front-month).
            use_cache (bool): Si True, utiliza el archivo guardado si existe.
            
        Returns:
            pd.DataFrame: DataFrame estandarizado con columnas [Open, High, Low, Close, Volume]
        """
        safe_sym = symbol.replace("^", "").replace("/", "_").replace("-", "_")
        cache_file = self.cache_dir / f"{exchange}_{safe_sym}_{interval}_bars{n_bars}.parquet"
        csv_file = cache_file.with_suffix(".csv")

        if use_cache and cache_file.exists():
            return pd.read_parquet(cache_file)
        if use_cache and csv_file.exists():
            return pd.read_csv(csv_file, index_col=0, parse_dates=True)

        tv = self._get_tv_client()

        # Resolver intervalo
        if interval in self.TV_INTERVAL_MAP:
            attr_name = self.TV_INTERVAL_MAP[interval]
            tv_interval = getattr(Interval, attr_name)
        else:
            tv_interval = getattr(Interval, interval, Interval.in_1_hour)

        df = tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=tv_interval,
            n_bars=n_bars,
            fut_contract=fut_contract
        )

        if df is None or df.empty:
            raise ValueError(
                f"No se pudieron obtener datos para {exchange}:{symbol} con intervalo {interval}. "
                "Verifica el símbolo y la bolsa."
            )

        # Estandarizar columnas a TitleCase para compatibilidad con backtesting
        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }
        df = df.rename(columns=rename_map)
        cols_to_keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[cols_to_keep].copy()
        df.dropna(inplace=True)
        df.sort_index(inplace=True)

        # Asegurar índice datetime sin zona horaria para evitar discrepancias
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        if use_cache:
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass
            df.to_csv(csv_file)

        return df

    def fetch_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Descarga datos OHLCV desde Yahoo Finance o los lee de la caché local.
        
        Args:
            symbol (str): Ticker del activo (ej. 'SPY', 'QQQ', 'BTC-USD', 'AAPL')
            start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'
            end_date (str, optional): Fecha de fin en formato 'YYYY-MM-DD'
            interval (str): Frecuencia ('1d', '1h', etc.)
            use_cache (bool): Si True, utiliza el archivo guardado localmente si existe.
            
        Returns:
            pd.DataFrame: DataFrame estandarizado con columnas [Open, High, Low, Close, Volume]
        """
        safe_sym = symbol.replace("^", "").replace("/", "_").replace("-", "_")
        cache_file = self.cache_dir / f"{safe_sym}_{interval}_{start_date}_{end_date or 'latest'}.parquet"

        if use_cache:
            if cache_file.exists():
                return pd.read_parquet(cache_file)
            # Buscar si existe en caché de TradingView
            tv_parquet = self.cache_dir / f"CME_MINI_{safe_sym}_{interval}_bars5000.parquet"
            tv_csv = self.cache_dir / f"CME_MINI_{safe_sym}_{interval}_bars5000.csv"
            if tv_parquet.exists():
                df_tv = pd.read_parquet(tv_parquet)
                return self._filter_by_dates(df_tv, start_date, end_date)
            elif tv_csv.exists():
                df_tv = pd.read_csv(tv_csv, index_col=0, parse_dates=True)
                return self._filter_by_dates(df_tv, start_date, end_date)

        # Resolver ticker de futuros para Yahoo Finance si aplica (ej. ES -> ES=F)
        yf_symbol = symbol
        upper_sym = symbol.upper()
        if upper_sym in FUTURES_SPECS:
            yf_symbol = FUTURES_SPECS[upper_sym].get("yf", symbol)

        # Descarga desde yfinance
        df = yf.download(
            tickers=yf_symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=True
        )

        if df.empty:
            raise ValueError(f"No se encontraron datos para {symbol} en el rango indicado.")

        # Manejo de MultiIndex en columnas (común en versiones recientes de yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Estandarizar nombres de columnas
        column_map = {
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        }
        df = df[[c for c in column_map.keys() if c in df.columns]].copy()
        df.dropna(inplace=True)

        # Asegurar DatetimeIndex sin tz
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        if use_cache:
            try:
                df.to_parquet(cache_file)
            except Exception:
                # Si parquet falla, guardamos en csv
                csv_file = cache_file.with_suffix(".csv")
                df.to_csv(csv_file)

        return df

    def _filter_by_dates(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Filtra un DataFrame por rango de fechas si se especifican."""
        df_filtered = df.copy()
        if not isinstance(df_filtered.index, pd.DatetimeIndex):
            df_filtered.index = pd.to_datetime(df_filtered.index)
        if df_filtered.index.tz is not None:
            df_filtered.index = df_filtered.index.tz_localize(None)

        if start_date:
            df_filtered = df_filtered[df_filtered.index >= pd.to_datetime(start_date)]
        if end_date:
            df_filtered = df_filtered[df_filtered.index <= pd.to_datetime(end_date)]

        return df_filtered.sort_index()

    def fetch_extended_intraday(
        self,
        symbol: str,
        interval: str = "5m",
        period: str = "60d",
        use_cache: bool = True,
        save_as: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Descarga el rango máximo disponible de datos intradía desde Yahoo Finance (ej. ES=F, NQ=F),
        limpia las columnas a [Open, High, Low, Close, Volume], normaliza a 'America/New_York'
        y opcionalmente combina con datos existentes locales sin duplicados.
        """
        yf_symbol = symbol
        upper_sym = symbol.upper()
        if upper_sym in FUTURES_SPECS:
            yf_symbol = FUTURES_SPECS[upper_sym].get("yf", symbol)

        safe_sym = symbol.replace("^", "").replace("/", "_").replace("-", "_")
        cache_file = self.cache_dir / f"CME_EXT_{safe_sym}_{interval}.parquet"
        csv_file = cache_file.with_suffix(".csv")

        if use_cache and cache_file.exists():
            return pd.read_parquet(cache_file)
        if use_cache and csv_file.exists():
            return pd.read_csv(csv_file, index_col=0, parse_dates=True)

        df = yf.download(
            tickers=yf_symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False
        )

        if df.empty:
            raise ValueError(f"No se pudieron descargar datos para {symbol} ({yf_symbol}) en intervalo {interval}.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Estandarizar columnas
        req_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in req_cols if c in df.columns]].copy()
        df.dropna(inplace=True)
        df.sort_index(inplace=True)

        # Convertir a America/New_York
        if df.index.tz is not None:
            df = df.tz_convert("America/New_York")
        else:
            df = df.tz_localize("America/New_York")

        if use_cache or save_as:
            out_parquet = Path(save_as) if save_as else cache_file
            out_csv = out_parquet.with_suffix(".csv")
            try:
                df.to_parquet(out_parquet)
            except Exception:
                pass
            df.to_csv(out_csv)

        return df

    @staticmethod
    def stitch_dataframes(df_primary: pd.DataFrame, df_secondary: pd.DataFrame) -> pd.DataFrame:
        """
        Combina dos DataFrames de series temporales ordenadas eliminando duplicados
        y dando prioridad a los valores del DataFrame primario.
        """
        combined = pd.concat([df_secondary, df_primary])
        if not isinstance(combined.index, pd.DatetimeIndex):
            combined.index = pd.to_datetime(combined.index)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        return combined

