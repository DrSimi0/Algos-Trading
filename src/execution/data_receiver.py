"""
src/execution/data_receiver.py
Receptor y administrador de buffer de datos de mercado en tiempo real.
Mantiene las últimas 780 barras de 5 minutos necesarias para calibrar
el estimador OLS y el test ADF de cointegración (F4) y el VWAP diario (F3).
"""

from pathlib import Path
import logging
from typing import Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("DataReceiver")


class LiveDataBuffer:
    """
    Gestiona un buffer rodante sincronizado en memoria para dos instrumentos (ej. NQ y ES).
    Garantiza timestamps en zona horaria America/New_York.
    """

    def __init__(
        self,
        symbol_x: str = "ES",
        symbol_y: str = "NQ",
        window_bars: int = 780,
    ):
        self.symbol_x = symbol_x
        self.symbol_y = symbol_y
        self.window_bars = window_bars
        self.df_x = pd.DataFrame()
        self.df_y = pd.DataFrame()

    def preload_historical_parquet(
        self,
        path_x: str = "data/raw/CME_MINI_ES_5m_bars5000.parquet",
        path_y: str = "data/raw/CME_MINI_NQ_5m_bars5000.parquet",
    ) -> None:
        """
        Carga las barras históricas más recientes para inicializar el buffer de 780 barras.
        """
        px = Path(path_x)
        py = Path(path_y)

        if not px.exists() or not py.exists():
            logger.warning("Archivos históricos %s o %s no encontrados.", px, py)
            return

        df_x = pd.read_parquet(px)
        df_y = pd.read_parquet(py)

        self.df_x = self._standardize_df(df_x).tail(self.window_bars)
        self.df_y = self._standardize_df(df_y).tail(self.window_bars)

        # Sincronizar índices internos
        common_idx = self.df_x.index.intersection(self.df_y.index)
        self.df_x = self.df_x.loc[common_idx]
        self.df_y = self.df_y.loc[common_idx]

        logger.info(
            "Buffer inicializado con %d barras históricas coincidentes (%s a %s)",
            len(common_idx),
            common_idx[0] if len(common_idx) > 0 else "N/A",
            common_idx[-1] if len(common_idx) > 0 else "N/A",
        )

    def _standardize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estandariza columnas y zona horaria a America/New_York.
        """
        df = df.copy()
        # Si 'timestamp' está en las columnas, convertir en DatetimeIndex
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Asegurar zona horaria America/New_York
        if df.index.tz is None:
            # Si no tiene zona, asumir México/Central y convertir a NY (como en el pipeline)
            df.index = df.index.tz_localize("America/Mexico_City").tz_convert("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")

        # Asegurar nombres de columnas en minúsculas estándar
        col_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=col_map)
        return df.sort_index()

    def append_bar(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        volume: float,
    ) -> None:
        """
        Añade una nueva barra de 5 minutos al buffer del instrumento correspondiente.
        """
        if timestamp.tz is None:
            ts = timestamp.tz_localize("America/New_York")
        else:
            ts = timestamp.tz_convert("America/New_York")

        row = pd.DataFrame(
            [{
                "open": float(open_p),
                "high": float(high_p),
                "low": float(low_p),
                "close": float(close_p),
                "volume": float(volume),
            }],
            index=[ts],
        )

        sym = symbol.upper()
        if "ES" in sym:
            self.df_x = pd.concat([self.df_x, row])
            self.df_x = self.df_x[~self.df_x.index.duplicated(keep="last")].tail(self.window_bars)
        elif "NQ" in sym:
            self.df_y = pd.concat([self.df_y, row])
            self.df_y = self.df_y[~self.df_y.index.duplicated(keep="last")].tail(self.window_bars)

    def get_aligned_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Devuelve los DataFrames de X e Y alineados por timestamp para el cálculo de cointegración.
        """
        common_idx = self.df_x.index.intersection(self.df_y.index)
        return self.df_x.loc[common_idx], self.df_y.loc[common_idx]
