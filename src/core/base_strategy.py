"""
src/core/base_strategy.py
Interfaz base abstracta para todas las estrategias del portafolio.
Cualquier estrategia traída de un repositorio externo debe heredar de BaseStrategy.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class BaseStrategy(ABC):
    """
    Clase base para cualquier estrategia de trading cuantitativo.
    Garantiza una interfaz uniforme para el Portfolio Manager y el Backtest Engine.
    """

    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera una serie de posiciones o señales basada en los datos históricos.
        
        Args:
            data (pd.DataFrame): DataFrame con columnas estándar ['Open', 'High', 'Low', 'Close', 'Volume']
                                 y un DatetimeIndex.
                                 
        Returns:
            pd.Series: Serie indexada por fecha con valores:
                       1.0  = Posición Larga (Comprar / Mantener)
                       -1.0 = Posición Corta (Venta en corto, si aplica)
                       0.0  = Fuera de mercado / Cash
        """
        pass

    def get_parameters(self) -> Dict[str, Any]:
        """Retorna el diccionario de parámetros configurables."""
        return self.params.copy()

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Actualiza parámetros de la estrategia."""
        self.params.update(params)

    def validate_data(self, data: pd.DataFrame) -> None:
        """Valida que los datos contengan las columnas requeridas."""
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas en el DataFrame: {missing}")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("El DataFrame debe tener un DatetimeIndex válido.")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', params={self.params})>"
