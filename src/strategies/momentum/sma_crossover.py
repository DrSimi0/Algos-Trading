"""
src/strategies/momentum/sma_crossover.py
Estrategia clásica de seguimiento de tendencia por cruce de medias móviles (SMA Crossover).
"""

from typing import Dict, Any
import pandas as pd
from ...core.base_strategy import BaseStrategy


class SMACrossoverStrategy(BaseStrategy):
    """
    Estrategia de cruce de medias:
    - Señal 1.0: Cuando SMA rápida cruza por encima de SMA lenta.
    - Señal 0.0 o -1.0: Cuando SMA rápida cruza por debajo de SMA lenta.
    """

    def __init__(self, name: str = "SMA_Crossover", params: Dict[str, Any] = None):
        default_params = {
            "fast_window": 20,
            "slow_window": 50,
            "allow_short": False
        }
        if params:
            default_params.update(params)
        super().__init__(name=name, params=default_params)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        fast_window = self.params["fast_window"]
        slow_window = self.params["slow_window"]
        allow_short = self.params["allow_short"]

        if fast_window >= slow_window:
            raise ValueError("fast_window debe ser menor que slow_window.")

        close = data["Close"]
        sma_fast = close.rolling(window=fast_window).mean()
        sma_slow = close.rolling(window=slow_window).mean()

        signals = pd.Series(0.0, index=data.index)
        signals[sma_fast > sma_slow] = 1.0
        if allow_short:
            signals[sma_fast < sma_slow] = -1.0

        return signals
