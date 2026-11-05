"""
src/strategies/mean_reversion/rsi_reversion.py
Estrategia de reversión a la media basada en el Índice de Fuerza Relativa (RSI).
"""

from typing import Dict, Any
import pandas as pd
from ...core.base_strategy import BaseStrategy


class RSIReversionStrategy(BaseStrategy):
    """
    Estrategia de reversión a la media:
    - Compra cuando RSI está sobrevendido (< oversold_threshold).
    - Vende / sale de posición cuando RSI está sobrecomprado (> overbought_threshold).
    """

    def __init__(self, name: str = "RSI_Reversion", params: Dict[str, Any] = None):
        default_params = {
            "period": 14,
            "oversold": 30.0,
            "overbought": 70.0
        }
        if params:
            default_params.update(params)
        super().__init__(name=name, params=default_params)

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)

        period = self.params["period"]
        oversold = self.params["oversold"]
        overbought = self.params["overbought"]

        rsi = self._calculate_rsi(data["Close"], period)

        signals = pd.Series(0.0, index=data.index)
        position = 0.0

        # Iteración secuencial para mantener estado (state-machine)
        for i in range(len(data)):
            val = rsi.iloc[i]
            if pd.isna(val):
                signals.iloc[i] = 0.0
                continue

            if val < oversold:
                position = 1.0  # Entrar en largo
            elif val > overbought:
                position = 0.0  # Salir a cash

            signals.iloc[i] = position

        return signals
