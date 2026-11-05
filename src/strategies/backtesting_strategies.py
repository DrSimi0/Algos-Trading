"""
src/strategies/backtesting_strategies.py
Estrategias diseñadas para el motor interactivo basado en backtesting.py (kernc).
"""

from typing import Callable
import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover


def SMA(values: np.ndarray, n: int) -> np.ndarray:
    """Calcula la media móvil simple (Simple Moving Average)."""
    return pd.Series(values).rolling(n).mean().to_numpy()


def RSI(values: np.ndarray, period: int = 14) -> np.ndarray:
    """Calcula el índice de fuerza relativa (Relative Strength Index)."""
    series = pd.Series(values)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0).to_numpy()


class SmaCrossBacktest(Strategy):
    """
    Estrategia de Cruce de Medias Móviles (SMA Crossover).
    - Compra cuando la media rápida supera a la lenta.
    - Cierra posición cuando la rápida cruza por debajo de la lenta.
    """
    n_fast: int = 20
    n_slow: int = 50
    sl_pct: float = 0.0  # Porcentaje de Stop Loss (0.0 = deshabilitado)
    tp_pct: float = 0.0  # Porcentaje de Take Profit (0.0 = deshabilitado)

    def init(self):
        # self.I() envuelve indicadores para cálculo óptimo y visualización automática
        self.sma_fast = self.I(SMA, self.data.Close, self.n_fast)
        self.sma_slow = self.I(SMA, self.data.Close, self.n_slow)

    def next(self):
        price = self.data.Close[-1]

        # Señal de compra
        if crossover(self.sma_fast, self.sma_slow):
            if not self.position:
                sl = price * (1.0 - self.sl_pct) if self.sl_pct > 0 else None
                tp = price * (1.0 + self.tp_pct) if self.tp_pct > 0 else None
                self.buy(sl=sl, tp=tp)

        # Señal de salida / cruce a la baja
        elif crossover(self.sma_slow, self.sma_fast):
            if self.position:
                self.position.close()


class RsiMeanReversionBacktest(Strategy):
    """
    Estrategia de Reversión a la Media con RSI.
    - Compra cuando el activo está sobrevendido (RSI < oversold).
    - Vende cuando el activo se recupera o entra en sobrecompra (RSI > overbought).
    """
    rsi_period: int = 14
    oversold: int = 30
    overbought: int = 70
    sl_pct: float = 0.05  # 5% Stop loss por defecto
    tp_pct: float = 0.10  # 10% Take profit por defecto

    def init(self):
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)

    def next(self):
        price = self.data.Close[-1]
        current_rsi = self.rsi[-1]

        # Compra en sobreventa
        if current_rsi < self.oversold and not self.position:
            sl = price * (1.0 - self.sl_pct) if self.sl_pct > 0 else None
            tp = price * (1.0 + self.tp_pct) if self.tp_pct > 0 else None
            self.buy(sl=sl, tp=tp)

        # Venta / salida en sobrecompra
        elif current_rsi > self.overbought and self.position:
            self.position.close()
