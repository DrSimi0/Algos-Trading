"""
src/backtest
Módulo de simulación cuantitativa:
- BacktestEngine: Motor vectorial rápido para señales continuas / matriciales.
- InteractiveBacktestEngine: Motor basado en eventos con backtesting.py y reportes HTML.
"""

from .engine import BacktestEngine
from .interactive_engine import InteractiveBacktestEngine

__all__ = ["BacktestEngine", "InteractiveBacktestEngine"]
