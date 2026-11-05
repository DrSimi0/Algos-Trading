"""
src/portfolio/allocator.py
Asignador de capital para portafolios multi-estrategia.
Soporta Equal-Weight, Inverse-Volatility (Risk Parity base) y ponderación personalizada.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


class PortfolioAllocator:
    """
    Combina retornos de múltiples estrategias en una curva de capital consolidada.
    """

    def __init__(self, method: str = "equal_weight", custom_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            method: 'equal_weight', 'inverse_volatility', o 'custom'
            custom_weights: Diccionario de pesos {strategy_name: peso} si method='custom'
        """
        self.method = method
        self.custom_weights = custom_weights or {}

    def calculate_weights(self, strategy_returns: pd.DataFrame) -> pd.Series:
        """
        Calcula las ponderaciones normalizadas (suman 1.0) para cada estrategia.
        """
        strategies = strategy_returns.columns
        n = len(strategies)

        if n == 0:
            raise ValueError("No hay retornos de estrategias para asignar pesos.")

        if self.method == "equal_weight":
            weights = pd.Series(1.0 / n, index=strategies)

        elif self.method == "inverse_volatility":
            # Volatilidad de cada estrategia
            vols = strategy_returns.std()
            vols = vols.replace(0, 1e-6)
            inv_vols = 1.0 / vols
            weights = inv_vols / inv_vols.sum()

        elif self.method == "custom":
            weights = pd.Series([self.custom_weights.get(s, 1.0 / n) for s in strategies], index=strategies)
            weights = weights / weights.sum()

        else:
            raise ValueError(f"Método de asignación desconocido: {self.method}")

        return weights

    def combine(
        self,
        strategy_returns: pd.DataFrame,
        initial_capital: float = 10000.0,
        periods_per_year: int = 252
    ) -> Dict[str, Any]:
        """
        Combina los retornos de las estrategias y calcula la curva de capital del portafolio.
        
        Args:
            strategy_returns (pd.DataFrame): Columnas = nombres de estrategias, filas = retornos diarios
            initial_capital (float): Capital inicial del portafolio
            periods_per_year (int): 252 para acciones, 365 para cripto
            
        Returns:
            Dict con pesos, métricas consolidadas y serie temporal.
        """
        clean_returns = strategy_returns.dropna()
        weights = self.calculate_weights(clean_returns)

        # Retorno diario del portafolio ponderado
        portfolio_returns = (clean_returns * weights).sum(axis=1)

        # Curva de capital combinada
        equity_curve = (1.0 + portfolio_returns).cumprod() * initial_capital
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak

        # Métricas consolidadas
        total_return = (equity_curve.iloc[-1] / initial_capital) - 1.0
        n_periods = len(portfolio_returns)
        n_years = max(n_periods / periods_per_year, 1.0 / periods_per_year)
        cagr = ((equity_curve.iloc[-1] / initial_capital) ** (1.0 / n_years)) - 1.0
        volatility = portfolio_returns.std() * np.sqrt(periods_per_year)
        sharpe = (portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(periods_per_year)) if portfolio_returns.std() > 0 else 0.0
        max_dd = drawdown.min()

        metrics = {
            "Total Return (%)": round(total_return * 100, 2),
            "CAGR (%)": round(cagr * 100, 2),
            "Annual Volatility (%)": round(volatility * 100, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Max Drawdown (%)": round(max_dd * 100, 2),
            "Calmar Ratio": round(cagr / abs(max_dd), 2) if abs(max_dd) > 0 else 0.0
        }

        portfolio_df = pd.DataFrame({
            "portfolio_return": portfolio_returns,
            "equity_curve": equity_curve,
            "drawdown": drawdown
        })

        return {
            "weights": weights.to_dict(),
            "metrics": metrics,
            "data": portfolio_df,
            "correlation_matrix": clean_returns.corr()
        }
