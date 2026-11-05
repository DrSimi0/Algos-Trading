"""
src/backtest/engine.py
Motor de backtesting cuantitativo vectorial con prevención de Lookahead Bias
y consideración de comisiones y deslizamiento (slippage).
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


class BacktestEngine:
    """
    Simulador cuantitativo de estrategias.
    Ejecuta las señales con un desfase de 1 barra (shift=1) para evitar el sesgo de mirar al futuro.
    """

    def __init__(
        self,
        commission_bps: float = 5.0,  # 5 puntos base = 0.05% por operación
        slippage_bps: float = 5.0,    # 5 puntos base = 0.05% por operación
        risk_free_rate: float = 0.0,  # Tasa libre de riesgo anual
        periods_per_year: int = 252   # 252 para diario (acciones), 365 para cripto
    ):
        self.commission_pct = commission_bps / 10000.0
        self.slippage_pct = slippage_bps / 10000.0
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Ejecuta el backtest de una señal sobre una serie de precios.
        
        Args:
            data (pd.DataFrame): Datos de mercado con columna 'Close'
            signals (pd.Series): Serie con valores 1.0, -1.0 o 0.0
            initial_capital (float): Capital inicial en dólares
            
        Returns:
            Dict con métricas financieras y serie temporal del portafolio.
        """
        # Alinear datos
        df = pd.DataFrame(index=data.index)
        df["close"] = data["Close"]
        df["market_return"] = df["close"].pct_change().fillna(0.0)

        # Regla de oro de ejecución: la señal calculada al cierre de t se ejecuta para t+1
        df["position"] = signals.reindex(df.index).fillna(0.0).shift(1).fillna(0.0)

        # Detectar cambios de posición para calcular costos de transacción
        df["position_change"] = df["position"].diff().abs().fillna(df["position"].abs())
        total_friction_rate = self.commission_pct + self.slippage_pct
        df["friction_cost"] = df["position_change"] * total_friction_rate

        # Retorno de la estrategia = posición * retorno del activo - fricción
        df["strategy_return"] = (df["position"] * df["market_return"]) - df["friction_cost"]

        # Curva de capital
        df["equity_curve"] = (1.0 + df["strategy_return"]).cumprod() * initial_capital
        df["peak"] = df["equity_curve"].cummax()
        df["drawdown"] = (df["equity_curve"] - df["peak"]) / df["peak"]

        # Métricas financieras
        metrics = self._calculate_metrics(df, initial_capital)

        return {
            "metrics": metrics,
            "data": df
        }

    def _calculate_metrics(self, df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        returns = df["strategy_return"]
        total_return = (df["equity_curve"].iloc[-1] / initial_capital) - 1.0

        n_periods = len(returns)
        n_years = max(n_periods / self.periods_per_year, 1.0 / self.periods_per_year)

        # CAGR (Tasa de Crecimiento Anual Compuesto)
        if df["equity_curve"].iloc[-1] > 0:
            cagr = ((df["equity_curve"].iloc[-1] / initial_capital) ** (1.0 / n_years)) - 1.0
        else:
            cagr = -1.0

        # Volatilidad anualizada
        volatility = returns.std() * np.sqrt(self.periods_per_year)

        # Sharpe Ratio
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess_returns = returns - rf_per_period
        sharpe = (excess_returns.mean() / returns.std() * np.sqrt(self.periods_per_year)) if returns.std() > 0 else 0.0

        # Sortino Ratio (solo volatilidad negativa)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self.periods_per_year)
        sortino = (excess_returns.mean() / downside_std * np.sqrt(self.periods_per_year)) if (downside_std > 0 and not np.isnan(downside_std)) else 0.0

        # Max Drawdown
        max_drawdown = df["drawdown"].min()

        # Calmar Ratio
        calmar = (cagr / abs(max_drawdown)) if abs(max_drawdown) > 0 else 0.0

        # Win Rate (% de días ganadores en posición)
        active_returns = returns[df["position"] != 0]
        if len(active_returns) > 0:
            winning_trades = (active_returns > 0).sum()
            win_rate = winning_trades / len(active_returns)
            gains = active_returns[active_returns > 0].sum()
            losses = abs(active_returns[active_returns < 0].sum())
            profit_factor = (gains / losses) if losses > 0 else float("inf")
        else:
            win_rate = 0.0
            profit_factor = 0.0

        return {
            "Total Return (%)": round(total_return * 100, 2),
            "CAGR (%)": round(cagr * 100, 2),
            "Annual Volatility (%)": round(volatility * 100, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
            "Max Drawdown (%)": round(max_drawdown * 100, 2),
            "Calmar Ratio": round(calmar, 2),
            "Win Rate (%)": round(win_rate * 100, 2),
            "Profit Factor": round(profit_factor, 2),
            "Total Periods": n_periods
        }
