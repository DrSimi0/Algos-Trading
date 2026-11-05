"""
src/backtest/interactive_engine.py
Adaptador de alto nivel para ejecutar backtests interactivos basados en backtesting.py (kernc).
Soporta cálculo de métricas financieras, optimización de parámetros y generación de reportes HTML.
"""

from pathlib import Path
from typing import Type, Dict, Any, Optional
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy


class InteractiveBacktestEngine:
    """
    Motor interactivo de simulación basado en eventos usando backtesting.py.
    """

    def __init__(
        self,
        strategy: Type[Strategy],
        cash: float = 10000.0,
        commission: float = 0.001,  # 0.1% de comisión
        margin: float = 1.0,         # 1.0 = sin apalancamiento
        trade_on_close: bool = False,
        exclusive_orders: bool = True,
        finalize_trades: bool = True
    ):
        self.strategy_cls = strategy
        self.cash = cash
        self.commission = commission
        self.margin = margin
        self.trade_on_close = trade_on_close
        self.exclusive_orders = exclusive_orders
        self.finalize_trades = finalize_trades
        self.bt: Optional[Backtest] = None
        self.stats: Optional[pd.Series] = None

    def run(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Ejecuta el backtest sobre los datos proporcionados.
        
        Args:
            data (pd.DataFrame): DataFrame con columnas [Open, High, Low, Close, Volume]
            **kwargs: Parámetros opcionales para la estrategia (ej. n_fast=10, n_slow=30)
            
        Returns:
            pd.Series: Estadísticas completas del backtest.
        """
        self._validate_data(data)

        self.bt = Backtest(
            data=data,
            strategy=self.strategy_cls,
            cash=self.cash,
            commission=self.commission,
            margin=self.margin,
            trade_on_close=self.trade_on_close,
            exclusive_orders=self.exclusive_orders,
            finalize_trades=self.finalize_trades
        )

        self.stats = self.bt.run(**kwargs)
        return self.stats

    def optimize(
        self,
        maximize: str = "Sharpe Ratio",
        constraint: Optional[callable] = None,
        **kwargs
    ) -> pd.Series:
        """
        Ejecuta optimización de hiperparámetros de la estrategia.
        
        Args:
            maximize (str): Métrica a maximizar (ej. 'Sharpe Ratio', 'Return [%]', 'SQN')
            constraint (callable, optional): Función de restricción (ej. lambda p: p.n_fast < p.n_slow)
            **kwargs: Rangos de parámetros (ej. n_fast=range(5, 30, 5), n_slow=range(20, 100, 10))
            
        Returns:
            pd.Series: Mejores estadísticas obtenidas.
        """
        if self.bt is None:
            raise RuntimeError("Debe llamar a run() antes de optimizar.")

        self.stats = self.bt.optimize(
            maximize=maximize,
            constraint=constraint,
            **kwargs
        )
        return self.stats

    def plot(self, filename: str = "report.html", open_browser: bool = False) -> str:
        """
        Genera el gráfico interactivo HTML basado en Bokeh.
        
        Args:
            filename (str): Ruta o nombre del archivo HTML de salida.
            open_browser (bool): Si True, abre automáticamente el navegador por defecto.
            
        Returns:
            str: Ruta absoluta del archivo generado.
        """
        if self.bt is None or self.stats is None:
            raise RuntimeError("No se puede graficar sin haber ejecutado run() previamente.")

        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.bt.plot(filename=str(filepath), open_browser=open_browser)
        return str(filepath.resolve())

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Extrae un diccionario limpio con las métricas principales."""
        if self.stats is None:
            raise RuntimeError("No hay estadísticas disponibles. Ejecute run() primero.")

        keys = [
            "Start", "End", "Duration", "Exposure Time [%]",
            "Equity Final [$]", "Equity Peak [$]", "Return [%]",
            "Buy & Hold Return [%]", "Return (Ann.) [%]", "Volatility (Ann.) [%]",
            "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max. Drawdown [%]",
            "Avg. Drawdown [%]", "# Trades", "Win Rate [%]", "Best Trade [%]",
            "Worst Trade [%]", "Avg. Trade [%]", "Profit Factor"
        ]

        summary = {}
        for k in keys:
            if k in self.stats:
                val = self.stats[k]
                if isinstance(val, (float, int)):
                    summary[k] = round(val, 2)
                else:
                    summary[k] = str(val)
        return summary

    def get_trades(self) -> pd.DataFrame:
        """Retorna el DataFrame de operaciones cerradas del backtest."""
        if self.stats is None or "_trades" not in self.stats:
            return pd.DataFrame()
        return self.stats["_trades"].copy()

    def get_equity_curve(self) -> pd.Series:
        """Retorna la serie temporal de curva de balance ($)."""
        if self.stats is None or "_equity_curve" not in self.stats:
            return pd.Series(dtype=float)
        eq_df = self.stats["_equity_curve"]
        if "Equity" in eq_df.columns:
            return eq_df["Equity"].copy()
        return pd.Series(dtype=float)

    @staticmethod
    def compare_three_tiers(is_stats: pd.Series, oos_stats: pd.Series, holdout_stats: pd.Series) -> pd.DataFrame:
        """
        Genera una matriz comparativa entre In-Sample, Out-of-Sample y Holdout Ciego Definitivo.
        """
        metrics = [
            "Return [%]", "Return (Ann.) [%]", "Volatility (Ann.) [%]",
            "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
            "Max. Drawdown [%]", "Win Rate [%]", "Profit Factor", "# Trades"
        ]

        rows = []
        for m in metrics:
            is_v = is_stats.get(m, np.nan) if hasattr(is_stats, "get") else is_stats[m]
            oos_v = oos_stats.get(m, np.nan) if hasattr(oos_stats, "get") else oos_stats[m]
            hold_v = holdout_stats.get(m, np.nan) if hasattr(holdout_stats, "get") else holdout_stats[m]

            # Retención Holdout vs IS
            retention_str = "-"
            try:
                is_num = float(is_v)
                hold_num = float(hold_v)
                if m in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Return (Ann.) [%]", "Profit Factor"]:
                    if is_num != 0:
                        retention = (hold_num / is_num) * 100
                        retention_str = f"{retention:.1f}% retención"
                elif m in ["Max. Drawdown [%]", "Win Rate [%]"]:
                    diff = hold_num - is_num
                    retention_str = f"{diff:+.2f}% p.p."
            except (ValueError, TypeError, ZeroDivisionError):
                retention_str = "-"

            rows.append({
                "Métrica": m,
                "In-Sample (IS)": round(float(is_v), 2) if isinstance(is_v, (int, float)) else str(is_v),
                "Out-of-Sample (OOS)": round(float(oos_v), 2) if isinstance(oos_v, (int, float)) else str(oos_v),
                "Holdout Ciego": round(float(hold_v), 2) if isinstance(hold_v, (int, float)) else str(hold_v),
                "Retención Holdout/IS": retention_str
            })

        return pd.DataFrame(rows)

    @staticmethod
    def compare_is_oos(is_stats: pd.Series, oos_stats: pd.Series) -> pd.DataFrame:
        """
        Compara las métricas financieras entre In-Sample y Out-of-Sample,
        calculando la retención o degradación del rendimiento.
        """
        metrics_to_compare = [
            "Return [%]", "Return (Ann.) [%]", "Volatility (Ann.) [%]",
            "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
            "Max. Drawdown [%]", "Win Rate [%]", "Profit Factor", "# Trades"
        ]

        rows = []
        for metric in metrics_to_compare:
            is_val = is_stats.get(metric, np.nan) if hasattr(is_stats, "get") else is_stats[metric]
            oos_val = oos_stats.get(metric, np.nan) if hasattr(oos_stats, "get") else oos_stats[metric]

            # Calcular ratio o delta
            decay_str = "-"
            try:
                is_num = float(is_val)
                oos_num = float(oos_val)
                if metric in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Return (Ann.) [%]", "Profit Factor"]:
                    if is_num != 0:
                        retention = (oos_num / is_num) * 100
                        decay_str = f"{retention:.1f}% retención"
                elif metric in ["Max. Drawdown [%]"]:
                    diff = oos_num - is_num
                    decay_str = f"{diff:+.2f}% p.p."
                elif metric in ["Win Rate [%]"]:
                    diff = oos_num - is_num
                    decay_str = f"{diff:+.2f}% p.p."
            except (ValueError, TypeError, ZeroDivisionError):
                decay_str = "-"

            rows.append({
                "Métrica": metric,
                "In-Sample (IS)": round(float(is_val), 2) if isinstance(is_val, (int, float)) else str(is_val),
                "Out-of-Sample (OOS)": round(float(oos_val), 2) if isinstance(oos_val, (int, float)) else str(oos_val),
                "Comportamiento OOS": decay_str
            })

        return pd.DataFrame(rows)

    def _validate_data(self, data: pd.DataFrame) -> None:
        required = ["Open", "High", "Low", "Close"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"El DataFrame debe contener las columnas: {missing}")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("El DataFrame debe estar indexado con DatetimeIndex.")
