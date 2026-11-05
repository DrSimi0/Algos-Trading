"""
src/portfolio/risk_manager.py
Gestión de riesgos a nivel portafolio:
- Auditoría de correlaciones entre estrategias.
- Circuit breaker por Max Drawdown global.
- Límites de apalancamiento y concentración.
"""

from typing import Dict, List, Any
import pandas as pd


class RiskManager:
    """
    Supervisor de riesgos para el Portfolio Manager.
    """

    def __init__(
        self,
        max_portfolio_drawdown: float = 0.15,  # 15% Max Drawdown tolerable
        max_correlation_threshold: float = 0.70, # Correlación máxima permitida entre dos estrategias
        max_strategy_allocation: float = 0.40    # Máxima ponderación por estrategia individual (40%)
    ):
        self.max_portfolio_drawdown = max_portfolio_drawdown
        self.max_correlation_threshold = max_correlation_threshold
        self.max_strategy_allocation = max_strategy_allocation

    def audit_correlations(self, returns_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detecta pares de estrategias excesivamente correlacionadas.
        """
        corr = returns_df.corr()
        warnings = []
        cols = corr.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if abs(val) > self.max_correlation_threshold:
                    warnings.append({
                        "pair": (cols[i], cols[j]),
                        "correlation": round(float(val), 2),
                        "warning": f"Alta correlación ({val:.2f}) excede el umbral de {self.max_correlation_threshold}. "
                                   f"Riesgo de duplicidad de exposición."
                    })
        return warnings

    def check_drawdown_limit(self, current_drawdown: float) -> Dict[str, Any]:
        """
        Verifica si el drawdown actual del portafolio excede el límite de tolerancia.
        """
        is_breached = abs(current_drawdown) > self.max_portfolio_drawdown
        return {
            "current_drawdown_pct": round(abs(current_drawdown) * 100, 2),
            "max_drawdown_limit_pct": round(self.max_portfolio_drawdown * 100, 2),
            "circuit_breaker_triggered": is_breached,
            "action": "HALT_TRADING_OR_SCALE_DOWN" if is_breached else "OK"
        }

    def validate_weights(self, weights: Dict[str, float]) -> List[str]:
        """
        Valida que ninguna estrategia concentre más del límite fijado.
        """
        alerts = []
        for strat, weight in weights.items():
            if weight > self.max_strategy_allocation:
                alerts.append(
                    f"Estrategia '{strat}' tiene peso de {weight:.1%}, superando el límite de concentración de {self.max_strategy_allocation:.1%}."
                )
        return alerts
