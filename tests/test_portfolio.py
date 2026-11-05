import pytest
import pandas as pd
import numpy as np
from src.portfolio.allocator import PortfolioAllocator
from src.portfolio.risk_manager import RiskManager


def test_portfolio_allocator_equal_weight():
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    returns = pd.DataFrame({
        "Strat_A": [0.01] * 20,
        "Strat_B": [0.02] * 20
    }, index=dates)

    allocator = PortfolioAllocator(method="equal_weight")
    res = allocator.combine(returns)

    assert res["weights"]["Strat_A"] == 0.5
    assert res["weights"]["Strat_B"] == 0.5
    # Retorno ponderado diario: 0.5*0.01 + 0.5*0.02 = 0.015
    assert np.isclose(res["data"]["portfolio_return"].iloc[0], 0.015)


def test_risk_manager_correlation_and_limits():
    risk_mgr = RiskManager(max_portfolio_drawdown=0.10, max_correlation_threshold=0.80)

    # Correlación perfecta de 1.0
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    series = np.linspace(0.01, 0.05, 20)
    returns = pd.DataFrame({
        "Strat_1": series,
        "Strat_2": series
    }, index=dates)

    warnings = risk_mgr.audit_correlations(returns)
    assert len(warnings) == 1
    assert warnings[0]["correlation"] == 1.0

    # Drawdown check
    dd_check_ok = risk_mgr.check_drawdown_limit(-0.05)
    assert not dd_check_ok["circuit_breaker_triggered"]

    dd_check_breach = risk_mgr.check_drawdown_limit(-0.15)
    assert dd_check_breach["circuit_breaker_triggered"]
