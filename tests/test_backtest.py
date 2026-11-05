import pytest
import pandas as pd
import numpy as np
from src.backtest.engine import BacktestEngine


def test_backtest_engine_execution():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    # Precios que suben constantemente 1% cada día
    prices = [100.0 * (1.01 ** i) for i in range(10)]
    df = pd.DataFrame({"Close": prices}, index=dates)

    # Señal de compra en t=0 (debe ejecutarse en t=1)
    signals = pd.Series([1.0] * 10, index=dates)

    engine = BacktestEngine(commission_bps=0.0, slippage_bps=0.0)
    result = engine.run(df, signals, initial_capital=1000.0)

    metrics = result["metrics"]
    data = result["data"]

    # En t=0, la posición ejecutada debe ser 0.0 debido al shift(1)
    assert data["position"].iloc[0] == 0.0
    # En t=1, la posición ejecutada es 1.0
    assert data["position"].iloc[1] == 1.0

    # Total Return debe ser positivo
    assert metrics["Total Return (%)"] > 0
    assert "Sharpe Ratio" in metrics
    assert "Max Drawdown (%)" in metrics
