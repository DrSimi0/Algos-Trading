import pytest
import pandas as pd
import numpy as np
from src.strategies.momentum.sma_crossover import SMACrossoverStrategy
from src.strategies.mean_reversion.rsi_reversion import RSIReversionStrategy


@pytest.fixture
def sample_market_data():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100))
    df = pd.DataFrame({
        "Open": prices,
        "High": prices + 1.0,
        "Low": prices - 1.0,
        "Close": prices,
        "Volume": 1000.0
    }, index=dates)
    return df


def test_sma_crossover_signals(sample_market_data):
    strategy = SMACrossoverStrategy(params={"fast_window": 5, "slow_window": 15})
    signals = strategy.generate_signals(sample_market_data)

    assert len(signals) == len(sample_market_data)
    assert set(signals.unique()).issubset({0.0, 1.0})


def test_rsi_reversion_signals(sample_market_data):
    strategy = RSIReversionStrategy(params={"period": 14, "oversold": 30, "overbought": 70})
    signals = strategy.generate_signals(sample_market_data)

    assert len(signals) == len(sample_market_data)
    assert set(signals.unique()).issubset({0.0, 1.0})
