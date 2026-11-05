"""
tests/test_interactive_backtest.py
Pruebas unitarias para InteractiveBacktestEngine y las estrategias basadas en backtesting.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.backtest.interactive_engine import InteractiveBacktestEngine
from src.strategies.backtesting_strategies import SmaCrossBacktest, RsiMeanReversionBacktest


@pytest.fixture
def sample_ohlcv_data():
    """Genera datos OHLCV sintéticos reproducibles para pruebas."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=120, freq="B")
    
    # Simular una tendencia alcista con fluctuaciones
    trend = np.linspace(100, 150, 120)
    noise = np.random.normal(0, 1.5, 120)
    close = trend + noise
    
    high = close + np.random.uniform(0.5, 2.0, 120)
    low = close - np.random.uniform(0.5, 2.0, 120)
    open_price = low + (high - low) * np.random.uniform(0.2, 0.8, 120)
    volume = np.random.randint(100000, 500000, 120)

    df = pd.DataFrame({
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    }, index=dates)

    return df


def test_sma_cross_interactive_engine(sample_ohlcv_data, tmp_path):
    engine = InteractiveBacktestEngine(
        strategy=SmaCrossBacktest,
        cash=10000.0,
        commission=0.001
    )

    stats = engine.run(sample_ohlcv_data, n_fast=5, n_slow=15)
    assert stats is not None
    assert "Return [%]" in stats
    assert "Sharpe Ratio" in stats
    assert stats["Equity Final [$]"] > 0

    summary = engine.get_summary_metrics()
    assert "Return [%]" in summary
    assert "Sharpe Ratio" in summary

    # Probar generación de gráfico
    report_file = tmp_path / "test_report.html"
    generated_path = engine.plot(filename=str(report_file), open_browser=False)
    assert Path(generated_path).exists()
    assert Path(generated_path).stat().st_size > 0


def test_rsi_reversion_interactive_engine(sample_ohlcv_data):
    engine = InteractiveBacktestEngine(
        strategy=RsiMeanReversionBacktest,
        cash=10000.0,
        commission=0.001
    )

    stats = engine.run(sample_ohlcv_data, rsi_period=10, oversold=35, overbought=65)
    assert stats is not None
    assert "Return [%]" in stats
    assert "Win Rate [%]" in stats


def test_interactive_engine_validation():
    engine = InteractiveBacktestEngine(strategy=SmaCrossBacktest)
    invalid_df = pd.DataFrame({"Price": [10, 11, 12]})

    with pytest.raises(ValueError, match="El DataFrame debe contener las columnas"):
        engine.run(invalid_df)
