"""
tests/test_data_split.py
Pruebas unitarias para DataSplitManager, control de cuarentena OOS y validación anti-fuga temporal.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.core.data_split import DataSplitManager, SecurityGateError
from src.backtest.interactive_engine import InteractiveBacktestEngine


@pytest.fixture
def sample_timeseries_data():
    """Genera 200 días de datos sintéticos con DatetimeIndex."""
    dates = pd.date_range("2022-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "Open": np.linspace(100, 150, 200),
        "High": np.linspace(101, 152, 200),
        "Low": np.linspace(99, 149, 200),
        "Close": np.linspace(100.5, 151, 200),
        "Volume": [1000] * 200
    }, index=dates)
    return df


def test_split_by_ratio(sample_timeseries_data):
    manager = DataSplitManager(sample_timeseries_data, split_ratio=0.70)
    is_df = manager.get_in_sample()

    assert len(is_df) == 140
    # Comprobar que OOS no es accesible sin autorización
    with pytest.raises(SecurityGateError, match="CUARENTENA"):
        manager.get_out_of_sample(authorized=False)

    # Comprobar acceso con autorización
    oos_df = manager.get_out_of_sample(authorized=True, reason="test")
    assert len(oos_df) > 0
    assert is_df.index.max() < oos_df.index.min()


def test_split_by_date(sample_timeseries_data):
    split_date = "2022-05-01"
    manager = DataSplitManager(sample_timeseries_data, split_date=split_date)
    is_df = manager.get_in_sample()
    oos_df = manager.get_out_of_sample(authorized=True)
    holdout_df = manager.get_holdout(authorized=True)

    assert is_df.index.max() < pd.to_datetime(split_date)
    assert oos_df.index.min() >= pd.to_datetime(split_date)
    assert len(is_df) + len(oos_df) + len(holdout_df) == len(sample_timeseries_data)


def test_audit_log_written(sample_timeseries_data, tmp_path):
    log_file = tmp_path / "test_audit.log"
    manager = DataSplitManager(sample_timeseries_data, split_ratio=0.80, audit_log_path=log_file)

    # Acceso autorizado debe escribir en el log
    manager.get_out_of_sample(authorized=True, reason="Unit test strategy X")
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "OOS_UNLOCKED" in content
    assert "Unit test strategy X" in content


def test_compare_is_oos_metrics():
    is_stats = {
        "Return [%]": 50.0,
        "Return (Ann.) [%]": 15.0,
        "Volatility (Ann.) [%]": 10.0,
        "Sharpe Ratio": 1.5,
        "Sortino Ratio": 2.1,
        "Calmar Ratio": 1.8,
        "Max. Drawdown [%]": -8.5,
        "Win Rate [%]": 60.0,
        "Profit Factor": 2.0,
        "# Trades": 40
    }
    oos_stats = {
        "Return [%]": 20.0,
        "Return (Ann.) [%]": 12.0,
        "Volatility (Ann.) [%]": 11.0,
        "Sharpe Ratio": 1.2,
        "Sortino Ratio": 1.7,
        "Calmar Ratio": 1.3,
        "Max. Drawdown [%]": -9.2,
        "Win Rate [%]": 55.0,
        "Profit Factor": 1.6,
        "# Trades": 15
    }

    df_comp = InteractiveBacktestEngine.compare_is_oos(is_stats, oos_stats)
    assert isinstance(df_comp, pd.DataFrame)
    assert len(df_comp) == 10
    assert "In-Sample (IS)" in df_comp.columns
    assert "Out-of-Sample (OOS)" in df_comp.columns
    assert "Comportamiento OOS" in df_comp.columns
