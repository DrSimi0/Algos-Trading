"""
tests/test_three_tier_and_prop_rules.py
Pruebas unitarias para:
1. Partición en 3 universos temporales (IS, OOS, Holdout) con doble cuarentena.
2. Auditoría de reglas de pruebas de fondeo (Apex / Topstep).
3. Filtro de factor suerte y concentración de ganancias.
4. Análisis de mesetas de estabilidad de parámetros.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.core.data_split import DataSplitManager, SecurityGateError
from src.core.prop_firm_rules import get_prop_firm_profile, eval_prop_firm_compliance, PropFirmProfile
from src.core.robustness import evaluate_luck_and_significance, evaluate_parameter_stability
from src.backtest.interactive_engine import InteractiveBacktestEngine


@pytest.fixture
def sample_timeseries_300():
    """Genera 300 días de serie temporal sintética."""
    dates = pd.date_range("2022-01-01", periods=300, freq="D")
    df = pd.DataFrame({
        "Open": np.linspace(100, 150, 300),
        "High": np.linspace(101, 152, 300),
        "Low": np.linspace(99, 149, 300),
        "Close": np.linspace(100.5, 151, 300),
        "Volume": [1000] * 300
    }, index=dates)
    return df


def test_three_tier_split_and_quarantine(sample_timeseries_300):
    manager = DataSplitManager(sample_timeseries_300, split_ratios=(0.60, 0.25, 0.15))

    # In-Sample libre
    df_is = manager.get_in_sample()
    assert len(df_is) == 180  # 60% de 300

    # OOS protegido por Visto Bueno 1
    with pytest.raises(SecurityGateError, match="CUARENTENA NIVEL 1"):
        manager.get_out_of_sample(authorized=False)

    df_oos = manager.get_out_of_sample(authorized=True, reason="Unit test OOS")
    assert len(df_oos) == 75  # 25% de 300

    # Holdout protegido por Visto Bueno 2
    with pytest.raises(SecurityGateError, match="CUARENTENA NIVEL 2"):
        manager.get_holdout(authorized=False)

    df_holdout = manager.get_holdout(authorized=True, reason="Unit test Holdout")
    assert len(df_holdout) == 45  # 15% de 300

    # Orden temporal estricto sin solapamiento
    assert df_is.index.max() < df_oos.index.min()
    assert df_oos.index.max() < df_holdout.index.min()


def test_prop_firm_trailing_drawdown_breach():
    profile = get_prop_firm_profile("apex_50k")
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    
    # Simular una cuenta que sube a $52,000 y cae a $49,000 (drawdown de $3,000 > límite de $2,500)
    equity = pd.Series([50000, 51000, 52000, 51000, 49000, 49500, 50000, 50500, 51000, 51500], index=dates)
    res = eval_prop_firm_compliance(profile, equity)

    assert not res["Aprobada"]
    assert "CUENTA QUEMADA" in res["Estado"]
    assert any("Trailing Drawdown excedido" in v for v in res["Violaciones"])


def test_prop_firm_daily_loss_limit_breach():
    profile = get_prop_firm_profile("topstep_50k")  # DLL de $1,000
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    
    # En un solo día cae de $51,000 a $49,500 ($1,500 de pérdida diaria > $1,000)
    equity = pd.Series([50000, 51000, 49500, 50000, 50500], index=dates)
    res = eval_prop_firm_compliance(profile, equity)

    assert not res["Aprobada"]
    assert any("Daily Loss Limit excedido" in v for v in res["Violaciones"])


def test_prop_firm_successful_evaluation():
    profile = get_prop_firm_profile("apex_50k")
    dates = pd.date_range("2023-01-01", periods=15, freq="D")
    
    # Curva ascendente consistente de $50k a $53.5k sin caer nunca más de $500
    equity = pd.Series(np.linspace(50000, 53500, 15), index=dates)
    res = eval_prop_firm_compliance(profile, equity)

    assert res["Aprobada"] is True
    assert "PASÓ LA PRUEBA" in res["Estado"]


def test_luck_filter_few_trades():
    # Solo 5 operaciones
    pnl = pd.Series([100.0, -50.0, 200.0, -30.0, 150.0])
    res = evaluate_luck_and_significance(pnl, sharpe_ratio=1.5, min_trades=30)

    assert res["Total Trades"] == 5
    assert any("Operaciones insuficientes" in a for a in res["Alertas"])
    assert "POSIBLE FACTOR SUERTE" in res["Veredicto"]


def test_luck_filter_high_concentration():
    # 40 operaciones pero 1 trade aporta casi toda la ganancia ($10,000 de $10,500)
    gains = [10.0] * 39 + [10000.0]
    pnl = pd.Series(gains)
    res = evaluate_luck_and_significance(pnl, sharpe_ratio=1.8, min_trades=30)

    assert any("Ganancia concentrada" in a for a in res["Alertas"])


def test_parameter_stability_analysis():
    # Rejilla donde los vecinos tienen buen Sharpe
    grid_df = pd.DataFrame({
        "n_fast": [10, 10, 15, 15, 20, 20],
        "n_slow": [40, 50, 40, 50, 40, 50],
        "Sharpe Ratio": [1.2, 1.3, 1.4, 1.35, 1.1, 1.25]
    })
    best_params = {"n_fast": 15, "n_slow": 45}
    stability = evaluate_parameter_stability(grid_df, best_params, ["n_fast", "n_slow"])

    assert "MESETA ESTABLE" in stability["Veredicto Estabilidad"]


def test_compare_three_tiers_matrix():
    is_stats = {"Return [%]": 50.0, "Sharpe Ratio": 1.5, "# Trades": 40}
    oos_stats = {"Return [%]": 30.0, "Sharpe Ratio": 1.2, "# Trades": 20}
    hold_stats = {"Return [%]": 25.0, "Sharpe Ratio": 1.1, "# Trades": 15}

    matrix = InteractiveBacktestEngine.compare_three_tiers(is_stats, oos_stats, hold_stats)
    assert isinstance(matrix, pd.DataFrame)
    assert "In-Sample (IS)" in matrix.columns
    assert "Out-of-Sample (OOS)" in matrix.columns
    assert "Holdout Ciego" in matrix.columns
    assert "Retención Holdout/IS" in matrix.columns
