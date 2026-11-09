"""
tests/test_monte_carlo_and_portfolio.py
Pruebas unitarias e integradas para el motor de Simulación de Monte Carlo
y la evaluación de Portafolios Multi-Estrategia.
"""

import sys
import subprocess
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.core.monte_carlo import MonteCarloSimulator
from src.portfolio.allocator import PortfolioAllocator


def test_monte_carlo_simulator_reshuffle_deterministic():
    """Verifica que el Reshuffling sea reproducible con la misma semilla."""
    trades = pd.Series([100.0, -50.0, 200.0, -150.0, 80.0, -40.0, 120.0])
    sim1 = MonteCarloSimulator(trades, initial_capital=50000.0, max_trailing_drawdown=2000.0, profit_target=3000.0, seed=42)
    res1 = sim1.run_reshuffle(n_simulations=100)

    sim2 = MonteCarloSimulator(trades, initial_capital=50000.0, max_trailing_drawdown=2000.0, profit_target=3000.0, seed=42)
    res2 = sim2.run_reshuffle(n_simulations=100)

    assert res1["risk_of_ruin_pct"] == res2["risk_of_ruin_pct"]
    assert res1["drawdown_stats"]["median_dd_usd"] == res2["drawdown_stats"]["median_dd_usd"]
    assert res1["equity_stats"]["p50_median_equity_usd"] == res2["equity_stats"]["p50_median_equity_usd"]


def test_monte_carlo_ruin_detection():
    """Verifica que una estrategia con pérdidas catastróficas reporte 100% de ruina."""
    # Pérdidas acumuladas severas que superan los $2,000 de MTDD
    catastrophic_trades = pd.Series([-500.0, -800.0, -900.0, -1000.0])
    sim = MonteCarloSimulator(catastrophic_trades, initial_capital=50000.0, max_trailing_drawdown=2000.0, profit_target=3000.0, seed=42)
    res = sim.run_reshuffle(n_simulations=50)

    assert res["risk_of_ruin_pct"] == 100.0
    assert "INVIABLE" in res["verdict"]


def test_monte_carlo_target_detection():
    """Verifica que una estrategia con ganancias consistentes reporte alta probabilidad de target."""
    winning_trades = pd.Series([500.0, 400.0, 600.0, 700.0, 800.0, 500.0])
    sim = MonteCarloSimulator(winning_trades, initial_capital=50000.0, max_trailing_drawdown=2000.0, profit_target=3000.0, seed=42)
    res = sim.run_reshuffle(n_simulations=50)

    assert res["risk_of_ruin_pct"] == 0.0
    assert res["target_reach_pct"] == 100.0
    assert res["target_before_ruin_pct"] == 100.0


def test_monte_carlo_bootstrap():
    """Verifica que el modo Bootstrapping funcione y genere estimaciones válidas."""
    trades = pd.Series([150.0, -100.0, 200.0, -50.0, 300.0, -120.0])
    sim = MonteCarloSimulator(trades, initial_capital=50000.0, max_trailing_drawdown=2000.0, profit_target=3000.0, seed=123)
    res = sim.run_bootstrap(n_simulations=100)

    assert res["simulations"] == 100
    assert res["total_trades_analyzed"] == 6
    assert res["drawdown_stats"]["worst_case_dd_usd"] >= res["drawdown_stats"]["median_dd_usd"]


def test_portfolio_allocator_combine_trades_prop_firm():
    """Verifica la combinación de dos estrategias en una cuenta Prop Trading única."""
    dates_a = pd.date_range("2023-01-01", periods=5, freq="D")
    df_a = pd.DataFrame({
        "date": dates_a,
        "net_pnl_usd": [200.0, -100.0, 300.0, -50.0, 150.0]
    })

    dates_b = pd.date_range("2023-01-01", periods=5, freq="D")
    df_b = pd.DataFrame({
        "date": dates_b,
        "net_pnl_usd": [-50.0, 250.0, -100.0, 180.0, 200.0]
    })

    allocator = PortfolioAllocator(method="equal_weight")
    res = allocator.combine_trades_prop_firm(
        strategy_trades={"StratA": df_a, "StratB": df_b},
        initial_capital=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_drawdown=2000.0,
        profit_target=3000.0
    )

    assert "StratA" in res["weights"]
    assert "StratB" in res["weights"]
    assert res["weights"]["StratA"] == 0.5
    assert res["weights"]["StratB"] == 0.5

    pm = res["portfolio_metrics"]
    # PnL total esperado: 0.5 * sum(A) + 0.5 * sum(B) = 0.5 * 500 + 0.5 * 480 = 250 + 240 = 490
    assert np.isclose(pm["net_profit_usd"], 490.0)
    assert pm["final_equity"] == 50490.0
    assert not pm["mtdd_breached"]
    assert pm["dll_violations_count"] == 0


def test_portfolio_diversification_drawdown_reduction():
    """Verifica que dos estrategias descorrelacionadas o negativamente correlacionadas reduzcan el Drawdown."""
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    # Estrategia A gana en días pares y pierde en impares
    df_a = pd.DataFrame({"date": dates, "net_pnl_usd": [300.0, -200.0, 300.0, -200.0, 300.0, -200.0]})
    # Estrategia B pierde en pares y gana en impares (cobertura perfecta)
    df_b = pd.DataFrame({"date": dates, "net_pnl_usd": [-200.0, 300.0, -200.0, 300.0, -200.0, 300.0]})

    allocator = PortfolioAllocator(method="equal_weight")
    res = allocator.combine_trades_prop_firm(
        strategy_trades={"A": df_a, "B": df_b},
        initial_capital=50000.0
    )

    # La correlación debe ser fuertemente negativa
    assert res["pearson_correlation"] < -0.90
    # En combinación 50/50, cada día gana: 0.5*300 + 0.5*(-200) = 50 cada día -> CERO DRAWDOWN
    assert np.isclose(res["portfolio_metrics"]["portfolio_max_dd_usd"], 0.0)
    assert res["portfolio_metrics"]["dd_reduction_pct"] == 100.0


def test_cli_run_monte_carlo():
    """Verifica la ejecución de run_monte_carlo.py vía CLI."""
    cmd = [sys.executable, "run_monte_carlo.py", "--strategy", "f7", "--simulations", "50", "--method", "reshuffle"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0
    assert "SIMULACIÓN DE MONTE CARLO" in result.stdout
    assert "Riesgo de Ruina" in result.stdout


def test_cli_run_portfolio_backtest():
    """Verifica la ejecución de run_portfolio_backtest.py vía CLI con simulación Monte Carlo."""
    cmd = [
        sys.executable, "run_portfolio_backtest.py",
        "--strat1", "f7",
        "--strat2", "f4",
        "--method", "equal_weight",
        "--monte-carlo",
        "--simulations", "50"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0
    assert "EVALUACIÓN DE PORTAFOLIO MULTI-ESTRATEGIA" in result.stdout
    assert "BENEFICIO DE DIVERSIFICACIÓN" in result.stdout
    assert "SIMULACIÓN DE MONTE CARLO SOBRE EL PORTAFOLIO" in result.stdout
