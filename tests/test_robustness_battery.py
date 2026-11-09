"""
tests/test_robustness_battery.py
Suite completa de pruebas unitarias y de integración para la batería de robustez cuantitativa:
1. Teorema de Erdős-Rényi corregido.
2. Session Block Bootstrap con auditoría de Daily Loss Limit (DLL) y zonas horarias UTC/DST.
3. Deflated Sharpe Ratio (DSR - Bailey & López de Prado) y penalización de múltiples ensayos K.
4. Pruebas de estrés What-If (Slippage 2x/3x, Top-3 Drop, Win-Pruning y Gap adverso en SL).
5. Walk-Forward Engine y Walk-Forward Matrix (WFM - Robert Pardo) con cuarentena estricta.
6. PropFirmRobustnessScorecard (100 puntos y 6 Compuertas Duras / Hard Gates).
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.core.monte_carlo import (
    MonteCarloSimulator,
    compute_erdos_renyi_expected_loss_streak,
    run_session_block_bootstrap,
)
from src.core.robustness import (
    compute_deflated_sharpe_ratio,
    run_what_if_stress_suite,
    PropFirmRobustnessScorecard,
    evaluate_parameter_stability,
)
from src.core.walk_forward import WalkForwardEngine, WalkForwardMatrix


# ==============================================================================
# TEST 1: TEOREMA DE ERDŐS-RÉNYI CORREGIDO
# ==============================================================================

def test_erdos_renyi_corrected_formula():
    """Valida la fórmula de Erdős-Rényi corregida: E[L_max] = (ln(N*(1-q)) + gamma) / ln(1/q) - 0.5."""
    gamma = 0.5772156649
    N = 330
    q = 0.5697

    # Cálculo manual
    expected_val = (math.log(N * (1.0 - q)) + gamma) / math.log(1.0 / q) - 0.5
    res = compute_erdos_renyi_expected_loss_streak(N, q)

    assert math.isclose(res, expected_val, rel_tol=1e-6)
    assert 9.30 <= res <= 9.35, f"Esperado en [9.30, 9.35], obtenido {res:.4f}"

    # Casos límite
    assert compute_erdos_renyi_expected_loss_streak(0, 0.5) == 0.0
    assert compute_erdos_renyi_expected_loss_streak(100, 0.0) == 0.0
    assert compute_erdos_renyi_expected_loss_streak(100, 1.0) == 0.0
    assert compute_erdos_renyi_expected_loss_streak(100, -0.1) == 0.0

    # Monotonía: más trades -> racha esperada mayor
    streak_100 = compute_erdos_renyi_expected_loss_streak(100, 0.5)
    streak_500 = compute_erdos_renyi_expected_loss_streak(500, 0.5)
    assert streak_500 > streak_100

    # Monotonía: mayor tasa de pérdida -> racha esperada mayor
    streak_q40 = compute_erdos_renyi_expected_loss_streak(200, 0.4)
    streak_q60 = compute_erdos_renyi_expected_loss_streak(200, 0.6)
    assert streak_q60 > streak_q40


# ==============================================================================
# TEST 2: SESSION BLOCK BOOTSTRAP Y AUDITORÍA DLL
# ==============================================================================

def test_session_block_bootstrap_dll_violation():
    """Verifica que el Session Block Bootstrap detecte fielmente violaciones del DLL intradía."""
    # 3 sesiones: en la primera hay 2 trades perdedores que suman -$1,100 (viola DLL de -$1,000)
    dates = [
        "2023-01-02 10:00:00-05:00",
        "2023-01-02 11:00:00-05:00",
        "2023-01-03 10:00:00-05:00",
        "2023-01-04 10:00:00-05:00",
    ]
    pnls = [-600.0, -500.0, 400.0, 500.0]
    df = pd.DataFrame({"exit_time": dates, "net_pnl_usd": pnls})

    res = run_session_block_bootstrap(
        trades_df=df,
        initial_capital=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_drawdown=2000.0,
        profit_target=3000.0,
        n_simulations=100,
        seed=42,
    )

    assert res["total_sessions"] == 3
    assert res["dll_violation_pct"] > 0.0
    assert "INVIABLE" in res["verdict"]
    assert res["daily_loss_stats"]["worst_day_usd"] >= 1100.0


def test_session_block_bootstrap_clean_survival():
    """Verifica que una estrategia con sesiones dentro de los límites reporte cero violaciones DLL."""
    dates = [
        "2023-01-02 10:00:00-05:00",
        "2023-01-02 14:00:00-05:00",
        "2023-01-03 10:00:00-05:00",
        "2023-01-03 14:00:00-05:00",
        "2023-01-04 10:00:00-05:00",
    ]
    pnls = [200.0, 150.0, -100.0, 250.0, 300.0]
    df = pd.DataFrame({"exit_time": dates, "net_pnl_usd": pnls})

    res = run_session_block_bootstrap(
        trades_df=df,
        initial_capital=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_drawdown=2000.0,
        profit_target=500.0,
        n_simulations=100,
        seed=42,
    )

    assert res["dll_violation_pct"] == 0.0
    assert res["risk_of_ruin_pct"] == 0.0
    assert "EXCELENTE" in res["verdict"] or "ROBUSTA" in res["verdict"]


# ==============================================================================
# TEST 3: DEFLATED SHARPE RATIO (DSR) & MOMENTOS ESTADÍSTICOS
# ==============================================================================

def test_deflated_sharpe_ratio_single_trial():
    """Verifica que con K=1, el benchmark de Sharpe sea 0.0 (PSR clásico)."""
    np.random.seed(42)
    # 100 retornos positivos con media alta
    rets = np.random.normal(loc=150.0, scale=50.0, size=100)

    res = compute_deflated_sharpe_ratio(rets, num_trials=1)
    assert res["benchmark_sr"] == 0.0
    assert res["num_trials"] == 1
    assert res["dsr"] > 0.99
    assert res["statistically_significant"] is True


def test_deflated_sharpe_ratio_data_mining_penalty():
    """Verifica que probar muchos ensayos K=500 reduzca el DSR castigando el sobreajuste."""
    np.random.seed(42)
    # Retornos ligeramente positivos con Sharpe moderado
    rets = np.random.normal(loc=10.0, scale=100.0, size=80)

    res_1 = compute_deflated_sharpe_ratio(rets, num_trials=1)
    res_500 = compute_deflated_sharpe_ratio(rets, num_trials=500)

    # El benchmark SR* debe ser mayor con K=500
    assert res_500["benchmark_sr"] > res_1["benchmark_sr"]
    # El DSR debe ser menor con K=500 por la penalización
    assert res_500["dsr"] < res_1["dsr"]


# ==============================================================================
# TEST 4: PRUEBAS DE ESTRÉS WHAT-IF
# ==============================================================================

def test_what_if_stress_suite_mechanics():
    """Valida los escenarios de choque de deslizamiento (2x/3x), Top-3 Drop y Win-Pruning."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "exit_time": dates,
        "net_pnl_usd": [500.0, 300.0, 200.0, -100.0, 400.0, -50.0, 150.0, -80.0, 250.0, 100.0],
        "contracts": [1] * 10,
        "exit_reason": ["TAKE_PROFIT", "TAKE_PROFIT", "TAKE_PROFIT", "STOP_LOSS", "TAKE_PROFIT",
                        "STOP_LOSS", "TAKE_PROFIT", "STOP_LOSS", "TAKE_PROFIT", "TAKE_PROFIT"]
    })

    res = run_what_if_stress_suite(df, symbol="ES", initial_capital=50000.0, mtdd=2000.0, dll=1000.0, seed=42)

    base = res["baseline"]
    s2x = res["slippage_2x"]
    s3x = res["slippage_3x"]
    top3 = res["top3_drop"]
    wp = res["win_pruning"]
    gap = res["gap_adverse_sl"]

    # ES tick value = $12.50. 2 ticks RT = $25.00 por trade
    expected_slip2x_deduction = 10 * 25.0
    assert math.isclose(base["net_profit_usd"] - s2x["net_profit_usd"], expected_slip2x_deduction, rel_tol=1e-4)

    # 4 ticks RT = $50.00 por trade
    expected_slip3x_deduction = 10 * 50.0
    assert math.isclose(base["net_profit_usd"] - s3x["net_profit_usd"], expected_slip3x_deduction, rel_tol=1e-4)

    # Top-3 trades eliminados (500, 400, 300 sumaban 1200)
    assert math.isclose(base["net_profit_usd"] - top3["net_profit_usd"], 1200.0, rel_tol=1e-4)
    assert top3["total_trades"] == 7

    # Win-pruning
    assert wp["pruned_trades_per_iter"] >= 1
    assert wp["median_profit_usd"] < base["net_profit_usd"]

    # Gap en SL: 3 trades tuvieron STOP_LOSS -> 3 * 2 * $12.50 = $75.00
    assert math.isclose(base["net_profit_usd"] - gap["net_profit_usd"], 75.0, rel_tol=1e-4)


# ==============================================================================
# TEST 5: WALK-FORWARD ENGINE & WALK-FORWARD MATRIX (CUARENTENA IN-SAMPLE)
# ==============================================================================

def test_walk_forward_engine_quarantine_enforcement():
    """Valida que WalkForwardEngine no permita datos posteriores a 2023-12-31."""
    dates = pd.date_range("2023-06-01", "2024-03-01", freq="D", tz="UTC")
    df = pd.DataFrame({
        "exit_time": dates,
        "net_pnl_usd": np.random.normal(50, 10, len(dates))
    })

    engine = WalkForwardEngine(df, train_days=60, test_days=20, enforce_quarantine=True)
    # Todos los datos de 2024 debieron ser descartados
    assert engine.df["_parsed_dt"].max() <= pd.Timestamp("2023-12-31 23:59:59.999999", tz="UTC")

    folds = engine.generate_folds()
    assert len(folds) > 0
    for f in folds:
        assert f["train_end"] <= pd.Timestamp("2024-01-01", tz="UTC")
        assert f["test_end"] <= pd.Timestamp("2024-01-01", tz="UTC")


def test_walk_forward_matrix_evaluation():
    """Valida la generación de la rejilla 2D y el cálculo de métricas de Robert Pardo."""
    dates = pd.date_range("2023-01-01", "2023-12-30", freq="2D", tz="UTC")
    df = pd.DataFrame({
        "exit_time": dates,
        "net_pnl_usd": [100.0 if i % 3 != 0 else -60.0 for i in range(len(dates))]
    })

    wfm = WalkForwardMatrix(train_windows=[60, 90], test_windows=[20, 30], enforce_quarantine=True)
    res = wfm.evaluate_trades(df)

    assert "matrix_results" in res
    assert len(res["matrix_results"]) > 0
    assert "mean_wfe_return" in res
    assert "pct_positive_oos_windows" in res
    assert res["mean_wfe_return"] > 0.0


# ==============================================================================
# TEST 6: PROPFIRM SCORECARD & 6 HARD GATES
# ==============================================================================

def test_prop_firm_scorecard_hard_gates_triggers():
    """Verifica que cada una de las Compuertas Duras descalifique la estrategia a FAIL."""
    scorecard = PropFirmRobustnessScorecard(capital=50000.0, mtdd=2000.0, dll=1000.0, target=3000.0)

    # Base limpia que pasaría
    base_mc = {
        "reshuffle": {"risk_of_ruin_pct": 0.0, "target_before_ruin_pct": 80.0, "drawdown_stats": {"p95_dd_usd": 800.0}},
        "bootstrap": {"risk_of_ruin_pct": 0.5, "target_before_ruin_pct": 75.0, "drawdown_stats": {"p95_dd_usd": 1000.0}},
        "session_bootstrap": {"dll_violation_pct": 0.0, "risk_of_ruin_pct": 0.5, "drawdown_stats": {"p95_dd_usd": 1000.0}},
    }
    base_wi = {
        "baseline": {"profit_factor": 2.0},
        "slippage_2x": {"profit_factor": 1.5, "dll_violated": False},
        "top3_drop": {"profit_factor": 1.4, "dll_violated": False},
        "win_pruning": {"p5_profit_factor": 1.2},
        "gap_adverse_sl": {"dll_violated": False},
    }

    # 1. Prueba de aprobación limpia
    eval_pass = scorecard.evaluate(monte_carlo_res=base_mc, what_if_res=base_wi)
    assert eval_pass["hard_gates_passed"] is True
    assert eval_pass["status"] == "PASS"

    # 2. Gate 1: P_ruin Reshuffling > 0.0%
    mc_g1 = dict(base_mc)
    mc_g1["reshuffle"] = {"risk_of_ruin_pct": 0.1, "target_before_ruin_pct": 70.0, "drawdown_stats": {"p95_dd_usd": 800.0}}
    eval_g1 = scorecard.evaluate(monte_carlo_res=mc_g1, what_if_res=base_wi)
    assert eval_g1["hard_gates_passed"] is False
    assert eval_g1["status"] == "FAIL"
    assert any("Gate 1" in v for v in eval_g1["hard_gates_violations"])

    # 3. Gate 2: P_ruin Bootstrapping >= 5.0%
    mc_g2 = dict(base_mc)
    mc_g2["bootstrap"] = {"risk_of_ruin_pct": 5.2, "target_before_ruin_pct": 60.0, "drawdown_stats": {"p95_dd_usd": 1200.0}}
    eval_g2 = scorecard.evaluate(monte_carlo_res=mc_g2, what_if_res=base_wi)
    assert eval_g2["status"] == "FAIL"
    assert any("Gate 2" in v for v in eval_g2["hard_gates_violations"])

    # 4. Gate 3: P95 Drawdown > $1,700
    mc_g3 = dict(base_mc)
    mc_g3["bootstrap"] = {"risk_of_ruin_pct": 0.5, "target_before_ruin_pct": 70.0, "drawdown_stats": {"p95_dd_usd": 1850.0}}
    eval_g3 = scorecard.evaluate(monte_carlo_res=mc_g3, what_if_res=base_wi)
    assert eval_g3["status"] == "FAIL"
    assert any("Gate 3" in v for v in eval_g3["hard_gates_violations"])

    # 5. Gate 4: Profit Factor < 1.0 bajo Slippage 2x
    wi_g4 = dict(base_wi)
    wi_g4["slippage_2x"] = {"profit_factor": 0.85, "dll_violated": False}
    eval_g4 = scorecard.evaluate(monte_carlo_res=base_mc, what_if_res=wi_g4)
    assert eval_g4["status"] == "FAIL"
    assert any("Gate 4" in v for v in eval_g4["hard_gates_violations"])

    # 6. Gate 5: Profit Factor < 1.0 tras Top-3 Drop
    wi_g5 = dict(base_wi)
    wi_g5["top3_drop"] = {"profit_factor": 0.90, "dll_violated": False}
    eval_g5 = scorecard.evaluate(monte_carlo_res=base_mc, what_if_res=wi_g5)
    assert eval_g5["status"] == "FAIL"
    assert any("Gate 5" in v for v in eval_g5["hard_gates_violations"])

    # 7. Gate 6: DLL violado en cualquier escenario
    wi_g6 = dict(base_wi)
    wi_g6["slippage_2x"] = {"profit_factor": 1.2, "dll_violated": True}
    eval_g6 = scorecard.evaluate(monte_carlo_res=base_mc, what_if_res=wi_g6)
    assert eval_g6["status"] == "FAIL"
    assert any("Gate 6" in v for v in eval_g6["hard_gates_violations"])


# ==============================================================================
# TEST 7: NUEVAS PRUEBAS DE COBERTURA & CASOS LÍMITE (ROBUSTNESS ADVANCED)
# ==============================================================================

def test_intraday_dll_breach_when_session_recovers():
    """
    Verifica que tanto _calc_pnl_metrics como run_session_block_bootstrap
    detecten el quiebre de DLL intradía incluso si la sesión se recupera y cierra positiva o con pérdida menor.
    """
    from src.core.robustness import _calc_pnl_metrics

    # Sesión única: -$600, -$500 (cumsum = -$1,100 -> quiebra DLL de $1,000), luego +$1,000 (cierre = -$100)
    dates = pd.Series(["2023-01-02", "2023-01-02", "2023-01-02"])
    pnls = pd.Series([-600.0, -500.0, 1000.0])

    m = _calc_pnl_metrics(pnls, dates_series=dates, dll=1000.0)
    assert m["dll_violated"] is True, "El quiebre intradía de DLL no fue detectado por _calc_pnl_metrics"
    assert m["worst_daily_loss_usd"] >= 1100.0, f"Esperado >= 1100, obtenido {m['worst_daily_loss_usd']}"

    # Verificación en run_session_block_bootstrap
    df = pd.DataFrame({"exit_time": dates, "net_pnl_usd": pnls})
    res_boot = run_session_block_bootstrap(
        trades_df=df,
        daily_loss_limit=1000.0,
        n_simulations=50,
        seed=42,
    )
    assert res_boot["dll_violation_pct"] == 100.0
    assert res_boot["daily_loss_stats"]["worst_day_usd"] >= 1100.0


def test_what_if_is_micro_column_support():
    """Valida que una columna booleana 'is_micro' aplique costos de tick micro ($1.25) en vez de estándar ($12.50)."""
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    df = pd.DataFrame({
        "exit_time": dates,
        "net_pnl_usd": [200.0, 300.0, -100.0, 150.0],
        "contracts": [1, 1, 1, 1],
        "is_micro": [True, True, True, True],
        "exit_reason": ["TAKE_PROFIT", "TAKE_PROFIT", "STOP_LOSS", "TAKE_PROFIT"]
    })

    res = run_what_if_stress_suite(df, symbol="ES", seed=42)
    base = res["baseline"]
    s2x = res["slippage_2x"]

    # 4 trades * 2 ticks RT * $1.25 (MES micro) = $10.00
    expected_diff = 4 * 2.0 * 1.25
    assert math.isclose(base["net_profit_usd"] - s2x["net_profit_usd"], expected_diff, rel_tol=1e-4)


def test_prop_firm_scorecard_gate_6_all_scenarios():
    """Verifica que Gate 6 se active si baseline, slippage 3x o top-3 drop violan el Daily Loss Limit."""
    scorecard = PropFirmRobustnessScorecard(capital=50000.0, mtdd=2000.0, dll=1000.0, target=3000.0)

    base_mc = {
        "reshuffle": {"risk_of_ruin_pct": 0.0, "drawdown_stats": {"p95_dd_usd": 800.0}},
        "bootstrap": {"risk_of_ruin_pct": 0.5, "target_before_ruin_pct": 75.0, "drawdown_stats": {"p95_dd_usd": 1000.0}},
        "session_bootstrap": {"dll_violation_pct": 0.0, "risk_of_ruin_pct": 0.5, "drawdown_stats": {"p95_dd_usd": 1000.0}},
    }

    # Baseline con DLL violated
    wi_base_breach = {
        "baseline": {"profit_factor": 2.0, "dll_violated": True},
        "slippage_2x": {"profit_factor": 1.5, "dll_violated": False},
        "slippage_3x": {"profit_factor": 1.3, "dll_violated": False},
        "top3_drop": {"profit_factor": 1.4, "dll_violated": False},
        "win_pruning": {"p5_profit_factor": 1.2},
        "gap_adverse_sl": {"dll_violated": False},
    }
    eval_base = scorecard.evaluate(monte_carlo_res=base_mc, what_if_res=wi_base_breach)
    assert eval_base["status"] == "FAIL"
    assert any("Baseline Histórico DLL Breach" in v for v in eval_base["hard_gates_violations"])

    # Slippage 3x con DLL violated
    wi_slip3x_breach = dict(wi_base_breach)
    wi_slip3x_breach["baseline"] = {"profit_factor": 2.0, "dll_violated": False}
    wi_slip3x_breach["slippage_3x"] = {"profit_factor": 1.1, "dll_violated": True}
    eval_s3 = scorecard.evaluate(monte_carlo_res=base_mc, what_if_res=wi_slip3x_breach)
    assert eval_s3["status"] == "FAIL"
    assert any("Slippage 3x DLL Breach" in v for v in eval_s3["hard_gates_violations"])


def test_session_block_bootstrap_datetime_index_support():
    """Valida que una Serie con DatetimeIndex agrupe operaciones por fecha de sesión sin requerir trade_dates."""
    dates = pd.to_datetime(["2023-01-02 10:00", "2023-01-02 14:00", "2023-01-03 10:00"], utc=True)
    pnl_series = pd.Series([100.0, 150.0, 200.0], index=dates)

    res = run_session_block_bootstrap(trades_pnl=pnl_series, n_simulations=50, seed=42)
    assert res["total_sessions"] == 2
    assert res["total_trades_analyzed"] == 3

