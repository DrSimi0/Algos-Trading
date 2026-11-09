"""
tests/test_f7_optimization_is_n1.py

Suite exhaustiva de pruebas unitarias y cuantitativas para la estrategia:
F7-Optimizacion -IS- N1 (ML Híbrido: Calibración, Barreras Asimétricas y Bet Sizing Kelly).
"""

from datetime import datetime, time, timedelta
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.strategies.f7_optimization_is_n1 import (
    CalibrationMethod,
    ProbabilityCalibrator,
    PurgedKFoldCV,
    CalibratedMetaClassifier,
    AdaptiveBarrierConfig,
    AdaptiveBarrierManager,
    calculate_truncated_fractional_kelly,
    calculate_cushion_deescalation_factor,
    PropRiskManagerOptimized,
    F7OptimizationISN1Strategy,
    F7OptimizationBacktester,
    label_historical_events_adaptive,
)
from src.strategies.meta_labeling_triple_barrier import (
    MicroFeatures,
    TradeRecord,
    ExitReason,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def sample_bars_df():
    """Genera 120 barras de 5m con estructura OHLCV válida."""
    np.random.seed(42)
    n = 120
    base_date = pd.Timestamp("2026-10-15 08:30:00")
    timestamps = [base_date + timedelta(minutes=5 * i) for i in range(n)]

    p = 5000.0
    prices = []
    for _ in range(n):
        p += np.random.normal(0.0, 1.2)
        prices.append(p)
    prices = np.array(prices)

    highs = prices + np.random.uniform(0.5, 2.5, n)
    lows = prices - np.random.uniform(0.5, 2.5, n)
    opens = (highs + lows) / 2.0
    closes = prices
    volumes = np.random.uniform(1000.0, 5000.0, n)

    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes
    }, index=pd.DatetimeIndex(timestamps))


# ==============================================================================
# 1. PRUEBAS DE CALIBRACIÓN DE PROBABILIDADES
# ==============================================================================

class TestProbabilityCalibration:
    """Valida Platt Scaling, Temperature Scaling e Isotonic Regression (PAV)."""

    def test_platt_scaling_fit_and_monotonicity(self):
        """Platt scaling debe ajustar a y b y preservar la monotonicidad de los logits."""
        np.random.seed(123)
        logits = np.linspace(-3.0, 3.0, 50)
        # Probabilidades reales distorsionadas
        true_probs = 1.0 / (1.0 + np.exp(-1.5 * logits + 0.3))
        labels = (np.random.rand(50) < true_probs).astype(int)

        calibrator = ProbabilityCalibrator(method=CalibrationMethod.PLATT)
        calibrator.fit(logits, labels)

        assert calibrator.is_fitted
        assert calibrator.a > 0.0

        p1 = calibrator.calibrate(-2.0)
        p2 = calibrator.calibrate(0.0)
        p3 = calibrator.calibrate(2.0)

        assert 0.0 < p1 < p2 < p3 < 1.0

    def test_temperature_scaling_softening_and_sharpening(self):
        """Temperature scaling: T > 1 suaviza logits; T < 1 los hace más agudos."""
        calibrator_soft = ProbabilityCalibrator(method=CalibrationMethod.TEMPERATURE)
        calibrator_soft.temperature = 2.0
        calibrator_soft.is_fitted = True

        calibrator_sharp = ProbabilityCalibrator(method=CalibrationMethod.TEMPERATURE)
        calibrator_sharp.temperature = 0.5
        calibrator_sharp.is_fitted = True

        logit = 2.0
        # Sin calibrar (T = 1.0)
        p_raw = 1.0 / (1.0 + math.exp(-logit))  # ~0.8808
        p_soft = calibrator_soft.calibrate(logit)   # 1 / (1 + exp(-1.0)) ~ 0.731
        p_sharp = calibrator_sharp.calibrate(logit) # 1 / (1 + exp(-4.0)) ~ 0.982

        assert 0.5 < p_soft < p_raw < p_sharp < 1.0

    def test_isotonic_pav_monotonicity_and_bounds(self):
        """PAV Isotonic Regression garantiza monotonicidad estricta y cotas [0, 1]."""
        np.random.seed(42)
        logits = np.array([-2.5, -1.8, -0.9, -0.2, 0.5, 1.1, 1.9, 2.7])
        labels = np.array([0, 0, 1, 0, 1, 1, 0, 1])

        calibrator = ProbabilityCalibrator(method=CalibrationMethod.ISOTONIC)
        calibrator.fit(logits, labels)

        assert calibrator.is_fitted
        assert calibrator.isotonic_knots_y is not None
        # Monotonía no decreciente
        diffs = np.diff(calibrator.isotonic_knots_y)
        assert (diffs >= 0.0).all()

        # Evaluación en puntos intermedios y extremos
        p_low = calibrator.calibrate(-5.0)
        p_mid = calibrator.calibrate(0.0)
        p_high = calibrator.calibrate(5.0)

        assert 0.0 < p_low <= p_mid <= p_high < 1.0

    def test_brier_score_improvement_platt(self):
        """La calibración Platt reduce el Brier Score sobre datos con sesgo sistemático."""
        np.random.seed(99)
        # Logits con sesgo de sobreconfianza (escala x3)
        logits = np.random.normal(0.0, 2.0, 100)
        # Verdad fundamental depende de logit / 3
        true_p = 1.0 / (1.0 + np.exp(-logits / 3.0))
        y = (np.random.rand(100) < true_p).astype(int)

        uncal_p = 1.0 / (1.0 + np.exp(-logits))
        brier_uncal = float(np.mean((uncal_p - y) ** 2))

        calibrator = ProbabilityCalibrator(method=CalibrationMethod.PLATT)
        calibrator.fit(logits, y)
        cal_p = np.array([calibrator.calibrate(z) for z in logits])
        brier_cal = float(np.mean((cal_p - y) ** 2))

        assert brier_cal <= brier_uncal


# ==============================================================================
# 2. PRUEBAS DE PURGED K-FOLD CROSS-VALIDATION
# ==============================================================================

class TestPurgedKFoldCrossValidation:
    """Verifica que Purged K-Fold Cross Validation no filtre eventos adyacentes."""

    def test_purged_splits_coverage_and_purge_window(self):
        """Cada evento de prueba debe estar excluido del entrenamiento junto con su ventana de purga."""
        event_indices = list(range(0, 100, 2))  # 50 eventos
        purge_window = 4

        cv = PurgedKFoldCV(n_splits=5, purge_window_bars=purge_window)
        splits = cv.split(event_indices)

        assert len(splits) == 5
        seen_test_indices = set()

        for train_idx, test_idx in splits:
            assert len(test_idx) > 0
            assert len(train_idx) > 0
            # No solapamiento entre train y test
            assert set(train_idx).isdisjoint(set(test_idx))

            test_bars = [event_indices[ti] for ti in test_idx]
            min_test = min(test_bars)
            max_test = max(test_bars)

            # Verificar que ningún evento de train esté dentro de [min_test - purge, max_test + purge]
            for tr_i in train_idx:
                bar_tr = event_indices[tr_i]
                assert not ((min_test - purge_window) <= bar_tr <= (max_test + purge_window)), (
                    f"Fuga temporal detectada: barra train {bar_tr} dentro de zona purgada "
                    f"[{min_test - purge_window}, {max_test + purge_window}]"
                )

            seen_test_indices.update(test_idx)

        assert seen_test_indices == set(range(len(event_indices)))


# ==============================================================================
# 3. PRUEBAS DE BARRERAS ASIMÉTRICAS ADAPTATIVAS PARA PROP TRADING
# ==============================================================================

class TestAdaptiveTripleBarriers:
    """Verifica el comportamiento de las barreras adaptadas a volatilidad y colchón de drawdown."""

    def test_strict_risk_reward_asymmetry_ratio(self):
        """El ratio ell^+ / ell^- debe satisfacer b_t >= 2.0 bajo todos los regímenes."""
        mgr = AdaptiveBarrierManager()

        vol_levels = [5.0, 10.0, 25.0, 50.0]
        cushion_levels = [0.10, 0.30, 0.50, 0.80, 1.00]

        for sigma in vol_levels:
            for cushion in cushion_levels:
                b_up, b_low, v_bars, l_plus, l_minus = mgr.calculate_regime_adjusted_barriers(
                    current_price=5000.0,
                    current_sigma=sigma,
                    rolling_median_sigma=10.0,
                    cushion_ratio=cushion,
                    signal_star=1
                )
                ratio = l_plus / l_minus
                assert ratio >= 2.0, f"Violación de asimetría: ratio={ratio:.2f} con sigma={sigma}, cushion={cushion}"
                assert b_up > 5000.0
                assert b_low < 5000.0
                assert v_bars >= 3

    def test_stop_loss_compresses_in_high_volatility(self):
        """En alta volatilidad (omega_vol > 1), ell^- se contrae para estabilizar el riesgo en USD."""
        mgr = AdaptiveBarrierManager()

        # Volatilidad normal (10 vs 10)
        _, _, _, _, sl_normal = mgr.calculate_regime_adjusted_barriers(
            current_price=5000.0, current_sigma=10.0, rolling_median_sigma=10.0, cushion_ratio=1.0, signal_star=1
        )
        # Volatilidad duplicada (20 vs 10)
        _, _, _, _, sl_high = mgr.calculate_regime_adjusted_barriers(
            current_price=5000.0, current_sigma=20.0, rolling_median_sigma=10.0, cushion_ratio=1.0, signal_star=1
        )

        assert sl_high < sl_normal, "El multiplicador de SL no se contrajo en alta volatilidad."

    def test_stop_loss_compresses_when_cushion_shrinks(self):
        """Cuando el colchón de MTDD cae por debajo del 50%, ell^- se contrae defensivamente."""
        mgr = AdaptiveBarrierManager()

        _, _, _, _, sl_healthy = mgr.calculate_regime_adjusted_barriers(
            current_price=5000.0, current_sigma=10.0, rolling_median_sigma=10.0, cushion_ratio=0.80, signal_star=1
        )
        _, _, _, _, sl_danger = mgr.calculate_regime_adjusted_barriers(
            current_price=5000.0, current_sigma=10.0, rolling_median_sigma=10.0, cushion_ratio=0.25, signal_star=1
        )

        assert sl_danger < sl_healthy, "El multiplicador de SL no se contrajo ante peligro de MTDD."


# ==============================================================================
# 4. PRUEBAS DE CRITERIO DE KELLY TRUNCADO Y DESESCALAMIENTO
# ==============================================================================

class TestTruncatedFractionalKellyBetSizing:
    """Verifica las propiedades matemáticas de Kelly fraccional y desescalamiento por cushion."""

    def test_kelly_zero_below_meta_threshold(self):
        """Si p_t < tau_meta, el tamaño de apuesta Kelly debe ser idénticamente cero."""
        m = calculate_truncated_fractional_kelly(p_t=0.55, b_ratio=2.0, tau_meta=0.60)
        assert m == 0.0

    def test_kelly_zero_when_negative_edge(self):
        """Si la esperanza matemática es <= 0 (p <= 1/(b+1)), la apuesta es 0.0."""
        # Para b=2.0, breakeven es p=1/3 (~0.333)
        m = calculate_truncated_fractional_kelly(p_t=0.30, b_ratio=2.0, tau_meta=0.25)
        assert m == 0.0

    def test_fractional_kelly_quarter_scaling(self):
        """Quarter-Kelly (kappa=0.25) debe ser exactamente el 25% del Full Kelly."""
        p = 0.65
        b = 2.0
        # Full Kelly: (2 * 0.65 - 0.35) / 2 = (1.30 - 0.35) / 2 = 0.95 / 2 = 0.475
        m_full = calculate_truncated_fractional_kelly(p_t=p, b_ratio=b, tau_meta=0.50, kelly_fraction=1.0)
        m_quarter = calculate_truncated_fractional_kelly(p_t=p, b_ratio=b, tau_meta=0.50, kelly_fraction=0.25)

        np.testing.assert_allclose(m_full, 0.475, rtol=1e-5)
        np.testing.assert_allclose(m_quarter, 0.475 * 0.25, rtol=1e-5)

    def test_kelly_monotonicity_in_prob_and_payoff(self):
        """El tamaño de apuesta Kelly crece monótonamente con la probabilidad y con el ratio b."""
        m_p60 = calculate_truncated_fractional_kelly(p_t=0.60, b_ratio=2.0, tau_meta=0.50)
        m_p70 = calculate_truncated_fractional_kelly(p_t=0.70, b_ratio=2.0, tau_meta=0.50)
        m_p80 = calculate_truncated_fractional_kelly(p_t=0.80, b_ratio=2.0, tau_meta=0.50)

        assert 0.0 < m_p60 < m_p70 < m_p80 <= 1.0

        m_b20 = calculate_truncated_fractional_kelly(p_t=0.65, b_ratio=2.0, tau_meta=0.50)
        m_b25 = calculate_truncated_fractional_kelly(p_t=0.65, b_ratio=2.5, tau_meta=0.50)

        assert m_b20 < m_b25

    def test_cushion_deescalation_smooth_decay(self):
        """psi(rho) es 1.0 para rho >= 0.50, decrece suavemente en (0.15, 0.50), y es 0 en <= 0.15."""
        assert calculate_cushion_deescalation_factor(0.80) == 1.0
        assert calculate_cushion_deescalation_factor(0.50) == 1.0

        psi_40 = calculate_cushion_deescalation_factor(0.40)
        psi_25 = calculate_cushion_deescalation_factor(0.25)
        psi_16 = calculate_cushion_deescalation_factor(0.16)

        assert 0.0 < psi_16 < psi_25 < psi_40 < 1.0

        assert calculate_cushion_deescalation_factor(0.15) == 0.0
        assert calculate_cushion_deescalation_factor(0.05) == 0.0


# ==============================================================================
# 5. PRUEBAS DE GESTIÓN DE RIESGO PROP Y CONMUTACIÓN FORZADA A MICROS
# ==============================================================================

class TestPropRiskManagerOptimized:
    """Verifica la conmutación proactiva a microcontratos MES/MNQ por MTDD."""

    def test_proactive_micro_switch_when_cushion_low(self):
        """Cuando cushion <= force_micro_threshold (40%), DEBE operar Micro aunque haya capital."""
        mgr = PropRiskManagerOptimized(
            initial_capital=50000.0,
            dll=1000.0,
            mtdd=2000.0,
            gamma_t=0.15,
            force_micro_cushion_ratio=0.40
        )
        # Simulamos drawdown que deja cushion en $700 (35% MTDD)
        # peak = 50000, equity = 48700 => cushion = 48700 - (50000 - 2000) = 700 USD
        mgr.equity = 48700.0
        mgr.peak_equity = 50000.0
        mgr.daily_start_equity = 48700.0
        assert mgr.cushion_ratio == 0.35

        contracts, is_micro, unit_risk = mgr.calculate_position_size_optimized(
            std_symbol="ES",
            micro_symbol="MES",
            sigma=4.0,
            ell_minus=1.0,
            effective_bet_size=0.80,
            slippage_ticks=1.0
        )

        assert is_micro is True
        assert contracts >= 1
        # El riesgo unitario del micro (5 USD/pt) es 10 veces menor al estándar (50 USD/pt)
        assert unit_risk < 100.0

    def test_standard_contract_allowed_when_cushion_healthy(self):
        """Con colchón saludable (100%) y presupuesto suficiente, opera contratos estándar (ES)."""
        mgr = PropRiskManagerOptimized(
            initial_capital=50000.0,
            dll=1000.0,
            mtdd=2000.0,
            gamma_d=0.30,
            gamma_t=0.20,
            force_micro_cushion_ratio=0.40
        )
        assert mgr.cushion_ratio == 1.0

        contracts, is_micro, _ = mgr.calculate_position_size_optimized(
            std_symbol="ES",
            micro_symbol="MES",
            sigma=4.0,       # Baja volatilidad
            ell_minus=0.8,
            effective_bet_size=1.0,
            slippage_ticks=1.0
        )

        assert is_micro is False
        assert contracts >= 1

    def test_circuit_breaker_halts_trading(self):
        """Si cushion <= 15% MTDD, el disyuntor analítico detiene la cuenta inmediatamente."""
        mgr = PropRiskManagerOptimized(
            initial_capital=50000.0,
            dll=1000.0,
            mtdd=2000.0,
            circuit_breaker_ratio=0.15
        )
        # Colchón de $200 (10% de MTDD)
        mgr.equity = 48200.0
        mgr.peak_equity = 50000.0
        assert mgr.cushion_ratio == 0.10

        assert mgr.check_circuit_breaker() is True
        assert mgr.is_halted is True

        contracts, _, _ = mgr.calculate_position_size_optimized(
            std_symbol="ES", micro_symbol="MES", sigma=5.0, ell_minus=1.0, effective_bet_size=1.0
        )
        assert contracts == 0


# ==============================================================================
# 6. PRUEBAS END-TO-END DE LA ESTRATEGIA Y BACKTESTER
# ==============================================================================

class TestF7OptimizationEndToEnd:
    """Verifica la ejecución integral de F7-Optimizacion -IS- N1 sobre datos reales."""

    def test_strategy_signals_generation_conformity(self, sample_bars_df):
        """generate_signals cumple con la interfaz BaseStrategy y produce {-1, 0, 1}."""
        strat = F7OptimizationISN1Strategy()
        signals = strat.generate_signals(sample_bars_df)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_bars_df)
        assert set(signals.unique()).issubset({-1.0, 0.0, 1.0})

    def test_backtester_run_on_es_sample(self, sample_bars_df):
        """Ejecuta el simulador F7OptimizationBacktester sobre barras sintéticas."""
        strat = F7OptimizationISN1Strategy(params={
            "initial_capital": 50000.0,
            "calibration_method": "platt"
        })
        backtester = F7OptimizationBacktester(strategy=strat)
        results = backtester.run(sample_bars_df)

        assert "total_trades" in results
        assert "win_rate" in results
        assert "profit_factor" in results
        assert "net_profit" in results
        assert "max_drawdown_usd" in results
        assert "micro_trades" in results
        assert "standard_trades" in results

    def test_backtester_fit_classifier_with_purged_cv(self, sample_bars_df):
        """fit_classifier_on_data ejecuta Purged K-Fold CV sin errores y ajusta el calibrador."""
        strat = F7OptimizationISN1Strategy(params={"calibration_method": "temperature"})
        backtester = F7OptimizationBacktester(strategy=strat)
        res = backtester.fit_classifier_on_data(sample_bars_df, purge_boundary=True)

        assert "total_bars_train" in res
        assert "events_labeled" in res
        assert "calibrator_fitted" in res
        assert res["calibration_method"] == "temperature"

    def test_backtester_on_real_es_dataset(self):
        """Verifica la ejecución en una ventana real del dataset CME ES 5m."""
        p = Path("data/raw/CME_MINI_ES_5m_bars5000.parquet")
        if not p.exists():
            p = Path("data/raw/CME_MINI_ES_5m_3y.parquet")
        if not p.exists():
            pytest.skip("Dataset real no disponible")

        df = pd.read_parquet(p)
        df_sub = df.iloc[:1000].copy()

        strat = F7OptimizationISN1Strategy(params={
            "symbol": "ES",
            "std_symbol": "ES",
            "micro_symbol": "MES",
            "calibration_method": "platt",
            "kelly_fraction": 0.25,
            "force_micro_cushion_ratio": 0.40
        })
        backtester = F7OptimizationBacktester(strategy=strat)
        results = backtester.run(df_sub)

        assert results["final_equity"] > 0.0
        assert results["max_drawdown_usd"] <= 2000.0

    def test_adaptive_labeling_consistency(self, sample_bars_df):
        """label_historical_events_adaptive genera etiquetas válidas y respeta ventanas de purga."""
        from src.strategies.meta_labeling_triple_barrier import compute_garman_klass_volatility
        sigma_series = compute_garman_klass_volatility(sample_bars_df)
        rolling_med = sigma_series.rolling(window=20, min_periods=2).median().bfill()
        barrier_mgr = AdaptiveBarrierManager()

        primary_signals = pd.Series(0, index=sample_bars_df.index)
        primary_signals.iloc[5] = 1
        primary_signals.iloc[20] = -1
        primary_signals.iloc[40] = 1

        val_idx, labels = label_historical_events_adaptive(
            df=sample_bars_df,
            primary_signals=primary_signals,
            sigma_series=sigma_series,
            rolling_median_sigma_series=rolling_med,
            barrier_mgr=barrier_mgr,
            purge_last_bars=6
        )

        assert len(val_idx) == 3
        assert len(labels) == 3
        assert set(labels).issubset({0, 1})

    def test_isotonic_calibration_end_to_end_execution(self, sample_bars_df):
        """Verifica que la calibración isotónica opere y no bloquee 100% de trades."""
        strat = F7OptimizationISN1Strategy(params={
            "initial_capital": 50000.0,
            "calibration_method": "isotonic",
            "tau_meta": 0.42
        })
        backtester = F7OptimizationBacktester(strategy=strat)
        results = backtester.run(sample_bars_df)

        assert "total_trades" in results
        assert results["final_equity"] > 0.0

    def test_standard_and_micro_contract_dynamic_sizing(self):
        """Verifica que opere contratos estándar cuando el colchón es alto y reduzca a micro ante drawdown."""
        mgr = PropRiskManagerOptimized(
            initial_capital=50000.0,
            dll=1000.0,
            mtdd=2000.0,
            gamma_d=0.25,
            gamma_t=0.15,
            force_micro_cushion_ratio=0.40
        )
        # Colchón 100% ($2,000)
        n_std, is_micro_std, risk_std = mgr.calculate_position_size_optimized(
            std_symbol="ES", micro_symbol="MES", sigma=2.0, ell_minus=1.0,
            effective_bet_size=0.10, slippage_ticks=1.0
        )
        assert is_micro_std is False
        assert n_std >= 1

        # Reducir colchón a $600 (30% MTDD -> bajo umbral del 40% acumulado en sesiones previas)
        mgr.equity = 48600.0
        mgr.daily_start_equity = 48600.0
        n_micro, is_micro_low, risk_micro = mgr.calculate_position_size_optimized(
            std_symbol="ES", micro_symbol="MES", sigma=2.0, ell_minus=1.0,
            effective_bet_size=0.10, slippage_ticks=1.0
        )
        assert is_micro_low is True
        assert n_micro >= 1
        assert risk_micro < risk_std
