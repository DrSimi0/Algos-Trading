"""
tests/test_meta_labeling_triple_barrier.py

Suite exhaustiva de pruebas unitarias, matemáticas y de microestructura para la
Estrategia F7: Machine Learning Híbrido (Meta-Labeling & Triple Barrier).

Especificación Matemática de Referencia:
    reports/estrategias_matematicas/F7_meta_labeling_triple_barrier.md
"""

from datetime import datetime, time, timedelta
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.strategies.meta_labeling_triple_barrier import (
    MetaLabelingTripleBarrierStrategy,
    MetaLabelingBacktester,
    PropRiskManager,
    TradeRecord,
    ExitReason,
    PositionDirection,
    MicroFeatures,
    MetaLabelingClassifier,
    MetaModelWeights,
    calculate_barriers,
    evaluate_triple_barrier_exit,
    calculate_bet_size,
    apply_meta_veto,
    compute_garman_klass_volatility,
    compute_intraday_anchored_vwap,
    compute_order_flow_imbalance_zscore,
    compute_relative_volume,
    compute_range_compression,
    compute_session_fraction,
    label_historical_events,
    detect_data_time_offset,
)
from src.core.futures_specs import get_spec


# ==============================================================================
# FIXTURES DE DATOS SINTÉTICOS Y REALES
# ==============================================================================

@pytest.fixture
def sample_intraday_bars():
    """Genera 100 barras intradiarias de 5 minutos con estructura OHLCV consistente."""
    np.random.seed(42)
    n = 100
    base_date = pd.Timestamp("2026-10-15 08:30:00")
    timestamps = [base_date + timedelta(minutes=5 * i) for i in range(n)]

    prices = [5000.0]
    for _ in range(n - 1):
        prices.append(prices[-1] + np.random.normal(0.0, 1.5))
    prices = np.array(prices)

    highs = prices + np.random.uniform(0.5, 2.5, n)
    lows = prices - np.random.uniform(0.5, 2.5, n)
    opens = (highs + lows) / 2.0 + np.random.normal(0.0, 0.5, n)
    closes = prices
    volumes = np.random.uniform(1000.0, 5000.0, n)

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes
    }, index=pd.DatetimeIndex(timestamps))
    return df


# ==============================================================================
# 1. PRUEBAS DE CARACTERÍSTICAS MICROESTRUCTURALES (SECCIÓN 2)
# ==============================================================================

class TestMicrostructuralFeatures:
    """Verifica el cálculo analítico del vector de características X_t in R^6."""

    def test_garman_klass_volatility_strictly_positive(self, sample_intraday_bars):
        """sigma_t debe ser estrictamente positivo y expresado en puntos de índice."""
        sigma = compute_garman_klass_volatility(sample_intraday_bars)
        assert isinstance(sigma, pd.Series)
        assert len(sigma) == len(sample_intraday_bars)
        assert (sigma > 0.0).all()
        # Verificar que no hay NaN o inf
        assert not sigma.isna().any()
        assert not np.isinf(sigma).any()

    def test_anchored_vwap_and_delta(self, sample_intraday_bars):
        """VWAP intradiario debe acumularse correctamente y Delta P = Close - VWAP."""
        vwap, delta_vwap = compute_intraday_anchored_vwap(sample_intraday_bars)
        assert (vwap > 0.0).all()
        # Delta P debe ser exactamente Close - VWAP
        diff = sample_intraday_bars["Close"] - vwap
        np.testing.assert_allclose(delta_vwap.values, diff.values, rtol=1e-5)

    def test_ofi_zscore_bounded(self, sample_intraday_bars):
        """Z_OFI debe tener media cercana a 0 y estar acotado numéricamente."""
        z_ofi = compute_order_flow_imbalance_zscore(sample_intraday_bars, window=20)
        assert (z_ofi >= -4.0).all()
        assert (z_ofi <= 4.0).all()
        assert not z_ofi.isna().any()

    def test_rvol_strictly_positive(self, sample_intraday_bars):
        """RVOL_t debe ser un ratio adimensional positivo."""
        rvol = compute_relative_volume(sample_intraday_bars, rolling_window=20)
        assert (rvol >= 0.1).all()
        assert not rvol.isna().any()

    def test_range_compression_ratio(self, sample_intraday_bars):
        """Compresión de rango S_{range, t} debe ser un ratio acotado y positivo."""
        comp = compute_range_compression(sample_intraday_bars, recent_bars=12, baseline_days=5)
        assert (comp > 0.0).all()
        assert not comp.isna().any()

    def test_session_fraction_domain(self, sample_intraday_bars):
        """tau_tod in [0, 1] dentro de la sesión regular."""
        tod = compute_session_fraction(sample_intraday_bars)
        assert (tod >= 0.0).all()
        assert (tod <= 1.0).all()

    def test_micro_features_dataclass(self):
        """Verifica la serialización a array de MicroFeatures."""
        feat = MicroFeatures(
            sigma=12.5,
            z_ofi=1.2,
            delta_vwap=3.5,
            rvol=2.1,
            range_compression=0.45,
            tau_tod=0.35
        )
        arr = feat.to_array()
        assert len(arr) == 6
        assert arr[0] == 12.5
        assert arr[1] == 1.2
        assert arr[2] == 3.5

    def test_detect_data_time_offset_ct_vs_est(self):
        """Verifica la detección automática de horario Chicago (CT) frente a New York (EST)."""
        times_ct = [time(8, 30), time(8, 35), time(9, 30), time(14, 55)]
        df_ct = pd.DataFrame({
            "Volume": [10000.0, 8000.0, 2000.0, 1000.0],
            "Open": [5000.0] * 4,
            "High": [5005.0] * 4,
            "Low": [4995.0] * 4,
            "Close": [5000.0] * 4
        }, index=pd.to_datetime(["2026-10-15 08:30:00", "2026-10-15 08:35:00", "2026-10-15 09:30:00", "2026-10-15 14:55:00"]))
        assert detect_data_time_offset(df_ct) is True

        df_est = pd.DataFrame({
            "Volume": [1000.0, 1200.0, 15000.0, 1000.0],
            "Open": [5000.0] * 4,
            "High": [5005.0] * 4,
            "Low": [4995.0] * 4,
            "Close": [5000.0] * 4
        }, index=pd.to_datetime(["2026-10-15 08:30:00", "2026-10-15 08:35:00", "2026-10-15 09:30:00", "2026-10-15 14:55:00"]))
        assert detect_data_time_offset(df_est) is False

    def test_compute_range_compression_no_lookahead(self):
        """El cálculo de compresión no debe cambiar cuando barras futuras del mismo día se modifican."""
        dts = pd.date_range("2026-10-10 09:30", periods=50, freq="5min")
        df1 = pd.DataFrame({
            "Open": [5000.0] * 50,
            "High": [5010.0] * 50,
            "Low": [4990.0] * 50,
            "Close": [5000.0] * 50,
            "Volume": [1000.0] * 50
        }, index=dts)
        comp1 = compute_range_compression(df1, recent_bars=5, baseline_days=3)

        # Modificar únicamente las últimas 5 barras del mismo día
        df2 = df1.copy()
        df2.iloc[-5:, df2.columns.get_loc("High")] = 5500.0
        comp2 = compute_range_compression(df2, recent_bars=5, baseline_days=3)

        # Las primeras 20 barras deben ser idénticas (sin lookahead bias del pico futuro de la tarde)
        np.testing.assert_allclose(comp1.iloc[:20].values, comp2.iloc[:20].values)


# ==============================================================================
# 2. PRUEBAS DEL MODELO PRIMARIO Y META-MODELO DE CALIDAD (SECCIÓN 3)
# ==============================================================================

class TestMetaLabelingClassificationAndVeto:
    """Verifica las ecuaciones de señal primaria, logit, probabilidad y función de veto."""

    def test_primary_signal_values(self, sample_intraday_bars):
        """Señal primaria s_t^{(1)} in {-1, 0, +1}."""
        strat = MetaLabelingTripleBarrierStrategy()
        _, delta_vwap = compute_intraday_anchored_vwap(sample_intraday_bars)
        z_ofi = compute_order_flow_imbalance_zscore(sample_intraday_bars)
        signals = strat.generate_primary_signals(sample_intraday_bars, delta_vwap, z_ofi)
        unique_vals = set(signals.unique())
        assert unique_vals.issubset({-1, 0, 1})

    def test_meta_probability_sigmoid_range(self):
        """La probabilidad p_t estimada debe residir estrictamente en [0, 1]."""
        classifier = MetaLabelingClassifier()
        feat = MicroFeatures(10.0, 1.5, 5.0, 2.0, 0.4, 0.5)

        prob_long = classifier.predict_probability(feat, primary_signal=1)
        prob_short = classifier.predict_probability(feat, primary_signal=-1)

        assert 0.0 <= prob_long <= 1.0
        assert 0.0 <= prob_short <= 1.0

    def test_synergy_boosts_probability(self):
        """Cuando s^{(1)} está alineado con Z_OFI y VWAP, p_t debe ser superior que si se opone."""
        classifier = MetaLabelingClassifier()
        # OFI fuertemente positivo y precio sobre VWAP
        feat_bullish = MicroFeatures(sigma=10.0, z_ofi=2.0, delta_vwap=5.0, rvol=2.0, range_compression=0.5, tau_tod=0.5)

        p_aligned = classifier.predict_probability(feat_bullish, primary_signal=1)
        p_opposed = classifier.predict_probability(feat_bullish, primary_signal=-1)

        assert p_aligned > p_opposed
        assert p_aligned > 0.50

    def test_meta_veto_logic_and_orthogonality(self):
        """
        Verifica S_t^* = s_t^{(1)} * I(p_t >= tau_meta) y la ortogonalidad:
        S_{Long} * S_{Short} == 0.
        """
        tau_meta = 0.60

        # Caso 1: Señal primaria Long con alta probabilidad -> Aprobada
        s_star_approved = apply_meta_veto(primary_signal=1, p_t=0.72, tau_meta=tau_meta)
        assert s_star_approved == 1

        # Caso 2: Señal primaria Long con baja probabilidad -> Vetada
        s_star_vetoed = apply_meta_veto(primary_signal=1, p_t=0.54, tau_meta=tau_meta)
        assert s_star_vetoed == 0

        # Caso 3: Señal primaria Short con alta probabilidad -> Aprobada
        s_star_short_app = apply_meta_veto(primary_signal=-1, p_t=0.65, tau_meta=tau_meta)
        assert s_star_short_app == -1

        # Caso 4: Señal Neutral
        assert apply_meta_veto(primary_signal=0, p_t=0.85, tau_meta=tau_meta) == 0

        # Ortogonalidad analítica estricta
        s_long = 1 if s_star_approved == 1 else 0
        s_short = 1 if s_star_short_app == -1 else 0
        assert s_long * (1 if s_star_approved == -1 else 0) == 0

    def test_meta_classifier_fit_optimization(self):
        """Verifica que fit() optimice los pesos del clasificador y converja."""
        classifier = MetaLabelingClassifier()
        # Generar 20 características sintéticas con etiquetas conocidas
        feats = []
        signals = []
        labels = []
        for i in range(20):
            sign = 1 if i % 2 == 0 else -1
            lbl = 1 if (i >= 10) else 0
            f = MicroFeatures(
                sigma=10.0 + i,
                z_ofi=1.5 if lbl == 1 else -1.5,
                delta_vwap=5.0 if lbl == 1 else -5.0,
                rvol=1.5,
                range_compression=0.5,
                tau_tod=0.5
            )
            feats.append(f)
            signals.append(sign)
            labels.append(lbl)

        old_w = classifier.weights.to_vector().copy()
        classifier.fit(feats, signals, labels, l2_reg=0.01)
        new_w = classifier.weights.to_vector()
        # Los pesos deben actualizarse respecto al valor inicial
        assert not np.allclose(old_w, new_w)


# ==============================================================================
# 3. PRUEBAS DE MODULACIÓN DE APUESTA (BET SIZING - SECCIÓN 3.4)
# ==============================================================================

class TestBetSizingFormulations:
    """Verifica las ecuaciones continuas m(p_t) Gaussiana y polinómica."""

    def test_gaussian_bet_sizing_boundary_and_monotonicity(self):
        """m(p_t) = 0 si p < tau_meta, y estrictamente creciente en p_t >= tau_meta."""
        tau_meta = 0.60

        # Por debajo del umbral de veto debe ser exactamente 0.0
        assert calculate_bet_size(0.50, tau_meta=tau_meta, method="gaussian") == 0.0
        assert calculate_bet_size(0.599, tau_meta=tau_meta, method="gaussian") == 0.0

        # En y por encima del umbral
        m_60 = calculate_bet_size(0.60, tau_meta=tau_meta, method="gaussian")
        m_70 = calculate_bet_size(0.70, tau_meta=tau_meta, method="gaussian")
        m_85 = calculate_bet_size(0.85, tau_meta=tau_meta, method="gaussian")
        m_99 = calculate_bet_size(0.99, tau_meta=tau_meta, method="gaussian")

        assert 0.0 < m_60 < m_70 < m_85 < m_99 <= 1.0

    def test_polynomial_bet_sizing(self):
        """Formulación polinómica cóncava con exponente eta = 1.50."""
        tau_meta = 0.60
        assert calculate_bet_size(0.55, tau_meta=tau_meta, method="polynomial") == 0.0
        assert calculate_bet_size(0.60, tau_meta=tau_meta, method="polynomial") == 0.0

        m_80 = calculate_bet_size(0.80, tau_meta=tau_meta, method="polynomial", eta=1.50)
        expected = ((0.80 - 0.60) / (1.0 - 0.60)) ** 1.50
        assert pytest.approx(m_80, abs=1e-5) == expected

        m_100 = calculate_bet_size(1.00, tau_meta=tau_meta, method="polynomial")
        assert pytest.approx(m_100, abs=1e-5) == 1.0


# ==============================================================================
# 4. PRUEBAS DE LA GEOMETRÍA ANALÍTICA DE TRIPLE BARRERA (SECCIÓN 4)
# ==============================================================================

class TestTripleBarrierGeometry:
    """Verifica las barreras B_upper, B_lower, asimetría 2:1 y detección de first stopping time."""

    def test_barrier_levels_and_asymmetry(self):
        """Asimetría matemática estricta: ell^+ / ell^- = 2.0 / 1.0 = 2.0 >= 1.50."""
        p0 = 5000.0
        sigma0 = 10.0
        ell_plus = 2.00
        ell_minus = 1.00

        # Para Long (+1)
        b_up_long, b_low_long = calculate_barriers(p0, sigma0, 1, ell_plus, ell_minus)
        assert b_up_long == 5000.0 + 2.0 * 10.0 == 5020.0  # Take Profit
        assert b_low_long == 5000.0 - 1.0 * 10.0 == 4990.0  # Stop Loss
        assert (b_up_long - p0) / (p0 - b_low_long) == 2.0

        # Para Short (-1)
        b_up_short, b_low_short = calculate_barriers(p0, sigma0, -1, ell_plus, ell_minus)
        assert b_up_short == 5000.0 + 1.0 * 10.0 == 5010.0  # Stop Loss
        assert b_low_short == 5000.0 - 2.0 * 10.0 == 4980.0  # Take Profit
        assert (p0 - b_low_short) / (b_up_short - p0) == 2.0

    def test_exit_classification_long(self):
        """Clasificación de salida para Long: TP, SL y TIME."""
        b_upper = 5020.0
        b_lower = 4990.0

        # 1. Take Profit alcanzado
        exit_tp = evaluate_triple_barrier_exit(
            signal_star=1, current_high=5022.0, current_low=5005.0, current_close=5015.0,
            b_upper=b_upper, b_lower=b_lower, is_vertical_reached=False
        )
        assert exit_tp is not None
        assert exit_tp[0] == ExitReason.TAKE_PROFIT
        assert exit_tp[1] == b_upper

        # 2. Stop Loss alcanzado
        exit_sl = evaluate_triple_barrier_exit(
            signal_star=1, current_high=5005.0, current_low=4988.0, current_close=4995.0,
            b_upper=b_upper, b_lower=b_lower, is_vertical_reached=False
        )
        assert exit_sl is not None
        assert exit_sl[0] == ExitReason.STOP_LOSS
        assert exit_sl[1] == b_lower

        # 3. Doble toque intrabarra (evento extremo) -> Prudencia manda Stop Loss
        exit_dual = evaluate_triple_barrier_exit(
            signal_star=1, current_high=5025.0, current_low=4985.0, current_close=5000.0,
            b_upper=b_upper, b_lower=b_lower, is_vertical_reached=False
        )
        assert exit_dual is not None
        assert exit_dual[0] == ExitReason.STOP_LOSS

        # 4. Expiración de Barrera Vertical sin tocar TP ni SL
        exit_time = evaluate_triple_barrier_exit(
            signal_star=1, current_high=5010.0, current_low=5000.0, current_close=5008.0,
            b_upper=b_upper, b_lower=b_lower, is_vertical_reached=True
        )
        assert exit_time is not None
        assert exit_time[0] == ExitReason.TIME_BARRIER
        assert exit_time[1] == 5008.0

    def test_exit_classification_short(self):
        """Clasificación de salida para Short: TP (Low <= B_lower), SL (High >= B_upper)."""
        b_upper = 5010.0  # SL
        b_lower = 4980.0  # TP

        # 1. Take Profit Short
        exit_tp = evaluate_triple_barrier_exit(
            signal_star=-1, current_high=4995.0, current_low=4978.0, current_close=4982.0,
            b_upper=b_upper, b_lower=b_lower, is_vertical_reached=False
        )
        assert exit_tp is not None
        assert exit_tp[0] == ExitReason.TAKE_PROFIT
        assert exit_tp[1] == b_lower

        # 2. Stop Loss Short
        exit_sl = evaluate_triple_barrier_exit(
            signal_star=-1, current_high=5012.0, current_low=5002.0, current_close=5005.0,
            b_upper=b_upper, b_lower=b_lower, is_vertical_reached=False
        )
        assert exit_sl is not None
        assert exit_sl[0] == ExitReason.STOP_LOSS
        assert exit_sl[1] == b_upper

    def test_exit_classification_with_gap_open_long(self):
        """Verifica que aperturas con gap fuera de barreras ejecuten al Open y no a precios fantasma."""
        b_upper = 5020.0
        b_lower = 4990.0

        # Gap bajista abre por debajo del stop loss (4980 < 4990)
        exit_gap_sl = evaluate_triple_barrier_exit(
            signal_star=1, current_high=4985.0, current_low=4975.0, current_close=4980.0,
            b_upper=b_upper, b_lower=b_lower, is_vertical_reached=False, current_open=4980.0
        )
        assert exit_gap_sl is not None
        assert exit_gap_sl[0] == ExitReason.STOP_LOSS
        assert exit_gap_sl[1] == 4980.0  # Fills at open with gap slippage, NOT at 4990

        # Gap alcista abre por encima del take profit (5030 > 5020)
        exit_gap_tp = evaluate_triple_barrier_exit(
            signal_star=1, current_high=5035.0, current_low=5025.0, current_close=5030.0,
            b_upper=b_upper, b_lower=b_lower, is_vertical_reached=False, current_open=5030.0
        )
        assert exit_gap_tp is not None
        assert exit_gap_tp[0] == ExitReason.TAKE_PROFIT
        assert exit_gap_tp[1] == 5030.0

    def test_exit_classification_with_gap_open_short(self):
        """Verifica que aperturas con gap para cortos ejecuten realistamente al Open."""
        b_upper = 5010.0  # SL
        b_lower = 4980.0  # TP

        # Gap alcista abre por encima del stop loss (5020 > 5010)
        exit_gap_sl = evaluate_triple_barrier_exit(
            signal_star=-1, current_high=5025.0, current_low=5015.0, current_close=5022.0,
            b_upper=b_upper, b_lower=b_lower, is_vertical_reached=False, current_open=5020.0
        )
        assert exit_gap_sl is not None
        assert exit_gap_sl[0] == ExitReason.STOP_LOSS
        assert exit_gap_sl[1] == 5020.0  # Real gap execution

        # Gap bajista abre por debajo del take profit (4970 < 4980)
        exit_gap_tp = evaluate_triple_barrier_exit(
            signal_star=-1, current_high=4975.0, current_low=4965.0, current_close=4970.0,
            b_upper=b_upper, b_lower=b_lower, is_vertical_reached=False, current_open=4970.0
        )
        assert exit_gap_tp is not None
        assert exit_gap_tp[0] == ExitReason.TAKE_PROFIT
        assert exit_gap_tp[1] == 4970.0

    def test_label_historical_events_and_eod_cutoff(self):
        """Verifica que el etiquetador de triple barrera genere etiquetas válidas {0, 1} y corte al fin de sesión."""
        dts = pd.date_range("2026-10-15 09:30", periods=20, freq="5min")
        df = pd.DataFrame({
            "Open": [5000.0 + i for i in range(20)],
            "High": [5005.0 + i for i in range(20)],
            "Low": [4995.0 + i for i in range(20)],
            "Close": [5002.0 + i for i in range(20)],
            "Volume": [1000.0] * 20
        }, index=dts)
        sigs = pd.Series([1, 0, -1, 0, 1] + [0] * 15, index=dts)
        sigmas = pd.Series([5.0] * 20, index=dts)

        valid_idx, labels = label_historical_events(
            df, sigs, sigmas, ell_plus=2.0, ell_minus=1.0, delta_tau_bars=4, t_eod_close=time(15, 55)
        )
        assert len(valid_idx) == 3
        assert set(labels).issubset({0, 1})


# ==============================================================================
# 5. PRUEBAS DE CONTROL DE RIESGO PROP TRADING (SECCIÓN 5)
# ==============================================================================

class TestPropRiskManagementAndContractSizing:
    """Verifica límites DLL, MTDD, presupuesto de riesgo y conmutación estándar/micro."""

    def test_risk_budget_calculation(self):
        """B_{risk, t0} = min(gamma_D * Delta DLL, gamma_T * DD_cushion)."""
        mgr = PropRiskManager(
            initial_capital=50000.0,
            dll=1000.0,
            mtdd=2000.0,
            gamma_d=0.20,
            gamma_t=0.08
        )
        # Cuenta intacta: Delta DLL = 1000, Cushion = 2000
        # gamma_D * 1000 = 200, gamma_T * 2000 = 160 -> min = 160
        budget = mgr.calculate_risk_budget()
        assert budget == 160.0

    def test_daily_loss_limit_exhaustion_blocks_trading(self):
        """Cuando se consume el DLL, el presupuesto es 0 y se bloquea la operativa."""
        mgr = PropRiskManager(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        # Simular pérdida de $1050 en el día
        trade = TradeRecord(
            trade_id=1, date="2026-10-15", symbol="ES", direction=1, contracts=1, is_micro=False,
            entry_time="09:45:00", entry_price=5000.0, b_upper=5020.0, b_lower=4990.0,
            t_vertical="10:15:00", meta_prob=0.75, bet_size=0.8,
            exit_time="10:00:00", exit_price=4978.0, exit_reason=ExitReason.STOP_LOSS,
            net_pnl_usd=-1050.0
        )
        mgr.record_completed_trade(trade)
        assert mgr.is_halted is True
        assert "DAILY_LOSS_LIMIT_BREACH" in mgr.halt_reason
        assert mgr.calculate_risk_budget() == 0.0

    def test_circuit_breaker_trailing_drawdown_trigger(self):
        """Disyuntor analítico: Cushion <= 0.15 * MTDD => HALT_DRAWDOWN_DEFENSE."""
        mgr = PropRiskManager(initial_capital=50000.0, dll=2000.0, mtdd=2000.0, circuit_breaker_ratio=0.15)
        # 15% de 2000 = 300. Si la pérdida acumulada es de $1750, el cushion es 2000 - 1750 = 250 <= 300
        mgr.equity = 50000.0 - 1750.0
        is_triggered = mgr.check_circuit_breaker()
        assert is_triggered is True
        assert mgr.is_halted is True
        assert "HALT_DRAWDOWN_DEFENSE" in mgr.halt_reason

    def test_automatic_switching_to_micro_contracts(self):
        """
        Si el riesgo unitario estándar delta_L_std excede el presupuesto efectivo
        (N_std = 0), debe conmutar automáticamente a microcontratos MES.
        """
        mgr = PropRiskManager(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        # Supongamos presupuesto de 160 USD y volatilidad de 5 puntos en ES
        # Riesgo unitario ES: 1.0 * 5.0 * 50 = 250 USD + fricción ~ 260 USD > 160 USD -> N_std = 0
        # Riesgo unitario MES: 1.0 * 5.0 * 5 = 25 USD + fricción ~ 28 USD -> N_micro = floor(160 / 28) ~ 5
        n_contracts, is_micro, unit_risk = mgr.calculate_position_size(
            std_symbol="ES",
            micro_symbol="MES",
            sigma=5.0,
            ell_minus=1.0,
            bet_size=1.0,
            slippage_ticks=1.0
        )
        assert is_micro is True
        assert n_contracts >= 1
        assert n_contracts * unit_risk <= 160.0

    def test_extreme_volatility_blocks_trading(self):
        """Volatilidad extrema (ej. sigma = 60 puntos) debe resultar en Bloqueo Defensivo (0 contratos)."""
        mgr = PropRiskManager(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        n_contracts, is_micro, unit_risk = mgr.calculate_position_size(
            std_symbol="ES",
            micro_symbol="MES",
            sigma=60.0,
            ell_minus=1.0,
            bet_size=1.0,
            slippage_ticks=1.0
        )
        assert n_contracts == 0
        assert is_micro is False

    def test_daily_loss_accounting_with_profits(self):
        """Verifica que ganancias intradiarias compensen pérdidas y no se detenga erróneamente."""
        mgr = PropRiskManager(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        # Trade 1: Ganancia de +$800
        t1 = TradeRecord(
            trade_id=1, date="2026-10-15", symbol="MES", direction=1, contracts=5, is_micro=True,
            entry_time="09:45:00", entry_price=5000.0, b_upper=5020.0, b_lower=4990.0,
            t_vertical="10:15:00", meta_prob=0.8, bet_size=0.8, net_pnl_usd=800.0
        )
        mgr.record_completed_trade(t1)
        assert mgr.equity == 50800.0
        assert mgr.daily_realized_loss == 0.0

        # Trade 2: Pérdida de -$600
        t2 = TradeRecord(
            trade_id=2, date="2026-10-15", symbol="MES", direction=1, contracts=5, is_micro=True,
            entry_time="10:30:00", entry_price=5010.0, b_upper=5030.0, b_lower=5000.0,
            t_vertical="11:00:00", meta_prob=0.8, bet_size=0.8, net_pnl_usd=-600.0
        )
        mgr.record_completed_trade(t2)
        # Equidad actual 50200 > 50000 inicial del día: Pérdida neta del día = 0!
        assert mgr.equity == 50200.0
        assert mgr.daily_realized_loss == 0.0
        assert mgr.is_halted is False
        assert mgr.delta_dll == 1000.0


# ==============================================================================
# 6. PRUEBA INTEGRAL E2E CON DATOS HISTÓRICOS DE CME GLOBEX
# ==============================================================================

class TestEndToEndMetaLabelingBacktest:
    """Verifica la ejecución determinista del motor de backtest barra a barra."""

    def test_backtest_execution_on_es_dataset(self):
        """Ejecuta backtest sobre los datos CME_MINI_ES_5m_bars5000.csv."""
        csv_path = Path("data/raw/CME_MINI_ES_5m_bars5000.csv")
        if not csv_path.exists():
            pytest.skip("Dataset de ES no encontrado.")

        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        strategy = MetaLabelingTripleBarrierStrategy(params={
            "symbol": "ES",
            "std_symbol": "ES",
            "micro_symbol": "MES",
            "train_classifier": False,
            "tau_meta": 0.60
        })
        backtester = MetaLabelingBacktester(strategy=strategy)
        results = backtester.run(df)

        assert results["total_trades"] > 0
        assert "win_rate" in results
        assert "net_profit" in results
        assert "max_drawdown_usd" in results
        assert results["max_drawdown_usd"] <= 2000.0  # MTDD respetado
        assert len(results["trades"]) == results["total_trades"]

    def test_backtest_execution_on_nq_dataset(self):
        """Ejecuta backtest sobre los datos CME_MINI_NQ_5m_bars5000.csv."""
        csv_path = Path("data/raw/CME_MINI_NQ_5m_bars5000.csv")
        if not csv_path.exists():
            pytest.skip("Dataset de NQ no encontrado.")

        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        strategy = MetaLabelingTripleBarrierStrategy(params={
            "symbol": "NQ",
            "std_symbol": "NQ",
            "micro_symbol": "MNQ",
            "train_classifier": False,
            "tau_meta": 0.60
        })
        backtester = MetaLabelingBacktester(strategy=strategy)
        results = backtester.run(df)

        assert results["total_trades"] > 0
        # NQ conmuta a micros por volatilidad
        assert results["micro_trades"] > 0
        assert results["max_drawdown_usd"] <= 2000.0  # MTDD respetado

    def test_base_strategy_signals_generation(self, sample_intraday_bars):
        """generate_signals() de BaseStrategy debe retornar pd.Series con {-1.0, 0.0, 1.0}."""
        strat = MetaLabelingTripleBarrierStrategy(params={"train_classifier": False})
        signals = strat.generate_signals(sample_intraday_bars)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_intraday_bars)
        unique_vals = set(signals.unique())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_backtest_with_train_classifier_on_es(self):
        """Verifica que el backtest con entrenamiento activo (train_classifier=True) ejecute correctamente."""
        csv_path = Path("data/raw/CME_MINI_ES_5m_bars5000.csv")
        if not csv_path.exists():
            pytest.skip("Dataset de ES no encontrado.")

        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        strategy = MetaLabelingTripleBarrierStrategy(params={
            "symbol": "ES",
            "std_symbol": "ES",
            "micro_symbol": "MES",
            "train_classifier": True,
            "tau_meta": 0.60
        })
        backtester = MetaLabelingBacktester(strategy=strategy)
        results = backtester.run(df)

        assert results["total_trades"] > 0
        assert "win_rate" in results
        assert "net_profit" in results
        assert results["max_drawdown_usd"] <= 2000.0

    def test_label_historical_events_with_purging(self, sample_intraday_bars):
        """Verifica que el parámetro purge_last_bars excluya estrictamente las últimas barras del etiquetado."""
        df = sample_intraday_bars
        n = len(df)
        signals = pd.Series([1.0] * n, index=df.index)
        sigmas = pd.Series([10.0] * n, index=df.index)

        # Sin purgado
        val_idx_no_purge, _ = label_historical_events(
            df, signals, sigmas, purge_last_bars=0
        )
        # Con purgado de 10 barras
        purge_k = 10
        val_idx_purged, _ = label_historical_events(
            df, signals, sigmas, purge_last_bars=purge_k
        )

        assert max(val_idx_purged) < (n - 1 - purge_k)
        assert len(val_idx_purged) < len(val_idx_no_purge)

    def test_backtester_fit_classifier_on_train_df_and_eval_on_oos(self):
        """Verifica la separación estricta: fit_classifier_on_data en train_df y run en eval_df."""
        csv_path = Path("data/raw/CME_MINI_ES_5m_bars5000.csv")
        if not csv_path.exists():
            pytest.skip("Dataset de ES no encontrado.")

        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        # 70% In-Sample / 30% Out-of-Sample
        split_idx = int(len(df) * 0.70)
        train_df = df.iloc[:split_idx]
        eval_df = df.iloc[split_idx:]

        strategy = MetaLabelingTripleBarrierStrategy(params={
            "symbol": "ES",
            "std_symbol": "ES",
            "micro_symbol": "MES",
            "train_classifier": False,  # No reentrenar en eval_df
            "tau_meta": 0.60
        })
        backtester = MetaLabelingBacktester(strategy=strategy)

        # Entrenar en In-Sample
        fit_info = backtester.fit_classifier_on_data(train_df, purge_boundary=True)
        assert fit_info["total_bars_train"] == len(train_df)
        assert fit_info["purged_bars"] > 0
        assert fit_info["events_labeled"] > 0

        # Guardar pesos congelados
        frozen_weights = list(backtester.strategy.classifier.weights.to_vector())

        # Evaluar en OOS
        results_oos = backtester.run(eval_df)
        assert results_oos["total_trades"] >= 0
        assert results_oos["max_drawdown_usd"] <= 2000.0

        # Verificar que los pesos se mantuvieron estrictamente idénticos (congelados)
        current_weights = list(backtester.strategy.classifier.weights.to_vector())
        assert np.allclose(frozen_weights, current_weights)

    def test_data_split_manager_integration_in_meta_labeling(self):
        """Verifica la integración formal con DataSplitManager para partición de 3 universos."""
        from src.core.data_split import DataSplitManager

        csv_path = Path("data/raw/CME_MINI_ES_5m_bars5000.csv")
        if not csv_path.exists():
            pytest.skip("Dataset de ES no encontrado.")

        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        split_mgr = DataSplitManager(df, split_ratio=0.70)
        is_df = split_mgr.get_in_sample()
        oos_df = split_mgr.get_out_of_sample(authorized=True, reason="Test unitario OOS")
        holdout_df = split_mgr.get_holdout(authorized=True, reason="Test unitario Holdout")

        # Verificar ausencia total de intersección de índices
        assert len(is_df.index.intersection(oos_df.index)) == 0
        assert len(oos_df.index.intersection(holdout_df.index)) == 0
        assert is_df.index.max() < oos_df.index.min()
        assert oos_df.index.max() < holdout_df.index.min()

        # Concatenar bloque OOS completo (30%)
        eval_oos = pd.concat([oos_df, holdout_df]).sort_index()
        assert len(is_df) + len(eval_oos) == len(df)
