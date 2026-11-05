"""
tests/test_multi_level_ofi.py
Suite exhaustiva de pruebas unitarias y de estrés para la Estrategia F6:
Multi-Level Order Flow Imbalance en LOB Profundo (Xu, Gould & Howison, 2019).
"""

import math
import numpy as np
import pandas as pd
import pytest
from datetime import time, datetime, timedelta

from src.strategies.multi_level_ofi import (
    MultiLevelOFIStrategy,
    MultiLevelOFIBacktester,
    PropAccount,
    StrategyState,
    ExitReason,
    TradeRecord,
    calculate_bid_flow,
    calculate_ask_flow,
    calculate_unit_imbalance,
    calculate_spatial_betas,
    compute_pca_spectral,
    calculate_pc1,
    calculate_coherence_index,
    calculate_m_ofi,
    calculate_z_score,
    evaluate_entry_signal,
    calculate_absorption_limit,
    calculate_position_size,
)
from src.core.futures_specs import get_spec
from scripts.run_multi_level_ofi_backtest import generate_synthetic_multilevel_lob


# ==============================================================================
# 1. PRUEBAS DE EVENTOS DE LOB Y DESBALANCE UNITARIO (SECCIÓN 3.1)
# ==============================================================================

class TestLOBEventDynamics:
    """Verifica las ecuaciones exactas de variación de inventario por nivel."""

    def test_bid_flow_price_increase(self):
        """P_n^b > P_{n-1}^b -> e_n^b = +q_n^b"""
        flow = calculate_bid_flow(p_curr=5000.25, q_curr=150.0, p_prev=5000.00, q_prev=100.0)
        assert flow == 150.0

    def test_bid_flow_price_equal(self):
        """P_n^b == P_{n-1}^b -> e_n^b = q_n^b - q_{n-1}^b"""
        flow_up = calculate_bid_flow(p_curr=5000.00, q_curr=180.0, p_prev=5000.00, q_prev=120.0)
        assert flow_up == 60.0

        flow_down = calculate_bid_flow(p_curr=5000.00, q_curr=90.0, p_prev=5000.00, q_prev=120.0)
        assert flow_down == -30.0

    def test_bid_flow_price_decrease(self):
        """P_n^b < P_{n-1}^b -> e_n^b = -q_{n-1}^b"""
        flow = calculate_bid_flow(p_curr=4999.75, q_curr=80.0, p_prev=5000.00, q_prev=110.0)
        assert flow == -110.0

    def test_ask_flow_price_increase(self):
        """P_n^a > P_{n-1}^a -> e_n^a = -q_{n-1}^a"""
        flow = calculate_ask_flow(p_curr=5000.50, q_curr=140.0, p_prev=5000.25, q_prev=95.0)
        assert flow == -95.0

    def test_ask_flow_price_equal(self):
        """P_n^a == P_{n-1}^a -> e_n^a = q_n^a - q_{n-1}^a"""
        flow_up = calculate_ask_flow(p_curr=5000.25, q_curr=210.0, p_prev=5000.25, q_prev=150.0)
        assert flow_up == 60.0

        flow_down = calculate_ask_flow(p_curr=5000.25, q_curr=100.0, p_prev=5000.25, q_prev=150.0)
        assert flow_down == -50.0

    def test_ask_flow_price_decrease(self):
        """P_n^a < P_{n-1}^a -> e_n^a = -q_n^a"""
        flow = calculate_ask_flow(p_curr=5000.00, q_curr=130.0, p_prev=5000.25, q_prev=100.0)
        assert flow == -130.0

    def test_unit_imbalance(self):
        """e_n^{(l)} = e_n^{b, (l)} - e_n^{a, (l)}"""
        eb = 150.0
        ea = -95.0
        en = calculate_unit_imbalance(eb, ea)
        assert en == 245.0


# ==============================================================================
# 2. PRUEBAS DE DECAIMIENTO ESPACIAL DE POTENCIAS (SECCIÓN 3.3)
# ==============================================================================

class TestSpatialPowerLawDecay:
    """Verifica la jerarquía monotónica decreciente y decaimiento potencial."""

    def test_beta_monotonicity(self):
        betas = calculate_spatial_betas(levels=5, alpha_decay=1.0, d_bar_1=250.0)
        assert len(betas) == 5
        assert betas[0] == pytest.approx(1.0 / 500.0)
        for i in range(len(betas) - 1):
            assert betas[i] > betas[i + 1]
            assert betas[i] > 0.0

    def test_beta_scaling_with_alpha(self):
        betas_alpha_1 = calculate_spatial_betas(levels=3, alpha_decay=1.0, beta_1=0.01)
        # beta_1 = 0.01, beta_2 = 0.01 * 2^-1 = 0.005, beta_3 = 0.01 * 3^-1 = 0.003333
        assert pytest.approx(betas_alpha_1[0]) == 0.01
        assert pytest.approx(betas_alpha_1[1]) == 0.005
        assert pytest.approx(betas_alpha_1[2]) == 0.01 / 3.0

        betas_alpha_2 = calculate_spatial_betas(levels=3, alpha_decay=2.0, beta_1=0.01)
        assert pytest.approx(betas_alpha_2[1]) == 0.01 / 4.0

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            calculate_spatial_betas(levels=0)
        with pytest.raises(ValueError):
            calculate_spatial_betas(alpha_decay=-0.5)
        with pytest.raises(ValueError):
            calculate_spatial_betas(d_bar_1=0)


# ==============================================================================
# 3. PRUEBAS DE DESCOMPOSICIÓN ESPECTRAL PCA (SECCIÓN 3.4)
# ==============================================================================

class TestPCASpectralDecomposition:
    """Verifica descomposición espectral de covarianza de OFI y teorema Perron-Frobenius."""

    def test_eigenvalues_and_dominant_vector(self):
        np.random.seed(42)
        # Matriz correlacionada de 5 niveles
        corr = np.array([
            [1.00, 0.75, 0.60, 0.50, 0.40],
            [0.75, 1.00, 0.78, 0.65, 0.55],
            [0.60, 0.78, 1.00, 0.80, 0.70],
            [0.50, 0.65, 0.80, 1.00, 0.82],
            [0.40, 0.55, 0.70, 0.82, 1.00]
        ])
        stds = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
        cov = np.outer(stds, stds) * corr

        # Generar muestras sintéticas a partir de esta covarianza
        samples = np.random.multivariate_normal(mean=[0]*5, cov=cov, size=200)

        eigvals, eigvecs, v1 = compute_pca_spectral(samples)

        # Autovalores ordenados descendentemente y no negativos
        assert eigvals[0] >= eigvals[1] >= eigvals[2] >= eigvals[3] >= eigvals[4]
        assert np.all(eigvals >= -1e-6)

        # Varianza explicada por PC1 > 60%
        var_ratio_pc1 = eigvals[0] / np.sum(eigvals)
        assert var_ratio_pc1 > 0.60

        # Propiedad de Perron-Frobenius: autovector dominante orientado positivamente
        assert np.all(v1 > 0.0)

    def test_calculate_pc1(self):
        """Verifica la proyección PC1_k = v_1^T * OFI_k."""
        ofi_vec = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
        v1 = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        pc1 = calculate_pc1(ofi_vec, v1)
        expected = (100*0.5 + 80*0.4 + 60*0.3 + 40*0.2 + 20*0.1)
        assert pc1 == pytest.approx(expected)

    def test_strategy_pca_weighting_mode(self):
        """Verifica que MultiLevelOFIStrategy genere señales usando ponderación PCA."""
        df = generate_synthetic_multilevel_lob(symbol="ES", n_bars=100, levels=5, seed=42)
        strat = MultiLevelOFIStrategy(params={"symbol": "ES", "levels": 5, "weighting_mode": "pca"})
        signals = strat.generate_signals(df)
        assert len(signals) == len(df)
        assert set(signals.unique()).issubset({-1.0, 0.0, 1.0})



# ==============================================================================
# 4. PRUEBAS DE COHERENCIA ANTI-SPOOFING (SECCIÓN 3.5)
# ==============================================================================

class TestCoherenceIndexAntiSpoofing:
    """Verifica el cálculo de Phi_coherence y detección de trampas de liquidez."""

    def test_genuine_buying_accumulation(self):
        betas = np.array([0.02, 0.01, 0.0067, 0.005, 0.004])
        # Nivel 1 comprador (+200), niveles 2 a 5 fuertemente compradores
        ofi_vec = np.array([200.0, 180.0, 150.0, 120.0, 90.0])
        phi = calculate_coherence_index(ofi_vec, betas)
        assert phi > 0.40
        assert phi <= 1.0

    def test_genuine_selling_distribution(self):
        betas = np.array([0.02, 0.01, 0.0067, 0.005, 0.004])
        # Nivel 1 vendedor (-200), niveles 2 a 5 vendedores (-180, ...)
        ofi_vec = np.array([-200.0, -180.0, -150.0, -120.0, -90.0])
        phi = calculate_coherence_index(ofi_vec, betas)
        assert phi > 0.40  # Ambos lados son coherentes en venta!
        assert phi <= 1.0

    def test_spoofing_trap_detection(self):
        betas = np.array([0.02, 0.01, 0.0067, 0.005, 0.004])
        # Nivel 1 muestra +200 compra aparente, pero niveles 2..5 venden masivamente
        ofi_vec = np.array([200.0, -180.0, -150.0, -120.0, -90.0])
        phi = calculate_coherence_index(ofi_vec, betas)
        assert phi < -0.40  # Alerta de spoofing

    def test_deserted_deep_book_edge_case(self):
        betas = np.array([0.02, 0.01, 0.0067, 0.005, 0.004])
        ofi_vec = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
        phi = calculate_coherence_index(ofi_vec, betas)
        assert pytest.approx(phi, abs=1e-5) == 0.0
        assert not math.isnan(phi)


# ==============================================================================
# 5. PRUEBAS DE SEÑALES CERRADAS Y Z-SCORE (SECCIONES 3.6 Y 3.7)
# ==============================================================================

class TestSignalsAndZScore:
    """Verifica el cálculo de M-OFI, normalización Z y decisiones de entrada."""

    def test_m_ofi_and_z_score(self):
        betas = np.array([0.02, 0.01, 0.005])
        ofi_vec = np.array([100.0, 50.0, 20.0])
        m_ofi = calculate_m_ofi(ofi_vec, betas)
        # 0.02*100 + 0.01*50 + 0.005*20 = 2.0 + 0.5 + 0.1 = 2.6
        assert pytest.approx(m_ofi) == 2.6

        # Historial de 30 valores con media 0 y std 1
        window = [float(x) for x in range(-15, 15)]
        z = calculate_z_score(15.0, window)
        assert z > 1.5

    def test_entry_decision_rules(self):
        # Long válido
        sig_long = evaluate_entry_signal(
            z_m_ofi=2.50, phi_coherence=0.60, spread=0.25, tick_size=0.25, theta_entry=2.25, phi_min=0.40
        )
        assert sig_long == 1

        # Short válido
        sig_short = evaluate_entry_signal(
            z_m_ofi=-2.50, phi_coherence=0.60, spread=0.25, tick_size=0.25, theta_entry=2.25, phi_min=0.40
        )
        assert sig_short == -1

        # Veto por spread ancho (> 1 tick)
        sig_wide = evaluate_entry_signal(
            z_m_ofi=2.50, phi_coherence=0.60, spread=0.50, tick_size=0.25
        )
        assert sig_wide == 0

        # Veto por coherencia insuficiente
        sig_no_coh = evaluate_entry_signal(
            z_m_ofi=2.50, phi_coherence=0.20, spread=0.25, tick_size=0.25
        )
        assert sig_no_coh == 0

        # Veto por trampa de spoofing
        sig_spoof = evaluate_entry_signal(
            z_m_ofi=2.50, phi_coherence=-0.50, spread=0.25, tick_size=0.25
        )
        assert sig_spoof == 0


# ==============================================================================
# 6. PRUEBAS DE RIESGO PROP TRADING Y CONMUTACIÓN ESTÁNDAR/MICRO (SECCIÓN 5)
# ==============================================================================

class TestPropTradingRiskManagement:
    """Verifica reglas de capital, absorción en profundidad y conmutación Standard/Micro."""

    def test_depth_absorption_limit(self):
        bids = [200.0, 300.0, 250.0]
        asks = [220.0, 280.0, 250.0]
        # Promedio: nivel 1 = 210, nivel 2 = 290, nivel 3 = 250 -> Total = 750
        # Gamma_depth = 0.08 -> 750 * 0.08 = 60 contratos
        limit = calculate_absorption_limit(bids, asks, levels=3, gamma_depth=0.08)
        assert limit == 60

    def test_prop_account_risk_budget(self):
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        # Presupuesto inicial: min(1000 * 0.25, 2000 * 0.15) = min(250, 300) = 250
        budget = account.calculate_risk_budget(phi_dll=0.25, phi_tmdd=0.15)
        assert budget == 250.0

    def test_switch_standard_to_micro(self):
        """Si el riesgo del contrato estándar supera el presupuesto, debe conmutar a micro."""
        # En ES: SL = 1.5 pt -> 1.5 * 50 = $75 + comision ($4.5) + slippage ($12.5) = $92.0 por contrato
        # Si presupuesto = $60, no alcanza para 1 ES -> conmuta a MES (riesgo aprox $9.20 por contrato)
        budget = 60.0
        contracts, is_micro, status = calculate_position_size(
            risk_budget_usd=budget,
            symbol="ES",
            stop_loss_points=1.50,
            depth_limit_contracts=10,
            max_contracts_param=5
        )
        assert is_micro is True
        assert contracts >= 1
        assert status == "SWITCHED_TO_MICRO"

    def test_halt_when_micro_cannot_fit(self):
        """Si ni siquiera un micro contrato entra en el presupuesto, debe detener preventivamente."""
        budget = 5.0  # $5 no cubre ni un micro contrato MES (~$9.2)
        contracts, is_micro, status = calculate_position_size(
            risk_budget_usd=budget,
            symbol="ES",
            stop_loss_points=1.50
        )
        assert contracts == 0
        assert status == "HALT_VOLATILITY_EXCEEDS_BUDGET"

    def test_volatility_scaling_protective_reduction(self):
        """Si la volatilidad se duplica, los contratos se reducen a la mitad."""
        budget = 1000.0
        c_norm, _, _ = calculate_position_size(
            risk_budget_usd=budget, symbol="ES", har_volatility=0.010, ref_volatility=0.010, max_contracts_param=20
        )
        c_high_vol, _, _ = calculate_position_size(
            risk_budget_usd=budget, symbol="ES", har_volatility=0.020, ref_volatility=0.010, max_contracts_param=20
        )
        assert c_high_vol <= math.ceil(c_norm / 2)

    def test_prop_account_breach_detection(self):
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        account.start_session("2026-10-19")
        assert not account.is_breached

        # Registrar pérdida diaria de $1,050
        account.update_trade(-1050.0)
        assert account.is_breached
        assert "DAILY_LOSS_LIMIT_BREACHED" in account.breach_reason

    def test_prop_account_net_daily_loss(self):
        """Verifica que las ganancias diarias amorticen pérdidas y solo la pérdida neta active el DLL."""
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        account.start_session("2026-10-19")

        # Trade 1 gana +$500
        account.update_trade(500.0)
        assert account.daily_pnl == 500.0
        assert account.daily_realized_loss == 0.0
        assert not account.is_breached

        # Trade 2 pierde -$300 (PnL neto del día sigue positivo: +$200)
        account.update_trade(-300.0)
        assert account.daily_pnl == 200.0
        assert account.daily_realized_loss == 0.0
        assert not account.is_breached

        # Trade 3 pierde -$1300 (PnL neto del día cae a -$1,100)
        account.update_trade(-1300.0)
        assert account.daily_pnl == -1100.0
        assert account.daily_realized_loss == 1100.0
        assert account.is_breached
        assert "DAILY_LOSS_LIMIT_BREACHED" in account.breach_reason


# ==============================================================================
# 7. PRUEBAS DE INTEGRACIÓN: SIMULACIÓN DE BACKTEST CON MOTOR F6
# ==============================================================================

class TestMultiLevelOFIBacktestIntegration:
    """Verifica la ejecución determinista del motor de backtest en datos LOB simulados."""

    def test_backtest_execution_clean_run(self):
        df = generate_synthetic_multilevel_lob(symbol="ES", n_bars=200, levels=5, seed=123)
        strategy = MultiLevelOFIStrategy(params={"symbol": "ES", "levels": 5})
        backtester = MultiLevelOFIBacktester(
            strategy=strategy,
            account_size=50000.0,
            daily_loss_limit=1000.0,
            max_trailing_loss=2000.0,
            symbol="ES"
        )
        results = backtester.run(df)

        metrics = results["metrics"]
        assert "total_trades" in metrics
        assert "net_pnl_usd" in metrics
        assert "prop_status" in metrics
        assert results["status"] in ["PASS", "INCOMPLETE", "FAIL"]
        assert len(results["equity_curve"]) > 0

    def test_backtest_adverse_gap_triggers_prop_failure(self):
        """Verifica que una pérdida masiva que vulnere el límite diario reporte FAIL."""
        # Configurar cuenta con límite ultrarrígido
        account = PropAccount(initial_capital=50000.0, dll=50.0, mtdd=100.0)
        account.update_trade(-75.0)
        assert account.is_breached
        assert "DAILY_LOSS_LIMIT_BREACHED" in account.breach_reason

        # Crear dataset donde se garantice la apertura de un trade LONG en barra 31
        df_adverse = generate_synthetic_multilevel_lob(symbol="ES", n_bars=60, levels=5, seed=42)
        df_adverse.loc[df_adverse.index[31], "ofi_1"] = 600.0
        for l in range(2, 6):
            df_adverse.loc[df_adverse.index[31], f"ofi_{l}"] = 300.0

        # En la barra 32, gap bajista extremo que provoca pérdida de 1.5 pt (en ES: 1.5 * 50 = $75 > DLL $50)
        entry_p = df_adverse.loc[df_adverse.index[31], "Close"]
        df_adverse.loc[df_adverse.index[32], "Low"] = entry_p - 10.0
        df_adverse.loc[df_adverse.index[32], "Close"] = entry_p - 10.0

        backtester = MultiLevelOFIBacktester(
            account_size=50000.0,
            daily_loss_limit=50.0,
            max_trailing_loss=100.0,
            symbol="ES"
        )
        results = backtester.run(df_adverse)
        assert results["account"].is_breached is True
        assert results["status"] == "FAIL"

    def test_exit_hierarchy_z_reversal_takes_precedence_over_z_close(self):
        """
        Verifica que ante una inversión violenta contraria del desbalance (Z <= -theta_reverse),
        la salida se registre como STOP_LOSS_Z_REVERSE y NO como TAKE_PROFIT_Z_CLOSE,
        resolviendo el bug donde Z <= 0.60 capturaba indebidamente trades perdedores.
        """
        # Crear 40 barras sintéticas
        df = generate_synthetic_multilevel_lob(symbol="ES", n_bars=40, levels=5, seed=42)
        # Forzar barra 31 como señal de entrada LONG (Z > 2.25, Phi > 0.40)
        df.loc[df.index[31], "ofi_1"] = 500.0
        for l in range(2, 6):
            df.loc[df.index[31], f"ofi_{l}"] = 300.0

        # En la barra 32, inducir inversión violenta del flujo en contra de los largos (Z < -1.0)
        df.loc[df.index[32], "ofi_1"] = -600.0
        for l in range(2, 6):
            df.loc[df.index[32], f"ofi_{l}"] = -400.0
        # Mantener el precio dentro de los límites de stop para que se pruebe el stop de flujo Z
        df.loc[df.index[32], "Low"] = df.loc[df.index[31], "Close"] - 0.50
        df.loc[df.index[32], "Close"] = df.loc[df.index[31], "Close"] - 0.50

        strategy = MultiLevelOFIStrategy(params={"symbol": "ES", "levels": 5})
        backtester = MultiLevelOFIBacktester(strategy=strategy, symbol="ES")
        results = backtester.run(df)

        trades = results["trades"]
        assert len(trades) >= 1
        trade = trades[0]
        assert trade.direction == 1
        assert trade.exit_reason in [ExitReason.STOP_LOSS_Z_REVERSE, ExitReason.STOP_LOSS_COHERENCE_ABORT]
        assert trade.exit_reason != ExitReason.TAKE_PROFIT_Z_CLOSE

    def test_exit_hierarchy_price_stop_loss_takes_precedence(self):
        """Verifica que el toque del stop loss por precio active STOP_LOSS_PRICE antes de TP."""
        df = generate_synthetic_multilevel_lob(symbol="ES", n_bars=40, levels=5, seed=42)
        # Forzar entrada LONG en barra 31
        df.loc[df.index[31], "ofi_1"] = 500.0
        for l in range(2, 6):
            df.loc[df.index[31], f"ofi_{l}"] = 300.0

        # En la barra 32, el Low perfora el stop loss (más de 6 ticks = 1.5 pt)
        entry_p = df.loc[df.index[31], "Close"]
        df.loc[df.index[32], "Low"] = entry_p - 3.00
        df.loc[df.index[32], "Close"] = entry_p - 2.50

        strategy = MultiLevelOFIStrategy(params={"symbol": "ES", "levels": 5})
        backtester = MultiLevelOFIBacktester(strategy=strategy, symbol="ES")
        results = backtester.run(df)

        trades = results["trades"]
        assert len(trades) >= 1
        trade = trades[0]
        assert trade.exit_reason == ExitReason.STOP_LOSS_PRICE
        assert trade.exit_price == pytest.approx(entry_p - 1.50)

