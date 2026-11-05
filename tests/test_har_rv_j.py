"""
tests/test_har_rv_j.py
Suite de pruebas unitarias, matemáticas y de estrés para la Estrategia F8:
Modelos Econométricos Heterogéneos HAR-RV-J y Dimensionamiento por Volatilidad Dinámica
(Corsi 2009 / Andersen, Bollerslev & Diebold 2007 / Moreira & Muir 2017).

Especificación: reports/estrategias_matematicas/F8_modelos_econometricos_har_rv_j.md
"""

from datetime import time, datetime, timedelta
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.strategies.har_rv_j import (
    EconometricHARRVJStrategy,
    HARRVJBacktester,
    HARStateVector,
    HARExitReason,
    HARStrategyState,
    TradeRecord,
    PropAccount,
    calculate_log_returns,
    calculate_realized_variance,
    calculate_bipower_variation,
    calculate_tripower_quarticity,
    calculate_jump_statistic,
    decompose_realized_variance,
    calculate_heterogeneous_rv,
    fit_log_har_rv_j,
    predict_log_har_rv_j,
    calculate_point_volatility,
    evaluate_jump_gate,
    calculate_dynamic_barriers,
    check_circuit_breaker,
    calculate_position_sizing,
    calculate_volatility_managed_weight,
    quantize_price,
    MU_1,
    MU_1_INV_SQ,
    MU_4_3,
    MU_4_3_INV_CUBED,
    C_V_JUMP,
    Z_ALPHA_JUMP_001,
)
from src.core.futures_specs import get_spec


# ==============================================================================
# 1. PRUEBAS DE FORMULACIÓN MATEMÁTICA PURA (SECCIÓN 3 DE F8)
# ==============================================================================

class TestF8MathematicalFormulations:
    """Verifica las ecuaciones matemáticas exactas formuladas en F8."""

    def test_mathematical_constants(self):
        """Verifica las constantes asintóticas analíticas de Barndorff-Nielsen."""
        expected_mu1 = math.sqrt(2.0 / math.pi)
        assert pytest.approx(MU_1, abs=1e-8) == expected_mu1
        assert pytest.approx(MU_1_INV_SQ, abs=1e-8) == math.pi / 2.0

        # mu_{4/3} = 2^(2/3) * Gamma(7/6) / sqrt(pi)
        expected_mu43 = (2.0 ** (2.0 / 3.0)) * math.gamma(7.0 / 6.0) / math.sqrt(math.pi)
        assert pytest.approx(MU_4_3, abs=1e-8) == expected_mu43
        assert pytest.approx(MU_4_3_INV_CUBED, abs=1e-3) == 1.7433

        # c_v = (pi/2)^2 + pi - 5
        expected_cv = (math.pi / 2.0) ** 2 + math.pi - 5.0
        assert pytest.approx(C_V_JUMP, abs=1e-5) == expected_cv
        assert pytest.approx(C_V_JUMP, abs=1e-3) == 0.6091

        # z_0.001 = 3.09023
        assert pytest.approx(Z_ALPHA_JUMP_001, abs=1e-4) == 3.0902

    def test_log_returns_formula(self):
        """Verifica r_{t, j} = ln(P_{t, j}) - ln(P_{t, j-1})."""
        prices = np.array([5000.0, 5010.0, 5005.0])
        returns = calculate_log_returns(prices)
        assert len(returns) == 2
        assert pytest.approx(returns[0], abs=1e-9) == math.log(5010.0 / 5000.0)
        assert pytest.approx(returns[1], abs=1e-9) == math.log(5005.0 / 5010.0)

        # Precios no positivos deben lanzar ValueError
        with pytest.raises(ValueError):
            calculate_log_returns(np.array([5000.0, 0.0, 5010.0]))
        with pytest.raises(ValueError):
            calculate_log_returns(np.array([5000.0, -100.0]))

        # Vector vacío o de 1 elemento
        assert len(calculate_log_returns(np.array([5000.0]))) == 0
        assert len(calculate_log_returns(np.array([]))) == 0

    def test_realized_variance_calculation(self):
        """Verifica RV_t^{(d)} = sum r_{t, j}^2."""
        returns = np.array([0.001, -0.002, 0.0015])
        expected_rv = 0.001**2 + (-0.002)**2 + 0.0015**2
        calc_rv = calculate_realized_variance(returns)
        assert pytest.approx(calc_rv, abs=1e-12) == expected_rv

        # Retornos vacíos o ceros
        assert calculate_realized_variance(np.array([])) == 0.0
        assert calculate_realized_variance(np.zeros(10)) == 0.0

    def test_bipower_variation_jump_robustness(self):
        """
        Verifica que BV es robusta a la presencia de un salto aislado:
        Un salto grande infla drásticamente RV, mientras que BV se mantiene mucho menor.
        """
        # Difusión regular con retornos pequeños
        np.random.seed(42)
        regular_returns = np.random.normal(0, 0.001, 78)
        rv_regular = calculate_realized_variance(regular_returns)
        bv_regular = calculate_bipower_variation(regular_returns)
        # En difusión pura continua, RV y BV son asintóticamente equivalentes
        assert pytest.approx(rv_regular, rel=0.35) == bv_regular

        # Insertar un salto masivo en un solo intervalo
        jump_returns = regular_returns.copy()
        jump_returns[40] = 0.05  # Salto de 5%
        rv_jump = calculate_realized_variance(jump_returns)
        bv_jump = calculate_bipower_variation(jump_returns)

        # RV debe explotar (salto al cuadrado) mucho más que BV
        assert rv_jump > 3.0 * bv_jump
        assert rv_jump - bv_jump > 0.002

    def test_decomposition_orthogonality(self):
        """
        Verifica la propiedad analítica esencial (Sección 3.1.4):
        C_t > 0, J_t >= 0, C_t + J_t === RV_t^{(d)}
        """
        np.random.seed(123)
        returns = np.random.normal(0, 0.002, 78)
        # Añadir salto
        returns[25] = 0.03

        rv, bv, tq, jump, continuous = decompose_realized_variance(returns, alpha_j=0.001)

        assert rv > 0
        assert continuous > 0
        assert jump >= 0.0
        assert pytest.approx(continuous + jump, abs=1e-10) == rv

    def test_heterogeneous_rv_aggregation(self):
        """Verifica RV^{(w)} (5 días) y RV^{(m)} (22 días)."""
        daily_rvs = [0.0001 * (i + 1) for i in range(30)]

        rv_w, rv_m = calculate_heterogeneous_rv(daily_rvs)

        # Últimos 5 días: 26, 27, 28, 29, 30 -> promedio
        expected_w = np.mean(daily_rvs[-5:])
        expected_m = np.mean(daily_rvs[-22:])

        assert pytest.approx(rv_w, abs=1e-10) == expected_w
        assert pytest.approx(rv_m, abs=1e-10) == expected_m

        # Historial corto (menos de 5 días)
        short_rvs = [0.0002, 0.0004]
        rw_short, rm_short = calculate_heterogeneous_rv(short_rvs)
        assert pytest.approx(rw_short, abs=1e-10) == 0.0003
        assert pytest.approx(rm_short, abs=1e-10) == 0.0003

    def test_log_har_rv_j_strict_positivity(self):
        """
        Verifica la positividad estricta de la varianza pronosticada:
        RV_hat_{t+1}^{(d)} > 0 bajo perturbaciones extremas o inputs mínimos.
        """
        betas = np.array([-0.25, 0.42, 0.30, 0.20, -0.05])
        sigma_eps_sq = 0.04

        # Caso regular
        pred_reg = predict_log_har_rv_j(0.0002, 0.00025, 0.0003, 0.0, betas, sigma_eps_sq)
        assert pred_reg > 0.0
        assert not math.isnan(pred_reg)
        assert not math.isinf(pred_reg)

        # Caso extremo con varianza minúscula
        pred_tiny = predict_log_har_rv_j(1e-10, 1e-10, 1e-10, 0.0, betas, sigma_eps_sq)
        assert pred_tiny > 0.0

        # Caso extremo con gran salto
        pred_jump = predict_log_har_rv_j(0.001, 0.001, 0.001, 0.05, betas, sigma_eps_sq)
        assert pred_jump > 0.0

    def test_point_volatility_scaling(self):
        """Verifica sigma_hat^{pts} = P_t * sqrt( RV_hat / 252 )."""
        price = 5000.0
        rv_pred = 0.0004  # Varianza anual ≈ 0.0004 * 252 ≈ 0.1008 (vol ≈ 31.7%)
        sigma_pts = calculate_point_volatility(rv_pred, price)

        expected = 5000.0 * math.sqrt(0.0004 / 252.0)
        assert pytest.approx(sigma_pts, abs=1e-8) == expected
        assert sigma_pts > 0

        # Precio cero o RV cero
        assert calculate_point_volatility(0.0, 5000.0) == 0.0
        assert calculate_point_volatility(0.0004, 0.0) == 0.0

    def test_jump_gate_veto(self):
        """Verifica que rho_J >= 0.35 activa el veto analítico de entrada."""
        # Sin saltos (rho = 0 < 0.35) -> Pasa
        rho, passed = evaluate_jump_gate(jump=0.0, rv_daily=0.001, theta_j=0.35)
        assert rho == 0.0
        assert passed is True

        # Salto moderado (rho = 0.20 < 0.35) -> Pasa
        rho, passed = evaluate_jump_gate(jump=0.0002, rv_daily=0.001, theta_j=0.35)
        assert pytest.approx(rho, abs=1e-6) == 0.20
        assert passed is True

        # Salto severo (rho = 0.50 >= 0.35) -> VETO
        rho, passed = evaluate_jump_gate(jump=0.0005, rv_daily=0.001, theta_j=0.35)
        assert pytest.approx(rho, abs=1e-6) == 0.50
        assert passed is False

    def test_dynamic_barriers_asymmetry(self):
        """
        Verifica las barreras dinámicas y la relación asimétrica estructural:
        kappa_TP / kappa_SL = 1.80 >= 1.50.
        """
        entry_p = 5000.0
        sigma_pts = 10.0

        # Long (+1)
        tp_long, sl_long = calculate_dynamic_barriers(entry_p, 1, sigma_pts, kappa_tp=1.80, kappa_sl=1.00)
        assert tp_long == 5000.0 + 1.80 * 10.0  # 5018.0
        assert sl_long == 5000.0 - 1.00 * 10.0  # 4990.0
        assert (tp_long - entry_p) / (entry_p - sl_long) == 1.80

        # Short (-1)
        tp_short, sl_short = calculate_dynamic_barriers(entry_p, -1, sigma_pts, kappa_tp=1.80, kappa_sl=1.00)
        assert tp_short == 5000.0 - 1.80 * 10.0  # 4982.0
        assert sl_short == 5000.0 + 1.00 * 10.0  # 5010.0
        assert (entry_p - tp_short) / (sl_short - entry_p) == 1.80

        # Dirección inválida
        with pytest.raises(ValueError):
            calculate_dynamic_barriers(entry_p, 0, sigma_pts)

    def test_circuit_breaker_trigger(self):
        """Verifica RV_intra >= xi_circuit * RV_pred con xi_circuit = 2.50."""
        rv_pred = 0.0002
        # Normal
        assert check_circuit_breaker(intraday_rv=0.0003, predicted_rv=rv_pred, xi_circuit=2.50) is False
        # Disparo exacto en 2.50 * 0.0002 = 0.0005
        assert check_circuit_breaker(intraday_rv=0.0005, predicted_rv=rv_pred, xi_circuit=2.50) is True
        # Disparo por encima
        assert check_circuit_breaker(intraday_rv=0.0010, predicted_rv=rv_pred, xi_circuit=2.50) is True

    def test_loss_streak_attenuation(self):
        """Verifica phi_{streak}(k): 1.0 (k=0), 0.5 (k=1), 0.25 (k>=2)."""
        acc = PropAccount(dll=1000.0, mtdd=2000.0)

        acc.consecutive_losses = 0
        assert acc.calculate_streak_factor() == 1.00

        acc.consecutive_losses = 1
        assert acc.calculate_streak_factor() == 0.50

        acc.consecutive_losses = 2
        assert acc.calculate_streak_factor() == 0.25

        acc.consecutive_losses = 5
        assert acc.calculate_streak_factor() == 0.25


# ==============================================================================
# 2. PRUEBAS DE CONTROL DE RIESGO PROP TRADING (SECCIÓN 5)
# ==============================================================================

class TestF8PropTradingRiskControls:
    """Verifica el cumplimiento estricto de reglas de Prop Trading."""

    def test_daily_loss_limit_breach(self):
        """Verifica que una pérdida acumulada >= DLL descalifique la cuenta."""
        acc = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        assert acc.is_breached is False

        # Trade perdedor de $600
        trade1 = TradeRecord(
            date="2026-10-10", symbol="ES", direction=1, contracts=1, is_micro=False,
            entry_time="09:35", entry_price=5000.0, stop_loss=4988.0, take_profit=5020.0,
            sigma_pts=12.0, net_pnl_usd=-600.0
        )
        acc.record_trade(trade1)
        assert acc.is_breached is False
        assert acc.daily_realized_loss == 600.0
        assert acc.remaining_dll == 400.0

        # Segundo trade perdedor de $450 (Total día = $1050 > $1000)
        trade2 = TradeRecord(
            date="2026-10-10", symbol="ES", direction=-1, contracts=1, is_micro=False,
            entry_time="11:00", entry_price=4990.0, stop_loss=5002.0, take_profit=4970.0,
            sigma_pts=12.0, net_pnl_usd=-450.0
        )
        acc.record_trade(trade2)
        assert acc.is_breached is True
        assert "DAILY_LOSS_LIMIT_BREACH" in acc.breach_reason
        assert acc.calculate_risk_budget() == 0.0

    def test_trailing_drawdown_breach(self):
        """Verifica que el Drawdown respecto al High Water Mark descalifique si >= MTDD."""
        acc = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)

        # Ganar $1,500 en día 1 -> nuevo HWM = $51,500
        trade_win = TradeRecord(
            date="2026-10-10", symbol="ES", direction=1, contracts=1, is_micro=False,
            entry_time="09:35", entry_price=5000.0, stop_loss=4988.0, take_profit=5020.0,
            sigma_pts=12.0, net_pnl_usd=1500.0
        )
        acc.record_trade(trade_win)
        assert acc.equity == 51500.0
        assert acc.peak_equity == 51500.0
        assert acc.current_drawdown == 0.0

        acc.start_session("2026-10-11")
        # Perder $900 en día 2 (equity = $50,600, drawdown = $900)
        trade_loss1 = TradeRecord(
            date="2026-10-11", symbol="ES", direction=1, contracts=1, is_micro=False,
            entry_time="09:35", entry_price=5050.0, stop_loss=5038.0, take_profit=5070.0,
            sigma_pts=12.0, net_pnl_usd=-900.0
        )
        acc.record_trade(trade_loss1)
        acc.end_session()
        assert acc.current_drawdown == 900.0
        assert acc.is_breached is False

        acc.start_session("2026-10-12")
        # Perder $1,150 en día 3 (equity = $49,450, drawdown = $2,050 >= $2,000 MTDD)
        trade_loss2 = TradeRecord(
            date="2026-10-12", symbol="ES", direction=-1, contracts=1, is_micro=False,
            entry_time="09:35", entry_price=5040.0, stop_loss=5052.0, take_profit=5020.0,
            sigma_pts=12.0, net_pnl_usd=-1150.0
        )
        acc.record_trade(trade_loss2)
        assert acc.is_breached is True
        assert "TRAILING_DRAWDOWN_BREACH" in acc.breach_reason

    def test_automatic_micro_switching(self):
        """
        Verifica que si el presupuesto de VaR no alcanza para 1 contrato estándar (ES),
        el algoritmo conmuta automáticamente a microcontratos (MES).
        """
        acc = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        # Volatilidad moderada/alta: sigma_pts = 15.0 puntos
        # VaR unitario para ES: 2.326 * 15.0 * 50 = $1,744.50
        # Presupuesto: phi(0)*min(0.25*1000, 0.08*2000) = min(250, 160) = $160.00
        # N_std = floor(160 / 1744.50) = 0 contratos ES!
        # Pero para MES: M = 5, VaR unitario = 2.326 * 15.0 * 5 = $174.45
        # N_micro = floor(160 / 174.45) = 0 contratos si sigma=15.

        # Con sigma_pts = 10.0 y DLL=1000, MTDD=2000:
        # Budget = min(250, 160) = 160.0
        # ES VaR = 2.326 * 10 * 50 = $1,163 -> n_std = 0
        # MES VaR = 2.326 * 10 * 5 = $116.30 -> n_micro = floor(160 / 116.3) = 1 contrato MES!
        n_contracts, is_micro, budget = calculate_position_sizing(
            sigma_pts=10.0,
            contract_symbol="ES",
            account=acc,
            max_contracts=5
        )

        assert n_contracts == 1
        assert is_micro is True
        assert budget == 160.0


# ==============================================================================
# 3. PRUEBAS DE SIMULACIÓN Y BACKTEST INTEGRADO
# ==============================================================================

class TestF8EndToEndBacktest:
    """Verifica la ejecución cronológica del backtest sobre datos sintéticos y reales."""

    @pytest.fixture
    def synthetic_multiday_data(self) -> pd.DataFrame:
        """Crea 3 días de datos intradía sintéticos a 5 minutos."""
        dates = [
            datetime(2026, 10, 12),
            datetime(2026, 10, 13),
            datetime(2026, 10, 14),
        ]
        records = []
        base_price = 5000.0

        for d in dates:
            for bar_idx in range(78):
                bar_dt = d + timedelta(hours=9, minutes=30 + bar_idx * 5)
                # Oscilación determinista
                fluctuation = 2.0 * math.sin(bar_idx * 0.2)
                p_open = base_price + fluctuation
                p_close = p_open + 0.5
                p_high = max(p_open, p_close) + 1.0
                p_low = min(p_open, p_close) - 1.0

                records.append({
                    "datetime": bar_dt,
                    "Open": p_open,
                    "High": p_high,
                    "Low": p_low,
                    "Close": p_close,
                    "Volume": 1000.0
                })
            base_price += 10.0

        df = pd.DataFrame(records)
        df.set_index("datetime", inplace=True)
        return df

    def test_synthetic_backtest_execution(self, synthetic_multiday_data):
        """Verifica que el backtester ejecute sin errores y reporte métricas coherentes."""
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        strategy = EconometricHARRVJStrategy(symbol="ES")
        backtester = HARRVJBacktester(strategy=strategy, account=account)

        report = backtester.run(intraday_data=synthetic_multiday_data)

        assert "initial_capital" in report
        assert report["initial_capital"] == 50000.0
        assert report["is_prop_breached"] is False
        assert len(backtester.daily_states) > 0

    def test_real_data_execution_if_present(self):
        """Verifica la ejecución sobre el archivo real de CME si existe en data/raw."""
        file_path = Path("data/raw/CME_MINI_ES_5m_bars5000.csv")
        if not file_path.exists():
            pytest.skip(f"Archivo de datos no encontrado: {file_path}")

        df = pd.read_csv(file_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)

        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        strategy = EconometricHARRVJStrategy(symbol="ES")
        backtester = HARRVJBacktester(strategy=strategy, account=account)

        report = backtester.run(intraday_data=df)

        assert report["total_trades"] > 0
        assert report["total_commissions_usd"] > 0
        assert report["total_slippage_usd"] > 0
        assert report["is_prop_breached"] is False

    def test_cme_tick_quantization(self):
        """Verifica que los precios se discreticen estrictamente a múltiplos de tick_size (0.25)."""
        assert quantize_price(5000.123, 0.25) == 5000.00
        assert quantize_price(5000.13, 0.25) == 5000.25
        assert quantize_price(5000.374, 0.25) == 5000.25
        assert quantize_price(5000.38, 0.25) == 5000.50

        # Barreras con volatilidad fraccional
        tp, sl = calculate_dynamic_barriers(
            entry_price=5000.0,
            direction=1,
            sigma_pts=3.14159,
            kappa_tp=1.80,
            kappa_sl=1.00,
            tick_size=0.25
        )
        # 5000 + 1.8 * 3.14159 = 5005.65486 -> 5005.75
        # 5000 - 1.0 * 3.14159 = 4996.85841 -> 4996.75
        assert (tp % 0.25) == 0.0
        assert (sl % 0.25) == 0.0

    def test_outside_bar_stop_loss_precedence(self):
        """
        Verifica que en una barra extrema (outside bar) que toca tanto el TP como el SL,
        el control de riesgo de Prop Trading priorice conservadoramente el STOP_LOSS.
        """
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        strategy = EconometricHARRVJStrategy(symbol="ES")

        # Crear sesión sintética con 3 barras
        d = datetime(2026, 10, 15)
        # Barra 0: señal Long
        bar0 = {"Open": 5000.0, "High": 5002.0, "Low": 4999.0, "Close": 5001.0, "Volume": 100}
        # Barra 1: entrada Long a 5001.0, TP = 5001 + 1.8*10 = 5019.0, SL = 5001 - 10 = 4991.0
        # En esta misma barra el mercado se mueve violentamente de 4985.0 a 5025.0 (toca ambos)
        bar1 = {"Open": 5001.0, "High": 5025.0, "Low": 4985.0, "Close": 5010.0, "Volume": 100}
        bar2 = {"Open": 5010.0, "High": 5011.0, "Low": 5009.0, "Close": 5010.0, "Volume": 100}

        bars_df = pd.DataFrame([bar0, bar1, bar2], index=[
            datetime(2026, 10, 15, 9, 30),
            datetime(2026, 10, 15, 9, 35),
            datetime(2026, 10, 15, 9, 40)
        ])

        prior_state = HARStateVector(
            date="2026-10-14",
            rv_daily=0.0004,
            bv=0.0004,
            jump=0.0,
            continuous=0.0004,
            rv_weekly=0.0004,
            rv_monthly=0.0004,
            rv_predicted=0.0004,
            sigma_predicted_pts=10.0,
            jump_ratio=0.0,
            jump_gate_passed=True
        )

        trade = strategy.simulate_day(
            session_date=d.date(),
            day_bars=bars_df,
            prior_state=prior_state,
            account=account
        )

        assert trade is not None
        assert trade.exit_reason == HARExitReason.STOP_LOSS
        assert trade.exit_price <= 4991.0

    def test_gap_opening_stop_loss_fill(self):
        """Verifica que si la barra abre con gap más allá del stop loss, se llene al Open."""
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        strategy = EconometricHARRVJStrategy(symbol="ES")

        # Entrada Long a 5000, SL = 4990
        # Barra 2 abre con gap en 4980 (10 puntos por debajo de SL)
        bar0 = {"Open": 4998.0, "High": 5001.0, "Low": 4998.0, "Close": 5000.0, "Volume": 100}
        bar1 = {"Open": 5000.0, "High": 5002.0, "Low": 4998.0, "Close": 4999.0, "Volume": 100}
        bar2 = {"Open": 4980.0, "High": 4985.0, "Low": 4975.0, "Close": 4982.0, "Volume": 100}

        bars_df = pd.DataFrame([bar0, bar1, bar2], index=[
            datetime(2026, 10, 15, 9, 30),
            datetime(2026, 10, 15, 9, 35),
            datetime(2026, 10, 15, 9, 40)
        ])

        prior_state = HARStateVector(
            date="2026-10-14",
            rv_daily=0.0004,
            bv=0.0004,
            jump=0.0,
            continuous=0.0004,
            rv_weekly=0.0004,
            rv_monthly=0.0004,
            rv_predicted=0.0004,
            sigma_predicted_pts=10.0,
            jump_ratio=0.0,
            jump_gate_passed=True
        )

        trade = strategy.simulate_day(
            session_date=datetime(2026, 10, 15).date(),
            day_bars=bars_df,
            prior_state=prior_state,
            account=account
        )

        assert trade is not None
        assert trade.exit_reason == HARExitReason.STOP_LOSS
        # El fill no puede ser 4990.0 porque abrió en 4980.0
        assert trade.exit_price == 4980.0

    def test_flat_prices_and_zero_volatility_edge_case(self):
        """Verifica robustez ante congelamiento de precios (retornos idénticos a cero)."""
        flat_prices = np.full(78, 5000.0)
        returns = calculate_log_returns(flat_prices)
        assert np.all(returns == 0.0)

        rv, bv, tq, jump, continuous = decompose_realized_variance(returns)
        assert rv > 0.0
        assert jump == 0.0
        assert continuous > 0.0

        # Historial de 25 días planos para OLS
        flat_recs = [{
            "rv_daily": 1e-12, "bv": 1e-12, "jump": 0.0,
            "continuous": 1e-12, "rv_weekly": 1e-12, "rv_monthly": 1e-12
        } for _ in range(25)]

        betas, sig_sq = fit_log_har_rv_j(flat_recs)
        assert len(betas) == 5
        assert not np.any(np.isnan(betas))
        assert sig_sq > 0.0

    def test_volatility_managed_weight_calculation(self):
        """Verifica w_t^{vol} = sigma^* / sigma_hat^{pts} (Moreira & Muir 2017)."""
        # ES: sigma* = 25.0
        w_es_low_vol = calculate_volatility_managed_weight(sigma_pts=12.5, symbol="ES")
        assert pytest.approx(w_es_low_vol, abs=1e-6) == 2.0  # Reduce riesgo -> duplica peso

        w_es_high_vol = calculate_volatility_managed_weight(sigma_pts=50.0, symbol="ES")
        assert pytest.approx(w_es_high_vol, abs=1e-6) == 0.5  # Aumenta riesgo -> comprime a la mitad

        # NQ: sigma* = 90.0
        w_nq = calculate_volatility_managed_weight(sigma_pts=90.0, symbol="NQ")
        assert pytest.approx(w_nq, abs=1e-6) == 1.0

        # Entrada nula o negativa
        assert calculate_volatility_managed_weight(0.0) == 0.0
        assert calculate_volatility_managed_weight(-5.0) == 0.0

    def test_explicit_micro_selection(self):
        """Verifica que solicitar explícitamente MES configure y dimensione microcontratos."""
        acc = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        # Volatilidad baja (sigma_pts = 2.0)
        # Budget = 160.0. Para ES el VaR = 2.326 * 2.0 * 50 = $232.60 -> n_std sería 0
        # Para MES: VaR = 2.326 * 2.0 * 5 = $23.26 -> n_micro = floor(160 / 23.26) = 6 contratos MES!
        n_contracts, is_micro, budget = calculate_position_sizing(
            sigma_pts=2.0,
            contract_symbol="MES",
            account=acc,
            max_contracts=5
        )
        assert is_micro is True
        assert n_contracts == 6

    def test_neutral_opening_bar_no_trade(self):
        """Verifica que si Close == Open en la primera barra, la estrategia permanezca neutra."""
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        strategy = EconometricHARRVJStrategy(symbol="ES")

        bar0 = {"Open": 5000.0, "High": 5002.0, "Low": 4998.0, "Close": 5000.0, "Volume": 100}
        bar1 = {"Open": 5000.0, "High": 5005.0, "Low": 4999.0, "Close": 5004.0, "Volume": 100}
        bars_df = pd.DataFrame([bar0, bar1], index=[
            datetime(2026, 10, 15, 9, 30),
            datetime(2026, 10, 15, 9, 35)
        ])

        prior_state = HARStateVector(
            date="2026-10-14",
            rv_daily=0.0004,
            bv=0.0004,
            jump=0.0,
            continuous=0.0004,
            rv_weekly=0.0004,
            rv_monthly=0.0004,
            rv_predicted=0.0004,
            sigma_predicted_pts=10.0,
            jump_ratio=0.0,
            jump_gate_passed=True
        )

        trade = strategy.simulate_day(
            session_date=datetime(2026, 10, 15).date(),
            day_bars=bars_df,
            prior_state=prior_state,
            account=account
        )
        assert trade is None
