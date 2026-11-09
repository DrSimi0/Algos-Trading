"""
tests/test_har_rv_j_opt_n1.py
Suite de pruebas unitarias, matemáticas, microestructurales y de estrés
para la Versión Alterna Optimizada: F8-Optimizacion -IS- N1.

Verifica rigurosamente:
1. Descomposición matemática rigurosa entre Bipower Variation (BV) y Saltos discretos (J).
2. Propiedad de ortogonalidad analítica: C_t > 0, J_t >= 0, C_t + J_t === RV_t.
3. Función de penalización exponencial de saltos: Psi_t^{jump} = exp(-lambda * JumpMetric).
4. Monotonía decreciente estricta de la penalización respecto a la magnitud de los saltos.
5. Veto duro de seguridad ante saltos catastróficos (rho_J >= theta_hard_veto).
6. Modulación del presupuesto de riesgo y dimensionamiento con conmutación Estándar/Micro.
7. Cálculo de métricas de subasta de apertura (RV_15m, retorno y sesgo direccional).
8. Cumplimiento de microestructura CME Globex (ticks de 0.25, outside bars, gaps).
9. Control de riesgo Prop Trading (DLL, MTDD, racha perdedora phi_streak).
10. Simulación cronológica sin Lookahead Bias con HARRVJOptN1Backtester.
"""

from datetime import time, datetime, timedelta
import math
import numpy as np
import pandas as pd
import pytest

from src.strategies.har_rv_j import (
    PropAccount,
    HARExitReason,
    calculate_log_returns,
    calculate_realized_variance,
    calculate_bipower_variation,
    calculate_tripower_quarticity,
    calculate_jump_statistic,
    quantize_price,
    Z_ALPHA_JUMP_001,
)
from src.strategies.har_rv_j_opt_n1 import (
    HARStateVectorOptN1,
    HARRVJOptN1TradeRecord,
    EconometricHARRVJOptN1Strategy,
    HARRVJOptN1Backtester,
    calculate_continuous_jump_decomposition,
    calculate_jump_exponential_penalty,
    calculate_position_sizing_opt_n1,
    calculate_opening_15m_metrics,
)
from src.core.futures_specs import get_spec


# ==============================================================================
# 1. PRUEBAS DE DESCOMPOSICIÓN MATEMÁTICA BIPOTENCIA Y SALTOS
# ==============================================================================

class TestF8N1MathematicalDecomposition:
    """Verifica la separación matemática estricta entre BV_t y J_t."""

    def test_orthogonality_and_conservation(self):
        """
        Verifica que C_t > 0, J_t >= 0, y C_t + J_t === RV_t
        tanto en difusión regular como con saltos artificiales.
        """
        np.random.seed(42)
        # 78 barras de 5m con retornos gaussianos
        returns = np.random.normal(0, 0.0015, 78)

        # Caso A: Sin saltos significativos
        rv, bv, tq, z_stat, jump, continuous, is_sig = calculate_continuous_jump_decomposition(
            returns, alpha_j=0.001
        )
        assert rv > 0.0
        assert continuous > 0.0
        assert jump >= 0.0
        assert pytest.approx(continuous + jump, abs=1e-11) == rv

        # Caso B: Con un salto masivo aislado (+4%)
        returns_with_jump = returns.copy()
        returns_with_jump[30] = 0.04
        rv_j, bv_j, tq_j, z_stat_j, jump_j, continuous_j, is_sig_j = calculate_continuous_jump_decomposition(
            returns_with_jump, alpha_j=0.001
        )
        assert rv_j > 0.0
        assert continuous_j > 0.0
        assert jump_j > 0.0
        assert is_sig_j is True
        assert z_stat_j > Z_ALPHA_JUMP_001
        assert pytest.approx(continuous_j + jump_j, abs=1e-11) == rv_j

    def test_bipower_robustness_vs_rv(self):
        """
        Verifica que Bipower Variation no se contamina ante un salto puntual:
        RV absorbe el cuadrado del salto, mientras que BV se mantiene mucho menor.
        """
        np.random.seed(101)
        base_returns = np.random.normal(0, 0.001, 78)
        jump_returns = base_returns.copy()
        jump_returns[40] = 0.05  # Salto de 500 bps

        rv = calculate_realized_variance(jump_returns)
        bv = calculate_bipower_variation(jump_returns)

        # RV debe ser significativamente mayor que BV
        assert rv > 3.0 * bv
        assert rv - bv > 0.002

    def test_insignificant_jump_sets_jump_to_zero(self):
        """
        Si Z_{J, t} <= z_alpha, el salto estimado debe ser idénticamente cero
        y la varianza continua debe ser exactamente igual a RV_t.
        """
        np.random.seed(77)
        # Retornos puramente gaussianos de baja varianza
        returns = np.random.normal(0, 0.0005, 78)
        rv, bv, tq, z_stat, jump, continuous, is_sig = calculate_continuous_jump_decomposition(
            returns, alpha_j=0.001
        )
        if not is_sig:
            assert jump == 0.0
            assert continuous == rv

    def test_empty_or_degenerate_returns(self):
        """Manejo numéricamente seguro de vectores de retorno vacíos o de longitud < 3."""
        rv, bv, tq, z_stat, jump, continuous, is_sig = calculate_continuous_jump_decomposition(
            np.array([]), alpha_j=0.001
        )
        assert rv > 0.0
        assert jump == 0.0
        assert is_sig is False

        rv2, bv2, tq2, z_stat2, jump2, continuous2, is_sig2 = calculate_continuous_jump_decomposition(
            np.array([0.001, -0.001]), alpha_j=0.001
        )
        assert rv2 > 0.0
        assert jump2 == 0.0
        assert is_sig2 is False


# ==============================================================================
# 2. PRUEBAS DE LA FUNCIÓN DE PENALIZACIÓN EXPONENCIAL DE SALTOS
# ==============================================================================

class TestF8N1ExponentialPenaltyFunction:
    """Verifica las propiedades matemáticas de Psi_t^{jump} = exp(-lambda * JumpMetric)."""

    def test_no_penalty_when_no_jump_or_insignificant(self):
        """
        Si no hay salto significativo, Psi_t^{jump} debe ser exactamente 1.0 (sin castigo).
        """
        # Caso 1: jump = 0
        rho, metric, psi, passed = calculate_jump_exponential_penalty(
            jump=0.0, rv_daily=0.001, z_stat=0.5, is_significant=False, lambda_jump=4.0
        )
        assert rho == 0.0
        assert metric == 0.0
        assert psi == 1.0
        assert passed is True

        # Caso 2: is_significant = False aunque jump numérico > 0 (ruido aleatorio)
        rho2, metric2, psi2, passed2 = calculate_jump_exponential_penalty(
            jump=0.0001, rv_daily=0.001, z_stat=1.5, is_significant=False, lambda_jump=4.0
        )
        assert metric2 == 0.0
        assert psi2 == 1.0
        assert passed2 is True

    def test_strict_monotonic_decay(self):
        """
        A mayor proporción de salto rho_J = J / RV, la penalización Psi debe ser
        estrictamente menor (menor apalancamiento permitido).
        """
        rv = 0.001
        lamb = 4.0

        # rho = 0.10, 0.25, 0.40
        _, _, psi_10, _ = calculate_jump_exponential_penalty(0.0001, rv, 4.0, True, lambda_jump=lamb)
        _, _, psi_25, _ = calculate_jump_exponential_penalty(0.00025, rv, 4.5, True, lambda_jump=lamb)
        _, _, psi_40, _ = calculate_jump_exponential_penalty(0.0004, rv, 5.0, True, lambda_jump=lamb)

        assert 1.0 > psi_10 > psi_25 > psi_40 > 0.0
        # Valores analíticos esperados: exp(-4 * 0.10) = exp(-0.4) ≈ 0.6703
        assert pytest.approx(psi_10, abs=1e-4) == math.exp(-0.4)
        assert pytest.approx(psi_25, abs=1e-4) == math.exp(-1.0)
        assert pytest.approx(psi_40, abs=1e-4) == math.exp(-1.6)

    def test_lambda_sensitivity(self):
        """
        Un lambda mayor aplica un castigo exponencial más severo para el mismo salto.
        """
        rv = 0.001
        jump = 0.0002  # rho = 0.20

        _, _, psi_low, _ = calculate_jump_exponential_penalty(jump, rv, 4.0, True, lambda_jump=2.0)
        _, _, psi_med, _ = calculate_jump_exponential_penalty(jump, rv, 4.0, True, lambda_jump=4.0)
        _, _, psi_high, _ = calculate_jump_exponential_penalty(jump, rv, 4.0, True, lambda_jump=8.0)

        assert psi_low > psi_med > psi_high
        assert pytest.approx(psi_low, abs=1e-4) == math.exp(-0.4)
        assert pytest.approx(psi_med, abs=1e-4) == math.exp(-0.8)
        assert pytest.approx(psi_high, abs=1e-4) == math.exp(-1.6)

    def test_hard_veto_on_extreme_jump(self):
        """
        Si rho_J >= theta_hard_veto (por defecto 0.55), la compuerta se cierra totalmente
        (Psi = 0.0 y gate_passed = False), protegiendo la cuenta ante anomalías macro.
        """
        rv = 0.001
        jump_extreme = 0.0006  # rho = 0.60 >= 0.55

        rho, metric, psi, passed = calculate_jump_exponential_penalty(
            jump=jump_extreme, rv_daily=rv, z_stat=8.0, is_significant=True,
            lambda_jump=4.0, theta_hard_veto=0.55
        )
        assert rho == 0.60
        assert psi == 0.0
        assert passed is False


# ==============================================================================
# 3. PRUEBAS DE DIMENSIONAMIENTO DINÁMICO Y CONMUTACIÓN ESTÁNDAR / MICRO
# ==============================================================================

class TestF8N1PositionSizing:
    """Verifica que el dimensionamiento penalizado respete los límites de Prop Trading."""

    def test_sizing_reduction_under_jump_penalty(self):
        """
        Una penalización Psi < 1.0 reduce directamente el presupuesto efectivo
        y el número de contratos respecto al régimen sin salto.
        """
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)

        # Caso A: Volatilidad estándar intradía (~2.5 pts en ES)
        # Sin penalización (Psi = 1.0) -> Presupuesto $160 permite 5 microcontratos
        n_unpen, is_m_unpen, raw_b, eff_b1 = calculate_position_sizing_opt_n1(
            sigma_pts=2.5,
            contract_symbol="ES",
            account=account,
            jump_penalty_factor=1.0,
            max_contracts=5
        )
        assert n_unpen == 5
        assert is_m_unpen is True

        # Con penalización moderada (Psi = 0.40) -> Reduce a 2 microcontratos
        n_pen_40, _, _, eff_b2 = calculate_position_sizing_opt_n1(
            sigma_pts=2.5,
            contract_symbol="ES",
            account=account,
            jump_penalty_factor=0.40,
            max_contracts=5
        )
        assert n_pen_40 == 2
        assert n_unpen > n_pen_40

        # Con penalización severa (Psi = 0.20) -> Reduce a 1 microcontrato
        n_pen_20, _, _, eff_b3 = calculate_position_sizing_opt_n1(
            sigma_pts=2.5,
            contract_symbol="ES",
            account=account,
            jump_penalty_factor=0.20,
            max_contracts=5
        )
        assert n_pen_20 == 1
        assert n_pen_40 > n_pen_20

        # Caso B: Volatilidad elevada (15.0 pts)
        n_high_unpen, _, _, eff_h1 = calculate_position_sizing_opt_n1(
            sigma_pts=15.0,
            contract_symbol="ES",
            account=account,
            jump_penalty_factor=1.0,
            max_contracts=5
        )
        n_high_pen, _, _, eff_h2 = calculate_position_sizing_opt_n1(
            sigma_pts=15.0,
            contract_symbol="ES",
            account=account,
            jump_penalty_factor=0.40,
            max_contracts=5
        )
        assert eff_h1 == raw_b
        assert pytest.approx(eff_h2, abs=1e-6) == raw_b * 0.40
        assert n_high_unpen == 1
        assert n_high_pen == 0
        assert n_high_unpen > n_high_pen

    def test_automatic_commutation_to_micro_when_budget_insufficient_for_standard(self):
        """
        Si el presupuesto efectivo no alcanza para 1 contrato estándar de ES (50 USD/pt),
        debe conmutar automáticamente a microcontratos MES (5 USD/pt).
        """
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        sigma_pts = 35.0  # Alta volatilidad

        # Penalización fuerte (Psi = 0.25)
        n, is_micro, raw_b, eff_b = calculate_position_sizing_opt_n1(
            sigma_pts=sigma_pts,
            contract_symbol="ES",
            account=account,
            jump_penalty_factor=0.25,
            max_contracts=5
        )

        # Verificamos si conmuta a micro
        unit_var_std = 2.326 * sigma_pts * 50.0  # ≈ 4,070 USD > Budget
        if eff_b < unit_var_std:
            assert is_micro is True

    def test_zero_sizing_on_breached_account_or_zero_budget(self):
        """Una cuenta descalificada o con presupuesto 0 debe devolver 0 contratos."""
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        account.is_breached = True

        n, is_micro, raw_b, eff_b = calculate_position_sizing_opt_n1(
            sigma_pts=20.0,
            contract_symbol="ES",
            account=account,
            jump_penalty_factor=1.0
        )
        assert n == 0
        assert raw_b == 0.0
        assert eff_b == 0.0


# ==============================================================================
# 4. PRUEBAS DE MÉTRICAS DE APERTURA Y DESACOPLAMIENTO TEMPORAL
# ==============================================================================

class TestF8N1OpeningAuctionMetrics:
    """Verifica el cálculo de RV_15m y el filtro direccional de subasta matutina."""

    def test_opening_15m_metrics_directional_signal(self):
        """
        Verifica que una tendencia alcista nítida en los primeros 15 min genere
        direction = +1, bajista genere -1, y apertura plana genere 0.
        """
        # Barras alcistas (3 barras de 5m: 09:30, 09:35, 09:40)
        times = [datetime(2023, 5, 10, 9, 30), datetime(2023, 5, 10, 9, 35), datetime(2023, 5, 10, 9, 40)]
        df_up = pd.DataFrame({
            "Open": [5000.0, 5005.0, 5010.0],
            "High": [5006.0, 5011.0, 5016.0],
            "Low": [4999.0, 5004.0, 5009.0],
            "Close": [5005.0, 5010.0, 5015.0],
        }, index=times)

        rv_up, r_open_up, thresh_up, dir_up = calculate_opening_15m_metrics(df_up)
        assert rv_up > 0.0
        assert r_open_up > 0.0
        assert dir_up == 1

        # Barras bajistas
        df_down = pd.DataFrame({
            "Open": [5015.0, 5010.0, 5005.0],
            "High": [5016.0, 5011.0, 5006.0],
            "Low": [5009.0, 5004.0, 4999.0],
            "Close": [5010.0, 5005.0, 5000.0],
        }, index=times)

        rv_dn, r_open_dn, thresh_dn, dir_dn = calculate_opening_15m_metrics(df_down)
        assert r_open_dn < 0.0
        assert dir_dn == -1

        # Barras planas (dentro del umbral)
        df_flat = pd.DataFrame({
            "Open": [5000.0, 5000.1, 5000.0],
            "High": [5001.0, 5001.0, 5001.0],
            "Low": [4999.0, 4999.0, 4999.0],
            "Close": [5000.05, 5000.02, 5000.0],
        }, index=times)

        rv_fl, r_open_fl, thresh_fl, dir_fl = calculate_opening_15m_metrics(df_flat)
        assert dir_fl == 0

    def test_opening_vwap_confirmation_filter(self):
        """
        Si el retorno r_open supera el umbral pero el precio de cierre no confirma
        con respecto al VWAP (e.g. cierre por debajo de VWAP en subasta con sesgo alcista),
        la dirección debe ser 0 (indecisión).
        """
        times = [datetime(2023, 5, 10, 9, 30), datetime(2023, 5, 10, 9, 35), datetime(2023, 5, 10, 9, 40)]
        # Barra 1 abre en 5000, sube con volumen monstruoso a 5030, y luego retrocede a 5005
        # r_open = ln(5005 / 5000) > 0, pero VWAP estará cerca de 5025 > 5005
        df_vwap_reject = pd.DataFrame({
            "Open": [5000.0, 5025.0, 5015.0],
            "High": [5030.0, 5030.0, 5020.0],
            "Low": [5000.0, 5015.0, 5004.0],
            "Close": [5025.0, 5015.0, 5005.0],
            "Volume": [10000, 100, 100]  # La masa de volumen estuvo arriba
        }, index=times)

        rv, r_open, thresh, direction = calculate_opening_15m_metrics(df_vwap_reject)
        assert r_open > 0
        # Cierre 5005 está muy por debajo del VWAP ponderado (~5024.7)
        assert direction == 0


# ==============================================================================
# 5. PRUEBAS DE MICROESTRUCTURA CME Y DISYUNTOR
# ==============================================================================

class TestF8N1CMEPrinciples:
    """Verifica reglas CME: ticks indivisibles de 0.25, gaps y disyuntor."""

    def test_cme_tick_quantization(self):
        """Los precios de orden deben cuantizarse a múltiplos indivisibles de 0.25."""
        p1 = 5000.123
        p2 = 5000.374
        p3 = 5000.89

        assert quantize_price(p1, 0.25) == 5000.00
        assert quantize_price(p2, 0.25) == 5000.25
        assert quantize_price(p3, 0.25) == 5001.00

    def test_stop_loss_priority_in_high_volatility(self):
        """
        En barras de rango extremo que tocan simultáneamente el TP y el SL,
        la evaluación debe dar prioridad estricta al Stop-Loss por prudencia Prop Firm.
        """
        account = PropAccount(initial_capital=50000.0, dll=2000.0, mtdd=5000.0)
        strategy = EconometricHARRVJOptN1Strategy(symbol="ES", entry_delay_bars=1)

        # Crear barras donde la barra 1 abre posición Long, y la barra 2 tiene un rango masivo
        times = [
            datetime(2023, 6, 1, 9, 30),
            datetime(2023, 6, 1, 9, 35),
            datetime(2023, 6, 1, 9, 40),
            datetime(2023, 6, 1, 9, 45)
        ]
        df_day = pd.DataFrame({
            "Open": [5000.0, 5010.0, 5010.0, 5010.0],
            "High": [5012.0, 5015.0, 5090.0, 5015.0],  # 5090 toca TP
            "Low": [4998.0, 5008.0, 4920.0, 5005.0],   # 4920 toca SL
            "Close": [5010.0, 5012.0, 5010.0, 5010.0],
        }, index=times)

        state = HARStateVectorOptN1(
            date="2023-05-31",
            rv_daily=0.0002,
            bv=0.0002,
            jump=0.0,
            continuous=0.0002,
            rv_weekly=0.0002,
            rv_monthly=0.0002,
            rv_predicted=0.0002,
            sigma_predicted_pts=5.0,
            z_jump_stat=0.0,
            is_jump_significant=False,
            jump_ratio=0.0,
            jump_metric=0.0,
            jump_penalty_factor=1.0,
            volatility_managed_weight=1.0,
            effective_weight=1.0,
            jump_gate_passed=True
        )

        trade = strategy.simulate_day("2023-06-01", df_day, state, account)
        assert trade is not None
        # Ante conflicto simultáneo, debe prevalecer STOP_LOSS
        assert trade.exit_reason == HARExitReason.STOP_LOSS


# ==============================================================================
# 6. PRUEBAS END-TO-END DE ESTRATEGIA Y BACKTESTER
# ==============================================================================

class TestF8N1EndToEndSimulation:
    """Verifica el flujo cronológico integral con HARRVJOptN1Backtester."""

    def test_backtester_multi_day_run(self):
        """
        Ejecuta una simulación sintética de 5 días con datos a 5 minutos,
        verificando la transición de estados y la generación del informe de desempeño.
        """
        np.random.seed(999)
        n_days = 5
        bars_per_day = 78
        dates = [datetime(2023, 6, 1) + timedelta(days=i) for i in range(n_days)]

        all_bars = []
        curr_price = 5000.0

        for d in dates:
            for b in range(bars_per_day):
                dt = datetime(d.year, d.month, d.day, 9, 30) + timedelta(minutes=5 * b)
                ret = np.random.normal(0, 0.001)
                new_price = curr_price * math.exp(ret)
                high = max(curr_price, new_price) * 1.0005
                low = min(curr_price, new_price) * 0.9995

                all_bars.append({
                    "datetime": dt,
                    "Open": curr_price,
                    "High": high,
                    "Low": low,
                    "Close": new_price,
                    "Volume": 1000
                })
                curr_price = new_price

        df_sim = pd.DataFrame(all_bars).set_index("datetime")

        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        strategy = EconometricHARRVJOptN1Strategy(
            symbol="ES",
            lambda_jump=4.0,
            entry_delay_bars=3,
            kappa_tp=2.25,
            kappa_sl=1.25
        )
        backtester = HARRVJOptN1Backtester(strategy=strategy, account=account)

        report = backtester.run(df_sim)

        assert report is not None
        assert "final_equity" in report
        assert "total_trades" in report
        assert "profit_factor" in report
        assert "avg_jump_penalty_factor" in report
        assert report["is_prop_breached"] is False
        assert report["initial_capital"] == 50000.0

    def test_backtester_jump_penalty_execution(self):
        """
        Verifica que el backtester ejecute efectivamente operaciones penalizadas
        cuando la víspera experimenta un salto estadísticamente significativo,
        registrando jump_penalized_trades_count >= 1 y jump_penalty_factor < 1.0.
        """
        np.random.seed(888)
        n_days = 3
        bars_per_day = 78
        dates = [datetime(2023, 6, 1) + timedelta(days=i) for i in range(n_days)]

        all_bars = []
        curr_price = 5000.0

        for day_idx, d in enumerate(dates):
            for b in range(bars_per_day):
                dt = datetime(d.year, d.month, d.day, 9, 30) + timedelta(minutes=5 * b)
                # En el día 1, introducimos un salto moderado significativo (barra 35)
                if day_idx == 1 and b == 35:
                    ret = 0.010  # Salto de 100 bps (produce z ≈ 3.89, rho ≈ 0.35, psi ≈ 0.25)
                elif day_idx == 2 and b < 3:
                    # En el día 2, apertura alcista limpia
                    ret = 0.002
                else:
                    ret = np.random.normal(0, 0.001)

                new_price = curr_price * math.exp(ret)
                high = max(curr_price, new_price) * 1.0005
                low = min(curr_price, new_price) * 0.9995

                all_bars.append({
                    "datetime": dt,
                    "Open": curr_price,
                    "High": high,
                    "Low": low,
                    "Close": new_price,
                    "Volume": 2000
                })
                curr_price = new_price

        df_sim = pd.DataFrame(all_bars).set_index("datetime")

        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        strategy = EconometricHARRVJOptN1Strategy(
            symbol="ES",
            lambda_jump=4.0,
            entry_delay_bars=3,
            kappa_tp=2.25,
            kappa_sl=1.25
        )
        backtester = HARRVJOptN1Backtester(strategy=strategy, account=account)
        report = backtester.run(df_sim)

        # Debe haberse registrado al menos 1 trade con penalización activa
        penalized = [t for t in backtester.trades if t.jump_penalty_factor < 0.999]
        assert len(penalized) >= 1
        assert report["jump_penalized_trades_count"] >= 1
        assert report["avg_jump_penalty_factor"] < 1.0
        assert penalized[0].jump_stat > 3.09

    def test_base_f8_unchanged_and_compatible(self):
        """
        Verifica que la versión base F8 en src.strategies.har_rv_j
        siga funcionando perfectamente sin alteraciones colaterales.
        """
        from src.strategies.har_rv_j import (
            EconometricHARRVJStrategy,
            calculate_bipower_variation,
            evaluate_jump_gate
        )
        # Importaciones y funciones base intactas
        assert calculate_bipower_variation(np.array([0.001, -0.002, 0.001])) > 0
        rho, passed = evaluate_jump_gate(0.0, 0.001, 0.35)
        assert passed is True
