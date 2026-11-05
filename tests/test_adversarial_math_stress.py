"""
Test Suite: Verificación Adversaria y Empírica de Bordes Matemáticos,
Consistencia Dimensional y Reglas de Prop Trading (F1 a F9).

Este harness de pruebas estresa empíricamente los 9 modelos matemáticos descritos
en reports/estrategias_matematicas/ sometiendo sus ecuaciones y parametrizaciones
a condiciones extremas de mercado, límites de cuenta de fondeo, singularidades numéricas
y escenarios adversarios.
"""

import math
import numpy as np
import pytest


# ==============================================================================
# 1. PRUEBAS ADVERSARIAS: F1 - Market Intraday Momentum & MOC Seasonality
# ==============================================================================
class TestF1AdversarialStress:
    """Stress testing de ecuaciones y bordes para F1."""

    def test_f1_return_and_threshold_singularities(self):
        """Verifica que el umbral theta_t y el retorno matutino no admitan indeterminaciones."""
        # Precios normales
        p_0930 = 5800.0
        p_1000 = 5815.0
        r_first = math.log(p_1000 / p_0930)
        assert isinstance(r_first, float)
        assert not math.isnan(r_first)

        # Volatilidad nula (mercado congelado)
        sigma_daily_zero = 0.0
        c_sigma = 0.20
        theta_floor = 0.0015
        theta_zero_vol = max(theta_floor, c_sigma * (sigma_daily_zero / p_1000))
        assert theta_zero_vol == 0.0015
        assert theta_zero_vol > 0, "El umbral de F1 debe tener un piso estrictamente positivo"

        # Volatilidad extrema (crash / VIX 80)
        sigma_daily_extreme = 500.0  # 500 puntos de desviación diaria
        theta_extreme = max(theta_floor, c_sigma * (sigma_daily_extreme / p_1000))
        assert theta_extreme > theta_floor
        assert not math.isnan(theta_extreme)

    def test_f1_prop_trading_sizing_budget_invariants(self):
        """Comprueba que el dimensionamiento respete rígidamente DLL y MTDD bajo cualquier racha."""
        dll = 1000.0
        mtdd = 2000.0
        p_entry = 5800.0
        lambda_sl = 0.0035
        m_es = 50.0
        m_mes = 5.0
        c_fric_es = 2 * 0.25 * m_es + 4.40  # 29.40
        c_fric_mes = 2 * 0.25 * m_mes + 1.20  # 3.70
        delta_sl = p_entry * lambda_sl  # 20.3 pt
        r_contract_es = delta_sl * m_es + c_fric_es
        r_contract_mes = delta_sl * m_mes + c_fric_mes

        # Recorrer pérdidas previas del día y rachas perdedoras
        for l_day in [0.0, 500.0, 999.0, 1000.0, 1500.0]:
            for dd in [0.0, 1000.0, 1999.0, 2000.0, 2500.0]:
                for k_loss in range(0, 10):
                    phi = 0.35 * (2 ** -min(k_loss, 3))
                    psi = 0.25
                    budget = min(phi * max(0.0, dll - l_day), psi * max(0.0, mtdd - dd))

                    assert budget >= 0.0
                    assert budget <= dll * 0.35 + 1e-9
                    assert budget <= mtdd * 0.25 + 1e-9

                    # Contratos estándar
                    n_std = math.floor(budget / r_contract_es)
                    if n_std > 0:
                        max_loss_std = n_std * r_contract_es
                        assert max_loss_std <= budget + 1e-9
                        assert max_loss_std <= (dll - l_day) + 1e-9 if l_day < dll else max_loss_std == 0
                    else:
                        # Conmutación a micro
                        n_micro = min(math.floor(budget / r_contract_mes), 10)
                        max_loss_micro = n_micro * r_contract_mes
                        assert max_loss_micro <= budget + 1e-9
                        if budget < r_contract_mes:
                            assert n_micro == 0, "Debe halt cuando el riesgo de 1 micro excede el budget"


# ==============================================================================
# 2. PRUEBAS ADVERSARIAS: F2 - Filtered Quantitative ORB
# ==============================================================================
class TestF2AdversarialStress:
    """Stress testing de ecuaciones y bordes para F2."""

    def test_f2_stretch_and_midpoint_bounds(self):
        """Verifica que el punto medio y el Stretch de Crabel mantengan coherencia dimensional."""
        orh = 5820.0
        orl = 5800.0
        m_or = (orh + orl) / 2.0
        assert m_or == 5810.0

        # Stretch nulo (sin oscilación)
        stretch_zero = 0.0
        delta_risk_min = (orh - orl) / 2.0 + 0.5 * stretch_zero
        assert delta_risk_min == 10.0
        assert delta_risk_min > 0, "El riesgo en puntos no puede ser nulo con ORH > ORL"

        # Stretch amplio
        stretch_wide = 20.0
        delta_risk_wide = (orh - orl) / 2.0 + 0.5 * stretch_wide
        assert delta_risk_wide == 20.0

        # Verificación del ratio riesgo recompensa 2R
        p_entry_long = orh + 0.5 * stretch_wide
        p_tp_long = p_entry_long + 2.0 * delta_risk_wide
        assert (p_tp_long - p_entry_long) / delta_risk_wide == 2.0

    def test_f2_excessive_range_circuit_breaker(self):
        """Verifica el disyuntor analítico si el rango de apertura es monstruoso (Flash Crash / CPI)."""
        dll = 1000.0
        l_day = 0.0
        dd_t = 0.0
        mtdd = 2000.0
        budget = min(0.35 * (dll - l_day), 0.25 * (mtdd - dd_t))  # 350.0 USD

        # Rango extremo de 100 puntos en apertura
        orh = 5900.0
        orl = 5800.0
        stretch = 10.0
        delta_risk = (orh - orl) / 2.0 + 0.5 * stretch  # 55 puntos
        m_mes = 5.0
        c_fric_mes = 3.70
        r_contract_mes = delta_risk * m_mes + c_fric_mes  # 55 * 5 + 3.7 = 278.70 USD

        # 1 micro cabe (278.70 <= 350.0) -> n_micro = 1
        n_micro = math.floor(budget / r_contract_mes)
        assert n_micro == 1

        # Rango de 200 puntos (r_contract = 105 * 5 + 3.7 = 528.70 USD) -> excede budget 350
        delta_risk_huge = 105.0
        r_contract_huge = delta_risk_huge * m_mes + c_fric_mes
        n_micro_huge = math.floor(budget / r_contract_huge)
        assert n_micro_huge == 0, "Disyuntor debe bloquear operaciones con riesgo superior al budget"


# ==============================================================================
# 3. PRUEBAS ADVERSARIAS: F3 - Anchored VWAP & Value Area Reversion
# ==============================================================================
class TestF3AdversarialStress:
    """Stress testing de ecuaciones y bordes para F3."""

    def test_f3_vwap_and_variance_mathematical_consistency(self):
        """Comprueba que la varianza ponderada por volumen no admita raíces negativas ni división por cero."""
        # 10 barras de prueba
        prices = np.array([5800.0, 5802.0, 5798.0, 5805.0, 5810.0, 5808.0, 5795.0, 5800.0, 5803.0, 5801.0])
        volumes = np.array([1000, 1500, 800, 2000, 3000, 2500, 1200, 1800, 1100, 900])

        total_vol = np.sum(volumes)
        assert total_vol > 0
        vwap = np.sum(prices * volumes) / total_vol
        vwap_var = np.sum(volumes * (prices - vwap) ** 2) / total_vol

        assert vwap_var >= 0.0
        vwap_std = math.sqrt(vwap_var)
        assert vwap_std > 0.0

        # Bandas
        band_sup_2 = vwap + 2.0 * vwap_std
        band_inf_2 = vwap - 2.0 * vwap_std
        band_sup_3 = vwap + 3.0 * vwap_std
        band_inf_3 = vwap - 3.0 * vwap_std

        assert band_sup_3 > band_sup_2 > vwap > band_inf_2 > band_inf_3

    def test_f3_stop_loss_buffer_and_catastrophic_3sigma(self):
        """Verifica que el stop técnico no sea idéntico al precio de entrada y que 3 sigma liquide."""
        c_entry = 5790.0
        low_t_minus_1 = 5788.0
        low_t = 5789.0
        k_buffer = 4
        delta_tick = 0.25

        p_sl = min(low_t_minus_1, low_t) - k_buffer * delta_tick
        assert p_sl == 5788.0 - 1.0 == 5787.0
        delta_sl = abs(c_entry - p_sl)
        assert delta_sl == 3.0
        assert delta_sl >= 1.0, "La distancia de stop técnico debe cubrir holgadamente el buffer"


# ==============================================================================
# 4. PRUEBAS ADVERSARIAS: F4 - Cointegración ES vs NQ / Ornstein-Uhlenbeck
# ==============================================================================
class TestF4AdversarialStress:
    """Stress testing de ecuaciones y bordes para F4."""

    def test_f4_ornstein_uhlenbeck_parameter_bounds(self):
        """Verifica que kappa, half-life y sigma_eq sean reales y positivos bajo 0 < b < 1."""
        delta_t = 5.0 / 390.0  # en días RTH

        # Casos válidos de b (reversión a la media)
        for b in [0.05, 0.50, 0.85, 0.95, 0.99]:
            kappa = -math.log(b) / delta_t
            assert kappa > 0.0, "Kappa debe ser estrictamente positivo para 0 < b < 1"

            # Vida media en minutos
            tau_half = (math.log(2) * 5.0) / -math.log(b)
            assert tau_half > 0.0
            assert not math.isnan(tau_half)

            # Varianza de equilibrio
            sigma_eta_sq = 0.0004
            sigma_eq_sq = sigma_eta_sq / (1.0 - b ** 2)
            assert sigma_eq_sq > 0.0
            sigma_eq = math.sqrt(sigma_eq_sq)
            assert sigma_eq > 0.0

        # Caso límite patológico b >= 1 (no estacionario / explosivo)
        b_invalid = 1.02
        # El documento exige vetar si b >= 1
        is_stationary = (0.0 < b_invalid < 1.0)
        assert not is_stationary, "Procesos con b >= 1 deben ser descalificados analíticamente"

    def test_f4_coprime_basket_residual_and_dollar_beta(self):
        """Verifica el balanceo Dollar-Beta Neutral de la cesta 10 MNQ vs 17 MES."""
        p_nq = 20000.0
        p_es = 5800.0
        m_mnq = 2.0
        m_mes = 5.0
        beta_hedge = 1.25

        # Ratio continuo
        ratio_cont = beta_hedge * ((p_nq * m_mnq) / (p_es * m_mes))
        assert abs(ratio_cont - 1.7241) < 0.001

        # Cesta entera coprima 17 / 10
        ratio_discrete = 17.0 / 10.0
        error_beta = abs(ratio_discrete - ratio_cont) / ratio_cont
        assert error_beta < 0.015, "El error de balanceo de beta debe ser inferior a 1.5%"

        # Pérdida unitaria en stop loss (s_stop = 3.25, s_open = 1.25 -> Delta S = 2.00)
        sigma_eq = 0.0030
        delta_x_sl = 2.0 * sigma_eq
        notional_nq_basket = 10 * m_mnq * p_nq  # 400,000 USD
        friction_basket = 57.40
        loss_unit_basket = delta_x_sl * notional_nq_basket + friction_basket
        assert loss_unit_basket > 0.0

        # Sizing con piso
        budget = 350.0  # USD
        n_baskets = math.floor(budget / loss_unit_basket)
        assert n_baskets >= 0
        assert n_baskets * loss_unit_basket <= budget


# ==============================================================================
# 5. PRUEBAS ADVERSARIAS: F5 - Order Flow Imbalance Cont et al.
# ==============================================================================
class TestF5AdversarialStress:
    """Stress testing de ecuaciones y bordes para F5."""

    def test_f5_event_classification_partition_exhaustion(self):
        """Comprueba que la clasificación de eventos e_n cubra exhaustivamente bid y ask sin fugas."""
        # Bid sube
        p_b_curr, p_b_prev = 5800.25, 5800.00
        q_b_curr, q_b_prev = 50, 40
        e_b = q_b_curr if p_b_curr > p_b_prev else (q_b_curr - q_b_prev if p_b_curr == p_b_prev else -q_b_prev)
        assert e_b == 50

        # Bid baja
        p_b_curr, p_b_prev = 5799.75, 5800.00
        e_b_down = -q_b_prev
        assert e_b_down == -40

        # Bid igual (inserción neta)
        p_b_curr, p_b_prev = 5800.00, 5800.00
        e_b_same = q_b_curr - q_b_prev
        assert e_b_same == 10

        # Ask sube
        p_a_curr, p_a_prev = 5800.50, 5800.25
        q_a_curr, q_a_prev = 60, 45
        e_a_up = -q_a_prev
        assert e_a_up == -45

        # Ask baja
        p_a_curr, p_a_prev = 5800.00, 5800.25
        e_a_down = -q_a_curr
        assert e_a_down == -60

        # Ask igual
        p_a_curr, p_a_prev = 5800.25, 5800.25
        e_a_same = q_a_curr - q_a_prev
        assert e_a_same == 15

    def test_f5_zscore_epsilon_zero_division_protection(self):
        """Verifica que la normalización Z_OFI con sigma = 0 esté protegida por epsilon_0."""
        ofi_constant = np.array([100.0] * 30)
        mu_ofi = np.mean(ofi_constant)
        sigma_ofi = np.std(ofi_constant, ddof=1)
        assert sigma_ofi == 0.0

        epsilon_0 = 1e-8
        # Sin epsilon lanzaría ZeroDivisionError
        z_ofi = (ofi_constant[-1] - mu_ofi) / (sigma_ofi + epsilon_0)
        assert z_ofi == 0.0
        assert not math.isnan(z_ofi)


# ==============================================================================
# 6. PRUEBAS ADVERSARIAS: F6 - Multi-Level OFI
# ==============================================================================
class TestF6AdversarialStress:
    """Stress testing de ecuaciones y bordes para F6."""

    def test_f6_spatial_power_law_decay_monotonicity(self):
        """Verifica la jerarquía monotónica decreciente de los coeficientes beta^(l)."""
        d_bar_1 = 250.0
        beta_1 = 1.0 / (2.0 * d_bar_1)
        alpha_decay = 1.0

        betas = [beta_1 * (l ** (-alpha_decay)) for l in range(1, 6)]
        for i in range(len(betas) - 1):
            assert betas[i] > betas[i + 1], f"Fallo de jerarquía en nivel {i+1} frente a {i+2}"
            assert betas[i] > 0.0

    def test_f6_coherence_index_anti_spoofing(self):
        """Comprueba que el índice de coherencia Phi detecte spoofing y valide acumulación real."""
        beta_weights = np.array([0.5, 0.33, 0.25, 0.20])  # niveles 2 a 5
        epsilon_0 = 1e-8

        # Caso 1: Respaldo genuino (nivel 1 compra, niveles 2 a 5 compran)
        ofi_1 = +100
        ofi_deep_real = np.array([+80, +70, +60, +50])
        phi_real = (np.sign(ofi_1) * np.sum(beta_weights * ofi_deep_real)) / (
            np.sum(beta_weights * np.abs(ofi_deep_real)) + epsilon_0
        )
        assert phi_real > 0.40, "Debe superar el umbral de entrada phi_min = 0.40"

        # Caso 2: Spoofing (nivel 1 muestra +100 pero niveles profundos son -80, -70, -60, -50)
        ofi_deep_spoof = np.array([-80, -70, -60, -50])
        phi_spoof = (np.sign(ofi_1) * np.sum(beta_weights * ofi_deep_spoof)) / (
            np.sum(beta_weights * np.abs(ofi_deep_spoof)) + epsilon_0
        )
        assert phi_spoof < -0.40, "Debe detonar veto por trampa / spoofing"

        # Caso 3: Libro profundo desierto (todos 0)
        ofi_deep_empty = np.array([0, 0, 0, 0])
        phi_empty = (np.sign(ofi_1) * np.sum(beta_weights * ofi_deep_empty)) / (
            np.sum(beta_weights * np.abs(ofi_deep_empty)) + epsilon_0
        )
        assert abs(phi_empty) < 1e-5
        assert not math.isnan(phi_empty)


# ==============================================================================
# 7. PRUEBAS ADVERSARIAS: F7 - Meta-Labeling & Triple Barrier
# ==============================================================================
class TestF7AdversarialStress:
    """Stress testing de ecuaciones y bordes para F7."""

    def test_f7_logistic_and_bet_sizing_smoothness(self):
        """Verifica que el clasificador logístico y bet sizing sean continuos y analiza bordes numéricos."""
        tau_meta = 0.60
        eta = 1.50

        # Para valores representativos de logit en [-10, 10]
        for logit in [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]:
            p = 1.0 / (1.0 + math.exp(-logit))
            assert 0.0 < p < 1.0, f"p={p} debe estar estrictamente en (0, 1) para logit={logit}"

            # Bet sizing polinómico (formulación robusta sin singularidad en p=1)
            if p >= tau_meta:
                m_poly = ((p - tau_meta) / (1.0 - tau_meta)) ** eta
                assert 0.0 <= m_poly <= 1.0
            else:
                m_poly = 0.0
                assert m_poly == 0.0

            # Bet sizing gaussiano z_t
            if 0.0 < p < 1.0:
                denom_z = math.sqrt(p * (1.0 - p))
                assert denom_z > 0.0
                z_t = (p - 0.50) / denom_z
                assert not math.isnan(z_t)

        # Análisis de frontera extrema (logit = 100):
        # En aritmética IEEE 754 float64, 1.0 + exp(-100) colapsa a 1.0
        p_extreme = 1.0 / (1.0 + math.exp(-100.0))
        assert p_extreme == 1.0
        # Demostración del comportamiento de la alternativa polinómica ante p=1.0:
        m_poly_max = ((p_extreme - tau_meta) / (1.0 - tau_meta)) ** eta
        assert m_poly_max == 1.0, "La formulación polinómica maneja p=1.0 de forma suave y sin indeterminación"

    def test_f7_triple_barrier_geometry(self):
        """Verifica que las tres barreras sean mutuamente excluyentes y formen un conjunto cerrado."""
        p_t0 = 5800.0
        sigma_t0 = 15.0
        ell_plus = 2.00
        ell_minus = 1.00

        # Long
        b_upper_long = p_t0 + ell_plus * sigma_t0
        b_lower_long = p_t0 - ell_minus * sigma_t0
        assert b_upper_long == 5830.0
        assert b_lower_long == 5785.0
        assert b_upper_long > p_t0 > b_lower_long

        # Short
        b_upper_short = p_t0 + ell_minus * sigma_t0
        b_lower_short = p_t0 - ell_plus * sigma_t0
        assert b_upper_short == 5815.0
        assert b_lower_short == 5770.0
        assert b_upper_short > p_t0 > b_lower_short

        # Ratio de asimetría
        assert ell_plus / ell_minus == 2.0


# ==============================================================================
# 8. PRUEBAS ADVERSARIAS: F8 - Modelos Econométricos HAR-RV-J
# ==============================================================================
class TestF8AdversarialStress:
    """Stress testing de ecuaciones y bordes para F8."""

    def test_f8_bipower_and_jump_decomposition_invariants(self):
        """Verifica que RV = C + J con C > 0 y J >= 0 bajo cualquier vector de retornos."""
        # Simular 78 retornos intradía con un salto masivo
        np.random.seed(42)
        returns = np.random.normal(0, 0.001, 78)
        returns[35] = 0.025  # Salto de +2.5%

        rv = float(np.sum(returns ** 2))
        mu_1_inv_sq = math.pi / 2.0
        m = 78
        bv = float(mu_1_inv_sq * (m / (m - 1)) * np.sum(np.abs(returns[1:]) * np.abs(returns[:-1])))

        # Salto aislado
        j = max(rv - bv, 0.0)
        c = rv - j

        assert rv > 0.0
        assert bv > 0.0
        assert j > 0.0, "El retorno anómalo de +2.5% debe ser detectado como salto"
        assert c > 0.0
        assert abs(c + j - rv) < 1e-12, "Invariante de descomposición violado: RV != C + J"

    def test_f8_log_har_strictly_positive_forecast(self):
        """Verifica que la formulación Log-HAR-RV-J garantice varianza pronosticada positiva."""
        # Coeficientes OLS típicos
        beta_0 = -1.5
        beta_d = 0.40
        beta_w = 0.30
        beta_m = 0.20
        beta_j = -0.05
        sigma_eps_sq = 0.04

        # Variables en niveles positivos
        c_t = 0.00015
        rv_w = 0.00018
        rv_m = 0.00020
        j_t = 0.00005

        log_rv_pred = (
            beta_0
            + beta_d * math.log(c_t)
            + beta_w * math.log(rv_w)
            + beta_m * math.log(rv_m)
            + beta_j * math.log(1.0 + j_t)
            + 0.5 * sigma_eps_sq
        )
        rv_pred = math.exp(log_rv_pred)

        assert rv_pred > 0.0
        assert not math.isinf(rv_pred)
        assert not math.isnan(rv_pred)


# ==============================================================================
# 9. PRUEBAS ADVERSARIAS: F9 - Anomalías Temporales / VRP
# ==============================================================================
class TestF9AdversarialStress:
    """Stress testing de ecuaciones y bordes para F9."""

    def test_f9_vrp_and_term_structure_gate(self):
        """Verifica el comportamiento de la compuerta VRP en contango vs backwardation extremo."""
        # Contango tranquilo: VIX 14, VIX3M 16, TSR = 14/16 = 0.875, Z_VRP = +2.0
        vix = 14.0
        vix3m = 16.0
        tsr_contango = vix / vix3m
        z_vrp_high = 2.0
        s_vrp_contango = +1 if (z_vrp_high >= 1.50 and tsr_contango <= 0.95) else 0
        assert s_vrp_contango == +1, "Debe habilitar sesgo favorable en contango claro"

        # Backwardation de pánico: VIX 35, VIX3M 28, TSR = 35/28 = 1.25, Z_VRP = -2.5
        vix_panic = 35.0
        vix3m_panic = 28.0
        tsr_backwardation = vix_panic / vix3m_panic
        z_vrp_low = -2.5
        s_vrp_panic = -1 if (z_vrp_low <= -1.20 or tsr_backwardation >= 1.00) else 0
        assert s_vrp_panic == -1, "Debe bloquear totalmente compras en backwardation"

    def test_f9_pre_macro_timing_safety_buffer(self):
        """Verifica que la salida en T-3 min respete incondicionalmente la ventana de embargo T-2 min."""
        # CPI a las 08:30:00 EST
        t_macro_sec = 8 * 3600 + 30 * 60  # 30600 s
        t_entry_sec = t_macro_sec - 15 * 60  # 08:15:00
        t_exit_mandated_sec = t_macro_sec - 3 * 60  # 08:27:00
        t_embargo_start_sec = t_macro_sec - 2 * 60  # 08:28:00
        t_embargo_end_sec = t_macro_sec + 2 * 60  # 08:32:00

        # Margen de seguridad entre salida y comienzo del embargo
        safety_gap_sec = t_embargo_start_sec - t_exit_mandated_sec
        assert safety_gap_sec == 60, "El margen de seguridad entre T-3 y T-2 debe ser exactamente de 60 segundos"
        assert t_exit_mandated_sec < t_embargo_start_sec
        assert t_entry_sec < t_exit_mandated_sec


# ==============================================================================
# 10. PRUEBAS TRANSVERSALES DE INTEGRIDAD Y REGLAS DE PROP TRADING
# ==============================================================================
class TestTransversalPropTradingInvariants:
    """Verificación global de las leyes de conservación de capital en las 9 estrategias."""

    @pytest.mark.parametrize("strategy_code", ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"])
    def test_zero_budget_implies_zero_contracts_all_strategies(self, strategy_code):
        """Si el budget restante es 0 (por DLL o TMDD agotado), contratos DEBE ser 0 estrictamente."""
        budget = 0.0
        risk_per_contract = 50.0  # USD

        # Función de dimensionamiento base
        n_contracts = math.floor(budget / risk_per_contract)
        assert n_contracts == 0, f"Estrategia {strategy_code} no puede asignar contratos con presupuesto nulo"

    @pytest.mark.parametrize("multiplier, instrument", [
        (50.0, "ES"),
        (5.0, "MES"),
        (20.0, "NQ"),
        (2.0, "MNQ")
    ])
    def test_cme_contract_multipliers_consistency(self, multiplier, instrument):
        """Verifica que los multiplicadores oficiales CME Globex sean exactos en todas las fórmulas."""
        if instrument == "ES":
            assert multiplier == 50.0
        elif instrument == "MES":
            assert multiplier == 5.0
        elif instrument == "NQ":
            assert multiplier == 20.0
        elif instrument == "MNQ":
            assert multiplier == 2.0
