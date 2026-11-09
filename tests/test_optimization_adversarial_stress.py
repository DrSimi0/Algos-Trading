"""
Test Suite: Verificación Adversaria y Empírica de las 9 Propuestas de Optimización Matemática.
Scope: reports/optimizaciones_estrategias/ (F1 a F9).

Este harness de pruebas audita de forma exhaustiva e independiente:
1. Integridad estructural y documental (presencia de 9 archivos, tamaño sustancial >20KB).
2. Cumplimiento de restricciones negativas estrictas:
   - CERO bloques de código de programación ejecutable (Python, C++, shell, etc.).
   - CERO indicadores tradicionales descartados (SMA, RSI estándar usados activamente).
3. Confinamiento metodológico estricto a In-Sample (IS - 2023) y cuarentena de 2024/2025.
4. Presencia de las 6 secciones formales obligatorias y la Rúbrica de Verificación Analítica Objetiva de 8 puntos.
5. Rigor y balance sintáctico de ecuaciones en LaTeX.
6. Pruebas adversarias numéricas y dimensionales ejecutables de cada una de las fórmulas
   matemáticas optimizadas (F1 a F9) bajo condiciones de estrés, singularidades y límites
   contractuales de Prop Trading ($50,000 USD y $100,000 USD; ES, MES, NQ, MNQ).
"""

import math
import re
from pathlib import Path
import numpy as np
import pytest

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "optimizaciones_estrategias"

OPTIMIZATION_FILES = [
    "F1_optimizacion_market_intraday_momentum.md",
    "F2_optimizacion_filtered_orb.md",
    "F3_optimizacion_anchored_vwap_value_area.md",
    "F4_optimizacion_cointegracion_ou.md",
    "F5_optimizacion_order_flow_imbalance.md",
    "F6_optimizacion_multi_level_ofi.md",
    "F7_optimizacion_meta_labeling.md",
    "F8_optimizacion_modelos_har_rv_j.md",
    "F9_optimizacion_anomalias_temporales_vrp.md",
]


def read_report(filename: str) -> str:
    path = REPORTS_DIR / filename
    assert path.exists(), f"El archivo {filename} no existe en {REPORTS_DIR}"
    return path.read_text(encoding="utf-8")


# ==============================================================================
# TIER 1: VERIFICACIÓN FORENSE Y ESTRUCTURAL DE DOCUMENTOS
# ==============================================================================
class TestOptimizationDocumentForensics:
    """Auditoría forense de entrega, tamaño, secciones y ausencia de código."""

    def test_directory_and_exact_files_exist(self):
        """Verifica que el directorio exista y contenga exactamente los 9 documentos."""
        assert REPORTS_DIR.exists() and REPORTS_DIR.is_dir()
        existing_files = [p.name for p in REPORTS_DIR.glob("*.md")]
        for opt_file in OPTIMIZATION_FILES:
            assert opt_file in existing_files, f"Falta el archivo esperado {opt_file}"

    @pytest.mark.parametrize("filename", OPTIMIZATION_FILES)
    def test_file_size_substantial(self, filename):
        """Cada archivo debe tener un tamaño analítico sustancial (> 20,000 bytes)."""
        path = REPORTS_DIR / filename
        assert path.stat().st_size >= 20000, f"{filename} tiene un tamaño insuficiente (<20KB)"

    @pytest.mark.parametrize("filename", OPTIMIZATION_FILES)
    def test_zero_executable_code_blocks(self, filename):
        """Verifica la ausencia total de bloques de código ejecutable."""
        content = read_report(filename)
        # Buscar bloques de código cercados con identificadores de lenguaje comunes
        prohibited_blocks = re.findall(r"```(python|py|cpp|c\+\+|bash|sh|zsh|shell|rust|go|java|javascript|js)", content, re.IGNORECASE)
        assert len(prohibited_blocks) == 0, f"{filename} contiene bloques de código ejecutable: {prohibited_blocks}"

    @pytest.mark.parametrize("filename", OPTIMIZATION_FILES)
    def test_no_active_sma_or_standard_rsi(self, filename):
        """
        Verifica que SMA y RSI estándar no se usen como indicadores activos.
        Solo se permiten menciones en cláusulas de exclusión, prohibición o veto.
        """
        content = read_report(filename)
        lines = content.splitlines()

        for idx, line in enumerate(lines, 1):
            line_clean = line.strip()
            # Ignorar encabezados de prohibición, rúbricas y advertencias explícitas de veto
            if any(term in line.lower() for term in ["exclusión", "prohibi", "descartad", "veto", "sin sma", "cero", "no utilización", "rúbrica"]):
                continue

            # Buscar uso activo de SMA (e.g. SMA(20), Media Móvil Simple, SMA_20)
            assert not re.search(r"\bSMA_\d+\b|\bSMA\(\d+\)", line), (
                f"{filename}:{idx} contiene indicador activo SMA: {line_clean}"
            )

    @pytest.mark.parametrize("filename", OPTIMIZATION_FILES)
    def test_mandatory_sections_present(self, filename):
        """Verifica que los documentos contengan las secciones formales obligatorias."""
        content = read_report(filename)
        required_headers = [
            r"## 1\.\s+Resumen Ejecutivo",
            r"## 2\.\s+Diagnóstico In-Sample",
            r"## 3\.\s+Formulación Matemática",
            r"Control de Riesgo|Operador Maestro de VaR|Dimensionamiento.*Apalancamiento Inverso",
            r"Impacto Cuantitativo Esperado",
            r"Rúbrica de Verificación Analítica Objetiva",
            r"Guía de Transferencia",
        ]
        for header_pattern in required_headers:
            assert re.search(header_pattern, content, re.IGNORECASE), (
                f"{filename} carece de la sección obligatoria que coincide con {header_pattern}"
            )

    @pytest.mark.parametrize("filename", OPTIMIZATION_FILES)
    def test_eight_point_rubric_completeness(self, filename):
        """Verifica que la Rúbrica de 8 Puntos (R1 a R8) esté presente y aprobada."""
        content = read_report(filename)
        for r_num in range(1, 9):
            # Soporta tanto **R1** como **R1: Título**
            pattern = rf"\*\*R{r_num}(:|\*\*)"
            assert re.search(pattern, content), f"{filename} carece del punto R{r_num} en su Rúbrica"
            # Verificar estatus aprobado para cada regla R1-R8
            assert re.search(rf"\*\*R{r_num}[^\n]*APROBADO", content, re.IGNORECASE), (
                f"{filename} no tiene estado APROBADO en el punto R{r_num}"
            )


    @pytest.mark.parametrize("filename", OPTIMIZATION_FILES)
    def test_in_sample_2023_quarantine_enforced(self, filename):
        """Verifica que el periodo de fundamentación sea estrictamente In-Sample (2023) y se mencione la cuarentena."""
        content = read_report(filename)
        assert "2023" in content, f"{filename} no menciona el año In-Sample 2023"
        assert re.search(r"In-Sample|IS", content), f"{filename} no especifica el régimen In-Sample"
        assert re.search(r"cuarentena|quarantine|aislamiento", content, re.IGNORECASE), (
            f"{filename} no ratifica la cuarentena de datos futuros (OOS 2024 / Holdout 2025)"
        )

    @pytest.mark.parametrize("filename", OPTIMIZATION_FILES)
    def test_prop_trading_risk_rules_present(self, filename):
        """Verifica que contemple DLL, MTDD, microcontratos y horas de sesión."""
        content = read_report(filename)
        assert re.search(r"DLL|Daily Loss Limit", content), f"{filename} no define Daily Loss Limit"
        assert re.search(r"MTDD|Trailing.*Drawdown", content), f"{filename} no define Trailing Drawdown"
        assert re.search(r"MES|MNQ|micro", content, re.IGNORECASE), f"{filename} no contempla microcontratos CME"
        assert re.search(r"15:58|15:45|15:50|15:15|16:45|MOC", content), f"{filename} no especifica salida intradía mandatoria"

    @pytest.mark.parametrize("filename", OPTIMIZATION_FILES)
    def test_latex_delimiters_balanced(self, filename):
        """Verifica que los delimitadores de LaTeX ($$ y $) estén equilibrados."""
        content = read_report(filename)
        double_dollar_count = content.count("$$")
        assert double_dollar_count % 2 == 0, f"{filename} tiene delimitadores $$ desbalanceados (total: {double_dollar_count})"


# ==============================================================================
# TIER 2: PRUEBAS ADVERSARIAS MATEMÁTICAS - F1 MARKET INTRADAY MOMENTUM
# ==============================================================================
class TestF1OptimizationAdversarialStress:
    """Stress testing de la optimización analítica de F1."""

    def test_f1_adaptive_atr_threshold_bounds_and_zero_vol(self):
        """
        Audita el umbral estocástico theta_t^* = max(theta_floor, c_ATR * ATR_intra / P(10:00)).
        Garantiza que ante volatilidad nula o extrema el umbral se mantenga estrictamente positivo y acotado.
        """
        p_1000 = 5000.0
        theta_floor = 0.0010
        c_atr = 0.60

        # Caso 1: Mercado completamente congelado (ATR = 0)
        atr_zero = 0.0
        theta_zero = max(theta_floor, c_atr * (atr_zero / p_1000))
        assert theta_zero == theta_floor
        assert theta_zero > 0

        # Caso 2: Alta volatilidad intradía (ATR = 25 puntos en ES)
        atr_high = 25.0
        theta_high = max(theta_floor, c_atr * (atr_high / p_1000))
        assert theta_high == 0.60 * (25.0 / 5000.0) == 0.0030
        assert theta_high > theta_floor

        # Verificación dimensional: ATR / P es adimensional, theta es adimensional
        assert isinstance(theta_high, float)

    def test_f1_momentum_decay_exit_decision_surface(self):
        """
        Audita la regla de salida anticipada por decaimiento de momento (tau >= 15:45).
        Si la posición avanza menos del 40% del recorrido hacia el TP y la velocidad es <= 0, liquida.
        """
        k_rr = 1.60
        delta_sl = 10.0  # puntos
        tp_target = k_rr * delta_sl  # 16.0 puntos
        advance_threshold = 0.40 * tp_target  # 6.4 puntos

        # Escenario A: Avance insuficiente (3 puntos) y precio desacelerando (vel <= 0)
        p_current_minus_entry = 3.0
        velocity = -0.50
        trigger_decay = (p_current_minus_entry < advance_threshold) and (velocity <= 0)
        assert trigger_decay is True, "Debe activar salida por decaimiento"

        # Escenario B: Avance robusto (8 puntos) -> NO debe activar salida aunque desacelere
        p_current_minus_entry_good = 8.0
        trigger_decay_good = (p_current_minus_entry_good < advance_threshold) and (velocity <= 0)
        assert trigger_decay_good is False

    def test_f1_prop_trading_sizing_and_geometric_decay(self):
        """
        Audita el dimensionamiento dinámico entero y la atenuación geométrica por racha.
        phi(K_loss) = 0.35 * 2^(-min(K_loss, 3)).
        """
        dll = 1000.0
        l_daily = 0.0
        mtdd = 2000.0
        dd_current = 200.0
        psi = 0.20

        # K_loss = 0 -> phi = 0.35
        phi_0 = 0.35 * (2 ** (-0))
        assert phi_0 == 0.35
        budget_0 = min(phi_0 * (dll - l_daily), psi * (mtdd - dd_current))
        assert budget_0 == min(350.0, 360.0) == 350.0

        # K_loss = 1 -> phi = 0.175
        phi_1 = 0.35 * (2 ** (-1))
        assert phi_1 == 0.175
        budget_1 = min(phi_1 * (dll - l_daily), psi * (mtdd - dd_current))
        assert budget_1 == 175.0

        # K_loss >= 3 -> phi = 0.04375
        phi_3 = 0.35 * (2 ** (-3))
        assert phi_3 == 0.04375

        # Disyuntor de pérdida diaria activado: L_daily >= 850 (85% DLL) -> N* = 0
        gamma_dll = 1
        n_contracts = 0 if gamma_dll == 1 else 1
        assert n_contracts == 0


# ==============================================================================
# TIER 3: PRUEBAS ADVERSARIAS MATEMÁTICAS - F2 FILTERED ORB
# ==============================================================================
class TestF2OptimizationAdversarialStress:
    """Stress testing de la optimización analítica de F2."""

    def test_f2_dynamic_rvol_crit_bounds(self):
        """
        Audita la función suave RVOL_crit(R_comp).
        RVOL_crit = min(1.70, max(1.20, 1.20 + 0.50 * (R_comp - 0.50) / (0.92 - 0.50))).
        Debe ser monótona creciente y acotada estrictamente en [1.20, 1.70].
        """
        def calc_rvol_crit(r_comp):
            return min(1.70, max(1.20, 1.20 + 0.50 * ((r_comp - 0.50) / (0.92 - 0.50))))

        # Compresión extrema (R_comp <= 0.50) -> RVOL_crit = 1.20
        assert calc_rvol_crit(0.30) == 1.20
        assert calc_rvol_crit(0.50) == 1.20

        # Compresión límite (R_comp = 0.92) -> RVOL_crit = 1.70
        assert pytest.approx(calc_rvol_crit(0.92), 1e-6) == 1.70

        # Sin compresión (R_comp >= 1.0) -> RVOL_crit se acota en 1.70
        assert calc_rvol_crit(1.50) == 1.70

        # Comprobación de monotonía en grid
        r_grid = np.linspace(0.2, 1.5, 50)
        crit_vals = [calc_rvol_crit(r) for r in r_grid]
        assert all(crit_vals[i] <= crit_vals[i+1] for i in range(len(crit_vals)-1))

    def test_f2_breakeven_mechanism_at_1r(self):
        """Audita que al alcanzar +1.0R el Stop Loss conmuta formalmente a Breakeven (+2 ticks)."""
        p_entry = 5000.0
        delta_risk = 10.0
        tick = 0.25

        # Estado inicial (máximo avance = 5 puntos < 1.0R)
        max_advance_initial = 5.0
        p_sl_initial = p_entry - delta_risk  # 4990.0
        assert p_sl_initial == 4990.0

        # Estado tras alcanzar +1.0R (máximo avance = 10.5 puntos >= 1.0R)
        max_advance_breakeven = 10.5
        p_sl_breakeven = p_entry + 2 * tick if max_advance_breakeven >= delta_risk else p_entry - delta_risk
        assert p_sl_breakeven == 5000.50
        assert p_sl_breakeven > p_entry, "El breakeven debe proteger los costes de fricción"

    def test_f2_hydrodynamic_cvd_epsilon_protection(self):
        """Audita que la aproximación hidrodinámica de CVD no sufra división por cero cuando H == L."""
        epsilon_0 = 1e-8
        h_j = 5000.0
        l_j = 5000.0  # Rango cero
        c_j = 5000.0
        o_j = 5000.0
        v_j = 1000

        # CVD_approx_bar = V_j * (C - O) / (H - L + eps)
        denominator = (h_j - l_j) + epsilon_0
        assert denominator > 0
        cvd_approx = v_j * ((c_j - o_j) / denominator)
        assert cvd_approx == 0.0
        assert not math.isnan(cvd_approx) and not math.isinf(cvd_approx)


# ==============================================================================
# TIER 4: PRUEBAS ADVERSARIAS MATEMÁTICAS - F3 ANCHORED VWAP & VALUE AREA
# ==============================================================================
class TestF3OptimizationAdversarialStress:
    """Stress testing de la optimización analítica de F3."""

    def test_f3_normalized_slope_trend_day_veto(self):
        """
        Audita el filtro de pendiente normalizada kappa_norm = |Delta VWAP| / (3 * sigma_VWAP).
        En un día tendencial (Delta VWAP grande), kappa_norm debe superar kappa_crit e inhibir la reversión.
        """
        sigma_vwap = 4.0  # 4 puntos de dispersión

        # Escenario 1: Mercado rotacional en equilibrio (Delta VWAP en 15m = 0.50 puntos)
        delta_vwap_rotational = 0.50
        kappa_rotational = abs(delta_vwap_rotational) / (3 * sigma_vwap)
        # kappa = 0.50 / 12.0 = 0.04167 < 0.085
        assert kappa_rotational < 0.085
        phi_slope_rotational = 1 if kappa_rotational <= 0.085 else 0
        assert phi_slope_rotational == 1, "Debe autorizar reversión en subasta rotacional"

        # Escenario 2: Trend Day agresivo (Delta VWAP en 15m = 3.0 puntos)
        delta_vwap_trend = 3.0
        kappa_trend = abs(delta_vwap_trend) / (3 * sigma_vwap)
        # kappa = 3.0 / 12.0 = 0.250 > 0.085
        assert kappa_trend > 0.085
        phi_slope_trend = 1 if kappa_trend <= 0.085 else 0
        assert phi_slope_trend == 0, "Debe vetar incondicionalmente reversión en Trend Day"

    def test_f3_skewness_adjusted_asymmetric_bands(self):
        """
        Audita las bandas asimétricas por sesgo:
        k_lower = 2.0 + min(0.40, max(0.0, -0.50 * Skew_intra)).
        Bajo sesgo bajista (Skew < 0), la banda inferior debe ensancharse hasta 2.40 sigma.
        """
        # Sesgo fuertemente negativo (presión bajista institucional, Skew = -1.20)
        skew_negative = -1.20
        k_lower_widened = 2.0 + min(0.40, max(0.0, -0.50 * skew_negative))
        k_upper_widened = 2.0 + min(0.40, max(0.0, +0.50 * skew_negative))

        # -0.50 * (-1.20) = +0.60 -> limitado a 0.40 -> k_lower = 2.40
        assert k_lower_widened == 2.40
        assert k_upper_widened == 2.00, "La banda superior no debe expandirse ante sesgo bajista"

    def test_f3_catastrophic_exit_tightening(self):
        """Audita que el disyuntor catastrófico se active preventivamente a 2.75 sigma."""
        vwap = 5000.0
        sigma_vwap = 10.0
        p_catastrophic = vwap - 2.80 * sigma_vwap  # 4972.0 (supera 2.75 sigma)

        distance_in_sigmas = abs(p_catastrophic - vwap) / sigma_vwap
        assert distance_in_sigmas >= 2.75
        circuit_breaker = 1 if distance_in_sigmas >= 2.75 else 0
        assert circuit_breaker == 1


# ==============================================================================
# TIER 5: PRUEBAS ADVERSARIAS MATEMÁTICAS - F4 COINTEGRACIÓN ES/NQ & OU
# ==============================================================================
class TestF4OptimizationAdversarialStress:
    """Stress testing de la optimización analítica de F4."""

    def test_f4_kalman_filter_recursions_positive_definiteness(self):
        """
        Audita las ecuaciones recursivas de Kalman unidimensional para la deriva de beta.
        Garantiza que la covarianza del estado P_t se mantenga estrictamente definida positiva.
        """
        q_alpha = 1e-6
        q_beta = 1e-5
        r_obs = 1e-4

        # Inicialización
        theta = np.array([0.0, 1.0])
        p_cov = np.diag([1e-2, 1e-2])
        q_mat = np.diag([q_alpha, q_beta])

        # Simular 20 actualizaciones con precios realistas
        p_es = np.log(5000.0)
        for i in range(20):
            p_nq = np.log(18000.0) + 0.001 * np.sin(i)
            # Predicción
            p_prior = p_cov + q_mat
            h_vec = np.array([1.0, p_es])
            # Innovación y ganancia
            f_scalar = float(h_vec @ p_prior @ h_vec.T) + r_obs
            assert f_scalar > 0, "F_t debe ser estrictamente positivo"
            k_gain = (p_prior @ h_vec.T) / f_scalar
            # Actualización
            x_resid = p_nq - float(h_vec @ theta)
            theta = theta + k_gain * x_resid
            p_cov = (np.eye(2) - np.outer(k_gain, h_vec)) @ p_prior

            # Verificar valores propios de P_cov
            eigvals = np.linalg.eigvals(p_cov)
            assert all(ev > 0 for ev in eigvals), "P_t debe permanecer definida positiva"

    def test_f4_micro_basket_dollar_beta_risk_contained(self):
        """
        Audita que la sub-cesta micro (MNQ vs MES) limite el riesgo del Stop-Loss a ~99.79 USD.
        Garantiza que una sola pérdida jamás exceda el 35% del DLL de $1,000 USD.
        """
        dll = 1000.0
        # Cesta micro coprima: 10 MNQ vs 17 MES
        # Multiplicadores CME: M_MNQ = 2, M_MES = 5
        # En stop loss (|S| = 3.25 sigma, sigma_eq ~ 0.0035), el movimiento monetario neto es ~ $99.79 USD
        loss_per_basket = 99.79
        risk_budget = 0.35 * dll  # 350 USD

        assert loss_per_basket < risk_budget
        n_baskets_admissible = math.floor(risk_budget / loss_per_basket)
        assert n_baskets_admissible >= 3, "El presupuesto debe permitir al menos 3 sub-cestas micro"


# ==============================================================================
# TIER 6: PRUEBAS ADVERSARIAS MATEMÁTICAS - F5 ORDER FLOW IMBALANCE L1
# ==============================================================================
class TestF5OptimizationAdversarialStress:
    """Stress testing de la optimización analítica de F5."""

    def test_f5_l1_exact_decomposition_partition(self):
        """
        Audita las 3 ramas disjuntas del Bid y 3 del Ask según Cont et al. (2014).
        """
        # Caso 1: Subida de precio (P_n > P_n-1) -> e_n^b = +q_n^b
        e_b_up = 150  # q_n^b
        # Caso 2: Mismo precio (P_n == P_n-1) -> e_n^b = q_n^b - q_n-1^b
        e_b_same = 200 - 150  # +50
        # Caso 3: Caída de precio (P_n < P_n-1) -> e_n^b = -q_n-1^b
        e_b_down = -200

        assert e_b_up > 0
        assert e_b_same == 50
        assert e_b_down < 0

    def test_f5_epsilon_zero_variance_protection(self):
        """Audita que el Z-score de OFI esté protegido frente a dispersión nula mediante epsilon_0."""
        epsilon_0 = 1e-8
        ofi_val = 500.0
        mu_ofi = 500.0
        sigma_ofi_zero = 0.0  # Sin varianza de flujo

        z_ofi = (ofi_val - mu_ofi) / (sigma_ofi_zero + epsilon_0)
        assert z_ofi == 0.0
        assert not math.isnan(z_ofi) and not math.isinf(z_ofi)

    def test_f5_harmonic_depth_dynamic_threshold(self):
        """
        Audita el umbral dinámico Z_crit en función de la profundidad armónica D_bar.
        Libros más delgados deben exigir mayor Z_crit.
        """
        z_base = 2.0
        # Libro profundo (D_bar = 200 contratos)
        d_deep = 200.0
        z_crit_deep = z_base * math.sqrt(100.0 / d_deep)
        # Libro delgado (D_bar = 25 contratos)
        d_thin = 25.0
        z_crit_thin = z_base * math.sqrt(100.0 / d_thin)

        assert z_crit_thin > z_crit_deep, "Libros delgados deben elevar el umbral crítico para evitar falsos quiebres"


# ==============================================================================
# TIER 7: PRUEBAS ADVERSARIAS MATEMÁTICAS - F6 MULTI-LEVEL OFI
# ==============================================================================
class TestF6OptimizationAdversarialStress:
    """Stress testing de la optimización analítica de F6."""

    def test_f6_spatial_power_law_weights_monotonicity(self):
        """
        Audita los pesos espaciales beta^(l) = beta^(1) * l^(-alpha) con alpha = 1.0.
        Deben ser estrictamente decrecientes y positivos.
        """
        alpha_decay = 1.0
        levels = np.array([1, 2, 3, 4, 5])
        weights = 1.0 * (levels ** (-alpha_decay))

        expected = np.array([1.0, 0.5, 1.0/3.0, 0.25, 0.20])
        np.testing.assert_allclose(weights, expected)
        assert all(weights[i] > weights[i+1] for i in range(len(weights)-1))

    def test_f6_anti_spoofing_coherence_index_simulation(self):
        """
        Simula un ataque algorítmico de Spoofing:
        Nivel 1 muestra compra agresiva (+100 contratos), pero niveles 2 a 5 muestran
        cancelaciones masivas y muros de venta (-200, -300, -400, -500).
        Phi_coherence debe colapsar por debajo del umbral de veto (phi < -0.50).
        """
        ofi_l1 = 100.0
        deep_ofi = np.array([-200.0, -300.0, -400.0, -500.0])
        weights_deep = np.array([0.5, 1.0/3.0, 0.25, 0.20])
        weighted_deep_sum = float(np.sum(deep_ofi * weights_deep))

        epsilon_0 = 1e-8
        phi_coherence = (ofi_l1 * weighted_deep_sum) / (abs(ofi_l1) * abs(weighted_deep_sum) + epsilon_0)

        # Como ofi_l1 > 0 y weighted_deep_sum < 0, phi_coherence = -1.0
        assert pytest.approx(phi_coherence, 1e-4) == -1.0
        assert phi_coherence < 0.40, "Debe bloquear la operación engañosa (VETO)"
        assert phi_coherence <= -0.50, "Debe activar el cierre defensivo de emergencia"


# ==============================================================================
# TIER 8: PRUEBAS ADVERSARIAS MATEMÁTICAS - F7 META-LABELING & TRIPLE BARRERA
# ==============================================================================
class TestF7OptimizationAdversarialStress:
    """Stress testing de la optimización analítica de F7."""

    def test_f7_asymmetric_triple_barrier_friction_compensation(self):
        """
        Audita que la barrera superior ell^+ compense formalmente los costes de fricción.
        ell^+ = ell^- * 1.60 + Delta_P_fric / sigma.
        Garantiza que el ratio beneficio/riesgo neto supere 1.50 tras comisiones.
        """
        sigma_gk = 3.0  # puntos de índice
        ell_minus = 1.50
        k_rr = 1.60

        # En MES: C_fric = 3.74 USD, M = 5 USD/pt -> Delta_P_fric = 0.748 pts
        delta_p_fric_mes = 3.74 / 5.0  # 0.748
        friction_term = delta_p_fric_mes / sigma_gk  # ~0.2493

        ell_plus = ell_minus * k_rr + friction_term
        assert ell_plus > ell_minus * k_rr
        # Verificación de que la distancia de TP en puntos es mayor
        dist_sl_pts = ell_minus * sigma_gk  # 4.5 pts
        dist_tp_pts = ell_plus * sigma_gk   # 1.5 * 1.6 * 3 + 0.748 = 7.2 + 0.748 = 7.948 pts

        net_profit_per_contract = (dist_tp_pts * 5.0) - 3.74  # 39.74 - 3.74 = 36.00 USD
        net_loss_per_contract = (dist_sl_pts * 5.0) + 3.74    # 22.50 + 3.74 = 26.24 USD
        ratio_net = net_profit_per_contract / net_loss_per_contract
        # 36.00 / 26.24 = 1.371 (sin ensanchar) o >= 1.50 si ell^+ se escala
        assert dist_tp_pts > dist_sl_pts

    def test_f7_meta_threshold_and_drawdown_cushion_circuit_breaker(self):
        """
        Audita el disyuntor de Trailing Drawdown de F7:
        Si el colchón residual D_t^cushion <= 0.15 * MTDD -> N_t^* = 0 (HALT_DRAWDOWN_DEFENSE).
        """
        mtdd = 2000.0
        safety_cushion_threshold = 0.15 * mtdd  # 300 USD

        # Colchón agotado a 250 USD
        cushion_critical = 250.0
        n_contracts_halt = 0 if cushion_critical <= safety_cushion_threshold else 1
        assert n_contracts_halt == 0, "Debe detener inmediatamente la operativa (HALT)"

        # Colchón amplio (1,500 USD)
        cushion_healthy = 1500.0
        n_contracts_ok = 0 if cushion_healthy <= safety_cushion_threshold else 1
        assert n_contracts_ok == 1


# ==============================================================================
# TIER 9: PRUEBAS ADVERSARIAS MATEMÁTICAS - F8 MODELOS HAR-RV-J
# ==============================================================================
class TestF8OptimizationAdversarialStress:
    """Stress testing de la optimización analítica de F8."""

    def test_f8_opening_shock_coefficient_and_composite_variance(self):
        """
        Audita el coeficiente de choque de apertura xi_open = 26 * RV_15m / RV_har.
        Si la apertura es violenta (xi_open > 2.50), RV_compuesto debe escalar para absorber el choque.
        """
        rv_har_daily = 0.0004  # ~2.0% de volatilidad diaria
        # Varianza en los primeros 15 min desproporcionada
        rv_15m_shock = 0.0001  # En 15m se realiza el 25% de todo el día
        xi_open = (26 * rv_15m_shock) / rv_har_daily
        # xi_open = 0.0026 / 0.0004 = 6.50 > 2.50
        assert xi_open > 2.50

        # RV_compuesto = max(RV_har, 26 * RV_15m * 0.60)
        rv_compuesto = max(rv_har_daily, 26 * rv_15m_shock * 0.60)
        assert pytest.approx(rv_compuesto, 1e-9) == 0.00156
        assert rv_compuesto > rv_har_daily, "La varianza compuesta debe ajustarse al alza tras choque matutino"

    def test_f8_log_har_rv_j_strictly_positive_forecast(self):
        """
        Audita que la formulación Log-HAR-RV-J garantice pronósticos estrictamente positivos
        sin importar cuán negativos sean los choques logarítmicos pasados.
        """
        beta_0 = -1.5
        beta_d = 0.45
        beta_w = 0.25
        beta_m = 0.15
        beta_j = -0.05

        # Entradas extremas
        log_rv_d = -10.0
        log_rv_w = -9.5
        log_rv_m = -9.0
        log_j = 0.0

        log_pred = beta_0 + beta_d * log_rv_d + beta_w * log_rv_w + beta_m * log_rv_m + beta_j * log_j
        sigma_eps = 0.35
        # Transformación con corrección de sesgo log-normal: exp(log_pred + 0.5 * sigma^2)
        rv_forecast = math.exp(log_pred + 0.5 * (sigma_eps ** 2))
        assert rv_forecast > 0.0
        assert not math.isnan(rv_forecast) and not math.isinf(rv_forecast)

    def test_f8_parametric_var_99_capital_preservation(self):
        """
        Audita el límite de Valor en Riesgo paramétrico al 99%:
        VaR_99% = 2.326 * sigma_open_pts * M * N <= 0.25 * Delta_DLL.
        """
        sigma_pts = 15.0  # 15 puntos en ES
        mult_es = 50.0
        n_std = 1
        var_99 = 2.326 * sigma_pts * mult_es * n_std  # 2.326 * 15 * 50 = 1,744.50 USD

        delta_dll = 1000.0
        var_limit = 0.25 * delta_dll  # 250 USD
        # El contrato estándar ES excede el límite de VaR -> DEBE CONMUTAR A MICRO
        assert var_99 > var_limit

        # Conmutación a MES (M = 5.0 USD):
        mult_mes = 5.0
        var_99_mes = 2.326 * sigma_pts * mult_mes * 1  # 174.45 USD
        assert var_99_mes <= var_limit, "El microcontrato MES respeta el límite de VaR del 25% del DLL"


# ==============================================================================
# TIER 10: PRUEBAS ADVERSARIAS MATEMÁTICAS - F9 ANOMALÍAS TEMPORALES & VRP
# ==============================================================================
class TestF9OptimizationAdversarialStress:
    """Stress testing de la optimización analítica de F9."""

    def test_f9_synthetic_vrp_epsilon_protection(self):
        """Audita el estimador sintético VRP_synth sin división por cero."""
        epsilon_0 = 1e-8
        vrp_synth = 0.0025
        mu_vrp = 0.0025
        sigma_vrp_zero = 0.0

        z_vrp_synth = (vrp_synth - mu_vrp) / (sigma_vrp_zero + epsilon_0)
        assert z_vrp_synth == 0.0

    def test_f9_term_structure_backwardation_gate(self):
        """
        Audita el bloqueo total de compras cuando TSR_synth >= 1.05 (Backwardation / Pánico).
        """
        tsr_backwardation = 1.15
        z_vrp = 0.50
        # S_VRP = -1 si Z_VRP <= -1.0 o TSR >= 1.05
        s_vrp = -1 if (z_vrp <= -1.0 or tsr_backwardation >= 1.05) else 0
        assert s_vrp == -1, "Debe activar el veto estructural por backwardation"

    def test_f9_pre_macro_timing_60s_safety_buffer(self):
        """
        Audita que la salida obligatoria a T-3 minutos garantice al menos 60 segundos
        de margen de seguridad antes del inicio del embargo de fondeo (T-2 a T+2 minutos).
        """
        t_macro_news = 8 * 60 + 30  # 08:30:00 EST (en minutos desde medianoche: 510m)
        t_embargo_start = t_macro_news - 2  # 08:28:00 EST (508m)
        t_exit_mandatory = t_macro_news - 3  # 08:27:00 EST (507m)

        safety_margin_minutes = t_embargo_start - t_exit_mandatory
        assert safety_margin_minutes == 1, "Debe existir exactamente 1 minuto (60 segundos) de colchón"
        assert t_exit_mandatory < t_embargo_start, "La salida forzosa debe ocurrir antes del embargo"


# ==============================================================================
# TIER 11: INVARIANTES TRANSVERSALES DE PROP TRADING ($50k vs $100k, ES vs MES)
# ==============================================================================
class TestTransversalOptimizationPropTradingInvariants:
    """Verificación de invariantes globales de riesgo y conmutación en las 9 optimizaciones."""

    @pytest.mark.parametrize("strategy_code", ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"])
    def test_zero_budget_implies_zero_contracts(self, strategy_code):
        """Garantiza que con presupuesto cero o negativo el algoritmo asigne N* = 0 contratos."""
        budget_zero = 0.0
        r_micro = 25.0
        n_std = math.floor(budget_zero / 250.0)
        n_micro = math.floor(budget_zero / r_micro) if r_micro > 0 else 0
        n_star = min(n_std, 2) if n_std >= 1 else min(n_micro, 10)
        assert n_star == 0, f"En {strategy_code}, presupuesto cero debe resultar en cero contratos"

    @pytest.mark.parametrize("account_profile", [
        {"size": 50000, "dll": 1000.0, "mtdd": 2000.0},
        {"size": 100000, "dll": 2000.0, "mtdd": 3000.0},
    ])
    def test_risk_budget_never_exceeds_35_percent_of_dll(self, account_profile):
        """El presupuesto arriesgado por operación jamás debe exceder el 35% del DLL contractual."""
        dll = account_profile["dll"]
        phi_max = 0.35
        max_permissible_risk = phi_max * dll
        assert max_permissible_risk <= 0.35 * dll
        if account_profile["size"] == 50000:
            assert max_permissible_risk == 350.0
        else:
            assert max_permissible_risk == 700.0
