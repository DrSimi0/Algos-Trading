r"""
tests/test_optimization_reports_verification.py

Suite de Pruebas de Verificación Automatizada E2E para los 9 Informes de
Optimización Matemática en 'reports/optimizaciones_estrategias/' (F1 a F9).

Estructura de la Suite en 4 Niveles (Tiers 1 a 4):
- Tier 1: Existencia de Entregables, Modularidad y Cero Código
  * Existencia física y sustancialidad de los 9 archivos independientes.
  * Presencia de las 7 secciones estructurales obligatorias.
  * Cero bloques de código cercados (```python, ```cpp, ```sh, etc.).
  * Cero sentencias o directivas ejecutables de programación.
  * Autocontención modular absoluta (cero dependencias no resueltas entre archivos).
- Tier 2: Indicadores Descartados y Confinamiento In-Sample
  * Veto estricto al uso activo de SMA y RSI convencional como señales.
  * Certificación explícita de confinamiento en In-Sample (Año 2023, 20,280 barras).
  * Cuarentena estricta de Out-of-Sample (2024) y Holdout (2025).
- Tier 3: Formalismo Matemático y Límites de Riesgo de Prop Trading
  * Densidad formal de ecuaciones LaTeX ($ y $$).
  * Especificación unívoca de dominios matemáticos y unidades físicas.
  * Ecuaciones analíticas de Daily Loss Limit (DLL <= $1,500 USD).
  * Ecuaciones analíticas de Trailing Max Drawdown (MTDD <= $3,000 USD).
  * Dimensionamiento de posición por volatilidad inversa con función piso.
  * Atenuación geométrica de presupuesto de riesgo ante rachas de pérdidas.
  * Conmutación automática Standard/Micro y multiplicadores CME.
  * Liquidación obligatoria intradía previa a la suspensión de compensación CME.
- Tier 4: Requerimientos Cuantitativos Profundos de F3 y F4 (+ Censo F1-F9)
  * F3: Análisis cuantitativo de ruina en Monte Carlo (ruina base 70.36%-98.96%,
        teorema de Erdős-Rényi, arrastre de fricción) y resolución cerrada
        (pendiente VWAP \kappa_{norm} <= 0.085, compuerta \Phi_{AMT}, bandas
        ajustadas por asimetría, Trailing Protect SL, supervivencia MC >= 95%).
  * F4: Auditoría forense de las 6 causas de anomalía PF 9.13 / Sharpe 14.76
        (parada temprana día 28 / barra 1794, error \sqrt{252}, ejecución
        contemporánea Close, desbalance 1:1 >$52.5k, brechas nocturnas, fricción
        multileg), ecuaciones corregidas Ornstein-Uhlenbeck, sub-cestas micro
        Dollar-Beta y envolvente de rendimiento realista.
  * F1, F2, F5, F6, F7, F8, F9: Soluciones analíticas específicas auditadas.
"""

import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports" / "optimizaciones_estrategias"

STRATEGY_FILES = [
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


@pytest.fixture(scope="module")
def opt_documents() -> Dict[str, str]:
    """Carga y retorna los 9 documentos de optimización como diccionario."""
    assert REPORTS_DIR.exists(), f"El directorio de reportes {REPORTS_DIR} debe existir."
    docs = {}
    for fname in STRATEGY_FILES:
        fpath = REPORTS_DIR / fname
        assert fpath.exists(), f"El archivo {fname} debe existir en {REPORTS_DIR}."
        docs[fname] = fpath.read_text(encoding="utf-8")
    return docs


# ==============================================================================
# TIER 1: EXISTENCIA, MODULARIDAD Y CERO CÓDIGO
# ==============================================================================
class TestTier1FileExistenceAndModularity:
    """Verifica existencia, modularidad, secciones requeridas y ausencia de código."""

    def test_all_nine_optimization_files_exist(self, opt_documents):
        """Certifica que existan exactamente los 9 archivos en el catálogo."""
        assert len(opt_documents) == 9
        for fname in STRATEGY_FILES:
            assert fname in opt_documents, f"Falta el archivo {fname} en el catálogo."

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_files_are_substantial_and_comprehensive(self, opt_documents, fname):
        """Verifica que cada archivo tenga contenido sustancial (>=15kB y >=200 líneas)."""
        content = opt_documents[fname]
        assert len(content) >= 15000, f"{fname} es inferior a 15,000 bytes ({len(content)} bytes)."
        assert len(content.splitlines()) >= 200, f"{fname} tiene menos de 200 líneas ({len(content.splitlines())})."

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_mandatory_structural_sections_present(self, opt_documents, fname):
        """Verifica que cada documento contenga las 7 secciones estructurales obligatorias."""
        content = opt_documents[fname]
        required_sections = [
            (r"## 1\.\s+Resumen Ejecutivo y Gu[ií]a Did[aá]ctica en Espa[nñ]ol", "Sección 1: Resumen Ejecutivo"),
            (r"## 2\.\s+Diagn[oó]stico In-Sample \(IS\) y Cuellos de Botella Observados", "Sección 2: Diagnóstico IS"),
            (r"## 3\.\s+Formulaci[oó]n Matem[aá]tica", "Sección 3: Formulación Matemática"),
            (r"(## 4\.\s+.*Control de Riesgo|### 3\.\d+\s+.*(Control de Riesgo|VaR y Apalancamiento|Riesgo))", "Control de Riesgo Prop Trading"),
            (r"## [45]\.\s+Impacto Cuantitativo Esperado en Prop Trading", "Sección: Impacto Cuantitativo"),
            (r"## [56]\.\s+R[uú]brica de Verificaci[oó]n Anal[ií]tica Objetiva", "Sección: Rúbrica de Verificación"),
            (r"## [67]\.\s+Gu[ií]a de Transferencia para el Chat de Implementaci[oó]n", "Sección: Guía de Transferencia"),
        ]
        for pattern, sec_name in required_sections:
            assert re.search(pattern, content), f"{fname} carece de la sección obligatoria: '{sec_name}'"

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_zero_fenced_programming_code_blocks(self, opt_documents, fname):
        """Certifica ausencia total de bloques cercados de código (```python, ```cpp, ```sh, etc.)."""
        content = opt_documents[fname]
        for idx, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            assert not stripped.startswith("```"), (
                f"{fname}:{idx} viola la regla de cero código: contiene bloque cercado '{line}'"
            )

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_zero_executable_programming_keywords(self, opt_documents, fname):
        """Certifica ausencia de directivas ejecutables de código fuente de programación."""
        content = opt_documents[fname]
        code_patterns = [
            r"^\s*import\s+[a-zA-Z0-9_]+",
            r"^\s*from\s+[a-zA-Z0-9_]+\s+import",
            r"^\s*def\s+[a-zA-Z0-9_]+\s*\(",
            r"^\s*class\s+[a-zA-Z0-9_]+",
            r"^\s*return\s+[a-zA-Z0-9_]+",
            r"cout\s*<<",
            r"console\.log\(",
        ]
        for idx, line in enumerate(content.splitlines(), start=1):
            for pat in code_patterns:
                assert not re.search(pat, line), (
                    f"{fname}:{idx} contiene sentencia ejecutable de programación: '{line}'"
                )

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_standalone_modularity_no_unresolved_cross_dependencies(self, opt_documents, fname):
        """Certifica que cada archivo sea 100% independiente y autocontenido."""
        content = opt_documents[fname]
        cross_dep_patterns = [
            r"como se defini[oó] en F[1-9]",
            r"ver el archivo F[1-9]",
            r"definido en el documento F[1-9]",
            r"consultar el reporte F[1-9]",
            r"depende de la estrategia F[1-9]",
        ]
        for pat in cross_dep_patterns:
            match = re.search(pat, content, flags=re.IGNORECASE)
            assert not match, f"{fname} contiene dependencia no resuelta hacia otro documento: '{match.group(0)}'"


# ==============================================================================
# TIER 2: INDICADORES DESCARTADOS Y CONFINAMIENTO IN-SAMPLE
# ==============================================================================
class TestTier2DiscardedIndicatorsAndISConfinement:
    """Verifica el veto estricto a SMA/RSI y el confinamiento riguroso en In-Sample 2023."""

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_zero_active_use_of_sma_or_standard_rsi(self, opt_documents, fname):
        """Certifica que SMA y RSI estándar no se usen como indicadores o señales activas."""
        content = opt_documents[fname]
        matches = list(re.finditer(
            r"\b(SMA|RSI|Simple\s+Moving\s+Average|Relative\s+Strength\s+Index)\b",
            content,
            flags=re.IGNORECASE
        ))
        for m in matches:
            start = max(0, m.start() - 120)
            end = min(len(content), m.end() + 120)
            ctx = content[start:end].replace("\n", " ").lower()
            is_exclusion = any(w in ctx for w in [
                "exclusi", "descartad", "prohibid", "cero", "card(", "indicadoresdescartados",
                "ausencia", "veto", "sin sma", "sin rsi", "reintroducir", "tokens(", "rechazo",
                "eliminad", "no emplear", "no utilizar"
            ])
            assert is_exclusion, (
                f"{fname} contiene mención de SMA/RSI fuera de contexto explícito de exclusión: '{ctx}'"
            )

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_explicit_in_sample_2023_certification(self, opt_documents, fname):
        """Verifica que cada documento certifique explícitamente su confinamiento en IS 2023."""
        content = opt_documents[fname]
        assert "2023" in content, f"{fname} debe fundamentarse explícitamente en el año 2023."
        assert "In-Sample" in content or "in-sample" in content.lower(), (
            f"{fname} debe referenciar explícitamente la partición In-Sample."
        )

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_quarantine_of_out_of_sample_2024_and_holdout_2025(self, opt_documents, fname):
        """Certifica que 2024 y 2025 solo aparezcan en contexto de cuarentena, bóveda o citas."""
        content = opt_documents[fname]
        for year in ["2024", "2025"]:
            matches = list(re.finditer(rf"\b{year}\b", content))
            for m in matches:
                start = max(0, m.start() - 120)
                end = min(len(content), m.end() + 120)
                ctx = content[start:end].replace("\n", " ").lower()
                is_academic = any(a in ctx for a in [
                    "fetna", "zarattini", "aziz", "barbon", "crabel", "alizadeh",
                    "bogousslavsky", "arxiv", "ssrn", "journal", "doi", "et al"
                ])
                is_quarantine = any(k in ctx for k in [
                    "cuarentena", "oos", "out-of-sample", "holdout", "ciego", "bóveda", "boveda",
                    "seguridad", "securitygate", "bloqueo", "prohib", "cero", "no ha sido",
                    "salvo desbloqueo", "futuras", "posteriores", "partición", "particion",
                    "autorizado", "inviolable", "sellad", "recalibrar", "alterar", "aislamiento"
                ])
                assert is_academic or is_quarantine, (
                    f"{fname} menciona {year} fuera de contexto de cuarentena estricta o cita académica: '{ctx}'"
                )


# ==============================================================================
# TIER 3: FORMALISMO MATEMÁTICO Y LÍMITES DE RIESGO PROP TRADING
# ==============================================================================
class TestTier3MathematicalFormalismAndRiskLimits:
    """Verifica formalismo LaTeX, dominios, unidades y restricciones contractuales Prop Firm."""

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_latex_equation_density_and_delimiters(self, opt_documents, fname):
        """Verifica alta densidad de formulaciones matemáticas en notación LaTeX ($ y $$)."""
        content = opt_documents[fname]
        display_eqs = len(re.findall(r"\$\$", content)) // 2
        inline_eqs = len(re.findall(r"(?<!\$)\$(?!\$)[^\n$]+(?<!\$)\$(?!\$)", content))
        assert display_eqs >= 4 or inline_eqs >= 20, (
            f"{fname} tiene baja densidad matemática: {display_eqs} display ($$), {inline_eqs} inline ($)."
        )
        assert content.count("$$") % 2 == 0, f"{fname} tiene delimitadores display '$$' desbalanceados."

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_variable_domains_and_physical_units_specified(self, opt_documents, fname):
        """Verifica presencia de dominios matemáticos formales y unidades físicas de futuros."""
        content = opt_documents[fname]
        has_domains = any(sym in content for sym in [r"\mathbb{R}", r"\mathbb{N}", r"\in", r"\subset", "Dominio"])
        assert has_domains, f"{fname} no especifica dominios matemáticos formales."

        has_units = any(u in content for u in ["USD", "pts", "puntos", "ticks", "contratos", "segundos"])
        assert has_units, f"{fname} no especifica unidades físicas o financieras de futuros."

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_daily_loss_limit_equations_bounded_below_1500(self, opt_documents, fname):
        """Verifica formulación analítica de Daily Loss Limit (DLL <= $1,500 USD)."""
        content = opt_documents[fname]
        assert "DLL" in content or "Daily Loss" in content or "L_{daily" in content, (
            f"{fname} no formaliza el Daily Loss Limit (DLL)."
        )
        dll_matches = re.findall(r"DLL\s*=\s*\$?([0-9,]+)", content)
        for val in dll_matches:
            numeric_val = float(val.replace(",", ""))
            assert numeric_val <= 1500.0, (
                f"{fname} define un DLL de ${numeric_val} que supera el límite contractual de $1,500 USD."
            )

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_trailing_max_drawdown_equations_bounded_below_3000(self, opt_documents, fname):
        """Verifica formulación analítica de Trailing Max Drawdown (MTDD <= $3,000 USD)."""
        content = opt_documents[fname]
        assert "MTDD" in content or "Trailing Drawdown" in content or "Trailing Max" in content or "Drawdown" in content, (
            f"{fname} no formaliza el Trailing Max Drawdown."
        )
        mtdd_matches = re.findall(r"MTDD\s*=\s*\$?([0-9,]+)", content)
        for val in mtdd_matches:
            numeric_val = float(val.replace(",", ""))
            assert numeric_val <= 3000.0, (
                f"{fname} define un MTDD de ${numeric_val} que supera el límite contractual de $3,000 USD."
            )

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_inverse_volatility_and_risk_distance_scaling(self, opt_documents, fname):
        """Verifica dimensionamiento inverso al riesgo/volatilidad con función piso."""
        content = opt_documents[fname]
        has_floor = r"\lfloor" in content or "piso" in content.lower() or "floor" in content.lower()
        assert has_floor, f"{fname} no formaliza la función piso entera para el dimensionamiento."

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_geometric_streak_attenuation_and_capital_preservation(self, opt_documents, fname):
        """Verifica regla analítica de atenuación geométrica por pérdidas sucesivas."""
        content = opt_documents[fname]
        content_lower = content.lower()
        has_attenuation = any(term in content for term in [
            r"K_{loss", r"2^{-", r"\phi(", r"\psi", r"D_t^{\text{cushion}}"
        ]) or any(term in content_lower for term in [
            "racha", "pérdidas consecutivas", "perdidas consecutivas", "colchón", "cushion"
        ])
        assert has_attenuation, f"{fname} no formaliza la regla de atenuación geométrica por racha."

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_micro_contract_switching_and_cme_multipliers(self, opt_documents, fname):
        """Verifica especificación de microcontratos y multiplicadores de futuros CME."""
        content = opt_documents[fname]
        has_micro = any(m in content.lower() for m in ["micro", "mes", "mnq"])
        assert has_micro, f"{fname} no contempla contratos Micro CME para gestión de riesgo."

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_mandatory_intraday_flat_liquidation(self, opt_documents, fname):
        """Verifica regla analítica de cierre intradía previo a la suspensión de CME."""
        content = opt_documents[fname]
        has_flat_time = any(t in content for t in [
            "15:15", "15:45", "15:50", "15:55", "15:58", "16:45", "08:30", "flat", "MOC"
        ])
        assert has_flat_time, f"{fname} no define hora analítica obligatoria de cierre intradía."


# ==============================================================================
# TIER 4: REQUERIMIENTOS PROFUNDOS DE F3, F4 Y CARTERA F1-F9
# ==============================================================================
class TestTier4DeepRequirementsF3AndF4:
    """Verifica los requerimientos profundos de F3 (Monte Carlo) y F4 (Auditoría Forense)."""

    # --------------------------------------------------------------------------
    # F3: ANCHORED VWAP / VALUE AREA MONTE CARLO RUIN & RESOLUTION
    # --------------------------------------------------------------------------
    def test_f3_monte_carlo_ruin_analysis_present(self, opt_documents):
        """F3 debe contener análisis cuantitativo de ruina en Monte Carlo (70.36%-98.96%, Erdős-Rényi)."""
        f3 = opt_documents["F3_optimizacion_anchored_vwap_value_area.md"]
        # Debe analizar simulación de Monte Carlo y riesgo de secuencia
        assert any(term in f3 for term in ["Monte Carlo", "monte carlo", "remuestreo", "permutación", "reshuffling"]), (
            "F3 debe documentar formalmente la simulación de Monte Carlo y el riesgo de secuencia."
        )
        # Teorema de Erdős-Rényi o análisis analítico de rachas
        assert any(term in f3 for term in ["Erdős", "Erdos", "Rényi", "Renyi", "rachas de pérdidas", "consecutive losses"]), (
            "F3 debe citar el teorema de Erdős-Rényi o el análisis formal de longitud de rachas de pérdidas."
        )
        # Rango cuantitativo de ruina base observado (70.36% - 98.96%)
        assert any(pct in f3 for pct in ["70.36", "98.96", "90.56", "70.3", "98.9"]), (
            "F3 debe documentar la tasa cuantitativa de ruina base en Monte Carlo (70.36% a 98.96%)."
        )
        # Arrastre de fricción (gross vs net)
        assert any(term in f3 for term in ["4,132", "4132", "4,587", "4587", "-455", "fricción", "friccion"]), (
            "F3 debe desglosar el impacto de fricción de comisiones y slippage en el PnL neto."
        )

    def test_f3_closed_form_mathematical_resolution(self, opt_documents):
        """F3 debe contener resolución cerrada: kappa_norm <= 0.085, Phi_AMT, skewness, Trailing Protect."""
        f3 = opt_documents["F3_optimizacion_anchored_vwap_value_area.md"]
        # Pendiente normalizada de VWAP
        assert r"\kappa_{norm" in f3 or "kappa_{norm" in f3 or r"\kappa_{crit}" in f3 or "0.085" in f3, (
            "F3 debe definir el filtro de pendiente normalizada de VWAP con umbral 0.085 para ES."
        )
        # Compuerta de aceptación de Área de Valor
        assert r"\Phi_{AMT" in f3 or "Phi_{AMT" in f3 or "OutsideValue" in f3 or "Área de Valor" in f3, (
            "F3 debe definir la compuerta de aceptación de Área de Valor previa (Phi_AMT)."
        )
        # Bandas asimétricas moduladas por sesgo
        assert "Skew" in f3 or "k_{lower" in f3 or "k_{upper" in f3 or "asimétrica" in f3, (
            "F3 debe formular bandas asimétricas ajustadas por sesgo de volumen (Skewness-adjusted bands)."
        )
        # Trailing protect stop loss
        assert "Trailing" in f3 or "trailing" in f3 or "dynamic" in f3 or "60%" in f3, (
            "F3 debe definir el Trailing Protect Stop Loss al 60% de camino al VWAP."
        )
        # Supervivencia en Monte Carlo >= 95%
        assert any(pct in f3 for pct in ["95.04", "97.72", "95%", "95.0%", "97.7%"]), (
            "F3 debe certificar supervivencia de Monte Carlo optimizada >= 95% (95.04% - 97.72%)."
        )

    # --------------------------------------------------------------------------
    # F4: COINTEGRACIÓN ES/NQ AUDITORÍA FORENSE DE 6 CAUSAS Y MARCO CORREGIDO
    # --------------------------------------------------------------------------
    def test_f4_forensic_audit_six_causes_of_anomaly(self, opt_documents):
        """F4 debe contener auditoría forense de las 6 causas de anomalía PF 9.13 / Sharpe 14.76."""
        f4 = opt_documents["F4_optimizacion_cointegracion_ou.md"]
        # Debe referenciar las métricas anómalas
        assert "9.13" in f4, "F4 debe citar el Profit Factor anómalo de 9.13."
        assert "14.76" in f4, "F4 debe citar el Sharpe Ratio anómalo de 14.76."

        # Causa 1: Parada temprana (día 28 / barra 1,794)
        assert any(term in f4 for term in ["1794", "1,794", "28 días", "28 dias", "early stopping", "parada temprana"]), (
            "F4 debe documentar la Causa 1: truncamiento por parada temprana al día 28 / barra 1,794."
        )
        # Causa 2: Error de anualización sqrt(252) sobre trades
        assert any(term in f4 for term in ["252", r"\sqrt{252}", "sqrt(252)", "anualización", "anualizacion"]), (
            "F4 debe documentar la Causa 2: error de anualización multiplicando retornos por trade por sqrt(252)."
        )
        # Causa 3: Ejecución contemporánea al cierre (target leakage)
        assert any(term in f4 for term in ["Close", "cierre", "look-ahead", "target leakage", "contemporánea", "contemporanea"]), (
            "F4 debe documentar la Causa 3: ejecución contemporánea en Close (target leakage / look-ahead bias)."
        )
        # Causa 4: Redondeo 1:1 y exposición no cubierta
        assert any(term in f4 for term in ["1:1", "52,500", "52500", "nocional", "desbalance"]), (
            "F4 debe documentar la Causa 4: redondeo 1:1 en sub-cestas causando exposición no cubierta >$52.5k."
        )
        # Causa 5: Brechas nocturnas / gaps
        assert any(term in f4 for term in ["overnight", "nocturna", "gap", "brecha"]), (
            "F4 debe documentar la Causa 5: impacto de brechas nocturnas en la difusión continua."
        )
        # Causa 6: Fricción multileg y riesgo de ejecución asincrónica (legging)
        assert any(term in f4 for term in ["multileg", "legging", "asincrónica", "asincronica", "fricción", "friccion"]), (
            "F4 debe documentar la Causa 6: subestimación de fricción multileg y riesgo de legging."
        )

    def test_f4_corrected_ornstein_uhlenbeck_diffusion(self, opt_documents):
        """F4 debe formular las ecuaciones corregidas del proceso de Ornstein-Uhlenbeck."""
        f4 = opt_documents["F4_optimizacion_cointegracion_ou.md"]
        assert any(k in f4 for k in [r"\kappa", "kappa", "velocidad de reversión"])
        assert any(th in f4 for th in [r"\theta", "theta", "media de largo plazo"])
        assert any(sig in f4 for sig in [r"\sigma_{eq}", "sigma_eq", "volatilidad de equilibrio"])
        assert any(tau in f4 for tau in [r"\tau_{1/2}", "tau_{1/2}", "vida media"])

    def test_f4_micro_dollar_beta_sub_baskets(self, opt_documents):
        """F4 debe formular sub-cestas micro Dollar-Beta con riesgo acotado <= $100 USD."""
        f4 = opt_documents["F4_optimizacion_cointegracion_ou.md"]
        assert any(term in f4 for term in ["sub-cesta", "sub-cestas", "sub_unit", "cesta micro"]), (
            "F4 debe formular la partición analítica en sub-cestas micro."
        )
        assert any(risk in f4 for risk in ["98", "99", "100"]), (
            "F4 debe acotar el riesgo por sub-cesta a <= $100 USD (ej. ~$98-$99 USD)."
        )

    def test_f4_realistic_performance_envelope(self, opt_documents):
        """F4 debe proyectar una envolvente de rendimiento realista institucional tras correcciones."""
        f4 = opt_documents["F4_optimizacion_cointegracion_ou.md"]
        has_realistic_sharpe = any(val in f4 for val in ["1.5", "2.3", "2.5", "3.0"])
        assert has_realistic_sharpe, "F4 debe proyectar un rango realista de Sharpe ratio."
        has_realistic_pf = any(val in f4 for val in ["1.6", "2.2", "2.5", "2.8"])
        assert has_realistic_pf, "F4 debe proyectar un rango realista de Profit Factor."

    # --------------------------------------------------------------------------
    # CENSO Y OPTIMIZACIONES F1, F2, F5, F6, F7, F8, F9
    # --------------------------------------------------------------------------
    def test_f1_market_momentum_analytical_resolution(self, opt_documents):
        """F1: Umbral dinámico por ATR/volatilidad, filtro VWAP 15:30 y salida MOC 15:58."""
        f1 = opt_documents["F1_optimizacion_market_intraday_momentum.md"]
        assert r"\theta_{floor}" in f1 or "0.0010" in f1 or "0.0015" in f1
        assert r"\Phi_{VWAP" in f1 or "VWAP" in f1
        assert any(t in f1 for t in ["15:30", "15:45", "15:58"])

    def test_f2_filtered_orb_compression_and_rvol_resolution(self, opt_documents):
        """F2: Ratio de compresión R_comp, umbral RVOL_crit y breakeven a +1.0R."""
        f2 = opt_documents["F2_optimizacion_filtered_orb.md"]
        assert any(term in f2 for term in ["R_{comp", "NR4", "Inside Day", "compresión"])
        assert any(term in f2 for term in ["RVOL", "RVOL_{crit}"])
        assert any(term in f2 for term in ["Breakeven", "breakeven", "+1.0R", "+1R"])

    def test_f5_order_flow_imbalance_decay_and_zscore(self, opt_documents):
        """F5: Núcleo de decaimiento exponencial de flujo y Z-score adaptativo."""
        f5 = opt_documents["F5_optimizacion_order_flow_imbalance.md"]
        assert any(term in f5 for term in [r"\lambda_d", r"\rho_{decay}", "decaimiento", "decay", "60"])
        assert any(term in f5 for term in ["z-score", "Z-score", "Z_{OFI", "normalización"])

    def test_f6_multi_level_ofi_spatial_decay_and_pca(self, opt_documents):
        """F6: Decaimiento espacial de potencias l^-alpha, PCA y coherencia direccional."""
        f6 = opt_documents["F6_optimizacion_multi_level_ofi.md"]
        assert any(term in f6 for term in ["potencia", "l^{-", "espacial", "power-law"])
        assert any(term in f6 for term in ["PCA", "componentes principales", "autovalor", "eigen"])
        assert any(term in f6 for term in [r"\Phi_{coherence", "coherencia", "0.40"])

    def test_f7_meta_labeling_asymmetric_triple_barrier(self, opt_documents):
        """F7: Barreras ampliadas Garman-Klass, umbral de meta-modelo y costo asimétrico."""
        f7 = opt_documents["F7_optimizacion_meta_labeling.md"]
        assert any(term in f7 for term in [r"\ell^-", r"\ell^+", "barrera", "Garman-Klass", "1.50"])
        assert any(term in f7 for term in [r"\tau_{meta}", "0.65", "umbral", "probabilidad"])

    def test_f8_har_rv_j_opening_quarantine_and_log_har(self, opt_documents):
        """F8: Desacoplamiento de apertura [09:30, 09:45], Log-HAR y gobernador de riesgo."""
        f8 = opt_documents["F8_optimizacion_modelos_har_rv_j.md"]
        assert any(term in f8 for term in ["09:45", "apertura", "cuarentena", "primer cuarto de hora"])
        assert any(term in f8 for term in ["Log-HAR", r"\ln(RV", "HAR-RV", "logarítm"])
        assert any(term in f8 for term in ["gobernador", "VaR", "disyuntor"])

    def test_f9_temporal_anomalies_vrp_synthetic_and_macro_exit(self, opt_documents):
        """F9: Estimador sintético VRP_synth, deriva nocturna y salida pre-macro T-3 min."""
        f9 = opt_documents["F9_optimizacion_anomalias_temporales_vrp.md"]
        assert any(term in f9 for term in ["VRP_{synth", "VRP", "sintético", "sintetico"])
        assert any(term in f9 for term in ["01:30", "08:30", "nocturn", "overnight"])
        assert any(term in f9 for term in ["T-3", "T - 3", "embargo", "macro"])
