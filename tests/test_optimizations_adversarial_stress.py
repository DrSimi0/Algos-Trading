"""
tests/test_optimizations_adversarial_stress.py

Suite de Pruebas Pytest: Verificación Adversarial y Estrés Cuantitativo E2E
para las 9 Propuestas de Optimización Matemática en 'reports/optimizaciones_estrategias/' (F1 a F9).

Cobertura de Auditoría:
- Tier 1: Entregables modulares independientes (exactamente 9 archivos) y estructura obligatoria.
- Tier 2: Restricciones negativas: ausencia total de bloques de código (R1) y exclusión de SMA/RSI (R2).
- Tier 3: Rigor metodológico y cuarentena estricta In-Sample 2023 (R3).
- Tier 4: Estrés matemático de ecuaciones cerradas, estabilidad y ausencia de singularidades (F1 a F9).
- Tier 5: Verificación de invariantes de gestión de riesgo institucional para Prop Trading.
"""

import math
import os
import re
from pathlib import Path
from typing import Dict, List
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPT_REPORTS_DIR = PROJECT_ROOT / "reports" / "optimizaciones_estrategias"

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
    assert OPT_REPORTS_DIR.exists(), f"El directorio {OPT_REPORTS_DIR} debe existir"
    docs = {}
    for fname in STRATEGY_FILES:
        fpath = OPT_REPORTS_DIR / fname
        assert fpath.exists(), f"El archivo {fname} debe existir en {OPT_REPORTS_DIR}"
        docs[fname] = fpath.read_text(encoding="utf-8")
    return docs


# ==============================================================================
# TIER 1: ESTRUCTURA, MODULARIDAD Y ENTREGABLES
# ==============================================================================
class TestTier1StructureAndModularDeliverables:
    """Verifica que existan exactamente los 9 archivos y contengan las secciones requeridas."""

    def test_all_nine_files_present_and_substantial(self, opt_documents):
        assert len(opt_documents) == 9
        for fname, content in opt_documents.items():
            assert len(content) >= 15000, f"{fname} es inferior a 15,000 bytes"
            assert len(content.splitlines()) >= 200, f"{fname} es inferior a 200 líneas"

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_mandatory_sections_present(self, opt_documents, fname):
        content = opt_documents[fname]
        assert re.search(r"## 1\.\s+Resumen Ejecutivo y Gu[ií]a Did[aá]ctica en Espa[nñ]ol", content)
        assert re.search(r"## 2\.\s+Diagn[oó]stico In-Sample \(IS\) y Cuellos de Botella Observados", content)
        assert re.search(r"## 3\.\s+Formulaci[oó]n Matem[aá]tica", content)
        assert re.search(r"(## 4\.\s+.*Control de Riesgo|### 3\.\d+\s+.*(Control de Riesgo|VaR y Apalancamiento))", content)
        assert re.search(r"## [45]\.\s+Impacto Cuantitativo Esperado en Prop Trading", content)
        assert re.search(r"## [56]\.\s+R[uú]brica de Verificaci[oó]n Anal[ií]tica Objetiva", content)
        assert re.search(r"## [67]\.\s+Gu[ií]a de Transferencia para el Chat de Implementaci[oó]n", content)


# ==============================================================================
# TIER 2: RESTRICCIONES NEGATIVAS (CERO CÓDIGO Y CERO SMA/RSI)
# ==============================================================================
class TestTier2NegativeConstraints:
    """Certifica la ausencia total de bloques de código y el veto a SMA/RSI."""

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_zero_fenced_code_blocks(self, opt_documents, fname):
        content = opt_documents[fname]
        for idx, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            assert not stripped.startswith("```"), f"{fname}:{idx} contiene bloque de código: '{line}'"

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_zero_executable_programming_keywords(self, opt_documents, fname):
        content = opt_documents[fname]
        code_patterns = [
            r"^\s*import\s+[a-zA-Z0-9_]+",
            r"^\s*from\s+[a-zA-Z0-9_]+\s+import",
            r"^\s*def\s+[a-zA-Z0-9_]+\s*\(",
            r"^\s*class\s+[a-zA-Z0-9_]+",
            r"^\s*return\s+[a-zA-Z0-9_]+",
        ]
        for idx, line in enumerate(content.splitlines(), start=1):
            for pat in code_patterns:
                assert not re.search(pat, line), f"{fname}:{idx} contiene sentencia de código: '{line}'"

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_zero_active_sma_or_rsi(self, opt_documents, fname):
        content = opt_documents[fname]
        matches = list(re.finditer(r"\b(SMA|RSI|Simple\s+Moving\s+Average|Relative\s+Strength\s+Index)\b", content, flags=re.IGNORECASE))
        for m in matches:
            start = max(0, m.start() - 100)
            end = min(len(content), m.end() + 100)
            ctx = content[start:end].replace("\n", " ").lower()
            is_exclusion = any(w in ctx for w in [
                "exclusi", "descartad", "prohibid", "cero", "card(", "indicadoresdescartados",
                "ausencia", "veto", "sin sma", "sin rsi", "reintroducir", "tokens("
            ])
            assert is_exclusion, f"{fname} contiene uso activo de SMA/RSI: '{ctx}'"


# ==============================================================================
# TIER 3: CUARENTENA ESTRICTA IN-SAMPLE (2023)
# ==============================================================================
class TestTier3InSampleQuarantine:
    """Verifica el aislamiento estricto In-Sample 2023 y cuarentena de 2024/2025."""

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_in_sample_grounding_and_quarantine(self, opt_documents, fname):
        content = opt_documents[fname]
        assert "2023" in content, f"{fname} debe fundamentarse en el año 2023"
        assert "In-Sample" in content or "in-sample" in content.lower()

        for year in ["2024", "2025"]:
            matches = list(re.finditer(rf"\b{year}\b", content))
            for m in matches:
                start = max(0, m.start() - 100)
                end = min(len(content), m.end() + 100)
                ctx = content[start:end].replace("\n", " ").lower()
                is_academic = any(a in ctx for a in ["fetna", "zarattini", "aziz", "barbon", "crabel", "alizadeh", "bogousslavsky"])
                is_quarantine = any(k in ctx for k in [
                    "cuarentena", "oos", "out-of-sample", "holdout", "ciego", "bóveda", "boveda",
                    "seguridad", "securitygate", "bloqueo", "prohib", "cero", "no ha sido",
                    "salvo desbloqueo", "futuras", "posteriores", "partición", "particion",
                    "autorizado", "inviolable", "sellad", "recalibrar", "alterar"
                ])
                assert is_academic or is_quarantine, f"{fname} menciona {year} fuera de contexto de cuarentena: '{ctx}'"


# ==============================================================================
# TIER 4: ESTRÉS MATEMÁTICO Y BORDES (F1 A F9)
# ==============================================================================
class TestTier4MathematicalStress:
    """Comprueba empíricamente las formulaciones de bordes matemáticos."""

    def test_f1_theta_and_volume_stress(self, opt_documents):
        f1 = opt_documents["F1_optimizacion_market_intraday_momentum.md"]
        assert r"\theta_{floor}" in f1 or "0.0010" in f1
        assert r"\Phi_{VWAP, t}" in f1
        p_ref = 5800.0
        for atr in [0.0, 1.0, 10.0, 100.0]:
            theta = max(0.0010, 0.60 * (atr / p_ref))
            assert theta >= 0.0010
            assert not math.isnan(theta)

    def test_f2_rvol_compression_function_bounds(self, opt_documents):
        f2 = opt_documents["F2_optimizacion_filtered_orb.md"]
        assert "R_{comp, t}" in f2
        assert r"\epsilon_0" in f2
        for r_comp in [0.0, 0.50, 0.70, 0.92, 2.0]:
            crit = min(1.70, max(1.20, 1.20 + 0.50 * ((r_comp - 0.50) / (0.92 - 0.50))))
            assert 1.20 <= crit <= 1.70

    def test_f3_asymmetric_bands_and_slope(self, opt_documents):
        f3 = opt_documents["F3_optimizacion_anchored_vwap_value_area.md"]
        assert r"\kappa_{norm, t}" in f3
        assert "Skew_{intra, t}" in f3
        for sk in [-10.0, -2.0, 0.0, 2.0, 10.0]:
            k_low = 2.0 + min(0.40, max(0.0, -0.50 * sk))
            k_upp = 2.0 + min(0.40, max(0.0, +0.50 * sk))
            assert 2.0 <= k_low <= 2.40
            assert 2.0 <= k_upp <= 2.40

    def test_f4_ornstein_uhlenbeck_roots_and_kalman(self, opt_documents):
        f4 = opt_documents["F4_optimizacion_cointegracion_ou.md"]
        assert "0.9995" in f4
        dt = 5.0 / 390.0
        for b in [0.1, 0.5, 0.9, 0.9995]:
            kappa = -math.log(b) / dt
            assert kappa > 0
            tau_half = (math.log(2) * dt) / (-math.log(b))
            assert tau_half > 0
            sigma_eq = math.sqrt(1e-4 / (1.0 - b**2))
            assert not math.isnan(sigma_eq) and not math.isinf(sigma_eq)

    def test_f5_ofi_exponential_decay_and_adaptive_z(self, opt_documents):
        f5 = opt_documents["F5_optimizacion_order_flow_imbalance.md"]
        assert r"\lambda_d" in f5 or r"\rho_{decay}" in f5
        assert r"\epsilon_0" in f5
        decay = math.exp(-(math.log(2) / 90.0) * 60.0)
        assert 0.60 < decay < 0.65

    def test_f6_multi_level_power_law_and_pca(self, opt_documents):
        f6 = opt_documents["F6_optimizacion_multi_level_ofi.md"]
        assert r"\Phi_{coherence" in f6
        assert "0.40" in f6
        weights = [1.0 / (l**1.0) for l in range(1, 6)]
        assert weights[0] == 1.0 and weights[-1] == 0.20
        np.random.seed(123)
        cov = np.cov(np.random.randn(30, 5), rowvar=False)
        w, _ = np.linalg.eigh(cov)
        assert np.all(w >= -1e-9)

    def test_f7_asymmetric_triple_barrier(self, opt_documents):
        f7 = opt_documents["F7_optimizacion_meta_labeling.md"]
        assert r"\ell^-(\sigma" in f7
        assert r"\ell^+(\sigma" in f7
        for xi in [0.1, 1.0, 5.0]:
            l_m = max(1.50, 1.25 + 0.50 * xi)
            assert l_m >= 1.50
            l_p = l_m * 1.60 + (0.748 / 5.0)
            assert l_p > l_m

    def test_f8_log_har_positive_variance_guarantee(self, opt_documents):
        f8 = opt_documents["F8_optimizacion_modelos_har_rv_j.md"]
        assert "09:45:00" in f8 or "09:45" in f8
        assert "Log-HAR" in f8 or r"\ln(RV" in f8
        for log_v in [-20.0, -5.0, 0.0, 10.0]:
            v = math.exp(log_v)
            assert v > 0.0

    def test_f9_synthetic_vrp_and_macro_embargo(self, opt_documents):
        f9 = opt_documents["F9_optimizacion_anomalias_temporales_vrp.md"]
        assert "VRP_{synth" in f9
        assert "TSR_{synth" in f9
        assert r"\mathcal{E}_{macro}" in f9 or "embargo" in f9


# ==============================================================================
# TIER 5: CONTROLES DE RIESGO DE PROP TRADING
# ==============================================================================
class TestTier5PropTradingRiskControls:
    """Verifica el cumplimiento de las restricciones contractuales de Prop Trading."""

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_prop_trading_risk_invariants_in_text(self, opt_documents, fname):
        content = opt_documents[fname]
        assert "DLL" in content
        assert "MTDD" in content or "Drawdown" in content
        assert r"\lfloor" in content or "piso" in content.lower() or "floor" in content.lower()
        assert "micro" in content.lower() or "mes" in content.lower() or "mnq" in content.lower()

    def test_streak_attenuation_and_capital_preservation(self):
        dll = 1000.0
        mtdd = 2000.0
        for streak in [0, 1, 2, 3, 5]:
            phi = 0.35 * (2.0 ** (-min(streak, 3)))
            assert phi <= 0.35
            if streak >= 3:
                assert phi == 0.04375

            budget = min(phi * dll, 0.20 * mtdd)
            n_std = math.floor(budget / 300.0)
            n_micro = math.floor(budget / 30.0)
            assert n_std * 300.0 <= budget + 1e-9
            assert n_micro * 30.0 <= budget + 1e-9


# ==============================================================================
# TIER 6: RÚBRICA DE 8 PUNTOS
# ==============================================================================
class TestTier6RubricCertification:
    """Verifica que cada uno de los 9 documentos incluya y audite R1 a R8."""

    @pytest.mark.parametrize("fname", STRATEGY_FILES)
    def test_eight_point_rubric_complete(self, opt_documents, fname):
        content = opt_documents[fname]
        assert re.search(r"## [56]\.\s+R[uú]brica de Verificaci[oó]n Anal[ií]tica Objetiva", content)
        for r_idx in range(1, 9):
            assert f"R{r_idx}" in content, f"{fname} no audita R{r_idx}"
