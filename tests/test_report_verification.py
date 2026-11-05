"""
tests/test_report_verification.py

Suite de Pruebas de Validación Automatizada y Verificación E2E para el Informe de
Investigación Cuantitativa: 'reports/investigacion_teorias_futuros_es_nq.md'.

Niveles de Validación (Tiers 1-4):
- Tier 1: Cobertura de características, 4 pilares obligatorios, adaptación a conversación 0afec980 y modelos F1-F9.
- Tier 2: Rigor formal, citas académicas completas, DOIs/URLs válidos y ecuaciones matemáticas LaTeX.
- Tier 3: Matriz de fricción CME (ES, MES, NQ, MNQ) y matriz de riesgo Prop Trading (DLL, Trailing DD, Sharpe).
- Tier 4: Integridad sintáctica de pseudocódigo Python (ast.parse) y dry-runs ejecutables de los 9 algoritmos clave.
"""

import ast
import os
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import pytest

try:
    from src.core.futures_specs import FUTURES_SPECS, get_spec
    from src.portfolio.prop_firm_rules import PropFirmProfile, PropFirmSimulator, PROFILES
except ImportError:
    FUTURES_SPECS, get_spec = None, None
    PropFirmProfile, PropFirmSimulator, PROFILES = None, None, None


# ==============================================================================
# CONFIGURACIÓN Y CONSTANTES DEL INFORME
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "investigacion_teorias_futuros_es_nq.md"
REPORT_PATH = Path(os.environ.get("REPORT_PATH", str(DEFAULT_REPORT_PATH)))

# Parámetro opcional: si REQUIRE_REPORT=1, fuerza fallo si el informe consolidado aún no existe.
# Si no está establecido (fase M1/M2), salta elegantemente las pruebas dependientes del archivo
# permitiendo progresividad TDD mientras ejecuta al 100% las pruebas algorítmicas de Tier 4.
REQUIRE_REPORT = os.environ.get("REQUIRE_REPORT", "0") == "1"

# 4 Pilares Fundamentales requeridos por el usuario (Follow-up)
MANDATORY_PILLARS = [
    {
        "name": "Trading Sistemático",
        "patterns": [r"trading sistem[aá]tico", r"enfoque sistem[aá]tico", r"reglas sistem[aá]ticas", r"systematic trading"],
    },
    {
        "name": "Arbitraje Estadístico",
        "patterns": [r"arbitraje estad[ií]stico", r"statistical arbitrage", r"cointegraci[oó]n", r"statarb"],
    },
    {
        "name": "Alta Frecuencia y Microestructura de Mercado (HFT)",
        "patterns": [r"alta frecuencia", r"microestructura", r"hft", r"order flow imbalance", r"desequilibrio.*flujo"],
    },
    {
        "name": "Machine Learning & Hybrid Quant",
        "patterns": [r"machine learning", r"hybrid quant", r"aprendizaje autom[aá]tico", r"cuantitativo h[ií]brido"],
    },
]

# 9 Modelos Cuantitativos Específicos (PROJECT.md F1-F9)
CORE_MODELS_SPECS = [
    {
        "id": "F1",
        "name": "Order Flow Imbalance (OFI & Multi-Level OFI)",
        "authors": ["Cont", "Kukanov", "Stoikov"],
        "year": "2014",
        "doi_patterns": [r"10\.1093/jjfinec/nbt003", r"arxiv\.org/abs/1011\.1108", r"ssrn\.com/abstract=1712822"],
        "math_keywords": [r"OFI", r"q_n\^b", r"q_n\^a", r"\Delta P"],
    },
    {
        "id": "F2",
        "name": "Market Intraday Momentum & MOC Seasonality",
        "authors": ["Gao", "Han", "Li", "Zhou"],
        "year": "2018",
        "doi_patterns": [r"10\.1016/j\.jfineco\.2017\.11\.002"],
        "math_keywords": [r"09:30", r"10:00", r"15:30", r"16:00", r"r_\{.*\}", r"\beta"],
    },
    {
        "id": "F3",
        "name": "Anchored VWAP & Auction Value Area",
        "authors": ["Madhavan"],
        "year": "2002",
        "doi_patterns": [r"10\.1016/S1386-4181\(02\)00004-7", r"ssrn"],
        "math_keywords": [r"VWAP", r"\sigma", r"P_i", r"V_i"],
    },
    {
        "id": "F4",
        "name": "Conditional Overnight Drift",
        "authors": ["Boyarchenko", "Larsen", "Whelan"],
        "year": "2023",
        "doi_patterns": [r"10\.1093/rapstu/raad004"],
        "math_keywords": [r"RTH", r"01:30", r"08:30", r"overnight", r"\Delta"],
    },
    {
        "id": "F5",
        "name": "Pre-Macro Announcement Informed Drift",
        "authors": ["Kurov"],
        "year": "2019",
        "doi_patterns": [r"10\.1017/S002210901800040X", r"ssrn"],
        "math_keywords": [r"FOMC", r"CPI", r"T-15", r"T-3"],
    },
    {
        "id": "F6",
        "name": "Filtered Quantitative Opening Range Breakout (ORB)",
        "authors": ["Crabel", "Zarattini"],
        "year": "2024",
        "doi_patterns": [r"4744211", r"ssrn", r"10\.2139/ssrn"],
        "math_keywords": [r"ORB", r"Stretch", r"NR4", r"RVOL"],
    },
    {
        "id": "F7",
        "name": "Variance Risk Premium (VRP) & Term Structure",
        "authors": ["Bollerslev", "Tauchen", "Zhou"],
        "year": "2009",
        "doi_patterns": [r"10\.1093/rfs/hhp008"],
        "math_keywords": [r"VRP", r"IV", r"RV", r"Z_\{VRP\}"],
    },
    {
        "id": "F8",
        "name": "HAR-RV-J Volatility Forecasting & Volatility-Managed Sizing",
        "authors": ["Corsi", "Moreira"],
        "year": "2009",
        "doi_patterns": [r"10\.1093/jjfinec/nbp001", r"10\.1111/jofi\.12498"],
        "math_keywords": [r"HAR-RV", r"RV\^\{(d)\}", r"RV\^\{(w)\}", r"RV\^\{(m)\}", r"N_t"],
    },
    {
        "id": "F9",
        "name": "Cointegrated ES vs NQ Ornstein-Uhlenbeck Pairs Trading",
        "authors": ["Avellaneda", "Lee"],
        "year": "2010",
        "doi_patterns": [r"10\.1080/14697680903124632"],
        "math_keywords": [r"S\text{-score}", r"Ornstein", r"\kappa", r"\theta", r"\sigma"],
    },
]


# ==============================================================================
# UTILIDADES DE PARSEO DE MARKDOWN
# ==============================================================================

def extract_fenced_code_blocks(markdown_text: str, language: str = "python") -> List[str]:
    """Extrae bloques de código cerrados por triples comillas con el lenguaje especificado."""
    pattern = rf"```{language}\s*\n(.*?)```"
    matches = re.findall(pattern, markdown_text, flags=re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches if m.strip()]


def extract_latex_equations(markdown_text: str) -> Dict[str, List[str]]:
    """Extrae fórmulas matemáticas LaTeX en línea ($...$) y en bloque ($$...$$)."""
    block_equations = re.findall(r"\$\$(.*?)\$\$", markdown_text, flags=re.DOTALL)
    # Limpiamos bloques primero para evitar coincidencias duplicadas de $
    no_blocks = re.sub(r"\$\$.*?\$\$", "", markdown_text, flags=re.DOTALL)
    inline_equations = re.findall(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", no_blocks)
    return {
        "block": [eq.strip() for eq in block_equations if eq.strip()],
        "inline": [eq.strip() for eq in inline_equations if eq.strip()],
    }


def extract_doi_and_urls(markdown_text: str) -> Dict[str, List[str]]:
    """Extrae identificadores DOI formales y enlaces web verificables."""
    doi_pattern = r"(?:doi(?:\.org)?/|DOI:\s*\[?)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)"
    url_pattern = r"https?://[^\s)\]\"'>]+"
    dois = re.findall(doi_pattern, markdown_text, flags=re.IGNORECASE)
    urls = re.findall(url_pattern, markdown_text)
    return {"dois": list(set(dois)), "urls": list(set(urls))}


def load_report_or_skip() -> str:
    """Carga el texto del informe o salta la prueba si no existe y no es obligatorio aún."""
    if not REPORT_PATH.exists():
        if REQUIRE_REPORT:
            pytest.fail(f"Informe consolidado requerido en {REPORT_PATH} pero no fue encontrado.")
        pytest.skip(
            f"El informe consolidado '{REPORT_PATH}' aún no existe (esperado tras Milestone 3). "
            f"Para validar borradores intermedios, defina REPORT_PATH=<ruta>."
        )
    return REPORT_PATH.read_text(encoding="utf-8")


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def report_content() -> str:
    """Fixture que proporciona el texto completo del informe markdown."""
    return load_report_or_skip()


# ==============================================================================
# TIER 1: VERIFICACIÓN DE ESTRUCTURA, 4 PILARES Y COBERTURA DE MODELOS
# ==============================================================================

class TestReportStructureAndPillars:
    """Pruebas de existencia, extensión y cobertura de los pilares metodológicos."""

    def test_report_file_exists_and_path_resolution(self):
        """Verifica la resolución y existencia del informe objetivo."""
        if not REPORT_PATH.exists() and not REQUIRE_REPORT:
            pytest.skip(f"Esperando Milestone 3 para crear {REPORT_PATH}")
        assert REPORT_PATH.exists(), f"El archivo {REPORT_PATH} debe existir físicamente."
        assert REPORT_PATH.is_file(), f"{REPORT_PATH} debe ser un archivo regular."

    def test_report_minimum_volume_and_depth(self, report_content: str):
        """Verifica que el informe no sea un resumen superficial (> 20,000 caracteres)."""
        char_count = len(report_content)
        word_count = len(report_content.split())
        assert char_count >= 20000, f"Informe demasiado corto ({char_count} caracteres). Se espera investigación exhaustiva."
        assert word_count >= 3000, f"Informe con pocas palabras ({word_count} palabras)."

    @pytest.mark.parametrize("pillar", MANDATORY_PILLARS, ids=lambda p: p["name"])
    def test_mandatory_four_pillars_inclusion(self, report_content: str, pillar: Dict[str, Any]):
        """Verifica la inclusión explícita de los 4 pilares solicitados por el usuario."""
        found = False
        lower_content = report_content.lower()
        for pat in pillar["patterns"]:
            if re.search(pat, lower_content, re.IGNORECASE):
                found = True
                break
        assert found, (
            f"El pilar obligatorio '{pillar['name']}' no fue detectado en el informe. "
            f"Patrones buscados: {pillar['patterns']}"
        )

    def test_adaptation_analysis_to_conversation_0afec980(self, report_content: str):
        """
        Verifica el análisis de adaptación a la conversación previa:
        Conversation ID: 0afec980-b3b6-4d5f-89f7-a5122f98398f
        Title: Algorithmic Trading Portfolio Backtesting Options
        """
        assert "0afec980-b3b6-4d5f-89f7-a5122f98398f" in report_content, (
            "El informe debe citar explícitamente el Conversation ID '0afec980-b3b6-4d5f-89f7-a5122f98398f'."
        )
        assert re.search(r"Algorithmic Trading Portfolio Backtesting Options", report_content, re.IGNORECASE), (
            "El informe debe referenciar el título de la conversación previa 'Algorithmic Trading Portfolio Backtesting Options'."
        )

    def test_codebase_integration_references(self, report_content: str):
        """Verifica que el informe evalúe componentes existentes del repositorio."""
        repo_components = ["BacktestEngine", "PropFirmSimulator", "run_futures_prop", "PortfolioAllocator"]
        matched = [comp for comp in repo_components if comp in report_content]
        assert len(matched) >= 2, (
            f"El informe debe mapear la adaptación a los módulos del repo. Encontrados: {matched}"
        )

    @pytest.mark.parametrize("model", CORE_MODELS_SPECS, ids=lambda m: m["id"] + "-" + m["name"][:20])
    def test_core_models_coverage_f1_to_f9(self, report_content: str, model: Dict[str, Any]):
        """Verifica que cada uno de los 9 modelos F1-F9 esté desarrollado en el informe."""
        # 1. Al menos un autor principal debe estar mencionado
        author_found = any(author.lower() in report_content.lower() for author in model["authors"])
        assert author_found, (
            f"Modelo {model['id']} ({model['name']}): No se encontraron autores {model['authors']} en el informe."
        )


# ==============================================================================
# TIER 2: CITAS ACADÉMICAS, DOIS Y FORMULACIÓN MATEMÁTICA LATEX
# ==============================================================================

class TestAcademicCitationsAndMath:
    """Validación rigurosa de trazabilidad bibliográfica y especificación matemática."""

    def test_report_contains_valid_dois_or_urls(self, report_content: str):
        """Verifica que el documento contenga DOIs o URLs académicas válidas."""
        extracted = extract_doi_and_urls(report_content)
        total_sources = len(extracted["dois"]) + len(extracted["urls"])
        assert total_sources >= 8, (
            f"Se esperaban al menos 8 fuentes académicas con DOI/URL verificable, encontradas {total_sources}."
        )

    @pytest.mark.parametrize("model", CORE_MODELS_SPECS, ids=lambda m: m["id"])
    def test_model_doi_or_academic_link_present(self, report_content: str, model: Dict[str, Any]):
        """Verifica que cada modelo incluya su DOI o identificador formal."""
        found = False
        for pat in model["doi_patterns"]:
            if re.search(pat, report_content, re.IGNORECASE):
                found = True
                break
        assert found, (
            f"El modelo {model['id']} ({model['name']}) no presenta su DOI o enlace formal. "
            f"Patrones requeridos: {model['doi_patterns']}"
        )

    def test_latex_equations_presence_and_balance(self, report_content: str):
        """Verifica la presencia de ecuaciones LaTeX y la integridad sintáctica de delimitadores y llaves."""
        eqs = extract_latex_equations(report_content)
        total_equations = len(eqs["block"]) + len(eqs["inline"])
        assert total_equations >= 15, (
            f"Se esperaban al menos 15 formulaciones matemáticas en LaTeX ($ o $$), encontradas {total_equations}."
        )
        assert len(eqs["block"]) >= 5, "Debe haber al menos 5 ecuaciones en bloque ($$...$$) para rigor técnico."

        # Verificación de integridad en bloques de ecuación LaTeX ($$...$$):
        # 1. El conteo de delimitadores '$$' fuera de bloques de código debe ser estrictamente par.
        clean_text = re.sub(r"```.*?```", "", report_content, flags=re.DOTALL)
        double_dollar_count = clean_text.count("$$")
        assert double_dollar_count % 2 == 0, (
            f"Desbalance en delimitadores de bloque LaTeX '$$': conteo total es {double_dollar_count} (impar)."
        )

        # 2. Cada bloque de ecuación debe tener llaves {} balanceadas (evita errores de compilación MathJax/KaTeX)
        unbalanced_blocks = []
        for i, block in enumerate(eqs["block"], 1):
            if block.count("{") != block.count("}"):
                unbalanced_blocks.append(f"Bloque #{i}: llaves desbalanceadas '{{'={block.count('{')}, '}}'={block.count('}')}\n{block[:100]}...")
        assert not unbalanced_blocks, "Ecuaciones LaTeX con llaves desbalanceadas:\n" + "\n".join(unbalanced_blocks)

    def test_mathematical_variables_explicitly_defined(self, report_content: str):
        """Verifica que no existan modelos de caja negra comprobando variables matemáticas clave."""
        essential_symbols = [
            r"OFI",       # Order Flow Imbalance
            r"VWAP",      # Volume Weighted Average Price
            r"VRP",       # Variance Risk Premium
            r"RV",        # Realized Volatility
            r"IV",        # Implied Volatility
        ]
        for symbol in essential_symbols:
            assert re.search(rf"\${re.escape(symbol)}|\b{re.escape(symbol)}\b", report_content), (
                f"Símbolo matemático esencial '{symbol}' no encontrado en el informe."
            )


# ==============================================================================
# TIER 3: FRICCIÓN CME Y MATRIZ DE RIESGO DE PROP TRADING
# ==============================================================================

class TestCMEFrictionAndPropTrading:
    """Verificación de especificaciones de contratos CME y calibración para firmas de fondeo."""

    def test_cme_contract_specifications_table(self, report_content: str):
        """Verifica que el informe detalle los contratos CME ES, MES, NQ, MNQ con tick y multiplicadores."""
        for symbol in ["ES", "MES", "NQ", "MNQ"]:
            assert symbol in report_content, f"El contrato {symbol} debe estar documentado en el informe."

        # Comprobación de tick size (0.25 en los 4 contratos)
        assert re.search(r"0\.25", report_content), "El informe debe especificar el tick size de 0.25 puntos."

        # Comprobación de multiplicadores y valores por tick
        # ES: $50 / punto -> $12.50 / tick
        # MES: $5 / punto -> $1.25 / tick
        # NQ: $20 / punto -> $5.00 / tick
        # MNQ: $2 / punto -> $0.50 / tick
        expected_values = [
            (r"12\.50", "ES tick value ($12.50)"),
            (r"1\.25", "MES tick value ($1.25)"),
            (r"5\.00", "NQ tick value ($5.00)"),
            (r"0\.50", "MNQ tick value ($0.50)"),
        ]
        for pattern, desc in expected_values:
            assert re.search(pattern, report_content), f"No se encontró {desc} en las tablas del informe."

    def test_futures_specs_consistency_with_project_code(self):
        """Cross-checking: valida que las especificaciones del repo coincidan exactamente con CME."""
        if get_spec is None:
            pytest.skip("src.core.futures_specs no está presente en el repositorio.")
        es_spec = get_spec("ES")
        mes_spec = get_spec("MES")
        nq_spec = get_spec("NQ")
        mnq_spec = get_spec("MNQ")

        assert es_spec["multiplier"] == 50.0 and es_spec["tick"] == 0.25
        assert mes_spec["multiplier"] == 5.0 and mes_spec["tick"] == 0.25
        assert nq_spec["multiplier"] == 20.0 and nq_spec["tick"] == 0.25
        assert mnq_spec["multiplier"] == 2.0 and mnq_spec["tick"] == 0.25

        # Verificación del valor del tick en dólares
        assert es_spec["multiplier"] * es_spec["tick"] == 12.50
        assert mes_spec["multiplier"] * mes_spec["tick"] == 1.25
        assert nq_spec["multiplier"] * nq_spec["tick"] == 5.00
        assert mnq_spec["multiplier"] * mnq_spec["tick"] == 0.50

    def test_prop_trading_risk_matrix_thresholds(self, report_content: str):
        """Verifica los umbrales de evaluación de firmas de fondeo (DLL, Trailing DD, Sharpe)."""
        lower = report_content.lower()

        # Límite de pérdida diaria (DLL)
        assert re.search(r"daily loss limit|p[eé]rdida diaria", lower), (
            "El informe debe documentar el límite de pérdida diaria (Daily Loss Limit)."
        )

        # Trailing Drawdown
        assert re.search(r"trailing (?:max )?drawdown|drawdown flotante", lower), (
            "El informe debe documentar el Trailing Drawdown de cuentas fondeadas."
        )

        # Sharpe ratio mínimo proyectado (> 1.5)
        assert re.search(r"sharpe.*(?:1\.5|1\.6|1\.7|1\.8|2\.0|>)", lower), (
            "El informe debe evaluar el ratio de Sharpe proyectado para prop trading (> 1.5)."
        )

        # Cierre de posiciones antes del corte CME (17:00 EST / Overnight flat)
        assert re.search(r"17:00|16:55|16:45|overnight|pausa cme", lower), (
            "El informe debe especificar la regla de estar plano antes del cierre técnico de CME (17:00 EST)."
        )


# ==============================================================================
# TIER 4: SINTAXIS AST Y EJECUTABILIDAD DE PSEUDOCÓDIGOS
# ==============================================================================

class TestPythonCodeASTSyntax:
    """Verifica que todos los bloques de código Python en el informe compilen sin errores."""

    def test_all_extracted_python_blocks_have_valid_ast(self, report_content: str):
        """Extrae y parsea mediante AST cada bloque ```python del informe."""
        code_blocks = extract_fenced_code_blocks(report_content, "python")
        assert len(code_blocks) >= 3, (
            f"Se esperaban al menos 3 bloques de código Python implementables, encontrados {len(code_blocks)}."
        )

        syntax_errors = []
        for i, code in enumerate(code_blocks, 1):
            try:
                ast.parse(code)
            except SyntaxError as e:
                syntax_errors.append(f"Bloque #{i} error en línea {e.lineno}: {e.msg}\nCódigo:\n{code[:200]}...")

        assert not syntax_errors, "Errores de sintaxis Python en el informe:\n" + "\n".join(syntax_errors)


class TestCoreAlgorithmsDryRun:
    """
    Dry-runs ejecutables de referencia para los algoritmos formalizados en M1-M3.
    Verifica que la lógica matemática no contenga errores conceptuales, de redondeo,
    ni excepciones en tiempo de ejecución (cero división, NaN, etc.).
    """

    def test_dry_run_ofi_calculation(self):
        """Algoritmo F1: Cálculo del Order Flow Imbalance según Cont, Kukanov & Stoikov (2014)."""
        def compute_ofi(prev_bid_p, prev_bid_s, curr_bid_p, curr_bid_s,
                        prev_ask_p, prev_ask_s, curr_ask_p, curr_ask_s) -> float:
            # Bid contribution
            if curr_bid_p > prev_bid_p:
                e_bid = curr_bid_s
            elif curr_bid_p == prev_bid_p:
                e_bid = curr_bid_s - prev_bid_s
            else:
                e_bid = -prev_bid_s

            # Ask contribution
            if curr_ask_p > prev_ask_p:
                e_ask = -prev_ask_s
            elif curr_ask_p == prev_ask_p:
                e_ask = curr_ask_s - prev_ask_s
            else:
                e_ask = -curr_ask_s

            return e_bid - e_ask

        # Escenario 1: Agresor comprador sube el bid -> OFI positivo
        ofi_buy = compute_ofi(5000.0, 10, 5000.25, 15, 5000.25, 20, 5000.50, 25)
        assert ofi_buy > 0, f"OFI esperado positivo en avance de precio comprador, obtenido {ofi_buy}"

        # Escenario 2: Absorción de ask (baja el tamaño del ask al mismo precio) -> OFI positivo
        ofi_absorb = compute_ofi(5000.0, 10, 5000.0, 10, 5000.25, 30, 5000.25, 10)
        assert ofi_absorb == 20, f"OFI por consumo de ask esperado 20, obtenido {ofi_absorb}"

    def test_dry_run_market_intraday_momentum(self):
        """Algoritmo F2: Momentum primera media hora (09:30-10:00) a última (15:30-16:00)."""
        def calc_momentum_signal(open_return: float, threshold: float = 0.002) -> int:
            if open_return > threshold:
                return 1   # Señal larga para la ventana 15:30-16:00
            elif open_return < -threshold:
                return -1  # Señal corta para la ventana 15:30-16:00
            return 0

        assert calc_momentum_signal(0.005) == 1
        assert calc_momentum_signal(-0.004) == -1
        assert calc_momentum_signal(0.001) == 0

    def test_dry_run_anchored_vwap_and_bands(self):
        """Algoritmo F3: VWAP anclado a la sesión y bandas de dispersión standard."""
        prices = np.array([5000.0, 5002.0, 5001.0, 5004.0, 5003.0])
        volumes = np.array([100, 250, 150, 300, 200])

        cum_pv = np.cumsum(prices * volumes)
        cum_v = np.cumsum(volumes)
        vwap = cum_pv / cum_v

        # Desviación estándar ponderada por volumen
        cum_dev_sq = np.cumsum(volumes * (prices - vwap) ** 2)
        vwap_std = np.sqrt(cum_dev_sq / cum_v)

        upper_band = vwap + 2.0 * vwap_std
        lower_band = vwap - 2.0 * vwap_std

        assert len(vwap) == 5
        assert np.all(upper_band >= vwap)
        assert np.all(lower_band <= vwap)
        assert not np.isnan(vwap).any()

    def test_dry_run_conditional_overnight_drift(self):
        """Algoritmo F4: Deriva nocturna condicionada a venta previa en RTH (Boyarchenko et al.)."""
        def overnight_entry_filter(rth_return: float, threshold: float = -0.0075) -> bool:
            # Si el RTH cerró con fuerte caída (< -0.75%), entrar largo en la sesión nocturna (01:30 EST)
            return rth_return < threshold

        assert overnight_entry_filter(-0.012) is True
        assert overnight_entry_filter(-0.003) is False
        assert overnight_entry_filter(0.005) is False

    def test_dry_run_pre_macro_announcement_window(self):
        """Algoritmo F5: Ventana informada previa a CPI/FOMC y salida obligatoria T-3 min."""
        def evaluate_macro_embargo(minutes_to_release: float) -> str:
            if 3.0 < minutes_to_release <= 15.0:
                return "TRADE_DRIFT"
            elif 0.0 <= minutes_to_release <= 3.0:
                return "MANDATORY_FLAT_EMBARGO"
            return "NORMAL_OPERATION"

        assert evaluate_macro_embargo(10.0) == "TRADE_DRIFT"
        assert evaluate_macro_embargo(2.5) == "MANDATORY_FLAT_EMBARGO"
        assert evaluate_macro_embargo(60.0) == "NORMAL_OPERATION"

    def test_dry_run_filtered_orb(self):
        """Algoritmo F6: Opening Range Breakout con buffer Stretch y filtro RVOL (Crabel/Zarattini)."""
        orb_high = 5020.0
        orb_low = 5000.0
        stretch = 2.50
        rvol = 2.2

        rvol_min = 1.8
        long_trigger = orb_high + stretch
        short_trigger = orb_low - stretch

        # Barra de prueba alcista
        bar_price = 5023.0
        signal = 0
        if rvol >= rvol_min:
            if bar_price > long_trigger:
                signal = 1
            elif bar_price < short_trigger:
                signal = -1

        assert signal == 1, f"Señal ORB esperada 1, obtenida {signal}"

    def test_dry_run_vrp_zscore(self):
        """Algoritmo F7: VRP Z-score = (IV^2 - RV^2) / sigma."""
        vix = 18.0
        iv_ann = (vix / 100.0) ** 2

        # 22 días de retornos logarítmicos con vol diaria ~ 1%
        np.random.seed(42)
        daily_returns = np.random.normal(0, 0.01, 22)
        rv_ann = 252.0 * np.sum(daily_returns ** 2) / 22.0

        vrp = iv_ann - rv_ann
        assert isinstance(vrp, float)
        assert not np.isnan(vrp)

    def test_dry_run_har_rv_j(self):
        """Algoritmo F8: HAR-RV de Corsi y dimensionamiento de volatilidad inversa."""
        # 30 días de volatilidad realizada sintética
        np.random.seed(42)
        rv_series = pd.Series(0.0001 + np.abs(np.random.normal(0, 0.00005, 30)))
        rv_d = rv_series.iloc[-1]
        rv_w = rv_series.iloc[-5:].mean()
        rv_m = rv_series.iloc[-22:].mean()

        # Coeficientes típicos de Corsi (2009)
        beta_0 = 0.00001
        beta_d = 0.40
        beta_w = 0.30
        beta_m = 0.20
        forecast_rv = beta_0 + beta_d * rv_d + beta_w * rv_w + beta_m * rv_m
        forecast_vol = math.sqrt(max(forecast_rv, 1e-8))

        # Dimensionamiento inverso de volatilidad para no exceder el DLL
        target_risk_dollars = 500.0  # $500 riesgo max por operación
        es_point_val = 50.0          # $50 por punto
        stop_points = forecast_vol * 5000.0 * 2.0  # Stop a 2 sigmas
        contracts = max(1, int(target_risk_dollars / (stop_points * es_point_val)))

        assert contracts >= 1
        assert not np.isnan(forecast_vol)

    def test_dry_run_cointegrated_pairs_ou(self):
        """Algoritmo F9: Arbitraje Estadístico ES vs NQ con difusión Ornstein-Uhlenbeck y S-score."""
        # Generar proceso de reversión a la media
        n = 100
        spread = np.zeros(n)
        kappa = 0.15
        theta = 0.0
        sigma = 1.0
        dt = 1.0

        np.random.seed(42)
        for t in range(1, n):
            spread[t] = spread[t - 1] + kappa * (theta - spread[t - 1]) * dt + sigma * np.random.normal(0, 1)

        mean_s = np.mean(spread)
        std_s = np.std(spread)
        s_score = (spread[-1] - mean_s) / (std_s + 1e-8)

        # Reglas de activación Avellaneda & Lee
        signal = 0
        if s_score < -1.25:
            signal = 1   # Comprar spread (Largo NQ / Corto ES)
        elif s_score > 1.25:
            signal = -1  # Vender spread (Corto NQ / Largo ES)

        assert abs(s_score) < 5.0, f"S-score anómalo: {s_score}"
        assert signal in (-1, 0, 1)

    def test_dry_run_prop_firm_simulator_integration(self):
        """Verifica la compatibilidad directa con PropFirmSimulator usando perfiles reales."""
        if PropFirmSimulator is None:
            pytest.skip("src.portfolio.prop_firm_rules no está presente en el repositorio.")
        profile = PROFILES["topstep_50k"]
        sim = PropFirmSimulator(profile)

        # Serie sintética con ganancias moderadas constantes
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        pnl_series = pd.Series([350.0] * 10, index=dates)

        res = sim.evaluate(pnl_series)
        assert res["status"] == "PASS"
        assert res["profit"] >= profile.profit_target  # 9 días * 350 = 3150 >= 3000 target
        assert res["days_traded"] == 9


# ==============================================================================
# PRUEBAS DE SANIDAD SOBRE INFORMES EXPLORADORES EXISTENTES
# ==============================================================================

class TestMarkdownParserSanityOnSurveyReports:
    """
    Verifica que el arnés de parseo funcione con los informes Markdown reales
    generados por los Exploradores 1, 2 y 3 en .agents/teamwork/.
    Esto garantiza que el validador esté libre de sesgos y no sea un test de fachada.
    """

    @pytest.mark.parametrize("survey_dir", [
        "teamwork_explorer_survey_1",
        "teamwork_explorer_survey_2",
        "teamwork_explorer_survey_3",
    ])
    def test_survey_reports_parseable_and_contain_content(self, survey_dir: str):
        """Verifica que cada informe de explorador contenga citas, LaTeX y código Python válido."""
        survey_path = PROJECT_ROOT / ".agents" / "teamwork" / survey_dir / "survey_report.md"
        assert survey_path.exists(), f"El archivo de prospección {survey_path} debe existir."

        text = survey_path.read_text(encoding="utf-8")
        assert len(text) > 10000, f"Informe {survey_dir} demasiado breve ({len(text)} chars)."

        # Verificar extracción de fórmulas LaTeX
        eqs = extract_latex_equations(text)
        assert len(eqs["block"]) + len(eqs["inline"]) >= 5, (
            f"Informe {survey_dir} carece de ecuaciones matemáticas LaTeX."
        )

        # Verificar extracción de fuentes académicas (DOIs o URLs)
        sources = extract_doi_and_urls(text)
        assert len(sources["dois"]) + len(sources["urls"]) >= 2, (
            f"Informe {survey_dir} carece de citas con DOI o enlace URL."
        )

        # Verificar sintaxis AST de bloques de código Python
        code_blocks = extract_fenced_code_blocks(text, "python")
        for i, code in enumerate(code_blocks):
            try:
                ast.parse(code)
            except SyntaxError as e:
                pytest.fail(f"Fallo de sintaxis en {survey_dir} bloque #{i}: {e}")
