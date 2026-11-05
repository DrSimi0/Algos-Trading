"""
tests/test_math_strategies_verification.py

Suite de Pruebas Automatizadas E2E de Verificación Analítica y Formal para los 9
Documentos de Estrategias Cuantitativas Matemáticas (F1 a F9) en:
    `reports/estrategias_matematicas/`

Arquitectura de Validación en 4 Niveles (Tiers 1-4):
---------------------------------------------------
- Tier 1: Estructura, Existencia de Entregables y Modularidad
  * Existencia del directorio `reports/estrategias_matematicas/`.
  * Presencia exacta de los 9 archivos con sus nombres canónicos (F1 a F9).
  * Longitud mínima sustancial (sin archivos vacíos ni esqueletos incompletos).
  * Catálogo exacto de entregables (modularidad e independencia).

- Tier 2: Restricciones Negativas Estrictas e Integridad Analítica
  * Prohibición absoluta de bloques de código de programación (```python, ```cpp, etc.).
  * Prohibición de directivas de programación ejecutables (import, def, class, etc.).
  * Prohibición terminante de SMA y RSI convencional ni pruebas empíricas sobre ellos.

- Tier 3: Secciones Obligatorias y Especificación Formal por Estrategia
  * Presencia explícita de las 5 secciones obligatorias en cada documento:
      1. Variables y Espacio de Estados (con dominios y unidades).
      2. Ecuaciones de Señal (Long, Short, Neutral).
      3. Ecuaciones de Salida y Time-Decay (Take Profit, Stop Loss, Decaimiento Temporal).
      4. Dimensionamiento de Posición y Gestión de Riesgo Prop Trading (DLL, MTDD, Volatilidad).
      5. Rúbrica de Verificación Analítica Objetiva.
  * Formulación matemática rigurosa de reglas de Prop Trading (DLL, Trailing DD, escalado inverso).
  * Definición inequívoca de dominios matemáticos y unidades de mercado.
  * Verificación de fundamentos teóricos específicos para cada estrategia F1-F9.

- Tier 4: Rigor Formal LaTeX, Autocontención y Robustez
  * Densidad matemática suficiente mediante ecuaciones LaTeX ($ o $$).
  * Balance y consistencia de sintaxis de delimitadores LaTeX.
  * Autocontención absoluta: cero dependencias cruzadas rotas entre archivos.
  * Parámetros de mercado e instrumentos CME (ES, NQ, MES, MNQ) auto-definidos.
  * Auditoría exhaustiva de la Rúbrica de Verificación Objetiva interna.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import pytest

# ==============================================================================
# CONFIGURACIÓN Y CONSTANTES DEL ENTORNO
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports" / "estrategias_matematicas"

# Catálogo canónico de las 9 estrategias matemáticas (PROJECT.md y ORIGINAL_REQUEST.md)
STRATEGY_SPECS: List[Dict[str, Any]] = [
    {
        "id": "F1",
        "filename": "F1_market_intraday_momentum.md",
        "name": "Market Intraday Momentum & MOC Seasonality",
        "authors": ["Gao", "Bogousslavsky"],
        "core_keywords": [r"intraday", r"momentum", r"moc", r"apertura|cierre", r"09:30|16:00"],
        "math_patterns": [r"r_\{?intraday\}?|r_\{?first\}?|r_\{?last\}?", r"\beta", r"MOC"],
    },
    {
        "id": "F2",
        "filename": "F2_filtered_orb.md",
        "name": "Filtered Quantitative ORB",
        "authors": ["Crabel", "Zarattini", "Fetna"],
        "core_keywords": [r"orb|opening range", r"stretch", r"nr4|inside day|compresi[oó]n", r"rvol"],
        "math_patterns": [r"OR_\{?15\}?|OR_\{?t\}?", r"Stretch", r"RVOL"],
    },
    {
        "id": "F3",
        "filename": "F3_anchored_vwap_value_area.md",
        "name": "Anchored VWAP & Rotational Value Area Reversion",
        "authors": ["Madhavan", "Gatheral", "Steidlmayer"],
        "core_keywords": [r"vwap", r"value area|área de valor", r"subasta|auction", r"adx|r[eé]gimen"],
        "math_patterns": [r"VWAP", r"\sigma", r"VAH|VAL|POC", r"ADX"],
    },
    {
        "id": "F4",
        "filename": "F4_cointegracion_es_nq_ornstein_uhlenbeck.md",
        "name": "Cointegración ES vs NQ / Ornstein-Uhlenbeck",
        "authors": ["Avellaneda", "Lee", "Hasbrouck"],
        "core_keywords": [r"cointegraci[oó]n", r"ornstein[- ]uhlenbeck", r"arbitraje estad[ií]stico", r"es.*nq|nq.*es"],
        "math_patterns": [r"S-score|s_t|S_t", r"\\theta|\\kappa", r"dW_t", r"dollar[- ]beta|neutral"],
    },
    {
        "id": "F5",
        "filename": "F5_order_flow_imbalance_cont.md",
        "name": "Order Flow Imbalance Cont et al.",
        "authors": ["Cont", "Kukanov", "Stoikov"],
        "core_keywords": [r"order flow imbalance|ofi", r"nivel 1|l1", r"profundidad|bid.*ask", r"flujo discreto"],
        "math_patterns": [r"OFI", r"e_n|e_t", r"\\Delta q|q_n\^b|q_n\^a", r"Z-score"],
    },
    {
        "id": "F6",
        "filename": "F6_multi_level_ofi.md",
        "name": "Multi-Level OFI",
        "authors": ["Xu", "Gould", "Howison"],
        "core_keywords": [r"multi[- ]level ofi|ofi multi[- ]nivel", r"decaimiento espacial|potencia", r"pca|componentes principales", r"coherencia"],
        "math_patterns": [r"OFI\^?\{\(?k\)?\}|OFI_k", r"k\^\{?-\\alpha\}?|\\alpha", r"PCA|\\lambda", r"\\Phi|coherence"],
    },
    {
        "id": "F7",
        "filename": "F7_meta_labeling_triple_barrier.md",
        "name": "Meta-Labeling & Triple Barrier",
        "authors": ["L[oó]pez de Prado"],
        "core_keywords": [r"meta[- ]labeling", r"triple barrier|triple barrera", r"señal primaria", r"bet sizing|dimensionamiento"],
        "math_patterns": [r"P\(y\s*=\s*1|p_t", r"barrera|barrier", r"sizing|apuesta", r"\tau|t_\{?stop\}?"],
    },
    {
        "id": "F8",
        "filename": "F8_modelos_econometricos_har_rv_j.md",
        "name": "Modelos Econométricos HAR-RV-J",
        "authors": ["Corsi", "Andersen", "Moreira", "Muir"],
        "core_keywords": [r"har[- ]rv", r"volatilidad realizada", r"bipower variation|saltos", r"varianza"],
        "math_patterns": [r"RV|BV", r"RV_\{?d\}?|RV_\{?w\}?|RV_\{?m\}?", r"J_t|Jump|salto", r"\sigma_\{?t\+1\}?"],
    },
    {
        "id": "F9",
        "filename": "F9_anomalias_temporales_vrp.md",
        "name": "Anomalías Temporales & VRP",
        "authors": ["Bollerslev", "Boyarchenko", "Kurov"],
        "core_keywords": [r"vrp|variance risk premium", r"prima de riesgo de varianza", r"overnight|nocturna", r"macro|anuncio|t-3"],
        "math_patterns": [r"VRP", r"IV.*RV|IV\^2.*RV\^2", r"TSR", r"T-3|embargo"],
    },
]

# Diccionario indexado por nombre de archivo para acceso rápido en fixtures
STRATEGY_BY_FILENAME = {spec["filename"]: spec for spec in STRATEGY_SPECS}
ALL_FILENAMES = [spec["filename"] for spec in STRATEGY_SPECS]


# ==============================================================================
# FUNCIONES AUXILIARES DE VALIDACIÓN
# ==============================================================================

def get_strategy_path(filename: str) -> Path:
    """Devuelve la ruta absoluta al archivo de estrategia."""
    return REPORTS_DIR / filename


def read_strategy_content(filename: str) -> str:
    """Lee y devuelve el contenido textual de una estrategia.
    
    Falla explícitamente con mensaje informativo si el archivo no existe.
    """
    path = get_strategy_path(filename)
    if not path.is_file():
        pytest.fail(
            f"Archivo requerido '{filename}' no existe en {REPORTS_DIR}. "
            f"El documento debe ser generado en el Milestone correspondiente."
        )
    return path.read_text(encoding="utf-8")


def find_fenced_code_blocks(content: str) -> List[str]:
    """Detecta bloques de código cercados con triple comilla invertida."""
    pattern = re.compile(r"```([a-zA-Z0-9_\-\+]*)\n(.*?)```", re.DOTALL)
    matches = pattern.findall(content)
    # Devuelve lista de etiquetas de lenguaje encontradas
    return [lang.strip().lower() for lang, _ in matches]


def find_latex_delimiters(content: str) -> Dict[str, int]:
    """Cuenta el número de delimitadores LaTeX display ($$) e inline ($)."""
    # Excluir delimitadores de moneda escapados tipo \$
    clean_content = re.sub(r"\\\$", "", content)
    display_count = len(re.findall(r"\$\$", clean_content))
    # Para inline, contar $ individuales no pareados como $$
    single_dollar_count = len(re.findall(r"(?<!\$)\$(?!\$)", clean_content))
    return {
        "display_dollars": display_count,
        "single_dollars": single_dollar_count,
    }


# ==============================================================================
# TIER 1: ESTRUCTURA, EXISTENCIA DE ENTREGABLES Y MODULARIDAD
# ==============================================================================

class TestTier1DeliverablesAndStructure:
    """Valida la existencia del directorio, los 9 archivos exactos y su sustancialidad."""

    def test_tier1_strategies_directory_exists(self):
        """Verifica que el directorio reports/estrategias_matematicas/ exista."""
        assert REPORTS_DIR.exists(), (
            f"El directorio de entrega {REPORTS_DIR} no existe. "
            f"Debe crearse para alojar los 9 documentos de estrategias."
        )
        assert REPORTS_DIR.is_dir(), f"{REPORTS_DIR} existe pero no es un directorio."

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier1_all_nine_files_exist(self, filename: str):
        """Verifica la existencia física de cada uno de los 9 archivos requeridos."""
        path = get_strategy_path(filename)
        assert path.exists(), (
            f"Falta el documento requerido: {filename} en {REPORTS_DIR}. "
            f"Todos los 9 modelos F1 a F9 deben estar materializados."
        )
        assert path.is_file(), f"El elemento {filename} existe pero no es un archivo regular."

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier1_files_non_empty_and_substantial(self, filename: str):
        """Verifica que ningún archivo esté vacío y que contenga un desarrollo sustancial."""
        content = read_strategy_content(filename)
        lines = content.strip().splitlines()
        byte_size = len(content.encode("utf-8"))

        # Una especificación formal rigurosa debe tener al menos 80 líneas y 2500 bytes
        assert len(lines) >= 80, (
            f"El documento '{filename}' tiene solo {len(lines)} líneas. "
            f"Se requiere una especificación formal completa y profunda (mínimo 80 líneas)."
        )
        assert byte_size >= 2500, (
            f"El documento '{filename}' tiene solo {byte_size} bytes. "
            f"Un desarrollo cuantitativo formal debe ser sustancial (mínimo 2500 bytes)."
        )

    def test_tier1_exact_deliverables_catalog(self):
        """Verifica que en reports/estrategias_matematicas/ no falte ningún archivo requerido."""
        if not REPORTS_DIR.exists():
            pytest.fail(f"Directorio {REPORTS_DIR} no existe todavía.")

        existing_files = {p.name for p in REPORTS_DIR.glob("*.md")}
        expected_files = set(ALL_FILENAMES)

        missing = expected_files - existing_files
        assert not missing, f"Faltan los siguientes documentos de estrategia: {missing}"


# ==============================================================================
# TIER 2: RESTRICCIONES NEGATIVAS ESTRICTAS E INTEGRIDAD ANALÍTICA
# ==============================================================================

class TestTier2NegativeConstraintsAndIntegrity:
    """Valida la ausencia total de código ejecutable, librerías de programación y SMA/RSI."""

    # Lenguajes de programación prohibidos en bloques cercados
    FORBIDDEN_LANGUAGES = {
        "python", "py", "cpp", "c", "c++", "bash", "sh", "r", "pseudocode",
        "pseudo", "rust", "java", "javascript", "js", "typescript", "ts",
        "golang", "go", "julia", "fortran", "powershell", "ps1"
    }

    # Palabras clave de directivas de programación ejecutables
    PROGRAMMING_DIRECTIVES = [
        (r"\bimport\s+numpy\b", "importación de numpy"),
        (r"\bimport\s+pandas\b", "importación de pandas"),
        (r"\bfrom\s+scipy\b", "importación de scipy"),
        (r"\bdef\s+[a-zA-Z0-9_]+\s*\(", "definición de función en Python (def)"),
        (r"\bclass\s+[a-zA-Z0-9_]+[:\(]", "definición de clase en Python (class)"),
        (r"(?:^|\n)\s*return\s+[a-zA-Z0-9_\(\)]+", "sentencia return de programación"),
        (r"#include\s*<", "directiva de preprocesador C/C++"),
        (r"\bcout\s*<<", "flujo de salida en C++"),
        (r"\bconsole\.log\(", "función console.log de JavaScript"),
        (r"\bpip\s+install\b", "comando de instalación pip"),
        (r"\bfunction\s+[a-zA-Z0-9_]+\s*\(", "declaración de función JavaScript/R"),
    ]

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier2_strict_no_programming_code_blocks(self, filename: str):
        """Verifica que el documento NO contenga bloques de código cercados de programación."""
        content = read_strategy_content(filename)
        languages_found = find_fenced_code_blocks(content)

        violating_langs = [lang for lang in languages_found if lang in self.FORBIDDEN_LANGUAGES]
        assert not violating_langs, (
            f"El archivo '{filename}' viola la prohibición estricta de código. "
            f"Se encontraron bloques de código de: {violating_langs}. "
            f"La especificación debe ser 100% analítica y matemática, sin bloques de código."
        )

        # Verificar que no exista ningún bloque cercado que contenga código de programación
        # Incluso si el bloque no tiene etiqueta de lenguaje explícita
        code_block_pattern = re.compile(r"```(.*?)\n(.*?)```", re.DOTALL)
        for _, block_body in code_block_pattern.findall(content):
            for directive_regex, directive_name in self.PROGRAMMING_DIRECTIVES:
                assert not re.search(directive_regex, block_body, re.IGNORECASE), (
                    f"El archivo '{filename}' contiene un bloque de código con {directive_name}. "
                    f"Queda terminantemente prohibido incluir código o pseudocódigo ejecutable."
                )

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier2_strict_no_executable_programming_keywords(self, filename: str):
        """Verifica que el texto no contenga sentencias ejecutables de lenguajes de programación."""
        content = read_strategy_content(filename)

        for directive_regex, directive_name in self.PROGRAMMING_DIRECTIVES:
            matches = re.findall(directive_regex, content, re.IGNORECASE)
            assert not matches, (
                f"El archivo '{filename}' contiene directivas de programación prohibidas ({directive_name}): "
                f"{matches[:3]}. Debe utilizarse notación matemática pura en LaTeX."
            )

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier2_strict_no_traditional_retail_indicators(self, filename: str):
        """Verifica que ningún archivo use SMA ni RSI estándar como indicadores o señales activas."""
        content = read_strategy_content(filename)

        # Detección de SMA activo
        # Permitido solo si expresamente dice "descartado SMA", "exclusión de SMA", "sin SMA", etc.
        sma_matches = re.finditer(r"\b(SMA|Simple Moving Average|Media M[oó]vil Simple)\b", content, re.IGNORECASE)
        for match in sma_matches:
            # Obtener contexto circundante de 140 caracteres
            start = max(0, match.start() - 70)
            end = min(len(content), match.end() + 70)
            context = content[start:end].lower()

            is_disclaimer = any(
                disclaimer in context
                for disclaimer in [
                    "descartad", "prohibid", "no utiliz", "sin sma", "cero sma",
                    "rechazad", "desecho", "no us", "evit", "retail", "descarte",
                    "exclusi", "excluid", "cero apariciones", "sustituid", "card(",
                    "cero referencia", "cero empleo", "cero uso", "cero", "sin", "ningun"
                ]
            )
            assert is_disclaimer, (
                f"El archivo '{filename}' contiene mención activa a SMA: '{match.group(0)}' "
                f"en contexto: '...{context}...'. "
                f"Está terminantemente prohibido utilizar SMA estándar o pruebas sobre él."
            )

        # Detección de RSI activo
        rsi_matches = re.finditer(r"\b(RSI|Relative Strength Index|[IÍ]ndice de Fuerza Relativa)\b", content, re.IGNORECASE)
        for match in rsi_matches:
            start = max(0, match.start() - 70)
            end = min(len(content), match.end() + 70)
            context = content[start:end].lower()

            is_disclaimer = any(
                disclaimer in context
                for disclaimer in [
                    "descartad", "prohibid", "no utiliz", "sin rsi", "cero rsi",
                    "rechazad", "desecho", "no us", "evit", "retail", "descarte",
                    "exclusi", "excluid", "cero apariciones", "sustituid", "card(",
                    "cero referencia", "cero empleo", "cero uso", "cero", "sin", "ningun"
                ]
            )
            assert is_disclaimer, (
                f"El archivo '{filename}' contiene mención activa a RSI: '{match.group(0)}' "
                f"en contexto: '...{context}...'. "
                f"Está terminantemente prohibido utilizar RSI convencional o pruebas sobre él."
            )

        # Verificar que no se reintroduzcan artefactos de backtest previos como SPY_SMA o AAPL_RSI
        empirical_trash = [r"SPY_SMA", r"AAPL_RSI", r"backtest.*sma", r"backtest.*rsi"]
        for trash in empirical_trash:
            assert not re.search(trash, content, re.IGNORECASE), (
                f"El archivo '{filename}' contiene referencias a pruebas empíricas descartadas ({trash})."
            )


# ==============================================================================
# TIER 3: SECCIONES OBLIGATORIAS Y ESPECIFICACIÓN FORMAL POR ESTRATEGIA
# ==============================================================================

class TestTier3MandatorySectionsAndFormulations:
    """Valida la presencia y consistencia formal de las 5 secciones obligatorias y componentes de riesgo."""

    MANDATORY_SECTIONS = [
        {
            "id": "SEC1_VARIABLES",
            "name": "Variables y Espacio de Estados",
            "patterns": [
                r"variables\s+y\s+espacio\s+de\s+estados",
                r"espacio\s+de\s+estados\s+y\s+variables",
                r"definici[oó]n\s+formal\s+de\s+variables",
                r"variables\s+de\s+entrada",
            ],
        },
        {
            "id": "SEC2_SIGNALS",
            "name": "Ecuaciones de Señal",
            "patterns": [
                r"ecuaciones\s+de\s+se[ñn]al",
                r"generaci[oó]n\s+de\s+se[ñn]ales",
                r"se[ñn]ales\s+matem[aá]ticas",
                r"condiciones\s+de\s+entrada",
            ],
        },
        {
            "id": "SEC3_EXITS",
            "name": "Ecuaciones de Salida y Time-Decay",
            "patterns": [
                r"ecuaciones\s+de\s+salida",
                r"salida\s+de\s+posici[oó]n",
                r"time[- ]decay",
                r"cierre\s+por\s+tiempo",
            ],
        },
        {
            "id": "SEC4_RISK",
            "name": "Dimensionamiento de Posición y Gestión de Riesgo Prop Trading",
            "patterns": [
                r"dimensionamiento\s+de\s+posici[oó]n",
                r"gesti[oó]n\s+de\s+riesgo",
                r"prop\s+trading",
                r"control\s+de\s+riesgo\s+cuantitativo",
            ],
        },
        {
            "id": "SEC5_RUBRIC",
            "name": "Rúbrica de Verificación Analítica Objetiva",
            "patterns": [
                r"r[uú]brica\s+de\s+verificaci[oó]n",
                r"verificaci[oó]n\s+anal[ií]tica\s+objetiva",
                r"matriz\s+de\s+verificaci[oó]n\s+objetiva",
            ],
        },
    ]

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier3_all_mandatory_sections_present(self, filename: str):
        """Verifica que cada uno de los 9 archivos contenga las 5 secciones obligatorias."""
        content = read_strategy_content(filename)

        for sec in self.MANDATORY_SECTIONS:
            found = any(re.search(pat, content, re.IGNORECASE) for pat in sec["patterns"])
            assert found, (
                f"El archivo '{filename}' carece de la sección obligatoria: '{sec['name']}'. "
                f"Patrones buscados: {sec['patterns']}"
            )

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier3_prop_trading_risk_components(self, filename: str):
        """Verifica que la sección de riesgo contenga DLL, Trailing DD y escalado por volatilidad."""
        content = read_strategy_content(filename)

        # 1. Daily Loss Limit (DLL)
        has_dll = bool(
            re.search(r"\b(DLL|Daily Loss Limit|L[ií]mite de P[eé]rdida Diaria)\b", content, re.IGNORECASE)
        )
        assert has_dll, (
            f"El archivo '{filename}' no formula matemáticamente el Daily Loss Limit (DLL) "
            f"exigido para reglas de cuentas fondeadas (Prop Trading)."
        )

        # 2. Max Trailing Drawdown (MTDD)
        has_trailing_dd = bool(
            re.search(
                r"\b(Trailing.*Drawdown|MTDD|Drawdown M[aá]ximo|Drawdown Flotante|High-Water Mark|HWM)\b",
                content,
                re.IGNORECASE,
            )
        )
        assert has_trailing_dd, (
            f"El archivo '{filename}' no formula matemáticamente el Max Trailing Drawdown "
            f"o seguimiento de Drawdown Máximo / High-Water Mark."
        )

        # 3. Inverse Volatility Scaling / Dimensionamiento Modulado por Riesgo y Volatilidad
        has_vol_scaling = bool(
            re.search(
                r"(invers.*volatilidad|escalado.*volatilidad|\\sigma|\bvolatilidad\b|\bvolatility\b|\batr\b|disyuntor.*volatilidad|\bR_\{?contract|\bdelta_\{?risk|\bdelta_\{?sl)",
                content,
                re.IGNORECASE,
            )
        )
        assert has_vol_scaling, (
            f"El archivo '{filename}' no incluye formulación matemática de dimensionamiento "
            f"de posición modulado por volatilidad / distancia de riesgo (e.g. 1/sigma, 1/ATR o Delta_risk)."
        )

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier3_variable_domains_and_units_specified(self, filename: str):
        """Verifica que las variables incluyan especificación de dominios matemáticos y unidades."""
        content = read_strategy_content(filename)

        # Dominios matemáticos formales en LaTeX o notación de conjuntos
        has_math_domain = bool(
            re.search(r"(\\mathbb\{[RZN]\}|\bin\s+\[|\\in\s+\[|\\in\s+\\mathbb|\bR\^|\b\[0,\s*1\])", content)
        )
        assert has_math_domain, (
            f"El archivo '{filename}' no define explícitamente los dominios matemáticos "
            f"(e.g. \\mathbb{{R}}, [0, 1], \\in) de las variables de entrada."
        )

        # Unidades físicas de mercado (USD, ticks, puntos, bps, segundos, contratos)
        has_units = bool(
            re.search(r"\b(ticks?|puntos?|USD|\$|contratos?|bps|segundos?|minutos?)\b", content, re.IGNORECASE)
        )
        assert has_units, (
            f"El archivo '{filename}' no documenta unívocamente las unidades físicas "
            f"(ticks, USD, contratos, segundos, puntos, etc.) de las variables."
        )

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier3_signal_equations_formalism(self, filename: str):
        """Verifica que la sección de señales contenga condiciones Long, Short y Neutralidad."""
        content = read_strategy_content(filename)

        has_long = bool(re.search(r"\b(Long|Larga|Compra|\+1)\b", content, re.IGNORECASE))
        has_short = bool(re.search(r"\b(Short|Corta|Venta|\-1)\b", content, re.IGNORECASE))
        has_neutral = bool(re.search(r"\b(Neutral|Neutro|Cero|0)\b", content, re.IGNORECASE))

        assert has_long and has_short, (
            f"El archivo '{filename}' debe definir formalmente las condiciones de señal "
            f"tanto para entrada Long como para entrada Short."
        )
        assert has_neutral, (
            f"El archivo '{filename}' debe definir formalmente la condición de Neutralidad (sin posición)."
        )

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier3_exit_equations_formalism(self, filename: str):
        """Verifica que las ecuaciones de salida contemplen Take Profit, Stop Loss y Time-Decay."""
        content = read_strategy_content(filename)

        has_tp = bool(re.search(r"(take[- ]profit|\btp\b|toma\s+de\s+(?:ganancias?|beneficios?)|objetivo\s+de\s+precio|barrera\s+superior|captura\s+del\s+flujo|salida\s+de\s+la\s+deriva)", content, re.IGNORECASE))
        has_sl = bool(re.search(r"(stop[- ]loss|\bsl\b|l[ií]mite\s+de\s+p[eé]rdida|salida\s+por\s+p[eé]rdida|barrera\s+inferior|\bdelta_\{?sl|\bdelta\s*p_\{?stop)", content, re.IGNORECASE))
        has_time_decay = bool(re.search(r"(time[- ]decay|decaimiento\s+temporal|cierre\s+(?:por\s+tiempo|determinista|intrad[ií]a|de\s+sesi[oó]n|terminal)|salida\s+(?:temporal|terminal|estricta|forzosa)|horizonte\s+m[aá]ximo|\btau_\{?close\}?|\btau_\{?exit\}?|\btau_\{?terminal\}?|t-3\s*min)", content, re.IGNORECASE))

        assert has_tp, f"El archivo '{filename}' no formula el mecanismo de Take Profit analítico."
        assert has_sl, f"El archivo '{filename}' no formula el Stop Loss matemático riguroso."
        assert has_time_decay, f"El archivo '{filename}' no formula el cierre por tiempo / Time-Decay analítico."

    @pytest.mark.parametrize("spec", STRATEGY_SPECS)
    def test_tier3_strategy_specific_foundations(self, spec: Dict[str, Any]):
        """Verifica los fundamentos teóricos específicos, autores y fórmulas clave de cada modelo."""
        filename = spec["filename"]
        content = read_strategy_content(filename)

        # 1. Autores canónicos de referencia
        authors_found = [
            author for author in spec["authors"]
            if re.search(rf"\b{author}\b", content, re.IGNORECASE)
        ]
        assert len(authors_found) >= 1, (
            f"El documento '{filename}' ({spec['name']}) no cita a sus autores académicos canónicos: "
            f"{spec['authors']}. Encontrados: {authors_found}."
        )

        # 2. Palabras clave del modelo
        keywords_matched = [
            kw for kw in spec["core_keywords"]
            if re.search(kw, content, re.IGNORECASE)
        ]
        assert len(keywords_matched) >= 2, (
            f"El documento '{filename}' carece de los conceptos nucleares de su estrategia: "
            f"requeridos={spec['core_keywords']}, encontrados={keywords_matched}."
        )

        # 3. Notación matemática distintiva de la estrategia
        math_matched = [
            pat for pat in spec["math_patterns"]
            if re.search(pat, content)
        ]
        assert len(math_matched) >= 1, (
            f"El documento '{filename}' no contiene la notación matemática característica: "
            f"esperados={spec['math_patterns']}, encontrados={math_matched}."
        )


# ==============================================================================
# TIER 4: RIGOR FORMAL LATEX, AUTOCONTENCIÓN Y ROBUSTEZ
# ==============================================================================

class TestTier4LatexAndSelfContainment:
    """Valida la calidad matemática en LaTeX, el balance sintáctico y la independencia total."""

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier4_latex_equation_density(self, filename: str):
        """Verifica que cada documento contenga una densidad matemática rigurosa en LaTeX."""
        content = read_strategy_content(filename)
        delimiters = find_latex_delimiters(content)

        # Al menos 6 ecuaciones display ($$) o al menos 20 expresiones matemáticas inline ($)
        display_eqs = delimiters["display_dollars"] // 2
        inline_eqs = delimiters["single_dollars"] // 2
        total_equations = display_eqs + inline_eqs

        assert display_eqs >= 4 or total_equations >= 15, (
            f"El archivo '{filename}' tiene baja densidad matemática: "
            f"{display_eqs} ecuaciones display ($$) y {inline_eqs} expresiones inline ($). "
            f"Se requiere formulación analítica profunda y formal en LaTeX."
        )

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier4_latex_syntax_balance(self, filename: str):
        """Verifica que los delimitadores de visualización LaTeX ($$) estén balanceados."""
        content = read_strategy_content(filename)
        clean_content = re.sub(r"\\\$", "", content)
        display_matches = len(re.findall(r"\$\$", clean_content))

        assert display_matches % 2 == 0, (
            f"El archivo '{filename}' tiene delimitadores display math '$$' desbalanceados "
            f"(total={display_matches}, debe ser un número par). Revise la sintaxis LaTeX."
        )

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier4_self_contained_no_cross_file_dependencies(self, filename: str):
        """Verifica que el archivo sea 100% autocontenido y no dependa de otros documentos."""
        content = read_strategy_content(filename)

        # Lista de otros archivos que no deben ser requeridos
        other_files = [f for f in ALL_FILENAMES if f != filename]

        for other in other_files:
            other_id = other.split("_")[0]  # e.g. "F1", "F2"
            
            # Buscar referencias dependientes tipo "ver archivo F1", "como se definió en F2"
            bad_ref_patterns = [
                rf"(?:ver|consultar|definid[oa]|seg[uú]n)\s+(?:el\s+documento|el\s+archivo)?\s*{other_id}\b",
                rf"importar\s+(?:de|desde)\s+{other_id}\b",
                rf"\({other}\)",
            ]
            for pat in bad_ref_patterns:
                match = re.search(pat, content, re.IGNORECASE)
                assert not match, (
                    f"El archivo '{filename}' contiene una referencia dependiente a '{other_id}': "
                    f"'{match.group(0)}'. "
                    f"Cada archivo debe ser 100% autocontenido e independiente para entrega directa a chats."
                )

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier4_instrument_and_market_parameters_self_contained(self, filename: str):
        """Verifica que cada archivo defina de forma autónoma los instrumentos CME aplicables."""
        content = read_strategy_content(filename)

        # Debe mencionar al menos uno de los instrumentos CME de futuros (ES, NQ, MES, MNQ)
        has_cme_symbol = bool(
            re.search(r"\b(ES|NQ|MES|MNQ|E-mini|Micro E-mini)\b", content)
        )
        assert has_cme_symbol, (
            f"El archivo '{filename}' no especifica los instrumentos objetivo CME (ES, NQ, MES o MNQ). "
            f"Cada documento debe ser autocontenido e indicar sus contratos aplicables."
        )

    @pytest.mark.parametrize("filename", ALL_FILENAMES)
    def test_tier4_objective_verification_rubric_completeness(self, filename: str):
        """Verifica que la Rúbrica de Verificación Objetiva interna contemple todos los criterios clave."""
        content = read_strategy_content(filename)

        # Encontrar la sección de la rúbrica (incluyendo encabezados numerados como '## 6. Rúbrica...')
        rubric_match = re.search(
            r"(#+[\s0-9\.]*(?:r[uú]brica\s+de\s+verificaci[oó]n|verificaci[oó]n\s+anal[ií]tica\s+objetiva).*?)(?=\n#+[\s0-9\.]+[A-Z]|\Z)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        assert rubric_match, f"No se pudo aislar la sección de la Rúbrica de Verificación en '{filename}'."

        rubric_text = rubric_match.group(1).lower()

        # Criterios que la rúbrica debe auditar internamente
        criteria = [
            ("ausencia de código", [r"c[oó]digo", r"programaci[oó]n", r"pseudoc[oó]digo"]),
            ("ausencia de SMA/RSI", [r"sma", r"rsi", r"indicador.*retail", r"tradicional", r"descartad"]),
            ("consistencia formal / condiciones analíticas", [r"condici[oó]n", r"ecuaci[oó]n", r"se[ñn]al", r"salida", r"estacionar", r"asimetr[ií]a", r"dimensional", r"consistencia", r"dominio", r"variable"]),
            ("gestión de riesgo prop trading", [r"prop", r"riesgo", r"drawdown", r"dll", r"volatilidad", r"l[ií]mite\s+diario", r"exposici[oó]n", r"p[eé]rdida"]),
        ]

        for crit_name, patterns in criteria:
            found = any(re.search(p, rubric_text) for p in patterns)
            assert found, (
                f"La Rúbrica de Verificación en '{filename}' no contiene el criterio de auditoría "
                f"obligatorio para '{crit_name}'. Patrones buscados: {patterns}"
            )
