# TEST_INFRA — Arquitectura y Metodología de Verificación Automatizada E2E

## 1. Misión y Filosofía de Pruebas

El presente documento establece la filosofía, arquitectura y especificación metodológica de la infraestructura de pruebas automatizadas E2E implementada en `tests/test_math_strategies_verification.py`.

El objetivo central de este sistema de verificación es auditar con rigor analítico, estático y semántico los 9 documentos de especificación matemática formal de estrategias cuantitativas para futuros CME (ES y NQ) generados en el directorio `reports/estrategias_matematicas/`.

### Principios Rectores:
1. **Auditoría Forense sin Concesiones**: Ninguna afirmación teórica se da por válida sin una prueba automatizada que certifique su presencia y consistencia formal.
2. **Prohibición Terminante de Código (Zero-Code Integrity)**: Los documentos de entrega están concebidos como especificaciones matemáticas puras listas para ser provistas a chats especializados de implementación. Se prohíbe taxativamente la inclusión de código ejecutable, bloques cercados de lenguajes de programación (Python, C++, R, Bash, etc.) o scripts.
3. **Erradicación de Indicadores Convencionales Descartados**: Vetar de forma absoluta el empleo de indicadores minoristas estándar (SMA o RSI convencional) o referencias empíricas preliminares que degraden la calidad institucional del marco cuantitativo.
4. **Formalismo Matemático en LaTeX**: Cada variable, modelo de señal, regla de desinversión y criterio de dimensionamiento debe formularse mediante ecuaciones analíticas rigurosas en notación LaTeX ($ o $$).
5. **Autocontención Modular (100% Standalone)**: Cada uno de los 9 documentos debe ser plenamente autosuficiente, permitiendo a cualquier operador o desarrollador transferir un archivo individual sin requerir dependencias externas ni otros documentos complementarios.

---

## 2. Arquitectura de Verificación en 4 Niveles (Tiers 1 a 4)

La suite de pruebas `tests/test_math_strategies_verification.py` organiza sus 146 casos de prueba en 4 niveles jerárquicos ortogonales:

```
+-----------------------------------------------------------------------------------+
|                        SUITE DE VERIFICACIÓN E2E (146 TESTS)                     |
+-----------------------------------------------------------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
+-------------------+           +-------------------+           +-------------------+
|      TIER 1       |           |      TIER 2       |           |      TIER 3       |
| Estructura y      |           | Restricciones     |           | Secciones         |
| Modularidad       |           | Negativas         |           | Obligatorias      |
| (20 tests)        |           | (27 tests)        |           | (54 tests)        |
+-------------------+           +-------------------+           +-------------------+
         |                               |                               |
         +-------------------------------+-------------------------------+
                                         |
                                         v
                                +-------------------+
                                |      TIER 4       |
                                | Rigor LaTeX y     |
                                | Autocontención    |
                                | (45 tests)        |
                                +-------------------+
```

### Tier 1: Estructura, Existencia de Entregables y Modularidad
- **Objetivo**: Asegurar la existencia física, nomenclatura exacta y sustancialidad de cada entregable.
- **Validaciones**:
  * `test_tier1_strategies_directory_exists`: Existencia y validez del directorio `reports/estrategias_matematicas/`.
  * `test_tier1_all_nine_files_exist`: Presencia verificada de los 9 archivos con nombres exactos (`F1` a `F9`).
  * `test_tier1_files_non_empty_and_substantial`: Umbral mínimo de profundidad analítica ($\ge 80$ líneas y $\ge 2,500$ bytes por archivo).
  * `test_tier1_exact_deliverables_catalog`: Integridad del conjunto sin ausencias ni archivos parásitos.

### Tier 2: Restricciones Negativas Estrictas e Integridad Analítica
- **Objetivo**: Blindar el marco teórico contra contaminaciones de código y métricas minoristas descartadas.
- **Validaciones**:
  * `test_tier2_strict_no_programming_code_blocks`: Inspección de bloques cercados (```python, ```cpp, ```sh, etc.). Cero tolerancia a bloques ejecutables.
  * `test_tier2_strict_no_executable_programming_keywords`: Detección estática de directivas de lenguajes (`import numpy`, `def `, `class `, `pip install`, `cout <<`, etc.).
  * `test_tier2_strict_no_traditional_retail_indicators`: Detección y veto estricto de SMA y RSI convencional como señales o indicadores activos, permitiendo únicamente menciones en declaraciones explícitas de descarte o tablas de auditoría.

### Tier 3: Secciones Obligatorias y Especificación Formal por Estrategia
- **Objetivo**: Auditar la completitud estructural y teórica exigida en `ORIGINAL_REQUEST.md` y `PROJECT.md`.
- **Validaciones**:
  * `test_tier3_all_mandatory_sections_present`: Existencia de las 5 secciones obligatorias:
    1. Variables y Espacio de Estados
    2. Ecuaciones de Señal
    3. Ecuaciones de Salida y Time-Decay
    4. Dimensionamiento de Posición y Control de Riesgo Prop Trading
    5. Rúbrica de Verificación Analítica Objetiva
  * `test_tier3_prop_trading_risk_components`: Verificación explícita de los 3 componentes contractuales de Prop Trading: Límite de Pérdida Diaria ($DLL$), Trailing Max Drawdown ($MTDD$) y escalado modular por volatilidad / distancia de riesgo.
  * `test_tier3_variable_domains_and_units_specified`: Especificación unívoca de dominios matemáticos ($\mathbb{R}$, $\mathbb{N}$, $[a, b]$, $\in$) y unidades físicas de futuros (USD, ticks, puntos, contratos).
  * `test_tier3_signal_equations_formalism`: Formulación analítica de señales para condiciones Long, Short y Neutral.
  * `test_tier3_exit_equations_formalism`: Ecuaciones formales de salida con Take Profit (o captura de beneficio), Stop Loss matemático y Time-Decay / salida terminal.
  * `test_tier3_strategy_specific_foundations`: Citas de autores académicos canónicos, conceptos teóricos distintivos y ecuaciones representativas de cada modelo (F1 a F9).

### Tier 4: Rigor Formal LaTeX, Autocontención y Robustez
- **Objetivo**: Evaluar la densidad matemática, balance sintáctico y modularidad 100% aislada.
- **Validaciones**:
  * `test_tier4_latex_equation_density`: Exigencia de alta densidad matemática ($\ge 4$ ecuaciones display `$$` o $\ge 15$ expresiones LaTeX totales).
  * `test_tier4_latex_syntax_balance`: Verificación de consistencia y número par de delimitadores display `$$`.
  * `test_tier4_self_contained_no_cross_file_dependencies`: Cero referencias dependientes no resueltas entre estrategias ("ver archivo F1", "como se definió en F2", etc.).
  * `test_tier4_instrument_and_market_parameters_self_contained`: Definición autónoma de instrumentos CME (ES, NQ, MES o MNQ) y sus multiplicadores/ticks en cada documento.
  * `test_tier4_objective_verification_rubric_completeness`: Auditoría de la tabla de rúbrica interna con verificación de los 4 ejes clave (cero código, exclusión SMA/RSI, condiciones analíticas y gestión de riesgo prop firm).

---

## 3. Catálogo de Estrategias y Mapeo Canónico (F1 a F9)

| ID | Archivo Canónico | Título y Autores Clave | Núcleo Analítico |
|---|---|---|---|
| **F1** | `F1_market_intraday_momentum.md` | Market Intraday Momentum (Gao et al., Bogousslavsky) | Retorno matutino $r_{first, t}$, filtro RVOL institucional, umbral por volatilidad $\sigma_{daily, t}$, salida MOC $15:58$ EST. |
| **F2** | `F2_filtered_orb.md` | Filtered Quantitative ORB (Crabel, Zarattini et al.) | ORB 15-min, compresión NR4/Inside Day, filtro $RVOL \ge 1.8$, amortiguador Stretch, invalidación en punto medio del rango. |
| **F3** | `F3_anchored_vwap_value_area.md` | Anchored VWAP & Value Area (Madhavan, Gatheral, Steidlmayer) | Bandas $\pm 2\sigma$ ancladas, Auction Market Theory (VAH, VAL, POC), filtro de régimen ADX $< 25$, reversión a la media rotacional. |
| **F4** | `F4_cointegracion_es_nq_ornstein_uhlenbeck.md` | Cointegración ES vs NQ / O-U (Avellaneda & Lee, Hasbrouck) | Difusión de Ornstein-Uhlenbeck sobre spread residual, S-score, vida media $\tau_{1/2}$, cobertura Dollar-Beta Neutral (10 MNQ vs 17 MES). |
| **F5** | `F5_order_flow_imbalance_cont.md` | Order Flow Imbalance Cont et al. (Cont, Kukanov & Stoikov) | OFI Nivel 1, función de impacto de flujo discreto $e_n$, variaciones de profundidad bid/ask $\Delta q$, Z-score rodante. |
| **F6** | `F6_multi_level_ofi.md` | Multi-Level OFI (Xu, Gould & Howison) | Desequilibrio multiescala $OFI^{(k)}$, decaimiento espacial de potencias $k^{-\alpha}$, descomposición PCA espectral, índice $\Phi_{coherence}$. |
| **F7** | `F7_meta_labeling_triple_barrier.md` | Meta-Labeling & Triple Barrier (López de Prado) | Clasificador primario desacoplado de meta-modelo probabilístico $P(y=1|x)$, bet sizing continuo, geometría dinámica de triple barrera. |
| **F8** | `F8_modelos_econometricos_har_rv_j.md` | Modelos Econométricos HAR-RV-J (Corsi, Andersen et al., Moreira & Muir) | Descomposición continua y de saltos mediante Bipower Variation ($BV_t$), componentes diario, semanal y mensual, escalado inverso $1/\sigma$. |
| **F9** | `F9_anomalias_temporales_vrp.md` | Anomalías Temporales & VRP (Bollerslev et al., Boyarchenko et al., Kurov et al.) | Variance Risk Premium $IV^2 - RV^2$, captura de deriva nocturna (01:30–08:30 EST), salida macro obligatoria en $T-3$ min (previa a embargo). |

---

## 4. Comandos de Ejecución y Diagnóstico

Para ejecutar la suite de verificación automatizada:

```powershell
# Ejecución completa con reporte detallado
python -m pytest tests/test_math_strategies_verification.py -v

# Ejecución concisa con resumen de fallos
python -m pytest tests/test_math_strategies_verification.py -q

# Ejecución focalizada por Tier específico
python -m pytest tests/test_math_strategies_verification.py -k "TestTier1" -v
python -m pytest tests/test_math_strategies_verification.py -k "TestTier2" -v
python -m pytest tests/test_math_strategies_verification.py -k "TestTier3" -v
python -m pytest tests/test_math_strategies_verification.py -k "TestTier4" -v

# Ejecución focalizada sobre una estrategia individual (ejemplo F4)
python -m pytest tests/test_math_strategies_verification.py -k "F4" -v
```

---

## 5. Protocolo de Escalación de Defectos

En cumplimiento con el rol de **Test Writer / QA Specialist**:
1. Los agentes de pruebas **únicamente modifican código de pruebas** (`tests/test_math_strategies_verification.py` y artefactos de documentación de pruebas).
2. Si un documento viola una restricción formal (ejemplo: presencia de bloque de código o ausencia de una sección obligatoria), el test fallará de manera estricta y el fallo será **escalado al agente redactor / worker** responsable mediante un reporte descriptivo.
3. No se toleran pruebas fachada (*facade tests*) que pasen sin verificar la verdadera consistencia matemática ni pruebas diseñadas ad-hoc para ocultar omisiones en los documentos.
