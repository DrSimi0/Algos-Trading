# TEST_INFRA — Arquitectura y Metodología de Verificación Automatizada E2E para Informes de Optimización

## 1. Misión y Filosofía de Pruebas

El presente documento formaliza la filosofía, arquitectura metodológica y catálogo de especificaciones de la infraestructura de pruebas automatizadas E2E implementada en `tests/test_optimization_reports_verification.py`.

El objetivo central de este sistema de verificación es auditar con rigor analítico, estático y semántico los 9 documentos de propuesta de optimización matemática para futuros CME (ES, MES, NQ, MNQ) en cuentas de Prop Trading ($50k–$100k USD) ubicados en el directorio `reports/optimizaciones_estrategias/`:
- `F1_optimizacion_market_intraday_momentum.md`
- `F2_optimizacion_filtered_orb.md`
- `F3_optimizacion_anchored_vwap_value_area.md`
- `F4_optimizacion_cointegracion_ou.md`
- `F5_optimizacion_order_flow_imbalance.md`
- `F6_optimizacion_multi_level_ofi.md`
- `F7_optimizacion_meta_labeling.md`
- `F8_optimizacion_modelos_har_rv_j.md`
- `F9_optimizacion_anomalias_temporales_vrp.md`

### Principios Rectores:
1. **Auditoría Forense sin Concesiones (Zero-Facade Testing)**: Cada prueba evalúa invariantes matemáticas y forenses reales derivadas de las exploraciones de Milestone 0 y los requisitos contractuales de Prop Trading. No se admiten pruebas triviales o fachada que aprueben sin validar el contenido crítico.
2. **Prohibición Terminante de Código (Zero-Code Integrity)**: Los documentos de entrega son especificaciones analíticas puras destinadas a chats de implementación externos. Se prohíbe de forma absoluta la inclusión de bloques cercados de código (```` ```python ````, ```` ```cpp ````, ```` ```sh ````, etc.) o sentencias ejecutables (`import`, `def`, `class`, `return`).
3. **Erradicación de Indicadores Convencionales Descartados**: Vetar de forma rigurosa el empleo de medias móviles tradicionales (SMA) o RSI estándar como indicadores o señales activas. Únicamente se toleran menciones en declaraciones explícitas de exclusión o rúbricas de descarte.
4. **Confinamiento Estricto In-Sample (IS 2023)**: Todas las optimizaciones deben sustentarse exclusivamente en las 20,280 barras de 5 minutos del año calendario 2023 (260 sesiones RTH). Los años 2024 (Out-of-Sample) y 2025 (Holdout) permanecen en cuarentena sellada e inviolable.
5. **Formalismo Matemático en LaTeX y Reglas Prop Trading**: Exigir notación formal LaTeX ($ y $$), dominios unívocos ($\mathbb{R}, \mathbb{N}$) y unidades físicas (USD, pts, ticks), con límites estrictos de Daily Loss Limit ($DLL \le \$1,500$ USD) y Trailing Max Drawdown ($MTDD \le \$3,000$ USD).
6. **Autocontención Modular (100% Standalone)**: Cada uno de los 9 documentos debe ser plenamente independiente y transferible sin dependencias cruzadas hacia otros reportes.

---

## 2. Arquitectura de Verificación en 4 Niveles (Tiers 1 a 4)

La suite de pruebas `tests/test_optimization_reports_verification.py` organiza sus 158 casos de prueba en 4 niveles jerárquicos ortogonales:

```
+-----------------------------------------------------------------------------------+
|               SUITE DE VERIFICACIÓN E2E DE REPORTES DE OPTIMIZACIÓN               |
|                                   (158 TESTS)                                     |
+-----------------------------------------------------------------------------------+
                                          |
         +--------------------------------+--------------------------------+
         |                                |                                |
         v                                v                                v
+-------------------+            +-------------------+            +-------------------+
|      TIER 1       |            |      TIER 2       |            |      TIER 3       |
| Existencia,       |            | Restricciones     |            | Formalismo        |
| Modularidad y     |            | Negativas e IS    |            | Matemático y Prop |
| Cero Código       |            | Confinamiento     |            | Trading Risk      |
| (55 tests)        |            | (27 tests)        |            | (63 tests)        |
+-------------------+            +-------------------+            +-------------------+
         |                                |                                |
         +--------------------------------+--------------------------------+
                                          |
                                          v
                                 +-------------------+
                                 |      TIER 4       |
                                 | Requerimientos    |
                                 | Profundos F3 y F4 |
                                 | (+ Censo F1-F9)   |
                                 | (13 tests)        |
                                 +-------------------+
```

### Tier 1: Existencia de Entregables, Modularidad y Cero Código
- **Propósito**: Garantizar la integridad física, sustancialidad analítica y ausencia absoluta de código ejecutable.
- **Validaciones**:
  * `test_all_nine_optimization_files_exist`: Existencia confirmada de los 9 archivos en `reports/optimizaciones_estrategias/`.
  * `test_files_are_substantial_and_comprehensive`: Cada archivo debe tener al menos 15,000 bytes y 200 líneas.
  * `test_mandatory_structural_sections_present`: Presencia de las 7 secciones obligatorias (Resumen, Diagnóstico IS, Formulación Matemática, Control de Riesgo Prop Trading, Impacto Cuantitativo, Rúbrica 8 Puntos R1-R8, Guía de Transferencia).
  * `test_zero_fenced_programming_code_blocks`: Inspección estática línea por línea; cero tolerancia a bloques cercados (```` ``` ````).
  * `test_zero_executable_programming_keywords`: Ausencia de directivas de lenguajes (`import`, `from`, `def`, `class`, `return`, `cout`, `console.log`).
  * `test_standalone_modularity_no_unresolved_cross_dependencies`: Cero referencias cruzadas no resueltas entre estrategias.

### Tier 2: Indicadores Descartados y Confinamiento In-Sample
- **Propósito**: Blindar el marco teórico contra heurísticas minoristas y sobreajuste por fuga de datos futuros.
- **Validaciones**:
  * `test_zero_active_use_of_sma_or_standard_rsi`: Veto al uso activo de SMA o RSI; permitidas solo en cláusulas explícitas de exclusión.
  * `test_explicit_in_sample_2023_certification`: Certificación explícita de fundamentación en el año 2023 (In-Sample).
  * `test_quarantine_of_out_of_sample_2024_and_holdout_2025`: Cuarentena inviolable de 2024 (OOS) y 2025 (Holdout); cero calibración ni uso de datos futuros.

### Tier 3: Formalismo Matemático y Límites de Riesgo Prop Trading
- **Propósito**: Verificar la densidad analítica y el cumplimiento de las reglas contractuales institucionales.
- **Validaciones**:
  * `test_latex_equation_density_and_delimiters`: Alta densidad matemática ($\ge 4$ ecuaciones display `$$` o $\ge 20$ inline `$`) y balance sintáctico.
  * `test_variable_domains_and_physical_units_specified`: Especificación unívoca de dominios ($\mathbb{R}, \mathbb{N}, \in$) y unidades físicas (USD, pts, ticks, sec, contratos).
  * `test_daily_loss_limit_equations_bounded_below_1500`: Ecuaciones formales de Daily Loss Limit ($DLL \le \$1,500$ USD).
  * `test_trailing_max_drawdown_equations_bounded_below_3000`: Ecuaciones formales de Trailing Max Drawdown ($MTDD \le \$3,000$ USD).
  * `test_inverse_volatility_and_risk_distance_scaling`: Dimensionamiento de posición inverso a la distancia de riesgo con función piso entera ($\lfloor \cdot \rfloor$).
  * `test_geometric_streak_attenuation_and_capital_preservation`: Presupuesto de riesgo modulado geométricamente por racha de pérdidas ($\phi(K) \propto 2^{-\min(K, \dots)}$).
  * `test_micro_contract_switching_and_cme_multipliers`: Parámetros contractuales CME (ES/MES, NQ/MNQ) y conmutación Standard/Micro.
  * `test_mandatory_intraday_flat_liquidation`: Liquidación forzosa intradiaria/previa al clearing CME (15:15, 15:45, 15:50, 16:45 EST).

### Tier 4: Requerimientos Cuantitativos Profundos de F3, F4 y Censo F1-F9
- **Propósito**: Auditar las soluciones a las fallas críticas descubiertas en Milestone 0.
- **Validaciones**:
  * `test_f3_monte_carlo_ruin_analysis_present`: Cuantificación de ruina base en Monte Carlo (70.36%-98.96%), teorema de Erdős-Rényi para longitud de rachas y arrastre de comisiones ($4,132 vs -$455 USD).
  * `test_f3_closed_form_mathematical_resolution`: Formulación cerrada de pendiente normalizada de VWAP ($\kappa_{norm} \le 0.085$), compuerta de Área de Valor ($\Phi_{AMT}$), bandas asimétricas por sesgo, Trailing Protect SL y certificación de supervivencia MC $\ge 95\%$.
  * `test_f4_forensic_audit_six_causes_of_anomaly`: Auditoría forense exhaustiva de las 6 causas de la anomalía PF 9.13 / Sharpe 14.76 (parada temprana en día 28 / barra 1,794, error $\sqrt{252}$ en Sharpe sobre trades, ejecución contemporánea en Close, desbalance 1:1 de $52.5k USD, brechas nocturnas, fricción multileg).
  * `test_f4_corrected_ornstein_uhlenbeck_diffusion`: Ecuaciones corregidas de Ornstein-Uhlenbeck ($\kappa, \theta, \sigma_{eq}, \tau_{1/2}$).
  * `test_f4_micro_dollar_beta_sub_baskets`: Sub-cestas micro Dollar-Beta con ratio continuo coprimo y riesgo acotado ($\le \$100$ USD).
  * `test_f4_realistic_performance_envelope`: Envolvente de rendimiento realista (Sharpe 1.5–2.3, PF 1.6–2.2, Win Rate 58%–66%).
  * `test_f1_to_f9_optimizations_coverage`: Soluciones específicas auditadas en F1, F2, F5, F6, F7, F8 y F9.

---

## 3. Comandos de Ejecución y Diagnóstico

Para ejecutar la suite de verificación automatizada:

```powershell
# Ejecución completa con reporte detallado
python -m pytest tests/test_optimization_reports_verification.py -v

# Ejecución concisa con resumen de fallos
python -m pytest tests/test_optimization_reports_verification.py -q

# Ejecución focalizada por Tier específico
python -m pytest tests/test_optimization_reports_verification.py -k "TestTier1" -v
python -m pytest tests/test_optimization_reports_verification.py -k "TestTier2" -v
python -m pytest tests/test_optimization_reports_verification.py -k "TestTier3" -v
python -m pytest tests/test_optimization_reports_verification.py -k "TestTier4" -v

# Ejecución focalizada sobre F3 o F4
python -m pytest tests/test_optimization_reports_verification.py -k "f3" -v
python -m pytest tests/test_optimization_reports_verification.py -k "f4" -v
```

---

## 4. Protocolo de Escalación de Defectos para Milestone 2

En estricto cumplimiento del rol de **Test Writer / QA Specialist**:
1. El Test Writer **únicamente modifica código de pruebas** (`tests/test_optimization_reports_verification.py`) y documentación de infraestructura de pruebas (`TEST_INFRA.md`, `TEST_READY.md`).
2. Los fallos detectados en Tier 4 constituyen la **lista oficial de defectos y requerimientos de enriquecimiento** que el Worker de Milestone 2 (`teamwork_worker_opt_m2`) debe consolidar en `reports/optimizaciones_estrategias/F3_optimizacion_anchored_vwap_value_area.md` y `F4_optimizacion_cointegracion_ou.md`.
3. Una vez que el Worker de M2 complete la consolidación e integración de los hallazgos cuantitativos de Milestone 0, la suite debe ejecutarse nuevamente hasta alcanzar **158 PASSED / 0 FAILED (100% de aprobación)** antes del paso a la compuerta de auditoría forense (Milestone 3).
