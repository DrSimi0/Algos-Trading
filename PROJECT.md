# Project: Especificación Matemática Formal de Estrategias Cuantitativas CME (ES & NQ) para Prop Trading

## Architecture
- **Domain**: Modelado matemático analítico, microestructura de mercado, procesos estocásticos y control de riesgo estricto para cuentas fondeadas (Prop Trading).
- **Target Instruments**: Futuros CME E-mini S&P 500 (`ES`), Micro E-mini S&P 500 (`MES`), CME E-mini Nasdaq 100 (`NQ`), Micro E-mini Nasdaq 100 (`MNQ`).
- **Target Deliverables**: 9 documentos modulares, individuales, autocontenidos e independientes en `reports/estrategias_matematicas/`.
- **Integrity & Constraints**:
  - Cero código de programación (Python, C++, etc.), scripts o pseudocódigo ejecutable.
  - Cero indicadores minoristas tradicionales descartados (sin SMA ni RSI estándar).
  - Pura formulación analítica, declarativa y formal en LaTeX con definiciones unívocas de variables, dominios y unidades.
  - Ecuaciones cerradas de generación de señales, reglas analíticas de salida (take profit, stop loss, time-decay) y control de riesgo prop trading (Daily Loss Limit, Max Trailing Drawdown, escalado por volatilidad).
  - Rúbrica de verificación analítica objetiva incorporada en cada archivo.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Market Intraday Momentum & MOC Seasonality | Gao et al. (2018); Bogousslavsky (2021). Retorno matutino prediciendo cierre, TWAP/VWAP institucional y balance MOC. | M2 | Survey Explorer 1 |
| F2 | Filtered Quantitative ORB | Crabel (1990); Zarattini et al. (2024); Fetna (2024). ORB 15-min con compresión previa NR4/ID, amortiguador Stretch y filtro RVOL >= 1.8. | M2 | Survey Explorer 1 |
| F3 | Anchored VWAP & Rotational Value Area Reversion | Madhavan (2002); Gatheral (2010); Steidlmayer (1986). Bandas $\pm 2\sigma$ ancladas, Auction Market Theory, filtro de régimen ADX < 25. | M2 | Survey Explorer 1 |
| F4 | Cointegración ES vs NQ / Ornstein-Uhlenbeck | Avellaneda & Lee (2010); Hasbrouck (2003). Arbitraje estadístico entre NQ y ES, difusión O-U, S-score, cobertura Dollar-Beta Neutral. | M3 | Survey Explorer 2 |
| F5 | Order Flow Imbalance Cont et al. | Cont, Kukanov & Stoikov (2014). Nivel 1 LOB imbalance, función de flujo discreto $e_n$, Z-score rodante, absorción del libro. | M3 | Survey Explorer 2 |
| F6 | Multi-Level OFI | Xu, Gould & Howison (2019). Imbalance multi-nivel, decaimiento espacial de potencias, descomposición en componentes principales (PCA), índice $\Phi_{coherence}$. | M3 | Survey Explorer 2 |
| F7 | Meta-Labeling & Triple Barrier | López de Prado (2018). Desacoplamiento de señal primaria y meta-modelo probabilístico, dimensionamiento continuo bet sizing, barreras dinámicas. | M4 | Survey Explorer 3 |
| F8 | Modelos Econométricos HAR-RV-J | Corsi (2009); Andersen et al. (2007); Moreira & Muir (2017). Descomposición continua y de saltos (Bipower Variation), Log-HAR-RV-J, dimensionamiento por volatilidad inversa. | M4 | Survey Explorer 3 |
| F9 | Anomalías Temporales & VRP | Bollerslev et al. (2009); Boyarchenko et al. (2023); Kurov et al. (2019). VRP Z-score, TSR, deriva nocturna condicional, deriva pre-macro con salida $T-3$ min. | M4 | Survey Explorer 3 |
| F10 | E2E Automated Verification Infrastructure | Test suite automatizado en `tests/test_math_strategies_verification.py` que audita los 9 documentos contra código, SMA/RSI, completitud de secciones y consistencia formal. | M1 | Test Track |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | E2E Test Suite Creation & Verification Infra | Diseñar suite de tests automatizada de 4 niveles en `tests/test_math_strategies_verification.py` y publicar `TEST_READY.md`. | none | DONE |
| M2 | Mathematical Documents Authoring: F1, F2, F3 | Generar `F1_market_intraday_momentum.md`, `F2_filtered_orb.md`, `F3_anchored_vwap_value_area.md` en `reports/estrategias_matematicas/`. | M1 | DONE |
| M3 | Mathematical Documents Authoring: F4, F5, F6 | Generar `F4_cointegracion_es_nq_ornstein_uhlenbeck.md`, `F5_order_flow_imbalance_cont.md`, `F6_multi_level_ofi.md` en `reports/estrategias_matematicas/`. | M1 | DONE |
| M4 | Mathematical Documents Authoring: F7, F8, F9 | Generar `F7_meta_labeling_triple_barrier.md`, `F8_modelos_econometricos_har_rv_j.md`, `F9_anomalias_temporales_vrp.md` en `reports/estrategias_matematicas/`. | M1 | DONE |
| M5 | Final Milestone: Review, Challenge & Forensic Audit Gate | Pasar 100% de la suite de tests E2E, revisión independiente por pares, stress test adversarial y veredicto CLEAN del Auditor Forense. | M2, M3, M4 | DONE |

## Code Layout
- `reports/estrategias_matematicas/F1_market_intraday_momentum.md`
- `reports/estrategias_matematicas/F2_filtered_orb.md`
- `reports/estrategias_matematicas/F3_anchored_vwap_value_area.md`
- `reports/estrategias_matematicas/F4_cointegracion_es_nq_ornstein_uhlenbeck.md`
- `reports/estrategias_matematicas/F5_order_flow_imbalance_cont.md`
- `reports/estrategias_matematicas/F6_multi_level_ofi.md`
- `reports/estrategias_matematicas/F7_meta_labeling_triple_barrier.md`
- `reports/estrategias_matematicas/F8_modelos_econometricos_har_rv_j.md`
- `reports/estrategias_matematicas/F9_anomalias_temporales_vrp.md`
- `tests/test_math_strategies_verification.py`: Suite de verificación automatizada.
- `TEST_READY.md`: Manifiesto de ejecución de pruebas.
- `TEST_INFRA.md`: Especificación metodológica de pruebas.
