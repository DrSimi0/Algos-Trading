# Project: Archivo de Datos Sintéticos y Re-Calibración de F4 sobre Datos Reales CME Globex

## Architecture
- **Dominio**: Archivo definitivo del generador sintético ad-hoc, persistencia de reglas institucionales anti-contaminación, implementación en F4 de piso de volatilidad ($\sigma_{floor} \ge 0.001$) y filtro de desacoplamiento sectorial (correlación rodante 30 días XLK/SPY $\ge 0.85$), priorización de estrategias reales como F3-N3 (reversión intradía a VWAP), y unificación de pipelines de calibración, backtesting y Monte Carlo exclusivamente sobre datos históricos reales de CME Globex (`data/raw/CME_MINI_ES_2021_2023_5m.parquet` y `CME_MINI_NQ_2021_2023_5m.parquet`, 56,298 barras limpias).
- **Instrumentos CME**: Futuros E-mini y Micro E-mini (ES, MES, NQ, MNQ) y ETFs sectoriales (XLK, SPY).
- **Entregables Principales**:
  1. `scripts/archive/generate_synthetic_futures_3y.py`: Archivo histórico del generador sintético fuera de la ruta activa.
  2. `docs/REGLAS_BACKTESTING_Y_ANTI_CONTAMINACION.md`: Regla 6 de prohibición estricta de calibración sobre datos sintéticos/simulados y uso exclusivo de CME real In-Sample.
  3. `src/strategies/f4_optimizacion_is_n1.py`, `src/strategies/f4_optimizacion_is_n2.py`, `src/strategies/cointegration_ou_base.py`: piso de volatilidad $\sigma_{floor} \ge 0.001$ en $\sigma_{eq}$ y oscilador S-score.
  4. Módulo de filtro sectorial XLK/SPY 30 días desfasado ($shift(1)$ anti-lookahead) con bloqueo de órdenes cuando $\rho < 0.85$.
  5. `run_f4_optimizacion_is_n1_backtest.py`, `run_f4_optimizacion_is_n2_backtest.py`, `run_f3_optimization_n3_backtest.py`: rutas por defecto a series reales `_2021_2023_5m.parquet`.
  6. `src/core/data_split.py`: soporte para universo In-Sample completo 2021-2023.
  7. `run_monte_carlo.py` y `run_portfolio_backtest.py`: integración y priorización de F3-N3 como estrategia base (~65% riesgo) y F4 como overlay condicional (~35% riesgo).
  8. Suites de pruebas unitarias y de integración actualizadas en `tests/`.
- **Restricciones Vinculantes**:
  - Prohibición terminante de calibrar sobre series sintéticas o simuladas ad-hoc.
  - Confinamiento estricto a In-Sample (2021-2023). Queda terminantemente PROHIBIDO consultar, leer, ejecutar o evaluar datos OOS (2024) o Holdout ciego (2025).
  - Veto binario del Forensic Auditor: veredicto obligatorio CLEAN.
  - Plausibilidad estadística estricta: Sharpe anualizado en rango real (1.20 a 3.50), Profit Factor (1.30 a 2.20), rechazando Sharpe > 8.0 o PF > 4.0 como anomalía/sobreajuste.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Synthetic Script Archiving | Traslado de `scripts/generate_synthetic_futures_3y.py` a `scripts/archive/` y limpieza de referencias activas en el repositorio. | M1 | Survey Explorer 1 |
| F2 | Institutional Rules Persistence | Registro formal de la Regla 6 (prohibición de calibrar con datos sintéticos y exclusividad de CME real) en documentación y reglas del proyecto. | M1 | Survey Explorer 1 |
| F3 | Z-Score Volatility Floor | Implementación de $\sigma_{floor} \ge 0.001$ sobre $\sigma_{eq}$ en el estimador OU, cálculo contemporáneo y congelado del S-score y stop loss de F4. | M2 | Survey Explorer 2 |
| F4 | Sector Decoupling Filter | Filtro de correlación rodante de 30 días XLK/SPY con desfase no anticipativo de 1 día ($shift(1)$) bloqueando nuevas entradas si $\rho < 0.85$. | M2 | Survey Explorer 2 |
| F5 | CME Globex Real Data Unification | Configuración de runners y pipelines de F4 y F3 para consumir exclusivamente `data/raw/CME_MINI_ES_2021_2023_5m.parquet` y contraparte NQ. | M3 | Survey Explorer 1 |
| F6 | Data Split IS 2021-2023 Support | Ajuste en `src/core/data_split.py` para reconocer el trienio 2021-2023 como periodo In-Sample en datasets históricos consolidados. | M3 | Survey Explorer 3 |
| F7 | Real Strategy Prioritization (F3-N3) | Priorización de F3-N3 como estrategia base (~65% margen) y F4 como overlay condicional (~35% margen) en runners de portafolio y Monte Carlo. | M3 | Survey Explorer 3 |
| F8 | Confinement & Plausibility Audit | Auditoría de air-gap criptográfico y verificación de métricas estadísticamente verosímiles en Monte Carlo (TMDD $\le \$3,000$, DLL $\le \$1,500$, Ruina $< 10\%$). | M4 | Survey Explorer 3 |
| F9 | E2E Verification Gate | Ejecución y certificación integral por Reviewers (APPROVE), Challengers (APPROVE) y Forensic Auditor (CLEAN). | M4 | All Explorers |
| F10 | Synthesis & Human Reporting | Elaboración de reporte de cierre estructurado en español para el usuario y Sentinel. | M5 | Orchestrator |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Synthetic Archiving & Policy Persistence | Archivar generador sintético a `scripts/archive/`, actualizar referencias en código y documentar Regla 6. | none | **DONE** |
| M2 | F4 Volatility Floor & Sector Filter | Implementar $\sigma_{floor} \ge 0.001$ y filtro rodante XLK/SPY $\ge 0.85$ en F4 con tests unitarios. | none | **DONE** |
| M3 | CME Data Unification & Strategy Prioritization | Unificar runners en CME real 2021-2023, ajustar `data_split.py`, y conectar F3-N3 en portafolio y Monte Carlo. | M1, M2 | **DONE** |
| M4 | End-to-End Verification Gate | Ejecutar backtests IS, Monte Carlo, suites de tests, revisión de código, pruebas de estrés y auditoría forense CLEAN. | M3 | **DONE** |
| M5 | Synthesis & Human Reporting | Generar reporte ejecutivo consolidado y comunicar resultados al usuario y Sentinel. | M4 | **DONE** |

## Interface Contracts
### F4 Strategy ↔ Sector Filter
- Función `compute_sector_correlation_series(df_xlk, df_spy, window=30)`:
  * Entradas: `df_xlk` (DataFrame diario OHLCV), `df_spy` (DataFrame diario OHLCV), `window: int = 30`.
  * Retorno: `pd.Series` con índice `datetime.date` y valores float de correlación de retornos logarítmicos desfasados con `shift(1)`.
- En `F4OptimizationISN1Strategy.generate_signals` y `F4OptimizationISN2Strategy.generate_signals`:
  * Parámetros: `"min_sector_corr": 0.85`, `"sector_corr_window": 30`, `"sigma_floor": 0.001`.
  * Condición de veto: `sector_corr_val < min_sector_corr` impide nuevas órdenes pendientes pero permite cerrar posiciones activas.
- En `fit_ou_process_intraday`, `calculate_s_score`, `calculate_frozen_s_score`:
  * `sigma_eq_eff = max(sigma_eq, sigma_floor)`.

### Runners ↔ Datasets
- Entradas estándar por defecto:
  * ES: `data/raw/CME_MINI_ES_2021_2023_5m.parquet`
  * NQ: `data/raw/CME_MINI_NQ_2021_2023_5m.parquet`
  * XLK: `data/raw/XLK_1d_2021-01-01_2023-12-31.parquet`
  * SPY: `data/raw/SPY_1d_2021-01-01_2023-12-31.parquet`

## Code Layout
- `scripts/archive/generate_synthetic_futures_3y.py`: Generador archivado.
- `docs/REGLAS_BACKTESTING_Y_ANTI_CONTAMINACION.md`: Reglas del sistema y anti-contaminación.
- `src/strategies/f4_optimizacion_is_n1.py` / `src/strategies/f4_optimizacion_is_n2.py` / `src/strategies/f4_optimizacion_is_n3.py`: Estrategias F4 con sigma floor, filtro sectorial y cestas coprimas.
- `src/strategies/cointegration_ou_base.py`: Modelo base OU con sigma floor.
- `src/strategies/mean_reversion/f3_optimization_is_n3.py`: Estrategia prioritaria F3-N3.
- `src/core/data_split.py`: Gestor de partición In-Sample / Vault.
- `run_f4_optimizacion_is_n1_backtest.py` / `run_f4_optimizacion_is_n2_backtest.py` / `run_f4_optimizacion_is_n3_backtest.py`: Runners F4 en CME real.
- `run_f3_optimization_n3_backtest.py`: Runner F3-N3 en CME real.
- `run_monte_carlo.py` / `run_portfolio_backtest.py`: Simuladores estocásticos y de portafolio.
- `tests/test_f4_optimizacion_is_n1.py` / `tests/test_f4_optimizacion_is_n2.py` / `tests/test_f4_optimizacion_is_n3.py`: Suites de pruebas de F4.
- `tests/test_f3_optimization_is_n2.py` / `tests/test_f3_optimization_is_n3.py` / `tests/test_monte_carlo_and_portfolio.py`: Suites de pruebas de F3 y portafolio.
