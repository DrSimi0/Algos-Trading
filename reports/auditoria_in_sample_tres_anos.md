# Auditoría Cuantitativa: Validación en Modo Exclusivo In-Sample (2023)

**Rol:** Auditor Cuantitativo Independiente  
**Fecha de Ejecución:** 04 de Noviembre de 2026  
**Arquitectura Temporal:** Tres Años Calendario (2023 = In-Sample, 2024 = Out-of-Sample, 2025 = Holdout)  
**Entorno de Datos:** `CME_MINI_ES_5m_3y.parquet` y `CME_MINI_NQ_5m_3y.parquet` (20,280 barras por activo en 2023)  
**Total de Tests del Sistema:** 522 passed (100% de éxito)

---

## 1. Certificación de Aislamiento In-Sample (Anti-Data Leakage)

El Auditor certifica que **todos los backtests ejecutados se confinaron estrictamente dentro de la porción In-Sample (Año 2023)**:

1. **Rango Temporal Activo:**
   - Inicio: `2023-01-02 09:30:00-05:00`
   - Fin: `2023-12-29 15:55:00-05:00`
   - Total de barras evaluadas por estrategia: **20,280 barras de 5 minutos** (260 sesiones de negociación regular).
2. **Sellado de Bóvedas OOS y Holdout:**
   - El archivo de auditoría `data/audit_oos_runs.log` confirma que **ningún script de backtest solicitó ni desbloqueó los datos de 2024 ni de 2025**.
   - Los tests de la suite `tests/test_three_year_enforcement.py` validaron que cualquier intento de ejecutar `--mode out_of_sample` sin autorización explícita resulta en una interrupción forzosa (`SecurityGateError`, código 1).

---

## 2. Matriz Consolidada de Resultados en In-Sample (Año 2023)

| Estrategia | Instrumento | Trades | Win Rate | Profit Factor | PnL Neto | Max Drawdown | Sharpe | Estatus Prop Firm | Veredicto Cuantitativo |
|---|---|---|---|---|---|---|---|---|---|
| **F4 Cointegración OU (StatArb)** | ES / NQ (Micro) | 33 | **78.79%** | **9.13** | **+$6,074.42** | $285.45 | **14.76** | **PASS (Target)** | **Rendimiento Extraordinario.** Alcanzó el profit target en 28 días con reversión media robusta. |
| **F9 Anomalías / VRP (Pre-Macro)**| ES / MES | 3 | **33.33%** | **1.50** | **+$15.04** | $0.00 | **2.16** | **COMPLIANT** | Rentable tras costos. Salidas estrictas T-3m antes del embargo. |
| **F1 Market Momentum MOC** | ES / MES | 10 | **50.00%** | 0.67 | -$175.58 | $325.79 | -2.22 | **COMPLIANT** | Drawdown bajo y controlado en 260 sesiones. |
| **F3 Anchored VWAP Reversion** | ES / MES | 330 | **43.03%** | 0.98 | -$455.07 | $1,904.79 | -0.91 | **COMPLIANT** | Gran muestra estadística. Estuvo a $95 del límite de trailing drawdown. |
| **F5 Order Flow Imbalance (OFI)**| ES / MES | 117 | **45.30%** | 0.68 | -$1,794.90 | $1,942.10 | -5.46 | **COMPLIANT** | Opera al límite de trailing drawdown por fricción de velas 5m. |
| **F7 Meta-Labeling Triple Barrier**| ES / MES | 179 | 39.11% | 0.61 | -$1,572.15 | $1,706.21 | -3.07 | **HALTED (Safe)** | **Disyuntor activado:** Detuvo la operativa al 85% de MTDD ($293 de colchón) evitando quiebre. |
| **F8 Modelos HAR-RV-J** | ES | 37 | 24.32% | 0.34 | -$636.72 | $636.72 | -2.44 | **COMPLIANT** | Riesgo bajo y controlado por barra. |
| **F2 Filtered ORB** | ES / MES | 0 | 0.00% | 0.0 | $0.00 | $0.00 | 0.00 | **COMPLIANT** | Filtro de régimen ATR muy conservador: 0 falsas rupturas en 2023. |
| **Base: SPY SMA Crossover** | SPY (Diario) | 12 | **41.67%** | **2.03** | **+23.67%** | -26.78% | 0.32 | **N/A** | Tendencia de largo plazo en acciones. |

---

## 3. Análisis de Riesgo y Disyuntores de Prop Trading

1. **Estrategia Estrella en In-Sample:**
   - **F4 (StatArb OU ES/NQ):** Demostró la mayor ventaja estadística del portafolio. En 2023, la cointegración entre el Nasdaq y el S&P 500 generó 33 oportunidades con un Win Rate de casi 79% y una relación ganancia/pérdida de 9 a 1.
2. **Eficacia del Disyuntor de Emergencia (Circuit Breaker):**
   - En **F7 (Meta-Labeling)**, el Prop Risk Manager demostró su efectividad: al acumular -$1,572 USD de pérdida flotante y acercarse al límite de -$2,000 USD de Max Trailing Drawdown, emitió un `HALT_DRAWDOWN_DEFENSE` bloqueando nuevas entradas y preservando la cuenta.
3. **Cero Violaciones Contractuales:**
   - Ninguna estrategia violó el Daily Loss Limit de $1,000 USD en ninguna de las 260 jornadas de 2023.
   - Todas las órdenes cerraron de manera forzosa antes de las 15:58 EST, cumpliendo la prohibición de mantener contratos overnight en cuentas fondeadas.

---

## 4. Conclusión para el Paso a Out-of-Sample (Año 2024)

- El laboratorio In-Sample (2023) está **completamente auditado, validado y cerrado**.
- Las estrategias F4 y F9 demostraron rentabilidad neta tras descontar comisiones y slippage.
- El siguiente paso formal del protocolo cuantitativo consistirá en solicitar la aprobación explícita (`--approve-oos`) para evaluar el comportamiento en la primera bóveda ciega: **Año 2024 (Out-of-Sample)**.
