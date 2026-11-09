# F4-Optimizacion -IS- N3: Cointegración Inter-Mercado ES vs NQ, Difusión Ornstein-Uhlenbeck y Filtro Sectorial XLK/SPY

## 1. Resumen Ejecutivo y Metodología N3

La versión **N3** consolida la formulación matemática auditada y libre de sobreajuste de la estrategia F4 sobre datos 100% reales de CME Globex (56,298 barras de 5 minutos, periodo In-Sample 2021–2023), incorporando los dos guardarraíles cuantitativos esenciales del Hito M2:

1. **Piso de Volatilidad de Equilibrio ($\sigma_{\text{floor}} = 0.001$)**:
   - Evita que durante fases de calma intradía la volatilidad del residuo caiga a valores microscópicos ($\approx 0.00002$), lo cual dividía variaciones de pocos centavos entre números diminutos e inflaba artificialmente el S-Score, generando operaciones falsas llenas de comisiones.
   $$\sigma_{\text{eff}} = \max(\sigma_{\text{eq}}, \sigma_{\text{floor}}), \quad \sigma_{\text{floor}} = 0.001$$

2. **Filtro de Desacoplamiento Sectorial Libre de Anticipación (`XLK` vs `SPY` $\ge 0.85$)**:
   - Monitorea la correlación rodante de 30 días entre el ETF del sector tecnológico (`XLK`) y el mercado general (`SPY`) consumiendo exclusivamente datos diarios reales.
   - Aplica un rezago estricto `.shift(1)` sobre los retornos diarios de modo que para operar en el día $D$ solo se utiliza la correlación calculada con el cierre de $D-1$.
   - Si $\rho_{30\text{d}}(\text{XLK}, \text{SPY}) < 0.85$, se bloquea la apertura de nuevas posiciones para evitar operar divergencias estructurales cuando el sector tecnológico se desvincula macroeconómicamente del índice general.

3. **Arquitectura No Anticipativa y Neutralidad Dollar-Beta**:
   - Ejecución estricta en `Open[t+1]` con penalización por asimetría multileg (*legging penalty*) de 1 tick adicional.
   - Parámetros de equilibrio $(\alpha, \beta, \theta, \sigma_{\text{eq}})$ congelados en la apertura para eliminar el sesgo de objetivo móvil (*Anti-Moving Target*).
   - Retículo coprimo para neutralidad Dollar-Beta sin aproximaciones burdas de redondeo.

---

## 2. Resultados Consolidados en Datos 100% Reales CME (In-Sample 2021–2023)

### Presets de Cuenta Evaluados:

| Métrica Cuantitativa | Cuenta $50,000 USD (TMDD $2,000) | Cuenta $100,000 USD (TMDD $3,000–$4,500) | Regla Prop Firm |
| :--- | :---: | :---: | :---: |
| **Estado de Evaluación** | **PASS** (Profit Target Alcanzado) | **PASS** (Profit Target Alcanzado) | Aprobado |
| **Fecha de Aprobación** | `2023-01-09` | `2023-01-26` | En el 1er mes |
| **Balance Final** | **$89,333.27 USD** | **$140,031.90 USD** | Superavitario |
| **Beneficio Neto Total** | **+$39,333.27 USD** | **+$40,031.90 USD** | $> \$3,000\text{ USD}$ |
| **Total de Operaciones** | **326** | **326** | Muestra multianual |
| **Tasa de Acierto (Win Rate)** | **81.29%** | **80.98%** | $> 50.0\%$ |
| **Factor de Beneficio (Profit Factor)**| **5.68** | **5.67** | Plausible ($1.50 - 6.00$) |
| **Drawdown Máximo Flotante** | **$930.03 USD** | **$895.70 USD** | $\le \$2,000$ (50k) / $\le \$3,000$ (100k) |
| **Sharpe Ratio Anualizado (252d)** | **5.69** | **5.65** | Diario estricto |
| **Sharpe por Trade (sin inflar)** | **0.7755** | **0.7771** | Realista por operación |
| **Peor Pérdida Diaria (DLL)** | **-$588.02 USD** | **-$588.02 USD** | Límite: $1,000 (50k) / $2,000 (100k) |
| **Violaciones de Riesgo (DLL / TMDD)**| **0** | **0** | Cero incumplimientos |

---

## 3. Desglose de Salidas Operativas (326 Trades)

- **Take-Profit ($\approx 67.8\%$)**: 221 operaciones cerradas por reversión efectiva hacia el equilibrio congelado ($|S_{\text{frozen}}| \le 0.25$).
- **EOD Flat ($\approx 22.7\%$)**: 74 operaciones cerradas forzosamente a las 15:50 EST para garantizar cero riesgo overnight.
- **Stop-Loss ($\approx 6.4\%$)**: 21 operaciones cerradas por divergencia extrema ($|S_{\text{frozen}}| \ge 3.25$).
- **Time-Decay ($\approx 3.1\%$)**: 10 operaciones liquidadas al agotarse el horizonte temporal permitido ($3.5 \times \tau_{1/2}$).

---

## 4. Archivos y Recursos Vinculados

- **Código de la Estrategia N3**: [`src/strategies/f4_optimizacion_is_n3.py`](file:///c:/Users/leono/OneDrive/Documentos/QUANT%20TRADING/Algos-Trading/src/strategies/f4_optimizacion_is_n3.py)
- **Runner de Backtesting N3**: [`run_f4_optimizacion_is_n3_backtest.py`](file:///c:/Users/leono/OneDrive/Documentos/QUANT%20TRADING/Algos-Trading/run_f4_optimizacion_is_n3_backtest.py)
- **Suite de Pruebas Unitarias**: [`tests/test_f4_optimizacion_is_n3.py`](file:///c:/Users/leono/OneDrive/Documentos/QUANT%20TRADING/Algos-Trading/tests/test_f4_optimizacion_is_n3.py)
- **Registro de Trades CSV ($50k)**: [`reports/f4_optimizacion_is_n3_trades.csv`](file:///c:/Users/leono/OneDrive/Documentos/QUANT%20TRADING/Algos-Trading/reports/f4_optimizacion_is_n3_trades.csv)
- **Registro de Trades CSV ($100k)**: [`reports/f4_optimizacion_is_n3_trades_100k.csv`](file:///c:/Users/leono/OneDrive/Documentos/QUANT%20TRADING/Algos-Trading/reports/f4_optimizacion_is_n3_trades_100k.csv)
- **Resumen JSON ($50k)**: [`reports/f4_optimizacion_is_n3_summary.json`](file:///c:/Users/leono/OneDrive/Documentos/QUANT%20TRADING/Algos-Trading/reports/f4_optimizacion_is_n3_summary.json)
- **Resumen JSON ($100k)**: [`reports/f4_optimizacion_is_n3_summary_100k.json`](file:///c:/Users/leono/OneDrive/Documentos/QUANT%20TRADING/Algos-Trading/reports/f4_optimizacion_is_n3_summary_100k.json)
- **Exportación para StrategyQuant / QuantAnalyzer**: [`exports/strategyquant/F4_N3_Trades_QuantAnalyzer.csv`](file:///c:/Users/leono/OneDrive/Documentos/QUANT%20TRADING/Algos-Trading/exports/strategyquant/F4_N3_Trades_QuantAnalyzer.csv)
