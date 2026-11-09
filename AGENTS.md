# Quantitative Trading Workspace Rules (Algos-Trading)

Estas reglas son obligatorias e innegociables para cualquier agente que opere en este repositorio:

## 1. Prohibición Estricta de Datos Sintéticos
- **Cero Calibración Sintética**: Queda terminantemente prohibido entrenar, optimizar, calibrar o validar estrategias sobre series generadas sintéticamente o mediante fórmulas ad-hoc (el script `scripts/archive/generate_synthetic_futures_3y.py` está archivado y obsoleto).
- **Datos 100% Reales CME Globex**: Todo backtest y calibración debe consumir exclusivamente series históricas reales de CME Globex (`data/raw/CME_MINI_ES_2021_2023_5m.parquet`, `data/raw/CME_MINI_NQ_2021_2023_5m.parquet`) y series diarias reales (`data/raw/XLK_1d_2021-01-01_2023-12-31.parquet`, `data/raw/SPY_1d_2021-01-01_2023-12-31.parquet`).

## 2. Confinamiento In-Sample (IS) y Cuarentena Criptográfica
- **Aislamiento Estricto In-Sample (2021–2023)**: Toda exploración, diseño, ajuste paramétrico y simulación de Monte Carlo debe realizarse exclusivamente dentro de la partición In-Sample (`years <= 2023`).
- **Prohibido Tocar OOS (2024) y Holdout Ciego (2025)**: Nunca desbloquear, desencriptar ni inspeccionar `data/vault/` (`oos_2024_*.enc`, `holdout_2025_*.enc`) salvo instrucción explícita y autorizada del usuario para una corrida de validación final.

## 3. Guardarraíles Matemáticos y Estadísticos
- **Piso de Volatilidad ($\sigma_{\text{floor}} \ge 0.001$)**: En todo modelo de reversión a la media o arbitraje estadístico (como F4 Ornstein-Uhlenbeck), imponer siempre `sigma_eq = max(sigma_eq_raw, sigma_floor)` con $\sigma_{\text{floor}} \ge 0.001$ para evitar que fluctuaciones de pocos centavos disparen el Z-Score/S-Score.
- **Filtro de Desacoplamiento Sectorial Libre de Lookahead**: En estrategias inter-mercado ES vs NQ (F4), exigir correlación rodante de 30 días $\rho(\text{XLK}, \text{SPY}) \ge 0.85$ aplicando estrictamente `.shift(1)` sobre retornos diarios para garantizar cero sesgo de anticipación intradía.
- **Sin Indicadores Retail Descartados**: No utilizar ni reintroducir `SMA` ni `RSI` estándar en ninguna estrategia.
- **Plausibilidad y Prop Trading**: Desconfiar y auditar de inmediato cualquier resultado con `Sharpe > 10` o `Profit Factor > 8` como alerta de sobreajuste o fuga de datos. Priorizar estrategias reales con ventaja microestructural comprobada (como `F3-N3` y `F4-N2` combinadas en portafolio bajo límites de Trailing Max Drawdown y Daily Loss Limit).
