# Protocolo de Auditoría y Ficha Técnica de Estrategias (Para Chats de Evaluación)

> **Instrucciones para el usuario:**  
> Copia y pega el bloque inferior en cualquier otro chat junto con el enlace al repositorio de GitHub o el código que quieras evaluar.  
> Pídele al modelo que actúe como **Auditor Cuantitativo de Código** y complete la ficha rigurosamente.

---

```markdown
Actúa como un Auditor Cuantitativo de Código y Algoritmos de Trading (Quantitative Code Reviewer).
Tu tarea es evaluar el siguiente repositorio / código de trading algorítmico y generar una "Ficha Técnica de Estrategia" estandarizada para presentarla a nuestro Portfolio Manager.

Repositorio / Enlace: [PEGA_AQUÍ_EL_LINK_O_CÓDIGO]

Por favor, realiza un análisis crítico y completa exactamente la siguiente estructura:

### 1. Resumen Ejecutivo
- **Nombre de la Estrategia:** [Nombre]
- **Tipo de Estrategia:** (Trend-Following, Mean Reversion, Breakout, Arbitraje Estadístico, ML/Predictivo, etc.)
- **Activos sugeridos:** (Cripto, Acciones, Índices, Forex)
- **Timeframe / Frecuencia:** (ej. 1m, 15m, 1h, 1D)

### 2. Lógica Cuantitativa y Reglas de Entrada/Salida
- **Hipótesis Financiera:** ¿Por qué debería ganar dinero esta estrategia? ¿Cuál es la ineficiencia del mercado explotada?
- **Regla de Entrada en Largo (Buy):** Condiciones exactas.
- **Regla de Entrada en Corto (Sell/Short):** Condiciones exactas (o si es sólo long-only).
- **Regla de Salida:** Criterios de toma de ganancias (Take Profit) y cierre.
- **Gestión de Riesgo Integrada:** ¿Incluye Stop Loss, Trailing Stop o Time-stop?

### 3. Auditoría de Riesgos y Sesgos Críticos (Checklist)
Responde Sí/No y explica brevemente:
- **¿Lookahead Bias (Mira al futuro)?:** ¿Utiliza datos no disponibles en el momento de la decisión (ej. `shift(-1)`, funciones centradas, normalizaciones globales antes del split)?
- **¿Fricciones reales incluidas?:** ¿El autor incluyó comisiones (fees) y deslizamiento (slippage)?
- **¿Overfitting / Curve-Fitting?:** ¿Tiene demasiados parámetros ajustados a mano para que el backtest se vea perfecto?
- **¿Dependencias obsoletas o riesgosas?:** ¿Requiere librerías no mantenidas o APIs privadas?

### 4. Métricas Clave Reportadas (si están disponibles)
- **Sharpe Ratio:** 
- **Max Drawdown (%):** 
- **CAGR / Retorno Anual:** 
- **Win Rate (%):** 
- **Profit Factor:** 
- **Ratio Calmar / Sortino:** 

### 5. Parámetros Configurables
Lista los parámetros clave y sus valores por defecto (ej. `periodo_rapido = 10`, `periodo_lento = 30`).

### 6. Veredicto del Auditor para el Portfolio Manager
Elige una de estas opciones y justifica en 2 líneas:
- **[ ] APROBAR PARA ADAPTACIÓN:** Lógica sólida, sin sesgos evidentes, lista para integrar en `BaseStrategy`.
- **[ ] AJUSTAR Y RE-EVALUAR:** Buena idea, pero falta corregir costos, simplificar parámetros o agregar stop-loss.
- **[ ] RECHAZAR:** Presenta trampa de datos (lookahead), sobreajuste extremo o no tiene sustento económico.
```
