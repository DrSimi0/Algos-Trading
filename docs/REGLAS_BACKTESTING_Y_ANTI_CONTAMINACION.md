# Manual de Reglas de Oro del Backtesting Cuantitativo y Prevención de Contaminación de Datos

Este documento establece las normas cuantitativas obligatorias para el diseño, evaluación y validación de estrategias de trading en este repositorio.

---

## 1. La Filosofía Cuantitativa: ¿Por qué fracasan las estrategias en vivo?

El 90% de las estrategias algorítmicas que muestran resultados espectaculares en un backtest fracasan en el mercado real por dos motivos:
1. **Contaminación de Datos (*Data Leakage* y *Lookahead Bias*)**: El algoritmo utilizó información del futuro sin que el desarrollador se diera cuenta.
2. **Sobreajuste (*Overfitting / Curve-Fitting*)**: Los parámetros fueron calibrados para memorizar el pasado en lugar de capturar una ineficiencia estructural de mercado.

---

## 2. Las 5 Reglas de Oro No Negociables

### Regla 1: Cuarentena Estricta del Periodo Out-of-Sample (OOS)
- **Definición**: Los datos históricos se dividen en dos universos temporales independientes:
  - **In-Sample (IS)**: El laboratorio de pruebas. Aquí se investiga la hipótesis, se calculan indicadores y se calibran parámetros.
  - **Out-of-Sample (OOS)**: La bóveda sellada. Es una simulación ciega que emula el futuro desconocido.
- **La Bóveda no se puede abrir dos veces**: Cada vez que ejecutas una prueba en el OOS y luego regresas al código a cambiar un parámetro porque el resultado no te gustó, **has destruido la pureza del OOS**, convirtiéndolo efectivamente en datos In-Sample (*Data Snooping Bias* o *p-hacking*).
- **El Visto Bueno Previo**: Nunca se ejecuta el OOS hasta que la estrategia pase los criterios de viabilidad en In-Sample:
  - Sharpe Ratio $\ge 1.0$ (o Calmar $\ge 1.0$)
  - Número de operaciones representativo ($N \ge 30$)
  - Hipótesis económica clara (¿por qué debería ganar dinero?)
  - Drawdown tolerable.

### Regla 2: Eliminación Total del Lookahead Bias (Sesgo de Anticipación)
- **Regla del Desfase de Ejecución**: Si una señal se decide con el precio de cierre de la barra en el tiempo $t$ (`Close[t]`), la orden solo se puede ejecutar en el tiempo $t+1$ (en el `Open[t+1]` o al siguiente tick disponible).
- **Anti-patrón Prohibido**:
  ```python
  # ❌ ERROR GRAVE: Mira al futuro usando el retorno del día siguiente en la decisión de hoy
  signals = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
  ```
- **Forma Correcta**:
  ```python
  # ✅ CORRECTO: La señal de hoy se ejecuta mañana (shift=1 en posición)
  df["position"] = signals.shift(1).fillna(0.0)
  ```
- **Cálculo de Indicadores**: Todos los indicadores (medias, RSI, volatilidad, percentiles) deben ser causales, usando ventanas rodantes (`rolling()`) y jamás funciones centradas (`center=True`).

### Regla 3: Aislamiento de Transformaciones Estadísticas y Machine Learning
- Si utilizas estandarización (Z-Score), escaladores (MinMaxScaler) o modelos predictivos:
  - El ajuste (`fit`) debe realizarse **únicamente** con datos In-Sample.
  - La transformación (`transform`) se aplica a los datos Out-of-Sample utilizando los parámetros calculados en IS (media y desviación estándar históricas).
- **Anti-patrón Prohibido**:
  ```python
  # ❌ ERROR GRAVE: Normalizar todo el dataset antes de dividirlo en train y test
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(full_dataset) # ¡Fuga de varianza y media futura!
  train = scaled_data[:split]
  test = scaled_data[split:]
  ```

### Regla 4: Inclusión Obligatoria de Fricciones Realistas (Costos y Deslizamiento)
- Una estrategia sin costos no es más que una ilusión matemática.
- Toda simulación debe incluir:
  - **Comisiones del Broker y Exchange**: Por ejemplo, 5 a 10 puntos básicos (0.05% - 0.10%) en acciones, o el costo *round-turn* por contrato en futuros CME (`commission_rt` definido en `src/core/futures_specs.py`).
  - **Deslizamiento (*Slippage*)**: Fricción de entrada y salida debida al spread bid-ask y velocidad de llenado (mínimo 1 tick o 0.05%).
  - **Precios Realistas de Ejecución**: Nunca asumir que las órdenes de mercado se llenan en el precio mínimo (*Low*) de una compra o en el máximo (*High*) de una venta.

### Regla 5: Representatividad de Regímenes de Mercado en In-Sample
- Un periodo In-Sample de 6 meses en medio de un mercado alcista parabólico dará resultados engañosos.
- El In-Sample debe abarcar al menos un ciclo económico completo con:
  1. Tendencias alcistas fuertes (*Bull Market*).
  2. Mercados bajistas pronunciados (*Bear Market* / pánicos).
  3. Periodos laterales prolongados sin tendencia (*Chop / Range*).
  4. Entornos de alta volatilidad (VIX elevado) y baja volatilidad.

---

## 3. Calendario Recomendado de Periodos (IS vs OOS)

| Activo | Timeframe | Periodo In-Sample (IS) | Periodo Out-of-Sample (OOS) | Justificación |
| :--- | :--- | :--- | :--- | :--- |
| **SPY / QQQ / Acciones** | 1D (Diario) | `2018-01-01` a `2023-12-31` | `2024-01-01` a Presente | IS incluye Crash 2020, Rally 2021 y Caída 2022. OOS evalúa el mercado reciente. |
| **Futuros CME (MES, MNQ)** | 1h / 15m / 5m | Primeras 75% de las velas | Últimas 25% de las velas | Muestra estadística suficiente ($>100$ trades) y reserva reciente intacta. |
| **Criptomonedas (BTC, ETH)** | 1D / 4h | `2019-01-01` a `2023-12-31` | `2024-01-01` a Presente | Abarca un ciclo completo de Halving (fase bajista y alcista previa). |

---

## 4. Checklist para Aprobar el Paso de In-Sample a Out-of-Sample

Antes de ejecutar `--approve-oos`, el operador cuantitativo debe responder afirmativamente a las siguientes preguntas:

- [ ] ¿La lógica de la estrategia tiene sentido económico o es pura coincidencia estadística?
- [ ] ¿El Sharpe Ratio en In-Sample es $\ge 1.0$ tras deducir comisiones y slippage?
- [ ] ¿La estrategia generó al menos 30 operaciones independientes en In-Sample?
- [ ] ¿El número de parámetros optimizados es pequeño ($\le 3$ variables clave) para evitar el sobreajuste?
- [ ] ¿Estás listo para congelar la estrategia tal como está sin volver a ajustarla si el OOS sale negativo?
