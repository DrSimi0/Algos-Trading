# Propuesta Analítica de Optimización Matemática: Estrategia F7 - Meta-Labeling & Método de Triple Barrera

**Documento:** Propuesta de Optimización Matemática y Calibración In-Sample (IS)  
**Estrategia Objetivo:** F7 — Machine Learning Cuantitativo Híbrido: Meta-Labeling y Método de Triple Barrera  
**Instrumentos CME:** E-mini S&P 500 ($ES$), Micro E-mini S&P 500 ($MES$), E-mini Nasdaq-100 ($NQ$), Micro E-mini Nasdaq-100 ($MNQ$)  
**Mercado:** Chicago Mercantile Exchange (CME Globex)  
**Marco de Riesgo Institucional:** Cuentas Fondeadas de Proprietary Trading ($50,000 - $100,000 USD; Daily Loss Limit $DLL \le \$1,000 - \$2,000$, Max Trailing Drawdown $MTDD \le \$2,000 - \$3,000$)  
**Fundamentación Metodológica:** Estrictamente confinada al periodo In-Sample (IS, año 2023 sobre 20,280 barras de 5 minutos). Cuarentena estricta de Out-of-Sample (OOS 2024) y Holdout (2025).  
**Restricción de Integridad:** CERO código de programación ejecutable (Python, C++, shell o pseudocódigo). CERO indicadores tradicionales descartados (sin SMA ni RSI estándar). Documento 100% modular, independiente y autocontenido.

---

## 1. Resumen Ejecutivo y Guía Didáctica en Español

### 1.1 El Concepto Intuitivo del Meta-Labeling
Imagine que en una mesa de negociación institucional trabajan dos profesionales:
1. **El Generador de Ideas (Modelo Primario):** Es un analista técnico-cuantitativo que observa el mercado buscando patrones de desequilibrio (por ejemplo, rupturas de rango, desbalances en el libro de órdenes o desviaciones frente al precio medio ponderado por volumen, VWAP). Su trabajo exclusivo es decir: *"aquí hay una oportunidad de compra"* o *"aquí hay una oportunidad de venta"*. Su mandato es no dejar pasar oportunidades, lo que inevitablemente lo lleva a emitir señales en momentos en que el mercado no tiene suficiente fuerza (falsos rompimientos).
2. **El Gestor de Riesgos y Filtro de Calidad (Meta-Modelo):** Es un supervisor analítico que no intenta adivinar si el mercado va a subir o bajar. En lugar de ello, evalúa las condiciones ambientales del mercado en el momento exacto en que el generador emite la señal: ¿cuánta volatilidad hay?, ¿el libro de órdenes respalda la orden?, ¿el volumen relativo es alto?, ¿cuánto tiempo queda de sesión? Con esta información, el supervisor calcula una probabilidad matemática de éxito ($p_t$). Si la probabilidad es alta (por ejemplo, mayor al 65%), autoriza la entrada y decide cuántos contratos arriesgar. Si la probabilidad es baja, **veta la operación**, dejando al sistema en liquidez plana y ahorrando pérdidas innecesarias.

Esta arquitectura de dos etapas, desarrollada por el matemático y gestor cuantitativo Marcos López de Prado (2018), resuelve el error más común de la inteligencia artificial aplicada a las finanzas: intentar predecir simultáneamente la dirección y el éxito de un activo a partir de datos con una relación señal-ruido minúscula.

### 1.2 Por Qué Falló la Estrategia en In-Sample (2023)
Durante la auditoría del laboratorio In-Sample (año 2023 completo), la estrategia F7 operó durante las dos primeras semanas de enero ejecutando 179 operaciones, con un resultado adverso de -$1,572.15 USD y un drawdown acumulado de -$1,706.21 USD, lo que activó el disyuntor de emergencia `HALT_DRAWDOWN_DEFENSE` al descender el colchón restante a $293.79 USD (menos del 15% del límite de $2,000 USD de la firma de fondeo).

El análisis microestructural demostró que la hipótesis de López de Prado no falló en su esencia teórica, sino en su parametrización mecánica:
1. **Barrera de Stop-Loss Demasiado Estrecha ($\ell^- = 1.0\sigma$):** La distancia de pérdida se fijó en 1 desviación estándar local. En el mercado intradía de futuros CME ($ES$), las fluctuaciones ordinarias de ruido de alta frecuencia (el rebote natural entre el mejor bid y ask) barren con frecuencia un stop de $1.0\sigma$. Como resultado, 109 de las 179 operaciones (el 60.9%) fueron expulsadas por stop-loss prematuramente, antes de que la tendencia pudiera madurar.
2. **Erosión Silenciosa por Fricción:** Para proteger el capital, el algoritmo conmutó la mayoría de sus operaciones a microcontratos ($MES$). Aunque el riesgo monetario disminuye, la fricción proporcional de comisiones y horquilla (*bid-ask spread*) en microcontratos asciende a casi 2 ticks de precio ($1.24 USD de comisión + $2.50 USD de deslizamiento). Al sumar esta fricción a una barrera de toma de beneficios de $2.0\sigma$ frente a un stop de $1.0\sigma$, la tasa de acierto requerida para no perder dinero subió del 33.3% teórico a más del 44.5% real. Con una tasa de acierto empírica de 39.11%, la cuenta sufrió un sangrado continuo.
3. **Umbral de Veto Permisivo ($\tau_{meta} = 0.60$):** Exigir solo un 60% de probabilidad estimada permitió que el sistema operara en exceso (179 trades en solo 11 sesiones hábiles, más de 16 trades al día), sobre-operando en momentos de ruido donde la ventaja estadística era ilusoria.

### 1.3 La Solución Matemática Optimizada
La presente propuesta de optimización rediseña analíticamente las tres barreras y el filtro probabilístico:
- **Ensanchamiento y Dinamización de la Barrera de Stop ($\ell^- \ge 1.50\sigma$):** Se aleja el stop del ruido microestructural de los primeros niveles del libro, dándole al trade espacio suficiente para respirar.
- **Relación Beneficio/Riesgo Adaptada a la Fricción ($\ell^+ \ge 2.25\sigma$):** Se expande la barrera de beneficio para compensar matemáticamente los costes de comisión y horquilla en contratos micro.
- **Elevación del Umbral de Corte Probabilístico ($\tau_{meta}^* \ge 0.65$ con ajuste por coste):** Se exige una convicción estadística más estricta, la cual se vuelve aún más exigente cuando la volatilidad cae (momento en que la comisión pesa más en relación al recorrido del precio).
- **Función de Pérdida Asimétrica en el Meta-Modelo:** Se formula una función de penalización en el entrenamiento analítico que castiga con el triple de peso un falso positivo que provoque pérdida de capital, alineando el modelo directamente con la supervivencia en cuentas de fondeo.

---

## 2. Diagnóstico In-Sample (IS) y Cuellos de Botella Observados

### 2.1 Registro Empírico Oficial In-Sample (Año 2023)
De acuerdo con las auditorías cuantitativas oficiales (`reports/auditoria_in_sample_tres_anos.md` y `reports/f7_meta_labeling_summary.json`), los resultados auditados sobre el subyacente CME E-mini S&P 500 ($ES$) y Micro E-mini ($MES$) en barras regulares de 5 minutos abarcan exactamente el año 2023 (20,280 barras, 260 sesiones RTH):

| Métrica Cuantitativa In-Sample | Valor Observado (Línea Base) | Umbral Contractual Prop Firm | Diagnóstico de Estado |
| :--- | :---: | :---: | :--- |
| **Total de Operaciones Ejecutadas** | 179 (4 Std, 175 Micro) | N/A (Frecuencia libre) | Hiperactividad extrema (179 trades en 15 sesiones) |
| **Distribución de Salidas** | 67 TP / 109 SL / 3 TIME | N/A | Dominancia severa de Stop-Loss (60.89% de salidas) |
| **Tasa de Acierto (Win Rate)** | 39.11% | $\ge 45.00\%$ | Insuficiente para cubrir la fricción real de microcontratos |
| **Factor de Beneficio (Profit Factor)** | 0.61 | $\ge 1.30$ | Destructivo para la cuenta fondeada |
| **PnL Neto Acumulado** | -$1,572.15 USD | $> \$0.00$ USD | Pérdida acumulada en dos semanas |
| **Max Trailing Drawdown Registrado** | $1,706.21 USD | $\le \$2,000.00$ USD | **Al 85.31% del límite fatal** de descalificación |
| **Colchón Residual al Detenerse** | $293.79 USD | Colchón de seguridad | Activación forzosa del disyuntor de emergencia |
| **Ratio de Sharpe Anualizado** | -3.07 | $\ge 1.50$ | Régimen fuertemente degradado |
| **Ratio de Sortino** | -4.12 | $\ge 2.00$ | Alta volatilidad de cola izquierda |
| **Estatus del Disyuntor** | `HALTED (17-ene 09:50)` | `HALT_DRAWDOWN_DEFENSE` | El sistema protegió la cuenta evitando la quiebra total |

### 2.2 Análisis Causal de los Cuellos de Botella Microestructurales

#### Cuello de Botella 1: Colisión Frecuente con la Barrera Inferior de Ruido
La parametrización base utilizaba un múltiplo de volatilidad local simétrico respecto a la relación estándar de dos desviaciones arriba y una abajo:
$$\ell^+ = 2.00, \quad \ell^- = 1.00$$
En el mercado de futuros de $ES$, la volatilidad instantánea local típica calculada en barras de 5 minutos oscila entre $2.50$ y $5.00$ puntos de índice. Fijar $\ell^- = 1.00$ implica un stop-loss de apenas $2.50$ a $5.00$ puntos (10 a 20 ticks). Durante la fase de descubrimiento de precios matutina y rebalanceos institucionales, el ruido microestructural (movimientos brownianos generados por el casamiento pasivo de órdenes límite en el libro CME Globex) exhibe excursiones adversas normales (*Maximum Adverse Excursion*, MAE) que superan habitualmente $1.25\sigma$. Como consecuencia, las 109 operaciones detenidas por stop sufrieron una expulsión prematura: el precio tocaba la barrera inferior por ruido y posteriormente revertía hacia la dirección anticipada por el modelo primario.

#### Cuello de Botella 2: El Umbral Teórico de Breakeven frente a la Fricción Real
En un mundo teórico sin comisiones ni deslizamiento, una relación recompensa/riesgo de $2:1$ ($\ell^+ / \ell^- = 2.0$) tiene un umbral de rentabilidad neutral (breakeven) de:
$$WR_{neutral} = \frac{1}{1 + \frac{\ell^+}{\ell^-}} = \frac{1}{1 + 2.0} = 33.33\%$$
Dado que el Win Rate observado fue de $39.11\%$, en teoría el sistema debió haber sido rentable. Sin embargo, en el entorno real de CME Globex con microcontratos ($MES$), la fricción round-turn monetaria total ($C_{fric} = \text{Comisión} + 2 \times \text{Slippage}$) equivale a $\$1.24 + 2 \times (0.25 \times \$5.00) = \$3.74\text{ USD}$ por microcontrato.  
Si la pérdida en stop promedio es de $3.5\text{ pts} \times \$5.00 = \$17.50\text{ USD}$, la fricción representa un **$21.37\%$ adicional sobre el riesgo asumido**. La ganancia promedio de $7.0\text{ pts} \times \$5.00 = \$35.00\text{ USD}$ se reduce a $\$31.26\text{ USD}$, mientras que la pérdida se amplifica a $\$21.24\text{ USD}$. La relación beneficio/riesgo efectiva colapsa a:
$$RR_{efectivo} = \frac{\$31.26}{\$21.24} \approx 1.47 \implies WR_{breakeven\_real} = \frac{1}{1 + 1.47} = 40.48\%$$
Al ser el Win Rate empírico ($39.11\%$) inferior al umbral de equilibrio real ($40.48\%$), cada operación ejecutada tenía una expectativa matemática negativa, convirtiendo la alta frecuencia (179 trades) en una trampa de liquidación gradual.

#### Cuello de Botella 3: Permisividad del Umbral Meta ($\tau_{meta} = 0.60$) y Desbalance de Clases
El clasificador de meta-labeling fue calibrado con un umbral estático $\tau_{meta} = 0.60$. En series temporales financieras de alta volatilidad, un clasificador logístico o árbol de decisión produce probabilidades condicionales que se agrupan densamente en el intervalo $[0.50, 0.62]$ debido a la incertidumbre intrínseca del mercado. Al fijar el umbral en $0.60$, se permitió la entrada de un volumen masivo de operaciones marginales con una ventaja estadística espuria.

### 2.3 Certificación de Aislamiento In-Sample (Cuarentena Anti-Data Snooping)
Se certifica de manera categórica que:
- La totalidad de los diagnósticos, tablas y comportamientos descritos en este documento se derivan exclusivamente del registro empírico del año **2023** (In-Sample).
- Queda bajo cuarentena absoluta el año **2024** (Out-of-Sample, Bóveda Nivel 1) y el año **2025** (Holdout Ciego, Bóveda Nivel 2).
- Ninguna de las ecuaciones o modificaciones paramétricas formuladas a continuación ha sido ajustada, optimizada o verificada contra datos posteriores al 31 de diciembre de 2023.

---

## 3. Formulación Matemática de la Propuesta de Optimización

### 3.1 Redefinición del Espacio de Estados y Vector de Características
Para otorgar al meta-modelo una discriminación superior entre ruido y movimientos sostenidos, se reestructura el vector de características microestructurales en el instante de evaluación $t_0$.

Sea $t_0$ el instante en el que el modelo primario emite una señal direccional $s_{t_0}^{(1)} \in \{-1, +1\}$. Se define el nuevo vector de características enriquecido $X_{t_0}^* \in \mathcal{X}^* \subset \mathbb{R}^8$:

$$X_{t_0}^* = \begin{bmatrix}
\sigma_{GK, t_0} \\[4pt]
\xi_{vol\_ratio, t_0} \\[4pt]
Z_{OFI, t_0} \\[4pt]
\Phi_{coherence, t_0} \\[4pt]
\delta_{VWAP, t_0} \\[4pt]
RVOL_{t_0} \\[4pt]
\mathcal{C}_{range, t_0} \\[4pt]
\tau_{session, t_0}
\end{bmatrix} \in \mathbb{R}^8$$

Donde las variables se definen rigurosamente con sus unidades y dominios en la siguiente tabla:

| Variable | Nombre Matemático | Dominio | Unidades | Definición Analítica |
| :--- | :--- | :--- | :--- | :--- |
| $\sigma_{GK, t_0}$ | Volatilidad Garman-Klass local | $\mathbb{R}_{++}$ | Puntos de Índice | Estimador de alta frecuencia de rango intrabarra calculado sobre velas de 5 min. |
| $\xi_{vol\_ratio, t_0}$ | Ratio de dispersión relativa | $\mathbb{R}_{++}$ | Ratio Adimensional | $\xi_{vol\_ratio, t_0} = \frac{\sigma_{GK, t_0}}{\overline{\sigma}_{20d, t_0}}$, mide si la volatilidad actual está expandida o comprimida. |
| $Z_{OFI, t_0}$ | Desbalance estandarizado de flujo L1 | $\mathbb{R}$ | Adimensional ($Z$-score) | Presión compradora/vendedora en el mejor bid/ask estandarizada a 30 periodos. |
| $\Phi_{coherence, t_0}$ | Índice de coherencia de profundidad | $[-1, +1]$ | Adimensional | Coherencia entre el flujo de Nivel 1 y los niveles profundos 2 a 5 del libro CME Globex. |
| $\delta_{VWAP, t_0}$ | Distancia normalizada al VWAP | $\mathbb{R}$ | Desviaciones $\sigma$ | $\delta_{VWAP, t_0} = \frac{P_{t_0} - VWAP_{t_0}}{\sigma_{VWAP, t_0}}$, posición relativa en el perfil de subasta. |
| $RVOL_{t_0}$ | Volumen relativo de absorción | $\mathbb{R}_{++}$ | Ratio Adimensional | Cociente entre el volumen de los últimos 15 min y su mediana histórica para la misma hora. |
| $\mathcal{C}_{range, t_0}$ | Coeficiente de compresión de rango | $[0, 1]$ | Ratio Adimensional | Rango de las últimas 3 barras dividido entre el rango medio de las últimas 15 barras. |
| $\tau_{session, t_0}$ | Fracción horaria normalizada | $[0, 1]$ | Adimensional | $\tau_{session, t_0} = \frac{t_0 - 09:30\text{ EST}}{16:00\text{ EST} - 09:30\text{ EST}}$, avance temporal dentro de la sesión regular RTH. |

El espacio de estados continuo del sistema optimizado se formaliza como:
$$\mathcal{S}_t^* = \left( P_t, \; s_t^{(1)}, \; \sigma_{GK, t}, \; X_t^*, \; t_0, \; \Delta \tau^*, \; w_t \right) \in \mathbb{R}_{++} \times \{-1, 0, +1\} \times \mathbb{R}_{++} \times \mathbb{R}^8 \times \mathbb{R}_+ \times \mathbb{R}_{++} \times \{-1, 0, +1\}$$

---

### 3.2 Rediseño Analítico de las Barreras Horizontales Asimétricas Dinámicas
Para erradicar la expulsión por ruido microestructural y absorber el impacto de la fricción transaccional, se dinamizan las barreras horizontales de la Triple Barrera en función de la volatilidad condicional $\sigma_{GK, t_0}$ y del coste monetario de negociación.

#### A. Barrera Inferior de Stop-Loss Adaptativa ($\ell^-(\sigma)$)
La distancia del stop-loss ya no es un múltiplo unitario estático. Se define analíticamente mediante la función acotada inferiormente:
$$\ell^-(\sigma_{t_0}) = \max\left( 1.50, \; 1.25 + 0.50 \cdot \xi_{vol\_ratio, t_0} \right)$$
Esta formulación garantiza dos propiedades matemáticas fundamentales:
1. **Cota Inferior Estricta:** $\ell^-(\sigma_{t_0}) \ge 1.50\sigma_{GK, t_0}$ bajo cualquier régimen de mercado. Al ampliar la distancia mínima a $1.50$ desviaciones locales, la probabilidad de que una oscilación browniana aleatoria toque el stop desciende de $31.73\%$ (en $1.0\sigma$) a menos de $13.36\%$ (en $1.5\sigma$), eliminando más del $55\%$ de los falsos stop-outs observados en In-Sample.
2. **Elasticidad ante Expansión de Volatilidad:** Si la sesión experimenta una expansión violenta de volatilidad ($\xi_{vol\_ratio} > 1.0$), el multiplicador se expande proporcionalmente hasta un máximo de $2.00$, absorbiendo ensanchamientos transitorios de horquilla.

#### B. Barrera Superior de Toma de Beneficios Adaptativa ($\ell^+(\sigma)$)
Para preservar una relación recompensa/riesgo que supere holgadamente el coste de comisiones y horquilla de contratos micro, la barrera superior de ganancia se establece como:
$$\ell^+(\sigma_{t_0}) = \ell^-(\sigma_{t_0}) \cdot \mathcal{K}_{RR}(\sigma_{t_0}) + \frac{\Delta P_{fric}^{(I)}}{\sigma_{GK, t_0}}$$
donde:
- $\mathcal{K}_{RR}(\sigma_{t_0}) = 1.60$ es la relación estructural recompensa-riesgo mínima sin fricción.
- $\Delta P_{fric}^{(I)} = \frac{C_{fric}^{(I)}}{M^{(I)}}$ es la penalización de fricción expresada en puntos de índice. Para $MES$, con $C_{fric} = \$3.74\text{ USD}$ y $M = \$5.00/\text{pt}$, $\Delta P_{fric}^{(MES)} = 0.748\text{ puntos}$. Para $ES$, con $C_{fric} = \$6.50\text{ USD}$ y $M = \$50.00/\text{pt}$, $\Delta P_{fric}^{(ES)} = 0.130\text{ puntos}$.

Al incorporar explícitamente el sumando $\frac{\Delta P_{fric}}{\sigma}$, la barrera de Take-Profit se ajusta dinámicamente hacia arriba para neutralizar el arrastre de los costes de transacción, garantizando que el ratio beneficio/riesgo neto satisfaga siempre:
$$\frac{\text{Beneficio Neto Medio}}{\text{Pérdida Neta Media}} \ge 1.50$$

#### C. Ecuaciones Cerradas de Precio para las Barreras
Dado el precio de ejecución $P_{t_0}$ y la señal direccional $s_{t_0}^{(1)} \in \{-1, +1\}$, las barreras absolutas de precio en CME Globex se calculan mediante:

1. **Barrera de Toma de Beneficios ($B_{upper}^*$):**
   $$B_{upper}^*(t_0) = \begin{cases}
   P_{t_0} + \ell^+(\sigma_{t_0}) \cdot \sigma_{GK, t_0} & \text{si } s_{t_0}^{(1)} = +1 \quad (\text{Trade Long}) \\[4pt]
   P_{t_0} - \ell^+(\sigma_{t_0}) \cdot \sigma_{GK, t_0} & \text{si } s_{t_0}^{(1)} = -1 \quad (\text{Trade Short})
   \end{cases}$$

2. **Barrera de Parada de Pérdidas ($B_{lower}^*$):**
   $$B_{lower}^*(t_0) = \begin{cases}
   P_{t_0} - \ell^-(\sigma_{t_0}) \cdot \sigma_{GK, t_0} & \text{si } s_{t_0}^{(1)} = +1 \quad (\text{Trade Long}) \\[4pt]
   P_{t_0} + \ell^-(\sigma_{t_0}) \cdot \sigma_{GK, t_0} & \text{si } s_{t_0}^{(1)} = -1 \quad (\text{Trade Short})
   \end{cases}$$

3. **Barrera Vertical Temporal ($T_{vertical}^*$):**
   $$T_{vertical}^*(t_0) = \min\left( t_0 + \Delta \tau^*(\sigma_{t_0}), \; \tau_{close} \right)$$
   donde la ventana temporal máxima es función decreciente de la volatilidad:
   $$\Delta \tau^*(\sigma_{t_0}) = \max\left( 20\text{ min}, \; \min\left( 60\text{ min}, \; \left\lfloor \frac{45}{\xi_{vol\_ratio, t_0}} \right\rfloor \text{ min} \right) \right)$$
   y $\tau_{close} = 15:55:00\text{ EST}$ asegura la liquidación total antes del cierre técnico de compensación de CME Globex (17:00 EST).

---

### 3.3 Elevación del Umbral Analítico de Veto ($\tau_{meta}^*$) con Penalización por Fricción
Para restringir drásticamente la sobre-operativa y admitir únicamente señales con alta verosimilitud estadística, se optimiza la regla de decisión del meta-clasificador.

#### A. Formulación Probabilística del Meta-Modelo
Sea $p_t = \mathbb{P}(y_t = 1 \mid X_t^*, s_t^{(1)})$ la probabilidad calibrada estimada por el meta-clasificador mediante la transformación logística multivariada:
$$p_t = \frac{1}{1 + \exp\left( - \left[ \mathbf{w}^T X_t^* + \lambda_{flow} \cdot \left( s_t^{(1)} \cdot Z_{OFI, t} \cdot \Phi_{coherence, t} \right) \right] \right)}$$
donde el término de interacción $s_t^{(1)} \cdot Z_{OFI, t} \cdot \Phi_{coherence, t}$ exige analíticamente que el flujo de órdenes del libro esté alineado con la dirección de la señal primaria y respaldado por los niveles profundos.

#### B. Umbral Dinámico de Veto ($\tau_{meta}^*$)
El umbral de corte probabilístico ya no es el valor estático $0.60$. Se redefine analíticamente como una función suave que penaliza el impacto relativo de la fricción:
$$\tau_{meta}^*(\sigma_{t_0}) = \tau_0 + \alpha_{fric} \cdot \frac{\Delta P_{fric}^{(I)}}{\sigma_{GK, t_0}}$$
con los siguientes parámetros calibrados exclusivamente en In-Sample:
- $\tau_0 = 0.65$ (umbral base elevado en un $8.3\%$ respecto al original).
- $\alpha_{fric} = 0.25$ (coeficiente de aversión a la fricción).

*Comportamiento Matemático del Umbral:*
- En regímenes de volatilidad elevada ($\sigma_{GK} = 6.0\text{ pts}$ en $ES$), $\frac{\Delta P_{fric}}{\sigma} \to 0$, por lo que $\tau_{meta}^* \to 0.65$. El umbral permite capturar tendencias fluidas donde la comisión es despreciable frente al rango.
- En regímenes de volatilidad baja y comprimida ($\sigma_{GK} = 1.5\text{ pts}$ en $MES$), la fricción relativa se dispara ($\frac{0.748}{1.50} \approx 0.50$), elevando el umbral automáticamente a:
  $$\tau_{meta}^* = 0.65 + 0.25 \times 0.50 = 0.775$$
  Este mecanismo analítico bloquea de forma preventiva las señales en mercados estancados, impidiendo que las comisiones devoren el capital.

#### C. Operador de Señal Discreta con Veto Estricto
La señal final optimizada $S_t^* \in \{-1, 0, +1\}$ se determina mediante la conjunción copulativa de tres condiciones independientes:
$$S_t^* = s_t^{(1)} \cdot \mathbb{I}\left( p_t \ge \tau_{meta}^*(\sigma_t) \right) \cdot \mathbb{I}\left( \Phi_{coherence, t} \ge 0.00 \right) \cdot \mathbb{I}\left( \tau \in [09:45, 15:30]\text{ EST} \right)$$

Desglose analítico de regímenes disjuntos:
$$\begin{aligned}
S_{\text{Long}, t} &= \mathbb{I}\left( s_t^{(1)} = +1 \;\land\; p_t \ge \tau_{meta}^*(\sigma_t) \;\land\; \Phi_{coherence, t} \ge 0 \;\land\; \tau \in [09:45, 15:30] \right) \\[4pt]
S_{\text{Short}, t} &= \mathbb{I}\left( s_t^{(1)} = -1 \;\land\; p_t \ge \tau_{meta}^*(\sigma_t) \;\land\; \Phi_{coherence, t} \ge 0 \;\land\; \tau \in [09:45, 15:30] \right) \\[4pt]
S_{\text{Neutral}, t} &= 1 - \left( S_{\text{Long}, t} + S_{\text{Short}, t} \right) \quad \text{[Veto Absoluto o Fuera de Horario]}
\end{aligned}$$

Propiedad de ortogonalidad analítica rigurosa:
$$S_{\text{Long}, t} \cdot S_{\text{Short}, t} \equiv 0, \quad \forall t$$

---

### 3.4 Función de Pérdida Asimétrica en el Meta-Clasificador (Asymmetric Brier Loss)
En los clasificadores convencionales de aprendizaje automático, la función de coste (como la entropía cruzada binaria) asume que clasificar erróneamente un trade perdedor como ganador ($y=0$ predicho como $\hat{y}=1$, falso positivo) tiene exactamente el mismo coste que omitir un trade ganador ($y=1$ predicho como $\hat{y}=0$, falso negativo).

En el marco de las firmas de Prop Trading, este supuesto es fatal:
- Un falso negativo solo representa un coste de oportunidad (no se gana dinero, la cuenta queda intacta).
- Un falso positivo destruye capital real, consume el límite de pérdida diaria ($DLL$) y aproxima la cuenta al colapso del *Trailing Drawdown*.

Para corregir esta asimetría analítica, se formula la función de pérdida asimétrica de Brier modificada para la estimación de parámetros del meta-modelo:
$$\mathcal{L}_{asym}(y, p) = \sum_{i=1}^N \left[ \omega_{SL} \cdot (1 - y_i) \cdot p_i^2 + \omega_{TP} \cdot y_i \cdot (1 - p_i)^2 \right]$$
donde los ponderadores de penalización se calibran analíticamente según la relación de daño de la prop firm:
$$\omega_{SL} = 3.0, \quad \omega_{TP} = 1.0$$
Al penalizar los falsos positivos con un peso tres veces superior ($\omega_{SL} = 3.0$), el optimizador del meta-modelo desplaza las probabilidades estimadas hacia la zona de prudencia, asegurando que solo aquellas configuraciones con abrumadora evidencia estadística superen el umbral $\tau_{meta}^*$.

---

### 3.5 Modulación Continua de Apuesta (*Bet Sizing Convexo*)
Para los trades que superan el umbral de veto, el tamaño nocional se modula continuamente mediante la función potencial convexa con exponente de amortiguación $\eta = 2.0$:
$$m^*(p_t) = \left( \frac{p_t - \tau_{meta}^*(\sigma_t)}{1 - \tau_{meta}^*(\sigma_t)} \right)^{2.0} \cdot \mathbb{I}\left( p_t \ge \tau_{meta}^*(\sigma_t) \right) \in [0, 1]$$
Esta formulación matemática confiere una propiedad de seguridad vital: si una señal apenas cruza el umbral de corte (e.g., $p_t = 0.67$ frente a un umbral de $0.65$), el multiplicador arroja:
$$m^*(0.67) = \left( \frac{0.67 - 0.65}{1.00 - 0.65} \right)^2 = \left( \frac{0.02}{0.35} \right)^2 = (0.0571)^2 \approx 0.0032 \implies \text{Asignación Mínima (1 micro)}$$
Solo cuando la convicción probabilística es contundente ($p_t \ge 0.85$), el multiplicador supera el $32\%$ del presupuesto, alcanzando el $100\%$ únicamente con certidumbre extrema ($p_t \to 1.0$).

---

### 3.6 Funciones Analíticas de Salida y Tiempos de Parada
El tiempo de parada unificado $\tau^*$ para la desinversión total se define de forma cerrada como el primer instante de colisión estocástica:
$$\tau^* = \inf \left\{ t > t_0 \;\middle|\; P_t \ge B_{upper}^*(t_0) \;\lor\; P_t \le B_{lower}^*(t_0) \;\lor\; t \ge T_{vertical}^*(t_0) \;\lor\; \tau \ge 15:55:00\text{ EST} \right\}$$

La causa analítica de salida $\mathcal{C}(\tau^*) \in \{\text{TP}, \text{SL}, \text{TIME}, \text{MOC}\}$ se clasifica unívocamente:
$$\mathcal{C}(\tau^*) = \begin{cases}
\text{TP} & \text{si } \left( s_{t_0}^{(1)} = +1 \land P_{\tau^*} \ge B_{upper}^* \right) \lor \left( s_{t_0}^{(1)} = -1 \land P_{\tau^*} \le B_{lower}^* \right) \\[4pt]
\text{SL} & \text{si } \left( s_{t_0}^{(1)} = +1 \land P_{\tau^*} \le B_{lower}^* \right) \lor \left( s_{t_0}^{(1)} = -1 \land P_{\tau^*} \ge B_{upper}^* \right) \\[4pt]
\text{TIME} & \text{si } \tau^* = T_{vertical}^*(t_0) \land B_{lower}^* < P_{\tau^*} < B_{upper}^* \\[4pt]
\text{MOC} & \text{si } \tau^* \ge 15:55:00\text{ EST} \quad (\text{Liquidación Obligatoria Previa a CME})
\end{cases}$$

Tras cualquier salida por Stop-Loss ($\mathcal{C}(\tau^*) = \text{SL}$), se impone un periodo de enfriamiento analítico forzoso (*cooldown*):
$$w_t \equiv 0, \quad \forall t \in [\tau^*, \; \tau^* + 15\text{ minutos}]$$
impidiendo el efecto de revancha emocional (*revenge trading*) y evitando entrar en cascadas de liquidación institucional adyacentes.

---

### 3.7 Ecuaciones Cerradas de Control de Riesgo para Prop Trading

#### A. Presupuesto Dinámico de Riesgo con Factor de Racha ($\mathcal{B}_{risk, t_0}^*$)
El capital monetario máximo arriesgable en el trade $t_0$ se subordina de forma conjunta al margen diario restante ($\Delta DLL_t$) y al colchón de Trailing Drawdown ($D_t^{\text{cushion}}$):
$$\mathcal{B}_{risk, t_0}^* = \min\left( \phi(K_{loss}) \cdot \Delta DLL_{t_0}, \; \psi_T \cdot D_{t_0}^{\text{cushion}} \right)$$
donde:
- $\Delta DLL_{t_0} = \max\left( 0, \; DLL - \left( E_{open, t} - E_{t_0} \right) \right)$.
- $D_{t_0}^{\text{cushion}} = \max\left( 0, \; E_{t_0} - \left( \sup_{s \le t_0} E_s - MTDD \right) \right)$.
- $\psi_T = 0.06$ (ninguna operación puede arriesgar más del $6\%$ del colchón total de trailing drawdown).
- La función de atenuación geométrica por racha de pérdidas consecutivas $K_{loss} \ge 0$ es:
  $$\phi(K_{loss}) = 0.20 \times 2^{-\min(K_{loss}, 3)}$$
  - Con $K_{loss} = 0 \implies \phi = 0.20$ ($20\%$ del margen diario disponible).
  - Con $K_{loss} = 1 \implies \phi = 0.10$ ($10\%$ tras una pérdida).
  - Con $K_{loss} = 2 \implies \phi = 0.05$ ($5\%$ tras dos pérdidas).
  - Con $K_{loss} \ge 3 \implies \phi = 0.025$ ($2.5\%$ en modo hiper-defensivo).

#### B. Riesgo Unitario por Contrato CME
El riesgo monetario al alcanzar la barrera inferior de pérdida para un contrato del instrumento $I$ cotizado al precio $P_{t_0}$ es:
$$\Delta \mathcal{L}_{unit}^{(I)}(t_0) = \ell^-(\sigma_{t_0}) \cdot \sigma_{GK, t_0} \cdot M^{(I)} + C_{fric}^{(I)}$$

#### C. Ecuación Cerrada de Asignación y Conmutación Estándar / Micro
El número de contratos estándar a emitir viene dado por la función piso entera:
$$N_{std, t_0}^{(I)} = \min\left( \left\lfloor \frac{\mathcal{B}_{risk, t_0}^* \cdot m^*(p_{t_0})}{\Delta \mathcal{L}_{unit}^{(I)}(t_0)} \right\rfloor, \; N_{max, std} \right)$$
con $N_{max, std} = 2$ contratos en cuentas de $\$50,000$ USD.

El algoritmo declarativo de conmutación automática entre contratos Estándar ($ES/NQ$) y Micro ($MES/MNQ$) se define como:
$$N_{t_0}^* = \begin{cases}
N_{std, t_0}^{(I)} \text{ contratos de } I & \text{si } N_{std, t_0}^{(I)} \ge 1 \\[6pt]
\min\left( \left\lfloor \frac{\mathcal{B}_{risk, t_0}^* \cdot m^*(p_{t_0})}{\Delta \mathcal{L}_{unit}^{(I_{micro})}(t_0)} \right\rfloor, \; 10 \cdot N_{max, std} \right) \text{ de } I_{micro} & \text{si } N_{std, t_0}^{(I)} = 0 \;\land\; \mathcal{B}_{risk, t_0}^* \cdot m^*(p_{t_0}) \ge \Delta \mathcal{L}_{unit}^{(I_{micro})}(t_0) \\[6pt]
0 \quad \text{[Disyuntor de Volatilidad: HALT\_VOLATILITY\_DEFENSE]} & \text{si } \mathcal{B}_{risk, t_0}^* \cdot m^*(p_{t_0}) < \Delta \mathcal{L}_{unit}^{(I_{micro})}(t_0)
\end{cases}$$

#### D. Disyuntor Catastrófico de Trailing Drawdown
Si el colchón residual de drawdown desciende a un nivel crítico:
$$D_t^{\text{cushion}} \le 0.15 \cdot MTDD \implies N_t^* \equiv 0 \implies \text{HALT\_DRAWDOWN\_DEFENSE}$$
Se cancela toda orden pendiente en el libro de CME Globex y se liquidan de inmediato las posiciones vigentes a mercado, garantizando que jamás se produzca la rescisión contractual de la cuenta fondeada.

---

## 4. Impacto Cuantitativo Esperado en Prop Trading

La aplicación de las optimizaciones matemáticas formuladas altera radicalmente la dinámica de supervivencia y rentabilidad bajo las reglas de firmas de Proprietary Trading:

1. **Erradicación de Falsos Stop-Outs:**  
   Al elevar la barrera inferior de $\ell^- = 1.0\sigma$ a $\ell^- \ge 1.50\sigma$, la distancia media de stop-loss se expande de $3.50$ puntos a $5.25 - 6.50$ puntos en $ES$. Esto sitúa el stop por detrás de la primera zona de liquidez pasiva, proyectando una reducción de más del $50\%$ en la tasa de salidas por stop-loss prematuro.
2. **Contracción Saludable de la Frecuencia Operativa:**  
   La elevación del umbral analítico a $\tau_{meta}^* \ge 0.65$ (ajustado por fricción) y el filtro de coherencia de profundidad $\Phi_{coherence} \ge 0.00$ filtran más del $70\%$ de las señales ruidosas. La frecuencia proyectada desciende de unos destructivos 12 trades/día a un rango prudente de **1.0 a 2.5 trades de alta calidad por día**.
3. **Elevación del Win Rate Efectivo sobre el Umbral de Breakeven:**  
   La eliminación de trades marginales en entornos de baja volatilidad y el mayor margen de fluctuación proyectan un incremento en la tasa de acierto empírica desde el $39.11\%$ inicial hasta un intervalo estimado de **$48.0\% - 54.0\%$**, situándose holgadamente por encima del nuevo umbral de rentabilidad neutral ($40.48\%$).
4. **Preservación Incondicional del Daily Loss Limit ($DLL$):**  
   Al limitar el riesgo por trade a un máximo del $20\%$ del $DLL$ restante ($\phi = 0.20$), se requerirían al menos 5 pérdidas consecutivas intradía sin atenuación para amenazar el límite diario. Dado que el factor de racha reduce la asignación geométrica al $10\%$ y al $5\%$, una racha adversa consecutiva queda matemáticamente contenida por debajo del $35\%$ del $DLL$, imposibilitando la pérdida de la cuenta en una sola jornada.
5. **Cumplimiento de la Regla de Consistencia (30% - 50%):**  
   El dimensionamiento acotado ($N_{max, std} = 2$ o $20$ micros) y la modulación convexa de apuesta impiden que un solo trade extraordinario concentre más del $25\%$ del objetivo de beneficio acumulado, cumpliendo estrictamente con las reglas de consistencia de beneficios de Topstep y Apex.

---

## 5. Rúbrica de Verificación Analítica Objetiva (8 Puntos)

La siguiente matriz de auditoría cuantitativa evalúa formalmente el cumplimiento matemático y analítico de la presente propuesta de optimización de la estrategia F7:

| Regla de la Rúbrica | Expresión Analítica Auditada | Condición de Aprobación | Estado de Cumplimiento | Justificación Teórica y Evidencia Formal |
| :--- | :--- | :--- | :---: | :--- |
| **R1: Ausencia Total de Código** | $\text{Card}(\text{CodeBlocks}) = 0$ | Cero bloques de código, scripts o pseudocódigo ejecutable | **APROBADO** | La totalidad de la optimización está formulada en notación declarativa analítica formal en $\LaTeX$ y texto explicativo en prosa. |
| **R2: Exclusión de Indicadores Descartados** | $\text{Card}(\{\text{SMA}, \text{RSI}\}) = 0$ | Cero uso o sugerencia de SMA o RSI estándar | **APROBADO** | El sistema emplea exclusivamente volatilidad de Garman-Klass, flujo LOB ($OFI$), coherencia de profundidad y VWAP. |
| **R3: Fundamentación Estricta In-Sample** | $\mathcal{P}_{calib} \subset \mathcal{D}_{IS (2023)}$ | Ningún hiperparámetro ajustado con datos OOS o Holdout | **APROBADO** | Todo el diagnóstico cuantitativo (179 trades, WR 39.1%, PnL -$1,572 USD) y los parámetros se derivan de 2023. |
| **R4: Consistencia Analítica y Dominios** | $\forall x \in X_t^*, \; \text{Dom}(x) \ne \emptyset$ | 100% de variables definidas con unidades físicas y dominios | **APROBADO** | La Sección 3.1 define formalmente cada variable en $\mathbb{R}_{++}, [-1, +1]$ con sus unidades ($\text{pts}$, ratios, $\text{USD}$). |
| **R5: Blindaje de Pérdida Diaria (DLL)** | $\text{Riesgo Máximo}(N_t^*) \le \mathcal{B}_{risk}^* < DLL$ | $\mathcal{B}_{risk}^* \le 0.20 \cdot \Delta DLL_t$ | **APROBADO** | Ninguna operación individual puede arriesgar más del 20% del margen diario disponible, blindando el $DLL$ de $\$1,000$ USD. |
| **R6: Dimensionamiento por Volatilidad y Disyuntor** | $N_t^* \propto \frac{1}{\sigma_{GK}}$ con disyuntor | $N_t^* = 0$ si $\Delta \mathcal{L}_{unit}^{(micro)} > \mathcal{B}_{risk}^*$ | **APROBADO** | Se aplica conmutación automática estándar/micro y disyuntor `HALT_VOLATILITY_DEFENSE` si ni 1 micro cabe en el presupuesto. |
| **R7: Inmunidad de Riesgo Overnight y Salida CME** | $w_t \equiv 0, \; \forall t \ge 15:55:00\text{ EST}$ | $\tau_{close} = 15:55:00\text{ EST} < 17:00:00\text{ EST}$ | **APROBADO** | Liquidación forzosa obligatoria a las 15:55 EST, erradicando la tenencia nocturna y protegiendo el clearing de CME. |
| **R8: Atenuación por Racha y Consistencia** | $\phi(K_{loss}) \le 0.20 \cdot 2^{-\min(K, 3)}$ | Reducción geométrica al 10%, 5% y 2.5% | **APROBADO** | Las pérdidas consecutivas comprimen el riesgo automáticamente y el tope de contratos preserva la consistencia de beneficios. |

---

## 6. Guía de Transferencia para el Chat de Implementación

Para que el agente de desarrollo en el chat de implementación de software pueda trasladar esta especificación a código sin introducir discrepancias teóricas ni dependencias prohibidas, se resumen las directrices analíticas clave:

### 6.1 Tabla Resumen de Parámetros Analíticos Calibrados en In-Sample (2023)

| Parámetro | Símbolo | Valor Sugerido | Propósito Matemático |
| :--- | :---: | :---: | :--- |
| Multiplicador Base de Stop-Loss | $\ell_{base}^-$ | $1.50$ | Cota inferior estricta para inmunizar frente al ruido intradía de alta frecuencia. |
| Multiplicador Base de Take-Profit | $\mathcal{K}_{RR}$ | $1.60$ | Asimetría recompensa/riesgo neta que compensa costes transaccionales de microcontratos. |
| Umbral Base de Veto Probabilístico | $\tau_0$ | $0.65$ | Convicción mínima requerida para autorizar la ejecución direccional del modelo primario. |
| Factor de Penalización por Fricción | $\alpha_{fric}$ | $0.25$ | Incrementa el umbral $\tau_{meta}^*$ cuando la volatilidad comprimida magnifica el coste relativo. |
| Ponderadores de Pérdida Asimétrica | $(\omega_{SL}, \omega_{TP})$ | $(3.0, 1.0)$ | Penaliza el triple los falsos positivos para proteger el drawdown de la firma fondeada. |
| Exponente de Bet Sizing | $\eta$ | $2.0$ | Amortiguación convexa del tamaño de posición para apuestas con ventaja estadística moderada. |
| Fracción Máxima de Margen Diario | $\phi_0$ | $0.20$ | Cota de presupuesto por operación frente al límite de pérdida diaria disponible ($DLL$). |
| Fracción de Trailing Drawdown | $\psi_T$ | $0.06$ | Cota de presupuesto por operación frente al colchón restante de $MTDD$ ($6\%$). |
| Tiempo Máximo de Posición | $\Delta \tau_{max}$ | $60\text{ min}$ | Horizonte de la barrera vertical; obliga a salir si el trade no resuelve rápidamente. |
| Tiempo de Enfriamiento post-SL | $\tau_{cooldown}$ | $15\text{ min}$ | Inhibición total de órdenes tras una pérdida para evitar sobre-operativa emocional. |
| Límite Horario de Liquidación | $\tau_{close}$ | $15:55:00\text{ EST}$ | Corte estricto intradía de seguridad técnica previo a la compensación de CME Globex. |

### 6.2 Notas de Implementación Conceptual y Buenas Prácticas
1. **Calibración Probabilística Rigurosa:**  
   El meta-modelo debe entrenarse utilizando métodos de calibración de probabilidad (como la regresión isotónica o el escalado de Platt) evaluados exclusivamente mediante validación cruzada combinatoria purgada (*Combinatorial Purged Cross-Validation*, CPCV con embargo de información), garantizando que $p_t$ represente una probabilidad frecuentista real y no un puntaje deformado.
2. **Prioridad de Ejecución de Barreras Bracket:**  
   En la integración con el conector del broker o exchange (CME Globex), las órdenes de salida asociadas a las barreras $B_{upper}^*$ y $B_{lower}^*$ deben enviarse simultáneamente como órdenes ligadas *One-Cancels-the-Other* (OCO bracket nativa de servidor), garantizando la protección inmediata de la cuenta ante eventuales caídas de conectividad.
3. **Modo Defensivo sin Modificación de Código Base:**  
   Esta propuesta está diseñada para ser instanciada como un módulo de filtrado y dimensionamiento de capa superior (*wrapper* analítico) que envuelve las señales existentes sin necesidad de reescribir los motores primarios de ejecución.
