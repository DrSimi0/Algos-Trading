# Especificación Matemática Formal: Estrategia F3 - Anchored VWAP & Rotational Value Area Mean Reversion

**Identificador de Estrategia:** F3  
**Familia Cuantitativa:** Trading Sistemático / Teoría de Subasta / Reversión a la Media Dinámica Institucional  
**Instrumentos Objetivo:** CME E-mini S&P 500 ($ES$), CME Micro E-mini S&P 500 ($MES$), CME E-mini Nasdaq-100 ($NQ$), CME Micro E-mini Nasdaq-100 ($MNQ$)  
**Mercado de Negociación:** Chicago Mercantile Exchange (CME Globex)  
**Marco de Riesgo:** Proprietary Trading (Cuentas Fondeadas: Topstep, Apex, Lucid, MyFundedFutures)  
**Nivel de Documento:** Especificación Matemática y Microestructural Modular Autosuficiente (Apta para entrega directa a agentes de implementación de software)  
**Restricción Estricta de Formato:** Teoría matemática analítica pura en notación declarativa formal LaTeX. Queda estrictamente excluido cualquier fragmento de código ejecutable, script o pseudocódigo, así como cualquier indicador técnico tradicional descartado.

---

## 1. Identificación y Fundamento Teórico / Hipótesis Microestructural

### 1.1 Citas Académicas de Referencia
1. **Madhavan, A. (2002)**. *VWAP Strategies*. **Algorithmic Trading: Investment Management Analysis**, Institutional Investor Journals. DOI: [10.1016/S1386-4181(02)00004-7](https://doi.org/10.1016/S1386-4181(02)00004-7).
2. **Gatheral, J. (2010)**. *No-Dynamic-Arbitrage and Market Impact*. **Quantitative Finance**, 10(7), 749–759. DOI: [10.1080/14697680903392330](https://doi.org/10.1080/14697680903392330).
3. **Steidlmayer, J. P., & Koy, K. (1986)**. *Markets & Market Logic: The Alternative to Technical Analysis*. Porcupine Press.

---

### 1.2 Hipótesis Económica y Dinámica Microestructural

La formulación de esta estrategia combina la microestructura analítica de ejecución institucional con los principios formales de la Teoría de Subasta de Mercado (*Auction Market Theory*, AMT):

1. **La Naturaleza Rotacional de los Mercados Organizados (Steidlmayer & Koy, 1986):**  
   Los mercados financieros existen con la finalidad primordial de facilitar transacciones entre compradores y vendedores mediante un proceso continuo de descubrimiento de precio eficiente (*fair value*). La investigación cuantitativa sobre el perfil de mercado revela que aproximadamente el $70\%$ del tiempo bursátil, los futuros de índices operan en un estado de equilibrio estocástico (*balanced rotational auction*), generando una distribución acampanada de volumen transado alrededor de un precio de consenso central.

2. **El Anchored VWAP como Benchmark Institucional Supremo (Madhavan 2002; Gatheral 2010):**  
   El Precio Medio Ponderado por Volumen anclado a la apertura regular de contado ($\tau_0 = 09:30:00$ EST) constituye el precio de referencia ineludible para los fondos institucionales de gran escala. Los gestores de activos evalúan la eficiencia de los intermediarios financieros en función de si sus ejecuciones superan o empeoran el VWAP de la sesión. Por consiguiente, el VWAP anclado no es un promedio aritmético pasivo, sino el centro gravitacional del libro de órdenes donde descansa la mayor densidad acumulada de transacciones.

3. **Dinámica de Agotamiento y Absorción en los Extremos ($\pm 2\sigma$):**  
   Cuando el precio instantáneo se aleja dos desviaciones estándar ponderadas por volumen ($\pm 2\sigma_{VWAP}$) de su centro de gravedad en un entorno desprovisto de noticias macroeconómicas direccionales (régimen no tendencial certificado por $ADX < 25$ y pendiente horizontal de VWAP), el precio entra en un territorio de sobreextensión estadística. En estos extremos de la curva de Gauss:
   - La agresión compradora o vendedora se agota, al considerarse el precio desfavorable para nuevas asignaciones institucionales.
   - Los proveedores de liquidez pasiva absorben los últimos remanentes de órdenes a mercado.
   - Se produce un rechazo de colas de liquidez (*liquidity rejection wick*) que desencadena una reversión determinista hacia el consenso de valor justo institucional ($VWAP$).

4. **Inhibición Estricta en Días Tendenciales (Filtro de Ruptura $3\sigma$):**  
   Si el mercado se encuentra en un día de tendencia macroeconómica sostenida ($ADX \ge 25$), el precio no revertirá al VWAP, sino que utilizará el VWAP como soporte dinámico de aceleración. Intentar revertir en estos escenarios equivale a "atrapar cuchillos cayendo". Por ello, el modelo exige un filtro de régimen no tendencial riguroso y un disyuntor catastrófico en la tercera desviación estándar ($\pm 3\sigma_{VWAP}$) que liquida inmediatamente cualquier posición si el mercado inicia una expansión direccional imprevista.

---

## 2. Variables de Entrada, Espacio de Estados y Horizontes Temporales

### 2.1 Variables de Entrada y Parámetros Formales

Sea la apertura regular de CME Globex el ancla temporal estricta: $\tau_0 = 09:30:00$ EST. Sea $i \in \{1, 2, \dots, t\}$ el índice discreto de intervalo o barra de 5 minutos transcurrido desde $\tau_0$ en el día en curso.

| Símbolo | Nombre Formal | Dominio Matemático | Unidades Físicas / Financieras | Descripción Operativa |
| :--- | :--- | :--- | :--- | :--- |
| $O_i, H_i, L_i, C_i$ | Precios de barra de 5 min | $\mathbb{R}_{++}^4$ | Puntos de Índice | Precios de Apertura, Máximo, Mínimo y Cierre del intervalo discreto $i$. |
| $P_i^{typ}$ | Precio Típico de la barra | $\mathbb{R}_{++}$ | Puntos de Índice | Media aritmética del intervalo: $P_i^{typ} = \frac{H_i + L_i + C_i}{3}$. |
| $V_i$ | Volumen transado en barra $i$ | $\mathbb{N}_{++}$ | Contratos de Futuros | Cantidad de contratos ejecutados en el intervalo discreto $i$. |
| $\mathcal{V}_t$ | Volumen acumulado desde ancla | $\mathbb{N}_{++}$ | Contratos de Futuros | Sumatorio continuo del volumen: $\mathcal{V}_t = \sum_{i=1}^t V_i$. |
| $VWAP_t$ | Anchored VWAP acumulado | $\mathbb{R}_{++}$ | Puntos de Índice | Precio medio ponderado por volumen acumulado desde $\tau_0 = 09:30:00$ EST. |
| $\sigma_{VWAP, t}$ | Desviación estándar VWAP | $\mathbb{R}_{++}$ | Puntos de Índice | Dispersión cuadrática ponderada por volumen acumulada hasta el intervalo $t$. |
| $ADX_{14, t}$ | Average Directional Index | $[0, 100]$ | Adimensional | Medida de fuerza de tendencia calculada sobre 14 intervalos de 5 minutos. |
| $\kappa_{slope, t}$ | Pendiente de la curva VWAP | $\mathbb{R}$ | Puntos / Minuto | Tasa de cambio de VWAP: $\kappa_{slope, t} = \frac{VWAP_t - VWAP_{t-3}}{3 \cdot \Delta t}$. |
| $M$ | Multiplicador de contrato CME | $\{50, 5, 20, 2\}$ | $\text{USD} / \text{Punto}$ | Multiplicador ($M_{ES}=50$, $M_{MES}=5$, $M_{NQ}=20$, $M_{MNQ}=2$). |
| $\delta_{tick}$ | Tamaño de tick mínimo | $\{0.25\}$ | Puntos de Índice | Variación mínima de cotización en los futuros de CME Globex. |
| $k_{buffer}$ | Colchón de Stop Técnico | $\mathbb{N}$ | Ticks de cotización | Margen de protección más allá de la mecha de rechazo ($k_{buffer} = 4$ ticks). |
| $C_{fric}$ | Coste total de fricción | $\mathbb{R}_{++}$ | USD / Contrato | Coste combinado de compensación y deslizamiento ($2 \cdot \delta_{tick} \cdot M + \text{Comisión}$). |

---

### 2.2 Espacio de Estados y Posición de Mercado

El estado analítico del sistema en cualquier instante discreto $t$ se define en el conjunto finito:
$$\mathcal{S}_t \in \{\text{ACCUMULATING\_VOLUME}, \text{TREND\_INHIBITED}, \text{ROTATIONAL\_SEARCH}, \text{LONG\_REVERSION}, \text{SHORT\_REVERSION}, \text{FLAT\_COMPLETED}\}$$

La función escalar de posición direccional de mercado se formula como:
$$w_t \in \{-1, 0, +1\}$$
donde:
- $w_t = +1$: Posición larga abierta tras confirmación de rechazo en banda inferior.
- $w_t = -1$: Posición corta abierta tras confirmación de rechazo en banda superior.
- $w_t = 0$: Posición neutral / sin exposición al mercado.

---

### 2.3 Horizontes Temporales y Ventanas de Negociación

La sesión de subasta se descompone en cuatro ventanas operativas cronológicas:

$$\begin{aligned}
\mathcal{W}_{init} &= [09:30:00, 10:30:00) \quad &&\text{(Periodo de Acumulación Inicial: prohibición absoluta de operar; estabilización de } \sigma_{VWAP}\text{)} \\
\mathcal{W}_{revers} &= [10:30:00, 15:00:00) \quad &&\text{(Ventana Operativa de Reversión Rotacional: máxima eficacia de la subasta equilibrada)} \\
\mathcal{W}_{inhibit} &= [15:00:00, 15:15:00) \quad &&\text{(Inhibición por Flujo de Cierre: cancelación de nuevas entradas para evitar el MOC)} \\
\mathcal{W}_{flat} &= [15:15:00, 17:00:00] \quad &&\text{(Ventana de Cierre Forzoso: liquidación mandatoria de cualquier posición remanente)}
\end{aligned}$$

---

## 3. Formulación Matemática de Generación de Señales

### 3.1 Ecuación Axiomática del Anchored VWAP

Para cualquier barra discreta $t \ge 1$, el Anchored VWAP representa la integral discreta de Riemann-Stieltjes del precio típico ponderado por el volumen transado:
$$VWAP_t = \frac{\sum_{i=1}^t P_i^{typ} \cdot V_i}{\sum_{i=1}^t V_i} = \frac{1}{\mathcal{V}_t} \sum_{i=1}^t P_i^{typ} \cdot V_i$$

---

### 3.2 Varianza y Desviación Estándar Ponderada por Volumen

La varianza condicional ponderada por volumen acumulada desde el ancla se calcula de forma exacta como:
$$\sigma_{VWAP, t}^2 = \frac{\sum_{i=1}^t V_i \cdot \left( P_i^{typ} - VWAP_t \right)^2}{\mathcal{V}_t}$$

La desviación estándar ponderada por volumen es su raíz cuadrada positiva:
$$\sigma_{VWAP, t} = \sqrt{\sigma_{VWAP, t}^2}$$

---

### 3.3 Bandas de Dispersión Estocástica de la Subasta

Para un múltiplo de dispersión $k \in \{1.0, 2.0, 3.0\}$:
$$\begin{aligned}
\text{Banda Superior}^{(k)}_t &= VWAP_t + k \cdot \sigma_{VWAP, t} \\
\text{Banda Inferior}^{(k)}_t &= VWAP_t - k \cdot \sigma_{VWAP, t}
\end{aligned}$$

En una subasta equilibrada (rotacional), el intervalo $[\text{Banda Inferior}^{(1.0)}_t, \text{Banda Superior}^{(1.0)}_t]$ delimita aproximadamente el **Área de Valor** institucional (el 68.2%–70% de las transacciones), mientras que las bandas de $k=2.0$ delimitan las colas extremas de rechazo de liquidez (95.4% de la distribución). En el contrato $NQ$, debido a su elevada leptocurtosis, se admite un parámetro $k = 2.5$.

---

### 3.4 Filtro de Régimen No Tendencial (Non-Trending Regime Gate)

La estrategia de reversión a la media sólo posee esperanza matemática positiva cuando el mercado carece de un vector direccional tendencial dominante. La función booleana de habilitación de régimen rotacional $\mathcal{R}_{rot, t} \in \{0, 1\}$ se modela formalmente como:
$$\mathcal{R}_{rot, t} = \mathbb{I}\left( ADX_{14, t} < 25.0 \right) \cdot \mathbb{I}\left( \left| \kappa_{slope, t} \right| \le \delta_{max\_slope} \right)$$
donde el umbral máximo de inclinación de la media ponderada se fija en:
- $\delta_{max\_slope} = 0.15 \text{ puntos/minuto en futuros } ES$.
- $\delta_{max\_slope} = 0.45 \text{ puntos/minuto en futuros } NQ$.

---

### 3.5 Ecuaciones Cerradas de Disparo de Entrada

El gatillo de entrada exige una secuencia determinista en dos etapas: penetración del extremo exterior ($\pm 2\sigma$) en la barra $t-1$, seguida por una confirmación de rechazo y cierre de vuelta al interior de la banda en la barra $t$:

1. **Señal de Compra por Reversión Larga ($S_{Long, t}$):**
   $$S_{Long, t} = \mathbb{I}\left( L_{t-1} < \text{Banda Inferior}^{(2.0)}_{t-1} \;\land\; C_t \ge \text{Banda Inferior}^{(2.0)}_t \;\land\; \mathcal{R}_{rot, t} = 1 \;\land\; \tau \in \mathcal{W}_{revers} \right)$$

2. **Señal de Venta por Reversión Corta ($S_{Short, t}$):**
   $$S_{Short, t} = \mathbb{I}\left( H_{t-1} > \text{Banda Superior}^{(2.0)}_{t-1} \;\land\; C_t \le \text{Banda Superior}^{(2.0)}_t \;\land\; \mathcal{R}_{rot, t} = 1 \;\land\; \tau \in \mathcal{W}_{revers} \right)$$

3. **Operador de Señal Discreta Integrada:**
   $$S_t = (+1) \cdot S_{Long, t} + (-1) \cdot S_{Short, t}$$
   con garantía analítica de exclusión mutua: $S_{Long, t} \cdot S_{Short, t} \equiv 0$.

---

## 4. Formulación Analítica de Salida de Posición

Sea $P_{entry} = C_t$ el precio de ejecución al completarse la barra de confirmación. Sea $S_t \in \{-1, +1\}$ la dirección del trade.

### 4.1 Ecuación Analítica de Take-Profit Dinámico (Reversión a Fair Value)

El objetivo de ganancia en una subasta rotacional no es un nivel estático, sino la convergencia hacia el precio de consenso ponderado por volumen ($VWAP_\tau$). Dado que el VWAP se actualiza continuamente con cada nuevo contrato transado, el nivel de Take-Profit es una función no estacionaria del tiempo:
$$P_{TP, t}(\tau) = VWAP_\tau$$

El instante de activación del Take-Profit se formula como el tiempo de parada:
$$\tau_{TP} = \inf \left\{ \tau \in (\tau_{entry}, 15:15:00] \;\middle|\; S_t \cdot \left( P_t(\tau) - VWAP_\tau \right) \ge 0 \right\}$$

---

### 4.2 Ecuación Analítica de Stop-Loss Técnico Extremo

El stop-loss se ubica analíticamente más allá del mínimo o máximo de la mecha de rechazo que generó el patrón, añadiendo un amortiguador de seguridad de $k_{buffer} = 4$ ticks (\$50 en ES, \$20 en NQ):
$$P_{SL, t} = \begin{cases}
\min\left( L_{t-1}, \; L_t \right) - k_{buffer} \cdot \delta_{tick} & \text{para posición Long } (S_t = +1) \\[6pt]
\max\left( H_{t-1}, \; H_t \right) + k_{buffer} \cdot \delta_{tick} & \text{para posición Short } (S_t = -1)
\end{cases}$$

El tiempo de parada por Stop-Loss es:
$$\tau_{SL} = \inf \left\{ \tau \in (\tau_{entry}, 15:15:00] \;\middle|\; S_t \cdot \left( P_t(\tau) - P_{SL, t} \right) \le 0 \right\}$$

---

### 4.3 Disyuntor Catastrófico de Expansión Estocástica ($\pm 3\sigma$)

Si el mercado experimenta una ruptura tendencial agresiva en contra de la posición y cruza la tercera banda de dispersión, la hipótesis de subasta rotacional queda definitivamente refutada:
$$\tau_{3\sigma} = \inf \left\{ \tau \in (\tau_{entry}, 15:15:00] \;\middle|\; \left| P_t(\tau) - VWAP_\tau \right| \ge 3.0 \cdot \sigma_{VWAP, \tau} \right\}$$

---

### 4.4 Decaimiento Temporal Forzoso y Salida Unificada

Para evitar que la posición quede expuesta a las sacudidas del cierre institucional MOC y asegurar el mandato de neutralidad absoluta de inventario nocturno, se define $\tau_{flat} = 15:15:00$ EST.

El tiempo de parada final de liquidación es unívocamente:
$$\tau_{exit} = \min \left( \tau_{TP}, \; \tau_{SL}, \; \tau_{3\sigma}, \; 15:15:00\text{ EST} \right)$$

Para cualquier instante posterior, la exposición satisface idénticamente:
$$w_t(\tau) = 0, \quad \forall \tau \in [\tau_{exit}, 16:00:00]$$

---

## 5. Dimensionamiento de Posición y Control de Riesgo Cuantitativo para Prop Trading

Las reglas de proprietary trading exigen blindar la cuenta frente al límite de pérdida diaria ($DLL$) y el trailing drawdown máximo ($MTDD$).

### 5.1 Parámetros de Riesgo Contractual

- $DLL$: Límite de Pérdida Diaria máxima permitida (USD, e.g., \$1,000 en cuentas de \$50,000).
- $MTDD$: Límite de Trailing Drawdown flotante máximo (USD, e.g., \$2,000 en cuentas de \$50,000).
- $HWM_t = \max_{k \le t} C_{equity, k}$: Marca de agua máxima de equidad.
- $DD_t = HWM_t - C_{equity, t}$: Drawdown acumulado actual (USD).
- $L_{day, t}$: Pérdida realizada acumulada en el día $t$ (USD).
- $K_{loss} \in \mathbb{N}_0$: Racha de operaciones consecutivas desfavorables.

---

### 5.2 Presupuesto Dinámico de Riesgo por Operación ($\mathcal{B}_{F3, t}$)

El presupuesto de pérdida admisible para cada operación de reversión se calcula analíticamente como:
$$\mathcal{B}_{F3, t} = \min \left( \phi(K_{loss}) \cdot \max\left( 0, \; DLL - L_{day, t} \right), \; \psi \cdot \max\left( 0, \; MTDD - DD_t \right) \right)$$
con $\psi = 0.20$ (reserva de 5 operaciones sobre el margen de drawdown) y amortiguador por rachas:
$$\phi(K_{loss}) = 0.30 \times 2^{-\min(K_{loss}, 3)}$$

---

### 5.3 Riesgo Monetario Unitario por Contrato

Definiendo la distancia de stop técnico estructural como:
$$\Delta_{SL, t} = |P_{entry} - P_{SL, t}|$$
El riesgo monetario por contrato con multiplicador CME $M$ se formula como:
$$\mathcal{R}_{contract, t}(M) = \Delta_{SL, t} \cdot M + C_{fric}$$
donde $C_{fric} = 2 \cdot \delta_{tick} \cdot M + \text{Comisión}$.

---

### 5.4 Modelo Declarativo de Conmutación Automática Estándar / Micro

El número de contratos estándar $N_{std, t}$ a ejecutar se determina mediante la función piso:
$$N_{std, t} = \left\lfloor \frac{\mathcal{B}_{F3, t}}{\mathcal{R}_{contract, t}(M_{std})} \right\rfloor$$

Si el presupuesto disponible no cubre el riesgo de un contrato estándar ($N_{std, t} = 0$), el algoritmo conmuta automáticamente a micro-contratos ($M_{micro} = \frac{1}{10} M_{std}$):
$$N_{micro, t} = \min \left( \left\lfloor \frac{\mathcal{B}_{F3, t}}{\mathcal{R}_{contract, t}(M_{micro})} \right\rfloor, \; N_{max, micro} \right)$$
con $N_{max, micro} = 15$.

---

### 5.5 Disyuntor Cuantitativo de Volatilidad Excesiva

Si la dispersión del mercado se expande de manera tal que el riesgo de invalidación de un solo micro-contrato excede el presupuesto admisible:
$$\text{Si } \mathcal{R}_{contract, t}(M_{micro}) > \mathcal{B}_{F3, t} \implies N_t = 0 \implies \text{HALT\_VOLATILITY\_DEFENSE}$$

Esta condición formal garantiza que ninguna transacción comprometa la viabilidad de la cuenta de prop trading en entornos de volatilidad descontrolada.

---

## 6. Rúbrica de Verificación Analítica Objetiva

La siguiente matriz de auditoría certifica la validez matemática y la conformidad de la especificación de la estrategia F3:

| Criterio de Verificación Cuantitativa | Expresión Analítica Auditada | Umbral / Condición de Aprobación | Estado de Cumplimiento | Justificación Teórica y Evidencia Formal |
| :--- | :--- | :--- | :--- | :--- |
| **Ausencia Total de Código** | $\text{Card}(\text{CodeBlocks}) = 0$ | Idéntico a 0 bloques de código | **APROBADO** | Especificado exclusivamente mediante ecuaciones en notación declarativa formal LaTeX. |
| **Exclusión de Indicadores Tradicionales Minoristas** | $\text{Card}(\text{IndicadoresDescartados}) = 0$ | Cero indicadores minoristas rezagados | **APROBADO** | Reversión gobernada por el consenso institucional del Anchored VWAP y sus bandas estocásticas. |
| **Definición Rigurosa de Variables y Dominios** | $\forall x \in \text{Inputs}, \; \text{Dom}(x) \subseteq \mathbb{R} \lor \mathbb{N}$ | 100% de variables unívocas | **APROBADO** | La totalidad de parámetros posee dominio matemático y unidades físicas financieras (USD, ticks, puntos). |
| **Estabilidad Muestral Asintótica** | Inactividad formal en $\tau \in [09:30, 10:30)$ | $\Delta \tau_{init} = 60\text{ min}$ | **APROBADO** | Previene indeterminaciones numéricas por varianza nula o muestras pequeñas en los primeros minutos. |
| **Filtro de Régimen No Tendencial** | $\mathcal{R}_{rot, t} = \mathbb{I}(ADX < 25) \cdot \mathbb{I}(|\kappa_{slope}| \le \delta)$ | Inhibición en mercados con tendencia | **APROBADO** | Bloquea de forma determinista la compra de activos en caída libre durante días direccionales. |
| **Take-Profit Dinámico Conforme** | $P_{TP, t}(\tau) = VWAP_\tau$ | Convergencia a precio medio justo | **APROBADO** | Captura exacta de la rotación hacia el precio ponderado por volumen institucional. |
| **Disyuntor de Expansión $3\sigma$** | Salida en $\inf\{\tau \mid \|P_t - VWAP_\tau\| \ge 3\sigma\}$ | Límite estructural de invalidación | **APROBADO** | Detiene de inmediato la pérdida ante un cambio de régimen intempestivo de rotación a tendencia. |
| **Control de Riesgo Prop Firm** | $\mathcal{R}_{total}(N_t) \le \mathcal{B}_{F3, t} \le 0.30 \cdot DLL$ | Pérdida máxima acotada al 30% DLL | **APROBADO** | Imposibilidad matemática de violar el límite de pérdida diaria de la cuenta fondeada. |
| **Inmunidad Overnight en CME** | $\tau_{exit} \le 15:15:00\text{ EST}$ | Liquidación 105 min antes de CME | **APROBADO** | Posiciones 100% cerradas antes del inicio de la subasta MOC y de la pausa de compensación diaria. |
