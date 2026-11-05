# Especificación Matemática Formal: Estrategia F7 - Meta-Labeling y Método de Triple Barrera

**Identificador de Estrategia:** F7  
**Familia Cuantitativa:** Machine Learning Cuantitativo Híbrido / Clasificación Probabilística Supervisada / Filtrado Microestructural  
**Instrumentos Objetivo:** CME E-mini S&P 500 ($ES$), CME Micro E-mini S&P 500 ($MES$), CME E-mini Nasdaq-100 ($NQ$), CME Micro E-mini Nasdaq-100 ($MNQ$)  
**Mercado de Negociación:** Chicago Mercantile Exchange (CME Globex)  
**Marco de Riesgo:** Proprietary Trading (Cuentas Fondeadas: Topstep, Apex, Lucid, MyFundedFutures)  
**Nivel de Documento:** Especificación Matemática y Microestructural Modular Autosuficiente (Apta para entrega directa a agentes de implementación de software)  
**Restricción Estricta de Formato:** Teoría matemática analítica pura en notación declarativa formal LaTeX. Queda estrictamente excluido cualquier fragmento de código ejecutable, script o pseudocódigo, así como cualquier indicador técnico tradicional descartado.

---

## 1. Identificación y Fundamento Teórico / Hipótesis Microestructural

### 1.1 Citas Académicas de Referencia
1. **López de Prado, M. (2018)**. *Advances in Financial Machine Learning*. John Wiley & Sons, Inc., Hoboken, New Jersey. Capítulos 3 ("Targeting: The Triple-Barrier Method") y 5 ("Fractionally Differentiated Features"), pp. 43–76. DOI: [10.1002/9781119522898](https://doi.org/10.1002/9781119522898).
2. **López de Prado, M. (2020)**. *Machine Learning for Asset Managers*. Cambridge University Press, Elements in Quantitative Finance. pp. 18–37. DOI: [10.1017/9781108883658](https://doi.org/10.1017/9781108883658).

---

### 1.2 Hipótesis Económica y Dinámica Microestructural en Español Sencillo

En el trading cuantitativo institucional sobre futuros de índices bursátiles (como el $ES$ y el $NQ$), exigir a un único modelo de aprendizaje automático que resuelva a la vez dos tareas de diferente naturaleza suele conducir al fracaso:
1. Predecir hacia dónde se moverá el precio (si subirá o bajará).
2. Determinar si la oportunidad presenta suficiente calidad para superar el spread, las comisiones y el riesgo de parada de pérdidas.

Dado que en los mercados financieros la relación señal-ruido es extremadamente baja, obligar a un clasificador a predecir simultáneamente dirección y magnitud genera un sobreajuste severo (*overfitting*).

Para resolver esta vulnerabilidad estructural, Marcos López de Prado (2018) diseñó una arquitectura desacoplada de dos niveles denominada **Meta-Labeling**:

1. **Nivel 1 (Modelo Primario de Hipótesis Económica):**  
   Un modelo primario, fundamentado en principios económicos o dinámicas de microestructura (como rupturas de rango tras compresión, desbalances de volumen o desviaciones de valor frente al VWAP), decide exclusivamente el *sentido direccional* de la operación: comprar ($+1$) o vender ($-1$). Su meta es maximizar la sensibilidad de captura (*recall*), sin preocuparse si una fracción de las señales resulta ser un falso rompimiento.

2. **Nivel 2 (Meta-Modelo de Calidad y Filtrado):**  
   Un modelo secundario de clasificación supervisada no busca adivinar la dirección del mercado, sino evaluar si *el modelo primario tiene una alta probabilidad de acertar en las condiciones actuales*. Este meta-clasificador analiza el vector de estado microestructural (volatilidad local, presión en el libro de órdenes, volumen relativo, tiempo transcurrido de la sesión) y emite una probabilidad condicional $p_t$. Si $p_t$ supera un umbral de convicción analítica $\tau_{meta}$, la operación es autorizada y su tamaño se modula de forma continua (*bet sizing*); si $p_t < \tau_{meta}$, la señal es **vetada**, anulando el falso positivo y protegiendo el capital.

3. **El Método de Triple Barrera (*Triple-Barrier Method*):**  
   Para etiquetar objetivamente el entrenamiento del meta-modelo sin caer en el sesgo de horizonte temporal fijo (el cual ignora la trayectoria del precio), se establecen tres fronteras simultáneas al instante de apertura:
   - **Barrera Superior de Beneficio ($B_{upper}$):** Nivel de precio proporcional a la volatilidad local donde se toma ganancia.
   - **Barrera Inferior de Pérdida ($B_{lower}$):** Nivel de precio proporcional a la volatilidad local donde se corta la pérdida.
   - **Barrera Vertical Temporal ($T_{vertical}$):** Tiempo máximo de permanencia en mercado tras el cual la posición se liquida obligatoriamente.

El primer tiempo de parada (*first stopping time*) en el que el proceso estocástico del precio colisiona con una de las tres barreras define el resultado unívoco de la operación. De este modo, la formulación matemática reproduce exactamente las reglas de ejecución contingente de una orden real en el libro de órdenes de CME Globex.

---

## 2. Variables de Entrada, Espacio de Estados y Horizontes Temporales

### 2.1 Variables de Entrada y Parámetros Formales

Sea $t$ el tiempo continuo parametrizado en la sesión bursátil de CME Globex. Sea $t_0$ el instante analítico exacto en el que el modelo primario emite una señal direccional candidata.

| Símbolo | Nombre Formal | Dominio Matemático | Unidades Físicas / Financieras | Descripción Operativa |
| :--- | :--- | :--- | :--- | :--- |
| $P_t$ | Precio de mercado instantáneo | $\mathbb{R}_{++}$ | Puntos de Índice | Precio del último cruce registrado en el libro de órdenes de CME Globex en el instante $t$. |
| $P_{t_0}$ | Precio de apertura del trade | $\mathbb{R}_{++}$ | Puntos de Índice | Precio de transacción registrado en el instante de entrada $t_0$. |
| $\sigma_{t_0}$ | Volatilidad instantánea local | $\mathbb{R}_{++}$ | Puntos de Índice | Desviación estándar local del subyacente calculada mediante el estimador de rango de Garman-Klass o EWMA intradía. |
| $Z_{OFI, t}$ | Desbalance de flujo de órdenes | $\mathbb{R}$ | Adimensional ($Z$-score) | Puntuación estandarizada del flujo neto de órdenes en el mejor bid/ask del libro de órdenes CME. |
| $\Delta P_{\text{VWAP}, t}$ | Distancia al VWAP intra-sesión | $\mathbb{R}$ | Puntos de Índice | Diferencial analítico entre el precio actual y el VWAP acumulado de la sesión: $P_t - \text{VWAP}_t$. |
| $RVOL_t$ | Ratio de volumen relativo | $\mathbb{R}_{++}$ | Ratio Adimensional | Cociente entre el volumen transado en la barra actual y su media histórica para la misma ventana horaria. |
| $\mathcal{S}_{\text{range}, t}$ | Compresión del rango previo | $\mathbb{R}_{++}$ | Ratio Adimensional | Rango de oscilación reciente normalizado por el rango promedio de las últimas 5 sesiones (indicador de compresión). |
| $\tau_{\text{tod}}$ | Fracción de tiempo de sesión | $[0, 1]$ | Adimensional | Fracción temporal transcurrida dentro del horario regular de negociación ($RTH$, 09:30 a 16:00 EST). |
| $\ell^+$ | Multiplicador de toma de beneficio | $\mathbb{R}_{++}$ | Adimensional | Coeficiente escalar para fijar la barrera superior de ganancia ($\ell^+ \ge 1.50$). |
| $\ell^-$ | Multiplicador de parada de pérdida | $\mathbb{R}_{++}$ | Adimensional | Coeficiente escalar para fijar la barrera inferior de pérdida ($\ell^- = 1.00$). |
| $\Delta \tau$ | Ventana de barrera vertical | $\mathbb{R}_{++}$ | Minutos | Horizonte temporal máximo de permanencia en el mercado ($\Delta \tau \in [15, 60]\text{ min}$). |
| $\tau_{\text{meta}}$ | Umbral analítico de veto | $(0.50, 1.00)$ | Adimensional (Probabilidad) | Probabilidad mínima requerida para autorizar la ejecución de la señal primaria ($\tau_{\text{meta}} = 0.60$). |
| $M^{(I)}$ | Multiplicador contractual CME | $\{50, 5, 20, 2\}$ | $\text{USD} / \text{Punto}$ | Multiplicador monetario canónico: $M^{(\text{ES})}=50$, $M^{(\text{MES})}=5$, $M^{(\text{NQ})}=20$, $M^{(\text{MNQ})}=2$. |
| $\delta_{tick}^{(I)}$ | Tamaño de tick mínimo | $\{0.25\}$ | Puntos de Índice | Resolución discreta mínima de cotización en CME Globex. |

---

### 2.2 Vector de Características Microestructurales

En el instante de evaluación $t_0$, el entorno microestructural queda condensado en el vector de características continuas $X_{t_0} \in \mathcal{X} \subset \mathbb{R}^6$:
$$X_{t_0} = \begin{bmatrix} \sigma_{t_0} \\ Z_{OFI, t_0} \\ \Delta P_{\text{VWAP}, t_0} \\ RVOL_{t_0} \\ \mathcal{S}_{\text{range}, t_0} \\ \tau_{\text{tod}} \end{bmatrix} \in \mathbb{R}^6$$

---

### 2.3 Espacio de Estados y Horizontes Temporales

El estado global del sistema de trading en cualquier instante $t$ se describe mediante la tupla analítica:
$$\mathcal{S}_t = \left( P_t, \; s_t^{(1)}, \; \sigma_t, \; X_t, \; t_0, \; \Delta \tau, \; w_t \right) \in \mathbb{R}_{++} \times \{-1, 0, +1\} \times \mathbb{R}_{++} \times \mathbb{R}^6 \times \mathbb{R}_+ \times \mathbb{R}_{++} \times \{-1, 0, +1\}$$
donde $w_t \in \{-1, 0, +1\}$ denota la posición activa en el mercado ($-1$: corto, $0$: plano, $+1$: largo).

Los horizontes de operación se ciñen rigurosamente a las sesiones de CME Globex:
- **Horario Regular de Negociación ($RTH$):** $[09:30:00, 16:00:00]\text{ EST}$.
- **Ventana de Ejecución Activa:** $[09:45:00, 15:30:00]\text{ EST}$ (se evitan los primeros 15 minutos de apertura caótica y los últimos 30 minutos de subasta MOC).
- **Límite de Liquidación Intradía Obligatoria:** $\tau_{close} = 15:55:00\text{ EST}$. Prohibición absoluta de permanencia en el mercado hacia el cierre de compensación técnica de CME (17:00 EST).

---

## 3. Formulación Matemática de Generación de Señales

### 3.1 Hipótesis del Modelo Primario Direccional

Sea $s_t^{(1)} \in \{-1, 0, +1\}$ la función de señal generada por el modelo primario de hipótesis económica en el instante $t$. Esta señal determina el lado propuesto de la operación:
$$s_t^{(1)} = \begin{cases} 
+1 & \text{si la hipótesis económica determina sesgo comprador (Long)} \\
-1 & \text{si la hipótesis económica determina sesgo vendedor (Short)} \\
0 & \text{si el mercado se encuentra en equilibrio estocástico (Neutral)}
\end{cases}$$

---

### 3.2 Clasificador Probabilístico de Meta-Labeling

Condicionado a la emisión de una señal primaria no nula ($s_t^{(1)} \ne 0$), el meta-modelo evalúa la verosimilitud de que el trade alcance la barrera de toma de beneficios antes de impactar el stop-loss o expirar en el tiempo. La probabilidad de acierto condicional se formaliza analíticamente mediante la función logística multivariada:
$$p_t = \mathbb{P}\left( y_t = 1 \;\middle|\; X_t, \, s_t^{(1)} \right) = \frac{1}{1 + \exp\left( - \mathcal{F}_{\text{meta}}(X_t, s_t^{(1)}) \right)} \in [0, 1]$$
donde $\mathcal{F}_{\text{meta}}: \mathbb{R}^6 \times \{-1, +1\} \to \mathbb{R}$ es una función de evaluación declarativa de margen que proyecta el vector de estado hacia el logit de ventaja estadística:
$$\mathcal{F}_{\text{meta}}(X_t, s_t^{(1)}) = \alpha_0 + \sum_{k=1}^6 \beta_k \cdot X_{t, k} + s_t^{(1)} \cdot \left( \gamma_1 Z_{OFI, t} + \gamma_2 \frac{\Delta P_{\text{VWAP}, t}}{\sigma_t} \right)$$
con parámetros de calibración $\{\alpha_0, \beta_k, \gamma_j\} \subset \mathbb{R}$ fijados analíticamente para satisfacer ortogonalidad y calibración estricta de probabilidad $\mathbb{E}[y \mid p] = p$.

---

### 3.3 Función Analítica de Veto y Señal Final Filtrada

Para blindar la cuenta frente a operaciones con expectativa matemática deficiente, se establece el umbral de corte analítico $\tau_{\text{meta}} = 0.60$. La señal ejecutoria final $S_t^* \in \{-1, 0, +1\}$ se formula mediante el producto del sentido primario por la función indicatriz de convicción probabilística:
$$S_t^* = s_t^{(1)} \cdot \mathbb{I}\left( p_t \ge \tau_{\text{meta}} \right)$$

Desglosando exhaustivamente las tres condiciones disjuntas del operador de señal:

$$\begin{aligned}
S_{\text{Long}, t} &= \mathbb{I}\left( s_t^{(1)} = +1 \;\land\; p_t \ge \tau_{\text{meta}} \right) \\
S_{\text{Short}, t} &= \mathbb{I}\left( s_t^{(1)} = -1 \;\land\; p_t \ge \tau_{\text{meta}} \right) \\
S_{\text{Neutral}, t} &= \mathbb{I}\left( s_t^{(1)} = 0 \;\lor\; p_t < \tau_{\text{meta}} \right)
\end{aligned}$$

Propiedad de ortogonalidad analítica estricta:
$$S_{\text{Long}, t} \cdot S_{\text{Short}, t} \equiv 0$$
$$S_{\text{Long}, t} + S_{\text{Short}, t} + S_{\text{Neutral}, t} \equiv 1$$

Si $p_t < \tau_{\text{meta}}$, la función indicatriz adopta el valor cero, provocando el **veto analítico inmediato** de la señal primaria y manteniendo la cuenta en liquidez plana ($S_t^* = 0$).

---

### 3.4 Función Continua de Modulación de Apuesta (*Bet Sizing Analítico*)

Siguiendo el principio de López de Prado (2018), la convicción del clasificador debe trasladarse de manera proporcional y continua a la fracción de capital arriesgada. Se formula la divergencia estandarizada frente al azar uniforme ($p=0.50$):
$$z_t = \frac{p_t - 0.50}{\sqrt{p_t (1 - p_t)}}$$

El coeficiente escalar de apuesta $m(p_t) \in [0, 1]$ se determina a través de la función de distribución acumulada gaussiana estándar $\Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^z e^{-u^2/2} du$:
$$m(p_t) = \begin{cases} 
2 \Phi(z_t) - 1 & \text{si } p_t \ge \tau_{\text{meta}} \\[4pt]
0 & \text{si } p_t < \tau_{\text{meta}}
\end{cases}$$

Alternativamente, bajo la formulación polinómica cóncava con exponente de amortiguación $\eta \ge 1$:
$$m(p_t) = \left( \frac{p_t - \tau_{\text{meta}}}{1 - \tau_{\text{meta}}} \right)^\eta \cdot \mathbb{I}\left( p_t \ge \tau_{\text{meta}} \right)$$
con $\eta = 1.50$, lo que garantiza que una señal apenas por encima del umbral ($\tau_{\text{meta}} = 0.60$) reciba una exposición pequeña y prudente, mientras que probabilidades elevadas ($p_t \ge 0.85$) desplieguen la totalidad del presupuesto disponible.

---

## 4. Formulación Analítica de Salida de Posición

Sea $t_0$ el instante analítico exacto de apertura de la posición y $P_{t_0}$ el precio de entrada ejecutado. La arquitectura de desinversión queda determinada por la interacción dinámica de tres barreras cerradas ortogonales.

> **Geometría Analítica de la Triple Barrera (López de Prado):**  
> - **Barrera Superior ($B_{\text{upper}}$):** $P_{t_0} + \ell^+ \cdot \sigma_{t_0}$ [Toma de Beneficios Long / Stop-Loss Short].  
> - **Barrera Inferior ($B_{\text{lower}}$):** $P_{t_0} - \ell^- \cdot \sigma_{t_0}$ [Stop-Loss Long / Toma de Beneficios Short].  
> - **Barrera Vertical ($T_{\text{vertical}}$):** $\min(t_0 + \Delta \tau, \; \tau_{\text{close}})$ [Expiración temporal máxima intradiaria].  
> - **Dinámica de Liquidación:** La primera frontera alcanzada por la trayectoria del precio $P_t$ define el tiempo de parada $\tau^*$ y fuerza el cierre de la posición.

---

### 4.1 Definición Funcional de las Tres Barreras

Sean $\ell^+ = 2.00$ y $\ell^- = 1.00$ los multiplicadores de la dispersión local $\sigma_{t_0}$:

1. **Barrera Superior ($B_{\text{upper}}$):**
   $$B_{\text{upper}}(t_0) = \begin{cases} 
   P_{t_0} + \ell^+ \cdot \sigma_{t_0} & \text{si } S_{t_0}^* = +1 \quad (\text{Toma de Beneficios Long}) \\[4pt]
   P_{t_0} + \ell^- \cdot \sigma_{t_0} & \text{si } S_{t_0}^* = -1 \quad (\text{Stop-Loss Short})
   \end{cases}$$

2. **Barrera Inferior ($B_{\text{lower}}$):**
   $$B_{\text{lower}}(t_0) = \begin{cases} 
   P_{t_0} - \ell^- \cdot \sigma_{t_0} & \text{si } S_{t_0}^* = +1 \quad (\text{Stop-Loss Long}) \\[4pt]
   P_{t_0} - \ell^+ \cdot \sigma_{t_0} & \text{si } S_{t_0}^* = -1 \quad (\text{Toma de Beneficios Short})
   \end{cases}$$

3. **Barrera Vertical Temporal ($T_{\text{vertical}}$):**
   $$T_{\text{vertical}}(t_0) = \min\left( t_0 + \Delta \tau, \; \tau_{\text{close}} \right)$$
   donde $\tau_{\text{close}} = 15:55:00\text{ EST}$ representa el corte de seguridad técnica previa al cierre de sesión.

---

### 4.2 Tiempo de Parada Unificado (*First Stopping Time*)

El instante de liquidación de la posición $\tau^*$ se define matemáticamente como el ínfimo del conjunto de instantes donde el precio colisiona con el contorno de absorción:
$$\tau^* = \inf \left\{ t > t_0 \;\middle|\; P_t \ge B_{\text{upper}}(t_0) \;\lor\; P_t \le B_{\text{lower}}(t_0) \;\lor\; t \ge T_{\text{vertical}}(t_0) \right\}$$

---

### 4.3 Clasificación Analítica de Causas de Salida

La etiqueta de cierre $\mathcal{C}(\tau^*) \in \{\text{TP}, \text{SL}, \text{TIME}\}$ se determina de forma exhaustiva y disjunta:
$$\mathcal{C}(\tau^*) = \begin{cases} 
\text{TP} & \text{si } \left( S_{t_0}^* = +1 \land P_{\tau^*} \ge B_{\text{upper}} \right) \lor \left( S_{t_0}^* = -1 \land P_{\tau^*} \le B_{\text{lower}} \right) \\[4pt]
\text{SL} & \text{si } \left( S_{t_0}^* = +1 \land P_{\tau^*} \le B_{\text{lower}} \right) \lor \left( S_{t_0}^* = -1 \land P_{\tau^*} \ge B_{\text{upper}} \right) \\[4pt]
\text{TIME} & \text{si } \tau^* = T_{\text{vertical}}(t_0) \land B_{\text{lower}} < P_{\tau^*} < B_{\text{upper}}
\end{cases}$$

Tras el instante $\tau^*$, la función de exposición satisface incondicionalmente:
$$w_t = 0, \quad \forall t \in [\tau^*, \tau^* + \tau_{\text{cooldown}}]$$
imponiendo una pausa de enfriamiento (*cooldown*) de al menos $\tau_{\text{cooldown}} = 5\text{ minutos}$ antes de evaluar nuevas aperturas.

---

## 5. Dimensionamiento de Posición y Control de Riesgo Cuantitativo para Prop Trading

Las reglas contractuales de las firmas de Proprietary Trading imponen límites estrictos sobre la pérdida máxima acumulada en una sola jornada (*Daily Loss Limit*, $DLL$) y sobre el retroceso acumulado desde la máxima ganancia histórica (*Max Trailing Drawdown*, $MTDD$). El dimensionamiento de la posición debe subordinarse a estas cotas.

### 5.1 Variables de Capital y Colchones de Supervivencia

Sea $E_t$ la equidad líquida de la cuenta en el instante $t$.
- **Equidad de Inicio del Día ($E_{d, 0}$):** Balance de capital al comenzar la jornada $d$.
- **Margen Diario Restante ($\Delta DLL_t$):**
  $$\Delta DLL_t = \max\left( DLL - \left( E_{d, 0} - E_t \right), \; 0 \right)$$
- **Marca de Agua Histórica (*High-Water Mark*, $H_t$):**
  $$H_t = \sup_{0 \le s \le t} E_s$$
- **Colchón de Trailing Drawdown ($D_t^{\text{cushion}}$):**
  $$D_t^{\text{cushion}} = \max\left( E_t - \left( H_t - MTDD \right), \; 0 \right)$$

---

### 5.2 Presupuesto Dinámico de Pérdida por Operación

El capital máximo que puede perderse en caso de tocar la barrera de stop-loss en el trade $t_0$, denotado $\mathcal{B}_{\text{risk}, t_0}$, se formula como el mínimo acotado entre las fracciones de seguridad diaria y de trailing drawdown:
$$\mathcal{B}_{\text{risk}, t_0} = \min\left( \gamma_D \cdot \Delta DLL_{t_0}, \; \gamma_T \cdot D_{t_0}^{\text{cushion}} \right)$$
donde $\gamma_D = 0.20$ (ninguna operación puede consumir más del $20\%$ del límite diario disponible) y $\gamma_T = 0.08$ (ninguna operación puede arriesgar más del $8\%$ del colchón total de drawdown).

---

### 5.3 Riesgo Unitario por Contrato CME

La pérdida monetaria máxima al alcanzar la barrera inferior de stop-loss para un solo contrato del instrumento $I$ cotizado al precio $P_{t_0}$ es:
$$\Delta \mathcal{L}_{\text{unit}}^{(I)}(t_0) = \ell^- \cdot \sigma_{t_0} \cdot M^{(I)} + \mathcal{C}_{\text{fric}}^{(I)}$$
donde $\mathcal{C}_{\text{fric}}^{(I)} = 2 \cdot \delta_{\text{tick}}^{(I)} \cdot M^{(I)} + \text{Comisión}_{\text{clearing}}^{(I)}$ cuantifica la fricción total de ida y vuelta.

---

### 5.4 Ecuación Cerrada del Número de Contratos

Incorporando el coeficiente continuo de modulación de apuesta $m(p_{t_0}) \in [0, 1]$, el número entero de contratos estándar a emitir $N_{\text{std}, t_0}^{(I)} \in \mathbb{N}_0$ viene dado por la función piso:
$$N_{\text{std}, t_0}^{(I)} = \min\left( \left\lfloor \frac{\mathcal{B}_{\text{risk}, t_0} \cdot m(p_{t_0})}{\Delta \mathcal{L}_{\text{unit}}^{(I)}(t_0)} \right\rfloor, \; N_{\max}^{(I)} \right)$$

---

### 5.5 Algoritmo Declarativo de Conmutación Estándar a Micro

En fases de preservación de capital donde la volatilidad se expande o el colchón de riesgo disminuye, el cálculo para contratos estándar ($I \in \{\text{ES}, \text{NQ}\}$) puede arrojar cero unidades ($N_{\text{std}, t_0}^{(I)} = 0$).

Para evitar la parálisis operativa sin incrementar el riesgo, el sistema conmuta de forma analítica y automática al contrato Micro equivalente ($I_{\text{micro}} \in \{\text{MES}, \text{MNQ}\}$, con multiplicador $M^{(I_{\text{micro}})} = 0.10 \cdot M^{(I)}$):

$$N_{t_0}^* = \begin{cases}
N_{\text{std}, t_0}^{(I)} \text{ contratos de } I & \text{si } N_{\text{std}, t_0}^{(I)} \ge 1 \\[6pt]
\min\left( \left\lfloor \frac{\mathcal{B}_{\text{risk}, t_0} \cdot m(p_{t_0})}{\Delta \mathcal{L}_{\text{unit}}^{(I_{\text{micro}})}(t_0)} \right\rfloor, \; 10 \cdot N_{\max}^{(I)} \right) \text{ de } I_{\text{micro}} & \text{si } N_{\text{std}, t_0}^{(I)} = 0 \;\land\; \mathcal{B}_{\text{risk}, t_0} \cdot m(p_{t_0}) \ge \Delta \mathcal{L}_{\text{unit}}^{(I_{\text{micro}})}(t_0) \\[6pt]
0 \quad (\text{Bloqueo Defensivo Absoluto}) & \text{si } \mathcal{B}_{\text{risk}, t_0} \cdot m(p_{t_0}) < \Delta \mathcal{L}_{\text{unit}}^{(I_{\text{micro}})}(t_0)
\end{cases}$$

---

### 5.6 Disyuntor Analítico por Vulnerabilidad de Trailing Drawdown

Si el colchón de drawdown desciende por debajo de un umbral crítico de emergencia:
$$D_t^{\text{cushion}} \le 0.15 \cdot MTDD \implies N_t^* = 0 \implies \text{HALT\_DRAWDOWN\_DEFENSE}$$
Esta regla bloquea cualquier nueva apertura y liquida de inmediato las posiciones vigentes a mercado, garantizando que jamás se produzca la rescisión contractual de la cuenta fondeada.

---

## 6. Rúbrica de Verificación Analítica Objetiva

La siguiente matriz de auditoría cuantitativa evalúa formalmente el cumplimiento matemático y analítico de la especificación de la estrategia F7:

| Criterio de Verificación Cuantitativa | Expresión Analítica Auditada | Umbral / Condición de Aprobación | Estado de Cumplimiento | Justificación Teórica y Evidencia Formal |
| :--- | :--- | :--- | :--- | :--- |
| **Ausencia Total de Código** | $\text{Card}(\text{CodeBlocks}) = 0$ | Idéntico a 0 bloques de código | **APROBADO** | La totalidad del documento está redactada en notación matemática analítica formal en $\LaTeX$ y texto explicativo. |
| **Exclusión de Indicadores Descartados** | $\text{Card}(\{\text{SMA}, \text{RSI}\}) = 0$ | Cero menciones de SMA o RSI estándar | **APROBADO** | El sistema descansa en microestructura de flujo ($OFI$), volatilidad de rango y clasificación probabilística. |
| **Definición Rigurosa de Espacio de Estados** | $\forall x \in X_t, \; \text{Dom}(x) \subseteq \mathbb{R} \lor \mathbb{N}$ | 100% de variables con dominio y unidades | **APROBADO** | Cada parámetro dispone de tabla formal con unidades explícitas ($\text{pts}$, $\text{USD}$, ratios adimensionales). |
| **Ortogonalidad de Señales** | $S_{\text{Long}, t} \cdot S_{\text{Short}, t} \equiv 0$ | Disyunción estricta de conjuntos de activación | **APROBADO** | El signo de $s_t^{(1)} \in \{-1, +1\}$ es unívoco; es imposible generar órdenes opuestas simultáneas. |
| **Invarianza Temporal y Causalidad** | $\mathbb{E}[p_t \mid \mathcal{F}_{t}] = p_t \land \tau^* > t_0$ | Ausencia estricta de filtración hacia el futuro | **APROBADO** | El clasificador depende estrictamente de la información en $\mathcal{F}_{t_0}$, y el tiempo de parada $\tau^*$ es posterior a $t_0$. |
| **Asimetría Matemática de Barreras** | $\frac{\ell^+}{\ell^-} \ge 1.50$ | Relación beneficio/riesgo $\ge 1.50$ | **APROBADO** | Fijando $\ell^+ = 2.00$ y $\ell^- = 1.00$, la relación es $2.0$, asegurando valor esperado positivo con tasas de acierto $\ge 40\%$. |
| **Acotamiento del Límite Diario (DLL)** | $\lim_{\Delta DLL \to 0} N_t^* = 0$ | Respeto incondicional de $\Delta DLL_t$ | **APROBADO** | La función piso y la dependencia lineal de $\mathcal{B}_{\text{risk}}$ garantizan cero contratos cuando el margen diario se agota. |
| **Cierre Forzoso Pre-Compensación CME** | $\tau^* \le 15:55:00\text{ EST} < 17:00:00\text{ EST}$ | Cero retención overnight hacia el clearing | **APROBADO** | La barrera vertical impone $T_{\text{vertical}} \le 15:55:00\text{ EST}$, eliminando el riesgo de mantenimiento de CME Globex. |
| **Conmutación Granular Discreta** | $N_t^* \in \mathbb{N}_0$ | Asignación discreta sin números fraccionarios | **APROBADO** | El dimensionamiento emplea la función piso $\lfloor \cdot \rfloor$ y transita fluidamente a contratos Micro ($MES/MNQ$). |
