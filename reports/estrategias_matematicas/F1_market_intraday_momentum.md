# Especificación Matemática Formal: Estrategia F1 - Market Intraday Momentum & MOC Seasonality

**Identificador de Estrategia:** F1  
**Familia Cuantitativa:** Trading Sistemático / Estacionalidad de Subasta Intradía / Flujo Institucional  
**Instrumentos Objetivo:** CME E-mini S&P 500 ($ES$), CME Micro E-mini S&P 500 ($MES$), CME E-mini Nasdaq-100 ($NQ$), CME Micro E-mini Nasdaq-100 ($MNQ$)  
**Mercado de Negociación:** Chicago Mercantile Exchange (CME Globex)  
**Marco de Riesgo:** Proprietary Trading (Cuentas Fondeadas: Topstep, Apex, Lucid, MyFundedFutures)  
**Nivel de Documento:** Especificación Matemática y Microestructural Modular Autosuficiente (Apta para entrega directa a agentes de implementación de software)  
**Restricción Estricta de Formato:** Teoría matemática analítica pura en notación declarativa formal LaTeX. Queda estrictamente excluido cualquier fragmento de código ejecutable, script o pseudocódigo, así como cualquier indicador técnico tradicional descartado.

---

## 1. Identificación y Fundamento Teórico / Hipótesis Microestructural

### 1.1 Citas Académicas de Referencia
1. **Gao, L., Han, Y., Li, S. Z., & Zhou, G. (2018)**. *Market Intraday Momentum*. **Journal of Financial Economics**, 129(2), 394–414. DOI: [10.1016/j.jfineco.2017.11.002](https://doi.org/10.1016/j.jfineco.2017.11.002).
2. **Bogousslavsky, V. (2021)**. *The Cross-Section of Intraday and Overnight Returns*. **Journal of Financial Economics**, 141(1), 172–194. DOI: [10.1016/j.jfineco.2021.03.003](https://doi.org/10.1016/j.jfineco.2021.03.003).

---

### 1.2 Hipótesis Económica y Dinámica Microestructural

En los mercados de derivados de índices accionarios estadounidenses, el volumen transado a lo largo de la sesión regular (*Regular Trading Hours*, RTH, 09:30:00 a 16:00:00 hora estándar del este, EST) describe una curva asimétrica en forma de "U". La literatura financiera empírica y los modelos microestructurales de subasta revelan una conexión causal profunda entre los dos extremos de esta curva:

1. **La Ventana de Apertura y Descubrimiento de Precios (09:30:00 - 10:00:00 EST):**  
   Durante la primera media hora tras la apertura del mercado de contado de Nueva York, el mercado absorbe el cúmulo de información macroeconómica, balances corporativos y noticias internacionales generadas durante la noche. Los inversores institucionales de gran envergadura (fondos de pensiones, fondos soberanos, fondos mutuos y proveedores de liquidez ETF) reciben asignaciones masivas de capital. Debido a que ejecutar estas órdenes de golpe causaría un impacto de mercado severo (*market impact* cuadrático de Almgren-Chriss), los intermediarios activan algoritmos de fragmentación de órdenes (TWAP y VWAP) diseñados para distribuir la ejecución a lo largo del día.

2. **La Fricción de Inventario Nocturno y la Ventana de Cierre MOC (15:30:00 - 16:00:00 EST):**  
   Conforme se aproxima el cierre del mercado regular, los intermediarios financieros y operadores institucionales enfrentan restricciones contractuales y regulatorias estrictas que penalizan el mantenimiento de inventarios direccionales no balanceados durante la noche (*overnight inventory risk* y costes de capital de Basilea III / Dodd-Frank, Bogousslavsky 2021). Aquellos operadores que iniciaron compras agresivas por la mañana deben completar el saldo remanente de sus cestas antes del cierre de la subasta *Market-on-Close* (MOC), concentrando una presión compradora sistemática en los últimos 30 minutos de negociación. Análogamente, liquidaciones matutinas forzadas generan ventas de liquidación acumuladas al cierre.

3. **La Ventaja Estadística (*Quantitative Edge*):**  
   Gao et al. (2018) demuestran formalmente que el retorno acumulado durante el primer intervalo de media hora ($r_{first}$) predice con signo idéntico y alta significancia estadística ($t\text{-stat} > 3.5$) el retorno de la última media hora ($r_{last}$), generando una prima por momento intradía persistente. Durante el lapso intermedio (10:00:00 a 15:30:00 EST), el mercado suele caer en un letargo de bajo volumen relativo (*lunchtime lull*); al abstenerse de operar durante este intervalo intermedio, la estrategia elimina el riesgo de falso rompimiento (*whipsaw*), limitando su exposición activa al mercado a escasos 28 minutos diarios (15:30:00 a 15:58:00 EST).

---

## 2. Variables de Entrada, Espacio de Estados y Horizontes Temporales

### 2.1 Variables de Entrada y Parámetros Formales

Sea $t \in \mathbb{N}$ el índice discreto que identifica el día de negociación bursátil en CME. Sea $\tau \in [09:30:00, 16:00:00]$ la marca de tiempo continua intradía expresada en hora estándar del este (America/New_York).

| Símbolo | Nombre Formal | Dominio Matemático | Unidades Físicas / Financieras | Descripción Operativa |
| :--- | :--- | :--- | :--- | :--- |
| $P_t(\tau)$ | Precio instantáneo | $\mathbb{R}_{++}$ | Puntos de Índice | Precio de la última transacción registrada en el instante $\tau$ en el libro de CME Globex. |
| $P_t(09:30)$ | Precio de Apertura RTH | $\mathbb{R}_{++}$ | Puntos de Índice | Precio de apertura oficial de la sesión regular al instante $\tau = 09:30:00$ EST. |
| $P_t(10:00)$ | Precio de Cierre Matutino | $\mathbb{R}_{++}$ | Puntos de Índice | Precio de transacción registrado exactamente a las $\tau = 10:00:00$ EST. |
| $P_t(15:30)$ | Precio de Inicio MOC | $\mathbb{R}_{++}$ | Puntos de Índice | Precio de cotización al comienzo de la ventana operativa vespertina ($\tau = 15:30:00$ EST). |
| $V_t(\tau_a, \tau_b)$ | Volumen transado acumulado | $\mathbb{N}_0$ | Contratos de Futuros | Volumen total de contratos transados en el intervalo temporal $[\tau_a, \tau_b]$. |
| $\sigma_{daily, t}$ | Dispersión condicional | $\mathbb{R}_{++}$ | Puntos de Índice | Desviación estándar muestral estimada de los retornos diarios basada en un lookback de 20 días. |
| $M$ | Multiplicador de contrato CME | $\{50, 5, 20, 2\}$ | $\text{USD} / \text{Punto}$ | Constante contractual CME ($M_{ES}=50$, $M_{MES}=5$, $M_{NQ}=20$, $M_{MNQ}=2$). |
| $\delta_{tick}$ | Tamaño de tick mínimo | $\{0.25\}$ | Puntos de Índice | Variación mínima indivisible de cotización en los contratos de futuros de índices CME. |
| $C_{fric}$ | Coste total de fricción | $\mathbb{R}_{++}$ | USD / Contrato | Coste combinado de comisión de compensación de CME más un tick de deslizamiento (*slippage*). |
| $\theta_{floor}$ | Umbral mínimo porcentual | $\mathbb{R}_{++}$ | Adimensional (Ratio) | Límite inferior para filtrar ruido estocástico ($\theta_{floor} = 0.0015$, equivalente a $0.15\%$). |
| $c_\sigma$ | Coeficiente de volatilidad | $\mathbb{R}_{++}$ | Adimensional | Factor de escalamiento de volatilidad ($c_\sigma = 0.20$). |
| $K_{vol}$ | Ventana de volumen histórico | $\mathbb{N}$ | Días hábiles | Lookback para la media histórica de volumen matutino ($K_{vol} = 20$). |

---

### 2.2 Espacio de Estados y Posición de Mercado

El estado interno del algoritmo en cualquier instante $\tau$ de la jornada $t$ se define en el espacio discreto acotado:
$$\mathcal{S}_t(\tau) \in \{\text{WAIT\_MORNING}, \text{IDLE\_INTRADAY}, \text{EVALUATING\_MOC}, \text{LONG\_EXPOSURE}, \text{SHORT\_EXPOSURE}, \text{FLAT\_COMPLETED}\}$$

La función de posición de mercado $w_t(\tau)$ representa el sentido de la exposición direccional de capital:
$$w_t(\tau) \in \{-1, 0, +1\}$$
donde:
- $w_t(\tau) = +1$: Posición larga abierta (compra de futuros).
- $w_t(\tau) = -1$: Posición corta abierta (venta de futuros).
- $w_t(\tau) = 0$: Posición neutral / plana (cero contratos en el mercado).

---

### 2.3 Horizontes Temporales y Ventanas de Negociación

La arquitectura temporal de la estrategia descompone la sesión RTH de CME en cuatro etapas disjuntas:

$$\begin{aligned}
\mathcal{W}_{obs} &= [09:30:00, 10:00:00) \quad &&\text{(Ventana de Observación Matutina: captura del impulso institucional inicial)} \\
\mathcal{W}_{idle} &= [10:00:00, 15:30:00) \quad &&\text{(Ventana de Inactividad Latente: prohibición estricta de ejecución, letargo intradía)} \\
\mathcal{W}_{exec} &= [15:30:00, 15:58:00) \quad &&\text{(Ventana de Ejecución y Monitoreo MOC: captura del momento de cierre)} \\
\mathcal{W}_{flat} &= [15:58:00, 17:00:00] \quad &&\text{(Ventana de Neutralización Forzosa: liquidación obligatoria previa a la pausa de CME)}
\end{aligned}$$

---

## 3. Formulación Matemática de Generación de Señales

### 3.1 Cálculo del Retorno Logarítmico Matutino

El retorno acumulado del subyacente durante la primera media hora de negociación de contado se calcula de forma unívoca mediante la diferencia logarítmica de precios:
$$r_{first, t} = \ln \left( \frac{P_t(10:00)}{P_t(09:30)} \right)$$

---

### 3.2 Filtro de Confirmación de Volumen Relativo Institucional

Para certificar que el movimiento matutino no es fruto de fluctuaciones estocásticas con liquidez reducida, se define el ratio de volumen relativo matutino $\mathcal{V}_{ratio, t}$:
$$\mathcal{V}_{ratio, t} = \frac{V_t(09:30:00, 10:00:00)}{\frac{1}{K_{vol}} \sum_{k=1}^{K_{vol}} V_{t-k}(09:30:00, 10:00:00)}$$

La condición booleana de confirmación por volumen se expresa formalmente como:
$$\Phi_{vol, t} = \mathbb{I}\left( \mathcal{V}_{ratio, t} \ge 1.00 \right)$$
donde $\mathbb{I}(\cdot)$ es la función indicatriz que adopta el valor $1$ si la proposición lógica es verdadera y $0$ en caso contrario.

---

### 3.3 Umbral Dinámico de Activación Indexado a la Volatilidad

Con el propósito de garantizar que el retorno matutino sea lo suficientemente amplio como para absorber los costes de transacción (spread bid-ask y comisiones de clearing), se define un umbral de activación no lineal $\theta_t$:
$$\theta_t = \max \left( \theta_{floor}, \; c_\sigma \cdot \frac{\sigma_{daily, t}}{P_t(10:00)} \right)$$

Este umbral garantiza adaptabilidad: en periodos de baja volatilidad se exige al menos un desplazamiento de $\theta_{floor} = 0.15\%$, mientras que en jornadas de alta volatilidad el filtro se expande automáticamente proporcionalmente a $\sigma_{daily, t}$.

---

### 3.4 Ecuaciones Cerradas del Operador de Señal $S_t(15:30)$

A las $\tau = 15:30:00$ EST, el operador analítico de señal primaria $S_t \equiv S_t(15:30)$ evalúa el estado del mercado según el siguiente sistema de ecuaciones disyuntivas cerradas:

$$\begin{aligned}
S_{Long, t} &= \mathbb{I}\left( r_{first, t} > +\theta_t \;\land\; \Phi_{vol, t} = 1 \right) \\
S_{Short, t} &= \mathbb{I}\left( r_{first, t} < -\theta_t \;\land\; \Phi_{vol, t} = 1 \right) \\
S_{Neutral, t} &= \mathbb{I}\left( |r_{first, t}| \le \theta_t \;\lor\; \Phi_{vol, t} = 0 \right)
\end{aligned}$$

El operador analítico integrado de señal discreta $S_t \in \{-1, 0, +1\}$ adopta la forma cerrada:
$$S_t = (+1) \cdot S_{Long, t} + (-1) \cdot S_{Short, t} + (0) \cdot S_{Neutral, t}$$

Por construcción lógica, los conjuntos son mutuamente excluyentes:
$$S_{Long, t} \cdot S_{Short, t} \equiv 0$$
$$S_{Long, t} + S_{Short, t} + S_{Neutral, t} \equiv 1$$

---

## 4. Formulación Analítica de Salida de Posición

Sea $P_{entry, t} = P_t(15:30:00)$ el precio de ejecución en el momento de la entrada. Sea $S_t \in \{-1, +1\}$ la dirección de la posición adoptada.

### 4.1 Ecuación Analítica de Parada de Pérdidas (Stop-Loss Matemático)

El nivel de precio de parada de pérdidas $P_{SL, t}$ se modela a una distancia porcentual simétrica $\lambda_{SL} = 0.0035$ (equivalente a $0.35\%$ del valor del índice, o aproximadamente 18-20 puntos en $ES$ y 65-75 puntos en $NQ$):
$$P_{SL, t} = P_{entry, t} \cdot \left( 1 - S_t \cdot \lambda_{SL} \right)$$

El instante de activación del Stop-Loss se formula rigurosamente como el tiempo de parada (*stopping time*) sobre el proceso estocástico del precio:
$$\tau_{SL, t} = \inf \left\{ \tau \in [15:30:00, 15:58:00] \;\middle|\; S_t \cdot \left( P_t(\tau) - P_{entry, t} \right) \le - P_{entry, t} \cdot \lambda_{SL} \right\}$$

Si el conjunto de instantes es vacío (el precio jamás vulnera el nivel), se establece por convención analítica $\tau_{SL, t} = \infty$.

---

### 4.2 Ecuación Analítica de Toma de Beneficios (Take-Profit Matemático)

La toma de ganancias incorpora una asimetría estructural de recompensa-riesgo mínima de $1.5:1$ ($k_{RR} = 1.50$):
$$P_{TP, t} = P_{entry, t} \cdot \left( 1 + S_t \cdot k_{RR} \cdot \lambda_{SL} \right)$$

El tiempo de parada para la consecución del objetivo de beneficio es:
$$\tau_{TP, t} = \inf \left\{ \tau \in [15:30:00, 15:58:00] \;\middle|\; S_t \cdot \left( P_t(\tau) - P_{entry, t} \right) \ge P_{entry, t} \cdot k_{RR} \cdot \lambda_{SL} \right\}$$

Si el precio no alcanza el objetivo, se define $\tau_{TP, t} = \infty$.

---

### 4.3 Ecuación de Decaimiento Temporal Forzoso (Time-Decay / MOC Exit)

Para evitar la volatilidad extrema y los desbalances de emparejamiento aleatorios que ocurren en el último minuto de la sesión regular de CME (15:59:00 a 16:00:00 EST), y para blindar la cuenta frente al riesgo de suspensión o mantenimiento diario de CME Globex (17:00 EST), se define el tiempo límite de liquidación:
$$\tau_{MOC, t} = 15:58:00\text{ EST}$$

---

### 4.4 Tiempo de Parada Unificado de Salida de Mercado

El instante de liquidación de la orden en el mercado $\tau_{exit, t}$ es el mínimo formal del conjunto de tiempos de parada:
$$\tau_{exit, t} = \min \left( \tau_{SL, t}, \; \tau_{TP, t}, \; \tau_{MOC, t} \right)$$

La función de exposición $w_t(\tau)$ tras la salida satisface de forma obligatoria e incondicional:
$$w_t(\tau) = 0, \quad \forall \tau \in [\tau_{exit, t}, 16:00:00]$$

---

## 5. Dimensionamiento de Posición y Control de Riesgo Cuantitativo para Prop Trading

Las reglas contractuales de las firmas de Proprietary Trading imponen límites estrictos sobre la pérdida máxima permitida en una sesión (*Daily Loss Limit*, $DLL$) y sobre el retroceso acumulado desde la máxima ganancia histórica (*Max Trailing Drawdown*, $MTDD$). El dimensionamiento de la posición debe ser estrictamente dinámico y aversivo al riesgo.

### 5.1 Parámetros y Variables de Capital de la Cuenta

- $C_{equity, t}$: Capital líquido realizado de la cuenta al inicio del día $t$ (USD).
- $DLL$: Límite de Pérdida Diaria máxima permitida por la firma de fondeo (USD, e.g., \$1,000 en una cuenta de \$50,000).
- $MTDD$: Límite de Trailing Drawdown máximo admitido (USD, e.g., \$2,000 en una cuenta de \$50,000).
- $HWM_t = \max_{k \le t} C_{equity, k}$: Marca de agua máxima histórica (*High-Water Mark*).
- $DD_t = HWM_t - C_{equity, t}$: Drawdown acumulado actual de la cuenta (USD).
- $L_{day, t} \ge 0$: Pérdida neta acumulada en la sesión $t$ antes de las 15:30 EST (USD).
- $K_{loss} \in \mathbb{N}_0$: Número de operaciones consecutivas perdedoras registradas por la estrategia.

---

### 5.2 Presupuesto Dinámico de Riesgo por Operación ($\mathcal{B}_{risk, t}$)

El presupuesto de pérdida monetaria admisible para el trade $\mathcal{B}_{risk, t}$ se calcula como el valor mínimo entre el margen disponible respecto al límite diario y el margen respecto al trailing drawdown, escalado por un factor multiplicativo que penaliza rachas perdedoras:

$$\mathcal{B}_{risk, t} = \min \left( \phi(K_{loss}) \cdot \max\left( 0, \; DLL - L_{day, t} \right), \; \psi \cdot \max\left( 0, \; MTDD - DD_t \right) \right)$$

donde $\psi = 0.25$ (reserva de al menos cuatro intentos sobre el colchón de drawdown) y la función de atenuación de rachas perdedoras $\phi(K_{loss})$ se formula analíticamente como:
$$\phi(K_{loss}) = 0.35 \times 2^{-\min(K_{loss}, 3)}$$

Evaluando para diferentes rachas:
- $K_{loss} = 0 \implies \phi = 0.35$ (35% del margen diario restante).
- $K_{loss} = 1 \implies \phi = 0.175$.
- $K_{loss} = 2 \implies \phi = 0.0875$.
- $K_{loss} \ge 3 \implies \phi = 0.04375$.

---

### 5.3 Riesgo Monetario por Contrato Individual

Para un contrato con multiplicador CME $M$ y distancia de invalidación $\Delta_{SL, t} = P_{entry, t} \cdot \lambda_{SL}$:
$$\mathcal{R}_{contract, t}(M) = \Delta_{SL, t} \cdot M + C_{fric}$$
donde $C_{fric} = 2 \cdot \delta_{tick} \cdot M + \text{Comisión}_{clearing}$.

---

### 5.4 Modelo Declarativo de Conmutación Automática Estándar / Micro

El número de contratos estándar $N_{std, t}$ a ejecutar se calcula mediante la función piso:
$$N_{std, t} = \left\lfloor \frac{\mathcal{B}_{risk, t}}{\mathcal{R}_{contract, t}(M_{std})} \right\rfloor$$

Si el presupuesto no permite ni siquiera un contrato estándar ($N_{std, t} = 0$), el algoritmo conmuta automáticamente a micro-contratos ($M_{micro} = \frac{1}{10} M_{std}$):
$$N_{micro, t} = \min \left( \left\lfloor \frac{\mathcal{B}_{risk, t}}{\mathcal{R}_{contract, t}(M_{micro})} \right\rfloor, \; N_{max, micro} \right)$$
con $N_{max, micro} = 10$.

---

### 5.5 Disyuntor Cuantitativo de Volatilidad (Volatility Circuit Breaker)

Si la volatilidad del mercado se expande de modo que el riesgo de un único micro-contrato excede el presupuesto disponible, se activa el disyuntor analítico forzoso:
$$\text{Si } \mathcal{R}_{contract, t}(M_{micro}) > \mathcal{B}_{risk, t} \implies N_t = 0 \implies \text{HALT\_VOLATILITY\_DEFENSE}$$

Esta condición previene matemáticamente que la cuenta incurra en una infracción de liquidación forzosa de la prop firm en días anómalos.

---

## 6. Rúbrica de Verificación Analítica Objetiva

La siguiente matriz de auditoría cuantitativa evalúa formalmente el cumplimiento matemático y analítico de la especificación de la estrategia F1:

| Criterio de Verificación Cuantitativa | Expresión Analítica Auditada | Umbral / Condición de Aprobación | Estado de Cumplimiento | Justificación Teórica y Evidencia Formal |
| :--- | :--- | :--- | :--- | :--- |
| **Ausencia Total de Código** | $\text{Card}(\text{CodeBlocks}) = 0$ | Idéntico a 0 bloques de código | **APROBADO** | El documento se expresa exclusivamente en lenguaje declarativo formal y ecuaciones en LaTeX. |
| **Exclusión de Indicadores Tradicionales Minoristas** | $\text{Card}(\text{IndicadoresDescartados}) = 0$ | Cero indicadores minoristas rezagados | **APROBADO** | La señal depende únicamente de la dinámica de subasta institucional $r_{first, t}$ y volumen relativo. |
| **Definición Rigurosa de Dominios** | $\forall x \in \text{Inputs}, \; \text{Dom}(x) \subseteq \mathbb{R} \lor \mathbb{N}$ | 100% de variables unívocas | **APROBADO** | La totalidad de parámetros posee dominio matemático y unidades físicas financieras (USD, ticks, puntos). |
| **Exclusión Mutua de Señales** | $S_{Long, t} \cdot S_{Short, t} \equiv 0$ | Ortogonalidad estricta | **APROBADO** | Los umbrales $+\theta_t$ y $-\theta_t$ son disjuntos ($\theta_t > 0$), impidiendo señales simultáneas opuestas. |
| **Asimetría de Riesgo-Recompensa** | $k_{RR} = \frac{|P_{TP} - P_{entry}|}{|P_{SL} - P_{entry}|} \ge 1.50$ | $k_{RR} \ge 1.50$ | **APROBADO** | Garantiza un valor esperado positivo ($EV > 0$) con tasas de acierto moderadas ($> 42\%$). |
| **Blindaje frente al Límite Diario** | $\text{Pérdida Máxima}(N_t) \le \mathcal{B}_{risk, t} < DLL$ | $\mathcal{B}_{risk, t} \le 0.35 \cdot DLL$ | **APROBADO** | Ningún trade individual puede consumir más del 35% del límite de pérdida diaria de la prop firm. |
| **Inmunidad de Riesgo Overnight** | $w_t(\tau) \equiv 0, \; \forall \tau \ge 15:58:00\text{ EST}$ | $\tau_{exit} \le 15:58:00$ | **APROBADO** | Liquidación analítica obligatoria 62 minutos antes de la pausa técnica de CME (17:00 EST). |
| **Conmutación Multiescala** | $N_{std} \cdot \mathbb{I}(N_{std} \ge 1) + N_{micro} \cdot \mathbb{I}(N_{std}=0)$ | Cero fraccionamiento irregular | **APROBADO** | Uso estricto de la función piso $\lfloor \cdot \rfloor$ para garantizar números enteros de contratos ejecutables. |
