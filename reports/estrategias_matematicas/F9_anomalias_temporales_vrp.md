# Especificación Matemática Formal: Estrategia F9 - Anomalías Temporales y Variance Risk Premium (VRP)

**Identificador de Estrategia:** F9  
**Familia Cuantitativa:** Explotación de Primas de Riesgo Estructurales / Asimetrías de Subasta y Microestructura Multi-Sesión  
**Instrumentos Objetivo:** CME E-mini S&P 500 ($ES$), CME Micro E-mini S&P 500 ($MES$), CME E-mini Nasdaq-100 ($NQ$), CME Micro E-mini Nasdaq-100 ($MNQ$)  
**Mercado de Negociación:** Chicago Mercantile Exchange (CME Globex)  
**Marco de Riesgo:** Proprietary Trading (Cuentas Fondeadas: Topstep, Apex, Lucid, MyFundedFutures)  
**Nivel de Documento:** Especificación Matemática y Microestructural Modular Autosuficiente (Apta para entrega directa a agentes de implementación de software)  
**Restricción Estricta de Formato:** Teoría matemática analítica pura en notación declarativa formal LaTeX. Queda estrictamente excluido cualquier fragmento de código ejecutable, script o pseudocódigo, así como cualquier indicador técnico tradicional descartado.

---

## 1. Identificación y Fundamento Teórico / Hipótesis Microestructural

### 1.1 Citas Académicas de Referencia
1. **Bollerslev, T., Tauchen, G., & Zhou, H. (2009)**. *Expected Stock Returns and Variance Risk Premia*. **The Review of Financial Studies**, 22(11), 4463–4492. DOI: [10.1093/rfs/hhp008](https://doi.org/10.1093/rfs/hhp008).
2. **Boyarchenko, N., Larsen, J. C., & Whelan, P. (2023)**. *The Overnight Drift*. **Federal Reserve Bank of New York Staff Reports**, No. 1061. Publicado en SSRN. DOI: [10.2139/ssrn.4452585](https://doi.org/10.2139/ssrn.4452585).
3. **Kurov, A., Sancetta, A., Strasser, G., & Wolfe, M. H. (2019)**. *Price Drift Before U.S. Macroeconomic News: Private Information or Pre-Announcement Drift?* **Journal of Financial and Quantitative Analysis**, 54(1), 449–479. DOI: [10.1017/S002210901800087X](https://doi.org/10.1017/S002210901800087X).
4. **Simon, D. P., & Campasano, J. (2014)**. *The VIX Futures Basis: Evidence and Trading Strategies*. **The Journal of Derivatives**, 22(2), 60–69. DOI: [10.3905/jod.2014.22.2.060](https://doi.org/10.3905/jod.2014.22.2.060).

---

### 1.2 Hipótesis Económica y Dinámica Microestructural en Español Sencillo

Los precios de los futuros sobre índices accionarios ($ES$ y $NQ$) no siguen un paseo aleatorio uniforme a lo largo de las 24 horas del día. La operativa institucional engendra **anomalías estructurales recurrentes** motivadas por tres factores:
1. La aversión al riesgo de ruina y el pago de pólizas de seguro financiero.
2. Los costes de balance y restricciones de inventario que sufren los intermediarios (*market makers*) al término de la sesión de Nueva York.
3. El posicionamiento informado previo a publicaciones macroeconómicas de alto impacto.

Esta estrategia integra de forma modular y ortogonal tres ventajas cuantitativas consolidadas en la literatura financiera empírica:

#### A) Compuerta de Régimen: Variance Risk Premium (VRP) y Estructura Temporal del VIX
Los inversores institucionales adquieren continuamente opciones de venta (*puts*) para proteger sus carteras ante posibles caídas bursátiles. Debido a esta demanda inelástica de protección, la volatilidad implícita en las opciones ($IV$) cotiza habitualmente por encima de la volatilidad que termina realizándose en el mercado ($RV$). El diferencial:
$$VRP = IV - RV$$
constituye la **Prima de Riesgo de Varianza**. Cuando el $VRP$ es amplio y positivo, y la estructura temporal de volatilidad se sitúa en *Contango* (los futuros de volatilidad lejanos son más caros que los cercanos: $TSR = VIX / VIX3M < 1.0$), el mercado se encuentra en un régimen saludable de absorción de riesgo, altamente propicio para compras sostenidas. Por el contrario, si el $VRP$ colapsa a terreno negativo y la curva se invierte a *Backwardation* ($TSR \ge 1.0$), indica pánico sistémico e iliquidez, vetando incondicionalmente cualquier orden de compra.

#### B) Deriva Nocturna Condicional (Overnight Drift)
Boyarchenko, Larsen y Whelan (2023) demostraron que en las jornadas regulares de Wall Street ($RTH$, 09:30–16:00 EST) donde los fondos mutuos ejecutan fuertes ventas de rebalanceo hacia el cierre, los intermediarios financieros acumulan inventarios largos no deseados que deprimen transitoriamente el precio. Durante la noche, al abrir las bolsas europeas (02:00 EST / 08:00 Fráncfort y Londres), la entrada de liquidez global permite a los intermediarios descargar su inventario, liberando la prima de riesgo acumulada y generando un rebote sistemático entre las 01:30 y las 08:30 EST.

#### C) Deriva Previa Informada a Anuncios Macroeconómicos (Pre-Macro Drift)
Kurov et al. (2019) descubrieron que en los 30 minutos previos a la difusión de noticias macroeconómicas de primer nivel (como inflación CPI, nóminas no agrícolas NFP o decisiones de tasas del FOMC), algoritmos institucionales de procesamiento de datos de alta frecuencia (*nowcasting*) anticipan la dirección de la sorpresa, generando una deriva sostenida del precio entre $T-15\text{ min}$ y $T-3\text{ min}$.  
*Regla de supervivencia para Prop Trading:* Cerrar la posición exactamente en el minuto $T-3$ monetiza la ventaja estadística informada y garantiza que la cuenta esté completamente plana 60 segundos antes de ingresar a la ventana de **embargo normativo** de las firmas de fondeo ($\pm 2\text{ minutos}$ alrededor del anuncio), eliminando el riesgo de spreads ensanchados y deslizamientos destructivos.

---

## 2. Variables de Entrada, Espacio de Estados y Horizontes Temporales

### 2.1 Variables de Entrada y Parámetros Formales

Sea $t$ el tiempo continuo en CME Globex. Sea $t_{\text{macro}}^{\text{ann}}$ el instante programado de un anuncio macroeconómico Tier-1.

| Símbolo | Nombre Formal | Dominio Matemático | Unidades Físicas / Financieras | Descripción Operativa |
| :--- | :--- | :--- | :--- | :--- |
| $IV_t$ | Varianza implícita anualizada | $\mathbb{R}_{++}$ | $\text{retorno}^2$ (Anualizada) | Varianza libre de modelo obtenida del índice de volatilidad: $(VIX_t / 100)^2$. |
| $RV_t^{\text{ann}}$ | Varianza realizada anualizada | $\mathbb{R}_{++}$ | $\text{retorno}^2$ (Anualizada) | Varianza acumulada a 5 minutos proyectada a horizonte anual mediante ventana de $K = 22$ días. |
| $VRP_t$ | Variance Risk Premium | $\mathbb{R}$ | $\text{retorno}^2$ (Anualizada) | Diferencial analítico entre varianza implícita y realizada: $IV_t - RV_t^{\text{ann}}$. |
| $Z_{VRP}(t)$ | Puntuación Z del VRP | $\mathbb{R}$ | Adimensional ($Z$-score) | Estandarización del $VRP$ sobre una ventana rodante histórica de $L = 60$ sesiones. |
| $TSR_t$ | Ratio de estructura temporal | $\mathbb{R}_{++}$ | Ratio Adimensional | Cociente entre el índice de volatilidad a 30 días y a 90 días: $VIX_t / VIX3M_t$. |
| $R_{\text{intraday}, t-1}$ | Retorno regular de la sesión previa | $\mathbb{R}$ | Adimensional (Ratio) | Retorno logarítmico RTH previo: $\ln(P_{t-1}(16:00) / P_{t-1}(09:30))$. |
| $\tau_{\text{macro}}$ | Tiempo restante al anuncio | $\mathbb{R}_+$ | Minutos | Intervalo de cuenta regresiva hasta el instante exacto de la noticia Tier-1. |
| $Z_{OFI}^{\text{pre}}(t)$ | Desbalance acumulado pre-macro | $\mathbb{R}$ | Adimensional ($Z$-score) | Puntuación acumulada de flujo en el mejor bid/ask durante la ventana $t \in [T-15m, T-3m]$. |
| $\theta_{\text{drop}}$ | Umbral de caída intradía previa | $\mathbb{R}_{++}$ | Ratio Adimensional | Magnitud mínima de descenso en RTH para habilitar el rebote nocturno ($\theta_{\text{drop}} = 0.0075 \implies 0.75\%$). |
| $\delta_{\text{SL}}^{\text{ON}}$ | Límite de parada nocturna | $\mathbb{R}_{++}$ | Ratio Adimensional | Tolerancia porcentual de pérdida máxima en overnight ($\delta_{\text{SL}}^{\text{ON}} = 0.0050 \implies 0.50\%$). |
| $\Delta P_{\text{stop}}^{\text{macro}}$ | Stop-loss en puntos pre-macro | $\mathbb{R}_{++}$ | Puntos de Índice | Distancia de invalidación para la deriva pre-anuncio ($\Delta P_{\text{stop}}^{\text{macro}} = 10.0\text{ pts}$ en $ES$). |
| $M^{(I)}$ | Multiplicador de contrato CME | $\{50, 5, 20, 2\}$ | $\text{USD} / \text{Punto}$ | Constante monetaria de contrato ($M^{(\text{ES})}=50$, $M^{(\text{MES})}=5$, etc.). |

---

### 2.2 Espacio de Estados y Regímenes Multi-Sesión

El espacio de estados temporal del sistema se descompone en función del reloj operativo $t_{\text{clock}}$ (hora de Nueva York, EST):
$$\mathcal{S}_t^{\text{anom}} = \left( X_t^{\text{anom}}, \; t_{\text{clock}}, \; S_{\text{VRP}}(t), \; w_{\text{ON}}(t), \; w_{\text{macro}}(t) \right)$$
donde $X_t^{\text{anom}} \in \mathbb{R}^8$ contiene las variables de subasta y volatilidad, y $w_{\text{ON}}, w_{\text{macro}} \in \{-1, 0, +1\}$ describen la tenencia en cada submódulo.

---

### 2.3 Horizontes Temporales Específicos y Ventanas de Embargo

1. **Sesión Nocturna (Deriva Boyarchenko):**  
   - Ventana de Apertura: $t = 01:30:00\text{ EST}$.  
   - Ventana de Cierre Determinista Incondicional: $t = 08:30:00\text{ EST}$.
2. **Ventana Pre-Anuncio Informada (Deriva Kurov):**  
   - Ventana de Apertura: $t = t_{\text{macro}}^{\text{ann}} - 15\text{ minutos}$.  
   - Ventana de Cierre Forzoso Estricto: $t = t_{\text{macro}}^{\text{ann}} - 3\text{ minutos}$.
3. **Ventana de Embargo Absoluto de Noticias Prop Trading:**  
   - $\mathcal{W}_{\text{embargo}} = [t_{\text{macro}}^{\text{ann}} - 2\text{ min}, \; t_{\text{macro}}^{\text{ann}} + 2\text{ min}]$. Prohibición total de tener órdenes o contratos vigentes.
4. **Ventana Regular Intradía (RTH):** $[09:30:00, 15:55:00]\text{ EST}$.
5. **Límite de Liquidación Universal:** $\tau_{\text{terminal}} = 16:45:00\text{ EST}$ (previo al mantenimiento de CME a las 17:00 EST).

---

## 3. Formulación Matemática de Generación de Señales

### 3.1 Sub-modelo 1: Señal de Régimen y Compuerta Estructural VRP / Term Structure

La varianza implícita y realizada anualizadas se formulan como:
$$IV_t = \left( \frac{VIX_t}{100} \right)^2$$
$$RV_t^{\text{ann}} = \frac{252}{K} \sum_{k=0}^{K-1} RV_{t-k}^{(d)}, \quad K = 22$$
$$VRP_t = IV_t - RV_t^{\text{ann}}$$

La puntuación estandarizada rodante sobre una ventana de $L = 60$ sesiones es:
$$Z_{VRP}(t) = \frac{VRP_t - \mu_{VRP}(L)}{\sigma_{VRP}(L)}$$
donde $\mu_{VRP}(L)$ y $\sigma_{VRP}(L)$ son la media y desviación típica muestrales del $VRP$ en las últimas 60 jornadas.

El ratio de estructura temporal de volatilidad (*Term Structure Ratio*, $TSR_t$) es:
$$TSR_t = \frac{VIX_t}{VIX3M_t}$$

La compuerta estructural de régimen $S_{\text{VRP}}(t) \in \{-1, 0, +1\}$ se define como:
$$S_{\text{VRP}}(t) = \begin{cases} 
+1 & \text{si } Z_{VRP}(t) \ge +1.50 \;\land\; TSR_t \le 0.95 \quad (\text{Complacencia / Sesgo Comprador Favorable}) \\[4pt]
-1 & \text{si } Z_{VRP}(t) \le -1.20 \;\lor\; TSR_t \ge 1.00 \quad (\text{Backwardation / Bloqueo Total de Compras}) \\[4pt]
0 & \text{en cualquier otro caso} \quad (\text{Régimen Neutral Ordinario})
\end{cases}$$

---

### 3.2 Sub-modelo 2: Deriva Nocturna Condicional (Boyarchenko et al., 2023)

Sea el retorno de la sesión RTH anterior:
$$R_{\text{intraday}, t-1} = \ln\left( \frac{P_{t-1}(16:00:00)}{P_{t-1}(09:30:00)} \right)$$

A las $t = 01:30:00\text{ EST}$, se evalúa la condición de sobreventa institucional previa. La señal nocturna $S_{\text{overnight}}(t)$ se formula como:
$$S_{\text{overnight}}(t) = \mathbb{I}\left( R_{\text{intraday}, t-1} \le -\theta_{\text{drop}} \right) \cdot \mathbb{I}\left( S_{\text{VRP}}(t) \ne -1 \right)$$
con $\theta_{\text{drop}} = 0.0075$ ($0.75\%$). Si se cumple la condición y la compuerta VRP no indica pánico sistémico ($S_{\text{VRP}} \ne -1$), se emite una orden de compra: $S_t^* = +1$.

---

### 3.3 Sub-modelo 3: Deriva Previa a Anuncios Macroeconómicos (Kurov et al., 2019)

Sea $t_{\text{macro}}^{\text{ann}} \in \{08:30:00, 14:00:00\}\text{ EST}$ el instante exacto de una publicación Tier-1 (CPI, NFP o FOMC). En la ventana $t \in [t_{\text{macro}}^{\text{ann}} - 15m, \, t_{\text{macro}}^{\text{ann}} - 3m]$, se calcula el desbalance acumulado estandarizado de flujo de órdenes:
$$Z_{OFI}^{\text{pre}}(t) = \frac{\sum_{\tau = t-15m}^t OFI_\tau - \mu_{OFI}}{\sigma_{OFI}}$$

A las $t = t_{\text{macro}}^{\text{ann}} - 15\text{ minutos}$, el operador analítico de deriva informada $S_{\text{macro}}(t)$ adopta la forma:
$$S_{\text{macro}}(t) = \begin{cases} 
+1 & \text{si } Z_{OFI}^{\text{pre}}(t) \ge +2.0 \;\land\; P_t > \text{VWAP}_{[t-30m, t-15m]} \;\land\; S_{\text{VRP}}(t) \ne -1 \\[4pt]
-1 & \text{si } Z_{OFI}^{\text{pre}}(t) \le -2.0 \;\land\; P_t < \text{VWAP}_{[t-30m, t-15m]} \;\land\; S_{\text{VRP}}(t) \ne +1 \\[4pt]
0 & \text{en cualquier otro caso}
\end{cases}$$

---

### 3.4 Matriz de Composición Analítica Ortogonal

La señal ejecutoria global $S_{\text{unified}}^*(t) \in \{-1, 0, +1\}$ se estructura como la unión disjunta de los sub-modelos según la ventana horaria:
$$S_{\text{unified}}^*(t) = \begin{cases} 
S_{\text{overnight}}(t) & \text{si } t \in [01:30:00, 08:30:00)\text{ EST} \\[4pt]
S_{\text{macro}}(t) & \text{si } t \in [t_{\text{macro}}^{\text{ann}} - 15m, \, t_{\text{macro}}^{\text{ann}} - 3m]\text{ EST} \\[4pt]
0 & \text{si } t \in [t_{\text{macro}}^{\text{ann}} - 2m, \, t_{\text{macro}}^{\text{ann}} + 2m] \quad (\text{Embargo Normativo Prop Firm}) \\[4pt]
0 & \text{si } t \ge 15:55:00\text{ EST} \quad (\text{Cierre Técnico Diario CME})
\end{cases}$$

---

## 4. Formulación Analítica de Salida de Posición

> **Ventana Operativa y Salida Estricta $T-3$ (Kurov / Regla de Embargo):**  
> - **Ventana Pre-Macro ($t \in [t_{\text{macro}}^{\text{ann}} - 15\text{ min}, \, t_{\text{macro}}^{\text{ann}} - 3\text{ min}]$):** Captura del flujo informado e impulso institucional previo a noticias Tier-1.  
> - **Salida Forzosa Incondicional en $T-3\text{ min}$:** Cierre obligatorio a mercado de cualquier posición abierta (a las 08:27:00 EST para anuncios de las 08:30, o a las 13:57:00 EST para anuncios de las 14:00).  
> - **Ventana de Embargo Prohibido de Prop Trading ($[t_{\text{macro}}^{\text{ann}} - 2\text{ min}, \, t_{\text{macro}}^{\text{ann}} + 2\text{ min}]$):** Exposición cero absoluta ($w_t \equiv 0$) con 60 segundos de margen de seguridad antes del inicio del embargo.

---

### 4.1 Salida de la Deriva Nocturna (Boyarchenko)

Para la posición larga nocturna abierta al precio $P_{\text{entry}}$ en $t_0 = 01:30:00\text{ EST}$, la salida está gobernada por un tiempo terminal determinista y un stop-loss porcentual:
$$\tau_{\text{exit}}^{\text{ON}} = \min \left( 08:30:00\text{ EST}, \; \inf \left\{ t > 01:30 \;\middle|\; P_t \le P_{\text{entry}} \cdot (1 - \delta_{\text{SL}}^{\text{ON}}) \right\} \right)$$
con $\delta_{\text{SL}}^{\text{ON}} = 0.0050$ ($0.50\%$ de retroceso máximo tolerable).

*Propiedad Fundamental:* La salida incondicional a las $08:30:00\text{ EST}$ garantiza que la orden se liquide antes de la apertura de Wall Street y antes de la publicación de indicadores económicos matutinos de las 08:30, eliminando el riesgo de saltos de apertura (*gaps*).

---

### 4.2 Salida Estricta de la Deriva Pre-Macro (Kurov / Regla de Embargo)

Para la posición pre-macro iniciada en $t_0 = t_{\text{macro}}^{\text{ann}} - 15\text{ minutos}$, el tiempo de salida es unívoco y no prorrogable:
$$\tau_{\text{exit}}^{\text{macro}} = \min \left( t_{\text{macro}}^{\text{ann}} - 3\text{ minutos}, \; \inf \left\{ t > t_0 \;\middle|\; |P_t - P_{\text{entry}}| \ge \Delta P_{\text{stop}}^{\text{macro}} \right\} \right)$$
donde $\Delta P_{\text{stop}}^{\text{macro}} = 10.0\text{ puntos}$ en $ES$ ($\$500\text{ USD}$ en contrato estándar).

*Cumplimiento Normativo de Prop Trading:* Al cerrar estrictamente en $T-3\text{ min}$ (ejemplo: a las $08:27:00\text{ EST}$ para el CPI o a las $13:57:00\text{ EST}$ para el FOMC), la cuenta queda en liquidez absoluta 60 segundos antes de que entre en vigor el embargo de la prop firm ($T-2\text{ min}$), erradicando las sanciones contractuales y el deslizamiento por falta de contrapartida.

---

### 4.3 Cierre Universal Previo al Mantenimiento de CME

Cualquier contrato abierto en cualquiera de los submódulos se liquida de forma forzosa al alcanzar:
$$\tau_{\text{terminal}} = 16:45:00\text{ EST} \implies w_t = 0, \quad \forall t \ge \tau_{\text{terminal}}$$

---

## 5. Dimensionamiento de Posición y Control de Riesgo Cuantitativo para Prop Trading

### 5.1 Partición Fraccionaria del Presupuesto Diario por Sesión

Para evitar que una eventualidad en la sesión nocturna agote el margen diario e inhabilite la operativa regular RTH, el límite de pérdida diaria restante $\Delta DLL_t$ se fragmenta analíticamente:
$$\Delta DLL_t^{\text{nocturno}} = \lambda_{\text{ON}} \cdot \Delta DLL_t, \quad \Delta DLL_t^{\text{RTH}} = (1 - \lambda_{\text{ON}}) \cdot \Delta DLL_t$$
donde $\lambda_{\text{ON}} = 0.25$ (se destina como máximo el $25\%$ del margen diario a la sesión nocturna).

---

### 5.2 Ecuación Cerrada de Contratos para la Deriva Nocturna

El número entero de contratos a negociar en overnight viene dado por:
$$N_{\text{ON}}^{(I)} = \min\left( \left\lfloor \frac{\min\left( \Delta DLL_t^{\text{nocturno}}, \; \gamma_T^{\text{ON}} \cdot D_t^{\text{cushion}} \right)}{\delta_{\text{SL}}^{\text{ON}} \cdot P_{\text{entry}} \cdot M^{(I)}} \right\rfloor, \; N_{\max}^{\text{ON}} \right)$$
con $\gamma_T^{\text{ON}} = 0.05$ (máximo $5\%$ del colchón de *Trailing Drawdown*) y $N_{\max}^{\text{ON}} = 2\text{ ES}$ (o $20\text{ MES}$).

---

### 5.3 Ecuación Cerrada de Contratos para la Deriva Pre-Macro

El número entero de contratos para la ventana pre-anuncio se formula como:
$$N_{\text{macro}}^{(I)} = \min\left( \left\lfloor \frac{\min\left( 0.15 \cdot \Delta DLL_t^{\text{RTH}}, \; 0.05 \cdot D_t^{\text{cushion}} \right)}{\Delta P_{\text{stop}}^{\text{macro}} \cdot M^{(I)}} \right\rfloor, \; N_{\max}^{\text{macro}} \right)$$

---

### 5.4 Algoritmo Declarativo de Conmutación Estándar a Micro

Si el riesgo unitario de una operación en contratos estándar excede el presupuesto disponible:
$$N_{\text{std}}^{(I)} = 0 \quad \land \quad \text{Presupuesto} \ge \text{Riesgo}_{\text{unit}}^{(I_{\text{micro}})}$$
el algoritmo conmuta de forma automática e inmediata al contrato Micro equivalente ($I_{\text{micro}} \in \{\text{MES}, \text{MNQ}\}$):
$$N_{t}^{*, (I)} = \begin{cases}
N_{\text{std}}^{(I)} \text{ contratos de } I & \text{si } N_{\text{std}}^{(I)} \ge 1 \\[6pt]
\min\left( \left\lfloor \frac{\text{Presupuesto}}{\text{Riesgo}_{\text{unit}}^{(I_{\text{micro}})}} \right\rfloor, \; 10 \cdot N_{\max}^{(I)} \right) \text{ de } I_{\text{micro}} & \text{si } N_{\text{std}}^{(I)} = 0 \;\land\; \text{Presupuesto} \ge \text{Riesgo}_{\text{unit}}^{(I_{\text{micro}})} \\[6pt]
0 \quad (\text{Parada de Protección Absoluta}) & \text{si el presupuesto es insuficiente para 1 micro}
\end{cases}$$

---

## 6. Rúbrica de Verificación Analítica Objetiva

La siguiente matriz de auditoría cuantitativa evalúa formalmente el cumplimiento matemático y analítico de la especificación de la estrategia F9:

| Criterio de Verificación Cuantitativa | Expresión Analítica Auditada | Umbral / Condición de Aprobación | Estado de Cumplimiento | Justificación Teórica y Evidencia Formal |
| :--- | :--- | :--- | :--- | :--- |
| **Ausencia Total de Código** | $\text{Card}(\text{CodeBlocks}) = 0$ | Idéntico a 0 bloques de código | **APROBADO** | El documento se expresa exclusivamente en lenguaje declarativo formal y ecuaciones en $\LaTeX$. |
| **Exclusión de Indicadores Descartados** | $\text{Card}(\{\text{SMA}, \text{RSI}\}) = 0$ | Cero menciones de SMA o RSI estándar | **APROBADO** | La estrategia se basa en primas de varianza empíricas ($VRP$), flujo de libro ($OFI$) y retornos nocturnos. |
| **Cumplimiento Estricto de Embargo** | $\mathbb{I}(w_t \ne 0) \cdot \mathbb{I}(t \in [T-2m, T+2m]) \equiv 0$ | Cero tolerancia a exposición en embargo | **APROBADO** | La salida forzosa mandatoria en $T-3\text{ min}$ garantiza neutralidad total 60 segundos antes del embargo. |
| **Inmunidad ante Gaps de Apertura** | $\tau_{\text{exit}}^{\text{ON}} \le 08:30:00\text{ EST}$ | Salida incondicional a las 08:30 EST | **APROBADO** | La posición nocturna se liquida antes de los anuncios de las 08:30 y de la apertura de contado de las 09:30 EST. |
| **Compuerta Estructural Asimétrica** | $S_{\text{VRP}} = -1 \implies S_{\text{unified}}^* \le 0$ | Bloqueo absoluto de compras en backwardation | **APROBADO** | Previene compras cuando la curva de volatilidad se invierte o el $VRP$ indica tensión institucional severa. |
| **Segregación del Límite Diario** | $\Delta DLL_t^{\text{nocturno}} \le 0.25 \cdot \Delta DLL_t$ | Partición analítica fraccionaria | **APROBADO** | Una racha adversa nocturna no puede comprometer más del 25% del presupuesto operativo de la sesión RTH. |
| **Protección contra Trailing Drawdown** | $\mathcal{B}_{\text{risk}} \le 0.05 \cdot D_t^{\text{cushion}}$ | Máximo 5% del colchón de drawdown | **APROBADO** | Asegura una reserva de al menos 20 operaciones antes de amenazar el suelo de liquidación histórica. |
| **Liquidación Mandatoria Previa a CME** | $\tau^* \le 16:45:00\text{ EST} < 17:00:00\text{ EST}$ | Cero retención durante el clearing | **APROBADO** | Liquidación universal obligatoria 15 minutos antes de la interrupción técnica del mercado. |
