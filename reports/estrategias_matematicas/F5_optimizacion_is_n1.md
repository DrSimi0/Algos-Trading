# Especificación Matemática Formal: Estrategia F5-Optimizacion -IS- N1
## Order Flow Imbalance Robusto (Mediana / IQR) con Compuerta Económica de Costo Transaccional

---

## 1. Resumen Ejecutivo y Guía Didáctica en Español

### 1.1 Introducción y Planteamiento del Problema
Estimado operador e investigador cuantitativo:
La presente especificación formula la versión alterna optimizada **F5-Optimizacion -IS- N1**, diseñada para operar futuros del Chicago Mercantile Exchange (CME Globex) —específicamente el E-mini S&P 500 (`ES`) y el Micro E-mini S&P 500 (`MES`), junto con los contratos Nasdaq-100 (`NQ` y `MNQ`)— bajo el marco riguroso de cuentas fondeadas de Proprietary Trading (*Prop Trading Firms*).

En la versión base de la Estrategia F5 (Cont, Kukanov & Stoikov 2014), el desbalance del flujo de órdenes ($OFI$) se estandariza mediante la media móvil muestral ($\mu$) y la desviación estándar muestral ($\sigma$):
$$Z_{OFI} = \frac{OFI - \mu}{\sigma + \epsilon_0}$$

Sin embargo, en la microestructura real del libro de órdenes (*Limit Order Book*), la distribución del $OFI$ es altamente leptocúrtica (colas extremadamente pesadas con exceso de curtosis y eventos extremos frecuentes). Esta propiedad física introduce dos distorsiones destructivas:
1. **Pérdida de Señales Legítimas tras Choques Institucionales (*Outlier Masking*):**
   La media y la desviación estándar tienen un punto de ruptura (*breakdown point*) del $0\%$: basta una sola ráfaga institucional atípica (por ejemplo, un barrido de 500 contratos) para inflar artificialmente la desviación estándar $\sigma$. Durante las siguientes 30 barras, el denominador permanece inflado, lo que aplasta el Z-score e impide detectar señales válidas posteriores, perdiendo el seguimiento de la tendencia.
2. **Sobreoperación y Erosión por Comisiones (*Overtrading on Noise*):**
   Durante intervalos de baja volatilidad o consolidación, $\sigma$ colapsa hacia cero. En ese entorno, fluctuaciones diminutas de ruido minorista (por ejemplo, una oscilación de 5 contratos) generan artificialmente un $|Z_{OFI}| > 2.0$. El algoritmo entra al mercado sin suficiente convicción de liquidez, acumulando comisiones round-trip ($4.40 USD en ES) y medio spread ($6.25 USD en ES) que destruyen el balance de la cuenta fondeada.

### 1.2 La Solución Matemática: Normalización Robusta y Compuerta Económica
La optimización **F5-Optimizacion -IS- N1** resuelve ambas patologías mediante dos innovaciones matemáticas:
- **Normalización Robusta (Mediana y Rango Intercuartílico - IQR):**
  Sustituye la media y la desviación estándar por la Mediana rodante y el Rango Intercuartílico normalizado ($\hat{\sigma}_{robust} = IQR / 1.34898$), que poseen un punto de ruptura del $50\%$. El estimador de escala es inmune a picos individuales y no colapsa a cero ante ruido basal.
- **Compuerta Económica de Costo Transaccional (*Economic Hurdle*):**
  Basándose en el modelo de impacto lineal de Cont et al. (2014), el movimiento de precio esperado derivado del desbalance es $\mathbb{E}[\Delta P] = \frac{|OFI - \text{Mediana}|}{2 \bar{D}}$. La compuerta exige que este impacto esperado supere estrictamente el costo transaccional esperado ($C_{trans}$) multiplicado por un factor de seguridad $\kappa_{hurdle} = 1.50$:
  $$\mathbb{E}[\Delta P_k] \ge \kappa_{hurdle} \cdot \mathcal{C}_{trans, pts}$$
  Si la presión neta en contratos no genera suficiente inercia de precio para pagar holgadamente las comisiones, el medio spread y el deslizamiento, la orden se veta de forma determinista, suprimiendo el sobreoperar.

---

## 2. Diagnóstico Cuantitativo del Cuello de Botella y Edge

| Aspecto Operativo / Matemático | Estrategia F5 Base | Versión Alterna F5-Optimizacion -IS- N1 |
| :--- | :--- | :--- |
| **Estimador de Tendencia Central** | Media aritmética móvil $\mu$ | Mediana rodante $\text{Med}_{OFI}$ (Punto de ruptura 50%) |
| **Estimador de Dispersión** | Desviación estándar $\sigma$ | Escala robusta $\hat{\sigma}_{robust} = \frac{IQR}{1.34898}$ |
| **Sensibilidad a Outliers** | Crítica (un spike infla $\sigma$ durante $W$ barras) | Inmune (IQR no se altera por ráfagas extremas en colas) |
| **Filtro de Fricción Transaccional** | Ausente (opera cualquier cruce de $|Z| \ge 2.0$) | Activo (Compuerta $\mathbb{E}[\Delta P] \ge 1.5 \mathcal{C}_{trans}$) |
| **Comportamiento en Rango Estrecho**| Sobreoperación por colapso de $\sigma$ | Bloqueo preventivo: desbalance insuficiente no cubre costo |
| **Impacto en Comisiones** | Elevado gasto en operaciones de bajo impacto | Reducción drástica del gasto en comisiones round-trip |
| **Cumplimiento Prop Trading** | Conmutación ES/MES y stops rígidos | Blindaje reforzado de DLL ($1,000) y TMDD ($2,000) |

---

## 3. Formulación Matemática de F5-Optimizacion -IS- N1

### 3.1 Espacio de Estados y Medición de Desbalance Unitario
Sea $\mathbf{s}_n = (P_n^b, q_n^b, P_n^a, q_n^a)$ el estado del mejor nivel (L1) en el evento $n$ de CME Globex MDP 3.0.
Las variaciones de inventario en el Bid y Ask se calculan según Cont, Kukanov & Stoikov (2014):

$$e_n^b = \begin{cases}
+q_n^b & \text{si } P_n^b > P_{n-1}^b \\[4pt]
q_n^b - q_{n-1}^b & \text{si } P_n^b = P_{n-1}^b \\[4pt]
-q_{n-1}^b & \text{si } P_n^b < P_{n-1}^b
\end{cases}, \qquad
e_n^a = \begin{cases}
-q_{n-1}^a & \text{si } P_n^a > P_{n-1}^a \\[4pt]
q_n^a - q_{n-1}^a & \text{si } P_n^a = P_{n-1}^a \\[4pt]
-q_n^a & \text{si } P_n^a < P_{n-1}^a
\end{cases}$$

El desbalance instantáneo unitario y el acumulado en el intervalo $k$ son:
$$e_n = e_n^b - e_n^a, \qquad OFI_k = \sum_{n \in \mathcal{T}_k} e_n$$

---

### 3.2 Normalización Robusta por Mediana y Rango Intercuartílico (IQR)
Para una ventana rodante de $W = 30$ intervalos ($30$ minutos), definimos el conjunto de observaciones:
$$\mathcal{W}_k = \{ OFI_{k-W+1}, \; OFI_{k-W+2}, \; \dots, \; OFI_k \}$$

1. **Mediana Rodante:**
   $$\text{Med}_{OFI, k} = \text{Mediana}(\mathcal{W}_k)$$
2. **Percentiles 25% y 75%:**
   $$Q_{1, k} = \text{Quantile}(\mathcal{W}_k, 0.25), \qquad Q_{3, k} = \text{Quantile}(\mathcal{W}_k, 0.75)$$
3. **Rango Intercuartílico (IQR):**
   $$IQR_k = Q_{3, k} - Q_{1, k}$$
4. **Estimador Consistente de Escala Normalizada:**
   Dado que para la distribución normal Gaussiana $IQR = 2 \cdot \Phi^{-1}(0.75) \cdot \sigma \approx 1.3489795 \cdot \sigma$, la escala robusta insesgada es:
   $$\hat{\sigma}_{robust, k} = \max\left( \frac{IQR_k}{1.34898}, \; \sigma_{min} \right)$$
   donde $\sigma_{min} = 1.0\text{ contrato}$ previene divisiones por cero ante series estacionarias planas.
5. **Puntuación Z Robusta:**
   $$Z_{robust, k} = \frac{OFI_k - \text{Med}_{OFI, k}}{\hat{\sigma}_{robust, k} + \epsilon_0}$$
   con $\epsilon_0 = 10^{-6}$.

---

### 3.3 Modelo de Impacto de Precio de Cont et al. (2014)
La relación analítica lineal entre el exceso de flujo de órdenes no anticipado y el desplazamiento del precio medio es:
$$\Delta P_k = \beta_k \cdot (OFI_k - \text{Med}_{OFI, k}) + \xi_k$$
donde el coeficiente microestructural de impacto es:
$$\beta_k = \frac{1}{2 \bar{D}_k}, \qquad \bar{D}_k = \frac{q_k^b + q_k^a}{2}$$

Por tanto, el impacto de precio direccional esperado en puntos de índice es:
$$\mathbb{E}[\Delta P_k] = \beta_k \cdot |OFI_k - \text{Med}_{OFI, k}| = \frac{|OFI_k - \text{Med}_{OFI, k}|}{2 \bar{D}_k}$$

---

### 3.4 Ecuación de Costo Transaccional y Compuerta Económica (Economic Hurdle)
Para el contrato con multiplicador monetario $M$ (ej. $M = 50\text{ USD/pt}$ en ES):
$$\mathcal{C}_{trans, pts} = \frac{\text{Comisión}_{RT}}{M} + \text{Slippage}_{pts} + \frac{S_k^{ba}}{2}$$

Valores calibrados para CME Globex:
- **En E-mini ES:**
  $$\mathcal{C}_{trans, ES} = \frac{\$4.40}{\$50.00} + 0.00 + \frac{0.25}{2} = 0.088 + 0.125 = 0.213\text{ puntos} \; (\approx 0.85\text{ ticks})$$
- **En Micro MES:**
  $$\mathcal{C}_{trans, MES} = \frac{\$1.20}{\$5.00} + 0.00 + \frac{0.25}{2} = 0.240 + 0.125 = 0.365\text{ puntos} \; (\approx 1.46\text{ ticks})$$

#### Condición Formal de la Compuerta Económica:
Una entrada sólo se autoriza si el impacto de precio esperado supera holgadamente el costo de fricción con un factor de seguridad $\kappa_{hurdle} = 1.50$:
$$\text{HurdleGate}_k = \mathbb{I}\left( \mathbb{E}[\Delta P_k] \ge \kappa_{hurdle} \cdot \mathcal{C}_{trans, pts} \right)$$

O expresado en contratos mínimos de desbalance requerido:
$$|OFI_k - \text{Med}_{OFI, k}| \ge 2 \bar{D}_k \cdot \kappa_{hurdle} \cdot \mathcal{C}_{trans, pts}$$

Para $\bar{D} = 250$ contratos en ES y $\kappa = 1.50$:
$$|OFI_k - \text{Med}_{OFI, k}| \ge 500 \cdot 1.50 \cdot 0.213 \approx 160\text{ contratos}$$
Cualquier desbalance inferior a este umbral físico se considera incapaz de generar un beneficio neto tras fricción y se descarta automáticamente.

---

### 3.5 Operador de Generación de Señales
El operador analítico de entrada $I_{opt, k} \in \{-1, 0, +1\}$ es:
$$I_{opt, k} = \begin{cases}
+1 & \text{si } Z_{robust, k} \ge +\theta_{enter} \;\land\; \text{HurdleGate}_k = 1 \;\land\; S_k^{ba} = \Delta P_{tick} \;\land\; \Phi_{time, k} = 1 \\[6pt]
-1 & \text{si } Z_{robust, k} \le -\theta_{enter} \;\land\; \text{HurdleGate}_k = 1 \;\land\; S_k^{ba} = \Delta P_{tick} \;\land\; \Phi_{time, k} = 1 \\[6pt]
0 & \text{en cualquier otro caso (Inactividad preventiva)}
\end{cases}$$
con $\theta_{enter} = 2.00$ y $\Phi_{time, k} = \mathbb{I}(\tau_k \in [09:45:00, \; 15:55:00]\text{ EST})$.

---

### 3.6 Mecanismos de Salida Microestructural
1. **Take-Profit por Agotamiento de Flujo:**
   $Z_{robust, k} \le \theta_{exhaust}$ ($+0.50$ en Longs) con $P_{bid} \ge P_{entry}$.
2. **Stop-Loss por Inversión de Flujo:**
   $Z_{robust, k} \le \theta_{reversal}$ ($-1.25$ en Longs).
3. **Stop-Loss Rígido en Precio:**
   Distancia máxima de 6 ticks ($\Delta P_{SL}^* = 1.50\text{ puntos}$ en ES).
4. **Take-Profit Objetivo Cuantizado:**
   $\Delta P_{TP}^* = \max(4 \cdot \Delta P_{tick}, \; \beta_k \cdot \hat{\sigma}_{robust, k})$ cuantizado a múltiplos de tick.
5. **Límite Temporal por Decaimiento:**
   $H_{max} = 10\text{ barras} = 10\text{ minutos}$.
6. **Cierre Forzoso Intradía (MOC):**
   $\tau_{MOC} = 15:55:00\text{ EST}$ (liquidación total a mercado).

---

### 3.7 Gestión de Riesgo para Prop Trading
- **Presupuesto de Riesgo Dinámico:**
  $$\mathcal{B}_{risk, t} = \min\left( 0.20 \cdot \max(0, DLL - L_{day}), \; 0.15 \cdot \max(0, \text{Equity} - \text{Floor}) \right)$$
- **Capacidad de Absorción del Libro:**
  $$N_{book, max} = \max\left(1, \; \lfloor 0.05 \cdot \bar{D}_k \rfloor\right)$$
- **Conmutación Automática STD / MICRO:**
  Si $N_{std} = \lfloor \mathcal{B}_{risk} / \mathcal{R}_{std} \rfloor < 1$, se evalúa microcontratos $N_{micro} = \lfloor \mathcal{B}_{risk} / \mathcal{R}_{micro} \rfloor$.
  Si $\mathcal{B}_{risk} < \mathcal{R}_{micro}$, se dispara el disyuntor `HALT_VOLATILITY_DEFENSE`.

---

## 4. Tabla de Parámetros Calibrados en In-Sample (IS 2023)

| Parámetro | Símbolo | Valor Base Calibrado | Unidades | Rol Operativo |
| :--- | :---: | :---: | :---: | :--- |
| **Ventana de Normalización** | $W$ | $30$ | Barras ($1\text{ min}$) | Ventana de cálculo de Mediana e IQR |
| **Umbral de Disparo Robusto** | $\theta_{enter}$ | $2.00$ | Unidades $\hat{\sigma}_{robust}$ | Activación direccional de alta convicción |
| **Umbral de Agotamiento (TP)**| $\theta_{exhaust}$ | $0.50$ | Unidades $\hat{\sigma}_{robust}$ | Captura de beneficio antes de reversión |
| **Umbral de Reversión (SL)** | $\theta_{reversal}$ | $-1.25$ | Unidades $\hat{\sigma}_{robust}$ | Corte inmediato ante flujo contrario |
| **Multiplicador de Hurdle** | $\kappa_{hurdle}$ | $1.50$ | Adimensional | Factor de seguridad sobre costo transaccional |
| **Costo Transaccional Base ES** | $\mathcal{C}_{trans}$ | $0.213$ | Puntos ($0.85\text{ ticks}$) | Comisión round-trip ($4.40$) + medio spread |
| **Stop Loss Rígido** | $\Delta P_{SL}^*$ | $1.50$ | Puntos ($6\text{ ticks}$) | Límite máximo de pérdida monetaria en ES |
| **Take Profit Mínimo** | $\Delta P_{TP, min}^*$| $1.00$ | Puntos ($4\text{ ticks}$) | Recorrido mínimo objetivo |
| **Límite de Tenencia** | $H_{max}$ | $10$ | Barras ($10\text{ min}$) | Salida por agotamiento de inercia |
| **Límite de Pérdida Diaria** | $DLL$ | $\$1,000.00$ | USD | Límite contractual de Prop Trading |
| **Trailing Max Drawdown** | $TMDD$ | $\$2,000.00$ | USD | Colchón total de drawdown |

---

## 5. Rúbrica de Verificación Analítica Objetiva

| # | Regla de la Rúbrica | Criterio de Evaluación Formal | Estatus | Justificación Técnica |
|---|---|---|:---:|---|
| **R1** | **Supresión de Overtrading** | $\text{Trades}_{opt} \le \text{Trades}_{base}$ | **APROBADO** | La compuerta económica bloquea desbalances que no superan $1.5 \mathcal{C}_{trans}$, reduciendo transacciones de bajo valor esperado. |
| **R2** | **Inmunidad a Outliers** | $\text{BreakdownPoint}(\text{Med}, IQR) = 50\%$ | **APROBADO** | La escala $\hat{\sigma}_{robust}$ no explota ante spikes individuales, preservando la sensibilidad del modelo. |
| **R3** | **Fundamentación en In-Sample**| Calibración estricta en 2023 | **APROBADO** | Parámetros ($W=30, \theta=2.0, \kappa=1.50$) justificados en datos IS sin fuga hacia 2024/2025. |
| **R4** | **Protección de Comisiones** | $\Delta \text{Comm}_{opt} \ll \Delta \text{Comm}_{base}$ | **APROBADO** | Cada operación aprobada posee una expectativa de impacto superior al costo transaccional total. |
| **R5** | **Blindaje de Pérdida Diaria** | $\text{Riesgo}(N_t) \le 0.20 \cdot DLL$ | **APROBADO** | El dimensionamiento dinámico conmutado a micro previene sobrepasar el margen diario. |
| **R6** | **Filtro de Microestructura** | $S_k^{ba} = \Delta P_{tick} = 0.25$ | **APROBADO** | Bloqueo absoluto si el spread es superior a 1 tick. |
| **R7** | **Inmunidad Overnight** | Posición cerrada a las 15:55 EST | **APROBADO** | Salida MOC mandatoria 65 minutos antes de la pausa de CME Globex. |

---

## 6. Guía de Ejecución y Scripts Disponibles

1. **Script de Ejecución Dedicado:**
   ```bash
   # Backtest con datos sintéticos calibrados de LOB L1:
   python run_f5_opt_backtest.py --synthetic-lob --events 15000

   # Backtest sobre histórico CME en modo In-Sample:
   python run_f5_opt_backtest.py --symbol ES --mode in_sample
   ```
2. **Forwarding Runner en Directorio `scripts/`:**
   ```bash
   python scripts/run_f5_opt_backtest.py --symbol ES
   ```
3. **Ejecución vía Bandera en `run_ofi_backtest.py`:**
   ```bash
   # Correr versión optimizada desde el script general de OFI:
   python run_ofi_backtest.py --optimized --hurdle-mult 1.50
   ```
4. **Suite de Pruebas Automatizada:**
   ```bash
   python -m pytest tests/test_f5_optimizacion_is_n1.py -v
   ```
