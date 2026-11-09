# Guía de Instalación y Uso: SQX ResultsPlugin para Estrategia F4

## Descripción
Este plugin especializado para **StrategyQuant X (SQX)** añade una pestaña analítica dedicada en la sección **Results** diseñada específicamente para la estrategia **F4 (Arbitraje Estadístico por Cointegración ES vs NQ / Difusión Ornstein-Uhlenbeck)** y para la evaluación de reglas estrictas de **Prop Trading (Cuentas de Fondeo)**.

---

## 🚀 Instalación en 3 Pasos

1. **Localiza tu carpeta de instalación de StrategyQuant X:**
   En Windows típicamente se encuentra en:
   `C:\StrategyQuantX\` (o donde hayas instalado SQX).

2. **Copia la carpeta del plugin:**
   Copia la carpeta `F4_PropTrading_Analyzer` dentro de la ruta:
   ```text
   <Ruta_Instalacion_SQX>/user/extend/ResultsPlugins/F4_PropTrading_Analyzer/
   ```
   Asegúrate de que el archivo `index.html` quede ubicado exactamente en:
   ```text
   <Ruta_Instalacion_SQX>/user/extend/ResultsPlugins/F4_PropTrading_Analyzer/index.html
   ```

3. **Reinicia StrategyQuant X:**
   Al abrir SQX e ir a cualquier Databank o sección **Results**, verás la nueva pestaña **`F4_PropTrading_Analyzer`**.

---

## 📊 Capacidades y Métricas Especializadas

### 1. Auditoría de Cuentas de Fondeo (Prop Trading Pass/Fail)
- **Límite de Pérdida Diaria (Daily Loss Limit - DLL):**
  - Agrupa automáticamente todos los trades cerrados por día hábil.
  - Alerta roja inmediata (`BREACH`) si alguna sesión excede el límite contractual (por ejemplo, $-\$1,500$ USD en cuenta de $\$50\text{k}$).
- **Trailing Max Drawdown:**
  - Monitorea el drawdown dinámico desde el balance más alto alcanzado (*High-Water Mark*), no el drawdown clásico de pico a valle.
- **Progreso de Profit Target:**
  - Barra visual de avance hacia la meta de fondeo ($\$3,000$ o $\$6,000$ USD).

### 2. Microestructura y Proceso Ornstein-Uhlenbeck
- **Tiempo de Retención vs. Vida Media ($\tau_{OU}$):**
  - Evalúa la duración media de los trades frente a la vida media teórica de reversión a la media ($\tau_{OU} = \ln(2)/\theta$).
- **Desglose de Tipos de Cierre (`CloseType`):**
  - Gráfico de dona que distingue salidas por convergencia de spread ($S \to 0$, señal o take profit), salidas por stop loss catastrófico y cierres al fin de la sesión (*End of Day*).
- **Control de Fricción y Piso Operativo ($105 USD):**
  - Compara el promedio neto de ganancia por trade frente al umbral mínimo requerido para absorber comisiones y deslizamiento en futuros CME Globex.

---

## 🧪 Modo Demostración Autónomo
Puedes abrir `index.html` directamente en cualquier navegador web (Google Chrome, Edge, Firefox) con doble clic. El plugin detectará automáticamente que no está dentro del iframe de SQX y cargará un conjunto de datos sintéticos representativos de F4 para que puedas explorar la interfaz y sus gráficos de inmediato.
