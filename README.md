# Algos-Trading: Sistema Gestor de Portafolio Cuantitativo Multi-Estrategia

Plataforma de gobernanza, backtesting y asignación de capital para portafolios de trading algorítmico multi-estrategia.

---

## 🏛️ Dinámica de Trabajo: Manager vs. Scouts

Este proyecto opera bajo un modelo de división de funciones:

```
[Chats Externos: Strategy Scouts]  ---> [Ficha Técnica: EVALUATION_TEMPLATE.md]
                                                       │
                                                       ▼
[Este Chat: Portfolio Manager]     <--- [Evaluación, Filtro y Asignación]
```

1. **Portfolio Manager (Este chat):**
   - Vela por la integridad del portafolio completo (control de correlación, drawdown global y sizing).
   - Define la interfaz estándar (`BaseStrategy`).
   - Adapta el código aprobado y lo somete al motor de backtesting (`BacktestEngine`).
2. **Strategy Scouts (Tus otros chats):**
   - Auditan repositorios externos de GitHub usando la plantilla [templates/EVALUATION_TEMPLATE.md](file:///c:/Users/leono/OneDrive/Documentos/QUANT TRADING/Algos-Trading/templates/EVALUATION_TEMPLATE.md).
   - Detectan sesgos fatales (*lookahead bias*, ausencia de comisiones, sobreajuste).

---

## 📂 Estructura del Proyecto

```
Algos-Trading/
├── config/                      # Ajustes generales y de riesgo
├── data/raw/                    # Almacenamiento local en caché (parquet/csv)
├── src/
│   ├── core/
│   │   ├── base_strategy.py     # Clase abstracta BaseStrategy
│   │   └── data_loader.py       # Descargador OHLCV estandarizado (yfinance)
│   ├── strategies/
│   │   ├── momentum/            # Estrategias de tendencia (ej. SMA Crossover)
│   │   └── mean_reversion/      # Estrategias de reversión (ej. RSI Reversion)
│   ├── portfolio/
│   │   ├── allocator.py         # Asignador de capital (Equal Weight, Inverse Vol)
│   │   └── risk_manager.py      # Circuit breakers y matriz de correlación
│   └── backtest/
│       └── engine.py            # Motor de simulación libre de lookahead bias
├── templates/
│   └── EVALUATION_TEMPLATE.md   # Plantilla para auditar repos en otros chats
├── tests/                       # Pruebas unitarias automatizadas (pytest)
├── run_portfolio.py             # Demostrador ejecutable del portafolio
└── README.md
```

---

## 🚀 Inicio Rápido

### 1. Ejecutar las Pruebas Unitarias
```bash
python -m pytest tests/ -v
```

### 2. Ejecutar la Simulación del Portafolio
```bash
python run_portfolio.py
```

---

## 📋 ¿Cómo evaluar un nuevo repositorio de GitHub?

1. Abre un nuevo chat con cualquier LLM.
2. Copia y pega las instrucciones de [templates/EVALUATION_TEMPLATE.md](file:///c:/Users/leono/OneDrive/Documentos/QUANT TRADING/Algos-Trading/templates/EVALUATION_TEMPLATE.md) junto con el link del repositorio.
3. El chat te generará la **Ficha Técnica**.
4. Pega esa ficha técnica aquí en el chat del **Manager**. Yo analizaré si la estrategia aporta valor descorrelacionado al portafolio y la convertiremos en un módulo dentro de `src/strategies/`.
