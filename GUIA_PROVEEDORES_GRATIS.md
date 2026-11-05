# Guía de Proveedores de LLM Gratuitos para Trading Algorítmico

Esta guía resume las mejores opciones del repositorio [awesome-free-llm-apis](https://github.com/mnfst/awesome-free-llm-apis) que ofrecen **capas gratuitas permanentes sin requerir tarjeta de crédito**.

---

## 1. Los 5 Proveedores Principales Recomendados

| Proveedor | Enlace para Clave | Modelos Destacados | Ventaja Principal en Trading |
| :--- | :--- | :--- | :--- |
| **Groq** | [console.groq.com/keys](https://console.groq.com/keys) | `llama-3.3-70b-versatile`, `deepseek-r1-distill-llama-70b` | **Ultra-rápido (LPU)**: Ideal para analizar noticias y generar señales en milisegundos. |
| **Google Gemini** | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | `gemini-2.5-flash`, `gemini-2.5-flash-lite` | **Contexto gigante (1M tokens)**: Permite cargar reportes de ganancias y libros contables completos. |
| **OpenRouter** | [openrouter.ai/keys](https://openrouter.ai/keys) | Modelos con sufijo `:free` (ej. `meta-llama/llama-3.3-70b-instruct:free`) | **Variedad**: Acceso a múltiples modelos abiertos en un solo lugar. |
| **Mistral AI** | [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys) | `mistral-small-latest`, `codestral-latest` | **Código y Razonamiento**: Excelente para escribir y auditar estrategias cuantitativas. |
| **Cohere** | [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) | `command-r`, `c4ai-aya-expanse-32b` | **1,000 llamadas/mes gratis**: Análisis semántico y procesamiento de lenguaje financiero. |

---

## 2. Cómo obtener tus claves (Paso a Paso)

### A. Groq (Recomendación #1 por velocidad)
1. Ve a [https://console.groq.com/keys](https://console.groq.com/keys).
2. Inicia sesión con tu cuenta de Google o GitHub.
3. Haz clic en **Create API Key**.
4. Copia la clave generada y pégala en tu archivo `.env` en `GROQ_API_KEY=gsk_...`.

### B. Google Gemini / AI Studio (Recomendación #2 por capacidad de contexto)
1. Ve a [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).
2. Inicia sesión con tu cuenta de Google.
3. Haz clic en **Create API key** (en un proyecto nuevo o existente).
4. Copia la clave y pégala en `.env` en `GEMINI_API_KEY=AIzaSy...`.

### C. OpenRouter (Modelos gratuitos variados)
1. Ve a [https://openrouter.ai/keys](https://openrouter.ai/keys).
2. Inicia sesión y crea una clave nueva.
3. Añádela en `.env` en `OPENROUTER_API_KEY=sk-or-v1-...`.
4. *Nota*: Solo utiliza modelos que terminen en `:free` para que no consuman créditos de pago.

---

## 3. Configuración en este Proyecto

### Paso 1: Crear el archivo `.env`
Copia la plantilla `.env.example`:
```powershell
Copy-Item .env.example .env
```
Abre `.env` y pega tus claves obtenidas (no uses comillas ni espacios adicionales).

### Paso 2: Instalar dependencias
Crea un entorno virtual y luego instala las dependencias:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Paso 3: Probar la conexión
Ejecuta el script de diagnóstico:
```powershell
python test_providers.py
```
El script probará cada clave configurada y te indicará si responde correctamente y el tiempo de respuesta en segundos.

---

## 4. Uso en tus Algoritmos de Trading

Puedes utilizar `FreeLLMRouter` directamente en cualquier script o notebook de tu proyecto:

```python
from llm_router import FreeLLMRouter

# Inicializa el enrutador
router = FreeLLMRouter()

# Ejemplo: Analizar el sentimiento de un titular
titular = "La Reserva Federal anuncia una pausa en las tasas de interés pero advierte sobre presiones inflacionarias."

respuesta = router.chat(
    prompt=f"Clasifica el sentimiento de este titular para el mercado bursátil (Positivo/Neutral/Negativo) y da una breve justificación: '{titular}'",
    system_prompt="Eres un analista cuantitativo de mercados financieros."
)

print(f"Respuesta de {respuesta['provider_name']} ({respuesta['model']}):")
print(respuesta['content'])
```

### ¿Qué hace el sistema de Fallback?
Si por alguna razón Groq alcanza su límite de peticiones por minuto (error 429), `FreeLLMRouter` enviará la consulta inmediatamente a Google Gemini o OpenRouter sin detener tu algoritmo de trading.
