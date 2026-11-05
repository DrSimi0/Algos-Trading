"""
ai_config.py - Configuración y catálogo de proveedores de LLM con capas gratuitas.
Basado en https://github.com/mnfst/awesome-free-llm-apis
"""

PROVIDERS = {
    "groq": {
        "name": "Groq",
        "env_var": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "fallback_models": [
            "llama-3.1-8b-instant",
            "deepseek-r1-distill-llama-70b"
        ],
        "website": "https://console.groq.com/keys",
        "description": "Inferencia de velocidad extrema (LPU). Excelente para trading algorítmico y señales en tiempo real.",
        "requires_key": True,
    },
    "gemini": {
        "name": "Google Gemini",
        "env_var": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-flash",
        "fallback_models": [
            "gemini-2.5-flash-lite",
            "gemini-1.5-flash"
        ],
        "website": "https://aistudio.google.com/app/apikey",
        "description": "Ventana de contexto masiva (1M tokens). Ideal para analizar balances financieros extensos y libros de órdenes.",
        "requires_key": True,
    },
    "openrouter": {
        "name": "OpenRouter",
        "env_var": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "fallback_models": [
            "google/gemini-2.0-flash-exp:free",
            "deepseek/deepseek-r1:free",
            "mistralai/mistral-7b-instruct:free"
        ],
        "website": "https://openrouter.ai/keys",
        "description": "Multi-proveedor. Permite acceder a múltiples modelos gratuitos que lleven la etiqueta ':free'.",
        "requires_key": True,
    },
    "mistral": {
        "name": "Mistral AI",
        "env_var": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-small-latest",
        "fallback_models": [
            "codestral-latest",
            "open-mistral-nemo"
        ],
        "website": "https://console.mistral.ai/api-keys",
        "description": "Modelos europeos de alta precisión, excelentes para razonamiento y programación de estrategias.",
        "requires_key": True,
    },
    "cohere": {
        "name": "Cohere",
        "env_var": "COHERE_API_KEY",
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "default_model": "command-r",
        "fallback_models": [
            "command-r-plus",
            "c4ai-aya-expanse-32b"
        ],
        "website": "https://dashboard.cohere.com/api-keys",
        "description": "1,000 llamadas gratis al mes. Muy bueno para análisis semántico y clasificación de noticias.",
        "requires_key": True,
    }
}
