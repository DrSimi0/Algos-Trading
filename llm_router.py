"""
llm_router.py - Cliente unificado con enrutador y tolerancia a fallos (Fallback).
Permite consultar modelos gratuitos de IA y cambiar automáticamente de proveedor
si se alcanza el límite de velocidad (Rate Limit / Error 429).
"""

import os
import time
from typing import Optional, Dict, Any, List
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from ai_config import PROVIDERS


class FreeLLMRouter:
    def __init__(self, preferred_order: Optional[List[str]] = None):
        """
        Inicializa clientes para los proveedores que tengan clave de API en las variables de entorno.
        :param preferred_order: Lista de proveedores en orden de prioridad (ej. ['groq', 'gemini', 'openrouter']).
        """
        self.clients: Dict[str, OpenAI] = {}
        self.available_providers: List[str] = []
        
        # Orden por defecto de intento: Groq primero (rapidez), luego Gemini (contexto), OpenRouter, Mistral, Cohere
        default_order = ["groq", "gemini", "openrouter", "mistral", "cohere"]
        self.priority_order = preferred_order or default_order

        self._init_clients()

    def _init_clients(self):
        """Detecta las claves configuradas y crea los clientes OpenAI correspondientes."""
        for key, p_data in PROVIDERS.items():
            api_key = os.getenv(p_data["env_var"])
            if api_key and api_key.strip():
                try:
                    client = OpenAI(
                        api_key=api_key.strip(),
                        base_url=p_data["base_url"]
                    )
                    self.clients[key] = client
                    self.available_providers.append(key)
                except Exception as e:
                    print(f"[!] Error al inicializar {p_data['name']}: {e}")

    def get_available_providers(self) -> List[str]:
        """Devuelve los proveedores con claves activas detectadas."""
        return list(self.clients.keys())

    def chat(
        self,
        prompt: str,
        system_prompt: str = "Eres un asistente experto en finanzas cuantitativas y trading algorítmico.",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Envía un mensaje a un LLM. Si falla por Rate Limit o error de red,
        intenta automáticamente con el siguiente proveedor disponible.
        """
        if not self.clients:
            raise RuntimeError(
                "No hay proveedores de IA configurados. Agrega tus claves en el archivo .env "
                "(Consulta GUIA_PROVEEDORES_GRATIS.md para obtenerlas gratis)."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Determinar la secuencia de proveedores a intentar
        if provider and provider in self.clients:
            attempt_sequence = [provider] + [p for p in self.priority_order if p in self.clients and p != provider]
        else:
            attempt_sequence = [p for p in self.priority_order if p in self.clients]
            # Agregar cualquier otro disponible no listado en priority_order
            for p in self.clients:
                if p not in attempt_sequence:
                    attempt_sequence.append(p)

        last_error = None

        for p_name in attempt_sequence:
            client = self.clients[p_name]
            p_data = PROVIDERS[p_name]
            target_model = model if (provider == p_name and model) else p_data["default_model"]

            start_time = time.time()
            try:
                # Caso especial para OpenRouter: encabezados requeridos para identificar la app
                extra_headers = {}
                if p_name == "openrouter":
                    extra_headers = {
                        "HTTP-Referer": "https://github.com/DrSimi0/Algos-Trading",
                        "X-Title": "Algos-Trading"
                    }

                response = client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers if extra_headers else None
                )

                latency = round(time.time() - start_time, 2)
                content = response.choices[0].message.content

                return {
                    "success": True,
                    "content": content,
                    "provider": p_name,
                    "provider_name": p_data["name"],
                    "model": target_model,
                    "latency_seconds": latency,
                    "fallback_triggered": p_name != attempt_sequence[0]
                }

            except Exception as e:
                latency = round(time.time() - start_time, 2)
                print(f"[!] Falló el proveedor '{p_data['name']}' ({target_model}) en {latency}s: {e}")
                last_error = e
                # Continúa con el siguiente proveedor en attempt_sequence

        return {
            "success": False,
            "error": str(last_error),
            "content": None,
            "provider": None,
            "model": None
        }


if __name__ == "__main__":
    router = FreeLLMRouter()
    print("Proveedores disponibles:", router.get_available_providers())
    if router.get_available_providers():
        res = router.chat("Explica en una oración corta qué es una media móvil exponencial (EMA).")
        print("\nResultado:")
        print(f"Proveedor: {res.get('provider_name')} | Modelo: {res.get('model')} | Latencia: {res.get('latency_seconds')}s")
        print(f"Respuesta: {res.get('content')}")
    else:
        print("Crea un archivo .env con tus claves de API para comenzar a probar.")
