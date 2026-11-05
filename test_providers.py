"""
test_providers.py - Diagnóstico y prueba de conexión de proveedores de IA gratuitos.
"""

import os
import sys
import time
from ai_config import PROVIDERS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] Falta la librería 'openai'.")
    print("Ejecuta en tu terminal: pip install -r requirements.txt")
    sys.exit(1)


def check_provider(provider_key: str, p_data: dict):
    env_var = p_data["env_var"]
    api_key = os.getenv(env_var)

    print(f"\n=======================================================")
    print(f"Probando: {p_data['name']}")
    print(f"Variable: {env_var}")
    print(f"Modelo:   {p_data['default_model']}")
    print(f"Base URL: {p_data['base_url']}")

    if not api_key or not api_key.strip():
        print(f"Estado:   [NO CONFIGURADO]")
        print(f"Detalle:  No se encontró {env_var} en tu archivo .env")
        print(f"Obtener:  {p_data['website']}")
        return False, None

    # Intentar conexión
    client = OpenAI(
        api_key=api_key.strip(),
        base_url=p_data["base_url"]
    )

    extra_headers = {}
    if provider_key == "openrouter":
        extra_headers = {
            "HTTP-Referer": "https://github.com/DrSimi0/Algos-Trading",
            "X-Title": "Algos-Trading"
        }

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=p_data["default_model"],
            messages=[
                {"role": "user", "content": "Di 'Conexión exitosa' en máximo 5 palabras."}
            ],
            max_tokens=30,
            temperature=0.2,
            extra_headers=extra_headers if extra_headers else None
        )
        latency = round(time.time() - start, 2)
        content = response.choices[0].message.content.strip()
        print(f"Estado:   [OK - FUNCIONANDO]")
        print(f"Latencia: {latency} segundos")
        print(f"Respuesta:{content}")
        return True, latency

    except Exception as e:
        latency = round(time.time() - start, 2)
        print(f"Estado:   [ERROR]")
        print(f"Latencia: {latency} segundos")
        print(f"Mensaje:  {e}")
        return False, latency


def main():
    print("=" * 55)
    print(" DIAGNÓSTICO DE PROVEEDORES LLM GRATUITOS (Algos-Trading)")
    print("=" * 55)

    if not os.path.exists(".env"):
        print("\n[AVISO] No se encontró el archivo '.env'.")
        print("Crea un archivo '.env' copiando '.env.example' y añade tus claves.")
        print("Ejemplo en PowerShell: Copy-Item .env.example .env\n")

    results = {}
    for key, data in PROVIDERS.items():
        ok, latency = check_provider(key, data)
        results[data["name"]] = (ok, latency)

    print("\n" + "=" * 55)
    print(" RESUMEN DE PROVEEDORES")
    print("=" * 55)
    for name, (ok, latency) in results.items():
        status = "ACTIVO " if ok else "INACTIVO"
        lat_text = f"({latency}s)" if latency is not None and ok else ""
        print(f" - {name:<18} [{status}] {lat_text}")

    print("\nPara instrucciones paso a paso, revisa: GUIA_PROVEEDORES_GRATIS.md\n")


if __name__ == "__main__":
    main()
