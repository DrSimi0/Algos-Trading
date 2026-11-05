"""
example_sentiment_analysis.py - Ejemplo práctico de análisis de sentimiento financiero
utilizando la rotación de proveedores gratuitos de IA.
"""

from llm_router import FreeLLMRouter

def main():
    print("Inicializando enrutador de modelos gratuitos...")
    router = FreeLLMRouter()
    
    activos = router.get_available_providers()
    if not activos:
        print("[!] No se encontraron claves en .env.")
        print("Revisa GUIA_PROVEEDORES_GRATIS.md para configurar tus claves sin costo.")
        return

    print(f"Proveedores activos detectados: {activos}\n")

    noticia_ejemplo = (
        "El IPC de EE. UU. subió un 0.2% intermensual en línea con lo esperado, "
        "mientras que las solicitudes iniciales de desempleo cayeron inesperadamente, "
        "demostrando la solidez del mercado laboral."
    )

    prompt = f"""
Analiza la siguiente noticia macroeconómica:
"{noticia_ejemplo}"

Proporciona:
1. Sentimiento general (Alcista / Neutral / Bajista para el S&P 500).
2. Impacto probable en rendimientos de bonos del tesoro.
3. Breve razonamiento (máximo 3 líneas).
"""

    print("Enviando consulta al router...")
    resultado = router.chat(
        prompt=prompt,
        system_prompt="Eres un estratega macroeconómico cuantitativo experto en análisis de sentimiento de mercado.",
        temperature=0.3
    )

    if resultado["success"]:
        print(f"\n[Éxito] Proveedor utilizado: {resultado['provider_name']}")
        print(f"Modelo: {resultado['model']}")
        print(f"Latencia: {resultado['latency_seconds']} s")
        print(f"¿Activó fallback secundario?: {'Sí' if resultado['fallback_triggered'] else 'No'}")
        print("\n--- Respuesta del Modelo ---")
        print(resultado["content"])
    else:
        print(f"\n[Error] No se pudo obtener respuesta: {resultado['error']}")

if __name__ == "__main__":
    main()
