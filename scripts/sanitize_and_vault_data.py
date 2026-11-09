"""
scripts/sanitize_and_vault_data.py
Script de Sanitización y Contención Física de Datos.

1. Extrae los datos de 2024 (OOS) y 2025 (Holdout) de CME_MINI_ES_5m_3y.parquet y CME_MINI_NQ_5m_3y.parquet.
2. Cifra los datos en data/vault/oos_2024_5m.enc y data/vault/holdout_2025_5m.enc usando AES-256-GCM.
3. Trunca físicamente data/raw/CME_MINI_ES_5m_3y.parquet y NQ para contener exclusivamente 2023 (20,280 barras).
4. Mueve y purga de reports/ los archivos de operaciones de OOS y Holdout hacia reports/vault/.
5. Planta el honeypot / canary en data/vault/canary_trap.enc.
"""

import os
import shutil
from pathlib import Path
import pandas as pd

from src.core.vault_security import (
    encrypt_dataset_to_vault,
    decrypt_vault_to_memory,
    generate_master_vault_key,
    plant_canary_trap,
)


def main():
    print("=" * 78)
    print(" SANITIZACIÓN Y AISLAMIENTO CRIPTOGRÁFICO DE DATOS (TRUE AIR-GAP)")
    print("=" * 78)

    # 1. Rutas
    es_parquet = Path("data/raw/CME_MINI_ES_5m_3y.parquet")
    nq_parquet = Path("data/raw/CME_MINI_NQ_5m_3y.parquet")
    es_csv = Path("data/raw/CME_MINI_ES_5m_3y.csv")
    nq_csv = Path("data/raw/CME_MINI_NQ_5m_3y.csv")
    vault_dir = Path("data/vault")
    reports_dir = Path("reports")
    reports_vault_dir = reports_dir / "vault"

    vault_dir.mkdir(parents=True, exist_ok=True)
    reports_vault_dir.mkdir(parents=True, exist_ok=True)

    if not es_parquet.exists() or not nq_parquet.exists():
        raise FileNotFoundError("No se encontraron los datasets originales en data/raw/")

    print("\n[1/5] Cargando datasets originales completos (2023, 2024, 2025)...")
    df_es = pd.read_parquet(es_parquet)
    df_nq = pd.read_parquet(nq_parquet)

    print(f"  ES Total: {len(df_es):,} barras | Años: {sorted(df_es.index.year.unique())}")
    print(f"  NQ Total: {len(df_nq):,} barras | Años: {sorted(df_nq.index.year.unique())}")

    # Separar particiones por año
    es_2023 = df_es[df_es.index.year == 2023].copy()
    es_2024 = df_es[df_es.index.year == 2024].copy()
    es_2025 = df_es[df_es.index.year == 2025].copy()

    nq_2023 = df_nq[df_nq.index.year == 2023].copy()
    nq_2024 = df_nq[df_nq.index.year == 2024].copy()
    nq_2025 = df_nq[df_nq.index.year == 2025].copy()

    print(f"  2023 In-Sample : ES={len(es_2023):,}, NQ={len(nq_2023):,} barras")
    print(f"  2024 OOS       : ES={len(es_2024):,}, NQ={len(nq_2024):,} barras")
    print(f"  2025 Holdout   : ES={len(es_2025):,}, NQ={len(nq_2025):,} barras")

    # 2. Generar Clave Maestra de Bóveda
    master_key = generate_master_vault_key()
    print(f"\n[2/5] Clave Maestra Generada para el Orquestador:\n  >>> {master_key} <<<")

    # 3. Cifrado en Bóveda
    print("\n[3/5] Cifrando particiones 2024 (OOS) y 2025 (Holdout) con AES-256-GCM...")
    oos_container = {"ES": es_2024, "NQ": nq_2024}
    holdout_container = {"ES": es_2025, "NQ": nq_2025}

    oos_enc = vault_dir / "oos_2024_5m.enc"
    holdout_enc = vault_dir / "holdout_2025_5m.enc"

    encrypt_dataset_to_vault(oos_container, oos_enc, master_key)
    encrypt_dataset_to_vault(holdout_container, holdout_enc, master_key)

    # También cifrar archivos individuales para conveniencia directa
    encrypt_dataset_to_vault(es_2024, vault_dir / "oos_2024_es_5m.enc", master_key)
    encrypt_dataset_to_vault(nq_2024, vault_dir / "oos_2024_nq_5m.enc", master_key)
    encrypt_dataset_to_vault(es_2025, vault_dir / "holdout_2025_es_5m.enc", master_key)
    encrypt_dataset_to_vault(nq_2025, vault_dir / "holdout_2025_nq_5m.enc", master_key)

    plant_canary_trap(vault_dir)

    print("  [+] Bóveda 2024 OOS creada:", oos_enc)
    print("  [+] Bóveda 2025 Holdout creada:", holdout_enc)
    print("  [+] Honeypot Canary plantado en data/vault/canary_trap.enc")

    # Verificar descifrado en RAM antes de truncar
    dec_oos_es = decrypt_vault_to_memory(oos_enc, master_key, symbol="ES")
    dec_hold_nq = decrypt_vault_to_memory(holdout_enc, master_key, symbol="NQ")
    assert dec_oos_es.equals(es_2024), "Error de integridad en descifrado OOS ES"
    assert dec_hold_nq.equals(nq_2025), "Error de integridad en descifrado Holdout NQ"
    print("  [OK] Verificacion de integridad bit a bit en RAM EXITOSA.")

    # 4. Truncamiento físico de data/raw/ a 2023 exclusivamente
    print("\n[4/5] Truncando físicamente data/raw/ a 2023 exclusivamente...")
    es_2023.to_parquet(es_parquet)
    nq_2023.to_parquet(nq_parquet)
    if es_csv.exists():
        es_2023.to_csv(es_csv)
    if nq_csv.exists():
        nq_2023.to_csv(nq_csv)

    print(f"  [+] {es_parquet.name} truncado a {len(es_2023):,} barras (solo 2023).")
    print(f"  [+] {nq_parquet.name} truncado a {len(nq_2023):,} barras (solo 2023).")

    # 5. Mover reportes OOS y Holdout hacia reports/vault/
    print("\n[5/5] Purgando reportes de OOS y Holdout hacia reports/vault/...")
    purged_count = 0
    for f in reports_dir.glob("*"):
        if f.is_file() and any(x in f.name.lower() for x in ["oos", "holdout"]):
            dest = reports_vault_dir / f.name
            shutil.move(str(f), str(dest))
            purged_count += 1
            print(f"  [Mover] {f.name} -> reports/vault/{f.name}")

    print(f"  [+] Total archivos de reportes purgados: {purged_count}")
    print("\n" + "=" * 78)
    print(" PROCESO COMPLETADO EXITOSAMENTE")
    print(f" GUARDA LA CLAVE DEL ORQUESTADOR: {master_key}")
    print("=" * 78)


if __name__ == "__main__":
    main()
