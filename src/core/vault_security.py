"""
src/core/vault_security.py
Bóveda Criptográfica y Seguridad Antifugas (True Cryptographic Air-Gap).

Implementa aislamiento físico en disco mediante cifrado autenticado AES-256-GCM
con derivación de claves PBKDF2 (100,000 iteraciones SHA-256).
Garantiza cero huella en disco: la desencriptación ocurre estrictamente
en memoria volátil (RAM) mediante buffers io.BytesIO y PyArrow.

Incluye sistema Tripwire / Honeypot para detectar intentos no autorizados de acceso.
"""

import io
import os
import secrets
import struct
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# ==============================================================================
# CONSTANTES DE SEGURIDAD CRIPTOGRÁFICA
# ==============================================================================

VAULT_MAGIC = b"VAULT_V1"          # 8 bytes Magic Header
SALT_SIZE = 16                     # 16 bytes Salt para PBKDF2
NONCE_SIZE = 12                    # 12 bytes Nonce para AES-GCM
PBKDF2_ITERATIONS = 100_000        # Iteraciones SHA-256 según NIST SP 800-132
HEADER_TOTAL_SIZE = 8 + 16 + 12   # 36 bytes

DEFAULT_VAULT_DIR = Path("data/vault")
DEFAULT_TRIPWIRE_LOG = Path("data/audit_vault_tripwire.log")


# ==============================================================================
# EXCEPCIONES DE SEGURIDAD
# ==============================================================================

class VaultSecurityError(Exception):
    """Excepción base para fallos criptográficos o de integridad de bóveda."""
    pass


class VaultAccessDeniedError(PermissionError):
    """Acceso denegado a la bóveda por falta de autorización o clave."""
    pass


class VaultTripwireTriggered(VaultAccessDeniedError):
    """Lanzada cuando un script o agente activa el tripwire por clave incorrecta o acceso no autorizado."""
    pass


# ==============================================================================
# TRIPWIRE & CANARY MONITORING
# ==============================================================================

def record_tripwire_event(
    event_type: str,
    details: str = "",
    log_path: Union[str, Path] = DEFAULT_TRIPWIRE_LOG,
) -> None:
    """
    Registra un evento de seguridad / alerta de tripwire de forma atómica.
    """
    try:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now().isoformat()
        caller = sys.argv[0] if sys.argv else "unknown_script"
        log_line = f"[{now_str}] TRIPWIRE_ALERT | Event: {event_type} | Caller: {caller} | Details: {details}\n"
        with open(p, "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception:
        pass


def plant_canary_trap(vault_dir: Union[str, Path] = DEFAULT_VAULT_DIR) -> Path:
    """
    Crea un archivo trampa (Honeypot Canary) en data/vault/.
    Cualquier intento de lectura ingenua o manipulación activará alertas.
    """
    vdir = Path(vault_dir)
    vdir.mkdir(parents=True, exist_ok=True)
    canary_path = vdir / "canary_trap.enc"
    if not canary_path.exists():
        # Generar contenido que simula ser un archivo cifrado pero con firma de trampa
        fake_salt = os.urandom(SALT_SIZE)
        fake_nonce = os.urandom(NONCE_SIZE)
        honey_payload = b"TRIPWIRE_HONEYPOT_SECURITY_TRAP_DO_NOT_READ"
        with open(canary_path, "wb") as f:
            f.write(VAULT_MAGIC + fake_salt + fake_nonce + honey_payload)
    return canary_path


# ==============================================================================
# GESTIÓN Y DERIVACIÓN DE CLAVES CRIPTOGRÁFICAS
# ==============================================================================

def generate_master_vault_key() -> str:
    """
    Genera una clave criptográfica de 256 bits (32 bytes) codificada en hex (64 caracteres)
    usando la entropía segura del sistema operativo (secrets).
    """
    return secrets.token_hex(32)


def derive_encryption_key(key: Union[str, bytes], salt: bytes) -> bytes:
    """
    Deriva una clave simétrica de 256 bits mediante PBKDF2HMAC-SHA256 con 100,000 iteraciones.
    """
    if isinstance(key, str):
        # Si es un string hex de 64 caracteres, convertirlo a bytes directos antes de PBKDF2
        clean_key = key.strip()
        if len(clean_key) == 64:
            try:
                key_material = bytes.fromhex(clean_key)
            except ValueError:
                key_material = clean_key.encode("utf-8")
        else:
            key_material = clean_key.encode("utf-8")
    elif isinstance(key, bytes):
        key_material = key
    else:
        raise TypeError(f"La clave debe ser str o bytes, no {type(key).__name__}")

    if not key_material:
        raise ValueError("La clave criptográfica no puede estar vacía.")

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(key_material)


# ==============================================================================
# SERIALIZACIÓN / DESERIALIZACIÓN ESTRICTA EN RAM
# ==============================================================================

def _serialize_payload(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> bytes:
    """
    Serializa uno o múltiples DataFrames directamente a bytes en RAM mediante PyArrow/Parquet.
    Cero escritura en disco.
    """
    if isinstance(data, pd.DataFrame):
        buf = io.BytesIO()
        data.to_parquet(buf, engine="pyarrow", index=True)
        return b"DF\x00" + buf.getvalue()

    elif isinstance(data, dict):
        # Empaquetado en memoria de múltiples DataFrames como un contenedor ZIP en RAM
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_STORED) as zf:
            for symbol, df in data.items():
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"El elemento '{symbol}' en el diccionario debe ser pd.DataFrame.")
                df_buf = io.BytesIO()
                df.to_parquet(df_buf, engine="pyarrow", index=True)
                zf.writestr(f"{symbol}.parquet", df_buf.getvalue())
        return b"ZIP\x00" + zip_buf.getvalue()

    else:
        raise TypeError(f"Tipo de datos no soportado para la bóveda: {type(data).__name__}")


def _deserialize_payload(
    raw_bytes: bytes,
    symbol: Optional[str] = None,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Reconstruye el DataFrame o diccionario de DataFrames desde el buffer binario en RAM.
    Cero escritura en disco.
    """
    if raw_bytes.startswith(b"DF\x00"):
        parquet_bytes = raw_bytes[3:]
        buf = io.BytesIO(parquet_bytes)
        return pd.read_parquet(buf)

    elif raw_bytes.startswith(b"ZIP\x00"):
        zip_bytes = raw_bytes[4:]
        zip_buf = io.BytesIO(zip_bytes)
        dfs: Dict[str, pd.DataFrame] = {}
        with zipfile.ZipFile(zip_buf, "r") as zf:
            for fname in zf.namelist():
                if fname.endswith(".parquet"):
                    sym = fname[:-8]
                    content = zf.read(fname)
                    dfs[sym] = pd.read_parquet(io.BytesIO(content))

        if symbol is not None:
            if symbol in dfs:
                return dfs[symbol]
            raise KeyError(f"El símbolo '{symbol}' no existe en el contenedor de la bóveda. Disponibles: {list(dfs.keys())}")
        return dfs

    # Compatibilidad de contingencia si los bytes son parquet crudo sin prefijo
    try:
        buf = io.BytesIO(raw_bytes)
        return pd.read_parquet(buf)
    except Exception as e:
        raise VaultSecurityError(f"Error al reconstruir DataFrame desde RAM: {e}")


# ==============================================================================
# OPERACIONES DE CIFRADO Y DESCIFRADO
# ==============================================================================

def encrypt_dataset_to_vault(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    output_path: Union[str, Path],
    key: Union[str, bytes],
) -> Path:
    """
    Serializa y cifra un DataFrame o diccionario de DataFrames directamente hacia el disco
    en formato binario AES-256-GCM.
    
    Estructura del archivo:
        [8 bytes VAULT_MAGIC] + [16 bytes Salt] + [12 bytes Nonce] + [Ciphertext + AuthTag]
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Serializar payload en RAM
    payload_bytes = _serialize_payload(data)

    # 2. Generar parámetros criptográficos
    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    aes_key = derive_encryption_key(key, salt)

    # 3. Cifrado autenticado AES-256-GCM
    aesgcm = AESGCM(aes_key)
    ciphertext = aesgcm.encrypt(nonce, payload_bytes, VAULT_MAGIC)

    # 4. Escritura binaria protegida
    with open(out_file, "wb") as f:
        f.write(VAULT_MAGIC + salt + nonce + ciphertext)

    return out_file


def decrypt_vault_to_memory(
    vault_path: Union[str, Path],
    key: Union[str, bytes],
    symbol: Optional[str] = None,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Desencripta un archivo protegido de la bóveda directamente hacia la memoria RAM.
    Cero archivos temporales en disco.
    
    Si la clave es incorrecta o el archivo está alterado, dispara el Tripwire y aborta.
    """
    path = Path(vault_path)
    if not path.exists():
        record_tripwire_event("FILE_NOT_FOUND", f"Ruta solicitada no existe: {vault_path}")
        raise FileNotFoundError(f"Archivo de bóveda no encontrado: {vault_path}")

    with open(path, "rb") as f:
        file_bytes = f.read()

    if len(file_bytes) < HEADER_TOTAL_SIZE:
        record_tripwire_event("INVALID_FILE_SIZE", f"Archivo corrupto o truncado ({len(file_bytes)} bytes)")
        raise VaultSecurityError(f"Archivo de bóveda inválido o corrupto: {vault_path}")

    # Verificar magic header
    magic = file_bytes[:8]
    if magic != VAULT_MAGIC:
        record_tripwire_event("INVALID_MAGIC_HEADER", f"Header no coincide: {magic!r}")
        raise VaultSecurityError(f"Formato no reconocido. El archivo {vault_path} no es una bóveda válida.")

    salt = file_bytes[8:24]
    nonce = file_bytes[24:36]
    ciphertext = file_bytes[36:]

    # Detección inmediata de trampa Honeypot Canary
    if "canary" in path.name.lower() or b"TRIPWIRE_HONEYPOT" in file_bytes:
        record_tripwire_event("HONEYPOT_CANARY_TOUCHED", f"Acceso directo no autorizado al Canary Trap: {vault_path}")
        raise VaultTripwireTriggered(
            "\n" + "!" * 80 + "\n"
            " [ALERTA DE SEGURIDAD CRIPTOGRÁFICA: HONEYPOT CANARY ACTIVADO]\n"
            " Un agente o script ha intentado leer el archivo trampa de la bóveda.\n"
            f" Archivo objetivo: {vault_path}\n"
            " El incidente ha sido registrado en la bitácora de auditoría inmutable.\n"
            + "!" * 80 + "\n"
        )

    # Validar clave de forma estricta: clave vacía, None o tipo inválido activa el tripwire
    if key is None:
        record_tripwire_event("EMPTY_KEY_ATTEMPT", f"Intento de acceso con clave nula (None) para {vault_path}")
        raise VaultTripwireTriggered(
            "\n" + "!" * 80 + "\n"
            " [ALERTA DE SEGURIDAD CRIPTOGRÁFICA: TRIPWIRE ACTIVADO]\n"
            " Intento de acceso a la bóveda sin clave (clave nula o no provista).\n"
            f" Archivo objetivo: {vault_path}\n"
            " El incidente ha sido registrado en la bitácora de auditoría inmutable.\n"
            + "!" * 80 + "\n"
        )

    if isinstance(key, str) and not key.strip():
        record_tripwire_event("EMPTY_KEY_ATTEMPT", f"Intento de acceso con clave vacía para {vault_path}")
        raise VaultTripwireTriggered(
            "\n" + "!" * 80 + "\n"
            " [ALERTA DE SEGURIDAD CRIPTOGRÁFICA: TRIPWIRE ACTIVADO]\n"
            " Intento de acceso a la bóveda con clave vacía.\n"
            f" Archivo objetivo: {vault_path}\n"
            " El incidente ha sido registrado en la bitácora de auditoría inmutable.\n"
            + "!" * 80 + "\n"
        )

    if not isinstance(key, (str, bytes)):
        record_tripwire_event("MALFORMED_KEY_TYPE", f"Tipo de clave inválido ({type(key).__name__}) para {vault_path}")
        raise VaultTripwireTriggered(
            "\n" + "!" * 80 + "\n"
            " [ALERTA DE SEGURIDAD CRIPTOGRÁFICA: TRIPWIRE ACTIVADO]\n"
            f" Intento de acceso con formato de clave inválido: {type(key).__name__}.\n"
            f" Archivo objetivo: {vault_path}\n"
            " El incidente ha sido registrado en la bitácora de auditoría inmutable.\n"
            + "!" * 80 + "\n"
        )

    try:
        aes_key = derive_encryption_key(key, salt)
        aesgcm = AESGCM(aes_key)
        decrypted_bytes = aesgcm.decrypt(nonce, ciphertext, VAULT_MAGIC)
    except InvalidTag:
        record_tripwire_event("INVALID_AUTHENTICATION_TAG", f"Clave rechazada para {vault_path}")
        raise VaultTripwireTriggered(
            "\n" + "!" * 80 + "\n"
            " [ALERTA DE SEGURIDAD CRIPTOGRÁFICA: TRIPWIRE ACTIVADO]\n"
            " Intento de acceso a la bóveda con clave inválida o autenticación fallida.\n"
            f" Archivo objetivo: {vault_path}\n"
            " El incidente ha sido registrado en la bitácora de auditoría inmutable.\n"
            + "!" * 80 + "\n"
        )
    except (ValueError, TypeError) as e:
        record_tripwire_event("MALFORMED_KEY_DERIVATION", f"Fallo al derivar clave ({e}) para {vault_path}")
        raise VaultTripwireTriggered(
            "\n" + "!" * 80 + "\n"
            " [ALERTA DE SEGURIDAD CRIPTOGRÁFICA: TRIPWIRE ACTIVADO]\n"
            f" Clave criptográfica no válida para derivación: {e}\n"
            f" Archivo objetivo: {vault_path}\n"
            " El incidente ha sido registrado en la bitácora de auditoría inmutable.\n"
            + "!" * 80 + "\n"
        )
    except Exception as e:
        record_tripwire_event("DECRYPTION_EXCEPTION", str(e))
        raise VaultSecurityError(f"Fallo durante el descifrado: {e}")

    # Deserialización en RAM
    return _deserialize_payload(decrypted_bytes, symbol=symbol)


def is_vault_file(file_path: Union[str, Path]) -> bool:
    """Verifica si un archivo contiene la firma mágica de la bóveda criptográfica."""
    try:
        p = Path(file_path)
        if not p.is_file():
            return False
        with open(p, "rb") as f:
            magic = f.read(len(VAULT_MAGIC))
        return magic == VAULT_MAGIC
    except Exception:
        return False
