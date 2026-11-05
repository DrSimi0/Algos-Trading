"""
src/core/futures_specs.py
Especificaciones de contratos de futuros CME.
- multiplier: dólares por 1 punto de movimiento del precio.
- tick: movimiento mínimo de precio.
- commission_rt: costo ida y vuelta (round-turn) aproximado por contrato, todo incluido
  (broker + exchange + NFA). Verificar con tu broker / prop firm.
- yf: ticker continuo en Yahoo Finance (gratis, útil para prototipos diarios).
"""

from typing import Dict, Any

FUTURES_SPECS: Dict[str, Dict[str, Any]] = {
    # Micro E-mini (CME / CBOT / NYMEX / COMEX)
    "MES": {
        "multiplier": 5.0,
        "tick": 0.25,
        "commission_rt": 1.24,
        "yf": "MES=F",
    },
    "MNQ": {
        "multiplier": 2.0,
        "tick": 0.25,
        "commission_rt": 1.24,
        "yf": "MNQ=F",
    },
    "M2K": {
        "multiplier": 5.0,
        "tick": 0.1,
        "commission_rt": 1.24,
        "yf": "M2K=F",
    },
    "MYM": {
        "multiplier": 0.5,
        "tick": 1.0,
        "commission_rt": 1.24,
        "yf": "MYM=F",
    },
    "MGC": {
        "multiplier": 10.0,
        "tick": 0.1,
        "commission_rt": 1.6,
        "yf": "MGC=F",
    },
    "MCL": {
        "multiplier": 100.0,
        "tick": 0.01,
        "commission_rt": 1.24,
        "yf": "MCL=F",
    },

    # Estándar E-mini / Full
    "ES": {
        "multiplier": 50.0,
        "tick": 0.25,
        "commission_rt": 4.5,
        "yf": "ES=F",
    },
    "NQ": {
        "multiplier": 20.0,
        "tick": 0.25,
        "commission_rt": 4.5,
        "yf": "NQ=F",
    },
    "CL": {
        "multiplier": 1000.0,
        "tick": 0.01,
        "commission_rt": 4.5,
        "yf": "CL=F",
    },
    "GC": {
        "multiplier": 100.0,
        "tick": 0.1,
        "commission_rt": 4.5,
        "yf": "GC=F",
    },
}


def get_spec(symbol: str) -> dict:
    """
    Obtiene las especificaciones de un contrato de futuros de CME por símbolo.
    
    Args:
        symbol (str): Ticker del contrato (ej. 'ES', 'MES', 'NQ', 'MNQ', 'MES=F').
        
    Returns:
        dict: Diccionario con multiplier, tick, commission_rt y yf.
        
    Raises:
        KeyError: Si el símbolo no está registrado en FUTURES_SPECS.
    """
    key = symbol.upper()
    if key.endswith("=F") and key[:-2] in FUTURES_SPECS:
        key = key[:-2]
    if key not in FUTURES_SPECS:
        raise KeyError(f"Contrato '{symbol}' no definido. Disponibles: {list(FUTURES_SPECS)}")
    return FUTURES_SPECS[key]
