"""
scripts/run_ofi_backtest.py
Forwarding runner para la Estrategia F5: Order Flow Imbalance de Nivel 1.
Ejecuta run_ofi_backtest.py desde el directorio scripts/.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import run_ofi_backtest

if __name__ == "__main__":
    run_ofi_backtest.main()
