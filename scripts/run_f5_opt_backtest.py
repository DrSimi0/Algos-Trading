"""
scripts/run_f5_opt_backtest.py
Forwarding runner para la Estrategia F5-Optimizacion -IS- N1.
Ejecuta run_f5_opt_backtest.py desde el directorio scripts/.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import run_f5_opt_backtest

if __name__ == "__main__":
    run_f5_opt_backtest.main()
