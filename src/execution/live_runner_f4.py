"""
src/execution/live_runner_f4.py
Motor de ejecución en vivo y paper trading para la estrategia de
Cointegración Ornstein-Uhlenbeck (F4) conectado a NinjaTrader 8.
"""

import sys
from pathlib import Path
import logging
import argparse
import time
from datetime import datetime
import numpy as np
import pandas as pd

# Añadir raíz al sys.path para importar módulos locales
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.strategies.cointegration_ou import CointegrationOUStrategy
from src.execution.ninjatrader_ati import NinjaTraderATIClient
from src.execution.data_receiver import LiveDataBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("live_execution_f4.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("LiveRunnerF4")


class LiveRunnerF4:
    """
    Orquestador de trading en vivo para la estrategia F4.
    """

    def __init__(
        self,
        account: str = "Sim101",
        mode: str = "file",
        dry_run: bool = False,
        symbol_nq: str = "MNQ",
        symbol_es: str = "MES",
    ):
        self.account = account
        self.dry_run = dry_run
        self.symbol_nq = symbol_nq
        self.symbol_es = symbol_es

        self.ati = NinjaTraderATIClient(account=account, mode=mode)
        self.buffer = LiveDataBuffer()
        self.strategy = CointegrationOUStrategy(
            params={
                "k_calib": 780,
                "s_open": 1.25,
                "s_close": 0.0,
                "s_stop": 3.25,
                "adf_t_threshold": -2.86,
                "tau_max": 78.0 * 5.0,  # Minutos
            }
        )

        self.current_position = 0  # 0: Flat, +1: Long spread (Long NQ / Short ES), -1: Short spread
        self.position_sub_baskets = 0

    def initialize(self) -> None:
        """
        Carga el historial previo para arrancar con el modelo calibrado.
        """
        logger.info("Iniciando precarga de datos históricos para F4...")
        self.buffer.preload_historical_parquet()
        df_x, df_y = self.buffer.get_aligned_data()
        logger.info("Buffer listo con %d barras alineadas.", len(df_x))

    def evaluate_live_market(self) -> dict:
        """
        Calcula el estado actual del spread de cointegración con las barras del buffer.
        """
        df_es, df_nq = self.buffer.get_aligned_data()
        k_calib = self.strategy.params.get("k_calib", 780)
        
        if len(df_es) < 200:
            logger.warning("Barras insuficientes para evaluar cointegración: %d", len(df_es))
            return {"status": "INSUFFICIENT_DATA"}

        # Extraer precios de cierre recientes
        hist_nq = df_nq["close"].tail(k_calib)
        hist_es = df_es["close"].tail(k_calib)

        log_hist_nq = np.log(hist_nq.values)
        log_hist_es = np.log(hist_es.values)

        # 1. Cointegración OLS
        beta, alpha, res_hist = self.strategy.model.estimate_cointegration(log_hist_nq, log_hist_es)

        # 2. Proceso Ornstein-Uhlenbeck
        ou_params = self.strategy.model.fit_ou_process(res_hist, dt_minutes=5.0)

        # 3. S-score actual
        p_nq_cur = float(hist_nq.iloc[-1])
        p_es_cur = float(hist_es.iloc[-1])
        res_cur = np.log(p_nq_cur) - (alpha + beta * np.log(p_es_cur))
        s_score = self.strategy.model.calculate_s_score(res_cur, ou_params["theta"], ou_params["sigma_eq"])

        # 4. Dimensionamiento de sub-cesta real (1 MNQ : 4 MES)
        coint_ratio = beta * (p_nq_cur * 2.0) / (p_es_cur * 5.0)
        qty_nq = 1
        qty_es = max(1, int(round(coint_ratio)))

        # 5. Test de Estacionariedad ADF
        is_stat, t_stat, p_val = self.strategy.model.test_stationarity(
            res_hist, t_stat_threshold=self.strategy.params.get("adf_t_threshold", -2.86)
        )

        return {
            "status": "OK",
            "p_nq": p_nq_cur,
            "p_es": p_es_cur,
            "beta": beta,
            "coint_ratio": coint_ratio,
            "sub_basket": (qty_nq, qty_es),
            "ou_valid": ou_params["is_valid"],
            "tau_half_min": ou_params.get("tau_half_min", 0.0),
            "s_score": s_score,
            "is_stat": is_stat,
            "adf_t_stat": t_stat,
            "timestamp": df_es.index[-1],
        }

    def execute_signal(self, target_position: int, state: dict) -> None:
        """
        Gestiona la transición de posición enviando órdenes a NinjaTrader ATI.
        """
        if target_position == self.current_position:
            return

        logger.info(
            "CAMBIO DE POSICIÓN: %d -> %d | S-Score: %.2f | Sub-cesta: %s",
            self.current_position,
            target_position,
            state.get("s_score", 0.0),
            state.get("sub_basket", (1, 4)),
        )

        # 1. Si la señal es cerrar a FLAT (0)
        if target_position == 0:
            logger.info("Cerrando posición abierta en F4...")
            if not self.dry_run:
                self.ati.close_strategy("F4_StatArb")
                self.ati.close_position(self.symbol_nq)
                self.ati.close_position(self.symbol_es)
            self.current_position = 0
            self.position_sub_baskets = 0
            return

        # 2. Si es apertura nueva
        qty_nq, qty_es = state.get("sub_basket", (1, 4))

        if target_position == 1:
            # Long spread: Long NQ, Short ES
            action_nq = "BUY"
            action_es = "SELL"
        else:
            # Short spread: Short NQ, Long ES
            action_nq = "SELL"
            action_es = "BUY"

        logger.info(
            "DISPARANDO SPREAD F4: %s %d %s y %s %d %s",
            action_nq,
            qty_nq,
            self.symbol_nq,
            action_es,
            qty_es,
            self.symbol_es,
        )

        if not self.dry_run:
            # Enviar primero NQ y luego ES inmediatamente
            self.ati.place_market_order(self.symbol_nq, action_nq, qty_nq, strategy_id="F4_StatArb")
            self.ati.place_market_order(self.symbol_es, action_es, qty_es, strategy_id="F4_StatArb")

        self.current_position = target_position
        self.position_sub_baskets = 1

    def run_dry_run_check(self) -> None:
        """
        Prueba rápida de validación sin bucle infinito.
        """
        logger.info("=== EJECUTANDO DRY-RUN CHECK DE F4 ===")
        self.initialize()
        state = self.evaluate_live_market()
        logger.info(
            "Diagnóstico Spread | NQ: %.2f | ES: %.2f | Beta: %.4f | S-Score: %.2f | ADF: %s | Sub-cesta: %s",
            state.get("p_nq", 0.0),
            state.get("p_es", 0.0),
            state.get("beta", 0.0),
            state.get("s_score", 0.0),
            state.get("is_stat", False),
            state.get("sub_basket", (1, 4)),
        )
        logger.info("Prueba de formateo de orden ATI Long Spread:")
        self.execute_signal(1, state)
        logger.info("Prueba de cierre de orden ATI:")
        self.execute_signal(0, state)
        logger.info("=== DRY-RUN COMPLETADO EXITOSAMENTE ===")


def main():
    parser = argparse.ArgumentParser(description="Live Runner F4 Cointegration OU")
    parser.add_argument("--account", type=str, default="Sim101", help="Cuenta NinjaTrader (ej. Sim101)")
    parser.add_argument("--mode", type=str, choices=["file", "socket"], default="file", help="Modo ATI")
    parser.add_argument("--dry-run", action="store_true", help="Modo simulación sin enviar órdenes reales")
    args = parser.parse_args()

    runner = LiveRunnerF4(account=args.account, mode=args.mode, dry_run=args.dry_run)
    if args.dry_run:
        runner.run_dry_run_check()
    else:
        runner.initialize()
        logger.info("Runner listo. Esperando eventos de mercado...")


if __name__ == "__main__":
    main()
