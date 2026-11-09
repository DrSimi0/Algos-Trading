"""
src/execution/ninjatrader_ati.py
Cliente de interfaz automatizada (ATI) para NinjaTrader 8.
Permite enviar órdenes a NinjaTrader tanto por archivo (.txt en incoming/)
como por socket TCP local (puerto por defecto 36973).
"""

from pathlib import Path
import socket
import logging
import time
from typing import Optional

logger = logging.getLogger("NinjaTraderATI")


class NinjaTraderATIClient:
    """
    Controlador para el Automated Trading Interface (ATI) de NinjaTrader 8.
    Comandos soportados según especificación oficial de NinjaTrader:
    PLACE, CANCEL, CANCELALLORDERS, CHANGE, CLOSEPOSITION, CLOSESTRATEGY.
    """

    def __init__(
        self,
        account: str = "Sim101",
        mode: str = "file",
        incoming_dir: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 36973,
    ):
        """
        :param account: Nombre de la cuenta (ej. 'Sim101' para demo).
        :param mode: 'file' (escribe en incoming/) o 'socket' (conecta por TCP).
        :param incoming_dir: Ruta al directorio 'incoming' de NinjaTrader 8.
        :param host: IP del host ATI (por defecto localhost).
        :param port: Puerto TCP ATI (por defecto 36973).
        """
        self.account = account
        self.mode = mode
        self.host = host
        self.port = port

        if incoming_dir:
            self.incoming_path = Path(incoming_dir)
        else:
            # Buscar en ubicaciones habituales de Windows (incluyendo OneDrive)
            home = Path.home()
            candidates = [
                home / "Documents" / "NinjaTrader 8" / "incoming",
                home / "OneDrive" / "Documents" / "NinjaTrader 8" / "incoming",
                home / "OneDrive" / "Documentos" / "NinjaTrader 8" / "incoming",
                Path("C:/NinjaTrader 8/incoming"),
            ]
            found = None
            for c in candidates:
                if c.parent.exists():
                    found = c
                    break
            self.incoming_path = found or candidates[0]

        # Asegurar que la carpeta exista si operamos en modo archivo
        if self.mode == "file":
            self.incoming_path.mkdir(parents=True, exist_ok=True)
            logger.info("NinjaTrader ATI inicializado en modo FILE: %s", self.incoming_path)
        else:
            logger.info("NinjaTrader ATI inicializado en modo SOCKET: %s:%d", self.host, self.port)

    def send_command(self, command: str) -> bool:
        """
        Envía un comando crudo ATI a NinjaTrader.
        """
        command = command.strip() + "\r\n"
        if self.mode == "file":
            try:
                filename = f"order_{int(time.time() * 1000)}_{time.perf_counter_ns() % 10000}.txt"
                filepath = self.incoming_path / filename
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(command)
                logger.debug("Comando ATI escrito a %s: %s", filepath.name, command.strip())
                return True
            except Exception as e:
                logger.error("Error al escribir archivo de comando ATI: %s", e)
                return False
        else:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2.0)
                    s.connect((self.host, self.port))
                    s.sendall(command.encode("utf-8"))
                logger.debug("Comando ATI enviado por socket: %s", command.strip())
                return True
            except Exception as e:
                logger.error("Error al enviar comando ATI por socket: %s", e)
                return False

    def place_market_order(
        self,
        symbol: str,
        action: str,
        qty: int,
        order_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
    ) -> bool:
        """
        Envía una orden Market.
        Sintaxis NT8: PLACE;<Account>;<Instrument>;<Action>;<Qty>;<OrderType>;<LimitPrice>;<StopPrice>;<Tif>;<Oco>;<OrderId>;<Strategy>;<StrategyId>
        """
        act = action.upper()
        if act not in ("BUY", "SELL"):
            raise ValueError(f"Acción inválida: {action}. Debe ser BUY o SELL.")
        
        oid = order_id or f"ORD_{int(time.time() * 1000)}"
        strat = strategy_id or "F4_StatArb"
        
        cmd = f"PLACE;{self.account};{symbol};{act};{qty};MARKET;;;DAY;;{oid};;{strat}"
        logger.info("Enviando orden MARKET: %s %d %s en %s", act, qty, symbol, self.account)
        return self.send_command(cmd)

    def close_position(self, symbol: str) -> bool:
        """
        Cierra cualquier posición abierta en el símbolo especificado.
        Sintaxis NT8: CLOSEPOSITION;<Account>;<Instrument>
        """
        cmd = f"CLOSEPOSITION;{self.account};{symbol}"
        logger.info("Cerrando posición en %s (%s)", symbol, self.account)
        return self.send_command(cmd)

    def close_strategy(self, strategy_id: str = "F4_StatArb") -> bool:
        """
        Cierra todas las órdenes y posiciones de una estrategia específica.
        Sintaxis NT8: CLOSESTRATEGY;<StrategyId>
        """
        cmd = f"CLOSESTRATEGY;{strategy_id}"
        logger.info("Cerrando estrategia completa ATI: %s", strategy_id)
        return self.send_command(cmd)

    def emergency_flatten_all(self, symbols=("MNQ", "MES")) -> None:
        """
        Protocolo de pánico o cierre diario forzado EOD (15:50 EST):
        Cierra todas las posiciones en los símbolos operados.
        """
        logger.warning("DISPARANDO PROTOCOLO DE CIERRE TOTAL (EMERGENCY FLATTEN)...")
        self.close_strategy("F4_StatArb")
        for sym in symbols:
            self.close_position(sym)
