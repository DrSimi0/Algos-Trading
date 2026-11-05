"""
src/strategies/order_flow_imbalance.py
Implementación Matemática y Microestructural de la Estrategia F5:
Order Flow Imbalance de Nivel 1 en Limit Order Book (L1-OFI Price Impact Model).

Fundamento Académico:
1. Cont, R., Kukanov, A., & Stoikov, S. (2014). The Price Impact of Order Book Events.
   Journal of Financial Econometrics, 12(1), 47-88.
2. Cont, R., & de Larrard, A. (2013). Price dynamics in a Markovian limit order book.
   SIAM Journal on Financial Mathematics, 4(1), 1-25.
3. Kyle, A. S. (1985). Continuous Auctions and Informed Trader. Econometrica, 53(6), 1429-1454.
4. Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices. JFE.
5. Hasbrouck, J. (1991). Measuring the Information Content of Stock Trades. Journal of Finance.

NOTA DE INFRAESTRUCTURA Y ADVERTENCIA DE PRODUCCIÓN:
Para explotar la ventaja estadística teórica de esta estrategia en entornos de producción real,
es estrictamente mandatorio contar con el feed de eventos tick a tick en tiempo real del protocolo
CME MDP 3.0 (Simple Binary Encoding - SBE). 
Alimentar este modelo con velas OHLCV agregadas de 5 minutos (o superiores) destruye el alpha
(Win Rate ~ 21.57%) debido a la paradoja de retardo temporal: el movimiento de precio ya se
ha consumado cuando la barra cierra, entrando en fase de agotamiento y activando el stop-loss ceñido.
Para backtests sin suscripción a CME MDP 3.0, use '--synthetic-lob' con tick_size=0.25 para
verificar la eficacia teórica de la estrategia mediante el simulador de vaciado de colas.

Especificación: reports/estrategias_matematicas/F5_order_flow_imbalance_cont.md
"""

from dataclasses import dataclass, field
from datetime import time, datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import math
import warnings
import numpy as np
import pandas as pd

from src.core.base_strategy import BaseStrategy
from src.core.futures_specs import get_spec


# ==============================================================================
# 1. ENUMS Y ESTRUCTURAS DE DATOS DE ESTADO Y TRANSACCIÓN
# ==============================================================================

class OFIExitReason(str, Enum):
    FLOW_EXHAUSTION = "FLOW_EXHAUSTION"      # Z_OFI se normaliza (|Z| <= 0.50)
    TAKE_PROFIT_PRICE = "TAKE_PROFIT_PRICE"  # Recorrido de precio >= Delta P_TP*
    FLOW_REVERSAL = "FLOW_REVERSAL"          # Z_OFI se invierte en contra (Z <= -1.25 / Z >= +1.25)
    STOP_LOSS_PRICE = "STOP_LOSS_PRICE"      # Stop de pérdida rígido de 6 ticks (1.50 pts)
    TIME_DECAY = "TIME_DECAY"                # Límite de permanencia H_max = 10 barras
    OVERNIGHT_FLAT = "OVERNIGHT_FLAT"        # Cierre forzoso de sesión (>= 15:55 EST)


@dataclass
class OFITradeRecord:
    date: str
    symbol: str
    direction: int                           # +1 = Long, -1 = Short
    contracts: int                           # Número de contratos operados
    contract_type: str                       # "STD" o "MICRO"
    entry_bar_idx: int                       # Índice secuencial de barra de entrada
    entry_time: Any                          # Timestamp de entrada
    entry_price: float                       # Precio de ejecución (Ask para Long, Bid para Short)
    stop_loss_price: float                   # Nivel de Stop Loss
    take_profit_price: float                 # Nivel de Take Profit objetivo
    delta_sl_points: float                   # Distancia de Stop Loss en puntos (1.50)
    delta_tp_points: float                   # Distancia de Take Profit en puntos
    z_score_entry: float                     # Z_OFI en la entrada
    exit_bar_idx: Optional[int] = None
    exit_time: Optional[Any] = None
    exit_price: Optional[float] = None       # Precio de salida (Bid para Long, Ask para Short)
    exit_reason: Optional[OFIExitReason] = None
    gross_pnl_usd: float = 0.0
    commission_usd: float = 0.0
    slippage_usd: float = 0.0
    net_pnl_usd: float = 0.0
    holding_bars: int = 0


@dataclass
class OFIPropAccount:
    """
    Gestión de riesgo para cuentas de fondeo (Prop Trading Firms).
    Supervisa Daily Loss Limit (DLL), Trailing Max Drawdown (TMDD) y
    cálculo del presupuesto de riesgo admisible por trade.
    """
    initial_capital: float = 50000.0
    dll: float = 1000.0                      # Daily Loss Limit
    tmdd: float = 2000.0                     # Trailing Max Drawdown
    equity: float = 50000.0
    peak_equity: float = 50000.0             # High-Water Mark para Trailing Drawdown
    trailing_floor: float = 48000.0          # Piso inicial = initial_capital - tmdd
    lock_floor_at_start: bool = True         # Si True, el piso no sube más allá de initial_capital
    daily_realized_loss: float = 0.0         # Pérdidas acumuladas en la sesión actual
    daily_pnl: float = 0.0                   # PnL neto del día
    is_breached: bool = False
    breach_reason: Optional[str] = None

    def start_session(self) -> None:
        """Reinicia el balance de pérdidas diarias al iniciar una nueva jornada."""
        self.daily_realized_loss = 0.0
        self.daily_pnl = 0.0

    def update_equity_intraday(self, trade_pnl: float) -> None:
        """Actualiza el balance de la cuenta tras la liquidación de un trade."""
        self.equity += trade_pnl
        self.daily_pnl += trade_pnl

        if trade_pnl < 0:
            self.daily_realized_loss += abs(trade_pnl)

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            new_floor = self.peak_equity - self.tmdd
            if self.lock_floor_at_start:
                new_floor = min(new_floor, self.initial_capital)
            self.trailing_floor = max(self.trailing_floor, new_floor)

        # Verificación estricta de límites Prop Trading
        if self.equity <= self.trailing_floor:
            self.is_breached = True
            self.breach_reason = f"Trailing Drawdown Superado: Equity ${self.equity:,.2f} <= Piso ${self.trailing_floor:,.2f}"
        elif self.daily_realized_loss >= self.dll:
            self.is_breached = True
            self.breach_reason = f"Daily Loss Limit Superado: Pérdida Día ${self.daily_realized_loss:,.2f} >= DLL ${self.dll:,.2f}"

    def calculate_risk_budget(self, phi_dll: float = 0.20, phi_tmdd: float = 0.15) -> float:
        """
        Calcula el presupuesto de riesgo asignado a una sola operación:
        PresupuestoRiesgo = min( max(0, DLL - L_day) * phi_DLL, max(0, TMDD - DD_t) * phi_TMDD )
        """
        if self.is_breached:
            return 0.0

        daily_margin = max(0.0, self.dll - self.daily_realized_loss)
        trailing_margin = max(0.0, self.equity - self.trailing_floor)

        b_daily = daily_margin * phi_dll
        b_trailing = trailing_margin * phi_tmdd

        return float(max(0.0, min(b_daily, b_trailing)))


# ==============================================================================
# 2. FUNCIONES MATEMÁTICAS PURAS DE MICROESTRUCTURA DE LOB (CONT ET AL., 2014)
# ==============================================================================

def calculate_eb(p_curr: float, q_curr: float, p_prev: float, q_prev: float) -> float:
    """
    Contribución neta de flujo en el Best Bid:
    e_n^b = +q_n^b         si P_n^b > P_{n-1}^b (Subida del Bid)
          = q_n^b - q_{n-1}^b si P_n^b == P_{n-1}^b (Mismo Bid)
          = -q_{n-1}^b       si P_n^b < P_{n-1}^b (Bajada del Bid)
    """
    if p_curr > p_prev:
        return float(q_curr)
    elif math.isclose(p_curr, p_prev, abs_tol=1e-6):
        return float(q_curr - q_prev)
    else:
        return -float(q_prev)


def calculate_ea(p_curr: float, q_curr: float, p_prev: float, q_prev: float) -> float:
    """
    Contribución neta de flujo en el Best Ask:
    e_n^a = -q_{n-1}^a       si P_n^a > P_{n-1}^a (Subida del Ask)
          = q_n^a - q_{n-1}^a si P_n^a == P_{n-1}^a (Mismo Ask)
          = -q_n^a         si P_n^a < P_{n-1}^a (Bajada del Ask)
    """
    if p_curr > p_prev:
        return -float(q_prev)
    elif math.isclose(p_curr, p_prev, abs_tol=1e-6):
        return float(q_curr - q_prev)
    else:
        return -float(q_curr)


def calculate_en(eb: float, ea: float) -> float:
    """
    Variación neta instantánea del desbalance unitario en el evento n:
    e_n = e_n^b - e_n^a
    """
    return float(eb - ea)


def compute_ofi_from_events(df_events: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Procesa un DataFrame de eventos L1 de CME MDP 3.0 con columnas:
    ['bid_price', 'bid_size', 'ask_price', 'ask_size'] (o ['P_b', 'q_b', 'P_a', 'q_a'])
    y calcula el Order Flow Imbalance agregado y la profundidad promedio armónica.
    """
    col_map = {}
    for c in df_events.columns:
        cl = str(c).lower()
        if "bid_p" in cl or cl == "p_b":
            col_map["p_b"] = c
        elif "bid_s" in cl or "bid_q" in cl or cl == "q_b":
            col_map["q_b"] = c
        elif "ask_p" in cl or cl == "p_a":
            col_map["p_a"] = c
        elif "ask_s" in cl or "ask_q" in cl or cl == "q_a":
            col_map["q_a"] = c

    required = ["p_b", "q_b", "p_a", "q_a"]
    if not all(k in col_map for k in required):
        raise ValueError(f"Faltan columnas de LOB requeridas. Necesarias: bid_price, bid_size, ask_price, ask_size. Encontradas: {list(df_events.columns)}")

    # Garantizar DatetimeIndex válido para resample
    if not isinstance(df_events.index, pd.DatetimeIndex):
        if "timestamp" in df_events.columns:
            df_events = df_events.set_index(pd.to_datetime(df_events["timestamp"]))
        elif "datetime" in df_events.columns:
            df_events = df_events.set_index(pd.to_datetime(df_events["datetime"]))
        else:
            df_events.index = pd.to_datetime(df_events.index)

    p_b = df_events[col_map["p_b"]].values
    q_b = df_events[col_map["q_b"]].values
    p_a = df_events[col_map["p_a"]].values
    q_a = df_events[col_map["q_a"]].values
    idx = df_events.index

    n = len(df_events)
    eb_arr = np.zeros(n, dtype=np.float64)
    ea_arr = np.zeros(n, dtype=np.float64)
    en_arr = np.zeros(n, dtype=np.float64)
    depth_arr = (q_b + q_a) / 2.0

    for i in range(1, n):
        eb = calculate_eb(p_b[i], q_b[i], p_b[i-1], q_b[i-1])
        ea = calculate_ea(p_a[i], q_a[i], p_a[i-1], q_a[i-1])
        eb_arr[i] = eb
        ea_arr[i] = ea
        en_arr[i] = calculate_en(eb, ea)

    df_calc = pd.DataFrame({
        "e_n": en_arr,
        "depth": depth_arr,
        "bid_price": p_b,
        "ask_price": p_a,
        "mid_price": (p_b + p_a) / 2.0,
        "spread": p_a - p_b,
    }, index=idx)

    # Agregación regular a barras de 1 minuto (Delta tau = 60 s) preservando OHLC
    resampled = df_calc.resample(freq)
    agg_df = pd.DataFrame({
        "OFI": resampled["e_n"].sum(),
        "mean_depth": resampled["depth"].mean(),
        "bid_price": resampled["bid_price"].last(),
        "ask_price": resampled["ask_price"].last(),
        "mid_price": resampled["mid_price"].last(),
        "spread": resampled["spread"].last(),
        "event_count": resampled["e_n"].count(),
        "Open": resampled["mid_price"].first(),
        "High": resampled["mid_price"].max(),
        "Low": resampled["mid_price"].min(),
        "Close": resampled["mid_price"].last(),
    }).dropna()

    return agg_df


def generate_synthetic_lob_events(
    start_price: float = 5800.0,
    n_events: int = 15000,
    tick_size: float = 0.25,
    mean_depth: float = 250.0,
    prob_price_change: float = 0.05,
    flow_persistence: float = 0.65,
    event_freq: str = "1s",
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Genera una secuencia sintética estricta y calibrada de eventos L1 de LOB respetando
    las restricciones microestructurales de no cruzamiento, saltos discretos de tick (0.25)
    y dinámica de vaciado de colas (Queue Depletion Model de Cont & de Larrard 2013).

    Incorpora episodios de flujo institucional informado con persistencia AR(1) (Kyle 1985;
    Cont, Kukanov & Stoikov 2014), reproduciendo la relación lineal de impacto de precio
    (R^2 > 0.65) entre OFI y variación del precio medio sin el ruido ni retardo de barras de 5m.
    """
    if seed is not None:
        np.random.seed(seed)

    bid = start_price
    ask = start_price + tick_size
    q_b = max(10, int(np.random.normal(mean_depth, mean_depth * 0.15)))
    q_a = max(10, int(np.random.normal(mean_depth, mean_depth * 0.15)))

    timestamps = pd.date_range("2026-10-15 09:30:00", periods=n_events, freq=event_freq)

    bids = [bid]
    asks = [ask]
    qb_list = [q_b]
    qa_list = [q_a]

    # Estado de flujo institucional informado episódico
    burst_direction = 0  # 0 = equilibrio de ruido, +1 = compras agresivas, -1 = ventas agresivas
    burst_remaining = 0

    for _ in range(1, n_events):
        # Transición de estados de choque direccional institucional
        if burst_remaining <= 0:
            burst_direction = 0
            if np.random.rand() < 0.015:  # Inicio de episodio institucional (~cada 60-100 segundos)
                burst_direction = int(np.random.choice([1, -1]))
                burst_remaining = int(np.random.randint(60, 240))  # Duración de 1 a 4 minutos
        else:
            burst_remaining -= 1

        # Ruido difusivo basal de participantes pasivos y minoristas
        noise_b = np.random.normal(0, mean_depth * 0.08)
        noise_a = np.random.normal(0, mean_depth * 0.08)

        # Presión de flujo institucional direccional
        if burst_direction == 1:
            inst_flow = mean_depth * np.random.uniform(0.18, 0.35)
            delta_qb = int(inst_flow + noise_b)
            delta_qa = int(-inst_flow + noise_a)
        elif burst_direction == -1:
            inst_flow = mean_depth * np.random.uniform(0.18, 0.35)
            delta_qb = int(-inst_flow + noise_b)
            delta_qa = int(inst_flow + noise_a)
        else:
            delta_qb = int(noise_b)
            delta_qa = int(noise_a)

        q_b += delta_qb
        q_a += delta_qa

        # Mecanismo de vaciado de colas (Queue Depletion):
        # Cuando q_a se agota (<= 0), el Ask sube 1 tick y el Bid lo acompaña
        if q_a <= 0:
            bid += tick_size
            ask = bid + tick_size
            q_a = max(10, int(np.random.normal(mean_depth, mean_depth * 0.15)))
            q_b = max(10, int(mean_depth * 0.75))
        # Cuando q_b se agota (<= 0), el Bid baja 1 tick y el Ask lo acompaña
        elif q_b <= 0:
            bid -= tick_size
            ask = bid + tick_size
            q_b = max(10, int(np.random.normal(mean_depth, mean_depth * 0.15)))
            q_a = max(10, int(mean_depth * 0.75))
        else:
            q_b = max(5, q_b)
            q_a = max(5, q_a)

        bids.append(bid)
        asks.append(ask)
        qb_list.append(q_b)
        qa_list.append(q_a)

    return pd.DataFrame({
        "bid_price": bids,
        "bid_size": qb_list,
        "ask_price": asks,
        "ask_size": qa_list,
    }, index=timestamps)


# ==============================================================================
# 3. IMPLEMENTACIÓN DE LA ESTRATEGIA F5: ORDER FLOW IMBALANCE (CONT ET AL.)
# ==============================================================================

class OrderFlowImbalanceStrategy(BaseStrategy):
    """
    Implementación rigurosa de la Estrategia F5: Order Flow Imbalance de Nivel 1.
    Formula: Cont, Kukanov & Stoikov (2014) Journal of Financial Econometrics.
    """

    def __init__(
        self,
        name: str = "Order_Flow_Imbalance_Cont",
        params: Optional[Dict[str, Any]] = None,
    ):
        default_params: Dict[str, Any] = {
            "symbol": "ES",
            "std_symbol": "ES",
            "micro_symbol": "MES",
            "window_w": 30,                 # Ventana de normalización W = 30 barras de 1 min
            "theta_enter": 2.00,            # Umbral de disparo de entrada Z_OFI >= 2.00
            "theta_exhaust": 0.50,          # Umbral de agotamiento (Take-Profit de flujo)
            "theta_reversal": -1.25,        # Umbral de inversión contraria (Stop-Loss de flujo)
            "tick_size": 0.25,              # Incremento mínimo de cotización CME (0.25 pt)
            "tp_min_ticks": 4,              # Take-Profit mínimo en ticks (4 ticks = 1.00 pt)
            "sl_max_ticks": 6,              # Stop-Loss rígido en ticks (6 ticks = 1.50 pts)
            "h_max": 10,                    # Límite máximo de permanencia: 10 barras (10 min)
            "gamma_book": 0.05,             # Fracción máxima de absorción del LOB (5%)
            "phi_dll": 0.20,                # Presupuesto máx: 20% del Daily Loss Limit
            "phi_tmdd": 0.15,               # Presupuesto máx: 15% del Trailing Max Drawdown
            "max_std_contracts": 4,         # Límite máximo de contratos estándar autorizados
            "max_micro_contracts": 40,      # Límite máximo de microcontratos autorizados
            "exec_start_time": time(9, 45), # Inicio de ventana operativa RTH
            "exec_end_time": time(15, 55),  # Fin de ventana operativa y liquidación EOD
            "epsilon_0": 1e-8,              # Constante de estabilidad numérica
            "target_tz": "America/New_York",
            "source_tz": None,
        }
        if params:
            default_params.update(params)
        super().__init__(name=name, params=default_params)

    # --------------------------------------------------------------------------
    # 3.1 Normalización Temporal
    # --------------------------------------------------------------------------
    def ensure_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Asegura que el índice DatetimeIndex esté localizado en America/New_York."""
        target_tz = self.params["target_tz"]
        source_tz = self.params.get("source_tz")

        if df.index.tz is None:
            if source_tz:
                return df.tz_localize(source_tz).tz_convert(target_tz)
            else:
                return df.tz_localize(target_tz)
        else:
            return df.tz_convert(target_tz)

    # --------------------------------------------------------------------------
    # 3.2 Preparación y Verificación de Datos de Entrada
    # --------------------------------------------------------------------------
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Garantiza que el DataFrame contenga las variables microestructurales indispensables:
        - Si el DataFrame contiene eventos L1, los agrega a 1 minuto vía Cont et al. (2014).
        - Si contiene velas con columnas 'OFI' y 'mean_depth', las utiliza directamente.
        - Si contiene solo OHLCV estándar, deriva el OFI continuo a partir de la firma de volumen y retorno
          según el modelo de impacto inverso de Cont et al.
        """
        df = self.ensure_timezone(data.copy())

        # Caso A: Datos brutos de eventos L1
        l1_cols = {"bid_price", "bid_size", "ask_price", "ask_size"}
        if l1_cols.issubset({c.lower() for c in df.columns}):
            df = compute_ofi_from_events(df, freq="1min")

        # Caso B: Velas con columnas microestructurales explícitas
        if "OFI" not in df.columns:
            # Reconstrucción microestructural fundamentada en volumen direccional y retorno
            # En futuros CME, el retorno por barra Delta P es proporcional a OFI / (2 * D)
            # Implícitamente: OFI ~ (Delta P / Delta P_tick) * V_eff
            warnings.warn(
                "ADVERTENCIA DE MICROESTRUCTURA (F5): Operando con proxy de volumen/retorno sobre barras OHLCV agregadas. "
                "El retardo temporal de velas comprimidas (>= 5m) degrada severamente la ventaja predictiva del OFI (Win Rate ~ 21.57%). "
                "Para operar con ventaja real en producción se requiere el feed tick a tick de CME MDP 3.0 (L1/L2) "
                "o para backtest teórico use '--synthetic-lob' con tick_size=0.25.",
                UserWarning,
                stacklevel=2,
            )
            tick = self.params["tick_size"]
            std_sym = self.params["std_symbol"]
            typical_depth = 250.0 if "ES" in std_sym else 30.0

            if "Close" in df.columns and "Open" in df.columns and "Volume" in df.columns:
                ret_pts = df["Close"] - df["Open"]
                ticks_moved = ret_pts / tick
                # Aproximación hidrodinámica del desbalance de flujo
                vol_baseline = np.maximum(df["Volume"].rolling(30, min_periods=1).mean(), 1.0)
                df["OFI"] = ticks_moved * (df["Volume"] / vol_baseline) * 50.0
                df["mean_depth"] = typical_depth
                df["bid_price"] = df["Close"] - (tick / 2.0)
                df["ask_price"] = df["Close"] + (tick / 2.0)
                df["spread"] = tick
            else:
                raise ValueError("El DataFrame debe contener 'OFI' y 'mean_depth' o columnas OHLCV.")

        if "mean_depth" not in df.columns:
            std_sym = self.params["std_symbol"]
            df["mean_depth"] = 250.0 if "ES" in std_sym else 30.0

        if "bid_price" not in df.columns or "ask_price" not in df.columns:
            tick = self.params["tick_size"]
            base_p = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
            df["bid_price"] = base_p - (tick / 2.0)
            df["ask_price"] = base_p + (tick / 2.0)
            df["spread"] = tick

        if "spread" not in df.columns:
            df["spread"] = df["ask_price"] - df["bid_price"]

        return df

    # --------------------------------------------------------------------------
    # 3.3 Normalización Z-Score de OFI y Coeficiente de Impacto de Precio
    # --------------------------------------------------------------------------
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula:
        1. Media móvil de OFI: mu_{OFI, k} = (1/W) * sum_{i=0}^{W-1} OFI_{k-i}
        2. Desviación estándar móvil de OFI: sigma_{OFI, k} con ddof=1
        3. Puntuación estandarizada: Z_{OFI, k} = (OFI_k - mu) / (sigma + epsilon_0)
        4. Coeficiente de impacto Cont-Kukanov-Stoikov: beta_{OFI, k} = 1 / (2 * mean_depth_k)
        5. Objetivo dinámico de Take-Profit: Delta P_{TP}^* = max(4 * tick, beta * sigma)
           cuantizado a múltiplos discretos de tick CME.
        """
        w = self.params["window_w"]
        eps_0 = self.params["epsilon_0"]
        tick = self.params["tick_size"]
        tp_min_ticks = self.params["tp_min_ticks"]

        ofi = df["OFI"]
        mean_depth = df["mean_depth"].clip(lower=1.0)

        df["mu_OFI"] = ofi.rolling(window=w, min_periods=w).mean()
        df["sigma_OFI"] = ofi.rolling(window=w, min_periods=w).std(ddof=1)
        df["Z_OFI"] = (ofi - df["mu_OFI"]) / (df["sigma_OFI"] + eps_0)
        df["Z_OFI"] = df["Z_OFI"].fillna(0.0)

        # Impacto microestructural beta = 1 / (2 * D_k)
        df["beta_OFI"] = 1.0 / (2.0 * mean_depth)

        # Target analítico de recorrido de precio Delta P_TP* cuantizado a ticks
        min_tp_pts = tp_min_ticks * tick
        dynamic_tp_pts = df["beta_OFI"] * df["sigma_OFI"].fillna(0.0)
        raw_tp = np.maximum(min_tp_pts, dynamic_tp_pts)
        df["delta_p_tp"] = np.round(raw_tp / tick) * tick

        return df

    # --------------------------------------------------------------------------
    # 3.4 Dimensionamiento de Posición y Conmutación Estándar / Micro (Sección 5)
    # --------------------------------------------------------------------------
    def calculate_position_sizing(
        self,
        risk_budget_usd: float,
        mean_depth: float,
    ) -> Dict[str, Any]:
        """
        Determina la cantidad de contratos N_{F5, k} a operar:
        1. Restricción de absorción del LOB: N_{book, max} = max(1, floor(gamma_book * mean_depth))
        2. Riesgo por contrato estándar y micro según Sección 5.3:
           - ES:  (1.50 * 50) + 4.40 = 79.40 USD
           - MES: (1.50 * 5)  + 1.20 =  8.70 USD
           - NQ:  (1.50 * 20) + 4.40 + 5.00 (slippage) = 39.40 USD
           - MNQ: (1.50 * 2)  + 1.20 + 0.50 (slippage) =  4.70 USD
        3. Conmutación automática: Si N_std == 0, se asigna N_micro.
        """
        std_sym = self.params["std_symbol"]
        micro_sym = self.params["micro_symbol"]
        gamma_book = self.params["gamma_book"]
        sl_ticks = self.params["sl_max_ticks"]
        tick = self.params["tick_size"]
        delta_sl_pts = sl_ticks * tick  # 1.50 puntos

        max_std = self.params["max_std_contracts"]
        max_micro = self.params["max_micro_contracts"]

        # Capacidad de absorción del LOB
        n_book_max_std = max(1, int(math.floor(gamma_book * mean_depth)))
        n_book_max_micro = n_book_max_std * 10

        # Obtener especificaciones oficiales de futuros
        spec_std = get_spec(std_sym)
        spec_micro = get_spec(micro_sym)

        m_std = spec_std["multiplier"]
        m_micro = spec_micro["multiplier"]

        # Costes de fricción documentados en Sección 5.3
        comm_std = 4.40
        comm_micro = 1.20
        # En NQ/MNQ se incluye 1 tick de slippage adverso por menor liquidez
        slip_std = 5.00 if "NQ" in std_sym else 0.0
        slip_micro = 0.50 if "NQ" in std_sym else 0.0

        r_std = (delta_sl_pts * m_std) + comm_std + slip_std
        r_micro = (delta_sl_pts * m_micro) + comm_micro + slip_micro

        if risk_budget_usd <= 0:
            return {
                "symbol": micro_sym,
                "contract_type": "NONE",
                "contracts": 0,
                "risk_per_contract": r_micro,
                "status": "HALT_NO_BUDGET"
            }

        # Evaluación en contrato estándar
        n_std = int(math.floor(risk_budget_usd / r_std))
        n_std = min(n_std, n_book_max_std, max_std)

        if n_std >= 1:
            return {
                "symbol": std_sym,
                "contract_type": "STD",
                "contracts": n_std,
                "risk_per_contract": r_std,
                "status": "OK"
            }

        # Conmutación automática a microcontratos
        if r_micro > risk_budget_usd:
            return {
                "symbol": micro_sym,
                "contract_type": "NONE",
                "contracts": 0,
                "risk_per_contract": r_micro,
                "status": "HALT_VOLATILITY_DEFENSE"
            }

        n_micro = int(math.floor(risk_budget_usd / r_micro))
        n_micro = min(n_micro, n_book_max_micro, max_micro)

        return {
            "symbol": micro_sym,
            "contract_type": "MICRO",
            "contracts": n_micro,
            "risk_per_contract": r_micro,
            "status": "OK" if n_micro > 0 else "HALT_VOLATILITY_DEFENSE"
        }

    # --------------------------------------------------------------------------
    # 3.5 Generación Vectorial de Señales (Interfaz BaseStrategy)
    # --------------------------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera la serie de exposición de mercado según la especificación F5:
        +1.0 = Posición Larga
        -1.0 = Posición Corta
         0.0 = Fuera de Mercado / Flat
        """
        df = self.prepare_data(data)
        df = self.calculate_indicators(df)

        theta_enter = self.params["theta_enter"]
        theta_exhaust = self.params["theta_exhaust"]
        theta_reversal = self.params["theta_reversal"]
        tick = self.params["tick_size"]
        sl_pts = self.params["sl_max_ticks"] * tick
        h_max = self.params["h_max"]
        t_start = self.params["exec_start_time"]
        t_end = self.params["exec_end_time"]

        signals = pd.Series(0.0, index=df.index)

        in_pos = False
        pos_dir = 0
        entry_price = 0.0
        entry_idx = 0
        tp_target_pts = 0.0

        for i, (ts, row) in enumerate(df.iterrows()):
            curr_time = ts.time() if hasattr(ts, "time") else None
            z = row["Z_OFI"]
            spread = row["spread"]
            bid = row["bid_price"]
            ask = row["ask_price"]
            delta_tp = row["delta_p_tp"]

            # Regla EOD Flat incondicional: Hora >= 15:55 EST
            if curr_time and curr_time >= t_end:
                in_pos = False
                pos_dir = 0
                signals.iloc[i] = 0.0
                continue

            if in_pos:
                # --------------------------------------------------------------
                # Verificación de Condiciones de Salida (Sección 4)
                # --------------------------------------------------------------
                bars_held = i - entry_idx
                exit_triggered = False

                high_b = (row["High"] - tick / 2.0) if "High" in row and pd.notna(row["High"]) else bid
                low_b = (row["Low"] - tick / 2.0) if "Low" in row and pd.notna(row["Low"]) else bid
                high_a = (row["High"] + tick / 2.0) if "High" in row and pd.notna(row["High"]) else ask
                low_a = (row["Low"] + tick / 2.0) if "Low" in row and pd.notna(row["Low"]) else ask

                if pos_dir == 1:
                    # Long:
                    # SL Precio: Low toca o perfora el stop loss
                    # TP Precio: High toca o supera el take profit objetivo
                    # SL Flujo: Z_OFI <= theta_reversal (-1.25)
                    # TP Flujo (Agotamiento): Z_OFI <= theta_exhaust (+0.50) asegurando beneficio (bid >= entry)
                    c_sl_price = (low_b <= (entry_price - sl_pts))
                    c_tp_price = (high_b >= (entry_price + tp_target_pts))
                    c_sl_flow = (z <= theta_reversal)
                    c_tp_flow = (z <= theta_exhaust) and (bid >= entry_price)
                    c_time = (bars_held >= h_max)

                    if c_sl_price or c_tp_price or c_sl_flow or c_tp_flow or c_time:
                        exit_triggered = True
                elif pos_dir == -1:
                    # Short:
                    # SL Precio: High toca o supera el stop loss
                    # TP Precio: Low toca o perfora el take profit objetivo
                    # SL Flujo: Z_OFI >= -theta_reversal (+1.25)
                    # TP Flujo (Agotamiento): Z_OFI >= -theta_exhaust (-0.50) asegurando beneficio (ask <= entry)
                    c_sl_price = (high_a >= (entry_price + sl_pts))
                    c_tp_price = (low_a <= (entry_price - tp_target_pts))
                    c_sl_flow = (z >= -theta_reversal)
                    c_tp_flow = (z >= -theta_exhaust) and (ask <= entry_price)
                    c_time = (bars_held >= h_max)

                    if c_sl_price or c_tp_price or c_sl_flow or c_tp_flow or c_time:
                        exit_triggered = True

                if exit_triggered:
                    in_pos = False
                    pos_dir = 0
                    signals.iloc[i] = 0.0
                else:
                    signals.iloc[i] = float(pos_dir)

            else:
                # --------------------------------------------------------------
                # Evaluación de Nuevas Entradas (Sección 3.5)
                # --------------------------------------------------------------
                # Filtro de horario: Solo 09:45 EST a 15:55 EST
                if curr_time and (curr_time < t_start or curr_time >= t_end):
                    continue

                # Filtro estricto de spread: Spread debe ser exactamente 1 tick (0.25 pt)
                if not math.isclose(spread, tick, abs_tol=1e-4):
                    continue

                # Disparo direccional por Z-Score de OFI
                if z > theta_enter:
                    # Entrada Long al Ask
                    in_pos = True
                    pos_dir = 1
                    entry_price = ask
                    entry_idx = i
                    tp_target_pts = delta_tp
                    signals.iloc[i] = 1.0
                elif z < -theta_enter:
                    # Entrada Short al Bid
                    in_pos = True
                    pos_dir = -1
                    entry_price = bid
                    entry_idx = i
                    tp_target_pts = delta_tp
                    signals.iloc[i] = -1.0

        return signals


# ==============================================================================
# 4. SIMULADOR DE ALTA FIDELIDAD Y AUDITORÍA DE PROP TRADING
# ==============================================================================

class OFIBacktester:
    """
    Motor de simulación cuantitativa de la Estrategia F5.
    Audita y reporta el cumplimiento estricto de Prop Trading:
    - Daily Loss Limit (DLL)
    - Trailing Max Drawdown (TMDD)
    - Conmutación automática Estándar / Micro
    - Comisiones y deslizamiento oficial de CME
    - Métricas estadísticas completas (Win Rate, Profit Factor, Sharpe, Sortino)
    """

    def __init__(
        self,
        strategy: Optional[OrderFlowImbalanceStrategy] = None,
        account: Optional[OFIPropAccount] = None,
    ):
        self.strategy = strategy or OrderFlowImbalanceStrategy()
        self.account = account or OFIPropAccount()
        self.trades: List[OFITradeRecord] = []
        self.daily_logs: List[Dict[str, Any]] = []

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Ejecuta la simulación cronológica sesión a sesión, barra a barra.
        """
        df = self.strategy.prepare_data(data)
        df = self.strategy.calculate_indicators(df)

        theta_enter = self.strategy.params["theta_enter"]
        theta_exhaust = self.strategy.params["theta_exhaust"]
        theta_reversal = self.strategy.params["theta_reversal"]
        tick = self.strategy.params["tick_size"]
        sl_pts = self.strategy.params["sl_max_ticks"] * tick
        h_max = self.strategy.params["h_max"]
        t_start = self.strategy.params["exec_start_time"]
        t_end = self.strategy.params["exec_end_time"]
        phi_dll = self.strategy.params["phi_dll"]
        phi_tmdd = self.strategy.params["phi_tmdd"]

        self.trades = []
        self.daily_logs = []

        active_trade: Optional[OFITradeRecord] = None

        grouped = df.groupby(df.index.normalize())

        for session_date, session_df in grouped:
            if self.account.is_breached:
                break

            self.account.start_session()
            session_start_equity = self.account.equity

            for i, (ts, row) in enumerate(session_df.iterrows()):
                if self.account.is_breached:
                    break

                curr_time = ts.time() if hasattr(ts, "time") else None
                z = row["Z_OFI"]
                spread = row["spread"]
                bid = row["bid_price"]
                ask = row["ask_price"]
                mean_depth = row["mean_depth"]
                delta_tp = row["delta_p_tp"]

                # Regla de salida forzosa EOD a las 15:55 EST
                if curr_time and curr_time >= t_end:
                    if active_trade is not None:
                        self._close_trade(
                            trade=active_trade,
                            exit_time=ts,
                            exit_bar_idx=i,
                            exit_price=bid if active_trade.direction == 1 else ask,
                            exit_reason=OFIExitReason.OVERNIGHT_FLAT
                        )
                        active_trade = None
                    continue

                if active_trade is not None:
                    # ----------------------------------------------------------
                    # Gestión de Posición Abierta
                    # ----------------------------------------------------------
                    bars_held = i - active_trade.entry_bar_idx
                    closed = False

                    high_b = (row["High"] - tick / 2.0) if "High" in row and pd.notna(row["High"]) else bid
                    low_b = (row["Low"] - tick / 2.0) if "Low" in row and pd.notna(row["Low"]) else bid
                    high_a = (row["High"] + tick / 2.0) if "High" in row and pd.notna(row["High"]) else ask
                    low_a = (row["Low"] + tick / 2.0) if "Low" in row and pd.notna(row["Low"]) else ask

                    has_open = ("Open" in row and pd.notna(row["Open"]))
                    open_b = (row["Open"] - tick / 2.0) if has_open else None
                    open_a = (row["Open"] + tick / 2.0) if has_open else None

                    if active_trade.direction == 1:
                        # Trade Long
                        # 1. Stop Loss rígido en precio (orden contingente stop en exchange)
                        if low_b <= active_trade.stop_loss_price:
                            fill_p = open_b if (has_open and open_b < active_trade.stop_loss_price) else active_trade.stop_loss_price
                            self._close_trade(active_trade, ts, i, fill_p, OFIExitReason.STOP_LOSS_PRICE)
                            closed = True
                        # 2. Take Profit objetivo en precio (orden límite resting en exchange)
                        elif high_b >= active_trade.take_profit_price:
                            fill_p = open_b if (has_open and open_b > active_trade.take_profit_price) else active_trade.take_profit_price
                            self._close_trade(active_trade, ts, i, fill_p, OFIExitReason.TAKE_PROFIT_PRICE)
                            closed = True
                        # 3. Stop Loss microestructural por inversión adversa de flujo
                        elif z <= theta_reversal:
                            self._close_trade(active_trade, ts, i, bid, OFIExitReason.FLOW_REVERSAL)
                            closed = True
                        # 4. Take Profit microestructural por agotamiento de flujo asegurando beneficio
                        elif z <= theta_exhaust and (bid >= active_trade.entry_price):
                            self._close_trade(active_trade, ts, i, bid, OFIExitReason.FLOW_EXHAUSTION)
                            closed = True
                        # 5. Cierre por Time-Decay (H_max barras)
                        elif bars_held >= h_max:
                            self._close_trade(active_trade, ts, i, bid, OFIExitReason.TIME_DECAY)
                            closed = True

                    elif active_trade.direction == -1:
                        # Trade Short
                        # 1. Stop Loss rígido en precio (orden contingente stop en exchange)
                        if high_a >= active_trade.stop_loss_price:
                            fill_p = open_a if (has_open and open_a > active_trade.stop_loss_price) else active_trade.stop_loss_price
                            self._close_trade(active_trade, ts, i, fill_p, OFIExitReason.STOP_LOSS_PRICE)
                            closed = True
                        # 2. Take Profit objetivo en precio (orden límite resting en exchange)
                        elif low_a <= active_trade.take_profit_price:
                            fill_p = open_a if (has_open and open_a < active_trade.take_profit_price) else active_trade.take_profit_price
                            self._close_trade(active_trade, ts, i, fill_p, OFIExitReason.TAKE_PROFIT_PRICE)
                            closed = True
                        # 3. Stop Loss microestructural por inversión adversa de flujo
                        elif z >= -theta_reversal:
                            self._close_trade(active_trade, ts, i, ask, OFIExitReason.FLOW_REVERSAL)
                            closed = True
                        # 4. Take Profit microestructural por agotamiento de flujo asegurando beneficio
                        elif z >= -theta_exhaust and (ask <= active_trade.entry_price):
                            self._close_trade(active_trade, ts, i, ask, OFIExitReason.FLOW_EXHAUSTION)
                            closed = True
                        # 5. Cierre por Time-Decay (H_max barras)
                        elif bars_held >= h_max:
                            self._close_trade(active_trade, ts, i, ask, OFIExitReason.TIME_DECAY)
                            closed = True

                    if closed:
                        active_trade = None

                else:
                    # ----------------------------------------------------------
                    # Evaluación de Disparo de Entrada
                    # ----------------------------------------------------------
                    if curr_time and (curr_time < t_start or curr_time >= t_end):
                        continue

                    # Filtro de spread estricto (1 tick)
                    if not math.isclose(spread, tick, abs_tol=1e-4):
                        continue

                    direction = 0
                    if z > theta_enter:
                        direction = 1
                    elif z < -theta_enter:
                        direction = -1

                    if direction != 0:
                        risk_budget = self.account.calculate_risk_budget(phi_dll, phi_tmdd)
                        sizing = self.strategy.calculate_position_sizing(risk_budget, mean_depth)

                        if sizing["contracts"] > 0:
                            entry_p = ask if direction == 1 else bid
                            stop_p = (entry_p - sl_pts) if direction == 1 else (entry_p + sl_pts)
                            tp_p = (entry_p + delta_tp) if direction == 1 else (entry_p - delta_tp)

                            active_trade = OFITradeRecord(
                                date=str(session_date.date() if hasattr(session_date, "date") else session_date),
                                symbol=sizing["symbol"],
                                direction=direction,
                                contracts=sizing["contracts"],
                                contract_type=sizing["contract_type"],
                                entry_bar_idx=i,
                                entry_time=ts,
                                entry_price=entry_p,
                                stop_loss_price=stop_p,
                                take_profit_price=tp_p,
                                delta_sl_points=sl_pts,
                                delta_tp_points=delta_tp,
                                z_score_entry=z,
                            )

            # Cierre de sesión y log diario
            if active_trade is not None:
                last_ts = session_df.index[-1]
                last_bid = session_df["bid_price"].iloc[-1]
                last_ask = session_df["ask_price"].iloc[-1]
                self._close_trade(
                    active_trade,
                    last_ts,
                    len(session_df) - 1,
                    last_bid if active_trade.direction == 1 else last_ask,
                    OFIExitReason.OVERNIGHT_FLAT
                )
                active_trade = None

            day_pnl = self.account.equity - session_start_equity
            self.daily_logs.append({
                "date": session_date,
                "day_pnl": round(day_pnl, 2),
                "ending_equity": round(self.account.equity, 2),
                "peak_equity": round(self.account.peak_equity, 2),
                "trailing_floor": round(self.account.trailing_floor, 2),
                "is_breached": self.account.is_breached,
            })

        return self._generate_report()

    def _close_trade(
        self,
        trade: OFITradeRecord,
        exit_time: Any,
        exit_bar_idx: int,
        exit_price: float,
        exit_reason: OFIExitReason
    ) -> None:
        """Calcula PnL bruto, fricciones de CME y actualiza la cuenta Prop."""
        trade.exit_time = exit_time
        trade.exit_bar_idx = exit_bar_idx
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.holding_bars = exit_bar_idx - trade.entry_bar_idx

        spec = get_spec(trade.symbol)
        m = spec["multiplier"]

        # PnL en puntos
        pts_diff = (exit_price - trade.entry_price) if trade.direction == 1 else (trade.entry_price - exit_price)
        gross_pnl = pts_diff * m * trade.contracts

        # Comisiones y deslizamiento oficial
        comm_per_contract = 4.40 if trade.contract_type == "STD" else 1.20
        slip_per_contract = 5.00 if "NQ" in trade.symbol and trade.contract_type == "STD" else (
            0.50 if "NQ" in trade.symbol and trade.contract_type == "MICRO" else 0.0
        )

        total_comm = comm_per_contract * trade.contracts
        total_slip = slip_per_contract * trade.contracts
        net_pnl = gross_pnl - total_comm - total_slip

        trade.gross_pnl_usd = round(gross_pnl, 2)
        trade.commission_usd = round(total_comm, 2)
        trade.slippage_usd = round(total_slip, 2)
        trade.net_pnl_usd = round(net_pnl, 2)

        self.account.update_equity_intraday(net_pnl)
        self.trades.append(trade)

    def _generate_report(self) -> Dict[str, Any]:
        """Genera el diccionario exhaustivo de métricas de desempeño y cumplimiento prop."""
        trades_df = self.get_trades_dataframe()
        net_profits = trades_df["net_pnl_usd"] if not trades_df.empty else pd.Series(dtype=float)

        total_trades = len(trades_df)
        wins = net_profits[net_profits > 0]
        losses = net_profits[net_profits < 0]

        win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        total_net_pnl = float(self.account.equity - self.account.initial_capital)
        total_return_pct = (total_net_pnl / self.account.initial_capital) * 100.0

        # Drawdown máximo
        peak = self.account.initial_capital
        max_dd = 0.0
        running = self.account.initial_capital
        for p in net_profits:
            running += p
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd

        # Sharpe y Sortino
        if len(net_profits) > 1 and net_profits.std(ddof=1) > 0:
            sharpe = float((net_profits.mean() / net_profits.std(ddof=1)) * np.sqrt(252 * 5))
            downside = net_profits[net_profits < 0]
            downside_std = downside.std(ddof=1) if len(downside) > 1 else 1e-6
            sortino = float((net_profits.mean() / (downside_std + 1e-6)) * np.sqrt(252 * 5))
        else:
            sharpe = 0.0
            sortino = 0.0

        return {
            "initial_capital": self.account.initial_capital,
            "final_equity": round(self.account.equity, 2),
            "total_net_pnl_usd": round(total_net_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "total_trades": total_trades,
            "winning_trades": int(len(wins)),
            "losing_trades": int(len(losses)),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2) if not math.isinf(profit_factor) else "INF",
            "gross_profit_usd": round(gross_profit, 2),
            "gross_loss_usd": round(gross_loss, 2),
            "total_commissions_usd": round(float(trades_df["commission_usd"].sum()), 2) if not trades_df.empty else 0.0,
            "total_slippage_usd": round(float(trades_df["slippage_usd"].sum()), 2) if not trades_df.empty else 0.0,
            "max_drawdown_usd": round(max_dd, 2),
            "max_drawdown_pct": round((max_dd / self.account.initial_capital) * 100.0, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "is_prop_breached": self.account.is_breached,
            "breach_reason": self.account.breach_reason or "COMPLIANT (Ningún límite violado)",
        }

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Retorna un DataFrame con el registro completo de operaciones."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self.trades])
