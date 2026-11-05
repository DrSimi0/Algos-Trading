"""
src/strategies/filtered_orb.py
Implementación Matemática y Microestructural de la Estrategia F2:
Filtered Quantitative Opening Range Breakout (ORB).

Referencias Académicas:
1. Crabel, T. (1990). Day Trading with Short Term Price Patterns. Traders Press.
2. Zarattini, C., Barbon, A., & Aziz, A. (2024). A Profitable Day Trading Strategy
   For The U.S. Equity Market. Swiss Finance Institute Research Paper No. 24-98.
3. Fetna, M. (2024). Opening-Range Breakout Does Not Survive Trading Costs.

Especificación: reports/estrategias_matematicas/F2_filtered_orb.md
"""

from dataclasses import dataclass, field
from datetime import time, datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Sequence
import math
import numpy as np
import pandas as pd

from src.core.futures_specs import get_spec


# ==============================================================================
# 1. ESTADOS FINITOS DEL SISTEMA (SECCIÓN 2.2)
# ==============================================================================

class StrategyState(str, Enum):
    MEASURING_OR = "MEASURING_OR"
    EVALUATING_FILTERS = "EVALUATING_FILTERS"
    ARMED_BREAKOUT = "ARMED_BREAKOUT"
    IN_POSITION = "IN_POSITION"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    FLAT_COMPLETED = "FLAT_COMPLETED"
    HALT_EXCESSIVE_RANGE_DEFENSE = "HALT_EXCESSIVE_RANGE_DEFENSE"
    HALT_ACCOUNT_BREACHED = "HALT_ACCOUNT_BREACHED"


class ExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TIME_DECAY = "TIME_DECAY"


# ==============================================================================
# 2. REGISTRO DE TRANSACCIÓN Y ESTADO DE CUENTA PROP TRADING
# ==============================================================================

@dataclass
class TradeRecord:
    date: str
    symbol: str
    direction: int                      # +1 = Long, -1 = Short
    contracts: int                     # Número de contratos
    is_micro: bool                     # True si usa microcontratos
    entry_time: str
    entry_price: float
    stop_loss: float
    take_profit: float
    delta_risk: float
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    gross_pnl_points: float = 0.0
    gross_pnl_usd: float = 0.0
    commissions_usd: float = 0.0
    slippage_usd: float = 0.0
    net_pnl_usd: float = 0.0
    running_equity: float = 0.0
    running_drawdown: float = 0.0


@dataclass
class PropAccount:
    """
    Control de riesgo cuantitativo para firmas de Prop Trading (Topstep, Apex, Lucid, MFF).
    Gestiona Límite de Pérdida Diaria (DLL), Trailing Drawdown flotante (MTDD)
    y atenuación exponencial por racha perdedora (K_loss).
    """
    initial_capital: float = 50000.0
    dll: float = 1000.0                # Daily Loss Limit
    mtdd: float = 2000.0               # Maximum Trailing Drawdown
    equity: float = 50000.0
    peak_equity: float = 50000.0       # High Water Mark (HWM)
    current_drawdown: float = 0.0
    daily_realized_loss: float = 0.0   # Pérdida acumulada en el día (positiva)
    daily_pnl: float = 0.0             # PnL neto del día actual
    k_loss: int = 0                    # Contador de pérdidas consecutivas
    is_breached: bool = False
    breach_reason: Optional[str] = None

    def start_session(self, session_date: Any) -> None:
        """Reinicia los contadores diarios al inicio de una nueva sesión bursátil."""
        self.daily_realized_loss = 0.0
        self.daily_pnl = 0.0

    def calculate_risk_budget(self, psi: float = 0.25) -> float:
        """
        Calcula el presupuesto dinámico de riesgo B_{F2, t} según sección 5.2:
        B_{F2, t} = min( phi(K_loss) * max(0, DLL - L_day,t), psi * max(0, MTDD - DD_t) )
        donde phi(K_loss) = 0.35 * 2^(-min(K_loss, 3)).
        """
        if self.is_breached:
            return 0.0

        rem_daily = max(0.0, self.dll - self.daily_realized_loss)
        rem_trailing = max(0.0, self.mtdd - self.current_drawdown)

        if rem_daily <= 0.0 or rem_trailing <= 0.0:
            return 0.0

        phi = 0.35 * (2.0 ** (-min(self.k_loss, 3)))
        budget = min(phi * rem_daily, psi * rem_trailing)
        return max(0.0, budget)

    def record_trade(self, trade: TradeRecord) -> None:
        """Actualiza el balance, drawdown, HWM y racha tras completar un trade."""
        self.equity += trade.net_pnl_usd
        self.daily_pnl += trade.net_pnl_usd

        if trade.net_pnl_usd < 0:
            self.daily_realized_loss += abs(trade.net_pnl_usd)
            self.k_loss += 1
        else:
            self.k_loss = 0

        # Actualizar High Water Mark y Drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        self.current_drawdown = self.peak_equity - self.equity

        # Verificar límites de Prop Trading
        if self.current_drawdown >= self.mtdd:
            self.is_breached = True
            self.breach_reason = f"TRAILING_DRAWDOWN_BREACH: Drawdown ${self.current_drawdown:.2f} >= MTDD ${self.mtdd:.2f}"
        elif self.daily_realized_loss >= self.dll:
            self.is_breached = True
            self.breach_reason = f"DAILY_LOSS_LIMIT_BREACH: Pérdida Diaria ${self.daily_realized_loss:.2f} >= DLL ${self.dll:.2f}"

        trade.running_equity = self.equity
        trade.running_drawdown = self.current_drawdown


# ==============================================================================
# 3. FUNCIONES MATEMÁTICAS FORMALES DEL ORB FILTRADO
# ==============================================================================

def _parse_datetime_index_robust(raw_dates: Any, target_tz: str = "America/New_York") -> pd.DatetimeIndex:
    """Parsea fechas soportando tanto marcas naive como zonas horarias mixtas por cambio de horario de verano (DST)."""
    try:
        idx = pd.to_datetime(raw_dates)
        if isinstance(idx, pd.Series):
            return pd.DatetimeIndex(idx)
        return pd.DatetimeIndex(idx)
    except Exception:
        utc_parsed = pd.to_datetime(raw_dates, utc=True)
        if hasattr(utc_parsed, "dt"):
            return pd.DatetimeIndex(utc_parsed.dt.tz_convert(target_tz))
        elif hasattr(utc_parsed, "tz_convert"):
            return pd.DatetimeIndex(utc_parsed.tz_convert(target_tz))
        return pd.DatetimeIndex(utc_parsed)


def normalize_to_eastern(
    df: pd.DataFrame,
    source_tz: Optional[str] = None,
    target_tz: str = "America/New_York"
) -> pd.DataFrame:
    """
    Garantiza que el índice DatetimeIndex esté localizado en America/New_York (EST/EDT).
    Si el DataFrame es naive y source_tz no se especifica, detecta automáticamente
    la zona horaria de origen analizando la distribución intradía del volumen matutino.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.copy()
            df.index = _parse_datetime_index_robust(df["datetime"], target_tz=target_tz)
            df.drop(columns=["datetime"], inplace=True)
        elif "Unnamed: 0" in df.columns:
            df = df.copy()
            df.index = _parse_datetime_index_robust(df["Unnamed: 0"], target_tz=target_tz)
            df.drop(columns=["Unnamed: 0"], inplace=True)
        else:
            df = df.copy()
            df.index = _parse_datetime_index_robust(df.index, target_tz=target_tz)

    df_out = df.copy()
    if df_out.index.tz is None:
        if source_tz:
            return df_out.tz_localize(source_tz).tz_convert(target_tz)

        # Detección automática por perfil de volumen matutino
        times = df_out.index.time
        df_vol = df_out.groupby(times)["Volume"].mean()
        vol_0730 = df_vol.get(time(7, 30), 0.0)
        vol_0830 = df_vol.get(time(8, 30), 0.0)
        vol_0930 = df_vol.get(time(9, 30), 0.0)

        if vol_0730 > max(vol_0830, vol_0930) and vol_0730 > 15000:
            detected_tz = "America/Mexico_City"  # UTC-6
        elif vol_0830 > max(vol_0730, vol_0930) and vol_0830 > 15000:
            detected_tz = "America/Chicago"      # UTC-5/UTC-6
        else:
            detected_tz = target_tz

        return df_out.tz_localize(detected_tz).tz_convert(target_tz)
    else:
        return df_out.tz_convert(target_tz)


def check_nr4(daily_ranges: Sequence[float]) -> bool:
    """
    Condición Narrow Range 4 (NR4):
    Range_{t-1} < min(Range_{t-2}, Range_{t-3}, Range_{t-4}).
    Requiere al menos 4 rangos diarios históricos previos.
    """
    if len(daily_ranges) < 4:
        return False
    r_prev = daily_ranges[-1]
    r_prior_3 = daily_ranges[-4:-1]
    return bool(r_prev < min(r_prior_3))


def check_inside_day(daily_highs: Sequence[float], daily_lows: Sequence[float]) -> bool:
    """
    Condición Inside Day (ID):
    High_{t-1} <= High_{t-2} AND Low_{t-1} >= Low_{t-2}.
    """
    if len(daily_highs) < 2 or len(daily_lows) < 2:
        return False
    return bool(daily_highs[-1] <= daily_highs[-2] and daily_lows[-1] >= daily_lows[-2])


def check_prior_compression(daily_bars: pd.DataFrame) -> bool:
    """
    Función de compresión estructural previa C_t = max(NR4_t, ID_t).
    Se evalúa estrictamente sobre las sesiones previas (t-1, t-2, t-3, t-4)
    sin ningún dato de la sesión actual (Lookahead bias = 0).
    """
    if len(daily_bars) < 4:
        return False
    highs = daily_bars["High"].values
    lows = daily_bars["Low"].values

    if np.any(highs < lows):
        raise ValueError("Datos corruptos detectados en daily_bars: High < Low.")

    ranges = highs - lows

    nr4 = check_nr4(ranges[-4:])
    inside_day = check_inside_day(highs[-2:], lows[-2:])
    return bool(nr4 or inside_day)


def calculate_crabel_stretch(daily_bars: pd.DataFrame, lookback: int = 10) -> float:
    """
    Cálculo formal del Amortiguador de Ruido de Crabel (Stretch):
    D_{t-k} = min(High_{t-k} - Open_{t-k}, Open_{t-k} - Low_{t-k}) para k in {1, ..., 10}.
    Stretch_t = (1/10) * sum_{k=1}^{10} D_{t-k}.
    """
    if len(daily_bars) < lookback:
        raise ValueError(f"Se requieren al menos {lookback} barras diarias para calcular el Stretch.")

    df_subset = daily_bars.iloc[-lookback:]
    opens = df_subset["Open"].values
    highs = df_subset["High"].values
    lows = df_subset["Low"].values

    if np.any(highs < lows):
        raise ValueError("Datos corruptos detectados: High < Low en cálculo de Stretch.")

    d_values = np.minimum(highs - opens, opens - lows)
    stretch = float(np.mean(d_values))
    return max(0.0, stretch)


def calculate_rvol(
    or_volume: float,
    historical_or_volumes: Sequence[float],
    lookback: int = 20,
    rvol_min: float = 1.80
) -> Tuple[float, bool]:
    """
    Filtro de Volumen Relativo Institucional (RVOL):
    Media de volumen de apertura V_bar = (1/20) * sum_{m=1}^{20} V_{OR, t-m}.
    RVOL_t = V_{OR, t} / V_bar.
    Condición agresora: RVOL_t >= 1.80.
    Protección contra división por cero y volumen nulo.
    """
    if or_volume <= 0.0 or len(historical_or_volumes) == 0:
        return 0.0, False

    recent_volumes = historical_or_volumes[-lookback:]
    avg_vol = float(np.mean(recent_volumes))
    if avg_vol <= 0.0:
        return 0.0, False

    rvol = float(or_volume / avg_vol)
    is_valid = bool(rvol >= rvol_min)
    return rvol, is_valid


def calculate_triggers(
    orh: float,
    orl: float,
    stretch: float,
    lambda_stretch: float = 0.50
) -> Tuple[float, float, float]:
    """
    Gatillo_Long = ORH + lambda * Stretch
    Gatillo_Short = ORL - lambda * Stretch
    Punto Medio M_OR = (ORH + ORL) / 2
    """
    if orh < orl:
        raise ValueError(f"Valores de Opening Range inválidos: ORH ({orh}) < ORL ({orl}).")
    if stretch < 0.0:
        raise ValueError(f"Amortiguador Stretch no puede ser negativo: {stretch}.")

    buffer = lambda_stretch * stretch
    gatillo_long = orh + buffer
    gatillo_short = orl - buffer
    midpoint = (orh + orl) / 2.0
    return gatillo_long, gatillo_short, midpoint


def calculate_delta_risk(
    orh: float,
    orl: float,
    stretch: float,
    lambda_stretch: float = 0.50
) -> float:
    """
    Distancia analítica de riesgo Delta_risk = |P_entry - M_OR|:
    Delta_risk = (ORH - ORL) / 2 + lambda * Stretch.
    """
    if orh < orl:
        raise ValueError(f"ORH ({orh}) < ORL ({orl}) inválido para cálculo de Delta_risk.")
    if stretch < 0.0:
        raise ValueError(f"Stretch ({stretch}) no puede ser negativo.")
    return ((orh - orl) / 2.0) + (lambda_stretch * stretch)


def calculate_position_size(
    risk_budget: float,
    delta_risk: float,
    std_symbol: str = "ES",
    micro_symbol: str = "MES",
    max_micros: int = 15,
    slippage_ticks: float = 1.0
) -> Tuple[str, int, float, str]:
    """
    Modelo de dimensionamiento y conmutación automática Estándar / Micro (Sección 5.4).
    Calcula:
      R_contract(M) = Delta_risk * M + C_fric
      donde C_fric = 2 * delta_tick * M + comision_rt.
    Retorna: (símbolo_a_operar, cantidad_contratos, riesgo_por_contrato, razón/estado)
    """
    if risk_budget <= 0.0 or delta_risk <= 0.0:
        return "", 0, 0.0, "HALT_NO_BUDGET"

    spec_std = get_spec(std_symbol)
    m_std = spec_std["multiplier"]
    tick_std = spec_std["tick"]
    comm_std = spec_std["commission_rt"]
    fric_std = (2.0 * slippage_ticks * tick_std * m_std) + comm_std
    r_contract_std = (delta_risk * m_std) + fric_std

    # Intentar contratos estándar
    n_std = math.floor(risk_budget / r_contract_std)
    if n_std >= 1:
        return std_symbol, n_std, r_contract_std, "STANDARD_ORDER"

    # Conmutar a microcontratos
    spec_micro = get_spec(micro_symbol)
    m_micro = spec_micro["multiplier"]
    tick_micro = spec_micro["tick"]
    comm_micro = spec_micro["commission_rt"]
    fric_micro = (2.0 * slippage_ticks * tick_micro * m_micro) + comm_micro
    r_contract_micro = (delta_risk * m_micro) + fric_micro

    # Disyuntor por Rango Excesivo (Sección 5.5)
    if r_contract_micro > risk_budget:
        return "", 0, r_contract_micro, "HALT_EXCESSIVE_RANGE_DEFENSE"

    n_micro = min(math.floor(risk_budget / r_contract_micro), max_micros)
    if n_micro >= 1:
        return micro_symbol, n_micro, r_contract_micro, "MICRO_SWITCH_ACTIVE"

    return "", 0, r_contract_micro, "HALT_EXCESSIVE_RANGE_DEFENSE"


# ==============================================================================
# 4. CLASE PRINCIPAL: FilteredQuantitativeORB
# ==============================================================================

class FilteredQuantitativeORB:
    """
    Estrategia Filtered Quantitative Opening Range Breakout (ORB).
    Implementación formal completa y modular sin indicadores tradicionales rezagados.
    """

    def __init__(
        self,
        std_symbol: str = "ES",
        micro_symbol: str = "MES",
        lambda_stretch: float = 0.50,
        rvol_min: float = 1.80,
        stretch_lookback: int = 10,
        rvol_lookback: int = 20,
        max_micros: int = 15,
        market_open: time = time(9, 30),
        or_end: time = time(9, 45),
        entry_cutoff: time = time(12, 0),
        flat_time: time = time(15, 45),
        slippage_ticks: float = 1.0
    ):
        self.std_symbol = std_symbol
        self.micro_symbol = micro_symbol
        self.lambda_stretch = lambda_stretch
        self.rvol_min = rvol_min
        self.stretch_lookback = stretch_lookback
        self.rvol_lookback = rvol_lookback
        self.max_micros = max_micros

        self.market_open = market_open
        self.or_end = or_end
        self.entry_cutoff = entry_cutoff
        self.flat_time = flat_time
        self.slippage_ticks = slippage_ticks

        # Estado del día actual
        self.state: StrategyState = StrategyState.MEASURING_OR
        self.current_trade: Optional[TradeRecord] = None

        # Variables analíticas del día
        self.is_compressed: bool = False
        self.stretch: float = 0.0
        self.orh: float = 0.0
        self.orl: float = 0.0
        self.midpoint: float = 0.0
        self.or_volume: float = 0.0
        self.rvol: float = 0.0
        self.has_volume_expansion: bool = False
        self.delta_risk: float = 0.0
        self.buy_trigger: float = 0.0
        self.sell_trigger: float = 0.0

    def prepare_daily_context(
        self,
        daily_history: pd.DataFrame,
        historical_or_volumes: List[float]
    ) -> bool:
        """
        Calcula compresión previa y amortiguador Stretch usando estrictamente barras diarias anteriores.
        Retorna True si la compresión previa está activa.
        """
        if len(daily_history) < max(4, self.stretch_lookback):
            self.is_compressed = False
            self.stretch = 0.0
            return False

        self.is_compressed = check_prior_compression(daily_history)
        self.stretch = calculate_crabel_stretch(daily_history, lookback=self.stretch_lookback)
        return self.is_compressed

    def evaluate_opening_range(
        self,
        or_bars: pd.DataFrame,
        historical_or_volumes: List[float]
    ) -> bool:
        """
        Construye el Opening Range [09:30, 09:45]:
        Mide ORH, ORL, V_OR y calcula RVOL institucional.
        """
        if or_bars.empty:
            self.orh = 0.0
            self.orl = 0.0
            self.or_volume = 0.0
            self.midpoint = 0.0
            self.rvol = 0.0
            self.has_volume_expansion = False
            self.delta_risk = 0.0
            self.buy_trigger = 0.0
            self.sell_trigger = 0.0
            self.state = StrategyState.ORDER_CANCELLED
            return False


        if (or_bars["High"] < or_bars["Low"]).any():
            raise ValueError("Barras corruptas en Opening Range: High < Low.")

        self.orh = float(or_bars["High"].max())
        self.orl = float(or_bars["Low"].min())
        self.or_volume = float(or_bars["Volume"].sum())
        if self.or_volume < 0:
            raise ValueError("Volumen negativo detectado en Opening Range.")

        self.midpoint = (self.orh + self.orl) / 2.0

        # RVOL
        self.rvol, self.has_volume_expansion = calculate_rvol(
            self.or_volume,
            historical_or_volumes,
            lookback=self.rvol_lookback,
            rvol_min=self.rvol_min
        )

        self.buy_trigger, self.sell_trigger, _ = calculate_triggers(
            self.orh, self.orl, self.stretch, lambda_stretch=self.lambda_stretch
        )
        self.delta_risk = calculate_delta_risk(
            self.orh, self.orl, self.stretch, lambda_stretch=self.lambda_stretch
        )

        return True

    def process_day(
        self,
        session_date: Any,
        intraday_bars: pd.DataFrame,
        daily_history: pd.DataFrame,
        historical_or_volumes: List[float],
        account: PropAccount
    ) -> Optional[TradeRecord]:
        """
        Ejecuta el ciclo de vida completo de la sesión para el modelo F2.
        Respeta ventanas temporales, órdenes stop de ruptura, midpoint SL, 2R TP y salida forzosa.
        """
        self.state = StrategyState.MEASURING_OR
        self.current_trade = None

        if intraday_bars.empty:
            return None

        # 1. Filtro Previo de Compresión y Amortiguador Stretch
        self.prepare_daily_context(daily_history, historical_or_volumes)

        # Separar barras según ventanas temporales
        times = intraday_bars.index.time
        # Opening Range estricto de 15 minutos: [market_open, or_end)
        # En barras de 5 minutos rotuladas al inicio: 09:30, 09:35, 09:40
        or_mask = (times >= self.market_open) & (times < self.or_end)
        or_bars = intraday_bars[or_mask]

        if or_bars.empty:
            return None

        # 2. Medición del Opening Range
        self.evaluate_opening_range(or_bars, historical_or_volumes)

        # Registrar el volumen del OR para el historial futuro
        if self.or_volume > 0:
            historical_or_volumes.append(self.or_volume)

        # 3. Evaluación de Filtros Conjuntos C_t y Phi_{vol, t}
        self.state = StrategyState.EVALUATING_FILTERS
        if not (self.is_compressed and self.has_volume_expansion):
            self.state = StrategyState.ORDER_CANCELLED
            return None

        # 4. Dimensionamiento y Verificación de Presupuesto Prop Trading
        risk_budget = account.calculate_risk_budget()
        symbol_to_trade, n_contracts, r_contract, status_code = calculate_position_size(
            risk_budget=risk_budget,
            delta_risk=self.delta_risk,
            std_symbol=self.std_symbol,
            micro_symbol=self.micro_symbol,
            max_micros=self.max_micros,
            slippage_ticks=self.slippage_ticks
        )

        if n_contracts == 0:
            if status_code == "HALT_EXCESSIVE_RANGE_DEFENSE":
                self.state = StrategyState.HALT_EXCESSIVE_RANGE_DEFENSE
            elif status_code == "HALT_NO_BUDGET":
                self.state = StrategyState.HALT_ACCOUNT_BREACHED
            else:
                self.state = StrategyState.ORDER_CANCELLED
            return None

        # 5. Armado de Órdenes Condicionadas en W_entry (09:45, 12:00]
        self.state = StrategyState.ARMED_BREAKOUT
        spec = get_spec(symbol_to_trade)
        tick_size = spec["tick"]
        multiplier = spec["multiplier"]
        comm_rt = spec["commission_rt"]
        is_micro = (symbol_to_trade == self.micro_symbol)

        # Barras posteriores al Opening Range (a partir de or_end, e.g. 09:45)
        post_or_mask = (times >= self.or_end)
        post_or_bars = intraday_bars[post_or_mask]

        for bar_time, bar in post_or_bars.iterrows():
            current_bar_time = bar_time.time()

            # A) Si estamos esperando entrada
            if self.state == StrategyState.ARMED_BREAKOUT:
                # Cancelación por Trampa de Mediodía
                if current_bar_time > self.entry_cutoff:
                    self.state = StrategyState.ORDER_CANCELLED
                    break

                b_open = float(bar["Open"])
                b_high = float(bar["High"])
                b_low = float(bar["Low"])
                b_close = float(bar["Close"])

                # Ruptura Alcista
                if b_high >= self.buy_trigger:
                    fill_price = self.buy_trigger + (self.slippage_ticks * tick_size)
                    sl_price = self.midpoint
                    tp_price = fill_price + (2.0 * self.delta_risk)

                    self.current_trade = TradeRecord(
                        date=str(session_date),
                        symbol=symbol_to_trade,
                        direction=1,
                        contracts=n_contracts,
                        is_micro=is_micro,
                        entry_time=str(bar_time),
                        entry_price=fill_price,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        delta_risk=self.delta_risk
                    )
                    self.state = StrategyState.IN_POSITION

                    # Evaluación de salida en la misma barra de entrada:
                    # 1. Take Profit intra-barra (si la vela continuó al alza tras la ruptura)
                    if b_high >= tp_price:
                        self._finalize_trade(
                            trade=self.current_trade,
                            exit_time=str(bar_time),
                            exit_price=tp_price,
                            reason=ExitReason.TAKE_PROFIT,
                            multiplier=multiplier,
                            comm_rt=comm_rt,
                            tick_size=tick_size
                        )
                        account.record_trade(self.current_trade)
                        self.state = StrategyState.FLAT_COMPLETED
                        return self.current_trade

                    # 2. Stop Loss en la barra de entrada: solo se activa si hay reversión comprobada
                    # posterior a la entrada (cierre por debajo del SL, o si abrió sobre el gatillo y cayó bajo el SL)
                    is_same_bar_sl = (b_close <= sl_price) or (b_open >= self.buy_trigger and b_low <= sl_price)
                    if is_same_bar_sl:
                        exit_fill = sl_price - (self.slippage_ticks * tick_size)
                        self._finalize_trade(
                            trade=self.current_trade,
                            exit_time=str(bar_time),
                            exit_price=exit_fill,
                            reason=ExitReason.STOP_LOSS,
                            multiplier=multiplier,
                            comm_rt=comm_rt,
                            tick_size=tick_size
                        )
                        account.record_trade(self.current_trade)
                        self.state = StrategyState.FLAT_COMPLETED
                        return self.current_trade

                    continue

                # Ruptura Bajista
                elif b_low <= self.sell_trigger:
                    fill_price = self.sell_trigger - (self.slippage_ticks * tick_size)
                    sl_price = self.midpoint
                    tp_price = fill_price - (2.0 * self.delta_risk)

                    self.current_trade = TradeRecord(
                        date=str(session_date),
                        symbol=symbol_to_trade,
                        direction=-1,
                        contracts=n_contracts,
                        is_micro=is_micro,
                        entry_time=str(bar_time),
                        entry_price=fill_price,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        delta_risk=self.delta_risk
                    )
                    self.state = StrategyState.IN_POSITION

                    # Evaluación de salida en la misma barra de entrada:
                    # 1. Take Profit intra-barra
                    if b_low <= tp_price:
                        self._finalize_trade(
                            trade=self.current_trade,
                            exit_time=str(bar_time),
                            exit_price=tp_price,
                            reason=ExitReason.TAKE_PROFIT,
                            multiplier=multiplier,
                            comm_rt=comm_rt,
                            tick_size=tick_size
                        )
                        account.record_trade(self.current_trade)
                        self.state = StrategyState.FLAT_COMPLETED
                        return self.current_trade

                    # 2. Stop Loss en la barra de entrada
                    is_same_bar_sl = (b_close >= sl_price) or (b_open <= self.sell_trigger and b_high >= sl_price)
                    if is_same_bar_sl:
                        exit_fill = sl_price + (self.slippage_ticks * tick_size)
                        self._finalize_trade(
                            trade=self.current_trade,
                            exit_time=str(bar_time),
                            exit_price=exit_fill,
                            reason=ExitReason.STOP_LOSS,
                            multiplier=multiplier,
                            comm_rt=comm_rt,
                            tick_size=tick_size
                        )
                        account.record_trade(self.current_trade)
                        self.state = StrategyState.FLAT_COMPLETED
                        return self.current_trade

                    continue

            # B) Si ya estamos en posición en barras subsiguientes
            elif self.state == StrategyState.IN_POSITION:
                trade = self.current_trade
                b_high = float(bar["High"])
                b_low = float(bar["Low"])
                b_close = float(bar["Close"])

                # Cierre Forzoso a las 15:45 EST (Time-Decay)
                if current_bar_time >= self.flat_time:
                    exit_fill = b_close - (trade.direction * self.slippage_ticks * tick_size)
                    self._finalize_trade(
                        trade=trade,
                        exit_time=str(bar_time),
                        exit_price=exit_fill,
                        reason=ExitReason.TIME_DECAY,
                        multiplier=multiplier,
                        comm_rt=comm_rt,
                        tick_size=tick_size
                    )
                    account.record_trade(trade)
                    self.state = StrategyState.FLAT_COMPLETED
                    return trade

                # Verificación de Stop Loss y Take Profit en barras posteriores
                if trade.direction == 1:
                    if b_low <= trade.stop_loss:
                        exit_fill = trade.stop_loss - (self.slippage_ticks * tick_size)
                        self._finalize_trade(
                            trade=trade,
                            exit_time=str(bar_time),
                            exit_price=exit_fill,
                            reason=ExitReason.STOP_LOSS,
                            multiplier=multiplier,
                            comm_rt=comm_rt,
                            tick_size=tick_size
                        )
                        account.record_trade(trade)
                        self.state = StrategyState.FLAT_COMPLETED
                        return trade
                    elif b_high >= trade.take_profit:
                        exit_fill = trade.take_profit
                        self._finalize_trade(
                            trade=trade,
                            exit_time=str(bar_time),
                            exit_price=exit_fill,
                            reason=ExitReason.TAKE_PROFIT,
                            multiplier=multiplier,
                            comm_rt=comm_rt,
                            tick_size=tick_size
                        )
                        account.record_trade(trade)
                        self.state = StrategyState.FLAT_COMPLETED
                        return trade

                elif trade.direction == -1:
                    if b_high >= trade.stop_loss:
                        exit_fill = trade.stop_loss + (self.slippage_ticks * tick_size)
                        self._finalize_trade(
                            trade=trade,
                            exit_time=str(bar_time),
                            exit_price=exit_fill,
                            reason=ExitReason.STOP_LOSS,
                            multiplier=multiplier,
                            comm_rt=comm_rt,
                            tick_size=tick_size
                        )
                        account.record_trade(trade)
                        self.state = StrategyState.FLAT_COMPLETED
                        return trade
                    elif b_low <= trade.take_profit:
                        exit_fill = trade.take_profit
                        self._finalize_trade(
                            trade=trade,
                            exit_time=str(bar_time),
                            exit_price=exit_fill,
                            reason=ExitReason.TAKE_PROFIT,
                            multiplier=multiplier,
                            comm_rt=comm_rt,
                            tick_size=tick_size
                        )
                        account.record_trade(trade)
                        self.state = StrategyState.FLAT_COMPLETED
                        return trade

        # Si finalizó el día y quedó abierta por alguna razón
        if self.state == StrategyState.IN_POSITION and self.current_trade:
            last_bar = post_or_bars.iloc[-1]
            exit_fill = float(last_bar["Close"]) - (self.current_trade.direction * self.slippage_ticks * tick_size)
            self._finalize_trade(
                trade=self.current_trade,
                exit_time=str(post_or_bars.index[-1]),
                exit_price=exit_fill,
                reason=ExitReason.TIME_DECAY,
                multiplier=multiplier,
                comm_rt=comm_rt,
                tick_size=tick_size
            )
            account.record_trade(self.current_trade)
            self.state = StrategyState.FLAT_COMPLETED
            return self.current_trade

        return None

    def _finalize_trade(
        self,
        trade: TradeRecord,
        exit_time: str,
        exit_price: float,
        reason: ExitReason,
        multiplier: float,
        comm_rt: float,
        tick_size: float
    ) -> None:
        """Calcula el PnL bruto, neto, comisiones y slippage total."""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason

        pnl_points = (trade.exit_price - trade.entry_price) * trade.direction
        trade.gross_pnl_points = round(pnl_points, 4)
        trade.gross_pnl_usd = round(pnl_points * multiplier * trade.contracts, 2)
        trade.commissions_usd = round(comm_rt * trade.contracts, 2)

        # Slippage total estimado en dólares
        entry_slip = self.slippage_ticks * tick_size * multiplier * trade.contracts
        exit_slip = (self.slippage_ticks * tick_size * multiplier * trade.contracts) if reason != ExitReason.TAKE_PROFIT else 0.0
        trade.slippage_usd = round(entry_slip + exit_slip, 2)

        trade.net_pnl_usd = round(trade.gross_pnl_usd - trade.commissions_usd, 2)


# ==============================================================================
# 5. SIMULADOR DE BACKTESTING DE ALTA RESOLUCIÓN: FilteredORBBacktester
# ==============================================================================

class FilteredORBBacktester:
    """
    Motor de backtest especializado para Filtered Quantitative ORB.
    Maneja doble timeframe: barras diarias históricas y velas intradía.
    Genera métricas financieras institucionales y auditoría de Prop Trading.
    """

    def __init__(
        self,
        strategy: FilteredQuantitativeORB,
        account: PropAccount
    ):
        self.strategy = strategy
        self.account = account
        self.trades: List[TradeRecord] = []
        self.daily_equity_curve: Dict[str, float] = {}

    def run(
        self,
        intraday_data: pd.DataFrame,
        daily_data: Optional[pd.DataFrame] = None,
        source_tz: Optional[str] = None,
        auto_normalize_tz: bool = True
    ) -> Dict[str, Any]:
        """
        Ejecuta la simulación día a día sobre el dataset intradía.
        Si daily_data no se suministra, se genera automáticamente agregando las barras intradía.
        """
        # Asegurar DatetimeIndex y normalizar a America/New_York si corresponde
        if auto_normalize_tz:
            df_intra = normalize_to_eastern(intraday_data, source_tz=source_tz)
        else:
            df_intra = intraday_data.copy()
            if not isinstance(df_intra.index, pd.DatetimeIndex):
                if "datetime" in df_intra.columns:
                    df_intra["datetime"] = pd.to_datetime(df_intra["datetime"])
                    df_intra.set_index("datetime", inplace=True)
                else:
                    raise ValueError("intraday_data debe tener DatetimeIndex o columna 'datetime'.")

        df_intra.sort_index(inplace=True)

        # Construir o preparar datos diarios
        if daily_data is None:
            df_daily = df_intra.resample("1D").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }).dropna()
        else:
            df_daily = daily_data.copy()
            if not isinstance(df_daily.index, pd.DatetimeIndex):
                if "datetime" in df_daily.columns:
                    df_daily["datetime"] = pd.to_datetime(df_daily["datetime"])
                    df_daily.set_index("datetime", inplace=True)
                else:
                    raise ValueError("daily_data debe tener DatetimeIndex o columna 'datetime'.")
            df_daily.sort_index(inplace=True)

        unique_dates = sorted(list(set(df_intra.index.date)))
        historical_or_volumes: List[float] = []

        for d in unique_dates:
            day_mask = (df_intra.index.date == d)
            day_intra = df_intra[day_mask]

            # Historial diario estrictamente anterior a hoy (Lookahead Bias = 0)
            prior_daily = df_daily[df_daily.index.date < d]

            self.account.start_session(d)

            # Si la cuenta ya fue liquidada por violación de reglas
            if self.account.is_breached:
                self.daily_equity_curve[str(d)] = self.account.equity
                continue

            trade = self.strategy.process_day(
                session_date=d,
                intraday_bars=day_intra,
                daily_history=prior_daily,
                historical_or_volumes=historical_or_volumes,
                account=self.account
            )

            if trade:
                self.trades.append(trade)

            self.daily_equity_curve[str(d)] = self.account.equity

        return self.generate_performance_report()

    def generate_performance_report(self) -> Dict[str, Any]:
        """Calcula el conjunto completo de métricas cuantitativas y de cumplimiento de fondeo."""
        initial_cap = self.account.initial_capital
        final_equity = self.account.equity
        total_pnl = final_equity - initial_cap
        total_return_pct = (total_pnl / initial_cap) * 100.0

        n_trades = len(self.trades)
        wins = [t for t in self.trades if t.net_pnl_usd > 0]
        losses = [t for t in self.trades if t.net_pnl_usd < 0]
        ties = [t for t in self.trades if t.net_pnl_usd == 0]

        win_rate = (len(wins) / n_trades * 100.0) if n_trades > 0 else 0.0
        gross_profit = sum(t.net_pnl_usd for t in wins)
        gross_loss = abs(sum(t.net_pnl_usd for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        total_commissions = sum(t.commissions_usd for t in self.trades)
        total_slippage = sum(t.slippage_usd for t in self.trades)

        # Retornos diarios para Sharpe / Sortino
        eq_series = pd.Series(self.daily_equity_curve)
        daily_returns = eq_series.pct_change().dropna()

        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
            neg_ret = daily_returns[daily_returns < 0]
            sortino = float((daily_returns.mean() / neg_ret.std()) * np.sqrt(252)) if (len(neg_ret) > 1 and neg_ret.std() > 0) else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        max_dd_usd = self.account.current_drawdown
        max_dd_pct = (max_dd_usd / self.account.peak_equity * 100.0) if self.account.peak_equity > 0 else 0.0

        # Razón de Calmar
        calmar = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

        return {
            "initial_capital": initial_cap,
            "final_equity": round(final_equity, 2),
            "total_net_pnl_usd": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "total_trades": n_trades,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "break_even_trades": len(ties),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2) if not math.isinf(profit_factor) else "inf",
            "gross_profit_usd": round(gross_profit, 2),
            "gross_loss_usd": round(gross_loss, 2),
            "total_commissions_usd": round(total_commissions, 2),
            "total_slippage_usd": round(total_slippage, 2),
            "max_drawdown_usd": round(max_dd_usd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "calmar_ratio": round(calmar, 2),
            "is_prop_breached": self.account.is_breached,
            "breach_reason": self.account.breach_reason or "None (Compliant)"
        }

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Exporta el log completo de trades a un DataFrame de pandas."""
        if not self.trades:
            return pd.DataFrame()
        records = [t.__dict__ for t in self.trades]
        return pd.DataFrame(records)
