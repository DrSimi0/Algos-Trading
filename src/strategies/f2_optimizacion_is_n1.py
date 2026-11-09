"""
src/strategies/f2_optimizacion_is_n1.py
Implementación Matemática y Microestructural de la Versión Alterna Optimizada:
Estrategia "F2-Optimizacion -IS- N1": Filtered Quantitative Opening Range Breakout (ORB).

Fundamento Académico y Cuantitativo:
1. Crabel, T. (1990). Day Trading with Short Term Price Patterns. Traders Press.
2. Zarattini, C., Barbon, A., & Aziz, A. (2024). A Profitable Day Trading Strategy
   For The U.S. Equity Market. Swiss Finance Institute Research Paper No. 24-98.
3. Fetna, M. (2024). Opening-Range Breakout Does Not Survive Trading Costs.
4. Kyle, A. S. (1985). Continuous Auctions and Informed Trader. Econometrica.
5. Cont, R., Kukanov, A., & Stoikov, S. (2014). The Price Impact of Order Book Events. JFE.

Especificación: reports/optimizaciones_estrategias/F2_optimizacion_filtered_orb.md

Mejoras Matemáticas Principales en F2-Optimizacion -IS- N1 frente a F2 Base:
1. Eliminación de Fakeouts por Filtro de Asimetría de Flujo de Órdenes (OFA / CVD):
   - Mide el Cumulative Volume Delta (CVD) en la subasta de apertura [09:30, 09:45).
   - Exige que la asimetría direccional del flujo supere un umbral dinámico ajustado
     por volatilidad (theta_flow), garantizando que las rupturas alcistas estén respaldadas
     por compras agresivas netas y las bajistas por ventas agresivas netas.
2. Dinamización Analítica del Umbral de Volumen Relativo Crítico (RVOL_crit):
   - Sustituye el corte rígido binario RVOL >= 1.80 por una función continua y decreciente
     de la tasa de compresión previa (R_comp), acotada en [1.20, 1.70].
   - Desbloquea la parálisis operativa del modelo base en regímenes de alta compresión.
3. Reemplazo del Stop-Loss Rígido por una Barrera Elástica Proporcional:
   - Sustituye el SL estático en el punto medio (M_OR) por una barrera elástica proporcional
     a la anchura del rango inicial (k_elastic * Range_OR), acotada por la volatilidad
     diaria (0.85 * ATR_daily) para blindar el balance ante aperturas desmedidas.
4. Mecanismo de Breakeven Protector Dinámico:
   - Al alcanzar una expansión favorable de +1.0R (mitad del objetivo Take-Profit 2.0R),
     la barrera elástica se ajusta inmediatamente a precio de entrada + amortiguador de fricción
     (2 ticks), garantizando la asimetría positiva de la distribución de retornos.
5. Cumplimiento Estricto de Prop Trading:
   - Daily Loss Limit ($1,000), Max Trailing Drawdown ($2,000), disyuntor preventivo al 85% de DLL,
     atenuación geométrica por racha perdedora phi(K_loss) y conmutación automática estándar/micro (ES/MES, NQ/MNQ).
"""

from dataclasses import dataclass, field
from datetime import time, datetime
from enum import Enum
import math
from typing import Dict, Any, List, Optional, Tuple, Sequence
import numpy as np
import pandas as pd

from src.core.base_strategy import BaseStrategy
from src.core.futures_specs import get_spec
from src.strategies.filtered_orb import (
    StrategyState,
    ExitReason,
    TradeRecord,
    PropAccount,
    check_nr4,
    check_inside_day,
    calculate_crabel_stretch,
    calculate_position_size,
    normalize_to_eastern,
)


# ==============================================================================
# 1. REGISTRO EXTENDIDO DE OPERACIÓN (F2-OPT-IS-N1)
# ==============================================================================

@dataclass
class F2OptTradeRecord(TradeRecord):
    """
    Registro extendido de trade para la estrategia F2-Optimizacion -IS- N1.
    Registra métricas microestructurales de volumen, flujo de órdenes y barrera elástica.
    """
    r_comp: float = 0.0                      # Ratio continuo de compresión previa
    rvol_crit: float = 0.0                   # Umbral dinámico de RVOL requerido
    rvol: float = 0.0                        # RVOL efectivamente medido en OR
    cvd_or: float = 0.0                      # Delta de volumen acumulado en OR
    ofa_ratio: float = 0.0                   # Asimetría normalizada de flujo (CVD / Volume)
    theta_flow: float = 0.0                  # Umbral dinámico exigido de asimetría
    atr_daily: float = 0.0                   # Volatilidad diaria ATR
    elastic_barrier_risk: float = 0.0        # Distancia analítica Delta_risk^* adoptada
    initial_stop_loss: float = 0.0           # Nivel inicial de la barrera de stop
    breakeven_triggered: bool = False        # True si se activó el ajuste elástico a breakeven


# ==============================================================================
# 2. FUNCIONES MATEMÁTICAS FORMALES DE OPTIMIZACIÓN
# ==============================================================================

def calculate_rcomp(daily_bars: pd.DataFrame) -> Tuple[float, bool]:
    """
    Calcula el ratio continuo de compresión previa de volatilidad (R_{comp, t}):
    R_{comp, t} = Range_{t-1} / ( (1/3) * sum_{k=2}^4 Range_{t-k} )
    donde Range_{t-k} = High_{t-k} - Low_{t-k}.

    Condición de admisibilidad de compresión previa (Phi_{comp, t}):
    Phi_{comp, t} = (R_{comp, t} <= 0.92) OR (Inside Day en t-1 frente a t-2).

    Requiere al menos 4 barras diarias previas sin lookahead bias.
    """
    if len(daily_bars) < 4:
        return 1.0, False

    highs = daily_bars["High"].values
    lows = daily_bars["Low"].values

    if np.any(highs < lows):
        raise ValueError("Datos corruptos detectados en daily_bars: High < Low.")

    ranges = highs - lows
    r_prev = ranges[-1]
    r_prior_3 = ranges[-4:-1]
    mean_prior_3 = float(np.mean(r_prior_3))

    if mean_prior_3 <= 0.0:
        return 1.0, False

    r_comp = float(r_prev / mean_prior_3)

    # Condición disyuntiva: ratio continuo <= 0.92 o Inside Day
    inside_day = check_inside_day(highs[-2:], lows[-2:])
    is_compressed = bool((r_comp <= 0.92) or inside_day)

    return r_comp, is_compressed


def calculate_rvol_dynamic_threshold(
    r_comp: float,
    rvol_floor: float = 1.20,
    rvol_cap: float = 1.70,
    rcomp_min: float = 0.50,
    rcomp_max: float = 0.92
) -> float:
    """
    Dinamización analítica suave del umbral de volumen relativo crítico RVOL_{crit}(R_comp):
    RVOL_{crit}(R_{comp}) = min(1.70, max(1.20, 1.20 + 0.50 * ( (R_{comp} - 0.50) / (0.92 - 0.50) ) ))

    Propiedad:
    - Compresión extrema (R_comp <= 0.50) -> RVOL_{crit} = 1.20 (20% sobre la media).
    - Compresión moderada (R_comp = 0.92) -> RVOL_{crit} = 1.70 (70% sobre la media).
    - Monótona creciente y acotada en [1.20, 1.70].
    """
    if math.isnan(r_comp) or r_comp < 0.0:
        return rvol_cap

    span = rcomp_max - rcomp_min
    if span <= 0.0:
        return rvol_cap

    normalized_comp = (r_comp - rcomp_min) / span
    scaled_threshold = rvol_floor + (rvol_cap - rvol_floor) * normalized_comp
    return float(min(rvol_cap, max(rvol_floor, scaled_threshold)))


def calculate_order_flow_asymmetry(
    or_bars: pd.DataFrame,
    eps_0: float = 1e-6
) -> Tuple[float, float]:
    """
    Calcula el Delta de Volumen Acumulado y la Asimetría del Flujo de Órdenes
    en los primeros 15 minutos (09:30 - 09:45 EST).

    Aproximación hidrodinámica analítica para datos OHLCV:
    CVD_{OR, t}^{approx} = sum_{j in OR} V_j * ( (C_j - O_j) / (H_j - L_j + eps_0) )
    OFA_{ratio, t} = CVD_{OR, t}^{approx} / ( sum_{j in OR} V_j + eps_0 )

    Retorna: (CVD_{OR}, OFA_{ratio} in [-1.0, +1.0])
    """
    if or_bars.empty:
        return 0.0, 0.0

    opens = or_bars["Open"].values
    highs = or_bars["High"].values
    lows = or_bars["Low"].values
    closes = or_bars["Close"].values
    volumes = or_bars["Volume"].values

    if (
        np.any(np.isnan(opens))
        or np.any(np.isnan(highs))
        or np.any(np.isnan(lows))
        or np.any(np.isnan(closes))
        or np.any(np.isnan(volumes))
    ):
        return 0.0, 0.0

    if np.any(highs < lows):
        raise ValueError("Datos corruptos detectados en Opening Range: High < Low.")

    total_volume = float(np.sum(volumes))
    if total_volume <= 0.0:
        return 0.0, 0.0

    bar_ranges = np.where(highs > lows, highs - lows, eps_0)
    directional_weights = (closes - opens) / bar_ranges
    bar_deltas = volumes * directional_weights

    cvd_or = float(np.sum(bar_deltas))
    if math.isnan(cvd_or):
        return 0.0, 0.0

    ofa_ratio = float(cvd_or / total_volume) if total_volume > 0 else 0.0
    if math.isnan(ofa_ratio):
        return 0.0, 0.0

    # Acotar numéricamente en [-1.0, 1.0]
    ofa_ratio = max(-1.0, min(1.0, ofa_ratio))
    return cvd_or, ofa_ratio


def calculate_dynamic_flow_threshold(
    r_comp: float,
    base_flow_threshold: float = 0.05,
    max_flow_threshold: float = 0.35
) -> float:
    """
    Calcula el umbral dinámico de asimetría de flujo de órdenes ajustado por volatilidad:
    theta_{flow}(R_comp) = min(max_flow, max(0.0, base_flow * (R_comp / 0.92)))

    A mayor R_comp (menor compresión / mayor dispersión), mayor es la propensión a fakeouts,
    por lo que se exige mayor dominancia neta compradora/vendedora en la subasta de apertura.
    """
    if math.isnan(r_comp) or r_comp <= 0.0:
        return base_flow_threshold

    ratio = r_comp / 0.92
    theta = base_flow_threshold * ratio
    return float(min(max_flow_threshold, max(0.0, theta)))


def evaluate_flow_confirmation(
    direction: int,
    ofa_ratio: float,
    theta_flow: float
) -> bool:
    """
    Verifica la confirmación direccional por asimetría de flujo:
    - Long (+1): ofa_ratio >= +theta_flow (compradores agresivos dominantes)
    - Short (-1): ofa_ratio <= -theta_flow (vendedores agresivos dominantes)
    """
    if direction == 1:
        return bool(ofa_ratio >= theta_flow)
    elif direction == -1:
        return bool(ofa_ratio <= -theta_flow)
    return False


def calculate_daily_atr(daily_bars: pd.DataFrame, lookback: int = 14) -> float:
    """
    Calcula el Rango Verdadero Medio (Average True Range) sobre las barras diarias históricas:
    TR_{t-k} = max( H_{t-k} - L_{t-k}, |H_{t-k} - C_{t-k-1}|, |L_{t-k} - C_{t-k-1}| )
    ATR = (1/K) * sum TR_{t-k}.
    """
    if len(daily_bars) < 2:
        return 0.0

    subset = daily_bars.iloc[-(lookback + 1):] if len(daily_bars) >= (lookback + 1) else daily_bars
    highs = subset["High"].values
    lows = subset["Low"].values
    closes = subset["Close"].values

    if np.any(np.isnan(highs)) or np.any(np.isnan(lows)) or np.any(np.isnan(closes)):
        return 0.0

    if np.any(highs < lows):
        raise ValueError("Datos corruptos detectados en daily_bars: High < Low.")

    tr_list = []
    for i in range(1, len(subset)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_list.append(max(hl, hc, lc))

    if not tr_list:
        return float(np.mean(highs - lows)) if len(highs) > 0 else 0.0

    return float(np.mean(tr_list))


def calculate_elastic_delta_risk(
    range_or: float,
    stretch: float,
    atr_daily: float,
    k_elastic: float = 0.50,
    lambda_stretch: float = 0.40,
    atr_cap_mult: float = 0.85
) -> float:
    r"""
    Calcula la barrera elástica proporcional a la anchura del rango inicial acotada por volatilidad:
    Delta_{risk, t}^* = min( k_elastic * Range_{OR, t} + lambda_stretch * Stretch_t, \; atr_cap_mult * ATR_{daily, t} )

    Si ATR_daily es cero o no está disponible, se utiliza el valor proporcional al rango sin acotar.
    """
    if math.isnan(range_or) or range_or < 0.0:
        raise ValueError(f"Range_OR no puede ser negativo ni NaN: {range_or}.")
    if math.isnan(stretch) or stretch < 0.0:
        raise ValueError(f"Stretch no puede ser negativo ni NaN: {stretch}.")

    proportional_risk = (k_elastic * range_or) + (lambda_stretch * stretch)

    if not math.isnan(atr_daily) and atr_daily > 0.0:
        vol_cap = atr_cap_mult * atr_daily
        bounded_risk = min(proportional_risk, vol_cap)
        return float(max(0.25, bounded_risk))

    return float(max(0.25, proportional_risk))


# ==============================================================================
# 3. CLASE PRINCIPAL: F2OptimizationISN1
# ==============================================================================

class F2OptimizationISN1(BaseStrategy):
    """
    Estrategia F2-Optimizacion -IS- N1:
    Filtered Quantitative Opening Range Breakout con Filtro Dinámico de Asimetría de Flujo
    y Barrera Elástica de Riesgo Proporcional.
    """

    def __init__(
        self,
        std_symbol: str = "ES",
        micro_symbol: str = "MES",
        lambda_stretch: float = 0.40,
        base_flow_threshold: float = 0.05,
        k_elastic: float = 0.50,
        atr_cap_mult: float = 0.85,
        rcomp_threshold: float = 0.92,
        rvol_floor: float = 1.20,
        rvol_cap: float = 1.70,
        stretch_lookback: int = 10,
        rvol_lookback: int = 20,
        atr_lookback: int = 14,
        enable_breakeven: bool = True,
        breakeven_r_mult: float = 1.0,
        max_micros: int = 12,
        market_open: time = time(9, 30),
        or_end: time = time(9, 45),
        entry_cutoff: time = time(12, 0),
        flat_time: time = time(15, 45),
        slippage_ticks: float = 1.0,
        dll_soft_limit_pct: float = 0.85
    ):
        super().__init__(name="F2-Optimizacion -IS- N1", params={})

        self.std_symbol = std_symbol
        self.micro_symbol = micro_symbol
        self.lambda_stretch = lambda_stretch
        self.base_flow_threshold = base_flow_threshold
        self.k_elastic = k_elastic
        self.atr_cap_mult = atr_cap_mult
        self.rcomp_threshold = rcomp_threshold
        self.rvol_floor = rvol_floor
        self.rvol_cap = rvol_cap

        self.stretch_lookback = stretch_lookback
        self.rvol_lookback = rvol_lookback
        self.atr_lookback = atr_lookback

        self.enable_breakeven = enable_breakeven
        self.breakeven_r_mult = breakeven_r_mult
        self.max_micros = max_micros

        self.market_open = market_open
        self.or_end = or_end
        self.entry_cutoff = entry_cutoff
        self.flat_time = flat_time
        self.slippage_ticks = slippage_ticks
        self.dll_soft_limit_pct = dll_soft_limit_pct

        # Variables de estado diario
        self.state: StrategyState = StrategyState.MEASURING_OR
        self.current_trade: Optional[F2OptTradeRecord] = None

        # Variables analíticas de la sesión actual
        self.r_comp: float = 1.0
        self.is_compressed: bool = False
        self.stretch: float = 0.0
        self.atr_daily: float = 0.0

        self.orh: float = 0.0
        self.orl: float = 0.0
        self.range_or: float = 0.0
        self.midpoint: float = 0.0
        self.or_volume: float = 0.0

        self.rvol: float = 0.0
        self.rvol_crit: float = 1.70
        self.has_volume_expansion: bool = False

        self.cvd_or: float = 0.0
        self.ofa_ratio: float = 0.0
        self.theta_flow: float = 0.0
        self.flow_long_confirmed: bool = False
        self.flow_short_confirmed: bool = False

        self.delta_risk: float = 0.0
        self.buy_trigger: float = 0.0
        self.sell_trigger: float = 0.0

        # Contadores de auditoría y diagnóstico
        self.fakeouts_avoided_long: int = 0
        self.fakeouts_avoided_short: int = 0
        self.breakeven_activations: int = 0

    def prepare_daily_context(
        self,
        daily_history: pd.DataFrame,
        historical_or_volumes: List[float]
    ) -> bool:
        """
        Calcula compresión previa continua R_comp, Stretch de Crabel y ATR diario
        utilizando estrictamente barras previas a la sesión actual (Lookahead bias = 0).
        """
        if len(daily_history) < max(4, self.stretch_lookback):
            self.r_comp = 1.0
            self.is_compressed = False
            self.stretch = 0.0
            self.atr_daily = 0.0
            return False

        self.r_comp, self.is_compressed = calculate_rcomp(daily_history)
        self.stretch = calculate_crabel_stretch(daily_history, lookback=self.stretch_lookback)
        self.atr_daily = calculate_daily_atr(daily_history, lookback=self.atr_lookback)
        return self.is_compressed

    def evaluate_opening_range(
        self,
        or_bars: pd.DataFrame,
        historical_or_volumes: List[float]
    ) -> bool:
        """
        Construye el Opening Range [09:30, 09:45):
        - Mide ORH, ORL, Range_OR, Volumen OR.
        - Calcula RVOL institucional y el umbral dinámico crítico RVOL_crit(R_comp).
        - Calcula el Cumulative Volume Delta (CVD) y la Asimetría del Flujo (OFA).
        - Obtiene el umbral dinámico de flujo theta_flow(R_comp) y valida confirmación direccional.
        - Calcula la barrera elástica Delta_risk^* y los gatillos de entrada.
        """
        if or_bars.empty:
            self._reset_daily_metrics()
            self.state = StrategyState.ORDER_CANCELLED
            return False

        if (or_bars["High"] < or_bars["Low"]).any():
            raise ValueError("Barras corruptas detectadas en Opening Range: High < Low.")

        self.orh = float(or_bars["High"].max())
        self.orl = float(or_bars["Low"].min())
        self.range_or = max(0.0, self.orh - self.orl)
        self.midpoint = (self.orh + self.orl) / 2.0
        self.or_volume = float(or_bars["Volume"].sum())

        if self.or_volume < 0:
            raise ValueError("Volumen negativo detectado en Opening Range.")

        # 1. RVOL dinámico ajustado por compresión
        self.rvol_crit = calculate_rvol_dynamic_threshold(
            self.r_comp,
            rvol_floor=self.rvol_floor,
            rvol_cap=self.rvol_cap
        )

        # Cálculo de RVOL empírico contra histórico de aperturas
        if self.or_volume > 0.0 and len(historical_or_volumes) > 0:
            recent_vols = historical_or_volumes[-self.rvol_lookback:]
            avg_vol = float(np.mean(recent_vols))
            self.rvol = float(self.or_volume / avg_vol) if avg_vol > 0.0 else 0.0
        else:
            self.rvol = 0.0

        self.has_volume_expansion = bool(self.rvol >= self.rvol_crit)

        # 2. Asimetría de Flujo de Órdenes (OFA / CVD)
        self.cvd_or, self.ofa_ratio = calculate_order_flow_asymmetry(or_bars)
        self.theta_flow = calculate_dynamic_flow_threshold(
            self.r_comp,
            base_flow_threshold=self.base_flow_threshold
        )

        self.flow_long_confirmed = evaluate_flow_confirmation(1, self.ofa_ratio, self.theta_flow)
        self.flow_short_confirmed = evaluate_flow_confirmation(-1, self.ofa_ratio, self.theta_flow)

        # 3. Barrera Elástica Proporcional a Range_OR acotada por ATR
        self.delta_risk = calculate_elastic_delta_risk(
            range_or=self.range_or,
            stretch=self.stretch,
            atr_daily=self.atr_daily,
            k_elastic=self.k_elastic,
            lambda_stretch=self.lambda_stretch,
            atr_cap_mult=self.atr_cap_mult
        )

        # Gatillos de ruptura con amortiguador Stretch
        buffer = self.lambda_stretch * self.stretch
        self.buy_trigger = self.orh + buffer
        self.sell_trigger = self.orl - buffer

        return True

    def process_day(
        self,
        session_date: Any,
        intraday_bars: pd.DataFrame,
        daily_history: pd.DataFrame,
        historical_or_volumes: List[float],
        account: PropAccount
    ) -> Optional[F2OptTradeRecord]:
        """
        Ejecuta el ciclo de vida diario de F2-Optimizacion -IS- N1:
        1. Contexto previo y compresión R_comp.
        2. Medición de Opening Range, RVOL dinámico y asimetría de flujo OFA.
        3. Filtro conjunto: Phi_comp AND Phi_vol^* AND (Phi_flow^Long OR Phi_flow^Short).
        4. Dimensionamiento conmutado estándar/micro y disyuntor de pérdida diaria.
        5. Ejecución con barrera elástica proporcional y breakeven a +1.0R.
        """
        self.state = StrategyState.MEASURING_OR
        self.current_trade = None

        if intraday_bars.empty:
            return None

        # 1. Filtro Previo de Compresión de Volatilidad
        self.prepare_daily_context(daily_history, historical_or_volumes)

        # Separar barras según ventanas temporales
        times = intraday_bars.index.time
        or_mask = (times >= self.market_open) & (times < self.or_end)
        or_bars = intraday_bars[or_mask]

        if or_bars.empty:
            return None

        # 2. Medición y Calibración del Opening Range
        self.evaluate_opening_range(or_bars, historical_or_volumes)

        # Registrar volumen para el historial institucional
        if self.or_volume > 0:
            historical_or_volumes.append(self.or_volume)

        # 3. Disyuntor Preventivo de Pérdida Diaria (85% DLL)
        if account.daily_realized_loss >= (account.dll * self.dll_soft_limit_pct):
            self.state = StrategyState.HALT_ACCOUNT_BREACHED
            return None

        # 4. Evaluación de Filtros Conjuntos
        self.state = StrategyState.EVALUATING_FILTERS
        if not (self.is_compressed and self.has_volume_expansion):
            self.state = StrategyState.ORDER_CANCELLED
            return None

        # Al menos una de las direcciones debe estar respaldada por el flujo de órdenes
        if not (self.flow_long_confirmed or self.flow_short_confirmed):
            self.state = StrategyState.ORDER_CANCELLED
            return None

        # 5. Dimensionamiento y Presupuesto de Riesgo Prop Trading
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

        # 6. Armado de Órdenes Condicionadas en W_entry (09:45, 12:00]
        self.state = StrategyState.ARMED_BREAKOUT
        spec = get_spec(symbol_to_trade)
        tick_size = spec["tick"]
        multiplier = spec["multiplier"]
        comm_rt = spec["commission_rt"]
        is_micro = (symbol_to_trade == self.micro_symbol)

        post_or_mask = (times >= self.or_end)
        post_or_bars = intraday_bars[post_or_mask]

        fakeout_flagged_long = False
        fakeout_flagged_short = False

        for bar_time, bar in post_or_bars.iterrows():
            current_bar_time = bar_time.time()

            # A) Si estamos esperando entrada
            if self.state == StrategyState.ARMED_BREAKOUT:
                # Cancelación de orden al vencer la ventana de entrada (12:00 EST)
                if current_bar_time > self.entry_cutoff:
                    self.state = StrategyState.ORDER_CANCELLED
                    break

                b_open = float(bar["Open"])
                b_high = float(bar["High"])
                b_low = float(bar["Low"])
                b_close = float(bar["Close"])

                bullish_attempt = (b_high >= self.buy_trigger)
                bearish_attempt = (b_low <= self.sell_trigger)

                # Auditoría de fakeouts evitados (contabilizado una vez por dirección por sesión)
                if bullish_attempt and not self.flow_long_confirmed:
                    if not fakeout_flagged_long:
                        self.fakeouts_avoided_long += 1
                        fakeout_flagged_long = True

                if bearish_attempt and not self.flow_short_confirmed:
                    if not fakeout_flagged_short:
                        self.fakeouts_avoided_short += 1
                        fakeout_flagged_short = True

                can_enter_long = bullish_attempt and self.flow_long_confirmed
                can_enter_short = bearish_attempt and self.flow_short_confirmed

                # Si ambas direcciones son posibles en la misma vela, desempatar por signo del flujo OFA
                if can_enter_long and can_enter_short:
                    if self.ofa_ratio < 0:
                        can_enter_long = False
                    else:
                        can_enter_short = False

                # Intento de Ruptura Alcista
                if can_enter_long:
                    fill_price = self.buy_trigger + (self.slippage_ticks * tick_size)
                    # Barrera elástica proporcional al rango inicial
                    sl_price = fill_price - self.delta_risk
                    tp_price = fill_price + (2.0 * self.delta_risk)

                    self.current_trade = F2OptTradeRecord(
                        date=str(session_date),
                        symbol=symbol_to_trade,
                        direction=1,
                        contracts=n_contracts,
                        is_micro=is_micro,
                        entry_time=str(bar_time),
                        entry_price=fill_price,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        delta_risk=self.delta_risk,
                        r_comp=round(self.r_comp, 4),
                        rvol_crit=round(self.rvol_crit, 2),
                        rvol=round(self.rvol, 2),
                        cvd_or=round(self.cvd_or, 2),
                        ofa_ratio=round(self.ofa_ratio, 4),
                        theta_flow=round(self.theta_flow, 4),
                        atr_daily=round(self.atr_daily, 2),
                        elastic_barrier_risk=round(self.delta_risk, 4),
                        initial_stop_loss=sl_price
                    )
                    self.state = StrategyState.IN_POSITION

                    # Evaluación intra-barra en la vela de entrada:
                    # 1. Take Profit intra-barra
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

                    # 2. Mecanismo elástico de Breakeven en la misma barra si alcanza +1.0R
                    if self.enable_breakeven:
                        be_trigger_price = fill_price + (self.breakeven_r_mult * self.delta_risk)
                        if b_high >= be_trigger_price:
                            be_sl = fill_price + (2.0 * tick_size)
                            if be_sl > self.current_trade.stop_loss:
                                self.current_trade.stop_loss = be_sl
                                self.current_trade.breakeven_triggered = True
                                self.breakeven_activations += 1

                    # 3. Stop Loss en la barra de entrada
                    effective_sl = self.current_trade.stop_loss
                    is_same_bar_sl = (b_close <= effective_sl) or (b_open >= self.buy_trigger and b_low <= effective_sl)
                    if is_same_bar_sl:
                        exit_fill = effective_sl - (self.slippage_ticks * tick_size)
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

                # Intento de Ruptura Bajista
                elif can_enter_short:
                    fill_price = self.sell_trigger - (self.slippage_ticks * tick_size)
                    # Barrera elástica proporcional al rango inicial
                    sl_price = fill_price + self.delta_risk
                    tp_price = fill_price - (2.0 * self.delta_risk)

                    self.current_trade = F2OptTradeRecord(
                        date=str(session_date),
                        symbol=symbol_to_trade,
                        direction=-1,
                        contracts=n_contracts,
                        is_micro=is_micro,
                        entry_time=str(bar_time),
                        entry_price=fill_price,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        delta_risk=self.delta_risk,
                        r_comp=round(self.r_comp, 4),
                        rvol_crit=round(self.rvol_crit, 2),
                        rvol=round(self.rvol, 2),
                        cvd_or=round(self.cvd_or, 2),
                        ofa_ratio=round(self.ofa_ratio, 4),
                        theta_flow=round(self.theta_flow, 4),
                        atr_daily=round(self.atr_daily, 2),
                        elastic_barrier_risk=round(self.delta_risk, 4),
                        initial_stop_loss=sl_price
                    )
                    self.state = StrategyState.IN_POSITION

                    # Evaluación intra-barra en la vela de entrada:
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

                    # 2. Mecanismo elástico de Breakeven en la misma barra si alcanza +1.0R
                    if self.enable_breakeven:
                        be_trigger_price = fill_price - (self.breakeven_r_mult * self.delta_risk)
                        if b_low <= be_trigger_price:
                            be_sl = fill_price - (2.0 * tick_size)
                            if be_sl < self.current_trade.stop_loss:
                                self.current_trade.stop_loss = be_sl
                                self.current_trade.breakeven_triggered = True
                                self.breakeven_activations += 1

                    # 3. Stop Loss en la barra de entrada
                    effective_sl = self.current_trade.stop_loss
                    is_same_bar_sl = (b_close >= effective_sl) or (b_open <= self.sell_trigger and b_high >= effective_sl)
                    if is_same_bar_sl:
                        exit_fill = effective_sl + (self.slippage_ticks * tick_size)
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

                # Cierre Forzoso a las 15:45 EST (Neutralización determinista)
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

                # Actualización de Breakeven elástico si alcanza +1.0R
                if self.enable_breakeven and not trade.breakeven_triggered:
                    if trade.direction == 1:
                        if b_high >= (trade.entry_price + self.breakeven_r_mult * trade.delta_risk):
                            be_sl = trade.entry_price + (2.0 * tick_size)
                            if be_sl > trade.stop_loss:
                                trade.stop_loss = be_sl
                                trade.breakeven_triggered = True
                                self.breakeven_activations += 1
                    elif trade.direction == -1:
                        if b_low <= (trade.entry_price - self.breakeven_r_mult * trade.delta_risk):
                            be_sl = trade.entry_price - (2.0 * tick_size)
                            if be_sl < trade.stop_loss:
                                trade.stop_loss = be_sl
                                trade.breakeven_triggered = True
                                self.breakeven_activations += 1

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

        # Cierre al término de sesión si la orden siguió abierta
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
        trade: F2OptTradeRecord,
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

        entry_slip = self.slippage_ticks * tick_size * multiplier * trade.contracts
        exit_slip = (self.slippage_ticks * tick_size * multiplier * trade.contracts) if reason != ExitReason.TAKE_PROFIT else 0.0
        trade.slippage_usd = round(entry_slip + exit_slip, 2)

        trade.net_pnl_usd = round(trade.gross_pnl_usd - trade.commissions_usd, 2)

    def _reset_daily_metrics(self) -> None:
        """Reinicia las métricas intra-día."""
        self.orh = 0.0
        self.orl = 0.0
        self.range_or = 0.0
        self.or_volume = 0.0
        self.midpoint = 0.0
        self.rvol = 0.0
        self.rvol_crit = 1.70
        self.has_volume_expansion = False
        self.cvd_or = 0.0
        self.ofa_ratio = 0.0
        self.theta_flow = 0.0
        self.flow_long_confirmed = False
        self.flow_short_confirmed = False
        self.delta_risk = 0.0
        self.buy_trigger = 0.0
        self.sell_trigger = 0.0

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Interfaz uniforme compatible con BaseStrategy.
        Genera una serie temporal con las señales adoptadas (+1.0, -1.0, 0.0).
        """
        self.validate_data(data)
        signals = pd.Series(0.0, index=data.index)

        # Normalizar datos a Eastern para asegurar alineación con la subasta de apertura
        df_norm = normalize_to_eastern(data)

        # Agrupar por días calendario
        dates = sorted(list(set(df_norm.index.date)))
        account = PropAccount(initial_capital=50000.0)
        hist_vols: List[float] = []

        # Construir diario
        daily_df = df_norm.resample("1D").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna()

        for d in dates:
            account.start_session(d)

            day_mask = (df_norm.index.date == d)
            day_data = df_norm[day_mask]
            prior_daily = daily_df[daily_df.index.date < d]

            trade = self.process_day(
                session_date=d,
                intraday_bars=day_data,
                daily_history=prior_daily,
                historical_or_volumes=hist_vols,
                account=account
            )
            if trade and trade.entry_time and trade.exit_time:
                try:
                    t_in = pd.to_datetime(trade.entry_time)
                    t_out = pd.to_datetime(trade.exit_time)
                    if df_norm.index.tz is not None:
                        if t_in.tzinfo is None:
                            t_in = t_in.tz_localize(df_norm.index.tz)
                        else:
                            t_in = t_in.tz_convert(df_norm.index.tz)
                        if t_out.tzinfo is None:
                            t_out = t_out.tz_localize(df_norm.index.tz)
                        else:
                            t_out = t_out.tz_convert(df_norm.index.tz)

                    pos_mask = (df_norm.index >= t_in) & (df_norm.index <= t_out)
                    signals.iloc[pos_mask] = float(trade.direction)
                except Exception:
                    pass

        return signals


# ==============================================================================
# 4. MOTOR DE BACKTEST: F2OptimizationISN1Backtester
# ==============================================================================

class F2OptimizationISN1Backtester:
    """
    Motor de simulación y evaluación cuantitativa para F2-Optimizacion -IS- N1.
    Audita el rendimiento financiero y el cumplimiento estricto de reglas de Prop Trading.
    """

    def __init__(
        self,
        strategy: F2OptimizationISN1,
        account: PropAccount
    ):
        self.strategy = strategy
        self.account = account
        self.trades: List[F2OptTradeRecord] = []
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
        if intraday_data.empty:
            raise ValueError("intraday_data no puede estar vacío.")

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
            prior_daily = df_daily[df_daily.index.date < d]

            self.account.start_session(d)

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
        """Calcula el conjunto completo de métricas financieras y auditoría de Prop Trading."""
        initial_cap = self.account.initial_capital
        final_equity = self.account.equity
        total_pnl = final_equity - initial_cap
        total_return_pct = (total_pnl / initial_cap) * 100.0 if initial_cap > 0 else 0.0

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
        calmar = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

        # Métricas específicas de optimización F2
        total_fakeouts_avoided = self.strategy.fakeouts_avoided_long + self.strategy.fakeouts_avoided_short
        be_trades = [t for t in self.trades if t.breakeven_triggered]

        return {
            "strategy": "F2-Optimizacion -IS- N1",
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
            "fakeouts_avoided_total": total_fakeouts_avoided,
            "fakeouts_avoided_long": self.strategy.fakeouts_avoided_long,
            "fakeouts_avoided_short": self.strategy.fakeouts_avoided_short,
            "breakeven_activations": self.strategy.breakeven_activations,
            "trades_with_breakeven": len(be_trades),
            "is_prop_breached": self.account.is_breached,
            "breach_reason": self.account.breach_reason or "None (Compliant)"
        }

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Exporta el log completo de trades a un DataFrame de pandas."""
        if not self.trades:
            return pd.DataFrame()
        records = [t.__dict__ for t in self.trades]
        return pd.DataFrame(records)


# ==============================================================================
# 5. ALIASES DE COMPATIBILIDAD
# ==============================================================================

F2OptimizacionISN1 = F2OptimizationISN1
F2OptimizacionISN1Backtester = F2OptimizationISN1Backtester
F2OptimizationISN1Strategy = F2OptimizationISN1
F2OptimizacionISN1Strategy = F2OptimizationISN1
