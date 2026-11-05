"""
src/strategies/anomalias_temporales_vrp.py
Implementación Matemática y Microestructural de la Estrategia F9:
Anomalías Temporales y Variance Risk Premium (VRP).

Fundamento Académico:
1. Bollerslev, T., Tauchen, G., & Zhou, H. (2009). Expected Stock Returns and Variance
   Risk Premia. The Review of Financial Studies, 22(11), 4463–4492.
2. Boyarchenko, N., Larsen, J. C., & Whelan, P. (2023). The Overnight Drift.
   Federal Reserve Bank of New York Staff Reports, No. 1061.
3. Kurov, A., Sancetta, A., Strasser, G., & Wolfe, M. H. (2019). Price Drift Before
   U.S. Macroeconomic News: Private Information or Pre-Announcement Drift? JFQA, 54(1), 449–479.
4. Simon, D. P., & Campasano, J. (2014). The VIX Futures Basis: Evidence and Trading Strategies.
   The Journal of Derivatives, 22(2), 60–69.

Especificación Formal: reports/estrategias_matematicas/F9_anomalias_temporales_vrp.md
"""

from dataclasses import dataclass, field
from datetime import time, datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Sequence, Set
import math
import numpy as np
import pandas as pd

from src.core.base_strategy import BaseStrategy
from src.core.futures_specs import get_spec


# ==============================================================================
# 1. ENUMS Y ESTRUCTURAS DE DATOS DE ESTADO Y TRANSACCIÓN (SECCIÓN 2)
# ==============================================================================

class VRPExitReason(str, Enum):
    DETERMINISTIC_TIME_EXIT = "DETERMINISTIC_TIME_EXIT"  # 08:30:00 EST (ON) o T-3m (Macro)
    STOP_LOSS = "STOP_LOSS"                              # 0.50% (ON) o 10.0 pts (Macro)
    EMBARGO_FLAT = "EMBARGO_FLAT"                        # Cierre de emergencia previo a ventana de embargo
    TERMINAL_EOD = "TERMINAL_EOD"                        # 16:45:00 EST cierre previo a CME maintenance
    RISK_BREACH = "RISK_BREACH"                          # Violación de DLL o MTDD


class VRPRegimeState(int, Enum):
    COMPLACENCY_BULLISH = 1    # Z_VRP >= 1.50 y TSR <= 0.95 (Sesgo comprador favorable)
    NEUTRAL = 0                # Régimen ordinario neutral
    BACKWARDATION_PANIC = -1   # Z_VRP <= -1.20 o TSR >= 1.00 (Bloqueo total de compras)


class SubModelType(str, Enum):
    OVERNIGHT = "OVERNIGHT"    # Deriva nocturna condicional (Boyarchenko et al., 2023)
    PRE_MACRO = "PRE_MACRO"    # Deriva pre-anuncio informada (Kurov et al., 2019)


@dataclass
class VRPTradeRecord:
    """Registro analítico detallado de una operación de anomalía temporal."""
    trade_id: int
    date: str
    submodel: str                              # "OVERNIGHT" o "PRE_MACRO"
    direction: int                             # +1 = Long, -1 = Short
    contracts: int                             # Cantidad de contratos operados
    contract_type: str                         # "STD" o "MICRO"
    symbol: str                                # "ES", "MES", "NQ", "MNQ"
    entry_time: str                            # Timestamp de entrada (EST)
    entry_price: float                         # Precio de llenado en entrada (con slippage)
    stop_loss_price: float                     # Precio de Stop Loss
    raw_entry_price: float = 0.0               # Precio bruto sin slippage
    raw_exit_price: Optional[float] = None     # Precio bruto de salida sin slippage
    exit_time: Optional[str] = None            # Timestamp de salida (EST)
    exit_price: Optional[float] = None         # Precio de llenado en salida (con slippage)
    exit_reason: Optional[VRPExitReason] = None
    bars_held: int = 0
    gross_pnl_pts: float = 0.0
    gross_pnl_usd: float = 0.0
    commissions_usd: float = 0.0
    slippage_usd: float = 0.0
    friction_usd: float = 0.0
    net_pnl_usd: float = 0.0
    running_balance: float = 0.0
    running_drawdown: float = 0.0
    vrp_regime: int = 0                        # S_VRP al momento de la entrada
    z_vrp: float = 0.0
    tsr: float = 0.0
    macro_event: Optional[str] = None


@dataclass
class VRPPropAccount:
    """
    Gestión cuantitativa de saldo y límites de Prop Trading (Topstep, Apex, Lucid, MFF).
    Controla el Daily Loss Limit (DLL), Maximum Trailing Drawdown (MTDD)
    y la partición fraccionaria estricta entre sesiones.
    """
    initial_capital: float = 50000.0
    dll: float = 1000.0                        # Límite de pérdida diaria
    mtdd: float = 2000.0                       # Máximo trailing drawdown
    trailing_mode: str = "eod"                 # "eod" o "intraday"
    profit_target: float = 3000.0              # Objetivo de aprobación
    lambda_on: float = 0.25                    # Partición del 25% de DLL para sesión nocturna
    gamma_t_on: float = 0.05                   # Máximo 5% del colchón MTDD por trade nocturno
    phi_macro_dll: float = 0.15                # 15% del DLL RTH para trade pre-macro
    gamma_macro_dd: float = 0.05               # 5% del colchón MTDD para trade pre-macro
    n_max_std: int = 2                         # Máximo de contratos estándar (2 ES/NQ)
    n_max_micro: int = 20                      # Máximo de microcontratos (20 MES/MNQ)

    equity: Optional[float] = None
    peak_equity: Optional[float] = None        # High Water Mark
    current_drawdown: float = 0.0
    daily_realized_loss: float = 0.0           # Pérdidas acumuladas en la sesión (magnitud positiva)
    daily_net_pnl: float = 0.0                 # PnL neto del día
    is_breached: bool = False
    breach_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.equity is None:
            self.equity = self.initial_capital
        if self.peak_equity is None:
            self.peak_equity = self.equity

    def start_session(self) -> None:
        """Reinicia el balance diario al comenzar una nueva sesión de negociación."""
        self.daily_realized_loss = 0.0
        self.daily_net_pnl = 0.0

    @property
    def cushion_mtdd(self) -> float:
        """Colchón restante de Trailing Drawdown: D_t^cushion = max(0, MTDD - DD_t)."""
        return max(0.0, self.mtdd - self.current_drawdown)

    @property
    def delta_dll(self) -> float:
        """Margen restante del límite diario: Delta DLL_t = max(0, DLL - DailyLoss_t)."""
        return max(0.0, self.dll - self.daily_realized_loss)

    @property
    def delta_dll_overnight(self) -> float:
        """Margen nocturno segregado: Delta DLL_t^nocturno = lambda_ON * Delta DLL_t."""
        return self.lambda_on * self.delta_dll

    @property
    def delta_dll_rth(self) -> float:
        """Margen RTH segregado: Delta DLL_t^RTH = (1 - lambda_ON) * Delta DLL_t."""
        return (1.0 - self.lambda_on) * self.delta_dll

    def update_balance(self, pnl: float, bar_ts: Optional[datetime] = None) -> None:
        """Actualiza el balance, el High Water Mark y evalúa posibles violaciones de reglas."""
        self.equity += pnl
        self.daily_net_pnl += pnl
        if pnl < 0.0:
            self.daily_realized_loss += abs(pnl)

        # Actualización de pico según modo trailing
        if self.trailing_mode == "intraday":
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
            self.current_drawdown = self.peak_equity - self.equity
        else:
            # En modo EOD, el pico se actualiza al final del día
            self.current_drawdown = max(0.0, self.peak_equity - self.equity)

        # Comprobación de límites de Prop Trading
        if self.current_drawdown >= self.mtdd:
            self.is_breached = True
            self.breach_reason = f"Violación de Max Trailing Drawdown ({self.current_drawdown:.2f} >= {self.mtdd:.2f})"

        if self.daily_realized_loss >= self.dll:
            self.is_breached = True
            self.breach_reason = f"Violación de Daily Loss Limit ({self.daily_realized_loss:.2f} >= {self.dll:.2f})"

    def close_eod_session(self) -> None:
        """Actualiza el trailing floor al cierre de sesión (modo EOD)."""
        if self.trailing_mode == "eod":
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
            self.current_drawdown = max(0.0, self.peak_equity - self.equity)


# ==============================================================================
# 2. FUNCIONES MATEMÁTICAS FORMALES Y MODELOS DE REGIMEN (SECCIONES 3 Y 5)
# ==============================================================================

def calculate_implied_variance(vix: float) -> float:
    """
    Varianza implícita anualizada:
    IV_t = (VIX_t / 100)^2
    """
    if vix < 0:
        raise ValueError(f"El VIX no puede ser negativo: {vix}")
    return float((vix / 100.0) ** 2)


def calculate_daily_realized_variance(returns_5m: Sequence[float]) -> float:
    """
    Varianza realizada diaria a partir de retornos logarítmicos de 5 minutos:
    RV_t^{(d)} = sum_{tau=1}^M r_{5m, tau}^2
    """
    if len(returns_5m) == 0:
        return 0.0
    arr = np.asarray(returns_5m, dtype=np.float64)
    return float(np.sum(arr ** 2))


def calculate_annualized_realized_variance(daily_rvs: Sequence[float], k: int = 22) -> float:
    """
    Varianza realizada acumulada proyectada a horizonte anual (ventana K = 22 días):
    RV_t^ann = (252 / K) * sum_{k=0}^{K-1} RV_{t-k}^{(d)}
    """
    if len(daily_rvs) == 0:
        return 0.0
    window = daily_rvs[-k:]
    k_eff = len(window)
    if k_eff == 0:
        return 0.0
    mean_daily_rv = float(np.mean(window))
    return float(252.0 * mean_daily_rv)


def calculate_vrp(iv: float, rv_ann: float) -> float:
    """
    Variance Risk Premium (Bollerslev et al., 2009):
    VRP_t = IV_t - RV_t^ann
    """
    return float(iv - rv_ann)


def calculate_z_score_vrp(
    current_vrp: float,
    history_vrp: Sequence[float],
    lookback: int = 60,
    eps: float = 1e-6
) -> float:
    """
    Estandarización rodante del VRP sobre L = 60 sesiones:
    Z_VRP(t) = (VRP_t - mu_VRP(L)) / sigma_VRP(L)
    """
    if len(history_vrp) == 0:
        return 0.0
    window = list(history_vrp[-(lookback - 1):]) if lookback > 1 else []
    window.append(current_vrp)
    arr = np.asarray(window, dtype=np.float64)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else float(np.std(arr))
    if sigma < eps:
        sigma = eps
    return float((current_vrp - mu) / sigma)


def calculate_term_structure_ratio(vix: float, vix3m: float) -> float:
    """
    Ratio de estructura temporal de volatilidad (Simon & Campasano, 2014):
    TSR_t = VIX_t / VIX3M_t
    """
    if vix3m <= 0:
        raise ValueError(f"VIX3M debe ser estrictamente positivo: {vix3m}")
    return float(vix / vix3m)


def evaluate_vrp_regime_gate(z_vrp: float, tsr: float) -> int:
    """
    Compuerta estructural de régimen S_VRP(t) in {-1, 0, +1} (Sección 3.1):
    +1: si Z_VRP >= +1.50 and TSR <= 0.95 (Complacencia / Sesgo Comprador Favorable)
    -1: si Z_VRP <= -1.20 or  TSR >= 1.00 (Backwardation / Bloqueo Total de Compras)
     0: en cualquier otro caso             (Régimen Neutral Ordinario)
    """
    if z_vrp >= 1.50 and tsr <= 0.95:
        return 1
    if z_vrp <= -1.20 or tsr >= 1.00:
        return -1
    return 0


def calculate_intraday_return(p_930: float, p_1600: float) -> float:
    """
    Retorno regular de la sesión previa (RTH):
    R_{intraday, t-1} = ln( P_{t-1}(16:00:00) / P_{t-1}(09:30:00) )
    """
    if p_930 <= 0 or p_1600 <= 0:
        raise ValueError(f"Precios deben ser estrictamente positivos: {p_930}, {p_1600}")
    return float(np.log(p_1600 / p_930))


def evaluate_overnight_signal(
    r_intraday_prev: float,
    s_vrp: int,
    theta_drop: float = 0.0075
) -> int:
    """
    Señal de la deriva nocturna condicional (Boyarchenko et al., 2023, Sección 3.2):
    S_overnight(t) = I( R_{intraday, t-1} <= -theta_drop ) * I( S_VRP(t) != -1 )
    con theta_drop = 0.0075 (0.75%).
    Retorna +1 (Comprar) o 0 (Neutral).
    """
    if r_intraday_prev <= -theta_drop and s_vrp != -1:
        return 1
    return 0


def calculate_window_vwap(prices: Sequence[float], volumes: Sequence[float]) -> float:
    """
    Calcula el Volume-Weighted Average Price (VWAP) en una ventana de observación:
    VWAP = sum( P_i * V_i ) / sum( V_i )
    """
    if len(prices) == 0 or len(volumes) == 0:
        return 0.0
    p = np.asarray(prices, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)
    sum_v = float(np.sum(v))
    if sum_v <= 0.0:
        return float(np.mean(p))
    return float(np.sum(p * v) / sum_v)


def evaluate_pre_macro_signal(
    z_ofi_pre: float,
    current_price: float,
    vwap_prior: float,
    s_vrp: int
) -> int:
    """
    Operador analítico de deriva pre-macro informada (Kurov et al., 2019, Sección 3.3):
    +1: Z_OFI_pre >= +2.0 and P_t > VWAP and S_VRP != -1
    -1: Z_OFI_pre <= -2.0 and P_t < VWAP and S_VRP != +1
     0: en cualquier otro caso
    """
    if z_ofi_pre >= 2.0 and current_price > vwap_prior and s_vrp != -1:
        return 1
    if z_ofi_pre <= -2.0 and current_price < vwap_prior and s_vrp != 1:
        return -1
    return 0


def calculate_overnight_sizing(
    budget_daily_on: float,
    budget_cushion_on: float,
    p_entry: float,
    multiplier_std: float,
    multiplier_micro: float,
    delta_sl_on: float = 0.0050,
    n_max_std: int = 2,
    n_max_micro: int = 20,
    friction_std: float = 0.0,
    friction_micro: float = 0.0
) -> Tuple[int, str]:
    """
    Ecuación cerrada de contratos para la Deriva Nocturna con conmutación Estándar a Micro (Sección 5.2 & 5.4):
    N_ON^{(I)} = min( floor( min( Delta DLL_t^nocturno, gamma_T^ON * D_t^cushion ) / (delta_SL^ON * P_entry * M^{(I)}) ), N_max^ON )
    """
    budget = min(budget_daily_on, budget_cushion_on)
    if budget <= 0.0 or p_entry <= 0.0:
        return 0, "NONE"

    risk_unit_std = (delta_sl_on * p_entry * multiplier_std) + friction_std
    risk_unit_micro = (delta_sl_on * p_entry * multiplier_micro) + friction_micro

    if risk_unit_std <= 0 or risk_unit_micro <= 0:
        return 0, "NONE"

    n_std = int(budget // risk_unit_std)
    n_std = min(n_std, n_max_std)

    if n_std >= 1:
        return n_std, "STD"

    # Conmutación a microcontratos
    if budget >= risk_unit_micro:
        n_micro = int(budget // risk_unit_micro)
        n_micro = min(n_micro, n_max_micro)
        if n_micro >= 1:
            return n_micro, "MICRO"

    return 0, "NONE"


def calculate_pre_macro_sizing(
    budget_daily_macro: float,
    budget_cushion_macro: float,
    delta_p_stop: float,
    multiplier_std: float,
    multiplier_micro: float,
    n_max_std: int = 2,
    n_max_micro: int = 20,
    friction_std: float = 0.0,
    friction_micro: float = 0.0
) -> Tuple[int, str]:
    """
    Ecuación cerrada de contratos para la Deriva Pre-Macro con conmutación Estándar a Micro (Sección 5.3 & 5.4):
    N_macro^{(I)} = min( floor( min( 0.15 * Delta DLL_t^RTH, 0.05 * D_t^cushion ) / (Delta P_stop^macro * M^{(I)}) ), N_max^macro )
    """
    budget = min(budget_daily_macro, budget_cushion_macro)
    if budget <= 0.0 or delta_p_stop <= 0.0:
        return 0, "NONE"

    risk_unit_std = (delta_p_stop * multiplier_std) + friction_std
    risk_unit_micro = (delta_p_stop * multiplier_micro) + friction_micro

    if risk_unit_std <= 0 or risk_unit_micro <= 0:
        return 0, "NONE"

    n_std = int(budget // risk_unit_std)
    n_std = min(n_std, n_max_std)

    if n_std >= 1:
        return n_std, "STD"

    # Conmutación a microcontratos
    if budget >= risk_unit_micro:
        n_micro = int(budget // risk_unit_micro)
        n_micro = min(n_micro, n_max_micro)
        if n_micro >= 1:
            return n_micro, "MICRO"

    return 0, "NONE"


# ==============================================================================
# 3. IMPLEMENTACIÓN DE ESTRATEGIA: ANOMALÍAS TEMPORALES Y VRP (F9)
# ==============================================================================

class TemporalAnomaliesVRPStrategy(BaseStrategy):
    """
    Estrategia F9: Anomalías Temporales Multi-Sesión y Variance Risk Premium (VRP).
    Modela rigurosamente:
    - Deriva nocturna condicional (Boyarchenko et al., 2023): entrada 01:30 EST, salida 08:30 EST, SL 0.50%.
    - Deriva pre-macro informada (Kurov et al., 2019): entrada T-15m, salida T-3m, SL 10.0 pts.
    - Compuerta de régimen estructural VRP / Term Structure (Simon & Campasano, 2014; Bollerslev et al., 2009).
    - Embargo estricto [T-2m, T+2m] para Prop Trading.
    - Cierre universal previo al mantenimiento de CME a las 16:45 EST.
    """

    def __init__(
        self,
        name: str = "Anomalias_Temporales_VRP",
        params: Optional[Dict[str, Any]] = None,
    ):
        default_params: Dict[str, Any] = {
            "symbol": "ES",
            "std_symbol": "ES",
            "micro_symbol": "MES",
            "theta_drop": 0.0075,              # 0.75% caída intradía mínima para activar overnight
            "delta_sl_on": 0.0050,             # 0.50% Stop Loss nocturno
            "delta_p_stop_macro": 10.0,        # 10.0 puntos stop loss pre-macro (ES)
            "k_rv_lookback": 22,               # Días de proyección de varianza realizada anualizada
            "l_vrp_lookback": 60,              # Ventana rodante de estandarización Z_VRP
            "default_vix": 16.0,               # VIX por defecto si no se incluye columna externa
            "default_tsr": 0.90,               # TSR (VIX/VIX3M) por defecto (Contango saludable = 0.90)
            "slippage_ticks": 1.0,             # 1 tick entrada + 1 tick salida = 2 ticks slippage total
            "target_tz": "America/New_York",
            "source_tz": None,
            "macro_schedule": None,            # Lista o dict opcional de timestamps de anuncios macro Tier-1
        }
        if params:
            default_params.update(params)
        super().__init__(name=name, params=default_params)

        # Cargar especificaciones de futuros
        self.spec_std = get_spec(self.params["std_symbol"])
        self.spec_micro = get_spec(self.params["micro_symbol"])

    def ensure_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Localiza y convierte el índice temporal a America/New_York (EST/EDT)."""
        target_tz = self.params["target_tz"]
        source_tz = self.params.get("source_tz")

        if df.index.tz is None:
            if source_tz:
                return df.tz_localize(source_tz).tz_convert(target_tz)
            else:
                return df.tz_localize(target_tz)
        else:
            return df.tz_convert(target_tz)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera la serie temporal de posición teórica unificada S_unified^*(t) in {-1, 0, +1}.
        Cumple la matriz de composición ortogonal (Sección 3.4):
        - Overnight en [01:30:00, 08:30:00) EST.
        - Pre-Macro en [T-15m, T-3m] EST.
        - 0 en ventana de embargo [T-2m, T+2m].
        - 0 a partir de 15:55:00 EST.
        """
        df = self.ensure_timezone(data.copy())
        self.validate_data(df)

        signals = pd.Series(0.0, index=df.index, name="Signal")

        # 1. Calcular retornos de 5m y varianza realizada
        df["ret_log"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)

        # VIX y VIX3M (si no están en el DataFrame, usar valores por defecto)
        vix_series = df["VIX"] if "VIX" in df.columns else pd.Series(self.params["default_vix"], index=df.index)
        tsr_series = df["TSR"] if "TSR" in df.columns else pd.Series(self.params["default_tsr"], index=df.index)

        # 2. Agrupación por días para retornos de sesión regular (09:30 a 16:00)
        daily_rth_returns: Dict[Any, float] = {}
        daily_realized_variances: Dict[Any, float] = {}

        for day_date, day_group in df.groupby(df.index.date):
            # Retornos RTH: P(16:00) / P(09:30)
            rth_bars = day_group.between_time("09:30", "16:00")
            if len(rth_bars) >= 2:
                p_open = float(rth_bars.iloc[0]["Open"])
                p_close = float(rth_bars.iloc[-1]["Close"])
                daily_rth_returns[day_date] = calculate_intraday_return(p_open, p_close)
            elif len(rth_bars) == 1:
                p_open = float(rth_bars.iloc[0]["Open"])
                p_close = float(rth_bars.iloc[0]["Close"])
                daily_rth_returns[day_date] = calculate_intraday_return(p_open, p_close) if p_open > 0 and p_close > 0 else 0.0
            else:
                daily_rth_returns[day_date] = 0.0

            # Varianza realizada de 5 minutos
            daily_realized_variances[day_date] = calculate_daily_realized_variance(day_group["ret_log"])

        # 3. Calcular serie diaria de VRP y Z_VRP
        sorted_dates = sorted(daily_realized_variances.keys())
        daily_vrp_map: Dict[Any, float] = {}
        daily_z_vrp_map: Dict[Any, float] = {}

        vrp_history: List[float] = []
        for i, dt in enumerate(sorted_dates):
            past_rvs = [daily_realized_variances[d] for d in sorted_dates[max(0, i - self.params["k_rv_lookback"] + 1): i + 1]]
            rv_ann = calculate_annualized_realized_variance(past_rvs, k=self.params["k_rv_lookback"])
            
            # VIX promedio del día
            day_vix = float(vix_series[df.index.date == dt].mean()) if len(vix_series[df.index.date == dt]) > 0 else self.params["default_vix"]
            iv = calculate_implied_variance(day_vix)
            vrp_val = calculate_vrp(iv, rv_ann)
            daily_vrp_map[dt] = vrp_val

            z_vrp_val = calculate_z_score_vrp(vrp_val, vrp_history, lookback=self.params["l_vrp_lookback"])
            daily_z_vrp_map[dt] = z_vrp_val
            vrp_history.append(vrp_val)

        # 4. Generación barra por barra de señales unificadas
        macro_events = self.params.get("macro_schedule") or []
        macro_dt_set: Set[pd.Timestamp] = set()
        if isinstance(macro_events, (list, set)):
            for me in macro_events:
                val = me[0] if isinstance(me, tuple) else me
                ts_obj = pd.Timestamp(val)
                norm_ts = ts_obj.tz_convert(self.params["target_tz"]) if ts_obj.tzinfo else ts_obj.tz_localize(self.params["target_tz"])
                macro_dt_set.add(norm_ts)

        for i in range(len(df)):
            ts = df.index[i]
            cur_time = ts.time()
            cur_date = ts.date()

            # Encontrar fecha de la sesión anterior
            prev_dates = [d for d in sorted_dates if d < cur_date]
            prev_date = prev_dates[-1] if prev_dates else None
            r_intraday_prev = daily_rth_returns.get(prev_date, 0.0) if prev_date else 0.0

            z_vrp = daily_z_vrp_map.get(prev_date, 0.0) if prev_date is not None else 0.0
            tsr = float(tsr_series.iloc[i])
            s_vrp = evaluate_vrp_regime_gate(z_vrp, tsr)

            # A) Evaluación de Ventana de Embargo Prohibido [T-2m, T+2m]
            in_embargo = False
            for m_dt in macro_dt_set:
                if (m_dt - timedelta(minutes=2)) <= ts <= (m_dt + timedelta(minutes=2)):
                    in_embargo = True
                    break

            if in_embargo:
                signals.iloc[i] = 0.0
                continue

            # B) Cierre técnico de fin de día (>= 15:55:00 EST)
            if cur_time >= time(15, 55):
                signals.iloc[i] = 0.0
                continue

            # C) Sub-modelo 2: Deriva Pre-Macro (Kurov et al., 2019)
            # Para cada evento macro, ventana [T-15m, T-3m]
            # Tiene prioridad sobre Overnight para evitar que anuncios de las 08:30 queden eclipsados
            in_macro_window = False
            for m_dt in macro_dt_set:
                t_minus_15 = m_dt - timedelta(minutes=15)
                t_minus_3 = m_dt - timedelta(minutes=3)
                if t_minus_15 <= ts <= t_minus_3:
                    in_macro_window = True
                    # Calcular Z_OFI y VWAP previo
                    # VWAP ventana previa [T-30m, T-15m]
                    sub_prev = df.loc[(df.index >= m_dt - timedelta(minutes=30)) & (df.index < t_minus_15)]
                    vwap_prior = calculate_window_vwap(sub_prev["Close"], sub_prev["Volume"]) if len(sub_prev) > 0 else float(df["Close"].iloc[i])
                    
                    z_ofi = float(df["Z_OFI"].iloc[i]) if "Z_OFI" in df.columns else 2.5  # Si no hay OFI explícito en datos sintéticos
                    p_cur = float(df["Close"].iloc[i])
                    s_macro = evaluate_pre_macro_signal(z_ofi, p_cur, vwap_prior, s_vrp)
                    signals.iloc[i] = float(s_macro)
                    break

            if in_macro_window:
                continue

            # D) Sub-modelo 1: Deriva Nocturna (Boyarchenko et al., 2023)
            # Ventana operativa: [01:30:00, 08:30:00) EST
            if time(1, 30) <= cur_time < time(8, 30):
                s_on = evaluate_overnight_signal(r_intraday_prev, s_vrp, theta_drop=self.params["theta_drop"])
                if s_on == 1:
                    signals.iloc[i] = 1.0
                continue

            # Fuera de las ventanas específicas: 0 (Neutral)
            signals.iloc[i] = 0.0

        return signals


# ==============================================================================
# 4. MOTOR DE BACKTEST Y AUDITORÍA DE PROP TRADING PARA F9
# ==============================================================================

class TemporalAnomaliesVRPBacktester:
    """
    Simulador determinista de ejecución y evaluación de Prop Trading para la Estrategia F9.
    Monitorea barra a barra:
    1. Entrada en 01:30:00 EST sujeta a caída previa >= 0.75% y filtro S_VRP != -1.
    2. Salida en 08:30:00 EST incondicional o por Stop Loss de 0.50%.
    3. Entrada Pre-Macro en T-15m y salida forzosa en T-3m (o Stop Loss 10.0 pts en ES).
    4. Cero contratos en ventana de embargo [T-2m, T+2m].
    5. Liquidación forzosa CME a las 16:45:00 EST.
    6. Auditoría de Prop Trading: Daily Loss Limit (DLL), Trailing Max Drawdown (MTDD),
       comisiones CME y slippage institucional de 2 ticks round-turn.
    """

    def __init__(
        self,
        strategy: TemporalAnomaliesVRPStrategy,
        account_size: float = 50000.0,
        daily_loss_limit: float = 1000.0,
        max_trailing_loss: float = 2000.0,
        trailing_mode: str = "eod",
        profit_target: float = 3000.0,
    ):
        self.strategy = strategy
        self.account = VRPPropAccount(
            initial_capital=account_size,
            dll=daily_loss_limit,
            mtdd=max_trailing_loss,
            trailing_mode=trailing_mode,
            profit_target=profit_target,
        )
        self.trades: List[VRPTradeRecord] = []

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Ejecuta la simulación completa sobre el conjunto de datos históricos.
        """
        df = self.strategy.ensure_timezone(data.copy())
        self.strategy.validate_data(df)

        spec_std = self.strategy.spec_std
        spec_micro = self.strategy.spec_micro
        slippage_ticks = self.strategy.params["slippage_ticks"]

        tick_std = spec_std["tick"]
        m_std = spec_std["multiplier"]
        comm_std = spec_std["commission_rt"]

        tick_micro = spec_micro["tick"]
        m_micro = spec_micro["multiplier"]
        comm_micro = spec_micro["commission_rt"]

        # Fricción combinada (2 ticks slippage + comisión RT)
        friction_unit_std = (2.0 * slippage_ticks * tick_std * m_std) + comm_std
        friction_unit_micro = (2.0 * slippage_ticks * tick_micro * m_micro) + comm_micro

        # Variables de VIX y TSR
        vix_series = df["VIX"] if "VIX" in df.columns else pd.Series(self.strategy.params["default_vix"], index=df.index)
        tsr_series = df["TSR"] if "TSR" in df.columns else pd.Series(self.strategy.params["default_tsr"], index=df.index)

        # Retornos logarítmicos
        df["ret_log"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)

        # Calcular retornos previos de sesión RTH (09:30 a 16:00) y varianzas
        daily_rth_returns: Dict[Any, float] = {}
        daily_realized_variances: Dict[Any, float] = {}

        for day_date, day_group in df.groupby(df.index.date):
            rth_bars = day_group.between_time("09:30", "16:00")
            if len(rth_bars) >= 2:
                p_open = float(rth_bars.iloc[0]["Open"])
                p_close = float(rth_bars.iloc[-1]["Close"])
                daily_rth_returns[day_date] = calculate_intraday_return(p_open, p_close)
            elif len(rth_bars) == 1:
                p_open = float(rth_bars.iloc[0]["Open"])
                p_close = float(rth_bars.iloc[0]["Close"])
                daily_rth_returns[day_date] = calculate_intraday_return(p_open, p_close) if p_open > 0 and p_close > 0 else 0.0
            else:
                daily_rth_returns[day_date] = 0.0

            daily_realized_variances[day_date] = calculate_daily_realized_variance(day_group["ret_log"])

        sorted_dates = sorted(daily_realized_variances.keys())
        daily_z_vrp_map: Dict[Any, float] = {}
        vrp_history: List[float] = []

        for i, dt in enumerate(sorted_dates):
            past_rvs = [daily_realized_variances[d] for d in sorted_dates[max(0, i - self.strategy.params["k_rv_lookback"] + 1): i + 1]]
            rv_ann = calculate_annualized_realized_variance(past_rvs, k=self.strategy.params["k_rv_lookback"])
            day_vix = float(vix_series[df.index.date == dt].mean()) if len(vix_series[df.index.date == dt]) > 0 else self.strategy.params["default_vix"]
            iv = calculate_implied_variance(day_vix)
            vrp_val = calculate_vrp(iv, rv_ann)
            z_vrp_val = calculate_z_score_vrp(vrp_val, vrp_history, lookback=self.strategy.params["l_vrp_lookback"])
            daily_z_vrp_map[dt] = z_vrp_val
            vrp_history.append(vrp_val)

        # Macro schedule
        macro_events = self.strategy.params.get("macro_schedule") or []
        macro_dt_list: List[Tuple[pd.Timestamp, str]] = []
        if isinstance(macro_events, list):
            for me in macro_events:
                if isinstance(me, tuple):
                    ts_m, name_m = me
                else:
                    ts_m, name_m = me, "Tier-1 News"
                ts_obj = pd.Timestamp(ts_m)
                norm_ts = ts_obj.tz_convert(self.strategy.params["target_tz"]) if ts_obj.tzinfo else ts_obj.tz_localize(self.strategy.params["target_tz"])
                macro_dt_list.append((norm_ts, name_m))

        # Estado del backtest
        current_trade: Optional[VRPTradeRecord] = None
        trade_id_counter = 1
        current_day: Optional[Any] = None

        bar_delta = (df.index[1] - df.index[0]) if len(df) >= 2 else timedelta(minutes=5)

        for i in range(len(df)):
            ts = df.index[i]
            cur_time = ts.time()
            cur_date = ts.date()
            bar_open = float(df["Open"].iloc[i])
            bar_high = float(df["High"].iloc[i])
            bar_low = float(df["Low"].iloc[i])
            bar_close = float(df["Close"].iloc[i])

            # Nueva sesión diaria
            if cur_date != current_day:
                if current_day is not None:
                    self.account.close_eod_session()
                current_day = cur_date
                self.account.start_session()

            # Si la cuenta fue quebrada, detener operaciones
            if self.account.is_breached:
                if current_trade is not None:
                    # Cerrar inmediatamente a precio de apertura
                    self._close_trade(
                        trade=current_trade,
                        exit_price=bar_open,
                        exit_time=ts,
                        exit_reason=VRPExitReason.RISK_BREACH,
                        slippage_ticks=slippage_ticks,
                    )
                    current_trade = None
                break

            # ------------------------------------------------------------------
            # GESTIÓN DE POSICIÓN ABIERTA (SALIDAS Y STOPS)
            # ------------------------------------------------------------------
            if current_trade is not None:
                current_trade.bars_held += 1

                # 1. Cierre técnico forzoso previo a CME (16:45:00 EST)
                if cur_time >= time(16, 45):
                    self._close_trade(
                        trade=current_trade,
                        exit_price=bar_close,
                        exit_time=ts,
                        exit_reason=VRPExitReason.TERMINAL_EOD,
                        slippage_ticks=slippage_ticks,
                    )
                    current_trade = None
                    continue

                # 2. Gestión de Trade Overnight (Boyarchenko)
                if current_trade.submodel == SubModelType.OVERNIGHT.value:
                    # A) Comprobar Stop Loss (0.50% de caída)
                    # En overnight el trade siempre es Long
                    if bar_low <= current_trade.stop_loss_price:
                        # Ejecutado en Stop Loss con deslizamiento
                        self._close_trade(
                            trade=current_trade,
                            exit_price=current_trade.stop_loss_price,
                            exit_time=ts,
                            exit_reason=VRPExitReason.STOP_LOSS,
                            slippage_ticks=slippage_ticks,
                        )
                        current_trade = None
                        continue

                    # B) Cierre de seguridad si hay noticia macro a las 08:30 EST (antes del embargo [08:28, 08:32])
                    macro_dt_830 = None
                    for m_dt, _ in macro_dt_list:
                        if m_dt.date() == cur_date and m_dt.time() == time(8, 30):
                            macro_dt_830 = m_dt
                            break

                    if macro_dt_830 is not None:
                        t_exit_macro = macro_dt_830 - timedelta(minutes=3)
                        t_embargo_start = macro_dt_830 - timedelta(minutes=2)
                        if ts >= t_exit_macro:
                            self._close_trade(
                                trade=current_trade,
                                exit_price=bar_open,
                                exit_time=t_exit_macro,
                                exit_reason=VRPExitReason.EMBARGO_FLAT,
                                slippage_ticks=slippage_ticks,
                            )
                            current_trade = None
                            continue
                        elif (ts + bar_delta) > t_embargo_start:
                            exit_p = bar_close if ts < t_exit_macro else bar_open
                            self._close_trade(
                                trade=current_trade,
                                exit_price=exit_p,
                                exit_time=t_exit_macro,
                                exit_reason=VRPExitReason.EMBARGO_FLAT,
                                slippage_ticks=slippage_ticks,
                            )
                            current_trade = None
                            continue

                    # C) Salida determinista incondicional a las 08:30:00 EST
                    if cur_time >= time(8, 30):
                        self._close_trade(
                            trade=current_trade,
                            exit_price=bar_open if cur_time == time(8, 30) else bar_close,
                            exit_time=ts,
                            exit_reason=VRPExitReason.DETERMINISTIC_TIME_EXIT,
                            slippage_ticks=slippage_ticks,
                        )
                        current_trade = None
                        continue

                # 3. Gestión de Trade Pre-Macro (Kurov)
                elif current_trade.submodel == SubModelType.PRE_MACRO.value:
                    # Encontrar evento macro correspondiente
                    macro_dt = None
                    for m_dt, _ in macro_dt_list:
                        if m_dt.date() == cur_date and (m_dt - timedelta(minutes=20)) <= ts <= (m_dt + timedelta(minutes=5)):
                            macro_dt = m_dt
                            break

                    # A) Comprobar Stop Loss en puntos (10.0 pts en ES)
                    if current_trade.direction == 1 and bar_low <= current_trade.stop_loss_price:
                        self._close_trade(
                            trade=current_trade,
                            exit_price=current_trade.stop_loss_price,
                            exit_time=ts,
                            exit_reason=VRPExitReason.STOP_LOSS,
                            slippage_ticks=slippage_ticks,
                        )
                        current_trade = None
                        continue
                    elif current_trade.direction == -1 and bar_high >= current_trade.stop_loss_price:
                        self._close_trade(
                            trade=current_trade,
                            exit_price=current_trade.stop_loss_price,
                            exit_time=ts,
                            exit_reason=VRPExitReason.STOP_LOSS,
                            slippage_ticks=slippage_ticks,
                        )
                        current_trade = None
                        continue

                    # B) Salida determinista forzosa en T-3m (antes del embargo T-2m)
                    if macro_dt is not None:
                        t_exit_macro = macro_dt - timedelta(minutes=3)
                        t_embargo_start = macro_dt - timedelta(minutes=2)

                        if ts >= t_exit_macro:
                            self._close_trade(
                                trade=current_trade,
                                exit_price=bar_open,
                                exit_time=t_exit_macro,
                                exit_reason=VRPExitReason.DETERMINISTIC_TIME_EXIT,
                                slippage_ticks=slippage_ticks,
                            )
                            current_trade = None
                            continue
                        elif (ts + bar_delta) > t_embargo_start:
                            # Esta es la última barra antes de que inicie la ventana de embargo [T-2m, T+2m].
                            # Liquidación forzosa en T-3m para garantizar cuenta plana 60s antes de embargo.
                            exit_p = bar_close if ts < t_exit_macro else bar_open
                            self._close_trade(
                                trade=current_trade,
                                exit_price=exit_p,
                                exit_time=t_exit_macro,
                                exit_reason=VRPExitReason.DETERMINISTIC_TIME_EXIT,
                                slippage_ticks=slippage_ticks,
                            )
                            current_trade = None
                            continue

                    # C) Cierre de seguridad ante aproximación o entrada a embargo
                    if macro_dt is not None and ts >= (macro_dt - timedelta(minutes=2)):
                        self._close_trade(
                            trade=current_trade,
                            exit_price=bar_open,
                            exit_time=ts,
                            exit_reason=VRPExitReason.EMBARGO_FLAT,
                            slippage_ticks=slippage_ticks,
                        )
                        current_trade = None
                        continue

            # ------------------------------------------------------------------
            # EVALUACIÓN DE NUEVAS ENTRADAS (SI NO ESTAMOS EN POSICIÓN)
            # ------------------------------------------------------------------
            if current_trade is None and not self.account.is_breached:
                # Comprobar embargo estricto [T-2m, T+2m]
                is_embargo = False
                for m_dt, _ in macro_dt_list:
                    if (m_dt - timedelta(minutes=2)) <= ts <= (m_dt + timedelta(minutes=2)):
                        is_embargo = True
                        break
                if is_embargo:
                    continue

                # Cierre intradía: no abrir después de 15:55 EST
                if cur_time >= time(15, 55):
                    continue

                # Obtener régimen VRP previo (sin lookahead en sesión 0)
                prev_dates = [d for d in sorted_dates if d < cur_date]
                prev_date = prev_dates[-1] if prev_dates else None
                r_intraday_prev = daily_rth_returns.get(prev_date, 0.0) if prev_date else 0.0

                z_vrp = daily_z_vrp_map.get(prev_date, 0.0) if prev_date is not None else 0.0
                tsr = float(tsr_series.iloc[i])
                s_vrp = evaluate_vrp_regime_gate(z_vrp, tsr)

                # A) Evaluar Sub-modelo Overnight (Boyarchenko) a las 01:30 EST
                # Nota: en barras de 5m, evaluamos en la barra de 01:30:00 (o entre 01:30 y 01:35)
                if time(1, 30) <= cur_time <= time(1, 35):
                    s_on = evaluate_overnight_signal(r_intraday_prev, s_vrp, theta_drop=self.strategy.params["theta_drop"])
                    if s_on == 1:
                        # Dimensionamiento de posición nocturna
                        p_entry = bar_close
                        budget_daily_on = self.account.delta_dll_overnight
                        budget_cushion_on = self.account.gamma_t_on * self.account.cushion_mtdd
                        inc_fric = self.strategy.params.get("include_friction_in_sizing", False)
                        fric_s = friction_unit_std if inc_fric else 0.0
                        fric_m = friction_unit_micro if inc_fric else 0.0

                        n_contracts, c_type = calculate_overnight_sizing(
                            budget_daily_on=budget_daily_on,
                            budget_cushion_on=budget_cushion_on,
                            p_entry=p_entry,
                            multiplier_std=m_std,
                            multiplier_micro=m_micro,
                            delta_sl_on=self.strategy.params["delta_sl_on"],
                            n_max_std=self.account.n_max_std,
                            n_max_micro=self.account.n_max_micro,
                            friction_std=fric_s,
                            friction_micro=fric_m,
                        )

                        if n_contracts >= 1:
                            sym = self.strategy.params["micro_symbol"] if c_type == "MICRO" else self.strategy.params["std_symbol"]
                            sl_price = p_entry * (1.0 - self.strategy.params["delta_sl_on"])
                            # Aplicar 1 tick slippage en entrada
                            tick_val = tick_micro if c_type == "MICRO" else tick_std
                            actual_entry = p_entry + (slippage_ticks * tick_val)

                            current_trade = VRPTradeRecord(
                                trade_id=trade_id_counter,
                                date=str(cur_date),
                                submodel=SubModelType.OVERNIGHT.value,
                                direction=1,
                                contracts=n_contracts,
                                contract_type=c_type,
                                symbol=sym,
                                entry_time=str(ts),
                                entry_price=actual_entry,
                                raw_entry_price=p_entry,
                                stop_loss_price=sl_price,
                                vrp_regime=s_vrp,
                                z_vrp=z_vrp,
                                tsr=tsr,
                            )
                            trade_id_counter += 1
                            continue

                # B) Evaluar Sub-modelo Pre-Macro (Kurov) a las T-15m
                for m_dt, m_name in macro_dt_list:
                    t_trigger = m_dt - timedelta(minutes=15)
                    # Ventana de disparo: dentro de los 5 minutos a partir de T-15m
                    if t_trigger <= ts <= (t_trigger + timedelta(minutes=5)):
                        # VWAP ventana [T-30m, T-15m]
                        sub_vwap = df.loc[(df.index >= m_dt - timedelta(minutes=30)) & (df.index < t_trigger)]
                        vwap_prior = calculate_window_vwap(sub_vwap["Close"], sub_vwap["Volume"]) if len(sub_vwap) > 0 else bar_close
                        
                        z_ofi = float(df["Z_OFI"].iloc[i]) if "Z_OFI" in df.columns else (2.5 if bar_close > vwap_prior else -2.5)
                        s_macro = evaluate_pre_macro_signal(z_ofi, bar_close, vwap_prior, s_vrp)

                        if s_macro != 0:
                            p_entry = bar_close
                            budget_daily_macro = self.account.phi_macro_dll * self.account.delta_dll_rth
                            budget_cushion_macro = self.account.gamma_macro_dd * self.account.cushion_mtdd
                            delta_pts = self.strategy.params["delta_p_stop_macro"]

                            inc_fric = self.strategy.params.get("include_friction_in_sizing", False)
                            fric_s = friction_unit_std if inc_fric else 0.0
                            fric_m = friction_unit_micro if inc_fric else 0.0

                            n_contracts, c_type = calculate_pre_macro_sizing(
                                budget_daily_macro=budget_daily_macro,
                                budget_cushion_macro=budget_cushion_macro,
                                delta_p_stop=delta_pts,
                                multiplier_std=m_std,
                                multiplier_micro=m_micro,
                                n_max_std=self.account.n_max_std,
                                n_max_micro=self.account.n_max_micro,
                                friction_std=fric_s,
                                friction_micro=fric_m,
                            )

                            if n_contracts >= 1:
                                sym = self.strategy.params["micro_symbol"] if c_type == "MICRO" else self.strategy.params["std_symbol"]
                                tick_val = tick_micro if c_type == "MICRO" else tick_std
                                actual_entry = p_entry + (s_macro * slippage_ticks * tick_val)
                                sl_price = actual_entry - (s_macro * delta_pts)

                                current_trade = VRPTradeRecord(
                                    trade_id=trade_id_counter,
                                    date=str(cur_date),
                                    submodel=SubModelType.PRE_MACRO.value,
                                    direction=s_macro,
                                    contracts=n_contracts,
                                    contract_type=c_type,
                                    symbol=sym,
                                    entry_time=str(ts),
                                    entry_price=actual_entry,
                                    raw_entry_price=p_entry,
                                    stop_loss_price=sl_price,
                                    vrp_regime=s_vrp,
                                    z_vrp=z_vrp,
                                    tsr=tsr,
                                    macro_event=m_name,
                                )
                                trade_id_counter += 1
                                break

        # Si queda trade abierto al final de los datos, cerrarlo
        if current_trade is not None:
            last_ts = df.index[-1]
            last_close = float(df["Close"].iloc[-1])
            self._close_trade(
                trade=current_trade,
                exit_price=last_close,
                exit_time=last_ts,
                exit_reason=VRPExitReason.TERMINAL_EOD,
                slippage_ticks=slippage_ticks,
            )

        # Resumen final de sesión
        self.account.close_eod_session()

        return self._generate_report(df)

    def _close_trade(
        self,
        trade: VRPTradeRecord,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: VRPExitReason,
        slippage_ticks: float,
    ) -> None:
        """Calcula el P&L y actualiza la cuenta tras el cierre de una operación."""
        is_micro = (trade.contract_type == "MICRO")
        spec = self.strategy.spec_micro if is_micro else self.strategy.spec_std
        multiplier = spec["multiplier"]
        comm_rt = spec["commission_rt"]
        tick_size = spec["tick"]

        # Aplicar 1 tick slippage en salida
        raw_exit = exit_price
        actual_exit = raw_exit - (trade.direction * slippage_ticks * tick_size)
        trade.raw_exit_price = round(raw_exit, 2)
        trade.exit_price = round(actual_exit, 2)
        trade.exit_time = str(exit_time)
        trade.exit_reason = exit_reason

        # Cálculos de PnL: Gross PnL sobre precio bruto sin slippage
        raw_entry = trade.raw_entry_price if trade.raw_entry_price > 0.0 else trade.entry_price
        raw_pts = (raw_exit - raw_entry) * trade.direction
        trade.gross_pnl_pts = round(raw_pts, 2)
        trade.gross_pnl_usd = round(raw_pts * multiplier * trade.contracts, 2)

        # Fricción completa: comisiones CME + 2 ticks slippage total (1 tick entrada + 1 tick salida)
        trade.commissions_usd = round(comm_rt * trade.contracts, 2)
        trade.slippage_usd = round(2.0 * slippage_ticks * tick_size * multiplier * trade.contracts, 2)
        trade.friction_usd = round(trade.commissions_usd + trade.slippage_usd, 2)
        # Net PnL = Gross PnL - Friction
        trade.net_pnl_usd = round(trade.gross_pnl_usd - trade.friction_usd, 2)

        # Actualizar cuenta
        self.account.update_balance(trade.net_pnl_usd, exit_time)
        trade.running_balance = round(self.account.equity, 2)
        trade.running_drawdown = round(self.account.current_drawdown, 2)

        self.trades.append(trade)

    def _generate_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compila las métricas estadísticas y la auditoría formal de Prop Trading."""
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
        total_days = len(set(df.index.date))

        if trades_df.empty:
            metrics = {
                "total_trades": 0,
                "win_rate": 0.0,
                "net_profit": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
            }
            prop_eval = {
                "status": "ACTIVE",
                "reason": "Sin operaciones ejecutadas en la ventana analizada",
                "final_balance": self.account.equity,
                "net_profit": 0.0,
                "total_days_evaluated": total_days,
            }
            return {
                "trades": trades_df,
                "metrics": metrics,
                "prop_evaluation": prop_eval,
                "account": self.account,
            }

        net_profits = trades_df["net_pnl_usd"]
        wins = net_profits[net_profits > 0]
        losses = net_profits[net_profits < 0]

        total_trades = len(trades_df)
        win_trades = len(wins)
        loss_trades = len(losses)
        win_rate = (win_trades / total_trades) * 100.0 if total_trades > 0 else 0.0

        gross_profit = float(wins.sum()) if not wins.empty else 0.0
        gross_loss = float(abs(losses.sum())) if not losses.empty else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        net_profit_total = float(net_profits.sum())

        # Sharpe ratio anualizado sobre trades
        if len(net_profits) > 1 and net_profits.std() > 0:
            sharpe = float((net_profits.mean() / net_profits.std()) * np.sqrt(252))
        else:
            sharpe = 0.0

        metrics = {
            "total_trades": total_trades,
            "winning_trades": win_trades,
            "losing_trades": loss_trades,
            "win_rate_pct": round(win_rate, 2),
            "gross_profit_usd": round(gross_profit, 2),
            "gross_loss_usd": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "net_profit_usd": round(net_profit_total, 2),
            "total_commissions_usd": round(float(trades_df["commissions_usd"].sum()), 2),
            "total_slippage_usd": round(float(trades_df["slippage_usd"].sum()), 2),
            "max_drawdown_usd": round(self.account.current_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
        }

        # Evaluación de Prop Trading
        status = "FAIL" if self.account.is_breached else (
            "PASS" if (self.account.equity - self.account.initial_capital) >= self.account.profit_target else "ACTIVE"
        )
        reason = self.account.breach_reason if self.account.is_breached else (
            "Profit Target Alcanzado Exitosamente" if status == "PASS" else "En curso / Sin violaciones de riesgo"
        )

        prop_eval = {
            "status": status,
            "reason": reason,
            "final_balance": round(self.account.equity, 2),
            "net_profit": round(net_profit_total, 2),
            "total_days_evaluated": total_days,
            "is_breached": self.account.is_breached,
        }

        return {
            "trades": trades_df,
            "metrics": metrics,
            "prop_evaluation": prop_eval,
            "account": self.account,
        }
