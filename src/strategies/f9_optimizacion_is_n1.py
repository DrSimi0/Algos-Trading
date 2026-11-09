"""
src/strategies/f9_optimizacion_is_n1.py
Implementación Matemática y Microestructural de la Versión Alterna Optimizada:
Estrategia "F9-Optimizacion -IS- N1": Anomalías Temporales y Variance Risk Premium (VRP).

Especificación y Fundamentos Teóricos:
- Bollerslev, T., Tauchen, G., & Zhou, H. (2009). Expected Stock Returns and Variance Risk Premia. RFS.
- Boyarchenko, N., Larsen, J. C., & Whelan, P. (2023). The Overnight Drift. Federal Reserve Bank of NY.
- Kurov, A., Sancetta, A., Strasser, G., & Wolfe, M. H. (2019). Pre-Announcement Drift. JFQA.
- Simon, D. P., & Campasano, J. (2014). The VIX Futures Basis: Evidence and Trading Strategies. JOD.
- Alizadeh, S., Brandt, M. W., & Diebold, F. X. (2002). Range-Based Estimation of Stochastic Volatility Models.

Mejoras Matemáticas Principales en F9-Optimizacion -IS- N1:
1. Condicionamiento estricto a la pendiente de la estructura de plazos de volatilidad (TSR / Contango vs. Backwardation):
   - Bloqueo total de entradas largas cuando la curva se invierte (TSR >= 1.00 o TSR_synth >= 1.05) para erradicar
     el riesgo de cola izquierda (left-tail risk) y drawdowns severos en aperturas.
2. Estimador Endógeno Sintético de Prima de Varianza (VRP_synth) y Ratio de Estructura Temporal (TSR_synth):
   - Elimina la dependencia de feeds externos de pago a CBOE (VIX/VIX3M) mediante el estimador de Garman-Klass
     y la varianza realizada multiescala (5d vs 22d), permitiendo ejecución autónoma en futuros CME.
3. Umbral de caída previa dinamizado por volatilidad diaria (Z_drop <= -1.25 sigma) en lugar de porcentaje fijo rígido.
4. Desacoplamiento modular con bandera `enable_overnight` para permitir cumplimiento en prop firms con regla RTH estricta.
5. Inclusión de Take Profit nocturno (delta_TP_on = 0.80%) con asimetría 1.6:1 frente al Stop Loss (0.50%).
6. Blindaje determinista pre-macro con Stop Loss de 8.0 pts y salida obligatoria a T-3 minutos (60s antes de embargo).
7. Gestión de riesgo Prop Trading con segregación fraccionaria optimizada (20% Noche / 80% RTH).
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
# 1. ENUMS Y ESTRUCTURAS DE DATOS DE ESTADO Y TRANSACCIÓN (F9-N1)
# ==============================================================================

class F9ExitReason(str, Enum):
    DETERMINISTIC_TIME_EXIT = "DETERMINISTIC_TIME_EXIT"  # 08:30:00 EST (ON) o T-3m (Macro)
    TAKE_PROFIT = "TAKE_PROFIT"                          # 0.80% Take Profit Nocturno
    STOP_LOSS = "STOP_LOSS"                              # 0.50% (ON) o 8.0 pts (Macro)
    EMBARGO_FLAT = "EMBARGO_FLAT"                        # Cierre de seguridad previo a ventana de embargo [T-2m, T+2m]
    TERMINAL_EOD = "TERMINAL_EOD"                        # 16:45:00 EST cierre obligatorio previo a CME maintenance
    RISK_BREACH = "RISK_BREACH"                          # Violación de DLL o MTDD


class F9RegimeState(int, Enum):
    COMPLACENCY_BULLISH = 1    # Z_VRP >= 1.00 y TSR <= 0.98 (Contango / Sesgo comprador favorable)
    NEUTRAL = 0                # Régimen neutral ordinario
    BACKWARDATION_PANIC = -1   # Z_VRP <= -1.00 o TSR >= 1.05 (Curva invertida / Bloqueo total de compras)


class F9SubModelType(str, Enum):
    OVERNIGHT = "OVERNIGHT"    # Deriva nocturna condicional dinamizada
    PRE_MACRO = "PRE_MACRO"    # Deriva pre-anuncio informada con salida en T-3m


@dataclass
class F9TradeRecord:
    """Registro analítico detallado de una operación en F9-Optimizacion -IS- N1."""
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
    take_profit_price: Optional[float] = None  # Precio de Take Profit (para Overnight)
    raw_entry_price: float = 0.0               # Precio bruto sin slippage
    raw_exit_price: Optional[float] = None     # Precio bruto de salida sin slippage
    exit_time: Optional[str] = None            # Timestamp de salida (EST)
    exit_price: Optional[float] = None         # Precio de llenado en salida (con slippage)
    exit_reason: Optional[F9ExitReason] = None
    bars_held: int = 0
    gross_pnl_pts: float = 0.0
    gross_pnl_usd: float = 0.0
    commissions_usd: float = 0.0
    slippage_usd: float = 0.0
    friction_usd: float = 0.0
    net_pnl_usd: float = 0.0
    running_balance: float = 0.0
    running_drawdown: float = 0.0
    vrp_regime: int = 0                        # S_VRP^* al momento de la entrada
    z_vrp: float = 0.0
    tsr: float = 0.0
    tsr_slope: float = 0.0                     # Pendiente de la curva (1.0 - TSR)
    is_curve_inverted: bool = False            # True si TSR >= 1.00 (Backwardation)
    macro_event: Optional[str] = None


@dataclass
class F9PropAccount:
    """
    Gestión cuantitativa de saldo y límites de Prop Trading (Topstep, Apex, Lucid, MFF)
    para la versión F9-Optimizacion -IS- N1.
    Controla el Daily Loss Limit (DLL), Maximum Trailing Drawdown (MTDD)
    y la segregación fraccionaria optimizada (lambda_on = 0.20, 20% Noche / 80% RTH).
    """
    initial_capital: float = 50000.0
    dll: float = 1000.0                        # Límite de pérdida diaria contractual
    mtdd: float = 2000.0                       # Máximo trailing drawdown
    trailing_mode: str = "eod"                 # "eod" o "intraday"
    profit_target: float = 3000.0              # Objetivo de aprobación de la prueba
    lambda_on: float = 0.20                    # 20% del DLL asignado a la sesión nocturna
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
        """Margen nocturno segregado optimizado: Delta DLL_t^nocturno = 0.20 * Delta DLL_t."""
        return self.lambda_on * self.delta_dll

    @property
    def delta_dll_rth(self) -> float:
        """Margen RTH segregado optimizado: Delta DLL_t^RTH = 0.80 * Delta DLL_t."""
        return (1.0 - self.lambda_on) * self.delta_dll

    def update_balance(self, pnl: float, bar_ts: Optional[datetime] = None) -> None:
        """Actualiza el balance, el High Water Mark y evalúa posibles violaciones de reglas."""
        self.equity += pnl
        self.daily_net_pnl += pnl
        if pnl < 0.0:
            self.daily_realized_loss += abs(pnl)

        if self.trailing_mode == "intraday":
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
            self.current_drawdown = self.peak_equity - self.equity
        else:
            self.current_drawdown = max(0.0, self.peak_equity - self.equity)

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
# 2. FUNCIONES MATEMÁTICAS FORMALES Y MODELOS DE VOLATILIDAD / RÉGIMEN (F9-N1)
# ==============================================================================

def calculate_garman_klass_variance(open_p: float, high_p: float, low_p: float, close_p: float) -> float:
    """
    Estimador insesgado de varianza de Garman-Klass (1980):
    sigma_GK^2 = 0.5 * (ln(H / L))^2 - (2 * ln(2) - 1) * (ln(C / O))^2

    Captura eficientemente la volatilidad intradía incorporando rangos extremos y apertura/cierre.
    """
    if open_p <= 0.0 or high_p <= 0.0 or low_p <= 0.0 or close_p <= 0.0:
        return 0.0
    if high_p < low_p:
        high_p, low_p = low_p, high_p

    term_hl = math.log(max(high_p / low_p, 1.0))
    term_co = math.log(close_p / open_p)
    coef = 2.0 * math.log(2.0) - 1.0  # aprox 0.386294
    gk_var = 0.5 * (term_hl ** 2) - coef * (term_co ** 2)
    return float(max(0.0, gk_var))


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
    Varianza realizada acumulada proyectada a horizonte anual (ventana K = 22 días bursátiles):
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


def calculate_proxy_implied_variance(daily_gk_variances: Sequence[float], k_vol: int = 5) -> float:
    """
    Proxy de volatilidad implícita anualizada mediante rango de Garman-Klass a corto plazo:
    sigma_{proxy, t}^2 = 252 * (1 / K_vol) * sum_{k=1}^{K_vol} sigma_{GK, t-k}^2
    con K_vol = 5 días hábiles (Alizadeh, Brandt & Diebold, 2002).
    """
    if len(daily_gk_variances) == 0:
        return 0.0
    window = daily_gk_variances[-k_vol:]
    if len(window) == 0:
        return 0.0
    mean_gk = float(np.mean(window))
    return float(252.0 * mean_gk)


def calculate_synthetic_vrp(sigma_proxy_sq: float, rv_ann: float) -> float:
    """
    Diferencial de Prima de Varianza Sintética (VRP_synth):
    VRP_{synth, t} = sigma_{proxy, t}^2 - RV_{22d, t}^ann
    """
    return float(sigma_proxy_sq - rv_ann)


def calculate_z_score_synthetic_vrp(
    current_vrp: float,
    history_vrp: Sequence[float],
    lookback: int = 60,
    eps: float = 1e-8
) -> float:
    """
    Estandarización móvil robusta de la prima de varianza sintética sobre L = 60 sesiones:
    Z_{VRP}^{synth}(t) = (VRP_{synth, t} - mu_{VRP}^{synth}(L)) / (sigma_{VRP}^{synth}(L) + eps)
    con eps = 1e-8 para protección absoluta frente a división por cero.
    """
    if len(history_vrp) == 0:
        return 0.0
    window = list(history_vrp[-(lookback - 1):]) if lookback > 1 else []
    window.append(current_vrp)
    arr = np.asarray(window, dtype=np.float64)
    mu = float(np.mean(arr))
    if abs(current_vrp - mu) < 1e-12:
        return 0.0
    sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else float(np.std(arr))
    return float((current_vrp - mu) / (sigma + eps))


def calculate_synthetic_term_structure_ratio(
    daily_rvs: Sequence[float],
    short_window: int = 5,
    long_window: int = 22,
    eps: float = 1e-8
) -> float:
    """
    Ratio sintético de estructura temporal de volatilidad (TSR_synth):
    TSR_{synth, t} = sqrt( (1/5) * sum_{k=1}^5 RV_{t-k}^{(d)} ) / ( sqrt( (1/22) * sum_{k=1}^{22} RV_{t-k}^{(d)} ) + eps )

    Interpretación analítica:
    - TSR <= 0.98: Volatilidad reciente inferior a la mensual -> Contango (Calma institucional).
    - TSR >= 1.05: Volatilidad reciente superior a la mensual -> Backwardation (Inversión / Pánico).
    """
    if len(daily_rvs) == 0:
        return 1.0  # Neutral por defecto
    short_w = daily_rvs[-short_window:] if len(daily_rvs) >= short_window else daily_rvs
    long_w = daily_rvs[-long_window:] if len(daily_rvs) >= long_window else daily_rvs

    mean_short = float(np.mean(short_w)) if len(short_w) > 0 else 0.0
    mean_long = float(np.mean(long_w)) if len(long_w) > 0 else 0.0

    vol_short = math.sqrt(max(0.0, mean_short))
    vol_long = math.sqrt(max(0.0, mean_long))

    return float(vol_short / (vol_long + eps))


def calculate_term_structure_ratio(vix: float, vix3m: float) -> float:
    """
    Ratio clásico de estructura temporal de volatilidad CBOE (Simon & Campasano, 2014):
    TSR_t = VIX_t / VIX3M_t
    """
    if vix3m <= 0:
        raise ValueError(f"VIX3M debe ser estrictamente positivo: {vix3m}")
    return float(vix / vix3m)


def evaluate_term_structure_slope(tsr: float) -> float:
    """
    Calcula la pendiente analítica de la estructura temporal de volatilidad:
    slope = 1.0 - TSR_t

    Propiedades matemáticas:
    - slope > 0: Curva en Contango (TSR < 1.0), estructura ascendente saludable.
    - slope == 0: Curva plana (TSR = 1.0).
    - slope < 0: Curva en Backwardation / Invertida (TSR > 1.0), tensión y riesgo de cola izquierda.
    """
    return float(1.0 - tsr)


def is_term_structure_inverted(tsr: float, backwardation_threshold: float = 1.00) -> bool:
    """
    Evalúa si la estructura de plazos de volatilidad se encuentra invertida (Backwardation):
    Retorna True si TSR >= backwardation_threshold (por defecto 1.00).
    """
    return bool(tsr >= backwardation_threshold)


def evaluate_vrp_regime_gate_n1(
    z_vrp: float,
    tsr: float,
    z_bull_threshold: float = 1.00,
    tsr_contango_threshold: float = 0.98,
    z_bear_threshold: float = -1.00,
    tsr_backwardation_threshold: float = 1.05,
) -> int:
    """
    Compuerta estructural optimizada de régimen S_VRP^*(t) in {-1, 0, +1}:
    +1: si Z_VRP >= +1.00 and TSR <= 0.98 (Contango / Sesgo Comprador Favorable)
    -1: si Z_VRP <= -1.00 or  TSR >= 1.05 (Backwardation / Bloqueo Total de Compras)
     0: en cualquier otro caso             (Régimen Neutral Ordinario)

    MEJORA MATEMÁTICA CRÍTICA:
    El condicionamiento estricto a TSR >= 1.05 (o curva invertida) veta incondicionalmente
    las compras, eliminando las pérdidas severas de apertura durante picos de volatilidad.
    """
    if z_vrp >= z_bull_threshold and tsr <= tsr_contango_threshold:
        return 1
    if z_vrp <= z_bear_threshold or tsr >= tsr_backwardation_threshold:
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


def calculate_standardized_intraday_drop(
    r_intraday: float,
    sigma_daily: float,
    eps: float = 1e-6
) -> float:
    """
    Retorno intradía previo estandarizado por la volatilidad diaria:
    Z_{drop, t-1} = R_{intraday, t-1} / (sigma_{daily, t-1} + eps)
    """
    return float(r_intraday / (sigma_daily + eps))


def evaluate_overnight_signal_n1(
    z_drop_prev: float,
    s_vrp: int,
    tsr: float,
    enable_overnight: bool = True,
    z_drop_crit: float = -1.25,
    max_tsr_long: float = 1.00,
) -> int:
    """
    Señal de la deriva nocturna condicional optimizada (Boyarchenko et al., 2023):
    S_{overnight}^*(t) = I(enable_overnight == 1) * I(Z_{drop, t-1} <= z_drop_crit) * I(S_VRP^* != -1) * I(TSR < max_tsr_long)

    Retorna +1 (Comprar) o 0 (Neutral).

    BLOQUEO ESTRICTO:
    Si la curva está invertida (TSR >= max_tsr_long, típicamente 1.00) o S_VRP^* == -1,
    las compras se bloquean incondicionalmente para evitar drawdowns severos de cola izquierda.
    """
    if not enable_overnight:
        return 0
    # Veto absoluto por backwardation (curva invertida / pendiente <= 0) o régimen de pánico
    if s_vrp == -1 or is_term_structure_inverted(tsr, backwardation_threshold=max_tsr_long):
        return 0
    if z_drop_prev <= z_drop_crit:
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


def evaluate_pre_macro_signal_n1(
    z_ofi_pre: float,
    current_price: float,
    vwap_prior: float,
    s_vrp: int,
    tsr: float,
    z_ofi_threshold: float = 1.80,
    max_tsr_long: float = 1.00,
) -> int:
    """
    Operador analítico optimizado de deriva pre-macro informada (Kurov et al., 2019):
    +1: Z_OFI >= +1.80 and P_t > VWAP and S_VRP^* != -1 and TSR < max_tsr_long
    -1: Z_OFI <= -1.80 and P_t < VWAP and S_VRP^* != +1
     0: en cualquier otro caso

    MEJORA MATEMÁTICA:
    Bloquea las entradas largas (+1) si la curva de volatilidad está en backwardation (TSR >= 1.00),
    mientras permite capturar ventas en corto (-1) cuando el flujo institucional confirma pánico bajista.
    """
    # Condición Long (bloqueada estrictamente si la estructura de plazos se invierte)
    if z_ofi_pre >= z_ofi_threshold and current_price > vwap_prior:
        if s_vrp != -1 and not is_term_structure_inverted(tsr, backwardation_threshold=max_tsr_long):
            return 1
        return 0

    # Condición Short
    if z_ofi_pre <= -z_ofi_threshold and current_price < vwap_prior:
        if s_vrp != 1:
            return -1
        return 0

    return 0


def calculate_overnight_sizing_n1(
    budget_daily_on: float,
    budget_cushion_on: float,
    p_entry: float,
    multiplier_std: float,
    multiplier_micro: float,
    delta_sl_on: float = 0.0050,
    n_max_std: int = 2,
    n_max_micro: int = 20,
    friction_std: float = 0.0,
    friction_micro: float = 0.0,
) -> Tuple[int, str]:
    """
    Ecuación cerrada optimizada de dimensionamiento para la Deriva Nocturna:
    N_{ON}^{(I)} = min( floor( min( Delta DLL_t^nocturno, 0.05 * D_t^cushion ) / (delta_SL^ON * P_entry * M^{(I)} + C_fric) ), N_max^ON )
    con conmutación automática Estándar a Micro y disyuntor si ni 1 micro cabe en presupuesto.
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


def calculate_pre_macro_sizing_n1(
    budget_daily_macro: float,
    budget_cushion_macro: float,
    delta_p_stop: float,
    multiplier_std: float,
    multiplier_micro: float,
    n_max_std: int = 2,
    n_max_micro: int = 20,
    friction_std: float = 0.0,
    friction_micro: float = 0.0,
) -> Tuple[int, str]:
    """
    Ecuación cerrada optimizada de dimensionamiento para la Deriva Pre-Macro:
    N_{macro}^{(I)} = min( floor( min( 0.15 * Delta DLL_t^RTH, 0.05 * D_t^cushion ) / (Delta P_stop * M^{(I)} + C_fric) ), N_max^macro )
    con conmutación automática Estándar a Micro y disyuntor si ni 1 micro cabe en presupuesto.
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
# 3. ESTRATEGIA OPTIMIZADA: F9-OPTIMIZACION -IS- N1
# ==============================================================================

class F9OptimizationISN1Strategy(BaseStrategy):
    """
    Estrategia F9-Optimizacion -IS- N1:
    Anomalías Temporales Multi-Sesión y Variance Risk Premium (VRP) Optimizada.

    Modela rigurosamente:
    - Estimador sintético de VRP (VRP_synth) y ratio de estructura temporal (TSR_synth) sin suscripciones de pago.
    - Condicionamiento estricto a la pendiente de la estructura temporal de volatilidad (TSR).
    - Bloqueo incondicional de compras durante backwardation (curva invertida) para evitar riesgo de cola izquierda.
    - Dinamización de la sobreventa previa RTH con Z_drop <= -1.25 sigma.
    - Desacoplamiento modular con bandera `enable_overnight`.
    - Take Profit nocturno del 0.80% (asimetría 1.6:1 frente a SL 0.50%).
    - Salida obligatoria pre-macro en T-3 minutos (60 segundos antes del embargo de prop trading).
    - Cierre terminal CME Globex a las 16:45:00 EST.
    """

    def __init__(
        self,
        name: str = "F9-Optimizacion -IS- N1",
        params: Optional[Dict[str, Any]] = None,
    ):
        default_params: Dict[str, Any] = {
            "symbol": "ES",
            "std_symbol": "ES",
            "micro_symbol": "MES",
            "enable_overnight": True,          # Desacoplamiento modular para prop firms RTH-only
            "use_synthetic_vrp": True,         # Cálculo endógeno de VRP y TSR con Garman-Klass y RV
            "z_drop_crit": -1.25,              # Umbral dinámico de retorno diario estandarizado
            "theta_drop": 0.0075,              # 0.75% umbral fijo clásico (fallback opcional)
            "use_standardized_drop": True,     # True usa Z_drop <= -1.25, False usa theta_drop fijo
            "delta_sl_on": 0.0050,             # 0.50% Stop Loss nocturno (25 pts en ES)
            "delta_tp_on": 0.0080,             # 0.80% Take Profit nocturno (40 pts en ES)
            "delta_p_stop_macro": 8.0,         # 8.0 puntos Stop Loss pre-macro (ES)
            "k_vol": 5,                        # Ventana de 5 días para proxy de volatilidad Garman-Klass
            "k_rv_lookback": 22,               # Días de proyección de varianza realizada (1 mes bursátil)
            "l_vrp_lookback": 60,              # Ventana rodante de estandarización Z_VRP (3 meses)
            "z_bull_threshold": 1.00,          # Umbral Z_VRP para régimen favorable alcista
            "tsr_contango_threshold": 0.98,    # TSR máximo para régimen alcista (contango)
            "z_bear_threshold": -1.00,         # Umbral Z_VRP para veto estructural bajista
            "tsr_backwardation_threshold": 1.05,# TSR mínimo para régimen de pánico/backwardation
            "max_tsr_long": 1.00,              # TSR límite absoluto: bloquea compras si TSR >= 1.00 (inversión)
            "z_ofi_threshold": 1.80,           # Umbral de flujo de órdenes pre-macro |Z_OFI| >= 1.80
            "default_vix": 16.0,               # VIX de respaldo si se usa modo clásico externo
            "default_tsr": 0.90,               # TSR de respaldo si se usa modo clásico externo
            "slippage_ticks": 1.0,             # 1 tick entrada + 1 tick salida = 2 ticks slippage total
            "target_tz": "America/New_York",
            "source_tz": None,
            "macro_schedule": None,            # Lista o dict de timestamps de noticias macro Tier-1
        }
        if params:
            default_params.update(params)
        super().__init__(name=name, params=default_params)

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
        Genera la serie temporal de posición teórica consolidada S_unified^*(t) in {-1, 0, +1}.
        Acondiciona la señal a la pendiente de la estructura temporal de volatilidad (TSR).
        """
        df = self.ensure_timezone(data.copy())
        self.validate_data(df)

        signals = pd.Series(0.0, index=df.index, name="Signal")

        # 1. Retornos logarítmicos de 5 minutos y varianza Garman-Klass barra a barra
        df["ret_log"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)
        df["gk_var"] = [
            calculate_garman_klass_variance(o, h, l, c)
            for o, h, l, c in zip(df["Open"], df["High"], df["Low"], df["Close"])
        ]

        # 2. Métricas agregadas por día (RTH return, Daily RV, Daily GK Variance)
        daily_rth_returns: Dict[Any, float] = {}
        daily_realized_variances: Dict[Any, float] = {}
        daily_gk_variances: Dict[Any, float] = {}
        daily_volatilities: Dict[Any, float] = {}

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

            rv_d = calculate_daily_realized_variance(day_group["ret_log"])
            daily_realized_variances[day_date] = rv_d
            daily_volatilities[day_date] = math.sqrt(max(0.0, rv_d))
            daily_gk_variances[day_date] = float(np.sum(day_group["gk_var"]))

        sorted_dates = sorted(daily_realized_variances.keys())
        daily_vrp_map: Dict[Any, float] = {}
        daily_z_vrp_map: Dict[Any, float] = {}
        daily_tsr_map: Dict[Any, float] = {}

        use_synth = self.params["use_synthetic_vrp"]
        has_ext_vix = "VIX" in df.columns and not use_synth
        has_ext_tsr = "TSR" in df.columns and not use_synth

        vix_series = df["VIX"] if "VIX" in df.columns else None
        ext_tsr_series = df["TSR"] if "TSR" in df.columns else None

        vrp_history: List[float] = []
        for i, dt in enumerate(sorted_dates):
            past_rvs = [daily_realized_variances[d] for d in sorted_dates[max(0, i - self.params["k_rv_lookback"] + 1): i + 1]]
            rv_ann = calculate_annualized_realized_variance(past_rvs, k=self.params["k_rv_lookback"])

            if use_synth or not has_ext_vix:
                past_gks = [daily_gk_variances[d] for d in sorted_dates[max(0, i - self.params["k_vol"] + 1): i + 1]]
                sigma_proxy_sq = calculate_proxy_implied_variance(past_gks, k_vol=self.params["k_vol"])
                vrp_val = calculate_synthetic_vrp(sigma_proxy_sq, rv_ann)
            else:
                day_vix = float(vix_series[df.index.date == dt].mean()) if len(vix_series[df.index.date == dt]) > 0 else self.params["default_vix"]
                iv = (day_vix / 100.0) ** 2
                vrp_val = float(iv - rv_ann)

            daily_vrp_map[dt] = vrp_val
            z_vrp_val = calculate_z_score_synthetic_vrp(vrp_val, vrp_history, lookback=self.params["l_vrp_lookback"])
            daily_z_vrp_map[dt] = z_vrp_val
            vrp_history.append(vrp_val)

            # TSR (Sintético o Externo)
            if use_synth or not has_ext_tsr:
                tsr_val = calculate_synthetic_term_structure_ratio(past_rvs, short_window=5, long_window=self.params["k_rv_lookback"])
            else:
                tsr_val = float(ext_tsr_series[df.index.date == dt].mean()) if len(ext_tsr_series[df.index.date == dt]) > 0 else self.params["default_tsr"]
            daily_tsr_map[dt] = tsr_val

        # 3. Macro events parsing (agrupado por fecha para velocidad O(1))
        macro_events = self.params.get("macro_schedule") or []
        macro_by_date: Dict[Any, List[pd.Timestamp]] = {}
        if isinstance(macro_events, (list, set)):
            for me in macro_events:
                val = me[0] if isinstance(me, tuple) else me
                ts_obj = pd.Timestamp(val)
                norm_ts = ts_obj.tz_convert(self.params["target_tz"]) if ts_obj.tzinfo else ts_obj.tz_localize(self.params["target_tz"])
                macro_by_date.setdefault(norm_ts.date(), []).append(norm_ts)

        # 4. Evaluación barra a barra
        for i in range(len(df)):
            ts = df.index[i]
            cur_time = ts.time()
            cur_date = ts.date()

            prev_dates = [d for d in sorted_dates if d < cur_date]
            prev_date = prev_dates[-1] if prev_dates else None
            r_intraday_prev = daily_rth_returns.get(prev_date, 0.0) if prev_date else 0.0
            sigma_daily_prev = daily_volatilities.get(prev_date, 0.01) if prev_date else 0.01

            z_vrp = daily_z_vrp_map.get(prev_date, 0.0) if prev_date is not None else 0.0
            tsr = daily_tsr_map.get(prev_date, self.params["default_tsr"]) if prev_date is not None else self.params["default_tsr"]

            s_vrp = evaluate_vrp_regime_gate_n1(
                z_vrp=z_vrp,
                tsr=tsr,
                z_bull_threshold=self.params["z_bull_threshold"],
                tsr_contango_threshold=self.params["tsr_contango_threshold"],
                z_bear_threshold=self.params["z_bear_threshold"],
                tsr_backwardation_threshold=self.params["tsr_backwardation_threshold"],
            )

            # A) Ventana de Embargo Prohibido [T-2m, T+2m]
            day_macros = macro_by_date.get(cur_date, [])
            in_embargo = False
            for m_dt in day_macros:
                if (m_dt - timedelta(minutes=2)) <= ts <= (m_dt + timedelta(minutes=2)):
                    in_embargo = True
                    break

            if in_embargo:
                signals.iloc[i] = 0.0
                continue

            # B) Cierre preventivo de fin de sesión (>= 15:55 EST)
            if cur_time >= time(15, 55):
                signals.iloc[i] = 0.0
                continue

            # C) Sub-modelo Pre-Macro (Prioridad sobre Overnight a las 08:30)
            in_macro_window = False
            for m_dt in day_macros:
                t_minus_15 = m_dt - timedelta(minutes=15)
                t_minus_3 = m_dt - timedelta(minutes=3)
                if t_minus_15 <= ts <= t_minus_3:
                    in_macro_window = True
                    sub_prev = df.loc[(df.index >= m_dt - timedelta(minutes=30)) & (df.index < t_minus_15)]
                    p_cur = float(df["Close"].iloc[i])
                    vwap_prior = calculate_window_vwap(sub_prev["Close"], sub_prev["Volume"]) if len(sub_prev) > 0 else p_cur
                    z_ofi = float(df["Z_OFI"].iloc[i]) if "Z_OFI" in df.columns else (2.5 if p_cur > vwap_prior else -2.5)
                    s_macro = evaluate_pre_macro_signal_n1(
                        z_ofi_pre=z_ofi,
                        current_price=p_cur,
                        vwap_prior=vwap_prior,
                        s_vrp=s_vrp,
                        tsr=tsr,
                        z_ofi_threshold=self.params["z_ofi_threshold"],
                        max_tsr_long=self.params["max_tsr_long"],
                    )
                    signals.iloc[i] = float(s_macro)
                    break

            if in_macro_window:
                continue

            # D) Sub-modelo Overnight (Boyarchenko et al., 2023)
            # Ventana operativa [01:30, 08:30) EST
            if time(1, 30) <= cur_time < time(8, 30):
                if self.params["use_standardized_drop"]:
                    z_drop = calculate_standardized_intraday_drop(r_intraday_prev, sigma_daily_prev)
                    s_on = evaluate_overnight_signal_n1(
                        z_drop_prev=z_drop,
                        s_vrp=s_vrp,
                        tsr=tsr,
                        enable_overnight=self.params["enable_overnight"],
                        z_drop_crit=self.params["z_drop_crit"],
                        max_tsr_long=self.params["max_tsr_long"],
                    )
                else:
                    # Modo clásico con umbral porcentual fijo
                    drop_val = -1.5 if r_intraday_prev <= -self.params["theta_drop"] else 0.0
                    s_on = evaluate_overnight_signal_n1(
                        z_drop_prev=drop_val,
                        s_vrp=s_vrp,
                        tsr=tsr,
                        enable_overnight=self.params["enable_overnight"],
                        z_drop_crit=-1.0,
                        max_tsr_long=self.params["max_tsr_long"],
                    )

                if s_on == 1:
                    signals.iloc[i] = 1.0
                continue

            # Fuera de ventanas: posición plana (0.0)
            signals.iloc[i] = 0.0

        return signals


# ==============================================================================
# 4. MOTOR DE BACKTEST Y AUDITORÍA DE PROP TRADING (F9-N1)
# ==============================================================================

class F9OptimizationISN1Backtester:
    """
    Simulador determinista de ejecución y evaluación de Prop Trading para F9-Optimizacion -IS- N1.
    Audita y aplica:
    1. Entrada overnight en 01:30:00 EST sujeta a Z_drop <= -1.25 y bloqueo estricto por backwardation (TSR >= 1.00 o S_VRP = -1).
    2. Salida overnight por Take Profit (0.80%), Stop Loss (0.50%) o salida determinista a las 08:30:00 EST.
    3. Entrada Pre-Macro en T-15m y salida determinista forzosa en T-3m (o Stop Loss 8.0 pts en ES).
    4. Cero contratos durante el embargo normativo [T-2m, T+2m] (60s de colchón preventivo).
    5. Cierre forzoso universal CME a las 16:45:00 EST.
    6. Auditoría de Prop Trading: DLL, MTDD, comisiones CME y slippage institucional round-turn.
    """

    def __init__(
        self,
        strategy: F9OptimizationISN1Strategy,
        account_size: float = 50000.0,
        daily_loss_limit: float = 1000.0,
        max_trailing_loss: float = 2000.0,
        trailing_mode: str = "eod",
        profit_target: float = 3000.0,
    ):
        self.strategy = strategy
        self.account = F9PropAccount(
            initial_capital=account_size,
            dll=daily_loss_limit,
            mtdd=max_trailing_loss,
            trailing_mode=trailing_mode,
            profit_target=profit_target,
            lambda_on=self.strategy.params.get("lambda_on", 0.20),
        )
        self.trades: List[F9TradeRecord] = []

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Ejecuta la simulación completa sobre el conjunto de datos históricos."""
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

        friction_unit_std = (2.0 * slippage_ticks * tick_std * m_std) + comm_std
        friction_unit_micro = (2.0 * slippage_ticks * tick_micro * m_micro) + comm_micro

        # Retornos logarítmicos y Garman-Klass
        df["ret_log"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)
        df["gk_var"] = [
            calculate_garman_klass_variance(o, h, l, c)
            for o, h, l, c in zip(df["Open"], df["High"], df["Low"], df["Close"])
        ]

        # Métricas agregadas por día
        daily_rth_returns: Dict[Any, float] = {}
        daily_realized_variances: Dict[Any, float] = {}
        daily_gk_variances: Dict[Any, float] = {}
        daily_volatilities: Dict[Any, float] = {}

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

            rv_d = calculate_daily_realized_variance(day_group["ret_log"])
            daily_realized_variances[day_date] = rv_d
            daily_volatilities[day_date] = math.sqrt(max(0.0, rv_d))
            daily_gk_variances[day_date] = float(np.sum(day_group["gk_var"]))

        sorted_dates = sorted(daily_realized_variances.keys())
        daily_vrp_map: Dict[Any, float] = {}
        daily_z_vrp_map: Dict[Any, float] = {}
        daily_tsr_map: Dict[Any, float] = {}

        use_synth = self.strategy.params["use_synthetic_vrp"]
        has_ext_vix = "VIX" in df.columns and not use_synth
        has_ext_tsr = "TSR" in df.columns and not use_synth

        vix_series = df["VIX"] if "VIX" in df.columns else None
        ext_tsr_series = df["TSR"] if "TSR" in df.columns else None

        vrp_history: List[float] = []
        for i, dt in enumerate(sorted_dates):
            past_rvs = [daily_realized_variances[d] for d in sorted_dates[max(0, i - self.strategy.params["k_rv_lookback"] + 1): i + 1]]
            rv_ann = calculate_annualized_realized_variance(past_rvs, k=self.strategy.params["k_rv_lookback"])

            if use_synth or not has_ext_vix:
                past_gks = [daily_gk_variances[d] for d in sorted_dates[max(0, i - self.strategy.params["k_vol"] + 1): i + 1]]
                sigma_proxy_sq = calculate_proxy_implied_variance(past_gks, k_vol=self.strategy.params["k_vol"])
                vrp_val = calculate_synthetic_vrp(sigma_proxy_sq, rv_ann)
            else:
                day_vix = float(vix_series[df.index.date == dt].mean()) if len(vix_series[df.index.date == dt]) > 0 else self.strategy.params["default_vix"]
                iv = (day_vix / 100.0) ** 2
                vrp_val = float(iv - rv_ann)

            daily_vrp_map[dt] = vrp_val
            z_vrp_val = calculate_z_score_synthetic_vrp(vrp_val, vrp_history, lookback=self.strategy.params["l_vrp_lookback"])
            daily_z_vrp_map[dt] = z_vrp_val
            vrp_history.append(vrp_val)

            if use_synth or not has_ext_tsr:
                tsr_val = calculate_synthetic_term_structure_ratio(past_rvs, short_window=5, long_window=self.strategy.params["k_rv_lookback"])
            else:
                tsr_val = float(ext_tsr_series[df.index.date == dt].mean()) if len(ext_tsr_series[df.index.date == dt]) > 0 else self.strategy.params["default_tsr"]
            daily_tsr_map[dt] = tsr_val

        # Macro schedule (agrupado por fecha para O(1))
        macro_events = self.strategy.params.get("macro_schedule") or []
        macro_by_date: Dict[Any, List[Tuple[pd.Timestamp, str]]] = {}
        if isinstance(macro_events, list):
            for me in macro_events:
                if isinstance(me, tuple):
                    ts_m, name_m = me
                else:
                    ts_m, name_m = me, "Tier-1 News"
                ts_obj = pd.Timestamp(ts_m)
                norm_ts = ts_obj.tz_convert(self.strategy.params["target_tz"]) if ts_obj.tzinfo else ts_obj.tz_localize(self.strategy.params["target_tz"])
                macro_by_date.setdefault(norm_ts.date(), []).append((norm_ts, name_m))

        current_trade: Optional[F9TradeRecord] = None
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
            day_macro_list = macro_by_date.get(cur_date, [])

            if cur_date != current_day:
                if current_day is not None:
                    self.account.close_eod_session()
                current_day = cur_date
                self.account.start_session()

            if self.account.is_breached:
                if current_trade is not None:
                    self._close_trade(
                        trade=current_trade,
                        exit_price=bar_open,
                        exit_time=ts,
                        exit_reason=F9ExitReason.RISK_BREACH,
                        slippage_ticks=slippage_ticks,
                    )
                    current_trade = None
                break

            # ------------------------------------------------------------------
            # GESTIÓN DE POSICIÓN ABIERTA
            # ------------------------------------------------------------------
            if current_trade is not None:
                current_trade.bars_held += 1

                # 1. Cierre terminal CME Globex (16:45:00 EST)
                if cur_time >= time(16, 45):
                    self._close_trade(
                        trade=current_trade,
                        exit_price=bar_close,
                        exit_time=ts,
                        exit_reason=F9ExitReason.TERMINAL_EOD,
                        slippage_ticks=slippage_ticks,
                    )
                    current_trade = None
                    continue

                # 2. Gestión Trade Overnight
                if current_trade.submodel == F9SubModelType.OVERNIGHT.value:
                    # A) Stop Loss (0.50% de caída con modelado de gaps)
                    if bar_low <= current_trade.stop_loss_price:
                        exit_price_sl = min(bar_open, current_trade.stop_loss_price)
                        self._close_trade(
                            trade=current_trade,
                            exit_price=exit_price_sl,
                            exit_time=ts,
                            exit_reason=F9ExitReason.STOP_LOSS,
                            slippage_ticks=slippage_ticks,
                        )
                        current_trade = None
                        continue

                    # B) Take Profit (0.80% de rebote nocturno con modelado de gaps)
                    if current_trade.take_profit_price is not None and bar_high >= current_trade.take_profit_price:
                        exit_price_tp = max(bar_open, current_trade.take_profit_price)
                        self._close_trade(
                            trade=current_trade,
                            exit_price=exit_price_tp,
                            exit_time=ts,
                            exit_reason=F9ExitReason.TAKE_PROFIT,
                            slippage_ticks=slippage_ticks,
                        )
                        current_trade = None
                        continue

                    # C) Cierre si hay noticia macro a las 08:30 EST
                    macro_dt_830 = None
                    for m_dt, _ in day_macro_list:
                        if m_dt.time() == time(8, 30):
                            macro_dt_830 = m_dt
                            break

                    if macro_dt_830 is not None:
                        t_exit_macro = macro_dt_830 - timedelta(minutes=3)
                        t_embargo_start = macro_dt_830 - timedelta(minutes=2)
                        if ts >= t_exit_macro or (ts + bar_delta) > t_embargo_start:
                            exit_p = bar_open if ts >= t_exit_macro else bar_close
                            self._close_trade(
                                trade=current_trade,
                                exit_price=exit_p,
                                exit_time=t_exit_macro,
                                exit_reason=F9ExitReason.EMBARGO_FLAT,
                                slippage_ticks=slippage_ticks,
                            )
                            current_trade = None
                            continue

                    # D) Salida determinista incondicional a las 08:30:00 EST
                    if cur_time >= time(8, 30):
                        self._close_trade(
                            trade=current_trade,
                            exit_price=bar_open if cur_time == time(8, 30) else bar_close,
                            exit_time=ts,
                            exit_reason=F9ExitReason.DETERMINISTIC_TIME_EXIT,
                            slippage_ticks=slippage_ticks,
                        )
                        current_trade = None
                        continue

                # 3. Gestión Trade Pre-Macro
                elif current_trade.submodel == F9SubModelType.PRE_MACRO.value:
                    macro_dt = None
                    for m_dt, _ in day_macro_list:
                        if (m_dt - timedelta(minutes=20)) <= ts <= (m_dt + timedelta(minutes=5)):
                            macro_dt = m_dt
                            break

                    # A) Stop Loss en puntos (8.0 pts en ES con modelado de gaps)
                    if current_trade.direction == 1 and bar_low <= current_trade.stop_loss_price:
                        exit_price_sl = min(bar_open, current_trade.stop_loss_price)
                        self._close_trade(
                            trade=current_trade,
                            exit_price=exit_price_sl,
                            exit_time=ts,
                            exit_reason=F9ExitReason.STOP_LOSS,
                            slippage_ticks=slippage_ticks,
                        )
                        current_trade = None
                        continue
                    elif current_trade.direction == -1 and bar_high >= current_trade.stop_loss_price:
                        exit_price_sl = max(bar_open, current_trade.stop_loss_price)
                        self._close_trade(
                            trade=current_trade,
                            exit_price=exit_price_sl,
                            exit_time=ts,
                            exit_reason=F9ExitReason.STOP_LOSS,
                            slippage_ticks=slippage_ticks,
                        )
                        current_trade = None
                        continue

                    # B) Salida obligatoria en T-3m (60s antes de embargo T-2m)
                    if macro_dt is not None:
                        t_exit_macro = macro_dt - timedelta(minutes=3)
                        t_embargo_start = macro_dt - timedelta(minutes=2)

                        if ts >= t_exit_macro or (ts + bar_delta) > t_embargo_start:
                            exit_p = bar_open if ts >= t_exit_macro else bar_close
                            self._close_trade(
                                trade=current_trade,
                                exit_price=exit_p,
                                exit_time=t_exit_macro,
                                exit_reason=F9ExitReason.DETERMINISTIC_TIME_EXIT,
                                slippage_ticks=slippage_ticks,
                            )
                            current_trade = None
                            continue

                    # C) Cierre preventivo ante proximidad a embargo
                    if macro_dt is not None and ts >= (macro_dt - timedelta(minutes=2)):
                        self._close_trade(
                            trade=current_trade,
                            exit_price=bar_open,
                            exit_time=ts,
                            exit_reason=F9ExitReason.EMBARGO_FLAT,
                            slippage_ticks=slippage_ticks,
                        )
                        current_trade = None
                        continue

            # ------------------------------------------------------------------
            # EVALUACIÓN DE NUEVAS ENTRADAS
            # ------------------------------------------------------------------
            if current_trade is None and not self.account.is_breached:
                is_embargo = False
                for m_dt, _ in day_macro_list:
                    if (m_dt - timedelta(minutes=2)) <= ts <= (m_dt + timedelta(minutes=2)):
                        is_embargo = True
                        break
                if is_embargo:
                    continue

                if cur_time >= time(15, 55):
                    continue

                prev_dates = [d for d in sorted_dates if d < cur_date]
                prev_date = prev_dates[-1] if prev_dates else None
                r_intraday_prev = daily_rth_returns.get(prev_date, 0.0) if prev_date else 0.0
                sigma_daily_prev = daily_volatilities.get(prev_date, 0.01) if prev_date else 0.01

                z_vrp = daily_z_vrp_map.get(prev_date, 0.0) if prev_date is not None else 0.0
                tsr = daily_tsr_map.get(prev_date, self.strategy.params["default_tsr"]) if prev_date is not None else self.strategy.params["default_tsr"]
                tsr_slope = evaluate_term_structure_slope(tsr)
                curve_inverted = is_term_structure_inverted(tsr, self.strategy.params["max_tsr_long"])

                s_vrp = evaluate_vrp_regime_gate_n1(
                    z_vrp=z_vrp,
                    tsr=tsr,
                    z_bull_threshold=self.strategy.params["z_bull_threshold"],
                    tsr_contango_threshold=self.strategy.params["tsr_contango_threshold"],
                    z_bear_threshold=self.strategy.params["z_bear_threshold"],
                    tsr_backwardation_threshold=self.strategy.params["tsr_backwardation_threshold"],
                )

                # A) Evaluar Sub-modelo Overnight (01:30 EST)
                if time(1, 30) <= cur_time <= time(1, 35) and self.strategy.params["enable_overnight"]:
                    if self.strategy.params["use_standardized_drop"]:
                        z_drop = calculate_standardized_intraday_drop(r_intraday_prev, sigma_daily_prev)
                        s_on = evaluate_overnight_signal_n1(
                            z_drop_prev=z_drop,
                            s_vrp=s_vrp,
                            tsr=tsr,
                            enable_overnight=True,
                            z_drop_crit=self.strategy.params["z_drop_crit"],
                            max_tsr_long=self.strategy.params["max_tsr_long"],
                        )
                    else:
                        drop_val = -1.5 if r_intraday_prev <= -self.strategy.params["theta_drop"] else 0.0
                        s_on = evaluate_overnight_signal_n1(
                            z_drop_prev=drop_val,
                            s_vrp=s_vrp,
                            tsr=tsr,
                            enable_overnight=True,
                            z_drop_crit=-1.0,
                            max_tsr_long=self.strategy.params["max_tsr_long"],
                        )

                    if s_on == 1:
                        p_entry = bar_close
                        budget_daily_on = self.account.delta_dll_overnight
                        budget_cushion_on = self.account.gamma_t_on * self.account.cushion_mtdd
                        inc_fric = self.strategy.params.get("include_friction_in_sizing", False)
                        fric_s = friction_unit_std if inc_fric else 0.0
                        fric_m = friction_unit_micro if inc_fric else 0.0

                        n_contracts, c_type = calculate_overnight_sizing_n1(
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
                            tp_price = p_entry * (1.0 + self.strategy.params["delta_tp_on"])
                            tick_val = tick_micro if c_type == "MICRO" else tick_std
                            actual_entry = p_entry + (slippage_ticks * tick_val)

                            current_trade = F9TradeRecord(
                                trade_id=trade_id_counter,
                                date=str(cur_date),
                                submodel=F9SubModelType.OVERNIGHT.value,
                                direction=1,
                                contracts=n_contracts,
                                contract_type=c_type,
                                symbol=sym,
                                entry_time=str(ts),
                                entry_price=actual_entry,
                                raw_entry_price=p_entry,
                                stop_loss_price=sl_price,
                                take_profit_price=tp_price,
                                vrp_regime=s_vrp,
                                z_vrp=z_vrp,
                                tsr=tsr,
                                tsr_slope=tsr_slope,
                                is_curve_inverted=curve_inverted,
                            )
                            trade_id_counter += 1
                            continue

                # B) Evaluar Sub-modelo Pre-Macro (T-15m)
                for m_dt, m_name in day_macro_list:
                    t_trigger = m_dt - timedelta(minutes=15)
                    if t_trigger <= ts <= (t_trigger + timedelta(minutes=5)):
                        sub_vwap = df.loc[(df.index >= m_dt - timedelta(minutes=30)) & (df.index < t_trigger)]
                        vwap_prior = calculate_window_vwap(sub_vwap["Close"], sub_vwap["Volume"]) if len(sub_vwap) > 0 else bar_close
                        z_ofi = float(df["Z_OFI"].iloc[i]) if "Z_OFI" in df.columns else (2.5 if bar_close > vwap_prior else -2.5)
                        s_macro = evaluate_pre_macro_signal_n1(
                            z_ofi_pre=z_ofi,
                            current_price=bar_close,
                            vwap_prior=vwap_prior,
                            s_vrp=s_vrp,
                            tsr=tsr,
                            z_ofi_threshold=self.strategy.params["z_ofi_threshold"],
                            max_tsr_long=self.strategy.params["max_tsr_long"],
                        )

                        if s_macro != 0:
                            p_entry = bar_close
                            budget_daily_macro = self.account.phi_macro_dll * self.account.delta_dll_rth
                            budget_cushion_macro = self.account.gamma_macro_dd * self.account.cushion_mtdd
                            delta_pts = self.strategy.params["delta_p_stop_macro"]

                            inc_fric = self.strategy.params.get("include_friction_in_sizing", False)
                            fric_s = friction_unit_std if inc_fric else 0.0
                            fric_m = friction_unit_micro if inc_fric else 0.0

                            n_contracts, c_type = calculate_pre_macro_sizing_n1(
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

                                current_trade = F9TradeRecord(
                                    trade_id=trade_id_counter,
                                    date=str(cur_date),
                                    submodel=F9SubModelType.PRE_MACRO.value,
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
                                    tsr_slope=tsr_slope,
                                    is_curve_inverted=curve_inverted,
                                    macro_event=m_name,
                                )
                                trade_id_counter += 1
                                break

        if current_trade is not None:
            last_ts = df.index[-1]
            last_close = float(df["Close"].iloc[-1])
            self._close_trade(
                trade=current_trade,
                exit_price=last_close,
                exit_time=last_ts,
                exit_reason=F9ExitReason.TERMINAL_EOD,
                slippage_ticks=slippage_ticks,
            )

        self.account.close_eod_session()
        return self._generate_report(df)

    def _close_trade(
        self,
        trade: F9TradeRecord,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: F9ExitReason,
        slippage_ticks: float,
    ) -> None:
        """Calcula el P&L exacto y actualiza la cuenta tras el cierre de una operación."""
        is_micro = (trade.contract_type == "MICRO")
        spec = self.strategy.spec_micro if is_micro else self.strategy.spec_std
        multiplier = spec["multiplier"]
        comm_rt = spec["commission_rt"]
        tick_size = spec["tick"]

        raw_exit = exit_price
        actual_exit = raw_exit - (trade.direction * slippage_ticks * tick_size)
        trade.raw_exit_price = round(raw_exit, 2)
        trade.exit_price = round(actual_exit, 2)
        trade.exit_time = str(exit_time)
        trade.exit_reason = exit_reason

        raw_entry = trade.raw_entry_price if trade.raw_entry_price > 0.0 else trade.entry_price
        raw_pts = (raw_exit - raw_entry) * trade.direction
        trade.gross_pnl_pts = round(raw_pts, 2)
        trade.gross_pnl_usd = round(raw_pts * multiplier * trade.contracts, 2)

        trade.commissions_usd = round(comm_rt * trade.contracts, 2)
        trade.slippage_usd = round(2.0 * slippage_ticks * tick_size * multiplier * trade.contracts, 2)
        trade.friction_usd = round(trade.commissions_usd + trade.slippage_usd, 2)
        trade.net_pnl_usd = round(trade.gross_pnl_usd - trade.friction_usd, 2)

        self.account.update_balance(trade.net_pnl_usd, exit_time)
        trade.running_balance = round(self.account.equity, 2)
        trade.running_drawdown = round(self.account.current_drawdown, 2)

        self.trades.append(trade)

    def _generate_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compila métricas cuantitativas y auditoría formal de Prop Trading."""
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
        total_days = len(set(df.index.date))

        if trades_df.empty:
            metrics = {
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "net_profit_usd": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_usd": 0.0,
                "sharpe_ratio": 0.0,
            }
            prop_eval = {
                "status": "ACTIVE",
                "reason": "Sin operaciones ejecutadas en la ventana analizada",
                "final_balance": self.account.equity,
                "net_profit": 0.0,
                "total_days_evaluated": total_days,
                "is_breached": self.account.is_breached,
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
