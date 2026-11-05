"""
src/strategies/har_rv_j.py
Implementación Matemática y Microestructural de la Estrategia F8:
Modelos Econométricos Heterogéneos HAR-RV-J y Dimensionamiento por Volatilidad Dinámica.

Fundamento Académico:
1. Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility.
   Journal of Financial Econometrics, 7(2), 174-196. DOI: 10.1093/jjfinec/nbp001.
2. Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007). Roughing It Up: Including
   Jump Components in the Measurement, Modeling, and Forecasting of Return Volatility.
   The Review of Economics and Statistics, 89(4), 701-720. DOI: 10.1162/rest.89.4.701.
3. Moreira, A., & Muir, T. (2017). Volatility-Managed Portfolios.
   The Journal of Finance, 72(4), 1611-1644. DOI: 10.1111/jofi.12513.
4. Barndorff-Nielsen, O. E., & Shephard, N. (2004). Power and Bipower Variation with
   Quantum Jumps. Journal of Financial Econometrics, 2(1), 1-37. DOI: 10.1093/jjfinec/nbh001.

Especificación: reports/estrategias_matematicas/F8_modelos_econometricos_har_rv_j.md
"""

from dataclasses import dataclass, field
from datetime import time, datetime
from enum import Enum
import math
from typing import Dict, Any, List, Optional, Tuple, Sequence, Union
import numpy as np
import pandas as pd

from src.core.futures_specs import get_spec


# ==============================================================================
# 1. CONSTANTES MATEMÁTICAS FORMALES (BARNDORFF-NIELSEN & SHEPHARD 2004)
# ==============================================================================

# mu_1 = E[|Z|] = sqrt(2/pi) para Z ~ N(0, 1)
MU_1: float = math.sqrt(2.0 / math.pi)
MU_1_INV_SQ: float = math.pi / 2.0  # (mu_1)^(-2) = pi / 2 ≈ 1.5707963268

# mu_{4/3} = 2^(2/3) * Gamma(7/6) / Gamma(1/2) = 2^(2/3) * Gamma(7/6) / sqrt(pi)
MU_4_3: float = (2.0 ** (2.0 / 3.0)) * math.gamma(7.0 / 6.0) / math.sqrt(math.pi)
MU_4_3_INV_CUBED: float = MU_4_3 ** (-3.0)  # ≈ 1.74332

# Constante de corrección asintótica en el denominador del estadístico Z_J:
# c_v = (pi / 2)^2 + pi - 5 ≈ 0.60907
C_V_JUMP: float = (math.pi / 2.0) ** 2 + math.pi - 5.0

# Valor crítico para detección de saltos bajo nivel de significancia alpha_J = 0.001
# Phi^(-1)(1 - 0.001) = Phi^(-1)(0.999) ≈ 3.0902323
Z_ALPHA_JUMP_001: float = 3.090232306167813


# ==============================================================================
# 2. ENUMS Y ESTRUCTURAS DE DATOS DE ESTADO Y TRANSACCIÓN
# ==============================================================================

class HARExitReason(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    SESSION_CLOSE = "SESSION_CLOSE"


class HARStrategyState(str, Enum):
    AWAITING_OPEN = "AWAITING_OPEN"
    EVALUATING_GATE = "EVALUATING_GATE"
    GATE_VETOED = "GATE_VETOED"
    IN_POSITION = "IN_POSITION"
    POSITION_CLOSED = "POSITION_CLOSED"
    CIRCUIT_BREAKER_HALT = "CIRCUIT_BREAKER_HALT"
    ACCOUNT_BREACHED = "ACCOUNT_BREACHED"


@dataclass
class HARStateVector:
    """Vector de estado econométrico S_t^{HAR} in R_+^7 (Sección 2.2)."""
    date: str
    rv_daily: float
    bv: float
    jump: float
    continuous: float
    rv_weekly: float
    rv_monthly: float
    rv_predicted: float
    sigma_predicted_pts: float
    jump_ratio: float
    jump_gate_passed: bool
    volatility_managed_weight: float = 1.0


@dataclass
class TradeRecord:
    date: str
    symbol: str
    direction: int                      # +1 = Long, -1 = Short
    contracts: int                     # Número de contratos
    is_micro: bool                     # True si se ejecutó con microcontratos
    entry_time: str
    entry_price: float
    stop_loss: float
    take_profit: float
    sigma_pts: float
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[HARExitReason] = None
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
    Gestión cuantitativa de riesgo para cuentas de Prop Trading (Sección 5).
    Controla el Daily Loss Limit (DLL), Trailing Drawdown (MTDD),
    colchón de trailing (D_t^cushion) y atenuación de racha phi_{streak}(k).
    """
    initial_capital: float = 50000.0
    dll: float = 1000.0                # Daily Loss Limit
    mtdd: float = 2000.0               # Maximum Trailing Drawdown
    equity: float = 50000.0
    peak_equity: float = 50000.0       # High Water Mark (HWM)
    current_drawdown: float = 0.0
    daily_realized_loss: float = 0.0   # Pérdida acumulada en el día (positiva)
    daily_pnl: float = 0.0             # PnL neto del día actual
    consecutive_losses: int = 0        # k = número de días perdedores consecutivos
    is_breached: bool = False
    breach_reason: Optional[str] = None

    def start_session(self, session_date: Any) -> None:
        """Reinicia los acumuladores diarios al inicio de cada nueva sesión."""
        self.daily_realized_loss = 0.0
        self.daily_pnl = 0.0

    @property
    def remaining_dll(self) -> float:
        """Delta DLL_t = max(0, DLL - daily_loss_t)"""
        return max(0.0, self.dll - self.daily_realized_loss)

    @property
    def trailing_cushion(self) -> float:
        """D_t^{cushion} = max(0, MTDD - current_drawdown_t)"""
        return max(0.0, self.mtdd - self.current_drawdown)

    def calculate_streak_factor(self) -> float:
        """
        Factor defensivo de atenuación por racha perdedora (Sección 5.3):
        phi_streak(k) = 1.00 si k == 0
                        0.50 si k == 1
                        0.25 si k >= 2
        """
        if self.consecutive_losses == 0:
            return 1.00
        elif self.consecutive_losses == 1:
            return 0.50
        else:
            return 0.25

    def calculate_risk_budget(self, psi_risk: float = 0.25, gamma_t: float = 0.08) -> float:
        """
        Presupuesto de riesgo combinado:
        Budget = phi_streak(k) * min( psi_risk * Delta DLL_t, gamma_t * D_t^{cushion} )
        """
        if self.is_breached:
            return 0.0

        rem_dll = self.remaining_dll
        cushion = self.trailing_cushion
        if rem_dll <= 0.0 or cushion <= 0.0:
            return 0.0

        phi = self.calculate_streak_factor()
        budget = phi * min(psi_risk * rem_dll, gamma_t * cushion)
        return max(0.0, budget)

    def record_trade(self, trade: TradeRecord) -> None:
        """Actualiza balance, HWM, drawdown flotante, rachas y verifica límites."""
        self.equity += trade.net_pnl_usd
        self.daily_pnl += trade.net_pnl_usd

        # En prop trading, la pérdida diaria corresponde a la pérdida neta acumulada en la sesión
        if self.daily_pnl < 0:
            self.daily_realized_loss = abs(self.daily_pnl)
        else:
            self.daily_realized_loss = 0.0

        # Actualizar High Water Mark y Trailing Drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        self.current_drawdown = self.peak_equity - self.equity

        # Verificar violación de reglas de prop trading
        if self.current_drawdown >= self.mtdd:
            self.is_breached = True
            self.breach_reason = (
                f"TRAILING_DRAWDOWN_BREACH: Drawdown ${self.current_drawdown:.2f} >= MTDD ${self.mtdd:.2f}"
            )
        elif self.daily_realized_loss >= self.dll:
            self.is_breached = True
            self.breach_reason = (
                f"DAILY_LOSS_LIMIT_BREACH: Pérdida diaria ${self.daily_realized_loss:.2f} >= DLL ${self.dll:.2f}"
            )

        trade.running_equity = self.equity
        trade.running_drawdown = self.current_drawdown

    def end_session(self) -> None:
        """Actualiza el contador de días perdedores consecutivos al final de la jornada."""
        if self.daily_pnl < 0:
            self.consecutive_losses += 1
        elif self.daily_pnl > 0:
            self.consecutive_losses = 0


# ==============================================================================
# 3. FORMULACIÓN MATEMÁTICA PURA (SECCIÓN 3 DE F8)
# ==============================================================================

def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calcula los retornos logarítmicos intradía:
    r_{t, j} = ln(P_{t, j}) - ln(P_{t, j-1})
    """
    if len(prices) < 2:
        return np.array([], dtype=float)
    if np.any(prices <= 0):
        raise ValueError("Todos los precios deben ser estrictamente positivos para calcular retornos logarítmicos.")
    log_p = np.log(prices)
    return np.diff(log_p)


def calculate_realized_variance(returns: np.ndarray) -> float:
    """
    Varianza Realizada Diaria RV_t^{(d)} (Ecuación 92):
    RV_t^{(d)} = sum_{j=1}^M r_{t, j}^2
    """
    if len(returns) == 0:
        return 0.0
    return float(np.sum(returns ** 2))


def calculate_bipower_variation(returns: np.ndarray) -> float:
    """
    Variación Bipower Consistente frente a Saltos BV_t (Ecuación 96):
    BV_t = mu_1^{-2} * (M / (M - 1)) * sum_{j=2}^M |r_{t, j}| * |r_{t, j-1}|
    donde mu_1^{-2} = pi / 2.
    """
    m = len(returns)
    if m < 2:
        return 0.0
    abs_r = np.abs(returns)
    sum_prod = float(np.sum(abs_r[1:] * abs_r[:-1]))
    scale = (m / (m - 1.0)) * MU_1_INV_SQ
    return float(scale * sum_prod)


def calculate_tripower_quarticity(returns: np.ndarray) -> float:
    """
    Tri-Power Quarticity TQ_t (Ecuación 102):
    TQ_t = M * mu_{4/3}^{-3} * (M / (M - 2)) * sum_{j=3}^M |r_{t, j}|^{4/3} * |r_{t, j-1}|^{4/3} * |r_{t, j-2}|^{4/3}
    donde mu_{4/3}^{-3} ≈ 1.74332.
    """
    m = len(returns)
    if m < 3:
        return 0.0
    abs_r_43 = np.abs(returns) ** (4.0 / 3.0)
    sum_prod = float(np.sum(abs_r_43[2:] * abs_r_43[1:-1] * abs_r_43[:-2]))
    scale = m * MU_4_3_INV_CUBED * (m / (m - 2.0))
    return float(scale * sum_prod)


def calculate_jump_statistic(rv: float, bv: float, tq: float, m: int) -> float:
    """
    Estadístico asintótico de detección de saltos Z_{J, t} (Ecuación 107):
    Z_{J, t} = [ (RV_t - BV_t) / RV_t ] / sqrt( c_v * (1/M) * max(1, TQ_t / BV_t^2) )
    donde c_v = (pi/2)^2 + pi - 5 ≈ 0.60907.
    """
    if rv <= 1e-14 or bv <= 1e-14 or m < 3:
        return 0.0

    ratio_diff = (rv - bv) / rv
    ratio_tq_bv = tq / (bv ** 2) if bv > 0 else 1.0
    adj_factor = max(1.0, ratio_tq_bv)

    radicand = C_V_JUMP * (1.0 / m) * adj_factor
    if radicand <= 1e-14:
        return 0.0

    z_stat = ratio_diff / math.sqrt(radicand)
    return float(z_stat)


def decompose_realized_variance(
    returns: np.ndarray,
    alpha_j: float = 0.001
) -> Tuple[float, float, float, float, float]:
    """
    Descomposición no paramétrica de varianza (Sección 3.1):
    Retorna: (RV_t, BV_t, TQ_t, J_t, C_t)
    Cumple rigurosamente:
    - J_t = I(Z_{J, t} > z_{alpha_J}) * max(RV_t - BV_t, 0) >= 0
    - C_t = RV_t - J_t > 0
    - C_t + J_t === RV_t
    """
    m = len(returns)
    rv = calculate_realized_variance(returns)
    if m < 3 or rv <= 1e-14:
        # Casos degenerate: cero saltos, varianza continua mínima
        rv_safe = max(rv, 1e-12)
        return rv_safe, rv_safe, 0.0, 0.0, rv_safe

    bv = calculate_bipower_variation(returns)
    tq = calculate_tripower_quarticity(returns)
    z_stat = calculate_jump_statistic(rv, bv, tq, m)

    # Nivel crítico para alpha_J = 0.001
    z_crit = Z_ALPHA_JUMP_001 if alpha_j == 0.001 else 3.0902323

    if z_stat > z_crit:
        jump = max(rv - bv, 0.0)
    else:
        jump = 0.0

    continuous = max(rv - jump, 1e-12)
    return rv, bv, tq, jump, continuous


def calculate_heterogeneous_rv(daily_rv_series: Sequence[float]) -> Tuple[float, float]:
    """
    Cálculo de componentes heterogéneos agregados (Ecuación 122):
    RV_t^{(w)} = (1/5) * sum_{i=0}^4 RV_{t-i}^{(d)}
    RV_t^{(m)} = (1/22) * sum_{i=0}^{21} RV_{t-i}^{(d)}
    """
    if not daily_rv_series:
        return 1e-4, 1e-4

    n = len(daily_rv_series)
    # Semanal (5 días)
    w_len = min(5, n)
    rv_w = float(np.mean(daily_rv_series[-w_len:]))

    # Mensual (22 días)
    m_len = min(22, n)
    rv_m = float(np.mean(daily_rv_series[-m_len:]))

    return max(rv_w, 1e-12), max(rv_m, 1e-12)


def fit_log_har_rv_j(
    historical_records: List[Dict[str, float]],
    min_obs: int = 22
) -> Tuple[np.ndarray, float]:
    """
    Ajuste econométrico del modelo Log-HAR-RV-J mediante MCO (Sección 3.2):
    ln(RV_{t+1}^{(d)}) = beta_0 + beta_d * ln(C_t) + beta_w * ln(RV_t^{(w)}) +
                         beta_m * ln(RV_t^{(m)}) + beta_j * ln(1 + J_t) + eps_{t+1}

    Retorna: (betas [beta_0, beta_d, beta_w, beta_m, beta_j], sigma_eps_sq)
    Si no hay suficientes observaciones, retorna coeficientes canónicos robustos de la literatura.
    """
    # Coeficientes canónicos por defecto (Corsi 2009 / Andersen et al. 2007)
    default_betas = np.array([-0.25, 0.42, 0.30, 0.20, -0.05], dtype=float)
    default_sigma_eps_sq = 0.04

    if len(historical_records) < min_obs + 1:
        return default_betas, default_sigma_eps_sq

    # Construir matrices X e y
    # Cada registro i predice el RV del registro i+1
    x_list = []
    y_list = []

    for i in range(len(historical_records) - 1):
        rec_t = historical_records[i]
        rec_next = historical_records[i + 1]

        c_t = max(rec_t["continuous"], 1e-12)
        rv_w = max(rec_t["rv_weekly"], 1e-12)
        rv_m = max(rec_t["rv_monthly"], 1e-12)
        j_t = max(rec_t["jump"], 0.0)
        rv_target = max(rec_next["rv_daily"], 1e-12)

        x_row = [
            1.0,
            math.log(c_t),
            math.log(rv_w),
            math.log(rv_m),
            math.log(1.0 + j_t)
        ]
        x_list.append(x_row)
        y_list.append(math.log(rv_target))

    X = np.array(x_list, dtype=float)
    y = np.array(y_list, dtype=float)

    if X.shape[0] < X.shape[1] + 2:
        return default_betas, default_sigma_eps_sq

    try:
        # OLS con regularización Ridge minúscula para evitar singularidad
        ridge = 1e-6 * np.eye(X.shape[1])
        xtx = X.T @ X + ridge
        xty = X.T @ y
        betas = np.linalg.solve(xtx, xty)

        residuals = y - X @ betas
        dof = max(1, len(y) - len(betas))
        sigma_eps_sq = float(np.sum(residuals ** 2) / dof)
        sigma_eps_sq = max(1e-6, min(sigma_eps_sq, 1.0))
        return betas, sigma_eps_sq
    except Exception:
        return default_betas, default_sigma_eps_sq


def predict_log_har_rv_j(
    c_t: float,
    rv_w: float,
    rv_m: float,
    j_t: float,
    betas: np.ndarray,
    sigma_eps_sq: float
) -> float:
    """
    Pronóstico insesgado con corrección log-normal (Ecuación 130):
    RV_hat_{t+1}^{(d)} = exp( beta_0 + beta_d*ln(C_t) + beta_w*ln(RV_t^{(w)}) +
                              beta_m*ln(RV_t^{(m)}) + beta_j*ln(1+J_t) + 0.5 * sigma_eps^2 )
    Garantiza positividad estricta RV_hat_{t+1}^{(d)} > 0 bajo cualquier perturbación.
    """
    c_safe = max(c_t, 1e-12)
    rv_w_safe = max(rv_w, 1e-12)
    rv_m_safe = max(rv_m, 1e-12)
    j_safe = max(j_t, 0.0)

    ln_pred = (
        betas[0]
        + betas[1] * math.log(c_safe)
        + betas[2] * math.log(rv_w_safe)
        + betas[3] * math.log(rv_m_safe)
        + betas[4] * math.log(1.0 + j_safe)
        + 0.5 * sigma_eps_sq
    )

    # Limitar el exponente a un rango razonable para evitar overflow/underflow
    ln_pred = max(-30.0, min(ln_pred, 5.0))
    rv_pred = math.exp(ln_pred)
    return float(max(rv_pred, 1e-12))


def calculate_point_volatility(predicted_rv: float, price: float) -> float:
    """
    Proyección de varianza a volatilidad diaria en puntos de índice (Ecuación 137):
    sigma_hat_{t+1}^{pts} = P_t * sqrt( RV_hat_{t+1}^{(d)} / 252 )
    """
    if price <= 0 or predicted_rv <= 0:
        return 0.0
    return float(price * math.sqrt(predicted_rv / 252.0))


def evaluate_jump_gate(jump: float, rv_daily: float, theta_j: float = 0.35) -> Tuple[float, bool]:
    """
    Compuerta analítica de régimen de saltos (Sección 3.4):
    rho_{J, t} = J_t / RV_t^{(d)}
    Si rho_{J, t} >= theta_J (0.35) -> VETO (mercado bajo choque discontinuo severo).
    Retorna: (rho_J, is_passed)
    """
    if rv_daily <= 1e-14:
        return 0.0, True

    rho_j = float(jump / rv_daily)
    passed = bool(rho_j < theta_j)
    return rho_j, passed


def quantize_price(price: float, tick_size: float = 0.25) -> float:
    """Discretiza el precio al múltiplo indivisible de tick de CME Globex (Sección 2.1)."""
    if tick_size <= 0:
        return price
    return round(round(price / tick_size) * tick_size, 4)


SIGMA_STAR_MAP: Dict[str, float] = {
    "ES": 25.0,
    "MES": 25.0,
    "NQ": 90.0,
    "MNQ": 90.0
}


def calculate_volatility_managed_weight(sigma_pts: float, symbol: str = "ES") -> float:
    """
    Escalado Inverso de Volatilidad (Moreira & Muir, 2017) (Sección 5.1):
    w_t^{vol} = sigma^* / sigma_hat_{t+1}^{pts}
    con sigma^* = 25.0 para ES/MES y 90.0 para NQ/MNQ.
    """
    if sigma_pts <= 0.0:
        return 0.0
    sigma_star = SIGMA_STAR_MAP.get(symbol.upper(), 25.0)
    return float(sigma_star / sigma_pts)


def calculate_dynamic_barriers(
    entry_price: float,
    direction: int,
    sigma_pts: float,
    kappa_tp: float = 1.80,
    kappa_sl: float = 1.00,
    tick_size: float = 0.25
) -> Tuple[float, float]:
    """
    Barreras dinámicas de salida escaladas por volatilidad HAR-RV-J (Sección 4.1):
    P_exit^{TP} = P_{t_0} + S_{t+1}^* * kappa_{TP} * sigma_{t+1}^{pts}
    P_exit^{SL} = P_{t_0} - S_{t+1}^* * kappa_{SL} * sigma_{t+1}^{pts}
    Garantiza una distancia mínima de al menos 1 tick y cuantiza los niveles
    exactamente a la escala de ticks de CME Globex.
    Relación de asimetría favorable: kappa_{TP} / kappa_{SL} = 1.80.
    """
    if direction not in (1, -1):
        raise ValueError("direction debe ser +1 (Long) o -1 (Short).")
    if sigma_pts <= 0:
        raise ValueError("sigma_pts debe ser estrictamente positivo.")

    # Asegurar distancia mínima de al menos 1 tick
    tp_distance = max(kappa_tp * sigma_pts, tick_size)
    sl_distance = max(kappa_sl * sigma_pts, tick_size)

    raw_tp = entry_price + direction * tp_distance
    raw_sl = entry_price - direction * sl_distance

    tp = quantize_price(raw_tp, tick_size)
    sl = quantize_price(raw_sl, tick_size)
    return tp, sl


def check_circuit_breaker(
    intraday_rv: float,
    predicted_rv: float,
    xi_circuit: float = 2.50
) -> bool:
    """
    Disyuntor analítico de choque de volatilidad intradía (Sección 4.2):
    RV_{t+1, tau}^{intra} >= xi_circuit * RV_hat_{t+1}^{(d)} con xi_circuit = 2.50.
    """
    if predicted_rv <= 0:
        return False
    return bool(intraday_rv >= xi_circuit * predicted_rv)


def calculate_position_sizing(
    sigma_pts: float,
    contract_symbol: str,
    account: PropAccount,
    max_contracts: int = 5,
    z_score: float = 2.326,     # Confianza del 99%
    psi_risk: float = 0.25,     # Fracción de DLL
    gamma_t: float = 0.08       # Fracción de Trailing Cushion
) -> Tuple[int, bool, float]:
    """
    Algoritmo declarativo de dimensionamiento y conmutación Estándar a Micro (Sección 5):
    1. Calcula el presupuesto Budget = phi_streak(k) * min(psi_risk * Delta DLL_t, gamma_t * D_t^{cushion})
    2. Si el símbolo solicitado es directamente un microcontrato (MES, MNQ), dimensiona para micro.
    3. Si es estándar (ES, NQ): calcula N_{std}. Si N_{std} == 0, conmuta automáticamente a microcontratos.
    Retorna: (num_contratos, es_micro, presupuesto_usd)
    """
    if account.is_breached or sigma_pts <= 0:
        return 0, False, 0.0

    budget = account.calculate_risk_budget(psi_risk=psi_risk, gamma_t=gamma_t)
    if budget <= 0.0:
        return 0, False, 0.0

    sym_clean = contract_symbol.upper()
    is_explicit_micro = sym_clean in ("MES", "MNQ")

    if sym_clean in ("ES", "MES"):
        std_sym, micro_sym = "ES", "MES"
    elif sym_clean in ("NQ", "MNQ"):
        std_sym, micro_sym = "NQ", "MNQ"
    else:
        std_sym = sym_clean
        micro_sym = sym_clean

    spec_micro = get_spec(micro_sym)
    m_micro = spec_micro["multiplier"]
    unit_var_micro = z_score * sigma_pts * m_micro
    max_micros = 10 * max_contracts

    # Si se solicitó explícitamente microcontrato
    if is_explicit_micro:
        if unit_var_micro <= 0:
            return 0, True, budget
        n_micro = int(math.floor(budget / unit_var_micro))
        n_micro = min(n_micro, max_micros)
        return (n_micro, True, budget) if n_micro >= 1 else (0, True, budget)

    # Contrato estándar solicitado
    spec_std = get_spec(std_sym)
    m_std = spec_std["multiplier"]
    unit_var_std = z_score * sigma_pts * m_std
    if unit_var_std <= 0:
        return 0, False, budget

    n_std = int(math.floor(budget / unit_var_std))
    n_std = min(n_std, max_contracts)

    if n_std >= 1:
        return n_std, False, budget

    # Conmutación automática a microcontrato si n_std == 0
    if unit_var_micro <= 0:
        return 0, True, budget

    n_micro = int(math.floor(budget / unit_var_micro))
    n_micro = min(n_micro, max_micros)

    if n_micro >= 1:
        return n_micro, True, budget
    else:
        # Parada de protección absoluta
        return 0, True, budget


# ==============================================================================
# 4. CLASE ESTRATÉGICA PRINCIPAL HAR-RV-J
# ==============================================================================

class EconometricHARRVJStrategy:
    """
    Implementación formal completa de la Estrategia F8.
    Encapsula el ciclo de vida:
    - Estimación y actualización de variables intradía a 5 minutos
    - Descomposición RV / BV / TQ / Jump / Continuous
    - Pronóstico Log-HAR-RV-J insesgado
    - Compuerta de Saltos
    - Generación de Señales de Entrada
    - Gestión de Barreras Dinámicas y Disyuntor Intradía
    """

    def __init__(
        self,
        symbol: str = "ES",
        kappa_tp: float = 1.80,
        kappa_sl: float = 1.00,
        theta_jump: float = 0.35,
        xi_circuit: float = 2.50,
        max_standard_contracts: int = 5,
        session_open_time: time = time(9, 30),
        session_close_time: time = time(15, 55),
        slippage_ticks: float = 1.0
    ):
        self.symbol = symbol.upper()
        self.pair_map = {
            "ES": ("ES", "MES"),
            "MES": ("ES", "MES"),
            "NQ": ("NQ", "MNQ"),
            "MNQ": ("NQ", "MNQ"),
        }
        self.std_sym, self.micro_sym = self.pair_map.get(self.symbol, (self.symbol, self.symbol))
        self.spec_std = get_spec(self.std_sym)
        self.spec_micro = get_spec(self.micro_sym)

        self.kappa_tp = kappa_tp
        self.kappa_sl = kappa_sl
        self.theta_jump = theta_jump
        self.xi_circuit = xi_circuit
        self.max_contracts = max_standard_contracts
        self.session_open_time = session_open_time
        self.session_close_time = session_close_time
        self.slippage_ticks = slippage_ticks

        # Historial de estados diarios para ajuste de Log-HAR
        self.history: List[Dict[str, float]] = []
        self.betas = np.array([-0.25, 0.42, 0.30, 0.20, -0.05], dtype=float)
        self.sigma_eps_sq = 0.04

    def update_model_coefficients(self) -> None:
        """Re-calibra los coeficientes MCO del modelo si hay suficiente historial."""
        if len(self.history) >= 22:
            self.betas, self.sigma_eps_sq = fit_log_har_rv_j(self.history, min_obs=22)

    def process_prior_session(
        self,
        date_str: str,
        intraday_bars: pd.DataFrame
    ) -> Optional[HARStateVector]:
        """
        Procesa la sesión RTH de t al cierre (16:00 EST) y genera el pronóstico para t+1.
        """
        if intraday_bars.empty:
            return None

        # Si el DataFrame contiene índice temporal, filtrar estrictamente la sesión regular RTH
        if isinstance(intraday_bars.index, pd.DatetimeIndex):
            times = intraday_bars.index.time
            rth_mask = (times >= self.session_open_time) & (times <= time(16, 0))
            bars_to_process = intraday_bars[rth_mask]
            if len(bars_to_process) < 3:
                return None
        else:
            if len(intraday_bars) < 3:
                return None
            bars_to_process = intraday_bars

        closes = bars_to_process["Close"].values
        # Retornos de 5 minutos
        returns = calculate_log_returns(closes)
        if len(returns) == 0:
            return None

        rv, bv, tq, jump, continuous = decompose_realized_variance(returns, alpha_j=0.001)

        # Agregar varianzas heterogéneas previas
        hist_rvs = [h["rv_daily"] for h in self.history] + [rv]
        rv_w, rv_m = calculate_heterogeneous_rv(hist_rvs)

        # Predecir RV para t+1
        rv_pred = predict_log_har_rv_j(
            c_t=continuous,
            rv_w=rv_w,
            rv_m=rv_m,
            j_t=jump,
            betas=self.betas,
            sigma_eps_sq=self.sigma_eps_sq
        )

        last_price = float(closes[-1])
        sigma_pts = calculate_point_volatility(rv_pred, last_price)
        rho_j, gate_passed = evaluate_jump_gate(jump, rv, theta_j=self.theta_jump)
        w_vol = calculate_volatility_managed_weight(sigma_pts, self.symbol)

        state = HARStateVector(
            date=date_str,
            rv_daily=rv,
            bv=bv,
            jump=jump,
            continuous=continuous,
            rv_weekly=rv_w,
            rv_monthly=rv_m,
            rv_predicted=rv_pred,
            sigma_predicted_pts=sigma_pts,
            jump_ratio=rho_j,
            jump_gate_passed=gate_passed,
            volatility_managed_weight=w_vol
        )

        # Registrar en el historial para futuros ajustes MCO
        self.history.append({
            "rv_daily": rv,
            "bv": bv,
            "jump": jump,
            "continuous": continuous,
            "rv_weekly": rv_w,
            "rv_monthly": rv_m,
            "rv_pred": rv_pred
        })

        self.update_model_coefficients()
        return state

    def simulate_day(
        self,
        session_date: Any,
        day_bars: pd.DataFrame,
        prior_state: Optional[HARStateVector],
        account: PropAccount
    ) -> Optional[TradeRecord]:
        """
        Ejecuta la simulación intradía barra a barra de la sesión t+1.
        """
        if prior_state is None or day_bars.empty:
            return None

        if account.is_breached:
            return None

        date_str = str(session_date)
        day_bars = day_bars.sort_index()

        # 1. Verificar compuerta de saltos previa
        if not prior_state.jump_gate_passed:
            # Veto por salto extremo (Sección 3.4)
            return None

        # 2. Filtrar horario regular RTH
        times = day_bars.index.time
        open_time = self.session_open_time
        close_time = self.session_close_time

        rth_bars = day_bars[(times >= open_time) & (times <= close_time)]
        if len(rth_bars) < 3:
            return None

        # 3. Señal direccional primaria intradía s_{t+1}^{dir} (Sección 3.4)
        # Se evalúa en la primera barra completa de la sesión
        first_bar = rth_bars.iloc[0]
        p_open = float(first_bar["Open"])
        p_close = float(first_bar["Close"])

        if p_close > p_open:
            direction = 1  # Long
        elif p_close < p_open:
            direction = -1  # Short
        else:
            # Neutro: s_{t+1}^{dir} = 0 -> S_{Neutral, t+1} = 1 (Ecuación 157)
            return None

        # 4. Dimensionamiento de posición dinámico
        sigma_pts = prior_state.sigma_predicted_pts
        if sigma_pts <= 0:
            return None

        contracts, is_micro, budget = calculate_position_sizing(
            sigma_pts=sigma_pts,
            contract_symbol=self.symbol,
            account=account,
            max_contracts=self.max_contracts
        )

        if contracts <= 0:
            return None

        # Determinar especificación según el tipo de contrato ejecutado
        active_spec = self.spec_micro if is_micro else self.spec_std
        active_symbol = self.micro_sym if is_micro else self.std_sym
        multiplier = active_spec["multiplier"]
        commission_rt = active_spec["commission_rt"]
        tick_size = active_spec["tick"]

        # 5. Entrada en la segunda barra tras confirmación de dirección
        entry_bar = rth_bars.iloc[1]
        entry_time_str = str(entry_bar.name)
        entry_price = quantize_price(float(entry_bar["Open"]), tick_size)

        # Costo de slippage monetario
        slippage_pts = self.slippage_ticks * tick_size
        slippage_usd = slippage_pts * multiplier * contracts
        commissions_usd = commission_rt * contracts

        # 6. Barreras dinámicas
        tp_price, sl_price = calculate_dynamic_barriers(
            entry_price=entry_price,
            direction=direction,
            sigma_pts=sigma_pts,
            kappa_tp=self.kappa_tp,
            kappa_sl=self.kappa_sl,
            tick_size=tick_size
        )

        trade = TradeRecord(
            date=date_str,
            symbol=active_symbol,
            direction=direction,
            contracts=contracts,
            is_micro=is_micro,
            entry_time=entry_time_str,
            entry_price=entry_price,
            stop_loss=sl_price,
            take_profit=tp_price,
            sigma_pts=sigma_pts,
            commissions_usd=commissions_usd,
            slippage_usd=slippage_usd
        )

        # 7. Monitoreo intradía y paradas de salida
        # Inicializar varianza realizada acumulada de la sesión t+1 (Ecuación 199)
        first_open = float(first_bar["Open"])
        first_close = float(first_bar["Close"])
        r0 = math.log(first_close / first_open) if (first_open > 0 and first_close > 0) else 0.0
        curr_intra_rv = r0 ** 2

        exit_time = None
        exit_price = None
        exit_reason = None

        prev_close = first_close
        bars_after_entry = rth_bars.iloc[1:]
        for idx, (bar_dt, bar) in enumerate(bars_after_entry.iterrows()):
            curr_bar_time = bar_dt.time()
            b_open = float(bar["Open"])
            high = float(bar["High"])
            low = float(bar["Low"])
            close = float(bar["Close"])

            # Acumular varianza realizada intradía de 5 minutos (Ecuación 199)
            if prev_close > 0 and close > 0:
                r_step = math.log(close / prev_close)
                curr_intra_rv += r_step ** 2
            prev_close = close

            # A. Disyuntor de volatilidad intradía
            if check_circuit_breaker(curr_intra_rv, prior_state.rv_predicted, self.xi_circuit):
                exit_time = str(bar_dt)
                exit_price = quantize_price(close, tick_size)
                exit_reason = HARExitReason.CIRCUIT_BREAKER
                break

            # B. Evaluación de Barreras Dinámicas (Take-Profit / Stop-Loss)
            # En Prop Trading, el Stop-Loss tiene prioridad absoluta ante barras de alta volatilidad
            if direction == 1:  # Long
                sl_hit = (low <= sl_price)
                tp_hit = (high >= tp_price)

                if sl_hit:
                    exit_time = str(bar_dt)
                    fill_p = min(sl_price, b_open)
                    exit_price = quantize_price(fill_p, tick_size)
                    exit_reason = HARExitReason.STOP_LOSS
                    break
                elif tp_hit:
                    exit_time = str(bar_dt)
                    fill_p = max(tp_price, b_open)
                    exit_price = quantize_price(fill_p, tick_size)
                    exit_reason = HARExitReason.TAKE_PROFIT
                    break
            else:  # Short
                sl_hit = (high >= sl_price)
                tp_hit = (low <= tp_price)

                if sl_hit:
                    exit_time = str(bar_dt)
                    fill_p = max(sl_price, b_open)
                    exit_price = quantize_price(fill_p, tick_size)
                    exit_reason = HARExitReason.STOP_LOSS
                    break
                elif tp_hit:
                    exit_time = str(bar_dt)
                    fill_p = min(tp_price, b_open)
                    exit_price = quantize_price(fill_p, tick_size)
                    exit_reason = HARExitReason.TAKE_PROFIT
                    break

            # C. Cierre determinista de sesión (15:55:00 EST)
            if curr_bar_time >= close_time:
                exit_time = str(bar_dt)
                exit_price = quantize_price(close, tick_size)
                exit_reason = HARExitReason.SESSION_CLOSE
                break

        # Si al final de la jornada la posición sigue abierta, liquidar al último cierre
        if exit_price is None:
            last_bar = rth_bars.iloc[-1]
            exit_time = str(last_bar.name)
            exit_price = quantize_price(float(last_bar["Close"]), tick_size)
            exit_reason = HARExitReason.SESSION_CLOSE

        # 8. Cálculo de PnL y registro en PropAccount
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason

        pnl_points = direction * (exit_price - entry_price)
        trade.gross_pnl_points = float(pnl_points)
        trade.gross_pnl_usd = float(pnl_points * multiplier * contracts)
        trade.net_pnl_usd = float(trade.gross_pnl_usd - trade.commissions_usd - trade.slippage_usd)

        account.record_trade(trade)
        return trade


# ==============================================================================
# 5. MOTOR DE BACKTESTING INTEGRADO (HARRVJBacktester)
# ==============================================================================

class HARRVJBacktester:
    """
    Motor de backtest para la estrategia F8:
    Itera día a día, mantiene el estado econométrico sin contaminación ni Lookahead Bias,
    registra métricas de Prop Trading y emite informe de rendimiento.
    """

    def __init__(
        self,
        strategy: EconometricHARRVJStrategy,
        account: PropAccount
    ):
        self.strategy = strategy
        self.account = account
        self.trades: List[TradeRecord] = []
        self.daily_states: Dict[str, HARStateVector] = {}
        self.daily_equity_curve: Dict[str, float] = {}

    def run(self, intraday_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Ejecuta el backtest día a día sobre el dataset intradía de 5 minutos.
        """
        df = intraday_data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df.set_index("datetime", inplace=True)
            else:
                raise ValueError("intraday_data debe contener DatetimeIndex o columna 'datetime'.")

        df.sort_index(inplace=True)
        unique_dates = sorted(list(set(df.index.date)))

        prior_state: Optional[HARStateVector] = None

        for d in unique_dates:
            day_mask = (df.index.date == d)
            day_bars = df[day_mask]

            # Si la cuenta ya fue descalificada en días previos
            if self.account.is_breached:
                self.daily_equity_curve[str(d)] = self.account.equity
                continue

            # Verificar si la jornada tiene barras en horario RTH
            if isinstance(day_bars.index, pd.DatetimeIndex):
                times = day_bars.index.time
                has_rth = bool(np.any((times >= self.strategy.session_open_time) & (times <= time(16, 0))))
                if not has_rth:
                    # Sesión no regular (ej. Domingo Globex por la noche) -> omitir
                    continue

            self.account.start_session(d)

            # 1. Si existe estado previo (calculado al cierre RTH anterior), simular sesión de t
            if prior_state is not None:
                trade = self.strategy.simulate_day(
                    session_date=d,
                    day_bars=day_bars,
                    prior_state=prior_state,
                    account=self.account
                )
                if trade:
                    self.trades.append(trade)

            # 2. Al término de la sesión de t, procesar datos de t para generar estado de t+1
            new_state = self.strategy.process_prior_session(
                date_str=str(d),
                intraday_bars=day_bars
            )
            if new_state is not None:
                prior_state = new_state
                self.daily_states[str(d)] = prior_state

            self.account.end_session()
            self.daily_equity_curve[str(d)] = self.account.equity

        return self.generate_performance_report()

    def generate_performance_report(self) -> Dict[str, Any]:
        """Calcula el informe exhaustivo de rendimiento cuantitativo y cumplimiento Prop Trading."""
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

        # Retornos diarios para Sharpe y Sortino
        eq_series = pd.Series(self.daily_equity_curve)
        daily_returns = eq_series.pct_change().dropna()

        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
            neg_ret = daily_returns[daily_returns < 0]
            sortino = float((daily_returns.mean() / neg_ret.std()) * np.sqrt(252)) if (len(neg_ret) > 0 and neg_ret.std() > 0) else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        max_dd_usd = self.account.current_drawdown
        max_dd_pct = (max_dd_usd / self.account.peak_equity * 100.0) if self.account.peak_equity > 0 else 0.0
        calmar = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

        # Conteo por razones de salida
        exit_counts = {}
        for r in HARExitReason:
            exit_counts[r.value] = sum(1 for t in self.trades if t.exit_reason == r)

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
            "exit_reasons": exit_counts,
            "is_prop_breached": self.account.is_breached,
            "breach_reason": self.account.breach_reason or "None (Compliant)"
        }

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Exporta el log completo de trades a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        records = [t.__dict__ for t in self.trades]
        return pd.DataFrame(records)
