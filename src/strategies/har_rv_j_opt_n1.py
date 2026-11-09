"""
src/strategies/har_rv_j_opt_n1.py
Implementación Matemática y Microestructural de la Versión Alterna:
F8-Optimizacion -IS- N1: Descomposición de Varianza Bipotencia y Penalización
Exponencial de Saltos Estocásticos sobre el Apalancamiento Dinámico.

Fundamento Académico y Cuantitativo:
1. Barndorff-Nielsen, O. E., & Shephard, N. (2004). Power and Bipower Variation
   with Quantum Jumps. Journal of Financial Econometrics, 2(1), 1-37.
2. Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility.
   Journal of Financial Econometrics, 7(2), 174-196.
3. Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007). Roughing It Up: Including
   Jump Components in the Measurement, Modeling, and Forecasting of Return Volatility.
   The Review of Economics and Statistics, 89(4), 701-720.
4. Moreira, A., & Muir, T. (2017). Volatility-Managed Portfolios.
   The Journal of Finance, 72(4), 1611-1644.

Mejora Matemática frente a F8 Base:
- Separa formalmente la varianza continua (Bipower Variation BV_t) del componente
  de saltos bruscos (J_t = max(RV_t - BV_t, 0) condicionado a Z_{J, t} > z_alpha).
- En lugar de un veto binario abrupto que descarte días aprovechables o permita
  sobreapalancamiento en regímenes con saltos latentes, F8-Optimizacion -IS- N1 aplica
  una función de penalización exponencial continua sobre el apalancamiento y el
  presupuesto de riesgo:
      Psi_t^{jump} = exp(-lambda * JumpMetric_t)
  modulando el apalancamiento efectivo:
      w_t^{opt} = w_t^{vol} * Psi_t^{jump}
      Budget_t^{eff} = Budget_t * Psi_t^{jump}
- Desacoplamiento de la apertura RTH (observación de los primeros 15 minutos entre
  09:30 y 09:45 EST) y escalado de barreras asimétricas (kappa_tp >= 2.0, kappa_sl >= 1.25)
  para mitigar el barrido prematuro por ruido de subasta en futuros CME.
- Respeto integral de reglas de microestructura CME Globex (ticks indivisibles de 0.25,
  outside bars, gaps) y cumplimiento estricto de Prop Trading (DLL, MTDD, comisiones y slippage).
"""

from dataclasses import dataclass, field
from datetime import time, datetime
from enum import Enum
import math
from typing import Dict, Any, List, Optional, Tuple, Sequence, Union
import numpy as np
import pandas as pd

from src.core.futures_specs import get_spec
from src.strategies.har_rv_j import (
    HARExitReason,
    HARStrategyState,
    PropAccount,
    TradeRecord,
    MU_1,
    MU_1_INV_SQ,
    MU_4_3,
    MU_4_3_INV_CUBED,
    C_V_JUMP,
    Z_ALPHA_JUMP_001,
    calculate_log_returns,
    calculate_realized_variance,
    calculate_bipower_variation,
    calculate_tripower_quarticity,
    calculate_jump_statistic,
    decompose_realized_variance,
    calculate_heterogeneous_rv,
    fit_log_har_rv_j,
    predict_log_har_rv_j,
    calculate_point_volatility,
    quantize_price,
    SIGMA_STAR_MAP,
    calculate_volatility_managed_weight,
    calculate_dynamic_barriers,
    check_circuit_breaker,
)


# ==============================================================================
# 1. ESTRUCTURAS DE DATOS EXTENDIDAS: F8-OPTIMIZACION -IS- N1
# ==============================================================================

@dataclass
class HARStateVectorOptN1:
    """
    Vector de estado econométrico enriquecido para F8-Optimizacion -IS- N1.
    Almacena la descomposición continua/salto, el estadístico Z_J, la métrica
    de salto y la penalización exponencial calculada.
    """
    date: str
    rv_daily: float
    bv: float
    jump: float
    continuous: float
    rv_weekly: float
    rv_monthly: float
    rv_predicted: float
    sigma_predicted_pts: float
    z_jump_stat: float
    is_jump_significant: bool
    jump_ratio: float
    jump_metric: float
    jump_penalty_factor: float          # Psi_t^{jump} = exp(-lambda * JumpMetric)
    volatility_managed_weight: float    # w_t^{vol} = sigma* / sigma_pts
    effective_weight: float             # w_t^{opt} = w_t^{vol} * Psi_t^{jump}
    jump_gate_passed: bool
    opening_rv_15m: float = 0.0
    opening_shock_ratio: float = 1.0


@dataclass
class HARRVJOptN1TradeRecord(TradeRecord):
    """
    Registro extendido de trade para F8-Optimizacion -IS- N1.
    Almacena factores de penalización de apalancamiento y métricas de apertura.
    """
    jump_stat: float = 0.0
    jump_metric: float = 0.0
    jump_penalty_factor: float = 1.0
    effective_budget_usd: float = 0.0
    opening_rv_15m: float = 0.0


# ==============================================================================
# 2. FUNCIONES MATEMÁTICAS PURAS: SEPARACIÓN Y PENALIZACIÓN EXPONENCIAL DE SALTOS
# ==============================================================================

def calculate_continuous_jump_decomposition(
    returns: np.ndarray,
    alpha_j: float = 0.001,
    z_crit_override: Optional[float] = None
) -> Tuple[float, float, float, float, float, float, bool]:
    """
    Descomposición matemática rigurosa entre Varianza Bipotencia (continua) y Saltos:
    Retorna: (RV_t, BV_t, TQ_t, Z_{J, t}, J_t, C_t, is_jump_significant)

    Ecuaciones:
    - RV_t = sum_{j=1}^M r_{t, j}^2
    - BV_t = mu_1^{-2} * (M / (M - 1)) * sum_{j=2}^M |r_{t, j}| * |r_{t, j-1}|
    - TQ_t = M * mu_{4/3}^{-3} * (M / (M - 2)) * sum |r_{t, j}|^{4/3} |r_{t, j-1}|^{4/3} |r_{t, j-2}|^{4/3}
    - Z_{J, t} = [ (RV_t - BV_t) / RV_t ] / sqrt( c_v * (1/M) * max(1, TQ_t / BV_t^2) )
    - Salto estadísticamente significativo si Z_{J, t} > z_alpha:
        J_t = I(Z_{J, t} > z_alpha) * max(RV_t - BV_t, 0.0)
        C_t = RV_t - J_t
    Propiedades analíticas garantizadas:
        J_t >= 0, C_t > 0, C_t + J_t === RV_t.
    """
    m = len(returns)
    rv = calculate_realized_variance(returns)
    if m < 3 or rv <= 1e-14:
        rv_safe = max(rv, 1e-12)
        return rv_safe, rv_safe, 0.0, 0.0, 0.0, rv_safe, False

    bv = calculate_bipower_variation(returns)
    tq = calculate_tripower_quarticity(returns)
    z_stat = calculate_jump_statistic(rv, bv, tq, m)

    if z_crit_override is not None:
        z_crit = float(z_crit_override)
    else:
        # Nivel crítico estándar z_{0.001} ≈ 3.09023
        z_crit = Z_ALPHA_JUMP_001 if alpha_j == 0.001 else 3.0902323

    is_significant = bool(z_stat > z_crit)
    if is_significant:
        jump = max(rv - bv, 0.0)
    else:
        jump = 0.0

    continuous = max(rv - jump, 1e-12)
    return rv, bv, tq, z_stat, jump, continuous, is_significant


def calculate_jump_exponential_penalty(
    jump: float,
    rv_daily: float,
    z_stat: float,
    is_significant: bool,
    lambda_jump: float = 4.0,
    gamma_z: float = 0.0,
    z_crit: float = Z_ALPHA_JUMP_001,
    theta_hard_veto: float = 0.55
) -> Tuple[float, float, float, bool]:
    """
    Función de penalización exponencial del apalancamiento ante saltos discretos:
    Psi_t^{jump} = exp( -lambda * JumpMetric_t )

    donde:
    - Si el salto no es estadísticamente significativo (is_significant is False o jump <= 0):
      JumpMetric_t = 0.0 ==> Psi_t^{jump} = exp(0) = 1.0 (sin penalización).
    - Si el salto es estadísticamente significativo:
      JumpMetric_t = (J_t / RV_t) + gamma_z * max(0.0, Z_{J, t} - z_crit)
      Psi_t^{jump} = exp( -lambda * JumpMetric_t ) in (0, 1].

    - Veto de seguridad dura (Hard Veto):
      Si rho_J = J_t / RV_t >= theta_hard_veto (def: 0.55):
      Psi_t^{jump} = 0.0 y gate_passed = False.

    Retorna: (jump_ratio rho_J, JumpMetric, Psi_t^{jump}, gate_passed)
    """
    if rv_daily <= 1e-14 or jump <= 0.0 or not is_significant:
        return 0.0, 0.0, 1.0, True

    rho_j = float(jump / rv_daily)
    # Acotar ratio en [0, 1)
    rho_j = max(0.0, min(rho_j, 0.9999))

    # Comprobación de veto por salto catastrófico
    if rho_j >= theta_hard_veto:
        return rho_j, rho_j, 0.0, False

    # Métrica de salto compuesta (proporción de salto + exceso de significancia opcional)
    z_excess = max(0.0, z_stat - z_crit) if gamma_z > 0 else 0.0
    jump_metric = rho_j + gamma_z * z_excess

    # Penalización exponencial continua
    lambda_safe = max(0.0, float(lambda_jump))
    exponent = -lambda_safe * jump_metric
    exponent_clamped = max(-20.0, min(exponent, 0.0))
    psi_jump = float(math.exp(exponent_clamped))

    # Umbral mínimo de viabilidad: si psi es menor a 0.05, desactiva la entrada
    gate_passed = bool(psi_jump >= 0.05)
    return rho_j, jump_metric, psi_jump, gate_passed


def calculate_position_sizing_opt_n1(
    sigma_pts: float,
    contract_symbol: str,
    account: PropAccount,
    jump_penalty_factor: float = 1.0,
    max_contracts: int = 5,
    z_score: float = 2.326,
    psi_risk: float = 0.25,
    gamma_t: float = 0.08,
    kappa_sl: float = 1.25
) -> Tuple[int, bool, float, float]:
    """
    Dimensionamiento cuantitativo de posición modulado por penalización de salto:
    Budget_t^{eff} = Budget_t * Psi_t^{jump}
    donde Budget_t = phi_streak(k) * min(psi_risk * Delta DLL_t, gamma_t * D_t^{cushion})

    Garantiza:
    - Modulación continua proporcional con Psi_t^{jump} in (0, 1].
    - Conmutación automática de estándar a microcontratos.
    - Prevención de bloqueo por cuantización discreta cuando el presupuesto cubre el riesgo
      real de stop-loss de 1 microcontrato.

    Retorna: (num_contratos, es_micro, presupuesto_base_usd, presupuesto_efectivo_usd)
    """
    if account.is_breached or sigma_pts <= 0 or jump_penalty_factor <= 0.0:
        return 0, False, 0.0, 0.0

    raw_budget = account.calculate_risk_budget(psi_risk=psi_risk, gamma_t=gamma_t)
    effective_budget = max(0.0, raw_budget * jump_penalty_factor)

    if effective_budget <= 0.0:
        return 0, False, raw_budget, 0.0

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
    unit_sl_risk_micro = kappa_sl * sigma_pts * m_micro
    max_micros = 10 * max_contracts

    # 1. Solicitud directa de microcontrato
    if is_explicit_micro:
        if unit_var_micro <= 0:
            return 0, True, raw_budget, effective_budget
        n_micro = int(math.floor(effective_budget / unit_var_micro))
        min_dollar_thresh = 15.0 if "ES" in micro_sym else 10.0
        if n_micro == 0 and effective_budget >= unit_sl_risk_micro and effective_budget >= min_dollar_thresh:
            n_micro = 1
        n_micro = min(n_micro, max_micros)
        return (n_micro, True, raw_budget, effective_budget) if n_micro >= 1 else (0, True, raw_budget, effective_budget)

    # 2. Contrato estándar solicitado
    spec_std = get_spec(std_sym)
    m_std = spec_std["multiplier"]
    unit_var_std = z_score * sigma_pts * m_std
    if unit_var_std <= 0:
        return 0, False, raw_budget, effective_budget

    n_std = int(math.floor(effective_budget / unit_var_std))
    n_std = min(n_std, max_contracts)

    if n_std >= 1:
        return n_std, False, raw_budget, effective_budget

    # 3. Conmutación automática a microcontratos si el estándar excede el presupuesto efectivo
    if unit_var_micro <= 0:
        return 0, True, raw_budget, effective_budget

    n_micro = int(math.floor(effective_budget / unit_var_micro))
    min_dollar_thresh = 15.0 if "ES" in micro_sym else 10.0
    if n_micro == 0 and effective_budget >= unit_sl_risk_micro and effective_budget >= min_dollar_thresh:
        n_micro = 1

    n_micro = min(n_micro, max_micros)

    if n_micro >= 1:
        return n_micro, True, raw_budget, effective_budget
    else:
        return 0, True, raw_budget, effective_budget


def calculate_opening_15m_metrics(
    opening_bars: pd.DataFrame
) -> Tuple[float, float, float, int]:
    """
    Calcula las métricas de la subasta de apertura regular (e.g. 09:30 - 09:45 EST):
    - Realized Variance de apertura RV_{15m} sobre retornos telescópicos de las 3 barras de 5m
    - Retorno acumulado r_{open} = ln(P_{end} / P_{start})
    - Desviación típica del retorno de apertura y umbral de significancia theta_thresh
    - Señal direccional condicionada por umbral y confirmación de VWAP_{15m}:
        direction = +1 si r_{open} > theta_thresh y P_{end} >= VWAP_{15m}
                    -1 si r_{open} < -theta_thresh y P_{end} <= VWAP_{15m}
                     0 de lo contrario (indecisión de apertura / rango)
    """
    if len(opening_bars) == 0:
        return 0.0, 0.0, 0.0, 0

    closes = opening_bars["Close"].values
    opens = opening_bars["Open"].values

    p_first_open = float(opens[0])
    p_last_close = float(closes[-1])

    if p_first_open <= 0 or p_last_close <= 0:
        return 0.0, 0.0, 0.0, 0

    r_open = float(math.log(p_last_close / p_first_open))

    # Retornos logarítmicos continuos incluyendo la primera barra
    prices = np.insert(closes, 0, p_first_open)
    r_intra = np.diff(np.log(prices))
    rv_obs = float(np.sum(r_intra ** 2)) if len(r_intra) > 0 else 0.0

    # Umbral de significancia de apertura (0.50 * sigma intrabarra)
    m_bars = max(len(r_intra), 1)
    sigma_bar = math.sqrt(rv_obs / m_bars) if rv_obs > 0 else 0.0
    theta_thresh = 0.50 * sigma_bar

    # Cálculo de VWAP de apertura de 15m
    if "Volume" in opening_bars.columns and opening_bars["Volume"].sum() > 0:
        vols = opening_bars["Volume"].values
        if "High" in opening_bars.columns and "Low" in opening_bars.columns:
            tp = (opening_bars["High"].values + opening_bars["Low"].values + closes) / 3.0
        else:
            tp = closes
        vwap_15m = float(np.sum(tp * vols) / np.sum(vols))
    else:
        vwap_15m = float(np.mean(closes))

    if r_open > theta_thresh and p_last_close >= vwap_15m:
        direction = 1
    elif r_open < -theta_thresh and p_last_close <= vwap_15m:
        direction = -1
    else:
        direction = 0

    return rv_obs, r_open, theta_thresh, direction


# ==============================================================================
# 3. CLASE ESTRATÉGICA F8-OPTIMIZACION -IS- N1
# ==============================================================================

class EconometricHARRVJOptN1Strategy:
    """
    Implementación formal de la Estrategia Optimizada F8-Optimizacion -IS- N1:
    - Descomposición analítica BV_t y J_t bajo Z_J > z_alpha.
    - Atenuación exponencial de apalancamiento Psi_t^{jump} = exp(-lambda * JumpMetric).
    - Desacoplamiento de la apertura RTH (observación de los primeros 15 min).
    - Escalado dinámico de barreras asimétricas (kappa_tp, kappa_sl).
    - Manejo integral de microestructura CME y control de riesgo Prop Trading.
    """

    def __init__(
        self,
        symbol: str = "ES",
        kappa_tp: float = 2.25,
        kappa_sl: float = 1.25,
        lambda_jump: float = 4.0,
        theta_hard_veto: float = 0.55,
        alpha_jump: float = 0.001,
        gamma_z: float = 0.0,
        entry_delay_bars: int = 3,       # 3 barras de 5m = 15m (entrada 09:45 EST)
        use_opening_rv: bool = True,     # Amortiguador compuesto de apertura
        opening_damp_factor: float = 0.60,
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
        self.lambda_jump = lambda_jump
        self.theta_hard_veto = theta_hard_veto
        self.alpha_jump = alpha_jump
        self.gamma_z = gamma_z
        self.entry_delay_bars = entry_delay_bars
        self.use_opening_rv = use_opening_rv
        self.opening_damp_factor = opening_damp_factor
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
    ) -> Optional[HARStateVectorOptN1]:
        """
        Procesa la sesión RTH de t al cierre (16:00 EST) y genera el pronóstico para t+1,
        descomponiendo la varianza continua y evaluando la penalización exponencial de saltos.
        """
        if intraday_bars.empty:
            return None

        # Filtrar sesión regular RTH
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
        returns = calculate_log_returns(closes)
        if len(returns) == 0:
            return None

        # 1. Descomposición matemática rigurosa entre Bipower Variation y Saltos
        rv, bv, tq, z_stat, jump, continuous, is_significant = calculate_continuous_jump_decomposition(
            returns, alpha_j=self.alpha_jump
        )

        # 2. Varianza heterogénea previa (semanal y mensual)
        hist_rvs = [h["rv_daily"] for h in self.history] + [rv]
        rv_w, rv_m = calculate_heterogeneous_rv(hist_rvs)

        # 3. Pronóstico Log-HAR-RV-J insesgado para t+1
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

        # 4. Función de penalización exponencial sobre apalancamiento
        rho_j, jump_metric, psi_jump, gate_passed = calculate_jump_exponential_penalty(
            jump=jump,
            rv_daily=rv,
            z_stat=z_stat,
            is_significant=is_significant,
            lambda_jump=self.lambda_jump,
            gamma_z=self.gamma_z,
            theta_hard_veto=self.theta_hard_veto
        )

        w_vol = calculate_volatility_managed_weight(sigma_pts, self.symbol)
        w_eff = w_vol * psi_jump

        state = HARStateVectorOptN1(
            date=date_str,
            rv_daily=rv,
            bv=bv,
            jump=jump,
            continuous=continuous,
            rv_weekly=rv_w,
            rv_monthly=rv_m,
            rv_predicted=rv_pred,
            sigma_predicted_pts=sigma_pts,
            z_jump_stat=z_stat,
            is_jump_significant=is_significant,
            jump_ratio=rho_j,
            jump_metric=jump_metric,
            jump_penalty_factor=psi_jump,
            volatility_managed_weight=w_vol,
            effective_weight=w_eff,
            jump_gate_passed=gate_passed
        )

        # Registrar en el historial
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
        prior_state: Optional[HARStateVectorOptN1],
        account: PropAccount
    ) -> Optional[HARRVJOptN1TradeRecord]:
        """
        Ejecuta la simulación intradía barra a barra de la sesión t+1.
        Incorpora desacoplamiento de apertura, modulación exponencial de tamaño,
        barreras asimétricas ampliadas y reglas de salida deterministas.
        """
        if prior_state is None or day_bars.empty or account.is_breached:
            return None

        # 1. Comprobar compuerta de salto y penalización
        if not prior_state.jump_gate_passed or prior_state.jump_penalty_factor <= 0.0:
            return None

        date_str = str(session_date)
        day_bars = day_bars.sort_index()

        times = day_bars.index.time
        open_time = self.session_open_time
        close_time = self.session_close_time

        rth_bars = day_bars[(times >= open_time) & (times <= close_time)]
        # Requiere al menos el número de barras de observación + 2 barras para operar
        min_bars_needed = self.entry_delay_bars + 2
        if len(rth_bars) < min_bars_needed:
            return None

        # 2. Desacoplamiento de apertura: observación de las primeras barras
        obs_bars = rth_bars.iloc[:self.entry_delay_bars]
        rv_15m, r_open, theta_thresh, direction = calculate_opening_15m_metrics(obs_bars)

        # Si el sesgo direccional de apertura es nulo (indecisión / rango estrecho) -> no operar
        if direction == 0:
            return None

        # 3. Proyección de volatilidad compuesta (incorporando amortiguador de apertura)
        sigma_pts = prior_state.sigma_predicted_pts
        if self.use_opening_rv and rv_15m > 0:
            # Varianza compuesta escalada: max(RV_pred, 26 * RV_15m * damp)
            comp_rv = max(prior_state.rv_predicted, 26.0 * rv_15m * self.opening_damp_factor)
            p_entry_bar = float(rth_bars.iloc[self.entry_delay_bars]["Open"])
            sigma_pts_comp = calculate_point_volatility(comp_rv, p_entry_bar)
            sigma_pts = max(sigma_pts, sigma_pts_comp)

        if sigma_pts <= 0:
            return None

        # 4. Dimensionamiento con penalización exponencial
        contracts, is_micro, raw_budget, eff_budget = calculate_position_sizing_opt_n1(
            sigma_pts=sigma_pts,
            contract_symbol=self.symbol,
            account=account,
            jump_penalty_factor=prior_state.jump_penalty_factor,
            max_contracts=self.max_contracts,
            kappa_sl=self.kappa_sl
        )

        if contracts <= 0:
            return None

        # Determinar especificación
        active_spec = self.spec_micro if is_micro else self.spec_std
        active_symbol = self.micro_sym if is_micro else self.std_sym
        multiplier = active_spec["multiplier"]
        commission_rt = active_spec["commission_rt"]
        tick_size = active_spec["tick"]

        # 5. Entrada en la barra inmediatamente posterior al periodo de observación
        entry_bar = rth_bars.iloc[self.entry_delay_bars]
        entry_time_str = str(entry_bar.name)
        entry_price = quantize_price(float(entry_bar["Open"]), tick_size)

        # Costos transaccionales
        slippage_pts = self.slippage_ticks * tick_size
        slippage_usd = slippage_pts * multiplier * contracts
        commissions_usd = commission_rt * contracts

        # 6. Barreras dinámicas asimétricas ampliadas
        tp_price, sl_price = calculate_dynamic_barriers(
            entry_price=entry_price,
            direction=direction,
            sigma_pts=sigma_pts,
            kappa_tp=self.kappa_tp,
            kappa_sl=self.kappa_sl,
            tick_size=tick_size
        )

        trade = HARRVJOptN1TradeRecord(
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
            slippage_usd=slippage_usd,
            jump_stat=prior_state.z_jump_stat,
            jump_metric=prior_state.jump_metric,
            jump_penalty_factor=prior_state.jump_penalty_factor,
            effective_budget_usd=eff_budget,
            opening_rv_15m=rv_15m
        )

        # 7. Simulación barra a barra post-entrada
        curr_intra_rv = rv_15m
        prev_close = float(entry_bar["Open"])

        exit_time = None
        exit_price = None
        exit_reason = None

        bars_after_entry = rth_bars.iloc[self.entry_delay_bars:]
        for bar_dt, bar in bars_after_entry.iterrows():
            curr_bar_time = bar_dt.time()
            b_open = float(bar["Open"])
            high = float(bar["High"])
            low = float(bar["Low"])
            close = float(bar["Close"])

            # Acumular varianza realizada intradía
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

            # B. Evaluación de Barreras Dinámicas (Prioridad al Stop-Loss en Prop Trading)
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

        # Si no hubo salida por barrera o disyuntor, liquidar al último cierre
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
# 4. MOTOR DE BACKTESTING INTEGRADO (HARRVJOptN1Backtester)
# ==============================================================================

class HARRVJOptN1Backtester:
    """
    Motor de backtest para F8-Optimizacion -IS- N1.
    Garantiza causalidad temporal estricta (Lookahead Bias = 0), registra
    métricas de microestructura CME y de cumplimiento Prop Trading.
    """

    def __init__(
        self,
        strategy: EconometricHARRVJOptN1Strategy,
        account: PropAccount
    ):
        self.strategy = strategy
        self.account = account
        self.trades: List[HARRVJOptN1TradeRecord] = []
        self.daily_states: Dict[str, HARStateVectorOptN1] = {}
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

        prior_state: Optional[HARStateVectorOptN1] = None

        for d in unique_dates:
            day_mask = (df.index.date == d)
            day_bars = df[day_mask]

            if self.account.is_breached:
                self.daily_equity_curve[str(d)] = self.account.equity
                continue

            # Verificar si contiene barras RTH
            if isinstance(day_bars.index, pd.DatetimeIndex):
                times = day_bars.index.time
                has_rth = bool(np.any((times >= self.strategy.session_open_time) & (times <= time(16, 0))))
                if not has_rth:
                    continue

            self.account.start_session(d)

            # 1. Simular sesión de t usando el estado de t-1
            if prior_state is not None:
                trade = self.strategy.simulate_day(
                    session_date=d,
                    day_bars=day_bars,
                    prior_state=prior_state,
                    account=self.account
                )
                if trade:
                    self.trades.append(trade)

            # 2. Procesar datos al cierre de t para generar el estado de t+1
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
        """Genera el reporte cuantitativo completo y cumplimiento Prop Trading."""
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
            sortino = float((daily_returns.mean() / neg_ret.std()) * np.sqrt(252)) if (len(neg_ret) > 0 and neg_ret.std() > 0) else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        max_dd_usd = self.account.current_drawdown
        max_dd_pct = (max_dd_usd / self.account.peak_equity * 100.0) if self.account.peak_equity > 0 else 0.0
        calmar = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

        exit_counts = {}
        for r in HARExitReason:
            exit_counts[r.value] = sum(1 for t in self.trades if t.exit_reason == r)

        # Métricas de penalización de saltos
        jump_penalties = [t.jump_penalty_factor for t in self.trades]
        avg_jump_penalty = float(np.mean(jump_penalties)) if jump_penalties else 1.0
        penalized_trades = sum(1 for p in jump_penalties if p < 0.999)

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
            "avg_jump_penalty_factor": round(avg_jump_penalty, 4),
            "jump_penalized_trades_count": penalized_trades,
            "is_prop_breached": self.account.is_breached,
            "breach_reason": self.account.breach_reason or "None (Compliant)"
        }

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Exporta el log completo de operaciones."""
        if not self.trades:
            return pd.DataFrame()
        records = [t.__dict__ for t in self.trades]
        return pd.DataFrame(records)
