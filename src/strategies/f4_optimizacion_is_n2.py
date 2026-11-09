"""
src/strategies/f4_optimizacion_is_n2.py
Implementación Matemática Formal de F4-Optimizacion -IS- N2:
Cointegración Inter-Mercado ES vs NQ y Difusión Ornstein-Uhlenbeck (StatArb OU).

Corrige punto por punto las 6 anomalías técnicas detectadas en la auditoría:
1. Parada Prematura (Early Stopping Bias): Opción early_stopping=False para evaluación
   completa del año (260 sesiones / 20,280 barras).
2. Anualización del Ratio de Sharpe: Anualización sobre retornos diarios (252 días continuos)
   en lugar de multiplicar la desviación por trade por sqrt(252).
3. Bug del "Objetivo Móvil" (Moving-Target S-Score): Congela los parámetros de equilibrio
   (alpha_entry, beta_entry, theta_entry, sigma_eq_entry, tau_half_entry) al abrir la posición.
   El S-Score de salida se calcula estrictamente respecto al equilibrio congelado.
4. Fuga de Ejecución Contemporánea en Close[t]: Ejecución estrictamente no anticipativa
   en la apertura de la barra siguiente (Open[t+1]) con deslizamiento realista y comisiones CME.
5. Desbalance de Cobertura Dollar-Beta: Asignación óptima en retículo entero coprimo
   gcd(N_NQ, N_ES) = 1 con minimización del residuo direccional y sub-cestas micro balanceadas.
6. Calibración Libre de Saltos Nocturnos y Filtro de Kalman: Filtrado estricto de parejas intradía
   D_intraday en AR(1) y soporte de Filtro de Kalman recursivo para el beta dinámico.

Referencia de Especificación:
reports/optimizaciones_estrategias/F4_optimizacion_cointegracion_ou.md
"""

from dataclasses import dataclass, field
from datetime import time, datetime
from enum import Enum
import math
from typing import Dict, Any, List, Optional, Tuple, Sequence
import warnings
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts

from src.core.base_strategy import BaseStrategy
from src.core.futures_specs import get_spec
from src.strategies.f4_optimizacion_is_n1 import compute_sector_correlation_series

warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# 1. ENUMERACIONES Y ESTRUCTURAS DE DATOS (F4-OPTIMIZACION -IS- N1)
# ==============================================================================

class TradeSignalN2(int, Enum):
    SELL_SPREAD = -1   # Corto NQ / Largo ES (S-score > +s_open)
    NEUTRAL = 0        # Fuera de mercado / Zona de indiferencia
    BUY_SPREAD = 1     # Largo NQ / Corto ES (S-score < -s_open)


class ExitReasonN2(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"           # Reversión a la media (|S_frozen| <= s_close)
    STOP_LOSS = "STOP_LOSS"               # Ruptura estructural (|S_frozen| >= s_stop)
    TIME_DECAY = "TIME_DECAY"             # Decaimiento temporal (tiempo >= multiplier * tau_entry)
    COINT_DEGRADED = "COINT_DEGRADED"     # Descalificación de estacionariedad (p-val > 0.10 x 3 barras)
    EOD_FLAT = "EOD_FLAT"                 # Cierre forzoso intradía MOC (>= 15:50 EST)
    RISK_BREACH = "RISK_BREACH"           # Violación de Daily Loss Limit o Trailing DD


@dataclass
class TradeRecordN2:
    """Registro detallado de una operación de F4-Optimizacion -IS- N2."""
    trade_id: int
    signal_date: str
    signal_time: str
    entry_date: str
    entry_time: str
    signal: str                        # "BUY_SPREAD" o "SELL_SPREAD"
    direction: int                     # +1 = Long NQ / Short ES, -1 = Short NQ / Long ES
    contracts_nq: int                  # Contratos NQ / MNQ
    contracts_es: int                  # Contratos ES / MES
    symbol_nq: str
    symbol_es: str
    n_baskets: int
    entry_price_nq: float              # Ejecutado en Open[t+1] con slippage
    entry_price_es: float              # Ejecutado en Open[t+1] con slippage
    signal_s_score: float              # S-Score en Close[t]
    # Parámetros de equilibrio CONGELADOS al abrir la operación (Punto 3)
    entry_alpha: float
    entry_beta: float
    entry_theta: float
    entry_sigma_eq: float
    entry_tau_half: float              # Vida media en minutos congelada
    coprime_ratio: float = 0.0         # N_ES / N_NQ
    target_dollar_ratio: float = 0.0   # Ratio teórico R*
    dollar_imbalance_usd: float = 0.0  # Desbalance nocional residual
    exit_date: Optional[str] = None
    exit_time: Optional[str] = None
    exit_price_nq: Optional[float] = None
    exit_price_es: Optional[float] = None
    exit_s_score_frozen: Optional[float] = None
    exit_reason: Optional[ExitReasonN2] = None
    bars_held: int = 0
    duration_minutes: float = 0.0
    gross_pnl_usd: float = 0.0
    commissions_usd: float = 0.0
    slippage_usd: float = 0.0
    legging_usd: float = 0.0
    friction_usd: float = 0.0
    net_pnl_usd: float = 0.0
    running_balance: float = 0.0
    running_drawdown: float = 0.0


# ==============================================================================
# 2. FILTRO DE KALMAN RECURSIVO PARA BETA DINÁMICO (SECCIÓN 3.6)
# ==============================================================================

class KalmanFilterBeta:
    """
    Filtro de Kalman recursivo para estimar la elasticidad dinámica (beta_K) y alpha
    entre log-precios sin rezago de ventana ni discontinuidades bruscas:
        p_NQ(t) = alpha(t) + beta(t) * p_ES(t) + v(t),  v(t) ~ N(0, R)
        theta(t) = theta(t-1) + w(t),                   w(t) ~ N(0, Q)
    """

    def __init__(
        self,
        q_alpha: float = 1e-6,
        q_beta: float = 1e-5,
        r_measurement: float = 1e-4,
        initial_alpha: float = 0.0,
        initial_beta: float = 1.25,
        initial_cov: float = 1e-2,
    ):
        self.q = np.array([[q_alpha, 0.0], [0.0, q_beta]], dtype=np.float64)
        self.r = float(r_measurement)
        self.state = np.array([initial_alpha, initial_beta], dtype=np.float64)
        self.cov = np.eye(2, dtype=np.float64) * initial_cov

    def initialize_from_ols(self, alpha: float, beta: float, cov_scale: float = 1e-3):
        """Inicializa el estado del filtro con estimadores OLS de calentamiento."""
        self.state = np.array([alpha, beta], dtype=np.float64)
        self.cov = np.eye(2, dtype=np.float64) * cov_scale

    def update(self, p_nq: float, p_es: float) -> Tuple[float, float]:
        """
        Paso recursivo de predicción y corrección barra a barra O(1).
        Retorna (alpha_k, beta_k).
        """
        # 1. Predicción de estado y covarianza
        cov_prior = self.cov + self.q

        # 2. Vector de observación H_t = [1.0, p_es]
        h = np.array([1.0, p_es], dtype=np.float64)

        # 3. Innovación (error de predicción)
        y = p_nq - float(np.dot(h, self.state))

        # 4. Varianza de la innovación S_t = H * P * H^T + R
        s = float(np.dot(h, np.dot(cov_prior, h))) + self.r
        if s <= 1e-12:
            return float(self.state[0]), float(self.state[1])

        # 5. Ganancia de Kalman K_t = P * H^T / S_t
        k = np.dot(cov_prior, h) / s

        # 6. Actualización de estado y covarianza
        self.state = self.state + k * y
        self.cov = cov_prior - np.outer(k, np.dot(h, cov_prior))

        return float(self.state[0]), float(self.state[1])


# ==============================================================================
# 3. MODELO MATEMÁTICO FORMAL OPTIMIZADO (F4-OPTIMIZACION -IS- N1)
# ==============================================================================

class CointegrationOUModelN2:
    """
    Núcleo analítico de F4-Optimizacion -IS- N2.
    Implementa:
    - OLS Engle-Granger hacia atrás estricto (Sección 3.2).
    - ADF con valor crítico de MacKinnon -3.37 (Sección 3.2).
    - AR(1) sobre D_intraday excluyendo saltos nocturnos de 17.5h (Sección 3.3).
    - Fórmulas continuas exactas OU: kappa, theta, sigma_eq, tau_1/2.
    - S-Score estandarizado y S-Score congelado durante trade abierto (Sección 3.4).
    - Optimización en retículo coprimo gcd(N_NQ, N_ES)=1 para neutralidad Dollar-Beta (Sección 3.6).
    - Presupuesto de riesgo con disyuntor de racha geometrica (Sección 4.2).
    """

    @staticmethod
    def estimate_cointegration_backward(
        p_nq: np.ndarray,
        p_es: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estima la relación OLS histórica sobre la ventana hacia atrás:
        p_NQ = alpha + beta * p_ES + X
        """
        if len(p_nq) != len(p_es):
            raise ValueError("Las series de log-precios deben tener igual longitud.")
        if len(p_nq) < 10:
            raise ValueError("Se requieren al menos 10 observaciones.")

        var_es = np.var(p_es, ddof=1)
        if var_es <= 1e-12:
            return 1.0, 0.0, np.zeros_like(p_nq)

        cov_matrix = np.cov(p_nq, p_es, ddof=1)
        cov_nq_es = cov_matrix[0, 1]
        beta_hedge = float(cov_nq_es / var_es)
        alpha_coint = float(np.mean(p_nq) - beta_hedge * np.mean(p_es))
        residuals = p_nq - (alpha_coint + beta_hedge * p_es)

        return beta_hedge, alpha_coint, residuals

    @staticmethod
    def test_stationarity_mackinnon(
        residuals: np.ndarray,
        t_stat_threshold: float = -3.37,
        p_val_threshold: float = 0.05,
        maxlag: int = 1,
        autolag: Optional[str] = None
    ) -> Tuple[bool, float, float]:
        """
        Test ADF de cointegración con umbral asintótico de MacKinnon para N=2 variables (-3.37).
        Usa maxlag=1 por defecto (acorde a la formulación Engle-Granger) garantizando alta velocidad.
        """
        if len(residuals) < 20:
            return False, 0.0, 1.0

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_result = ts.adfuller(residuals, maxlag=maxlag, autolag=autolag)
            t_stat = float(adf_result[0])
            p_val = float(adf_result[1])
            is_stationary = (t_stat < t_stat_threshold) and (p_val < p_val_threshold)
            return bool(is_stationary), t_stat, p_val
        except Exception:
            return False, 0.0, 1.0

    @staticmethod
    def fit_ou_process_intraday(
        residuals: np.ndarray,
        timestamps: Sequence[datetime],
        dt_minutes: float = 5.0,
        sigma_floor: float = 0.001
    ) -> Dict[str, float]:
        """
        Ajuste analítico AR(1) de Ornstein-Uhlenbeck filtrando saltos nocturnos (Sección 3.3).
        Excluye las transiciones donde timestamps[k] y timestamps[k-1] pertenecen a distintos días.
        Aplica piso de volatilidad sigma_floor (R3).
        """
        n = len(residuals)
        if n < 10 or len(timestamps) != n:
            return {
                "b": 0.0, "a": 0.0, "kappa": 0.0, "theta": 0.0,
                "sigma_eta": 0.0, "sigma_eq": 0.0, "sigma": 0.0,
                "tau_half_min": 0.0, "is_valid": False, "intraday_pairs": 0
            }

        # Construir D_intraday seleccionando únicamente pares de la misma fecha
        if isinstance(timestamps, pd.DatetimeIndex):
            same_day = (timestamps.dayofyear.values[1:] == timestamps.dayofyear.values[:-1]) & (timestamps.year.values[1:] == timestamps.year.values[:-1])
        elif hasattr(timestamps, "values") and np.issubdtype(np.asarray(timestamps).dtype, np.integer):
            t_vals = np.asarray(timestamps)
            same_day = (t_vals[1:] == t_vals[:-1])
        else:
            try:
                dt_idx = pd.DatetimeIndex(timestamps)
                same_day = (dt_idx.dayofyear.values[1:] == dt_idx.dayofyear.values[:-1]) & (dt_idx.year.values[1:] == dt_idx.year.values[:-1])
            except Exception:
                dates = [getattr(t, "date", lambda: t)() for t in timestamps]
                same_day = np.array([dates[k] == dates[k - 1] for k in range(1, len(dates))], dtype=bool)

        res_arr = np.asarray(residuals, dtype=np.float64)
        x_prev = res_arr[:-1][same_day]
        x_curr = res_arr[1:][same_day]

        if len(x_prev) < 10:
            return {
                "b": 0.0, "a": 0.0, "kappa": 0.0, "theta": 0.0,
                "sigma_eta": 0.0, "sigma_eq": 0.0, "sigma": 0.0,
                "tau_half_min": 0.0, "is_valid": False, "intraday_pairs": len(x_prev)
            }

        var_prev = np.var(x_prev, ddof=1)
        if var_prev <= 1e-14:
            return {
                "b": 0.0, "a": 0.0, "kappa": 0.0, "theta": 0.0,
                "sigma_eta": 0.0, "sigma_eq": 0.0, "sigma": 0.0,
                "tau_half_min": 0.0, "is_valid": False, "intraday_pairs": len(x_prev)
            }

        cov_matrix = np.cov(x_curr, x_prev, ddof=1)
        cov_curr_prev = cov_matrix[0, 1]
        b = float(cov_curr_prev / var_prev)
        a = float(np.mean(x_curr) - b * np.mean(x_prev))

        # Restricción de ergodicidad estricta (0 < b < 0.9995)
        if b <= 0.0 or b >= 0.9995:
            return {
                "b": b, "a": a, "kappa": 0.0, "theta": 0.0,
                "sigma_eta": 0.0, "sigma_eq": 0.0, "sigma": 0.0,
                "tau_half_min": 0.0, "is_valid": False, "intraday_pairs": len(x_prev)
            }

        eta = x_curr - (a + b * x_prev)
        sigma_eta_sq = float(np.var(eta, ddof=2)) if len(eta) > 2 else float(np.var(eta))
        if sigma_eta_sq <= 1e-14:
            sigma_eta_sq = 1e-14

        # Velocidad de reversión continua en [minutos^-1]
        kappa = float(-np.log(b) / dt_minutes)
        theta = float(a / (1.0 - b))
        sigma_eq_sq = float(sigma_eta_sq / (1.0 - b**2))
        sigma_eq_raw = float(np.sqrt(max(1e-14, sigma_eq_sq)))
        sigma_eq = float(max(sigma_eq_raw, sigma_floor))
        sigma = float(sigma_eq * np.sqrt(2.0 * kappa))
        tau_half_min = float(np.log(2.0) / kappa)

        return {
            "b": b,
            "a": a,
            "kappa": kappa,
            "kappa_daily": float(kappa * 390.0),
            "theta": theta,
            "sigma_eta": float(np.sqrt(sigma_eta_sq)),
            "sigma_eq": sigma_eq,
            "sigma": sigma,
            "tau_half_min": tau_half_min,
            "is_valid": True,
            "intraday_pairs": len(x_prev),
        }

    @staticmethod
    def calculate_s_score(x_k: float, theta: float, sigma_eq: float, sigma_floor: float = 0.001) -> float:
        """S-score estandarizado con piso de volatilidad: S_k = (X_k - theta) / eff_sigma."""
        eff_sigma = max(sigma_eq, sigma_floor)
        if eff_sigma <= 1e-12:
            return 0.0
        return float((x_k - theta) / eff_sigma)

    @staticmethod
    def calculate_frozen_s_score(
        p_nq: float,
        p_es: float,
        entry_alpha: float,
        entry_beta: float,
        entry_theta: float,
        entry_sigma_eq: float,
        sigma_floor: float = 0.001
    ) -> float:
        """
        S-score calculado estrictamente contra los parámetros de equilibrio CONGELADOS
        al momento de la entrada (Punto 3 - Erradicación de Moving-Target S-Score) con piso de volatilidad (R3).
        """
        eff_sigma = max(entry_sigma_eq, sigma_floor)
        if eff_sigma <= 1e-12 or p_nq <= 0 or p_es <= 0:
            return 0.0
        log_nq = float(np.log(p_nq))
        log_es = float(np.log(p_es))
        res_frozen = log_nq - (entry_alpha + entry_beta * log_es)
        return float((res_frozen - entry_theta) / eff_sigma)

    @staticmethod
    def calculate_dollar_beta_ratio(
        p_nq: float,
        p_es: float,
        beta_hedge: float,
        m_nq: float = 2.0,
        m_es: float = 5.0
    ) -> float:
        """Ratio analítico Dollar-Beta: R* = beta * ((P_NQ * M_NQ) / (P_ES * M_ES))."""
        if p_es <= 0 or m_es <= 0:
            return 1.0
        return float(beta_hedge * ((p_nq * m_nq) / (p_es * m_es)))

    @staticmethod
    def find_optimal_coprime_basket(
        target_ratio: float,
        max_nq_contracts: int = 10,
        prefer_sub_baskets: bool = True
    ) -> Tuple[int, int, float]:
        """
        Búsqueda en retículo entero coprimo:
        min |(N_ES / N_NQ) - R*| sujeto a gcd(N_NQ, N_ES) = 1 (Sección 3.6).
        
        Retorna:
            (N_NQ, N_ES, error_relativo)
        """
        if target_ratio <= 0.0:
            return 1, 1, 0.0

        best_tuple = (1, max(1, int(round(target_ratio))))
        best_err = abs((best_tuple[1] / best_tuple[0]) - target_ratio) / target_ratio

        # Si prefer_sub_baskets está activo para cuentas micro, probar n_nq desde 1 hasta max
        for n_nq in range(1, max_nq_contracts + 1):
            n_es_candidates = [
                int(math.floor(n_nq * target_ratio)),
                int(math.ceil(n_nq * target_ratio)),
            ]
            for n_es in n_es_candidates:
                if n_es <= 0:
                    continue
                if math.gcd(n_nq, n_es) == 1:
                    ratio = n_es / n_nq
                    err = abs(ratio - target_ratio) / target_ratio
                    if err < best_err:
                        best_err = err
                        best_tuple = (n_nq, n_es)

        return best_tuple[0], best_tuple[1], float(best_err)

    @staticmethod
    def calculate_risk_budget_attenuated(
        dll: float,
        tmdd: float,
        realized_loss_today: float = 0.0,
        current_drawdown: float = 0.0,
        loss_streak: int = 0,
        psi_tmdd: float = 0.20,
        phi_dll: float = 0.35,
        survival_floor: float = 0.0
    ) -> float:
        """
        Presupuesto de riesgo con atenuación geométrica por racha perdedora y suelo de supervivencia (Sección 4.2):
        phi(K_loss) = phi_dll * 2^(-min(K_loss, 3))
        B_risk = min( phi(K_loss) * (DLL - L_daily), psi * (TMDD - DD) )
        """
        daily_cushion = max(0.0, dll - realized_loss_today)
        tmdd_cushion = max(0.0, tmdd - current_drawdown)

        streak_exp = min(max(0, loss_streak), 3)
        phi = phi_dll * (2.0 ** (-streak_exp))

        budget_daily = phi * daily_cushion
        budget_tmdd = psi_tmdd * tmdd_cushion
        raw_budget = min(budget_daily, budget_tmdd)

        # Suelo operativo de supervivencia ($105 USD) si existe margen regulatorio mínimo:
        if survival_floor > 0.0 and daily_cushion >= 150.0 and tmdd_cushion >= 250.0:
            return float(max(survival_floor, raw_budget))

        return float(max(0.0, raw_budget))

    @staticmethod
    def calculate_unit_stop_loss_loss(
        sigma_eq: float,
        p_nq: float,
        n_nq_unit: int,
        m_nq: float,
        friction_basket_usd: float,
        delta_s_sl: float = 2.00,
        sigma_floor: float = 0.001
    ) -> float:
        """Pérdida monetaria máxima esperada al tocar el Stop-Loss estructural con piso de volatilidad (R3)."""
        eff_sigma = max(sigma_eq, sigma_floor)
        delta_x_sl = delta_s_sl * eff_sigma
        notional_nq = n_nq_unit * m_nq * p_nq
        loss_pts = delta_x_sl * notional_nq
        return float(loss_pts + friction_basket_usd)

    @staticmethod
    def calculate_baskets_allocation(
        risk_budget_usd: float,
        unit_sl_loss_usd: float,
        n_basket_max: int = 3,
        min_survival_budget: float = 100.0
    ) -> int:
        """Función piso con disyuntor de volatilidad y suelo operativo de supervivencia."""
        if risk_budget_usd <= 0 or unit_sl_loss_usd <= 0:
            return 0
        raw = int(np.floor(risk_budget_usd / unit_sl_loss_usd))
        # Suelo operativo: si el presupuesto permite al menos la cesta mínima ($100-$105 USD), asignar al menos 1
        if raw == 0 and risk_budget_usd >= min_survival_budget:
            raw = 1
        return int(max(0, min(raw, n_basket_max)))


# ==============================================================================
# 4. ESTRATEGIA OPTIMIZADA (INTERFAZ BASESTRATEGY)
# ==============================================================================

class F4OptimizationISN2Strategy(BaseStrategy):
    """
    Estrategia Cuantitativa F4-Optimizacion -IS- N2:
    Cointegración Inter-Mercado ES vs NQ y Difusión Ornstein-Uhlenbeck.
    """

    def __init__(
        self,
        name: str = "F4-Optimizacion -IS- N2",
        params: Optional[Dict[str, Any]] = None,
    ):
        default_params: Dict[str, Any] = {
            # Instrumentos y contratos
            "symbol_nq": "NQ",
            "symbol_es": "ES",
            "micro_nq": "MNQ",
            "micro_es": "MES",
            "contract_mode": "micro",           # "micro" o "standard"
            "use_kalman": False,                # Filtro de Kalman para beta recursivo
            "unify_beta_ols": True,             # Unificar elasticidad con OLS rodante por defecto
            "max_coprime_nq": 7,                # Límite superior en retículo coprimo
            "allow_sub_baskets": True,          # Sub-cestas micro (1 MNQ vs exact MES)
            "dynamic_basket_ratio": True,       # Cobertura exacta Dollar-Beta
            # Ventana de calibración histórica hacia atrás
            "k_calib": 2340,                    # 30 sesiones RTH x 78 barras (Sec 3.2)
            "min_calib_bars": 390,              # Calibración mínima permitida (5 sesiones RTH)
            "dt_minutes": 5.0,                  # Barra de 5 minutos
            # Umbrales analíticos de Avellaneda & Lee (Sec 3.4 & 3.5)
            "s_open": 1.25,                     # Umbral de activación
            "s_close": 0.25,                    # Umbral de toma de ganancias (reversión a equilibrio)
            "s_stop": 3.25,                     # Umbral de Stop-Loss estructural
            # Filtro de vida media de reversión
            "tau_min": 15.0,                    # 15 minutos
            "tau_max": 390.0,                   # 390 minutos (1 sesión RTH completa)
            # Filtro ADF de cointegración con MacKinnon
            "adf_t_threshold": -3.37,           # Valor crítico MacKinnon 95% para N=2
            "adf_p_threshold": 0.05,
            # Control de degradación de estacionariedad
            "enable_degrade_exit": True,
            "degrade_p_threshold": 0.10,
            "degrade_consecutive_bars": 3,
            # Horarios RTH en America/New_York
            "rth_start": "09:30",
            "rth_end": "16:00",
            "entry_start": "10:00",
            "entry_end": "15:40",
            "eod_flat": "15:50",                # Cierre forzoso MOC a las 15:50 EST
            "time_decay_multiplier": 3.5,       # Multiplicador de decaimiento temporal
            # Control de riesgo para Prop Trading (Sec 4.2 & 4.4)
            "phi_dll": 0.35,                    # Base 35% del DLL (el preset 50k calibra a 0.60)
            "psi_tmdd": 0.20,                   # Base 20% del TMDD (el preset 50k calibra a 0.30)
            "survival_floor": 105.0,            # Suelo de supervivencia operativa ($105 USD)
            "n_basket_max": 3,                  # Máximo 3 sub-cestas micro
            "slippage_ticks": 1.0,              # 1 tick por leg = 2 ticks round-turn
            "legging_penalty_ticks": 1.0,       # Penalización asincrónica de orden multileg (Sec 3.7)
            # Filtros R3 y R4 (Piso de volatilidad y Desacoplamiento Sectorial)
            "sigma_floor": 0.001,               # Piso mínimo de volatilidad para sigma_eq (R3)
            "min_sector_corr": 0.85,            # Umbral mínimo de correlación XLK/SPY (R4)
            "sector_corr_window": 30,           # Ventana rodante en días de correlación sectorial
            # Zonas horarias
            "target_tz": "America/New_York",
            "source_tz": "America/Mexico_City",
        }
        if params:
            default_params.update(params)
        super().__init__(name=name, params=default_params)
        self.model = CointegrationOUModelN2()

    def ensure_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        target_tz = self.params["target_tz"]
        source_tz = self.params.get("source_tz")
        if df.index.tz is None:
            if source_tz:
                return df.tz_localize(source_tz).tz_convert(target_tz)
            else:
                return df.tz_localize(target_tz)
        else:
            return df.tz_convert(target_tz)

    def generate_signals(
        self,
        data: pd.DataFrame,
        sector_corr_series: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Generación de señales de entrada para F4-Optimizacion -IS- N2.
        Las señales se evalúan al cierre de la barra t, diseñadas para ser
        ejecutadas estrictamente en la apertura de la barra t+1.
        """
        if "NQ" in data.columns and "ES" in data.columns:
            nq_close = data["NQ"]
            es_close = data["ES"]
        elif "Close_NQ" in data.columns and "Close_ES" in data.columns:
            nq_close = data["Close_NQ"]
            es_close = data["Close_ES"]
        else:
            raise ValueError("Se requieren columnas de precios para NQ y ES.")

        sector_corr_lookup: Dict[Any, float] = {}
        if sector_corr_series is not None and not sector_corr_series.empty:
            sector_corr_lookup = dict(sector_corr_series)

        df_pair = pd.DataFrame({"NQ": nq_close, "ES": es_close}).dropna()
        df_pair = self.ensure_timezone(df_pair)
        rth_bars = df_pair.between_time(self.params["rth_start"], self.params["rth_end"])
        signals = pd.Series(0.0, index=df_pair.index)

        k_calib = self.params["k_calib"]
        min_calib = min(self.params["min_calib_bars"], k_calib)

        # Inicialización del Filtro de Kalman si está activo
        kf = KalmanFilterBeta() if self.params.get("use_kalman", True) else None
        kf_initialized = False

        current_signal = 0
        entry_idx = 0
        frozen_alpha = 0.0
        frozen_beta = 1.0
        frozen_theta = 0.0
        frozen_sigma_eq = 1.0
        frozen_tau = 0.0

        for i in range(len(rth_bars)):
            ts = rth_bars.index[i]
            p_nq_cur = float(rth_bars["NQ"].iloc[i])
            p_es_cur = float(rth_bars["ES"].iloc[i])
            bar_time_str = ts.strftime("%H:%M")

            if i < min_calib:
                continue

            win_start = max(0, i - k_calib)
            hist_nq = rth_bars["NQ"].iloc[win_start:i]
            hist_es = rth_bars["ES"].iloc[win_start:i]
            hist_ts = rth_bars.index[win_start:i]

            log_nq = np.log(hist_nq.values)
            log_es = np.log(hist_es.values)

            # 1. Cointegración hacia atrás OLS (residuos exactos para Engle-Granger / MacKinnon)
            beta_ols, alpha_ols, res_hist = self.model.estimate_cointegration_backward(log_nq, log_es)
            if kf is not None:
                if not kf_initialized:
                    kf.initialize_from_ols(alpha_ols, beta_ols)
                    kf_initialized = True
                else:
                    kf.update(float(np.log(p_nq_cur)), float(np.log(p_es_cur)))
                if not self.params.get("unify_beta_ols", False):
                    beta_used = float(kf.state[1])
                else:
                    beta_used = beta_ols
                alpha_used = alpha_ols
            else:
                alpha_used, beta_used = alpha_ols, beta_ols

            # 2. Ajuste OU libre de saltos nocturnos con piso de volatilidad (R3)
            ou = self.model.fit_ou_process_intraday(
                res_hist, hist_ts,
                dt_minutes=self.params["dt_minutes"],
                sigma_floor=self.params.get("sigma_floor", 0.001)
            )
            if not ou["is_valid"]:
                if current_signal != 0:
                    current_signal = 0
                signals.loc[ts] = 0.0
                continue

            # ------------------------------------------------------------------
            # EVALUACIÓN DE SALIDA CON EQUILIBRIO CONGELADO (PUNTO 3)
            # ------------------------------------------------------------------
            if current_signal != 0:
                elapsed_min = (i - entry_idx) * self.params["dt_minutes"]
                s_score_frozen = self.model.calculate_frozen_s_score(
                    p_nq=p_nq_cur, p_es=p_es_cur,
                    entry_alpha=frozen_alpha, entry_beta=frozen_beta,
                    entry_theta=frozen_theta, entry_sigma_eq=frozen_sigma_eq,
                    sigma_floor=self.params.get("sigma_floor", 0.001)
                )
                should_exit = False

                # Take-Profit: reversión al equilibrio congelado (|S_frozen| <= s_close)
                if current_signal == -1 and s_score_frozen <= self.params["s_close"]:
                    should_exit = True
                elif current_signal == 1 and s_score_frozen >= -self.params["s_close"]:
                    should_exit = True

                # Stop-Loss: ruptura estructural congelada (|S_frozen| >= s_stop)
                if current_signal == -1 and s_score_frozen >= self.params["s_stop"]:
                    should_exit = True
                elif current_signal == 1 and s_score_frozen <= -self.params["s_stop"]:
                    should_exit = True

                # Time-Decay: tiempo >= multiplier * tau_entry
                if elapsed_min >= self.params["time_decay_multiplier"] * frozen_tau:
                    should_exit = True

                # Cierre MOC EOD
                if bar_time_str >= self.params["eod_flat"]:
                    should_exit = True

                if should_exit:
                    current_signal = 0
                    signals.loc[ts] = 0.0
                else:
                    signals.loc[ts] = float(current_signal)
                continue

            # ------------------------------------------------------------------
            # EVALUACIÓN DE ENTRADA (CON FILTROS MACKINNON, TAU Y SECTORIAL)
            # ------------------------------------------------------------------
            in_window = (self.params["entry_start"] <= bar_time_str <= self.params["entry_end"])
            tau_ok = (self.params["tau_min"] <= ou["tau_half_min"] <= self.params["tau_max"])

            # Filtro R4: Desacoplamiento sectorial XLK/SPY
            bar_date = ts.date() if hasattr(ts, "date") else ts
            sector_corr_val = sector_corr_lookup.get(bar_date, 1.0)
            sector_ok = (sector_corr_val >= self.params.get("min_sector_corr", 0.85))

            if in_window and tau_ok and sector_ok and current_signal == 0:
                res_cur = np.log(p_nq_cur) - (alpha_ols + beta_ols * np.log(p_es_cur))
                s_score_cur = self.model.calculate_s_score(
                    res_cur, ou["theta"], ou["sigma_eq"],
                    sigma_floor=self.params.get("sigma_floor", 0.001)
                )

                # Comprobar candidatos antes del test ADF
                candidate_sig = 0
                if self.params["s_open"] < s_score_cur < self.params["s_stop"]:
                    candidate_sig = -1  # SELL_SPREAD
                elif -self.params["s_stop"] < s_score_cur < -self.params["s_open"]:
                    candidate_sig = 1   # BUY_SPREAD

                if candidate_sig != 0:
                    is_stat, _, _ = self.model.test_stationarity_mackinnon(
                        res_hist,
                        t_stat_threshold=self.params["adf_t_threshold"],
                        p_val_threshold=self.params["adf_p_threshold"]
                    )
                    if is_stat:
                        current_signal = candidate_sig
                        entry_idx = i
                        # CONGELAR PARÁMETROS DE EQUILIBRIO AL ABRIR
                        frozen_alpha = alpha_ols
                        frozen_beta = beta_ols
                        frozen_theta = ou["theta"]
                        frozen_sigma_eq = ou["sigma_eq"]
                        frozen_tau = ou["tau_half_min"]
                        signals.loc[ts] = float(candidate_sig)

        return signals.reindex(data.index).fillna(0.0)


# ==============================================================================
# 5. SIMULADOR Y MOTOR DE BACKTESTING CORREGIDO (F4-OPTIMIZACION -IS- N1)
# ==============================================================================

class F4OptimizationISN2Backtester:
    """
    Simulador cuantitativo de alta fidelidad para F4-Optimizacion -IS- N2.

    Garantiza:
    1. Evaluación de muestra completa In-Sample (early_stopping=False por defecto).
    2. Anualización del Sharpe sobre retornos diarios (252 días).
    3. Parámetros de equilibrio congelados con posición abierta.
    4. Ejecución estrictamente no anticipativa en Open[t+1] con deslizamiento y comisiones.
    5. Cobertura Dollar-Beta en retículo coprimo y disyuntor de volatilidad.
    6. Eliminación de saltos nocturnos en la discretización AR(1) y Kalman Filter.
    """

    def __init__(
        self,
        strategy: Optional[F4OptimizationISN2Strategy] = None,
        account_size: float = 50000.0,
        daily_loss_limit: float = 1000.0,
        max_trailing_loss: float = 2000.0,
        trailing_mode: str = "eod",           # "eod" o "intraday"
        lock_floor_at_start: bool = True,
        profit_target: float = 3000.0,
        min_days: int = 5,
        early_stopping: bool = False,         # Por defecto False para evaluar año completo
    ):
        self.strategy = strategy or F4OptimizationISN2Strategy()
        self.account_size = account_size
        self.daily_loss_limit = daily_loss_limit
        self.max_trailing_loss = max_trailing_loss
        self.trailing_mode = trailing_mode
        self.lock_floor_at_start = lock_floor_at_start
        self.profit_target = profit_target
        self.min_days = min_days
        self.early_stopping = early_stopping
        self.model = self.strategy.model

    def _update_floor(self, peak: float) -> float:
        floor = peak - self.max_trailing_loss
        if self.lock_floor_at_start:
            floor = min(floor, self.account_size)
        return floor

    def run(
        self,
        df_nq: pd.DataFrame,
        df_es: pd.DataFrame,
        df_xlk: Optional[pd.DataFrame] = None,
        df_spy: Optional[pd.DataFrame] = None,
        sector_corr_series: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Ejecuta la simulación cuantitativa de F4-Optimizacion -IS- N2."""
        if sector_corr_series is None and df_xlk is not None and df_spy is not None:
            sector_corr_series = compute_sector_correlation_series(
                df_xlk, df_spy, window=self.strategy.params.get("sector_corr_window", 30)
            )

        sector_corr_lookup: Dict[Any, float] = {}
        if sector_corr_series is not None and not sector_corr_series.empty:
            sector_corr_lookup = dict(sector_corr_series)

        df_nq_ny = self.strategy.ensure_timezone(df_nq.copy())
        df_es_ny = self.strategy.ensure_timezone(df_es.copy())

        common_index = df_nq_ny.index.intersection(df_es_ny.index)
        if len(common_index) == 0:
            raise ValueError("No hay coincidencia temporal entre las series NQ y ES.")

        df_nq_ny = df_nq_ny.loc[common_index].sort_index()
        df_es_ny = df_es_ny.loc[common_index].sort_index()

        rth_mask = (
            (df_nq_ny.index.time >= time(9, 30)) &
            (df_nq_ny.index.time <= time(16, 0))
        )
        rth_nq = df_nq_ny[rth_mask]
        rth_es = df_es_ny[rth_mask]

        if len(rth_nq) < 20:
            raise ValueError("Se requieren al menos 20 barras RTH.")

        is_micro = (self.strategy.params["contract_mode"] == "micro")
        sym_nq = self.strategy.params["micro_nq"] if is_micro else self.strategy.params["symbol_nq"]
        sym_es = self.strategy.params["micro_es"] if is_micro else self.strategy.params["symbol_es"]

        spec_nq = get_spec(sym_nq)
        spec_es = get_spec(sym_es)

        m_nq = spec_nq["multiplier"]
        m_es = spec_es["multiplier"]
        tick_nq = spec_nq["tick"]
        tick_es = spec_es["tick"]
        comm_nq = spec_nq["commission_rt"]
        comm_es = spec_es["commission_rt"]
        slip_pts_nq = self.strategy.params["slippage_ticks"] * tick_nq
        slip_pts_es = self.strategy.params["slippage_ticks"] * tick_es

        balance = self.account_size
        peak = balance
        floor = self._update_floor(peak)

        failed = False
        fail_reason: Optional[str] = None
        fail_date: Optional[Any] = None
        passed = False
        pass_date: Optional[Any] = None

        trades: List[TradeRecordN2] = []
        daily_logs: List[Dict[str, Any]] = []

        k_calib = self.strategy.params["k_calib"]
        min_calib = min(self.strategy.params["min_calib_bars"], k_calib)

        # Filtro de Kalman online
        kf = KalmanFilterBeta() if self.strategy.params.get("use_kalman", True) else None
        kf_initialized = False

        grouped = rth_nq.groupby(rth_nq.index.normalize())

        in_position = False
        current_trade: Optional[TradeRecordN2] = None
        trade_counter = 0

        # Buffer para ejecución no anticipativa en Open[t+1] (Punto 4)
        pending_order: Optional[Dict[str, Any]] = None

        # Contador de barras y auditoría
        global_bar_idx = 0
        loss_streak = 0
        degrade_consecutive_bars = 0

        audit_bars_evaluated = 0
        audit_valid_ou = 0
        audit_adf_passed = 0
        audit_tau_passed = 0
        audit_both_passed = 0
        audit_betas = []
        audit_taus_min = []
        audit_sigmas = []

        last_is_stat = False
        last_p_val_coint = 1.0

        for day_dt, day_nq in grouped:
            if failed or (passed and self.early_stopping):
                break

            day_es = rth_es.loc[day_nq.index]
            day_start_balance = balance
            daily_realized_loss = 0.0
            loss_streak = 0

            for i in range(len(day_nq)):
                ts = day_nq.index[i]
                open_nq_cur = float(day_nq["Open"].iloc[i])
                open_es_cur = float(day_es["Open"].iloc[i])
                p_nq_cur = float(day_nq["Close"].iloc[i])
                p_es_cur = float(day_es["Close"].iloc[i])
                bar_time_str = ts.strftime("%H:%M")

                n_prior = global_bar_idx
                global_bar_idx += 1

                # --------------------------------------------------------------
                # EJECUCIÓN NO ANTICIPATIVA DE ORDEN PENDIENTE EN OPEN[t+1]
                # --------------------------------------------------------------
                if pending_order is not None and not in_position and not failed:
                    trade_counter += 1
                    sig_dir = pending_order["direction"]
                    n_baskets = pending_order["n_baskets"]
                    if "total_contracts_nq" in pending_order and "total_contracts_es" in pending_order:
                        n_nq = pending_order["total_contracts_nq"]
                        n_es = pending_order["total_contracts_es"]
                    else:
                        n_nq = pending_order["n_nq_unit"] * n_baskets
                        n_es = max(1, int(round(n_nq * pending_order.get("target_dollar_ratio", 1.0))))

                    # Precio de entrada con deslizamiento realista
                    # Si sig_dir == +1 (BUY_SPREAD: Long NQ, Short ES):
                    # NQ se compra al Ask (Open + slip), ES se vende al Bid (Open - slip)
                    # Si sig_dir == -1 (SELL_SPREAD: Short NQ, Long ES):
                    # NQ se vende al Bid (Open - slip), ES se compra al Ask (Open + slip)
                    entry_p_nq = open_nq_cur + (sig_dir * slip_pts_nq)
                    entry_p_es = open_es_cur - (sig_dir * slip_pts_es)

                    current_trade = TradeRecordN2(
                        trade_id=trade_counter,
                        signal_date=pending_order["signal_date"],
                        signal_time=pending_order["signal_time"],
                        entry_date=ts.strftime("%Y-%m-%d"),
                        entry_time=ts.strftime("%H:%M:%S"),
                        signal=pending_order["signal"],
                        direction=sig_dir,
                        contracts_nq=n_nq,
                        contracts_es=n_es,
                        symbol_nq=sym_nq,
                        symbol_es=sym_es,
                        n_baskets=n_baskets,
                        entry_price_nq=round(entry_p_nq, 2),
                        entry_price_es=round(entry_p_es, 2),
                        signal_s_score=round(pending_order["signal_s_score"], 3),
                        # Parámetros congelados
                        entry_alpha=pending_order["alpha"],
                        entry_beta=pending_order["beta"],
                        entry_theta=pending_order["theta"],
                        entry_sigma_eq=pending_order["sigma_eq"],
                        entry_tau_half=pending_order["tau_half"],
                        coprime_ratio=pending_order["coprime_ratio"],
                        target_dollar_ratio=pending_order["target_dollar_ratio"],
                        dollar_imbalance_usd=pending_order["dollar_imbalance_usd"],
                    )
                    in_position = True
                    pending_order = None

                if n_prior < min_calib:
                    continue

                # Ventana rodante hacia atrás sin fuga
                calib_start_idx = max(0, n_prior - k_calib)
                calib_nq = rth_nq.iloc[calib_start_idx:n_prior]["Close"].values
                calib_es = rth_es.iloc[calib_start_idx:n_prior]["Close"].values
                calib_ts = rth_nq.index[calib_start_idx:n_prior]

                log_calib_nq = np.log(calib_nq)
                log_calib_es = np.log(calib_es)

                # 1. Cointegración hacia atrás OLS y Kalman
                beta_ols, alpha_ols, res_hist = self.model.estimate_cointegration_backward(log_calib_nq, log_calib_es)
                if kf is not None:
                    if not kf_initialized:
                        kf.initialize_from_ols(alpha_ols, beta_ols)
                        kf_initialized = True
                    else:
                        kf.update(float(np.log(p_nq_cur)), float(np.log(p_es_cur)))

                if kf is not None and not self.strategy.params.get("unify_beta_ols", False):
                    beta_hedge = float(kf.state[1])
                else:
                    beta_hedge = beta_ols
                alpha_bar, beta_bar = alpha_ols, beta_hedge

                # 2. Ajuste OU libre de saltos nocturnos sobre D_intraday con piso de volatilidad (R3)
                ou = self.model.fit_ou_process_intraday(
                    res_hist, calib_ts,
                    dt_minutes=self.strategy.params["dt_minutes"],
                    sigma_floor=self.strategy.params.get("sigma_floor", 0.001)
                )

                # 3. S-score contemporáneo unificado (res_cur sobre residuos Engle-Granger OLS)
                res_cur = np.log(p_nq_cur) - (alpha_ols + beta_ols * np.log(p_es_cur))
                s_score_cur = self.model.calculate_s_score(
                    res_cur, ou["theta"], ou["sigma_eq"],
                    sigma_floor=self.strategy.params.get("sigma_floor", 0.001)
                ) if ou["is_valid"] else 0.0

                # Auditoría estadística
                audit_bars_evaluated += 1
                is_stat = False
                p_val_coint = 1.0
                if ou["is_valid"]:
                    audit_valid_ou += 1
                    audit_betas.append(beta_bar)
                    audit_taus_min.append(ou["tau_half_min"])
                    audit_sigmas.append(ou["sigma_eq"])

                    tau_ok = (self.strategy.params["tau_min"] <= ou["tau_half_min"] <= self.strategy.params["tau_max"])
                    if tau_ok:
                        audit_tau_passed += 1

                    eval_adf = (
                        (self.strategy.params["entry_start"] <= bar_time_str <= self.strategy.params["entry_end"]) and
                        (self.strategy.params["s_open"] < abs(s_score_cur) < self.strategy.params["s_stop"])
                    ) or in_position or (global_bar_idx % 20 == 0)

                    if eval_adf:
                        is_stat, t_stat, p_val_coint = self.model.test_stationarity_mackinnon(
                            res_hist,
                            t_stat_threshold=self.strategy.params["adf_t_threshold"],
                            p_val_threshold=self.strategy.params["adf_p_threshold"]
                        )
                        last_is_stat = is_stat
                        last_p_val_coint = p_val_coint
                    else:
                        is_stat = last_is_stat
                        p_val_coint = last_p_val_coint

                    if is_stat:
                        audit_adf_passed += 1
                    if is_stat and tau_ok:
                        audit_both_passed += 1

                # --------------------------------------------------------------
                # GESTIÓN DE POSICIÓN ABIERTA (CON PARÁMETROS CONGELADOS)
                # --------------------------------------------------------------
                if in_position and current_trade is not None:
                    current_trade.bars_held += 1
                    current_trade.duration_minutes += self.strategy.params["dt_minutes"]
                    elapsed_min = current_trade.duration_minutes

                    # Monitoreo de equity flotante
                    pts_nq_float = (p_nq_cur - current_trade.entry_price_nq) * current_trade.direction
                    pts_es_float = (p_es_cur - current_trade.entry_price_es) * (-current_trade.direction)
                    unrealized_gross = (
                        pts_nq_float * m_nq * current_trade.contracts_nq +
                        pts_es_float * m_es * current_trade.contracts_es
                    )
                    commissions_est = (
                        current_trade.contracts_nq * comm_nq +
                        current_trade.contracts_es * comm_es
                    )
                    slippage_est = (
                        2.0 * self.strategy.params["slippage_ticks"] * tick_nq * m_nq * current_trade.contracts_nq +
                        2.0 * self.strategy.params["slippage_ticks"] * tick_es * m_es * current_trade.contracts_es
                    )
                    legging_est = self.strategy.params.get("legging_penalty_ticks", 1.0) * tick_es * m_es * current_trade.contracts_es
                    friction_est = commissions_est + slippage_est + legging_est

                    unrealized_net = unrealized_gross - friction_est
                    unrealized_bal = balance + unrealized_net
                    unrealized_loss_today = max(0.0, day_start_balance - unrealized_bal)

                    if self.trailing_mode == "intraday" and unrealized_bal > peak:
                        peak = unrealized_bal
                        floor = self._update_floor(peak)

                    # Cálculo del S-Score CONGELADO (Punto 3) con piso de volatilidad (R3)
                    s_score_frozen = self.model.calculate_frozen_s_score(
                        p_nq=p_nq_cur,
                        p_es=p_es_cur,
                        entry_alpha=current_trade.entry_alpha,
                        entry_beta=current_trade.entry_beta,
                        entry_theta=current_trade.entry_theta,
                        entry_sigma_eq=current_trade.entry_sigma_eq,
                        sigma_floor=self.strategy.params.get("sigma_floor", 0.001)
                    )

                    exit_triggered = False
                    exit_reason_code: Optional[ExitReasonN2] = None

                    # 0. Disyuntor de límites de Prop Trading (DLL y TMDD)
                    if unrealized_loss_today >= self.daily_loss_limit or unrealized_bal <= floor:
                        exit_triggered = True
                        exit_reason_code = ExitReasonN2.RISK_BREACH

                    # 1. Take-Profit contra equilibrio CONGELADO (|S_frozen| <= s_close)
                    if not exit_triggered:
                        if current_trade.direction == -1 and s_score_frozen <= self.strategy.params["s_close"]:
                            exit_triggered = True
                            exit_reason_code = ExitReasonN2.TAKE_PROFIT
                        elif current_trade.direction == 1 and s_score_frozen >= -self.strategy.params["s_close"]:
                            exit_triggered = True
                            exit_reason_code = ExitReasonN2.TAKE_PROFIT

                    # 2. Stop-Loss estructural contra equilibrio CONGELADO (|S_frozen| >= s_stop)
                    if not exit_triggered:
                        if current_trade.direction == -1 and s_score_frozen >= self.strategy.params["s_stop"]:
                            exit_triggered = True
                            exit_reason_code = ExitReasonN2.STOP_LOSS
                        elif current_trade.direction == 1 and s_score_frozen <= -self.strategy.params["s_stop"]:
                            exit_triggered = True
                            exit_reason_code = ExitReasonN2.STOP_LOSS

                    # 3. Descalificación dinámica de estacionariedad (tau_degrade)
                    if not exit_triggered and self.strategy.params.get("enable_degrade_exit", True):
                        if p_val_coint > self.strategy.params.get("degrade_p_threshold", 0.10):
                            degrade_consecutive_bars += 1
                        else:
                            degrade_consecutive_bars = 0
                        if degrade_consecutive_bars >= self.strategy.params.get("degrade_consecutive_bars", 3):
                            exit_triggered = True
                            exit_reason_code = ExitReasonN2.COINT_DEGRADED

                    # 4. Decaimiento temporal (Time-Decay)
                    if not exit_triggered:
                        max_time = self.strategy.params["time_decay_multiplier"] * current_trade.entry_tau_half
                        if elapsed_min >= max_time:
                            exit_triggered = True
                            exit_reason_code = ExitReasonN2.TIME_DECAY

                    # 5. Cierre forzoso intradía MOC (>= 15:50 EST)
                    if not exit_triggered and (bar_time_str >= self.strategy.params["eod_flat"] or i == len(day_nq) - 1):
                        exit_triggered = True
                        exit_reason_code = ExitReasonN2.EOD_FLAT

                    # Ejecución del cierre
                    if exit_triggered:
                        # Para la salida al precio actual con deslizamiento
                        exit_p_nq = p_nq_cur - (current_trade.direction * slip_pts_nq)
                        exit_p_es = p_es_cur + (current_trade.direction * slip_pts_es)

                        pts_nq = (exit_p_nq - current_trade.entry_price_nq) * current_trade.direction
                        pts_es = (exit_p_es - current_trade.entry_price_es) * (-current_trade.direction)

                        gross_pnl = (
                            pts_nq * m_nq * current_trade.contracts_nq +
                            pts_es * m_es * current_trade.contracts_es
                        )
                        commissions = (
                            current_trade.contracts_nq * comm_nq +
                            current_trade.contracts_es * comm_es
                        )
                        slippage = (
                            2.0 * self.strategy.params["slippage_ticks"] * tick_nq * m_nq * current_trade.contracts_nq +
                            2.0 * self.strategy.params["slippage_ticks"] * tick_es * m_es * current_trade.contracts_es
                        )
                        legging = self.strategy.params.get("legging_penalty_ticks", 1.0) * tick_es * m_es * current_trade.contracts_es
                        total_friction = commissions + slippage + legging
                        net_pnl = gross_pnl - total_friction

                        balance += net_pnl
                        daily_realized_loss = max(0.0, day_start_balance - balance)
                        if net_pnl < 0:
                            loss_streak += 1
                        else:
                            loss_streak = 0

                        if self.trailing_mode == "intraday" and balance > peak:
                            peak = balance
                            floor = self._update_floor(peak)

                        current_dd = max(0.0, peak - balance)

                        current_trade.exit_date = ts.strftime("%Y-%m-%d")
                        current_trade.exit_time = ts.strftime("%H:%M:%S")
                        current_trade.exit_price_nq = round(exit_p_nq, 2)
                        current_trade.exit_price_es = round(exit_p_es, 2)
                        current_trade.exit_s_score_frozen = round(s_score_frozen, 3)
                        current_trade.exit_reason = exit_reason_code
                        current_trade.gross_pnl_usd = round(gross_pnl, 2)
                        current_trade.commissions_usd = round(commissions, 2)
                        current_trade.slippage_usd = round(slippage, 2)
                        current_trade.legging_usd = round(legging, 2)
                        current_trade.friction_usd = round(total_friction, 2)
                        current_trade.net_pnl_usd = round(net_pnl, 2)
                        current_trade.running_balance = round(balance, 2)
                        current_trade.running_drawdown = round(current_dd, 2)

                        trades.append(current_trade)
                        in_position = False
                        current_trade = None
                        degrade_consecutive_bars = 0

                        if balance <= floor:
                            failed = True
                            fail_reason = "Max Trailing Drawdown Breached"
                            fail_date = ts
                            break
                        elif daily_realized_loss >= self.daily_loss_limit:
                            failed = True
                            fail_reason = "Daily Loss Limit Breached"
                            fail_date = ts
                            break

                        continue

                # --------------------------------------------------------------
                # GENERACIÓN DE SEÑALES DE ENTRADA (EN COPRIMA DOLLAR-BETA)
                # --------------------------------------------------------------
                if not in_position and pending_order is None and not failed:
                    in_window = (self.strategy.params["entry_start"] <= bar_time_str <= self.strategy.params["entry_end"])
                    valid_ou = ou["is_valid"]
                    valid_tau = (self.strategy.params["tau_min"] <= ou["tau_half_min"] <= self.strategy.params["tau_max"]) if valid_ou else False

                    # Filtro R4: Desacoplamiento sectorial XLK/SPY
                    current_date = day_dt.date() if hasattr(day_dt, "date") else day_dt
                    sector_corr_val = sector_corr_lookup.get(current_date, 1.0)
                    sector_ok = (sector_corr_val >= self.strategy.params.get("min_sector_corr", 0.85))

                    if in_window and valid_ou and valid_tau and sector_ok:
                        sig_dir = 0
                        sig_name = "NEUTRAL"
                        if self.strategy.params["s_open"] < s_score_cur < self.strategy.params["s_stop"]:
                            sig_dir = -1
                            sig_name = "SELL_SPREAD"
                        elif -self.strategy.params["s_stop"] < s_score_cur < -self.strategy.params["s_open"]:
                            sig_dir = 1
                            sig_name = "BUY_SPREAD"

                        if sig_dir != 0 and is_stat:
                            # 1. Presupuesto de riesgo atenuado
                            tmdd_cushion = max(0.0, balance - floor)
                            risk_budget = self.model.calculate_risk_budget_attenuated(
                                dll=self.daily_loss_limit,
                                tmdd=self.max_trailing_loss,
                                realized_loss_today=daily_realized_loss,
                                current_drawdown=max(0.0, peak - balance),
                                loss_streak=loss_streak,
                                psi_tmdd=self.strategy.params.get("psi_tmdd", 0.30),
                                phi_dll=self.strategy.params.get("phi_dll", 0.60),
                                survival_floor=self.strategy.params.get("survival_floor", 105.0)
                            )

                            # 2. Ratio analítico Dollar-Beta
                            target_ratio = self.model.calculate_dollar_beta_ratio(
                                p_nq_cur, p_es_cur, beta_bar, m_nq, m_es
                            )

                            # 3. Retículo coprimo para neutralidad Dollar-Beta
                            max_nq = self.strategy.params.get("max_coprime_nq", 7)
                            opt_nq, opt_es, cop_err = self.model.find_optimal_coprime_basket(
                                target_ratio=target_ratio,
                                max_nq_contracts=max_nq,
                                prefer_sub_baskets=self.strategy.params.get("allow_sub_baskets", True)
                            )

                            # Neutralidad Dollar-Beta sin forzar ratio math.ceil 1:2:
                            if self.strategy.params.get("allow_sub_baskets", True) and is_micro:
                                f_opt = (
                                    opt_nq * comm_nq + opt_es * comm_es +
                                    2.0 * self.strategy.params["slippage_ticks"] * tick_nq * m_nq * opt_nq +
                                    2.0 * self.strategy.params["slippage_ticks"] * tick_es * m_es * opt_es +
                                    self.strategy.params.get("legging_penalty_ticks", 1.0) * tick_es * m_es * opt_es
                                )
                                sl_opt = self.model.calculate_unit_stop_loss_loss(
                                    sigma_eq=ou["sigma_eq"],
                                    p_nq=p_nq_cur,
                                    n_nq_unit=opt_nq,
                                    m_nq=m_nq,
                                    friction_basket_usd=f_opt,
                                    delta_s_sl=(self.strategy.params["s_stop"] - self.strategy.params["s_open"]),
                                    sigma_floor=self.strategy.params.get("sigma_floor", 0.001)
                                )
                                if sl_opt <= risk_budget and opt_nq > 1:
                                    unit_nq = opt_nq
                                    unit_es = opt_es
                                else:
                                    unit_nq = 1
                                    unit_es = max(1, int(round(unit_nq * target_ratio)))
                            else:
                                unit_nq = opt_nq
                                unit_es = opt_es

                            # Fricción unitaria de la cesta
                            friction_unit = (
                                unit_nq * comm_nq + unit_es * comm_es +
                                2.0 * self.strategy.params["slippage_ticks"] * tick_nq * m_nq * unit_nq +
                                2.0 * self.strategy.params["slippage_ticks"] * tick_es * m_es * unit_es +
                                self.strategy.params.get("legging_penalty_ticks", 1.0) * tick_es * m_es * unit_es
                            )

                            # Pérdida en Stop-Loss estructural (s_stop - s_open) con piso de volatilidad (R3)
                            unit_sl_loss = self.model.calculate_unit_stop_loss_loss(
                                sigma_eq=ou["sigma_eq"],
                                p_nq=p_nq_cur,
                                n_nq_unit=unit_nq,
                                m_nq=m_nq,
                                friction_basket_usd=friction_unit,
                                delta_s_sl=(self.strategy.params["s_stop"] - self.strategy.params["s_open"]),
                                sigma_floor=self.strategy.params.get("sigma_floor", 0.001)
                            )

                            # Asignación piso de sub-cestas con protección de supervivencia
                            n_baskets = self.model.calculate_baskets_allocation(
                                risk_budget_usd=risk_budget,
                                unit_sl_loss_usd=unit_sl_loss,
                                n_basket_max=self.strategy.params.get("n_basket_max", 3),
                                min_survival_budget=self.strategy.params.get("survival_floor", 100.0)
                            )

                            if n_baskets >= 1:
                                total_nq = unit_nq * n_baskets
                                total_es = max(1, int(round(total_nq * target_ratio)))
                                notional_nq = total_nq * p_nq_cur * m_nq
                                notional_es = total_es * p_es_cur * m_es
                                dollar_imbalance = abs(notional_nq * beta_bar - notional_es)

                                # Colocar orden para ejecución estricta en Open[t+1]
                                pending_order = {
                                    "signal_date": ts.strftime("%Y-%m-%d"),
                                    "signal_time": ts.strftime("%H:%M:%S"),
                                    "signal": sig_name,
                                    "direction": sig_dir,
                                    "signal_s_score": s_score_cur,
                                    "n_baskets": n_baskets,
                                    "n_nq_unit": unit_nq,
                                    "n_es_unit": unit_es,
                                    "total_contracts_nq": total_nq,
                                    "total_contracts_es": total_es,
                                    "alpha": alpha_bar,
                                    "beta": beta_bar,
                                    "theta": ou["theta"],
                                    "sigma_eq": ou["sigma_eq"],
                                    "tau_half": ou["tau_half_min"],
                                    "coprime_ratio": round(total_es / total_nq, 4),
                                    "target_dollar_ratio": round(target_ratio, 4),
                                    "dollar_imbalance_usd": round(dollar_imbalance, 2),
                                }

            if self.trailing_mode == "eod" and balance > peak:
                peak = balance
                floor = self._update_floor(peak)

            day_pnl = balance - day_start_balance
            curr_d = day_dt.date() if hasattr(day_dt, "date") else day_dt
            daily_logs.append({
                "date": day_dt.strftime("%Y-%m-%d") if hasattr(day_dt, "strftime") else str(day_dt),
                "day_pnl": round(day_pnl, 2),
                "ending_balance": round(balance, 2),
                "peak": round(peak, 2),
                "floor": round(floor, 2),
                "daily_realized_loss": round(daily_realized_loss, 2),
                "sector_corr": round(sector_corr_lookup.get(curr_d, 1.0), 4),
            })

            # Evaluación de Target de Prop Trading
            if (balance - self.account_size) >= self.profit_target and len(daily_logs) >= self.min_days:
                if not passed:
                    passed = True
                    pass_date = day_dt

        # Cierre forzoso de seguridad si queda posición al agotar datos
        if in_position and current_trade is not None:
            last_ts = rth_nq.index[-1]
            last_p_nq = float(rth_nq["Close"].iloc[-1])
            last_p_es = float(rth_es["Close"].iloc[-1])
            pts_nq = (last_p_nq - current_trade.entry_price_nq) * current_trade.direction
            pts_es = (last_p_es - current_trade.entry_price_es) * (-current_trade.direction)
            gross = pts_nq * m_nq * current_trade.contracts_nq + pts_es * m_es * current_trade.contracts_es
            comm = current_trade.contracts_nq * comm_nq + current_trade.contracts_es * comm_es
            slip = (
                2.0 * self.strategy.params["slippage_ticks"] * tick_nq * m_nq * current_trade.contracts_nq +
                2.0 * self.strategy.params["slippage_ticks"] * tick_es * m_es * current_trade.contracts_es
            )
            fric = comm + slip
            net = gross - fric
            balance += net
            current_trade.exit_date = last_ts.strftime("%Y-%m-%d")
            current_trade.exit_time = last_ts.strftime("%H:%M:%S")
            current_trade.exit_price_nq = round(last_p_nq, 2)
            current_trade.exit_price_es = round(last_p_es, 2)
            current_trade.exit_reason = ExitReasonN2.EOD_FLAT
            current_trade.gross_pnl_usd = round(gross, 2)
            current_trade.commissions_usd = round(comm, 2)
            current_trade.slippage_usd = round(slip, 2)
            current_trade.friction_usd = round(fric, 2)
            current_trade.net_pnl_usd = round(net, 2)
            current_trade.running_balance = round(balance, 2)
            trades.append(current_trade)
            in_position = False
            current_trade = None

        trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame()
        daily_df = pd.DataFrame(daily_logs) if daily_logs else pd.DataFrame()
        metrics = self._calculate_metrics(trades_df, daily_df, balance)

        prop_evaluation = {
            "status": "PASS" if passed else ("FAIL" if failed else "INCOMPLETE"),
            "reason": fail_reason if failed else ("Profit Target Achieved" if passed else "Evaluation Ongoing"),
            "target_achieved_date": str(pass_date) if passed else None,
            "final_balance": round(balance, 2),
            "net_profit": round(balance - self.account_size, 2),
            "peak_equity": round(peak, 2),
            "final_floor": round(floor, 2),
            "total_days_evaluated": len(daily_logs),
            "early_stopping_disabled": not self.early_stopping,
        }

        cointegration_audit = {
            "bars_evaluated": audit_bars_evaluated,
            "valid_ou_count": audit_valid_ou,
            "valid_ou_pct": round((audit_valid_ou / audit_bars_evaluated * 100.0) if audit_bars_evaluated > 0 else 0.0, 2),
            "adf_passed_count": audit_adf_passed,
            "adf_passed_pct": round((audit_adf_passed / audit_valid_ou * 100.0) if audit_valid_ou > 0 else 0.0, 2),
            "tau_passed_count": audit_tau_passed,
            "tau_passed_pct": round((audit_tau_passed / audit_valid_ou * 100.0) if audit_valid_ou > 0 else 0.0, 2),
            "both_passed_count": audit_both_passed,
            "both_passed_pct": round((audit_both_passed / audit_valid_ou * 100.0) if audit_valid_ou > 0 else 0.0, 2),
            "mean_beta": round(float(np.mean(audit_betas)), 4) if audit_betas else 0.0,
            "min_beta": round(float(np.min(audit_betas)), 4) if audit_betas else 0.0,
            "max_beta": round(float(np.max(audit_betas)), 4) if audit_betas else 0.0,
            "mean_tau_half_min": round(float(np.mean(audit_taus_min)), 1) if audit_taus_min else 0.0,
            "mean_sigma_eq": round(float(np.mean(audit_sigmas)), 6) if audit_sigmas else 0.0,
        }

        return {
            "metrics": metrics,
            "prop_evaluation": prop_evaluation,
            "trades": trades_df,
            "daily_logs": daily_df,
            "cointegration_audit": cointegration_audit,
            "sector_correlation": sector_corr_series,
        }

    def _calculate_metrics(
        self,
        trades_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        final_balance: float
    ) -> Dict[str, Any]:
        """
        Cálculo riguroso de métricas cuantitativas corregidas (Punto 2):
        - Ratio de Sharpe anualizado sobre la serie completa de retornos diarios (252 días).
        - Ratio de Sharpe por trade reportado por separado sin inflar por sqrt(252).
        """
        if trades_df.empty:
            return {
                "Total Trades": 0,
                "Winning Trades": 0,
                "Losing Trades": 0,
                "Win Rate (%)": 0.0,
                "Profit Factor": 0.0,
                "Net Profit ($)": 0.0,
                "Total Return (%)": 0.0,
                "Max Drawdown ($)": 0.0,
                "Sharpe Ratio (Daily 252d)": 0.0,
                "Sharpe Ratio (Per Trade)": 0.0,
                "Sortino Ratio (Daily 252d)": 0.0,
                "Annualized Return (%)": 0.0,
            }

        pnl = trades_df["net_pnl_usd"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        win_rate = (len(wins) / len(pnl)) * 100.0 if len(pnl) > 0 else 0.0
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        balances = trades_df["running_balance"]
        peak = balances.cummax()
        dd = peak - balances
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0

        # Sharpe por trade (sin inflar por 252)
        returns_per_trade = pnl / self.account_size
        std_ret_trade = float(returns_per_trade.std(ddof=1)) if len(returns_per_trade) > 1 else 0.0
        sharpe_per_trade = (float(returns_per_trade.mean()) / std_ret_trade) if std_ret_trade > 0 else 0.0

        # Anualización rigurosa sobre retornos diarios (252 días)
        if not daily_df.empty and "day_pnl" in daily_df.columns:
            daily_returns = daily_df["day_pnl"] / self.account_size
            mean_daily = float(daily_returns.mean())
            std_daily = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0
            sharpe_daily = (mean_daily / std_daily * np.sqrt(252.0)) if std_daily > 0 else 0.0

            downside_daily = daily_returns[daily_returns < 0]
            downside_std_daily = float(downside_daily.std(ddof=1)) if len(downside_daily) > 1 else 0.0
            sortino_daily = (mean_daily / downside_std_daily * np.sqrt(252.0)) if downside_std_daily > 0 else 0.0

            n_days = len(daily_returns)
            annualized_ret = (float(daily_returns.sum()) * (252.0 / n_days)) * 100.0 if n_days > 0 else 0.0
        else:
            sharpe_daily = sharpe_per_trade
            sortino_daily = 0.0
            annualized_ret = 0.0

        exit_counts = trades_df["exit_reason"].value_counts().to_dict()

        return {
            "Total Trades": int(len(trades_df)),
            "Winning Trades": int(len(wins)),
            "Losing Trades": int(len(losses)),
            "Win Rate (%)": round(win_rate, 2),
            "Gross Profit ($)": round(gross_profit, 2),
            "Gross Loss ($)": round(gross_loss, 2),
            "Profit Factor": round(profit_factor, 2) if not np.isinf(profit_factor) else "INF",
            "Net Profit ($)": round(final_balance - self.account_size, 2),
            "Total Return (%)": round(((final_balance / self.account_size) - 1.0) * 100.0, 2),
            "Annualized Return (%)": round(annualized_ret, 2),
            "Max Drawdown ($)": round(max_dd, 2),
            "Sharpe Ratio (Daily 252d)": round(sharpe_daily, 2),
            "Sharpe Ratio (Per Trade)": round(sharpe_per_trade, 4),
            "Sortino Ratio (Daily 252d)": round(sortino_daily, 2),
            "Exit Reasons": exit_counts,
        }
