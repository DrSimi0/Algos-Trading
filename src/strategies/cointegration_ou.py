"""
src/strategies/cointegration_ou.py
Implementación Matemática Formal y Backtester de la Estrategia F4:
Cointegración Inter-Mercado ES vs NQ y Proceso de Difusión Ornstein-Uhlenbeck (StatArb OU).

Referencias Académicas:
1. Avellaneda, M., & Lee, J. H. (2010). Statistical arbitrage in the U.S. equities market.
   Quantitative Finance, 10(7), 761-782.
2. Hasbrouck, J. (2003). Intraday Price Formation in U.S. Equity Index Markets.
   The Journal of Finance, 58(6), 2375-2400.
3. Hou, K. (2007). Industry Information Diffusion and the Lead-Lag Effect in Stock Returns.
   The Review of Financial Studies, 20(4), 1113-1138.
4. Engle, R. F., & Granger, C. W. (1987). Co-Integration and Error Correction:
   Representation, Estimation, and Testing. Econometrica, 55(2), 251-276.
5. Uhlenbeck, G. E., & Ornstein, L. S. (1930). On the Theory of the Brownian Motion.
   Physical Review, 36(5), 823-841.

Especificación: reports/estrategias_matematicas/F4_cointegracion_es_nq_ornstein_uhlenbeck.md
"""

from dataclasses import dataclass, field
from datetime import time, datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Sequence
import warnings
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts

from src.core.base_strategy import BaseStrategy
from src.core.futures_specs import get_spec, FUTURES_SPECS

# Suprimir advertencias futuras de statsmodels en ADF
warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# 1. ENUMERACIONES Y ESTRUCTURAS DE DATOS (SECCIONES 2.1 & 4.5)
# ==============================================================================

class TradeSignal(int, Enum):
    SELL_SPREAD = -1   # Corto NQ / Largo ES (S-score > +s_open)
    NEUTRAL = 0        # Fuera de mercado / Canal de indiferencia
    BUY_SPREAD = 1     # Largo NQ / Corto ES (S-score < -s_open)


class ExitReason(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"       # Reversión a la media (|S| <= s_close)
    STOP_LOSS = "STOP_LOSS"           # Ruptura estructural (|S| >= s_stop)
    TIME_DECAY = "TIME_DECAY"         # Decaimiento temporal (tiempo >= 2 * tau_1/2)
    EOD_FLAT = "EOD_FLAT"             # Cierre forzoso intradía (>= 15:50 EST)
    RISK_BREACH = "RISK_BREACH"       # Violación de DLL o TMDD


@dataclass
class TradeRecord:
    """Registro detallado de una operación de arbitraje estadístico de pares."""
    trade_id: int
    entry_date: str
    entry_time: str
    signal: str                        # "BUY_SPREAD" o "SELL_SPREAD"
    direction: int                     # +1 = Buy Spread (Long NQ, Short ES), -1 = Sell Spread
    contracts_nq: int                  # Contratos NQ o MNQ
    contracts_es: int                  # Contratos ES o MES
    symbol_nq: str                     # "MNQ" o "NQ"
    symbol_es: str                     # "MES" o "ES"
    n_baskets: int                     # Número de cestas ejecutadas
    entry_price_nq: float
    entry_price_es: float
    entry_s_score: float
    entry_beta: float
    entry_tau_half: float              # Vida media en minutos
    exit_date: Optional[str] = None
    exit_time: Optional[str] = None
    exit_price_nq: Optional[float] = None
    exit_price_es: Optional[float] = None
    exit_s_score: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    bars_held: int = 0
    duration_minutes: float = 0.0
    gross_pnl_usd: float = 0.0
    commissions_usd: float = 0.0
    slippage_usd: float = 0.0
    friction_usd: float = 0.0
    net_pnl_usd: float = 0.0
    running_balance: float = 0.0
    running_drawdown: float = 0.0


# ==============================================================================
# 2. MODELO MATEMÁTICO FORMAL: COINTEGRACIÓN Y DIFUSIÓN ORNSTEIN-UHLENBECK
# ==============================================================================

class CointegrationOUModel:
    """
    Núcleo analítico de formulación matemática pura de la Estrategia F4.
    Implementa:
    - Estimación OLS de Cointegración Engle-Granger (Sección 3.1)
    - Test Dickey-Fuller Aumentado de Estacionariedad (Sección 3.2)
    - Discretización analítica exacta en forma AR(1) del proceso OU (Sección 3.3 - 3.4)
    - Vida media de reversión y filtro de admisibilidad (Sección 3.5)
    - S-Score estandarizado de Avellaneda & Lee (Sección 3.6)
    - Neutralidad Dollar-Beta y dimensionamiento de riesgo Prop Trading (Sección 5)
    """

    @staticmethod
    def estimate_cointegration(
        p_nq: np.ndarray,
        p_es: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estima la relación de cointegración Engle-Granger entre log-precios:
        p_NQ(t) = alpha_coint + beta_hedge * p_ES(t) + X(t)
        
        Retorna:
            (beta_hedge, alpha_coint, residuals_X)
        """
        if len(p_nq) != len(p_es):
            raise ValueError("Las series de log-precios deben tener igual longitud.")
        if len(p_nq) < 10:
            raise ValueError("Se requieren al menos 10 observaciones para estimar cointegración.")

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
    def test_stationarity(
        residuals: np.ndarray,
        t_stat_threshold: float = -3.45,
        p_val_threshold: float = 0.05
    ) -> Tuple[bool, float, float]:
        """
        Test de Dickey-Fuller Aumentado (ADF) sobre el residuo del spread.
        Condición matemática de admisibilidad (Sección 3.2):
        C_stationarity = 1 si t_stat < -3.45 y p_val < 0.05.
        
        Retorna:
            (is_stationary, t_stat, p_val)
        """
        if len(residuals) < 20:
            return False, 0.0, 1.0

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_result = ts.adfuller(residuals, autolag="AIC")
            t_stat = float(adf_result[0])
            p_val = float(adf_result[1])
            is_stationary = (t_stat < t_stat_threshold) and (p_val < p_val_threshold)
            return bool(is_stationary), t_stat, p_val
        except Exception:
            return False, 0.0, 1.0

    @staticmethod
    def fit_ou_process(
        residuals: np.ndarray,
        dt_minutes: float = 5.0
    ) -> Dict[str, float]:
        """
        Ajuste analítico del proceso de difusión Ornstein-Uhlenbeck vía AR(1):
        X_k = a + b * X_{k-1} + eta_k
        
        Fórmulas exactas invertibles (Sección 3.4):
        - kappa = -ln(b) / dt
        - theta = a / (1 - b)
        - sigma_eq = sqrt( sigma_eta^2 / (1 - b^2) )
        - sigma = sigma_eq * sqrt(2 * kappa)
        - tau_half = ln(2) / kappa = ln(2) * dt / -ln(b)
        """
        n = len(residuals)
        if n < 10:
            return {
                "b": 0.0, "a": 0.0, "kappa": 0.0, "theta": 0.0,
                "sigma_eta": 0.0, "sigma_eq": 0.0, "sigma": 0.0,
                "tau_half_min": 0.0, "is_valid": False
            }

        x_prev = residuals[:-1]
        x_curr = residuals[1:]

        var_prev = np.var(x_prev, ddof=1)
        if var_prev <= 1e-14:
            return {
                "b": 0.0, "a": 0.0, "kappa": 0.0, "theta": 0.0,
                "sigma_eta": 0.0, "sigma_eq": 0.0, "sigma": 0.0,
                "tau_half_min": 0.0, "is_valid": False
            }

        cov_matrix = np.cov(x_curr, x_prev, ddof=1)
        cov_curr_prev = cov_matrix[0, 1]

        b = float(cov_curr_prev / var_prev)
        a = float(np.mean(x_curr) - b * np.mean(x_prev))

        # Condición estricta de ergodicidad y reversión a la media: 0 < b < 1
        if b <= 0.0 or b >= 1.0:
            return {
                "b": b, "a": a, "kappa": 0.0, "theta": 0.0,
                "sigma_eta": 0.0, "sigma_eq": 0.0, "sigma": 0.0,
                "tau_half_min": 0.0, "is_valid": False
            }

        eta = x_curr - (a + b * x_prev)
        sigma_eta_sq = float(np.var(eta, ddof=2)) if len(eta) > 2 else float(np.var(eta))
        if sigma_eta_sq <= 1e-14:
            sigma_eta_sq = 1e-14

        # Velocidad de reversión continua en unidades de [minutos^-1]
        kappa = float(-np.log(b) / dt_minutes)

        # Nivel de equilibrio asintótico
        theta = float(a / (1.0 - b))

        # Desviación estándar de equilibrio estacionario
        sigma_eq_sq = float(sigma_eta_sq / (1.0 - b**2))
        sigma_eq = float(np.sqrt(max(1e-14, sigma_eq_sq)))

        # Volatilidad instantánea continua
        sigma = float(sigma_eq * np.sqrt(2.0 * kappa))

        # Vida media en minutos
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
            "is_valid": True
        }

    @staticmethod
    def calculate_s_score(
        x_k: float,
        theta: float,
        sigma_eq: float
    ) -> float:
        """
        Puntuación estandarizada S-Score de Avellaneda & Lee (Sección 3.6):
        S_k = (X_k - theta) / sigma_eq
        """
        if sigma_eq <= 1e-12:
            return 0.0
        return float((x_k - theta) / sigma_eq)

    @staticmethod
    def calculate_dollar_beta_ratio(
        p_nq: float,
        p_es: float,
        beta_hedge: float,
        m_nq: float = 2.0,
        m_es: float = 5.0
    ) -> float:
        """
        Ratio analítico de cobertura neutral Dollar-Beta (Sección 5.1):
        Ratio_coint = beta_hedge * ( (P_NQ * M_NQ) / (P_ES * M_ES) )
        """
        if p_es <= 0 or m_es <= 0:
            return 1.0
        return float(beta_hedge * ((p_nq * m_nq) / (p_es * m_es)))

    @staticmethod
    def calculate_risk_budget(
        dll: float,
        tmdd: float,
        realized_loss_today: float = 0.0,
        current_drawdown: float = 0.0,
        phi_dll: float = 0.35,
        phi_tmdd: float = 0.20
    ) -> float:
        """
        Presupuesto de riesgo monetario para Prop Trading (Sección 5.3):
        PresupuestoRiesgo_k = min( DLL * phi_DLL, TMDD * phi_TMDD )
        Ajustado por pérdidas ya realizadas hoy y drawdown flotante actual.
        """
        daily_cushion = max(0.0, dll - realized_loss_today)
        tmdd_cushion = max(0.0, tmdd - current_drawdown)

        budget_daily = phi_dll * daily_cushion
        budget_tmdd = phi_tmdd * tmdd_cushion

        return float(max(0.0, min(budget_daily, budget_tmdd)))

    @staticmethod
    def calculate_unit_stop_loss_loss(
        sigma_eq: float,
        p_nq: float,
        n_nq_unit: int,
        m_nq: float,
        friction_basket_usd: float,
        delta_s_sl: float = 2.00
    ) -> float:
        """
        Pérdida monetaria máxima esperada al tocar el Stop-Loss por cada cesta unitaria (Sección 5.4):
        Delta_X_SL = (s_stop - s_open) * sigma_eq = 2.00 * sigma_eq
        PerdidaUnitaria_SL = Delta_X_SL * (N_NQ^(u) * M_NQ * P_NQ) + Friccion_$
        """
        delta_x_sl = delta_s_sl * sigma_eq
        notional_nq = n_nq_unit * m_nq * p_nq
        loss_pts = delta_x_sl * notional_nq
        total_unit_loss = loss_pts + friction_basket_usd
        return float(total_unit_loss)

    @staticmethod
    def calculate_baskets_allocation(
        risk_budget_usd: float,
        unit_sl_loss_usd: float,
        n_basket_max: int = 3
    ) -> int:
        """
        Función entera suelo de asignación de cestas y blindaje por volatilidad (Sección 5.5):
        N_baskets = max(0, min( floor( PresupuestoRiesgo / PerdidaUnitaria_SL ), N_basket_max ))
        Si PerdidaUnitaria_SL > PresupuestoRiesgo => N_baskets = 0 (HALT_VOLATILITY_DEFENSE).
        """
        if risk_budget_usd <= 0 or unit_sl_loss_usd <= 0:
            return 0
        raw_baskets = int(np.floor(risk_budget_usd / unit_sl_loss_usd))
        return int(max(0, min(raw_baskets, n_basket_max)))


# ==============================================================================
# 3. ESTRATEGIA PRINCIPAL (INTERFAZ BASESTRATEGY Y LÓGICA COMPLETA)
# ==============================================================================

class CointegrationOUStrategy(BaseStrategy):
    """
    Estrategia Cuantitativa F4: Cointegración Inter-Mercado ES vs NQ y Difusión OU.
    """

    def __init__(
        self,
        name: str = "Cointegration_OU_ES_NQ",
        params: Optional[Dict[str, Any]] = None,
    ):
        default_params: Dict[str, Any] = {
            # Instrumentos
            "symbol_nq": "NQ",
            "symbol_es": "ES",
            "micro_nq": "MNQ",
            "micro_es": "MES",
            "contract_mode": "micro",           # "micro" o "standard"
            "basket_unit_mnq": 10,              # Cesta unitaria canónica 10 MNQ vs 17 MES (Sec 5.2)
            "basket_unit_mes": 17,
            "basket_unit_nq": 1,                # Cesta unitaria standard 1 NQ vs 2 ES (Sec 5.6)
            "basket_unit_es": 2,
            "allow_sub_baskets": True,          # Permite sub-cestas (1 MNQ vs exact MES) si el presupuesto es < 10 MNQ
            "dynamic_basket_ratio": True,       # Si True, calcula MES según Ratio_coint exacto para cumplir RUB-F4-04
            # Ventana de calibración histórica (Sec 2.2)
            "k_calib": 2340,                    # 30 sesiones RTH x 78 barras (Sec 2.3)
            "min_calib_bars": 390,              # Calibración mínima permitida (5 sesiones RTH)
            "dt_minutes": 5.0,                  # Barra de 5 minutos (Delta t = 5 min)
            # Umbrales analíticos de Avellaneda & Lee (Sec 2.2 & 3.7)
            "s_open": 1.25,                     # Activación de señal
            "s_close": 0.25,                    # Take-profit por reversión a la media
            "s_stop": 3.25,                     # Stop-loss estructural
            # Filtro de velocidad de reversión (Sec 3.5)
            "tau_min": 15.0,                    # Vida media mínima (15 minutos)
            "tau_max": 780.0,                   # Vida media máxima (2 sesiones RTH = 780 minutos)
            # Filtro de significancia de cointegración ADF (Sec 3.2)
            "adf_t_threshold": -3.45,
            "adf_p_threshold": 0.05,
            # Horarios operativos RTH en America/New_York (Sec 2.3 & 4.4)
            "rth_start": "09:30",
            "rth_end": "16:00",
            "entry_start": "10:00",             # Entradas permitidas solo entre 10:00 y 15:40 EST
            "entry_end": "15:40",
            "eod_flat": "15:50",                # Cierre forzoso a las 15:50 EST (cero riesgo overnight)
            "time_decay_multiplier": 2.0,       # Salida por time-decay si tiempo >= 2 * tau_1/2
            # Restricciones Prop Trading (Sec 5.3 & 5.5)
            "phi_dll": 0.35,                    # Máx 35% de margen diario en un trade
            "phi_tmdd": 0.20,                   # Máx 20% de colchón de trailing drawdown
            "n_basket_max": 3,                  # Límite superior de cestas micro
            "slippage_ticks": 1.0,              # 1 tick entrada + 1 tick salida = 2 ticks slippage
            # Zonas horarias
            "target_tz": "America/New_York",
            "source_tz": "America/Mexico_City", # Formato habitual de los datos brutos UTC-6
        }
        if params:
            default_params.update(params)
        super().__init__(name=name, params=default_params)
        self.model = CointegrationOUModel()

    def ensure_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Localiza y convierte el índice al huso horario de Nueva York (EST/EDT)."""
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
        Interfaz canónica de BaseStrategy.
        Para pares de activos, 'data' puede ser:
        1. DataFrame con columnas 'NQ' y 'ES' (precios de cierre), o
        2. DataFrame con columnas 'Close_NQ' y 'Close_ES'.
        
        Retorna:
            pd.Series con valores -1.0 (Corto Spread), +1.0 (Largo Spread), 0.0 (Neutral).
        """
        if "NQ" in data.columns and "ES" in data.columns:
            nq_close = data["NQ"]
            es_close = data["ES"]
        elif "Close_NQ" in data.columns and "Close_ES" in data.columns:
            nq_close = data["Close_NQ"]
            es_close = data["Close_ES"]
        elif "Close" in data.columns:
            raise ValueError(
                "La estrategia de Cointegración requiere precios de NQ y ES. "
                "Proporcione columnas ['NQ', 'ES'] o use CointegrationOUBacktester."
            )
        else:
            raise ValueError("No se encontraron columnas de precios para NQ y ES.")

        df_pair = pd.DataFrame({"NQ": nq_close, "ES": es_close}).dropna()
        df_pair = self.ensure_timezone(df_pair)

        rth_bars = df_pair.between_time(self.params["rth_start"], self.params["rth_end"])
        signals = pd.Series(0.0, index=df_pair.index)

        k_calib = self.params["k_calib"]
        min_calib = min(self.params["min_calib_bars"], k_calib)

        current_signal = 0
        entry_idx = 0
        current_tau = 0.0

        for i in range(len(rth_bars)):
            ts = rth_bars.index[i]
            if i < min_calib:
                continue

            win_start = max(0, i - k_calib)
            hist_nq = rth_bars["NQ"].iloc[win_start:i]
            hist_es = rth_bars["ES"].iloc[win_start:i]

            log_hist_nq = np.log(hist_nq.values)
            log_hist_es = np.log(hist_es.values)

            # 1. Cointegración OLS
            beta, alpha, res_hist = self.model.estimate_cointegration(log_hist_nq, log_hist_es)

            # 2. Proceso OU
            ou_params = self.model.fit_ou_process(res_hist, dt_minutes=self.params["dt_minutes"])
            if not ou_params["is_valid"]:
                if current_signal != 0:
                    current_signal = 0
                signals.loc[ts] = 0.0
                continue

            # 3. S-score actual
            p_nq_cur = float(rth_bars["NQ"].iloc[i])
            p_es_cur = float(rth_bars["ES"].iloc[i])
            res_cur = np.log(p_nq_cur) - (alpha + beta * np.log(p_es_cur))
            s_score = self.model.calculate_s_score(res_cur, ou_params["theta"], ou_params["sigma_eq"])

            bar_time_str = ts.strftime("%H:%M")

            # Salida si estamos en posición
            if current_signal != 0:
                elapsed_min = (i - entry_idx) * self.params["dt_minutes"]
                should_exit = False

                # Take-Profit
                if current_signal == -1 and s_score <= self.params["s_close"]:
                    should_exit = True
                elif current_signal == 1 and s_score >= -self.params["s_close"]:
                    should_exit = True

                # Stop-Loss
                if current_signal == -1 and s_score >= self.params["s_stop"]:
                    should_exit = True
                elif current_signal == 1 and s_score <= -self.params["s_stop"]:
                    should_exit = True

                # Time-Decay
                if elapsed_min >= self.params["time_decay_multiplier"] * current_tau:
                    should_exit = True

                # Cierre forzoso EOD
                if bar_time_str >= self.params["eod_flat"]:
                    should_exit = True

                if should_exit:
                    current_signal = 0
                    signals.loc[ts] = 0.0
                    continue
                else:
                    signals.loc[ts] = float(current_signal)
                    continue

            # Entrada
            in_entry_window = (self.params["entry_start"] <= bar_time_str <= self.params["entry_end"])
            tau_ok = (self.params["tau_min"] <= ou_params["tau_half_min"] <= self.params["tau_max"])

            if in_entry_window and tau_ok and current_signal == 0:
                # Solo evaluar ADF al evaluar nueva entrada
                is_stat, _, _ = self.model.test_stationarity(
                    res_hist,
                    t_stat_threshold=self.params["adf_t_threshold"],
                    p_val_threshold=self.params["adf_p_threshold"]
                )
                if is_stat:
                    # Condición de entrada analítica acotada por ruptura (|S| < s_stop)
                    if self.params["s_open"] < s_score < self.params["s_stop"]:
                        current_signal = -1
                        entry_idx = i
                        current_tau = ou_params["tau_half_min"]
                        signals.loc[ts] = -1.0
                    elif -self.params["s_stop"] < s_score < -self.params["s_open"]:
                        current_signal = 1
                        entry_idx = i
                        current_tau = ou_params["tau_half_min"]
                        signals.loc[ts] = 1.0

        return signals.reindex(data.index).fillna(0.0)


# ==============================================================================
# 4. SIMULADOR Y MOTOR DE BACKTESTING DE ALTA FIDELIDAD PARA PROP TRADING
# ==============================================================================

class CointegrationOUBacktester:
    """
    Simulador cuantitativo de alta fidelidad para la Estrategia F4: Cointegración ES/NQ.
    
    Implementa:
    - Ejecución simultánea de pares de futuros CME (ES/NQ o MES/MNQ).
    - Control de límites de cuentas de Prop Trading (Topstep, Apex, Lucid, MFF):
      * Límite de Pérdida Diaria (Daily Loss Limit - DLL).
      * Trailing Drawdown Máximo (MTDD) con anclaje de piso (lock_floor_at_start).
    - Cestas neutrales en valor nocional Dollar-Beta (10 MNQ vs 17 MES o sub-cestas).
    - Comisiones reales por contrato y deslizamiento exacto por tick en cada pata.
    - Blindaje de cuenta por disyuntor de volatilidad (Sec 5.5).
    - Cierre determinista a las 15:50 EST (cero riesgo overnight).
    """

    def __init__(
        self,
        strategy: Optional[CointegrationOUStrategy] = None,
        account_size: float = 50000.0,
        daily_loss_limit: float = 1000.0,
        max_trailing_loss: float = 2000.0,
        trailing_mode: str = "eod",           # "eod" o "intraday"
        lock_floor_at_start: bool = True,
        profit_target: float = 3000.0,
        min_days: int = 5,
    ):
        self.strategy = strategy or CointegrationOUStrategy()
        self.account_size = account_size
        self.daily_loss_limit = daily_loss_limit
        self.max_trailing_loss = max_trailing_loss
        self.trailing_mode = trailing_mode
        self.lock_floor_at_start = lock_floor_at_start
        self.profit_target = profit_target
        self.min_days = min_days
        self.model = self.strategy.model

    def _update_floor(self, peak: float) -> float:
        """Actualiza el piso de trailing drawdown respetando el anclaje al capital inicial."""
        floor = peak - self.max_trailing_loss
        if self.lock_floor_at_start:
            floor = min(floor, self.account_size)
        return floor

    def run(
        self,
        df_nq: pd.DataFrame,
        df_es: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Ejecuta la simulación completa de arbitraje estadístico sobre las series de NQ y ES.
        """
        # Validar y armonizar series temporales
        df_nq_ny = self.strategy.ensure_timezone(df_nq.copy())
        df_es_ny = self.strategy.ensure_timezone(df_es.copy())

        common_index = df_nq_ny.index.intersection(df_es_ny.index)
        if len(common_index) == 0:
            raise ValueError("No hay coincidencia de marcas temporales entre NQ y ES.")

        df_nq_ny = df_nq_ny.loc[common_index].sort_index()
        df_es_ny = df_es_ny.loc[common_index].sort_index()

        rth_mask = (
            (df_nq_ny.index.time >= time(9, 30)) &
            (df_nq_ny.index.time <= time(16, 0))
        )
        rth_nq = df_nq_ny[rth_mask]
        rth_es = df_es_ny[rth_mask]

        if len(rth_nq) < 20:
            raise ValueError("Se requieren al menos 20 barras RTH para ejecutar la simulación.")

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

        unit_nq = self.strategy.params["basket_unit_mnq"] if is_micro else self.strategy.params["basket_unit_nq"]
        unit_es = self.strategy.params["basket_unit_mes"] if is_micro else self.strategy.params["basket_unit_es"]

        balance = self.account_size
        peak = balance
        floor = self._update_floor(peak)

        failed = False
        fail_reason: Optional[str] = None
        fail_date: Optional[Any] = None
        passed = False
        pass_date: Optional[Any] = None

        trades: List[TradeRecord] = []
        daily_logs: List[Dict[str, Any]] = []

        k_calib = self.strategy.params["k_calib"]
        min_calib = min(self.strategy.params["min_calib_bars"], k_calib)

        grouped = rth_nq.groupby(rth_nq.index.normalize())

        in_position = False
        current_trade: Optional[TradeRecord] = None
        trade_counter = 0

        # Contador global continuo O(1) de barras RTH
        global_bar_idx = 0

        # Auditoría cuantitativa de cointegración y difusión OU
        audit_bars_evaluated = 0
        audit_valid_ou = 0
        audit_adf_passed = 0
        audit_tau_passed = 0
        audit_both_passed = 0
        audit_betas = []
        audit_taus_min = []
        audit_sigmas = []
        latest_sl_sub_usd = 0.0
        latest_sl_std_usd = 0.0

        for day_dt, day_nq in grouped:
            if failed or passed:
                break

            day_es = rth_es.loc[day_nq.index]
            day_start_balance = balance
            daily_realized_loss = 0.0

            for i in range(len(day_nq)):
                ts = day_nq.index[i]
                p_nq_cur = float(day_nq["Close"].iloc[i])
                p_es_cur = float(day_es["Close"].iloc[i])
                bar_time_str = ts.strftime("%H:%M")

                n_prior = global_bar_idx
                global_bar_idx += 1

                if n_prior < min_calib:
                    continue

                calib_start_idx = max(0, n_prior - k_calib)
                calib_nq = rth_nq.iloc[calib_start_idx:n_prior]["Close"].values
                calib_es = rth_es.iloc[calib_start_idx:n_prior]["Close"].values

                log_calib_nq = np.log(calib_nq)
                log_calib_es = np.log(calib_es)

                # 1. Cointegración OLS
                beta, alpha, res_hist = self.model.estimate_cointegration(log_calib_nq, log_calib_es)

                # 2. Proceso OU
                ou = self.model.fit_ou_process(res_hist, dt_minutes=self.strategy.params["dt_minutes"])

                # 3. S-score actual
                res_cur = np.log(p_nq_cur) - (alpha + beta * np.log(p_es_cur))
                s_score = self.model.calculate_s_score(res_cur, ou["theta"], ou["sigma_eq"]) if ou["is_valid"] else 0.0

                # Auditoría cuantitativa continua de parámetros y condiciones de cointegración
                audit_bars_evaluated += 1
                is_stat = False
                if ou["is_valid"]:
                    audit_valid_ou += 1
                    audit_betas.append(beta)
                    audit_taus_min.append(ou["tau_half_min"])
                    audit_sigmas.append(ou["sigma_eq"])
                    tau_ok = (self.strategy.params["tau_min"] <= ou["tau_half_min"] <= self.strategy.params["tau_max"])
                    if tau_ok:
                        audit_tau_passed += 1

                    is_stat, _, _ = self.model.test_stationarity(
                        res_hist,
                        t_stat_threshold=self.strategy.params["adf_t_threshold"],
                        p_val_threshold=self.strategy.params["adf_p_threshold"]
                    )
                    if is_stat:
                        audit_adf_passed += 1
                    if is_stat and tau_ok:
                        audit_both_passed += 1

                    exact_ratio = self.model.calculate_dollar_beta_ratio(p_nq_cur, p_es_cur, beta, m_nq, m_es)
                    sub_es_cnt = max(1, int(round(1 * exact_ratio)))
                    sub_fric = 1 * comm_nq + sub_es_cnt * comm_es + 2.0 * self.strategy.params["slippage_ticks"] * (tick_nq * m_nq + tick_es * m_es * sub_es_cnt)
                    latest_sl_sub_usd = self.model.calculate_unit_stop_loss_loss(
                        ou["sigma_eq"], p_nq_cur, 1, m_nq, sub_fric,
                        self.strategy.params["s_stop"] - self.strategy.params["s_open"]
                    )
                    can_es_cnt = max(1, int(round(unit_nq * exact_ratio)))
                    can_fric = unit_nq * comm_nq + can_es_cnt * comm_es + 2.0 * self.strategy.params["slippage_ticks"] * (tick_nq * m_nq * unit_nq + tick_es * m_es * can_es_cnt)
                    latest_sl_std_usd = self.model.calculate_unit_stop_loss_loss(
                        ou["sigma_eq"], p_nq_cur, unit_nq, m_nq, can_fric,
                        self.strategy.params["s_stop"] - self.strategy.params["s_open"]
                    )

                # --------------------------------------------------------------
                # GESTIÓN DE POSICIÓN ABIERTA (SALIDAS Y GESTIÓN DE RIESGO)
                # --------------------------------------------------------------
                if in_position and current_trade is not None:
                    current_trade.bars_held += 1
                    current_trade.duration_minutes += self.strategy.params["dt_minutes"]
                    elapsed_min = current_trade.duration_minutes

                    # Monitoreo de equity flotante y riesgo intradía
                    pts_nq_float = (p_nq_cur - current_trade.entry_price_nq) * current_trade.direction
                    pts_es_float = (p_es_cur - current_trade.entry_price_es) * (-current_trade.direction)
                    unrealized_gross = (
                        pts_nq_float * m_nq * current_trade.contracts_nq +
                        pts_es_float * m_es * current_trade.contracts_es
                    )
                    friction_est = (
                        current_trade.contracts_nq * comm_nq +
                        current_trade.contracts_es * comm_es +
                        2.0 * self.strategy.params["slippage_ticks"] * tick_nq * m_nq * current_trade.contracts_nq +
                        2.0 * self.strategy.params["slippage_ticks"] * tick_es * m_es * current_trade.contracts_es
                    )
                    unrealized_net = unrealized_gross - friction_est
                    unrealized_bal = balance + unrealized_net
                    unrealized_loss_today = daily_realized_loss + max(0.0, -unrealized_net)

                    if self.trailing_mode == "intraday" and unrealized_bal > peak:
                        peak = unrealized_bal
                        floor = self._update_floor(peak)

                    exit_triggered = False
                    exit_reason_code: Optional[ExitReason] = None

                    # 0. Disyuntor de riesgo intradía (Límites Prop Trading DLL / TMDD)
                    if unrealized_loss_today >= self.daily_loss_limit or unrealized_bal <= floor:
                        exit_triggered = True
                        exit_reason_code = ExitReason.RISK_BREACH

                    # 1. Take-Profit (Sec 4.1)
                    if not exit_triggered:
                        if current_trade.direction == -1 and s_score <= self.strategy.params["s_close"]:
                            exit_triggered = True
                            exit_reason_code = ExitReason.TAKE_PROFIT
                        elif current_trade.direction == 1 and s_score >= -self.strategy.params["s_close"]:
                            exit_triggered = True
                            exit_reason_code = ExitReason.TAKE_PROFIT

                    # 2. Stop-Loss Estructural (Sec 4.2)
                    if not exit_triggered:
                        if current_trade.direction == -1 and s_score >= self.strategy.params["s_stop"]:
                            exit_triggered = True
                            exit_reason_code = ExitReason.STOP_LOSS
                        elif current_trade.direction == 1 and s_score <= -self.strategy.params["s_stop"]:
                            exit_triggered = True
                            exit_reason_code = ExitReason.STOP_LOSS

                    # 3. Time-Decay (Sec 4.3)
                    if not exit_triggered:
                        max_time = self.strategy.params["time_decay_multiplier"] * current_trade.entry_tau_half
                        if elapsed_min >= max_time:
                            exit_triggered = True
                            exit_reason_code = ExitReason.TIME_DECAY

                    # 4. Cierre forzoso intradía (Sec 4.4 - EOD Flat antes de 15:50 o fin de sesión)
                    if not exit_triggered and (bar_time_str >= self.strategy.params["eod_flat"] or i == len(day_nq) - 1):
                        exit_triggered = True
                        exit_reason_code = ExitReason.EOD_FLAT

                    # Ejecución de la salida
                    if exit_triggered:
                        exit_price_nq = p_nq_cur
                        exit_price_es = p_es_cur

                        pts_nq = (exit_price_nq - current_trade.entry_price_nq) * current_trade.direction
                        pts_es = (exit_price_es - current_trade.entry_price_es) * (-current_trade.direction)

                        gross_pnl_nq = pts_nq * m_nq * current_trade.contracts_nq
                        gross_pnl_es = pts_es * m_es * current_trade.contracts_es
                        gross_pnl = gross_pnl_nq + gross_pnl_es

                        commissions = (
                            current_trade.contracts_nq * comm_nq +
                            current_trade.contracts_es * comm_es
                        )
                        slippage = (
                            2.0 * self.strategy.params["slippage_ticks"] * tick_nq * m_nq * current_trade.contracts_nq +
                            2.0 * self.strategy.params["slippage_ticks"] * tick_es * m_es * current_trade.contracts_es
                        )
                        friction = commissions + slippage
                        net_pnl = gross_pnl - friction

                        balance += net_pnl
                        if net_pnl < 0:
                            daily_realized_loss += abs(net_pnl)

                        if self.trailing_mode == "intraday" and balance > peak:
                            peak = balance
                            floor = self._update_floor(peak)

                        current_dd = max(0.0, peak - balance)

                        current_trade.exit_date = ts.strftime("%Y-%m-%d")
                        current_trade.exit_time = ts.strftime("%H:%M:%S")
                        current_trade.exit_price_nq = round(exit_price_nq, 2)
                        current_trade.exit_price_es = round(exit_price_es, 2)
                        current_trade.exit_s_score = round(s_score, 3)
                        current_trade.exit_reason = exit_reason_code
                        current_trade.gross_pnl_usd = round(gross_pnl, 2)
                        current_trade.commissions_usd = round(commissions, 2)
                        current_trade.slippage_usd = round(slippage, 2)
                        current_trade.friction_usd = round(friction, 2)
                        current_trade.net_pnl_usd = round(net_pnl, 2)
                        current_trade.running_balance = round(balance, 2)
                        current_trade.running_drawdown = round(current_dd, 2)

                        trades.append(current_trade)
                        in_position = False
                        current_trade = None

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

                        # Evitar re-apertura en la misma barra donde se ejecutó la salida
                        continue

                # --------------------------------------------------------------
                # GESTIÓN DE ENTRADAS (SEÑALES)
                # --------------------------------------------------------------
                if not in_position and not failed:
                    in_window = (self.strategy.params["entry_start"] <= bar_time_str <= self.strategy.params["entry_end"])
                    valid_ou = ou["is_valid"]
                    valid_tau = (self.strategy.params["tau_min"] <= ou["tau_half_min"] <= self.strategy.params["tau_max"]) if valid_ou else False

                    if in_window and valid_ou and valid_tau:
                        # Test ADF solo cuando hay candidato potencial dentro de umbrales admisibles
                        signal_type = TradeSignal.NEUTRAL
                        if self.strategy.params["s_open"] < s_score < self.strategy.params["s_stop"]:
                            signal_type = TradeSignal.SELL_SPREAD
                        elif -self.strategy.params["s_stop"] < s_score < -self.strategy.params["s_open"]:
                            signal_type = TradeSignal.BUY_SPREAD

                        if signal_type != TradeSignal.NEUTRAL:
                            if is_stat:
                                # 1. Presupuesto de riesgo Prop Trading (Sec 5.3)
                                # tmdd_cushion es el colchón directo balance - floor
                                tmdd_cushion = max(0.0, balance - floor)
                                daily_cushion = max(0.0, self.daily_loss_limit - daily_realized_loss)
                                risk_budget = min(
                                    self.strategy.params["phi_dll"] * daily_cushion,
                                    self.strategy.params.get("phi_tmdd", 0.20) * tmdd_cushion
                                )
                                risk_budget = float(max(0.0, risk_budget))

                                # 2. Ratio analítico Dollar-Beta (Sec 5.1 & RUB-F4-04)
                                exact_ratio = self.model.calculate_dollar_beta_ratio(
                                    p_nq_cur, p_es_cur, beta, m_nq, m_es
                                )

                                # 3. Cesta unitaria base y pérdida en stop-loss (Sec 5.4)
                                cur_unit_nq = unit_nq
                                if self.strategy.params.get("dynamic_basket_ratio", True):
                                    cur_unit_es = max(1, int(round(cur_unit_nq * exact_ratio)))
                                else:
                                    cur_unit_es = unit_es

                                friction_unit = (
                                    cur_unit_nq * comm_nq + cur_unit_es * comm_es +
                                    2.0 * self.strategy.params["slippage_ticks"] * tick_nq * m_nq * cur_unit_nq +
                                    2.0 * self.strategy.params["slippage_ticks"] * tick_es * m_es * cur_unit_es
                                )

                                unit_sl_loss = self.model.calculate_unit_stop_loss_loss(
                                    sigma_eq=ou["sigma_eq"],
                                    p_nq=p_nq_cur,
                                    n_nq_unit=cur_unit_nq,
                                    m_nq=m_nq,
                                    friction_basket_usd=friction_unit,
                                    delta_s_sl=(self.strategy.params["s_stop"] - self.strategy.params["s_open"])
                                )

                                # 4. Asignación de cestas y blindaje por volatilidad (Sec 5.5)
                                n_baskets = self.model.calculate_baskets_allocation(
                                    risk_budget_usd=risk_budget,
                                    unit_sl_loss_usd=unit_sl_loss,
                                    n_basket_max=self.strategy.params["n_basket_max"]
                                )

                                # Si el riesgo de la cesta base excede el presupuesto pero
                                # allow_sub_baskets está habilitado, evaluar la sub-cesta calibrada
                                if n_baskets == 0 and self.strategy.params.get("allow_sub_baskets", False) and is_micro:
                                    sub_unit_nq = 1
                                    if self.strategy.params.get("dynamic_basket_ratio", True):
                                        sub_unit_es = max(1, int(round(sub_unit_nq * exact_ratio)))
                                    else:
                                        sub_unit_es = 2

                                    sub_friction = (
                                        sub_unit_nq * comm_nq + sub_unit_es * comm_es +
                                        2.0 * self.strategy.params["slippage_ticks"] * tick_nq * m_nq * sub_unit_nq +
                                        2.0 * self.strategy.params["slippage_ticks"] * tick_es * m_es * sub_unit_es
                                    )
                                    sub_sl_loss = self.model.calculate_unit_stop_loss_loss(
                                        sigma_eq=ou["sigma_eq"],
                                        p_nq=p_nq_cur,
                                        n_nq_unit=sub_unit_nq,
                                        m_nq=m_nq,
                                        friction_basket_usd=sub_friction,
                                        delta_s_sl=(self.strategy.params["s_stop"] - self.strategy.params["s_open"])
                                    )
                                    n_sub = self.model.calculate_baskets_allocation(
                                        risk_budget_usd=risk_budget,
                                        unit_sl_loss_usd=sub_sl_loss,
                                        n_basket_max=self.strategy.params["n_basket_max"] * 10
                                    )
                                    if n_sub >= 1:
                                        cur_unit_nq = sub_unit_nq
                                        cur_unit_es = sub_unit_es
                                        n_baskets = n_sub

                                if n_baskets >= 1:
                                    trade_counter += 1
                                    tot_contracts_nq = cur_unit_nq * n_baskets
                                    tot_contracts_es = cur_unit_es * n_baskets

                                    in_position = True
                                    current_trade = TradeRecord(
                                        trade_id=trade_counter,
                                        entry_date=ts.strftime("%Y-%m-%d"),
                                        entry_time=ts.strftime("%H:%M:%S"),
                                        signal="BUY_SPREAD" if signal_type == TradeSignal.BUY_SPREAD else "SELL_SPREAD",
                                        direction=1 if signal_type == TradeSignal.BUY_SPREAD else -1,
                                        contracts_nq=tot_contracts_nq,
                                        contracts_es=tot_contracts_es,
                                        symbol_nq=sym_nq,
                                        symbol_es=sym_es,
                                        n_baskets=n_baskets,
                                        entry_price_nq=p_nq_cur,
                                        entry_price_es=p_es_cur,
                                        entry_s_score=round(s_score, 3),
                                        entry_beta=round(beta, 4),
                                        entry_tau_half=round(ou["tau_half_min"], 2),
                                    )

            if self.trailing_mode == "eod" and balance > peak:
                peak = balance
                floor = self._update_floor(peak)

            day_pnl = balance - day_start_balance
            daily_logs.append({
                "date": day_dt.strftime("%Y-%m-%d") if hasattr(day_dt, "strftime") else str(day_dt),
                "day_pnl": round(day_pnl, 2),
                "ending_balance": round(balance, 2),
                "peak": round(peak, 2),
                "floor": round(floor, 2),
                "daily_realized_loss": round(daily_realized_loss, 2),
            })

            if (balance - self.account_size) >= self.profit_target and len(daily_logs) >= self.min_days:
                passed = True
                pass_date = day_dt

        # Cierre forzoso de seguridad si quedó alguna posición viva al terminar los datos
        if in_position and current_trade is not None:
            last_ts = rth_nq.index[-1]
            last_p_nq = float(rth_nq["Close"].iloc[-1])
            last_p_es = float(rth_es["Close"].iloc[-1])
            pts_nq = (last_p_nq - current_trade.entry_price_nq) * current_trade.direction
            pts_es = (last_p_es - current_trade.entry_price_es) * (-current_trade.direction)
            gross = pts_nq * m_nq * current_trade.contracts_nq + pts_es * m_es * current_trade.contracts_es
            fric = (
                current_trade.contracts_nq * comm_nq + current_trade.contracts_es * comm_es +
                2.0 * self.strategy.params["slippage_ticks"] * tick_nq * m_nq * current_trade.contracts_nq +
                2.0 * self.strategy.params["slippage_ticks"] * tick_es * m_es * current_trade.contracts_es
            )
            net = gross - fric
            balance += net
            current_trade.exit_date = last_ts.strftime("%Y-%m-%d")
            current_trade.exit_time = last_ts.strftime("%H:%M:%S")
            current_trade.exit_price_nq = round(last_p_nq, 2)
            current_trade.exit_price_es = round(last_p_es, 2)
            current_trade.exit_reason = ExitReason.EOD_FLAT
            current_trade.gross_pnl_usd = round(gross, 2)
            current_trade.commissions_usd = round(current_trade.contracts_nq * comm_nq + current_trade.contracts_es * comm_es, 2)
            current_trade.slippage_usd = round(fric - (current_trade.contracts_nq * comm_nq + current_trade.contracts_es * comm_es), 2)
            current_trade.friction_usd = round(fric, 2)
            current_trade.net_pnl_usd = round(net, 2)
            current_trade.running_balance = round(balance, 2)
            trades.append(current_trade)
            in_position = False
            current_trade = None

        trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame()
        daily_df = pd.DataFrame(daily_logs) if daily_logs else pd.DataFrame()
        metrics = self._calculate_metrics(trades_df, balance)

        prop_evaluation = {
            "status": "PASS" if passed else ("FAIL" if failed else "INCOMPLETE"),
            "reason": fail_reason if failed else ("Profit Target Achieved" if passed else "Evaluation Ongoing"),
            "date": str(pass_date) if passed else (str(fail_date) if failed else None),
            "final_balance": round(balance, 2),
            "net_profit": round(balance - self.account_size, 2),
            "peak_equity": round(peak, 2),
            "final_floor": round(floor, 2),
            "total_days_evaluated": len(daily_logs),
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
            "mean_tau_half_days": round(float(np.mean(audit_taus_min) / 390.0), 2) if audit_taus_min else 0.0,
            "mean_sigma_eq": round(float(np.mean(audit_sigmas)), 6) if audit_sigmas else 0.0,
            "latest_sub_basket_sl_usd": round(latest_sl_sub_usd, 2),
            "latest_standard_basket_sl_usd": round(latest_sl_std_usd, 2),
        }

        return {
            "metrics": metrics,
            "prop_evaluation": prop_evaluation,
            "trades": trades_df,
            "daily_logs": daily_df,
            "cointegration_audit": cointegration_audit,
        }

    def _calculate_metrics(
        self,
        trades_df: pd.DataFrame,
        final_balance: float
    ) -> Dict[str, Any]:
        """Calcula las métricas financieras clave de desempeño del portafolio."""
        if trades_df.empty:
            return {
                "Total Trades": 0,
                "Win Rate (%)": 0.0,
                "Profit Factor": 0.0,
                "Net Profit ($)": 0.0,
                "Total Return (%)": 0.0,
                "Max Drawdown ($)": 0.0,
                "Sharpe Ratio": 0.0,
                "Sortino Ratio": 0.0,
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

        returns_per_trade = pnl / self.account_size
        std_ret = float(returns_per_trade.std(ddof=1)) if len(returns_per_trade) > 1 else 0.0
        sharpe = (float(returns_per_trade.mean()) / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

        downside = returns_per_trade[returns_per_trade < 0]
        downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        sortino = (float(returns_per_trade.mean()) / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

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
            "Max Drawdown ($)": round(max_dd, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
            "Exit Reasons": exit_counts,
        }
