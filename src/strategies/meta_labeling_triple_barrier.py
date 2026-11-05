"""
src/strategies/meta_labeling_triple_barrier.py

Implementación Matemática y Microestructural Formal de la Estrategia F7:
Meta-Labeling y Método de Triple Barrera para Futuros CME (ES, MES, NQ, MNQ).

Referencias Académicas:
1. López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
   Capítulo 3 ("Targeting: The Triple-Barrier Method") y Capítulo 5 ("Fractionally Differentiated Features").
2. López de Prado, M. (2020). Machine Learning for Asset Managers. Cambridge University Press.
   pp. 18-37.

Especificación Matemática de Referencia:
    reports/estrategias_matematicas/F7_meta_labeling_triple_barrier.md

Reglas Principales:
1. Nivel 1 (Modelo Primario): Emite señal direccional s_t^{(1)} in {-1, 0, +1} basada en
   dinámicas de microestructura (divergencia VWAP, desbalance de flujo OFI y compresión).
2. Nivel 2 (Meta-Modelo de Clasificación Probabilística): Estima p_t = P(y_t = 1 | X_t, s_t^{(1)})
   mediante función logística multivariada sobre vector microestructural X_t in R^6.
3. Función Analítica de Veto: Si p_t < tau_meta (0.60), la señal se veta (S_t^* = 0).
4. Modulación de Apuesta Continua (Bet Sizing): m(p_t) calculado vía CDF Gaussiana o polinómica cóncava.
5. Método de Triple Barrera:
   - Barrera Superior B_upper (Take Profit Long / Stop Loss Short)
   - Barrera Inferior B_lower (Stop Loss Long / Take Profit Short)
   - Barrera Vertical Temporal T_vertical (min(t_0 + Delta_tau, tau_close))
6. Gestión de Riesgo Prop Trading:
   - Daily Loss Limit (DLL) con cota gamma_D = 0.20
   - Maximum Trailing Drawdown (MTDD) con cota gamma_T = 0.08
   - Conmutación automática a microcontratos (MES/MNQ) si N_std = 0
   - Disyuntor de emergencia por drawdown (cushion <= 0.15 * MTDD)
   - Cierre intradiario obligatorio a las 15:55:00 EST.
"""

from dataclasses import dataclass, field
from datetime import time, datetime, timedelta
from enum import Enum
import math
from typing import Dict, Any, List, Optional, Tuple, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from src.core.base_strategy import BaseStrategy
from src.core.futures_specs import get_spec


# ==============================================================================
# 1. ENUMS Y ESTRUCTURAS DE DATOS DE ESTADO Y EJECUCIÓN
# ==============================================================================

class PositionDirection(int, Enum):
    """Sentido direccional de la posición o señal."""
    LONG = 1
    FLAT = 0
    SHORT = -1


class ExitReason(str, Enum):
    """Causa analítica de cierre de posición según Sección 4.3."""
    TAKE_PROFIT = "TP"
    STOP_LOSS = "SL"
    TIME_BARRIER = "TIME"
    EOD_FORCED = "EOD"
    CIRCUIT_BREAKER = "HALT_DRAWDOWN"


class StrategyState(str, Enum):
    """Estados del ciclo de vida intradiario de la estrategia."""
    WARMUP = "WARMUP"
    MONITORING = "MONITORING"
    IN_POSITION = "IN_POSITION"
    COOLDOWN = "COOLDOWN"
    HALTED_DAILY_LOSS = "HALTED_DAILY_LOSS"
    HALTED_DRAWDOWN = "HALTED_DRAWDOWN"


@dataclass
class MicroFeatures:
    """Vector de características microestructurales X_t in R^6 (Sección 2.2)."""
    sigma: float                # Volatilidad local instantánea
    z_ofi: float                # Z-score de Order Flow Imbalance
    delta_vwap: float           # Distancia a VWAP acumulado: P_t - VWAP_t
    rvol: float                 # Ratio de volumen relativo institucional
    range_compression: float    # Compresión de rango reciente frente a media 5d
    tau_tod: float              # Fracción temporal transcurrida en sesión RTH [0, 1]

    def to_array(self) -> np.ndarray:
        return np.array([
            self.sigma,
            self.z_ofi,
            self.delta_vwap,
            self.rvol,
            self.range_compression,
            self.tau_tod
        ], dtype=float)


@dataclass
class TradeRecord:
    """Registro auditable de operación completada en el backtester."""
    trade_id: int
    date: str
    symbol: str
    direction: int                      # +1 Long, -1 Short
    contracts: int                     # Número de contratos operados
    is_micro: bool                     # True si operó con microcontratos
    entry_time: str
    entry_price: float
    b_upper: float
    b_lower: float
    t_vertical: str
    meta_prob: float                   # Probabilidad p_t estimada por el meta-modelo
    bet_size: float                    # Modulador m(p_t)
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


# ==============================================================================
# 2. CONTROL DE RIESGO CUANTITATIVO PARA PROP TRADING (SECCIÓN 5)
# ==============================================================================

@dataclass
class PropRiskManager:
    """
    Gestión de balance y límites de capital de Prop Trading (Topstep, Apex, Lucid, MFF).
    Aplica rigurosamente las ecuaciones de la Sección 5:
    - DLL: Límite diario disponible Delta DLL_t
    - MTDD: Colchón de drawdown trailing D_t^{cushion}
    - Presupuesto de riesgo por operación B_{risk, t0}
    - Conmutación automática a microcontratos
    - Disyuntor analítico por vulnerabilidad de trailing drawdown (cushion <= 0.15 MTDD).
    """
    initial_capital: float = 50000.0
    dll: float = 1000.0                 # Daily Loss Limit ($1000)
    mtdd: float = 2000.0                # Max Trailing Drawdown ($2000)
    gamma_d: float = 0.20               # Fracción máxima de DLL por trade (20%)
    gamma_t: float = 0.08               # Fracción máxima de MTDD cushion por trade (8%)
    circuit_breaker_ratio: float = 0.15 # Umbral crítico de parada de emergencia (15% MTDD)
    max_contracts_std: int = 5          # Límite máximo de contratos estándar
    
    equity: float = field(init=False)
    peak_equity: float = field(init=False)
    daily_start_equity: float = field(init=False)
    daily_realized_loss: float = 0.0
    is_halted: bool = False
    halt_reason: Optional[str] = None

    def __post_init__(self):
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.daily_start_equity = self.initial_capital

    def start_new_session(self, current_date: Any) -> None:
        """Reinicia contadores diarios al comienzo de una nueva sesión bursátil."""
        self.daily_start_equity = self.equity
        self.daily_realized_loss = 0.0
        # Si estaba suspendido por pérdida diaria (no trailing definitivo), reanuda
        if self.is_halted and "DAILY_LOSS_LIMIT" in str(self.halt_reason):
            self.is_halted = False
            self.halt_reason = None

    @property
    def current_drawdown(self) -> float:
        """Drawdown acumulado desde el High Water Mark."""
        return max(0.0, self.peak_equity - self.equity)

    @property
    def delta_dll(self) -> float:
        """Margen de pérdida diaria restante: Delta DLL_t = max(DLL - (E_{d,0} - E_t), 0)."""
        loss_today = max(0.0, self.daily_start_equity - self.equity)
        return max(0.0, self.dll - loss_today)

    @property
    def dd_cushion(self) -> float:
        """Colchón de trailing drawdown: D_t^{cushion} = max(E_t - (H_t - MTDD), 0)."""
        return max(0.0, self.equity - (self.peak_equity - self.mtdd))

    def check_circuit_breaker(self) -> bool:
        """
        Disyuntor analítico por vulnerabilidad de Trailing Drawdown (Sección 5.6):
        D_t^{cushion} <= 0.15 * MTDD => HALT_DRAWDOWN_DEFENSE.
        """
        if self.dd_cushion <= self.circuit_breaker_ratio * self.mtdd:
            self.is_halted = True
            self.halt_reason = (
                f"HALT_DRAWDOWN_DEFENSE: Colchón de drawdown ${self.dd_cushion:.2f} "
                f"<= {self.circuit_breaker_ratio * 100:.0f}% de MTDD (${self.mtdd:.2f})"
            )
            return True
        return False

    def calculate_risk_budget(self) -> float:
        """
        Presupuesto dinámico de riesgo por trade (Sección 5.2):
        B_{risk, t0} = min(gamma_D * Delta DLL_t0, gamma_T * D_{t0}^{cushion}).
        """
        if self.is_halted or self.check_circuit_breaker():
            return 0.0
        b_risk = min(self.gamma_d * self.delta_dll, self.gamma_t * self.dd_cushion)
        return max(0.0, b_risk)

    def calculate_position_size(
        self,
        std_symbol: str,
        micro_symbol: str,
        sigma: float,
        ell_minus: float,
        bet_size: float,
        slippage_ticks: float = 1.0
    ) -> Tuple[int, bool, float]:
        """
        Calcula el número de contratos a emitir aplicando Sección 5.3, 5.4 y 5.5:
        1. Riesgo unitario estándar y micro (pérdida en SL + fricción completa).
        2. Número de contratos estándar N_std.
        3. Si N_std == 0, conmutación automática a microcontratos.
        
        Returns:
            Tuple[num_contracts, is_micro, unit_risk_usd]
        """
        if bet_size <= 0.0 or self.is_halted:
            return 0, False, 0.0

        b_risk = self.calculate_risk_budget()
        effective_budget = b_risk * bet_size
        if effective_budget <= 0.0:
            return 0, False, 0.0

        # Especificaciones CME
        spec_std = get_spec(std_symbol)
        spec_micro = get_spec(micro_symbol)

        m_std = spec_std["multiplier"]
        m_micro = spec_micro["multiplier"]
        tick_std = spec_std["tick"]
        tick_micro = spec_micro["tick"]
        comm_std = spec_std["commission_rt"]
        comm_micro = spec_micro["commission_rt"]

        # Fricción total: 2 * slippage_ticks * tick * multiplier + comisión RT
        fric_std = (2.0 * slippage_ticks * tick_std * m_std) + comm_std
        fric_micro = (2.0 * slippage_ticks * tick_micro * m_micro) + comm_micro

        # Pérdida unitaria en stop loss
        delta_l_std = (ell_minus * sigma * m_std) + fric_std
        delta_l_micro = (ell_minus * sigma * m_micro) + fric_micro

        if delta_l_std <= 0.0 or delta_l_micro <= 0.0:
            return 0, False, 0.0

        # Contratos estándar
        n_std = min(int(math.floor(effective_budget / delta_l_std)), self.max_contracts_std)

        if n_std >= 1:
            return n_std, False, delta_l_std

        # Conmutación a Microcontratos si N_std == 0
        if effective_budget >= delta_l_micro:
            max_micro = 10 * self.max_contracts_std
            n_micro = min(int(math.floor(effective_budget / delta_l_micro)), max_micro)
            if n_micro >= 1:
                return n_micro, True, delta_l_micro

        # Bloqueo defensivo absoluto
        return 0, False, 0.0

    def record_completed_trade(self, trade: TradeRecord) -> None:
        """Actualiza la equidad, HWM, drawdown y evalúa límites contractuales."""
        self.equity += trade.net_pnl_usd
        # Pérdida neta del día según Sección 5.1: max(0.0, E_{d, 0} - E_t)
        daily_loss = max(0.0, self.daily_start_equity - self.equity)
        self.daily_realized_loss = daily_loss

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Verificar violación de DLL o MTDD
        if (self.peak_equity - self.equity) >= self.mtdd:
            self.is_halted = True
            self.halt_reason = f"MTDD_BREACH: Drawdown ${self.peak_equity - self.equity:.2f} >= ${self.mtdd:.2f}"
        elif daily_loss >= self.dll:
            self.is_halted = True
            self.halt_reason = f"DAILY_LOSS_LIMIT_BREACH: Pérdida del día ${daily_loss:.2f} >= ${self.dll:.2f}"
        else:
            self.check_circuit_breaker()

        trade.running_equity = self.equity
        trade.running_drawdown = self.current_drawdown


# ==============================================================================
# 3. FUNCIONES MATEMÁTICAS FORMALES DE CARACTERÍSTICAS Y META-LABELING
# ==============================================================================

def compute_garman_klass_volatility(
    df: pd.DataFrame,
    window: int = 12,
    fallback_window: int = 12
) -> pd.Series:
    """
    Estimador de volatilidad local instantánea de Garman-Klass (1980) adaptado a puntos:
    sigma_{GK}^2 = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2.
    Se expresa en puntos de índice multiplicando por el precio de cierre.
    """
    eps = 1e-8
    log_hl = np.log(np.maximum(df["High"], eps) / np.maximum(df["Low"], eps))
    log_co = np.log(np.maximum(df["Close"], eps) / np.maximum(df["Open"], eps))

    gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    gk_var = np.maximum(gk_var, 0.0)

    # Varianza rodante
    gk_rolling = pd.Series(gk_var, index=df.index).rolling(window=window, min_periods=2).mean()
    gk_rolling = gk_rolling.bfill().fillna(0.0)

    sigma_pct = np.sqrt(gk_rolling)
    sigma_points = sigma_pct * df["Close"]

    # Asegurar cota inferior mínima razonable (e.g. 1 tick = 0.25 pt o 0.5 pt)
    min_sigma = df["High"].rolling(window=fallback_window, min_periods=2).max() - \
                df["Low"].rolling(window=fallback_window, min_periods=2).min()
    min_sigma = (min_sigma / float(fallback_window)).fillna(1.0)
    min_sigma = np.maximum(min_sigma, 0.50)

    return pd.Series(np.maximum(sigma_points, min_sigma), index=df.index)


def compute_intraday_anchored_vwap(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Calcula el VWAP anclado a cada sesión intradiaria y la distancia P_t - VWAP_t.
    """
    dates = df.index.date
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = np.maximum(df["Volume"], 1.0)
    pv = typical_price * vol

    df_tmp = pd.DataFrame({"pv": pv, "vol": vol, "close": df["Close"], "date": dates}, index=df.index)
    cum_pv = df_tmp.groupby("date")["pv"].cumsum()
    cum_vol = df_tmp.groupby("date")["vol"].cumsum()

    vwap = cum_pv / np.maximum(cum_vol, 1e-8)
    delta_vwap = df["Close"] - vwap
    return vwap, delta_vwap


def compute_order_flow_imbalance_zscore(
    df: pd.DataFrame,
    window: int = 20
) -> pd.Series:
    """
    Estimación del Order Flow Imbalance (OFI) intradiario discreto a partir de barras:
    OFI_t = Volume_t * sign(Close_t - Close_{t-1}).
    Z_OFI = (OFI_t - mean(OFI)) / (std(OFI) + eps).
    """
    diff_close = df["Close"].diff().fillna(0.0)
    sign_dir = np.sign(diff_close)
    # Si la barra no cambia de precio, asignamos signo según relación con Open
    zero_mask = (sign_dir == 0.0)
    sign_dir[zero_mask] = np.sign(df["Close"][zero_mask] - df["Open"][zero_mask])

    ofi = df["Volume"] * sign_dir
    ofi_mean = ofi.rolling(window=window, min_periods=3).mean()
    ofi_std = ofi.rolling(window=window, min_periods=3).std().replace(0.0, np.nan).fillna(1.0)

    z_score = (ofi - ofi_mean) / (ofi_std + 1e-8)
    # Acotamos Z-score en [-4.0, 4.0] para estabilidad numérica
    return pd.Series(np.clip(z_score.fillna(0.0), -4.0, 4.0), index=df.index)


def compute_relative_volume(
    df: pd.DataFrame,
    rolling_window: int = 20
) -> pd.Series:
    """
    Calcula RVOL_t como cociente entre el volumen de la barra actual y su media histórica.
    """
    avg_vol = df["Volume"].rolling(window=rolling_window, min_periods=3).mean()
    rvol = df["Volume"] / np.maximum(avg_vol, 1.0)
    return pd.Series(np.clip(rvol.fillna(1.0), 0.1, 10.0), index=df.index)


def compute_range_compression(
    df: pd.DataFrame,
    recent_bars: int = 12,
    baseline_days: int = 5
) -> pd.Series:
    """
    Ratio de compresión de rango S_{range, t}:
    Rango reciente (ej. últimas 12 barras = 1 hora) dividido por el rango diario medio de 5 días.
    Utiliza estrictamente los días previos cerrados (shift 1) para prevenir filtración hacia el futuro.
    """
    recent_high = df["High"].rolling(window=recent_bars, min_periods=2).max()
    recent_low = df["Low"].rolling(window=recent_bars, min_periods=2).min()
    recent_range = recent_high - recent_low

    # Rango diario histórico causal (sin lookahead bias)
    daily_high = df.groupby(df.index.date)["High"].max()
    daily_low = df.groupby(df.index.date)["Low"].min()
    daily_range = daily_high - daily_low
    # Desplazar 1 día para basar la media de 5 días estrictamente en sesiones previas concluidas
    avg_5d_range = daily_range.shift(1).rolling(window=baseline_days, min_periods=1).mean().fillna(50.0)
    
    # Mapear rango medio diario al índice de las barras
    date_to_range = avg_5d_range.to_dict()
    mapped_daily_range = pd.Series([date_to_range.get(d, 50.0) for d in df.index.date], index=df.index)

    mapped_daily_range = np.maximum(mapped_daily_range, 5.0)

    compression = recent_range / mapped_daily_range
    return pd.Series(np.clip(compression.fillna(0.5), 0.05, 3.0), index=df.index)


def compute_session_fraction(df: pd.DataFrame, is_ct: Optional[bool] = None) -> pd.Series:
    """
    Fracción analítica del tiempo transcurrido en sesión regular RTH:
    - EST: [09:30, 16:00] (570 a 960 min)
    - CT:  [08:30, 15:00] (510 a 900 min)
    tau_tod in [0, 1].
    """
    if is_ct is None:
        is_ct = detect_data_time_offset(df)

    fractions = []
    if is_ct:
        t_start = 8 * 60 + 30     # 08:30 CT en minutos (510)
        t_end = 15 * 60           # 15:00 CT en minutos (900)
    else:
        t_start = 9 * 60 + 30     # 09:30 EST en minutos (570)
        t_end = 16 * 60           # 16:00 EST en minutos (960)
    total_minutes = t_end - t_start

    for dt in df.index:
        cur_min = dt.hour * 60 + dt.minute
        if cur_min < t_start:
            fractions.append(0.0)
        elif cur_min >= t_end:
            fractions.append(1.0)
        else:
            fractions.append((cur_min - t_start) / float(total_minutes))

    return pd.Series(fractions, index=df.index, dtype=float)


# ==============================================================================
# 4. CLASIFICADOR PROBABILÍSTICO DE META-LABELING (SECCIONES 3.2, 3.3, 3.4)
# ==============================================================================

@dataclass
class MetaModelWeights:
    """Pesos analíticos del meta-modelo de calidad (Sección 3.2)."""
    alpha_0: float = 0.05
    beta_sigma: float = -0.05
    beta_z_ofi: float = 0.25
    beta_vwap: float = 0.25
    beta_rvol: float = 0.20
    beta_compression: float = -0.15
    beta_tod: float = 0.05
    gamma_1: float = 0.45          # Sinergia s^{(1)} * Z_OFI
    gamma_2: float = 0.40          # Sinergia s^{(1)} * (Delta P_VWAP / sigma)

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.alpha_0,
            self.beta_sigma,
            self.beta_z_ofi,
            self.beta_vwap,
            self.beta_rvol,
            self.beta_compression,
            self.beta_tod,
            self.gamma_1,
            self.gamma_2
        ], dtype=float)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "MetaModelWeights":
        return cls(
            alpha_0=float(v[0]),
            beta_sigma=float(v[1]),
            beta_z_ofi=float(v[2]),
            beta_vwap=float(v[3]),
            beta_rvol=float(v[4]),
            beta_compression=float(v[5]),
            beta_tod=float(v[6]),
            gamma_1=float(v[7]),
            gamma_2=float(v[8])
        )


class MetaLabelingClassifier:
    """
    Clasificador probabilístico de Meta-Labeling según Sección 3.2.
    Evalúa F_meta(X_t, s_t^{(1)}) y genera p_t = 1 / (1 + exp(-F_meta)).
    Puede calibrarse analíticamente o ajustarse mediante máxima verosimilitud (cross-entropy con L2).
    """

    def __init__(self, weights: Optional[MetaModelWeights] = None):
        self.weights = weights or MetaModelWeights()

    def evaluate_logit(self, feat: MicroFeatures, primary_signal: int) -> float:
        """
        F_meta(X_t, s_t^{(1)}) = alpha_0 + sum(beta_k * X_k) + s_t^{(1)} * (gamma_1 * Z_OFI + gamma_2 * DeltaVWAP/sigma).
        """
        w = self.weights
        norm_vwap = feat.delta_vwap / max(feat.sigma, 1e-4)

        linear_part = (
            w.alpha_0
            + w.beta_sigma * (feat.sigma / 50.0)      # Escalamiento para estabilidad
            + w.beta_z_ofi * feat.z_ofi
            + w.beta_vwap * norm_vwap
            + w.beta_rvol * (feat.rvol - 1.0)
            + w.beta_compression * (feat.range_compression - 0.5)
            + w.beta_tod * (feat.tau_tod - 0.5)
        )

        synergy_part = float(primary_signal) * (
            w.gamma_1 * feat.z_ofi + w.gamma_2 * norm_vwap
        )

        return linear_part + synergy_part

    def predict_probability(self, feat: MicroFeatures, primary_signal: int) -> float:
        """
        p_t = 1 / (1 + exp(-F_meta)) in [0, 1].
        """
        logit = self.evaluate_logit(feat, primary_signal)
        # Acotamiento para prevenir overflow
        clipped_logit = float(np.clip(logit, -30.0, 30.0))
        return 1.0 / (1.0 + math.exp(-clipped_logit))

    def fit(
        self,
        features: List[MicroFeatures],
        signals: List[int],
        labels: List[int],
        l2_reg: float = 1.0
    ) -> "MetaLabelingClassifier":
        """
        Ajusta los parámetros del clasificador logístico multivariado con regularización Bayesiana L2
        hacia los pesos analíticos estructurales iniciales (MAP estimation).
        labels in {0, 1} proceden del etiquetado con el Método de Triple Barrera.
        """
        if len(features) < 10 or len(labels) != len(features):
            return self

        # Construir matriz de diseño Z donde cada fila es:
        # [1, sigma/50, z_ofi, norm_vwap, rvol-1, comp-0.5, tod-0.5, s*z_ofi, s*norm_vwap]
        z_rows = []
        for feat, s in zip(features, signals):
            norm_vwap = feat.delta_vwap / max(feat.sigma, 1e-4)
            row = [
                1.0,
                feat.sigma / 50.0,
                feat.z_ofi,
                norm_vwap,
                feat.rvol - 1.0,
                feat.range_compression - 0.5,
                feat.tau_tod - 0.5,
                float(s) * feat.z_ofi,
                float(s) * norm_vwap
            ]
            z_rows.append(row)

        z_mat = np.array(z_rows, dtype=float)
        y_vec = np.array(labels, dtype=float)

        n1 = max(float(np.sum(y_vec)), 1.0)
        n0 = max(float(len(y_vec) - n1), 1.0)
        w1 = len(y_vec) / (2.0 * n1)
        w0 = len(y_vec) / (2.0 * n0)
        sample_weights = np.where(y_vec == 1.0, w1, w0)

        prior_w = self.weights.to_vector()

        def loss_func(params: np.ndarray) -> float:
            logits = np.clip(np.dot(z_mat, params), -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            eps = 1e-12
            # Cross-entropy balanceada + regularización Bayesiana hacia el prior estructural
            ce = -np.mean(sample_weights * (y_vec * np.log(probs + eps) + (1.0 - y_vec) * np.log(1.0 - probs + eps)))
            reg = l2_reg * np.sum((params - prior_w) ** 2)
            return ce + reg

        initial_w = prior_w.copy()
        res = minimize(loss_func, initial_w, method="L-BFGS-B")
        if res.success:
            self.weights = MetaModelWeights.from_vector(res.x)

        return self



def calculate_bet_size(
    p_t: float,
    tau_meta: float = 0.60,
    method: str = "gaussian",
    eta: float = 1.50
) -> float:
    """
    Función continua de dimensionamiento de apuesta m(p_t) in [0, 1] (Sección 3.4).
    
    1. 'gaussian':
       z_t = (p_t - 0.50) / sqrt(p_t * (1 - p_t))
       m(p_t) = 2 * Phi(z_t) - 1 si p_t >= tau_meta, else 0
       
    2. 'polynomial':
       m(p_t) = ((p_t - tau_meta) / (1 - tau_meta))^eta * I(p_t >= tau_meta)
    """
    if p_t < tau_meta:
        return 0.0

    if method == "polynomial":
        factor = (p_t - tau_meta) / (1.0 - tau_meta)
        return float(np.clip(factor ** eta, 0.0, 1.0))

    # Formulación Gaussiana canónica de López de Prado (2018)
    p_clamped = min(max(p_t, 1e-4), 1.0 - 1e-4)
    z_t = (p_clamped - 0.50) / math.sqrt(p_clamped * (1.0 - p_clamped))
    m_val = 2.0 * norm.cdf(z_t) - 1.0
    return float(np.clip(m_val, 0.0, 1.0))


def apply_meta_veto(
    primary_signal: int,
    p_t: float,
    tau_meta: float = 0.60
) -> int:
    """
    Función analítica de veto y señal final filtrada (Sección 3.3):
    S_t^* = s_t^{(1)} * I(p_t >= tau_meta).
    
    Propiedades analíticas satisfechas:
    - S_{Long} * S_{Short} == 0
    - S_{Long} + S_{Short} + S_{Neutral} == 1
    """
    if primary_signal == 0 or p_t < tau_meta:
        return 0
    return primary_signal


# ==============================================================================
# 5. GEOMETRÍA ANALÍTICA DE LA TRIPLE BARRERA (SECCIÓN 4)
# ==============================================================================

def calculate_barriers(
    p_0: float,
    sigma_0: float,
    signal_star: int,
    ell_plus: float = 2.00,
    ell_minus: float = 1.00
) -> Tuple[float, float]:
    """
    Geometría Analítica de la Triple Barrera (Sección 4.1):
    Para Long (S^* = +1):
        B_upper = P_0 + ell^+ * sigma_0  (TP)
        B_lower = P_0 - ell^- * sigma_0  (SL)
    Para Short (S^* = -1):
        B_upper = P_0 + ell^- * sigma_0  (SL)
        B_lower = P_0 - ell^+ * sigma_0  (TP)
        
    Asimetría matemática estricta: ell^+ / ell^- = 2.00 / 1.00 = 2.0 >= 1.50.
    """
    if signal_star == 1:
        b_upper = p_0 + ell_plus * sigma_0
        b_lower = p_0 - ell_minus * sigma_0
    elif signal_star == -1:
        b_upper = p_0 + ell_minus * sigma_0
        b_lower = p_0 - ell_plus * sigma_0
    else:
        raise ValueError("Señal ejecutoria debe ser +1 (Long) o -1 (Short) para calcular barreras.")

    return b_upper, b_lower


def evaluate_triple_barrier_exit(
    signal_star: int,
    current_high: float,
    current_low: float,
    current_close: float,
    b_upper: float,
    b_lower: float,
    is_vertical_reached: bool,
    current_open: Optional[float] = None
) -> Optional[Tuple[ExitReason, float]]:
    """
    Evalúa si la barra actual impacta alguna de las tres barreras cerradas (Sección 4.2 y 4.3).
    
    Gestión realista de gaps y prioridad de toque intrabarra:
    - Si la barra abre con gap fuera de las barreras (current_open <= B_lower o >= B_upper),
      se ejecuta de inmediato en el Open (slippage por gap).
    - Si ambas barreras son penetradas en la misma barra (evento extremo de volatilidad),
      se asume Stop Loss prioritariamente por principio de prudencia cuantitativa.
    - Si se alcanza T_vertical y el precio está entre B_lower y B_upper => TIME.
    """
    if signal_star == 1:  # LONG
        # 1. Comprobación de gap en la apertura de la barra
        if current_open is not None:
            if current_open <= b_lower:
                return ExitReason.STOP_LOSS, current_open
            elif current_open >= b_upper:
                return ExitReason.TAKE_PROFIT, current_open

        hit_tp = (current_high >= b_upper)
        hit_sl = (current_low <= b_lower)

        if hit_tp and hit_sl:
            return ExitReason.STOP_LOSS, b_lower
        elif hit_sl:
            exit_px = min(current_open, b_lower) if current_open is not None else b_lower
            return ExitReason.STOP_LOSS, exit_px
        elif hit_tp:
            exit_px = max(current_open, b_upper) if current_open is not None else b_upper
            return ExitReason.TAKE_PROFIT, exit_px
        elif is_vertical_reached:
            return ExitReason.TIME_BARRIER, current_close

    elif signal_star == -1:  # SHORT
        # 1. Comprobación de gap en la apertura de la barra
        if current_open is not None:
            if current_open >= b_upper:
                return ExitReason.STOP_LOSS, current_open
            elif current_open <= b_lower:
                return ExitReason.TAKE_PROFIT, current_open

        hit_tp = (current_low <= b_lower)
        hit_sl = (current_high >= b_upper)

        if hit_tp and hit_sl:
            return ExitReason.STOP_LOSS, b_upper
        elif hit_sl:
            exit_px = max(current_open, b_upper) if current_open is not None else b_upper
            return ExitReason.STOP_LOSS, exit_px
        elif hit_tp:
            exit_px = min(current_open, b_lower) if current_open is not None else b_lower
            return ExitReason.TAKE_PROFIT, exit_px
        elif is_vertical_reached:
            return ExitReason.TIME_BARRIER, current_close

    return None


def label_historical_events(
    df: pd.DataFrame,
    primary_signals: pd.Series,
    sigma_series: pd.Series,
    ell_plus: float = 2.0,
    ell_minus: float = 1.0,
    delta_tau_bars: int = 6,  # 30 minutos en barras de 5m
    t_eod_close: Optional[time] = None,
    purge_last_bars: int = 0
) -> Tuple[List[int], List[int]]:
    """
    Etiqueta las señales primarias históricas mediante el Método de Triple Barrera:
    y = 1 si alcanza Take Profit antes de Stop Loss.
    y = 0 si alcanza Stop Loss o expira en tiempo sin ganancia.
    Respeta la barrera vertical T_vertical = min(t_0 + Delta_tau, tau_close) sin filtración entre sesiones.
    Si purge_last_bars > 0, purga las barras finales del dataset para evitar que las barreras
    miren hacia datos futuros fuera del conjunto de entrenamiento.
    
    Returns:
        Tuple[valid_indices, labels]
    """
    valid_indices = []
    labels = []
    n_rows = len(df)

    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    sigs = primary_signals.values
    sigmas = sigma_series.values
    dts = df.index

    # Limitar el rango de eventos excluyendo las últimas barras de purgado
    cutoff_idx = max(0, n_rows - 1 - max(0, purge_last_bars))

    for i in range(cutoff_idx):
        s = int(sigs[i])
        if s == 0:
            continue

        p0 = float(closes[i])
        sigma0 = float(sigmas[i])
        b_upper, b_lower = calculate_barriers(p0, sigma0, s, ell_plus, ell_minus)

        outcome_label = 0
        entry_date = dts[i].date()

        max_steps = min(delta_tau_bars, n_rows - 1 - i)
        for step in range(1, max_steps + 1):
            curr_idx = i + step
            curr_dt = dts[curr_idx]
            curr_time = curr_dt.time()
            curr_date = curr_dt.date()

            is_new_day = (curr_date != entry_date)
            is_eod = (t_eod_close is not None and curr_time >= t_eod_close)
            is_vert = (step == max_steps) or is_new_day or is_eod

            o = float(opens[curr_idx])
            h = float(highs[curr_idx])
            l = float(lows[curr_idx])
            c = float(closes[curr_idx])

            res = evaluate_triple_barrier_exit(
                s, h, l, c, b_upper, b_lower, is_vert, current_open=o
            )
            if res is not None:
                reason, exit_price = res
                if reason == ExitReason.TAKE_PROFIT:
                    outcome_label = 1
                elif reason in (ExitReason.TIME_BARRIER, ExitReason.EOD_FORCED):
                    ret = (exit_price - p0) * s
                    outcome_label = 1 if ret > 0 else 0
                else:
                    outcome_label = 0
                break

            if is_new_day or is_eod:
                break

        valid_indices.append(i)
        labels.append(outcome_label)

    return valid_indices, labels


def detect_data_time_offset(df: pd.DataFrame) -> bool:
    """
    Detecta si las barras intradía de CME provienen del horario local de Chicago (CT),
    donde la subasta abre a las 08:30 en vez de las 09:30 EST.
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return False
    times = df.index.time
    df_vol = df.groupby(times)["Volume"].mean()
    vol_0830 = df_vol.get(time(8, 30), 0.0)
    vol_0930 = df_vol.get(time(9, 30), 0.0)
    return bool(vol_0830 > vol_0930 * 1.3)



# ==============================================================================
# 6. ESTRATEGIA PRINCIPAL F7: META-LABELING TRIPLE BARRIER
# ==============================================================================

class MetaLabelingTripleBarrierStrategy(BaseStrategy):
    """
    Estrategia Cuantitativa Híbrida F7: Meta-Labeling y Triple Barrera.
    """

    def __init__(
        self,
        name: str = "MetaLabeling_TripleBarrier_F7",
        params: Optional[Dict[str, Any]] = None
    ):
        default_params: Dict[str, Any] = {
            "symbol": "ES",
            "std_symbol": "ES",
            "micro_symbol": "MES",
            "ell_plus": 2.00,              # Multiplicador TP (2.0 sigma)
            "ell_minus": 1.00,             # Multiplicador SL (1.0 sigma)
            "delta_tau_minutes": 30,       # Barrera vertical (minutos)
            "tau_meta": 0.60,              # Umbral analítico de veto probabilístico
            "bet_sizing_method": "gaussian", # 'gaussian' o 'polynomial'
            "eta_polynomial": 1.50,        # Exponente amortiguación polinómica
            "slippage_ticks": 1.0,         # Deslizamiento por ejecución
            "initial_capital": 50000.0,
            "dll": 1000.0,                 # Daily Loss Limit
            "mtdd": 2000.0,                # Max Trailing Drawdown
            "gamma_d": 0.20,               # Cota de DLL por trade (20%)
            "gamma_t": 0.08,               # Cota de MTDD cushion por trade (8%)
            "cooldown_minutes": 5,         # Tiempo de enfriamiento tras cierre
            "rth_open": "09:30",
            "exec_window_open": "09:45",
            "exec_window_close": "15:30",
            "eod_close": "15:55",
            "train_classifier": True,      # Entrenar meta-clasificador sobre historial
        }
        if params:
            default_params.update(params)
        super().__init__(name=name, params=default_params)

        self.classifier = MetaLabelingClassifier()
        self.risk_manager = PropRiskManager(
            initial_capital=self.params["initial_capital"],
            dll=self.params["dll"],
            mtdd=self.params["mtdd"],
            gamma_d=self.params["gamma_d"],
            gamma_t=self.params["gamma_t"]
        )

    def generate_primary_signals(self, df: pd.DataFrame, delta_vwap: pd.Series, z_ofi: pd.Series) -> pd.Series:
        """
        Nivel 1: Hipótesis económica microestructural de señal direccional s_t^{(1)} in {-1, 0, +1}.
        Combina divergencia del precio frente al VWAP intradiario con confirmación por desbalance de OFI.
        """
        signals = pd.Series(0, index=df.index, dtype=int)
        
        # Long: Precio por encima de VWAP y presión compradora institucional en OFI
        long_cond = (delta_vwap > 0.0) & (z_ofi >= 0.50)
        # Short: Precio por debajo de VWAP y presión vendedora institucional en OFI
        short_cond = (delta_vwap < 0.0) & (z_ofi <= -0.50)

        signals[long_cond] = 1
        signals[short_cond] = -1
        return signals

    def extract_features(
        self,
        df: pd.DataFrame,
        sigma_series: pd.Series,
        z_ofi_series: pd.Series,
        delta_vwap_series: pd.Series,
        rvol_series: pd.Series,
        compression_series: pd.Series,
        tod_series: pd.Series
    ) -> List[MicroFeatures]:
        """Extrae la lista de vectores X_t in R^6 para todas las barras del dataset."""
        features = []
        for i in range(len(df)):
            feat = MicroFeatures(
                sigma=float(sigma_series.iloc[i]),
                z_ofi=float(z_ofi_series.iloc[i]),
                delta_vwap=float(delta_vwap_series.iloc[i]),
                rvol=float(rvol_series.iloc[i]),
                range_compression=float(compression_series.iloc[i]),
                tau_tod=float(tod_series.iloc[i])
            )
            features.append(feat)
        return features

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera la serie de señales ejecutivas finales S_t^* in {-1.0, 0.0, +1.0}
        cumpliendo con la interfaz abstracta BaseStrategy.
        """
        self.validate_data(data)
        df_sorted = data.sort_index()

        is_ct = detect_data_time_offset(df_sorted)
        sigma_series = compute_garman_klass_volatility(df_sorted)
        _, delta_vwap_series = compute_intraday_anchored_vwap(df_sorted)
        z_ofi_series = compute_order_flow_imbalance_zscore(df_sorted)
        rvol_series = compute_relative_volume(df_sorted)
        comp_series = compute_range_compression(df_sorted)
        tod_series = compute_session_fraction(df_sorted, is_ct=is_ct)

        primary = self.generate_primary_signals(df_sorted, delta_vwap_series, z_ofi_series)
        feats = self.extract_features(
            df_sorted, sigma_series, z_ofi_series, delta_vwap_series, rvol_series, comp_series, tod_series
        )

        tau_meta = self.params.get("tau_meta", 0.60)
        final_signals = pd.Series(0.0, index=df_sorted.index, dtype=float)

        for i in range(len(df_sorted)):
            s_prim = int(primary.iloc[i])
            if s_prim != 0:
                p_t = self.classifier.predict_probability(feats[i], s_prim)
                s_star = apply_meta_veto(s_prim, p_t, tau_meta)
                final_signals.iloc[i] = float(s_star)

        return final_signals



# ==============================================================================
# 7. MOTOR DE BACKTESTING BARRA A BARRA CON REGLAS DE PROP TRADING
# ==============================================================================

class MetaLabelingBacktester:
    """
    Simulador cuantitativo determinista barra a barra para Meta-Labeling y Triple Barrera.
    Cumple rigurosamente con:
    - Causalidad estricta (información hasta t para ejecutar en t+1).
    - Horarios de CME Globex (RTH, ventana activa, cierre intradiario forzoso 15:55 EST).
    - Cooldown analítico de 5 minutos.
    - Control de límites de Prop Trading (DLL, MTDD, Disyuntor).
    - Conmutación automática a microcontratos y dimensionamiento continuo m(p_t).
    """

    def __init__(self, strategy: MetaLabelingTripleBarrierStrategy):
        self.strategy = strategy
        self.params = strategy.params
        self.risk_mgr = strategy.risk_manager

    def fit_classifier_on_data(
        self,
        train_df: pd.DataFrame,
        purge_boundary: bool = True
    ) -> Dict[str, Any]:
        """
        Entrena el meta-clasificador exclusivamente sobre el bloque de datos de entrenamiento (In-Sample).
        Garantiza que no exista filtración hacia el futuro:
        - Si purge_boundary es True, purga las últimas delta_tau_bars del entrenamiento
          para evitar que las barreras triples evalúen velas posteriores o fuera de In-Sample.
        """
        req_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not req_cols.issubset(train_df.columns):
            raise ValueError(f"El DataFrame debe contener las columnas: {req_cols}")

        if not isinstance(train_df.index, pd.DatetimeIndex):
            raise TypeError("El índice del DataFrame de entrenamiento debe ser pd.DatetimeIndex.")

        df_train_sorted = train_df.sort_index().copy()
        is_ct = detect_data_time_offset(df_train_sorted)
        t_eod_close = time(14, 55) if is_ct else time(15, 55)

        sigma_series = compute_garman_klass_volatility(df_train_sorted)
        _, delta_vwap_series = compute_intraday_anchored_vwap(df_train_sorted)
        z_ofi_series = compute_order_flow_imbalance_zscore(df_train_sorted)
        rvol_series = compute_relative_volume(df_train_sorted)
        comp_series = compute_range_compression(df_train_sorted)
        tod_series = compute_session_fraction(df_train_sorted, is_ct=is_ct)

        primary_signals = self.strategy.generate_primary_signals(df_train_sorted, delta_vwap_series, z_ofi_series)
        all_features = self.strategy.extract_features(
            df_train_sorted, sigma_series, z_ofi_series, delta_vwap_series, rvol_series, comp_series, tod_series
        )

        delta_tau_bars = max(1, int(self.params.get("delta_tau_minutes", 30) / 5))
        purge_bars = delta_tau_bars if purge_boundary else 0

        val_idx, labels = label_historical_events(
            df_train_sorted, primary_signals, sigma_series,
            self.params.get("ell_plus", 2.0), self.params.get("ell_minus", 1.0), delta_tau_bars,
            t_eod_close=t_eod_close,
            purge_last_bars=purge_bars
        )

        if len(val_idx) >= 10:
            train_feats = [all_features[i] for i in val_idx]
            train_sigs = [int(primary_signals.iloc[i]) for i in val_idx]
            self.strategy.classifier.fit(train_feats, train_sigs, labels)

        return {
            "total_bars_train": len(df_train_sorted),
            "events_labeled": len(labels),
            "purged_bars": purge_bars,
            "weights": self.strategy.classifier.weights.to_vector().tolist()
        }

    def run(self, df: pd.DataFrame, train_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Ejecuta el backtest completo sobre un DataFrame OHLCV intradiario (ej. 5 minutos).
        Si se especifica train_df, entrena el meta-clasificador exclusivamente sobre train_df
        (con purgado en frontera) y evalúa de forma ciega sobre df (Out-of-Sample).
        """
        # Validar columnas requeridas
        req_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not req_cols.issubset(df.columns):
            raise ValueError(f"El DataFrame debe contener las columnas: {req_cols}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("El índice del DataFrame debe ser pd.DatetimeIndex.")

        df_sorted = df.sort_index().copy()

        # 1. Parsear horarios de sesión con detección de horario CT / EST
        is_ct = detect_data_time_offset(df_sorted)
        if is_ct:
            t_exec_open = time(8, 45)
            t_exec_close = time(14, 30)
            t_eod_close = time(14, 55)
        else:
            t_exec_open = time(9, 45)
            t_exec_close = time(15, 30)
            t_eod_close = time(15, 55)

        # 2. Computar variables microestructurales
        sigma_series = compute_garman_klass_volatility(df_sorted)
        _, delta_vwap_series = compute_intraday_anchored_vwap(df_sorted)
        z_ofi_series = compute_order_flow_imbalance_zscore(df_sorted)
        rvol_series = compute_relative_volume(df_sorted)
        comp_series = compute_range_compression(df_sorted)
        tod_series = compute_session_fraction(df_sorted, is_ct=is_ct)

        # 3. Señal primaria s_t^{(1)}
        primary_signals = self.strategy.generate_primary_signals(df_sorted, delta_vwap_series, z_ofi_series)

        # 4. Vector de características X_t
        all_features = self.strategy.extract_features(
            df_sorted, sigma_series, z_ofi_series, delta_vwap_series, rvol_series, comp_series, tod_series
        )

        # 5. Entrenamiento / calibración opcional del meta-clasificador
        if train_df is not None and self.params.get("train_classifier", True):
            self.fit_classifier_on_data(train_df, purge_boundary=True)
        elif self.params.get("train_classifier", True):
            delta_tau_bars = max(1, int(self.params["delta_tau_minutes"] / 5))
            val_idx, labels = label_historical_events(
                df_sorted, primary_signals, sigma_series,
                self.params["ell_plus"], self.params["ell_minus"], delta_tau_bars,
                t_eod_close=t_eod_close,
                purge_last_bars=delta_tau_bars
            )
            if len(val_idx) >= 10:
                train_feats = [all_features[i] for i in val_idx]
                train_sigs = [int(primary_signals.iloc[i]) for i in val_idx]
                self.strategy.classifier.fit(train_feats, train_sigs, labels)

        # 6. Parámetros de simulación
        tau_meta = self.params["tau_meta"]
        ell_plus = self.params["ell_plus"]
        ell_minus = self.params["ell_minus"]
        delta_tau_td = timedelta(minutes=self.params["delta_tau_minutes"])
        cooldown_td = timedelta(minutes=self.params["cooldown_minutes"])
        slippage_ticks = self.params["slippage_ticks"]

        std_sym = self.params["std_symbol"]
        micro_sym = self.params["micro_symbol"]

        # Variables de estado intradiario
        trade_id = 0
        trades: List[TradeRecord] = []
        in_position = False
        current_trade: Optional[TradeRecord] = None
        cooldown_until: Optional[datetime] = None
        current_day: Optional[Any] = None

        # Arrays pre-extraídos para máxima eficiencia de simulación
        bar_dts = df_sorted.index
        bar_opens = df_sorted["Open"].values
        bar_highs = df_sorted["High"].values
        bar_lows = df_sorted["Low"].values
        bar_closes = df_sorted["Close"].values
        bar_sigs = primary_signals.values
        bar_sigmas = sigma_series.values

        # Bucle barra a barra
        n_bars = len(df_sorted)
        for i in range(n_bars):
            bar_dt = bar_dts[i]
            bar_date = bar_dt.date()
            bar_time = bar_dt.time()
            bar_open = float(bar_opens[i])
            bar_high = float(bar_highs[i])
            bar_low = float(bar_lows[i])
            bar_close = float(bar_closes[i])

            # Nueva sesión bursátil
            if current_day != bar_date:
                current_day = bar_date
                self.risk_mgr.start_new_session(bar_date)
                cooldown_until = None

            # --- A. GESTIÓN DE POSICIÓN ABIERTA (SI EXISTE) ---
            if in_position and current_trade is not None:
                # 1. Cierre forzoso intradía pre-clearing CME (15:55:00 EST / 14:55:00 CT)
                if bar_time >= t_eod_close:
                    exit_check = evaluate_triple_barrier_exit(
                        signal_star=current_trade.direction,
                        current_high=bar_high,
                        current_low=bar_low,
                        current_close=bar_close,
                        b_upper=current_trade.b_upper,
                        b_lower=current_trade.b_lower,
                        is_vertical_reached=True,
                        current_open=bar_open
                    )
                    if exit_check is not None and exit_check[0] in (ExitReason.TAKE_PROFIT, ExitReason.STOP_LOSS):
                        reason, exit_price = exit_check
                    else:
                        reason = ExitReason.EOD_FORCED
                        exit_price = bar_close

                    self._close_position(
                        current_trade, bar_dt, exit_price, reason, slippage_ticks
                    )
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None
                    cooldown_until = bar_dt + cooldown_td
                    continue

                # 2. Expiración de Barrera Vertical Temporal
                t_vert_dt = datetime.fromisoformat(current_trade.t_vertical)
                is_vert_expired = bool(bar_dt >= t_vert_dt)

                # 3. Evaluación de impacto en Triple Barrera
                exit_check = evaluate_triple_barrier_exit(
                    signal_star=current_trade.direction,
                    current_high=bar_high,
                    current_low=bar_low,
                    current_close=bar_close,
                    b_upper=current_trade.b_upper,
                    b_lower=current_trade.b_lower,
                    is_vertical_reached=is_vert_expired,
                    current_open=bar_open
                )

                if exit_check is not None:
                    reason, exit_price = exit_check
                    self._close_position(
                        current_trade, bar_dt, exit_price, reason, slippage_ticks
                    )
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None
                    cooldown_until = bar_dt + cooldown_td
                    continue

                # 4. Disyuntor por drawdown durante la posición
                if self.risk_mgr.check_circuit_breaker():
                    self._close_position(
                        current_trade, bar_dt, bar_close, ExitReason.CIRCUIT_BREAKER, slippage_ticks
                    )
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None
                    continue

            # --- B. EVALUACIÓN DE NUEVAS ENTRADAS ---
            if in_position or self.risk_mgr.is_halted:
                continue

            if cooldown_until is not None and bar_dt < cooldown_until:
                continue

            # Restricción horaria de ventana de ejecución activa [09:45, 15:30]
            if not (t_exec_open <= bar_time <= t_exec_close):
                continue

            # Señal primaria en barra actual
            s_primary = int(bar_sigs[i])
            if s_primary == 0:
                continue

            # Evaluar meta-modelo
            feat_i = all_features[i]
            p_t = self.strategy.classifier.predict_probability(feat_i, s_primary)
            s_star = apply_meta_veto(s_primary, p_t, tau_meta)

            if s_star == 0:
                continue  # Señal vetada analíticamente

            # Dimensionamiento continuo de apuesta
            bet_size = calculate_bet_size(
                p_t, tau_meta,
                method=self.params["bet_sizing_method"],
                eta=self.params["eta_polynomial"]
            )
            if bet_size <= 0.0:
                continue

            # Cálculo de contratos considerando límites y conmutación a micro
            sigma_i = float(bar_sigmas[i])
            n_contracts, is_micro, _ = self.risk_mgr.calculate_position_size(
                std_symbol=std_sym,
                micro_symbol=micro_sym,
                sigma=sigma_i,
                ell_minus=ell_minus,
                bet_size=bet_size,
                slippage_ticks=slippage_ticks
            )

            if n_contracts <= 0:
                continue

            # Abrir posición al cierre de la barra actual
            trade_id += 1
            b_upper, b_lower = calculate_barriers(bar_close, sigma_i, s_star, ell_plus, ell_minus)
            t_close_dt = bar_dt.replace(hour=t_eod_close.hour, minute=t_eod_close.minute, second=0)
            t_vertical_dt = min(bar_dt + delta_tau_td, t_close_dt)

            sym_operated = micro_sym if is_micro else std_sym

            current_trade = TradeRecord(
                trade_id=trade_id,
                date=str(bar_date),
                symbol=sym_operated,
                direction=s_star,
                contracts=n_contracts,
                is_micro=is_micro,
                entry_time=bar_dt.isoformat(),
                entry_price=bar_close,
                b_upper=b_upper,
                b_lower=b_lower,
                t_vertical=t_vertical_dt.isoformat(),
                meta_prob=round(p_t, 4),
                bet_size=round(bet_size, 4)
            )
            in_position = True

        # Liquidar cualquier trade remanente al final del histórico
        if in_position and current_trade is not None:
            last_dt = df_sorted.index[-1]
            last_close = float(df_sorted["Close"].iloc[-1])
            self._close_position(current_trade, last_dt, last_close, ExitReason.EOD_FORCED, slippage_ticks)
            trades.append(current_trade)

        # 6. Compilar métricas analíticas globales
        return self._compile_results(trades, df_sorted)

    def _close_position(
        self,
        trade: TradeRecord,
        exit_dt: datetime,
        exit_price: float,
        reason: ExitReason,
        slippage_ticks: float
    ) -> None:
        """Calcula el PnL bruto, fricciones (comisiones y slippage CME) y actualiza la cuenta."""
        spec = get_spec(trade.symbol)
        multiplier = spec["multiplier"]
        tick = spec["tick"]
        comm_rt = spec["commission_rt"]

        # PnL en puntos y USD
        pts_diff = (exit_price - trade.entry_price) * trade.direction
        gross_usd = pts_diff * multiplier * trade.contracts

        # Fricción CME (slippage de entrada y salida + comisiones round-turn)
        # 1 tick de entrada + 1 tick de salida = 2 ticks por contrato
        total_slippage_usd = (2.0 * slippage_ticks * tick * multiplier) * trade.contracts
        total_comm_usd = comm_rt * trade.contracts

        net_usd = gross_usd - (total_slippage_usd + total_comm_usd)

        trade.exit_time = exit_dt.isoformat()
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.gross_pnl_points = round(pts_diff, 4)
        trade.gross_pnl_usd = round(gross_usd, 2)
        trade.commissions_usd = round(total_comm_usd, 2)
        trade.slippage_usd = round(total_slippage_usd, 2)
        trade.net_pnl_usd = round(net_usd, 2)

        self.risk_mgr.record_completed_trade(trade)

    def _compile_results(self, trades: List[TradeRecord], df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estadísticas financieras completas para la rúbrica y reporte."""
        init_cap = self.params["initial_capital"]
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "net_profit": 0.0,
                "total_return_pct": 0.0,
                "final_equity": round(self.risk_mgr.equity, 2),
                "peak_equity": round(self.risk_mgr.peak_equity, 2),
                "max_drawdown_usd": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "standard_trades": 0,
                "micro_trades": 0,
                "exit_reasons": {},
                "is_halted": self.risk_mgr.is_halted,
                "halt_reason": self.risk_mgr.halt_reason,
                "trades": []
            }


        pnl_series = pd.Series([t.net_pnl_usd for t in trades])
        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]

        total_net_pnl = float(pnl_series.sum())
        total_return_pct = (total_net_pnl / init_cap) * 100.0
        win_rate = (len(wins) / len(pnl_series)) * 100.0 if len(pnl_series) > 0 else 0.0

        gross_gains = float(wins.sum())
        gross_losses = abs(float(losses.sum()))
        profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else (999.0 if gross_gains > 0 else 0.0)

        # Max Drawdown
        equities = [init_cap] + [t.running_equity for t in trades]
        eq_series = pd.Series(equities)
        peak_series = eq_series.cummax()
        dd_series = peak_series - eq_series
        max_dd_usd = float(dd_series.max())
        max_dd_pct = float((dd_series / peak_series).max() * 100.0)

        # Sharpe Ratio por trade
        std_pnl = float(pnl_series.std())
        sharpe = (float(pnl_series.mean()) / std_pnl * math.sqrt(252)) if std_pnl > 0 else 0.0

        # Conteo por causas de salida
        reasons_count: Dict[str, int] = {}
        for t in trades:
            r = t.exit_reason.value if t.exit_reason else "UNKNOWN"
            reasons_count[r] = reasons_count.get(r, 0) + 1

        # Conteo estándar vs micro
        micro_trades = sum(1 for t in trades if t.is_micro)
        std_trades = len(trades) - micro_trades

        return {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "net_profit": round(total_net_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "final_equity": round(self.risk_mgr.equity, 2),
            "peak_equity": round(self.risk_mgr.peak_equity, 2),
            "max_drawdown_usd": round(max_dd_usd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe_ratio": round(sharpe, 2),
            "standard_trades": std_trades,
            "micro_trades": micro_trades,
            "exit_reasons": reasons_count,
            "is_halted": self.risk_mgr.is_halted,
            "halt_reason": self.risk_mgr.halt_reason,
            "trades": [t.__dict__ for t in trades]
        }
