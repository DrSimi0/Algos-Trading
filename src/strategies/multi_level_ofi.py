"""
src/strategies/multi_level_ofi.py
Implementación Matemática y Microestructural de la Estrategia F6:
Multi-Level Order Flow Imbalance en LOB Profundo (Xu, Gould & Howison, 2019).

Referencias Académicas:
1. Xu, K., Gould, M. D., & Howison, S. D. (2019). Multi-Level Order-Flow Imbalance in a Limit Order Book.
   arXiv:1907.06230.
2. Cont, R., Kukanov, A., & Stoikov, S. (2014). The Price Impact of Order Book Events.
   Journal of Financial Econometrics, 12(1), 47-88.
3. Biais, B., Hillion, P., & Spatt, C. (1995). An Empirical Analysis of the Limit Order Book and
   the Order Flow in the Paris Bourse. Journal of Finance, 50(5), 1655-1689.
4. Jolliffe, I. T. (2002). Principal Component Analysis. Springer Series in Statistics.

Especificación Formal: reports/estrategias_matematicas/F6_multi_level_ofi.md
"""

from dataclasses import dataclass, field
from datetime import time, datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Sequence, Union
import math
import numpy as np
import pandas as pd

from src.core.base_strategy import BaseStrategy
from src.core.futures_specs import get_spec


# ==============================================================================
# 1. ESTADOS FINITOS Y ENUMS DEL SISTEMA (SECCIÓN 2 Y 4)
# ==============================================================================

class StrategyState(str, Enum):
    WARMING_UP = "WARMING_UP"
    SCANNING = "SCANNING"
    IN_POSITION = "IN_POSITION"
    FLAT_EOD = "FLAT_EOD"
    HALT_ACCOUNT_BREACHED = "HALT_ACCOUNT_BREACHED"
    HALT_VOLATILITY_EXCEEDS_BUDGET = "HALT_VOLATILITY_EXCEEDS_BUDGET"


class ExitReason(str, Enum):
    TAKE_PROFIT_Z_CLOSE = "TAKE_PROFIT_Z_CLOSE"
    TAKE_PROFIT_PRICE_TARGET = "TAKE_PROFIT_PRICE_TARGET"
    STOP_LOSS_COHERENCE_ABORT = "STOP_LOSS_COHERENCE_ABORT"
    STOP_LOSS_Z_REVERSE = "STOP_LOSS_Z_REVERSE"
    STOP_LOSS_PRICE = "STOP_LOSS_PRICE"
    TIME_DECAY = "TIME_DECAY"
    EOD_CLOSE = "EOD_CLOSE"


# ==============================================================================
# 2. REGISTRO DE TRADES Y CUENTA PROP TRADING (SECCIÓN 5)
# ==============================================================================

@dataclass
class TradeRecord:
    trade_id: int
    entry_bar: int
    entry_time: str
    symbol: str
    direction: int                      # +1 = Long, -1 = Short
    contracts: int                      # Número de contratos operados
    is_micro: bool                      # True si operó micro (MES / MNQ)
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    exit_bar: Optional[int] = None
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
    Gestiona Límite de Pérdida Diaria (DLL), Trailing Drawdown flotante (TMDD)
    y conmutación automática de contratos estándar a micro.
    """
    initial_capital: float = 50000.0
    dll: float = 1000.0                 # Daily Loss Limit ($)
    mtdd: float = 2000.0                # Maximum Trailing Drawdown ($)
    equity: float = 50000.0
    peak_equity: float = 50000.0        # High-Water Mark (HWM)
    current_drawdown: float = 0.0
    daily_realized_loss: float = 0.0    # Pérdida realizada acumulada en la sesión (positiva)
    daily_pnl: float = 0.0              # PnL neto del día
    is_breached: bool = False
    breach_reason: Optional[str] = None

    def start_session(self, session_date: Any = None) -> None:
        """Reinicia los contadores diarios al inicio de una nueva sesión."""
        self.daily_realized_loss = 0.0
        self.daily_pnl = 0.0

    def calculate_risk_budget(self, phi_dll: float = 0.25, phi_tmdd: float = 0.15) -> float:
        """
        Calcula el presupuesto de riesgo financiero asignable a la operación (Ecuación 5.2):
        PresupuestoRiesgo = min(DLL * phi_dll, TMDD(t_k) * phi_tmdd)
        """
        if self.is_breached:
            return 0.0

        # Trailing drawdown colchón remanente
        # El límite inferior de equity admisible es peak_equity - mtdd
        equity_floor = self.peak_equity - self.mtdd
        tmdd_remaining = max(0.0, self.equity - equity_floor)

        # Límite diario remanente
        dll_remaining = max(0.0, self.dll - self.daily_realized_loss)

        budget_from_dll = dll_remaining * phi_dll
        budget_from_tmdd = tmdd_remaining * phi_tmdd

        return max(0.0, min(budget_from_dll, budget_from_tmdd))

    def update_trade(self, net_pnl: float) -> None:
        """Actualiza el estado de la cuenta tras un trade liquidado."""
        self.equity += net_pnl
        self.daily_pnl += net_pnl

        if self.daily_pnl < 0:
            self.daily_realized_loss = abs(self.daily_pnl)
        else:
            self.daily_realized_loss = 0.0

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.current_drawdown = self.peak_equity - self.equity

        # Verificación estricta de límites de Prop Trading
        if self.daily_realized_loss >= self.dll:
            self.is_breached = True
            self.breach_reason = (
                f"DAILY_LOSS_LIMIT_BREACHED: Pérdida diaria ${self.daily_realized_loss:.2f} "
                f"supera DLL ${self.dll:.2f}"
            )

        if self.current_drawdown >= self.mtdd:
            self.is_breached = True
            self.breach_reason = (
                f"TRAILING_DRAWDOWN_BREACHED: Drawdown ${self.current_drawdown:.2f} "
                f"supera MTDD ${self.mtdd:.2f}"
            )



# ==============================================================================
# 3. FORMULACIÓN MATEMÁTICA PURA (SECCIONES 3, 4 Y 5)
# ==============================================================================

def calculate_bid_flow(p_curr: float, q_curr: float, p_prev: float, q_prev: float) -> float:
    """
    Flujo neto de compra en nivel l (Ecuación 3.1.1):
    e_n^{b, (l)} =
        +q_n^{b, (l)}        si P_n^{b, (l)} > P_{n-1}^{b, (l)}
        q_n - q_{n-1}        si P_n^{b, (l)} == P_{n-1}^{b, (l)}
        -q_{n-1}^{b, (l)}    si P_n^{b, (l)} < P_{n-1}^{b, (l)}
    """
    if p_curr > p_prev:
        return float(q_curr)
    elif p_curr == p_prev:
        return float(q_curr - q_prev)
    else:
        return float(-q_prev)


def calculate_ask_flow(p_curr: float, q_curr: float, p_prev: float, q_prev: float) -> float:
    """
    Flujo neto de venta en nivel l (Ecuación 3.1.2):
    e_n^{a, (l)} =
        -q_{n-1}^{a, (l)}    si P_n^{a, (l)} > P_{n-1}^{a, (l)}
        q_n - q_{n-1}        si P_n^{a, (l)} == P_{n-1}^{a, (l)}
        -q_n^{a, (l)}        si P_n^{a, (l)} < P_{n-1}^{a, (l)}
    """
    if p_curr > p_prev:
        return float(-q_prev)
    elif p_curr == p_prev:
        return float(q_curr - q_prev)
    else:
        return float(-q_curr)


def calculate_unit_imbalance(e_b: float, e_a: float) -> float:
    """
    Desbalance neto unitario del nivel l (Ecuación 3.1.3):
    e_n^{(l)} = e_n^{b, (l)} - e_n^{a, (l)}
    """
    return float(e_b - e_a)


def calculate_spatial_betas(
    levels: int = 5,
    alpha_decay: float = 1.0,
    beta_1: Optional[float] = None,
    d_bar_1: float = 250.0
) -> np.ndarray:
    """
    Vector de impacto marginal bajo decaimiento espacial de ley de potencias (Sección 3.3):
    beta^{(l)} = beta^{(1)} * l^(-alpha_decay)
    con beta^{(1)} = 1 / (2 * d_bar_1) por defecto.
    """
    if levels < 1:
        raise ValueError(f"levels debe ser >= 1, recibido: {levels}")
    if alpha_decay <= 0:
        raise ValueError(f"alpha_decay debe ser > 0, recibido: {alpha_decay}")

    if beta_1 is None:
        if d_bar_1 <= 0:
            raise ValueError(f"d_bar_1 debe ser > 0, recibido: {d_bar_1}")
        beta_1 = 1.0 / (2.0 * d_bar_1)

    l_indices = np.arange(1, levels + 1, dtype=float)
    betas = beta_1 * (l_indices ** (-alpha_decay))
    return betas


def compute_pca_spectral(ofi_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Descomposición espectral de la matriz de covarianza de OFI (Sección 3.4):
    Sigma_{OFI} = V * Lambda * V^T
    Retorna:
    - eigenvalues: array de autovalores ordenados descendentemente (lambda_1 >= ... >= lambda_L)
    - eigenvectors: matriz V cuyas columnas son autovectores ortonormales
    - v1: autovector dominante orientado positivamente (Teorema de Perron-Frobenius: v1^(l) > 0)
    """
    if ofi_matrix.ndim != 2 or ofi_matrix.shape[0] < 2:
        raise ValueError("ofi_matrix debe tener al menos 2 observaciones temporales.")

    cov = np.cov(ofi_matrix, rowvar=False)
    # Caso escalar si L=1
    if cov.ndim == 0:
        return np.array([float(cov)]), np.array([[1.0]]), np.array([1.0])

    eigvals, eigvecs = np.linalg.eigh(cov)
    # Ordenar de mayor a menor
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Teorema de Perron-Frobenius: todos los pesos de v_1 deben ser positivos
    v1 = eigvecs[:, 0].copy()
    if np.sum(v1) < 0:
        v1 = -v1
        eigvecs[:, 0] = v1

    return eigvals, eigvecs, v1


def calculate_pc1(ofi_vector: np.ndarray, v1: np.ndarray) -> float:
    """
    Primer Componente Principal del libro de órdenes (Sección 3.4):
    PC1_k = v_1^T * OFI_k = sum_{l=1}^L v_1^(l) * OFI_k^(l)
    """
    n = min(len(ofi_vector), len(v1))
    return float(np.dot(v1[:n], ofi_vector[:n]))



def calculate_coherence_index(
    ofi_vector: np.ndarray,
    beta_weights: np.ndarray,
    epsilon_0: float = 1e-8
) -> float:
    """
    Índice de Coherencia Direccional de Profundidad (Sección 3.5):
    Phi_{coherence, k} = ( sign(OFI_k^(1)) * sum_{l=2}^L beta^(l) * OFI_k^(l) ) /
                         ( sum_{l=2}^L beta^(l) * |OFI_k^(l)| + epsilon_0 )
    Retorna valor en [-1.0, 1.0].
    """
    if len(ofi_vector) < 2 or len(beta_weights) < len(ofi_vector):
        return 0.0

    ofi_1 = ofi_vector[0]
    if ofi_1 == 0:
        return 0.0

    deep_ofi = ofi_vector[1:]
    deep_betas = beta_weights[1:len(ofi_vector)]

    numerator = np.sign(ofi_1) * np.sum(deep_betas * deep_ofi)
    denominator = np.sum(deep_betas * np.abs(deep_ofi)) + epsilon_0

    if denominator <= 0:
        return 0.0

    coherence = float(numerator / denominator)
    return float(np.clip(coherence, -1.0, 1.0))


def calculate_m_ofi(ofi_vector: np.ndarray, beta_weights: np.ndarray) -> float:
    """
    Multi-Level OFI Compuesto (Sección 3.6):
    M-OFI_k = beta^T * OFI_k = sum_{l=1}^L beta^(l) * OFI_k^(l)
    """
    n = min(len(ofi_vector), len(beta_weights))
    return float(np.dot(beta_weights[:n], ofi_vector[:n]))


def calculate_z_score(
    current_m_ofi: float,
    m_ofi_window: Sequence[float],
    epsilon_0: float = 1e-8
) -> float:
    """
    Z-Score Compuesto Multi-Nivel sobre ventana W (Sección 3.6):
    Z_{M-OFI, k} = ( M-OFI_k - mu_{M, k} ) / ( sigma_{M, k} + epsilon_0 )
    """
    if len(m_ofi_window) < 2:
        return 0.0

    mu = float(np.mean(m_ofi_window))
    sigma = float(np.std(m_ofi_window, ddof=1))

    return float((current_m_ofi - mu) / (sigma + epsilon_0))


def evaluate_entry_signal(
    z_m_ofi: float,
    phi_coherence: float,
    spread: float,
    tick_size: float = 0.25,
    theta_entry: float = 2.25,
    phi_min: float = 0.40
) -> int:
    """
    Ecuaciones Matemáticas Cerradas de Decisión Direccional (Sección 3.7):
    I_{F6, k} =
       +1  si Z > +theta_entry AND phi_coherence >= phi_min AND spread == tick_size
       -1  si Z < -theta_entry AND phi_coherence >= phi_min AND spread == tick_size
        0  en cualquier otro caso
    """
    # Filtro de microestructura: el spread debe ser de 1 tick exacto
    if abs(spread - tick_size) > 1e-6:
        return 0

    if phi_coherence < phi_min:
        return 0

    if z_m_ofi > theta_entry:
        return 1
    elif z_m_ofi < -theta_entry:
        return -1
    return 0


def calculate_absorption_limit(
    lob_bid_sizes: Sequence[float],
    lob_ask_sizes: Sequence[float],
    levels: int = 3,
    gamma_depth: float = 0.08
) -> int:
    """
    Límite de Absorción de Liquidez Acumulada en Profundidad (Sección 5.1):
    Q_{depth, 3}(k) = sum_{l=1}^3 ( (q_k^{b, (l)} + q_k^{a, (l)}) / 2 )
    N_{depth, max} = max(1, floor( gamma_depth * Q_{depth, 3}(k) ))
    """
    n = min(levels, len(lob_bid_sizes), len(lob_ask_sizes))
    if n == 0:
        return 1

    total_depth_3 = 0.0
    for l in range(n):
        total_depth_3 += (lob_bid_sizes[l] + lob_ask_sizes[l]) / 2.0

    max_contracts = int(math.floor(gamma_depth * total_depth_3))
    return max(1, max_contracts)


def calculate_position_size(
    risk_budget_usd: float,
    symbol: str,
    stop_loss_points: float = 1.50,
    depth_limit_contracts: int = 10,
    max_contracts_param: int = 5,
    har_volatility: Optional[float] = None,
    ref_volatility: Optional[float] = None,
    force_micro: bool = False
) -> Tuple[int, bool, str]:
    """
    Dimensionamiento de contratos y Conmutación Automática Estándar / Micro (Secciones 5.3 y 5.4).
    Retorna: (contracts, is_micro, status_message)
    """
    if risk_budget_usd <= 0:
        return 0, False, "HALT_NO_RISK_BUDGET"

    # Determinar contratos estándar y micro
    key = symbol.upper()
    if key in ["MES", "MNQ", "M2K", "MYM", "MGC", "MCL"]:
        micro_key = key
        std_key = key[1:] if key.startswith("M") else key
    elif key in ["ES", "NQ", "RTY", "YM", "GC", "CL"]:
        std_key = key
        micro_key = "M" + key
    else:
        std_key = key
        micro_key = key

    spec_std = get_spec(std_key)
    spec_micro = get_spec(micro_key)

    # Volatilidad condicional y escalado dinámico F_vol_scale (Sección 5.3)
    if ref_volatility is None:
        ref_volatility = 0.0100 if "ES" in std_key else 0.0140

    if har_volatility is not None and har_volatility > 0:
        f_vol_scale = float(ref_volatility / har_volatility)
    else:
        f_vol_scale = 1.0

    # Riesgo por contrato estándar: SL * multiplier + friction (comisión RT + 1 tick slippage)
    slippage_std_usd = spec_std["tick"] * spec_std["multiplier"]
    friction_std_usd = spec_std["commission_rt"] + slippage_std_usd
    risk_per_contract_std = (stop_loss_points * spec_std["multiplier"]) + friction_std_usd

    # Riesgo por micro contrato:
    slippage_micro_usd = spec_micro["tick"] * spec_micro["multiplier"]
    friction_micro_usd = spec_micro["commission_rt"] + slippage_micro_usd
    risk_per_contract_micro = (stop_loss_points * spec_micro["multiplier"]) + friction_micro_usd

    # 1. Evaluar si es posible operar contrato estándar
    if not force_micro:
        n_candidate_std = int(math.floor((risk_budget_usd / risk_per_contract_std) * f_vol_scale))
    else:
        n_candidate_std = 0

    if n_candidate_std >= 1:
        # Operar contrato estándar
        final_contracts = min(n_candidate_std, depth_limit_contracts, max_contracts_param)
        return max(1, final_contracts), False, "STANDARD_EXECUTION"

    # 2. Conmutación Automática a Micro Contratos (Sección 5.4)
    n_candidate_micro = int(math.floor((risk_budget_usd / risk_per_contract_micro) * f_vol_scale))
    if n_candidate_micro >= 1:
        max_micro_limit = 10 * max_contracts_param
        max_micro_depth = 10 * depth_limit_contracts
        final_micro = min(n_candidate_micro, max_micro_depth, max_micro_limit)
        return max(1, final_micro), True, "SWITCHED_TO_MICRO"

    # Si ni 1 micro contrato entra en el presupuesto: parada preventiva
    return 0, True, "HALT_VOLATILITY_EXCEEDS_BUDGET"


# ==============================================================================
# 4. CLASE ESTRATEGIA PRINCIPAL: MultiLevelOFIStrategy
# ==============================================================================

class MultiLevelOFIStrategy(BaseStrategy):
    """
    Estrategia F6: Multi-Level Order Flow Imbalance en LOB Profundo.
    Implementa la evaluación completa de señales y gestión de estados para CME Globex.
    """

    def __init__(
        self,
        name: str = "MultiLevel_OFI_F6",
        params: Optional[Dict[str, Any]] = None
    ):
        default_params: Dict[str, Any] = {
            "symbol": "ES",
            "levels": 5,                       # Profundidad analizada L = 5
            "alpha_decay": 1.0,                # Exponente de decaimiento espacial de impacto
            "d_bar_1": 250.0,                  # Profundidad promedio esperada en nivel 1
            "window": 30,                      # Ventana móvil de calibración W = 30 barras
            "theta_entry": 2.25,               # Umbral Z de activación de entrada
            "phi_min": 0.40,                   # Umbral mínimo de coherencia anti-spoofing
            "theta_close": 0.60,               # Umbral Z de salida por normalización de desbalance
            "phi_abort": 0.50,                 # Umbral de escape por fuga de soporte (Phi <= -0.50)
            "theta_reverse": 1.00,             # Umbral Z de escape por inversión contraria
            "sl_ticks": 6,                     # Stop-Loss rígido en ticks (6 ticks = 1.50 pt)
            "tp_min_ticks": 6,                 # Take-Profit mínimo en ticks (6 ticks = 1.50 pt)
            "tp_beta_sigma_factor": 1.50,      # Factor k_tp para proyección TP analítico
            "max_holding_bars": 15,            # Límite de permanencia: 15 barras de 1 min
            "start_time": time(9, 45),         # Inicio operativo RTH
            "end_time": time(15, 55),          # Cierre mandatario EOD Flat
            "gamma_depth": 0.08,               # Factor de absorción profunda en 3 niveles
            "max_std_contracts": 5,            # Límite superior de contratos estándar
            "dll": 1000.0,                     # Daily Loss Limit ($)
            "mtdd": 2000.0,                    # Maximum Trailing Drawdown ($)
            "phi_dll": 0.25,                   # Presupuesto fracción DLL
            "phi_tmdd": 0.15,                  # Presupuesto fracción TMDD
            "weighting_mode": "power_law",     # Modo de ponderación: 'power_law' (Xu et al.) o 'pca' (espectral)
            "epsilon_0": 1e-8,
        }
        if params:
            default_params.update(params)
        super().__init__(name=name, params=default_params)

        self.levels = self.params["levels"]
        self.betas = calculate_spatial_betas(
            levels=self.levels,
            alpha_decay=self.params["alpha_decay"],
            d_bar_1=self.params["d_bar_1"]
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera serie de señales discretas {-1, 0, +1} a partir de datos LOB agregados por barra.
        Espera columnas:
        - 'bid_price_1', 'ask_price_1', ..., 'bid_price_L', 'ask_price_L'
        - 'bid_size_1', 'ask_size_1', ..., 'bid_size_L', 'ask_size_L'
        O alternativamente columnas directas de OFI:
        - 'ofi_1', 'ofi_2', ..., 'ofi_L'
        """
        signals = pd.Series(0.0, index=data.index)
        n_bars = len(data)
        window = self.params["window"]
        weighting_mode = self.params.get("weighting_mode", "power_law")

        # 1. Comprobar si vienen columnas precalculadas de OFI o cotizaciones LOB
        has_ofi_cols = all(f"ofi_{l}" in data.columns for l in range(1, self.levels + 1))

        if has_ofi_cols:
            ofi_matrix = data[[f"ofi_{l}" for l in range(1, self.levels + 1)]].to_numpy(dtype=float)
        else:
            # Calcular OFI a partir de cambios de cotizaciones LOB nivel por nivel
            ofi_matrix = self.compute_ofi_from_lob_bars(data)

        if weighting_mode == "pca":
            m_ofi_vals = np.zeros(n_bars, dtype=float)
            for idx in range(n_bars):
                if idx >= window:
                    sub_mat = ofi_matrix[max(0, idx - window):idx]
                    try:
                        _, _, v1 = compute_pca_spectral(sub_mat)
                        v1_norm = (v1 / np.maximum(np.sum(np.abs(v1)), 1e-8)) * np.sum(self.betas)
                        m_ofi_vals[idx] = calculate_m_ofi(ofi_matrix[idx], v1_norm)
                    except Exception:
                        m_ofi_vals[idx] = calculate_m_ofi(ofi_matrix[idx], self.betas)
                else:
                    m_ofi_vals[idx] = calculate_m_ofi(ofi_matrix[idx], self.betas)
        else:
            m_ofi_vals = np.array([calculate_m_ofi(ofi_matrix[i], self.betas) for i in range(n_bars)])

        tick_size = get_spec(self.params["symbol"])["tick"]

        for i in range(window, n_bars):
            current_time = data.index[i].time() if isinstance(data.index[i], pd.Timestamp) else None
            if current_time is not None:
                if not (self.params["start_time"] <= current_time < self.params["end_time"]):
                    continue

            # Spread en nivel 1
            if "ask_price_1" in data.columns and "bid_price_1" in data.columns:
                spread = data["ask_price_1"].iloc[i] - data["bid_price_1"].iloc[i]
            else:
                spread = tick_size

            # Z-Score y Coherencia
            z_val = calculate_z_score(
                m_ofi_vals[i],
                m_ofi_vals[max(0, i - window):i],
                self.params["epsilon_0"]
            )
            phi_val = calculate_coherence_index(
                ofi_matrix[i],
                self.betas,
                self.params["epsilon_0"]
            )

            sig = evaluate_entry_signal(
                z_m_ofi=z_val,
                phi_coherence=phi_val,
                spread=spread,
                tick_size=tick_size,
                theta_entry=self.params["theta_entry"],
                phi_min=self.params["phi_min"]
            )
            signals.iloc[i] = float(sig)

        return signals

    def compute_ofi_from_lob_bars(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calcula el vector de OFI contemporáneo en cada barra a partir de los niveles de cotización.
        """
        n = len(df)
        ofi_mat = np.zeros((n, self.levels), dtype=float)

        for l in range(1, self.levels + 1):
            bp_col = f"bid_price_{l}"
            ap_col = f"ask_price_{l}"
            bs_col = f"bid_size_{l}"
            as_col = f"ask_size_{l}"

            if not all(col in df.columns for col in [bp_col, ap_col, bs_col, as_col]):
                continue

            bp = df[bp_col].to_numpy(dtype=float)
            ap = df[ap_col].to_numpy(dtype=float)
            bs = df[bs_col].to_numpy(dtype=float)
            as_ = df[as_col].to_numpy(dtype=float)

            for i in range(1, n):
                eb = calculate_bid_flow(bp[i], bs[i], bp[i - 1], bs[i - 1])
                ea = calculate_ask_flow(ap[i], as_[i], ap[i - 1], as_[i - 1])
                ofi_mat[i, l - 1] = calculate_unit_imbalance(eb, ea)

        return ofi_mat


# ==============================================================================
# 5. MOTOR DE BACKTESTING CUANTITATIVO: MultiLevelOFIBacktester
# ==============================================================================

class MultiLevelOFIBacktester:
    """
    Simulador determinista de microestructura y backtest para la Estrategia F6.
    Aplica rigurosamente:
    - Reglas de ejecución CME Globex (fricciones, comisiones, tick size).
    - Reglas de Prop Trading (DLL, MTDD, trailing drawdown flotante).
    - Salidas dinámicas microestructurales (Z-Score relaxation, fuga de soporte por Phi,
      stop-loss, time-decay de 15 barras y cierre mandatario EOD).
    - Dimensionamiento con escalado por volatilidad y conmutación Standard / Micro.
    """

    def __init__(
        self,
        strategy: Optional[MultiLevelOFIStrategy] = None,
        account_size: float = 50000.0,
        daily_loss_limit: float = 1000.0,
        max_trailing_loss: float = 2000.0,
        commission_per_contract: Optional[float] = None,
        slippage_ticks: float = 1.0,
        symbol: str = "ES",
    ):
        self.symbol = symbol.upper()
        self.spec = get_spec(self.symbol)
        self.strategy = strategy or MultiLevelOFIStrategy(params={"symbol": self.symbol})

        self.account_size = account_size
        self.daily_loss_limit = daily_loss_limit
        self.max_trailing_loss = max_trailing_loss
        self.slippage_ticks = slippage_ticks

        if commission_per_contract is not None:
            self.commission_rt = commission_per_contract
        else:
            self.commission_rt = self.spec["commission_rt"]

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Ejecuta el backtest barra a barra con control estricto de eventos y riesgo prop.
        """
        p = self.strategy.params
        symbol = self.symbol
        spec_std = get_spec("ES" if "ES" in symbol else "NQ")
        spec_micro = get_spec("MES" if "ES" in symbol else "MNQ")

        account = PropAccount(
            initial_capital=self.account_size,
            dll=self.daily_loss_limit,
            mtdd=self.max_trailing_loss,
            equity=self.account_size,
            peak_equity=self.account_size
        )

        n_bars = len(data)
        window = p["window"]
        tick_size = spec_std["tick"]
        sl_points = p["sl_ticks"] * tick_size  # 1.50 puntos

        # Precomputar matriz OFI y vector de betas
        betas = self.strategy.betas
        weighting_mode = p.get("weighting_mode", "power_law")
        if all(f"ofi_{l}" in data.columns for l in range(1, self.strategy.levels + 1)):
            ofi_matrix = data[[f"ofi_{l}" for l in range(1, self.strategy.levels + 1)]].to_numpy(dtype=float)
        else:
            ofi_matrix = self.strategy.compute_ofi_from_lob_bars(data)

        if weighting_mode == "pca":
            m_ofi_vals = np.zeros(n_bars, dtype=float)
            for idx in range(n_bars):
                if idx >= window:
                    sub_mat = ofi_matrix[max(0, idx - window):idx]
                    try:
                        _, _, v1 = compute_pca_spectral(sub_mat)
                        v1_norm = (v1 / np.maximum(np.sum(np.abs(v1)), 1e-8)) * np.sum(betas)
                        m_ofi_vals[idx] = calculate_m_ofi(ofi_matrix[idx], v1_norm)
                    except Exception:
                        m_ofi_vals[idx] = calculate_m_ofi(ofi_matrix[idx], betas)
                else:
                    m_ofi_vals[idx] = calculate_m_ofi(ofi_matrix[idx], betas)
        else:
            m_ofi_vals = np.array([calculate_m_ofi(ofi_matrix[i], betas) for i in range(n_bars)])


        # Columnas de precios
        if "Close" in data.columns:
            close_prices = data["Close"].to_numpy(dtype=float)
            high_prices = data["High"].to_numpy(dtype=float) if "High" in data.columns else close_prices
            low_prices = data["Low"].to_numpy(dtype=float) if "Low" in data.columns else close_prices
        elif "bid_price_1" in data.columns and "ask_price_1" in data.columns:
            bid1 = data["bid_price_1"].to_numpy(dtype=float)
            ask1 = data["ask_price_1"].to_numpy(dtype=float)
            close_prices = (bid1 + ask1) / 2.0
            high_prices = ask1
            low_prices = bid1
        else:
            raise ValueError("El DataFrame debe contener precios 'Close' o 'bid_price_1'/'ask_price_1'.")

        trades: List[TradeRecord] = []
        active_trade: Optional[TradeRecord] = None
        state = StrategyState.WARMING_UP
        current_date = None

        equity_curve = [account.equity]
        drawdown_curve = [0.0]
        dates_recorded = [data.index[0]]

        trade_counter = 0

        for i in range(n_bars):
            ts = data.index[i]
            bar_time = ts.time() if hasattr(ts, "time") else None
            bar_date = ts.date() if hasattr(ts, "date") else None

            # 1. Gestión de nueva sesión diaria
            if bar_date != current_date:
                current_date = bar_date
                account.start_session(current_date)
                # Si la cuenta estaba en estado terminal breach, no se reinicia
                if not account.is_breached:
                    state = StrategyState.SCANNING

            # Si la cuenta vulneró los límites contractuales, detener permanentemente
            if account.is_breached:
                state = StrategyState.HALT_ACCOUNT_BREACHED
                equity_curve.append(account.equity)
                drawdown_curve.append(account.current_drawdown)
                dates_recorded.append(ts)
                continue

            current_close = close_prices[i]
            current_high = high_prices[i]
            current_low = low_prices[i]

            # 2. Gestión de posición viva (Evaluación de Salidas de Sección 4)
            if active_trade is not None:
                bars_held = i - active_trade.entry_bar
                exit_triggered = False
                exit_reason = None
                exit_price = current_close

                # Factores de contrato activo (Estándar vs Micro)
                contract_spec = spec_micro if active_trade.is_micro else spec_std
                mult = contract_spec["multiplier"]

                # A. Cierre Mandatorio al Final de Sesión (Overnight Flat Rule 4.4)
                if bar_time is not None and bar_time >= p["end_time"]:
                    exit_triggered = True
                    exit_reason = ExitReason.EOD_CLOSE
                    exit_price = current_close

                # B. Cierre por Time-Decay Multi-Nivel (Sección 4.3)
                elif bars_held >= p["max_holding_bars"]:
                    exit_triggered = True
                    exit_reason = ExitReason.TIME_DECAY
                    exit_price = current_close

                # C. Evaluación de Señales Dinámicas de LOB (TP y SL)
                else:
                    # Métricas actuales de la barra
                    z_val = calculate_z_score(
                        m_ofi_vals[i],
                        m_ofi_vals[max(0, i - window):i],
                        p["epsilon_0"]
                    )
                    phi_val = calculate_coherence_index(
                        ofi_matrix[i],
                        betas,
                        p["epsilon_0"]
                    )

                    # Posición Larga
                    if active_trade.direction == 1:
                        # 1. Stop-Loss rígido por precio CME (Sección 4.2 / límite de parada)
                        if current_low <= active_trade.stop_loss_price:
                            exit_triggered = True
                            exit_reason = ExitReason.STOP_LOSS_PRICE
                            exit_price = active_trade.stop_loss_price

                        # 2. Stop-Loss estructural por fuga de soporte pasivo profundo (Sección 4.2: Phi <= -0.50)
                        elif phi_val <= -p["phi_abort"]:
                            exit_triggered = True
                            exit_reason = ExitReason.STOP_LOSS_COHERENCE_ABORT
                            exit_price = current_close

                        # 3. Stop-Loss estructural por inversión contraria del desbalance (Sección 4.2: Z <= -1.00)
                        elif z_val <= -p["theta_reverse"]:
                            exit_triggered = True
                            exit_reason = ExitReason.STOP_LOSS_Z_REVERSE
                            exit_price = current_close

                        # 4. Take-Profit por objetivo analítico de precio (Sección 4.1: High >= TP)
                        elif current_high >= active_trade.take_profit_price:
                            exit_triggered = True
                            exit_reason = ExitReason.TAKE_PROFIT_PRICE_TARGET
                            exit_price = active_trade.take_profit_price

                        # 5. Take-Profit por relajación / normalización del desbalance (Sección 4.1: Z <= 0.60)
                        elif z_val <= p["theta_close"]:
                            exit_triggered = True
                            exit_reason = ExitReason.TAKE_PROFIT_Z_CLOSE
                            exit_price = current_close

                    # Posición Corta
                    elif active_trade.direction == -1:
                        # 1. Stop-Loss rígido por precio CME (Sección 4.2)
                        if current_high >= active_trade.stop_loss_price:
                            exit_triggered = True
                            exit_reason = ExitReason.STOP_LOSS_PRICE
                            exit_price = active_trade.stop_loss_price

                        # 2. Stop-Loss estructural por fuga de resistencia pasiva profunda (Sección 4.2: Phi <= -0.50)
                        elif phi_val <= -p["phi_abort"]:
                            exit_triggered = True
                            exit_reason = ExitReason.STOP_LOSS_COHERENCE_ABORT
                            exit_price = current_close

                        # 3. Stop-Loss estructural por inversión contraria del desbalance (Sección 4.2: Z >= +1.00)
                        elif z_val >= p["theta_reverse"]:
                            exit_triggered = True
                            exit_reason = ExitReason.STOP_LOSS_Z_REVERSE
                            exit_price = current_close

                        # 4. Take-Profit por objetivo analítico de precio (Sección 4.1: Low <= TP)
                        elif current_low <= active_trade.take_profit_price:
                            exit_triggered = True
                            exit_reason = ExitReason.TAKE_PROFIT_PRICE_TARGET
                            exit_price = active_trade.take_profit_price

                        # 5. Take-Profit por relajación / normalización del desbalance (Sección 4.1: Z >= -0.60)
                        elif z_val >= -p["theta_close"]:
                            exit_triggered = True
                            exit_reason = ExitReason.TAKE_PROFIT_Z_CLOSE
                            exit_price = current_close

                # Liquidar trade si detonó salida
                if exit_triggered:
                    active_trade.exit_bar = i
                    active_trade.exit_time = str(ts)
                    active_trade.exit_price = exit_price
                    active_trade.exit_reason = exit_reason

                    # Cálculo PnL
                    points = (exit_price - active_trade.entry_price) * active_trade.direction
                    active_trade.gross_pnl_points = points
                    gross_usd = points * mult * active_trade.contracts
                    active_trade.gross_pnl_usd = gross_usd

                    # Comisiones y Deslizamiento (Slippage de entrada y salida)
                    comm_cost = contract_spec["commission_rt"] * active_trade.contracts
                    slip_cost = (self.slippage_ticks * tick_size * mult * 2.0) * active_trade.contracts
                    active_trade.commissions_usd = comm_cost
                    active_trade.slippage_usd = slip_cost

                    net_usd = gross_usd - comm_cost - slip_cost
                    active_trade.net_pnl_usd = net_usd

                    # Actualizar Prop Account
                    account.update_trade(net_usd)
                    active_trade.running_equity = account.equity
                    active_trade.running_drawdown = account.current_drawdown

                    trades.append(active_trade)
                    active_trade = None

                    if account.is_breached:
                        state = StrategyState.HALT_ACCOUNT_BREACHED
                    else:
                        state = StrategyState.SCANNING

            # 3. Evaluación de Nuevas Entradas (Sección 3.7)
            elif state == StrategyState.SCANNING and i >= window:
                # Ventana operativa RTH
                in_time_window = True
                if bar_time is not None:
                    in_time_window = (p["start_time"] <= bar_time < p["end_time"])

                if in_time_window:
                    # Spread en nivel 1
                    if "ask_price_1" in data.columns and "bid_price_1" in data.columns:
                        spread = data["ask_price_1"].iloc[i] - data["bid_price_1"].iloc[i]
                    else:
                        spread = tick_size

                    # Z-Score y Coherencia
                    z_val = calculate_z_score(
                        m_ofi_vals[i],
                        m_ofi_vals[max(0, i - window):i],
                        p["epsilon_0"]
                    )
                    phi_val = calculate_coherence_index(
                        ofi_matrix[i],
                        betas,
                        p["epsilon_0"]
                    )

                    signal = evaluate_entry_signal(
                        z_m_ofi=z_val,
                        phi_coherence=phi_val,
                        spread=spread,
                        tick_size=tick_size,
                        theta_entry=p["theta_entry"],
                        phi_min=p["phi_min"]
                    )

                    if signal != 0:
                        # Calcular absorción profunda Q_{depth, 3} (Sección 5.1)
                        if "bid_size_1" in data.columns:
                            b_sizes = [data[f"bid_size_{l}"].iloc[i] for l in range(1, 4)]
                            a_sizes = [data[f"ask_size_{l}"].iloc[i] for l in range(1, 4)]
                            depth_max = calculate_absorption_limit(b_sizes, a_sizes, 3, p["gamma_depth"])
                        else:
                            depth_max = p["max_std_contracts"]

                        # Presupuesto de riesgo contractual Prop Firm
                        risk_budget = account.calculate_risk_budget(p["phi_dll"], p["phi_tmdd"])

                        # Volatilidad condicional de alta frecuencia para escalado dinámico (Sección 5.3)
                        if "har_vol" in data.columns:
                            current_vol = float(data["har_vol"].iloc[i])
                        elif "volatility" in data.columns:
                            current_vol = float(data["volatility"].iloc[i])
                        else:
                            w_start = max(0, i - window)
                            if i - w_start >= 5:
                                ret_window = np.diff(close_prices[w_start:i + 1]) / np.maximum(close_prices[w_start:i], 1e-8)
                                current_vol = float(np.std(ret_window, ddof=1) * np.sqrt(390.0))
                            else:
                                current_vol = None

                        # Dimensionamiento y switch automático Standard / Micro
                        contracts, is_micro, sizing_status = calculate_position_size(
                            risk_budget_usd=risk_budget,
                            symbol=symbol,
                            stop_loss_points=sl_points,
                            depth_limit_contracts=depth_max,
                            max_contracts_param=p["max_std_contracts"],
                            har_volatility=current_vol
                        )

                        if contracts >= 1:
                            trade_counter += 1
                            entry_price = current_close

                            # Estimación analítica del TP (Sección 4.1): max(6 ticks, 1.5 * beta^T * sigma_ofi)
                            ofi_sub = ofi_matrix[max(0, i - window):i]
                            if len(ofi_sub) >= 2:
                                sigma_vec = np.std(ofi_sub, axis=0, ddof=1)
                            else:
                                sigma_vec = np.ones(self.strategy.levels)

                            anal_tp_points = max(
                                p["tp_min_ticks"] * tick_size,
                                p["tp_beta_sigma_factor"] * float(np.dot(betas, sigma_vec))
                            )

                            if signal == 1:
                                sl_price = entry_price - sl_points
                                tp_price = entry_price + anal_tp_points
                            else:
                                sl_price = entry_price + sl_points
                                tp_price = entry_price - anal_tp_points

                            active_trade = TradeRecord(
                                trade_id=trade_counter,
                                entry_bar=i,
                                entry_time=str(ts),
                                symbol="MES" if is_micro and "ES" in symbol else ("MNQ" if is_micro else symbol),
                                direction=signal,
                                contracts=contracts,
                                is_micro=is_micro,
                                entry_price=entry_price,
                                stop_loss_price=sl_price,
                                take_profit_price=tp_price
                            )
                            state = StrategyState.IN_POSITION

            equity_curve.append(account.equity)
            drawdown_curve.append(account.current_drawdown)
            dates_recorded.append(ts)

        # 4. Compilación de Métricas Finales
        summary = self._calculate_performance_metrics(trades, account)
        equity_df = pd.DataFrame({
            "equity": equity_curve[:len(dates_recorded)],
            "drawdown": drawdown_curve[:len(dates_recorded)]
        }, index=dates_recorded)

        return {
            "metrics": summary,
            "trades": trades,
            "equity_curve": equity_df,
            "account": account,
            "status": "FAIL" if account.is_breached else ("PASS" if len(trades) > 0 and summary["net_pnl_usd"] > 0 else "INCOMPLETE")
        }

    def _calculate_performance_metrics(
        self,
        trades: List[TradeRecord],
        account: PropAccount
    ) -> Dict[str, Any]:
        """Calcula métricas financieras y estadísticas de prop trading."""
        total_trades = len(trades)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "net_pnl_usd": 0.0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_usd": account.current_drawdown,
                "prop_status": "FAIL" if account.is_breached else "NO_TRADES",
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0
            }

        pnls = np.array([t.net_pnl_usd for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        net_pnl = float(np.sum(pnls))

        # Sharpe y Sortino por trade
        std_pnl = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
        sharpe = (float(np.mean(pnls)) / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0.0

        downside_std = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
        sortino = (float(np.mean(pnls)) / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

        return {
            "initial_capital": account.initial_capital,
            "final_equity": round(account.equity, 2),
            "net_pnl_usd": round(net_pnl, 2),
            "total_trades": total_trades,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2) if not math.isinf(profit_factor) else 999.0,
            "max_drawdown_usd": round(account.current_drawdown, 2),
            "peak_equity": round(account.peak_equity, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "prop_status": "FAIL" if account.is_breached else ("PASS" if net_pnl > 0 else "FLAT_EVALUATION"),
            "breach_reason": account.breach_reason
        }
