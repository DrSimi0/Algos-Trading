"""
src/strategies/f5_optimizacion_is_n1.py
Implementación Matemática y Microestructural de la Versión Optimizada F5-Optimizacion -IS- N1:
Order Flow Imbalance de Nivel 1 con Normalización Robusta (Mediana / IQR)
y Filtro de Costo Transaccional Esperado (Economic Hurdle).

Fundamento Académico y Cuantitativo:
1. Cont, R., Kukanov, A., & Stoikov, S. (2014). The Price Impact of Order Book Events.
   Journal of Financial Econometrics, 12(1), 47-88.
2. Huber, P. J. (1981). Robust Statistics. John Wiley & Sons.
   (Fundamento de normalización por Mediana y Escala Normalizada por IQR / 1.34898).
3. Kyle, A. S. (1985). Continuous Auctions and Informed Trader. Econometrica.
4. Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices. JFE.

Mejora Matemática frente a F5 Base:
- Reduce el exceso de operaciones (overtrading) y la erosión por comisiones y slippage
  sustituyendo la media y desviación estándar tradicionales (vulnerables a colas pesadas y outliers)
  por la Mediana móvil y el Rango Intercuartílico (IQR / 1.34898).
- Implementa una compuerta económica (Economic Hurdle): solo ejecuta órdenes cuando el impacto
  de precio esperado E[Delta P] = |OFI - Mediana| / (2 * Profundidad) supera el costo
  transaccional esperado (comisión round-trip + medio spread + slippage esperado)
  multiplicado por un factor de seguridad kappa_hurdle (por defecto 1.5).
"""

from dataclasses import dataclass
from datetime import time, datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import math
import warnings
import numpy as np
import pandas as pd

from src.core.base_strategy import BaseStrategy
from src.core.futures_specs import get_spec
from src.strategies.order_flow_imbalance import (
    OFIExitReason,
    OFITradeRecord,
    OFIPropAccount,
    calculate_eb,
    calculate_ea,
    calculate_en,
    compute_ofi_from_events,
    generate_synthetic_lob_events,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. ESTRUCTURA DE DATOS EXTENDIDA DE OPERACIÓN OPTIMIZADA
# ==============================================================================

@dataclass
class RobustOFITradeRecord(OFITradeRecord):
    """
    Registro extendido de trade para F5-Optimizacion -IS- N1.
    Almacena las métricas analíticas de convicción y costo transaccional.
    """
    expected_impact_pts: float = 0.0     # Impacto de precio esperado E[Delta P] en puntos
    cost_hurdle_pts: float = 0.0         # Umbral de costo transaccional (kappa * C_trans) en puntos
    hurdle_ratio: float = 0.0            # Ratio E[Delta P] / (kappa * C_trans)
    z_score_robust: float = 0.0          # Puntuación estandarizada robusta por IQR


# ==============================================================================
# 2. FUNCIONES MATEMÁTICAS PURAS: NORMALIZACIÓN ROBUSTA Y COMPUERTA ECONÓMICA
# ==============================================================================

def calculate_robust_scale_iqr(iqr: float, eps_0: float = 1e-6, min_scale: float = 1.0) -> float:
    """
    Calcula el estimador de escala consistente e insesgado para distribuciones normales:
    sigma_robust = max(IQR / 1.3489795, min_scale)

    Propiedad: En una distribución normal N(mu, sigma^2), IQR = 2 * Phi^(-1)(0.75) * sigma ~ 1.34898 * sigma.
    Posee un punto de ruptura del 50%, siendo inmune a picos individuales o colas pesadas.
    """
    if not math.isfinite(iqr) or iqr < eps_0:
        return float(min_scale)
    return float(max(iqr / 1.3489795003921634, min_scale))


def calculate_robust_z_score(
    ofi_val: float,
    median_val: float,
    iqr_val: float,
    eps_0: float = 1e-6,
    min_scale: float = 1.0,
) -> float:
    """
    Calcula el Z-score robusto de Order Flow Imbalance:
    Z_robust = (OFI - Mediana) / ( (IQR / 1.34898) + eps_0 )

    Garantiza estabilidad numérica y finitud total en periodos de inactividad o varianza cero.
    """
    if not math.isfinite(ofi_val) or not math.isfinite(median_val):
        return 0.0

    scale = calculate_robust_scale_iqr(iqr_val, eps_0=eps_0, min_scale=min_scale)
    return float((ofi_val - median_val) / (scale + eps_0))


def calculate_expected_price_impact(
    ofi_residual: float,
    mean_depth: float,
    min_depth: float = 1.0,
) -> float:
    """
    Calcula el impacto de precio lineal esperado según el modelo de Cont, Kukanov & Stoikov (2014):
    E[Delta P] = beta * |OFI - Mediana| = |OFI - Mediana| / (2 * D_bar)

    Donde beta = 1 / (2 * D_bar) es la pendiente de impacto en unidades de precio por contrato.
    """
    if not math.isfinite(ofi_residual):
        return 0.0
    if not math.isfinite(mean_depth) or mean_depth < min_depth:
        depth = float(min_depth)
    else:
        depth = float(mean_depth)
    beta = 1.0 / (2.0 * depth)
    return float(beta * abs(ofi_residual))


def calculate_transaction_cost_pts(
    symbol: str,
    contract_type: str = "STD",
    tick_size: float = 0.25,
    commission_usd: Optional[float] = None,
    slippage_pts: Optional[float] = None,
) -> float:
    """
    Determina el costo transaccional total esperado expresado en puntos del subyacente:
    C_trans,pts = (Comisión_RT / Multiplicador) + Slippage_esperado + (Spread / 2)

    Cálculos oficiales en CME:
    - ES:  (4.40 USD / 50 USD/pt) + 0.00 pts slip + (0.25 / 2) = 0.088 + 0.125 = 0.213 pts (~0.85 ticks)
    - MES: (1.20 USD / 5 USD/pt)  + 0.00 pts slip + (0.25 / 2) = 0.240 + 0.125 = 0.365 pts (~1.46 ticks)
    - NQ:  (4.40 USD / 20 USD/pt) + 0.25 pts slip + (0.25 / 2) = 0.220 + 0.250 + 0.125 = 0.595 pts
    - MNQ: (1.20 USD / 2 USD/pt)  + 0.25 pts slip + (0.25 / 2) = 0.600 + 0.250 + 0.125 = 0.975 pts
    """
    spec = get_spec(symbol)
    multiplier = spec.get("multiplier", 50.0)

    if commission_usd is not None:
        comm = commission_usd
    else:
        comm = 4.40 if contract_type == "STD" else 1.20

    if slippage_pts is not None:
        slip = slippage_pts
    else:
        slip = 0.25 if "NQ" in symbol else 0.0

    half_spread = tick_size / 2.0
    comm_pts = comm / multiplier
    total_cost = comm_pts + slip + half_spread
    return float(total_cost)


def evaluate_economic_hurdle(
    expected_impact_pts: float,
    transaction_cost_pts: float,
    hurdle_multiplier: float = 1.5,
) -> Tuple[bool, float, float]:
    """
    Evalúa la compuerta analítica de costo económico:
    Aprobado si: E[Delta P] >= kappa_hurdle * C_trans,pts

    Retorna: (pasa_filtro, impacto_esperado_pts, costo_umbral_pts)
    """
    hurdle_threshold = hurdle_multiplier * transaction_cost_pts
    passed = bool(expected_impact_pts >= hurdle_threshold)
    return passed, float(expected_impact_pts), float(hurdle_threshold)


# ==============================================================================
# 3. IMPLEMENTACIÓN DE LA ESTRATEGIA: F5-OPTIMIZACION -IS- N1
# ==============================================================================

class F5OptimizacionISN1Strategy(BaseStrategy):
    """
    Estrategia F5-Optimizacion -IS- N1:
    Versión alterna optimizada de Order Flow Imbalance de Nivel 1.

    Principales mejoras matemáticas:
    1. Normalización robusta basada en Mediana rodante y Rango Intercuartílico (IQR / 1.34898).
       Inmune a ráfagas de outliers y distorsiones por colas pesadas del microflujo.
    2. Compuerta económica de costo transaccional (Economic Hurdle):
       Solo emite señales si el impacto de precio esperado supera holgadamente los costes de
       fricción (comisiones + medio spread + slippage), suprimiendo drásticamente el overtrading.
    3. Cumplimiento estricto del marco Prop Trading institucional de CME Globex:
       - Stop loss microestructural ceñido (6 ticks = 1.50 pts en ES).
       - Conmutación automática a microcontratos (`MES` / `MNQ`) cuando el presupuesto se reduce.
       - Disyuntor preventivo de Daily Loss Limit (DLL) y Trailing Drawdown.
       - Regla intradía de salida MOC forzosa (15:55 EST).
    """

    def __init__(
        self,
        name: str = "F5-Optimizacion -IS- N1",
        params: Optional[Dict[str, Any]] = None,
    ):
        default_params: Dict[str, Any] = {
            "symbol": "ES",
            "std_symbol": "ES",
            "micro_symbol": "MES",
            "window_w": 30,                 # Ventana de normalización W = 30 barras
            "theta_enter": 2.00,            # Umbral de disparo robusto |Z_robust| >= 2.00
            "theta_exhaust": 0.50,          # Umbral de agotamiento (Take-Profit de flujo)
            "theta_reversal": -1.25,        # Umbral de reversión adversa (Stop-Loss de flujo)
            "hurdle_multiplier": 1.50,      # Factor de seguridad kappa sobre costo transaccional
            "hurdle_enabled": True,         # Activar filtro económico de costos
            "tick_size": 0.25,              # Tick size oficial CME
            "tp_min_ticks": 4,              # Take-Profit mínimo en ticks (4 ticks = 1.00 pt)
            "sl_max_ticks": 6,              # Stop-Loss rígido en ticks (6 ticks = 1.50 pts)
            "h_max": 10,                    # Permanencia máxima permitida (10 barras)
            "gamma_book": 0.05,             # Máxima absorción permitida del LOB visible (5%)
            "phi_dll": 0.20,                # Presupuesto de riesgo: máx 20% del Daily Loss Limit
            "phi_tmdd": 0.15,               # Presupuesto de riesgo: máx 15% del Trailing Drawdown
            "max_std_contracts": 4,         # Máximo de contratos estándar
            "max_micro_contracts": 40,      # Máximo de microcontratos
            "exec_start_time": time(9, 45), # Inicio de ventana operativa regular RTH
            "exec_end_time": time(15, 55),  # Cierre mandatorio EOD MOC
            "epsilon_0": 1e-6,              # Guarda de estabilidad numérica
            "iqr_min_scale": 1.0,           # Escala mínima de seguridad para IQR
            "target_tz": "America/New_York",
            "source_tz": None,
        }
        if params:
            if "symbol" in params:
                sym = str(params["symbol"]).upper()
                if ("NQ" in sym or "MNQ" in sym) and "std_symbol" not in params:
                    default_params["std_symbol"] = "NQ"
                    default_params["micro_symbol"] = "MNQ"
                elif ("ES" in sym or "MES" in sym) and "std_symbol" not in params:
                    default_params["std_symbol"] = "ES"
                    default_params["micro_symbol"] = "MES"
            default_params.update(params)
        super().__init__(name=name, params=default_params)

    # --------------------------------------------------------------------------
    # 3.1 Normalización de Zona Horaria
    # --------------------------------------------------------------------------
    def ensure_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Garantiza la localización temporal uniforme en America/New_York."""
        target_tz = self.params["target_tz"]
        source_tz = self.params.get("source_tz")

        if df.index.tz is None:
            if source_tz:
                return df.tz_localize(source_tz).tz_convert(target_tz)
            else:
                return df.tz_localize(target_tz)
        else:
            return df.tz_convert(target_tz)

    # --------------------------------------------------------------------------
    # 3.2 Preparación de Datos y Reconstrucción de Microestructura
    # --------------------------------------------------------------------------
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara el DataFrame asegurando la existencia de variables indispensables:
        OFI, mean_depth, bid_price, ask_price y spread.
        Reutiliza la lógica no lineal de L1 de Cont et al. o deriva el OFI por proxy.
        """
        df = self.ensure_timezone(data.copy())

        l1_cols = {"bid_price", "bid_size", "ask_price", "ask_size"}
        if l1_cols.issubset({c.lower() for c in df.columns}):
            df = compute_ofi_from_events(df, freq="1min")

        if "OFI" not in df.columns:
            tick = self.params["tick_size"]
            std_sym = self.params["std_symbol"]
            typical_depth = 250.0 if "ES" in std_sym else 30.0

            if "Close" in df.columns and "Open" in df.columns and "Volume" in df.columns:
                ret_pts = df["Close"] - df["Open"]
                ticks_moved = ret_pts / tick
                vol_baseline = np.maximum(df["Volume"].rolling(30, min_periods=1).mean(), 1.0)
                df["OFI"] = ticks_moved * (df["Volume"] / vol_baseline) * 50.0
                df["mean_depth"] = typical_depth
                df["bid_price"] = df["Close"] - (tick / 2.0)
                df["ask_price"] = df["Close"] + (tick / 2.0)
                df["spread"] = tick
            else:
                raise ValueError("El DataFrame debe contener columnas microestructurales ('OFI', 'mean_depth') u OHLCV.")

        if "mean_depth" not in df.columns:
            std_sym = self.params["std_symbol"]
            df["mean_depth"] = 250.0 if "ES" in std_sym else 30.0

        if "bid_price" not in df.columns or "ask_price" not in df.columns:
            tick = self.params["tick_size"]
            base_p = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
            df["bid_price"] = base_p - (tick / 2.0)
            df["ask_price"] = base_p + (tick / 2.0)
            df["spread"] = tick

        if "spread" not in df.columns:
            df["spread"] = df["ask_price"] - df["bid_price"]

        return df

    # --------------------------------------------------------------------------
    # 3.3 Cálculo de Indicadores Robustos y Compuertas Económicas
    # --------------------------------------------------------------------------
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula:
        1. Mediana rodante de OFI: Med_{OFI, k} = Median(OFI_{k-W+1...k})
        2. Rango Intercuartílico móvil: IQR_k = Q75_k - Q25_k
        3. Escala robusta normalizada: sigma_{robust, k} = max(IQR_k / 1.34898, min_scale)
        4. Puntuación Z robusta: Z_{robust, k} = (OFI_k - Med_k) / (sigma_{robust, k} + eps_0)
        5. Impacto de precio esperado Cont et al.: E[Delta P_k] = |OFI_k - Med_k| / (2 * mean_depth_k)
        6. Costo transaccional esperado C_{trans, pts} y umbral de hurdle kappa * C_{trans}
        7. Compuerta económica de trade: hurdle_passed (True/False)
        8. Target analítico de Take Profit dinámico cuantizado a ticks.
        """
        w = self.params["window_w"]
        eps_0 = self.params["epsilon_0"]
        min_scale = self.params["iqr_min_scale"]
        tick = self.params["tick_size"]
        tp_min_ticks = self.params["tp_min_ticks"]
        hurdle_mult = self.params["hurdle_multiplier"]
        hurdle_enabled = self.params["hurdle_enabled"]
        std_sym = self.params["std_symbol"]

        ofi = df["OFI"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        typical_depth = 250.0 if "ES" in std_sym else 30.0
        mean_depth = df["mean_depth"].replace([np.inf, -np.inf], np.nan).fillna(typical_depth).clip(lower=1.0)

        # 1 & 2. Estadísticos robustos rodantes
        df["med_OFI"] = ofi.rolling(window=w, min_periods=w).median()
        df["q25_OFI"] = ofi.rolling(window=w, min_periods=w).quantile(0.25)
        df["q75_OFI"] = ofi.rolling(window=w, min_periods=w).quantile(0.75)
        df["iqr_OFI"] = (df["q75_OFI"] - df["q25_OFI"]).clip(lower=0.0)

        # 3. Escala robusta consistente
        norm_factor = 1.3489795003921634
        raw_sigma_robust = df["iqr_OFI"] / norm_factor
        df["sigma_robust"] = np.maximum(raw_sigma_robust.fillna(min_scale), min_scale)

        # 4. Z-Score Robusto
        df["Z_robust"] = (ofi - df["med_OFI"]) / (df["sigma_robust"] + eps_0)
        df["Z_robust"] = df["Z_robust"].fillna(0.0)

        # Alias Z_OFI para interoperabilidad con el motor de salida
        df["Z_OFI"] = df["Z_robust"]

        # 5. Impacto de precio esperado E[Delta P] = beta * |OFI - Mediana|
        df["beta_OFI"] = 1.0 / (2.0 * mean_depth)
        ofi_excess = (ofi - df["med_OFI"]).abs().fillna(0.0)
        df["expected_impact_pts"] = (df["beta_OFI"] * ofi_excess).fillna(0.0)

        # 6. Costo transaccional esperado en puntos
        active_sym = self.params.get("symbol", std_sym)
        c_type = "MICRO" if active_sym in ["MES", "MNQ"] else "STD"
        base_cost_pts = calculate_transaction_cost_pts(
            symbol=active_sym,
            contract_type=c_type,
            tick_size=tick,
        )
        df["trans_cost_pts"] = base_cost_pts
        df["hurdle_threshold_pts"] = hurdle_mult * base_cost_pts

        # 7. Compuerta económica
        if hurdle_enabled:
            df["hurdle_passed"] = df["expected_impact_pts"] >= df["hurdle_threshold_pts"]
        else:
            df["hurdle_passed"] = True

        # 8. Objetivo dinámico de Take Profit
        min_tp_pts = tp_min_ticks * tick
        dynamic_tp_pts = (df["beta_OFI"] * df["sigma_robust"].fillna(0.0)).fillna(0.0)
        raw_tp = np.maximum(min_tp_pts, dynamic_tp_pts)
        df["delta_p_tp"] = np.round(raw_tp / tick) * tick
        df["delta_p_tp"] = df["delta_p_tp"].fillna(min_tp_pts)

        return df

    # --------------------------------------------------------------------------
    # 3.4 Dimensionamiento de Posición y Conmutación Estándar / Micro
    # --------------------------------------------------------------------------
    def calculate_position_sizing(
        self,
        risk_budget_usd: float,
        mean_depth: float,
    ) -> Dict[str, Any]:
        """
        Determina contratos respetando:
        1. Capacidad de absorción del LOB: N_book_max = max(1, floor(gamma_book * mean_depth))
        2. Riesgo unitario oficial en CME (Stop 6 ticks + comisión round-trip + slippage)
        3. Conmutación automática a microcontratos si N_std == 0.
        """
        std_sym = self.params["std_symbol"]
        micro_sym = self.params["micro_symbol"]
        gamma_book = self.params["gamma_book"]
        sl_ticks = self.params["sl_max_ticks"]
        tick = self.params["tick_size"]
        delta_sl_pts = sl_ticks * tick

        max_std = self.params["max_std_contracts"]
        max_micro = self.params["max_micro_contracts"]

        if pd.isna(mean_depth) or not math.isfinite(mean_depth) or mean_depth <= 0:
            typical_depth = 250.0 if "ES" in std_sym else 30.0
            safe_depth = typical_depth
        else:
            safe_depth = float(mean_depth)

        n_book_max_std = max(1, int(math.floor(gamma_book * safe_depth)))
        n_book_max_micro = n_book_max_std * 10

        spec_std = get_spec(std_sym)
        spec_micro = get_spec(micro_sym)

        m_std = spec_std["multiplier"]
        m_micro = spec_micro["multiplier"]

        comm_std = 4.40
        comm_micro = 1.20
        slip_std = 5.00 if "NQ" in std_sym else 0.0
        slip_micro = 0.50 if "NQ" in std_sym else 0.0

        r_std = (delta_sl_pts * m_std) + comm_std + slip_std
        r_micro = (delta_sl_pts * m_micro) + comm_micro + slip_micro

        if risk_budget_usd <= 0:
            return {
                "symbol": micro_sym,
                "contract_type": "NONE",
                "contracts": 0,
                "risk_per_contract": r_micro,
                "status": "HALT_NO_BUDGET",
            }

        n_std = int(math.floor(risk_budget_usd / r_std))
        n_std = min(n_std, n_book_max_std, max_std)

        if n_std >= 1:
            return {
                "symbol": std_sym,
                "contract_type": "STD",
                "contracts": n_std,
                "risk_per_contract": r_std,
                "status": "OK",
            }

        if r_micro > risk_budget_usd:
            return {
                "symbol": micro_sym,
                "contract_type": "NONE",
                "contracts": 0,
                "risk_per_contract": r_micro,
                "status": "HALT_VOLATILITY_DEFENSE",
            }

        n_micro = int(math.floor(risk_budget_usd / r_micro))
        n_micro = min(n_micro, n_book_max_micro, max_micro)

        return {
            "symbol": micro_sym,
            "contract_type": "MICRO",
            "contracts": n_micro,
            "risk_per_contract": r_micro,
            "status": "OK" if n_micro > 0 else "HALT_VOLATILITY_DEFENSE",
        }

    # --------------------------------------------------------------------------
    # 3.5 Generación Vectorial de Señales (Interfaz BaseStrategy)
    # --------------------------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera la serie de posiciones de mercado (+1, -1, 0) aplicando:
        - Normalización robusta (Mediana / IQR)
        - Compuerta económica (Economic Hurdle anti-overtrading)
        - Filtro estricto de spread de 1 tick
        - Filtro de horario regular RTH
        - Tiempos de parada por SL, TP, agotamiento, reversión y decaimiento
        """
        df = self.prepare_data(data)
        df = self.calculate_indicators(df)

        theta_enter = self.params["theta_enter"]
        theta_exhaust = self.params["theta_exhaust"]
        theta_reversal = self.params["theta_reversal"]
        tick = self.params["tick_size"]
        sl_pts = self.params["sl_max_ticks"] * tick
        h_max = self.params["h_max"]
        t_start = self.params["exec_start_time"]
        t_end = self.params["exec_end_time"]

        signals = pd.Series(0.0, index=df.index)

        in_pos = False
        pos_dir = 0
        entry_price = 0.0
        entry_idx = 0
        tp_target_pts = 0.0

        for i, (ts, row) in enumerate(df.iterrows()):
            curr_time = ts.time() if hasattr(ts, "time") else None
            z = row["Z_robust"]
            spread = row["spread"]
            bid = row["bid_price"]
            ask = row["ask_price"]
            delta_tp = row["delta_p_tp"]
            hurdle_ok = row["hurdle_passed"]

            # Regla EOD Flat incondicional: Hora >= 15:55 EST
            if curr_time and curr_time >= t_end:
                in_pos = False
                pos_dir = 0
                signals.iloc[i] = 0.0
                continue

            if in_pos:
                bars_held = i - entry_idx
                exit_triggered = False

                high_b = (row["High"] - tick / 2.0) if "High" in row and pd.notna(row["High"]) else bid
                low_b = (row["Low"] - tick / 2.0) if "Low" in row and pd.notna(row["Low"]) else bid
                high_a = (row["High"] + tick / 2.0) if "High" in row and pd.notna(row["High"]) else ask
                low_a = (row["Low"] + tick / 2.0) if "Low" in row and pd.notna(row["Low"]) else ask

                if pos_dir == 1:
                    c_sl_price = (low_b <= (entry_price - sl_pts))
                    c_tp_price = (high_b >= (entry_price + tp_target_pts))
                    c_sl_flow = (z <= theta_reversal)
                    c_tp_flow = (z <= theta_exhaust) and (bid >= entry_price)
                    c_time = (bars_held >= h_max)

                    if c_sl_price or c_tp_price or c_sl_flow or c_tp_flow or c_time:
                        exit_triggered = True

                elif pos_dir == -1:
                    c_sl_price = (high_a >= (entry_price + sl_pts))
                    c_tp_price = (low_a <= (entry_price - tp_target_pts))
                    c_sl_flow = (z >= -theta_reversal)
                    c_tp_flow = (z >= -theta_exhaust) and (ask <= entry_price)
                    c_time = (bars_held >= h_max)

                    if c_sl_price or c_tp_price or c_sl_flow or c_tp_flow or c_time:
                        exit_triggered = True

                if exit_triggered:
                    in_pos = False
                    pos_dir = 0
                    signals.iloc[i] = 0.0
                else:
                    signals.iloc[i] = float(pos_dir)

            else:
                # Evaluación de nuevas entradas
                if curr_time and (curr_time < t_start or curr_time >= t_end):
                    continue

                # Filtro de spread estricto (1 tick)
                if not math.isclose(spread, tick, abs_tol=1e-4):
                    continue

                # Compuerta económica obligatoria: debe superar el costo transaccional esperado
                if not hurdle_ok:
                    continue

                if z >= theta_enter:
                    in_pos = True
                    pos_dir = 1
                    entry_price = ask
                    entry_idx = i
                    tp_target_pts = delta_tp
                    signals.iloc[i] = 1.0
                elif z <= -theta_enter:
                    in_pos = True
                    pos_dir = -1
                    entry_price = bid
                    entry_idx = i
                    tp_target_pts = delta_tp
                    signals.iloc[i] = -1.0

        return signals


# ==============================================================================
# 4. SIMULADOR CUANTITATIVO Y AUDITORÍA DE PROP TRADING PARA F5-OPTIMIZADA
# ==============================================================================

class F5OptimizacionISN1Backtester:
    """
    Motor de backtesting de alta fidelidad para la versión F5-Optimizacion -IS- N1.
    Audita:
    - Cumplimiento de Daily Loss Limit (DLL = $1,000)
    - Trailing Max Drawdown (TMDD = $2,000)
    - Filtrado de sobreoperación (reducción efectiva de comisiones)
    - Registro pormenorizado de trades con justificación analítica de hurdle
    """

    def __init__(
        self,
        strategy: Optional[F5OptimizacionISN1Strategy] = None,
        account: Optional[OFIPropAccount] = None,
    ):
        self.strategy = strategy or F5OptimizacionISN1Strategy()
        self.account = account or OFIPropAccount()
        self.trades: List[RobustOFITradeRecord] = []
        self.daily_logs: List[Dict[str, Any]] = []

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Ejecuta el backtest cronológico sesión a sesión."""
        df = self.strategy.prepare_data(data)
        df = self.strategy.calculate_indicators(df)

        theta_enter = self.strategy.params["theta_enter"]
        theta_exhaust = self.strategy.params["theta_exhaust"]
        theta_reversal = self.strategy.params["theta_reversal"]
        tick = self.strategy.params["tick_size"]
        sl_pts = self.strategy.params["sl_max_ticks"] * tick
        h_max = self.strategy.params["h_max"]
        t_start = self.strategy.params["exec_start_time"]
        t_end = self.strategy.params["exec_end_time"]
        phi_dll = self.strategy.params["phi_dll"]
        phi_tmdd = self.strategy.params["phi_tmdd"]

        self.trades = []
        self.daily_logs = []

        active_trade: Optional[RobustOFITradeRecord] = None
        grouped = df.groupby(df.index.normalize())

        for session_date, session_df in grouped:
            if self.account.is_breached:
                break

            self.account.start_session()
            session_start_equity = self.account.equity

            for i, (ts, row) in enumerate(session_df.iterrows()):
                if self.account.is_breached:
                    break

                curr_time = ts.time() if hasattr(ts, "time") else None
                z = row["Z_robust"]
                spread = row["spread"]
                bid = row["bid_price"]
                ask = row["ask_price"]
                mean_depth = row["mean_depth"]
                delta_tp = row["delta_p_tp"]
                hurdle_ok = row["hurdle_passed"]
                exp_impact = row["expected_impact_pts"]
                hurdle_threshold = row["hurdle_threshold_pts"]

                # Cierre MOC EOD 15:55 EST
                if curr_time and curr_time >= t_end:
                    if active_trade is not None:
                        self._close_trade(
                            trade=active_trade,
                            exit_time=ts,
                            exit_bar_idx=i,
                            exit_price=bid if active_trade.direction == 1 else ask,
                            exit_reason=OFIExitReason.OVERNIGHT_FLAT,
                        )
                        active_trade = None
                    continue

                if active_trade is not None:
                    bars_held = i - active_trade.entry_bar_idx
                    closed = False

                    high_b = (row["High"] - tick / 2.0) if "High" in row and pd.notna(row["High"]) else bid
                    low_b = (row["Low"] - tick / 2.0) if "Low" in row and pd.notna(row["Low"]) else bid
                    high_a = (row["High"] + tick / 2.0) if "High" in row and pd.notna(row["High"]) else ask
                    low_a = (row["Low"] + tick / 2.0) if "Low" in row and pd.notna(row["Low"]) else ask

                    has_open = ("Open" in row and pd.notna(row["Open"]))
                    open_b = (row["Open"] - tick / 2.0) if has_open else None
                    open_a = (row["Open"] + tick / 2.0) if has_open else None

                    if active_trade.direction == 1:
                        if low_b <= active_trade.stop_loss_price:
                            fill_p = open_b if (has_open and open_b < active_trade.stop_loss_price) else active_trade.stop_loss_price
                            self._close_trade(active_trade, ts, i, fill_p, OFIExitReason.STOP_LOSS_PRICE)
                            closed = True
                        elif high_b >= active_trade.take_profit_price:
                            fill_p = open_b if (has_open and open_b > active_trade.take_profit_price) else active_trade.take_profit_price
                            self._close_trade(active_trade, ts, i, fill_p, OFIExitReason.TAKE_PROFIT_PRICE)
                            closed = True
                        elif z <= theta_reversal:
                            self._close_trade(active_trade, ts, i, bid, OFIExitReason.FLOW_REVERSAL)
                            closed = True
                        elif z <= theta_exhaust and (bid >= active_trade.entry_price):
                            self._close_trade(active_trade, ts, i, bid, OFIExitReason.FLOW_EXHAUSTION)
                            closed = True
                        elif bars_held >= h_max:
                            self._close_trade(active_trade, ts, i, bid, OFIExitReason.TIME_DECAY)
                            closed = True

                    elif active_trade.direction == -1:
                        if high_a >= active_trade.stop_loss_price:
                            fill_p = open_a if (has_open and open_a > active_trade.stop_loss_price) else active_trade.stop_loss_price
                            self._close_trade(active_trade, ts, i, fill_p, OFIExitReason.STOP_LOSS_PRICE)
                            closed = True
                        elif low_a <= active_trade.take_profit_price:
                            fill_p = open_a if (has_open and open_a < active_trade.take_profit_price) else active_trade.take_profit_price
                            self._close_trade(active_trade, ts, i, fill_p, OFIExitReason.TAKE_PROFIT_PRICE)
                            closed = True
                        elif z >= -theta_reversal:
                            self._close_trade(active_trade, ts, i, ask, OFIExitReason.FLOW_REVERSAL)
                            closed = True
                        elif z >= -theta_exhaust and (ask <= active_trade.entry_price):
                            self._close_trade(active_trade, ts, i, ask, OFIExitReason.FLOW_EXHAUSTION)
                            closed = True
                        elif bars_held >= h_max:
                            self._close_trade(active_trade, ts, i, ask, OFIExitReason.TIME_DECAY)
                            closed = True

                    if closed:
                        active_trade = None

                else:
                    # Evaluación de entradas
                    if curr_time and (curr_time < t_start or curr_time >= t_end):
                        continue

                    if not math.isclose(spread, tick, abs_tol=1e-4):
                        continue

                    # Filtro económico de costo transaccional
                    if not hurdle_ok:
                        continue

                    direction = 0
                    if z >= theta_enter:
                        direction = 1
                    elif z <= -theta_enter:
                        direction = -1

                    if direction != 0:
                        risk_budget = self.account.calculate_risk_budget(phi_dll, phi_tmdd)
                        sizing = self.strategy.calculate_position_sizing(risk_budget, mean_depth)

                        if sizing["contracts"] > 0:
                            actual_cost_pts = calculate_transaction_cost_pts(
                                symbol=sizing["symbol"],
                                contract_type=sizing["contract_type"],
                                tick_size=tick,
                            )
                            actual_hurdle_threshold = self.strategy.params["hurdle_multiplier"] * actual_cost_pts
                            if self.strategy.params["hurdle_enabled"] and exp_impact < actual_hurdle_threshold:
                                continue

                            entry_p = ask if direction == 1 else bid
                            stop_p = (entry_p - sl_pts) if direction == 1 else (entry_p + sl_pts)
                            tp_p = (entry_p + delta_tp) if direction == 1 else (entry_p - delta_tp)

                            active_trade = RobustOFITradeRecord(
                                date=str(session_date.date() if hasattr(session_date, "date") else session_date),
                                symbol=sizing["symbol"],
                                direction=direction,
                                contracts=sizing["contracts"],
                                contract_type=sizing["contract_type"],
                                entry_bar_idx=i,
                                entry_time=ts,
                                entry_price=entry_p,
                                stop_loss_price=stop_p,
                                take_profit_price=tp_p,
                                delta_sl_points=sl_pts,
                                delta_tp_points=delta_tp,
                                z_score_entry=z,
                                expected_impact_pts=round(float(exp_impact), 4),
                                cost_hurdle_pts=round(float(actual_hurdle_threshold), 4),
                                hurdle_ratio=round(float(exp_impact / (actual_hurdle_threshold + 1e-8)), 2),
                                z_score_robust=round(float(z), 4),
                            )

            if active_trade is not None:
                last_ts = session_df.index[-1]
                last_bid = session_df["bid_price"].iloc[-1]
                last_ask = session_df["ask_price"].iloc[-1]
                self._close_trade(
                    active_trade,
                    last_ts,
                    len(session_df) - 1,
                    last_bid if active_trade.direction == 1 else last_ask,
                    OFIExitReason.OVERNIGHT_FLAT,
                )
                active_trade = None

            day_pnl = self.account.equity - session_start_equity
            self.daily_logs.append({
                "date": session_date,
                "day_pnl": round(day_pnl, 2),
                "ending_equity": round(self.account.equity, 2),
                "peak_equity": round(self.account.peak_equity, 2),
                "trailing_floor": round(self.account.trailing_floor, 2),
                "is_breached": self.account.is_breached,
            })

        return self._generate_report()

    def _close_trade(
        self,
        trade: RobustOFITradeRecord,
        exit_time: Any,
        exit_bar_idx: int,
        exit_price: float,
        exit_reason: OFIExitReason,
    ) -> None:
        """Calcula el resultado neto, fricciones de CME y actualiza la cuenta."""
        trade.exit_time = exit_time
        trade.exit_bar_idx = exit_bar_idx
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.holding_bars = exit_bar_idx - trade.entry_bar_idx

        spec = get_spec(trade.symbol)
        m = spec["multiplier"]

        pts_diff = (exit_price - trade.entry_price) if trade.direction == 1 else (trade.entry_price - exit_price)
        gross_pnl = pts_diff * m * trade.contracts

        comm_per_contract = 4.40 if trade.contract_type == "STD" else 1.20
        slip_per_contract = 5.00 if "NQ" in trade.symbol and trade.contract_type == "STD" else (
            0.50 if "NQ" in trade.symbol and trade.contract_type == "MICRO" else 0.0
        )

        total_comm = comm_per_contract * trade.contracts
        total_slip = slip_per_contract * trade.contracts
        net_pnl = gross_pnl - total_comm - total_slip

        trade.gross_pnl_usd = round(gross_pnl, 2)
        trade.commission_usd = round(total_comm, 2)
        trade.slippage_usd = round(total_slip, 2)
        trade.net_pnl_usd = round(net_pnl, 2)

        self.account.update_equity_intraday(net_pnl)
        self.trades.append(trade)

    def _generate_report(self) -> Dict[str, Any]:
        """Genera el resumen cuantitativo de métricas auditadas."""
        trades_df = self.get_trades_dataframe()
        net_profits = trades_df["net_pnl_usd"] if not trades_df.empty else pd.Series(dtype=float)

        total_trades = len(trades_df)
        wins = net_profits[net_profits > 0]
        losses = net_profits[net_profits < 0]

        win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        total_net_pnl = float(self.account.equity - self.account.initial_capital)
        total_return_pct = (total_net_pnl / self.account.initial_capital) * 100.0

        peak = self.account.initial_capital
        max_dd = 0.0
        running = self.account.initial_capital
        for p in net_profits:
            running += p
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd

        if len(net_profits) > 1 and net_profits.std(ddof=1) > 0:
            sharpe = float((net_profits.mean() / net_profits.std(ddof=1)) * np.sqrt(252 * 5))
            downside = net_profits[net_profits < 0]
            if len(downside) > 1 and downside.std(ddof=1) > 1e-6:
                sortino = float((net_profits.mean() / downside.std(ddof=1)) * np.sqrt(252 * 5))
            elif len(downside) == 0 and len(wins) > 0:
                sortino = float(sharpe * 1.5) if sharpe > 0 else 0.0
            else:
                sortino = 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        return {
            "strategy_name": self.strategy.name,
            "initial_capital": self.account.initial_capital,
            "final_equity": round(self.account.equity, 2),
            "total_net_pnl_usd": round(total_net_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "total_trades": total_trades,
            "winning_trades": int(len(wins)),
            "losing_trades": int(len(losses)),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2) if not math.isinf(profit_factor) else "INF",
            "gross_profit_usd": round(gross_profit, 2),
            "gross_loss_usd": round(gross_loss, 2),
            "total_commissions_usd": round(float(trades_df["commission_usd"].sum()), 2) if not trades_df.empty else 0.0,
            "total_slippage_usd": round(float(trades_df["slippage_usd"].sum()), 2) if not trades_df.empty else 0.0,
            "max_drawdown_usd": round(max_dd, 2),
            "max_drawdown_pct": round((max_dd / self.account.initial_capital) * 100.0, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "is_prop_breached": self.account.is_breached,
            "breach_reason": self.account.breach_reason or "COMPLIANT (Ningún límite violado)",
        }

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Retorna el DataFrame con el registro completo de operaciones."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self.trades])


# Alias semánticos para máxima flexibilidad de importación
RobustOrderFlowImbalanceStrategy = F5OptimizacionISN1Strategy
OrderFlowImbalanceOptimizedStrategy = F5OptimizacionISN1Strategy
RobustOFIBacktester = F5OptimizacionISN1Backtester
