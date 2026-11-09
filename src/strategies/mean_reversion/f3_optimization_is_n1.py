"""
src/strategies/mean_reversion/f3_optimization_is_n1.py

Implementación formal de la versión alterna optimizada:
F3-Optimizacion -IS- N1
Anchored VWAP & Rotational Value Area Reversion con Filtro Avanzado de Régimen
para Futuros CME (ES, MES, NQ, MNQ).

Especificación matemática de referencia:
    reports/optimizaciones_estrategias/F3_optimizacion_anchored_vwap_value_area.md

Mejoras matemáticas implementadas sobre la versión base F3:
    1. Filtro de Régimen Ultra-Sensible por Pendiente Normalizada de VWAP (kappa_norm):
       kappa_norm,t = |Delta_VWAP_t| / (k * sigma_VWAP,t) con k=3 barras (15 min).
       Invariante a escala de precios e inmune al retardo de fase del ADX.
       Veta de inmediato intentos de reversión cuando el consenso institucional se desplaza.
    2. Compuerta de Aceptación de Área de Valor Previa (Phi_AMT,t):
       Calcula el Área de Valor (70% de volumen RTH) de la sesión previa (VAH_t-1, VAL_t-1).
       Si la subasta abre fuera del rango y es aceptada en los primeros 30 minutos, se cataloga
       como Trend Day / Value Migration Day, anulando las operaciones contra-tendencia.
    3. Asimetría y Dispersión de Volumen Intradía (Volume Skewness & Kurtosis):
       Calcula momentos centrales de orden 3 y 4 ponderados por volumen:
         - Skew_intra,t: modula asimétricamente las bandas k_lower y k_upper en [2.0, 2.4] sigma.
         - Ratio de volumen transado fuera del Área de Valor previa (Phi_vol_outside).
    4. Disyuntor Catastrófico Estocástico Ajustado a +-2.75 sigma (reducción desde +-3.0 sigma).
    5. Trailing Stop Protector de Convergencia:
       Al alcanzar >= 60% del recorrido proyectado hacia el VWAP, el stop loss se actualiza
       al extremo de +-1.0 sigma para blindar la operación.
    6. Blindaje Contractual Prop Trading Reforzado:
       - Disyuntor de pérdida diaria a alpha_DLL = 0.85 * DLL.
       - Parada diaria tras K_loss_intraday >= 2 pérdidas consecutivas en el día.
       - Ventana de reversión ajustada W_revers [10:15, 15:00) EST.
       - Salida obligatoria por decaimiento temporal y cierre a las 15:15:00 EST.
"""

from typing import Dict, Any, List, Optional, Tuple
import datetime
import numpy as np
import pandas as pd

from ...core.base_strategy import BaseStrategy
from ...core.futures_specs import get_spec


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calcula el Average Directional Index (ADX) sobre barras OHLCV según el estándar de Wilder.
    
    Args:
        df (pd.DataFrame): Datos con 'High', 'Low', 'Close'.
        period (int): Período de suavizado (por defecto 14).
        
    Returns:
        pd.Series: Valores de ADX indexados por fecha.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    # True Range (TR)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement (+DM, -DM)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    # Suavizado exponencial de Wilder (alpha = 1 / period)
    alpha = 1.0 / period
    smoothed_tr = pd.Series(tr, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    smoothed_plus_dm = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    smoothed_minus_dm = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()

    # Indicadores Direccionales (+DI, -DI)
    tr_safe = smoothed_tr.replace(0.0, np.nan)
    plus_di = 100.0 * (smoothed_plus_dm / tr_safe)
    minus_di = 100.0 * (smoothed_minus_dm / tr_safe)

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * (di_diff / di_sum.replace(0.0, np.nan))
    dx = dx.fillna(0.0)

    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx


def compute_prior_day_value_areas(
    df: pd.DataFrame,
    rth_start: str = "09:30",
    rth_end: str = "16:00",
    value_area_pct: float = 0.70
) -> Dict[datetime.date, Tuple[float, float, float]]:
    """
    Calcula el Área de Valor (VAL, VAH y POC) para cada sesión regular (RTH) previa.
    
    El Área de Valor representa el rango de precios que concentra el 70%
    del volumen negociado durante la sesión regular de contado.
    
    Returns:
        Dict[datetime.date, Tuple[float, float, float]]: Mapeo de fecha actual -> (VAL_prev, VAH_prev, POC_prev).
    """
    start_h, start_m = [int(x) for x in rth_start.split(":")]
    end_h, end_m = [int(x) for x in rth_end.split(":")]
    t_start = datetime.time(start_h, start_m)
    t_end = datetime.time(end_h, end_m)

    p_typ = (df["High"] + df["Low"] + df["Close"]) / 3.0
    dates = df.index.date
    if len(dates) == 0:
        return {}

    times = [t.time() for t in df.index]
    change_idx = np.where(dates[1:] != dates[:-1])[0] + 1
    day_starts = np.r_[0, change_idx, len(dates)]

    unique_dates = []
    daily_va: Dict[datetime.date, Tuple[float, float, float]] = {}

    lower_pct = (1.0 - value_area_pct) / 2.0  # 0.15 para 70%
    upper_pct = 1.0 - lower_pct              # 0.85 para 70%

    for k in range(len(day_starts) - 1):
        s_idx = day_starts[k]
        e_idx = day_starts[k + 1]
        d = dates[s_idx]
        unique_dates.append(d)

        rth_indices = [idx for idx in range(s_idx, e_idx) if t_start <= times[idx] <= t_end]
        if len(rth_indices) == 0:
            continue

        prices = p_typ.iloc[rth_indices].values
        vols = df["Volume"].iloc[rth_indices].values

        total_vol = np.sum(vols)
        if total_vol <= 0.0:
            val = float(np.min(prices))
            vah = float(np.max(prices))
            poc = float(np.median(prices))
        else:
            sort_idx = np.argsort(prices)
            sorted_p = prices[sort_idx]
            sorted_v = vols[sort_idx]
            cum_v = np.cumsum(sorted_v)
            cum_pct = cum_v / total_vol

            # 15th percentile y 85th percentile de volumen acumulado
            idx_val = np.searchsorted(cum_pct, lower_pct)
            idx_val = min(idx_val, len(sorted_p) - 1)
            val = float(sorted_p[idx_val])

            idx_vah = np.searchsorted(cum_pct, upper_pct)
            idx_vah = min(idx_vah, len(sorted_p) - 1)
            vah = float(sorted_p[idx_vah])

            # POC: precio típico con mayor volumen en la sesión
            max_v_idx = np.argmax(vols)
            poc = float(prices[max_v_idx])

        daily_va[d] = (val, vah, poc)

    # Crear mapeo hacia el día siguiente
    prior_va_map: Dict[datetime.date, Tuple[float, float, float]] = {}
    prev_tuple: Optional[Tuple[float, float, float]] = None

    for d in unique_dates:
        if prev_tuple is not None:
            prior_va_map[d] = prev_tuple
        if d in daily_va:
            prev_tuple = daily_va[d]

    return prior_va_map


def compute_anchored_vwap_optimized_bands(
    df: pd.DataFrame,
    anchor_time: str = "09:30",
    session_end_time: str = "17:00",
    k_band: float = 2.0,
    k_catastrophic: float = 2.75,
    kappa_crit: float = 2.2,
    adx_period: int = 14,
    bar_minutes: float = 5.0,
    slope_lookback_bars: int = 3,
    max_vol_outside_ratio: float = 0.55,
    min_kurtosis: float = 1.7
) -> pd.DataFrame:
    """
    Calcula el Anchored VWAP, la varianza y desviación estándar ponderada,
    el sesgo (Skewness) y curtosis (Kurtosis) de volumen, las bandas asimétricas adaptadas,
    la pendiente normalizada del VWAP corregida por inercia, la compuerta de aceptación de Área de Valor
    previa (Phi_AMT) y el filtro unificado de régimen de equilibrio rotacional.
    """
    res = df.copy()
    anchor_h, anchor_m = [int(x) for x in anchor_time.split(":")]
    anchor_tod = datetime.time(anchor_h, anchor_m)

    end_h, end_m = [int(x) for x in session_end_time.split(":")]
    session_end_tod = datetime.time(end_h, end_m)

    # Precio típico Pi_typ = (Hi + Li + Ci) / 3
    res["P_typ"] = (res["High"] + res["Low"] + res["Close"]) / 3.0
    res["PV"] = res["P_typ"] * res["Volume"]
    res["PV2"] = (res["P_typ"] ** 2) * res["Volume"]

    # Calcular ADX global
    res["ADX"] = calculate_adx(res, period=adx_period)

    # Pre-calcular Áreas de Valor de la sesión previa
    prior_va_map = compute_prior_day_value_areas(res, rth_start=anchor_time, rth_end="16:00")

    dates = res.index.date
    unique_dates = list(dict.fromkeys(dates))
    times = [t.time() for t in res.index]

    n = len(res)
    vwap_arr = np.full(n, np.nan, dtype=np.float64)
    sigma_arr = np.full(n, np.nan, dtype=np.float64)
    skew_arr = np.zeros(n, dtype=np.float64)
    kurt_arr = np.full(n, 3.0, dtype=np.float64)

    upper_band_arr = np.full(n, np.nan, dtype=np.float64)
    lower_band_arr = np.full(n, np.nan, dtype=np.float64)
    upper_cat_arr = np.full(n, np.nan, dtype=np.float64)
    lower_cat_arr = np.full(n, np.nan, dtype=np.float64)

    slope_raw_arr = np.zeros(n, dtype=np.float64)
    slope_norm_arr = np.zeros(n, dtype=np.float64)
    slope_eff_arr = np.zeros(n, dtype=np.float64)
    is_anchored_arr = np.zeros(n, dtype=bool)

    prior_val_arr = np.full(n, np.nan, dtype=np.float64)
    prior_vah_arr = np.full(n, np.nan, dtype=np.float64)
    prior_poc_arr = np.full(n, np.nan, dtype=np.float64)

    phi_amt_arr = np.ones(n, dtype=int)
    vol_outside_ratio_arr = np.zeros(n, dtype=np.float64)
    phi_slope_arr = np.ones(n, dtype=int)
    phi_kurt_arr = np.ones(n, dtype=int)
    regime_rot_arr = np.zeros(n, dtype=int)

    p_typ_vals = res["P_typ"].values
    vol_vals = res["Volume"].values
    close_vals = res["Close"].values

    change_idx = np.where(dates[1:] != dates[:-1])[0] + 1
    day_starts = np.r_[0, change_idx, len(dates)]

    for k in range(len(day_starts) - 1):
        s_idx = day_starts[k]
        e_idx = day_starts[k + 1]
        day_indices = range(s_idx, e_idx)
        d = dates[s_idx]

        prior_va = prior_va_map.get(d)
        has_prior_va = prior_va is not None
        if has_prior_va:
            p_val, p_vah, p_poc = prior_va
            prior_val_arr[s_idx:e_idx] = p_val
            prior_vah_arr[s_idx:e_idx] = p_vah
            prior_poc_arr[s_idx:e_idx] = p_poc
        else:
            p_val, p_vah, p_poc = np.nan, np.nan, np.nan

        # Determinar Outside Value Acceptance en los primeros 30 minutos (primeras 6 barras de 5m)
        is_outside_value_day = False
        if has_prior_va:
            # Obtener índices de barras entre 09:30 y 10:00 EST
            open_indices = [idx for idx in day_indices if anchor_tod <= times[idx] < datetime.time(10, 0)]
            if len(open_indices) >= 6:
                first_6 = open_indices[:6]
                all_above = all(close_vals[i] > p_vah for i in first_6)
                all_below = all(close_vals[i] < p_val for i in first_6)
                if all_above or all_below:
                    is_outside_value_day = True

        day_phi_amt = 0 if is_outside_value_day else 1

        # Acumuladores de momentos O(1) con desplazamiento p0 para estabilidad numérica
        p0 = p_typ_vals[s_idx]
        cum_v = 0.0
        cum_vy = 0.0
        cum_vy2 = 0.0
        cum_vy3 = 0.0
        cum_vy4 = 0.0
        cum_v_outside = 0.0
        anchor_started = False
        anchor_local_indices = []

        for idx in day_indices:
            t = times[idx]
            if anchor_tod <= t <= session_end_tod:
                anchor_started = True
            elif t > session_end_tod:
                anchor_started = False

            if anchor_started and (anchor_tod <= t <= session_end_tod):
                is_anchored_arr[idx] = True
                anchor_local_indices.append(idx)
                v = vol_vals[idx]
                p = p_typ_vals[idx]
                c = close_vals[idx]
                y = p - p0

                cum_v += v
                cum_vy += y * v
                cum_vy2 += (y ** 2) * v
                cum_vy3 += (y ** 3) * v
                cum_vy4 += (y ** 4) * v

                # Monitoreo de volumen fuera del Área de Valor previa
                if has_prior_va:
                    if c > p_vah or c < p_val:
                        cum_v_outside += v
                    ratio_outside = (cum_v_outside / cum_v) if cum_v > 0.0 else 0.0
                else:
                    ratio_outside = 0.0
                vol_outside_ratio_arr[idx] = ratio_outside

                if cum_v > 0.0:
                    mu_y = cum_vy / cum_v
                    vw = p0 + mu_y
                    vwap_arr[idx] = vw

                    # Varianza y Desviación Estándar
                    var = max(0.0, (cum_vy2 / cum_v) - (mu_y ** 2))
                    sig = np.sqrt(var)
                    sigma_arr[idx] = sig

                    # Cálculo exacto de Sesgo (Skewness) y Curtosis (Kurtosis) en O(1)
                    if sig > 1e-6:
                        m3 = (cum_vy3 / cum_v) - 3.0 * mu_y * (cum_vy2 / cum_v) + 2.0 * (mu_y ** 3)
                        m4 = (cum_vy4 / cum_v) - 4.0 * mu_y * (cum_vy3 / cum_v) + 6.0 * (mu_y ** 2) * (cum_vy2 / cum_v) - 3.0 * (mu_y ** 4)
                        skew = m3 / (sig ** 3)
                        kurt = m4 / (sig ** 4)
                        skew = np.clip(skew, -4.0, 4.0)
                        kurt = np.clip(kurt, 1.0, 15.0)
                    else:
                        skew = 0.0
                        kurt = 3.0

                    skew_arr[idx] = skew
                    kurt_arr[idx] = kurt

                    # Modulación Asimétrica de Bandas según Skewness
                    # k_lower se expande si skew < 0; k_upper se expande si skew > 0
                    k_lower_eff = k_band + min(0.40, max(0.0, -0.50 * skew))
                    k_upper_eff = k_band + min(0.40, max(0.0, 0.50 * skew))

                    upper_band_arr[idx] = vw + k_upper_eff * sig
                    lower_band_arr[idx] = vw - k_lower_eff * sig
                    upper_cat_arr[idx] = vw + k_catastrophic * sig
                    lower_cat_arr[idx] = vw - k_catastrophic * sig

                    # Pendiente Crítica Normalizada del VWAP (lookback k = 3 barras = 15m)
                    lookback_k = slope_lookback_bars
                    if len(anchor_local_indices) >= (lookback_k + 1):
                        idx_lag = anchor_local_indices[-(lookback_k + 1)]
                        vw_lag = vwap_arr[idx_lag]
                        if not np.isnan(vw_lag):
                            delta_vwap = vw - vw_lag
                            raw_slope = delta_vwap / (lookback_k * bar_minutes)
                            # Normalización clásica invariante: |Delta_VWAP| / (k * sigma)
                            norm_slope = abs(delta_vwap) / (lookback_k * max(sig, 0.50))

                            # Normalización corregida por inercia de volumen: |Delta_VWAP| / (w_k * sigma)
                            v_k = sum(vol_vals[anchor_local_indices[-lookback_k:]])
                            w_k = (v_k / cum_v) if cum_v > 0.0 else 1.0
                            eff_disp = (abs(delta_vwap) / (w_k * max(sig, 0.50))) if w_k > 0.0 else 0.0

                            slope_raw_arr[idx] = raw_slope
                            slope_norm_arr[idx] = norm_slope
                            slope_eff_arr[idx] = eff_disp
                        else:
                            slope_raw_arr[idx] = 0.0
                            slope_norm_arr[idx] = 0.0
                            slope_eff_arr[idx] = 0.0
                    else:
                        slope_raw_arr[idx] = 0.0
                        slope_norm_arr[idx] = 0.0
                        slope_eff_arr[idx] = 0.0
                else:
                    vwap_arr[idx] = p
                    sigma_arr[idx] = 0.0
                    upper_band_arr[idx] = p
                    lower_band_arr[idx] = p
                    upper_cat_arr[idx] = p
                    lower_cat_arr[idx] = p
                    slope_raw_arr[idx] = 0.0
                    slope_norm_arr[idx] = 0.0
                    slope_eff_arr[idx] = 0.0

                # Clasificación de Régimen
                if kappa_crit <= 0.20:
                    phi_slope = 1 if slope_norm_arr[idx] <= kappa_crit else 0
                else:
                    phi_slope = 1 if slope_eff_arr[idx] <= kappa_crit else 0

                phi_amt = day_phi_amt
                phi_vol_outside = 1 if (ratio_outside <= max_vol_outside_ratio or t < datetime.time(10, 15)) else 0
                phi_kurt = 1 if (kurt >= min_kurtosis or t < datetime.time(10, 15)) else 0

                phi_amt_arr[idx] = phi_amt
                phi_slope_arr[idx] = phi_slope
                phi_kurt_arr[idx] = phi_kurt

                # Régimen Rotacional Equilibrado: pendiente estable, sin desbalance AMT, sin dispersión excesiva y perfil unimodal
                if phi_slope == 1 and phi_amt == 1 and phi_vol_outside == 1 and phi_kurt == 1:
                    regime_rot_arr[idx] = 1
                else:
                    regime_rot_arr[idx] = 0

    res["VWAP"] = vwap_arr
    res["Sigma_VWAP"] = sigma_arr
    res["Volume_Skewness"] = skew_arr
    res["Volume_Kurtosis"] = kurt_arr
    res["UpperBand"] = upper_band_arr
    res["LowerBand"] = lower_band_arr
    res["UpperCatastrophic"] = upper_cat_arr
    res["LowerCatastrophic"] = lower_cat_arr
    res["VWAP_Slope"] = slope_raw_arr
    res["VWAP_Slope_Norm"] = slope_norm_arr
    res["VWAP_Slope_Eff"] = slope_eff_arr
    res["IsAnchored"] = is_anchored_arr

    res["Prior_VAL"] = prior_val_arr
    res["Prior_VAH"] = prior_vah_arr
    res["Prior_POC"] = prior_poc_arr
    res["Phi_AMT"] = phi_amt_arr
    res["Phi_Slope"] = phi_slope_arr
    res["Phi_Kurtosis"] = phi_kurt_arr
    res["Ratio_Vol_Outside"] = vol_outside_ratio_arr
    res["Regime_Rotational"] = regime_rot_arr

    return res


class PropRiskManagerOptimized:
    """
    Gestor Cuantitativo de Riesgo y Dimensionamiento Optimizado para Prop Trading (F3-N1).
    
    Implementa:
        - Daily Loss Limit (DLL) con disyuntor preventivo en alpha_DLL = 0.85 * DLL.
        - Parada diaria preventiva si se registran K_loss_intraday >= 2 pérdidas consecutivas en el día.
        - Maximum Trailing Drawdown (MTDD) y marca de agua máxima (HWM).
        - Amortiguador por rachas de pérdidas consecutivas: phi(K_loss) = 0.30 * 2^(-min(K_loss, 3)).
        - Presupuesto de riesgo por operación:
          B_F3, t = min(phi(K_loss) * max(0, DLL - L_day, t), psi * max(0, MTDD - DD_t)) con psi = 0.20.
        - Conmutación automática Estándar (N_std <= 2) -> Micro (N_micro <= 10).
        - Disyuntor de volatilidad excesiva: si R_contract(M_micro) > B_F3 => N = 0 (HALT_VOLATILITY_DEFENSE).
    """

    def __init__(
        self,
        account_size: float = 50000.0,
        daily_loss_limit: float = 1000.0,
        max_trailing_drawdown: float = 2000.0,
        psi: float = 0.20,
        max_std_contracts: int = 1,
        max_micro_contracts: int = 10,
        alpha_dll: float = 0.85,
        max_intraday_consecutive_losses: int = 2,
        symbol: str = "ES",
        tick_size: float = 0.25,
        std_multiplier: float = 50.0,
        micro_multiplier: float = 5.0,
        std_commission_rt: float = 4.50,
        micro_commission_rt: float = 1.24,
        consecutive_losses: int = 0
    ):
        self.account_size = account_size
        self.daily_loss_limit = daily_loss_limit
        self.max_trailing_drawdown = max_trailing_drawdown
        self.psi = psi
        self.max_std_contracts = max_std_contracts
        self.max_micro_contracts = max_micro_contracts
        self.alpha_dll = alpha_dll
        self.max_intraday_consecutive_losses = max_intraday_consecutive_losses

        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.std_multiplier = std_multiplier
        self.micro_multiplier = micro_multiplier
        self.std_commission_rt = std_commission_rt
        self.micro_commission_rt = micro_commission_rt

        # Fricción combinada: 2 ticks round-turn de slippage + comisión
        self.c_fric_std = 2.0 * self.tick_size * self.std_multiplier + self.std_commission_rt
        self.c_fric_micro = 2.0 * self.tick_size * self.micro_multiplier + self.micro_commission_rt

        # Variables de estado
        self.current_equity = account_size
        self.peak_equity = account_size
        self.current_date: Optional[datetime.date] = None
        self.day_net_pnl = 0.0
        self.day_realized_loss = 0.0
        self.consecutive_losses = consecutive_losses
        self.intraday_consecutive_losses = 0
        self.is_halted_day = False
        self.is_halted_account = False

    def reset_day_if_new(self, date: datetime.date):
        """Reinicia la pérdida realizada y contador de racha intradía al comenzar una nueva fecha."""
        if self.current_date is None or date != self.current_date:
            self.current_date = date
            self.day_net_pnl = 0.0
            self.day_realized_loss = 0.0
            self.intraday_consecutive_losses = 0
            self.is_halted_day = False

    def update_floating_equity(self, equity: float):
        """
        Actualiza la marca de agua máxima (HWM) y disyuntor de cuenta durante posiciones abiertas (intraday floating).
        """
        if equity > self.peak_equity:
            self.peak_equity = equity
        trailing_dd = self.peak_equity - equity
        if trailing_dd >= self.max_trailing_drawdown:
            self.is_halted_account = True

    def phi_k_loss(self) -> float:
        """Calcula el factor de escala por racha perdedora: phi(K_loss) = 0.30 * 2^(-min(K_loss, 3))."""
        k = min(self.consecutive_losses, 3)
        return 0.30 * (2.0 ** (-k))

    def get_risk_budget(self) -> float:
        """
        Calcula el presupuesto analítico de riesgo admisible para el trade actual (B_F3, t).
        """
        if self.is_halted_account or self.is_halted_day:
            return 0.0

        trailing_dd = max(0.0, self.peak_equity - self.current_equity)
        rem_trailing_dd = max(0.0, self.max_trailing_drawdown - trailing_dd)
        rem_daily_budget = max(0.0, self.daily_loss_limit - self.day_realized_loss)

        if rem_trailing_dd <= 0.0:
            self.is_halted_account = True
            return 0.0

        # Disyuntor preventivo si pérdida diaria excede alpha_DLL * DLL
        if self.day_realized_loss >= (self.alpha_dll * self.daily_loss_limit):
            self.is_halted_day = True
            return 0.0

        if self.intraday_consecutive_losses >= self.max_intraday_consecutive_losses:
            self.is_halted_day = True
            return 0.0

        budget = min(
            self.phi_k_loss() * rem_daily_budget,
            self.psi * rem_trailing_dd
        )
        return max(0.0, budget)

    def calculate_position_size(
        self,
        entry_price: float,
        sl_price: float
    ) -> Tuple[int, str, float, str]:
        """
        Determina la cantidad de contratos, tipo (STANDARD vs MICRO) y riesgo por contrato.
        
        Returns:
            Tuple[int, str, float, str]:
                - n_contracts (int): número de contratos a operar.
                - contract_type (str): 'STANDARD', 'MICRO' o 'NONE'.
                - r_contract (float): riesgo monetario por contrato.
                - status (str): 'OK', 'HALT_VOLATILITY_DEFENSE', 'DAILY_LIMIT_REACHED', etc.
        """
        budget = self.get_risk_budget()
        if budget <= 0.0:
            if self.is_halted_account:
                return 0, "NONE", 0.0, "ACCOUNT_TRAILING_DD_BREACH"
            if self.intraday_consecutive_losses >= self.max_intraday_consecutive_losses:
                return 0, "NONE", 0.0, "MAX_INTRADAY_LOSSES_REACHED"
            return 0, "NONE", 0.0, "DAILY_LOSS_LIMIT_REACHED"

        delta_sl = abs(entry_price - sl_price)
        if delta_sl <= 0.0:
            return 0, "NONE", 0.0, "INVALID_STOP_DISTANCE"

        # Riesgo unitario por contrato
        r_contract_std = delta_sl * self.std_multiplier + self.c_fric_std
        r_contract_micro = delta_sl * self.micro_multiplier + self.c_fric_micro

        # 1. Intentar contrato estándar (acotado a max_std_contracts)
        n_std = int(np.floor(budget / r_contract_std))
        if n_std >= 1:
            n_std = min(n_std, self.max_std_contracts)
            return n_std, "STANDARD", r_contract_std, "OK"

        # 2. Conmutar a micro-contratos (acotado a max_micro_contracts)
        n_micro = int(np.floor(budget / r_contract_micro))
        n_micro = min(n_micro, self.max_micro_contracts)

        if n_micro >= 1:
            return n_micro, "MICRO", r_contract_micro, "OK"

        # 3. Disyuntor de volatilidad excesiva
        return 0, "NONE", r_contract_micro, "HALT_VOLATILITY_DEFENSE"

    def record_trade_result(self, net_pnl: float):
        """
        Actualiza el estado de la cuenta tras el cierre de un trade.
        """
        self.current_equity += net_pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        # Seguimiento de balance neto diario
        self.day_net_pnl += net_pnl
        self.day_realized_loss = max(0.0, -self.day_net_pnl)

        if net_pnl < 0.0:
            self.consecutive_losses += 1
            self.intraday_consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            self.intraday_consecutive_losses = 0

        # Verificar límites de Prop Trading
        trailing_dd = self.peak_equity - self.current_equity
        if trailing_dd >= self.max_trailing_drawdown:
            self.is_halted_account = True

        if self.day_realized_loss >= (self.alpha_dll * self.daily_loss_limit):
            self.is_halted_day = True

        if self.intraday_consecutive_losses >= self.max_intraday_consecutive_losses:
            self.is_halted_day = True


class F3OptimizationISN1Strategy(BaseStrategy):
    """
    Estrategia Cuantitativa F3-Optimizacion -IS- N1:
    Anchored VWAP & Rotational Value Area Reversion con Filtro Avanzado de Régimen.
    
    Corrige el problema de operar reversión en tendencias fuertes mediante:
      1. Pendiente normalizada del VWAP (kappa_norm).
      2. Compuerta de aceptación de Área de Valor previa (Phi_AMT) y distribución de volumen.
      3. Bandas asimétricas adaptadas al sesgo de volumen (Volume Skewness).
      4. Trailing Stop protector de convergencia (60% del recorrido al VWAP).
      5. Disyuntor catastrófico ajustado a 2.75 sigma.
    """

    def __init__(
        self,
        name: str = "F3-Optimizacion -IS- N1",
        params: Optional[Dict[str, Any]] = None
    ):
        default_params = {
            "symbol": "ES",
            "anchor_time": "09:30",
            "window_init_end": "10:15",        # 10:15 EST en la versión optimizada
            "window_revers_end": "15:00",
            "window_inhibit_end": "15:15",
            "window_flat_end": "16:00",
            "k_band": 2.0,                     # 2.0 para ES, 2.5 para NQ
            "k_catastrophic": 2.75,            # Disyuntor optimizado a 2.75 sigma
            "kappa_crit": 2.2,                 # Umbral crítico para pendiente normalizada de VWAP
            "slope_lookback_bars": 3,
            "max_vol_outside_ratio": 0.55,     # 0.55 para filtrar migración direccional de valor
            "min_kurtosis": 1.7,               # Curtosis mínima de volumen para balance/equilibrio
            "min_dist_to_vwap_ratio": 0.80,    # Recorrido mínimo (0.8 sigma) para entrada
            "adx_period": 14,
            "k_buffer_ticks": 3,               # Colchón de stop optimizado a 3 ticks (0.75 pts)
            "convergence_protect_ratio": 0.60, # Trailing stop al 60% de camino al VWAP
            "tick_size": 0.25,
            "account_size": 50000.0,
            "daily_loss_limit": 1000.0,
            "max_trailing_drawdown": 2000.0,
            "psi": 0.20,
            "max_std_contracts": 1,            # 1 contrato estándar para proteger estrictamente el MTDD
            "max_micro_contracts": 10,
            "alpha_dll": 0.85,
            "max_intraday_consecutive_losses": 2,
            "bar_minutes": 5.0,
            "session_end_time": "17:00"
        }
        if params:
            default_params.update(params)

        symbol_upper = default_params["symbol"].upper()
        if symbol_upper in ["NQ", "MNQ"]:
            if "k_band" not in (params or {}):
                default_params["k_band"] = 2.5
            if "kappa_crit" not in (params or {}):
                default_params["kappa_crit"] = 2.2

        super().__init__(name=name, params=default_params)

    def prepare_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los indicadores matemáticos requeridos para la estrategia optimizada."""
        self.validate_data(data)
        p = self.params
        df = compute_anchored_vwap_optimized_bands(
            df=data,
            anchor_time=p["anchor_time"],
            session_end_time=p.get("session_end_time", "17:00"),
            k_band=p["k_band"],
            k_catastrophic=p["k_catastrophic"],
            kappa_crit=p["kappa_crit"],
            adx_period=p["adx_period"],
            bar_minutes=p["bar_minutes"],
            slope_lookback_bars=p["slope_lookback_bars"],
            max_vol_outside_ratio=p["max_vol_outside_ratio"],
            min_kurtosis=p.get("min_kurtosis", 1.7)
        )
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera una serie de posiciones o señales sobre los datos.
        
        Valores:
            1.0  = Posición Larga
            -1.0 = Posición Corta
            0.0  = Posición Neutral / Flat
        """
        df = self.prepare_indicators(data)
        p = self.params

        w_revers_start = datetime.time(*[int(x) for x in p["window_init_end"].split(":")])
        w_revers_end = datetime.time(*[int(x) for x in p["window_revers_end"].split(":")])
        w_flat_time = datetime.time(*[int(x) for x in p["window_inhibit_end"].split(":")])

        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        if n < 2:
            return signals

        position = 0.0
        entry_price = 0.0
        entry_vwap = 0.0
        sl_price = 0.0
        buffer_pts = p["k_buffer_ticks"] * p["tick_size"]
        convergence_ratio = p["convergence_protect_ratio"]
        min_dist_ratio = p.get("min_dist_to_vwap_ratio", 0.80)

        for i in range(1, n):
            t = df.index[i].time()
            date = df.index[i].date()
            prev_date = df.index[i - 1].date()

            # Forzar cierre si es nuevo día
            if date != prev_date and position != 0.0:
                position = 0.0

            # Cierre forzoso intradía a las 15:15 EST
            if t >= w_flat_time and position != 0.0:
                position = 0.0
                signals.iloc[i] = 0.0
                continue

            current_vwap = df["VWAP"].iloc[i]
            current_sig = df["Sigma_VWAP"].iloc[i]
            current_upper_cat = df["UpperCatastrophic"].iloc[i]
            current_lower_cat = df["LowerCatastrophic"].iloc[i]

            # Si estamos en posición, evaluar trailing stop protector y salidas dinámicas
            # (Prioridad conservadora: STOP LOSS / disyuntor se evalúa antes de take profit)
            dist_target = abs(entry_vwap - entry_price) if position != 0.0 else 0.0
            if position == 1.0:
                stop_barrier = max(sl_price, current_lower_cat)
                if df["Low"].iloc[i] <= stop_barrier:
                    position = 0.0
                elif df["High"].iloc[i] >= current_vwap:
                    position = 0.0

                if position != 0.0 and dist_target > 0.0:
                    favorable_gain = df["High"].iloc[i] - entry_price
                    if favorable_gain >= (convergence_ratio * dist_target):
                        candidate_protect = max(entry_price, current_vwap - 1.0 * current_sig)
                        sl_price = max(sl_price, candidate_protect)

            elif position == -1.0:
                stop_barrier = min(sl_price, current_upper_cat)
                if df["High"].iloc[i] >= stop_barrier:
                    position = 0.0
                elif df["Low"].iloc[i] <= current_vwap:
                    position = 0.0

                if position != 0.0 and dist_target > 0.0:
                    favorable_gain = entry_price - df["Low"].iloc[i]
                    if favorable_gain >= (convergence_ratio * dist_target):
                        candidate_protect = min(entry_price, current_vwap + 1.0 * current_sig)
                        sl_price = min(sl_price, candidate_protect)

            # Si estamos neutrales, evaluar entradas en W_revers [10:15, 15:00)
            if position == 0.0 and (w_revers_start <= t < w_revers_end):
                regime = df["Regime_Rotational"].iloc[i]
                if regime == 1 and df["IsAnchored"].iloc[i]:
                    prev_low = df["Low"].iloc[i - 1]
                    prev_lower_band = df["LowerBand"].iloc[i - 1]
                    curr_close = df["Close"].iloc[i]
                    curr_lower_band = df["LowerBand"].iloc[i]

                    prev_high = df["High"].iloc[i - 1]
                    prev_upper_band = df["UpperBand"].iloc[i - 1]
                    curr_upper_band = df["UpperBand"].iloc[i]

                    min_dist = min_dist_ratio * current_sig

                    # Long Reversion (requiere distancia mínima a VWAP para justificar trade)
                    is_long = (prev_low < prev_lower_band) and (curr_close >= curr_lower_band) and (curr_close < current_vwap - min_dist)
                    # Short Reversion (requiere distancia mínima a VWAP para justificar trade)
                    is_short = (prev_high > prev_upper_band) and (curr_close <= curr_upper_band) and (curr_close > current_vwap + min_dist)

                    if is_long and not is_short:
                        position = 1.0
                        entry_price = curr_close
                        entry_vwap = current_vwap
                        sl_price = min(prev_low, df["Low"].iloc[i]) - buffer_pts
                    elif is_short and not is_long:
                        position = -1.0
                        entry_price = curr_close
                        entry_vwap = current_vwap
                        sl_price = max(prev_high, df["High"].iloc[i]) + buffer_pts

            signals.iloc[i] = position

        return signals

    def run_backtest(
        self,
        data: pd.DataFrame,
        risk_manager: Optional[PropRiskManagerOptimized] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta el backtest formal con gestión de riesgo Prop Trading intra-barra y modelo de fricción CME.
        """
        df = self.prepare_indicators(data)
        p = self.params

        sym = p["symbol"].upper()
        if sym in ["NQ", "MNQ"]:
            std_sym = "NQ"
            micro_sym = "MNQ"
        else:
            std_sym = "ES"
            micro_sym = "MES"

        spec_std = get_spec(std_sym)
        spec_micro = get_spec(micro_sym)

        if risk_manager is None:
            risk_manager = PropRiskManagerOptimized(
                account_size=p["account_size"],
                daily_loss_limit=p["daily_loss_limit"],
                max_trailing_drawdown=p["max_trailing_drawdown"],
                psi=p["psi"],
                max_std_contracts=p.get("max_std_contracts", 1),
                max_micro_contracts=p.get("max_micro_contracts", 10),
                alpha_dll=p.get("alpha_dll", 0.85),
                max_intraday_consecutive_losses=p.get("max_intraday_consecutive_losses", 2),
                symbol=std_sym,
                tick_size=p["tick_size"],
                std_multiplier=spec_std["multiplier"],
                micro_multiplier=spec_micro["multiplier"],
                std_commission_rt=spec_std["commission_rt"],
                micro_commission_rt=spec_micro["commission_rt"]
            )

        w_revers_start = datetime.time(*[int(x) for x in p["window_init_end"].split(":")])
        w_revers_end = datetime.time(*[int(x) for x in p["window_revers_end"].split(":")])
        w_flat_time = datetime.time(*[int(x) for x in p["window_inhibit_end"].split(":")])

        buffer_pts = p["k_buffer_ticks"] * p["tick_size"]
        convergence_ratio = p["convergence_protect_ratio"]
        min_dist_ratio = p.get("min_dist_to_vwap_ratio", 0.80)

        trades: List[Dict[str, Any]] = []
        equity_series = pd.Series(index=df.index, dtype=np.float64)

        in_position = False
        direction = 0  # +1 Long, -1 Short
        entry_idx = -1
        entry_time = None
        entry_price = 0.0
        entry_vwap = 0.0
        sl_price = 0.0
        initial_sl_price = 0.0
        trailing_sl_active = False
        contracts = 0
        contract_type = "NONE"
        contract_multiplier = 0.0
        contract_friction = 0.0

        n = len(df)
        for i in range(1, n):
            current_bar = df.iloc[i]
            t = df.index[i].time()
            date = df.index[i].date()
            prev_date = df.index[i - 1].date()

            # Reiniciar presupuesto diario al inicio del día
            risk_manager.reset_day_if_new(date)

            # Cierre forzoso de anomalía si posición cruza cambio de día
            if in_position and date != prev_date:
                exit_price = current_bar["Open"]
                exit_reason = "DAY_END_OVERNIGHT_CLOSE"
                in_position = False
                pnl_pts = (exit_price - entry_price) * direction
                gross_pnl = pnl_pts * contract_multiplier * contracts
                friction = contract_friction * contracts
                net_pnl = gross_pnl - friction
                risk_manager.record_trade_result(net_pnl)
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "contract_type": contract_type,
                    "contracts": contracts,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pts": pnl_pts,
                    "gross_pnl": gross_pnl,
                    "friction": friction,
                    "net_pnl": net_pnl,
                    "exit_reason": exit_reason,
                    "equity": risk_manager.current_equity
                })

            current_vwap = current_bar["VWAP"]
            current_sig = current_bar["Sigma_VWAP"]
            current_upper_cat = current_bar["UpperCatastrophic"]
            current_lower_cat = current_bar["LowerCatastrophic"]

            # Evaluar salidas si estamos en posición
            # Prioridad conservadora institucional: Stop Loss / Catastrophic stop se evalúa primero
            if in_position:
                exit_triggered = False
                exit_price = 0.0
                exit_reason = ""

                # Prioridad 1: Cierre Forzoso a las 15:15 EST
                if t >= w_flat_time:
                    exit_triggered = True
                    exit_price = current_bar["Close"]
                    exit_reason = "FORCED_TIME_DECAY_1515"

                elif direction == 1:
                    stop_barrier = max(sl_price, current_lower_cat)
                    if current_bar["Low"] <= stop_barrier:
                        exit_triggered = True
                        if sl_price >= current_lower_cat:
                            exit_reason = "TRAILING_PROTECT_SL" if trailing_sl_active else "STOP_LOSS"
                            exit_price = min(current_bar["Open"], sl_price)
                        else:
                            exit_reason = "2.75_SIGMA_CIRCUIT_BREAKER"
                            exit_price = min(current_bar["Open"], current_lower_cat)
                    elif current_bar["High"] >= current_vwap:
                        exit_triggered = True
                        exit_price = max(current_bar["Open"], current_vwap)
                        exit_reason = "TAKE_PROFIT_VWAP"

                elif direction == -1:
                    stop_barrier = min(sl_price, current_upper_cat)
                    if current_bar["High"] >= stop_barrier:
                        exit_triggered = True
                        if sl_price <= current_upper_cat:
                            exit_reason = "TRAILING_PROTECT_SL" if trailing_sl_active else "STOP_LOSS"
                            exit_price = max(current_bar["Open"], sl_price)
                        else:
                            exit_reason = "2.75_SIGMA_CIRCUIT_BREAKER"
                            exit_price = max(current_bar["Open"], current_upper_cat)
                    elif current_bar["Low"] <= current_vwap:
                        exit_triggered = True
                        exit_price = min(current_bar["Open"], current_vwap)
                        exit_reason = "TAKE_PROFIT_VWAP"

                # Si no hubo salida, actualizar Trailing Stop Protector para la siguiente barra
                dist_target = abs(entry_vwap - entry_price)
                if not exit_triggered and dist_target > 0.0:
                    if direction == 1:
                        favorable_gain = current_bar["High"] - entry_price
                        if favorable_gain >= (convergence_ratio * dist_target):
                            candidate_protect = max(entry_price, current_vwap - 1.0 * current_sig)
                            if candidate_protect > sl_price:
                                sl_price = candidate_protect
                                trailing_sl_active = True
                    elif direction == -1:
                        favorable_gain = entry_price - current_bar["Low"]
                        if favorable_gain >= (convergence_ratio * dist_target):
                            candidate_protect = min(entry_price, current_vwap + 1.0 * current_sig)
                            if candidate_protect < sl_price:
                                sl_price = candidate_protect
                                trailing_sl_active = True

                if exit_triggered:
                    in_position = False
                    pnl_pts = (exit_price - entry_price) * direction
                    gross_pnl = pnl_pts * contract_multiplier * contracts
                    friction = contract_friction * contracts
                    net_pnl = gross_pnl - friction
                    risk_manager.record_trade_result(net_pnl)

                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": df.index[i],
                        "direction": "LONG" if direction == 1 else "SHORT",
                        "contract_type": contract_type,
                        "contracts": contracts,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl_pts": pnl_pts,
                        "gross_pnl": gross_pnl,
                        "friction": friction,
                        "net_pnl": net_pnl,
                        "exit_reason": exit_reason,
                        "equity": risk_manager.current_equity
                    })

            # Evaluar entradas si NO estamos en posición y en W_revers [10:15, 15:00)
            if not in_position and (w_revers_start <= t < w_revers_end):
                if current_bar["Regime_Rotational"] == 1 and current_bar["IsAnchored"]:
                    prev_low = df["Low"].iloc[i - 1]
                    prev_lower_band = df["LowerBand"].iloc[i - 1]
                    curr_close = current_bar["Close"]
                    curr_lower_band = current_bar["LowerBand"]

                    prev_high = df["High"].iloc[i - 1]
                    prev_upper_band = df["UpperBand"].iloc[i - 1]
                    curr_upper_band = current_bar["UpperBand"]

                    min_dist = min_dist_ratio * current_sig
                    candidate_direction = 0
                    candidate_sl = 0.0

                    is_long = (prev_low < prev_lower_band) and (curr_close >= curr_lower_band) and (curr_close < current_vwap - min_dist)
                    is_short = (prev_high > prev_upper_band) and (curr_close <= curr_upper_band) and (curr_close > current_vwap + min_dist)

                    if is_long and not is_short:
                        candidate_direction = 1
                        candidate_sl = min(prev_low, current_bar["Low"]) - buffer_pts
                    elif is_short and not is_long:
                        candidate_direction = -1
                        candidate_sl = max(prev_high, current_bar["High"]) + buffer_pts

                    if candidate_direction != 0:
                        n_cnt, c_type, _, status = risk_manager.calculate_position_size(
                            entry_price=curr_close,
                            sl_price=candidate_sl
                        )

                        if n_cnt >= 1 and status == "OK":
                            in_position = True
                            direction = candidate_direction
                            entry_idx = i
                            entry_time = df.index[i]
                            entry_price = curr_close
                            entry_vwap = current_vwap
                            sl_price = candidate_sl
                            initial_sl_price = candidate_sl
                            trailing_sl_active = False
                            contracts = n_cnt
                            contract_type = c_type
                            if contract_type == "STANDARD":
                                contract_multiplier = risk_manager.std_multiplier
                                contract_friction = risk_manager.c_fric_std
                            else:
                                contract_multiplier = risk_manager.micro_multiplier
                                contract_friction = risk_manager.c_fric_micro

            # Tracking continuo de equidad con actualización de HWM flotante
            if in_position:
                open_pnl = (current_bar["Close"] - entry_price) * direction * contract_multiplier * contracts
                bar_equity = risk_manager.current_equity + open_pnl
                risk_manager.update_floating_equity(bar_equity)
                equity_series.iloc[i] = bar_equity
            else:
                equity_series.iloc[i] = risk_manager.current_equity

        equity_series.iloc[0] = risk_manager.account_size
        equity_series = equity_series.ffill().bfill()

        df_trades = pd.DataFrame(trades)
        metrics = self._calculate_backtest_metrics(
            df_trades=df_trades,
            equity_series=equity_series,
            initial_capital=risk_manager.account_size
        )

        return {
            "metrics": metrics,
            "trades": df_trades,
            "equity_curve": equity_series,
            "data": df
        }

    def _calculate_backtest_metrics(
        self,
        df_trades: pd.DataFrame,
        equity_series: pd.Series,
        initial_capital: float
    ) -> Dict[str, Any]:
        """Calcula métricas financieras y de control de riesgo Prop Trading."""
        final_equity = equity_series.iloc[-1]
        total_net_pnl = final_equity - initial_capital
        total_return_pct = (total_net_pnl / initial_capital) * 100.0

        peak = equity_series.cummax()
        drawdown_usd = peak - equity_series
        max_drawdown_usd = drawdown_usd.max()
        max_drawdown_pct = ((drawdown_usd / peak) * 100.0).max()

        total_trades = len(df_trades)
        if total_trades > 0:
            winning_trades = (df_trades["net_pnl"] > 0.0).sum()
            losing_trades = (df_trades["net_pnl"] < 0.0).sum()
            win_rate = (winning_trades / total_trades) * 100.0

            gross_profits = df_trades[df_trades["net_pnl"] > 0]["net_pnl"].sum()
            gross_losses = abs(df_trades[df_trades["net_pnl"] < 0]["net_pnl"].sum())
            profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (float("inf") if gross_profits > 0 else 0.0)

            avg_trade = df_trades["net_pnl"].mean()
            best_trade = df_trades["net_pnl"].max()
            worst_trade = df_trades["net_pnl"].min()

            standard_trades = (df_trades["contract_type"] == "STANDARD").sum()
            micro_trades = (df_trades["contract_type"] == "MICRO").sum()

            exit_reasons = df_trades["exit_reason"].value_counts().to_dict()
        else:
            winning_trades = 0
            losing_trades = 0
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade = 0.0
            best_trade = 0.0
            worst_trade = 0.0
            standard_trades = 0
            micro_trades = 0
            exit_reasons = {}

        return {
            "Capital Inicial ($)": round(initial_capital, 2),
            "Capital Final ($)": round(final_equity, 2),
            "P&L Neto Total ($)": round(total_net_pnl, 2),
            "Retorno Total (%)": round(total_return_pct, 2),
            "Max Trailing Drawdown ($)": round(max_drawdown_usd, 2),
            "Max Trailing Drawdown (%)": round(max_drawdown_pct, 2),
            "Total Trades": total_trades,
            "Trades Ganadores": winning_trades,
            "Trades Perdedores": losing_trades,
            "Win Rate (%)": round(win_rate, 2),
            "Profit Factor": round(profit_factor, 2) if profit_factor != float("inf") else "Inf",
            "Trade Promedio ($)": round(avg_trade, 2),
            "Mejor Trade ($)": round(best_trade, 2),
            "Peor Trade ($)": round(worst_trade, 2),
            "Trades Estándar": standard_trades,
            "Trades Micro": micro_trades,
            "Desglose de Salidas": exit_reasons
        }
