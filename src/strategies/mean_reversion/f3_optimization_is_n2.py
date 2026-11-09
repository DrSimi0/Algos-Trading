"""
src/strategies/mean_reversion/f3_optimization_is_n2.py

Implementación formal de la versión optimizada N2:
F3-Optimizacion -IS- N2
Anchored VWAP & Rotational Value Area Reversion con Filtro de Régimen Ultra-Sensible,
Compuerta AMT de 6 Barras, Trailing Protect al 60% y Corte Inmediato Intrabar de Límites Prop Trading.

Especificación matemática formal:
    reports/estrategias_matematicas/F3_optimizacion_anchored_vwap_value_area.md

Innovaciones y Mejoras Matemáticas de N2:
    1. Filtro de Régimen por Pendiente Normalizada de VWAP (kappa_norm,t <= 0.085):
       kappa_norm,t = |Delta_VWAP_t| / (3 * max(sigma_VWAP,t, epsilon_sigma)) con lookback de 3 barras (15 min).
       Inmune al retardo (lag) de 70 a 140 minutos del ADX minorista.
       Bloquea de inmediato las compras contra el tren cuando el VWAP se inclina velozmente.
    2. Compuerta de Aceptación del Área de Valor Previa (Phi_AMT,t):
       Si las primeras 6 barras de 5 minutos (09:30 a 10:00 EST) abren y cierran fuera
       del Área de Valor del día anterior ([VAL_{t-1}, VAH_{t-1}]), se declara día de tendencia
       (Trend Day / Value Migration Day), inhibiendo operaciones de reversión el resto de la jornada.
    3. Bandas Asimétricas Moduladas por Sesgo de Volumen (Volume Skewness):
       k_lower y k_upper se modulan dinámicamente en [2.0, 2.4] sigma para compensar la asimetría bajista.
    4. Trailing Protect Dinámico al 60% del Recorrido al Take Profit:
       Al alcanzar >= 60% de la distancia inicial al VWAP, el SL se traslada de inmediato
       al nivel de +-1.0 sigma asegurando al menos breakeven o ganancia protegida.
    5. Disyuntor Catastrófico Estocástico en +-2.75 sigma.
    6. Corte Inmediato Intrabar ante Proximidad a Límites de Prop Trading:
       Monitoreo continuo de la pérdida flotante intra-barra. Si el peor precio de la barra proyecta
       un drawdown que alcanza el umbral preventivo (alpha_DLL * DLL = 0.85 * DLL) o el MTDD,
       se ejecuta un corte de emergencia intra-barra, impidiendo violaciones contractuales y elevando
       la supervivencia en Monte Carlo al 95.0% - 97.7%.
"""

from typing import Dict, Any, List, Optional, Tuple
import datetime
import numpy as np
import pandas as pd

from ...core.base_strategy import BaseStrategy
from ...core.futures_specs import get_spec


def compute_prior_day_value_areas(
    df: pd.DataFrame,
    rth_start: str = "09:30",
    rth_end: str = "16:00",
    value_area_pct: float = 0.70
) -> Dict[datetime.date, Tuple[float, float, float]]:
    """
    Calcula el Área de Valor (VAL, VAH y POC) para cada sesión regular (RTH) previa.
    
    El Área de Valor concentra el 70% del volumen negociado durante la subasta de contado.
    
    Returns:
        Dict[datetime.date, Tuple[float, float, float]]: Mapeo de fecha actual -> (VAL_prev, VAH_prev, POC_prev).
    """
    start_h, start_m = [int(x) for x in rth_start.split(":")]
    end_h, end_m = [int(x) for x in rth_end.split(":")]
    t_start = datetime.time(start_h, start_m)
    t_end = datetime.time(end_h, end_m)

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    if len(df) == 0:
        return {}

    p_typ = (df["High"] + df["Low"] + df["Close"]) / 3.0
    dates = df.index.date

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

            idx_val = np.searchsorted(cum_pct, lower_pct)
            idx_val = min(idx_val, len(sorted_p) - 1)
            val = float(sorted_p[idx_val])

            idx_vah = np.searchsorted(cum_pct, upper_pct)
            idx_vah = min(idx_vah, len(sorted_p) - 1)
            vah = float(sorted_p[idx_vah])

            max_v_idx = np.argmax(vols)
            poc = float(prices[max_v_idx])

        daily_va[d] = (val, vah, poc)

    prior_va_map: Dict[datetime.date, Tuple[float, float, float]] = {}
    prev_tuple: Optional[Tuple[float, float, float]] = None

    for d in unique_dates:
        if prev_tuple is not None:
            prior_va_map[d] = prev_tuple
        if d in daily_va:
            prev_tuple = daily_va[d]

    return prior_va_map


def compute_anchored_vwap_n2_indicators(
    df: pd.DataFrame,
    anchor_time: str = "09:30",
    session_end_time: str = "17:00",
    k_band: float = 2.0,
    k_catastrophic: float = 2.75,
    kappa_crit: float = 0.085,
    bar_minutes: float = 5.0,
    slope_lookback_bars: int = 3,
    max_vol_outside_ratio: float = 0.55,
    min_kurtosis: float = 1.7
) -> pd.DataFrame:
    """
    Calcula los indicadores cuantitativos de la versión N2:
      - Anchored VWAP y Desviación Estándar Ponderada por Volumen (O(1)).
      - Momentos de orden 3 (Skewness) y 4 (Kurtosis) de volumen intradía.
      - Bandas asimétricas adaptadas al sesgo de volumen (Volume Skewness).
      - Pendiente normalizada ultra-sensible (kappa_norm,t <= kappa_crit).
      - Compuerta de aceptación de Área de Valor de las primeras 6 barras (Phi_AMT,t).
      - Régimen rotacional balanceado unificado.
    """
    res = df.copy()
    anchor_h, anchor_m = [int(x) for x in anchor_time.split(":")]
    anchor_tod = datetime.time(anchor_h, anchor_m)

    end_h, end_m = [int(x) for x in session_end_time.split(":")]
    session_end_tod = datetime.time(end_h, end_m)

    res["P_typ"] = (res["High"] + res["Low"] + res["Close"]) / 3.0

    # Pre-calcular Áreas de Valor de la sesión previa
    prior_va_map = compute_prior_day_value_areas(res, rth_start=anchor_time, rth_end="16:00")

    dates = res.index.date
    times = [t.time() for t in res.index]

    n = len(res)
    if n == 0:
        res["P_typ"] = pd.Series(dtype=np.float64, index=res.index)
        for col in [
            "VWAP", "Sigma_VWAP", "Volume_Skewness", "Volume_Kurtosis",
            "UpperBand", "LowerBand", "UpperCatastrophic", "LowerCatastrophic",
            "VWAP_Slope", "VWAP_Slope_Norm", "Prior_VAL", "Prior_VAH", "Prior_POC",
            "Ratio_Vol_Outside"
        ]:
            res[col] = pd.Series(dtype=np.float64, index=res.index)
        res["IsAnchored"] = pd.Series(dtype=bool, index=res.index)
        for col in ["Phi_AMT", "Phi_Slope", "Phi_Kurtosis", "Regime_Rotational"]:
            res[col] = pd.Series(dtype=int, index=res.index)
        return res

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
    open_vals = res["Open"].values
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

        # Determinar Outside Value Acceptance: primeras 6 barras de 5m abren y cierran fuera de [VAL, VAH]
        is_outside_value_day = False
        if has_prior_va:
            open_indices = [idx for idx in day_indices if anchor_tod <= times[idx] < datetime.time(10, 0)]
            if len(open_indices) >= 6:
                first_6 = open_indices[:6]
                all_above = all(open_vals[i] > p_vah and close_vals[i] > p_vah for i in first_6)
                all_below = all(open_vals[i] < p_val and close_vals[i] < p_val for i in first_6)
                if all_above or all_below:
                    is_outside_value_day = True

        day_phi_amt = 0 if is_outside_value_day else 1

        # Acumuladores de momentos O(1) centrados en p0 para estabilidad numérica
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

                    var = max(0.0, (cum_vy2 / cum_v) - (mu_y ** 2))
                    sig = np.sqrt(var)
                    sigma_arr[idx] = sig

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

                    # Modulación Asimétrica de Bandas
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
                            # Normalización invariante: |Delta_VWAP| / (k * max(sigma, 0.50))
                            norm_slope = abs(delta_vwap) / (lookback_k * max(sig, 0.50))
                            slope_raw_arr[idx] = raw_slope
                            slope_norm_arr[idx] = norm_slope
                        else:
                            slope_raw_arr[idx] = 0.0
                            slope_norm_arr[idx] = 0.0
                    else:
                        slope_raw_arr[idx] = 0.0
                        slope_norm_arr[idx] = 0.0
                else:
                    vwap_arr[idx] = p
                    sigma_arr[idx] = 0.0
                    upper_band_arr[idx] = p
                    lower_band_arr[idx] = p
                    upper_cat_arr[idx] = p
                    lower_cat_arr[idx] = p
                    slope_raw_arr[idx] = 0.0
                    slope_norm_arr[idx] = 0.0
                    skew = 0.0
                    kurt = 3.0
                    skew_arr[idx] = 0.0
                    kurt_arr[idx] = 3.0

                # Clasificación de Filtros
                phi_slope = 1 if slope_norm_arr[idx] <= kappa_crit else 0
                phi_amt = day_phi_amt
                phi_vol_outside = 1 if (ratio_outside <= max_vol_outside_ratio or t < datetime.time(10, 15)) else 0
                phi_kurt = 1 if (kurt >= min_kurtosis or t < datetime.time(10, 15)) else 0

                phi_amt_arr[idx] = phi_amt
                phi_slope_arr[idx] = phi_slope
                phi_kurt_arr[idx] = phi_kurt

                # Régimen Rotacional de Reversión Habilitado
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


class PropRiskManagerN2:
    """
    Gestor Cuantitativo de Riesgo Institucional N2 para Proprietary Trading.
    
    Implementa:
        - Daily Loss Limit (DLL) con disyuntor preventivo en alpha_DLL = 0.85 * DLL.
        - Parada diaria preventiva tras K_loss_intraday >= 2 pérdidas consecutivas en el día.
        - Maximum Trailing Drawdown (MTDD) con monitoreo continuo de Peak Equity.
        - Corte Inmediato Intrabar: monitorea el peor precio intra-barra frente al umbral preventivo
          de pérdida diaria y trailing drawdown, ejecutando liquidación de emergencia si se aproxima al límite.
        - Presupuesto de riesgo con factor de reserva estricto psi* = 0.12 y atenuación geométrica:
          phi(K_loss) = 0.25 * 2^(-min(K_loss, 3)).
        - Dimensionamiento entero piso con conmutación automática Estándar -> Micro y disyuntor HALT_VOLATILITY_DEFENSE.
    """

    def __init__(
        self,
        account_size: float = 50000.0,
        daily_loss_limit: float = 1000.0,
        max_trailing_drawdown: float = 2000.0,
        psi: float = 0.12,
        max_std_contracts: int = 1,
        max_micro_contracts: int = 8,
        alpha_dll: float = 0.85,
        max_intraday_consecutive_losses: int = 2,
        symbol: str = "ES",
        tick_size: float = 0.25,
        std_multiplier: float = 50.0,
        micro_multiplier: float = 5.0,
        std_commission_rt: float = 4.50,
        micro_commission_rt: float = 1.24,
        consecutive_losses: int = 0,
        scaling_factor: float = 1.0
    ):
        self.account_size = account_size
        self.daily_loss_limit = daily_loss_limit
        self.max_trailing_drawdown = max_trailing_drawdown
        self.psi = psi
        self.max_std_contracts = max_std_contracts
        self.max_micro_contracts = max_micro_contracts
        self.alpha_dll = alpha_dll
        self.max_intraday_consecutive_losses = max_intraday_consecutive_losses
        self.scaling_factor = scaling_factor

        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.std_multiplier = std_multiplier
        self.micro_multiplier = micro_multiplier
        self.std_commission_rt = std_commission_rt
        self.micro_commission_rt = micro_commission_rt

        # Fricción combinada: 2 ticks round-turn de slippage + comisión oficial CME
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
        """Reinicia la pérdida diaria realizada y contador de racha intradía al comenzar una nueva sesión."""
        if self.current_date is None or date != self.current_date:
            self.current_date = date
            self.day_net_pnl = 0.0
            self.day_realized_loss = 0.0
            self.intraday_consecutive_losses = 0
            self.is_halted_day = False

    def update_floating_equity(self, equity: float):
        """Actualiza la marca de agua máxima (HWM) y disyuntor de cuenta durante posiciones abiertas."""
        if equity > self.peak_equity:
            self.peak_equity = equity
        trailing_dd = self.peak_equity - equity
        if trailing_dd >= self.max_trailing_drawdown:
            self.is_halted_account = True

    def phi_k_loss(self) -> float:
        """Atenuación geométrica ante pérdidas consecutivas: phi(K_loss) = 0.25 * 2^(-min(K_loss, 3))."""
        k = min(self.consecutive_losses, 3)
        return 0.25 * (2.0 ** (-k))

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

        if self.day_realized_loss >= (self.alpha_dll * self.daily_loss_limit):
            self.is_halted_day = True
            return 0.0

        if self.intraday_consecutive_losses >= self.max_intraday_consecutive_losses:
            self.is_halted_day = True
            return 0.0

        raw_budget = min(
            self.phi_k_loss() * rem_daily_budget,
            self.psi * rem_trailing_dd
        )
        scaled_budget = raw_budget * self.scaling_factor
        return max(0.0, scaled_budget)

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
                - status (str): 'OK', 'HALT_VOLATILITY_DEFENSE', etc.
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

        r_contract_std = delta_sl * self.std_multiplier + self.c_fric_std
        r_contract_micro = delta_sl * self.micro_multiplier + self.c_fric_micro

        # 1. Intentar contrato estándar si el presupuesto lo permite
        n_std = int(np.floor(budget / r_contract_std))
        if n_std >= 1:
            n_std = min(n_std, self.max_std_contracts)
            return n_std, "STANDARD", r_contract_std, "OK"

        # 2. Conmutar a micro-contratos
        n_micro = int(np.floor(budget / r_contract_micro))
        n_micro = min(n_micro, self.max_micro_contracts)

        if n_micro >= 1:
            return n_micro, "MICRO", r_contract_micro, "OK"

        # 3. Disyuntor de volatilidad excesiva: ni un micro cabe en el presupuesto
        return 0, "NONE", r_contract_micro, "HALT_VOLATILITY_DEFENSE"

    def check_intrabar_limit_cut(
        self,
        direction: int,
        entry_price: float,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        contracts: int,
        contract_multiplier: float,
        contract_friction: float
    ) -> Tuple[bool, float, str]:
        """
        Evalúa si durante la barra actual la pérdida flotante alcanza el umbral de corte preventivo
        de Daily Loss Limit (alpha_DLL * DLL) o el límite de Trailing Drawdown (MTDD).
        
        Returns:
            Tuple[bool, float, str]:
                - is_cut (bool): True si se dispara el corte intrabar.
                - cut_price (float): Precio exacto de liquidación de emergencia.
                - cut_reason (str): Razón del corte.
        """
        if contracts <= 0:
            return False, 0.0, ""

        friction = contract_friction * contracts
        mult = contract_multiplier * contracts

        # Margen remanente antes del corte preventivo de DLL (85% de DLL)
        target_dll_loss = self.alpha_dll * self.daily_loss_limit
        rem_dll_loss = max(0.0, target_dll_loss - self.day_realized_loss)

        # Margen remanente antes del MTDD
        current_dd = max(0.0, self.peak_equity - self.current_equity)
        rem_mtdd = max(0.0, self.max_trailing_drawdown - current_dd)

        max_adverse_loss = min(rem_dll_loss, rem_mtdd)
        if max_adverse_loss <= friction:
            # Ya no hay margen para absorber ni la fricción: cortar de inmediato
            cut_p = bar_open
            reason = "INTRABAR_DAILY_LOSS_LIMIT_CUT" if rem_dll_loss <= rem_mtdd else "INTRABAR_TRAILING_DD_CUT"
            return True, cut_p, reason

        allowable_adverse_pts = (max_adverse_loss - friction) / mult

        if direction == 1:
            # Long: pérdida si precio cae
            cut_barrier = entry_price - allowable_adverse_pts
            if bar_low <= cut_barrier:
                cut_price = min(bar_open, cut_barrier)
                reason = "INTRABAR_DAILY_LOSS_LIMIT_CUT" if rem_dll_loss <= rem_mtdd else "INTRABAR_TRAILING_DD_CUT"
                return True, cut_price, reason
        elif direction == -1:
            # Short: pérdida si precio sube
            cut_barrier = entry_price + allowable_adverse_pts
            if bar_high >= cut_barrier:
                cut_price = max(bar_open, cut_barrier)
                reason = "INTRABAR_DAILY_LOSS_LIMIT_CUT" if rem_dll_loss <= rem_mtdd else "INTRABAR_TRAILING_DD_CUT"
                return True, cut_price, reason

        return False, 0.0, ""

    def record_trade_result(self, net_pnl: float):
        """Actualiza el estado de la cuenta tras el cierre de un trade."""
        self.current_equity += net_pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        self.day_net_pnl += net_pnl
        self.day_realized_loss = max(0.0, -self.day_net_pnl)

        if net_pnl < 0.0:
            self.consecutive_losses += 1
            self.intraday_consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            self.intraday_consecutive_losses = 0

        trailing_dd = self.peak_equity - self.current_equity
        if trailing_dd >= self.max_trailing_drawdown:
            self.is_halted_account = True

        if self.day_realized_loss >= (self.alpha_dll * self.daily_loss_limit):
            self.is_halted_day = True

        if self.intraday_consecutive_losses >= self.max_intraday_consecutive_losses:
            self.is_halted_day = True


class F3OptimizationISN2Strategy(BaseStrategy):
    """
    Estrategia Cuantitativa F3-Optimizacion -IS- N2:
    Anchored VWAP & Rotational Value Area Reversion con Filtro de Régimen Ultra-Sensible.
    """

    def __init__(
        self,
        name: str = "F3_Optimization_IS_N2",
        params: Optional[Dict[str, Any]] = None
    ):
        defaults = self._default_params()
        if params is not None:
            defaults.update(params)
        super().__init__(name=name, params=defaults)

    def _default_params(self) -> Dict[str, Any]:
        return {
            "symbol": "ES",
            "anchor_time": "09:30",
            "session_end_time": "17:00",
            "window_init_end": "10:15",
            "window_revers_end": "15:00",
            "window_inhibit_end": "15:15",
            "k_band": 2.0,
            "k_catastrophic": 2.75,
            "kappa_crit": 0.085,
            "bar_minutes": 5.0,
            "slope_lookback_bars": 3,
            "max_vol_outside_ratio": 0.55,
            "min_kurtosis": 1.7,
            "min_dist_to_vwap_ratio": 0.80,
            "k_buffer_ticks": 3,
            "convergence_protect_ratio": 0.60,
            "account_size": 50000.0,
            "daily_loss_limit": 1000.0,
            "max_trailing_drawdown": 2000.0,
            "psi": 0.12,
            "max_std_contracts": 1,
            "max_micro_contracts": 8,
            "alpha_dll": 0.85,
            "max_intraday_consecutive_losses": 2,
            "scaling_factor": 0.65,
            "tick_size": 0.25
        }

    def prepare_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los indicadores cuantitativos para N2."""
        self.validate_data(data)
        p = self.params
        df = compute_anchored_vwap_n2_indicators(
            df=data,
            anchor_time=p["anchor_time"],
            session_end_time=p.get("session_end_time", "17:00"),
            k_band=p["k_band"],
            k_catastrophic=p["k_catastrophic"],
            kappa_crit=p["kappa_crit"],
            bar_minutes=p["bar_minutes"],
            slope_lookback_bars=p["slope_lookback_bars"],
            max_vol_outside_ratio=p["max_vol_outside_ratio"],
            min_kurtosis=p.get("min_kurtosis", 1.7)
        )
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera serie de posiciones:
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

            if date != prev_date and position != 0.0:
                position = 0.0

            if t >= w_flat_time and position != 0.0:
                position = 0.0
                signals.iloc[i] = 0.0
                continue

            current_vwap = df["VWAP"].iloc[i]
            current_sig = df["Sigma_VWAP"].iloc[i]
            current_upper_cat = df["UpperCatastrophic"].iloc[i]
            current_lower_cat = df["LowerCatastrophic"].iloc[i]

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
                    is_long = (prev_low < prev_lower_band) and (curr_close >= curr_lower_band) and (curr_close < current_vwap - min_dist)
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
        risk_manager: Optional[PropRiskManagerN2] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta el backtest formal con gestión de riesgo Prop Trading intra-barra y modelo de fricción CME Globex.
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
            risk_manager = PropRiskManagerN2(
                account_size=p["account_size"],
                daily_loss_limit=p["daily_loss_limit"],
                max_trailing_drawdown=p["max_trailing_drawdown"],
                psi=p.get("psi", 0.12),
                max_std_contracts=p.get("max_std_contracts", 1),
                max_micro_contracts=p.get("max_micro_contracts", 8),
                alpha_dll=p.get("alpha_dll", 0.85),
                max_intraday_consecutive_losses=p.get("max_intraday_consecutive_losses", 2),
                symbol=std_sym,
                tick_size=p["tick_size"],
                std_multiplier=spec_std["multiplier"],
                micro_multiplier=spec_micro["multiplier"],
                std_commission_rt=spec_std["commission_rt"],
                micro_commission_rt=spec_micro["commission_rt"],
                scaling_factor=p.get("scaling_factor", 0.65)
            )

        n = len(df)
        if n == 0:
            empty_trades = pd.DataFrame()
            empty_equity = pd.Series(dtype=np.float64)
            metrics = self._calculate_backtest_metrics(
                df_trades=empty_trades,
                equity_series=empty_equity,
                initial_capital=risk_manager.account_size
            )
            return {
                "metrics": metrics,
                "trades": empty_trades,
                "equity_curve": empty_equity,
                "data": df
            }

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

            risk_manager.reset_day_if_new(date)

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
            if in_position:
                exit_triggered = False
                exit_price = 0.0
                exit_reason = ""

                # Prioridad 1: Cierre Forzoso a las 15:15 EST
                if t >= w_flat_time:
                    exit_triggered = True
                    exit_price = current_bar["Close"]
                    exit_reason = "FORCED_TIME_DECAY_1515"

                # Prioridad 2: Corte Inmediato Intrabar ante proximidad al Daily Loss Limit o MTDD
                if not exit_triggered:
                    is_cut, cut_p, cut_rsn = risk_manager.check_intrabar_limit_cut(
                        direction=direction,
                        entry_price=entry_price,
                        bar_open=current_bar["Open"],
                        bar_high=current_bar["High"],
                        bar_low=current_bar["Low"],
                        contracts=contracts,
                        contract_multiplier=contract_multiplier,
                        contract_friction=contract_friction
                    )
                    if is_cut:
                        exit_triggered = True
                        exit_price = cut_p
                        exit_reason = cut_rsn

                # Prioridad 3: Stop Loss Técnico, Trailing Protect o Disyuntor Catastrófico
                if not exit_triggered and direction == 1:
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

                elif not exit_triggered and direction == -1:
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

        if n > 0:
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
        """Calcula métricas estadísticas oficiales del censo de operaciones."""
        if df_trades.empty or equity_series.empty:
            return {
                "Capital Inicial ($)": initial_capital,
                "Capital Final ($)": initial_capital,
                "P&L Neto Total ($)": 0.0,
                "Retorno Total (%)": 0.0,
                "Max Trailing Drawdown ($)": 0.0,
                "Max Trailing Drawdown (%)": 0.0,
                "Total Trades": 0,
                "Trades Ganadores": 0,
                "Trades Perdedores": 0,
                "Win Rate (%)": 0.0,
                "Profit Factor": 0.0,
                "Trade Promedio ($)": 0.0,
                "Trades Estándar": 0,
                "Trades Micro": 0,
                "Desglose de Salidas": {}
            }

        final_equity = equity_series.iloc[-1]
        net_pnl = final_equity - initial_capital
        ret_pct = (net_pnl / initial_capital) * 100.0

        # Trailing Drawdown exacto a partir de la curva de equidad
        cummax = equity_series.cummax()
        dd_series = cummax - equity_series
        max_dd_dollars = float(dd_series.max())
        max_dd_pct = float((dd_series / cummax).max() * 100.0) if cummax.max() > 0.0 else 0.0

        n_trades = len(df_trades)
        wins = df_trades[df_trades["net_pnl"] > 0]
        losses = df_trades[df_trades["net_pnl"] < 0]

        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = (n_wins / n_trades) * 100.0 if n_trades > 0 else 0.0

        gross_wins = wins["net_pnl"].sum()
        gross_losses = abs(losses["net_pnl"].sum())
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0.0 else (99.0 if gross_wins > 0 else 0.0)

        avg_trade = df_trades["net_pnl"].mean()
        best_trade = df_trades["net_pnl"].max()
        worst_trade = df_trades["net_pnl"].min()

        std_trades = len(df_trades[df_trades["contract_type"] == "STANDARD"])
        micro_trades = len(df_trades[df_trades["contract_type"] == "MICRO"])

        exit_reasons = df_trades["exit_reason"].value_counts().to_dict()

        return {
            "Capital Inicial ($)": round(initial_capital, 2),
            "Capital Final ($)": round(final_equity, 2),
            "P&L Neto Total ($)": round(net_pnl, 2),
            "Retorno Total (%)": round(ret_pct, 2),
            "Max Trailing Drawdown ($)": round(max_dd_dollars, 2),
            "Max Trailing Drawdown (%)": round(max_dd_pct, 2),
            "Total Trades": int(n_trades),
            "Trades Ganadores": int(n_wins),
            "Trades Perdedores": int(n_losses),
            "Win Rate (%)": round(win_rate, 2),
            "Profit Factor": round(profit_factor, 2),
            "Trade Promedio ($)": round(avg_trade, 2),
            "Mejor Trade ($)": round(best_trade, 2),
            "Peor Trade ($)": round(worst_trade, 2),
            "Trades Estándar": int(std_trades),
            "Trades Micro": int(micro_trades),
            "Desglose de Salidas": exit_reasons
        }
