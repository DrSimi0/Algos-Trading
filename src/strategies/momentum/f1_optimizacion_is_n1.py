"""
src/strategies/momentum/f1_optimizacion_is_n1.py
Estrategia F1-Optimizacion -IS- N1: Market Intraday Momentum & MOC Seasonality Optimizado.

Fundamento Cuantitativo:
- Gao, Han, Li, & Zhou (2018). Market Intraday Momentum. Journal of Financial Economics.
- Bogousslavsky (2021). The Cross-Section of Intraday and Overnight Returns. JFE.
- Propuesta Analítica de Optimización: reports/optimizaciones_estrategias/F1_optimizacion_market_intraday_momentum.md

Mejoras Matemáticas Integradas:
1. Normalización del retorno de apertura [09:30, 10:00) EST contra la volatilidad realizada intradía (ATR y RV cuadrática).
2. Filtro bilateral de apertura:
   - Filtro de Puro Ruido: Descarta sesiones con señal direccional insuficiente respecto al ruido intradía (Z_norm < Z_noise_min).
   - Filtro de Agotamiento de Rango: Descarta sesiones donde el movimiento inicial ya consumió el rango esperado del día
     (Z_norm > Z_exhaust_max o Range_open / ATR_daily > max_range_fraction).
3. Escalado dinámico del tamaño de posición según el colchón remanente del Daily Loss Limit (DLL).
4. Filtro de coherencia tendencial de cierre con el VWAP institucional a las 15:30 EST.
5. Stop Loss adaptativo al ATR intradía y Take Profit asimétrico (1.6R).
6. Salida anticipada por decaimiento de momento (Momentum Decay Exit) a partir de las 15:45 EST.
7. Cierre forzoso determinista a las 15:58:00 EST (MOC).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from ...core.base_strategy import BaseStrategy
from ...core.futures_specs import get_spec


# ==============================================================================
# FUNCIONES MATEMÁTICAS ANALÍTICAS INDEPENDIENTES
# ==============================================================================

def calculate_intraday_realized_volatility(
    obs_bars: pd.DataFrame,
    p_1000: float
) -> Tuple[float, float, float]:
    """
    Calcula la volatilidad realizada intradía en la ventana de apertura [09:30, 10:00) EST.

    Retorna:
    - atr_intra: Rango verdadero medio (ATR) de las barras de 5 minutos en puntos.
    - rv_intra: Volatilidad realizada cuadrática: sqrt( sum( (ln(Close/Open))^2 ) ).
    - sigma_atr: ATR intradía relativo respecto al precio de las 10:00 (adimensional).
    """
    if len(obs_bars) == 0:
        return 0.0, 0.0, 0.0

    highs = obs_bars["High"].values
    lows = obs_bars["Low"].values
    closes = obs_bars["Close"].values
    opens = obs_bars["Open"].values

    m = len(obs_bars)
    tr_list = []
    log_rets = []

    for j in range(m):
        h = highs[j]
        l = lows[j]
        c = closes[j]
        o = opens[j]

        # Log return de la barra
        if o > 0 and c > 0:
            log_rets.append(np.log(c / o))

        # True Range
        if j == 0:
            tr = h - l
        else:
            prev_c = closes[j - 1]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        tr_list.append(tr)

    atr_intra = float(np.mean(tr_list)) if tr_list else 0.0
    rv_intra = float(np.sqrt(np.sum(np.square(log_rets)))) if log_rets else 0.0
    sigma_atr = (atr_intra / p_1000) if p_1000 > 0 else 0.0

    return atr_intra, rv_intra, sigma_atr


def calculate_opening_noise_and_exhaustion(
    r_first: float,
    obs_bars: pd.DataFrame,
    p_1000: float,
    daily_atr: float = 0.0,
    z_noise_min: float = 0.60,
    z_exhaust_max: float = 2.50,
    max_range_fraction: float = 0.65,
    theta_floor: float = 0.0010,
    c_atr: float = 0.60,
) -> Dict[str, Any]:
    """
    Normaliza el retorno de apertura contra la volatilidad realizada intradía y evalúa:
    1. Si el movimiento fue puro ruido (Z_norm < Z_noise_min o |r_first| < theta_t^*).
    2. Si ya agotó el rango del día (Z_norm > Z_exhaust_max o Range_open / ATR_daily > max_range_fraction).
    """
    atr_intra, rv_intra, sigma_atr = calculate_intraday_realized_volatility(obs_bars, p_1000)

    # Umbral adaptativo formal theta_t^* = max(theta_floor, c_atr * sigma_atr)
    theta_star = max(theta_floor, c_atr * sigma_atr)

    # Normalización del retorno de apertura frente a la volatilidad realizada
    denom = max(sigma_atr, 1e-7)
    z_norm = float(abs(r_first) / denom)

    # Rango de apertura en puntos
    range_open = float(obs_bars["High"].max() - obs_bars["Low"].min()) if len(obs_bars) > 0 else 0.0
    e_range = (range_open / daily_atr) if daily_atr > 0 else 0.0

    # Detección de Puro Ruido
    is_noise = bool(abs(r_first) < theta_star or z_norm < z_noise_min)

    # Detección de Agotamiento de Rango
    is_exhausted = bool(
        z_norm > z_exhaust_max or (daily_atr > 0 and e_range > max_range_fraction)
    )

    # Calificación
    if is_noise:
        filter_status = "FILTER_NOISE"
    elif is_exhausted:
        filter_status = "FILTER_RANGE_EXHAUSTED"
    else:
        filter_status = "VALID_MOMENTUM"

    return {
        "atr_intra": atr_intra,
        "rv_intra": rv_intra,
        "sigma_atr": sigma_atr,
        "theta_star": theta_star,
        "z_norm": z_norm,
        "range_open": range_open,
        "e_range": e_range,
        "is_noise": is_noise,
        "is_exhausted": is_exhausted,
        "filter_status": filter_status,
        "is_qualified": (filter_status == "VALID_MOMENTUM"),
    }


def calculate_dll_cushion_scaling(
    dll: float,
    l_day: float = 0.0,
    k_loss: int = 0,
    phi_base: float = 0.35,
    cushion_exponent: float = 1.0,
    mtdd: Optional[float] = None,
    dd_current: float = 0.0,
    psi: float = 0.20,
) -> Dict[str, Any]:
    """
    Escala el presupuesto de riesgo según el colchón restante del Daily Loss Limit (DLL).

    Fórmulas:
    C_DLL,t = max(0, DLL - L_day,t)
    cushion_ratio = C_DLL,t / DLL in [0, 1]
    phi(K_loss) = phi_base * 2^(-min(K_loss, 3))
    B_DLL,t = phi(K_loss) * C_DLL,t * (cushion_ratio)^cushion_exponent
    B_risk,t = min(B_DLL,t, psi * max(0, MTDD - DD_current))
    """
    if dll is None or dll <= 0:
        c_dll = float("inf")
        cushion_ratio = 1.0
        b_dll = float("inf")
    else:
        c_dll = max(0.0, dll - l_day)
        cushion_ratio = c_dll / dll
        phi_k = phi_base * (2.0 ** (-min(max(k_loss, 0), 3)))
        # Escalado por colchón restante con exponente de elasticidad
        scale_mult = (cushion_ratio ** max(cushion_exponent, 0.0))
        b_dll = phi_k * c_dll * scale_mult

    if mtdd is not None and mtdd > 0:
        c_mtdd = max(0.0, mtdd - dd_current)
        b_trailing = psi * c_mtdd
    else:
        c_mtdd = float("inf")
        b_trailing = float("inf")

    b_risk = min(b_dll, b_trailing)
    if np.isinf(b_risk):
        b_risk = 5000.0

    return {
        "c_dll": c_dll,
        "cushion_ratio": cushion_ratio,
        "c_mtdd": c_mtdd,
        "b_dll": b_dll,
        "b_trailing": b_trailing,
        "b_risk": float(max(0.0, b_risk)),
    }


def compute_vwap(df_session: pd.DataFrame) -> float:
    """
    Calcula el Precio Medio Ponderado por Volumen (VWAP) intradía acumulado.
    VWAP = sum( TypicalPrice * Volume ) / sum( Volume )
    """
    if len(df_session) == 0:
        return 0.0
    vol = df_session["Volume"].values
    total_vol = np.sum(vol)
    if total_vol <= 0:
        return float(df_session["Close"].iloc[-1])

    typ_price = (df_session["High"].values + df_session["Low"].values + df_session["Close"].values) / 3.0
    return float(np.sum(typ_price * vol) / total_vol)


# ==============================================================================
# CLASE PRINCIPAL: MarketIntradayMomentumOptN1
# ==============================================================================

class MarketIntradayMomentumOptN1(BaseStrategy):
    """
    Versión alterna y optimizada: F1-Optimizacion -IS- N1
    Market Intraday Momentum & MOC Seasonality.
    """

    def __init__(
        self,
        name: str = "F1-Optimizacion -IS- N1",
        params: Optional[Dict[str, Any]] = None,
    ):
        default_params: Dict[str, Any] = {
            "symbol": "ES",
            "std_symbol": "ES",
            "micro_symbol": "MES",
            "theta_floor": 0.0010,          # 0.10% umbral mínimo adaptativo (IS calibrado)
            "c_atr": 0.60,                  # Coeficiente de normalización por ATR intradía
            "z_noise_min": 0.60,            # Cota mínima para filtrar puro ruido (|r_first| / sigma_atr)
            "z_exhaust_max": 2.50,          # Cota máxima para filtrar agotamiento de rango
            "max_range_fraction": 0.65,     # Máxima fracción del ATR diario consumida por el rango matutino
            "k_vol": 20,                    # Lookback de volumen matutino
            "vol_ratio_threshold": 0.85,    # Umbral de volumen relativo suavizado
            "vwap_coherence": True,         # Filtro de coherencia tendencial con VWAP a las 15:30
            "vwap_min_ticks": 2,            # Separación mínima del VWAP en ticks
            "use_atr_sl": True,             # Stop Loss adaptativo al ATR intradía
            "k_sl": 1.25,                   # Multiplicador de Stop Loss sobre ATR intradía
            "lambda_sl": 0.0035,            # Fallback stop porcentual si no hay ATR
            "k_rr": 1.60,                   # Asimetría Recompensa/Riesgo (TP = 1.60 * SL)
            "enable_decay_exit": True,      # Salida anticipada por decaimiento de momento
            "decay_progress_threshold": 0.40,  # Progreso mínimo exigido hacia TP a las 15:45
            "psi": 0.20,                    # Fracción de margen MTDD por trade
            "phi_base": 0.35,               # Fracción base DLL por trade
            "cushion_exponent": 1.0,        # Exponente de elasticidad de escalado con colchón DLL
            "n_max_std": 2,                 # Máximo de contratos estándar
            "n_max_micro": 10,              # Máximo de contratos micro
            "obs_start": "09:30",           # Inicio ventana observación (ET)
            "obs_end": "10:00",             # Fin ventana observación (ET)
            "exec_start": "15:30",          # Inicio ejecución MOC (ET)
            "moc_exit": "15:58",            # Salida forzosa MOC (ET)
            "target_tz": "America/New_York",
            "source_tz": None,
        }
        if params:
            default_params.update(params)
        super().__init__(name=name, params=default_params)

    def ensure_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Garantiza localización en America/New_York (EST/EDT)."""
        target_tz = self.params["target_tz"]
        source_tz = self.params.get("source_tz")

        if df.index.tz is None:
            if source_tz:
                return df.tz_localize(source_tz).tz_convert(target_tz)
            else:
                return df.tz_localize(target_tz)
        else:
            return df.tz_convert(target_tz)

    @staticmethod
    def calculate_morning_return(p_open: float, p_close: float) -> float:
        """Retorno logarítmico matutino: r_first,t = ln( P_t(10:00) / P_t(09:30) )."""
        if p_open <= 0 or p_close <= 0:
            raise ValueError(f"Precios deben ser estrictamente positivos: {p_open}, {p_close}")
        return float(np.log(p_close / p_open))

    def calculate_relative_volume(
        self,
        v_current: float,
        v_history: List[float]
    ) -> Tuple[float, int]:
        """Ratio de volumen relativo institucional y condición booleana (umbral >= 0.85)."""
        k_vol = self.params["k_vol"]
        vol_threshold = self.params["vol_ratio_threshold"]

        if len(v_history) == 0:
            return 1.0, 1

        recent_history = v_history[-k_vol:]
        mean_hist = float(np.mean(recent_history))
        if mean_hist <= 0:
            return 1.0, 1

        v_ratio = float(v_current / mean_hist)
        phi_vol = 1 if v_ratio >= vol_threshold else 0
        return v_ratio, phi_vol

    def evaluate_opening_filter(
        self,
        r_first: float,
        obs_bars: pd.DataFrame,
        p_1000: float,
        daily_atr: float = 0.0
    ) -> Dict[str, Any]:
        """
        Aplica el filtro matemático bilateral de apertura normalizado por volatilidad realizada:
        - Filtra días donde el movimiento fue puro ruido (Z_norm < Z_noise_min).
        - Filtra días donde el movimiento ya agotó el rango del día.
        """
        return calculate_opening_noise_and_exhaustion(
            r_first=r_first,
            obs_bars=obs_bars,
            p_1000=p_1000,
            daily_atr=daily_atr,
            z_noise_min=self.params["z_noise_min"],
            z_exhaust_max=self.params["z_exhaust_max"],
            max_range_fraction=self.params["max_range_fraction"],
            theta_floor=self.params["theta_floor"],
            c_atr=self.params["c_atr"],
        )

    def evaluate_vwap_coherence(
        self,
        r_first: float,
        p_1530: float,
        session_to_1530: pd.DataFrame,
        tick_size: float = 0.25
    ) -> Tuple[bool, float]:
        """
        Filtro de Coherencia Tendencial de Cierre con el VWAP Institucional:
        sign(r_first) == sign(P_1530 - VWAP_1530) and |P_1530 - VWAP_1530| >= 2 * tick.
        """
        if not self.params.get("vwap_coherence", True):
            return True, 0.0

        vwap_val = compute_vwap(session_to_1530)
        delta_vwap = p_1530 - vwap_val
        min_distance = self.params["vwap_min_ticks"] * tick_size

        if abs(delta_vwap) < min_distance:
            return False, delta_vwap

        coherence = bool(
            (r_first > 0 and delta_vwap > 0) or
            (r_first < 0 and delta_vwap < 0)
        )
        return coherence, delta_vwap

    def evaluate_signal_opt(
        self,
        r_first: float,
        phi_vol: int,
        opening_analysis: Dict[str, Any],
        vwap_coherent: bool = True
    ) -> int:
        """
        Operador analítico integrado de señal para F1-Optimizacion -IS- N1:
        Retorna:
         +1: Largo calificado (impulso alcista válido, volumen, coherencia VWAP, ni ruido ni agotado).
         -1: Corto calificado (impulso bajista válido, volumen, coherencia VWAP, ni ruido ni agotado).
          0: Neutral (descartado por ruido, agotamiento, volumen o divergencia VWAP).
        """
        if not opening_analysis["is_qualified"]:
            return 0

        if phi_vol != 1:
            return 0

        if not vwap_coherent:
            return 0

        if r_first > opening_analysis["theta_star"]:
            return 1
        elif r_first < -opening_analysis["theta_star"]:
            return -1

        return 0

    def calculate_exit_levels_opt(
        self,
        p_entry: float,
        direction: int,
        atr_intra: float,
        tick_size: float = 0.25
    ) -> Tuple[float, float, float]:
        """
        Calcula niveles de Stop Loss y Take Profit adaptativos:
        Delta_SL = max(4 * delta_tick, k_SL * ATR_intra)
        P_SL = P_entry - direction * Delta_SL
        P_TP = P_entry + direction * (k_RR * Delta_SL)
        """
        if direction not in (-1, 1):
            raise ValueError(f"Dirección debe ser +1 o -1, recibido: {direction}")

        k_rr = self.params["k_rr"]

        if self.params.get("use_atr_sl", True) and atr_intra > 0:
            k_sl = self.params["k_sl"]
            delta_sl = max(4.0 * tick_size, k_sl * atr_intra)
        else:
            delta_sl = p_entry * self.params["lambda_sl"]

        p_sl = p_entry - direction * delta_sl
        p_tp = p_entry + direction * (k_rr * delta_sl)
        return float(p_sl), float(p_tp), float(delta_sl)

    def calculate_risk_budget_scaled_by_dll(
        self,
        dll: float,
        l_day: float = 0.0,
        k_loss: int = 0,
        mtdd: Optional[float] = None,
        dd_current: float = 0.0
    ) -> Dict[str, Any]:
        """Escala el presupuesto de riesgo directamente con el colchón restante del DLL."""
        return calculate_dll_cushion_scaling(
            dll=dll,
            l_day=l_day,
            k_loss=k_loss,
            phi_base=self.params["phi_base"],
            cushion_exponent=self.params["cushion_exponent"],
            mtdd=mtdd,
            dd_current=dd_current,
            psi=self.params["psi"]
        )

    def calculate_contract_risk(
        self,
        delta_sl_pts: float,
        symbol: str
    ) -> float:
        """Riesgo monetario individual por contrato considerando deslizamiento y comisiones."""
        spec = get_spec(symbol)
        m = spec["multiplier"]
        delta_tick = spec["tick"]
        commission_rt = spec["commission_rt"]

        c_fric = 2.0 * delta_tick * m + commission_rt
        r_contract = delta_sl_pts * m + c_fric
        return float(r_contract)

    def calculate_position_sizing_scaled(
        self,
        b_risk: float,
        delta_sl_pts: float,
        cushion_ratio: float = 1.0
    ) -> Dict[str, Any]:
        """
        Dimensionamiento de contratos con conmutación automática Estándar/Micro y escalado por colchón.
        Si el colchón restante de DLL o MTDD es insuficiente para asumir el riesgo mínimo de un micro,
        activa el disyuntor de preservación de capital.
        """
        std_symbol = self.params["std_symbol"]
        micro_symbol = self.params["micro_symbol"]
        n_max_std = self.params["n_max_std"]
        n_max_micro = self.params["n_max_micro"]

        if b_risk <= 0 or cushion_ratio <= 0.0:
            return {
                "contract_type": "NONE",
                "symbol": micro_symbol,
                "contracts": 0,
                "risk_per_contract": 0.0,
                "cushion_ratio": cushion_ratio,
                "status": "HALT_NO_CUSHION"
            }

        r_std = self.calculate_contract_risk(delta_sl_pts, std_symbol)
        n_std = int(np.floor(b_risk / r_std))

        if n_std >= 1:
            allocated_std = min(n_std, n_max_std)
            return {
                "contract_type": "STD",
                "symbol": std_symbol,
                "contracts": allocated_std,
                "risk_per_contract": r_std,
                "cushion_ratio": cushion_ratio,
                "status": "OK"
            }

        # Conmutación a contratos Micro
        r_micro = self.calculate_contract_risk(delta_sl_pts, micro_symbol)
        if r_micro > b_risk:
            return {
                "contract_type": "NONE",
                "symbol": micro_symbol,
                "contracts": 0,
                "risk_per_contract": r_micro,
                "cushion_ratio": cushion_ratio,
                "status": "HALT_INSUFFICIENT_CUSHION"
            }

        n_micro = min(int(np.floor(b_risk / r_micro)), n_max_micro)
        return {
            "contract_type": "MICRO",
            "symbol": micro_symbol,
            "contracts": n_micro,
            "risk_per_contract": r_micro,
            "cushion_ratio": cushion_ratio,
            "status": "OK" if n_micro > 0 else "HALT_INSUFFICIENT_CUSHION"
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Interfaz canónica vectorial BaseStrategy.
        Genera la serie temporal de señales (+1, -1, 0) para el DataFrame histórico.
        """
        self.validate_data(data)
        df = self.ensure_timezone(data.copy())

        sig_array = np.zeros(len(df), dtype=np.float64)
        daily_groups = df.groupby(df.index.normalize())

        morning_vols: List[float] = []
        daily_ranges: List[float] = []
        ts_to_idx = {ts: idx for idx, ts in enumerate(df.index)}

        spec = get_spec(self.params["symbol"])
        tick_size = spec["tick"]

        for day_dt, day_df in daily_groups:
            # Ventana de Observación Matutina [09:30, 10:00) EST
            obs_bars = day_df.between_time(
                self.params["obs_start"],
                self.params["obs_end"],
                inclusive="left"
            )
            if len(obs_bars) < 2:
                if len(day_df) > 0:
                    daily_ranges.append(float(day_df["High"].max() - day_df["Low"].min()))
                continue

            p_930 = float(obs_bars["Open"].iloc[0])
            p_1000 = float(obs_bars["Close"].iloc[-1])
            v_current = float(obs_bars["Volume"].sum())

            r_first = self.calculate_morning_return(p_930, p_1000)
            v_ratio, phi_vol = self.calculate_relative_volume(v_current, morning_vols)
            morning_vols.append(v_current)

            daily_atr = float(np.mean(daily_ranges[-20:])) if daily_ranges else 0.0
            daily_ranges.append(float(day_df["High"].max() - day_df["Low"].min()))

            # Evaluación de Filtro Bilateral de Apertura (Puro Ruido / Rango Agotado)
            opening_analysis = self.evaluate_opening_filter(
                r_first=r_first,
                obs_bars=obs_bars,
                p_1000=p_1000,
                daily_atr=daily_atr
            )

            # Ventana acumulada hasta las 15:30 EST
            session_to_1530 = day_df.between_time("09:30", "15:30", inclusive="both")
            exec_bars = day_df.between_time(
                self.params["exec_start"],
                self.params["moc_exit"],
                inclusive="left"
            )
            if len(exec_bars) == 0:
                continue

            p_1530 = float(exec_bars["Open"].iloc[0])
            vwap_coherent, _ = self.evaluate_vwap_coherence(
                r_first=r_first,
                p_1530=p_1530,
                session_to_1530=session_to_1530,
                tick_size=tick_size
            )

            sig = self.evaluate_signal_opt(
                r_first=r_first,
                phi_vol=phi_vol,
                opening_analysis=opening_analysis,
                vwap_coherent=vwap_coherent
            )

            if sig == 0:
                continue

            p_entry = p_1530
            p_sl, p_tp, delta_sl = self.calculate_exit_levels_opt(
                p_entry=p_entry,
                direction=sig,
                atr_intra=opening_analysis["atr_intra"],
                tick_size=tick_size
            )

            for ts, row in exec_bars.iterrows():
                high = float(row["High"])
                low = float(row["Low"])
                arr_i = ts_to_idx.get(ts)

                if sig == 1:
                    if low <= p_sl:
                        break
                    if arr_i is not None:
                        sig_array[arr_i] = 1.0
                    if high >= p_tp:
                        break
                elif sig == -1:
                    if high >= p_sl:
                        break
                    if arr_i is not None:
                        sig_array[arr_i] = -1.0
                    if low <= p_tp:
                        break

        return pd.Series(sig_array, index=data.index, dtype=np.float64)


# ==============================================================================
# MOTOR DE BACKTEST DEDICADO: IntradayMomentumOptN1Backtester
# ==============================================================================

class IntradayMomentumOptN1Backtester:
    """
    Simulador cuantitativo de alta fidelidad para F1-Optimizacion -IS- N1.
    Soporta:
    - Normalización de retorno de apertura contra volatilidad realizada intradía.
    - Filtro de puro ruido y filtro de rango agotado.
    - Escalado de tamaño de posición según colchón restante de Daily Loss Limit (DLL).
    - Filtro de coherencia VWAP vespertino.
    - Salida anticipada por decaimiento de momento (Momentum Decay Exit).
    - Verificación rigurosa de límites de fondeo (DLL, MTDD) y liquidación forzosa MOC 15:58.
    """

    def __init__(
        self,
        strategy: Optional[MarketIntradayMomentumOptN1] = None,
        account_size: float = 50000.0,
        daily_loss_limit: float = 1000.0,
        max_trailing_loss: float = 2000.0,
        trailing_mode: str = "eod",
        lock_floor_at_start: bool = True,
        profit_target: float = 3000.0,
        min_days: int = 2,
    ):
        self.strategy = strategy or MarketIntradayMomentumOptN1()
        self.account_size = account_size
        self.daily_loss_limit = daily_loss_limit
        self.max_trailing_loss = max_trailing_loss
        self.trailing_mode = trailing_mode
        self.lock_floor_at_start = lock_floor_at_start
        self.profit_target = profit_target
        self.min_days = min_days

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Ejecuta la simulación sesión a sesión."""
        self.strategy.validate_data(data)
        df = self.strategy.ensure_timezone(data.copy())

        balance = self.account_size
        peak = balance
        floor = self._update_floor(peak)

        k_loss = 0
        trades: List[Dict[str, Any]] = []
        daily_logs: List[Dict[str, Any]] = []
        skipped_sessions: List[Dict[str, Any]] = []

        morning_vols: List[float] = []
        daily_ranges: List[float] = []

        failed = False
        fail_reason: Optional[str] = None
        fail_date: Optional[Any] = None
        passed = False
        pass_date: Optional[Any] = None

        spec = get_spec(self.strategy.params["symbol"])
        tick_size = spec["tick"]

        grouped = df.groupby(df.index.normalize())

        for day_dt, day_df in grouped:
            if failed or passed:
                break

            day_start_balance = balance
            l_day = 0.0

            # 1. Ventana de observación matutina [09:30, 10:00) EST
            obs_bars = day_df.between_time(
                self.strategy.params["obs_start"],
                self.strategy.params["obs_end"],
                inclusive="left"
            )

            if len(obs_bars) < 2:
                if len(day_df) > 0:
                    daily_ranges.append(float(day_df["High"].max() - day_df["Low"].min()))
                continue

            p_930 = float(obs_bars["Open"].iloc[0])
            p_1000 = float(obs_bars["Close"].iloc[-1])
            v_current = float(obs_bars["Volume"].sum())

            r_first = self.strategy.calculate_morning_return(p_930, p_1000)
            v_ratio, phi_vol = self.strategy.calculate_relative_volume(v_current, morning_vols)
            morning_vols.append(v_current)

            daily_atr = float(np.mean(daily_ranges[-20:])) if daily_ranges else 0.0
            daily_ranges.append(float(day_df["High"].max() - day_df["Low"].min()))

            # 2. Filtro bilateral normalizado por volatilidad realizada
            opening_analysis = self.strategy.evaluate_opening_filter(
                r_first=r_first,
                obs_bars=obs_bars,
                p_1000=p_1000,
                daily_atr=daily_atr
            )

            session_to_1530 = day_df.between_time("09:30", "15:30", inclusive="both")
            exec_bars = day_df.between_time(
                self.strategy.params["exec_start"],
                self.strategy.params["moc_exit"],
                inclusive="left"
            )

            if len(exec_bars) == 0:
                continue

            p_entry = float(exec_bars["Open"].iloc[0])
            vwap_coherent, delta_vwap = self.strategy.evaluate_vwap_coherence(
                r_first=r_first,
                p_1530=p_entry,
                session_to_1530=session_to_1530,
                tick_size=tick_size
            )

            # Señal integrada optimizada
            sig = self.strategy.evaluate_signal_opt(
                r_first=r_first,
                phi_vol=phi_vol,
                opening_analysis=opening_analysis,
                vwap_coherent=vwap_coherent
            )

            if sig == 0:
                skipped_sessions.append({
                    "date": day_dt.date() if hasattr(day_dt, "date") else day_dt,
                    "r_first": r_first,
                    "filter_status": opening_analysis["filter_status"],
                    "z_norm": opening_analysis["z_norm"],
                    "e_range": opening_analysis["e_range"],
                    "phi_vol": phi_vol,
                    "vwap_coherent": vwap_coherent,
                })
                # Actualizar log del día sin trade
                daily_logs.append({
                    "date": day_dt,
                    "day_pnl": 0.0,
                    "ending_balance": round(balance, 2),
                    "peak": round(peak, 2),
                    "floor": round(floor, 2),
                })
                continue

            # 3. Niveles de salida adaptativos
            p_sl, p_tp, delta_sl_pts = self.strategy.calculate_exit_levels_opt(
                p_entry=p_entry,
                direction=sig,
                atr_intra=opening_analysis["atr_intra"],
                tick_size=tick_size
            )

            # 4. Dimensionamiento con escalado estricto por colchón restante de DLL
            current_dd = max(0.0, peak - balance)
            cushion_info = self.strategy.calculate_risk_budget_scaled_by_dll(
                dll=self.daily_loss_limit,
                l_day=l_day,
                k_loss=k_loss,
                mtdd=self.max_trailing_loss,
                dd_current=current_dd
            )

            sizing = self.strategy.calculate_position_sizing_scaled(
                b_risk=cushion_info["b_risk"],
                delta_sl_pts=delta_sl_pts,
                cushion_ratio=cushion_info["cushion_ratio"]
            )

            if sizing["contracts"] > 0:
                trade_spec = get_spec(sizing["symbol"])
                m = trade_spec["multiplier"]
                delta_tick = trade_spec["tick"]
                commission_rt = trade_spec["commission_rt"]

                exit_time = exec_bars.index[-1]
                exit_price = float(exec_bars["Close"].iloc[-1])
                exit_reason = "MOC"

                # 5. Simulación barra a barra en la ventana vespertina
                exec_indices = list(exec_bars.index)
                for b_idx, (ts, row) in enumerate(exec_bars.iterrows()):
                    high = float(row["High"])
                    low = float(row["Low"])
                    curr_open = float(row["Open"])
                    curr_close = float(row["Close"])

                    # Verificación de Stop Loss
                    if sig == 1 and low <= p_sl:
                        exit_time = ts
                        exit_price = min(curr_open, p_sl) if curr_open <= p_sl else p_sl
                        exit_reason = "SL"
                        break
                    elif sig == -1 and high >= p_sl:
                        exit_time = ts
                        exit_price = max(curr_open, p_sl) if curr_open >= p_sl else p_sl
                        exit_reason = "SL"
                        break

                    # Verificación de Take Profit
                    if sig == 1 and high >= p_tp:
                        exit_time = ts
                        exit_price = max(curr_open, p_tp) if curr_open >= p_tp else p_tp
                        exit_reason = "TP"
                        break
                    elif sig == -1 and low <= p_tp:
                        exit_time = ts
                        exit_price = min(curr_open, p_tp) if curr_open <= p_tp else p_tp
                        exit_reason = "TP"
                        break

                    # Regla de Salida Anticipada por Decaimiento de Momento (15:45 EST)
                    if self.strategy.params.get("enable_decay_exit", True):
                        bar_time_str = ts.strftime("%H:%M")
                        if bar_time_str >= "15:45":
                            pts_progress = (curr_close - p_entry) if sig == 1 else (p_entry - curr_close)
                            tp_dist = (p_tp - p_entry) if sig == 1 else (p_entry - p_tp)
                            min_progress = self.strategy.params["decay_progress_threshold"] * tp_dist

                            # Velocidad instantánea en los últimos compases
                            if b_idx >= 1:
                                prev_close = float(exec_bars["Close"].iloc[b_idx - 1])
                                velocity = (curr_close - prev_close) if sig == 1 else (prev_close - curr_close)
                            else:
                                velocity = 0.0

                            if pts_progress < min_progress and velocity <= 0.0:
                                exit_time = ts
                                exit_price = curr_close
                                exit_reason = "DECAY"
                                break

                # Cálculo de PnL y Fricciones exactas
                pts_diff = (exit_price - p_entry) if sig == 1 else (p_entry - exit_price)
                gross_pnl = pts_diff * m * sizing["contracts"]
                # 2 ticks de deslizamiento total (1 tick entrada + 1 tick salida)
                slippage = 2.0 * delta_tick * m * sizing["contracts"]
                commission = commission_rt * sizing["contracts"]
                friction = slippage + commission
                net_pnl = gross_pnl - friction

                balance += net_pnl
                if net_pnl < 0:
                    l_day += abs(net_pnl)
                    k_loss += 1
                else:
                    k_loss = 0

                if self.trailing_mode == "intraday" and balance > peak:
                    peak = balance
                    floor = self._update_floor(peak)

                trades.append({
                    "date": day_dt.date() if hasattr(day_dt, "date") else day_dt,
                    "signal": "LONG" if sig == 1 else "SHORT",
                    "symbol": sizing["symbol"],
                    "contract_type": sizing["contract_type"],
                    "contracts": sizing["contracts"],
                    "entry_time": exec_bars.index[0],
                    "entry_price": p_entry,
                    "exit_time": exit_time,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "gross_pnl": round(gross_pnl, 2),
                    "friction": round(friction, 2),
                    "net_pnl": round(net_pnl, 2),
                    "balance": round(balance, 2),
                    "k_loss": k_loss,
                    "cushion_ratio": round(cushion_info["cushion_ratio"], 3),
                    "z_norm": round(opening_analysis["z_norm"], 3),
                    "e_range": round(opening_analysis["e_range"], 3),
                    "delta_vwap": round(delta_vwap, 2),
                })

                # Verificación de límites de Prop Trading
                if balance <= floor:
                    failed = True
                    fail_reason = "Max Trailing Drawdown Breached"
                    fail_date = day_dt
                elif self.daily_loss_limit is not None and l_day >= self.daily_loss_limit:
                    failed = True
                    fail_reason = "Daily Loss Limit Breached"
                    fail_date = day_dt

            # Cierre EOD
            if self.trailing_mode == "eod" and balance > peak:
                peak = balance
                floor = self._update_floor(peak)

            day_pnl = balance - day_start_balance
            daily_logs.append({
                "date": day_dt,
                "day_pnl": round(day_pnl, 2),
                "ending_balance": round(balance, 2),
                "peak": round(peak, 2),
                "floor": round(floor, 2),
            })

            if (balance - self.account_size) >= self.profit_target and len(daily_logs) >= self.min_days:
                passed = True
                pass_date = day_dt

        trades_df = pd.DataFrame(trades)
        metrics = self._calculate_performance_metrics(trades_df, balance)

        prop_evaluation = {
            "status": "PASS" if passed else ("FAIL" if failed else "INCOMPLETE"),
            "reason": fail_reason if failed else ("Profit Target Achieved" if passed else "Ongoing"),
            "date": pass_date if passed else fail_date,
            "final_balance": round(balance, 2),
            "net_profit": round(balance - self.account_size, 2),
            "total_days_evaluated": len(daily_logs),
            "total_skipped_sessions": len(skipped_sessions),
        }

        return {
            "metrics": metrics,
            "prop_evaluation": prop_evaluation,
            "trades": trades_df,
            "daily_logs": pd.DataFrame(daily_logs),
            "skipped_sessions": pd.DataFrame(skipped_sessions),
        }

    def _update_floor(self, peak: float) -> float:
        fl = peak - self.max_trailing_loss
        if self.lock_floor_at_start:
            fl = min(fl, self.account_size)
        return fl

    def _calculate_performance_metrics(
        self,
        trades_df: pd.DataFrame,
        final_balance: float
    ) -> Dict[str, Any]:
        if trades_df.empty:
            return {
                "Total Trades": 0,
                "Win Rate (%)": 0.0,
                "Profit Factor": 0.0,
                "Net Profit ($)": 0.0,
                "Total Return (%)": 0.0,
                "Max Drawdown ($)": 0.0,
                "Sharpe Ratio": 0.0,
            }

        pnl = trades_df["net_pnl"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        win_rate = (len(wins) / len(pnl)) * 100.0 if len(pnl) > 0 else 0.0
        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        equity_curve = pd.Series([self.account_size] + trades_df["balance"].tolist())
        peak = equity_curve.cummax()
        dd = peak - equity_curve
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0

        ret_series = pnl / self.account_size
        sharpe = (ret_series.mean() / ret_series.std(ddof=1) * np.sqrt(252)) if len(pnl) >= 2 and ret_series.std(ddof=1) > 0 else 0.0

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
        }


# Alias formales
F1OptimizacionISN1 = MarketIntradayMomentumOptN1
F1OptimizacionISN1Backtester = IntradayMomentumOptN1Backtester
