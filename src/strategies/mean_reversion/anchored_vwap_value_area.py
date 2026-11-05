"""
src/strategies/mean_reversion/anchored_vwap_value_area.py

Implementación formal de la Estrategia F3:
Anchored VWAP & Rotational Value Area Mean Reversion para Futuros CME (ES, MES, NQ, MNQ).

Especificación matemática de referencia:
    reports/estrategias_matematicas/F3_anchored_vwap_value_area.md

Fundamento Teórico:
    - Madhavan, A. (2002): VWAP Strategies.
    - Gatheral, J. (2010): No-Dynamic-Arbitrage and Market Impact.
    - Steidlmayer, J. P., & Koy, K. (1986): Auction Market Theory.

Reglas Cuantitativas y de Prop Trading:
    1. Anclaje estricto de VWAP a la apertura regular de contado (tau_0 = 09:30:00 EST).
    2. Varianza y Desviación Estándar Ponderada por Volumen acumuladas desde tau_0.
    3. Bandas estocásticas de subasta (+-2.0 sigma para ES/MES, +-2.5 sigma para NQ/MNQ, +-3.0 sigma disyuntor).
    4. Filtro de Régimen No Tendencial: ADX(14) < 25.0 y |kappa_slope| <= delta_max_slope.
    5. Ventanas horarias:
       - W_init [09:30, 10:30): acumulación inicial (sin operaciones).
       - W_revers [10:30, 15:00): ventana operativa de reversión (entradas permitidas).
       - W_inhibit [15:00, 15:15): inhibición de nuevas entradas ante MOC.
       - W_flat [15:15, 17:00]: liquidación forzosa (inmunidad overnight CME).
    6. Gatillo de entrada de dos etapas:
       - Long: Low(t-1) < LowerBand(t-1) y Close(t) >= LowerBand(t) con filtro rotacional activo.
       - Short: High(t-1) > UpperBand(t-1) y Close(t) <= UpperBand(t) con filtro rotacional activo.
    7. Salidas:
       - Take Profit Dinámico: P_TP(tau) = VWAP(tau).
       - Stop Loss Técnico Estructural: min/max(t-1, t) +- 4 ticks.
       - Disyuntor Catastrófico 3 sigma: salida inmediata si el precio cruza +-3.0 sigma.
       - Cierre forzoso intradía a las 15:15 EST.
    8. Gestión de Riesgo Prop Trading:
       - Presupuesto de riesgo por operación: B_F3 = min(phi(K_loss) * max(0, DLL - L_day), psi * max(0, MTDD - DD)).
       - Dimensionamiento automático con conmutación Estándar / Micro y disyuntor de volatilidad.
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


def compute_anchored_vwap_bands(
    df: pd.DataFrame,
    anchor_time: str = "09:30",
    session_end_time: str = "17:00",
    k_band: float = 2.0,
    k_catastrophic: float = 3.0,
    delta_max_slope: float = 0.15,
    adx_period: int = 14,
    bar_minutes: float = 5.0
) -> pd.DataFrame:
    """
    Calcula el Anchored VWAP, la desviación estándar ponderada por volumen,
    las bandas estocásticas, la pendiente del VWAP (kappa_slope) y el filtro
    de régimen no tendencial (R_rot).
    
    Alinea el ancla temporal diario a `anchor_time` (por defecto 09:30:00 EST)
    y finaliza la sesión regular en `session_end_time` (por defecto 17:00:00 EST).
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

    # Calcular ADX global sobre la serie continua
    res["ADX"] = calculate_adx(res, period=adx_period)

    # Identificar sesiones por fecha
    dates = res.index.date
    unique_dates = np.unique(dates)

    vwap_arr = np.full(len(res), np.nan, dtype=np.float64)
    sigma_arr = np.full(len(res), np.nan, dtype=np.float64)
    upper_band_arr = np.full(len(res), np.nan, dtype=np.float64)
    lower_band_arr = np.full(len(res), np.nan, dtype=np.float64)
    upper_cat_arr = np.full(len(res), np.nan, dtype=np.float64)
    lower_cat_arr = np.full(len(res), np.nan, dtype=np.float64)
    slope_arr = np.zeros(len(res), dtype=np.float64)
    is_anchored_arr = np.zeros(len(res), dtype=bool)

    p_typ_vals = res["P_typ"].values
    vol_vals = res["Volume"].values
    times = [t.time() for t in res.index]

    idx_start = 0
    for d in unique_dates:
        mask = (dates == d)
        day_indices = np.where(mask)[0]
        if len(day_indices) == 0:
            continue

        cum_v = 0.0
        cum_pv = 0.0
        cum_pv2 = 0.0
        anchor_started = False
        anchor_local_indices = []

        for idx in day_indices:
            t = times[idx]
            # Determinar si estamos dentro de la sesión regular anclada [09:30, 17:00]
            if anchor_tod <= t <= session_end_tod:
                anchor_started = True
            elif t > session_end_tod:
                anchor_started = False

            if anchor_started and (anchor_tod <= t <= session_end_tod):
                is_anchored_arr[idx] = True
                anchor_local_indices.append(idx)
                v = vol_vals[idx]
                p = p_typ_vals[idx]
                cum_v += v
                cum_pv += p * v
                cum_pv2 += (p ** 2) * v

                if cum_v > 0.0:
                    vw = cum_pv / cum_v
                    vwap_arr[idx] = vw

                    # Varianza ponderada por volumen: sum(V * (P - VWAP)^2) / sum(V)
                    # Algebraicamente: (sum(V * P^2) / cum_v) - VWAP^2
                    var = max(0.0, (cum_pv2 / cum_v) - (vw ** 2))
                    sig = np.sqrt(var)
                    sigma_arr[idx] = sig

                    upper_band_arr[idx] = vw + k_band * sig
                    lower_band_arr[idx] = vw - k_band * sig
                    upper_cat_arr[idx] = vw + k_catastrophic * sig
                    lower_cat_arr[idx] = vw - k_catastrophic * sig

                    # Pendiente kappa_slope, t = (VWAP_t - VWAP_t-3) / (3 * Delta_t)
                    # Donde Delta_t = bar_minutes (5 min), 3 * Delta_t = 15 min
                    if len(anchor_local_indices) >= 4:
                        idx_lag3 = anchor_local_indices[-4]
                        vw_lag3 = vwap_arr[idx_lag3]
                        if not np.isnan(vw_lag3):
                            slope_arr[idx] = (vw - vw_lag3) / (3.0 * bar_minutes)
                        else:
                            slope_arr[idx] = 0.0
                    else:
                        slope_arr[idx] = 0.0
                else:
                    vwap_arr[idx] = p
                    sigma_arr[idx] = 0.0
                    upper_band_arr[idx] = p
                    lower_band_arr[idx] = p
                    upper_cat_arr[idx] = p
                    lower_cat_arr[idx] = p
                    slope_arr[idx] = 0.0

    res["VWAP"] = vwap_arr
    res["Sigma_VWAP"] = sigma_arr
    res["UpperBand"] = upper_band_arr
    res["LowerBand"] = lower_band_arr
    res["UpperCatastrophic"] = upper_cat_arr
    res["LowerCatastrophic"] = lower_cat_arr
    res["VWAP_Slope"] = slope_arr
    res["IsAnchored"] = is_anchored_arr

    # Filtro de Régimen No Tendencial R_rot, t = I(ADX < 25.0) * I(|kappa_slope| <= delta_max_slope)
    res["Regime_Rotational"] = (
        (res["ADX"] < 25.0) &
        (res["VWAP_Slope"].abs() <= delta_max_slope)
    ).astype(int)

    return res


class PropRiskManager:
    """
    Gestor Cuantitativo de Riesgo y Dimensionamiento de Posición para Prop Trading.
    
    Implementa:
        - Daily Loss Limit (DLL) y control de pérdida diaria acumulada L_day, t.
        - Maximum Trailing Drawdown (MTDD) y marca de agua máxima (HWM).
        - Amortiguador por rachas de pérdidas consecutivas: phi(K_loss) = 0.30 * 2^(-min(K_loss, 3)).
        - Presupuesto de riesgo por operación:
          B_F3, t = min(phi(K_loss) * max(0, DLL - L_day, t), psi * max(0, MTDD - DD_t)) con psi = 0.20.
        - Riesgo monetario por contrato con fricciones CME (deslizamiento de 2 ticks + comisión):
          R_contract = Delta_SL * M + C_fric.
        - Conmutación automática Estándar (N_std) -> Micro (N_micro <= 15).
        - Disyuntor de volatilidad excesiva: si R_contract(M_micro) > B_F3 => N = 0 (HALT_VOLATILITY_DEFENSE).
    """

    def __init__(
        self,
        account_size: float = 50000.0,
        daily_loss_limit: float = 1000.0,
        max_trailing_drawdown: float = 2000.0,
        psi: float = 0.20,
        max_micro_contracts: int = 15,
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
        self.max_micro_contracts = max_micro_contracts

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
        self.is_halted_day = False
        self.is_halted_account = False

    def reset_day_if_new(self, date: datetime.date):
        """Reinicia la pérdida realizada diaria al iniciar una nueva fecha."""
        if self.current_date is None or date != self.current_date:
            self.current_date = date
            self.day_net_pnl = 0.0
            self.day_realized_loss = 0.0
            self.is_halted_day = False

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

        if rem_daily_budget <= 0.0:
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
            return 0, "NONE", 0.0, "DAILY_LOSS_LIMIT_REACHED"

        delta_sl = abs(entry_price - sl_price)
        if delta_sl <= 0.0:
            return 0, "NONE", 0.0, "INVALID_STOP_DISTANCE"

        # Riesgo unitario por contrato
        r_contract_std = delta_sl * self.std_multiplier + self.c_fric_std
        r_contract_micro = delta_sl * self.micro_multiplier + self.c_fric_micro

        # 1. Intentar contrato estándar
        n_std = int(np.floor(budget / r_contract_std))
        if n_std >= 1:
            return n_std, "STANDARD", r_contract_std, "OK"

        # 2. Conmutar a micro-contratos si estándar no cabe en presupuesto
        n_micro = int(np.floor(budget / r_contract_micro))
        n_micro = min(n_micro, self.max_micro_contracts)

        if n_micro >= 1:
            return n_micro, "MICRO", r_contract_micro, "OK"

        # 3. Disyuntor de volatilidad excesiva: ni un solo micro cabe en el presupuesto
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
        else:
            self.consecutive_losses = 0

        # Verificar límites de Prop Trading
        trailing_dd = self.peak_equity - self.current_equity
        if trailing_dd >= self.max_trailing_drawdown:
            self.is_halted_account = True

        if self.day_realized_loss >= self.daily_loss_limit:
            self.is_halted_day = True


class AnchoredVWAPValueAreaStrategy(BaseStrategy):
    """
    Estrategia Cuantitativa F3: Anchored VWAP & Rotational Value Area Reversion.
    
    Implementa la generación de señales conforme a la interfaz BaseStrategy,
    así como el motor de simulación y auditoría específico de Prop Trading.
    """

    def __init__(
        self,
        name: str = "Anchored_VWAP_Value_Area_F3",
        params: Optional[Dict[str, Any]] = None
    ):
        default_params = {
            "symbol": "ES",
            "anchor_time": "09:30",
            "window_init_end": "10:30",
            "window_revers_end": "15:00",
            "window_inhibit_end": "15:15",
            "window_flat_end": "16:00",
            "k_band": 2.0,            # 2.0 para ES, 2.5 para NQ
            "k_catastrophic": 3.0,     # Disyuntor 3-sigma
            "delta_max_slope": 0.15,   # 0.15 para ES, 0.45 para NQ
            "adx_period": 14,
            "adx_threshold": 25.0,
            "k_buffer_ticks": 4,       # Colchón de stop (4 ticks)
            "tick_size": 0.25,
            "account_size": 50000.0,
            "daily_loss_limit": 1000.0,
            "max_trailing_drawdown": 2000.0,
            "psi": 0.20,
            "max_micro_contracts": 15,
            "bar_minutes": 5.0,
            "session_end_time": "17:00"
        }
        if params:
            default_params.update(params)

        # Ajustar valores automáticos para NQ si aplica
        symbol_upper = default_params["symbol"].upper()
        if symbol_upper in ["NQ", "MNQ"]:
            if "k_band" not in (params or {}):
                default_params["k_band"] = 2.5
            if "delta_max_slope" not in (params or {}):
                default_params["delta_max_slope"] = 0.45

        super().__init__(name=name, params=default_params)

    def prepare_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los indicadores matemáticos requeridos para la estrategia."""
        self.validate_data(data)
        p = self.params
        df = compute_anchored_vwap_bands(
            df=data,
            anchor_time=p["anchor_time"],
            session_end_time=p.get("session_end_time", "17:00"),
            k_band=p["k_band"],
            k_catastrophic=p["k_catastrophic"],
            delta_max_slope=p["delta_max_slope"],
            adx_period=p["adx_period"],
            bar_minutes=p["bar_minutes"]
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
        sl_price = 0.0
        buffer_pts = p["k_buffer_ticks"] * p["tick_size"]

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
            current_upper_cat = df["UpperCatastrophic"].iloc[i]
            current_lower_cat = df["LowerCatastrophic"].iloc[i]

            # Si estamos en posición, evaluar salidas dinámicas
            if position == 1.0:
                # 1. Stop Loss Técnico / Disyuntor 3-sigma (la barrera más cercana al precio de entrada)
                stop_barrier = max(sl_price, current_lower_cat)
                if df["Low"].iloc[i] <= stop_barrier:
                    position = 0.0
                # 2. Dynamic TP en VWAP
                elif df["High"].iloc[i] >= current_vwap:
                    position = 0.0

            elif position == -1.0:
                # 1. Stop Loss Técnico / Disyuntor 3-sigma
                stop_barrier = min(sl_price, current_upper_cat)
                if df["High"].iloc[i] >= stop_barrier:
                    position = 0.0
                # 2. Dynamic TP en VWAP
                elif df["Low"].iloc[i] <= current_vwap:
                    position = 0.0

            # Si estamos neutrales, evaluar entradas en W_revers [10:30, 15:00)
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

                    # Señal de Compra Long Reversion (debe cerrar dentro de la banda y por debajo del VWAP)
                    is_long = (prev_low < prev_lower_band) and (curr_close >= curr_lower_band) and (curr_close < current_vwap)
                    # Señal de Venta Short Reversion (debe cerrar dentro de la banda y por encima del VWAP)
                    is_short = (prev_high > prev_upper_band) and (curr_close <= curr_upper_band) and (curr_close > current_vwap)

                    if is_long and not is_short:
                        position = 1.0
                        entry_price = curr_close
                        sl_price = min(prev_low, df["Low"].iloc[i]) - buffer_pts
                    elif is_short and not is_long:
                        position = -1.0
                        entry_price = curr_close
                        sl_price = max(prev_high, df["High"].iloc[i]) + buffer_pts

            signals.iloc[i] = position

        return signals

    def run_backtest(
        self,
        data: pd.DataFrame,
        risk_manager: Optional[PropRiskManager] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta el backtest formal con gestión de riesgo Prop Trading intra-barra.
        
        Args:
            data (pd.DataFrame): Datos OHLCV con DatetimeIndex de 5 minutos.
            risk_manager (PropRiskManager, opcional): Gestor de riesgo a emplear.
            
        Returns:
            Dict[str, Any]: Reporte financiero completo con trade log y curva de equidad.
        """
        df = self.prepare_indicators(data)
        p = self.params

        sym = p["symbol"].upper()
        # Resolver multiplicadores y comisiones
        if sym in ["NQ", "MNQ"]:
            std_sym = "NQ"
            micro_sym = "MNQ"
        else:
            std_sym = "ES"
            micro_sym = "MES"

        spec_std = get_spec(std_sym)
        spec_micro = get_spec(micro_sym)

        if risk_manager is None:
            risk_manager = PropRiskManager(
                account_size=p["account_size"],
                daily_loss_limit=p["daily_loss_limit"],
                max_trailing_drawdown=p["max_trailing_drawdown"],
                psi=p["psi"],
                max_micro_contracts=p["max_micro_contracts"],
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

        trades: List[Dict[str, Any]] = []
        equity_series = pd.Series(index=df.index, dtype=np.float64)

        in_position = False
        direction = 0  # +1 Long, -1 Short
        entry_idx = -1
        entry_time = None
        entry_price = 0.0
        sl_price = 0.0
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

            # Si había posición abierta del día anterior (anomalía), cerrar
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
            current_upper_cat = current_bar["UpperCatastrophic"]
            current_lower_cat = current_bar["LowerCatastrophic"]

            # Evaluar salidas si estamos en posición
            if in_position:
                exit_triggered = False
                exit_price = 0.0
                exit_reason = ""

                # Prioridad 1: Cierre Forzoso de Decaimiento Temporal a las 15:15 EST
                if t >= w_flat_time:
                    exit_triggered = True
                    exit_price = current_bar["Close"]
                    exit_reason = "FORCED_TIME_DECAY_1515"

                elif direction == 1:
                    stop_barrier = max(sl_price, current_lower_cat)
                    if current_bar["Low"] <= stop_barrier:
                        exit_triggered = True
                        if sl_price >= current_lower_cat:
                            exit_reason = "STOP_LOSS"
                            exit_price = min(current_bar["Open"], sl_price)
                        else:
                            exit_reason = "3_SIGMA_CIRCUIT_BREAKER"
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
                            exit_reason = "STOP_LOSS"
                            exit_price = max(current_bar["Open"], sl_price)
                        else:
                            exit_reason = "3_SIGMA_CIRCUIT_BREAKER"
                            exit_price = max(current_bar["Open"], current_upper_cat)
                    elif current_bar["Low"] <= current_vwap:
                        exit_triggered = True
                        exit_price = min(current_bar["Open"], current_vwap)
                        exit_reason = "TAKE_PROFIT_VWAP"

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

            # Evaluar entradas si NO estamos en posición y en W_revers [10:30, 15:00)
            if not in_position and (w_revers_start <= t < w_revers_end):
                if current_bar["Regime_Rotational"] == 1 and current_bar["IsAnchored"]:
                    prev_low = df["Low"].iloc[i - 1]
                    prev_lower_band = df["LowerBand"].iloc[i - 1]
                    curr_close = current_bar["Close"]
                    curr_lower_band = current_bar["LowerBand"]

                    prev_high = df["High"].iloc[i - 1]
                    prev_upper_band = df["UpperBand"].iloc[i - 1]
                    curr_upper_band = current_bar["UpperBand"]

                    candidate_direction = 0
                    candidate_sl = 0.0

                    # Señal Long Reversion (debe cerrar dentro de la banda y por debajo del VWAP)
                    is_long = (prev_low < prev_lower_band) and (curr_close >= curr_lower_band) and (curr_close < current_vwap)
                    # Señal Short Reversion (debe cerrar dentro de la banda y por encima del VWAP)
                    is_short = (prev_high > prev_upper_band) and (curr_close <= curr_upper_band) and (curr_close > current_vwap)

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
                            sl_price = candidate_sl
                            contracts = n_cnt
                            contract_type = c_type
                            if contract_type == "STANDARD":
                                contract_multiplier = risk_manager.std_multiplier
                                contract_friction = risk_manager.c_fric_std
                            else:
                                contract_multiplier = risk_manager.micro_multiplier
                                contract_friction = risk_manager.c_fric_micro

            # Tracking continuo de equidad (mark-to-market si hay posición abierta)
            if in_position:
                open_pnl = (current_bar["Close"] - entry_price) * direction * contract_multiplier * contracts
                equity_series.iloc[i] = risk_manager.current_equity + open_pnl
            else:
                equity_series.iloc[i] = risk_manager.current_equity

        equity_series.iloc[0] = risk_manager.account_size
        equity_series = equity_series.ffill().bfill()

        # Construir resumen de métricas
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
