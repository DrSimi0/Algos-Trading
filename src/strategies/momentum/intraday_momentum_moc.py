"""
src/strategies/momentum/intraday_momentum_moc.py
Estrategia F1: Market Intraday Momentum & MOC Seasonality.

Fundamento Académico:
- Gao, Han, Li, & Zhou (2018). Market Intraday Momentum. Journal of Financial Economics.
- Bogousslavsky (2021). The Cross-Section of Intraday and Overnight Returns. JFE.

Especificación Matemática y Microestructural:
- Ventana de Observación Matutina: [09:30:00, 10:00:00) EST
- Ventana de Inactividad Latente:  [10:00:00, 15:30:00) EST
- Ventana de Ejecución MOC:       [15:30:00, 15:58:00) EST
- Cierre forzoso MOC:             15:58:00 EST
- Control de Riesgo Prop Trading: DLL, MTDD, escalado por rachas perdedoras,
  conmutación Estándar/Micro y disyuntor de volatilidad.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from ...core.base_strategy import BaseStrategy
from ...core.futures_specs import get_spec


class MarketIntradayMomentumMOC(BaseStrategy):
    """
    Implementación formal de la Estrategia F1: Market Intraday Momentum & MOC.
    """

    def __init__(
        self,
        name: str = "Market_Intraday_Momentum_MOC",
        params: Optional[Dict[str, Any]] = None,
    ):
        default_params: Dict[str, Any] = {
            "symbol": "ES",
            "std_symbol": "ES",
            "micro_symbol": "MES",
            "theta_floor": 0.0015,         # 0.15% umbral mínimo
            "c_sigma": 0.20,               # Coeficiente de volatilidad
            "k_vol": 20,                   # Lookback histórico volumen matutino
            "k_sigma": 20,                 # Lookback histórico volatilidad diaria
            "lambda_sl": 0.0035,           # 0.35% distancia de stop loss
            "k_rr": 1.50,                  # Asimetría recompensa/riesgo (TP = 1.5 * SL)
            "psi": 0.25,                   # Fracción de margen MTDD por trade
            "phi_base": 0.35,              # Fracción base DLL por trade
            "n_max_micro": 10,             # Máximo de contratos micro
            "obs_start": "09:30",          # Inicio ventana observación (ET)
            "obs_end": "10:00",            # Fin ventana observación (ET)
            "exec_start": "15:30",         # Inicio ejecución MOC (ET)
            "moc_exit": "15:58",           # Salida forzosa MOC (ET)
            "target_tz": "America/New_York",
            "source_tz": None,             # Si los datos son ingenuos (naive), timezone origen
        }
        if params:
            default_params.update(params)
        super().__init__(name=name, params=default_params)

    # --------------------------------------------------------------------------
    # 1. Utilidades de Normalización Temporal
    # --------------------------------------------------------------------------
    def ensure_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Garantiza que el índice DatetimeIndex esté localizado en America/New_York (EST/EDT).
        Si el DataFrame es naive y se especificó `source_tz`, se localiza y convierte.
        """
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
    # 2. Formulación Matemática de Señales (Sección 3)
    # --------------------------------------------------------------------------
    @staticmethod
    def calculate_morning_return(p_open: float, p_close: float) -> float:
        """
        Retorno logarítmico matutino:
        r_first,t = ln( P_t(10:00) / P_t(09:30) )
        """
        if p_open <= 0 or p_close <= 0:
            raise ValueError(f"Precios deben ser estrictamente positivos: {p_open}, {p_close}")
        return float(np.log(p_close / p_open))

    @staticmethod
    def calculate_relative_volume(
        v_current: float,
        v_history: List[float],
        k_vol: int = 20
    ) -> Tuple[float, int]:
        """
        Ratio de volumen relativo institucional y condición booleana:
        V_ratio,t = V_t(09:30, 10:00) / ( (1/K_vol) * sum_{k=1}^{K_vol} V_{t-k} )
        Phi_vol,t = 1 si V_ratio,t >= 1.00, sino 0.
        """
        if len(v_history) == 0:
            # Sin histórico previo disponible (ej. periodo de calentamiento), ratio neutral
            return 1.0, 1

        recent_history = v_history[-k_vol:]
        mean_hist = float(np.mean(recent_history))
        if mean_hist <= 0:
            return 1.0, 1

        v_ratio = float(v_current / mean_hist)
        phi_vol = 1 if v_ratio >= 1.00 else 0
        return v_ratio, phi_vol

    def calculate_dynamic_threshold(
        self,
        p_1000: float,
        daily_prices_or_returns: pd.Series
    ) -> float:
        """
        Umbral dinámico indexado a la volatilidad:
        theta_t = max( theta_floor, c_sigma * (sigma_daily,t / P_t(10:00)) )
        """
        theta_floor = self.params["theta_floor"]
        c_sigma = self.params["c_sigma"]
        k_sigma = self.params["k_sigma"]

        if len(daily_prices_or_returns) < 2:
            return float(theta_floor)

        window = daily_prices_or_returns.tail(k_sigma)
        # Si la serie ya contiene retornos porcentuales diarios
        if (window.abs() < 0.30).all():
            sigma_rel = float(window.std(ddof=1))
        else:
            # Serie de precios en puntos: sigma_puntos / P
            sigma_pts = float(window.std(ddof=1))
            sigma_rel = sigma_pts / p_1000 if p_1000 > 0 else 0.0

        if np.isnan(sigma_rel) or sigma_rel <= 0:
            return float(theta_floor)

        return float(max(theta_floor, c_sigma * sigma_rel))

    def evaluate_signal(
        self,
        r_first: float,
        phi_vol: int,
        theta: float
    ) -> int:
        """
        Operador analítico integrado de señal S_t in {-1, 0, +1}:
        S_Long,t    = 1 si r_first,t > +theta_t and Phi_vol,t == 1
        S_Short,t   = 1 si r_first,t < -theta_t and Phi_vol,t == 1
        S_Neutral,t = 1 si |r_first,t| <= theta_t or Phi_vol,t == 0
        S_t = (+1)*S_Long + (-1)*S_Short + (0)*S_Neutral
        """
        if phi_vol == 1:
            if r_first > theta:
                return 1
            elif r_first < -theta:
                return -1
        return 0

    # --------------------------------------------------------------------------
    # 3. Formulación Analítica de Salida (Sección 4)
    # --------------------------------------------------------------------------
    def calculate_exit_levels(
        self,
        p_entry: float,
        direction: int
    ) -> Tuple[float, float]:
        """
        Stop Loss: P_SL,t = P_entry,t * (1 - S_t * lambda_SL)
        Take Profit: P_TP,t = P_entry,t * (1 + S_t * k_RR * lambda_SL)
        """
        lambda_sl = self.params["lambda_sl"]
        k_rr = self.params["k_rr"]

        if direction not in (-1, 1):
            raise ValueError(f"Dirección debe ser +1 o -1, recibido: {direction}")

        p_sl = p_entry * (1.0 - direction * lambda_sl)
        p_tp = p_entry * (1.0 + direction * k_rr * lambda_sl)
        return float(p_sl), float(p_tp)

    # --------------------------------------------------------------------------
    # 4. Dimensionamiento de Posición y Control Prop Trading (Sección 5)
    # --------------------------------------------------------------------------
    def calculate_risk_budget(
        self,
        dll: float,
        mtdd: float,
        dd_current: float,
        l_day: float = 0.0,
        k_loss: int = 0
    ) -> float:
        """
        B_risk,t = min( phi(K_loss) * max(0, DLL - L_day,t), psi * max(0, MTDD - DD_t) )
        donde phi(K_loss) = 0.35 * 2^(-min(K_loss, 3)) y psi = 0.25.
        """
        psi = self.params["psi"]
        phi_k = 0.35 * (2.0 ** (-min(max(k_loss, 0), 3)))

        daily_margin = max(0.0, dll - l_day) if dll is not None else float("inf")
        trailing_margin = max(0.0, mtdd - dd_current) if mtdd is not None else float("inf")

        b_daily = phi_k * daily_margin if dll is not None else float("inf")
        b_trailing = psi * trailing_margin if mtdd is not None else float("inf")

        b_risk = min(b_daily, b_trailing)
        return float(b_risk) if not np.isinf(b_risk) else float(self.account_size if hasattr(self, "account_size") else 5000.0)

    def calculate_contract_risk(
        self,
        p_entry: float,
        symbol: str
    ) -> float:
        """
        Riesgo monetario por contrato individual:
        R_contract,t(M) = Delta_SL,t * M + C_fric
        donde Delta_SL,t = P_entry,t * lambda_SL,
        C_fric = 2 * delta_tick * M + Comision_clearing.
        """
        spec = get_spec(symbol)
        m = spec["multiplier"]
        delta_tick = spec["tick"]
        commission_rt = spec["commission_rt"]
        lambda_sl = self.params["lambda_sl"]

        delta_sl = p_entry * lambda_sl
        c_fric = 2.0 * delta_tick * m + commission_rt
        r_contract = delta_sl * m + c_fric
        return float(r_contract)

    def calculate_position_sizing(
        self,
        b_risk: float,
        p_entry: float
    ) -> Dict[str, Any]:
        """
        Modelo declarativo de conmutación automática Estándar / Micro:
        N_std,t = floor( B_risk,t / R_contract,t(M_std) )
        Si N_std,t == 0:
            N_micro,t = min( floor( B_risk,t / R_contract,t(M_micro) ), N_max_micro )
        Disyuntor de volatilidad:
        Si R_contract,t(M_micro) > B_risk,t => N_t = 0 => HALT_VOLATILITY_DEFENSE
        """
        std_symbol = self.params["std_symbol"]
        micro_symbol = self.params["micro_symbol"]
        n_max_micro = self.params["n_max_micro"]

        if b_risk <= 0:
            return {
                "contract_type": "NONE",
                "symbol": micro_symbol,
                "contracts": 0,
                "risk_per_contract": 0.0,
                "status": "HALT_NO_BUDGET"
            }

        r_std = self.calculate_contract_risk(p_entry, std_symbol)
        n_std = int(np.floor(b_risk / r_std))

        if n_std >= 1:
            return {
                "contract_type": "STD",
                "symbol": std_symbol,
                "contracts": n_std,
                "risk_per_contract": r_std,
                "status": "OK"
            }

        # Conmutación a contratos Micro
        r_micro = self.calculate_contract_risk(p_entry, micro_symbol)
        if r_micro > b_risk:
            # Disyuntor de volatilidad activado
            return {
                "contract_type": "NONE",
                "symbol": micro_symbol,
                "contracts": 0,
                "risk_per_contract": r_micro,
                "status": "HALT_VOLATILITY_DEFENSE"
            }

        n_micro = min(int(np.floor(b_risk / r_micro)), n_max_micro)
        return {
            "contract_type": "MICRO",
            "symbol": micro_symbol,
            "contracts": n_micro,
            "risk_per_contract": r_micro,
            "status": "OK" if n_micro > 0 else "HALT_VOLATILITY_DEFENSE"
        }

    # --------------------------------------------------------------------------
    # 5. Generación Vectorial de Señales (Interfaz BaseStrategy)
    # --------------------------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera la serie de señales e interfaz canónica para BaseStrategy:
        +1.0 durante la ventana de exposición larga,
        -1.0 durante la ventana corta,
         0.0 fuera de mercado (letargo intradía o cierre).
        """
        self.validate_data(data)
        df = self.ensure_timezone(data.copy())

        sig_array = np.zeros(len(df), dtype=np.float64)
        daily_groups = df.groupby(df.index.normalize())

        morning_vols: List[float] = []
        daily_closes: List[float] = []

        # Mapa de índice temporal a posición entera en el array
        ts_to_idx = {ts: idx for idx, ts in enumerate(df.index)}

        for day_dt, day_df in daily_groups:
            # 1. Ventana de Observación [09:30, 10:00) EST (intervalo semiabierto estricto)
            obs_bars = day_df.between_time(
                self.params["obs_start"],
                self.params["obs_end"],
                inclusive="left"
            )
            if len(obs_bars) < 2:
                if len(day_df) > 0:
                    daily_closes.append(float(day_df["Close"].iloc[-1]))
                continue

            p_930 = float(obs_bars["Open"].iloc[0])
            p_1000 = float(obs_bars["Close"].iloc[-1])
            v_current = float(obs_bars["Volume"].sum())

            # Log return y volumen relativo
            r_first = self.calculate_morning_return(p_930, p_1000)
            v_ratio, phi_vol = self.calculate_relative_volume(v_current, morning_vols, self.params["k_vol"])
            morning_vols.append(v_current)

            # Umbral de volatilidad
            theta = self.calculate_dynamic_threshold(p_1000, pd.Series(daily_closes))
            daily_closes.append(float(day_df["Close"].iloc[-1]))

            # Señal discreta a las 15:30
            sig = self.evaluate_signal(r_first, phi_vol, theta)
            if sig == 0:
                continue

            # Ventana de Ejecución [15:30, 15:58) EST
            exec_bars = day_df.between_time(
                self.params["exec_start"],
                self.params["moc_exit"],
                inclusive="left"
            )
            if len(exec_bars) == 0:
                continue

            p_entry = float(exec_bars["Open"].iloc[0])
            p_sl, p_tp = self.calculate_exit_levels(p_entry, sig)

            for ts, row in exec_bars.iterrows():
                high = float(row["High"])
                low = float(row["Low"])
                arr_i = ts_to_idx.get(ts)

                # Verificación de Stop Loss y Take Profit
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

        # Garantizar que la serie retornada tenga exactamente el mismo índice que data
        return pd.Series(sig_array, index=data.index, dtype=np.float64)


class IntradayMomentumBacktester:
    """
    Simulador de alta fidelidad cuantitativa para la Estrategia F1.
    Soporta:
    - Reglas rigurosas de Prop Trading (Daily Loss Limit, Max Trailing Drawdown EOD/Intradía).
    - Conmutación automática estándar (ES/NQ) a micro (MES/MNQ).
    - Deslizamiento exacto en ticks y comisiones de clearing CME.
    - Presupuesto de riesgo modulado por rachas consecutivas de pérdidas phi(K_loss).
    - Cierre determinista a las 15:58:00 EST (MOC).
    """

    def __init__(
        self,
        strategy: Optional[MarketIntradayMomentumMOC] = None,
        account_size: float = 50000.0,
        daily_loss_limit: float = 1000.0,
        max_trailing_loss: float = 2000.0,
        trailing_mode: str = "eod",          # "eod" o "intraday"
        lock_floor_at_start: bool = True,
        profit_target: float = 3000.0,
        min_days: int = 2,
    ):
        self.strategy = strategy or MarketIntradayMomentumMOC()
        self.account_size = account_size
        self.daily_loss_limit = daily_loss_limit
        self.max_trailing_loss = max_trailing_loss
        self.trailing_mode = trailing_mode
        self.lock_floor_at_start = lock_floor_at_start
        self.profit_target = profit_target
        self.min_days = min_days

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Ejecuta la simulación sesión a sesión.
        """
        self.strategy.validate_data(data)
        df = self.strategy.ensure_timezone(data.copy())

        balance = self.account_size
        peak = balance
        floor = self._update_floor(peak)

        k_loss = 0
        trades: List[Dict[str, Any]] = []
        daily_logs: List[Dict[str, Any]] = []

        morning_vols: List[float] = []
        daily_closes: List[float] = []

        failed = False
        fail_reason: Optional[str] = None
        fail_date: Optional[Any] = None
        passed = False
        pass_date: Optional[Any] = None

        grouped = df.groupby(df.index.normalize())

        for day_dt, day_df in grouped:
            if failed or passed:
                break

            day_start_balance = balance
            l_day = 0.0

            # 1. Ventana de observación matutina [09:30, 10:00) EST (semiabierto estricto)
            obs_bars = day_df.between_time(
                self.strategy.params["obs_start"],
                self.strategy.params["obs_end"],
                inclusive="left"
            )

            if len(obs_bars) < 2:
                if len(day_df) > 0:
                    daily_closes.append(float(day_df["Close"].iloc[-1]))
                continue

            p_930 = float(obs_bars["Open"].iloc[0])
            p_1000 = float(obs_bars["Close"].iloc[-1])
            v_current = float(obs_bars["Volume"].sum())

            r_first = self.strategy.calculate_morning_return(p_930, p_1000)
            v_ratio, phi_vol = self.strategy.calculate_relative_volume(
                v_current, morning_vols, self.strategy.params["k_vol"]
            )
            morning_vols.append(v_current)

            theta = self.strategy.calculate_dynamic_threshold(
                p_1000, pd.Series(daily_closes)
            )
            daily_closes.append(float(day_df["Close"].iloc[-1]))

            # 2. Evaluación de señal a las 15:30 EST
            sig = self.strategy.evaluate_signal(r_first, phi_vol, theta)

            exec_bars = day_df.between_time(
                self.strategy.params["exec_start"],
                self.strategy.params["moc_exit"],
                inclusive="left"
            )

            if sig != 0 and len(exec_bars) > 0:
                p_entry = float(exec_bars["Open"].iloc[0])
                current_dd = max(0.0, peak - balance)

                # Presupuesto de riesgo Prop Trading
                b_risk = self.strategy.calculate_risk_budget(
                    dll=self.daily_loss_limit,
                    mtdd=self.max_trailing_loss,
                    dd_current=current_dd,
                    l_day=l_day,
                    k_loss=k_loss
                )

                sizing = self.strategy.calculate_position_sizing(b_risk, p_entry)

                if sizing["contracts"] > 0:
                    spec = get_spec(sizing["symbol"])
                    m = spec["multiplier"]
                    delta_tick = spec["tick"]
                    commission_rt = spec["commission_rt"]
                    p_sl, p_tp = self.strategy.calculate_exit_levels(p_entry, sig)

                    exit_time = exec_bars.index[-1]
                    exit_price = float(exec_bars["Close"].iloc[-1])
                    exit_reason = "MOC"

                    # Simulación barra a barra en la ventana de ejecución
                    for ts, row in exec_bars.iterrows():
                        high = float(row["High"])
                        low = float(row["Low"])

                        if sig == 1:
                            if low <= p_sl:
                                exit_time = ts
                                exit_price = min(float(row["Open"]), p_sl) if float(row["Open"]) <= p_sl else p_sl
                                exit_reason = "SL"
                                break
                            elif high >= p_tp:
                                exit_time = ts
                                exit_price = max(float(row["Open"]), p_tp) if float(row["Open"]) >= p_tp else p_tp
                                exit_reason = "TP"
                                break
                        elif sig == -1:
                            if high >= p_sl:
                                exit_time = ts
                                exit_price = max(float(row["Open"]), p_sl) if float(row["Open"]) >= p_sl else p_sl
                                exit_reason = "SL"
                                break
                            elif low <= p_tp:
                                exit_time = ts
                                exit_price = min(float(row["Open"]), p_tp) if float(row["Open"]) <= p_tp else p_tp
                                exit_reason = "TP"
                                break


                    # Cálculo P&L y fricciones
                    pts_diff = (exit_price - p_entry) if sig == 1 else (p_entry - exit_price)
                    gross_pnl = pts_diff * m * sizing["contracts"]
                    # 1 tick entrada + 1 tick salida = 2 ticks slippage total
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

                    # Intraday trailing floor
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
                        "r_first_pct": round(r_first * 100, 3),
                        "theta_pct": round(theta * 100, 3),
                        "v_ratio": round(v_ratio, 2),
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

            # Cierre EOD del día
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

            # Comprobar Target de Aprobación
            if (balance - self.account_size) >= self.profit_target and len(daily_logs) >= self.min_days:
                passed = True
                pass_date = day_dt

        # Recopilación de estadísticas
        trades_df = pd.DataFrame(trades)
        metrics = self._calculate_performance_metrics(trades_df, balance)

        prop_evaluation = {
            "status": "PASS" if passed else ("FAIL" if failed else "INCOMPLETE"),
            "reason": fail_reason if failed else ("Profit Target Achieved" if passed else "Ongoing"),
            "date": pass_date if passed else fail_date,
            "final_balance": round(balance, 2),
            "net_profit": round(balance - self.account_size, 2),
            "total_days_evaluated": len(daily_logs),
        }

        return {
            "metrics": metrics,
            "prop_evaluation": prop_evaluation,
            "trades": trades_df,
            "daily_logs": pd.DataFrame(daily_logs),
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

        # Curva de equidad iniciando con el capital de la cuenta
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
