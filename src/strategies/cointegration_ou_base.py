"""
src/strategies/cointegration_ou_base.py
Versión "Base" (Baseline) de la Estrategia F4: Cointegración Inter-Mercado ES vs NQ y Proceso OU.

Esta versión preserva la lógica de línea base previa a la optimización IS-N1, permitiendo
volver a ella, comparar resultados o auditar el comportamiento original.
"""

from dataclasses import dataclass
from datetime import time
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts

from src.core.base_strategy import BaseStrategy
from src.core.futures_specs import get_spec

warnings.filterwarnings("ignore", category=FutureWarning)


class TradeSignalBase(int, Enum):
    SELL_SPREAD = -1
    NEUTRAL = 0
    BUY_SPREAD = 1


class ExitReasonBase(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TIME_DECAY = "TIME_DECAY"
    EOD_FLAT = "EOD_FLAT"
    RISK_BREACH = "RISK_BREACH"


@dataclass
class TradeRecordBase:
    trade_id: int
    entry_date: str
    entry_time: str
    signal: str
    direction: int
    contracts_nq: int
    contracts_es: int
    symbol_nq: str
    symbol_es: str
    n_baskets: int
    entry_price_nq: float
    entry_price_es: float
    entry_s_score: float
    entry_beta: float
    entry_tau_half: float
    exit_date: Optional[str] = None
    exit_time: Optional[str] = None
    exit_price_nq: Optional[float] = None
    exit_price_es: Optional[float] = None
    exit_s_score: Optional[float] = None
    exit_reason: Optional[ExitReasonBase] = None
    bars_held: int = 0
    duration_minutes: float = 0.0
    gross_pnl_usd: float = 0.0
    commissions_usd: float = 0.0
    slippage_usd: float = 0.0
    friction_usd: float = 0.0
    net_pnl_usd: float = 0.0
    running_balance: float = 0.0
    running_drawdown: float = 0.0


class CointegrationOUModelBase:
    @staticmethod
    def estimate_cointegration(p_nq: np.ndarray, p_es: np.ndarray) -> Tuple[float, float, np.ndarray]:
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
    def test_stationarity(residuals: np.ndarray, t_stat_threshold: float = -3.45, p_val_threshold: float = 0.05) -> Tuple[bool, float, float]:
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
    def fit_ou_process(residuals: np.ndarray, dt_minutes: float = 5.0, sigma_floor: float = 0.001) -> Dict[str, float]:
        n = len(residuals)
        if n < 10:
            return {"b": 0.0, "a": 0.0, "kappa": 0.0, "theta": 0.0, "sigma_eta": 0.0, "sigma_eq": 0.0, "sigma": 0.0, "tau_half_min": 0.0, "is_valid": False}
        x_prev = residuals[:-1]
        x_curr = residuals[1:]
        var_prev = np.var(x_prev, ddof=1)
        if var_prev <= 1e-14:
            return {"b": 0.0, "a": 0.0, "kappa": 0.0, "theta": 0.0, "sigma_eta": 0.0, "sigma_eq": 0.0, "sigma": 0.0, "tau_half_min": 0.0, "is_valid": False}
        cov_matrix = np.cov(x_curr, x_prev, ddof=1)
        cov_curr_prev = cov_matrix[0, 1]
        b = float(cov_curr_prev / var_prev)
        a = float(np.mean(x_curr) - b * np.mean(x_prev))
        if b <= 0.0 or b >= 1.0:
            return {"b": b, "a": a, "kappa": 0.0, "theta": 0.0, "sigma_eta": 0.0, "sigma_eq": 0.0, "sigma": 0.0, "tau_half_min": 0.0, "is_valid": False}
        eta = x_curr - (a + b * x_prev)
        sigma_eta_sq = float(np.var(eta, ddof=2)) if len(eta) > 2 else float(np.var(eta))
        if sigma_eta_sq <= 1e-14:
            sigma_eta_sq = 1e-14
        kappa = float(-np.log(b) / dt_minutes)
        theta = float(a / (1.0 - b))
        sigma_eq_sq = float(sigma_eta_sq / (1.0 - b**2))
        sigma_eq_raw = float(np.sqrt(max(1e-14, sigma_eq_sq)))
        sigma_eq = float(max(sigma_eq_raw, sigma_floor))
        sigma = float(sigma_eq * np.sqrt(2.0 * kappa))
        tau_half_min = float(np.log(2.0) / kappa)
        return {
            "b": b, "a": a, "kappa": kappa, "kappa_daily": float(kappa * 390.0),
            "theta": theta, "sigma_eta": float(np.sqrt(sigma_eta_sq)),
            "sigma_eq": sigma_eq, "sigma": sigma, "tau_half_min": tau_half_min, "is_valid": True
        }

    @staticmethod
    def calculate_s_score(x_k: float, theta: float, sigma_eq: float, sigma_floor: float = 0.001) -> float:
        eff_sigma = max(sigma_eq, sigma_floor)
        if eff_sigma <= 1e-12:
            return 0.0
        return float((x_k - theta) / eff_sigma)

    @staticmethod
    def calculate_dollar_beta_ratio(p_nq: float, p_es: float, beta_hedge: float, m_nq: float = 2.0, m_es: float = 5.0) -> float:
        if p_es <= 0 or m_es <= 0:
            return 1.0
        return float(beta_hedge * ((p_nq * m_nq) / (p_es * m_es)))

    @staticmethod
    def calculate_risk_budget(dll: float, tmdd: float, realized_loss_today: float = 0.0, current_drawdown: float = 0.0, phi_dll: float = 0.35, phi_tmdd: float = 0.20) -> float:
        daily_cushion = max(0.0, dll - realized_loss_today)
        tmdd_cushion = max(0.0, tmdd - current_drawdown)
        budget_daily = phi_dll * daily_cushion
        budget_tmdd = phi_tmdd * tmdd_cushion
        return float(max(0.0, min(budget_daily, budget_tmdd)))

    @staticmethod
    def calculate_unit_stop_loss_loss(sigma_eq: float, p_nq: float, n_nq_unit: int, m_nq: float, friction_basket_usd: float, delta_s_sl: float = 2.00, sigma_floor: float = 0.001) -> float:
        eff_sigma = max(sigma_eq, sigma_floor)
        delta_x_sl = delta_s_sl * eff_sigma
        notional_nq = n_nq_unit * m_nq * p_nq
        loss_pts = delta_x_sl * notional_nq
        return float(loss_pts + friction_basket_usd)

    @staticmethod
    def calculate_baskets_allocation(risk_budget_usd: float, unit_sl_loss_usd: float, n_basket_max: int = 3) -> int:
        if risk_budget_usd <= 0 or unit_sl_loss_usd <= 0:
            return 0
        raw_baskets = int(np.floor(risk_budget_usd / unit_sl_loss_usd))
        return int(max(0, min(raw_baskets, n_basket_max)))


class CointegrationOUStrategyBase(BaseStrategy):
    def __init__(self, name: str = "Cointegration_OU_ES_NQ_Base", params: Optional[Dict[str, Any]] = None):
        default_params = {
            "symbol_nq": "NQ", "symbol_es": "ES", "micro_nq": "MNQ", "micro_es": "MES",
            "contract_mode": "micro", "basket_unit_mnq": 10, "basket_unit_mes": 17,
            "basket_unit_nq": 1, "basket_unit_es": 2, "allow_sub_baskets": True,
            "dynamic_basket_ratio": True, "k_calib": 2340, "min_calib_bars": 390,
            "dt_minutes": 5.0, "s_open": 1.25, "s_close": 0.25, "s_stop": 3.25,
            "tau_min": 15.0, "tau_max": 780.0, "adf_t_threshold": -3.45, "adf_p_threshold": 0.05,
            "rth_start": "09:30", "rth_end": "16:00", "entry_start": "10:00", "entry_end": "15:40",
            "eod_flat": "15:50", "time_decay_multiplier": 2.0, "phi_dll": 0.35, "phi_tmdd": 0.20,
            "n_basket_max": 3, "slippage_ticks": 1.0, "target_tz": "America/New_York", "source_tz": "America/Mexico_City",
            "sigma_floor": 0.001,
        }
        if params:
            default_params.update(params)
        super().__init__(name=name, params=default_params)
        self.model = CointegrationOUModelBase()

    def ensure_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        target_tz = self.params["target_tz"]
        source_tz = self.params.get("source_tz")
        if df.index.tz is None:
            return df.tz_localize(source_tz).tz_convert(target_tz) if source_tz else df.tz_localize(target_tz)
        return df.tz_convert(target_tz)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if "NQ" in data.columns and "ES" in data.columns:
            nq_close = data["NQ"]
            es_close = data["ES"]
        elif "Close_NQ" in data.columns and "Close_ES" in data.columns:
            nq_close = data["Close_NQ"]
            es_close = data["Close_ES"]
        else:
            raise ValueError("No se encontraron columnas para NQ y ES.")

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
            beta, alpha, res_hist = self.model.estimate_cointegration(np.log(hist_nq.values), np.log(hist_es.values))
            ou_params = self.model.fit_ou_process(res_hist, dt_minutes=self.params["dt_minutes"])
            if not ou_params["is_valid"]:
                if current_signal != 0:
                    current_signal = 0
                signals.loc[ts] = 0.0
                continue

            p_nq_cur = float(rth_bars["NQ"].iloc[i])
            p_es_cur = float(rth_bars["ES"].iloc[i])
            res_cur = np.log(p_nq_cur) - (alpha + beta * np.log(p_es_cur))
            s_score = self.model.calculate_s_score(res_cur, ou_params["theta"], ou_params["sigma_eq"])
            bar_time_str = ts.strftime("%H:%M")

            if current_signal != 0:
                elapsed_min = (i - entry_idx) * self.params["dt_minutes"]
                should_exit = False
                if current_signal == -1 and s_score <= self.params["s_close"]:
                    should_exit = True
                elif current_signal == 1 and s_score >= -self.params["s_close"]:
                    should_exit = True
                if current_signal == -1 and s_score >= self.params["s_stop"]:
                    should_exit = True
                elif current_signal == 1 and s_score <= -self.params["s_stop"]:
                    should_exit = True
                if elapsed_min >= self.params["time_decay_multiplier"] * current_tau:
                    should_exit = True
                if bar_time_str >= self.params["eod_flat"]:
                    should_exit = True
                if should_exit:
                    current_signal = 0
                    signals.loc[ts] = 0.0
                else:
                    signals.loc[ts] = float(current_signal)
                continue

            in_entry_window = (self.params["entry_start"] <= bar_time_str <= self.params["entry_end"])
            tau_ok = (self.params["tau_min"] <= ou_params["tau_half_min"] <= self.params["tau_max"])
            if in_entry_window and tau_ok and current_signal == 0:
                is_stat, _, _ = self.model.test_stationarity(
                    res_hist, t_stat_threshold=self.params["adf_t_threshold"], p_val_threshold=self.params["adf_p_threshold"]
                )
                if is_stat:
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


class CointegrationOUBacktesterBase:
    def __init__(
        self,
        strategy: Optional[CointegrationOUStrategyBase] = None,
        account_size: float = 50000.0,
        daily_loss_limit: float = 1000.0,
        max_trailing_loss: float = 2000.0,
        trailing_mode: str = "eod",
        lock_floor_at_start: bool = True,
        profit_target: float = 3000.0,
        min_days: int = 5,
    ):
        self.strategy = strategy or CointegrationOUStrategyBase()
        self.account_size = account_size
        self.daily_loss_limit = daily_loss_limit
        self.max_trailing_loss = max_trailing_loss
        self.trailing_mode = trailing_mode
        self.lock_floor_at_start = lock_floor_at_start
        self.profit_target = profit_target
        self.min_days = min_days
        self.model = self.strategy.model

    def _update_floor(self, peak: float) -> float:
        floor = peak - self.max_trailing_loss
        if self.lock_floor_at_start:
            floor = min(floor, self.account_size)
        return floor

    def run(self, df_nq: pd.DataFrame, df_es: pd.DataFrame) -> Dict[str, Any]:
        df_nq_ny = self.strategy.ensure_timezone(df_nq.copy())
        df_es_ny = self.strategy.ensure_timezone(df_es.copy())
        common_index = df_nq_ny.index.intersection(df_es_ny.index)
        if len(common_index) == 0:
            raise ValueError("No hay coincidencia de marcas temporales entre NQ y ES.")

        df_nq_ny = df_nq_ny.loc[common_index].sort_index()
        df_es_ny = df_es_ny.loc[common_index].sort_index()

        rth_mask = (df_nq_ny.index.time >= time(9, 30)) & (df_nq_ny.index.time <= time(16, 0))
        rth_nq = df_nq_ny[rth_mask]
        rth_es = df_es_ny[rth_mask]

        is_micro = (self.strategy.params["contract_mode"] == "micro")
        sym_nq = self.strategy.params["micro_nq"] if is_micro else self.strategy.params["symbol_nq"]
        sym_es = self.strategy.params["micro_es"] if is_micro else self.strategy.params["symbol_es"]
        spec_nq = get_spec(sym_nq)
        spec_es = get_spec(sym_es)

        m_nq, m_es = spec_nq["multiplier"], spec_es["multiplier"]
        tick_nq, tick_es = spec_nq["tick"], spec_es["tick"]
        comm_nq, comm_es = spec_nq["commission_rt"], spec_es["commission_rt"]
        unit_nq = self.strategy.params["basket_unit_mnq"] if is_micro else self.strategy.params["basket_unit_nq"]
        unit_es = self.strategy.params["basket_unit_mes"] if is_micro else self.strategy.params["basket_unit_es"]

        balance = self.account_size
        peak = balance
        floor = self._update_floor(peak)
        failed, fail_reason, fail_date = False, None, None
        passed, pass_date = False, None

        trades: List[TradeRecordBase] = []
        daily_logs: List[Dict[str, Any]] = []
        k_calib = self.strategy.params["k_calib"]
        min_calib = min(self.strategy.params["min_calib_bars"], k_calib)

        grouped = rth_nq.groupby(rth_nq.index.normalize())
        in_position = False
        current_trade: Optional[TradeRecordBase] = None
        trade_counter = 0
        global_bar_idx = 0

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
                beta, alpha, res_hist = self.model.estimate_cointegration(np.log(calib_nq), np.log(calib_es))
                ou = self.model.fit_ou_process(res_hist, dt_minutes=self.strategy.params["dt_minutes"])
                res_cur = np.log(p_nq_cur) - (alpha + beta * np.log(p_es_cur))
                s_score = self.model.calculate_s_score(res_cur, ou["theta"], ou["sigma_eq"]) if ou["is_valid"] else 0.0

                is_stat = False
                if ou["is_valid"]:
                    tau_ok = (self.strategy.params["tau_min"] <= ou["tau_half_min"] <= self.strategy.params["tau_max"])
                    is_stat, _, _ = self.model.test_stationarity(
                        res_hist, t_stat_threshold=self.strategy.params["adf_t_threshold"], p_val_threshold=self.strategy.params["adf_p_threshold"]
                    )

                if in_position and current_trade is not None:
                    current_trade.bars_held += 1
                    current_trade.duration_minutes += self.strategy.params["dt_minutes"]
                    elapsed_min = current_trade.duration_minutes
                    pts_nq_float = (p_nq_cur - current_trade.entry_price_nq) * current_trade.direction
                    pts_es_float = (p_es_cur - current_trade.entry_price_es) * (-current_trade.direction)
                    unrealized_gross = pts_nq_float * m_nq * current_trade.contracts_nq + pts_es_float * m_es * current_trade.contracts_es
                    friction_est = (
                        current_trade.contracts_nq * comm_nq + current_trade.contracts_es * comm_es +
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
                    exit_reason_code: Optional[ExitReasonBase] = None
                    if unrealized_loss_today >= self.daily_loss_limit or unrealized_bal <= floor:
                        exit_triggered = True
                        exit_reason_code = ExitReasonBase.RISK_BREACH
                    if not exit_triggered:
                        if current_trade.direction == -1 and s_score <= self.strategy.params["s_close"]:
                            exit_triggered = True
                            exit_reason_code = ExitReasonBase.TAKE_PROFIT
                        elif current_trade.direction == 1 and s_score >= -self.strategy.params["s_close"]:
                            exit_triggered = True
                            exit_reason_code = ExitReasonBase.TAKE_PROFIT
                    if not exit_triggered:
                        if current_trade.direction == -1 and s_score >= self.strategy.params["s_stop"]:
                            exit_triggered = True
                            exit_reason_code = ExitReasonBase.STOP_LOSS
                        elif current_trade.direction == 1 and s_score <= -self.strategy.params["s_stop"]:
                            exit_triggered = True
                            exit_reason_code = ExitReasonBase.STOP_LOSS
                    if not exit_triggered:
                        max_time = self.strategy.params["time_decay_multiplier"] * current_trade.entry_tau_half
                        if elapsed_min >= max_time:
                            exit_triggered = True
                            exit_reason_code = ExitReasonBase.TIME_DECAY
                    if not exit_triggered and (bar_time_str >= self.strategy.params["eod_flat"] or i == len(day_nq) - 1):
                        exit_triggered = True
                        exit_reason_code = ExitReasonBase.EOD_FLAT

                    if exit_triggered:
                        exit_price_nq = p_nq_cur
                        exit_price_es = p_es_cur
                        pts_nq = (exit_price_nq - current_trade.entry_price_nq) * current_trade.direction
                        pts_es = (exit_price_es - current_trade.entry_price_es) * (-current_trade.direction)
                        gross_pnl = pts_nq * m_nq * current_trade.contracts_nq + pts_es * m_es * current_trade.contracts_es
                        commissions = current_trade.contracts_nq * comm_nq + current_trade.contracts_es * comm_es
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
                            failed, fail_reason, fail_date = True, "Max Trailing Drawdown Breached", ts
                            break
                        elif daily_realized_loss >= self.daily_loss_limit:
                            failed, fail_reason, fail_date = True, "Daily Loss Limit Breached", ts
                            break
                        continue

                if not in_position and not failed:
                    in_window = (self.strategy.params["entry_start"] <= bar_time_str <= self.strategy.params["entry_end"])
                    valid_ou = ou["is_valid"]
                    valid_tau = (self.strategy.params["tau_min"] <= ou["tau_half_min"] <= self.strategy.params["tau_max"]) if valid_ou else False
                    if in_window and valid_ou and valid_tau:
                        sig_type = TradeSignalBase.NEUTRAL
                        if self.strategy.params["s_open"] < s_score < self.strategy.params["s_stop"]:
                            sig_type = TradeSignalBase.SELL_SPREAD
                        elif -self.strategy.params["s_stop"] < s_score < -self.strategy.params["s_open"]:
                            sig_type = TradeSignalBase.BUY_SPREAD

                        if sig_type != TradeSignalBase.NEUTRAL and is_stat:
                            tmdd_cushion = max(0.0, balance - floor)
                            daily_cushion = max(0.0, self.daily_loss_limit - daily_realized_loss)
                            risk_budget = min(self.strategy.params["phi_dll"] * daily_cushion, self.strategy.params.get("phi_tmdd", 0.20) * tmdd_cushion)
                            risk_budget = float(max(0.0, risk_budget))
                            exact_ratio = self.model.calculate_dollar_beta_ratio(p_nq_cur, p_es_cur, beta, m_nq, m_es)
                            cur_unit_nq = unit_nq
                            cur_unit_es = max(1, int(round(cur_unit_nq * exact_ratio))) if self.strategy.params.get("dynamic_basket_ratio", True) else unit_es
                            friction_unit = (
                                cur_unit_nq * comm_nq + cur_unit_es * comm_es +
                                2.0 * self.strategy.params["slippage_ticks"] * tick_nq * m_nq * cur_unit_nq +
                                2.0 * self.strategy.params["slippage_ticks"] * tick_es * m_es * cur_unit_es
                            )
                            unit_sl_loss = self.model.calculate_unit_stop_loss_loss(
                                sigma_eq=ou["sigma_eq"], p_nq=p_nq_cur, n_nq_unit=cur_unit_nq, m_nq=m_nq,
                                friction_basket_usd=friction_unit, delta_s_sl=(self.strategy.params["s_stop"] - self.strategy.params["s_open"])
                            )
                            n_baskets = self.model.calculate_baskets_allocation(risk_budget, unit_sl_loss, self.strategy.params["n_basket_max"])

                            if n_baskets == 0 and self.strategy.params.get("allow_sub_baskets", False) and is_micro:
                                sub_unit_nq = 1
                                sub_unit_es = max(1, int(round(sub_unit_nq * exact_ratio))) if self.strategy.params.get("dynamic_basket_ratio", True) else 2
                                sub_fric = (
                                    sub_unit_nq * comm_nq + sub_unit_es * comm_es +
                                    2.0 * self.strategy.params["slippage_ticks"] * tick_nq * m_nq * sub_unit_nq +
                                    2.0 * self.strategy.params["slippage_ticks"] * tick_es * m_es * sub_unit_es
                                )
                                sub_sl_loss = self.model.calculate_unit_stop_loss_loss(
                                    sigma_eq=ou["sigma_eq"], p_nq=p_nq_cur, n_nq_unit=sub_unit_nq, m_nq=m_nq,
                                    friction_basket_usd=sub_fric, delta_s_sl=(self.strategy.params["s_stop"] - self.strategy.params["s_open"])
                                )
                                n_sub = self.model.calculate_baskets_allocation(risk_budget, sub_sl_loss, self.strategy.params["n_basket_max"] * 10)
                                if n_sub >= 1:
                                    cur_unit_nq, cur_unit_es, n_baskets = sub_unit_nq, sub_unit_es, n_sub

                            if n_baskets >= 1:
                                trade_counter += 1
                                in_position = True
                                current_trade = TradeRecordBase(
                                    trade_id=trade_counter,
                                    entry_date=ts.strftime("%Y-%m-%d"),
                                    entry_time=ts.strftime("%H:%M:%S"),
                                    signal="BUY_SPREAD" if sig_type == TradeSignalBase.BUY_SPREAD else "SELL_SPREAD",
                                    direction=1 if sig_type == TradeSignalBase.BUY_SPREAD else -1,
                                    contracts_nq=cur_unit_nq * n_baskets,
                                    contracts_es=cur_unit_es * n_baskets,
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
                "day_pnl": round(day_pnl, 2), "ending_balance": round(balance, 2),
                "peak": round(peak, 2), "floor": round(floor, 2), "daily_realized_loss": round(daily_realized_loss, 2),
            })
            if (balance - self.account_size) >= self.profit_target and len(daily_logs) >= self.min_days:
                passed, pass_date = True, day_dt

        trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame()
        daily_df = pd.DataFrame(daily_logs) if daily_logs else pd.DataFrame()
        metrics = self._calculate_metrics(trades_df, balance)
        return {
            "metrics": metrics,
            "prop_evaluation": {
                "status": "PASS" if passed else ("FAIL" if failed else "INCOMPLETE"),
                "reason": fail_reason if failed else ("Profit Target Achieved" if passed else "Evaluation Ongoing"),
                "date": str(pass_date) if passed else (str(fail_date) if failed else None),
                "final_balance": round(balance, 2),
                "net_profit": round(balance - self.account_size, 2),
                "peak_equity": round(peak, 2),
                "final_floor": round(floor, 2),
                "total_days_evaluated": len(daily_logs),
            },
            "trades": trades_df,
            "daily_logs": daily_df,
        }

    def _calculate_metrics(self, trades_df: pd.DataFrame, final_balance: float) -> Dict[str, Any]:
        if trades_df.empty:
            return {"Total Trades": 0, "Win Rate (%)": 0.0, "Profit Factor": 0.0, "Net Profit ($)": 0.0, "Total Return (%)": 0.0, "Max Drawdown ($)": 0.0, "Sharpe Ratio": 0.0, "Sortino Ratio": 0.0}
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
        return {
            "Total Trades": int(len(trades_df)), "Winning Trades": int(len(wins)), "Losing Trades": int(len(losses)),
            "Win Rate (%)": round(win_rate, 2), "Gross Profit ($)": round(gross_profit, 2), "Gross Loss ($)": round(gross_loss, 2),
            "Profit Factor": round(profit_factor, 2) if not np.isinf(profit_factor) else "INF",
            "Net Profit ($)": round(final_balance - self.account_size, 2),
            "Total Return (%)": round(((final_balance / self.account_size) - 1.0) * 100.0, 2),
            "Max Drawdown ($)": round(max_dd, 2), "Sharpe Ratio": round(sharpe, 2), "Sortino Ratio": round(sortino, 2),
        }
