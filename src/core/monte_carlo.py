"""
src/core/monte_carlo.py
Simulador de Monte Carlo para análisis de robustez y evaluación de riesgo de ruina.

Permite evaluar:
1. Reshuffling (Permutación sin reemplazo): Aísla el riesgo de secuencia (Sequence Risk).
2. Bootstrap con Reemplazo: Evalúa escenarios con rachas adversas extendidas.
3. Métricas de firmas de fondeo (Prop Firms):
   - Probabilidad de Ruina / Quiebra (tocar Max Trailing Drawdown o Daily Loss Limit).
   - Probabilidad de alcanzar el Profit Target antes de quebrar la cuenta.
   - Percentiles de Drawdown (P50, P95, Peor Caso) y Capital Final.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_erdos_renyi_expected_loss_streak(
    n_trades: int,
    loss_rate: float,
    gamma: float = 0.5772156649,
) -> float:
    """
    Fórmula de Erdős-Rényi corregida para la racha máxima esperada de pérdidas consecutivas:
    E[L_max] = (ln(N * (1 - q)) + gamma) / ln(1 / q) - 0.5
    donde:
        N = número total de operaciones
        q = tasa de pérdidas (loss rate = n_perdedoras / N)
        gamma = constante de Euler-Mascheroni (~0.577215)
    """
    if n_trades <= 1 or loss_rate <= 0.0 or loss_rate >= 1.0:
        return 0.0

    numerator = np.log(n_trades * (1.0 - loss_rate)) + gamma
    denominator = np.log(1.0 / loss_rate)
    if denominator <= 0:
        return 0.0

    expected_streak = (numerator / denominator) - 0.5
    return float(max(0.0, expected_streak))


class MonteCarloSimulator:
    """
    Motor cuantitativo de simulación de Monte Carlo para series de trades.
    """

    def __init__(
        self,
        trades_pnl: pd.Series,
        initial_capital: float = 50000.0,
        daily_loss_limit: float = 1000.0,
        max_trailing_drawdown: float = 2000.0,
        profit_target: float = 3000.0,
        trade_dates: Optional[pd.Series] = None,
        seed: Optional[int] = 42,
    ):
        """
        Args:
            trades_pnl: Serie o arreglo con el PnL neto de cada operación cerrada (en USD).
            initial_capital: Capital inicial de la cuenta (USD).
            daily_loss_limit: Límite de pérdida diaria permitido por la prop firm (USD).
            max_trailing_drawdown: Límite máximo de drawdown trailing permitido (USD).
            profit_target: Objetivo de beneficio contractual (USD).
            trade_dates: Serie de fechas de cada trade (opcional, para auditar DLL intradía).
            seed: Semilla para reproducibilidad estadística.
        """
        clean_pnl = pd.Series(trades_pnl).dropna().astype(float).values
        self.trades = clean_pnl
        self.initial_capital = float(initial_capital)
        self.dll = float(daily_loss_limit)
        self.mtdd = float(max_trailing_drawdown)
        self.target = float(profit_target)
        self.trade_dates = trade_dates
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def erdos_renyi_expected_loss_streak(self) -> float:
        """Calcula la racha máxima teórica de pérdidas según el teorema de Erdős-Rényi corregido."""
        if len(self.trades) == 0:
            return 0.0
        loss_rate = float((self.trades < 0).mean())
        return compute_erdos_renyi_expected_loss_streak(len(self.trades), loss_rate)

    def simulate_path(
        self,
        pnl_sequence: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Simula una única trayectoria de capital trade a trade evaluando
        violación de Max Trailing Drawdown y alcance de Profit Target.
        """
        equity = self.initial_capital
        peak_equity = self.initial_capital
        max_drawdown = 0.0

        hit_ruin = False
        hit_target = False
        first_event = None  # 'RUIN' o 'TARGET'

        n = len(pnl_sequence)
        equity_curve = np.empty(n + 1, dtype=float)
        equity_curve[0] = equity

        for i, trade in enumerate(pnl_sequence):
            equity += trade
            equity_curve[i + 1] = equity

            if equity > peak_equity:
                peak_equity = equity

            current_dd = peak_equity - equity
            if current_dd > max_drawdown:
                max_drawdown = current_dd

            # Evaluar quiebra por Trailing Drawdown
            if current_dd >= self.mtdd and not hit_ruin:
                hit_ruin = True
                if first_event is None:
                    first_event = "RUIN"

            # Evaluar meta de beneficio alcanzada (Target)
            if (equity - self.initial_capital) >= self.target and not hit_target:
                hit_target = True
                if first_event is None:
                    first_event = "TARGET"

        return {
            "final_equity": equity,
            "net_profit": equity - self.initial_capital,
            "max_drawdown": max_drawdown,
            "hit_ruin": hit_ruin,
            "hit_target": hit_target,
            "first_event": first_event,
            "equity_curve": equity_curve,
        }

    def run_reshuffle(
        self,
        n_simulations: int = 2500,
    ) -> Dict[str, Any]:
        """
        Ejecuta simulación de Monte Carlo mediante permutación sin reemplazo (Reshuffling).
        Evalúa el impacto estocástico de la secuencia de operaciones con el conjunto exacto de trades.
        """
        if len(self.trades) == 0:
            raise ValueError("No se proporcionaron operaciones válidas para la simulación de Monte Carlo.")

        if self.seed is not None:
            np.random.seed(self.seed)

        n_trades = len(self.trades)
        max_dds = np.empty(n_simulations, dtype=float)
        final_equities = np.empty(n_simulations, dtype=float)
        net_profits = np.empty(n_simulations, dtype=float)
        ruin_count = 0
        target_count = 0
        target_before_ruin_count = 0

        sample_curves = []

        for i in range(n_simulations):
            shuffled = np.random.permutation(self.trades)
            path_res = self.simulate_path(shuffled)

            max_dds[i] = path_res["max_drawdown"]
            final_equities[i] = path_res["final_equity"]
            net_profits[i] = path_res["net_profit"]

            if path_res["hit_ruin"]:
                ruin_count += 1
            if path_res["hit_target"]:
                target_count += 1
            if path_res["first_event"] == "TARGET":
                target_before_ruin_count += 1

            if i < 100:  # Guardar muestra de curvas para análisis visual / bandas
                sample_curves.append(path_res["equity_curve"])

        return self._format_results(
            method="Reshuffling (Permutación sin Reemplazo)",
            n_simulations=n_simulations,
            max_dds=max_dds,
            final_equities=final_equities,
            net_profits=net_profits,
            ruin_count=ruin_count,
            target_count=target_count,
            target_before_ruin_count=target_before_ruin_count,
            sample_curves=sample_curves,
        )

    def run_bootstrap(
        self,
        n_simulations: int = 2500,
        n_trades_per_sim: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta simulación de Monte Carlo mediante remuestreo con reemplazo (Bootstrapping).
        Permite evaluar escenarios donde rachas negativas se repiten con mayor frecuencia.
        """
        if len(self.trades) == 0:
            raise ValueError("No se proporcionaron operaciones válidas para la simulación de Monte Carlo.")

        if self.seed is not None:
            np.random.seed(self.seed)

        n_draws = n_trades_per_sim or len(self.trades)
        max_dds = np.empty(n_simulations, dtype=float)
        final_equities = np.empty(n_simulations, dtype=float)
        net_profits = np.empty(n_simulations, dtype=float)
        ruin_count = 0
        target_count = 0
        target_before_ruin_count = 0

        sample_curves = []

        for i in range(n_simulations):
            bootstrapped = np.random.choice(self.trades, size=n_draws, replace=True)
            path_res = self.simulate_path(bootstrapped)

            max_dds[i] = path_res["max_drawdown"]
            final_equities[i] = path_res["final_equity"]
            net_profits[i] = path_res["net_profit"]

            if path_res["hit_ruin"]:
                ruin_count += 1
            if path_res["hit_target"]:
                target_count += 1
            if path_res["first_event"] == "TARGET":
                target_before_ruin_count += 1

            if i < 100:
                sample_curves.append(path_res["equity_curve"])

        return self._format_results(
            method="Bootstrapping (Muestreo con Reemplazo)",
            n_simulations=n_simulations,
            max_dds=max_dds,
            final_equities=final_equities,
            net_profits=net_profits,
            ruin_count=ruin_count,
            target_count=target_count,
            target_before_ruin_count=target_before_ruin_count,
            sample_curves=sample_curves,
        )

    def _format_results(
        self,
        method: str,
        n_simulations: int,
        max_dds: np.ndarray,
        final_equities: np.ndarray,
        net_profits: np.ndarray,
        ruin_count: int,
        target_count: int,
        target_before_ruin_count: int,
        sample_curves: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Calcula estadísticos, percentiles y clasificaciones de riesgo."""
        ruin_prob = (ruin_count / n_simulations) * 100.0
        target_prob = (target_count / n_simulations) * 100.0
        target_before_ruin_prob = (target_before_ruin_count / n_simulations) * 100.0

        p5_dd = np.percentile(max_dds, 50)
        p95_dd = np.percentile(max_dds, 95)
        p99_dd = np.percentile(max_dds, 99)
        worst_dd = np.max(max_dds)

        p5_equity = np.percentile(final_equities, 5)
        p50_equity = np.percentile(final_equities, 50)
        p95_equity = np.percentile(final_equities, 95)

        # Diagnóstico cualitativo para Prop Trading
        if ruin_prob == 0.0 and target_before_ruin_prob >= 70.0:
            verdict = "🟢 EXCELENTE (Cero Ruina en Simulación / Alta Probabilidad de Aprobación)"
        elif ruin_prob < 5.0 and target_before_ruin_prob >= 50.0:
            verdict = "🟢 ROBUSTA (Riesgo de Ruina Bajo < 5% / Viable para Fondeo)"
        elif ruin_prob < 15.0:
            verdict = "🟡 MODERADA (Riesgo de Ruina Aceptable pero Requiere Gestión Estricta)"
        else:
            verdict = "🔴 INVIABLE (Alto Riesgo de Ruina / Probabilidad Excesiva de Quemar Cuenta)"

        return {
            "method": method,
            "simulations": n_simulations,
            "total_trades_analyzed": len(self.trades),
            "initial_capital": self.initial_capital,
            "mtdd_limit": self.mtdd,
            "profit_target": self.target,
            "risk_of_ruin_pct": round(ruin_prob, 2),
            "target_reach_pct": round(target_prob, 2),
            "target_before_ruin_pct": round(target_before_ruin_prob, 2),
            "drawdown_stats": {
                "median_dd_usd": round(float(p5_dd), 2),
                "p95_dd_usd": round(float(p95_dd), 2),
                "p99_dd_usd": round(float(p99_dd), 2),
                "worst_case_dd_usd": round(float(worst_dd), 2),
                "cvar_95_dd_usd": round(float(np.mean(max_dds[max_dds >= p95_dd])), 2),
            },
            "equity_stats": {
                "p5_equity_usd": round(float(p5_equity), 2),
                "p50_median_equity_usd": round(float(p50_equity), 2),
                "p95_equity_usd": round(float(p95_equity), 2),
                "mean_profit_usd": round(float(np.mean(net_profits)), 2),
            },
            "erdos_renyi_expected_loss_streak": round(self.erdos_renyi_expected_loss_streak(), 2),
            "verdict": verdict,
            "sample_curves": sample_curves,
        }

    def run_session_block_bootstrap(
        self,
        n_simulations: int = 2500,
        trades_df: Optional[pd.DataFrame] = None,
        trade_dates: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta simulación de Monte Carlo mediante Session Block Bootstrap (remuestreo de sesiones completas).
        Agrupa operaciones por fecha de sesión evaluando el Daily Loss Limit (DLL) intradía además del MTDD y Profit Target.
        """
        return run_session_block_bootstrap(
            trades_df=trades_df,
            trades_pnl=pd.Series(self.trades),
            trade_dates=trade_dates if trade_dates is not None else self.trade_dates,
            initial_capital=self.initial_capital,
            daily_loss_limit=self.dll,
            max_trailing_drawdown=self.mtdd,
            profit_target=self.target,
            n_simulations=n_simulations,
            seed=self.seed,
        )


def run_session_block_bootstrap(
    trades_df: Optional[pd.DataFrame] = None,
    trades_pnl: Optional[pd.Series] = None,
    trade_dates: Optional[pd.Series] = None,
    initial_capital: float = 50000.0,
    daily_loss_limit: float = 1000.0,
    max_trailing_drawdown: float = 2000.0,
    profit_target: float = 3000.0,
    n_simulations: int = 2500,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Función de primer nivel para ejecutar Session Block Bootstrap.
    Agrupa operaciones por fecha de sesión (utilizando pd.to_datetime(series, utc=True) para evitar problemas de DST/zonas horarias),
    remuestrea sesiones completas con reemplazo y audita tanto la quiebra por MTDD como por DLL intradía.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Extraer PnL y Fechas
    clean_pnl_list: List[float] = []
    parsed_dates_list: Optional[List[Any]] = None

    if trades_df is not None and not trades_df.empty:
        # Detectar columna de fecha/hora de forma robusta
        dt_series = None
        if "exit_date" in trades_df.columns:
            tc = "exit_time" if "exit_time" in trades_df.columns else "entry_time"
            if tc in trades_df.columns:
                dt_series = pd.to_datetime(trades_df["exit_date"].astype(str) + " " + trades_df[tc].astype(str), utc=True)
            else:
                dt_series = pd.to_datetime(trades_df["exit_date"], utc=True)
        elif "entry_date" in trades_df.columns:
            tc = "entry_time" if "entry_time" in trades_df.columns else "exit_time"
            if tc in trades_df.columns:
                dt_series = pd.to_datetime(trades_df["entry_date"].astype(str) + " " + trades_df[tc].astype(str), utc=True)
            else:
                dt_series = pd.to_datetime(trades_df["entry_date"], utc=True)
        else:
            for col in ["exit_time", "entry_time", "date", "Date", "timestamp", "Timestamp"]:
                if col in trades_df.columns:
                    sample_str = str(trades_df[col].dropna().iloc[0]) if len(trades_df[col].dropna()) > 0 else ""
                    if "-" in sample_str or "/" in sample_str:
                        dt_series = pd.to_datetime(trades_df[col], utc=True)
        if dt_series is None and isinstance(trades_df.index, pd.DatetimeIndex):
            dt_series = pd.Series(trades_df.index, index=trades_df.index)

        # Detectar columna de PnL
        pnl_col = None
        for col in ["net_pnl_usd", "net_pnl", "net_profit", "pnl", "PnL", "Net_PnL"]:
            if col in trades_df.columns:
                pnl_col = col
                break

        if pnl_col is None:
            raise ValueError(f"No se encontró columna de PnL válida en el DataFrame. Columnas: {list(trades_df.columns)}")

        valid_df = trades_df.dropna(subset=[pnl_col]).copy()
        clean_pnl_list = valid_df[pnl_col].astype(float).tolist()

        if dt_series is not None:
            parsed_dates_list = pd.to_datetime(dt_series.loc[valid_df.index], utc=True).dt.date.tolist()

    elif trades_pnl is not None:
        pnl_s = pd.Series(trades_pnl).dropna().astype(float)
        clean_pnl_list = pnl_s.tolist()
        if trade_dates is not None:
            valid_dates = pd.Series(trade_dates).loc[pnl_s.index]
            parsed_dates_list = pd.to_datetime(valid_dates, utc=True).dt.date.tolist()
        elif isinstance(trades_pnl.index, pd.DatetimeIndex):
            parsed_dates_list = pd.to_datetime(pnl_s.index, utc=True).date.tolist()

    if len(clean_pnl_list) == 0:
        raise ValueError("No se proporcionaron operaciones válidas para Session Block Bootstrap.")

    # 2. Agrupar trades por bloques de sesión
    session_blocks: List[np.ndarray] = []
    if parsed_dates_list is not None and len(parsed_dates_list) == len(clean_pnl_list):
        grouped: Dict[Any, List[float]] = {}
        for d, p in zip(parsed_dates_list, clean_pnl_list):
            if d not in grouped:
                grouped[d] = []
            grouped[d].append(p)
        session_blocks = [np.array(v, dtype=float) for v in grouped.values()]
    else:
        # Fallback: Cada trade como bloque individual si no hay fechas
        session_blocks = [np.array([p], dtype=float) for p in clean_pnl_list]

    n_sessions = len(session_blocks)
    max_dds = np.empty(n_simulations, dtype=float)
    worst_daily_losses = np.empty(n_simulations, dtype=float)
    final_equities = np.empty(n_simulations, dtype=float)
    net_profits = np.empty(n_simulations, dtype=float)

    ruin_count = 0
    dll_breach_count = 0
    mtdd_breach_count = 0
    target_count = 0
    target_before_ruin_count = 0
    sample_curves: List[np.ndarray] = []

    for i in range(n_simulations):
        sampled_indices = np.random.choice(n_sessions, size=n_sessions, replace=True)
        equity = float(initial_capital)
        peak_equity = float(initial_capital)
        max_dd = 0.0
        worst_day = 0.0

        hit_ruin_mtdd = False
        hit_dll = False
        hit_target = False
        first_event = None

        curve = [equity]

        for s_idx in sampled_indices:
            trades_in_day = session_blocks[s_idx]
            day_pnl = 0.0

            for trade in trades_in_day:
                equity += trade
                day_pnl += trade
                curve.append(equity)

                if equity > peak_equity:
                    peak_equity = equity

                current_dd = peak_equity - equity
                if current_dd > max_dd:
                    max_dd = current_dd

                # Evaluar MTDD
                if current_dd >= max_trailing_drawdown and not hit_ruin_mtdd:
                    hit_ruin_mtdd = True
                    if first_event is None:
                        first_event = "RUIN_MTDD"

                # Evaluar DLL intradía
                if day_pnl <= -daily_loss_limit and not hit_dll:
                    hit_dll = True
                    if first_event is None:
                        first_event = "RUIN_DLL"

                # Registrar la máxima pérdida intradía alcanzada en la sesión
                if day_pnl < worst_day:
                    worst_day = day_pnl

                # Evaluar Target
                if (equity - initial_capital) >= profit_target and not hit_target:
                    hit_target = True
                    if first_event is None:
                        first_event = "TARGET"

        max_dds[i] = max_dd
        worst_daily_losses[i] = abs(worst_day)
        final_equities[i] = equity
        net_profits[i] = equity - initial_capital

        hit_ruin = hit_ruin_mtdd or hit_dll
        if hit_ruin:
            ruin_count += 1
        if hit_dll:
            dll_breach_count += 1
        if hit_ruin_mtdd:
            mtdd_breach_count += 1
        if hit_target:
            target_count += 1
        if first_event == "TARGET":
            target_before_ruin_count += 1

        if i < 100:
            sample_curves.append(np.array(curve, dtype=float))

    ruin_prob = (ruin_count / n_simulations) * 100.0
    dll_prob = (dll_breach_count / n_simulations) * 100.0
    mtdd_prob = (mtdd_breach_count / n_simulations) * 100.0
    target_prob = (target_count / n_simulations) * 100.0
    target_before_ruin_prob = (target_before_ruin_count / n_simulations) * 100.0

    p50_dd = np.percentile(max_dds, 50)
    p95_dd = np.percentile(max_dds, 95)
    p99_dd = np.percentile(max_dds, 99)
    worst_dd = np.max(max_dds)

    p50_worst_day = np.percentile(worst_daily_losses, 50)
    p95_worst_day = np.percentile(worst_daily_losses, 95)
    worst_day_all = np.max(worst_daily_losses)

    p5_equity = np.percentile(final_equities, 5)
    p50_equity = np.percentile(final_equities, 50)
    p95_equity = np.percentile(final_equities, 95)

    loss_rate = float((np.array(clean_pnl_list) < 0).mean())
    erdos_renyi_val = compute_erdos_renyi_expected_loss_streak(len(clean_pnl_list), loss_rate)

    if dll_prob > 0.0:
        verdict = f"🔴 INVIABLE (Violación de Daily Loss Limit en {dll_prob:.2f}% de las simulaciones)"
    elif ruin_prob == 0.0 and target_before_ruin_prob >= 70.0:
        verdict = "🟢 EXCELENTE (Cero Ruina MTDD/DLL / Alta Probabilidad de Aprobación)"
    elif ruin_prob < 5.0 and target_before_ruin_prob >= 50.0:
        verdict = "🟢 ROBUSTA (Riesgo Bajo < 5% / Cero Violación DLL / Viable para Fondeo)"
    elif ruin_prob < 15.0:
        verdict = "🟡 MODERADA (Riesgo Aceptable pero Requiere Gestión Estricta)"
    else:
        verdict = "🔴 INVIABLE (Alto Riesgo de Ruina en Simulación)"

    return {
        "method": "Session Block Bootstrap (Remuestreo de Sesiones Diarias)",
        "simulations": n_simulations,
        "total_sessions": n_sessions,
        "total_trades_analyzed": len(clean_pnl_list),
        "initial_capital": initial_capital,
        "mtdd_limit": max_trailing_drawdown,
        "daily_loss_limit": daily_loss_limit,
        "profit_target": profit_target,
        "risk_of_ruin_pct": round(ruin_prob, 2),
        "dll_violation_pct": round(dll_prob, 2),
        "mtdd_violation_pct": round(mtdd_prob, 2),
        "target_reach_pct": round(target_prob, 2),
        "target_before_ruin_pct": round(target_before_ruin_prob, 2),
        "drawdown_stats": {
            "median_dd_usd": round(float(p50_dd), 2),
            "p95_dd_usd": round(float(p95_dd), 2),
            "p99_dd_usd": round(float(p99_dd), 2),
            "worst_case_dd_usd": round(float(worst_dd), 2),
            "cvar_95_dd_usd": round(float(np.mean(max_dds[max_dds >= p95_dd])), 2),
        },
        "daily_loss_stats": {
            "median_worst_day_usd": round(float(p50_worst_day), 2),
            "p95_worst_day_usd": round(float(p95_worst_day), 2),
            "worst_day_usd": round(float(worst_day_all), 2),
        },
        "equity_stats": {
            "p5_equity_usd": round(float(p5_equity), 2),
            "p50_median_equity_usd": round(float(p50_equity), 2),
            "p95_equity_usd": round(float(p95_equity), 2),
            "mean_profit_usd": round(float(np.mean(net_profits)), 2),
        },
        "erdos_renyi_expected_loss_streak": round(erdos_renyi_val, 2),
        "verdict": verdict,
        "sample_curves": sample_curves,
    }
