"""
src/core/robustness.py
Análisis estadístico de robustez cuantitativa:
1. Filtro y detector de Factor Suerte (pocas operaciones y concentración de beneficios).
2. Detector de Mesetas de Estabilidad de Parámetros frente a picos sobreajustados (curve-fitting).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def evaluate_luck_and_significance(
    trades_pnl: pd.Series,
    sharpe_ratio: float,
    min_trades: int = 30
) -> Dict[str, Any]:
    """
    Evalúa si el rendimiento de la estrategia es estadísticamente significativo
    o si presenta indicios de ser producto del azar o suerte (pocas operaciones / concentración).
    
    Args:
        trades_pnl: Serie con los rendimientos o PnL de cada operación cerrada.
        sharpe_ratio: Ratio de Sharpe anualizado de la estrategia.
        min_trades: Umbral mínimo de operaciones requeridas para validez estadística.
        
    Returns:
        Diccionario con métricas de significancia, alertas de suerte y nivel de confianza.
    """
    n_trades = len(trades_pnl)
    if n_trades == 0:
        return {
            "Total Trades": 0,
            "Alerta de Suerte": "🔴 CRÍTICA: Sin operaciones registradas",
            "Significancia Estadística": "Invalorable",
            "Concentración Top 3 Trades": "0.0%",
            "Confiable": False
        }

    # 1. Alerta por escasez de operaciones
    few_trades_warning = n_trades < min_trades

    # 2. Concentración de ganancias (¿El 80% de la ganancia vino de 1 o 2 trades?)
    positive_trades = trades_pnl[trades_pnl > 0]
    total_gains = positive_trades.sum()

    top1_pct = 0.0
    top3_pct = 0.0
    if total_gains > 0 and len(positive_trades) > 0:
        sorted_gains = positive_trades.sort_values(ascending=False)
        top1_pct = (sorted_gains.iloc[0] / total_gains) * 100
        top3_pct = (sorted_gains.iloc[:3].sum() / total_gains) * 100

    high_concentration = top3_pct > 50.0  # Alerta si los 3 mejores trades aportan más de la mitad

    # 3. Test de Significancia Estadística del Sharpe Ratio (t-test)
    # H0: Sharpe = 0 (pura suerte) vs H1: Sharpe > 0
    # Error estándar asintótico: SE = 1 / sqrt(N)
    if n_trades > 1:
        t_stat = sharpe_ratio * np.sqrt(n_trades / 252.0)
        # p-value unilateral
        p_val = 1.0 - scipy_stats.norm.cdf(t_stat)
    else:
        t_stat = 0.0
        p_val = 1.0

    statistically_significant = (p_val < 0.05) and not few_trades_warning

    # Clasificación visual
    alerts = []
    if few_trades_warning:
        alerts.append(f"⚠️ Operaciones insuficientes (N={n_trades} < {min_trades}). Alto riesgo de suerte.")
    if high_concentration:
        alerts.append(f"⚠️ Ganancia concentrada: Los mejores 3 trades aportaron el {top3_pct:.1f}% del total.")

    status_str = "🟢 ESTADÍSTICAMENTE SIGNIFICATIVA" if (statistically_significant and not high_concentration) else (
        "🟡 SOSPECHOSA (POSIBLE FACTOR SUERTE)" if few_trades_warning or high_concentration else "🔴 NO SIGNIFICATIVA (RUIDO)"
    )

    return {
        "Total Trades": n_trades,
        "Umbral Requerido": min_trades,
        "Veredicto": status_str,
        "Top 1 Trade (% Ganancia)": f"{top1_pct:.1f}%",
        "Top 3 Trades (% Ganancia)": f"{top3_pct:.1f}%",
        "t-statistic": round(t_stat, 2),
        "p-value": round(p_val, 4),
        "Nivel de Confianza": f"{(1.0 - p_val) * 100:.1f}%",
        "Alertas": alerts if alerts else ["Ninguna: Operaciones estadísticamente distribuidas"]
    }


def evaluate_parameter_stability(
    opt_results: pd.DataFrame,
    best_params: Dict[str, Any],
    param_keys: List[str]
) -> Dict[str, Any]:
    """
    Analiza si los parámetros óptimos pertenecen a una 'Meseta Robusta'
    o a un 'Pico Aislado' (sobreajuste / curve-fitting).
    
    Args:
        opt_results: DataFrame con las columnas de parámetros y métricas ('Sharpe Ratio', 'Return [%]').
        best_params: Diccionario con la combinación óptima seleccionada.
        param_keys: Lista con los nombres de los parámetros optimizados (ej. ['n_fast', 'n_slow']).
        
    Returns:
        Diccionario con tasa de retención vecinal, coeficiente de variación y diagnóstico.
    """
    if opt_results.empty or not param_keys:
        return {"Estado": "Sin datos de optimización"}

    # Filtrar vecinos en la rejilla de parámetros (distancia manhattan o diferencia de índice)
    # Consideramos vecino si cada parámetro difiere en no más de 1 o 2 pasos del valor óptimo
    df = opt_results.copy()
    
    is_neighbor = pd.Series(True, index=df.index)
    for key in param_keys:
        if key in df.columns and key in best_params:
            best_val = best_params[key]
            unique_sorted = sorted(df[key].unique())
            if best_val in unique_sorted:
                idx = unique_sorted.index(best_val)
                min_idx = max(0, idx - 1)
                max_idx = min(len(unique_sorted) - 1, idx + 1)
                allowed_vals = unique_sorted[min_idx:max_idx + 1]
                is_neighbor = is_neighbor & df[key].isin(allowed_vals)

    neighbors_df = df[is_neighbor]
    n_neighbors = len(neighbors_df)

    if n_neighbors <= 1:
        return {
            "Estado": "🟡 Insuficientes vecinos para evaluar estabilidad",
            "Vecinos Analizados": n_neighbors
        }

    # Evaluar métricas de los vecinos
    sharpe_col = "Sharpe Ratio" if "Sharpe Ratio" in neighbors_df.columns else None
    return_col = "Return [%]" if "Return [%]" in neighbors_df.columns else None

    if sharpe_col:
        neighbor_sharpes = neighbors_df[sharpe_col].dropna()
        positive_neighbors_pct = (neighbor_sharpes > 0).mean() * 100
        mean_sharpe = neighbor_sharpes.mean()
        std_sharpe = neighbor_sharpes.std()
        cv_sharpe = (std_sharpe / abs(mean_sharpe)) if abs(mean_sharpe) > 0 else float("inf")
    else:
        positive_neighbors_pct = 100.0
        cv_sharpe = 0.0

    # Clasificación de estabilidad
    # Es meseta si al menos el 75% de los vecinos tienen Sharpe positivo y la dispersión no es desmedida
    is_plateau = (positive_neighbors_pct >= 75.0) and (cv_sharpe < 1.0)

    verdict = "🟢 MESETA ESTABLE (Robusta)" if is_plateau else "🔴 PICO AISLADO (Inestable / Sobreajuste)"

    return {
        "Veredicto Estabilidad": verdict,
        "Vecinos Analizados": n_neighbors,
        "Vecinos con Sharpe Positivo (%)": f"{positive_neighbors_pct:.1f}%",
        "Dispersión Vecinal (CV)": round(cv_sharpe, 2),
        "Sharpe Promedio Vecindario": round(mean_sharpe, 2) if sharpe_col else "N/A"
    }


def compute_deflated_sharpe_ratio(
    trades_or_returns: Union[pd.Series, np.ndarray, list],
    num_trials: int = 1,
    sharpe_std: Optional[float] = None,
    annualized_sharpe: Optional[float] = None,
    trades_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Calcula el Deflated Sharpe Ratio (DSR) de Bailey & López de Prado (2014)
    para ajustar el Sharpe Ratio frente a:
    1. No-normalidad (asimetría / skewness y curtosis / fat tails).
    2. Sesgo de selección por múltiples pruebas / optimización de parámetros (K ensayos).
    
    Args:
        trades_or_returns: Serie o arreglo de rendimientos o PnL por trade/período.
        num_trials: Número total de combinaciones de parámetros evaluadas (K).
        sharpe_std: Desviación estándar de los Sharpe ratios de los K ensayos (opcional).
        annualized_sharpe: Sharpe ratio anualizado precalculado (opcional).
        trades_per_year: Frecuencia estimada de operaciones por año (def: 252).
        risk_free_rate: Tasa libre de riesgo por período (def: 0.0).
        
    Returns:
        Diccionario con DSR, p-value, estadísticos de orden superior y diagnóstico.
    """
    clean_series = pd.Series(trades_or_returns).dropna().astype(float).values
    n = len(clean_series)
    if n < 3:
        return {
            "dsr": 0.0,
            "p_value": 1.0,
            "observed_sr": 0.0,
            "observed_sr_annualized": 0.0,
            "benchmark_sr": 0.0,
            "benchmark_sr_annualized": 0.0,
            "num_trials": num_trials,
            "skewness": 0.0,
            "kurtosis": 3.0,
            "se_sharpe": 1.0,
            "statistically_significant": False,
            "verdict": "🔴 DATOS INSUFICIENTES (N < 3)"
        }

    mean_ret = float(np.mean(clean_series))
    std_ret = float(np.std(clean_series, ddof=1))
    sr = (mean_ret - risk_free_rate) / std_ret if std_ret > 1e-12 else 0.0

    # Momentos de distribución
    skewness = float(scipy_stats.skew(clean_series))
    # Pearson kurtosis (normal = 3.0)
    kurtosis = float(scipy_stats.kurtosis(clean_series, fisher=False))

    # Error estándar asintótico bajo no-normalidad (Mertens 2002 / Bailey & López de Prado 2014)
    # var(SR) = (1 - gamma_3 * SR + (gamma_4 - 1)/4 * SR^2) / (N - 1)
    variance_term = (1.0 - skewness * sr + ((kurtosis - 1.0) / 4.0) * (sr ** 2)) / (n - 1.0)
    se_sr = float(np.sqrt(max(1e-12, variance_term)))

    # Benchmark SR* penalizado por selección múltiple de K ensayos
    k = max(1, int(num_trials))
    if k <= 1:
        sr_benchmark = 0.0
    else:
        gamma_euler = 0.5772156649
        e_const = np.e
        sigma_k = sharpe_std if (sharpe_std is not None and sharpe_std > 0) else se_sr
        term1 = (1.0 - gamma_euler) * scipy_stats.norm.ppf(1.0 - 1.0 / k)
        term2 = gamma_euler * scipy_stats.norm.ppf(1.0 - 1.0 / (k * e_const))
        sr_benchmark = float(sigma_k * (term1 + term2))

    # Estadístico Z de Deflated Sharpe Ratio
    z_stat = (sr - sr_benchmark) / se_sr if se_sr > 1e-12 else 0.0
    dsr = float(scipy_stats.norm.cdf(z_stat))
    p_val = float(max(0.0, 1.0 - dsr))

    # Anualización
    scale_ann = np.sqrt(trades_per_year)
    sr_ann = annualized_sharpe if annualized_sharpe is not None else (sr * scale_ann)
    sr_benchmark_ann = sr_benchmark * scale_ann

    is_significant = bool(dsr >= 0.95)
    if dsr >= 0.95:
        verdict = "🟢 ROBUSTO (DSR >= 0.95, p < 0.05 - Sin evidencia de minería de datos)"
    elif dsr >= 0.85:
        verdict = "🟡 MODERADO (0.85 <= DSR < 0.95 - Sobreajuste leve o muestra moderada)"
    else:
        verdict = "🔴 VULNERABLE (DSR < 0.85 - Alto riesgo de sobreajuste / falsos descubrimientos)"

    return {
        "dsr": round(dsr, 4),
        "p_value": round(p_val, 4),
        "observed_sr": round(sr, 4),
        "observed_sr_annualized": round(sr_ann, 2),
        "benchmark_sr": round(sr_benchmark, 4),
        "benchmark_sr_annualized": round(sr_benchmark_ann, 2),
        "num_trials": k,
        "skewness": round(skewness, 3),
        "kurtosis": round(kurtosis, 3),
        "se_sharpe": round(se_sr, 4),
        "statistically_significant": is_significant,
        "verdict": verdict,
    }


def _calc_pnl_metrics(
    pnl_series: pd.Series,
    initial_capital: float = 50000.0,
    mtdd: float = 2000.0,
    dll: float = 1000.0,
    dates_series: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Función auxiliar para calcular métricas completas de trading y prop firm sobre una serie de PnL."""
    clean_pnl = pnl_series.dropna().astype(float)
    if len(clean_pnl) == 0:
        return {
            "net_profit_usd": 0.0,
            "profit_factor": 0.0,
            "win_rate_pct": 0.0,
            "total_trades": 0,
            "max_drawdown_usd": 0.0,
            "worst_daily_loss_usd": 0.0,
            "dll_violated": False,
            "mtdd_violated": False,
        }

    net_profit = float(clean_pnl.sum())
    wins = clean_pnl[clean_pnl > 0]
    losses = clean_pnl[clean_pnl < 0]
    gross_win = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    pf = (gross_win / gross_loss) if gross_loss > 1e-6 else (10.0 if gross_win > 0 else 0.0)
    win_rate = (len(wins) / len(clean_pnl)) * 100.0

    # Drawdown trade a trade
    equity = float(initial_capital) + clean_pnl.cumsum().values
    peak = np.maximum.accumulate(equity)
    dd_curve = peak - equity
    max_dd = float(np.max(dd_curve)) if len(dd_curve) > 0 else 0.0

    # Pérdida diaria intradía
    worst_day_loss = 0.0
    if dates_series is not None and len(dates_series) == len(clean_pnl):
        valid_dates = pd.to_datetime(dates_series, utc=True).dt.date
        # Calcular trayectoria acumulada intradía por sesión para detectar el peor valle
        intraday_cum = clean_pnl.groupby(valid_dates).cumsum()
        min_intraday = float(intraday_cum.min()) if len(intraday_cum) > 0 else 0.0
        if min_intraday < 0:
            worst_day_loss = abs(min_intraday)
    else:
        worst_loss = float(clean_pnl.min())
        if worst_loss < 0:
            worst_day_loss = abs(worst_loss)

    dll_breached = bool(worst_day_loss >= dll)
    mtdd_breached = bool(max_dd >= mtdd)

    return {
        "net_profit_usd": round(net_profit, 2),
        "profit_factor": round(pf, 2),
        "win_rate_pct": round(win_rate, 2),
        "total_trades": len(clean_pnl),
        "max_drawdown_usd": round(max_dd, 2),
        "worst_daily_loss_usd": round(worst_day_loss, 2),
        "dll_violated": dll_breached,
        "mtdd_violated": mtdd_breached,
    }


def run_what_if_stress_suite(
    trades_df: pd.DataFrame,
    symbol: str = "ES",
    contract_type: Optional[str] = None,
    initial_capital: float = 50000.0,
    mtdd: float = 2000.0,
    dll: float = 1000.0,
    n_prune_simulations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Ejecuta una batería de pruebas de estrés hostiles 'What-If':
    1. Slippage Shock 2x (+1 tick/lado = 2 ticks RT/contrato).
    2. Slippage Shock 3x (+2 ticks/lado = 4 ticks RT/contrato).
    3. Eliminación de los 3 mejores trades (Top-3 Trades Drop).
    4. Random Win-Pruning (elimina 10% de ganadores al azar en 1,000 iteraciones).
    5. Gap Adverso en Stop Loss (+2 ticks adicionales en trades con SL).
    
    Tick values CME:
        ES: $12.50 | MES: $1.25 | NQ: $5.00 | MNQ: $0.50
    """
    np.random.seed(seed)
    df = trades_df.copy()

    # Detectar columna de PnL
    pnl_col = None
    for col in ["net_pnl_usd", "net_pnl", "net_profit", "pnl", "PnL", "Net_PnL"]:
        if col in df.columns:
            pnl_col = col
            break

    if pnl_col is None:
        raise ValueError(f"No se encontró columna de PnL en trades_df. Columnas: {list(df.columns)}")

    df = df.dropna(subset=[pnl_col]).copy()
    original_pnl = df[pnl_col].astype(float)

    # Detectar fechas para auditoría de Daily Loss Limit de forma robusta
    dates_series = None
    if "exit_date" in df.columns:
        tc = "exit_time" if "exit_time" in df.columns else "entry_time"
        if tc in df.columns:
            dates_series = pd.to_datetime(df["exit_date"].astype(str) + " " + df[tc].astype(str), utc=True)
        else:
            dates_series = pd.to_datetime(df["exit_date"], utc=True)
    elif "entry_date" in df.columns:
        tc = "entry_time" if "entry_time" in df.columns else "exit_time"
        if tc in df.columns:
            dates_series = pd.to_datetime(df["entry_date"].astype(str) + " " + df[tc].astype(str), utc=True)
        else:
            dates_series = pd.to_datetime(df["entry_date"], utc=True)
    else:
        for col in ["exit_time", "entry_time", "date", "Date", "timestamp", "Timestamp"]:
            if col in df.columns:
                sample_str = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else ""
                if "-" in sample_str or "/" in sample_str:
                    dates_series = pd.to_datetime(df[col], utc=True)
                    break

    if dates_series is None and isinstance(df.index, pd.DatetimeIndex):
        dates_series = pd.Series(df.index, index=df.index)

    # Calcular costos por tick por contrato / pata
    sym = symbol.upper()
    cme_tick_values = {"ES": 12.50, "MES": 1.25, "NQ": 5.00, "MNQ": 0.50}

    # Comprobar si es spread multi-pata (ej. F4 con contracts_nq y contracts_es)
    if "contracts_nq" in df.columns and "contracts_es" in df.columns:
        c_nq = df["contracts_nq"].fillna(0).astype(float)
        c_es = df["contracts_es"].fillna(0).astype(float)
        # MNQ = $0.50, MES = $1.25
        tick_cost_per_side = c_nq * 0.50 + c_es * 1.25
    else:
        contracts = df["contracts"].fillna(1.0).astype(float) if "contracts" in df.columns else pd.Series(1.0, index=df.index)
        if "contract_type" in df.columns:
            is_micro = df["contract_type"].astype(str).str.upper() == "MICRO"
            micro_tv = 1.25 if "ES" in sym else 0.50
            std_tv = 12.50 if "ES" in sym else 5.00
            tick_cost_per_side = contracts * np.where(is_micro, micro_tv, std_tv)
        elif "is_micro" in df.columns:
            is_micro = df["is_micro"].astype(bool)
            micro_tv = 1.25 if "ES" in sym else 0.50
            std_tv = 12.50 if "ES" in sym else 5.00
            tick_cost_per_side = contracts * np.where(is_micro, micro_tv, std_tv)
        elif contract_type is not None and contract_type.upper() in ["MICRO", "MES", "MNQ"]:
            tv = 1.25 if "ES" in sym else 0.50
            tick_cost_per_side = contracts * tv
        else:
            tv = cme_tick_values.get(sym, 12.50)
            tick_cost_per_side = contracts * tv

    round_trip_tick_cost = 2.0 * tick_cost_per_side

    # 0. Escenario Base
    base_metrics = _calc_pnl_metrics(original_pnl, initial_capital, mtdd, dll, dates_series)

    # 1. Slippage Shock 2x (+1 tick/lado = 2 ticks RT adicionales)
    pnl_slip_2x = original_pnl - round_trip_tick_cost * 1.0
    slip_2x_metrics = _calc_pnl_metrics(pnl_slip_2x, initial_capital, mtdd, dll, dates_series)

    # 2. Slippage Shock 3x (+2 ticks/lado = 4 ticks RT adicionales)
    pnl_slip_3x = original_pnl - round_trip_tick_cost * 2.0
    slip_3x_metrics = _calc_pnl_metrics(pnl_slip_3x, initial_capital, mtdd, dll, dates_series)

    # 3. Top-3 Trades Drop
    if len(original_pnl) > 3:
        top3_indices = original_pnl.nlargest(3).index
        pnl_no_top3 = original_pnl.drop(top3_indices)
        dates_no_top3 = dates_series.drop(top3_indices) if dates_series is not None else None
        top3_drop_metrics = _calc_pnl_metrics(pnl_no_top3, initial_capital, mtdd, dll, dates_no_top3)
    else:
        top3_drop_metrics = base_metrics

    # 4. Random Win-Pruning (eliminar 10% de ganadores al azar en 1,000 iteraciones)
    win_indices = original_pnl[original_pnl > 0].index.tolist()
    n_wins = len(win_indices)
    k_prune = max(1, int(round(0.10 * n_wins))) if n_wins > 0 else 0

    pruned_profits = np.empty(n_prune_simulations, dtype=float)
    pruned_pfs = np.empty(n_prune_simulations, dtype=float)
    pruned_dds = np.empty(n_prune_simulations, dtype=float)

    if n_wins > 0 and k_prune > 0:
        for it in range(n_prune_simulations):
            drop_idx = np.random.choice(win_indices, size=k_prune, replace=False)
            pnl_iter = original_pnl.drop(drop_idx)
            m = _calc_pnl_metrics(pnl_iter, initial_capital, mtdd, dll, None)
            pruned_profits[it] = m["net_profit_usd"]
            pruned_pfs[it] = m["profit_factor"]
            pruned_dds[it] = m["max_drawdown_usd"]
    else:
        pruned_profits.fill(base_metrics["net_profit_usd"])
        pruned_pfs.fill(base_metrics["profit_factor"])
        pruned_dds.fill(base_metrics["max_drawdown_usd"])

    win_pruning_metrics = {
        "pruned_trades_per_iter": k_prune,
        "p5_profit_usd": round(float(np.percentile(pruned_profits, 5)), 2),
        "median_profit_usd": round(float(np.percentile(pruned_profits, 50)), 2),
        "p95_profit_usd": round(float(np.percentile(pruned_profits, 95)), 2),
        "p5_profit_factor": round(float(np.percentile(pruned_pfs, 5)), 2),
        "median_profit_factor": round(float(np.percentile(pruned_pfs, 50)), 2),
        "p95_profit_factor": round(float(np.percentile(pruned_pfs, 95)), 2),
        "p5_max_dd_usd": round(float(np.percentile(pruned_dds, 5)), 2),
        "median_max_dd_usd": round(float(np.percentile(pruned_dds, 50)), 2),
        "worst_max_dd_usd": round(float(np.max(pruned_dds)), 2),
        "profitable_iterations_pct": round(float((pruned_pfs >= 1.0).mean() * 100.0), 2),
    }

    # 5. Gap Adverso en Stop Loss (+2 ticks adicionales en salidas por SL)
    if "exit_reason" in df.columns:
        is_sl = df["exit_reason"].astype(str).str.upper().str.contains("STOP|SL")
    else:
        is_sl = original_pnl < 0

    gap_cost = np.where(is_sl, 2.0 * tick_cost_per_side, 0.0)
    pnl_gap = original_pnl - gap_cost
    gap_sl_metrics = _calc_pnl_metrics(pnl_gap, initial_capital, mtdd, dll, dates_series)

    return {
        "baseline": base_metrics,
        "slippage_2x": slip_2x_metrics,
        "slippage_3x": slip_3x_metrics,
        "top3_drop": top3_drop_metrics,
        "win_pruning": win_pruning_metrics,
        "gap_adverse_sl": gap_sl_metrics,
    }


class PropFirmRobustnessScorecard:
    """
    Sistema de Calificación Cuantitativa de 100 Puntos y 6 Compuertas Duras (Hard Gates)
    para auditar estrategias de trading con destino a cuentas de evaluación (Apex, Topstep).
    """

    def __init__(
        self,
        capital: float = 50000.0,
        mtdd: float = 2000.0,
        dll: float = 1000.0,
        target: float = 3000.0,
    ):
        self.capital = float(capital)
        self.mtdd = float(mtdd)
        self.dll = float(dll)
        self.target = float(target)

    def evaluate(
        self,
        monte_carlo_res: Dict[str, Any],
        what_if_res: Dict[str, Any],
        parameter_res: Optional[Dict[str, Any]] = None,
        walk_forward_res: Optional[Dict[str, Any]] = None,
        dsr_res: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calcula la puntuación desglosada por pilar y verifica las 6 Compuertas Duras.
        Soporta recalibración de ponderaciones cuando no se dispone de matriz de optimización (modo trade-log).
        """
        hard_gates_violations: List[str] = []

        # -------------------------------------------------------------
        # VERIFICACIÓN DE LAS 6 COMPUERTAS DURAS (HARD GATES)
        # -------------------------------------------------------------
        # Gate 1: P_ruin Reshuffling > 0.0%
        # Gate 2: P_ruin Bootstrapping >= 5.0%
        # Gate 3: P95 Drawdown Monte Carlo > $1,700 USD (evaluando todas las simulaciones MC disponibles)
        ruin_reshuffle = float(monte_carlo_res.get("reshuffle", {}).get("risk_of_ruin_pct", 0.0))
        ruin_bootstrap = float(monte_carlo_res.get("bootstrap", {}).get("risk_of_ruin_pct", 0.0))
        p95_candidates = [
            float(monte_carlo_res[k]["drawdown_stats"]["p95_dd_usd"])
            for k in ["bootstrap", "session_bootstrap", "reshuffle"]
            if k in monte_carlo_res and isinstance(monte_carlo_res[k], dict) and "drawdown_stats" in monte_carlo_res[k] and "p95_dd_usd" in monte_carlo_res[k]["drawdown_stats"]
        ]
        p95_dd = max(p95_candidates) if p95_candidates else 0.0

        if ruin_reshuffle > 0.0:
            hard_gates_violations.append(f"Gate 1: Riesgo de Ruina Reshuffling > 0.0% ({ruin_reshuffle:.2f}%)")

        if ruin_bootstrap >= 5.0:
            hard_gates_violations.append(f"Gate 2: Riesgo de Ruina Bootstrapping >= 5.0% ({ruin_bootstrap:.2f}%)")

        if p95_dd > 1700.0:
            hard_gates_violations.append(f"Gate 3: P95 Drawdown Monte Carlo > $1,700 USD (${p95_dd:,.2f})")

        # Gate 4: Profit Factor < 1.0 bajo Slippage Shock 2x
        pf_slip2x = float(what_if_res.get("slippage_2x", {}).get("profit_factor", 1.0))
        if pf_slip2x < 1.0:
            hard_gates_violations.append(f"Gate 4: Profit Factor < 1.0 bajo Slippage 2x (PF={pf_slip2x:.2f})")

        # Gate 5: Profit Factor < 1.0 tras eliminar los 3 mejores trades
        pf_top3 = float(what_if_res.get("top3_drop", {}).get("profit_factor", 1.0))
        if pf_top3 < 1.0:
            hard_gates_violations.append(f"Gate 5: Profit Factor < 1.0 tras eliminar Top-3 trades (PF={pf_top3:.2f})")

        # Gate 6: Violación del Daily Loss Limit ($1,000) en cualquier escenario
        session_dll_pct = float(monte_carlo_res.get("session_bootstrap", {}).get("dll_violation_pct", 0.0))
        base_dll = bool(what_if_res.get("baseline", {}).get("dll_violated", False))
        slip2x_dll = bool(what_if_res.get("slippage_2x", {}).get("dll_violated", False))
        slip3x_dll = bool(what_if_res.get("slippage_3x", {}).get("dll_violated", False))
        top3_dll = bool(what_if_res.get("top3_drop", {}).get("dll_violated", False))
        gap_dll = bool(what_if_res.get("gap_adverse_sl", {}).get("dll_violated", False))

        reasons = []
        if session_dll_pct > 0.0:
            reasons.append(f"Session Bootstrap DLL Breach ({session_dll_pct:.2f}%)")
        if base_dll:
            reasons.append("Baseline Histórico DLL Breach")
        if slip2x_dll:
            reasons.append("Slippage 2x DLL Breach")
        if slip3x_dll:
            reasons.append("Slippage 3x DLL Breach")
        if top3_dll:
            reasons.append("Top-3 Drop DLL Breach")
        if gap_dll:
            reasons.append("Gap adverso en SL DLL Breach")

        if reasons:
            hard_gates_violations.append(f"Gate 6: Violación de Daily Loss Limit (${self.dll:,.0f}): {', '.join(reasons)}")

        # -------------------------------------------------------------
        # SISTEMA DE PUNTUACIÓN (100 PUNTOS)
        # -------------------------------------------------------------
        points_earned = 0.0
        max_possible_points = 0.0
        pillar_breakdown: Dict[str, Any] = {}

        # Pilar 1: Monte Carlo & Ruina (30 pts)
        mc_pts = 0.0
        mc_boot = monte_carlo_res.get("bootstrap") or monte_carlo_res.get("session_bootstrap") or {}
        r_boot = float(mc_boot.get("risk_of_ruin_pct", 0.0))
        p_target = float(mc_boot.get("target_before_ruin_pct", 0.0))
        p95_val = float(mc_boot.get("drawdown_stats", {}).get("p95_dd_usd", 0.0))

        # Sub-criterio 1: Ruina Bootstrap (12 pts)
        if r_boot < 1.0:
            mc_pts += 12.0
        elif r_boot < 3.0:
            mc_pts += 8.0
        elif r_boot < 5.0:
            mc_pts += 4.0

        # Sub-criterio 2: Probabilidad de Aprobación (10 pts)
        if p_target >= 70.0:
            mc_pts += 10.0
        elif p_target >= 50.0:
            mc_pts += 6.0
        elif p_target >= 30.0:
            mc_pts += 3.0

        # Sub-criterio 3: P95 Drawdown (8 pts)
        if p95_val <= 1200.0:
            mc_pts += 8.0
        elif p95_val <= 1500.0:
            mc_pts += 5.0
        elif p95_val <= 1700.0:
            mc_pts += 2.0

        points_earned += mc_pts
        max_possible_points += 30.0
        pillar_breakdown["monte_carlo_points"] = {"earned": mc_pts, "max": 30.0}

        # Pilar 2: Estabilidad de Parámetros (25 pts)
        param_pts = 0.0
        param_max = 0.0

        # Si tenemos DSR (10 pts)
        if dsr_res is not None:
            param_max += 10.0
            dsr_val = float(dsr_res.get("dsr", 0.0))
            if dsr_val >= 0.95:
                param_pts += 10.0
            elif dsr_val >= 0.90:
                param_pts += 6.0
            elif dsr_val >= 0.80:
                param_pts += 3.0

        # Si tenemos vecindario de parámetros (15 pts)
        if parameter_res is not None and "Vecinos Analizados" in parameter_res:
            param_max += 15.0
            pos_pct_str = str(parameter_res.get("Vecinos con Sharpe Positivo (%)", "0")).replace("%", "")
            try:
                pos_pct = float(pos_pct_str)
            except ValueError:
                pos_pct = 0.0

            cv_nbr = float(parameter_res.get("Dispersión Vecinal (CV)", 99.0))

            # Retención vecinal (8 pts)
            if pos_pct >= 85.0:
                param_pts += 8.0
            elif pos_pct >= 75.0:
                param_pts += 5.0
            elif pos_pct >= 60.0:
                param_pts += 2.0

            # Dispersión vecinal (7 pts)
            if cv_nbr <= 0.35:
                param_pts += 7.0
            elif cv_nbr <= 0.50:
                param_pts += 4.0
            elif cv_nbr <= 0.70:
                param_pts += 2.0

        if param_max > 0:
            points_earned += param_pts
            max_possible_points += param_max
            pillar_breakdown["parameter_stability_points"] = {"earned": param_pts, "max": param_max}

        # Pilar 3: Walk-Forward Matrix (25 pts)
        wf_pts = 0.0
        if walk_forward_res is not None and walk_forward_res.get("total_evaluations", 0) > 0 and "mean_wfe_return" in walk_forward_res:
            max_possible_points += 25.0
            wfe_ret = float(walk_forward_res.get("mean_wfe_return", 0.0))
            oos_pos = float(walk_forward_res.get("pct_positive_oos_windows", 0.0))
            wfe_shp = float(walk_forward_res.get("mean_wfe_sharpe", 0.0))

            # WFE Return (12 pts)
            if wfe_ret >= 60.0:
                wf_pts += 12.0
            elif wfe_ret >= 50.0:
                wf_pts += 8.0
            elif wfe_ret >= 40.0:
                wf_pts += 4.0

            # Ventanas OOS Positivas (8 pts)
            if oos_pos >= 75.0:
                wf_pts += 8.0
            elif oos_pos >= 60.0:
                wf_pts += 5.0
            elif oos_pos >= 50.0:
                wf_pts += 2.0

            # WFE Sharpe / Estabilidad (5 pts)
            if wfe_shp >= 70.0:
                wf_pts += 5.0
            elif wfe_shp >= 50.0:
                wf_pts += 3.0

            points_earned += wf_pts
            pillar_breakdown["walk_forward_points"] = {"earned": wf_pts, "max": 25.0}

        # Pilar 4: What-If Stress Scenarios (20 pts)
        wi_pts = 0.0
        max_possible_points += 20.0

        # Sub-criterio 1: Slippage 2x (6 pts)
        if pf_slip2x >= 1.20:
            wi_pts += 6.0
        elif pf_slip2x >= 1.00:
            wi_pts += 4.0

        # Sub-criterio 2: Sin Top-3 Trades (6 pts)
        if pf_top3 >= 1.20:
            wi_pts += 6.0
        elif pf_top3 >= 1.00:
            wi_pts += 4.0

        # Sub-criterio 3: No viola DLL bajo Gap en SL (4 pts)
        if not gap_dll:
            wi_pts += 4.0

        # Sub-criterio 4: Rentable en Win-Pruning P5 (4 pts)
        p5_pf_pruned = float(what_if_res.get("win_pruning", {}).get("p5_profit_factor", 0.0))
        if p5_pf_pruned >= 1.00:
            wi_pts += 4.0
        elif p5_pf_pruned >= 0.90:
            wi_pts += 2.0

        points_earned += wi_pts
        pillar_breakdown["what_if_points"] = {"earned": wi_pts, "max": 20.0}

        # Normalización a escala de 100 puntos
        if max_possible_points > 0:
            normalized_score = round((points_earned / max_possible_points) * 100.0, 1)
        else:
            normalized_score = 0.0

        # -------------------------------------------------------------
        # CLASIFICACIÓN FINAL DEL "ÚLTIMO VEREDICTO"
        # -------------------------------------------------------------
        hard_gates_passed = (len(hard_gates_violations) == 0)

        if not hard_gates_passed:
            status = "FAIL"
            verdict = "🔴 FAIL - REPROBADO POR COMPUERTA DURA (HARD GATE)"
        elif normalized_score >= 80.0:
            status = "PASS"
            verdict = "🟢 PASS - CERTIFICADO PARA PRUEBA DE FONDEO"
        elif normalized_score >= 65.0:
            status = "CONDITIONAL_PASS"
            verdict = "🟡 CONDITIONAL PASS (Precaución - Requiere Micro-Contratos)"
        else:
            status = "FAIL"
            verdict = "🔴 FAIL - REPROBADO (Puntuación Insuficiente < 65/100)"

        return {
            "score": normalized_score,
            "raw_points_earned": points_earned,
            "max_possible_points": max_possible_points,
            "status": status,
            "verdict": verdict,
            "hard_gates_passed": hard_gates_passed,
            "hard_gates_violations": hard_gates_violations,
            "pillar_breakdown": pillar_breakdown,
        }

