"""
src/core/robustness.py
Análisis estadístico de robustez cuantitativa:
1. Filtro y detector de Factor Suerte (pocas operaciones y concentración de beneficios).
2. Detector de Mesetas de Estabilidad de Parámetros frente a picos sobreajustados (curve-fitting).
"""

from typing import Dict, Any, List, Optional
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
