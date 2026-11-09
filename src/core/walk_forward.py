"""
src/core/walk_forward.py
Motor de optimización y validación Walk-Forward (WFO) y Walk-Forward Matrix (WFM).

Restricción estricta de cuarentena:
Todas las particiones y ventanas deben residir estrictamente dentro del período
In-Sample (Año 2023). Queda terminantemente prohibido acceder a OOS (2024) y Holdout (2025).
"""

from typing import Dict, Any, List, Optional, Tuple, Generator
import numpy as np
import pandas as pd


class WalkForwardEngine:
    """
    Generador de particiones temporales rodantes (rolling) o ancladas (anchored)
    confinadas estrictamente dentro del rango In-Sample disponible.
    """

    def __init__(
        self,
        data_or_trades: pd.DataFrame,
        date_col: Optional[str] = None,
        train_days: int = 90,
        test_days: int = 30,
        step_days: Optional[int] = None,
        anchored: bool = False,
        enforce_quarantine: bool = True,
    ):
        """
        Args:
            data_or_trades: DataFrame con operaciones o barras temporales.
            date_col: Nombre de la columna de fecha (si None, se detecta automáticamente).
            train_days: Tamaño de la ventana de optimización/entrenamiento (días calendario).
            test_days: Tamaño de la ventana de evaluación fuera de muestra (días calendario).
            step_days: Paso de desplazamiento entre ventanas (def: igual a test_days).
            anchored: Si es True, la ventana de entrenamiento se expande desde el inicio.
            enforce_quarantine: Si es True, rechaza o corta datos posteriores al 31 de diciembre de 2023.
        """
        if data_or_trades.empty:
            raise ValueError("El DataFrame proporcionado a WalkForwardEngine está vacío.")

        df = data_or_trades.copy()

        # Detección de columna de fecha robusta
        dt_series = None
        if date_col is not None and date_col in df.columns:
            if date_col in ["exit_date", "entry_date"] and ("exit_time" in df.columns or "entry_time" in df.columns):
                tc = "exit_time" if "exit_time" in df.columns else "entry_time"
                dt_series = pd.to_datetime(df[date_col].astype(str) + " " + df[tc].astype(str), utc=True)
            else:
                dt_series = pd.to_datetime(df[date_col], utc=True)
        elif "exit_date" in df.columns:
            tc = "exit_time" if "exit_time" in df.columns else "entry_time"
            if tc in df.columns:
                dt_series = pd.to_datetime(df["exit_date"].astype(str) + " " + df[tc].astype(str), utc=True)
            else:
                dt_series = pd.to_datetime(df["exit_date"], utc=True)
        elif "entry_date" in df.columns:
            tc = "entry_time" if "entry_time" in df.columns else "exit_time"
            if tc in df.columns:
                dt_series = pd.to_datetime(df["entry_date"].astype(str) + " " + df[tc].astype(str), utc=True)
            else:
                dt_series = pd.to_datetime(df["entry_date"], utc=True)
        else:
            for col in ["exit_time", "entry_time", "date", "Date", "timestamp", "Timestamp"]:
                if col in df.columns:
                    sample_str = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else ""
                    if "-" in sample_str or "/" in sample_str:
                        dt_series = pd.to_datetime(df[col], utc=True)
                        break

        if dt_series is None and isinstance(df.index, pd.DatetimeIndex):
            dt_series = pd.to_datetime(df.index, utc=True)

        if dt_series is None:
            raise ValueError("No se pudo detectar una columna o índice DateTime válido en los datos.")

        df["_parsed_dt"] = dt_series

        # Ordenar cronológicamente
        df = df.sort_values("_parsed_dt").reset_index(drop=True)

        # Restricción de Cuarentena Cuantitativa
        if enforce_quarantine:
            max_is_limit = pd.Timestamp("2023-12-31 23:59:59.999999", tz="UTC")
            leaked_mask = df["_parsed_dt"] > max_is_limit
            if leaked_mask.any():
                # Cortar estrictamente para proteger la cuarentena
                df = df[~leaked_mask].copy()

        self.df = df
        self.train_days = int(train_days)
        self.test_days = int(test_days)
        self.step_days = int(step_days) if step_days is not None else int(test_days)
        self.anchored = bool(anchored)

    def generate_folds(self) -> List[Dict[str, Any]]:
        """
        Genera la lista de particiones Walk-Forward (Train / Test).
        """
        if len(self.df) == 0:
            return []

        start_time = self.df["_parsed_dt"].min()
        end_time = self.df["_parsed_dt"].max()

        folds: List[Dict[str, Any]] = []
        current_train_start = start_time

        train_delta = pd.Timedelta(days=self.train_days)
        test_delta = pd.Timedelta(days=self.test_days)
        step_delta = pd.Timedelta(days=self.step_days)

        fold_idx = 1
        while True:
            if self.anchored:
                train_start = start_time
            else:
                train_start = current_train_start

            train_end = current_train_start + train_delta
            test_start = train_end
            test_end = test_start + test_delta

            if test_end > (end_time + pd.Timedelta(days=1)):
                # Si el test sobrepasa el final de datos disponibles, verificar si hay suficientes datos
                if test_start < end_time and (end_time - test_start).days >= 10:
                    test_end = end_time + pd.Timedelta(seconds=1)
                else:
                    break

            train_mask = (self.df["_parsed_dt"] >= train_start) & (self.df["_parsed_dt"] < train_end)
            test_mask = (self.df["_parsed_dt"] >= test_start) & (self.df["_parsed_dt"] < test_end)

            train_slice = self.df[train_mask].copy()
            test_slice = self.df[test_mask].copy()

            if len(train_slice) > 0 and len(test_slice) > 0:
                folds.append({
                    "fold_idx": fold_idx,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_trades_count": len(train_slice),
                    "test_trades_count": len(test_slice),
                    "train_data": train_slice,
                    "test_data": test_slice,
                })
                fold_idx += 1

            current_train_start += step_delta

            if (current_train_start + train_delta) >= end_time:
                break

        return folds


class WalkForwardMatrix:
    """
    Rejilla 2D (Entrenamiento x Prueba) para evaluar la robustez y consistencia
    de la estrategia a través de múltiples horizontes de optimización y prueba.
    
    Calcula:
    1. Walk-Forward Efficiency (WFE - Robert Pardo):
       WFE_Return = (Annualized OOS Return / Mean Annualized IS Return) * 100%
       WFE_Sharpe = (OOS Sharpe / Mean IS Sharpe) * 100%
    2. Consistencia Temporal (% de ventanas OOS con beneficio positivo).
    """

    def __init__(
        self,
        train_windows: Optional[List[int]] = None,
        test_windows: Optional[List[int]] = None,
        enforce_quarantine: bool = True,
    ):
        """
        Args:
            train_windows: Lista de tamaños de ventana IS en días (def: [60, 90, 120]).
            test_windows: Lista de tamaños de ventana OOS en días (def: [20, 30, 45]).
            enforce_quarantine: Si es True, confina la evaluación dentro de In-Sample 2023.
        """
        self.train_windows = train_windows or [60, 90, 120]
        self.test_windows = test_windows or [20, 30, 45]
        self.enforce_quarantine = enforce_quarantine

    def evaluate_trades(
        self,
        trades_df: pd.DataFrame,
        date_col: Optional[str] = None,
        pnl_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evalúa el registro de operaciones a lo largo de la cuadrícula 2D de Walk-Forward.
        """
        if trades_df.empty:
            raise ValueError("El registro de operaciones trades_df está vacío.")

        df = trades_df.copy()

        # Detectar PnL
        chosen_pnl = pnl_col
        if chosen_pnl is None:
            for col in ["net_pnl_usd", "net_pnl", "net_profit", "pnl", "PnL", "Net_PnL"]:
                if col in df.columns:
                    chosen_pnl = col
                    break

        if chosen_pnl is None:
            raise ValueError(f"No se encontró columna de PnL en {list(df.columns)}")

        matrix_results: List[Dict[str, Any]] = []
        wfe_returns: List[float] = []
        wfe_sharpes: List[float] = []
        positive_window_pcts: List[float] = []

        grid_table = []

        for tw in self.train_windows:
            row_entry = {"train_days": tw}
            for tst in self.test_windows:
                engine = WalkForwardEngine(
                    data_or_trades=df,
                    date_col=date_col,
                    train_days=tw,
                    test_days=tst,
                    step_days=tst,
                    anchored=False,
                    enforce_quarantine=self.enforce_quarantine,
                )
                folds = engine.generate_folds()

                if len(folds) == 0:
                    row_entry[f"test_{tst}d"] = "N/A"
                    continue

                is_profits: List[float] = []
                is_sharpes: List[float] = []
                oos_profits: List[float] = []
                oos_trades_list: List[float] = []

                positive_folds_count = 0

                for f in folds:
                    tr_pnl = f["train_data"][chosen_pnl].astype(float).values
                    te_pnl = f["test_data"][chosen_pnl].astype(float).values

                    is_p = float(np.sum(tr_pnl))
                    is_profits.append(is_p)

                    # Sharpe IS
                    if len(tr_pnl) > 1 and np.std(tr_pnl, ddof=1) > 1e-9:
                        shp_is = (np.mean(tr_pnl) / np.std(tr_pnl, ddof=1)) * np.sqrt(252.0)
                    else:
                        shp_is = 0.0
                    is_sharpes.append(shp_is)

                    te_p = float(np.sum(te_pnl))
                    oos_profits.append(te_p)
                    oos_trades_list.extend(te_pnl.tolist())

                    if te_p > 0:
                        positive_folds_count += 1

                # WFE Robert Pardo
                total_oos_profit = float(np.sum(oos_profits))
                mean_is_profit = float(np.mean(is_profits)) if len(is_profits) > 0 else 0.0

                # Escala anualizada para comparar IS y OOS en períodos distintos
                is_ann_return = mean_is_profit * (365.0 / tw) if tw > 0 else 0.0
                total_oos_days = len(folds) * tst
                oos_ann_return = total_oos_profit * (365.0 / max(1, total_oos_days))

                if is_ann_return > 0:
                    wfe_ret = (oos_ann_return / is_ann_return) * 100.0
                elif is_ann_return == 0:
                    wfe_ret = 100.0 if oos_ann_return >= 0 else 0.0
                else:
                    wfe_ret = 0.0

                # OOS Sharpe concatenado
                oos_arr = np.array(oos_trades_list)
                if len(oos_arr) > 1 and np.std(oos_arr, ddof=1) > 1e-9:
                    oos_sharpe = float((np.mean(oos_arr) / np.std(oos_arr, ddof=1)) * np.sqrt(252.0))
                else:
                    oos_sharpe = 0.0

                mean_is_sharpe = float(np.mean(is_sharpes)) if len(is_sharpes) > 0 else 0.0
                if mean_is_sharpe > 0:
                    wfe_shp = (oos_sharpe / mean_is_sharpe) * 100.0
                else:
                    wfe_shp = 100.0 if oos_sharpe >= 0 else 0.0

                pct_positive = (positive_folds_count / len(folds)) * 100.0

                wfe_returns.append(wfe_ret)
                wfe_sharpes.append(wfe_shp)
                positive_window_pcts.append(pct_positive)

                cell_res = {
                    "train_days": tw,
                    "test_days": tst,
                    "total_folds": len(folds),
                    "wfe_return": round(wfe_ret, 1),
                    "wfe_sharpe": round(wfe_shp, 1),
                    "pct_positive_windows": round(pct_positive, 1),
                    "total_oos_profit_usd": round(total_oos_profit, 2),
                    "mean_is_profit_usd": round(mean_is_profit, 2),
                    "oos_sharpe": round(oos_sharpe, 2),
                    "mean_is_sharpe": round(mean_is_sharpe, 2),
                }
                matrix_results.append(cell_res)
                row_entry[f"test_{tst}d"] = f"WFE={wfe_ret:.1f}% (Pos={pct_positive:.0f}%)"

            grid_table.append(row_entry)

        mean_wfe_ret = float(np.mean(wfe_returns)) if wfe_returns else 0.0
        mean_wfe_shp = float(np.mean(wfe_sharpes)) if wfe_sharpes else 0.0
        mean_pos_windows = float(np.mean(positive_window_pcts)) if positive_window_pcts else 0.0

        # Veredicto de robustez temporal
        # Robert Pardo: WFE >= 55% y consistencia >= 70%
        if mean_wfe_ret >= 60.0 and mean_pos_windows >= 70.0:
            verdict = "🟢 EXCELENTE CONSISTENCIA WALK-FORWARD (WFE >= 60% y Positivas >= 70%)"
        elif mean_wfe_ret >= 50.0 and mean_pos_windows >= 55.0:
            verdict = "🟢 ROBUSTA WALK-FORWARD (WFE >= 50% y Rendimiento Sostenido)"
        elif mean_wfe_ret >= 40.0:
            verdict = "🟡 MODERADA (WFE 40-50% - Degradación Parcial en Prueba Ciega)"
        else:
            verdict = "🔴 INESTABLE / SOBREAJUSTADA (WFE < 40% - Falla en replicar IS)"

        return {
            "matrix_results": matrix_results,
            "grid_table": grid_table,
            "mean_wfe_return": round(mean_wfe_ret, 1),
            "mean_wfe_sharpe": round(mean_wfe_shp, 1),
            "pct_positive_oos_windows": round(mean_pos_windows, 1),
            "total_evaluations": len(matrix_results),
            "verdict": verdict,
        }
