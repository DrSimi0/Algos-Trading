"""
src/core/multi_year_wfm.py
Motor de Walk-Forward Matrix (WFM 2D) Multi-Año con Aislamiento Criptográfico.

Evalúa la robustez temporal de estrategias a través de la frontera In-Sample (2023) -> Out-of-Sample (2024)
mediante una rejilla bidimensional:
  T_train in [60, 90, 120, 180, 240] días RTH
  T_test  in [20, 30, 45, 60] días RTH

Calcula:
1. Walk-Forward Efficiency (WFE - Robert Pardo):
   - WFE_Return = (Annualized OOS Return / Mean Annualized IS Return) * 100% (Umbral >= 55%)
   - WFE_Sharpe = (OOS Sharpe / Mean IS Sharpe) * 100%
2. Consistencia Temporal (C_temp):
   - Porcentaje de ventanas OOS con beneficio neto positivo (Umbral >= 70%)
3. Matrix Stability Index (MSI):
   - MSI = sigma(WFE) / mu(WFE) (Umbral <= 0.40)
4. Auditoría Continua Prop Firm:
   - MTDD <= $2,000 USD, DLL <= $1,000 USD en cada ventana de prueba.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class WFMCellResult:
    train_days: int
    test_days: int
    total_folds: int
    wfe_return: float
    wfe_sharpe: float
    c_temp: float
    total_oos_profit: float
    mean_is_profit: float
    oos_sharpe: float
    mean_is_sharpe: float
    max_oos_drawdown: float
    worst_oos_daily_loss: float
    dll_violated: bool
    mtdd_violated: bool
    status: str
    folds_details: List[Dict[str, Any]] = field(default_factory=list)


class MultiYearWalkForwardMatrix:
    """
    Motor de Walk-Forward Matrix (WFM 2D) para evaluación multi-anual.
    """

    DEFAULT_TRAIN_WINDOWS = [60, 90, 120, 180, 240]
    DEFAULT_TEST_WINDOWS = [20, 30, 45, 60]

    def __init__(
        self,
        train_windows: Optional[List[int]] = None,
        test_windows: Optional[List[int]] = None,
        mtdd_limit: float = 2000.0,
        dll_limit: float = 1000.0,
        initial_capital: float = 50000.0,
        anchored: bool = False,
    ):
        self.train_windows = sorted(train_windows or self.DEFAULT_TRAIN_WINDOWS)
        self.test_windows = sorted(test_windows or self.DEFAULT_TEST_WINDOWS)
        self.mtdd_limit = float(mtdd_limit)
        self.dll_limit = float(dll_limit)
        self.initial_capital = float(initial_capital)
        self.anchored = bool(anchored)

    def _extract_datetime_series(
        self,
        df: pd.DataFrame,
        date_col: Optional[str] = None,
    ) -> pd.Series:
        """Extrae o infiere la serie de timestamps normalizados en UTC."""
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
            for c in ["_dt", "datetime", "Datetime", "exit_time", "entry_time", "date", "Date", "timestamp", "Timestamp"]:
                if c in df.columns:
                    sample_str = str(df[c].dropna().iloc[0]) if len(df[c].dropna()) > 0 else ""
                    if "-" in sample_str or "/" in sample_str or isinstance(df[c].iloc[0], (pd.Timestamp, datetime)):
                        dt_series = pd.to_datetime(df[c], utc=True)
                        break

        if dt_series is None and isinstance(df.index, pd.DatetimeIndex):
            dt_series = pd.to_datetime(df.index, utc=True)

        if dt_series is None:
            raise ValueError("No se pudo detectar una columna DateTime válida en el registro.")

        return dt_series

    def _extract_pnl_col(self, df: pd.DataFrame, pnl_col: Optional[str] = None) -> str:
        """Determina el nombre de la columna que contiene el PnL neto en USD."""
        if pnl_col is not None and pnl_col in df.columns:
            return pnl_col
        for c in ["net_pnl_usd", "net_pnl", "net_profit", "pnl", "PnL", "Net_PnL"]:
            if c in df.columns:
                return c
        raise ValueError(f"No se encontró columna de PnL en el DataFrame. Columnas: {list(df.columns)}")

    def generate_folds(
        self,
        df: pd.DataFrame,
        train_days: int,
        test_days: int,
        step_days: Optional[int] = None,
        date_col: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Genera la secuencia de particiones temporales Train/Test rodantes o ancladas.
        """
        if df.empty:
            return []

        work_df = df.copy()
        work_df["_dt"] = self._extract_datetime_series(work_df, date_col)
        work_df = work_df.sort_values("_dt").reset_index(drop=True)

        start_time = work_df["_dt"].min()
        end_time = work_df["_dt"].max()

        step_d = step_days if step_days is not None else test_days
        train_delta = pd.Timedelta(days=train_days)
        test_delta = pd.Timedelta(days=test_days)
        step_delta = pd.Timedelta(days=step_d)

        folds: List[Dict[str, Any]] = []
        current_train_start = start_time
        fold_idx = 1

        while True:
            train_start = start_time if self.anchored else current_train_start
            train_end = current_train_start + train_delta
            test_start = train_end
            test_end = test_start + test_delta

            if test_start >= end_time:
                break

            if test_end > end_time:
                if (end_time - test_start).days >= 10:
                    test_end = end_time + pd.Timedelta(seconds=1)
                else:
                    break

            train_mask = (work_df["_dt"] >= train_start) & (work_df["_dt"] < train_end)
            test_mask = (work_df["_dt"] >= test_start) & (work_df["_dt"] < test_end)

            train_slice = work_df[train_mask].copy()
            test_slice = work_df[test_mask].copy()

            if len(train_slice) > 0 and len(test_slice) > 0:
                folds.append({
                    "fold_idx": fold_idx,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_count": len(train_slice),
                    "test_count": len(test_slice),
                    "train_data": train_slice,
                    "test_data": test_slice,
                })
                fold_idx += 1

            current_train_start += step_delta
            if (current_train_start + train_delta) >= end_time:
                break

        return folds

    def evaluate_trades(
        self,
        trades_df: pd.DataFrame,
        date_col: Optional[str] = None,
        pnl_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evalúa el registro de operaciones a lo largo de toda la matriz 2D de Walk-Forward.
        """
        if trades_df.empty:
            raise ValueError("El DataFrame de operaciones trades_df está vacío.")

        work_df = trades_df.copy()
        work_df["_dt"] = self._extract_datetime_series(work_df, date_col)
        work_df = work_df.sort_values("_dt").reset_index(drop=True)
        chosen_pnl = self._extract_pnl_col(work_df, pnl_col)

        cell_results: List[WFMCellResult] = []
        grid_table: List[Dict[str, Any]] = []

        all_wfe_returns: List[float] = []
        all_wfe_sharpes: List[float] = []
        all_c_temps: List[float] = []
        global_dll_violations = 0
        global_mtdd_violations = 0
        worst_matrix_dd = 0.0
        worst_matrix_dll = 0.0

        for tw in self.train_windows:
            row_dict: Dict[str, Any] = {"train_days": tw}
            for tst in self.test_windows:
                folds = self.generate_folds(
                    df=work_df,
                    train_days=tw,
                    test_days=tst,
                    step_days=tst,
                    date_col="_dt",
                )

                if len(folds) == 0:
                    row_dict[f"test_{tst}d"] = "N/A (Folds insuficientes)"
                    continue

                is_profits: List[float] = []
                is_ann_returns: List[float] = []
                is_sharpes: List[float] = []
                oos_profits: List[float] = []
                oos_daily_pnls: List[float] = []
                positive_folds = 0
                cell_dll_violated = False
                cell_mtdd_violated = False
                cell_max_dd = 0.0
                cell_worst_day = 0.0
                total_oos_elapsed_days = 0.0
                fold_details = []

                for f in folds:
                    tr_pnl = f["train_data"][chosen_pnl].astype(float).values
                    te_pnl = f["test_data"][chosen_pnl].astype(float).values
                    tr_dates = f["train_data"]["_dt"].dt.date.values
                    te_dates = f["test_data"]["_dt"].dt.date.values

                    # Duración exacta del fold de entrenamiento y prueba
                    fold_train_days = max(1.0, (f["train_end"] - f["train_start"]).total_seconds() / 86400.0)
                    fold_test_days = max(1.0, (f["test_end"] - f["test_start"]).total_seconds() / 86400.0)
                    total_oos_elapsed_days += fold_test_days

                    # IS metrics
                    is_p = float(np.sum(tr_pnl))
                    is_profits.append(is_p)
                    is_ann_returns.append(is_p * (252.0 / fold_train_days))

                    if len(tr_dates) > 0:
                        daily_tr = pd.Series(tr_pnl).groupby(tr_dates).sum().values
                        shp_is = (np.mean(daily_tr) / np.std(daily_tr, ddof=1) * np.sqrt(252.0)) if len(daily_tr) > 1 and np.std(daily_tr, ddof=1) > 1e-9 else 0.0
                    else:
                        shp_is = 0.0
                    is_sharpes.append(float(shp_is))

                    # OOS metrics
                    te_p = float(np.sum(te_pnl))
                    oos_profits.append(te_p)
                    if len(te_dates) > 0:
                        daily_te = pd.Series(te_pnl).groupby(te_dates).sum().values
                        oos_daily_pnls.extend(daily_te.tolist())
                    if te_p > 0:
                        positive_folds += 1

                    # Prop firm audit in this test fold
                    # Drawdown: incluir capital inicial al inicio de la curva de equidad
                    eq_curve = np.concatenate(([self.initial_capital], self.initial_capital + np.cumsum(te_pnl)))
                    peak = np.maximum.accumulate(eq_curve)
                    dd = float(np.max(peak - eq_curve)) if len(eq_curve) > 0 else 0.0
                    if dd > cell_max_dd:
                        cell_max_dd = dd
                    if dd >= self.mtdd_limit:
                        cell_mtdd_violated = True
                        global_mtdd_violations += 1

                    # Intraday / Session daily loss
                    if len(te_dates) > 0:
                        daily_cum = pd.Series(te_pnl).groupby(te_dates).cumsum()
                        worst_session = float(abs(daily_cum.min())) if float(daily_cum.min()) < 0 else 0.0
                    else:
                        worst_session = float(abs(np.min(te_pnl))) if len(te_pnl) > 0 and np.min(te_pnl) < 0 else 0.0

                    if worst_session > cell_worst_day:
                        cell_worst_day = worst_session
                    if worst_session >= self.dll_limit:
                        cell_dll_violated = True
                        global_dll_violations += 1

                    fold_details.append({
                        "fold_idx": f["fold_idx"],
                        "train_start": str(f["train_start"].date()),
                        "train_end": str(f["train_end"].date()),
                        "test_start": str(f["test_start"].date()),
                        "test_end": str(f["test_end"].date()),
                        "is_profit": round(is_p, 2),
                        "oos_profit": round(te_p, 2),
                        "oos_dd": round(dd, 2),
                        "worst_day": round(worst_session, 2),
                    })

                # Robert Pardo WFE
                total_oos_p = float(np.sum(oos_profits))
                mean_is_p = float(np.mean(is_profits)) if is_profits else 0.0
                mean_is_ann_return = float(np.mean(is_ann_returns)) if is_ann_returns else 0.0

                oos_ann_return = total_oos_p * (252.0 / max(1.0, total_oos_elapsed_days))

                if mean_is_ann_return > 0:
                    wfe_ret = (oos_ann_return / mean_is_ann_return) * 100.0
                elif mean_is_ann_return == 0:
                    wfe_ret = 100.0 if oos_ann_return >= 0 else 0.0
                else:
                    wfe_ret = 0.0

                # Sharpe WFE sobre retornos diarios anualizados
                oos_daily_arr = np.array(oos_daily_pnls)
                if len(oos_daily_arr) > 1 and np.std(oos_daily_arr, ddof=1) > 1e-9:
                    oos_shp = float((np.mean(oos_daily_arr) / np.std(oos_daily_arr, ddof=1)) * np.sqrt(252.0))
                else:
                    oos_shp = 0.0

                mean_is_shp = float(np.mean(is_sharpes)) if is_sharpes else 0.0
                if mean_is_shp > 0:
                    wfe_shp = (oos_shp / mean_is_shp) * 100.0
                else:
                    wfe_shp = 100.0 if oos_shp >= 0 else 0.0

                c_temp = (positive_folds / len(folds)) * 100.0 if folds else 0.0

                if cell_max_dd > worst_matrix_dd:
                    worst_matrix_dd = cell_max_dd
                if cell_worst_day > worst_matrix_dll:
                    worst_matrix_dll = cell_worst_day

                # Cell status
                cell_passed = (wfe_ret >= 55.0) and (c_temp >= 70.0) and (not cell_dll_violated) and (not cell_mtdd_violated)
                cell_status = "PASS" if cell_passed else "FAIL"

                cell_res = WFMCellResult(
                    train_days=tw,
                    test_days=tst,
                    total_folds=len(folds),
                    wfe_return=round(wfe_ret, 1),
                    wfe_sharpe=round(wfe_shp, 1),
                    c_temp=round(c_temp, 1),
                    total_oos_profit=round(total_oos_p, 2),
                    mean_is_profit=round(mean_is_p, 2),
                    oos_sharpe=round(oos_shp, 2),
                    mean_is_sharpe=round(mean_is_shp, 2),
                    max_oos_drawdown=round(cell_max_dd, 2),
                    worst_oos_daily_loss=round(cell_worst_day, 2),
                    dll_violated=cell_dll_violated,
                    mtdd_violated=cell_mtdd_violated,
                    status=cell_status,
                    folds_details=fold_details,
                )
                cell_results.append(cell_res)

                all_wfe_returns.append(wfe_ret)
                all_wfe_sharpes.append(wfe_shp)
                all_c_temps.append(c_temp)

                cell_flag = "[PASS]" if cell_passed else "[FAIL]"
                row_dict[f"test_{tst}d"] = f"{cell_flag} WFE={wfe_ret:.1f}% C_temp={c_temp:.0f}%"

            grid_table.append(row_dict)

        # Matriz Completa: Métricas Globales
        mean_wfe_ret = float(np.mean(all_wfe_returns)) if all_wfe_returns else 0.0
        std_wfe_ret = float(np.std(all_wfe_returns, ddof=1)) if len(all_wfe_returns) > 1 else 0.0
        mean_wfe_shp = float(np.mean(all_wfe_sharpes)) if all_wfe_sharpes else 0.0
        mean_c_temp = float(np.mean(all_c_temps)) if all_c_temps else 0.0

        # Matrix Stability Index (MSI) = sigma(WFE) / mu(WFE)
        msi = (std_wfe_ret / mean_wfe_ret) if mean_wfe_ret > 1e-6 else 1.0

        # Criterios de Aprobación Global del Plan:
        # 1. WFE Retorno >= 55%
        # 2. Consistencia Temporal C_temp >= 70%
        # 3. MSI <= 0.40
        # 4. Cero violaciones de MTDD ($2,000) y DLL ($1,000)
        criterio_wfe = bool(mean_wfe_ret >= 55.0)
        criterio_ctemp = bool(mean_c_temp >= 70.0)
        criterio_msi = bool(msi <= 0.40)
        criterio_prop = bool((global_dll_violations == 0) and (global_mtdd_violations == 0))

        matrix_passed = criterio_wfe and criterio_ctemp and criterio_msi and criterio_prop

        if matrix_passed:
            verdict = "[PASS] - WALK-FORWARD MATRIX APROBADA (Robusta, Consistente y Estable)"
        else:
            reasons = []
            if not criterio_wfe:
                reasons.append(f"WFE={mean_wfe_ret:.1f}% < 55%")
            if not criterio_ctemp:
                reasons.append(f"C_temp={mean_c_temp:.1f}% < 70%")
            if not criterio_msi:
                reasons.append(f"MSI={msi:.2f} > 0.40")
            if not criterio_prop:
                reasons.append(f"Violaciones Prop Firm (DLL={global_dll_violations}, MTDD={global_mtdd_violations})")
            verdict = f"[FAIL] - REPROBADA ({', '.join(reasons)})"

        return {
            "matrix_passed": matrix_passed,
            "verdict": verdict,
            "mean_wfe_return": round(mean_wfe_ret, 1),
            "std_wfe_return": round(std_wfe_ret, 1),
            "mean_wfe_sharpe": round(mean_wfe_shp, 1),
            "mean_c_temp": round(mean_c_temp, 1),
            "matrix_stability_index_msi": round(msi, 3),
            "worst_matrix_drawdown_usd": round(worst_matrix_dd, 2),
            "worst_matrix_daily_loss_usd": round(worst_matrix_dll, 2),
            "total_dll_violations": global_dll_violations,
            "total_mtdd_violations": global_mtdd_violations,
            "total_evaluated_cells": len(cell_results),
            "grid_table": grid_table,
            "cells": [
                {
                    "train_days": c.train_days,
                    "test_days": c.test_days,
                    "total_folds": c.total_folds,
                    "wfe_return": c.wfe_return,
                    "wfe_sharpe": c.wfe_sharpe,
                    "c_temp": c.c_temp,
                    "total_oos_profit": c.total_oos_profit,
                    "mean_is_profit": c.mean_is_profit,
                    "oos_sharpe": c.oos_sharpe,
                    "mean_is_sharpe": c.mean_is_sharpe,
                    "max_oos_drawdown": c.max_oos_drawdown,
                    "worst_oos_daily_loss": c.worst_oos_daily_loss,
                    "dll_violated": c.dll_violated,
                    "mtdd_violated": c.mtdd_violated,
                    "status": c.status,
                }
                for c in cell_results
            ],
            "criteria_summary": {
                "wfe_return_passed": criterio_wfe,
                "c_temp_passed": criterio_ctemp,
                "msi_passed": criterio_msi,
                "prop_firm_passed": criterio_prop,
            }
        }
