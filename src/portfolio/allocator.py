"""
src/portfolio/allocator.py
Asignador de capital para portafolios multi-estrategia.
Soporta:
1. Ponderaciones: Equal-Weight, Inverse-Volatility (Risk Parity base) y personalizada.
2. Combinación de curvas de retornos porcentuales.
3. Fusión de logs de operaciones reales / PnL en dólares para cuentas de Prop Trading (CME Globex).
4. Auditoría de correlación, beneficio de diversificación y cumplimiento de límites de fondeo (DLL y MTDD).
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd


class PortfolioAllocator:
    """
    Combina retornos y secuencias de operaciones de múltiples estrategias
    en una curva de capital consolidada y audita el riesgo conjunto.
    """

    def __init__(self, method: str = "equal_weight", custom_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            method: 'equal_weight', 'inverse_volatility', o 'custom'
            custom_weights: Diccionario de pesos {strategy_name: peso} si method='custom'
        """
        self.method = method
        self.custom_weights = custom_weights or {}

    def calculate_weights(self, strategy_returns: pd.DataFrame) -> pd.Series:
        """
        Calcula las ponderaciones normalizadas (suman 1.0) para cada estrategia.
        """
        strategies = strategy_returns.columns
        n = len(strategies)

        if n == 0:
            raise ValueError("No hay retornos de estrategias para asignar pesos.")

        if self.method == "equal_weight":
            weights = pd.Series(1.0 / n, index=strategies)

        elif self.method == "inverse_volatility":
            vols = strategy_returns.std()
            vols = vols.replace(0, 1e-6)
            inv_vols = 1.0 / vols
            weights = inv_vols / inv_vols.sum()

        elif self.method == "custom":
            weights = pd.Series([self.custom_weights.get(s, 1.0 / n) for s in strategies], index=strategies)
            weights = weights / weights.sum()

        else:
            raise ValueError(f"Método de asignación desconocido: {self.method}")

        return weights

    def combine(
        self,
        strategy_returns: pd.DataFrame,
        initial_capital: float = 10000.0,
        periods_per_year: int = 252
    ) -> Dict[str, Any]:
        """
        Combina los retornos porcentuales de las estrategias y calcula la curva de capital del portafolio.
        """
        clean_returns = strategy_returns.dropna()
        weights = self.calculate_weights(clean_returns)

        # Retorno diario del portafolio ponderado
        portfolio_returns = (clean_returns * weights).sum(axis=1)

        # Curva de capital combinada
        equity_curve = (1.0 + portfolio_returns).cumprod() * initial_capital
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak

        # Métricas consolidadas
        total_return = (equity_curve.iloc[-1] / initial_capital) - 1.0
        n_periods = len(portfolio_returns)
        n_years = max(n_periods / periods_per_year, 1.0 / periods_per_year)
        cagr = ((equity_curve.iloc[-1] / initial_capital) ** (1.0 / n_years)) - 1.0
        volatility = portfolio_returns.std() * np.sqrt(periods_per_year)
        sharpe = (portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(periods_per_year)) if portfolio_returns.std() > 0 else 0.0
        max_dd = drawdown.min()

        metrics = {
            "Total Return (%)": round(total_return * 100, 2),
            "CAGR (%)": round(cagr * 100, 2),
            "Annual Volatility (%)": round(volatility * 100, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Max Drawdown (%)": round(max_dd * 100, 2),
            "Calmar Ratio": round(cagr / abs(max_dd), 2) if abs(max_dd) > 0 else 0.0
        }

        portfolio_df = pd.DataFrame({
            "portfolio_return": portfolio_returns,
            "equity_curve": equity_curve,
            "drawdown": drawdown
        })

        return {
            "weights": weights.to_dict(),
            "metrics": metrics,
            "data": portfolio_df,
            "correlation_matrix": clean_returns.corr()
        }

    def combine_trades_prop_firm(
        self,
        strategy_trades: Dict[str, pd.DataFrame],
        initial_capital: float = 50000.0,
        daily_loss_limit: float = 1000.0,
        max_trailing_drawdown: float = 2000.0,
        profit_target: float = 3000.0,
    ) -> Dict[str, Any]:
        """
        Combina los registros de trades reales en USD de 2 o más estrategias para una cuenta de Prop Trading.
        
        Args:
            strategy_trades: Dict {nombre_estrategia: df_trades} con columnas de fecha y net_pnl_usd.
            initial_capital: Capital inicial de la cuenta (USD).
            daily_loss_limit: DLL contractual (USD).
            max_trailing_drawdown: MTDD contractual (USD).
            profit_target: Target contractual (USD).
            
        Returns:
            Dict completo con correlación, reducción de drawdown, PnL consolidado y auditoría Prop Firm.
        """
        if len(strategy_trades) < 2:
            raise ValueError("Se requieren al menos 2 estrategias para formar un portafolio.")

        # 1. Normalizar DataFrames de trades
        daily_series_dict = {}
        all_trades_list = []

        strat_stats = {}

        for name, df in strategy_trades.items():
            if df.empty:
                daily_series_dict[name] = pd.Series(dtype=float)
                strat_stats[name] = {
                    "total_trades": 0,
                    "net_pnl_usd": 0.0,
                    "max_drawdown_usd": 0.0,
                    "win_rate_pct": 0.0,
                }
                continue

            df_clean = df.copy()

            # Encontrar columna de PnL neto
            pnl_col = None
            for c in ["net_pnl_usd", "net_pnl", "net_profit", "pnl", "PnL", "Net_PnL"]:
                if c in df_clean.columns:
                    pnl_col = c
                    break
            if pnl_col is None:
                raise ValueError(f"No se encontró columna de PnL en trades de '{name}'. Columnas: {list(df_clean.columns)}")

            # Encontrar columna de fecha / hora
            if "exit_date" in df_clean.columns and "exit_time" in df_clean.columns:
                combined_dt = df_clean["exit_date"].astype(str) + " " + df_clean["exit_time"].astype(str)
                s_dt = pd.to_datetime(combined_dt, utc=True, format="mixed").dt.tz_localize(None)
                df_clean["_dt"] = s_dt
                df_clean["_date"] = df_clean["_dt"].dt.date
            elif "entry_date" in df_clean.columns and "entry_time" in df_clean.columns:
                combined_dt = df_clean["entry_date"].astype(str) + " " + df_clean["entry_time"].astype(str)
                s_dt = pd.to_datetime(combined_dt, utc=True, format="mixed").dt.tz_localize(None)
                df_clean["_dt"] = s_dt
                df_clean["_date"] = df_clean["_dt"].dt.date
            else:
                time_col = None
                for c in ["exit_time", "entry_time", "exit_date", "entry_date", "date", "Date", "datetime", "timestamp"]:
                    if c in df_clean.columns:
                        time_col = c
                        break

                if time_col is not None:
                    s_dt = pd.to_datetime(df_clean[time_col], utc=True, format="mixed").dt.tz_localize(None)
                    df_clean["_dt"] = s_dt
                    df_clean["_date"] = df_clean["_dt"].dt.date
                else:
                    df_clean["_dt"] = pd.date_range("2023-01-01", periods=len(df_clean), freq="h")
                    df_clean["_date"] = df_clean["_dt"].dt.date

            # PnL diario acumulado por estrategia
            daily_pnl = df_clean.groupby("_date")[pnl_col].sum()
            daily_series_dict[name] = daily_pnl

            # Guardar trades etiquetados para secuencia unificada
            for _, row in df_clean.iterrows():
                all_trades_list.append({
                    "strategy": name,
                    "datetime": row["_dt"],
                    "date": row["_date"],
                    "net_pnl_usd": float(row[pnl_col]),
                })

            # Estadísticas individuales
            cum_pnl = df_clean[pnl_col].cumsum()
            hwm = cum_pnl.cummax()
            dd_series = hwm - cum_pnl
            wins = (df_clean[pnl_col] > 0).sum()
            strat_stats[name] = {
                "total_trades": len(df_clean),
                "net_pnl_usd": round(float(df_clean[pnl_col].sum()), 2),
                "max_drawdown_usd": round(float(dd_series.max() if len(dd_series) > 0 else 0.0), 2),
                "win_rate_pct": round(float((wins / len(df_clean)) * 100 if len(df_clean) > 0 else 0.0), 2),
            }

        # 2. Alinear retornos diarios de las estrategias
        daily_df = pd.DataFrame(daily_series_dict).fillna(0.0)
        daily_df.sort_index(inplace=True)

        # Matriz de Correlación
        corr_matrix = daily_df.corr()
        strat_names = list(strategy_trades.keys())
        pearson_corr = float(corr_matrix.iloc[0, 1]) if len(strat_names) >= 2 else 0.0

        # Ponderaciones de capital
        weights = self.calculate_weights(daily_df if (daily_df.std() > 0).all() else daily_df + 1e-6)

        # 3. PnL diario ponderado del portafolio
        daily_df["portfolio_pnl_usd"] = (daily_df[strat_names] * weights).sum(axis=1)
        daily_df["cum_portfolio_pnl"] = daily_df["portfolio_pnl_usd"].cumsum()
        daily_df["portfolio_equity"] = initial_capital + daily_df["cum_portfolio_pnl"]
        daily_df["portfolio_hwm"] = daily_df["portfolio_equity"].cummax()
        daily_df["portfolio_dd_usd"] = daily_df["portfolio_hwm"] - daily_df["portfolio_equity"]

        portfolio_max_dd = float(daily_df["portfolio_dd_usd"].max()) if not daily_df.empty else 0.0
        portfolio_net_profit = float(daily_df["portfolio_pnl_usd"].sum()) if not daily_df.empty else 0.0

        # 4. Beneficio de Diversificación y Reducción de Drawdown
        # Compara el Max Drawdown del portafolio contra el promedio ponderado y el mínimo de las individuales
        ind_dds = [strat_stats[s]["max_drawdown_usd"] for s in strat_names]
        min_ind_dd = min(ind_dds) if ind_dds else 1.0
        mean_ind_dd = np.mean(ind_dds) if ind_dds else 1.0

        dd_reduction_vs_mean_pct = 0.0
        if mean_ind_dd > 0:
            dd_reduction_vs_mean_pct = max(0.0, ((mean_ind_dd - portfolio_max_dd) / mean_ind_dd) * 100.0)

        # 5. Auditoría de Prop Firm sobre la cuenta conjunta
        dll_violations = (daily_df["portfolio_pnl_usd"] < -daily_loss_limit).sum()
        mtdd_breached = portfolio_max_dd >= max_trailing_drawdown
        target_reached = portfolio_net_profit >= profit_target

        if mtdd_breached:
            compliance_status = "🔴 CUENTA QUEMADA (Violación de Max Trailing Drawdown de $2,000)"
        elif dll_violations > 0:
            compliance_status = f"🔴 VIOLACIÓN DLL ({dll_violations} días excedieron límite de -$1,000)"
        elif target_reached:
            compliance_status = "🟢 PROFIT TARGET ALCANZADO (Aprobada sin violaciones)"
        else:
            compliance_status = "🟢 EN CURSO / CUMPLIMIENTO CONFORME (Sin violaciones)"

        # 6. Secuencia unificada cronológica de trades combinados
        all_trades_df = pd.DataFrame(all_trades_list)
        if not all_trades_df.empty:
            all_trades_df.sort_values(by="datetime", inplace=True)
            # Ponderar PnL de cada trade según el peso de su estrategia
            all_trades_df["weighted_pnl_usd"] = all_trades_df.apply(
                lambda r: r["net_pnl_usd"] * weights.get(r["strategy"], 1.0 / len(strat_names)),
                axis=1
            )
        else:
            all_trades_df = pd.DataFrame(columns=["strategy", "datetime", "date", "net_pnl_usd", "weighted_pnl_usd"])

        return {
            "strategies": strat_names,
            "weights": weights.to_dict(),
            "strat_stats": strat_stats,
            "pearson_correlation": round(pearson_corr, 4),
            "correlation_matrix": corr_matrix,
            "portfolio_metrics": {
                "initial_capital": initial_capital,
                "final_equity": round(initial_capital + portfolio_net_profit, 2),
                "net_profit_usd": round(portfolio_net_profit, 2),
                "portfolio_max_dd_usd": round(portfolio_max_dd, 2),
                "dd_reduction_pct": round(dd_reduction_vs_mean_pct, 2),
                "dll_violations_count": int(dll_violations),
                "mtdd_breached": bool(mtdd_breached),
                "target_reached": bool(target_reached),
                "compliance_status": compliance_status,
            },
            "daily_data": daily_df,
            "unified_trades": all_trades_df,
        }
