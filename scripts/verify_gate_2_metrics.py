"""
scripts/verify_gate_2_metrics.py
Independent verification script for Gate 2 review:
Validates Prop Trading compliance, statistical plausibility, and portfolio diversification.
"""

import pandas as pd
import numpy as np

def run_audit():
    print("=" * 60)
    print("1. F3-N3 STRATEGY AUDIT (reports/f3_optimization_n3_trades.csv)")
    print("=" * 60)
    df_f3 = pd.read_csv("reports/f3_optimization_n3_trades.csv")
    print(f"Total Trades: {len(df_f3)}")
    net_pnl_f3 = df_f3["net_pnl"].sum()
    gross_win_f3 = df_f3[df_f3["net_pnl"] > 0]["net_pnl"].sum()
    gross_loss_f3 = abs(df_f3[df_f3["net_pnl"] < 0]["net_pnl"].sum())
    pf_f3 = gross_win_f3 / gross_loss_f3 if gross_loss_f3 > 0 else 0.0
    win_rate_f3 = (df_f3["net_pnl"] > 0).mean() * 100
    cum_f3 = df_f3["net_pnl"].cumsum()
    tmdd_f3 = float((cum_f3.cummax() - cum_f3).max())
    
    df_f3["exit_dt"] = pd.to_datetime(df_f3["exit_time"])
    daily_f3 = df_f3.groupby(df_f3["exit_dt"].dt.date)["net_pnl"].sum()
    worst_day_f3 = float(daily_f3.min())
    dll_viol_f3 = int((daily_f3 < -1000.0).sum())
    
    print(f"Net PnL (USD): {net_pnl_f3:,.2f}")
    print(f"Gross Win: ${gross_win_f3:,.2f} | Gross Loss: ${gross_loss_f3:,.2f}")
    print(f"Profit Factor: {pf_f3:.2f}")
    print(f"Win Rate: {win_rate_f3:.2f}%")
    print(f"Max Trailing DD (trade-level): ${tmdd_f3:,.2f} (Limit: <= $2,000.00 -> {'PASS' if tmdd_f3 <= 2000 else 'FAIL'})")
    print(f"Worst Daily Loss: ${worst_day_f3:,.2f} (Limit: <= $1,000.00 -> {'PASS' if worst_day_f3 >= -1000 else 'FAIL'})")
    print(f"DLL Violations count: {dll_viol_f3}")
    
    print("\n" + "=" * 60)
    print("2. F4-N2 STRATEGY AUDIT (reports/f4_optimizacion_is_n2_trades.csv)")
    print("=" * 60)
    df_f4 = pd.read_csv("reports/f4_optimizacion_is_n2_trades.csv")
    print(f"Total Trades: {len(df_f4)}")
    net_pnl_f4 = df_f4["net_pnl_usd"].sum()
    gross_win_f4 = df_f4[df_f4["net_pnl_usd"] > 0]["net_pnl_usd"].sum()
    gross_loss_f4 = abs(df_f4[df_f4["net_pnl_usd"] < 0]["net_pnl_usd"].sum())
    pf_f4 = gross_win_f4 / gross_loss_f4 if gross_loss_f4 > 0 else 0.0
    win_rate_f4 = (df_f4["net_pnl_usd"] > 0).mean() * 100
    cum_f4 = df_f4["net_pnl_usd"].cumsum()
    tmdd_f4 = float((cum_f4.cummax() - cum_f4).max())
    
    df_f4["exit_dt"] = pd.to_datetime(df_f4["exit_date"].astype(str) + " " + df_f4["exit_time"].astype(str))
    daily_f4 = df_f4.groupby(df_f4["exit_dt"].dt.date)["net_pnl_usd"].sum()
    worst_day_f4 = float(daily_f4.min())
    dll_viol_f4 = int((daily_f4 < -1500.0).sum())
    
    all_dates = pd.date_range("2021-01-04", "2023-12-29", freq="B").date
    aligned_f4 = daily_f4.reindex(all_dates, fill_value=0.0)
    sharpe_daily_f4 = float((aligned_f4.mean() / aligned_f4.std()) * np.sqrt(252)) if aligned_f4.std() > 0 else 0.0
    per_trade_sharpe_f4 = float(df_f4["net_pnl_usd"].mean() / df_f4["net_pnl_usd"].std())
    
    print(f"Net PnL (USD): {net_pnl_f4:,.2f}")
    print(f"Gross Win: ${gross_win_f4:,.2f} | Gross Loss: ${gross_loss_f4:,.2f}")
    print(f"Profit Factor: {pf_f4:.2f}")
    print(f"Win Rate: {win_rate_f4:.2f}%")
    print(f"Max Trailing DD (trade-level): ${tmdd_f4:,.2f} (Limit: <= $3,000.00 -> {'PASS' if tmdd_f4 <= 3000 else 'FAIL'})")
    print(f"Worst Daily Loss: ${worst_day_f4:,.2f} (Limit: <= $1,500.00 -> {'PASS' if worst_day_f4 >= -1500 else 'FAIL'})")
    print(f"DLL Violations count: {dll_viol_f4}")
    print(f"Annualized Daily Sharpe (252d): {sharpe_daily_f4:.2f}")
    print(f"Per-Trade Sharpe: {per_trade_sharpe_f4:.4f}")
    
    print("\n" + "=" * 60)
    print("3. PORTFOLIO ORTHOGONALITY & COMBINED RISK AUDIT")
    print("=" * 60)
    aligned_df = pd.DataFrame({"F3": daily_f3, "F4": daily_f4}).fillna(0.0)
    pearson_corr = float(aligned_df["F3"].corr(aligned_df["F4"]))
    
    # Portfolio allocation: Equal weight (50% F3, 50% F4)
    port_daily = 0.5 * aligned_df["F3"] + 0.5 * aligned_df["F4"]
    cum_port = port_daily.cumsum()
    port_tmdd = float((cum_port.cummax() - cum_port).max())
    mean_ind_dd = (tmdd_f3 + tmdd_f4) / 2.0
    dd_reduction = ((mean_ind_dd - port_tmdd) / mean_ind_dd) * 100
    port_worst_day = float(port_daily.min())
    port_dll_viol = int((port_daily < -1500.0).sum())
    
    print(f"Pearson Correlation (F3 vs F4): {pearson_corr:.4f}")
    print(f"Combined Max Drawdown: ${port_tmdd:,.2f}")
    print(f"Mean Individual Max Drawdown: ${mean_ind_dd:,.2f}")
    print(f"Drawdown Reduction: {dd_reduction:.2f}%")
    print(f"Portfolio Worst Day: ${port_worst_day:,.2f} (Limit: <= $1,500.00 -> {'PASS' if port_worst_day >= -1500 else 'FAIL'})")
    print(f"Portfolio DLL Violations count: {port_dll_viol}")
    
    print("\n" + "=" * 60)
    print("4. INTEGRITY & DATA ARTIFACT AUDIT")
    print("=" * 60)
    # Check for hardcoded test results, zero frictions, duplicate timestamps
    dup_f3 = df_f3.duplicated(subset=["entry_time", "exit_time"]).sum()
    dup_f4 = df_f4.duplicated(subset=["entry_date", "entry_time", "exit_date", "exit_time"]).sum()
    zero_frict_f3 = (df_f3["friction"] <= 0).sum()
    zero_frict_f4 = (df_f4["friction_usd"] <= 0).sum()
    print(f"F3 Duplicate trades: {dup_f3} | Zero friction trades: {zero_frict_f3}")
    print(f"F4 Duplicate trades: {dup_f4} | Zero friction trades: {zero_frict_f4}")

if __name__ == "__main__":
    run_audit()
