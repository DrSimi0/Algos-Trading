import pandas as pd
import json

from src.strategies.f4_optimizacion_is_n3 import (
    F4OptimizationISN3Strategy,
    F4OptimizationISN3Backtester,
)

print("=" * 80)
print("BACKTESTING F4-N3 INTRADIA (K=390, M5) POR AÑO: 2021, 2022, 2023 Y TRIENIO")
print("=" * 80)

df_es = pd.read_parquet("data/raw/CME_MINI_ES_2021_2023_5m.parquet")
df_nq = pd.read_parquet("data/raw/CME_MINI_NQ_2021_2023_5m.parquet")
df_xlk = pd.read_parquet("data/raw/XLK_1d_2021-01-01_2023-12-31.parquet")
df_spy = pd.read_parquet("data/raw/SPY_1d_2021-01-01_2023-12-31.parquet")

for df in [df_es, df_nq, df_xlk, df_spy]:
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)

common = df_nq.index.intersection(df_es.index)
df_nq = df_nq.loc[common]
df_es = df_es.loc[common]

yearly_results = []

# Evaluar cada año individualmente
for yr in [2021, 2022, 2023]:
    sub_nq = df_nq[df_nq.index.year == yr]
    sub_es = df_es[df_es.index.year == yr]
    sub_xlk = df_xlk[df_xlk.index.year == yr]
    sub_spy = df_spy[df_spy.index.year == yr]
    
    strat = F4OptimizationISN3Strategy(params={"contract_mode": "micro", "k_calib": 390, "min_calib_bars": 390})
    bt = F4OptimizationISN3Backtester(
        strategy=strat,
        account_size=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_loss=2000.0,
        profit_target=3000.0,
        early_stopping=False
    )
    res = bt.run(sub_nq, sub_es, df_xlk=sub_xlk, df_spy=sub_spy)
    m = res["metrics"]
    p = res["prop_evaluation"]
    
    print(f"\n--- AÑO {yr} (Barras 5m: {len(sub_nq):,}) ---")
    print(f"Total Trades        : {m.get('Total Trades')}")
    print(f"Win Rate            : {m.get('Win Rate (%)', 0):.2f}%")
    print(f"Net Profit ($)      : ${m.get('Net Profit ($)', 0):,.2f}")
    print(f"Total Return (%)    : {m.get('Total Return (%)', 0):.2f}%")
    print(f"Max Drawdown ($)    : ${m.get('Max Drawdown ($)', 0):,.2f}")
    print(f"Profit Factor       : {m.get('Profit Factor', 0):.2f}")
    print(f"Sharpe (Daily 252d) : {m.get('Sharpe Ratio (Daily 252d)', 0):.2f}")
    print(f"Sortino (Daily 252d): {m.get('Sortino Ratio (Daily 252d)', 0):.2f}")
    print(f"Prop Status         : {p.get('status')} ({p.get('reason')})")
    
    yearly_results.append({
        "year": yr,
        "bars": len(sub_nq),
        "trades": m.get("Total Trades"),
        "win_rate": m.get("Win Rate (%)"),
        "net_pnl": m.get("Net Profit ($)"),
        "return_pct": m.get("Total Return (%)"),
        "max_drawdown": m.get("Max Drawdown ($)"),
        "profit_factor": m.get("Profit Factor"),
        "sharpe_daily": m.get("Sharpe Ratio (Daily 252d)"),
        "sortino_daily": m.get("Sortino Ratio (Daily 252d)"),
        "prop_status": p.get("status"),
        "prop_reason": p.get("reason"),
        "exit_reasons": m.get("Exit Reasons", {})
    })

# Periodo consolidado 2021-2023
strat_full = F4OptimizationISN3Strategy(params={"contract_mode": "micro", "k_calib": 390, "min_calib_bars": 390})
bt_full = F4OptimizationISN3Backtester(
    strategy=strat_full,
    account_size=50000.0,
    daily_loss_limit=1000.0,
    max_trailing_loss=2000.0,
    profit_target=3000.0,
    early_stopping=False
)
res_full = bt_full.run(df_nq, df_es, df_xlk=df_xlk, df_spy=df_spy)
m_full = res_full["metrics"]
p_full = res_full["prop_evaluation"]

yearly_results.append({
    "year": "2021-2023_Consolidado",
    "bars": len(df_nq),
    "trades": m_full.get("Total Trades"),
    "win_rate": m_full.get("Win Rate (%)"),
    "net_pnl": m_full.get("Net Profit ($)"),
    "return_pct": m_full.get("Total Return (%)"),
    "max_drawdown": m_full.get("Max Drawdown ($)"),
    "profit_factor": m_full.get("Profit Factor"),
    "sharpe_daily": m_full.get("Sharpe Ratio (Daily 252d)"),
    "sortino_daily": m_full.get("Sortino Ratio (Daily 252d)"),
    "prop_status": p_full.get("status"),
    "prop_reason": p_full.get("reason"),
    "exit_reasons": m_full.get("Exit Reasons", {})
})

with open("reports/f4_n3_intraday_2021_2023_summary.json", "w") as f:
    json.dump(yearly_results, f, indent=2)

print("\n[OK] Resultados exportados a reports/f4_n3_intraday_2021_2023_summary.json")
