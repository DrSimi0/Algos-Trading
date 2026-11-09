import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.strategies.f4_optimizacion_is_n2 import (
    F4OptimizationISN2Strategy,
    F4OptimizationISN2Backtester,
)

df_nq = pd.read_parquet(ROOT / "data/raw/CME_MINI_NQ_2021_2023_5m.parquet")
df_es = pd.read_parquet(ROOT / "data/raw/CME_MINI_ES_2021_2023_5m.parquet")

# Probar con cuenta de 100k (TMDD $3,000 / $4,500) y ver cuántos trades ejecuta
strat = F4OptimizationISN2Strategy()
bt_100k = F4OptimizationISN2Backtester(
    account_size=100000.0,
    daily_loss_limit=2000.0,
    max_trailing_loss=4500.0,
    profit_target=6000.0,
    early_stopping=False,
)
res = bt_100k.run(df_nq, df_es)
tdf = res["trades"]
m = res["metrics"]
pe = res["prop_evaluation"]

print("=== CUENTA 100K / TMDD $4,500 ===")
print(f"Estado: {pe['status']} ({pe.get('reason')})")
print(f"Total trades: {len(tdf)}")
print(f"Net Profit: ${pe['net_profit']:,.2f}")
print(f"Win Rate: {m.get('win_rate', 0)*100:.2f}%")
print(f"Profit Factor: {m.get('profit_factor', 0):.2f}")
print(f"Sharpe (252d): {m.get('sharpe_ratio_daily_252d', 0):.2f}")

# Desglose por año
if len(tdf) > 0:
    tdf["yr"] = pd.to_datetime(tdf["entry_date"]).dt.year
    for yr in sorted(tdf["yr"].unique()):
        sub = tdf[tdf["yr"] == yr]
        print(f"  Año {yr}: {len(sub)} trades | PnL: ${sub['net_pnl_usd'].sum():,.2f}")
