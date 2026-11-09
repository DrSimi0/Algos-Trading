import struct
import zipfile
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from pathlib import Path

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    raw = z.read("Results/Main: SPREAD_NQ_ES._LOM_M5/dailyEquity.bin")

pos = 13
records = []
while pos + 16 <= len(raw):
    ts, val = struct.unpack(">qd", raw[pos:pos+16])
    if ts > 1000000000000 and ts < 2500000000000:
        dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        records.append({"date": dt, "equity_pnl": val})
    else:
        # Fin de registros válidos
        break
    pos += 16

df_eq = pd.DataFrame(records)
df_eq.set_index("date", inplace=True)

print(f"Total Daily Bars in SQX: {len(df_eq)}")
print(f"Start Date: {df_eq.index[0]}")
print(f"End Date:   {df_eq.index[-1]}")
print(f"Final Net Profit in SQX: ${df_eq['equity_pnl'].iloc[-1]:,.2f}")
print(f"Peak Equity in SQX:       ${df_eq['equity_pnl'].max():,.2f}")
print(f"Max Trough (Lowest PnL):  ${df_eq['equity_pnl'].min():,.2f}")

# Calcular Max Drawdown diario en SQX
running_max = df_eq["equity_pnl"].cummax()
drawdown = running_max - df_eq["equity_pnl"]
print(f"Max Daily Closed Drawdown in SQX: ${drawdown.max():,.2f}")

# Desglose por año en SQX
for yr in sorted(df_eq.index.year.unique()):
    sub = df_eq[df_eq.index.year == yr]
    start_val = sub["equity_pnl"].iloc[0]
    end_val = sub["equity_pnl"].iloc[-1]
    net_yr = end_val - (df_eq.loc[df_eq.index < sub.index[0], "equity_pnl"].iloc[-1] if len(df_eq.loc[df_eq.index < sub.index[0]]) > 0 else 0)
    print(f"  Año {yr}: Final PnL = ${end_val:,.2f} | Ganancia Año = ${net_yr:,.2f} | Días: {len(sub)}")

# Calcular Sharpe diario de SQX (asumiendo capital base $100,000)
daily_diff = df_eq["equity_pnl"].diff().dropna()
mean_daily = daily_diff.mean()
std_daily = daily_diff.std()
sharpe_sqx = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
print(f"Sharpe Ratio Diario Anualizado en SQX (sobre 3Y): {sharpe_sqx:.2f}")

# Guardar curva de equidad diaria de SQX
df_eq.to_csv("exports/strategyquant/sqx_daily_equity.csv")
print("Saved exports/strategyquant/sqx_daily_equity.csv")
