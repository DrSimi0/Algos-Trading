import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    raw = z.read("Results/Main: SPREAD_NQ_ES._LOM_M5/dailyEquity.bin")

# Header es 12 bytes
# byte 10 y 11: count (short) -> 782
magic = raw[:4]
count = struct.unpack(">H", raw[10:12])[0]
print(f"Daily records count: {count}")

records = []
offset = 12
for i in range(count):
    if offset + 16 > len(raw):
        break
    ts_ms, val = struct.unpack(">qd", raw[offset:offset+16])
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    records.append((dt, val))
    offset += 16

print(f"Parsed {len(records)} daily equity records:")
print("First 5 days:")
for dt, val in records[:5]:
    print(f"  {dt.strftime('%Y-%m-%d')} -> Daily Equity/PnL: {val:,.2f}")

print("\nLast 5 days:")
for dt, val in records[-5:]:
    print(f"  {dt.strftime('%Y-%m-%d')} -> Daily Equity/PnL: {val:,.2f}")

# Calcular métricas acumuladas de Daily Equity en SQX
vals = [v for dt, v in records]
total_pnl = vals[-1] if vals else 0
print(f"\nFinal Equity PnL en SQX: ${total_pnl:,.2f}")
