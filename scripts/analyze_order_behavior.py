import struct
from datetime import datetime, timezone
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    raw = z.read("orders.bin")

records = []
for i in range(len(raw) - 100):
    if i + 37 <= len(raw):
        ts1 = struct.unpack(">q", raw[i:i+8])[0]
        ts2 = struct.unpack(">q", raw[i+17:i+25])[0]
        ts3 = struct.unpack(">q", raw[i+29:i+37])[0]
        if (1672531200000 <= ts1 <= 1767225600000 and 
            ts1 == ts2 and 
            1672531200000 <= ts3 <= 1767225600000 and 
            ts3 >= ts1):
            
            open_dt = datetime.fromtimestamp(ts1 / 1000.0, tz=timezone.utc)
            close_dt = datetime.fromtimestamp(ts3 / 1000.0, tz=timezone.utc)
            duration_min = (close_dt - open_dt).total_seconds() / 60.0
            
            records.append({
                "open_time": open_dt,
                "close_time": close_dt,
                "duration_min": duration_min,
                "open_hour": open_dt.hour,
                "open_minute": open_dt.minute,
                "is_eod_spam": (open_dt.hour == 15 and open_dt.minute >= 45) or duration_min == 0
            })

df_orders = pd.DataFrame(records)
print(f"Total Decoded Orders: {len(df_orders)}")
print(f"Orders with 0 min duration (Instant Open & Close): {(df_orders['duration_min'] == 0).sum()} ({((df_orders['duration_min'] == 0).sum() / len(df_orders))*100:.1f}%)")
print(f"Orders opened at or after 15:45: {(df_orders['is_eod_spam']).sum()} ({((df_orders['is_eod_spam']).sum() / len(df_orders))*100:.1f}%)")

# Filtrar órdenes normales (sin las de cierre instantáneo EOD)
df_valid = df_orders[~df_orders["is_eod_spam"]]
print(f"\nOrders REALES (sin apertura-cierre simultáneo de las 15:45+): {len(df_valid)}")
print(f"Promedio de duración de órdenes reales: {df_valid['duration_min'].mean():.1f} minutos")

# Conteo por año de órdenes reales
for yr in sorted(df_valid["open_time"].dt.year.unique()):
    cnt_real = (df_valid["open_time"].dt.year == yr).sum()
    cnt_total = (df_orders["open_time"].dt.year == yr).sum()
    print(f"  Año {yr}: {cnt_real} órdenes reales (de {cnt_total} totales con spam EOD)")
