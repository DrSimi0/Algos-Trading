import struct
from datetime import datetime, timezone
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    raw = z.read("orders.bin")

# Analizar la primera orden
# pos=152 es open_time
# Vamos a ver qué hay alrededor de pos 152:
# 152 - 12: ticket/orderId?
# Busquemos los floats / doubles entre pos 152 y 152+149
records = []
pos = 143 # inicio de primer registro
# Estructura del chunk:
# Buscamos cada ocurrencia de timestamp
for i in range(len(raw) - 100):
    # Detectar el patrón: open_ts a i, open_fill_ts a i+17, close_ts a i+29
    if i + 37 <= len(raw):
        ts1 = struct.unpack(">q", raw[i:i+8])[0]
        ts2 = struct.unpack(">q", raw[i+17:i+25])[0]
        ts3 = struct.unpack(">q", raw[i+29:i+37])[0]
        if (1672531200000 <= ts1 <= 1767225600000 and 
            ts1 == ts2 and 
            1672531200000 <= ts3 <= 1767225600000 and 
            ts3 >= ts1):
            
            # Hemos localizado una orden válida!
            # Extraer campos
            # Delante de ts1: direction, orderId
            # open_price, close_price, profit, etc.
            open_dt = datetime.fromtimestamp(ts1 / 1000.0, tz=timezone.utc)
            close_dt = datetime.fromtimestamp(ts3 / 1000.0, tz=timezone.utc)
            
            # Buscar doubles en el bloque de 100 bytes posteriores a ts3
            chunk = raw[i+37 : i+140]
            # Encontrar profit buscando doubles
            # Vamos a ver la estructura exacta del primer match
            if len(records) == 0:
                print("Primer match en i =", i)
                print("Byte antes de ts1 (direction?):", list(raw[i-10:i]))
                print("Bytes entre ts1 y ts2:", list(raw[i+8:i+17]))
                print("Bytes entre ts2 y ts3:", list(raw[i+25:i+29]))
            
            # Intentar decodificar open_price y close_price
            # Open price está entre ts1 y ts2? raw[i+9:i+17] es 8 bytes double!
            open_p = struct.unpack(">d", raw[i+9:i+17])[0]
            close_p = struct.unpack(">d", raw[i+21:i+29])[0] if i+29 <= len(raw) else 0
            
            records.append({
                "pos": i,
                "open_time": open_dt,
                "close_time": close_dt,
                "open_p": round(open_p, 5) if abs(open_p) < 100 else 0,
                "close_p": round(close_p, 5) if abs(close_p) < 100 else 0
            })

print(f"Total matching order timestamps found: {len(records)}")
if records:
    df_rec = pd.DataFrame(records)
    print(df_rec.head(10))
    print(df_rec.tail(5))
