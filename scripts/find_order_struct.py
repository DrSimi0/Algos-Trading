import struct
from datetime import datetime, timezone
from pathlib import Path
import zipfile

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    raw = z.read("orders.bin")

# Empezamos después del header (alrededor del offset 125)
print("Hex from 110 to 300:")
print(raw[110:300].hex())

# Buscar secuencias de timestamps de apertura y cierre (timestamps de 2023 empiezan con 0x00000185... o 0x00000186...)
ts_positions = []
for i in range(len(raw) - 8):
    ts = struct.unpack(">q", raw[i:i+8])[0]
    if 1672531200000 <= ts <= 1767225600000:
        ts_positions.append((i, ts))

print(f"Found {len(ts_positions)} timestamp occurrences!")
print("First 10 timestamp positions:")
for idx, (p, ts) in enumerate(ts_positions[:10]):
    dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
    print(f"  [{idx}] pos={p} | ts={ts} -> {dt.strftime('%Y-%m-%d %H:%M:%S')}")

# Analizar la distancia entre timestamps consecutivos
diffs = [ts_positions[i+1][0] - ts_positions[i][0] for i in range(min(50, len(ts_positions)-1))]
print("Distance between consecutive timestamps:", diffs[:20])
