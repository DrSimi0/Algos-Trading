import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    raw = z.read("orders.bin")

print(f"Total raw bytes in orders.bin: {len(raw)}")

# SQOrderFileFormat:11 header
# Let's inspect the first 200 bytes
pos = 0
header_end = raw.find(b"\n")
if header_end == -1:
    header_end = raw.find(b"\x00")
print("Header line:", raw[:50])

# Buscar si hay patrones de órdenes
# Cada orden en SQX típicamente contiene:
# - orderId / ticket (int o long)
# - symbol (string)
# - orderType (byte / int)
# - openTime (long ms)
# - openPrice (double)
# - closeTime (long ms)
# - closePrice (double)
# - size (double)
# - profit (double)
# - commission (double)
# - etc.
