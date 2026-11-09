import struct
import zipfile
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    raw = z.read("orders.bin")

# Buscar patrones de órdenes en orders.bin
# Cada orden en SQX serializado suele tener strings como "Entry", "Exit", o números de tipo de orden
print(f"Scanning {len(raw)} bytes of orders.bin...")

# Encontrar posiciones donde aparece SPREAD_NQ_ES.
pos = 0
matches = []
needle = b"SPREAD_NQ_ES."
while True:
    idx = raw.find(needle, pos)
    if idx == -1:
        break
    matches.append(idx)
    pos = idx + len(needle)

print(f"Total occurrences of SPREAD_NQ_ES.: {len(matches)}")

# Analizar los bytes alrededor de los primeros matches
for m in matches[:5]:
    snippet = raw[max(0, m-20) : min(len(raw), m+80)]
    print(f"\nMatch at {m}:")
    print("  Bytes:", snippet.hex())
    # Extract any readable text
    text = "".join(chr(b) if 32 <= b <= 126 else "." for b in snippet)
    print("  Text: ", text)
