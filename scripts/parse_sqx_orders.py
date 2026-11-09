import struct
import zipfile
from pathlib import Path

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    orders_bytes = z.read("orders.bin")
    print(f"orders.bin size: {len(orders_bytes)} bytes")

    # Inspeccionar strings o estructura en orders.bin
    # Las órdenes en SQX suelen guardarse como objetos serializados en Java o struct binario
    printable = ""
    strings_found = []
    current_str = []
    for b in orders_bytes:
        if 32 <= b <= 126:
            current_str.append(chr(b))
        else:
            if len(current_str) >= 4:
                strings_found.append("".join(current_str))
            current_str = []

    print(f"Found {len(strings_found)} strings in orders.bin")
    print("Sample strings:", strings_found[:40])
