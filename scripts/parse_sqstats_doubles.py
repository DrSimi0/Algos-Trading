import base64
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    s_xml = z.read("settings.xml").decode("utf-8", errors="ignore")

root = ET.fromstring(s_xml)

# Buscar SQStats
for stat_elem in root.iter():
    if "stats_" in stat_elem.tag:
        sqstats = stat_elem.find("SQStats")
        if sqstats is not None and sqstats.text:
            data = base64.b64decode(sqstats.text.strip())
            # Intentar extraer todos los floats y doubles válidos
            print(f"\nTag: {stat_elem.tag[:50]}... Data len: {len(data)}")
            doubles = []
            for i in range(0, len(data) - 8, 4):
                val_d = struct.unpack(">d", data[i:i+8])[0]
                if abs(val_d) > 0.001 and abs(val_d) < 1e8 and not (val_d != val_d):
                    doubles.append((i, round(val_d, 4)))
            print(f"Sample parsed doubles: {doubles[:25]}")
            break
