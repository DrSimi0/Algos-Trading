import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    s_xml = z.read("settings.xml").decode("utf-8", errors="ignore")

root = ET.fromstring(s_xml)

print("Settings XML elements:")
for elem in root.iter():
    # Print elements that have attributes or text
    if elem.tag not in ["SQStats"]:
        txt = (elem.text or "").strip()
        attribs = " ".join(f"{k}={v}" for k, v in elem.attrib.items())
        if txt or attribs:
            print(f"<{elem.tag} {attribs}> {txt[:60]}")
