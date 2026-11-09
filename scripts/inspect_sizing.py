import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    strat_xml = z.read("strategy_Portfolio.xml").decode("utf-8", errors="ignore")
    settings_xml = z.read("settings.xml").decode("utf-8", errors="ignore")

s_root = ET.fromstring(strat_xml)
print("=== Money Management in strategy_Portfolio.xml ===")
mm = s_root.find(".//MoneyManagement")
if mm is not None:
    print("MM Type:", mm.attrib.get("type"))
    for p in mm.iter("Param"):
        print(f"  Param {p.attrib.get('key')}: {p.text}")
    init_cap = mm.find("InitialCapital")
    if init_cap is not None:
        print("  Initial Capital:", init_cap.text)

print("\n=== Money Management in settings.xml ===")
set_root = ET.fromstring(settings_xml)
for elem in set_root.iter():
    if "MoneyManagement" in elem.tag or "Size" in elem.tag or "pointValue" in elem.attrib:
        print(f"<{elem.tag} {elem.attrib}>: {elem.text}")
