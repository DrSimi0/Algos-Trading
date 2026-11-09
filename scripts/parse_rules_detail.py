import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    xml_str = z.read("strategy_Portfolio.xml").decode("utf-8", errors="ignore")

root = ET.fromstring(xml_str)

for rule in root.iter("Rule"):
    rname = rule.attrib.get("name", "Unnamed")
    print(f"\n==================== RULE: {rname} ====================")
    for item in rule.iter("Item"):
        iname = item.attrib.get("name", item.attrib.get("key"))
        display = item.attrib.get("display", "")
        params = {p.attrib.get("key"): p.text for p in item.iter("Param")}
        print(f"  Item: {iname} | display: {display} | params: {params}")
