import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    xml_str = z.read("strategy_Portfolio.xml").decode("utf-8", errors="ignore")

root = ET.fromstring(xml_str)

for rule in root.iter("Rule"):
    rule_name = rule.attrib.get("name", "Unnamed")
    print("\n" + "=" * 70)
    print(f"REGLA: {rule_name}")
    print("=" * 70)

    # Convert rule to string or inspect children
    for child in rule:
        print(f"\n--- Seccion: <{child.tag}> ---")
        xml_child = ET.tostring(child, encoding="unicode")
        # Format / clean up lines
        for line in xml_child.splitlines()[:60]:
            print("  ", line)
        if len(xml_child.splitlines()) > 60:
            print("   ... (truncated)")
