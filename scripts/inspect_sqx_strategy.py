import base64
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    for name in z.namelist():
        print(f"File in zip: {name} ({z.getinfo(name).file_size} bytes)")

    # 1. Analizar strategy_Portfolio.xml
    strat_xml = z.read("strategy_Portfolio.xml").decode("utf-8", errors="ignore")
    root = ET.fromstring(strat_xml)

    print("\n" + "=" * 60)
    print("STRATEGY FILE STRUCTURE & RULES")
    print("=" * 60)

    # Info de la estrategia
    strat_elem = root.find("Strategy")
    if strat_elem is not None:
        print("Strategy Name:", strat_elem.attrib.get("name"))
        print("Engine:", strat_elem.attrib.get("engine"))

    mm = root.find(".//MoneyManagement")
    if mm is not None:
        print("Money Management Type:", mm.attrib.get("type"))
        for p in mm.iter("Param"):
            print(f"  MM Param {p.attrib.get('key')}: {p.text}")
        init_cap = mm.find("InitialCapital")
        if init_cap is not None:
            print("  Initial Capital:", init_cap.text)

    # Reglas
    for task_grp in root.iter("TaskGroup"):
        print(f"\nTaskGroup: {task_grp.attrib.get('name')}")
        for rule in task_grp.iter("Rule"):
            print(f"  Rule: {rule.attrib.get('name')}")

    for rule in root.iter("Rule"):
        rule_name = rule.attrib.get("name", "Unnamed")
        print(f"\n--- REGLA: {rule_name} ---")
        # Condiciones
        conditions = rule.find("Conditions")
        if conditions is not None:
            print("  [CONDICIONES]:")
            for cond in conditions.iter():
                if cond.tag in ["Condition", "Comparison", "Operator", "Left", "Right", "Function"]:
                    txt = (cond.text or "").strip()
                    attribs = " ".join(f"{k}={v}" for k, v in cond.attrib.items())
                    if attribs or txt:
                        print(f"    <{cond.tag} {attribs}> {txt}")
        # Acciones
        actions = rule.find("Actions")
        if actions is not None:
            print("  [ACCIONES]:")
            for act in actions.iter():
                if act.tag in ["Action", "Order", "Param", "Method"]:
                    txt = (act.text or "").strip()
                    attribs = " ".join(f"{k}={v}" for k, v in act.attrib.items())
                    print(f"    <{act.tag} {attribs}> {txt}")

    # 2. Analizar settings.xml
    print("\n" + "=" * 60)
    print("SETTINGS & BACKTEST STATS")
    print("=" * 60)
    settings_xml = z.read("settings.xml").decode("utf-8", errors="ignore")
    s_root = ET.fromstring(settings_xml)

    for res in s_root.iter("Result"):
        print("Result Key:", res.attrib.get("resultKey"))

    for stats in s_root.iter("SQStats"):
        ver = stats.attrib.get("version")
        e = stats.attrib.get("e")
        b64_data = stats.text.strip() if stats.text else ""
        print(f"SQStats (version={ver}, encoding={e}, len={len(b64_data)})")
        if e == "b64":
            raw_bytes = base64.b64decode(b64_data)
            print(f"  Raw Decoded Bytes Length: {len(raw_bytes)}")
            # Intentar buscar strings o números en los bytes
            printable = "".join(chr(b) if 32 <= b <= 126 else "." for b in raw_bytes[:200])
            print("  Preview ASCII:", printable)

    # 3. Analizar lastSettings.xml
    print("\n" + "=" * 60)
    print("LAST SETTINGS PARAMETERS")
    print("=" * 60)
    last_xml = z.read("lastSettings.xml").decode("utf-8", errors="ignore")
    l_root = ET.fromstring(last_xml)
    for p in l_root.iter("Param"):
        val = (p.text or "").strip()
        if val and val != "false" and val != "0":
            print(f"  {p.attrib.get('key')}: {val}")
