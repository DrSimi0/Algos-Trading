import base64
import struct
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from pathlib import Path

sqx_path = Path("exports/strategyquant/F4-EstrategiaCointegracion ESNQ.sqx")

with zipfile.ZipFile(sqx_path, "r") as z:
    for name in z.namelist():
        print(f"File in zip: {name} ({z.getinfo(name).file_size} bytes)")

    # 1. settings.xml
    s_xml = z.read("settings.xml").decode("utf-8", errors="ignore")
    s_root = ET.fromstring(s_xml)

    print("\n" + "=" * 60)
    print("NEW BACKTEST SETTINGS & METRICS IN SQX")
    print("=" * 60)

    # Fingerprint
    fp = s_root.find(".//Fingerprint")
    if fp is not None:
        print("Fingerprint attributes:")
        for k, v in fp.attrib.items():
            print(f"  {k}: {v}")

    # Fitnesses
    fit = s_root.find(".//Fitnesses")
    if fit is not None:
        print("Fitnesses:")
        for k, v in fit.attrib.items():
            if float(v) != 0:
                print(f"  {k}: {v}")

    # Parametros clave de ejecucion
    for elem in s_root.iter():
        if elem.tag in ["Commission", "Slippage", "MoneyManagement.InitialCapital", "AddCommissionSwapToPL"]:
            print(f"{elem.tag}: {elem.text or elem.attrib}")

    # InstrumentInfo
    for inst in s_root.iter("InstrumentInfo"):
        print(f"Instrument: {inst.attrib.get('instrument')} | pointValue={inst.attrib.get('pointValue')} | defaultSpread={inst.attrib.get('defaultSpread')} | defaultSlippage={inst.attrib.get('defaultSlippage')}")
        for comm in inst.iter("commissions"):
            print("  Commissions:", ET.tostring(comm, encoding="unicode").strip())

    # 2. Daily Equity
    print("\n" + "=" * 60)
    print("NEW DAILY EQUITY ANALYSIS")
    print("=" * 60)
    eq_name = [n for n in z.namelist() if "dailyEquity.bin" in n]
    if eq_name:
        raw_eq = z.read(eq_name[0])
        print(f"dailyEquity size: {len(raw_eq)} bytes")
        # Parse all timestamp records
        pos = 0
        valid_ts = []
        for i in range(len(raw_eq) - 16):
            ts = struct.unpack(">q", raw_eq[i:i+8])[0]
            if 1672531200000 <= ts <= 1767225600000:
                val = struct.unpack(">d", raw_eq[i+8:i+16])[0]
                if abs(val) < 1000000:
                    dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
                    valid_ts.append({"date": dt, "equity_pnl": val})
        
        # Eliminar duplicados si los hubiera
        if valid_ts:
            df_eq = pd.DataFrame(valid_ts).drop_duplicates(subset=["date"]).sort_values("date").set_index("date")
            print(f"Total Daily Bars: {len(df_eq)}")
            print(f"Start Date: {df_eq.index[0]}")
            print(f"End Date:   {df_eq.index[-1]}")
            print(f"Final Net Profit: ${df_eq['equity_pnl'].iloc[-1]:,.2f}")
            print(f"Peak Equity:       ${df_eq['equity_pnl'].max():,.2f}")
            print(f"Min Equity:        ${df_eq['equity_pnl'].min():,.2f}")
            running_max = df_eq["equity_pnl"].cummax()
            dd = running_max - df_eq["equity_pnl"]
            print(f"Max Daily Closed Drawdown: ${dd.max():,.2f}")

            # Anual
            for yr in sorted(df_eq.index.year.unique()):
                sub = df_eq[df_eq.index.year == yr]
                start_val = sub["equity_pnl"].iloc[0]
                end_val = sub["equity_pnl"].iloc[-1]
                net_yr = end_val - (df_eq.loc[df_eq.index < sub.index[0], "equity_pnl"].iloc[-1] if len(df_eq.loc[df_eq.index < sub.index[0]]) > 0 else 0)
                print(f"  Año {yr}: Final PnL = ${end_val:,.2f} | Ganancia Año = ${net_yr:,.2f} | Días: {len(sub)}")

            daily_diff = df_eq["equity_pnl"].diff().dropna()
            sharpe = (daily_diff.mean() / daily_diff.std()) * np.sqrt(252) if daily_diff.std() > 0 else 0
            print(f"Sharpe Ratio Diario Anualizado: {sharpe:.2f}")

    # 3. Rules check: Did the user update the rules?
    print("\n" + "=" * 60)
    print("RULES INSPECTION IN strategy_Portfolio.xml")
    print("=" * 60)
    strat_xml = z.read("strategy_Portfolio.xml").decode("utf-8", errors="ignore")
    strat_root = ET.fromstring(strat_xml)
    for rule in strat_root.iter("Rule"):
        rname = rule.attrib.get("name", "Unnamed")
        print(f"\nRule: {rname}")
        for item in rule.iter("Item"):
            iname = item.attrib.get("name", item.attrib.get("key"))
            display = item.attrib.get("display", "")
            params = {p.attrib.get("key"): p.text for p in item.iter("Param")}
            if "Time" in str(params) or "154" in str(params) or "CloseAll" in iname or "Enter" in iname:
                print(f"  Item: {iname} | params: {params}")

    # 4. Orders analysis
    print("\n" + "=" * 60)
    print("ORDERS ANALYSIS (orders.bin)")
    print("=" * 60)
    raw_ord = z.read("orders.bin")
    print(f"orders.bin size: {len(raw_ord)} bytes")
    # Parse orders timestamps
    orders = []
    for i in range(len(raw_ord) - 100):
        if i + 37 <= len(raw_ord):
            ts1 = struct.unpack(">q", raw_ord[i:i+8])[0]
            ts2 = struct.unpack(">q", raw_ord[i+17:i+25])[0]
            ts3 = struct.unpack(">q", raw_ord[i+29:i+37])[0]
            if (1672531200000 <= ts1 <= 1767225600000 and 
                ts1 == ts2 and 
                1672531200000 <= ts3 <= 1767225600000 and 
                ts3 >= ts1):
                open_dt = datetime.fromtimestamp(ts1 / 1000.0, tz=timezone.utc)
                close_dt = datetime.fromtimestamp(ts3 / 1000.0, tz=timezone.utc)
                dur = (close_dt - open_dt).total_seconds() / 60.0
                orders.append({
                    "open": open_dt, "close": close_dt, "dur": dur,
                    "is_instant": dur == 0,
                    "is_post_1545": (open_dt.hour == 15 and open_dt.minute >= 45)
                })

    df_o = pd.DataFrame(orders)
    print(f"Total Decoded Orders: {len(df_o)}")
    if len(df_o) > 0:
        print(f"Instant Orders (dur=0): {df_o['is_instant'].sum()} ({(df_o['is_instant'].sum()/len(df_o))*100:.1f}%)")
        print(f"Post 15:45 Orders:      {df_o['is_post_1545'].sum()} ({(df_o['is_post_1545'].sum()/len(df_o))*100:.1f}%)")
        df_real = df_o[~df_o["is_instant"]]
        print(f"Real Non-Instant Orders: {len(df_real)}")
        print(f"Mean Duration: {df_real['dur'].mean():.1f} min")
        for yr in sorted(df_o['open'].dt.year.unique()):
            print(f"  Año {yr}: {(df_o['open'].dt.year == yr).sum()} trades")
