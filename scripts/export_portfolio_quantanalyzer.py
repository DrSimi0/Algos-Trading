import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

f3_path = ROOT / "reports/f3_optimization_n3_trades.csv"
f4_path = ROOT / "reports/f4_optimizacion_is_n3_trades.csv"

all_records = []
ticket = 1000

# 1. Process F3-N3 trades
if f3_path.exists():
    df3 = pd.read_csv(f3_path)
    f3_records = []
    for _, r in df3.iterrows():
        open_dt = pd.to_datetime(r["entry_time"])
        close_dt = pd.to_datetime(r["exit_time"])
        open_time = open_dt.strftime("%Y.%m.%d %H:%M")
        close_time = close_dt.strftime("%Y.%m.%d %H:%M")
        t_type = "buy" if str(r["direction"]).upper() == "LONG" else "sell"
        size = int(r["contracts"])
        item = "ES_F3_N3"
        open_price = round(float(r["entry_price"]), 2)
        close_price = round(float(r["exit_price"]), 2)
        comm = round(-abs(float(r["friction"])), 2)
        profit = round(float(r["net_pnl"]), 2)
        comment = f"{r['exit_reason']}_{r['contract_type']}"

        rec = {
            "Ticket": ticket,
            "Open Time": open_time,
            "Type": t_type,
            "Size": size,
            "Item": item,
            "Open Price": open_price,
            "Close Time": close_time,
            "Close Price": close_price,
            "Commission": comm,
            "Swap": 0.0,
            "Profit": profit,
            "Comment": comment,
            "_raw_open_dt": open_dt,
        }
        all_records.append(rec)
        f3_records.append(rec)
        ticket += 1

    # Standalone F3-N3 export
    df_f3_qa = pd.DataFrame(f3_records).drop(columns=["_raw_open_dt"])
    f3_out = ROOT / "exports/strategyquant/F3_N3_Trades_QuantAnalyzer.csv"
    df_f3_qa.to_csv(f3_out, index=False)
    print(f"Exported F3-N3: {len(df_f3_qa)} trades -> {f3_out}")

# 2. Process F4-N3 trades
if f4_path.exists():
    df4 = pd.read_csv(f4_path)
    for _, r in df4.iterrows():
        open_dt = pd.to_datetime(f"{r['entry_date']} {r['entry_time']}")
        close_dt = pd.to_datetime(f"{r['exit_date']} {r['exit_time']}")
        open_time = open_dt.strftime("%Y.%m.%d %H:%M")
        close_time = close_dt.strftime("%Y.%m.%d %H:%M")
        t_type = "buy" if r["direction"] == 1 else "sell"
        size = int(r["contracts_nq"])
        item = "SPREAD_F4_N3"
        open_price = round(float(r["entry_price_nq"]), 2)
        close_price = round(float(r["exit_price_nq"]), 2)
        comm = round(-abs(float(r["commissions_usd"])), 2)
        profit = round(float(r["net_pnl_usd"]), 2)
        comment = f"{r['exit_reason']}_NQ{size}_ES{int(r['contracts_es'])}"

        rec = {
            "Ticket": ticket,
            "Open Time": open_time,
            "Type": t_type,
            "Size": size,
            "Item": item,
            "Open Price": open_price,
            "Close Time": close_time,
            "Close Price": close_price,
            "Commission": comm,
            "Swap": 0.0,
            "Profit": profit,
            "Comment": comment,
            "_raw_open_dt": open_dt,
        }
        all_records.append(rec)
        ticket += 1

# 3. Sort merged portfolio by chronological open time and re-index tickets
all_records.sort(key=lambda x: x["_raw_open_dt"])
for idx, r in enumerate(all_records):
    r["Ticket"] = 1000 + idx
    del r["_raw_open_dt"]

df_portfolio = pd.DataFrame(all_records)
portfolio_out = ROOT / "exports/strategyquant/Portfolio_F3_F4_QuantAnalyzer.csv"
df_portfolio.to_csv(portfolio_out, index=False)
print(f"Exported Combined Portfolio: {len(df_portfolio)} trades -> {portfolio_out}")
