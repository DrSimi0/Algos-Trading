import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

df_path = ROOT / "reports/f4_optimizacion_is_n3_trades.csv"
if not df_path.exists():
    print(f"Error: {df_path} does not exist.")
    exit(1)

df = pd.read_csv(df_path)
qa_records = []
for _, r in df.iterrows():
    open_time = f"{str(r['entry_date']).replace('-', '.')} {r['entry_time']}"
    close_time = f"{str(r['exit_date']).replace('-', '.')} {r['exit_time']}"
    t_type = "buy" if r["direction"] == 1 else "sell"
    size = int(r["contracts_nq"])
    item = "NQ_ES_SPREAD"
    open_price = round(float(r["entry_price_nq"]), 2)
    close_price = round(float(r["exit_price_nq"]), 2)
    comm = round(-abs(float(r["commissions_usd"])), 2)
    profit = round(float(r["net_pnl_usd"]), 2)
    comment = f"{r['exit_reason']}_NQ{size}_ES{int(r['contracts_es'])}"

    qa_records.append({
        "Ticket": int(r["trade_id"]),
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
    })

qa_df = pd.DataFrame(qa_records)
out_path = ROOT / "exports/strategyquant/F4_N3_Trades_QuantAnalyzer.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
qa_df.to_csv(out_path, index=False)
print(f"Exported {len(qa_df)} trades to {out_path}")
