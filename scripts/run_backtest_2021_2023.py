import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.strategies.f4_optimizacion_is_n2 import (
    F4OptimizationISN2Strategy,
    F4OptimizationISN2Backtester,
)


def run():
    print("=" * 78)
    print(" BACKTEST F4-OPTIMIZACION -IS- N2: PERIODO 2021 A 2023 (56,298 BARRAS)")
    print("=" * 78)

    nq_path = ROOT / "data/raw/CME_MINI_NQ_2021_2023_5m.parquet"
    es_path = ROOT / "data/raw/CME_MINI_ES_2021_2023_5m.parquet"

    print("\n[1/4] Cargando datasets 2021-2023...")
    df_nq = pd.read_parquet(nq_path)
    df_es = pd.read_parquet(es_path)

    # Convert tz if present
    if df_nq.index.tz is not None:
        df_nq.index = df_nq.index.tz_convert("America/New_York").tz_localize(None)
    if df_es.index.tz is not None:
        df_es.index = df_es.index.tz_convert("America/New_York").tz_localize(None)

    common_idx = df_nq.index.intersection(df_es.index)
    df_nq = df_nq.loc[common_idx]
    df_es = df_es.loc[common_idx]

    print(f"  Rango: {common_idx[0]} a {common_idx[-1]}")
    print(f"  Total barras sincronizadas: {len(common_idx):,}")
    print(f"  Años cubiertos: {sorted(common_idx.year.unique())}")

    print("\n[2/4] Configurando modelo cuantitativo F4 N2...")
    strat_params = {
        "contract_mode": "micro",
        "use_kalman": False,
        "unify_beta_ols": True,
        "max_coprime_nq": 7,
        "allow_sub_baskets": True,
        "dynamic_basket_ratio": True,
        "s_open": 1.25,
        "s_close": 0.25,
        "s_stop": 3.25,
        "k_calib": 2340,
        "min_calib_bars": 390,
        "phi_dll": 0.60,
        "psi_tmdd": 0.30,
        "survival_floor": 105.0,
        "time_decay_multiplier": 3.5,
        "n_basket_max": 3,
        "source_tz": "America/New_York",
        "target_tz": "America/New_York",
    }
    strategy = F4OptimizationISN2Strategy(params=strat_params)

    backtester = F4OptimizationISN2Backtester(
        strategy=strategy,
        account_size=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_loss=2000.0,
        profit_target=3000.0,
        early_stopping=False,
    )

    xlk_path = ROOT / "data/raw/XLK_1d_2021-01-01_2023-12-31.parquet"
    spy_path = ROOT / "data/raw/SPY_1d_2021-01-01_2023-12-31.parquet"
    df_xlk = pd.read_parquet(xlk_path) if xlk_path.exists() else None
    df_spy = pd.read_parquet(spy_path) if spy_path.exists() else None

    print("\n[3/4] Ejecutando simulación...")
    results = backtester.run(df_nq=df_nq, df_es=df_es, df_xlk=df_xlk, df_spy=df_spy)

    metrics = results["metrics"]
    trades_df = results["trades"]
    prop_eval = results["prop_evaluation"]

    print("\n" + "=" * 78)
    print(" RESULTADOS GLOBALES 2021-2023 (F4-OPTIMIZACION -IS- N2)")
    print("=" * 78)
    print(f"  Estado Prop Trading   : {prop_eval['status']}")
    print(f"  Total Operaciones     : {len(trades_df)}")
    print(f"  Balance Final         : ${prop_eval['final_balance']:,.2f}")
    win_rate = metrics.get("Win Rate (%)", metrics.get("win_rate", 0.0) * (100 if metrics.get("win_rate", 0.0) <= 1.0 else 1.0))
    profit_factor = metrics.get("Profit Factor", metrics.get("profit_factor", 0.0))
    sharpe = metrics.get("Sharpe Ratio (Daily 252d)", metrics.get("sharpe_ratio_daily_252d", 0.0))
    sortino = metrics.get("Sortino Ratio (Daily 252d)", metrics.get("sortino_ratio_daily_252d", 0.0))

    print(f"  Beneficio Neto Total  : ${prop_eval['net_profit']:,.2f}")
    print(f"  Drawdown Máximo       : ${metrics.get('Max Drawdown ($)', 0.0):,.2f}")
    print(f"  Win Rate              : {win_rate:.2f}%")
    print(f"  Profit Factor         : {profit_factor:.2f}")
    print(f"  Sharpe Ratio (252d)   : {sharpe:.2f}")
    print(f"  Sortino Ratio (252d)  : {sortino:.2f}")

    # Desglose por año
    print("\n" + "-" * 78)
    print(" DESGLOSE ANUALIZADO (2021, 2022, 2023)")
    print("-" * 78)
    trades_df["entry_dt"] = pd.to_datetime(trades_df["entry_date"])
    yearly_summary = {}

    for yr in [2021, 2022, 2023]:
        t_yr = trades_df[trades_df["entry_dt"].dt.year == yr]
        n_t = len(t_yr)
        pnl_yr = float(t_yr["net_pnl_usd"].sum()) if n_t > 0 else 0.0
        wins_yr = int((t_yr["net_pnl_usd"] > 0).sum()) if n_t > 0 else 0
        wr_yr = (wins_yr / n_t * 100) if n_t > 0 else 0.0
        gross_win = float(t_yr.loc[t_yr["net_pnl_usd"] > 0, "net_pnl_usd"].sum()) if n_t > 0 else 0.0
        gross_loss = float(abs(t_yr.loc[t_yr["net_pnl_usd"] < 0, "net_pnl_usd"].sum())) if n_t > 0 else 0.0
        pf_yr = (gross_win / gross_loss) if gross_loss > 0 else 999.0

        yearly_summary[yr] = {
            "trades": n_t,
            "net_pnl": round(pnl_yr, 2),
            "win_rate": round(wr_yr, 2),
            "profit_factor": round(pf_yr, 2),
        }
        print(f"  Año {yr}: Trades = {n_t:>4} | PnL = ${pnl_yr:>10,.2f} | WinRate = {wr_yr:>5.1f}% | PF = {pf_yr:>5.2f}")

    # Guardar reportes
    print("\n[4/4] Guardando reportes y archivos exportables...")
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    trades_csv = reports_dir / "f4_2021_2023_trades.csv"
    trades_df.to_csv(trades_csv, index=False)
    print(f"  [+] Trades guardados: {trades_csv}")

    summary_json = reports_dir / "f4_2021_2023_summary.json"
    full_summary = {
        "period": "2021-2023",
        "bars": len(common_idx),
        "prop_evaluation": prop_eval,
        "metrics": metrics,
        "yearly_breakdown": yearly_summary,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, default=str)
    print(f"  [+] Resumen guardado: {summary_json}")

    # Exportar para StrategyQuant / QuantAnalyzer
    sqx_dir = ROOT / "exports/strategyquant"
    sqx_dir.mkdir(parents=True, exist_ok=True)

    # 1. Spread OHLCV MetaTrader4
    spread_open = (df_nq["Open"] / df_es["Open"]).round(5)
    spread_high = (df_nq["High"] / df_es["Low"]).round(5)
    spread_low = (df_nq["Low"] / df_es["High"]).round(5)
    spread_close = (df_nq["Close"] / df_es["Close"]).round(5)
    volume = df_nq.get("Volume", 100).astype(int)

    mt4_lines = []
    for i in range(len(common_idx)):
        dt = common_idx[i]
        d_str = dt.strftime("%Y.%m.%d")
        t_str = dt.strftime("%H:%M")
        mt4_lines.append(
            f"{d_str},{t_str},{spread_open.iloc[i]:.5f},{spread_high.iloc[i]:.5f},{spread_low.iloc[i]:.5f},{spread_close.iloc[i]:.5f},{volume.iloc[i]}"
        )

    sq_spread_path = sqx_dir / "SPREAD_NQ_ES_2021_2023_M5.csv"
    with open(sq_spread_path, "w", encoding="utf-8") as f:
        f.write("\n".join(mt4_lines) + "\n")
    print(f"  [+] Spread MetaTrader4 SQX: {sq_spread_path} ({len(mt4_lines):,} barras)")

    # 2. Trades para QuantAnalyzer
    qa_records = []
    for _, r in trades_df.iterrows():
        open_time = f"{str(r['entry_date']).replace('-', '.')} {r['entry_time']}"
        close_time = f"{str(r['exit_date']).replace('-', '.')} {r['exit_time']}"
        t_type = "buy" if r["direction"] == 1 else "sell"
        size = int(r["contracts_nq"])
        item = "NQ_ES_SPREAD"
        open_price = round(float(r["entry_price_nq"]), 2)
        close_price = round(float(r["exit_price_nq"]), 2)
        comm = round(-abs(float(r["commissions_usd"])), 2)
        swap = 0.0
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
            "Swap": swap,
            "Profit": profit,
            "Comment": comment,
        })

    qa_df = pd.DataFrame(qa_records)
    qa_csv_path = sqx_dir / "F4_Trades_2021_2023_QuantAnalyzer.csv"
    qa_df.to_csv(qa_csv_path, index=False)
    print(f"  [+] Trades QuantAnalyzer: {qa_csv_path} ({len(qa_df)} trades)")


if __name__ == "__main__":
    run()
