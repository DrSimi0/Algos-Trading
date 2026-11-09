"""
Mismas configuraciones del barrido IS aplicadas a 2026 (SOLO REPORTE, no se usa para elegir parametros).
Resultado: reports/f4_n3_param_sweep_2026.json
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd

import run_f4_n3_param_sweep_is as S
from src.strategies.f4_optimizacion_is_n3 import (
    F4OptimizationISN3Strategy,
    F4OptimizationISN3Backtester,
)

_D = {}


def _init():
    xlk = pd.read_parquet("data/raw/XLK_1d_2021_2026.parquet")
    spy = pd.read_parquet("data/raw/SPY_1d_2021_2026.parquet")
    for d in (xlk, spy):
        if d.index.tz is not None:
            d.index = d.index.tz_convert("America/New_York").tz_localize(None)
    es = pd.read_parquet("data/raw/CME_EXT_ES_5m.parquet")
    nq = pd.read_parquet("data/raw/CME_EXT_NQ_5m.parquet")
    for d in (es, nq):
        if d.index.tz is not None:
            d.index = d.index.tz_convert("America/New_York").tz_localize(None)
    c = nq.index.intersection(es.index)
    _D.update(xlk=xlk, spy=spy, es=es.loc[c], nq=nq.loc[c])


def _run(task):
    name, value = task
    params = dict(S.BASE)
    params[name] = value
    if name == "k_calib":
        params["min_calib_bars"] = min(390, value)
    bt = F4OptimizationISN3Backtester(
        strategy=F4OptimizationISN3Strategy(params=params),
        account_size=50000.0, daily_loss_limit=1000.0,
        max_trailing_loss=2000.0, profit_target=3000.0, early_stopping=False,
    )
    res = bt.run(_D["nq"], _D["es"], df_xlk=_D["xlk"], df_spy=_D["spy"])
    m, a = res["metrics"], res.get("cointegration_audit", {})
    return {
        "param": name, "value": value, "year": 2026,
        "trades": m.get("Total Trades"), "win_rate": m.get("Win Rate (%)"),
        "pnl": m.get("Net Profit ($)"), "max_dd": m.get("Max Drawdown ($)"),
        "pf": m.get("Profit Factor"), "sharpe": m.get("Sharpe Ratio (Daily 252d)"),
        "prop": res["prop_evaluation"]["status"],
        "tau_ok_pct": a.get("tau_passed_pct"), "mean_tau": a.get("mean_tau_half_min"),
        "mean_beta": a.get("mean_beta"),
    }


if __name__ == "__main__":
    tasks = [(n, v) for n, vals in S.GRID.items() for v in vals]
    print(len(tasks), "corridas 2026", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=2, initializer=_init) as ex:
        for r in ex.map(_run, tasks):
            out.append(r)
            print(r["param"], r["value"], r["trades"], r["pnl"], flush=True)
    Path("reports/f4_n3_param_sweep_2026.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("OK", flush=True)
