"""
Barrido de sensibilidad de parametros F4-N3 (un factor a la vez), SOLO In-Sample 2021-2023.
No toca data/vault ni 2024/2025. Resultado: reports/f4_n3_param_sweep_is.json
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd

from src.strategies.f4_optimizacion_is_n3 import (
    F4OptimizationISN3Strategy,
    F4OptimizationISN3Backtester,
)

BASE = {
    "contract_mode": "micro", "k_calib": 390, "min_calib_bars": 390,
    "phi_dll": 0.60, "psi_tmdd": 0.30, "survival_floor": 105.0,
    "time_decay_multiplier": 3.5, "n_basket_max": 3,
    "source_tz": "America/New_York", "target_tz": "America/New_York",
}

# Un factor a la vez (el resto en BASE). sigma_floor y min_sector_corr nunca bajan de las reglas del repo.
GRID = {
    "k_calib": [195, 390, 780],
    "s_open": [1.0, 1.25, 1.5, 2.0],
    "s_close": [0.0, 0.25, 0.5],
    "s_stop": [2.5, 3.25, 4.0],
    "tau_min": [5.0, 15.0, 30.0],
    "time_decay_multiplier": [2.0, 3.5, 5.0],
    "min_sector_corr": [0.85, 0.90],
}
YEARS = [2021, 2022, 2023]
_D = {}


def _init():
    xlk = pd.read_parquet("data/raw/XLK_1d_2021_2026.parquet")
    spy = pd.read_parquet("data/raw/SPY_1d_2021_2026.parquet")
    for d in (xlk, spy):
        if d.index.tz is not None:
            d.index = d.index.tz_convert("America/New_York").tz_localize(None)
    es = pd.read_parquet("data/raw/CME_MINI_ES_2021_2023_5m.parquet")
    nq = pd.read_parquet("data/raw/CME_MINI_NQ_2021_2023_5m.parquet")
    for d in (es, nq):
        if d.index.tz is not None:
            d.index = d.index.tz_convert("America/New_York").tz_localize(None)
    c = nq.index.intersection(es.index)
    _D.update(xlk=xlk, spy=spy, es=es.loc[c], nq=nq.loc[c])


def _run(task):
    name, value, year = task
    params = dict(BASE)
    params[name] = value
    if name == "k_calib":
        params["min_calib_bars"] = min(390, value)
    nq = _D["nq"][_D["nq"].index.year == year]
    es = _D["es"][_D["es"].index.year == year]
    xlk = _D["xlk"][_D["xlk"].index.year == year]
    spy = _D["spy"][_D["spy"].index.year == year]
    bt = F4OptimizationISN3Backtester(
        strategy=F4OptimizationISN3Strategy(params=params),
        account_size=50000.0, daily_loss_limit=1000.0,
        max_trailing_loss=2000.0, profit_target=3000.0, early_stopping=False,
    )
    res = bt.run(nq, es, df_xlk=xlk, df_spy=spy)
    m, a = res["metrics"], res.get("cointegration_audit", {})
    return {
        "param": name, "value": value, "year": year,
        "trades": m.get("Total Trades"), "win_rate": m.get("Win Rate (%)"),
        "pnl": m.get("Net Profit ($)"), "max_dd": m.get("Max Drawdown ($)"),
        "pf": m.get("Profit Factor"), "sharpe": m.get("Sharpe Ratio (Daily 252d)"),
        "prop": res["prop_evaluation"]["status"],
        "tau_ok_pct": a.get("tau_passed_pct"), "mean_tau": a.get("mean_tau_half_min"),
        "mean_beta": a.get("mean_beta"),
    }


if __name__ == "__main__":
    tasks, seen = [], set()
    for name, vals in GRID.items():
        for v in vals:
            for y in YEARS:
                key = (BASE.get(name) == v and "base") or (name, v)
                if key == "base":
                    if ("base", y) in seen:
                        continue
                    seen.add(("base", y))
                tasks.append((name, v, y))
    print(f"{len(tasks)} corridas", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=10, initializer=_init) as ex:
        for r in ex.map(_run, tasks):
            out.append(r)
            print(r["param"], r["value"], r["year"], r["trades"], r["pnl"], flush=True)
    Path("reports/f4_n3_param_sweep_is.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("OK", flush=True)
