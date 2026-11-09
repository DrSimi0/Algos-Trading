import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.strategies.f4_optimizacion_is_n3 import (
    F4OptimizationISN3Strategy,
    F4OptimizationISN3Backtester,
)

print("=" * 80)
print("BACKTESTING F4-N3 EN 2026 (5 MINUTOS) & ANALISIS ESTOCASTICO MONTE CARLO")
print("=" * 80)

# Cargar datos de 2026
df_es = pd.read_parquet("data/raw/CME_EXT_ES_5m.parquet")
df_nq = pd.read_parquet("data/raw/CME_EXT_NQ_5m.parquet")

if df_es.index.tz is not None:
    df_es.index = df_es.index.tz_convert("America/New_York")
else:
    df_es.index = df_es.index.tz_localize("America/New_York")

if df_nq.index.tz is not None:
    df_nq.index = df_nq.index.tz_convert("America/New_York")
else:
    df_nq.index = df_nq.index.tz_localize("America/New_York")

common = df_nq.index.intersection(df_es.index)
df_nq = df_nq.loc[common].sort_index()
df_es = df_es.loc[common].sort_index()

print(f"Rango de datos: {common[0]} a {common[-1]} ({len(common):,} barras)")

# Cargar datos sectoriales XLK y SPY para 2026 (Filtro R4)
df_xlk = None
df_spy = None
xlk_file = Path("data/raw/XLK_1d_2026.parquet")
spy_file = Path("data/raw/SPY_1d_2018-01-01_latest.parquet")
if not spy_file.exists():
    spy_file = Path("data/raw/SPY_1d_2023-01-01_latest.parquet")

if xlk_file.exists() and spy_file.exists():
    df_xlk = pd.read_parquet(xlk_file)
    df_spy = pd.read_parquet(spy_file)
    print(f"Filtro sectorial activo: XLK ({len(df_xlk)} velas) y SPY ({len(df_spy)} velas)")
else:
    print("[!] Aviso: No se encontraron series XLK/SPY para 2026; ejecutando sin filtro sectorial.")

# Evaluacion de F4 N3 con k=2340 y k=390
backtest_results = {}
for k in [2340, 390]:
    strat = F4OptimizationISN3Strategy(params={
        "contract_mode": "micro",
        "k_calib": k,
        "min_calib_bars": 390,
        "source_tz": "America/New_York",
        "target_tz": "America/New_York",
        "use_session_demean": True,
    })
    bt = F4OptimizationISN3Backtester(
        strategy=strat,
        account_size=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_loss=2000.0,
        profit_target=3000.0
    )
    res = bt.run(df_nq, df_es, df_xlk=df_xlk, df_spy=df_spy)
    backtest_results[k] = res
    m = res["metrics"]
    tdf = res["trades"]
    print(f"\n--- F4-N3 (K={k}) ---")
    print(f"Total Trades     : {m.get('Total Trades')}")
    print(f"Win Rate         : {m.get('Win Rate (%)', 0):.2f}%")
    print(f"Net Profit       : ${m.get('Net Profit ($)', 0):,.2f}")
    print(f"Max Drawdown     : ${m.get('Max Drawdown ($)', 0):,.2f}")
    print(f"Profit Factor    : {m.get('Profit Factor', 0):.2f}")
    print(f"Sharpe (252d)    : {m.get('Sharpe Ratio (Daily 252d)', 0):.2f}")
    if len(tdf) > 0:
        cols = ["trade_id", "entry_date", "entry_time", "exit_time", "direction", "friction_usd", "net_pnl_usd", "exit_reason"]
        print(tdf[[c for c in cols if c in tdf.columns]].head(15).to_string())

# Monte Carlo Engine (2,500 iteraciones)
print("\n" + "=" * 80)
print("SIMULACION DE MONTE CARLO (2,500 ITERACIONES)")
print("=" * 80)

# Para Monte Carlo, evaluamos el universo de trades reales generados en 2026
# y contrastamos con la distribucion empírica de operaciones de F4
def run_mc_simulation(pnl_series: np.ndarray, n_sim: int = 2500, initial_capital: float = 50000.0, target: float = 3000.0, mtdd: float = 2000.0):
    np.random.seed(42)
    n_trades = len(pnl_series)
    if n_trades == 0:
        print("[!] No hay operaciones para simular Monte Carlo sobre el año 2026 solo.")
        return {}

    final_equities = []
    max_drawdowns = []
    ruin_count = 0
    target_count = 0

    # Reshuffling & Bootstrapping
    for _ in range(n_sim):
        # Bootstrap con reemplazo sobre los retornos observados en el mercado
        sampled_pnl = np.random.choice(pnl_series, size=n_trades, replace=True)
        equity = initial_capital + np.cumsum(sampled_pnl)
        peak = np.maximum.accumulate(np.insert(equity, 0, initial_capital))
        dd = peak[1:] - equity
        mdd = np.max(dd)
        
        final_equities.append(equity[-1])
        max_drawdowns.append(mdd)
        
        if mdd >= mtdd:
            ruin_count += 1
        if (equity[-1] - initial_capital) >= target:
            target_count += 1

    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)

    p50_pnl = np.percentile(final_equities - initial_capital, 50)
    p5_pnl = np.percentile(final_equities - initial_capital, 5)
    p95_pnl = np.percentile(final_equities - initial_capital, 95)
    p50_dd = np.percentile(max_drawdowns, 50)
    p95_dd = np.percentile(max_drawdowns, 95)
    cvar_95 = np.mean(max_drawdowns[max_drawdowns >= p95_dd])

    mc_results = {
        "n_simulations": n_sim,
        "n_trades": n_trades,
        "median_net_pnl": float(p50_pnl),
        "p5_pnl": float(p5_pnl),
        "p95_pnl": float(p95_pnl),
        "median_max_dd": float(p50_dd),
        "p95_max_dd": float(p95_dd),
        "cvar_95_dd": float(cvar_95),
        "prob_ruin_pct": float(ruin_count / n_sim * 100.0),
        "prob_target_pct": float(target_count / n_sim * 100.0)
    }

    print(f"Simulaciones        : {n_sim:,}")
    print(f"Mediana PnL Neto    : ${p50_pnl:,.2f}")
    print(f"Percentil 5 PnL     : ${p5_pnl:,.2f}")
    print(f"Percentil 95 PnL    : ${p95_pnl:,.2f}")
    print(f"Mediana Max DD      : ${p50_dd:,.2f}")
    print(f"Percentil 95 Max DD : ${p95_dd:,.2f}")
    print(f"CVaR 95% Drawdown   : ${cvar_95:,.2f}")
    print(f"Riesgo de Ruina     : {mc_results['prob_ruin_pct']:.2f}% (TMDD $2,000)")
    print(f"Prob. Meta ($3k)    : {mc_results['prob_target_pct']:.2f}%")
    return mc_results

# Realizar Monte Carlo sobre F4-N3 (K=390) trades
res_390 = backtest_results[390]
pnl_390 = res_390["trades"]["net_pnl_usd"].values if len(res_390["trades"]) > 0 else np.array([])

mc_390 = run_mc_simulation(pnl_390)

# Tambien para tener una muestra robusta multianual proyectada a 2026:
# simulamos Monte Carlo con el pool de trades de F4 N3 completos si existe
mc_robust = {}
pool_file = Path("reports/f4_optimizacion_is_n3_trades.csv")
if not pool_file.exists():
    pool_file = Path("reports/f4_optimizacion_is_n2_trades.csv")
if pool_file.exists():
    all_trades_pnl = pd.read_csv(pool_file)["net_pnl_usd"].values
    print(f"\n--- MONTE CARLO PROYECTIVO INSTITUCIONAL (Pool de Trades F4 sobre 2026 desde {pool_file.name}) ---")
    mc_robust = run_mc_simulation(all_trades_pnl)

# Exportar resultados a JSON
with open("reports/f4_n3_2026_monte_carlo.json", "w") as f:
    json.dump({
        "metrics_k2340": backtest_results[2340]["metrics"],
        "metrics_k390": backtest_results[390]["metrics"],
        "mc_2026_isolated": mc_390,
        "mc_projected_pool": mc_robust
    }, f, indent=2, default=str)

print("\n[OK] Informe exportado a reports/f4_n3_2026_monte_carlo.json")

