"""
scripts/verify_empirical_distributions_and_friction.py
Adversarial Verification Script for Challenger 2:
Empirical Sequence, Distribution Stress, Friction Sensitivity, and Temporal Dispersion.
"""

import math
import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Dict, Any, Tuple, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def analyze_trade_distribution(df: pd.DataFrame, pnl_col: str, name: str) -> Dict[str, Any]:
    pnl = df[pnl_col].dropna().astype(float).values
    n = len(pnl)
    if n == 0:
        return {"name": name, "n": 0}

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    zeros = pnl[pnl == 0]

    n_wins = len(wins)
    n_losses = len(losses)
    n_zeros = len(zeros)

    win_rate = n_wins / n if n > 0 else 0.0
    loss_rate = n_losses / n if n > 0 else 0.0

    gross_profit = float(np.sum(wins)) if n_wins > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if n_losses > 0 else 0.0
    net_pnl = float(np.sum(pnl))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = float(np.mean(wins)) if n_wins > 0 else 0.0
    avg_loss = float(np.abs(np.mean(losses))) if n_losses > 0 else 0.0
    payout_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

    # Moments
    mean_val = float(np.mean(pnl))
    std_val = float(np.std(pnl, ddof=1)) if n > 1 else 0.0
    median_val = float(np.median(pnl))
    skew_val = float(stats.skew(pnl, bias=False)) if n > 2 else 0.0
    kurt_val = float(stats.kurtosis(pnl, bias=False)) if n > 3 else 0.0 # excess kurtosis

    # Normality test
    jb_stat, jb_pval = stats.jarque_bera(pnl) if n >= 8 else (0.0, 1.0)

    # Quantiles
    p1 = float(np.percentile(pnl, 1))
    p5 = float(np.percentile(pnl, 5))
    p10 = float(np.percentile(pnl, 10))
    p25 = float(np.percentile(pnl, 25))
    p50 = float(np.percentile(pnl, 50))
    p75 = float(np.percentile(pnl, 75))
    p90 = float(np.percentile(pnl, 90))
    p95 = float(np.percentile(pnl, 95))
    p99 = float(np.percentile(pnl, 99))

    # VaR and CVaR (loss is positive)
    var_95 = -p5
    var_99 = -p1
    cvar_95 = -float(np.mean(pnl[pnl <= p5])) if np.sum(pnl <= p5) > 0 else var_95
    cvar_99 = -float(np.mean(pnl[pnl <= p1])) if np.sum(pnl <= p1) > 0 else var_99

    # Consecutive streak analysis
    # Longest loss streak
    max_loss_streak = 0
    current_loss_streak = 0
    loss_streaks = []

    max_win_streak = 0
    current_win_streak = 0
    win_streaks = []

    for x in pnl:
        if x < 0:
            current_loss_streak += 1
            if current_win_streak > 0:
                win_streaks.append(current_win_streak)
                current_win_streak = 0
            if current_loss_streak > max_loss_streak:
                max_loss_streak = current_loss_streak
        elif x > 0:
            current_win_streak += 1
            if current_loss_streak > 0:
                loss_streaks.append(current_loss_streak)
                current_loss_streak = 0
            if current_win_streak > max_win_streak:
                max_win_streak = current_win_streak
        else:
            # treat zero as break
            if current_loss_streak > 0:
                loss_streaks.append(current_loss_streak)
                current_loss_streak = 0
            if current_win_streak > 0:
                win_streaks.append(current_win_streak)
                current_win_streak = 0

    if current_loss_streak > 0:
        loss_streaks.append(current_loss_streak)
    if current_win_streak > 0:
        win_streaks.append(current_win_streak)

    # Erdős-Rényi bounds for consecutive losses
    # Given N trials and probability of loss q = loss_rate
    # E[L_N] ~ ln(N) / ln(1/q) - (gamma + ln(ln(1/q))) / ln(1/q) - 0.5
    # where gamma ~ 0.5772156649
    q = loss_rate
    gamma = 0.5772156649015329
    if 0 < q < 1:
        log_inv_q = math.log(1.0 / q)
        er_mean = math.log(n) / log_inv_q - (gamma + math.log(log_inv_q)) / log_inv_q - 0.5
        er_95 = (math.log(n) - math.log(-math.log(0.95))) / log_inv_q
        er_99 = (math.log(n) - math.log(-math.log(0.99))) / log_inv_q
    else:
        er_mean = 0.0
        er_95 = 0.0
        er_99 = 0.0

    return {
        "name": name,
        "n_trades": n,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_zeros": n_zeros,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": net_pnl,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payout_ratio": payout_ratio,
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "skewness": skew_val,
        "excess_kurtosis": kurt_val,
        "jb_stat": jb_stat,
        "jb_pval": jb_pval,
        "p1": p1, "p5": p5, "p10": p10, "p25": p25, "p50": p50,
        "p75": p75, "p90": p90, "p95": p95, "p99": p99,
        "var_95": var_95, "var_99": var_99,
        "cvar_95": cvar_95, "cvar_99": cvar_99,
        "max_loss_streak": max_loss_streak,
        "max_win_streak": max_win_streak,
        "er_mean_loss_streak": er_mean,
        "er_95_loss_streak": er_95,
        "er_99_loss_streak": er_99,
    }


def run_monte_carlo_adversarial(
    trades: np.ndarray,
    n_sims: int = 10000,
    initial_capital: float = 50000.0,
    mtdd: float = 2000.0,
    target: float = 3000.0,
    method: str = "reshuffle",
    win_haircut: float = 0.0,
    seed: int = 42
) -> Dict[str, Any]:
    np.random.seed(seed)
    pnl = trades.copy()
    if win_haircut > 0.0:
        pnl = np.where(pnl > 0, pnl * (1.0 - win_haircut), pnl)

    n_trades = len(pnl)
    ruin_count = 0
    target_count = 0
    target_before_ruin_count = 0
    max_dds = np.empty(n_sims, dtype=float)
    final_equities = np.empty(n_sims, dtype=float)

    for i in range(n_sims):
        if method == "reshuffle":
            seq = np.random.permutation(pnl)
        else: # bootstrap
            seq = np.random.choice(pnl, size=n_trades, replace=True)

        equity = initial_capital
        peak = initial_capital
        max_dd = 0.0
        hit_ruin = False
        hit_target = False
        first_event = None

        for trade in seq:
            equity += trade
            if equity > peak:
                peak = equity
            current_dd = peak - equity
            if current_dd > max_dd:
                max_dd = current_dd

            if current_dd >= mtdd and not hit_ruin:
                hit_ruin = True
                if first_event is None:
                    first_event = "RUIN"

            if (equity - initial_capital) >= target and not hit_target:
                hit_target = True
                if first_event is None:
                    first_event = "TARGET"

        max_dds[i] = max_dd
        final_equities[i] = equity
        if hit_ruin:
            ruin_count += 1
        if hit_target:
            target_count += 1
        if first_event == "TARGET":
            target_before_ruin_count += 1

    ruin_rate = (ruin_count / n_sims) * 100.0
    pass_rate = (target_before_ruin_count / n_sims) * 100.0
    target_rate = (target_count / n_sims) * 100.0

    p50_dd = float(np.percentile(max_dds, 50))
    p95_dd = float(np.percentile(max_dds, 95))
    p99_dd = float(np.percentile(max_dds, 99))
    max_mc_dd = float(np.max(max_dds))
    cvar_95_dd = float(np.mean(max_dds[max_dds >= p95_dd]))

    return {
        "method": method,
        "n_sims": n_sims,
        "mtdd": mtdd,
        "target": target,
        "win_haircut": win_haircut,
        "ruin_rate_pct": ruin_rate,
        "pass_rate_pct": pass_rate,
        "target_rate_pct": target_rate,
        "p50_dd": p50_dd,
        "p95_dd": p95_dd,
        "p99_dd": p99_dd,
        "max_mc_dd": max_mc_dd,
        "cvar_95_dd": cvar_95_dd,
    }


def stress_test_f4_friction(df_f4: pd.DataFrame, initial_capital: float = 50000.0, mtdd: float = 2000.0, dll: float = 1000.0):
    """
    Stress test F4 execution frictions:
    Baseline friction vs 2x friction, 3x friction, and custom commission scenarios.
    """
    # Columns in df_f4:
    # gross_pnl_usd, commissions_usd, slippage_usd, legging_usd, friction_usd, net_pnl_usd, entry_date, exit_date
    results = {}

    gross = df_f4["gross_pnl_usd"].values
    comm = df_f4["commissions_usd"].values
    slip = df_f4["slippage_usd"].values
    leg = df_f4["legging_usd"].values
    base_fric = df_f4["friction_usd"].values
    dates = pd.to_datetime(df_f4["exit_date"]).dt.date.values

    scenarios = {
        "Baseline (1x)": base_fric,
        "Doubled Friction (2x)": base_fric * 2.0,
        "Tripled Friction (3x)": base_fric * 3.0,
        "Doubled Slippage Only (2x Slip, 1x Comm)": comm + 2.0 * slip + 2.0 * leg,
        "High Commission ($0.85/contract/side = $1.70 RT)": (comm / 1.24 * 1.70) + slip + leg,
        "Doubled Comm + Doubled Slip ($1.70 RT + 2x Slip)": (comm / 1.24 * 1.70) + 2.0 * slip + 2.0 * leg,
    }

    for sc_name, fric_series in scenarios.items():
        net = gross - fric_series
        n_trades = len(net)
        wins = net[net > 0]
        losses = net[net < 0]
        gp = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gl = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
        pf = gp / gl if gl > 0 else float('inf')
        tot_net = float(np.sum(net))
        wr = len(wins) / n_trades * 100.0

        # Equity curve and Max Trailing Drawdown
        cum_pnl = np.cumsum(net)
        equity = initial_capital + cum_pnl
        running_peak = np.maximum.accumulate(np.insert(equity, 0, initial_capital))[1:]
        dds = running_peak - equity
        max_dd = float(np.max(dds))

        # Daily aggregations for DLL
        df_tmp = pd.DataFrame({"date": dates, "net": net})
        daily = df_tmp.groupby("date")["net"].sum()
        min_daily = float(daily.min())
        dll_breaches = int((daily < -dll).sum())

        results[sc_name] = {
            "net_pnl": tot_net,
            "profit_factor": pf,
            "win_rate_pct": wr,
            "total_friction": float(np.sum(fric_series)),
            "max_drawdown": max_dd,
            "worst_day_pnl": min_daily,
            "dll_breaches": dll_breaches,
            "tmdd_breach": max_dd >= mtdd,
        }

    return results


def analyze_f4_temporal_dispersion(df_f4: pd.DataFrame, data_es_path: str = "data/raw/CME_MINI_ES_5m_3y.parquet"):
    """
    Analyzes temporal dispersion across all 260 trading sessions of IS 2023.
    """
    df_f4["entry_dt"] = pd.to_datetime(df_f4["entry_date"] + " " + df_f4["entry_time"])
    df_f4["exit_dt"] = pd.to_datetime(df_f4["exit_date"] + " " + df_f4["exit_time"])
    df_f4["entry_date_only"] = pd.to_datetime(df_f4["entry_date"]).dt.date

    # Monthly breakdown
    df_f4["month"] = pd.to_datetime(df_f4["entry_date"]).dt.to_period("M")
    monthly_groups = df_f4.groupby("month")
    monthly_stats = []
    for m, g in monthly_groups:
        wins = g[g["net_pnl_usd"] > 0]
        losses = g[g["net_pnl_usd"] < 0]
        gp = wins["net_pnl_usd"].sum()
        gl = abs(losses["net_pnl_usd"].sum())
        pf = gp / gl if gl > 0 else float('inf')
        monthly_stats.append({
            "month": str(m),
            "trades": len(g),
            "net_pnl": round(float(g["net_pnl_usd"].sum()), 2),
            "win_rate_pct": round(len(wins) / len(g) * 100.0, 2),
            "profit_factor": round(pf, 2),
            "avg_trades_per_day": round(len(g) / g["entry_date_only"].nunique(), 2) if g["entry_date_only"].nunique() > 0 else 0.0
        })

    # Calendar session analysis
    # Load ES market data to know all RTH sessions in 2023
    es_df = pd.read_parquet(data_es_path)
    if not isinstance(es_df.index, pd.DatetimeIndex):
        es_df.index = pd.to_datetime(es_df.index)
    if es_df.index.tz is not None:
        es_df.index = es_df.index.tz_convert("America/New_York")
    else:
        es_df.index = es_df.index.tz_localize("UTC").tz_convert("America/New_York")

    # In-Sample 2023 mask
    es_2023 = es_df[(es_df.index >= "2023-01-01") & (es_df.index <= "2023-12-31 23:59:59")]
    # RTH mask
    rth_mask = (es_2023.index.time >= pd.to_datetime("09:30").time()) & (es_2023.index.time <= pd.to_datetime("16:00").time())
    es_rth_2023 = es_2023[rth_mask]
    all_sessions = sorted(list(set(es_rth_2023.index.date)))
    total_sessions_2023 = len(all_sessions)

    active_trade_sessions = sorted(list(df_f4["entry_date_only"].unique()))
    n_active_sessions = len(active_trade_sessions)
    n_inactive_sessions = total_sessions_2023 - n_active_sessions

    # Consecutive inactive sessions (starvation)
    active_set = set(active_trade_sessions)
    max_starvation_sessions = 0
    cur_starvation = 0
    starvation_periods = []

    for s in all_sessions:
        if s not in active_set:
            cur_starvation += 1
            if cur_starvation > max_starvation_sessions:
                max_starvation_sessions = cur_starvation
        else:
            if cur_starvation > 0:
                starvation_periods.append(cur_starvation)
                cur_starvation = 0
    if cur_starvation > 0:
        starvation_periods.append(cur_starvation)

    # Time between trades
    df_sorted = df_f4.sort_values("entry_dt")
    time_gaps = (df_sorted["entry_dt"].iloc[1:].values - df_sorted["entry_dt"].iloc[:-1].values)
    time_gaps_hours = [t / np.timedelta64(1, 'h') for t in time_gaps]

    return {
        "total_sessions_2023": total_sessions_2023,
        "n_active_sessions": n_active_sessions,
        "n_inactive_sessions": n_inactive_sessions,
        "pct_active_sessions": round(n_active_sessions / total_sessions_2023 * 100.0, 2),
        "max_consecutive_inactive_sessions": max_starvation_sessions,
        "mean_inactive_streak": round(float(np.mean(starvation_periods)), 2) if starvation_periods else 0.0,
        "max_time_gap_hours": round(float(np.max(time_gaps_hours)), 1) if time_gaps_hours else 0.0,
        "median_time_gap_hours": round(float(np.median(time_gaps_hours)), 1) if time_gaps_hours else 0.0,
        "monthly_stats": monthly_stats
    }


def main():
    print("=" * 80)
    print(" ADVERSARIAL STRESS TEST: SEQUENCE, DISTRIBUTION, FRICTION, & DISPERSION")
    print("=" * 80)

    f3_file = Path("reports/f3_optimization_n2_trades.csv")
    f4_file = Path("reports/f4_optimizacion_is_n1_trades.csv")

    if not f3_file.exists():
        print(f"Error: {f3_file} does not exist.")
        return
    if not f4_file.exists():
        print(f"Error: {f4_file} does not exist.")
        return

    df_f3 = pd.read_csv(f3_file)
    df_f4 = pd.read_csv(f4_file)

    # 1. Trade Distribution Analysis for F3
    print("\n[+] 1. ANALYZING F3 TRADE RETURN DISTRIBUTION & ERDŐS-RÉNYI BOUNDS")
    f3_dist = analyze_trade_distribution(df_f3, "net_pnl", "F3 (Anchored VWAP N2)")
    for k, v in f3_dist.items():
        if isinstance(v, float):
            print(f"  {k:<30} : {v:.4f}")
        else:
            print(f"  {k:<30} : {v}")

    # 2. Trade Distribution Analysis for F4
    print("\n[+] 2. ANALYZING F4 TRADE RETURN DISTRIBUTION & ERDŐS-RÉNYI BOUNDS")
    f4_dist = analyze_trade_distribution(df_f4, "net_pnl_usd", "F4 (Cointegration OU IS N1)")
    for k, v in f4_dist.items():
        if isinstance(v, float):
            print(f"  {k:<30} : {v:.4f}")
        else:
            print(f"  {k:<30} : {v}")

    # 3. Friction Sensitivity Stress for F4
    print("\n[+] 3. F4 EXECUTION FRICTION SENSITIVITY STRESS TEST")
    f4_friction_res = stress_test_f4_friction(df_f4)
    for sc, res in f4_friction_res.items():
        print(f"\n  Scenario: {sc}")
        for k, v in res.items():
            if isinstance(v, float):
                print(f"    {k:<26} : {v:,.2f}")
            else:
                print(f"    {k:<26} : {v}")

    # 4. F4 Temporal & Calendar Dispersion
    print("\n[+] 4. F4 TEMPORAL & CALENDAR DISPERSION ACROSS 260 SESSIONS (IS 2023)")
    f4_temporal = analyze_f4_temporal_dispersion(df_f4)
    print(f"  Total IS 2023 Sessions     : {f4_temporal['total_sessions_2023']}")
    print(f"  Active Trading Sessions    : {f4_temporal['n_active_sessions']} ({f4_temporal['pct_active_sessions']}%)")
    print(f"  Inactive Sessions          : {f4_temporal['n_inactive_sessions']}")
    print(f"  Max Starvation Streak      : {f4_temporal['max_consecutive_inactive_sessions']} sessions")
    print(f"  Mean Inactive Streak       : {f4_temporal['mean_inactive_streak']} sessions")
    print(f"  Max Trade Gap              : {f4_temporal['max_time_gap_hours']} hours")
    print(f"  Median Trade Gap           : {f4_temporal['median_time_gap_hours']} hours")
    print("\n  Monthly Distribution:")
    for ms in f4_temporal["monthly_stats"]:
        print(f"    {ms['month']} | Trades: {ms['trades']:<3} | Net: ${ms['net_pnl']:>8.2f} | WR: {ms['win_rate_pct']:>5.1f}% | PF: {ms['profit_factor']:>5.2f} | Trades/ActiveDay: {ms['avg_trades_per_day']}")

    # 5. Monte Carlo Adversarial Sweeps for F3
    print("\n[+] 5. MONTE CARLO ADVERSARIAL SWEEPS (F3 N2 - 10,000 SIMULATIONS)")
    f3_pnl = df_f3["net_pnl"].values

    mc_configs = [
        {"desc": "F3 Baseline Reshuffle", "method": "reshuffle", "mtdd": 2000.0, "target": 3000.0, "haircut": 0.0},
        {"desc": "F3 Baseline Bootstrap", "method": "bootstrap", "mtdd": 2000.0, "target": 3000.0, "haircut": 0.0},
        {"desc": "F3 Tighter MTDD ($1,500) Reshuffle", "method": "reshuffle", "mtdd": 1500.0, "target": 3000.0, "haircut": 0.0},
        {"desc": "F3 Tighter MTDD ($1,500) Bootstrap", "method": "bootstrap", "mtdd": 1500.0, "target": 3000.0, "haircut": 0.0},
        {"desc": "F3 Severe MTDD ($1,200) Reshuffle", "method": "reshuffle", "mtdd": 1200.0, "target": 3000.0, "haircut": 0.0},
        {"desc": "F3 Severe MTDD ($1,200) Bootstrap", "method": "bootstrap", "mtdd": 1200.0, "target": 3000.0, "haircut": 0.0},
        {"desc": "F3 Higher Target ($4,000) Bootstrap", "method": "bootstrap", "mtdd": 2000.0, "target": 4000.0, "haircut": 0.0},
        {"desc": "F3 10% Win Haircut Bootstrap", "method": "bootstrap", "mtdd": 2000.0, "target": 3000.0, "haircut": 0.10},
        {"desc": "F3 20% Win Haircut Bootstrap", "method": "bootstrap", "mtdd": 2000.0, "target": 3000.0, "haircut": 0.20},
    ]

    for cfg in mc_configs:
        res = run_monte_carlo_adversarial(
            trades=f3_pnl,
            n_sims=10000,
            initial_capital=50000.0,
            mtdd=cfg["mtdd"],
            target=cfg["target"],
            method=cfg["method"],
            win_haircut=cfg["haircut"],
            seed=42
        )
        print(f"  {cfg['desc']:<38} -> Ruin: {res['ruin_rate_pct']:>5.2f}% | Pass: {res['pass_rate_pct']:>5.2f}% | P50 DD: ${res['p50_dd']:>6.2f} | P95 DD: ${res['p95_dd']:>6.2f} | CVaR95: ${res['cvar_95_dd']:>6.2f}")

    # 6. Monte Carlo Adversarial Sweeps for F4
    print("\n[+] 6. MONTE CARLO ADVERSARIAL SWEEPS (F4 IS N1 - 10,000 SIMULATIONS)")
    f4_pnl = df_f4["net_pnl_usd"].values
    mc_configs_f4 = [
        {"desc": "F4 Baseline Reshuffle", "method": "reshuffle", "mtdd": 2000.0, "target": 3000.0, "haircut": 0.0},
        {"desc": "F4 Baseline Bootstrap", "method": "bootstrap", "mtdd": 2000.0, "target": 3000.0, "haircut": 0.0},
        {"desc": "F4 Severe MTDD ($1,000) Bootstrap", "method": "bootstrap", "mtdd": 1000.0, "target": 3000.0, "haircut": 0.0},
        {"desc": "F4 Higher Target ($6,000) Bootstrap", "method": "bootstrap", "mtdd": 2000.0, "target": 6000.0, "haircut": 0.0},
        {"desc": "F4 20% Win Haircut Bootstrap", "method": "bootstrap", "mtdd": 2000.0, "target": 3000.0, "haircut": 0.20},
        {"desc": "F4 Doubled Friction Bootstrap", "method": "bootstrap", "mtdd": 2000.0, "target": 3000.0, "haircut": 0.0, "pnl_override": df_f4["gross_pnl_usd"].values - df_f4["friction_usd"].values * 2.0},
    ]

    for cfg in mc_configs_f4:
        pnl_to_use = cfg.get("pnl_override", f4_pnl)
        res = run_monte_carlo_adversarial(
            trades=pnl_to_use,
            n_sims=10000,
            initial_capital=50000.0,
            mtdd=cfg["mtdd"],
            target=cfg["target"],
            method=cfg["method"],
            win_haircut=cfg["haircut"],
            seed=42
        )
        print(f"  {cfg['desc']:<38} -> Ruin: {res['ruin_rate_pct']:>5.2f}% | Pass: {res['pass_rate_pct']:>5.2f}% | P50 DD: ${res['p50_dd']:>6.2f} | P95 DD: ${res['p95_dd']:>6.2f} | CVaR95: ${res['cvar_95_dd']:>6.2f}")

    print("\n" + "=" * 80)
    print(" EMPIRICAL VERIFICATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
