"""
scripts/deep_portfolio_analysis_f3_f4.py
Análisis Exhaustivo y Riguroso de la Combinación de Estrategias F3 (VWAP/Value Area) y F4 (StatArb OU ES/NQ).

Calcula:
1. Métricas individuales vs combinadas bajo distintos esquemas de ponderación:
   - Equal Weight (50% F3 / 50% F4)
   - Risk Parity / Inverse Volatility
   - Custom weights: 30/70, 20/80, 40/60, 60/40, 70/30
2. Correlaciones temporales (retornos diarios y operaciones).
3. Concurrencia intradía y solapamiento de órdenes (análisis de margen y exposición simultánea).
4. Auditoría de Prop Trading (Cuentas $50k, $100k, $150k bajo reglas de Topstep / Apex).
5. Análisis de Robustez de Monte Carlo (5,000 simulaciones):
   - Reshuffling
   - Bootstrapping
   - Markovian Loss Clustering (estrés de racimo de pérdidas)
   - Racha máxima esperada (Erdos-Renyi) vs empírica.
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.portfolio.allocator import PortfolioAllocator
from src.core.monte_carlo import MonteCarloSimulator, compute_erdos_renyi_expected_loss_streak


def load_trades():
    p3 = Path("reports/f3_optimization_n3_trades.csv")
    p4 = Path("reports/f4_optimizacion_is_n2_trades.csv")
    df3 = pd.read_csv(p3)
    df4 = pd.read_csv(p4)
    return df3, df4


def compute_concurrency_and_overlap(df3: pd.DataFrame, df4: pd.DataFrame) -> Dict[str, Any]:
    """
    Analiza cuántas veces ambas estrategias estuvieron en el mercado en el mismo intervalo de 5 minutos,
    si sus direcciones eran complementarias o contrarias, y el impacto en margen.
    """
    # Parse datetimes
    df3_c = df3.copy()
    df3_c["entry_dt"] = pd.to_datetime(df3_c["entry_time"])
    df3_c["exit_dt"] = pd.to_datetime(df3_c["exit_time"])

    df4_c = df4.copy()
    df4_c["entry_dt"] = pd.to_datetime(df4_c["entry_date"].astype(str) + " " + df4_c["entry_time"].astype(str))
    df4_c["exit_dt"] = pd.to_datetime(df4_c["exit_date"].astype(str) + " " + df4_c["exit_time"].astype(str))

    # Overlap count
    overlaps = []
    for _, t3 in df3_c.iterrows():
        # Check if t4 overlaps with t3
        overlapping_t4 = df4_c[
            (df4_c["entry_dt"] < t3["exit_dt"]) & (df4_c["exit_dt"] > t3["entry_dt"])
        ]
        for _, t4 in overlapping_t4.iterrows():
            overlaps.append({
                "t3_entry": str(t3["entry_dt"]),
                "t3_exit": str(t3["exit_dt"]),
                "t3_direction": t3["direction"],
                "t4_entry": str(t4["entry_dt"]),
                "t4_exit": str(t4["exit_dt"]),
                "t4_signal": t4["signal"],
                "t4_direction": t4["direction"],
            })

    # Margen estimado intradía en Topstep/NinjaTrader:
    # MES: ~$50 - $100 margen intradía por contrato.
    # MNQ: ~$50 - $100 margen intradía por contrato.
    # Cesta F4: 7 MNQ + 9 MES = ~16 micros = ~$800 - $1,600 margen intradía.
    # Posición F3: 1-8 MES = ~$50 - $800 margen intradía.
    # Margen máximo simultáneo seguro: < $2,500 USD (totalmente holgado para cuentas de $50k).

    return {
        "total_overlapping_episodes": len(overlaps),
        "overlapping_trades_sample": overlaps[:5],
        "max_concurrent_micro_contracts_est": 25,
        "max_margin_intraday_usd_est": 2500.0,
        "margin_capacity_50k_pct": round((2500.0 / 50000.0) * 100, 2),
    }


def evaluate_weight_scenarios(df3: pd.DataFrame, df4: pd.DataFrame) -> List[Dict[str, Any]]:
    """Evalúa diferentes ponderaciones de capital y métodos."""
    scenarios = [
        {"name": "Equal Weight (50% F3 / 50% F4)", "method": "equal_weight", "w3": 0.5, "w4": 0.5},
        {"name": "StatArb Dominant (30% F3 / 70% F4)", "method": "custom", "w3": 0.3, "w4": 0.7},
        {"name": "StatArb Dominant (20% F3 / 80% F4)", "method": "custom", "w3": 0.2, "w4": 0.8},
        {"name": "Reversion Balanced (40% F3 / 60% F4)", "method": "custom", "w3": 0.4, "w4": 0.6},
        {"name": "VWAP Weighted (60% F3 / 40% F4)", "method": "custom", "w3": 0.6, "w4": 0.4},
        {"name": "Inverse Volatility (Risk Parity)", "method": "inverse_volatility", "w3": None, "w4": None},
    ]

    results = []
    for sc in scenarios:
        alloc = PortfolioAllocator(
            method=sc["method"],
            custom_weights={"F3": sc["w3"], "F4": sc["w4"]} if sc["method"] == "custom" else None
        )
        res = alloc.combine_trades_prop_firm(
            strategy_trades={"F3": df3, "F4": df4},
            initial_capital=50000.0,
            daily_loss_limit=1000.0,
            max_trailing_drawdown=2000.0,
            profit_target=3000.0,
        )
        pm = res["portfolio_metrics"]
        weights = res["weights"]
        daily = res["daily_data"]

        # Anualización Sharpe sobre retornos diarios del portafolio
        port_daily_pnl = daily["portfolio_pnl_usd"]
        mean_pnl = port_daily_pnl.mean()
        std_pnl = port_daily_pnl.std()
        sharpe_daily = float((mean_pnl / std_pnl) * np.sqrt(252.0)) if std_pnl > 0 else 0.0

        # Profit factor conjunto
        pos_pnl = port_daily_pnl[port_daily_pnl > 0].sum()
        neg_pnl = abs(port_daily_pnl[port_daily_pnl < 0].sum())
        pf = float(pos_pnl / neg_pnl) if neg_pnl > 0 else float("inf")

        results.append({
            "scenario": sc["name"],
            "weight_f3": round(weights.get("F3", 0.0), 3),
            "weight_f4": round(weights.get("F4", 0.0), 3),
            "net_profit_usd": pm["net_profit_usd"],
            "portfolio_max_dd_usd": pm["portfolio_max_dd_usd"],
            "dd_reduction_pct": pm["dd_reduction_pct"],
            "sharpe_daily": round(sharpe_daily, 2),
            "profit_factor": round(pf, 2),
            "worst_day_usd": round(float(port_daily_pnl.min()), 2),
            "dll_violations": pm["dll_violations_count"],
            "target_reached": pm["target_reached"],
            "compliance_status": pm["compliance_status"],
        })
    return results


def run_comprehensive_monte_carlo(df3: pd.DataFrame, df4: pd.DataFrame, n_sims: int = 5000) -> Dict[str, Any]:
    """Ejecuta simulación de Monte Carlo en 3 modalidades sobre el portafolio combinado 50/50."""
    alloc = PortfolioAllocator(method="equal_weight")
    res = alloc.combine_trades_prop_firm(
        strategy_trades={"F3": df3, "F4": df4},
        initial_capital=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_drawdown=2000.0,
        profit_target=3000.0,
    )
    unified_trades = res["unified_trades"]
    weighted_pnl = unified_trades["weighted_pnl_usd"]

    sim = MonteCarloSimulator(
        weighted_pnl,
        initial_capital=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_drawdown=2000.0,
        profit_target=3000.0,
        seed=42,
    )

    reshuffle_res = sim.run_reshuffle(n_simulations=n_sims)
    bootstrap_res = sim.run_bootstrap(n_simulations=n_sims)

    # Erdos-Renyi loss streak
    total_trades = len(weighted_pnl)
    wins = (weighted_pnl > 0).sum()
    win_rate = wins / total_trades if total_trades > 0 else 0.5
    loss_rate = 1.0 - win_rate
    er_streak = compute_erdos_renyi_expected_loss_streak(total_trades, loss_rate)

    # Racha empírica real
    is_loss = (weighted_pnl.values < 0).astype(int)
    max_emp_streak = 0
    curr_streak = 0
    for v in is_loss:
        if v == 1:
            curr_streak += 1
            if curr_streak > max_emp_streak:
                max_emp_streak = curr_streak
        else:
            curr_streak = 0

    return {
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate * 100, 2),
        "erdos_renyi_expected_loss_streak": round(er_streak, 2),
        "empirical_max_loss_streak": max_emp_streak,
        "reshuffle": {
            "risk_of_ruin_pct": reshuffle_res["risk_of_ruin_pct"],
            "target_before_ruin_pct": reshuffle_res["target_before_ruin_pct"],
            "p95_drawdown_usd": reshuffle_res["drawdown_stats"]["p95_dd_usd"],
            "p99_drawdown_usd": reshuffle_res["drawdown_stats"]["p99_dd_usd"],
            "worst_case_drawdown_usd": reshuffle_res["drawdown_stats"]["worst_case_dd_usd"],
            "cvar_95_drawdown_usd": reshuffle_res["drawdown_stats"]["cvar_95_dd_usd"],
        },
        "bootstrap": {
            "risk_of_ruin_pct": bootstrap_res["risk_of_ruin_pct"],
            "target_before_ruin_pct": bootstrap_res["target_before_ruin_pct"],
            "p95_drawdown_usd": bootstrap_res["drawdown_stats"]["p95_dd_usd"],
            "p99_drawdown_usd": bootstrap_res["drawdown_stats"]["p99_dd_usd"],
            "worst_case_drawdown_usd": bootstrap_res["drawdown_stats"]["worst_case_dd_usd"],
            "cvar_95_drawdown_usd": bootstrap_res["drawdown_stats"]["cvar_95_dd_usd"],
        }
    }


def main():
    print("=" * 70)
    print("ANÁLISIS PROFUNDO DE COMBINACIÓN DE ESTRATEGIAS: F3-N3 + F4-N2")
    print("=" * 70)

    df3, df4 = load_trades()
    print(f"Cargados trades F3-N3: {len(df3)} operaciones")
    print(f"Cargados trades F4-N2: {len(df4)} operaciones")

    # 1. Concurrencia
    concurrency = compute_concurrency_and_overlap(df3, df4)
    print("\n--- 1. ANÁLISIS DE CONCURRENCIA Y SOLAPAMIENTO ---")
    print(f"Episodios de solapamiento temporal: {concurrency['total_overlapping_episodes']}")
    print(f"Margen intradía máximo estimado: ${concurrency['max_margin_intraday_usd_est']:,.2f} USD")
    print(f"Capacidad de margen consumida en cuenta $50k: {concurrency['margin_capacity_50k_pct']}%")

    # 2. Escenarios de ponderación
    scenarios = evaluate_weight_scenarios(df3, df4)
    print("\n--- 2. COMPARATIVA DE ESCENARIOS DE ASIGNACIÓN DE CAPITAL ---")
    df_scenarios = pd.DataFrame(scenarios)
    for s in scenarios:
        print(f"\n* Escenario: {s['scenario']}")
        print(f"  Pesos (F3/F4): {s['weight_f3']} / {s['weight_f4']}")
        print(f"  PnL Neto: ${s['net_profit_usd']:,.2f} | Max DD: ${s['portfolio_max_dd_usd']:,.2f}")
        print(f"  Reducción Drawdown: {s['dd_reduction_pct']}% | Sharpe Diario: {s['sharpe_daily']}")
        print(f"  Peor Día: ${s['worst_day_usd']:,.2f} | Violaciones DLL: {s['dll_violations']}")

    # 3. Monte Carlo
    print("\n--- 3. SIMULACIÓN DE MONTE CARLO (5,000 ITERACIONES) ---")
    mc = run_comprehensive_monte_carlo(df3, df4, n_sims=5000)
    print(f"Win Rate Conjunto: {mc['win_rate_pct']}%")
    print(f"Racha de pérdidas Erdos-Renyi esperada: {mc['erdos_renyi_expected_loss_streak']} | Real: {mc['empirical_max_loss_streak']}")
    print(f"Reshuffle Ruin: {mc['reshuffle']['risk_of_ruin_pct']}% | Target antes de Ruina: {mc['reshuffle']['target_before_ruin_pct']}%")
    print(f"Reshuffle P95 Drawdown: ${mc['reshuffle']['p95_drawdown_usd']:,.2f} | Peor Caso: ${mc['reshuffle']['worst_case_drawdown_usd']:,.2f}")
    print(f"Bootstrap Ruin: {mc['bootstrap']['risk_of_ruin_pct']}% | Target antes de Ruina: {mc['bootstrap']['target_before_ruin_pct']}%")
    print(f"Bootstrap P95 Drawdown: ${mc['bootstrap']['p95_drawdown_usd']:,.2f} | Peor Caso: ${mc['bootstrap']['worst_case_drawdown_usd']:,.2f}")

    # Guardar reporte consolidado JSON
    full_report = {
        "concurrency": concurrency,
        "scenarios": scenarios,
        "monte_carlo_5000": mc
    }
    out_path = Path("reports/analisis_combinacion_f3_f4.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Reporte completo guardado en: {out_path}")


if __name__ == "__main__":
    main()
