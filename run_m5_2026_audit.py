import pandas as pd
import json
from src.strategies.f4_optimizacion_is_n2 import (
    F4OptimizationISN2Strategy,
    F4OptimizationISN2Backtester,
)
from src.strategies.mean_reversion.f3_optimization_is_n3 import (
    F3OptimizationISN3Strategy,
)

print("=" * 80)
print("BACKTESTING INTEGRAL CME M5 (5 MINUTOS) - DATOS DE MERCADO 2026")
print("=" * 80)

# Datasets 2026 disponibles
datasets = [
    {
        "name": "CME_EXT_2026 (Agosto a Noviembre 2026)",
        "es": "data/raw/CME_EXT_ES_5m.parquet",
        "nq": "data/raw/CME_EXT_NQ_5m.parquet",
        "is_tz": True
    },
    {
        "name": "CME_MINI_2026 (Octubre a Noviembre 2026)",
        "es": "data/raw/CME_MINI_ES_2026_5m.parquet",
        "nq": "data/raw/CME_MINI_NQ_2026_5m.parquet",
        "is_tz": False
    }
]

for d in datasets:
    print(f"\n>>> Evaluando Dataset: {d['name']} <<<")
    df_es = pd.read_parquet(d["es"])
    df_nq = pd.read_parquet(d["nq"])
    
    if df_es.index.tz is not None:
        df_es.index = df_es.index.tz_convert("America/New_York").tz_localize(None)
    if df_nq.index.tz is not None:
        df_nq.index = df_nq.index.tz_convert("America/New_York").tz_localize(None)
        
    common_idx = df_nq.index.intersection(df_es.index)
    df_nq = df_nq.loc[common_idx]
    df_es = df_es.loc[common_idx]
    
    print(f"Rango temporal: {common_idx[0]} hasta {common_idx[-1]} ({len(common_idx):,} barras de 5m)")

    # 1. F4-N2 con ventana estándar (k=2340 barras)
    strat_k2340 = F4OptimizationISN2Strategy(params={
        "contract_mode": "micro", "use_kalman": False, "unify_beta_ols": True,
        "max_coprime_nq": 7, "allow_sub_baskets": True, "dynamic_basket_ratio": True,
        "s_open": 1.25, "s_close": 0.25, "s_stop": 3.25, "k_calib": 2340, "min_calib_bars": 390,
        "phi_dll": 0.60, "psi_tmdd": 0.30, "survival_floor": 105.0, "time_decay_multiplier": 3.5,
        "n_basket_max": 3, "source_tz": "America/New_York", "target_tz": "America/New_York"
    })
    bt_k2340 = F4OptimizationISN2Backtester(
        strategy=strat_k2340, account_size=50000.0, daily_loss_limit=1000.0,
        max_trailing_loss=2000.0, profit_target=3000.0, early_stopping=False
    )
    res_k2340 = bt_k2340.run(df_nq, df_es)
    m_k2340 = res_k2340["metrics"]
    print(f"  [F4-N2 K=2340] Trades: {m_k2340['Total Trades']} | PnL: ${m_k2340['Net Profit ($)']:,.2f} | MaxDD: ${m_k2340['Max Drawdown ($)']:,.2f}")

    # 2. F4-N2 con ventana adaptativa intradía (k=390 barras)
    strat_k390 = F4OptimizationISN2Strategy(params={
        "contract_mode": "micro", "use_kalman": False, "unify_beta_ols": True,
        "max_coprime_nq": 7, "allow_sub_baskets": True, "dynamic_basket_ratio": True,
        "s_open": 1.25, "s_close": 0.25, "s_stop": 3.25, "k_calib": 390, "min_calib_bars": 390,
        "phi_dll": 0.60, "psi_tmdd": 0.30, "survival_floor": 105.0, "time_decay_multiplier": 3.5,
        "n_basket_max": 3, "source_tz": "America/New_York", "target_tz": "America/New_York"
    })
    bt_k390 = F4OptimizationISN2Backtester(
        strategy=strat_k390, account_size=50000.0, daily_loss_limit=1000.0,
        max_trailing_loss=2000.0, profit_target=3000.0, early_stopping=False
    )
    res_k390 = bt_k390.run(df_nq, df_es)
    m_k390 = res_k390["metrics"]
    print(f"  [F4-N2 K=390]  Trades: {m_k390['Total Trades']} | WinRate: {m_k390['Win Rate (%)']:.1f}% | PnL: ${m_k390['Net Profit ($)']:,.2f} | MaxDD: ${m_k390['Max Drawdown ($)']:,.2f} | PF: {m_k390['Profit Factor']:.2f}")

    # 3. F3-N3 Reversión a VWAP en ES 5m
    strat_f3 = F3OptimizationISN3Strategy(params={
        "symbol": "ES", "anchor_time": "09:30", "window_init_end": "10:15",
        "window_revers_end": "15:00", "window_inhibit_end": "15:15", "k_band": 2.20,
        "k_catastrophic": 2.75, "kappa_crit": 0.085, "slope_lookback_bars": 3,
        "max_vol_outside_ratio": 0.55, "min_kurtosis": 1.7, "min_dist_to_vwap_ratio": 0.80,
        "k_buffer_ticks": 3, "convergence_protect_ratio": 0.50, "account_size": 50000.0,
        "daily_loss_limit": 1000.0, "max_trailing_drawdown": 2000.0, "psi": 0.12,
        "max_std_contracts": 1, "max_micro_contracts": 8, "alpha_dll": 0.85,
        "max_intraday_consecutive_losses": 2, "scaling_factor": 0.65, "bar_minutes": 5.0
    })
    res_f3 = strat_f3.run_backtest(df_es)
    m_f3 = res_f3["metrics"]
    print(f"  [F3-N3 ES 5m]  Trades: {m_f3['Total Trades']} | WinRate: {m_f3['Win Rate (%)']:.1f}% | PnL: ${m_f3['P&L Neto Total ($)']:,.2f} | MaxDD: ${m_f3['Max Trailing Drawdown ($)']:,.2f} | PF: {m_f3['Profit Factor']:.2f}")

print("\n" + "=" * 80)
print("EJECUCION FINALIZADA CON EXITO")
print("=" * 80)
