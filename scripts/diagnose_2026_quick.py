import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.strategies.f4_optimizacion_is_n2 import CointegrationOUModelN2

df_es = pd.read_parquet("data/raw/CME_EXT_ES_5m.parquet")
df_nq = pd.read_parquet("data/raw/CME_EXT_NQ_5m.parquet")
if df_es.index.tz is not None:
    df_es.index = df_es.index.tz_convert("America/New_York").tz_localize(None)
if df_nq.index.tz is not None:
    df_nq.index = df_nq.index.tz_convert("America/New_York").tz_localize(None)

common_idx = df_nq.index.intersection(df_es.index)
df_nq = df_nq.loc[common_idx]
df_es = df_es.loc[common_idx]

model = CointegrationOUModelN2()
print(f"Total bars in 2026 CME_EXT: {len(common_idx):,} ({common_idx[0]} a {common_idx[-1]})")

for start in [0, 2500, 5000, 7500, 10000]:
    hist_nq = df_nq["Close"].iloc[start : start + 2340]
    hist_es = df_es["Close"].iloc[start : start + 2340]
    log_nq = np.log(hist_nq.values)
    log_es = np.log(hist_es.values)
    timestamps = list(hist_nq.index)

    beta_hedge, alpha_coint, residuals = model.estimate_cointegration_backward(log_nq, log_es)
    is_stat, t_stat, p_val = model.test_stationarity_mackinnon(residuals)
    ou_res = model.fit_ou_process_intraday(residuals, timestamps)

    # Stop loss
    stop_loss = model.calculate_unit_stop_loss_loss(
        sigma_eq=ou_res["sigma_eq"],
        p_nq=hist_nq.iloc[-1],
        n_nq_unit=3,
        m_nq=2.0,
        friction_basket_usd=25.0,
        delta_s_sl=2.0
    )
    b_alloc = model.calculate_baskets_allocation(
        risk_budget_usd=600.0,
        unit_sl_loss_usd=stop_loss,
        n_basket_max=3,
        min_survival_budget=105.0,
    )
    date_str = str(hist_nq.index[-1])
    print(
        f"End={date_str} | beta={beta_hedge:.4f} | adf_stat={t_stat:.2f} (p={p_val:.4f}, stat={is_stat}) | "
        f"tau_half={ou_res['tau_half_min']:.1f}m (valid={ou_res['is_valid']}) | stop_loss=${stop_loss:.2f} | "
        f"baskets={b_alloc}"
    )
