import pandas as pd
from src.strategies.f4_optimizacion_is_n2 import (
    F4OptimizationISN2Strategy,
    F4OptimizationISN2Backtester,
)

df_es = pd.read_parquet('data/raw/CME_EXT_ES_5m.parquet')
df_nq = pd.read_parquet('data/raw/CME_EXT_NQ_5m.parquet')
if df_es.index.tz is not None:
    df_es.index = df_es.index.tz_convert('America/New_York').tz_localize(None)
if df_nq.index.tz is not None:
    df_nq.index = df_nq.index.tz_convert('America/New_York').tz_localize(None)

common_idx = df_nq.index.intersection(df_es.index)
df_nq = df_nq.loc[common_idx]
df_es = df_es.loc[common_idx]

for k in [390, 780, 1170, 2340]:
    strat_params = {
        'contract_mode': 'micro',
        'use_kalman': False,
        'unify_beta_ols': True,
        'max_coprime_nq': 7,
        'allow_sub_baskets': True,
        'dynamic_basket_ratio': True,
        's_open': 1.25,
        's_close': 0.25,
        's_stop': 3.25,
        'k_calib': k,
        'min_calib_bars': 390,
        'source_tz': 'America/New_York',
        'target_tz': 'America/New_York',
    }
    strategy = F4OptimizationISN2Strategy(params=strat_params)
    backtester = F4OptimizationISN2Backtester(
        strategy=strategy,
        account_size=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_loss=2000.0,
        profit_target=3000.0,
        early_stopping=False
    )
    res = backtester.run(df_nq=df_nq, df_es=df_es)
    audit = res.get('cointegration_audit', {})
    eval_b = audit.get('bars_evaluated')
    valid_b = audit.get('valid_ou_count')
    adf_b = audit.get('adf_passed_count')
    tau_b = audit.get('tau_passed_count')
    trades_n = len(res['trades'])
    print(f"k_calib={k}: evaluated={eval_b}, valid_ou={valid_b}, adf_passed={adf_b}, tau_passed={tau_b}, trades={trades_n}")
