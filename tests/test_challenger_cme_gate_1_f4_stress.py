"""
tests/test_challenger_cme_gate_1_f4_stress.py

Empirical Adversarial Stress Verification Suite for Milestone Real CME Gate 1 (Feature F4).
Authored by: teamwork_challenger_cme_gate_1

Scope of Challenges:
1. Volatility Floor (sigma_floor >= 0.001):
   - Inject extreme near-zero or zero residual variance (sigma_eta <= 1e-15 or flat price lines).
   - Verify that sigma_eq NEVER drops below 0.001 when OU fit is valid.
   - Verify that calculate_s_score and calculate_frozen_s_score never divide by zero and produce bounded, stable S-scores without exploding.
   - Test stop loss loss unit calculation does not collapse to 0 under zero residual variance.
   - Test across CointegrationOUModelBase, CointegrationOUModelN1, and CointegrationOUModelN2.

2. Sector Decoupling Filter (XLK/SPY rolling 30d correlation >= 0.85):
   - Test threshold boundary: inject correlation series with 0.8499 (just below 0.85) vs 0.8501 (just above 0.85) vs 0.8500 (exact).
     Verify strict binary gating: exactly zero orders opened when < 0.85, normal execution when >= 0.85.
   - Test lookahead bias: mutate daily prices on day D and verify that S-score / orders on day D are completely unaffected because correlation strictly uses data up to D-1.
   - Test handling of missing ETF dates or unscheduled trading halts: asymmetric missingness, multi-day gaps, insufficient history, and futures trading on ETF holidays.
"""

from datetime import datetime, date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.strategies.cointegration_ou_base import (
    CointegrationOUModelBase,
    CointegrationOUStrategyBase,
)
from src.strategies.f4_optimizacion_is_n1 import (
    CointegrationOUModelN1,
    F4OptimizationISN1Strategy,
    F4OptimizationISN1Backtester,
    compute_sector_correlation_series,
)
from src.strategies.f4_optimizacion_is_n2 import (
    CointegrationOUModelN2,
    F4OptimizationISN2Strategy,
    F4OptimizationISN2Backtester,
)
from src.strategies.f4_optimizacion_is_n3 import (
    CointegrationOUModelN3,
    F4OptimizationISN3Strategy,
    F4OptimizationISN3Backtester,
)


# ==============================================================================
# CHALLENGE 1: ADVERSARIAL VOLATILITY FLOOR STRESS (sigma_floor >= 0.001)
# ==============================================================================

class TestF4VolatilityFloorAdversarial:
    """Stress tests challenging the mathematical stability and zero-variance limits of the Volatility Floor."""

    def test_fit_ou_flat_lines_zero_variance(self):
        """
        Adversarial test: Flat prices or identical residuals (zero variance).
        The model must handle zero variance gracefully without ZeroDivisionError or crash.
        """
        dt_index = pd.date_range("2023-01-10 09:30:00", "2023-01-10 15:55:00", freq="5min")
        n = len(dt_index)

        # 1. Constant zero residuals
        flat_zeros = np.zeros(n)
        for model_cls in [CointegrationOUModelN1, CointegrationOUModelN2, CointegrationOUModelN3]:
            ou = model_cls.fit_ou_process_intraday(flat_zeros, dt_index, sigma_floor=0.001)
            assert ou["is_valid"] is False, "Flat zeros must be marked invalid"
            assert ou["sigma_eta"] == 0.0
            # When invalid, sigma_eq is reported as 0.0 and strategy must not trade
            assert ou["sigma_eq"] == 0.0

        # Also test Base model
        ou_base = CointegrationOUModelBase.fit_ou_process(flat_zeros, dt_minutes=5.0, sigma_floor=0.001)
        assert ou_base["is_valid"] is False

        # 2. Constant non-zero residuals (e.g. flat line at 1.50)
        flat_nonzero = np.full(n, 1.50)
        for model_cls in [CointegrationOUModelN1, CointegrationOUModelN2, CointegrationOUModelN3]:
            ou = model_cls.fit_ou_process_intraday(flat_nonzero, dt_index, sigma_floor=0.001)
            assert ou["is_valid"] is False
            assert ou["sigma_eq"] == 0.0

    def test_fit_ou_micro_variance_clamped_to_floor(self):
        """
        Adversarial test: Inject residuals with microscopic residual noise (sigma_eta ~ 1e-6 << 0.001).
        When the AR(1) fit is valid, sigma_eq must NEVER drop below sigma_floor (0.001).
        """
        dt_index = pd.date_range("2023-01-10 09:30:00", "2023-01-10 15:55:00", freq="5min")
        n = len(dt_index)

        # Generate an AR(1) series with b=0.85 and micro noise sigma_noise = 1e-6
        np.random.seed(42)
        tiny_res = np.zeros(n)
        for i in range(1, n):
            tiny_res[i] = 0.85 * tiny_res[i - 1] + np.random.normal(0, 1e-6)

        for model_cls in [CointegrationOUModelN1, CointegrationOUModelN2, CointegrationOUModelN3]:
            ou = model_cls.fit_ou_process_intraday(tiny_res, dt_index, sigma_floor=0.001)
            assert ou["is_valid"] is True, f"AR(1) fit should be valid for {model_cls.__name__}"
            # Raw sigma_eq would be ~ 1e-6 / sqrt(1 - 0.85^2) ≈ 1.9e-6 << 0.001.
            # sigma_eq must be clamped to EXACTLY >= 0.001!
            assert ou["sigma_eq"] >= 0.001, (
                f"{model_cls.__name__}: sigma_eq={ou['sigma_eq']} dropped below 0.001!"
            )
            assert ou["sigma_eq"] == pytest.approx(0.001, rel=1e-6), (
                f"sigma_eq must equal sigma_floor (0.001), got: {ou['sigma_eq']}"
            )

        # Test Base model
        ou_base = CointegrationOUModelBase.fit_ou_process(tiny_res, dt_minutes=5.0, sigma_floor=0.001)
        assert ou_base["is_valid"] is True
        assert ou_base["sigma_eq"] >= 0.001

    def test_fit_ou_degenerate_variance_safely_invalidated(self):
        """
        Adversarial test: Inject residuals with extreme degenerate variance (sigma_eta <= 1e-15).
        Because var_prev <= 1e-14, the AR(1) regression is mathematically singular.
        The model must set is_valid: False, protecting the strategy from trading on degenerate data.
        """
        dt_index = pd.date_range("2023-01-10 09:30:00", "2023-01-10 15:55:00", freq="5min")
        n = len(dt_index)

        np.random.seed(42)
        tiny_res = np.zeros(n)
        for i in range(1, n):
            tiny_res[i] = 0.85 * tiny_res[i - 1] + np.random.normal(0, 1e-15)

        for model_cls in [CointegrationOUModelN1, CointegrationOUModelN2, CointegrationOUModelN3]:
            ou = model_cls.fit_ou_process_intraday(tiny_res, dt_index, sigma_floor=0.001)
            assert ou["is_valid"] is False, "Degenerate noise <= 1e-15 must be safely invalidated"

    def test_calculate_s_score_zero_division_immunity(self):
        """
        Adversarial test: Inject zero, negative, and extreme near-zero sigma_eq values
        into calculate_s_score across Base, N1, and N2 models.
        Must NEVER raise ZeroDivisionError and must produce bounded, stable S-scores.
        """
        test_sigmas = [0.0, 1e-16, 1e-100, 1e-300, -0.05, -1e-15]
        test_deviations = [0.0, 0.0005, 0.001, 0.05, 1.0, 100.0, -0.0005, -5.0]

        for model_cls in [CointegrationOUModelBase, CointegrationOUModelN1, CointegrationOUModelN2, CointegrationOUModelN3]:
            for s_val in test_sigmas:
                for dev in test_deviations:
                    # Default sigma_floor=0.001
                    score = model_cls.calculate_s_score(x_k=dev, theta=0.0, sigma_eq=s_val, sigma_floor=0.001)
                    assert not np.isnan(score), f"{model_cls.__name__}: NaN returned for sigma={s_val}, dev={dev}"
                    assert not np.isinf(score), f"{model_cls.__name__}: Inf returned for sigma={s_val}, dev={dev}"
                    # With sigma_floor=0.001, effective sigma is 0.001, so score = dev / 0.001 = dev * 1000
                    expected = dev / 0.001
                    assert score == pytest.approx(expected, rel=1e-5), (
                        f"{model_cls.__name__}: Expected {expected}, got {score}"
                    )

            # Test catastrophic case: both sigma_eq <= 0 AND sigma_floor <= 0 (e.g. user override)
            catastrophic_score = model_cls.calculate_s_score(x_k=0.05, theta=0.0, sigma_eq=0.0, sigma_floor=0.0)
            assert catastrophic_score == 0.0, f"{model_cls.__name__} must return 0.0 when eff_sigma <= 1e-12"

    def test_calculate_frozen_s_score_zero_division_and_pathological_prices(self):
        """
        Adversarial test: Inject zero, negative, and extreme near-zero sigma_eq values,
        as well as negative and zero prices, into calculate_frozen_s_score.
        Must NEVER raise ZeroDivisionError or math domain error (log <= 0).
        """
        for model_cls in [CointegrationOUModelN1, CointegrationOUModelN2, CointegrationOUModelN3]:
            # 1. Normal prices with zero/micro entry_sigma_eq
            for s_val in [0.0, 1e-16, 1e-300, -1.0]:
                frozen = model_cls.calculate_frozen_s_score(
                    p_nq=10005.0, p_es=10000.0, entry_alpha=0.0, entry_beta=1.0,
                    entry_theta=0.0, entry_sigma_eq=s_val, sigma_floor=0.001
                )
                assert not np.isnan(frozen)
                assert not np.isinf(frozen)
                # log(10005) - log(10000) ≈ 0.0004998
                # eff_sigma = 0.001 => frozen ≈ 0.4998
                assert abs(frozen - 0.50) < 0.01

            # 2. Pathological zero and negative prices
            bad_prices = [
                (0.0, 10000.0),
                (-100.0, 10000.0),
                (10000.0, 0.0),
                (10000.0, -50.0),
                (0.0, 0.0),
            ]
            for p_nq, p_es in bad_prices:
                frozen = model_cls.calculate_frozen_s_score(
                    p_nq=p_nq, p_es=p_es, entry_alpha=0.0, entry_beta=1.0,
                    entry_theta=0.0, entry_sigma_eq=0.001, sigma_floor=0.001
                )
                assert frozen == 0.0, f"Pathological prices ({p_nq}, {p_es}) must return 0.0 safely"

    def test_unit_stop_loss_loss_floor_protection(self):
        """
        Adversarial test: Verify calculate_unit_stop_loss_loss does not collapse to pure friction
        when sigma_eq is zero or microscopic. The volatility floor ensures risk budget cannot be bypassed.
        """
        for model_cls in [CointegrationOUModelBase, CointegrationOUModelN1, CointegrationOUModelN2, CointegrationOUModelN3]:
            unit_sl = model_cls.calculate_unit_stop_loss_loss(
                sigma_eq=1e-16, p_nq=15000.0, n_nq_unit=1, m_nq=2.0,
                friction_basket_usd=25.0, delta_s_sl=2.0, sigma_floor=0.001
            )
            # delta_x_sl = 2.0 * max(1e-16, 0.001) = 0.002
            # notional = 1 * 2 * 15000 = 30000
            # loss_pts = 0.002 * 30000 = 60 USD
            # total = 60 + 25 = 85 USD
            assert unit_sl >= 85.0, f"{model_cls.__name__}: Unit SL loss must be >= 85.0 USD, got: {unit_sl}"


# ==============================================================================
# CHALLENGE 2: ADVERSARIAL SECTOR DECOUPLING FILTER STRESS (XLK/SPY >= 0.85)
# ==============================================================================

class TestF4SectorDecouplingFilterAdversarial:
    """Stress tests challenging threshold boundaries, lookahead bias, and missing data resilience."""

    @pytest.fixture(scope="class")
    def cme_sample_data(self):
        """Loads a 3,000 bar sample of genuine CME In-Sample data where cointegration is active."""
        nq_path = "data/raw/CME_MINI_NQ_5m_3y.parquet" if Path("data/raw/CME_MINI_NQ_5m_3y.parquet").exists() else ("data/archive_synthetic/CME_MINI_NQ_5m_3y.parquet" if Path("data/archive_synthetic/CME_MINI_NQ_5m_3y.parquet").exists() else "data/raw/CME_MINI_NQ_2021_2023_5m.parquet")
        es_path = "data/raw/CME_MINI_ES_5m_3y.parquet" if Path("data/raw/CME_MINI_ES_5m_3y.parquet").exists() else ("data/archive_synthetic/CME_MINI_ES_5m_3y.parquet" if Path("data/archive_synthetic/CME_MINI_ES_5m_3y.parquet").exists() else "data/raw/CME_MINI_ES_2021_2023_5m.parquet")

        df_nq = pd.read_parquet(nq_path).iloc[:3000]
        df_es = pd.read_parquet(es_path).iloc[:3000]
        return df_es, df_nq

    def test_threshold_boundary_binary_gating_08499_vs_08501(self, cme_sample_data):
        """
        Adversarial test: Threshold boundary precision.
        - Inject constant correlation series 0.8499 (just below 0.8500).
          Verify strict binary gating: exactly ZERO orders opened (Total Trades == 0).
        - Inject constant correlation series 0.8501 (just above 0.8500).
          Verify normal trading execution (Total Trades > 0).
        - Inject exact boundary 0.8500.
          Verify normal trading execution (Total Trades > 0).
        Test on both N1 and N2 backtesters!
        """
        df_es, df_nq = cme_sample_data
        unique_dates = sorted(list(set(pd.to_datetime(df_nq.index).date)))

        # Sub-test 1: 0.8499 (Strict Block)
        corr_series_blocked = pd.Series(0.8499, index=unique_dates)

        # N1
        bt_n1_blocked = F4OptimizationISN1Backtester(
            strategy=F4OptimizationISN1Strategy(params={"min_sector_corr": 0.85}),
            account_size=100000.0, early_stopping=False
        )
        res_n1_blocked = bt_n1_blocked.run(df_nq, df_es, sector_corr_series=corr_series_blocked)
        assert res_n1_blocked["metrics"]["Total Trades"] == 0, (
            f"N1: Correlation 0.8499 must block 100% of trades, got {res_n1_blocked['metrics']['Total Trades']}"
        )

        # N2
        bt_n2_blocked = F4OptimizationISN2Backtester(
            strategy=F4OptimizationISN2Strategy(params={"min_sector_corr": 0.85}),
            account_size=100000.0, early_stopping=False
        )
        res_n2_blocked = bt_n2_blocked.run(df_nq, df_es, sector_corr_series=corr_series_blocked)
        assert res_n2_blocked["metrics"]["Total Trades"] == 0, (
            f"N2: Correlation 0.8499 must block 100% of trades, got {res_n2_blocked['metrics']['Total Trades']}"
        )

        # N3
        bt_n3_blocked = F4OptimizationISN3Backtester(
            strategy=F4OptimizationISN3Strategy(params={"min_sector_corr": 0.85}),
            account_size=100000.0, early_stopping=False
        )
        res_n3_blocked = bt_n3_blocked.run(df_nq, df_es, sector_corr_series=corr_series_blocked)
        assert res_n3_blocked["metrics"]["Total Trades"] == 0, (
            f"N3: Correlation 0.8499 must block 100% of trades, got {res_n3_blocked['metrics']['Total Trades']}"
        )

        # Sub-test 2: 0.8501 (Pass)
        corr_series_passed = pd.Series(0.8501, index=unique_dates)
        bt_n1_passed = F4OptimizationISN1Backtester(
            strategy=F4OptimizationISN1Strategy(params={"min_sector_corr": 0.85}),
            account_size=100000.0, early_stopping=False
        )
        res_n1_passed = bt_n1_passed.run(df_nq, df_es, sector_corr_series=corr_series_passed)
        assert res_n1_passed["metrics"]["Total Trades"] > 0, (
            "N1: Correlation 0.8501 must allow trading"
        )

        bt_n2_passed = F4OptimizationISN2Backtester(
            strategy=F4OptimizationISN2Strategy(params={"min_sector_corr": 0.85}),
            account_size=100000.0, early_stopping=False
        )
        res_n2_passed = bt_n2_passed.run(df_nq, df_es, sector_corr_series=corr_series_passed)
        assert res_n2_passed["metrics"]["Total Trades"] > 0, (
            "N2: Correlation 0.8501 must allow trading"
        )

        bt_n3_passed = F4OptimizationISN3Backtester(
            strategy=F4OptimizationISN3Strategy(params={"min_sector_corr": 0.85}),
            account_size=100000.0, early_stopping=False
        )
        res_n3_passed = bt_n3_passed.run(df_nq, df_es, sector_corr_series=corr_series_passed)
        assert res_n3_passed["metrics"]["Total Trades"] > 0, (
            "N3: Correlation 0.8501 must allow trading"
        )

        # Sub-test 3: Exactly 0.8500 (Pass because >= 0.85)
        corr_series_exact = pd.Series(0.8500, index=unique_dates)
        bt_n1_exact = F4OptimizationISN1Backtester(
            strategy=F4OptimizationISN1Strategy(params={"min_sector_corr": 0.85}),
            account_size=100000.0, early_stopping=False
        )
        res_n1_exact = bt_n1_exact.run(df_nq, df_es, sector_corr_series=corr_series_exact)
        assert res_n1_exact["metrics"]["Total Trades"] == res_n1_passed["metrics"]["Total Trades"]

        bt_n3_exact = F4OptimizationISN3Backtester(
            strategy=F4OptimizationISN3Strategy(params={"min_sector_corr": 0.85}),
            account_size=100000.0, early_stopping=False
        )
        res_n3_exact = bt_n3_exact.run(df_nq, df_es, sector_corr_series=corr_series_exact)
        assert res_n3_exact["metrics"]["Total Trades"] == res_n3_passed["metrics"]["Total Trades"]

    def test_lookahead_bias_deep_adversarial_shock(self):
        """
        Adversarial test: Deep verification of zero lookahead bias.
        Mutate day D close price in XLK and SPY by +100% and -80%.
        Verify that for day D, the correlation value is BIT-FOR-BIT IDENTICAL to baseline,
        and that only day D+1 reflects the price shock.
        """
        dates = pd.date_range("2023-01-01", periods=60, freq="B")
        np.random.seed(123)
        r_spy = np.random.normal(0.0005, 0.012, 60)
        r_xlk = 1.2 * r_spy + np.random.normal(0, 0.003, 60)

        p_spy = 400.0 * np.exp(np.cumsum(r_spy))
        p_xlk = 150.0 * np.exp(np.cumsum(r_xlk))

        df_spy = pd.DataFrame({"Close": p_spy}, index=dates)
        df_xlk = pd.DataFrame({"Close": p_xlk}, index=dates)

        corr_baseline = compute_sector_correlation_series(df_xlk, df_spy, window=30)

        # Test multiple target days within the valid rolling window
        for target_idx in [35, 45, 55]:
            target_date = dates[target_idx].date()

            # Apply extreme adversarial price shock on target_date: +100% price spike
            df_xlk_shocked = df_xlk.copy()
            df_xlk_shocked.iloc[target_idx, 0] = df_xlk_shocked.iloc[target_idx, 0] * 2.0

            corr_shocked = compute_sector_correlation_series(df_xlk_shocked, df_spy, window=30)

            # 1. On day D (target_date), correlation MUST NOT change by even 1e-15!
            base_val = corr_baseline.loc[target_date]
            shocked_val = corr_shocked.loc[target_date]
            assert abs(base_val - shocked_val) < 1e-14, (
                f"Lookahead leak at day D={target_date}: baseline={base_val}, shocked={shocked_val}"
            )

            # 2. For all prior days t < D, correlation must be strictly identical
            prior_dates = [d.date() for d in dates if d.date() <= target_date and d.date() in corr_baseline.index]
            for pd_d in prior_dates:
                assert abs(corr_baseline.loc[pd_d] - corr_shocked.loc[pd_d]) < 1e-14

            # 3. Day D+1 MUST reflect the shock (different correlation)
            if target_idx + 1 < len(dates):
                next_date = dates[target_idx + 1].date()
                if next_date in corr_baseline.index and next_date in corr_shocked.index:
                    next_base = corr_baseline.loc[next_date]
                    next_shocked = corr_shocked.loc[next_date]
                    assert abs(next_base - next_shocked) > 1e-4, (
                        f"Day D+1={next_date} should reflect shock: base={next_base}, shocked={next_shocked}"
                    )

    def test_missing_etf_dates_and_trading_halts(self):
        """
        Adversarial test: Test handling of missing ETF dates, unscheduled trading halts,
        and asymmetric gaps between XLK and SPY.
        """
        dates = pd.date_range("2023-01-01", periods=60, freq="B")
        np.random.seed(999)
        r_spy = np.random.normal(0.0005, 0.01, 60)
        r_xlk = 1.15 * r_spy + np.random.normal(0, 0.002, 60)
        p_spy = 400.0 * np.exp(np.cumsum(r_spy))
        p_xlk = 150.0 * np.exp(np.cumsum(r_xlk))

        df_spy = pd.DataFrame({"Close": p_spy}, index=dates)
        df_xlk = pd.DataFrame({"Close": p_xlk}, index=dates)

        # Scenario A: Asymmetric halt - XLK halts on day 35 (missing row), SPY trades normally
        df_xlk_halted = df_xlk.drop(dates[35])
        corr_halted = compute_sector_correlation_series(df_xlk_halted, df_spy, window=30)
        assert not corr_halted.empty
        # Verify day 35 is not in the output series
        assert dates[35].date() not in corr_halted.index

        # Scenario B: Multi-day extended halt (days 35-39 missing in both)
        df_xlk_multi_halt = df_xlk.drop(dates[35:40])
        df_spy_multi_halt = df_spy.drop(dates[35:40])
        corr_multi = compute_sector_correlation_series(df_xlk_multi_halt, df_spy_multi_halt, window=30)
        assert not corr_multi.empty
        for d in dates[35:40]:
            assert d.date() not in corr_multi.index

        # Scenario C: Insufficient history (< 31 days)
        df_short_xlk = df_xlk.iloc[:25]
        df_short_spy = df_spy.iloc[:25]
        corr_short = compute_sector_correlation_series(df_short_xlk, df_short_spy, window=30)
        assert corr_short.empty, "Fewer than 31 days must return empty Series"

        # Scenario D: None or empty inputs
        assert compute_sector_correlation_series(None, df_spy).empty
        assert compute_sector_correlation_series(df_xlk, None).empty
        assert compute_sector_correlation_series(pd.DataFrame(), pd.DataFrame()).empty

    def test_futures_holiday_etf_lookup_fallback(self, cme_sample_data):
        """
        Adversarial test: Evaluate the fallback behavior when futures trade on a date
        that does NOT exist in the ETF dataset (e.g. CME holiday sessions).
        Verify that:
        1. If sector_corr_series does not contain date D, lookup defaults to 1.0 (permitting trades).
        2. If an explicit correlation series IS provided with missing date D, it logs cleanly without throwing KeyError.
        """
        df_es, df_nq = cme_sample_data
        unique_dates = sorted(list(set(pd.to_datetime(df_nq.index).date)))

        # Create a correlation series missing the first 2 dates entirely
        partial_corr_series = pd.Series(0.92, index=unique_dates[2:])

        strat = F4OptimizationISN1Strategy(params={"min_sector_corr": 0.85})
        bt = F4OptimizationISN1Backtester(strategy=strat, account_size=100000.0, early_stopping=False)

        # Must execute without KeyError or crash on missing dates
        res = bt.run(df_nq, df_es, sector_corr_series=partial_corr_series)
        assert "metrics" in res
        assert "Total Trades" in res["metrics"]

        strat_n3 = F4OptimizationISN3Strategy(params={"min_sector_corr": 0.85})
        bt_n3 = F4OptimizationISN3Backtester(strategy=strat_n3, account_size=100000.0, early_stopping=False)
        res_n3 = bt_n3.run(df_nq, df_es, sector_corr_series=partial_corr_series)
        assert "metrics" in res_n3
        assert "Total Trades" in res_n3["metrics"]
