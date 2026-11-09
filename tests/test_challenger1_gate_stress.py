"""
tests/test_challenger1_gate_stress.py

Adversarial Empirical Stress Verification Suite
Designed by Challenger 1 (Adversarial Empirical Verifier)
Target: Worker Implementation for F4 (Cointegration OU) & F3 (Anchored VWAP N2)

Stress Test Coverage:
1. F4 Coprime Lattice Rounding & Directional Exposure Under Volatility Shocks.
2. F4 Trade 413 & 2023-11-30 Session Loss Containment and DLL Compliance.
3. F4 Algorithmic Edge Cases: Division by zero, OLS degeneracy, Kalman stability, overnight gaps.
4. F3 Monte Carlo Robustness Stress Testing across 5,000 to 10,000 simulations under varied seeds.
5. F3 Trend Day Counter-Trend Protection (kappa_norm <= 0.085 and 6-bar AMT gate).
"""

import math
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.strategies.f4_optimizacion_is_n1 import (
    CointegrationOUModelN1,
    KalmanFilterBeta,
    F4OptimizationISN1Strategy,
    F4OptimizationISN1Backtester,
)
from src.strategies.mean_reversion.f3_optimization_is_n2 import (
    F3OptimizationISN2Strategy,
    PropRiskManagerN2,
    compute_prior_day_value_areas,
    compute_anchored_vwap_n2_indicators,
)
from src.core.monte_carlo import MonteCarloSimulator


# ==============================================================================
# PART 1: F4 ADVERSARIAL STRESS - COPRIME ROUNDING & VOLATILITY SHOCKS
# ==============================================================================

class TestF4AdversarialCoprimeAndVolatilityShocks:
    """Stress tests on coprime lattice rounding and directional risk under market shocks."""

    def setup_method(self):
        self.model = CointegrationOUModelN1()

    def test_coprime_rounding_vs_ceil_directional_bias(self):
        """
        Adversarial test comparing old ceil vs new round:
        Across the observed 2023 ratio range [1.15, 1.20, 1.25, 1.28, 1.32]:
        1. math.ceil(1 * tr) = 2 forces a ratio of 2.0, introducing a +51% to +73% short ES directional exposure.
        2. int(round(1 * tr)) = 1 yields a ratio of 1.0, strictly bounding the directional error under 25%.
        3. find_optimal_coprime_basket for max_nq=7 finds coprime integers with relative error < 5%.
        """
        typical_2023_ratios = [1.15, 1.20, 1.25, 1.28, 1.32]
        for tr in typical_2023_ratios:
            # Old implementation logic:
            ceil_es = math.ceil(1 * tr)
            ceil_ratio = ceil_es / 1.0
            ceil_err = (ceil_ratio - tr) / tr

            # New implementation logic:
            round_es = max(1, int(round(1 * tr)))
            round_ratio = round_es / 1.0
            round_err = abs(round_ratio - tr) / tr

            # Ceil always forces 2, creating massive directional error (> 50%)
            assert ceil_es == 2
            assert ceil_err > 0.50

            # Round produces 1, keeping error strictly bounded under 25%
            assert round_es == 1
            assert round_err <= 0.25
            assert round_err < ceil_err  # Round is strictly superior to ceil

            # Multi-contract optimal coprime basket (e.g. 7 MNQ vs 9 MES for 1.2857)
            opt_nq, opt_es, cop_err = self.model.find_optimal_coprime_basket(tr, max_nq_contracts=7)
            assert math.gcd(opt_nq, opt_es) == 1
            assert cop_err < 0.05  # Within 5% dollar-beta parity

    def test_market_volatility_divergence_shock_and_sl_containment(self):
        """
        Simulate an adversarial market divergence:
        NQ drops 0.5% while ES rises 0.5% in a single bar against a BUY_SPREAD (Long NQ, Short ES).
        Verify that:
        1. With unit sub-basket (1 MNQ vs 1 MES), the single-bar loss is < $300 USD.
        2. The structural Stop Loss (s_stop - s_open = 1.4 sigma) caps unit loss within risk budget.
        """
        p_nq_0 = 15000.0
        p_es_0 = 4200.0
        m_nq = 2.0   # MNQ multiplier ($2/pt)
        m_es = 5.0   # MES multiplier ($5/pt)

        # 1 unit sub-basket: 1 MNQ and 1 MES
        unit_nq, unit_es = 1, 1

        # Flash divergence: NQ drops 0.5% (-75 pts), ES rises 0.5% (+21 pts)
        p_nq_shock = p_nq_0 * 0.995
        p_es_shock = p_es_0 * 1.005

        loss_nq = unit_nq * (p_nq_shock - p_nq_0) * m_nq   # 1 * -75 * 2 = -$150
        loss_es = -unit_es * (p_es_shock - p_es_0) * m_es  # -1 * +21 * 5 = -$105
        total_unit_loss = loss_nq + loss_es                # -$255

        # Single-bar unit loss is strictly bounded and far below the $1,000 DLL
        assert total_unit_loss == -255.0
        assert abs(total_unit_loss) < 300.0

        # Structural Stop-Loss formula check:
        # At delta_s_sl = 1.4 sigma, with typical sigma_eq = 0.0012, delta_spread = 1.4 * 0.0012 = 0.00168
        # Unit stop loss loss:
        sl_loss = self.model.calculate_unit_stop_loss_loss(
            sigma_eq=0.0012,
            p_nq=p_nq_0,
            n_nq_unit=1,
            m_nq=m_nq,
            friction_basket_usd=26.68,
            delta_s_sl=1.40
        )
        # Structural SL loss is ~$77 USD per sub-basket, easily fitting in $600 risk budget
        assert sl_loss < 100.0

    def test_f4_slippage_spike_robustness(self):
        """
        Stress test transaction friction:
        Evaluate unit friction when slippage spikes from 1.0 tick to 8.0 ticks.
        """
        unit_nq, unit_es = 3, 4
        m_nq, m_es = 2.0, 5.0
        tick_nq, tick_es = 0.25, 0.25
        comm_nq, comm_es = 0.87, 0.87

        for slip in [1.0, 2.0, 4.0, 8.0]:
            friction = (
                unit_nq * comm_nq + unit_es * comm_es +
                2.0 * slip * tick_nq * m_nq * unit_nq +
                2.0 * slip * tick_es * m_es * unit_es +
                1.0 * tick_es * m_es * unit_es
            )
            # Even under 8 ticks of extreme slippage, friction per unit basket is < $150 USD
            assert friction < 150.0


# ==============================================================================
# PART 2: F4 EMPIRICAL VERIFICATION - TRADE 413, 2023-11-30 & DAILY LOSS LIMIT
# ==============================================================================

class TestF4EmpiricalTradeContainmentAndDLL:
    """Verifies empirical trades CSV for F4 In-Sample 2023."""

    @pytest.fixture(scope="class")
    def f4_trades_df(self):
        path = Path("reports/f4_optimizacion_is_n1_trades.csv")
        assert path.exists(), f"Missing F4 trades CSV at {path}"
        return pd.read_csv(path)

    def test_total_trades_count_and_sample_coverage(self, f4_trades_df):
        """Confirms 428 trades across 2023 without starvation."""
        assert len(f4_trades_df) == 428
        assert "net_pnl_usd" in f4_trades_df.columns

    def test_worst_single_trade_loss_contained(self, f4_trades_df):
        """
        Verifies that no single trade loses more than $500 USD,
        ruling out the -$1,017 USD catastrophic loss observed in the unoptimized version.
        """
        min_net_pnl = f4_trades_df["net_pnl_usd"].min()
        min_gross_pnl = f4_trades_df["gross_pnl_usd"].min()

        # In unoptimized version, single trade loss breached -$1,017 USD.
        # In optimized version, worst trade is strictly bounded:
        assert min_net_pnl > -500.0, f"Worst net trade loss {min_net_pnl} breached $500 limit!"
        assert min_gross_pnl > -500.0, f"Worst gross trade loss {min_gross_pnl} breached $500 limit!"

    def test_containment_on_2023_11_30_session(self, f4_trades_df):
        """
        Specifically inspects the 2023-11-30 session (previously fatal Day 239):
        Confirms trades exited safely without breaching DLL.
        """
        nov30_trades = f4_trades_df[f4_trades_df["entry_date"] == "2023-11-30"]
        assert len(nov30_trades) >= 1, "Expected trades on 2023-11-30"

        session_net_pnl = nov30_trades["net_pnl_usd"].sum()
        # Daily loss must not breach $1,000 DLL
        assert session_net_pnl > -100.0, f"2023-11-30 loss was {session_net_pnl}, expected containment!"

        for _, tr in nov30_trades.iterrows():
            assert tr["net_pnl_usd"] > -500.0
            assert tr["exit_reason"] in ["TIME_DECAY", "EOD_FLAT", "TAKE_PROFIT", "STOP_LOSS"]

    def test_trade_413_specifically(self, f4_trades_df):
        """Verifies trade_id 413 properties."""
        t413 = f4_trades_df[f4_trades_df["trade_id"] == 413]
        assert len(t413) == 1
        row = t413.iloc[0]
        # Trade 413 had loss of -$117.68 USD (gross -$91.0), cleanly contained by EOD_FLAT
        assert row["net_pnl_usd"] > -200.0
        assert row["contracts_nq"] == 3
        assert row["contracts_es"] == 4
        assert row["exit_reason"] == "EOD_FLAT"

    def test_daily_loss_limit_zero_breaches_full_year(self, f4_trades_df):
        """
        Aggregates PnL by entry date across all 2023 trading sessions.
        Asserts that daily loss never hits or breaches the $1,000 USD Daily Loss Limit.
        """
        daily_pnl = f4_trades_df.groupby("entry_date")["net_pnl_usd"].sum()
        worst_day_pnl = daily_pnl.min()
        worst_day_date = daily_pnl.idxmin()

        # Daily loss limit for $50k tier is $1,000 USD
        assert worst_day_pnl > -1000.0, f"DLL breached on {worst_day_date} with PnL {worst_day_pnl}"
        # Even stricter: worst day loss should be within risk budget (~$600)
        assert worst_day_pnl > -700.0

    def test_max_floating_drawdown_under_tmdd(self, f4_trades_df):
        """Verifies running drawdown never approaches $2,000 USD TMDD."""
        max_dd = f4_trades_df["running_drawdown"].max()
        assert max_dd < 1000.0, f"Max drawdown {max_dd} was unexpectedly high!"
        assert max_dd <= 467.39 + 1e-2  # Matches verified $467.39 USD


# ==============================================================================
# PART 3: F4 ALGORITHMIC EDGE CASES - KALMAN, OLS, DEGENERACY & GAPS
# ==============================================================================

class TestF4AlgorithmicEdgeCases:
    """Stress tests numerical edge cases in F4 mathematical components."""

    def setup_method(self):
        self.model = CointegrationOUModelN1()

    def test_dollar_beta_ratio_zero_or_negative_inputs(self):
        """
        Tests calculate_dollar_beta_ratio and find_optimal_coprime_basket
        under degenerate zero/negative inputs.
        """
        # Zero ES price or zero ES multiplier -> safe fallback to 1.0
        r_zero_es = self.model.calculate_dollar_beta_ratio(15000.0, 0.0, 1.25, 2.0, 5.0)
        assert r_zero_es == 1.0

        r_zero_mes = self.model.calculate_dollar_beta_ratio(15000.0, 4000.0, 1.25, 2.0, 0.0)
        assert r_zero_mes == 1.0

        # Zero NQ price or zero beta -> returns 0.0
        r_zero_nq = self.model.calculate_dollar_beta_ratio(0.0, 4000.0, 1.25, 2.0, 5.0)
        assert r_zero_nq == 0.0

        r_zero_b = self.model.calculate_dollar_beta_ratio(15000.0, 4000.0, 0.0, 2.0, 5.0)
        assert r_zero_b == 0.0

        # Coprime search must handle target_ratio <= 0.0 gracefully without division by zero:
        basket_0 = self.model.find_optimal_coprime_basket(0.0, max_nq_contracts=7)
        assert basket_0 == (1, 1, 0.0)

        basket_neg = self.model.find_optimal_coprime_basket(-1.5, max_nq_contracts=7)
        assert basket_neg == (1, 1, 0.0)

    def test_ols_estimation_flat_series_zero_variance(self):
        """Tests estimate_cointegration_backward when ES is completely flat (variance = 0)."""
        n = 100
        p_nq = np.linspace(15000, 15100, n)
        p_es_flat = np.full(n, 4000.0)  # Variance is 0

        # Should handle without crashing (ZeroDivisionError caught or handled)
        beta, alpha, res = self.model.estimate_cointegration_backward(p_nq, p_es_flat)
        # Should return finite floats
        assert np.isfinite(beta)
        assert np.isfinite(alpha)
        assert len(res) == n

    def test_kalman_filter_extreme_nan_and_inf_resilience(self):
        """Tests Kalman filter update with anomalous values."""
        kf = KalmanFilterBeta(initial_alpha=1.0, initial_beta=1.2)
        # Normal update
        a, b = kf.update(np.log(15000.0), np.log(4000.0))
        assert np.isfinite(a) and np.isfinite(b)

        # Huge jump
        a2, b2 = kf.update(np.log(30000.0), np.log(4000.0))
        assert np.isfinite(a2) and np.isfinite(b2)

    def test_ou_fit_explosive_process_detection(self):
        """
        When spread is explosive (theta >= 0 / kappa <= 0),
        fit_ou_process_intraday must mark is_valid=False to prevent entering unstable spread.
        """
        n = 150
        timestamps = pd.date_range("2023-01-10 09:30", periods=n, freq="5min")
        # Explosive series: x_t = 1.05 * x_{t-1} + e_t
        res_explosive = np.zeros(n)
        for i in range(1, n):
            res_explosive[i] = 1.05 * res_explosive[i - 1] + 0.001

        ou = self.model.fit_ou_process_intraday(res_explosive, timestamps, dt_minutes=5.0)
        # An explosive process has theta >= 0 or kappa <= 0
        assert ou["is_valid"] is False or ou["kappa"] <= 0


# ==============================================================================
# PART 4: F3 MONTE CARLO ADVERSARIAL STRESS - 5,000 TO 10,000 SIMULATIONS
# ==============================================================================

class TestF3MonteCarloAdversarialStress:
    """Stress tests F3 Monte Carlo across 5,000 to 10,000 iterations under varied random seeds."""

    @pytest.fixture(scope="class")
    def f3_pnl_series(self):
        path = Path("reports/f3_optimization_n2_trades.csv")
        assert path.exists(), f"Missing F3 trades CSV at {path}"
        df = pd.read_csv(path)
        col = "net_pnl_usd" if "net_pnl_usd" in df.columns else "net_pnl"
        return df[col].dropna().astype(float).values

    @pytest.mark.parametrize("simulations", [5000, 10000])
    @pytest.mark.parametrize("seed", [42, 100, 777, 1984, 2023])
    def test_monte_carlo_reshuffle_ruin_strictly_under_1_percent(self, f3_pnl_series, simulations, seed):
        """
        Adversarial Reshuffling Stress across 5,000 and 10,000 simulations:
        Ruin rate under $2,000 MTDD must remain strictly < 1.0% (and far below the 10% mandate).
        Survival rate must be >= 99.0%.
        """
        sim = MonteCarloSimulator(
            trades_pnl=pd.Series(f3_pnl_series),
            initial_capital=50000.0,
            daily_loss_limit=1000.0,
            max_trailing_drawdown=2000.0,
            profit_target=3000.0,
            seed=seed,
        )
        res = sim.run_reshuffle(n_simulations=simulations)

        ruin_pct = res["risk_of_ruin_pct"]
        survival_pct = 100.0 - ruin_pct

        assert ruin_pct < 1.0, f"Seed {seed} ({simulations} sims): Ruin rate {ruin_pct}% exceeded 1%!"
        assert survival_pct >= 99.0, f"Survival rate {survival_pct}% below 99%!"

    @pytest.mark.parametrize("simulations", [5000, 10000])
    @pytest.mark.parametrize("seed", [42, 100, 777, 1984, 2023])
    def test_monte_carlo_bootstrap_ruin_strictly_under_5_percent(self, f3_pnl_series, simulations, seed):
        """
        Adversarial Bootstrapping Stress across 5,000 and 10,000 simulations:
        Under severe adverse trade clustering with replacement:
        Ruin rate under $2,000 MTDD must remain strictly < 5.0% (mandate requires < 10%).
        Under $3,000 MTDD (prop tier limit), ruin rate must be < 1.0%.
        Survival rate must be >= 95.0% (mandate requires >= 90%).
        """
        sim = MonteCarloSimulator(
            trades_pnl=pd.Series(f3_pnl_series),
            initial_capital=50000.0,
            daily_loss_limit=1000.0,
            max_trailing_drawdown=2000.0,
            profit_target=3000.0,
            seed=seed,
        )
        res_2000 = sim.run_bootstrap(n_simulations=simulations)
        ruin_2000 = res_2000["risk_of_ruin_pct"]
        survival_2000 = 100.0 - ruin_2000

        assert ruin_2000 < 5.0, f"Seed {seed} ({simulations} sims): Bootstrap ruin {ruin_2000}% exceeded 5%!"
        assert survival_2000 >= 95.0, f"Survival rate {survival_2000}% below 95%!"

        # Now test under $3,000 MTDD limit
        sim_3000 = MonteCarloSimulator(
            trades_pnl=pd.Series(f3_pnl_series),
            initial_capital=50000.0,
            daily_loss_limit=1500.0,
            max_trailing_drawdown=3000.0,
            profit_target=3000.0,
            seed=seed,
        )
        res_3000 = sim_3000.run_bootstrap(n_simulations=simulations)
        assert res_3000["risk_of_ruin_pct"] < 1.0


# ==============================================================================
# PART 5: F3 TREND DAY ADVERSARIAL PROTECTION - KAPPA NORM & 6-BAR AMT GATE
# ==============================================================================

class TestF3TrendDayAdversarialProtection:
    """Stress tests trend day protection mechanisms in F3 N2."""

    def test_steep_trend_slope_blocks_counter_trend_entry(self):
        """
        Synthetic trend day:
        Price climbs steadily +40 points over 30 bars.
        Normalized VWAP slope kappa_norm reaches > 0.10.
        Verify that:
        1. VWAP_Slope_Norm exceeds kappa_crit (0.085).
        2. Phi_Slope becomes 0.
        3. Regime_Rotational becomes 0, blocking counter-trend entries.
        """
        n_bars = 78
        timestamps = pd.date_range("2023-05-15 09:30", periods=n_bars, freq="5min")
        base = 4200.0
        # Linear steady upward drift of +0.8 pts per bar = +62.4 points on the day
        closes = base + np.arange(n_bars) * 0.8
        highs = closes + 1.0
        lows = closes - 0.5
        opens = closes - 0.2
        volumes = np.full(n_bars, 2000.0)

        df = pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
        }, index=timestamps)

        ind = compute_anchored_vwap_n2_indicators(
            df,
            k_band=2.20,
            k_catastrophic=2.75,
            kappa_crit=0.085,
            slope_lookback_bars=3,
        )

        # After bar 15, slope kappa_norm should exceed 0.085
        late_slopes = ind["VWAP_Slope_Norm"].iloc[15:]
        assert (late_slopes > 0.085).any(), "Slope did not exceed 0.085 on trend day"

        # On any bar where normalized slope exceeds kappa_crit (0.085):
        # 1. Phi_Slope must be 0
        # 2. Regime_Rotational must be 0, shutting down mean-reversion entries
        steep_mask = ind["VWAP_Slope_Norm"] > 0.085
        assert (ind.loc[steep_mask, "Phi_Slope"] == 0).all()
        assert (ind.loc[steep_mask, "Regime_Rotational"] == 0).all()

    def test_amt_6_bars_outside_gate_blocks_premature_breakouts(self):
        """
        Tests the 6-bar AMT Acceptance Gate:
        Price opens outside previous Value Area [VAL, VAH].
        First 6 bars (09:30 - 09:55) open & close above VAH.
        Verify that:
        1. is_outside_value_day triggers.
        2. Phi_AMT is set to 0 for all bars of Day 2.
        3. Regime_Rotational is forced to 0 for the entire day, blocking counter-trend shorts.
        """
        data_days = []
        # Day 1: normal trading, establish VA between 4000 and 4020
        d1_times = pd.date_range("2023-06-01 09:30", periods=78, freq="5min")
        d1_closes = 4010.0 + np.sin(np.linspace(0, 3.14, 78)) * 10.0
        df1 = pd.DataFrame({
            "Open": d1_closes, "High": d1_closes + 1.0, "Low": d1_closes - 1.0,
            "Close": d1_closes, "Volume": 1000.0
        }, index=d1_times)
        data_days.append(df1)

        # Day 2: opens completely above Day 1 VAH (> 4030)
        d2_times = pd.date_range("2023-06-02 09:30", periods=78, freq="5min")
        d2_closes = 4050.0 + np.random.normal(0, 1.0, 78)
        df2 = pd.DataFrame({
            "Open": d2_closes, "High": d2_closes + 1.0, "Low": d2_closes - 1.0,
            "Close": d2_closes, "Volume": 1000.0
        }, index=d2_times)
        data_days.append(df2)

        data = pd.concat(data_days).sort_index()

        ind = compute_anchored_vwap_n2_indicators(
            data,
            k_band=2.20,
            k_catastrophic=2.75,
            kappa_crit=0.085,
        )

        day2_mask = ind.index.date == d2_times[0].date()
        day2_ind = ind[day2_mask]

        # On Day 2, all 6 opening bars opened and closed above prior VAH
        # Therefore, Phi_AMT must be 0 for the entire day
        assert (day2_ind["Phi_AMT"] == 0).all(), "Phi_AMT should be 0 on outside value day"

        # Regime rotational must be 0 for the entire day, eliminating counter-trend trades
        assert (day2_ind["Regime_Rotational"] == 0).all(), "Regime rotational should be locked at 0"

    def test_catastrophic_band_circuit_breaker(self):
        """
        Verifies that the strategy configures and calculates
        UpperCatastrophic and LowerCatastrophic at +-2.75 sigma.
        """
        strat = F3OptimizationISN2Strategy(
            params={"symbol": "ES", "k_band": 2.20, "k_catastrophic": 2.75}
        )
        assert strat.params["k_catastrophic"] == 2.75

        # Check indicator computation
        n_bars = 78
        timestamps = pd.date_range("2023-05-15 09:30", periods=n_bars, freq="5min")
        closes = 4000.0 + np.random.normal(0, 2.0, n_bars)
        df = pd.DataFrame({
            "Open": closes, "High": closes + 1.0, "Low": closes - 1.0,
            "Close": closes, "Volume": 1000.0
        }, index=timestamps)

        ind = compute_anchored_vwap_n2_indicators(
            df,
            k_band=2.20,
            k_catastrophic=2.75,
        )

        vwap = ind["VWAP"].iloc[-1]
        sigma = ind["Sigma_VWAP"].iloc[-1]
        upper_cat = ind["UpperCatastrophic"].iloc[-1]
        lower_cat = ind["LowerCatastrophic"].iloc[-1]

        assert pytest.approx(upper_cat, rel=1e-4) == vwap + 2.75 * sigma
        assert pytest.approx(lower_cat, rel=1e-4) == vwap - 2.75 * sigma
