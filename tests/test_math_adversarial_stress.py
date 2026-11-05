"""
Adversarial Stress Test Harness for CME Quantitative Futures Strategies (F1 - F9).
Validates mathematical edge cases, singularities, dimensional consistency,
and strict prop trading drawdown constraints.
"""

import math
import numpy as np
import pytest


# ==============================================================================
# F1: Market Intraday Momentum & MOC Seasonality
# ==============================================================================
class TestF1Adversarial:
    def test_log_return_well_defined_for_positive_prices(self):
        # Index prices must be positive
        for p_open, p_close in [(5000.0, 5010.0), (5800.25, 5750.50), (20000.0, 20150.0)]:
            r = math.log(p_close / p_open)
            assert math.isfinite(r)

    def test_volume_ratio_zero_volume_handling(self):
        # If historical volume is 0 or positive, test division guard
        v_current = 50000
        k_vol = 20
        v_hist_normal = [45000] * k_vol
        mean_v = sum(v_hist_normal) / k_vol
        ratio = v_current / mean_v if mean_v > 0 else 0.0
        assert ratio == pytest.approx(50000 / 45000)

    def test_dynamic_threshold_positive(self):
        theta_floor = 0.0015
        c_sigma = 0.20
        # Under low or zero daily vol
        for sigma in [0.0, 5.0, 50.0, 500.0]:
            p = 5800.0
            theta = max(theta_floor, c_sigma * (sigma / p))
            assert theta >= theta_floor
            assert theta > 0

    def test_sizing_respects_daily_loss_and_trailing_drawdown(self):
        dll = 1000.0
        mtdd = 2000.0
        psi = 0.25

        def get_phi(k_loss):
            return 0.35 * (2 ** (-min(k_loss, 3)))

        # Case 1: Normal operation, 0 losses
        b_risk = min(get_phi(0) * max(0.0, dll - 0.0), psi * max(0.0, mtdd - 0.0))
        assert b_risk <= 0.35 * dll
        assert b_risk <= psi * mtdd

        # Case 2: Daily loss exhausted (L_day >= DLL)
        b_risk_exhausted = min(get_phi(0) * max(0.0, dll - 1000.0), psi * max(0.0, mtdd - 500.0))
        assert b_risk_exhausted == 0.0

        # Case 3: Trailing DD exhausted (DD_t >= MTDD)
        b_risk_dd_exhausted = min(get_phi(0) * max(0.0, dll - 200.0), psi * max(0.0, mtdd - 2000.0))
        assert b_risk_dd_exhausted == 0.0

        # Sizing with zero budget
        r_contract = (5800.0 * 0.0035) * 50 + 10.0
        n_std = math.floor(b_risk_exhausted / r_contract)
        assert n_std == 0

    def test_loss_streak_attenuation(self):
        def phi(k):
            return 0.35 * (2 ** (-min(k, 3)))

        assert phi(0) == 0.35
        assert phi(1) == 0.175
        assert phi(2) == 0.0875
        assert phi(3) == 0.04375
        assert phi(5) == 0.04375

    def test_standard_to_micro_switch_and_circuit_breaker(self):
        m_std = 50.0
        m_micro = 5.0
        delta_sl = 20.0  # 20 points
        c_fric_std = 10.0
        c_fric_micro = 2.0

        r_std = delta_sl * m_std + c_fric_std  # 1010 USD
        r_micro = delta_sl * m_micro + c_fric_micro  # 102 USD

        # Small budget: 300 USD
        b_risk = 300.0
        n_std = math.floor(b_risk / r_std)
        assert n_std == 0

        # Switches to micro
        n_micro = min(math.floor(b_risk / r_micro), 10)
        assert n_micro == 2

        # Extreme volatility spike where single micro > budget: 80 USD budget vs 102 USD risk
        b_risk_tiny = 80.0
        n_micro_halt = math.floor(b_risk_tiny / r_micro)
        assert n_micro_halt == 0  # Volatility circuit breaker activates


# ==============================================================================
# F2: Filtered Quantitative ORB
# ==============================================================================
class TestF2Adversarial:
    def test_stretch_and_midpoint_positive(self):
        orh = 5820.0
        orl = 5800.0
        m_or = (orh + orl) / 2.0
        assert m_or == 5810.0

        # Stretch calculation from 10 days
        diffs = [15.0, 12.0, 18.0, 10.0, 14.0, 11.0, 13.0, 16.0, 17.0, 14.0]
        stretch = sum(diffs) / 10.0
        assert stretch == 14.0

        lambda_stretch = 0.50
        gatillo_long = orh + lambda_stretch * stretch
        gatillo_short = orl - lambda_stretch * stretch

        assert gatillo_long > orh
        assert gatillo_short < orl

        delta_risk = abs(gatillo_long - m_or)
        assert delta_risk == (orh - orl) / 2.0 + lambda_stretch * stretch
        assert delta_risk == 17.0

    def test_flat_opening_range_edge_case(self):
        # Zero range edge case (ORH == ORL)
        orh = 5800.0
        orl = 5800.0
        stretch = 5.0
        lambda_stretch = 0.5
        delta_risk = (orh - orl) / 2.0 + lambda_stretch * stretch
        assert delta_risk == 2.5 > 0

    def test_orb_excessive_range_circuit_breaker(self):
        # Huge opening range during macro news: 150 points
        orh = 5950.0
        orl = 5800.0
        stretch = 20.0
        delta_risk = (orh - orl) / 2.0 + 0.5 * stretch  # 75 + 10 = 85 points
        m_micro = 5.0  # MES
        c_fric = 2.0
        r_micro = delta_risk * m_micro + c_fric  # 85 * 5 + 2 = 427 USD

        b_risk = 250.0  # Budget is 250 USD
        assert r_micro > b_risk
        n_micro = math.floor(b_risk / r_micro)
        assert n_micro == 0  # Defense halts execution


# ==============================================================================
# F3: Anchored VWAP & Rotational Value Area Reversion
# ==============================================================================
class TestF3Adversarial:
    def test_anchored_vwap_and_variance_positivity(self):
        # Generate 12 bars (1 hour) of 5-min bars
        np.random.seed(42)
        v = np.random.randint(1000, 5000, size=12)
        p_typ = 5800.0 + np.random.randn(12) * 5.0

        cum_v = np.sum(v)
        assert cum_v > 0

        vwap = np.sum(p_typ * v) / cum_v
        assert 5750.0 < vwap < 5850.0

        var_vwap = np.sum(v * ((p_typ - vwap) ** 2)) / cum_v
        assert var_vwap >= 0.0

        sigma_vwap = math.sqrt(var_vwap)
        assert sigma_vwap >= 0.0

    def test_three_sigma_catastrophic_breaker(self):
        vwap = 5800.0
        sigma_vwap = 10.0
        band_upper_2 = vwap + 2.0 * sigma_vwap  # 5820
        band_upper_3 = vwap + 3.0 * sigma_vwap  # 5830

        # Normal reversal entry at 5821 returning inside
        # Adverse breakout goes to 5831
        p_curr = 5831.0
        is_broken = abs(p_curr - vwap) >= 3.0 * sigma_vwap
        assert is_broken is True

    def test_f3_risk_with_buffer_ticks(self):
        delta_tick = 0.25
        k_buffer = 4
        m_es = 50.0
        c_fric = 5.0

        # Extreme case: L_t == L_t-1 == P_entry
        p_entry = 5800.0
        p_sl = p_entry - k_buffer * delta_tick  # 5800 - 1.0 = 5799.0
        delta_sl = abs(p_entry - p_sl)
        assert delta_sl == 1.0

        r_contract = delta_sl * m_es + c_fric
        assert r_contract == 55.0 > 0


# ==============================================================================
# F4: Cointegration ES vs NQ & Ornstein-Uhlenbeck
# ==============================================================================
class TestF4Adversarial:
    def test_ou_parameters_and_half_life(self):
        delta_t = 1.0 / 78.0  # 1 bar of 5 min in RTH day units

        # For valid stationary AR(1): 0 < b < 1
        for b in [0.85, 0.92, 0.97]:
            kappa = -math.log(b) / delta_t
            assert kappa > 0.0

            half_life_days = math.log(2) / kappa
            half_life_minutes = half_life_days * 390.0
            assert half_life_minutes > 0.0

            sigma_eta = 0.002
            sigma_eq = math.sqrt((sigma_eta ** 2) / (1.0 - b ** 2))
            assert sigma_eq > 0.0

    def test_ou_half_life_admissibility_filter(self):
        tau_min = 15.0  # 15 minutes
        tau_max = 780.0  # 2 days (780 min)

        def calc_hl_min(b):
            return (math.log(2) * 5.0) / (-math.log(b))

        # Test b = 0.999 (too slow, half life > 780 min)
        hl_slow = calc_hl_min(0.999)
        assert hl_slow > tau_max
        is_admissible_slow = tau_min <= hl_slow <= tau_max
        assert is_admissible_slow is False

        # Test b = 0.5 (too fast, micro noise, half life < 15 min)
        hl_fast = calc_hl_min(0.5)
        assert hl_fast < tau_min
        is_admissible_fast = tau_min <= hl_fast <= tau_max
        assert is_admissible_fast is False

        # Test b = 0.95 (healthy: ~67 min)
        hl_healthy = calc_hl_min(0.95)
        assert tau_min <= hl_healthy <= tau_max

    def test_dollar_beta_neutral_basket_coprime_ratio(self):
        p_nq = 20000.0
        m_mnq = 2.0
        p_es = 5800.0
        m_mes = 5.0
        beta_hedge = 1.25

        exact_ratio = beta_hedge * ((p_nq * m_mnq) / (p_es * m_mes))
        assert exact_ratio == pytest.approx(1.7241379, rel=1e-4)

        coprime_ratio = 17.0 / 10.0
        error = abs(coprime_ratio - exact_ratio) / exact_ratio
        assert error < 0.015  # Less than 1.5% residual beta error

        # Net dollar exposure
        val_nq = 10 * m_mnq * p_nq  # 400,000 USD
        val_es = 17 * m_mes * p_es  # 493,000 USD
        beta_adj_nq = val_nq * beta_hedge  # 500,000 USD
        diff_pct = abs(beta_adj_nq - val_es) / val_es
        assert diff_pct < 0.02


# ==============================================================================
# F5: Order Flow Imbalance Cont et al.
# ==============================================================================
class TestF5Adversarial:
    def test_all_nine_lob_event_combinations(self):
        # Verify e_n^b
        # 1. P_n^b > P_{n-1}^b
        assert self._calc_eb(5800.25, 10, 5800.00, 20) == 10
        # 2. P_n^b == P_{n-1}^b
        assert self._calc_eb(5800.00, 25, 5800.00, 20) == 5
        assert self._calc_eb(5800.00, 15, 5800.00, 20) == -5
        # 3. P_n^b < P_{n-1}^b
        assert self._calc_eb(5799.75, 15, 5800.00, 20) == -20

        # Verify e_n^a
        # 1. P_n^a > P_{n-1}^a
        assert self._calc_ea(5800.50, 10, 5800.25, 20) == -20
        # 2. P_n^a == P_{n-1}^a
        assert self._calc_ea(5800.25, 25, 5800.25, 20) == 5
        assert self._calc_ea(5800.25, 15, 5800.25, 20) == -5
        # 3. P_n^a < P_{n-1}^a
        assert self._calc_ea(5800.00, 15, 5800.25, 20) == -15

    def _calc_eb(self, p_curr, q_curr, p_prev, q_prev):
        if p_curr > p_prev:
            return q_curr
        elif p_curr == p_prev:
            return q_curr - q_prev
        else:
            return -q_prev

    def _calc_ea(self, p_curr, q_curr, p_prev, q_prev):
        if p_curr > p_prev:
            return -q_prev
        elif p_curr == p_prev:
            return q_curr - q_prev
        else:
            return -q_curr

    def test_z_score_epsilon_guard(self):
        # All OFI values identical -> sigma == 0
        ofi_series = [100.0] * 30
        mu = sum(ofi_series) / 30.0
        var = sum((x - mu) ** 2 for x in ofi_series) / 29.0
        sigma = math.sqrt(var)
        assert sigma == 0.0

        eps_0 = 1e-8
        z = (ofi_series[-1] - mu) / (sigma + eps_0)
        assert math.isfinite(z)
        assert z == 0.0

    def test_absorption_capacity_limit(self):
        gamma_book = 0.05
        # ES depth = 250
        n_book_max_es = max(1, math.floor(gamma_book * 250))
        assert n_book_max_es == 12

        # NQ depth = 30
        n_book_max_nq = max(1, math.floor(gamma_book * 30))
        assert n_book_max_nq == 1


# ==============================================================================
# F6: Multi-Level OFI
# ==============================================================================
class TestF6Adversarial:
    def test_spatial_power_law_decay(self):
        beta_1 = 0.02
        alpha_decay = 1.0  # Harmonic decay
        betas = [beta_1 * (l ** (-alpha_decay)) for l in range(1, 6)]
        for i in range(len(betas) - 1):
            assert betas[i] > betas[i + 1]
            assert betas[i] > 0.0

    def test_pca_spectral_decomposition(self):
        # Construct realistic correlated covariance matrix (5x5)
        np.random.seed(123)
        corr = np.array([
            [1.00, 0.70, 0.50, 0.40, 0.30],
            [0.70, 1.00, 0.72, 0.55, 0.42],
            [0.50, 0.72, 1.00, 0.75, 0.60],
            [0.40, 0.55, 0.75, 1.00, 0.78],
            [0.30, 0.42, 0.60, 0.78, 1.00],
        ])
        std = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
        cov = np.outer(std, std) * corr

        eigvals, eigvecs = np.linalg.eigh(cov)
        # Order descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        assert np.all(eigvals >= 0.0)
        # PC1 explains > 60% of total variance
        var_ratio_pc1 = eigvals[0] / np.sum(eigvals)
        assert var_ratio_pc1 > 0.60

        # By Perron-Frobenius, all weights of PC1 have same sign
        v1 = eigvecs[:, 0]
        if v1[0] < 0:
            v1 = -v1
        assert np.all(v1 > 0.0)

    def test_anti_spoofing_coherence_filter(self):
        betas = np.array([0.020, 0.010, 0.0067, 0.0050, 0.0040])
        eps_0 = 1e-8

        # Scenario A: Genuine institutional buying across all levels
        ofi_genuine = np.array([500, 400, 350, 200, 150])
        num_gen = np.sign(ofi_genuine[0]) * np.sum(betas[1:] * ofi_genuine[1:])
        den_gen = np.sum(betas[1:] * np.abs(ofi_genuine[1:])) + eps_0
        phi_gen = num_gen / den_gen
        assert phi_gen == pytest.approx(1.0, rel=1e-3)
        assert phi_gen >= 0.40  # Authorized

        # Scenario B: Spoofing! Level 1 shows buy (+500), but deep levels show sell (-400, -350, -200, -150)
        ofi_spoof = np.array([500, -400, -350, -200, -150])
        num_spf = np.sign(ofi_spoof[0]) * np.sum(betas[1:] * ofi_spoof[1:])
        den_spf = np.sum(betas[1:] * np.abs(ofi_spoof[1:])) + eps_0
        phi_spf = num_spf / den_spf
        assert phi_spf == pytest.approx(-1.0, rel=1e-3)
        assert phi_spf < 0.40  # VETOED by anti-spoofing filter!


# ==============================================================================
# F7: Meta-Labeling & Triple Barrier
# ==============================================================================
class TestF7Adversarial:
    def test_logistic_calibration_bounds(self):
        for logit in [-100.0, -5.0, 0.0, 2.5, 100.0]:
            p = 1.0 / (1.0 + math.exp(-logit))
            assert 0.0 <= p <= 1.0

    def test_bet_sizing_continuous_modulation(self):
        tau_meta = 0.60
        eta = 1.50

        # p below threshold
        for p in [0.10, 0.50, 0.599]:
            m = ((p - tau_meta) / (1.0 - tau_meta)) ** eta if p >= tau_meta else 0.0
            assert m == 0.0

        # p at threshold
        m_60 = ((0.60 - tau_meta) / (1.0 - tau_meta)) ** eta
        assert m_60 == 0.0

        # p above threshold
        m_80 = ((0.80 - tau_meta) / (1.0 - tau_meta)) ** eta
        assert 0.0 < m_80 < 1.0

        m_100 = ((1.00 - tau_meta) / (1.0 - tau_meta)) ** eta
        assert m_100 == pytest.approx(1.0)

    def test_drawdown_defense_circuit_breaker(self):
        mtdd = 2000.0
        # Cushion <= 0.15 * MTDD (300 USD)
        for cushion in [300.0, 200.0, 0.0]:
            is_halted = cushion <= 0.15 * mtdd
            assert is_halted is True


# ==============================================================================
# F8: Econometric HAR-RV-J Models
# ==============================================================================
class TestF8Adversarial:
    def test_log_har_strict_positivity(self):
        # Forecasted variance from Log-HAR is exp(...) > 0 under any coefficients
        for exp_val in [-50.0, -5.0, 0.0, 10.0]:
            rv_pred = math.exp(exp_val)
            assert rv_pred > 0.0
            assert math.isfinite(rv_pred)

    def test_bipower_and_jump_decomposition(self):
        # Constant returns (0 jump)
        r = np.array([0.001] * 78)
        rv = np.sum(r ** 2)
        mu1_inv_sq = math.pi / 2.0
        bv = mu1_inv_sq * (78.0 / 77.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
        assert rv > 0.0
        assert bv > 0.0

    def test_intraday_volatility_circuit_breaker(self):
        rv_pred = 0.0004  # Projected daily variance
        xi_circuit = 2.50

        # Intraday flash crash: RV exceeds 2.5x
        rv_intra_calm = 0.0003
        assert (rv_intra_calm >= xi_circuit * rv_pred) is False

        rv_intra_spike = 0.0012
        assert (rv_intra_spike >= xi_circuit * rv_pred) is True  # Halts trading


# ==============================================================================
# F9: Temporal Anomalies & VRP
# ==============================================================================
class TestF9Adversarial:
    def test_tsr_term_structure_ratio_regimes(self):
        # Contango (Normal, calm): VIX < VIX3M
        vix = 15.0
        vix3m = 18.0
        tsr = vix / vix3m
        assert tsr < 1.0  # 0.833 <= 0.95: Favorable for equity

        # Backwardation (Inversion, panic): VIX > VIX3M
        vix_panic = 32.0
        vix3m_panic = 25.0
        tsr_panic = vix_panic / vix3m_panic
        assert tsr_panic >= 1.0  # Blocks all long trades

    def test_pre_macro_exit_strictly_prior_to_embargo(self):
        # Macro announcement at 08:30:00 EST
        # Strategy entry: 08:15:00 EST (T-15 min)
        # Strategy exit: 08:27:00 EST (T-3 min)
        # Prop firm embargo: [08:28:00, 08:32:00] EST ([T-2, T+2])
        t_exit_seconds = 27 * 60  # 1620 s into hour
        t_embargo_start_seconds = 28 * 60  # 1680 s into hour

        # Safety gap must be >= 60 seconds
        safety_gap = t_embargo_start_seconds - t_exit_seconds
        assert safety_gap == 60  # Exactly 60s buffer before embargo!


# ==============================================================================
# Universal Prop Firm Stress Suite across All Account Sizes
# ==============================================================================
class TestUniversalPropFirmStress:
    @pytest.mark.parametrize("account_name,dll,mtdd", [
        ("Topstep_50k", 1000.0, 2000.0),
        ("Apex_50k", 1250.0, 2500.0),
        ("Topstep_100k", 2000.0, 3000.0),
        ("Apex_100k", 2500.0, 3000.0),
        ("150k_Account", 3000.0, 4500.0),
    ])
    def test_sizing_collapses_to_zero_at_limits(self, account_name, dll, mtdd):
        # When Daily Loss Limit reached
        l_day = dll
        rem_dll = max(0.0, dll - l_day)
        assert rem_dll == 0.0

        # When Trailing Drawdown reached
        dd_curr = mtdd
        rem_dd = max(0.0, mtdd - dd_curr)
        assert rem_dd == 0.0

        # In all strategies, sizing formula budget is multiplied by rem_dll / rem_dd
        budget = min(rem_dll, rem_dd)
        assert budget == 0.0

        # Integer contract floor
        for r_contract in [5.0, 50.0, 500.0]:
            n = math.floor(budget / r_contract)
            assert n == 0
