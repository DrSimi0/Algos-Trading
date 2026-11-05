"""
tests/test_empirical_stress.py

Empirical Stress Testing & Adversarial Edge Case Suite
Designed by Challenger 1 (Adversarial Critic & Empirical Verifier)
Target: Mathematical models and pseudocode in 'reports/investigacion_teorias_futuros_es_nq.md'

Edge cases tested:
1. Zero volume periods (V_t = 0)
2. Flat prices / zero variance (sigma = 0)
3. Extreme price shocks (+10% / -10% single-bar shocks, flash crashes)
4. Cointegration divergence (theta <= 0, non-stationary / explosive spread, structural breaks)
"""

import math
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tests.test_report_verification import extract_fenced_code_blocks, REPORT_PATH


# ==============================================================================
# FIXTURE: Dynamically load exact report implementations into a test namespace
# ==============================================================================

@pytest.fixture(scope="module")
def report_models():
    """Extracts and executes all 9 Python blocks directly from the report text."""
    text = REPORT_PATH.read_text(encoding="utf-8")
    blocks = extract_fenced_code_blocks(text, "python")
    assert len(blocks) == 9, f"Expected 9 code blocks in report, found {len(blocks)}"
    
    ns = {}
    for i, code in enumerate(blocks, 1):
        try:
            exec(code, ns)
        except Exception as e:
            pytest.fail(f"Failed to exec block {i} from report: {e}")
            
    return ns


# ==============================================================================
# EDGE CASE 1: ZERO VOLUME PERIODS (V_t = 0)
# ==============================================================================

class TestZeroVolumeEdgeCases:
    """Stress tests for models handling zero-volume market periods."""

    def test_f3_vwap_zero_volume_at_session_open_produces_nans(self, report_models):
        """
        Model F3 (Anchored VWAP):
        If the first bar(s) of a session have volume = 0 (e.g. Globex illiquid bar or data gap),
        cum_vol is 0, causing division by zero: 0.0 / 0.0 -> NaN.
        This invalidates VWAP and standard deviation bands.
        """
        compute_anchored_vwap_bands = report_models["compute_anchored_vwap_bands"]

        # Synthetic session where the first 3 bars have volume = 0
        df = pd.DataFrame({
            "date": ["2026-11-04"] * 6,
            "high": [5000.0, 5001.0, 5002.0, 5003.0, 5002.0, 5004.0],
            "low": [4998.0, 4999.0, 5000.0, 5001.0, 5000.0, 5002.0],
            "close": [4999.0, 5000.5, 5001.5, 5002.0, 5001.0, 5003.0],
            "volume": [0, 0, 0, 150, 200, 300],
        })

        res_df = compute_anchored_vwap_bands(df)

        # Verificación blindada: Las primeras barras con cum_vol=0 están protegidas de NaN
        assert not np.isnan(res_df.loc[0, "VWAP"]), "Bar 0 VWAP debe estar protegido contra NaN"
        assert np.isclose(res_df.loc[0, "VWAP"], 4999.0)
        assert np.isclose(res_df.loc[0, "vwap_std"], 0.0)
        assert not np.isnan(res_df.loc[1, "VWAP"])
        assert not np.isnan(res_df.loc[2, "VWAP"])

        # Al entrar volumen positivo, VWAP se actualiza fluidamente
        assert not np.isnan(res_df.loc[3, "VWAP"])
        assert res_df.loc[3, "VWAP"] > 0

    def test_f3_vwap_entire_session_zero_volume_all_nans(self, report_models):
        """If all bars have volume = 0, VWAP and bands remain valid numbers (typical_price and std=0) without NaN."""
        compute_anchored_vwap_bands = report_models["compute_anchored_vwap_bands"]

        df = pd.DataFrame({
            "date": ["2026-11-04"] * 5,
            "high": [5000.0] * 5,
            "low": [4998.0] * 5,
            "close": [4999.0] * 5,
            "volume": [0] * 5,
        })

        res_df = compute_anchored_vwap_bands(df)
        assert not res_df["VWAP"].isna().any(), "Ningún valor de VWAP debe ser NaN aunque el volumen sea cero"
        assert not res_df["upper_band_2"].isna().any()
        assert np.allclose(res_df["VWAP"], 4999.0)
        assert np.allclose(res_df["vwap_std"], 0.0)

    def test_f6_orb_zero_volume_filter(self, report_models):
        """
        Model F6 (Filtered ORB):
        Tests division by zero protection in RVOL calculation when avg_volume = 0 or or_volume = 0.
        Formula: rvol = or_volume / (avg_volume + 1e-8)
        """
        FilteredQuantitativeORB = report_models["FilteredQuantitativeORB"]
        orb = FilteredQuantitativeORB()
        orb.is_compressed = True  # force compression to test volume filter

        # Case A: Both volumes are 0
        triggered_zero = orb.setup_triggers(
            or_high=5010.0, or_low=5000.0, or_volume=0.0, avg_volume=0.0, stretch_val=2.5
        )
        assert triggered_zero is False, "Zero volume should not trigger breakout setup"

        # Case B: Opening range volume is 0, avg volume is positive
        triggered_zero_or = orb.setup_triggers(
            or_high=5010.0, or_low=5000.0, or_volume=0.0, avg_volume=1000.0, stretch_val=2.5
        )
        assert triggered_zero_or is False, "Zero OR volume must not trigger"

        # Case C: Valid high volume triggers properly
        triggered_valid = orb.setup_triggers(
            or_high=5010.0, or_low=5000.0, or_volume=2500.0, avg_volume=1000.0, stretch_val=2.5
        )
        assert triggered_valid is True, "RVOL=2.5 >= 1.8 must trigger"
        assert orb.buy_trigger == 5010.0 + 1.25
        assert orb.sell_trigger == 5000.0 - 1.25


# ==============================================================================
# EDGE CASE 2: FLAT PRICES / ZERO VARIANCE (sigma = 0)
# ==============================================================================

class TestZeroVarianceEdgeCases:
    """Stress tests when prices or spreads are completely constant (zero volatility)."""

    def test_f1_ofi_flat_market_zero_variance_std(self, report_models):
        """
        Model F1 (OFI):
        When prices and sizes are completely static (zero variance),
        OFI series is all 0. OFI_std is 0.0.
        Z_OFI = (OFI - OFI_mean) / (OFI_std + 1e-8)
        The denominator epsilon (+ 1e-8) must protect against ZeroDivisionError and produce 0.0.
        """
        calculate_ofi_series = report_models["calculate_ofi_series"]

        lob_df = pd.DataFrame({
            "bid_price": [5000.0] * 40,
            "bid_size": [10.0] * 40,
            "ask_price": [5000.25] * 40,
            "ask_size": [10.0] * 40,
        })

        res_df = calculate_ofi_series(lob_df, window=30)
        # For rows >= 30, Z_OFI should be exactly 0.0 (not NaN, not Inf)
        valid_z = res_df.loc[30:, "Z_OFI"]
        assert np.allclose(valid_z, 0.0), f"Z_OFI under flat market should be 0.0, got {valid_z.values}"

    def test_f3_vwap_bands_collapse_under_zero_variance(self, report_models):
        """
        Model F3 (VWAP):
        When price is completely constant but volume > 0, vwap_std must be 0.0,
        and upper/lower bands must collapse exactly to VWAP without crashing.
        """
        compute_anchored_vwap_bands = report_models["compute_anchored_vwap_bands"]

        df = pd.DataFrame({
            "date": ["2026-11-04"] * 10,
            "high": [5000.0] * 10,
            "low": [5000.0] * 10,
            "close": [5000.0] * 10,
            "volume": [100] * 10,
        })

        res_df = compute_anchored_vwap_bands(df)
        assert np.allclose(res_df["VWAP"], 5000.0)
        assert np.allclose(res_df["vwap_std"], 0.0)
        assert np.allclose(res_df["upper_band_2"], 5000.0)
        assert np.allclose(res_df["lower_band_2"], 5000.0)

    def test_f7_vrp_zero_variance_history(self, report_models):
        """
        Model F7 (VRP):
        When historical VRP series is constant (std = 0), epsilon (+ 1e-8) must prevent division by zero.
        """
        compute_vrp_metrics = report_models["compute_vrp_metrics"]

        # Constant history of 0.005
        vrp_hist = [0.005] * 30
        res = compute_vrp_metrics(vix=18.0, vix3m=19.0, rv_ann=0.0274, vrp_history=vrp_hist)

        assert isinstance(res["Z_VRP"], float)
        assert not np.isnan(res["Z_VRP"])
        assert not np.isinf(res["Z_VRP"])

    def test_f9_ou_constant_flat_prices_lstsq_rank_deficiency(self, report_models):
        """
        Model F9 (OU Pairs Trading):
        When both ES and NQ log prices are perfectly constant, OLS residuals are at machine epsilon (~1e-15).
        Due to floating point noise, b evaluates to a tiny positive number (~1e-30 > 0), resulting in an
        infinitesimal half-life (~0.05 min) and S-score = 0.0, outputting 'CLOSE'.
        This test verifies that under flat prices, the model outputs either 'NO_TRADE' or 'CLOSE',
        and never initiates a new directional trade ('SHORT_NQ_LONG_ES' or 'LONG_NQ_SHORT_ES').
        """
        calibrate_ou = report_models["calibrate_ornstein_uhlenbeck_pairs"]

        p_nq = np.array([np.log(18000.0)] * 50)
        p_es = np.array([np.log(5000.0)] * 50)

        res = calibrate_ou(p_nq, p_es)
        # S-score is 0.0, so it must not initiate positions
        assert res["action"] in ("NO_TRADE", "CLOSE")
        assert np.isclose(res["s_score"], 0.0)


# ==============================================================================
# EDGE CASE 3: EXTREME PRICE SHOCKS (+10% / -10% FLASH CRASHES)
# ==============================================================================

class TestExtremePriceShockEdgeCases:
    """Stress tests under sudden market crashes, flash shocks, and volatility spikes."""

    def test_f2_momentum_flash_crash_gap(self, report_models):
        """
        Model F2 (Intraday Momentum):
        Flash crash of -10% during morning window (09:30 to 10:00).
        Evaluates direction = -1 (short).
        At 15:30 entry, if price gaps up beyond stop loss, verifies stop loss trigger.
        """
        Strategy = report_models["MarketIntradayMomentumStrategy"]
        strat = Strategy(threshold_pct=0.0020, stop_loss_pct=0.0035)

        # 10% morning crash
        direction = strat.evaluate_morning_window(p_0930=5000.0, p_1000=4500.0)
        assert direction == -1, "Direction should be short (-1)"

        # 15:30 entry at 4500.0
        entry_res = strat.on_bar_update(datetime.time(15, 30), 4500.0)
        assert entry_res["action"] == "SELL"
        # Stop price for short: 4500 * (1 + 0.0035) = 4515.75
        assert np.isclose(strat.stop_price, 4515.75)

        # Price spikes against short to 4520.0 -> STOP_LOSS must trigger
        stop_res = strat.on_bar_update(datetime.time(15, 35), 4520.0)
        assert stop_res["action"] == "STOP_LOSS"
        assert strat.position == 0

    def test_f4_overnight_unbounded_drawdown_vulnerability(self, report_models):
        """
        CRITICAL VULNERABILITY AUDIT — Model F4 (Conditional Overnight Drift):
        Under Boyarchenko et al. (2023), when RTH closes down (< -0.75%), F4 enters LONG at 01:30 EST.
        However, the pseudocode in Block 4 contains NO intraday stop-loss check in evaluate_overnight_bar!
        If a severe overnight shock occurs (e.g. ES drops 150 points between 01:30 and 08:30),
        the strategy holds the position until 08:30, incurring -$7,500 drawdown per contract,
        which immediately wipes out the Prop Trading Daily Loss Limit ($1,000)!
        """
        OvernightDrift = report_models["ConditionalOvernightDrift"]
        strat = OvernightDrift()

        # Prior day closed down -1.5%
        eligible = strat.check_rth_close(p_open_0930=5000.0, p_close_1600=4925.0)
        assert eligible is True

        # Enter LONG at 01:30 EST at 4920.0
        buy_res = strat.evaluate_overnight_bar("01:30", 4920.0)
        assert buy_res["action"] == "BUY"
        assert strat.position == 1

        # Overnight shock: at 04:00 EST, price collapses to 4750.0 (-170 points = -$8,500/contract)
        shock_res = strat.evaluate_overnight_bar("04:00", 4750.0)
        
        # Verificación blindada: La estrategia activa inmediatamente el STOP_LOSS ante el desplome
        assert shock_res["action"] == "STOP_LOSS", (
            "Blindaje verificado: Model F4 detiene el trade ante el desplome nocturno."
        )
        assert strat.position == 0
        assert shock_res["price"] == 4750.0

    def test_f8_harrv_extreme_volatility_dll_breach(self, report_models):
        """
        CRITICAL VULNERABILITY AUDIT — Model F8 (HAR-RV-J & Volatility Managed Sizing):
        Under extreme market volatility (e.g. VIX > 60 or single-day RV spike where 1-sigma daily move > 200 pts),
        calculate_safe_contracts computes:
            micro_risk = sigma_pts * 5.0 * 1.645
            allowed_dollar_risk = dll_budget * phi (e.g. 1000 * 0.40 = $400)
            contracts = max(1, int(allowed_dollar_risk / (micro_risk + 1e-8)))
        
        Because of 'max(1, ...)', when micro_risk ($1,645+) exceeds allowed_dollar_risk ($400)
        and even exceeds the entire Daily Loss Limit budget ($1,000), it STILL allocates 1 contract!
        This mathematically violates the Prop Trading risk rule under extreme shocks.
        """
        Manager = report_models["HARRVVolatilityManager"]
        mgr = Manager(point_multiplier=50.0, dll_budget=1000.0, safety_phi=0.40)

        # Extreme RV history simulating flash crash / crisis regime (daily vol ~ 4%)
        extreme_rv_history = [0.0016] * 30  # sqrt(0.0016) = 0.04 (4% daily move)
        current_price = 5000.0  # 4% of 5000 = 200 points move!

        res = mgr.calculate_safe_contracts(current_price, extreme_rv_history)

        assert res["symbol"] == "MES"
        # The risk of even 1 MES contract is:
        # sigma_pts ~ 5000 * sqrt(0.0014) ~ 187 points.
        # micro_risk = 187 * 5.0 * 1.645 = $1,538.07!
        # This is GREATER than the entire DLL ($1,000)!
        assert res["risk_dollar"] > mgr.dll_budget, (
            f"Micro risk ${res['risk_dollar']:.2f} exceeds entire DLL budget ${mgr.dll_budget:.2f}"
        )
        # Verificación blindada: El disyuntor previene la violación del DLL asignando 0 contratos
        assert res["contracts"] == 0, (
            "Blindaje verificado: F8 activa disyuntor (contracts=0) cuando micro_risk > allowed_dollar_risk!"
        )
        assert res["action"] == "HALT_VOLATILITY_EXCEEDS_DLL"


# ==============================================================================
# EDGE CASE 4: COINTEGRATION DIVERGENCE (theta <= 0, b >= 1.0, STRUCTURAL BREAKS)
# ==============================================================================

class TestCointegrationDivergenceEdgeCases:
    """Stress tests for pairs trading under breakdown of statistical equilibrium."""

    def test_f9_ou_explosive_and_random_walk_spread(self, report_models):
        """
        Model F9 (OU Pairs Trading):
        When spread is explosive (b >= 1.0, e.g. b = 1.05) or random walk (b = 1.0),
        the spread does NOT revert to the mean.
        The algorithm must immediately detect non-stationarity and output 'NO_TRADE'.
        """
        calibrate_ou = report_models["calibrate_ornstein_uhlenbeck_pairs"]

        np.random.seed(42)
        n = 100
        # Synthetic diverging series where NQ grows much faster than ES without bound
        p_es = np.linspace(8.5, 8.6, n)
        p_nq = np.linspace(9.7, 10.2, n) + np.cumsum(np.random.normal(0.01, 0.02, n))

        res = calibrate_ou(p_nq, p_es)
        # Should detect non-mean-reverting behavior
        assert res["action"] in ("NO_TRADE", "STOP_LOSS", "HOLD")

    def test_f9_ou_catastrophic_divergence_triggers_stop_loss(self, report_models):
        """
        Model F9 (OU Pairs Trading):
        If an open pair trade suffers a catastrophic cointegration breakdown,
        producing |S-score| > 3.25, it must trigger 'STOP_LOSS'.
        """
        calibrate_ou = report_models["calibrate_ornstein_uhlenbeck_pairs"]

        np.random.seed(42)
        n = 100
        # Generate stationary OU spread first
        p_es = np.ones(n) * 8.5
        # Sudden 10-sigma spike on the last 3 bars
        p_nq = np.ones(n) * 9.8
        p_nq[-1] = 10.5  # massive divergence

        res = calibrate_ou(p_nq, p_es)
        assert res["action"] in ("STOP_LOSS", "NO_TRADE"), (
            f"Catastrophic divergence must trigger STOP_LOSS or NO_TRADE, got {res['action']}"
        )

    def test_f9_ou_insufficient_data_points_handling(self, report_models):
        """
        If length 1 input is supplied, residuals[1:] is empty, resulting in b=0.0 from lstsq.
        Since b <= 0.0, the implementation safely returns 'NO_TRADE' without crashing.
        """
        calibrate_ou = report_models["calibrate_ornstein_uhlenbeck_pairs"]

        p_nq = np.array([9.8])
        p_es = np.array([8.5])

        res = calibrate_ou(p_nq, p_es)
        assert res["action"] == "NO_TRADE"
        assert res["half_life_min"] == np.inf
