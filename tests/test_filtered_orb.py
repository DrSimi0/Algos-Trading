"""
tests/test_filtered_orb.py
Suite completa de pruebas unitarias y de estrés cuantitativo para la estrategia F2:
Filtered Quantitative Opening Range Breakout (ORB).
"""

from datetime import datetime, time
import math
import numpy as np
import pandas as pd
import pytest

from src.strategies.filtered_orb import (
    FilteredQuantitativeORB,
    PropAccount,
    FilteredORBBacktester,
    TradeRecord,
    StrategyState,
    ExitReason,
    check_nr4,
    check_inside_day,
    check_prior_compression,
    calculate_crabel_stretch,
    calculate_rvol,
    calculate_triggers,
    calculate_delta_risk,
    calculate_position_size,
    normalize_to_eastern,
)


# ==============================================================================
# 1. PRUEBAS DE COMPRESIÓN DE VOLATILIDAD PREVIA (NR4 / INSIDE DAY)
# ==============================================================================

class TestVolatilityCompression:
    def test_nr4_detection(self):
        # Range t-1 (8.0) < min(12.0, 15.0, 10.0) -> True
        ranges_true = [10.0, 15.0, 12.0, 8.0]
        assert check_nr4(ranges_true) is True

        # Range t-1 (11.0) >= min(10.0, 15.0, 12.0) -> False
        ranges_false = [10.0, 15.0, 12.0, 11.0]
        assert check_nr4(ranges_false) is False

        # Insuficientes barras
        assert check_nr4([10.0, 8.0]) is False

    def test_inside_day_detection(self):
        # High t-1 <= High t-2 AND Low t-1 >= Low t-2 -> True
        highs_true = [5850.0, 5840.0]
        lows_true = [5800.0, 5810.0]
        assert check_inside_day(highs_true, lows_true) is True

        # High t-1 > High t-2 -> False
        highs_false = [5850.0, 5855.0]
        lows_false = [5800.0, 5810.0]
        assert check_inside_day(highs_false, lows_false) is False

        # Low t-1 < Low t-2 -> False
        highs_false2 = [5850.0, 5840.0]
        lows_false2 = [5800.0, 5795.0]
        assert check_inside_day(highs_false2, lows_false2) is False

    def test_prior_compression_disjunction(self):
        # Caso 1: Solo NR4
        df_nr4 = pd.DataFrame({
            "Open": [5800, 5800, 5800, 5800],
            "High": [5830, 5825, 5820, 5810],
            "Low":  [5800, 5800, 5800, 5805],
            "Close": [5820, 5810, 5815, 5808]
        })
        assert check_prior_compression(df_nr4) is True

        # Caso 2: Solo Inside Day
        df_id = pd.DataFrame({
            "Open": [5800, 5800, 5800, 5800],
            "High": [5820, 5820, 5850, 5840],
            "Low":  [5800, 5800, 5800, 5810],
            "Close": [5810, 5810, 5830, 5825]
        })
        assert check_prior_compression(df_id) is True

        # Caso 3: Ninguna compresión
        df_none = pd.DataFrame({
            "Open": [5800, 5800, 5800, 5800],
            "High": [5810, 5815, 5820, 5860],
            "Low":  [5800, 5800, 5800, 5790],
            "Close": [5805, 5810, 5815, 5850]
        })
        assert check_prior_compression(df_none) is False


# ==============================================================================
# 2. PRUEBAS DEL AMORTIGUADOR DE RUIDO DE CRABEL (STRETCH)
# ==============================================================================

class TestCrabelStretch:
    def test_stretch_calculation_exact(self):
        # 10 barras con valores exactos
        opens = [5800.0] * 10
        highs = [5810.0] * 10
        lows = [5795.0] * 10  # H - O = 10, O - L = 5 -> D = 5
        closes = [5805.0] * 10

        df = pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows, "Close": closes
        })
        stretch = calculate_crabel_stretch(df, lookback=10)
        assert stretch == 5.0

    def test_stretch_lookback_error(self):
        df_short = pd.DataFrame({
            "Open": [5800.0] * 5, "High": [5810.0] * 5,
            "Low": [5790.0] * 5, "Close": [5805.0] * 5
        })
        with pytest.raises(ValueError, match="al menos 10 barras"):
            calculate_crabel_stretch(df_short, lookback=10)

    def test_stretch_zero_variance(self):
        # Mercado congelado (sin oscilación)
        df_flat = pd.DataFrame({
            "Open": [5800.0] * 10, "High": [5800.0] * 10,
            "Low": [5800.0] * 10, "Close": [5800.0] * 10
        })
        assert calculate_crabel_stretch(df_flat, lookback=10) == 0.0


# ==============================================================================
# 3. PRUEBAS DEL VOLUMEN RELATIVO INSTITUCIONAL (RVOL)
# ==============================================================================

class TestRelativeVolumeRVOL:
    def test_rvol_formula_and_threshold(self):
        hist_vols = [1000.0] * 20
        or_vol_valid = 1800.0  # Exactly 1.80
        rvol, is_valid = calculate_rvol(or_vol_valid, hist_vols, lookback=20, rvol_min=1.80)
        assert rvol == 1.80
        assert is_valid is True

        or_vol_low = 1799.0
        rvol_low, is_valid_low = calculate_rvol(or_vol_low, hist_vols, lookback=20, rvol_min=1.80)
        assert rvol_low < 1.80
        assert is_valid_low is False

    def test_rvol_zero_volume_protection(self):
        # Volumen de apertura nulo
        rvol, is_valid = calculate_rvol(0.0, [1000.0] * 20)
        assert rvol == 0.0
        assert is_valid is False

        # Historial nulo
        rvol, is_valid = calculate_rvol(2000.0, [])
        assert rvol == 0.0
        assert is_valid is False

        # Historial con ceros
        rvol, is_valid = calculate_rvol(2000.0, [0.0] * 20)
        assert rvol == 0.0
        assert is_valid is False


# ==============================================================================
# 4. PRUEBAS DE GATILLOS, STOP LOSS Y TAKE PROFIT (2R)
# ==============================================================================

class TestTriggersAndTargets:
    def test_triggers_and_midpoint(self):
        orh = 5820.0
        orl = 5800.0
        stretch = 10.0
        lambda_stretch = 0.50

        long_trig, short_trig, midpoint = calculate_triggers(orh, orl, stretch, lambda_stretch)
        assert midpoint == 5810.0
        assert long_trig == 5820.0 + 5.0   # 5825.0
        assert short_trig == 5800.0 - 5.0  # 5795.0

        delta_risk = calculate_delta_risk(orh, orl, stretch, lambda_stretch)
        assert delta_risk == 15.0

        # Verificación dimensional
        assert long_trig - midpoint == delta_risk
        assert midpoint - short_trig == delta_risk

    def test_asymmetry_2r(self):
        entry_price = 5825.0
        delta_risk = 15.0

        tp_long = entry_price + (2.0 * delta_risk)
        assert tp_long == 5855.0
        assert (tp_long - entry_price) / delta_risk == 2.0

        entry_short = 5795.0
        tp_short = entry_short - (2.0 * delta_risk)
        assert tp_short == 5765.0
        assert (entry_short - tp_short) / delta_risk == 2.0


# ==============================================================================
# 5. PRUEBAS DE GESTIÓN DE RIESGO PROP TRADING Y DIMENSIONAMIENTO
# ==============================================================================

class TestPropTradingRiskAndSizing:
    def test_risk_budget_streak_attenuation(self):
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)

        # Racha 0: phi = 0.35 * 2^0 = 0.35. rem_daily = 1000 -> 350. rem_mtdd = 2000 -> psi * 2000 = 500
        # Budget = min(350, 500) = 350.0
        assert account.calculate_risk_budget() == 350.0

        # Racha 1: phi = 0.35 / 2 = 0.175 -> budget = 175.0
        account.k_loss = 1
        assert account.calculate_risk_budget() == 175.0

        # Racha 2: phi = 0.35 / 4 = 0.0875 -> budget = 87.5
        account.k_loss = 2
        assert account.calculate_risk_budget() == 87.5

        # Racha 3: phi = 0.35 / 8 = 0.04375 -> budget = 43.75
        account.k_loss = 3
        assert account.calculate_risk_budget() == 43.75

        # Racha 4 (saturada en 3): budget = 43.75
        account.k_loss = 4
        assert account.calculate_risk_budget() == 43.75

    def test_prop_account_drawdown_breach(self):
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)

        # Simular trade perdedor de $2,100 -> viola MTDD
        trade = TradeRecord(
            date="2026-10-15",
            symbol="MES",
            direction=1,
            contracts=5,
            is_micro=True,
            entry_time="09:50",
            entry_price=5800.0,
            stop_loss=5790.0,
            take_profit=5820.0,
            delta_risk=10.0,
            net_pnl_usd=-2100.0
        )
        account.record_trade(trade)
        assert account.is_breached is True
        assert "TRAILING_DRAWDOWN_BREACH" in account.breach_reason
        assert account.calculate_risk_budget() == 0.0

    def test_prop_account_daily_loss_breach(self):
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)

        # Simular trade perdedor de $1,050 en el día -> viola DLL
        trade = TradeRecord(
            date="2026-10-15",
            symbol="MES",
            direction=1,
            contracts=2,
            is_micro=True,
            entry_time="09:50",
            entry_price=5800.0,
            stop_loss=5790.0,
            take_profit=5820.0,
            delta_risk=10.0,
            net_pnl_usd=-1050.0
        )
        account.record_trade(trade)
        assert account.is_breached is True
        assert "DAILY_LOSS_LIMIT_BREACH" in account.breach_reason
        assert account.calculate_risk_budget() == 0.0

    def test_position_sizing_switch_to_micro(self):
        # ES tiene M=50, tick=0.25, comm=4.50. Fricción = 2 * 0.25 * 50 + 4.5 = 29.50
        # Delta_risk = 10 puntos -> R_std = 10 * 50 + 29.5 = 529.50 USD
        # Si budget es 350 USD -> n_std = floor(350 / 529.5) = 0 -> Conmuta a MES
        # MES tiene M=5, tick=0.25, comm=1.24. Fricción = 2 * 0.25 * 5 + 1.24 = 3.74
        # R_micro = 10 * 5 + 3.74 = 53.74 USD
        # n_micro = floor(350 / 53.74) = 6 contratos
        sym, n, r_contract, status = calculate_position_size(
            risk_budget=350.0,
            delta_risk=10.0,
            std_symbol="ES",
            micro_symbol="MES",
            max_micros=15
        )
        assert sym == "MES"
        assert n == 6
        assert status == "MICRO_SWITCH_ACTIVE"

    def test_excessive_range_circuit_breaker(self):
        # Rango masivo de 100 puntos durante noticias macroeconómicas
        # R_micro = 100 * 5 + 3.74 = 503.74 USD
        # Con budget de 350 USD -> R_micro > budget -> Disyuntor se dispara
        sym, n, r_contract, status = calculate_position_size(
            risk_budget=350.0,
            delta_risk=100.0,
            std_symbol="ES",
            micro_symbol="MES"
        )
        assert n == 0
        assert status == "HALT_EXCESSIVE_RANGE_DEFENSE"


# ==============================================================================
# 6. PRUEBAS DE CICLO DE VIDA INTRADÍA Y VENTANAS TEMPORALES
# ==============================================================================

class TestIntradayExecutionLifecycle:
    @pytest.fixture
    def setup_synthetic_day(self):
        # Generar 10 días previos para contexto
        dates = pd.date_range("2026-10-01", periods=10, freq="B")
        daily_df = pd.DataFrame({
            "Open": [5800.0] * 10,
            "High": [5820.0, 5820.0, 5820.0, 5820.0, 5820.0, 5820.0, 5820.0, 5820.0, 5815.0, 5810.0],
            "Low":  [5780.0, 5780.0, 5780.0, 5780.0, 5780.0, 5780.0, 5780.0, 5780.0, 5795.0, 5800.0],
            "Close": [5800.0] * 10,
            "Volume": [1000000.0] * 10
        }, index=dates)

        # Velas intradía de 5 minutos para 2026-10-15 (09:30 a 16:00)
        times = pd.date_range("2026-10-15 09:30", "2026-10-15 16:00", freq="5min")
        n_bars = len(times)
        intraday_df = pd.DataFrame({
            "Open": [5805.0] * n_bars,
            "High": [5808.0] * n_bars,
            "Low":  [5802.0] * n_bars,
            "Close": [5805.0] * n_bars,
            "Volume": [5000.0] * n_bars
        }, index=times)

        return daily_df, intraday_df

    def test_lunchtime_order_cancellation(self, setup_synthetic_day):
        daily_df, intraday_df = setup_synthetic_day
        strat = FilteredQuantitativeORB()
        account = PropAccount()

        # El precio nunca quiebra los gatillos entre 09:45 y 12:00
        hist_vols = [2000.0] * 20  # RVOL alto (3 barras de 5000 = 15000 / 2000 = 7.5 > 1.8)
        trade = strat.process_day(
            session_date="2026-10-15",
            intraday_bars=intraday_df,
            daily_history=daily_df,
            historical_or_volumes=hist_vols,
            account=account
        )
        assert trade is None
        assert strat.state == StrategyState.ORDER_CANCELLED

    def test_time_decay_forced_flat_at_1545(self, setup_synthetic_day):
        daily_df, intraday_df = setup_synthetic_day
        strat = FilteredQuantitativeORB()
        account = PropAccount()

        # Provocar ruptura alcista a las 10:00 y mantener el precio por encima del punto medio (5805)
        post_breakout_mask = (intraday_df.index >= "2026-10-15 10:00:00")
        intraday_df.loc[post_breakout_mask, "Open"] = 5825.0
        intraday_df.loc[post_breakout_mask, "High"] = 5830.0
        intraday_df.loc[post_breakout_mask, "Low"] = 5820.0  # > midpoint 5805.0
        intraday_df.loc[post_breakout_mask, "Close"] = 5825.0

        hist_vols = [2000.0] * 20
        trade = strat.process_day(
            session_date="2026-10-15",
            intraday_bars=intraday_df,
            daily_history=daily_df,
            historical_or_volumes=hist_vols,
            account=account
        )

        assert trade is not None
        assert trade.exit_reason == ExitReason.TIME_DECAY
        assert "15:45" in trade.exit_time
        assert strat.state == StrategyState.FLAT_COMPLETED

    def test_take_profit_hit(self, setup_synthetic_day):
        daily_df, intraday_df = setup_synthetic_day
        strat = FilteredQuantitativeORB()
        account = PropAccount()

        # Ruptura alcista a las 10:00 manteniéndose por encima del midpoint
        post_breakout_mask = (intraday_df.index >= "2026-10-15 10:00:00")
        intraday_df.loc[post_breakout_mask, "Open"] = 5825.0
        intraday_df.loc[post_breakout_mask, "High"] = 5830.0
        intraday_df.loc[post_breakout_mask, "Low"] = 5820.0
        intraday_df.loc[post_breakout_mask, "Close"] = 5825.0

        # Vela posterior a las 11:00 toca Take Profit (> 5860)
        tp_idx = intraday_df.index.get_loc("2026-10-15 11:00:00")
        intraday_df.iloc[tp_idx, intraday_df.columns.get_loc("High")] = 5865.0

        hist_vols = [2000.0] * 20
        trade = strat.process_day(
            session_date="2026-10-15",
            intraday_bars=intraday_df,
            daily_history=daily_df,
            historical_or_volumes=hist_vols,
            account=account
        )


        assert trade is not None
        assert trade.exit_reason == ExitReason.TAKE_PROFIT
        assert trade.net_pnl_usd > 0
        assert account.k_loss == 0

    def test_stop_loss_hit_at_midpoint(self, setup_synthetic_day):
        daily_df, intraday_df = setup_synthetic_day
        strat = FilteredQuantitativeORB()
        account = PropAccount()

        # Ruptura alcista a las 10:00
        b_idx = intraday_df.index.get_loc("2026-10-15 10:00:00")
        intraday_df.iloc[b_idx, intraday_df.columns.get_loc("High")] = 5830.0

        # Reversión violenta a las 10:30 perforando el punto medio
        sl_idx = intraday_df.index.get_loc("2026-10-15 10:30:00")
        intraday_df.iloc[sl_idx, intraday_df.columns.get_loc("Low")] = 5790.0

        hist_vols = [2000.0] * 20
        trade = strat.process_day(
            session_date="2026-10-15",
            intraday_bars=intraday_df,
            daily_history=daily_df,
            historical_or_volumes=hist_vols,
            account=account
        )

        assert trade is not None
        assert trade.exit_reason == ExitReason.STOP_LOSS
        assert trade.net_pnl_usd < 0
        assert account.k_loss == 1


# ==============================================================================
# 7. PRUEBA DEL MOTOR DE BACKTEST COMPLETO (FilteredORBBacktester)
# ==============================================================================

class TestFilteredORBBacktester:
    def test_backtester_end_to_end_on_real_data(self):
        # Ejecutar backtest sobre el dataset real disponible
        df_intra = pd.read_csv("data/raw/CME_MINI_ES_5m_bars5000.csv", index_col=0, parse_dates=True)
        df_daily = pd.read_csv("data/raw/CME_MINI_ES_1d_bars5000.csv", index_col=0, parse_dates=True)

        strat = FilteredQuantitativeORB(
            std_symbol="ES",
            micro_symbol="MES",
            rvol_min=1.15  # Umbral adaptado al tamaño muestral de 23 días
        )
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        backtester = FilteredORBBacktester(strategy=strat, account=account)

        report = backtester.run(intraday_data=df_intra, daily_data=df_daily)
        df_trades = backtester.get_trades_dataframe()

        assert report["initial_capital"] == 50000.0
        assert report["total_trades"] == len(df_trades)
        assert report["is_prop_breached"] is False
        assert report["total_commissions_usd"] > 0
        assert report["total_slippage_usd"] > 0
        assert len(backtester.daily_equity_curve) > 0

    def test_backtester_end_to_end_on_nq(self):
        # Ejecutar backtest sobre NQ
        df_intra = pd.read_csv("data/raw/CME_MINI_NQ_5m_bars5000.csv", index_col=0, parse_dates=True)
        df_daily = pd.read_csv("data/raw/CME_MINI_NQ_1d_bars5000.csv", index_col=0, parse_dates=True)

        strat = FilteredQuantitativeORB(
            std_symbol="NQ",
            micro_symbol="MNQ",
            rvol_min=1.15
        )
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        backtester = FilteredORBBacktester(strategy=strat, account=account)

        report = backtester.run(intraday_data=df_intra, daily_data=df_daily)
        assert report["initial_capital"] == 50000.0
        assert report["is_prop_breached"] is False


# ==============================================================================
# 8. PRUEBAS DE VALIDACIÓN DE DATOS CORRUPTOS Y CASOS LÍMITE
# ==============================================================================

class TestCorruptDataAndEdgeCases:
    def test_corrupt_daily_bars_high_less_than_low(self):
        df_corrupt = pd.DataFrame({
            "Open": [5800.0] * 10,
            "High": [5790.0] * 10,  # High < Low corrupto
            "Low":  [5810.0] * 10,
            "Close": [5800.0] * 10
        })
        with pytest.raises(ValueError, match="High < Low"):
            calculate_crabel_stretch(df_corrupt, lookback=10)

        with pytest.raises(ValueError, match="High < Low"):
            check_prior_compression(df_corrupt)

    def test_corrupt_opening_range_triggers_and_delta(self):
        with pytest.raises(ValueError, match="ORH .* < ORL"):
            calculate_triggers(orh=5800.0, orl=5810.0, stretch=5.0)

        with pytest.raises(ValueError, match="no puede ser negativo"):
            calculate_triggers(orh=5820.0, orl=5800.0, stretch=-1.0)

        with pytest.raises(ValueError, match="ORH .* < ORL"):
            calculate_delta_risk(orh=5800.0, orl=5810.0, stretch=5.0)

        with pytest.raises(ValueError, match="no puede ser negativo"):
            calculate_delta_risk(orh=5820.0, orl=5800.0, stretch=-2.0)

    def test_corrupt_opening_range_bars_rejected(self):
        times = pd.date_range("2026-10-15 09:30", "2026-10-15 09:40", freq="5min")
        df_corrupt_bars = pd.DataFrame({
            "Open": [5800.0, 5800.0, 5800.0],
            "High": [5790.0, 5810.0, 5810.0],
            "Low":  [5810.0, 5790.0, 5790.0],
            "Close": [5800.0, 5800.0, 5800.0],
            "Volume": [1000.0, 1000.0, 1000.0]
        }, index=times)

        strat = FilteredQuantitativeORB()
        with pytest.raises(ValueError, match="High < Low"):
            strat.evaluate_opening_range(df_corrupt_bars, [])


# ==============================================================================
# 9. PRUEBAS DE NORMALIZACIÓN HORARIA (AMERICA/NEW_YORK)
# ==============================================================================

class TestTimezoneNormalization:
    def test_auto_detect_mexico_city_to_new_york(self):
        # Crear dataset con pico matutino a las 07:30
        times = pd.date_range("2026-10-15 06:00", "2026-10-15 15:00", freq="5min")
        vols = [1000.0] * len(times)
        df_raw = pd.DataFrame({
            "Open": [5800.0] * len(times),
            "High": [5810.0] * len(times),
            "Low":  [5795.0] * len(times),
            "Close": [5805.0] * len(times),
            "Volume": vols
        }, index=times)
        # Inyectar pico a las 07:30
        idx_0730 = df_raw.index.get_loc("2026-10-15 07:30:00")
        df_raw.iloc[idx_0730, df_raw.columns.get_loc("Volume")] = 35000.0

        df_ny = normalize_to_eastern(df_raw)
        # El pico de 07:30 Mexico City (UTC-6) corresponde a 09:30 New York (EDT, UTC-4)
        peak_ny_time = df_ny["Volume"].idxmax()
        assert peak_ny_time.hour == 9
        assert peak_ny_time.minute == 30
        assert str(df_ny.index.tz) == "America/New_York"


# ==============================================================================
# 10. PRUEBAS DE SEGURIDAD EN EJECUCIÓN INTRA-BARRA
# ==============================================================================

class TestSameBarExecutionSafety:
    def test_breakout_candle_pre_entry_dip_does_not_false_stop_out(self):
        """
        Una vela que abre abajo, marca un mínimo previo bajo el punto medio,
        luego rompe al alza el gatillo y cierra cerca del máximo
        NO debe ser detenida falsamente por el mínimo que ocurrió antes de la entrada.
        """
        # Contexto previo
        daily_df = pd.DataFrame({
            "Open": [5800.0] * 10,
            "High": [5820.0, 5820.0, 5820.0, 5820.0, 5820.0, 5820.0, 5820.0, 5820.0, 5815.0, 5810.0],
            "Low":  [5780.0, 5780.0, 5780.0, 5780.0, 5780.0, 5780.0, 5780.0, 5780.0, 5795.0, 5800.0],
            "Close": [5800.0] * 10,
            "Volume": [1000000.0] * 10
        }, index=pd.date_range("2026-10-01", periods=10, freq="B"))

        times = pd.date_range("2026-10-15 09:30", "2026-10-15 16:00", freq="5min")
        intraday_df = pd.DataFrame({
            "Open": [5805.0] * len(times),
            "High": [5808.0] * len(times),
            "Low":  [5802.0] * len(times),
            "Close": [5805.0] * len(times),
            "Volume": [5000.0] * len(times)
        }, index=times)

        # En la vela de ruptura a las 10:00:
        # Abre en 5804, mínimo en 5798 (bajo midpoint 5805), sube a 5825 (gatillo largo 5811.75) y cierra en 5823
        b_idx = intraday_df.index.get_loc("2026-10-15 10:00:00")
        intraday_df.iloc[b_idx, intraday_df.columns.get_loc("Open")] = 5804.0
        intraday_df.iloc[b_idx, intraday_df.columns.get_loc("Low")] = 5798.0
        intraday_df.iloc[b_idx, intraday_df.columns.get_loc("High")] = 5825.0
        intraday_df.iloc[b_idx, intraday_df.columns.get_loc("Close")] = 5823.0

        # Velas posteriores se mantienen arriba
        for t_idx in range(b_idx + 1, len(intraday_df)):
            intraday_df.iloc[t_idx, intraday_df.columns.get_loc("Open")] = 5823.0
            intraday_df.iloc[t_idx, intraday_df.columns.get_loc("High")] = 5826.0
            intraday_df.iloc[t_idx, intraday_df.columns.get_loc("Low")] = 5820.0
            intraday_df.iloc[t_idx, intraday_df.columns.get_loc("Close")] = 5824.0

        strat = FilteredQuantitativeORB()
        account = PropAccount()
        trade = strat.process_day(
            session_date="2026-10-15",
            intraday_bars=intraday_df,
            daily_history=daily_df,
            historical_or_volumes=[2000.0] * 20,
            account=account
        )

        assert trade is not None
        # No debe haber salido por Stop Loss en la vela de entrada a las 10:00
        assert trade.exit_reason == ExitReason.TIME_DECAY
        assert "15:45" in trade.exit_time
        assert trade.net_pnl_usd > 0

