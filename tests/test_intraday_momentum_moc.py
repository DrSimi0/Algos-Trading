"""
tests/test_intraday_momentum_moc.py
Suite de pruebas unitarias y de estrés para la Estrategia F1:
Market Intraday Momentum & MOC Seasonality.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.strategies.momentum.intraday_momentum_moc import (
    MarketIntradayMomentumMOC,
    IntradayMomentumBacktester,
)
from src.core.futures_specs import get_spec


# ==============================================================================
# 1. PRUEBAS DE FORMULACIÓN MATEMÁTICA PURA (Secciones 3 y 4 de F1)
# ==============================================================================

class TestF1MathematicalFormulations:
    """Verifica las ecuaciones matemáticas exactas formuladas en F1."""

    def test_morning_log_return_formula(self):
        """Verifica r_first,t = ln( P(10:00) / P(09:30) )."""
        p_930 = 5000.0
        p_1000 = 5025.0
        expected = math.log(5025.0 / 5000.0)
        calc = MarketIntradayMomentumMOC.calculate_morning_return(p_930, p_1000)
        assert pytest.approx(calc, abs=1e-8) == expected
        assert calc > 0

        # Retorno negativo
        p_1000_down = 4975.0
        calc_down = MarketIntradayMomentumMOC.calculate_morning_return(p_930, p_1000_down)
        assert pytest.approx(calc_down, abs=1e-8) == math.log(4975.0 / 5000.0)
        assert calc_down < 0

        # Precios no positivos deben lanzar ValueError
        with pytest.raises(ValueError):
            MarketIntradayMomentumMOC.calculate_morning_return(0.0, 5000.0)
        with pytest.raises(ValueError):
            MarketIntradayMomentumMOC.calculate_morning_return(5000.0, -10.0)

    def test_volume_confirmation_condition(self):
        """Verifica V_ratio,t y la indicatriz Phi_vol,t."""
        v_current = 25000.0
        v_hist = [20000.0] * 20
        v_ratio, phi_vol = MarketIntradayMomentumMOC.calculate_relative_volume(
            v_current, v_hist, k_vol=20
        )
        assert v_ratio == 1.25
        assert phi_vol == 1

        # Volumen inferior al promedio
        v_current_low = 18000.0
        v_ratio_low, phi_vol_low = MarketIntradayMomentumMOC.calculate_relative_volume(
            v_current_low, v_hist, k_vol=20
        )
        assert v_ratio_low == 0.90
        assert phi_vol_low == 0

        # Sin histórico: se asume ratio neutral 1.0 y phi_vol 1
        v_ratio_empty, phi_empty = MarketIntradayMomentumMOC.calculate_relative_volume(
            15000.0, [], k_vol=20
        )
        assert v_ratio_empty == 1.0
        assert phi_empty == 1

    def test_dynamic_threshold_floor_and_expansion(self):
        """Verifica theta_t = max(theta_floor, c_sigma * (sigma_daily / P_1000))."""
        strat = MarketIntradayMomentumMOC(params={
            "theta_floor": 0.0015,
            "c_sigma": 0.20,
            "k_sigma": 20
        })

        # Caso 1: Volatilidad muy baja -> activa theta_floor (0.0015)
        # Precios estables con mínima dispersión
        low_vol_series = pd.Series([5000.0 + (i % 2) * 0.25 for i in range(25)])
        theta_low = strat.calculate_dynamic_threshold(5000.0, low_vol_series)
        assert theta_low == 0.0015

        # Caso 2: Alta volatilidad -> expande por encima de theta_floor
        # Desviación de 100 puntos sobre precio 5000 = 2% volatilidad
        # c_sigma * 0.02 = 0.20 * 0.02 = 0.0040 > 0.0015
        high_vol_prices = pd.Series(np.linspace(4800, 5200, 25))
        theta_high = strat.calculate_dynamic_threshold(5000.0, high_vol_prices)
        assert theta_high > 0.0015

    def test_signal_operator_mutual_exclusivity(self):
        """Verifica S_t in {-1, 0, 1} y la exclusión mutua estricta."""
        strat = MarketIntradayMomentumMOC(params={"theta_floor": 0.0015})
        theta = 0.0020

        # Long: r_first > theta y phi_vol == 1
        assert strat.evaluate_signal(r_first=0.0025, phi_vol=1, theta=theta) == 1
        # Short: r_first < -theta y phi_vol == 1
        assert strat.evaluate_signal(r_first=-0.0025, phi_vol=1, theta=theta) == -1
        # Neutral por retorno dentro de banda [-theta, +theta]
        assert strat.evaluate_signal(r_first=0.0010, phi_vol=1, theta=theta) == 0
        assert strat.evaluate_signal(r_first=-0.0010, phi_vol=1, theta=theta) == 0
        # Neutral por falta de volumen a pesar de fuerte movimiento
        assert strat.evaluate_signal(r_first=0.0050, phi_vol=0, theta=theta) == 0
        assert strat.evaluate_signal(r_first=-0.0050, phi_vol=0, theta=theta) == 0

    def test_exit_levels_sl_and_tp(self):
        """Verifica niveles de Stop Loss y Take Profit con asimetría k_RR = 1.50."""
        strat = MarketIntradayMomentumMOC(params={
            "lambda_sl": 0.0035,
            "k_rr": 1.50
        })
        p_entry = 5000.0

        # Long: SL por debajo, TP por encima
        p_sl_long, p_tp_long = strat.calculate_exit_levels(p_entry, direction=1)
        assert p_sl_long == 5000.0 * (1.0 - 0.0035)  # 4982.50
        assert p_tp_long == 5000.0 * (1.0 + 1.5 * 0.0035)  # 5026.25
        # Comprobar ratio asimétrico exacto
        rr_long = (p_tp_long - p_entry) / (p_entry - p_sl_long)
        assert pytest.approx(rr_long, abs=1e-6) == 1.50

        # Short: SL por encima, TP por debajo
        p_sl_short, p_tp_short = strat.calculate_exit_levels(p_entry, direction=-1)
        assert p_sl_short == 5000.0 * (1.0 + 0.0035)  # 5017.50
        assert p_tp_short == 5000.0 * (1.0 - 1.5 * 0.0035)  # 4973.75
        rr_short = (p_entry - p_tp_short) / (p_sl_short - p_entry)
        assert pytest.approx(rr_short, abs=1e-6) == 1.50


# ==============================================================================
# 2. PRUEBAS DE GESTIÓN DE RIESGO Y CONTROL PROP TRADING (Sección 5 de F1)
# ==============================================================================

class TestF1PropTradingRiskRules:
    """Valida presupuesto dinámico, conmutación de contratos y disyuntor de volatilidad."""

    def test_risk_budget_and_consecutive_losses_attenuation(self):
        """Verifica phi(K_loss) = 0.35 * 2^(-min(K_loss, 3))."""
        strat = MarketIntradayMomentumMOC(params={"psi": 0.25})
        dll = 1000.0
        mtdd = 2000.0

        # K_loss = 0 -> phi = 0.35 -> b_daily = 350.0, b_trailing = 0.25 * 2000 = 500 -> min = 350.0
        b_0 = strat.calculate_risk_budget(dll, mtdd, dd_current=0.0, l_day=0.0, k_loss=0)
        assert pytest.approx(b_0, abs=1e-5) == 350.0

        # K_loss = 1 -> phi = 0.175 -> b_daily = 175.0
        b_1 = strat.calculate_risk_budget(dll, mtdd, dd_current=0.0, l_day=0.0, k_loss=1)
        assert pytest.approx(b_1, abs=1e-5) == 175.0

        # K_loss = 2 -> phi = 0.0875 -> b_daily = 87.5
        b_2 = strat.calculate_risk_budget(dll, mtdd, dd_current=0.0, l_day=0.0, k_loss=2)
        assert pytest.approx(b_2, abs=1e-5) == 87.5

        # K_loss >= 3 -> phi = 0.04375 -> b_daily = 43.75
        b_3 = strat.calculate_risk_budget(dll, mtdd, dd_current=0.0, l_day=0.0, k_loss=3)
        b_5 = strat.calculate_risk_budget(dll, mtdd, dd_current=0.0, l_day=0.0, k_loss=5)
        assert pytest.approx(b_3, abs=1e-5) == 43.75
        assert pytest.approx(b_5, abs=1e-5) == 43.75

        # Si el trailing drawdown está muy cerca del límite, el presupuesto se acota por MTDD
        b_near_dd = strat.calculate_risk_budget(dll, mtdd, dd_current=1900.0, l_day=0.0, k_loss=0)
        # Margen MTDD restante = 100 -> psi * 100 = 0.25 * 100 = 25.0 < 350.0
        assert pytest.approx(b_near_dd, abs=1e-5) == 25.0

    def test_contract_switching_standard_to_micro(self):
        """Verifica cambio automático a Micro cuando el presupuesto no alcanza para Estándar."""
        strat = MarketIntradayMomentumMOC(params={
            "std_symbol": "ES",
            "micro_symbol": "MES",
            "lambda_sl": 0.0035,
        })
        p_entry = 5000.0

        # Riesgo ES a 5000 con lambda_sl 0.0035 (17.5 pts * 50 = $875 + $29.50 fricción = $904.50)
        # Riesgo MES a 5000 (17.5 pts * 5 = $87.5 + $3.74 fricción = $91.24)

        # Caso A: Presupuesto amplio ($2000) -> ejecuta contratos Estándar ES
        sizing_std = strat.calculate_position_sizing(b_risk=2000.0, p_entry=p_entry)
        assert sizing_std["contract_type"] == "STD"
        assert sizing_std["symbol"] == "ES"
        assert sizing_std["contracts"] == int(2000.0 // 904.50)  # 2 contratos

        # Caso B: Presupuesto moderado ($350) -> conmuta automáticamente a Micro MES
        sizing_micro = strat.calculate_position_sizing(b_risk=350.0, p_entry=p_entry)
        assert sizing_micro["contract_type"] == "MICRO"
        assert sizing_micro["symbol"] == "MES"
        assert sizing_micro["contracts"] == int(350.0 // 91.24)  # 3 contratos
        assert sizing_micro["contracts"] <= 10

    def test_volatility_circuit_breaker(self):
        """Verifica activación de HALT_VOLATILITY_DEFENSE cuando el riesgo excede el presupuesto."""
        strat = MarketIntradayMomentumMOC(params={
            "std_symbol": "ES",
            "micro_symbol": "MES",
            "lambda_sl": 0.0035,
        })
        p_entry = 5000.0
        # Riesgo MES es ~91.24. Si el presupuesto disponible es solo $50:
        sizing_halt = strat.calculate_position_sizing(b_risk=50.0, p_entry=p_entry)
        assert sizing_halt["contract_type"] == "NONE"
        assert sizing_halt["contracts"] == 0
        assert sizing_halt["status"] == "HALT_VOLATILITY_DEFENSE"


# ==============================================================================
# 3. PRUEBAS DE SIMULACIÓN Y BACKTEST SESIÓN A SESIÓN
# ==============================================================================

class TestF1IntradaySimulationAndBacktester:
    """Verifica el motor de backtesting completo con datos sintéticos y reales."""

    @staticmethod
    def _create_synthetic_day(
        date_str: str,
        morning_move_pct: float,
        intraday_outcome: str = "TP",  # "TP", "SL", "MOC"
        volume: float = 10000.0,
        base_price: float = 5000.0
    ) -> pd.DataFrame:
        """Genera un día completo de barras de 5 minutos en America/New_York."""
        times = pd.date_range(f"{date_str} 09:30:00", f"{date_str} 16:00:00", freq="5min", tz="America/New_York")
        df = pd.DataFrame(index=times, columns=["Open", "High", "Low", "Close", "Volume"])

        p_930 = base_price
        p_1000 = p_930 * (1.0 + morning_move_pct)

        # Generar barras matutinas [09:30, 10:00) EST (exactamente 6 barras en 5m)
        morning_idx = df.between_time("09:30", "10:00", inclusive="left").index
        n_m = len(morning_idx)
        m_prices = np.linspace(p_930, p_1000, n_m)

        for i, ts in enumerate(morning_idx):
            p = m_prices[i]
            df.loc[ts, "Open"] = p
            df.loc[ts, "High"] = p + 1.0
            df.loc[ts, "Low"] = p - 1.0
            df.loc[ts, "Close"] = p
            df.loc[ts, "Volume"] = volume / n_m

        # Ventana intermedia (10:00 a 15:25): letargo
        idle_idx = df.between_time("10:00", "15:25").index
        for ts in idle_idx:
            df.loc[ts, "Open"] = p_1000
            df.loc[ts, "High"] = p_1000 + 1.0
            df.loc[ts, "Low"] = p_1000 - 1.0
            df.loc[ts, "Close"] = p_1000
            df.loc[ts, "Volume"] = 100.0

        # Ventana de ejecución (15:30 a 15:58)
        p_entry = p_1000
        exec_idx = df.between_time("15:30", "15:58", inclusive="left").index
        lambda_sl = 0.0035
        k_rr = 1.5

        for i, ts in enumerate(exec_idx):
            df.loc[ts, "Volume"] = 5000.0
            if i == 0:
                # Barra de entrada a las 15:30: cotiza en el precio de entrada sin vulnerar stops
                df.loc[ts, "Open"] = p_entry
                df.loc[ts, "High"] = p_entry + 0.5
                df.loc[ts, "Low"] = p_entry - 0.5
                df.loc[ts, "Close"] = p_entry
                continue

            if intraday_outcome == "TP":
                # Sube por encima de Take Profit
                df.loc[ts, "Open"] = p_entry
                df.loc[ts, "High"] = p_entry * (1.0 + 2.0 * lambda_sl * k_rr)
                df.loc[ts, "Low"] = p_entry - 0.5
                df.loc[ts, "Close"] = p_entry * (1.0 + 1.5 * lambda_sl * k_rr)
            elif intraday_outcome == "SL":
                # Cae por debajo de Stop Loss
                df.loc[ts, "Open"] = p_entry
                df.loc[ts, "High"] = p_entry + 0.5
                df.loc[ts, "Low"] = p_entry * (1.0 - 2.0 * lambda_sl)
                df.loc[ts, "Close"] = p_entry * (1.0 - 1.5 * lambda_sl)
            elif intraday_outcome == "GAP_DOWN_DLL":
                # Gap bajista extremo que perfora DLL
                df.loc[ts, "Open"] = p_entry * 0.96
                df.loc[ts, "High"] = p_entry * 0.965
                df.loc[ts, "Low"] = p_entry * 0.95
                df.loc[ts, "Close"] = p_entry * 0.955
            else:  # "MOC"
                # Fluctuación dentro del rango SL-TP hasta el cierre
                df.loc[ts, "Open"] = p_entry
                df.loc[ts, "High"] = p_entry + 2.0
                df.loc[ts, "Low"] = p_entry - 2.0
                df.loc[ts, "Close"] = p_entry + 1.0

        return df.astype(float)


    def test_backtest_winning_trade_tp(self):
        """Verifica ejecución de trade ganador por Take Profit."""
        day_df = self._create_synthetic_day("2026-10-15", morning_move_pct=+0.0050, intraday_outcome="TP")
        backtester = IntradayMomentumBacktester(account_size=50000.0)
        results = backtester.run(day_df)

        trades = results["trades"]
        assert len(trades) == 1
        trade = trades.iloc[0]
        assert trade["signal"] == "LONG"
        assert trade["exit_reason"] == "TP"
        assert trade["net_pnl"] > 0
        assert results["metrics"]["Winning Trades"] == 1
        assert results["metrics"]["Net Profit ($)"] > 0

    def test_backtest_losing_trade_sl(self):
        """Verifica ejecución de trade perdedor por Stop Loss y actualización de K_loss."""
        day_df = self._create_synthetic_day("2026-10-15", morning_move_pct=+0.0050, intraday_outcome="SL")
        backtester = IntradayMomentumBacktester(account_size=50000.0)
        results = backtester.run(day_df)

        trades = results["trades"]
        assert len(trades) == 1
        trade = trades.iloc[0]
        assert trade["exit_reason"] == "SL"
        assert trade["net_pnl"] < 0
        assert trade["k_loss"] == 1
        assert results["metrics"]["Losing Trades"] == 1

    def test_backtest_moc_time_decay_exit(self):
        """Verifica cierre forzoso por tiempo a las 15:58 / MOC."""
        day_df = self._create_synthetic_day("2026-10-15", morning_move_pct=+0.0050, intraday_outcome="MOC")
        backtester = IntradayMomentumBacktester(account_size=50000.0)
        results = backtester.run(day_df)

        trades = results["trades"]
        assert len(trades) == 1
        trade = trades.iloc[0]
        assert trade["exit_reason"] == "MOC"

    def test_backtest_daily_loss_limit_protection(self):
        """Verifica que un evento adverso extremo que vulnere el límite diario detenga la cuenta y reporte FAIL."""
        day_df = self._create_synthetic_day("2026-10-15", morning_move_pct=+0.0050, intraday_outcome="GAP_DOWN_DLL")
        backtester = IntradayMomentumBacktester(
            account_size=50000.0,
            daily_loss_limit=1000.0,
            max_trailing_loss=5000.0
        )
        results = backtester.run(day_df)
        prop_eval = results["prop_evaluation"]
        assert prop_eval["status"] == "FAIL"
        assert "Daily Loss Limit" in prop_eval["reason"]



    def test_backtest_real_es_data(self):
        """Verifica que el backtester corra sin errores sobre los datos históricos reales de ES 5m."""
        data_path = "data/raw/CME_MINI_ES_5m_bars5000.parquet"
        df = pd.read_parquet(data_path)
        # Los datos brutos están en hora local UTC-6 (America/Mexico_City)
        strat = MarketIntradayMomentumMOC(params={"source_tz": "America/Mexico_City"})
        backtester = IntradayMomentumBacktester(
            strategy=strat,
            account_size=50000.0,
            daily_loss_limit=1000.0,
            max_trailing_loss=2000.0
        )

        results = backtester.run(df)
        assert "metrics" in results
        assert "prop_evaluation" in results
        assert "trades" in results
        assert isinstance(results["trades"], pd.DataFrame)
        assert results["prop_evaluation"]["final_balance"] > 0

    def test_backtest_real_nq_data(self):
        """Verifica que el backtester corra sin errores sobre los datos históricos reales de NQ 5m."""
        data_path = "data/raw/CME_MINI_NQ_5m_bars5000.parquet"
        df = pd.read_parquet(data_path)
        strat = MarketIntradayMomentumMOC(params={
            "symbol": "NQ",
            "std_symbol": "NQ",
            "micro_symbol": "MNQ",
            "source_tz": "America/Mexico_City"
        })
        backtester = IntradayMomentumBacktester(
            strategy=strat,
            account_size=50000.0,
            daily_loss_limit=1000.0,
            max_trailing_loss=2000.0
        )

        results = backtester.run(df)
        assert "metrics" in results
        assert "trades" in results
        assert isinstance(results["trades"], pd.DataFrame)

    def test_signal_series_alignment_with_tz_naive_dataframe(self):
        """Verifica que generate_signals() sobre un DataFrame naive retorne un índice idéntico sin NaNs."""
        data_path = "data/raw/CME_MINI_ES_5m_bars5000.parquet"
        df = pd.read_parquet(data_path)
        assert df.index.tz is None  # Asegurar que es naive

        strat = MarketIntradayMomentumMOC(params={"source_tz": "America/Mexico_City"})
        sig = strat.generate_signals(df)

        assert len(sig) == len(df)
        assert sig.index.equals(df.index)

        # Asignación directa a columna de DataFrame no debe producir NaNs
        df["Signal"] = sig
        assert df["Signal"].isna().sum() == 0

    def test_half_day_session_cme_holiday_no_trade(self):
        """Verifica que en una sesión abreviada (cierre 13:00 EST), no se abran operaciones al faltar W_exec."""
        times = pd.date_range("2026-11-27 09:30:00", "2026-11-27 13:00:00", freq="5min", tz="America/New_York")
        df = pd.DataFrame(index=times, columns=["Open", "High", "Low", "Close", "Volume"])
        df["Open"] = 5000.0
        df["High"] = 5020.0
        df["Low"] = 4995.0
        df["Close"] = 5015.0
        df["Volume"] = 10000.0
        df = df.astype(float)

        backtester = IntradayMomentumBacktester(account_size=50000.0)
        results = backtester.run(df)

        assert len(results["trades"]) == 0
        assert results["prop_evaluation"]["status"] == "INCOMPLETE"
        assert results["prop_evaluation"]["final_balance"] == 50000.0

    def test_profile_without_daily_loss_limit_lucid(self):
        """Verifica que una firma sin límite de pérdida diaria (DLL=None, ej. Lucid 50K) funcione sin error."""
        day_df = self._create_synthetic_day("2026-10-15", morning_move_pct=+0.0050, intraday_outcome="TP")
        backtester = IntradayMomentumBacktester(
            account_size=50000.0,
            daily_loss_limit=None,
            max_trailing_loss=2000.0
        )
        results = backtester.run(day_df)
        assert results["prop_evaluation"]["status"] in ("PASS", "INCOMPLETE")
        assert len(results["trades"]) == 1

    def test_max_drawdown_includes_initial_capital(self):
        """Verifica que una racha perdedora inicial registre el Max Drawdown desde el capital inicial."""
        day_df = self._create_synthetic_day("2026-10-15", morning_move_pct=+0.0050, intraday_outcome="SL")
        backtester = IntradayMomentumBacktester(account_size=50000.0)
        results = backtester.run(day_df)

        metrics = results["metrics"]
        assert metrics["Losing Trades"] == 1
        assert metrics["Max Drawdown ($)"] > 0
        # El Max DD debe coincidir con el valor absoluto del trade perdedor inicial
        trade_loss = abs(results["trades"].iloc[0]["net_pnl"])
        assert pytest.approx(metrics["Max Drawdown ($)"], abs=1e-2) == trade_loss

    def test_morning_interval_exclusion_of_1000_bar(self):
        """Verifica formalmente que la vela de 10:00:00 NO forma parte de W_obs = [09:30, 10:00)."""
        times = pd.date_range("2026-10-15 09:30:00", "2026-10-15 10:10:00", freq="5min", tz="America/New_York")
        df = pd.DataFrame(index=times, columns=["Open", "High", "Low", "Close", "Volume"])
        df["Open"] = 5000.0
        df["High"] = 5010.0
        df["Low"] = 4990.0
        df["Close"] = 5005.0
        df["Volume"] = 1000.0
        df = df.astype(float)

        # Inyectar anomalía extrema en la barra de 10:00:00
        bar_1000 = pd.Timestamp("2026-10-15 10:00:00", tz="America/New_York")
        df.loc[bar_1000, "Close"] = 6000.0
        df.loc[bar_1000, "Volume"] = 999999.0

        strat = MarketIntradayMomentumMOC()
        df_ny = strat.ensure_timezone(df)
        obs_bars = df_ny.between_time(strat.params["obs_start"], strat.params["obs_end"], inclusive="left")

        # La barra de 10:00:00 NO debe estar presente
        assert bar_1000 not in obs_bars.index
        assert float(obs_bars["Close"].iloc[-1]) == 5005.0  # Close de 09:55
        assert float(obs_bars["Volume"].sum()) == 6000.0    # 6 barras de 1000, sin los 999999
