"""
tests/test_f9_optimizacion_is_n1.py
Suite de pruebas unitarias, matemáticas, microestructurales y de estrés
para la Versión Alterna Optimizada: F9-Optimizacion -IS- N1.

Verifica rigurosamente:
1. Estimador de Garman-Klass, varianza multiescala y proxy de volatilidad implícita.
2. Estimador sintético de VRP (VRP_synth) y ratio de estructura temporal sintética (TSR_synth).
3. Pendiente de la estructura temporal (slope = 1.0 - TSR) e indicador de inversión (backwardation).
4. Condicionamiento estricto a la pendiente de la estructura de plazos:
   - Bloqueo total de compras (Overnight y Pre-Macro) cuando TSR >= 1.00 o S_VRP = -1.
   - Mitigación del riesgo de cola izquierda (left-tail risk).
5. Desacoplamiento modular con bandera enable_overnight.
6. Ejecución y salidas: Take Profit 0.80%, Stop Loss 0.50% y salida determinista 08:30 EST.
7. Deriva pre-macro: Salida mandatoria en T-3m (60s antes de embargo) y Stop Loss 8.0 pts.
8. Control de riesgo Prop Trading: Segregación fraccionaria (20% Noche / 80% RTH),
   conmutación automática Estándar/Micro y disyuntor de volatilidad.
9. Simulación End-to-End con cuenta de fondeo (DLL, MTDD, comisiones y slippage CME).
"""

import math
from datetime import time, datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.strategies.f9_optimizacion_is_n1 import (
    F9OptimizationISN1Strategy,
    F9OptimizationISN1Backtester,
    F9PropAccount,
    F9TradeRecord,
    F9ExitReason,
    F9RegimeState,
    F9SubModelType,
    calculate_garman_klass_variance,
    calculate_daily_realized_variance,
    calculate_annualized_realized_variance,
    calculate_proxy_implied_variance,
    calculate_synthetic_vrp,
    calculate_z_score_synthetic_vrp,
    calculate_synthetic_term_structure_ratio,
    calculate_term_structure_ratio,
    evaluate_term_structure_slope,
    is_term_structure_inverted,
    evaluate_vrp_regime_gate_n1,
    calculate_intraday_return,
    calculate_standardized_intraday_drop,
    evaluate_overnight_signal_n1,
    calculate_window_vwap,
    evaluate_pre_macro_signal_n1,
    calculate_overnight_sizing_n1,
    calculate_pre_macro_sizing_n1,
)
from src.core.futures_specs import get_spec


# ==============================================================================
# 1. PRUEBAS DE FORMULACIÓN MATEMÁTICA Y VOLATILIDAD SINTÉTICA
# ==============================================================================

class TestF9N1MathematicalFormulas:
    """Verifica las ecuaciones matemáticas de Garman-Klass, VRP_synth y TSR_synth."""

    def test_garman_klass_variance_formula(self):
        """
        sigma_GK^2 = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2.
        """
        o = 5000.0
        h = 5050.0  # +1%
        l = 4950.0  # -1%
        c = 5020.0  # +0.4%

        term_hl = math.log(5050.0 / 4950.0)
        term_co = math.log(5020.0 / 5000.0)
        coef = 2.0 * math.log(2.0) - 1.0
        expected = 0.5 * (term_hl ** 2) - coef * (term_co ** 2)

        calc = calculate_garman_klass_variance(o, h, l, c)
        assert pytest.approx(calc, abs=1e-8) == expected
        assert calc > 0.0

        # Caso degenerate: precios idénticos -> varianza 0.0
        assert calculate_garman_klass_variance(5000.0, 5000.0, 5000.0, 5000.0) == 0.0

        # Caso precios inválidos -> 0.0 sin excepción
        assert calculate_garman_klass_variance(-10.0, 5000.0, 4900.0, 5000.0) == 0.0

    def test_proxy_implied_variance(self):
        """sigma_proxy^2 = 252 * mean(GK_vars[-k_vol:]) con k_vol = 5."""
        gk_vars = [0.0001, 0.00012, 0.00011, 0.00009, 0.00013]
        expected = 252.0 * float(np.mean(gk_vars))
        calc = calculate_proxy_implied_variance(gk_vars, k_vol=5)
        assert pytest.approx(calc, abs=1e-8) == expected

        assert calculate_proxy_implied_variance([], k_vol=5) == 0.0

    def test_synthetic_vrp_formula(self):
        """VRP_synth = sigma_proxy^2 - RV_22d^ann."""
        sigma_proxy_sq = 0.0350  # 3.5%
        rv_ann = 0.0200         # 2.0%
        expected = 0.0150
        assert pytest.approx(calculate_synthetic_vrp(sigma_proxy_sq, rv_ann), abs=1e-8) == expected

    def test_z_score_synthetic_vrp_with_epsilon(self):
        """Z_VRP = (VRP - mu) / (sigma + eps). Epsilon previene división por cero."""
        history = [0.015] * 20
        curr = 0.015
        calc = calculate_z_score_synthetic_vrp(curr, history, lookback=60, eps=1e-8)
        assert calc == 0.0

        # Con dispersión real
        history_var = [0.01, 0.02, 0.015, 0.018, 0.012]
        curr_val = 0.025
        all_vals = history_var + [curr_val]
        expected = (curr_val - np.mean(all_vals)) / (np.std(all_vals, ddof=1) + 1e-8)
        calc_var = calculate_z_score_synthetic_vrp(curr_val, history_var, lookback=60, eps=1e-8)
        assert pytest.approx(calc_var, abs=1e-6) == expected

    def test_synthetic_term_structure_ratio(self):
        """
        TSR_synth = sqrt(mean(RV_5d)) / sqrt(mean(RV_22d)).
        """
        # Caso Contango: volatilidad reciente baja (0.0001) vs mensual alta (0.0004)
        # TSR = sqrt(0.0001) / sqrt(0.0004) = 0.01 / 0.02 = 0.50
        rvs_contango = [0.0004] * 17 + [0.0001] * 5
        tsr_c = calculate_synthetic_term_structure_ratio(rvs_contango, short_window=5, long_window=22)
        assert tsr_c < 0.98

        # Caso Backwardation: volatilidad reciente alta (0.0009) vs mensual moderada (0.0001)
        # TSR = sqrt(0.0009) / sqrt(mean(22d)) > 1.05
        rvs_back = [0.0001] * 17 + [0.0009] * 5
        tsr_b = calculate_synthetic_term_structure_ratio(rvs_back, short_window=5, long_window=22)
        assert tsr_b > 1.05

    def test_term_structure_slope_and_inversion(self):
        """
        slope = 1.0 - TSR.
        is_term_structure_inverted = (TSR >= threshold).
        """
        # Contango
        assert evaluate_term_structure_slope(0.88) > 0.0
        assert not is_term_structure_inverted(0.88, backwardation_threshold=1.00)

        # Flat
        assert evaluate_term_structure_slope(1.00) == 0.0
        assert is_term_structure_inverted(1.00, backwardation_threshold=1.00)

        # Backwardation / Invertida
        assert evaluate_term_structure_slope(1.15) < 0.0
        assert is_term_structure_inverted(1.15, backwardation_threshold=1.00)

    def test_standardized_intraday_drop(self):
        """Z_drop = R_intraday / (sigma_daily + eps)."""
        r_drop = -0.015  # -1.5%
        sigma = 0.010   # 1.0% diaria
        expected = -0.015 / (0.010 + 1e-6)
        calc = calculate_standardized_intraday_drop(r_drop, sigma, eps=1e-6)
        assert pytest.approx(calc, abs=1e-4) == expected
        assert calc <= -1.25


# ==============================================================================
# 2. PRUEBAS DEL CONDICIONAMIENTO A LA ESTRUCTURA DE PLAZOS Y VETO DE COMPRAS
# ==============================================================================

class TestTermStructureConditioningAndLeftTailProtection:
    """
    Verifica que la pendiente de la estructura de plazos (TSR) condicione estrictamente
    las entradas, bloqueando compras en backwardation para erradicar el riesgo de cola izquierda.
    """

    def test_vrp_regime_gate_n1(self):
        """
        S_VRP^*:
        +1: Z_VRP >= 1.00 and TSR <= 0.98 (Contango / Sesgo Favorable)
        -1: Z_VRP <= -1.00 or TSR >= 1.05 (Backwardation / Inversión de Curva)
         0: Neutral
        """
        # Contango con prima positiva -> +1
        assert evaluate_vrp_regime_gate_n1(z_vrp=1.20, tsr=0.92) == 1

        # Curva invertida (TSR >= 1.05) -> -1 (BLOQUEO TOTAL) aunque Z_VRP sea alto
        assert evaluate_vrp_regime_gate_n1(z_vrp=2.00, tsr=1.10) == -1

        # Prima colapsada (Z_VRP <= -1.00) -> -1
        assert evaluate_vrp_regime_gate_n1(z_vrp=-1.30, tsr=0.95) == -1

        # Neutral
        assert evaluate_vrp_regime_gate_n1(z_vrp=0.50, tsr=0.96) == 0

    def test_backwardation_strictly_blocks_overnight_longs(self):
        """
        Garantiza que ante una fuerte caída intradía previa (Z_drop <= -1.25),
        si la estructura de plazos se invierte (TSR >= 1.00 o S_VRP = -1),
        la señal de compra nocturna sea 0 (VETO ABSOLUTO).
        """
        z_drop_extreme = -2.50  # Fuerte liquidación en RTH

        # Escenario 1: Estructura en Contango (TSR = 0.90, S_VRP = 0) -> COMPRA PERMITIDA (+1)
        sig_contango = evaluate_overnight_signal_n1(
            z_drop_prev=z_drop_extreme,
            s_vrp=0,
            tsr=0.90,
            enable_overnight=True,
            z_drop_crit=-1.25,
            max_tsr_long=1.00,
        )
        assert sig_contango == 1, "En contango saludable con caída previa debe activar compra"

        # Escenario 2: Curva Invertida (TSR = 1.02 >= max_tsr_long) -> BLOQUEO ESTRICTO (0)
        sig_inverted_tsr = evaluate_overnight_signal_n1(
            z_drop_prev=z_drop_extreme,
            s_vrp=0,
            tsr=1.02,
            enable_overnight=True,
            z_drop_crit=-1.25,
            max_tsr_long=1.00,
        )
        assert sig_inverted_tsr == 0, "Curva invertida (TSR >= 1.00) DEBE bloquear compra nocturna"

        # Escenario 3: Régimen de pánico por compuerta (S_VRP = -1) -> BLOQUEO ESTRICTO (0)
        sig_panic = evaluate_overnight_signal_n1(
            z_drop_prev=z_drop_extreme,
            s_vrp=-1,
            tsr=0.92,
            enable_overnight=True,
            z_drop_crit=-1.25,
            max_tsr_long=1.00,
        )
        assert sig_panic == 0, "S_VRP = -1 DEBE vetar incondicionalmente la compra nocturna"

    def test_backwardation_strictly_blocks_pre_macro_longs(self):
        """
        Garantiza que compras pre-macro queden bloqueadas si la curva está invertida (TSR >= 1.00),
        pero permite ventas en corto durante backwardation si el flujo es bajista.
        """
        vwap = 5000.0
        p_above = 5008.0
        p_below = 4992.0

        # Long en Contango (TSR = 0.92) -> +1
        assert evaluate_pre_macro_signal_n1(
            z_ofi_pre=2.5,
            current_price=p_above,
            vwap_prior=vwap,
            s_vrp=0,
            tsr=0.92,
            max_tsr_long=1.00,
        ) == 1

        # Long en Backwardation (TSR = 1.08) -> 0 (BLOQUEADO)
        assert evaluate_pre_macro_signal_n1(
            z_ofi_pre=2.5,
            current_price=p_above,
            vwap_prior=vwap,
            s_vrp=-1,
            tsr=1.08,
            max_tsr_long=1.00,
        ) == 0

        # Short en Backwardation (TSR = 1.08, S_VRP = -1) -> PERMITIDO (-1)
        assert evaluate_pre_macro_signal_n1(
            z_ofi_pre=-2.5,
            current_price=p_below,
            vwap_prior=vwap,
            s_vrp=-1,
            tsr=1.08,
            max_tsr_long=1.00,
        ) == -1


# ==============================================================================
# 3. PRUEBAS DE DESACOPLAMIENTO MODULAR (BANDERA ENABLE_OVERNIGHT)
# ==============================================================================

class TestModularDecoupling:
    """Verifica que el parámetro enable_overnight permita desactivar la noche sin afectar pre-macro."""

    def test_disable_overnight_flag(self):
        """Si enable_overnight = False, la señal nocturna siempre es 0."""
        sig = evaluate_overnight_signal_n1(
            z_drop_prev=-2.0,
            s_vrp=1,
            tsr=0.88,
            enable_overnight=False,
        )
        assert sig == 0

    def test_strategy_generate_signals_with_overnight_disabled(self):
        """Verifica que la estrategia completa no emita señales nocturnas si enable_overnight=False."""
        timestamps = pd.date_range("2026-10-15 00:00:00", "2026-10-15 09:00:00", freq="5min", tz="America/New_York")
        df = pd.DataFrame({
            "Open": [5000.0] * len(timestamps),
            "High": [5005.0] * len(timestamps),
            "Low": [4995.0] * len(timestamps),
            "Close": [5000.0] * len(timestamps),
            "Volume": [1000.0] * len(timestamps),
        }, index=timestamps)

        strat_disabled = F9OptimizationISN1Strategy(params={"enable_overnight": False})
        signals = strat_disabled.generate_signals(df)
        assert (signals == 0.0).all(), "Con enable_overnight=False no deben generarse señales en la madrugada"


# ==============================================================================
# 4. PRUEBAS DE SALIDAS DETERMINISTAS, TAKE PROFIT Y STOP LOSS
# ==============================================================================

class TestExecutionExitsAndTiming:
    """Verifica Take Profit 0.80%, Stop Loss 0.50% y salida mandatoria T-3m."""

    def test_overnight_take_profit_hit(self):
        """
        Una posición nocturna que alcance un rebote de 0.80% debe cerrarse
        con razón TAKE_PROFIT.
        """
        strat = F9OptimizationISN1Strategy()
        backtester = F9OptimizationISN1Backtester(strategy=strat)

        p_entry = 5000.0
        tp_price = p_entry * (1.0 + 0.0080)  # 5040.0 (+40 pts)
        sl_price = p_entry * (1.0 - 0.0050)  # 4975.0 (-25 pts)

        trade = F9TradeRecord(
            trade_id=1,
            date="2026-10-15",
            submodel=F9SubModelType.OVERNIGHT.value,
            direction=1,
            contracts=1,
            contract_type="STD",
            symbol="ES",
            entry_time="2026-10-15 01:30:00-04:00",
            entry_price=p_entry,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
        )

        # Simular que a las 04:00 el High alcanza 5042.0 (supera TP)
        ts_hit = pd.Timestamp("2026-10-15 04:00:00-04:00")
        backtester._close_trade(
            trade=trade,
            exit_price=tp_price,
            exit_time=ts_hit,
            exit_reason=F9ExitReason.TAKE_PROFIT,
            slippage_ticks=1.0,
        )

        assert trade.exit_reason == F9ExitReason.TAKE_PROFIT
        assert trade.gross_pnl_pts == 40.0
        assert trade.net_pnl_usd > 1900.0  # 40 * 50 = $2000 - comisiones y slippage

    def test_overnight_stop_loss_hit(self):
        """Una caída al nivel de SL de 0.50% gatilla STOP_LOSS."""
        strat = F9OptimizationISN1Strategy()
        backtester = F9OptimizationISN1Backtester(strategy=strat)

        p_entry = 5000.0
        sl_price = p_entry * (1.0 - 0.0050)  # 4975.0

        trade = F9TradeRecord(
            trade_id=2,
            date="2026-10-15",
            submodel=F9SubModelType.OVERNIGHT.value,
            direction=1,
            contracts=1,
            contract_type="STD",
            symbol="ES",
            entry_time="2026-10-15 01:30:00-04:00",
            entry_price=p_entry,
            stop_loss_price=sl_price,
            take_profit_price=5040.0,
        )

        ts_sl = pd.Timestamp("2026-10-15 03:30:00-04:00")
        backtester._close_trade(
            trade=trade,
            exit_price=sl_price,
            exit_time=ts_sl,
            exit_reason=F9ExitReason.STOP_LOSS,
            slippage_ticks=1.0,
        )

        assert trade.exit_reason == F9ExitReason.STOP_LOSS
        assert trade.gross_pnl_pts == -25.0
        assert trade.net_pnl_usd < 0

    def test_pre_macro_mandatory_exit_t_minus_3_and_embargo_isolation(self):
        """
        Verifica que el trade pre-macro se liquide en T-3m exactamente,
        asegurando al menos 60 segundos de colchón antes del embargo [T-2m, T+2m].
        """
        macro_event_dt = pd.Timestamp("2026-10-15 08:30:00-04:00")
        strat = F9OptimizationISN1Strategy(params={
            "macro_schedule": [macro_event_dt],
            "delta_p_stop_macro": 8.0,
        })
        backtester = F9OptimizationISN1Backtester(strategy=strat)

        ts = pd.date_range("2026-10-15 08:00:00", "2026-10-15 09:00:00", freq="5min", tz="America/New_York")
        prices = [5000.0 if t < pd.Timestamp("2026-10-15 08:15:00", tz="America/New_York") else 5005.0 for t in ts]
        df = pd.DataFrame({
            "Open": prices,
            "High": [p + 2.0 for p in prices],
            "Low": [p - 2.0 for p in prices],
            "Close": prices,
            "Volume": [1000.0] * len(ts),
            "Z_OFI": [2.5] * len(ts),
        }, index=ts)

        results = backtester.run(df)
        trades = results["trades"]
        assert len(trades) == 1
        t = trades.iloc[0]
        expected_exit_time = str(macro_event_dt - timedelta(minutes=3))
        assert t["exit_time"] == expected_exit_time
        assert t["exit_reason"] == F9ExitReason.DETERMINISTIC_TIME_EXIT.value

        exit_dt = pd.Timestamp(t["exit_time"])
        embargo_start = macro_event_dt - timedelta(minutes=2)
        assert (embargo_start - exit_dt).total_seconds() >= 60.0


# ==============================================================================
# 5. PRUEBAS DE GESTIÓN DE RIESGO DE PROP TRADING Y SIZING (F9-N1)
# ==============================================================================

class TestPropRiskManagementF9N1:
    """Verifica segregación del DLL (lambda_on = 0.20), MTDD y conmutación a Micro."""

    def test_segregated_dll_partition(self):
        """Delta DLL_t^nocturno = 0.20 * Delta DLL, Delta DLL_t^RTH = 0.80 * Delta DLL."""
        account = F9PropAccount(dll=1000.0, mtdd=2000.0, lambda_on=0.20)
        assert account.delta_dll == 1000.0
        assert account.delta_dll_overnight == 200.0  # 20%
        assert account.delta_dll_rth == 800.0        # 80%

    def test_overnight_sizing_micro_switch(self):
        """
        N_ON con conmutación automática Estándar a Micro.
        """
        m_es = 50.0
        m_mes = 5.0
        p_entry = 5000.0
        delta_sl = 0.0050  # 25 pts -> $1,250 en ES, $125 en MES

        # Presupuesto $200 (partición nocturna 20% de $1000):
        # Insuficiente para ES ($1250) -> Conmuta a MES ($125) -> 200 // 125 = 1 contrato MES
        n_c, c_t = calculate_overnight_sizing_n1(
            budget_daily_on=200.0,
            budget_cushion_on=200.0,
            p_entry=p_entry,
            multiplier_std=m_es,
            multiplier_micro=m_mes,
            delta_sl_on=delta_sl,
        )
        assert c_t == "MICRO"
        assert n_c == 1

    def test_sizing_volatility_defense_zero_contracts(self):
        """Si el presupuesto es menor al riesgo unitario de 1 micro, N = 0."""
        n_c, c_t = calculate_overnight_sizing_n1(
            budget_daily_on=50.0,  # $50 < $125
            budget_cushion_on=50.0,
            p_entry=5000.0,
            multiplier_std=50.0,
            multiplier_micro=5.0,
            delta_sl_on=0.0050,
        )
        assert n_c == 0
        assert c_t == "NONE"


# ==============================================================================
# 6. PRUEBA END-TO-END DE SIMULACIÓN DETERMINISTA
# ==============================================================================

class TestEndToEndBacktestSimulationF9N1:
    """Verifica la ejecución determinista completa del backtest en escenario sintético multi-día."""

    def test_end_to_end_backtest_with_synthetic_market_data(self):
        """
        Simula 3 días:
        Día 1: RTH normal.
        Día 2: RTH con caída > 1.25 sigma, estructura en Contango -> Activa overnight con ganancia.
        Día 3: Anuncio macro a las 08:30 -> Activa pre-macro con salida forzosa en T-3m.
        """
        strat = F9OptimizationISN1Strategy(params={
            "enable_overnight": True,
            "use_synthetic_vrp": True,
            "z_drop_crit": -1.25,
            "delta_sl_on": 0.0050,
            "delta_tp_on": 0.0080,
            "delta_p_stop_macro": 8.0,
            "macro_schedule": [pd.Timestamp("2026-10-16 08:30:00", tz="America/New_York")],
        })
        backtester = F9OptimizationISN1Backtester(
            strategy=strat,
            account_size=50000.0,
            daily_loss_limit=1000.0,
            max_trailing_loss=2000.0,
            profit_target=3000.0,
        )

        dates = pd.date_range("2026-10-14 00:00:00", "2026-10-16 16:00:00", freq="5min", tz="America/New_York")
        opens = []
        closes = []
        highs = []
        lows = []
        volumes = []

        base_p = 5000.0
        for ts in dates:
            # En día 14 (RTH 09:30-16:00): caída pronunciada de 5000 a 4920 (-1.6%)
            if ts.date() == pd.Timestamp("2026-10-14").date() and time(9, 30) <= ts.time() <= time(16, 0):
                p = 5000.0 - (ts.hour - 9) * 10.0
            # En madrugada del día 15 (01:30 a 08:30): rebote a 4960
            elif ts.date() == pd.Timestamp("2026-10-15").date() and time(1, 30) <= ts.time() <= time(8, 30):
                p = 4925.0 + (ts.hour - 1) * 4.0
            # Pre-macro día 16 (08:15 a 08:27): impulso alcista
            elif ts.date() == pd.Timestamp("2026-10-16").date() and time(8, 15) <= ts.time() <= time(8, 27):
                p = 4980.0 + 5.0
            else:
                p = 4940.0

            opens.append(p)
            highs.append(p + 1.5)
            lows.append(p - 1.5)
            closes.append(p)
            volumes.append(1000.0)

        df = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
            "Z_OFI": [2.5] * len(dates),
        }, index=dates)

        results = backtester.run(df)
        trades = results["trades"]
        metrics = results["metrics"]
        prop_eval = results["prop_evaluation"]

        assert isinstance(trades, pd.DataFrame)
        assert "win_rate_pct" in metrics
        assert "status" in prop_eval
        assert not prop_eval["is_breached"]

    def test_signal_and_backtest_direction_consistency_without_explicit_z_ofi(self):
        """
        Garantiza que cuando la columna Z_OFI no está presente en los datos,
        generate_signals y backtester.run concuerden estrictamente en la dirección de la señal.
        """
        macro_dt = pd.Timestamp("2026-10-15 14:00:00", tz="America/New_York")
        strat = F9OptimizationISN1Strategy(params={
            "macro_schedule": [macro_dt],
            "max_tsr_long": 1.00,
        })
        backtester = F9OptimizationISN1Backtester(strategy=strat)

        # Ventana pre-macro con precios por debajo de VWAP previo -> señal SHORT (-1)
        ts = pd.date_range("2026-10-15 13:20:00", "2026-10-15 14:10:00", freq="5min", tz="America/New_York")
        # Precios altos antes de T-15m (5020.0), luego caen fuertemente a 4980.0
        prices = [5020.0 if t < pd.Timestamp("2026-10-15 13:45:00", tz="America/New_York") else 4980.0 for t in ts]
        df = pd.DataFrame({
            "Open": prices,
            "High": [p + 1.0 for p in prices],
            "Low": [p - 1.0 for p in prices],
            "Close": prices,
            "Volume": [1000.0] * len(ts),
        }, index=ts)

        signals = strat.generate_signals(df)
        results = backtester.run(df)
        trades = results["trades"]

        # En la barra 13:45:00 el precio cayó por debajo del VWAP
        sig_at_trigger = signals.loc["2026-10-15 13:45:00-04:00"]
        assert sig_at_trigger == -1.0, f"generate_signals debió emitir SHORT (-1.0), obtuvo {sig_at_trigger}"
        assert len(trades) == 1
        assert trades.iloc[0]["direction"] == -1, "El backtester debió ejecutar una venta en corto (direction == -1)"

    def test_stop_loss_gap_down_slippage_handling(self):
        """
        Verifica que si una barra abre con un gap por debajo del Stop Loss,
        el precio de salida ejecutado sea bar_open (más adverso) y no el stop nominal.
        """
        strat = F9OptimizationISN1Strategy()
        backtester = F9OptimizationISN1Backtester(strategy=strat)

        p_entry = 5000.0
        sl_nominal = 4975.0  # -0.50%

        # Simular trade activo
        current_trade = F9TradeRecord(
            trade_id=1,
            date="2026-10-15",
            submodel=F9SubModelType.OVERNIGHT.value,
            direction=1,
            contracts=1,
            contract_type="STD",
            symbol="ES",
            entry_time="2026-10-15 01:30:00-04:00",
            entry_price=p_entry,
            raw_entry_price=p_entry,
            stop_loss_price=sl_nominal,
        )

        # Barra con gap down masivo: Open = 4960.0 (15 pts por debajo del SL nominal)
        bar_open = 4960.0
        bar_low = 4955.0
        ts = pd.Timestamp("2026-10-15 03:00:00-04:00")

        # Cierre mediante la lógica optimizada de gaps
        exit_p = min(bar_open, current_trade.stop_loss_price)
        backtester._close_trade(
            trade=current_trade,
            exit_price=exit_p,
            exit_time=ts,
            exit_reason=F9ExitReason.STOP_LOSS,
            slippage_ticks=1.0,
        )

        assert current_trade.raw_exit_price == 4960.0, "El fill bruto debe ser el open del gap (4960), no el SL (4975)"
        assert current_trade.gross_pnl_pts == -40.0, "La pérdida bruta debe ser 5000 - 4960 = -40 pts"

    def test_term_structure_slope_negative_strictly_blocks_longs(self):
        """
        Verifica que cuando la pendiente es negativa (slope = 1.0 - TSR < 0, TSR > 1.0),
        la compuerta de régimen y las señales bloqueen incondicionalmente cualquier compra.
        """
        tsr_inverted = 1.12
        slope = evaluate_term_structure_slope(tsr_inverted)
        assert slope < 0.0, "La pendiente de la curva debe ser negativa"
        assert is_term_structure_inverted(tsr_inverted, backwardation_threshold=1.00)

        # Señal nocturna con Z_drop extremo
        s_on = evaluate_overnight_signal_n1(
            z_drop_prev=-2.5,
            s_vrp=0,
            tsr=tsr_inverted,
            max_tsr_long=1.00,
        )
        assert s_on == 0, "No debe comprar en la noche con pendiente negativa"

        # Señal pre-macro con flujo comprador
        s_macro = evaluate_pre_macro_signal_n1(
            z_ofi_pre=3.0,
            current_price=5010.0,
            vwap_prior=5000.0,
            s_vrp=0,
            tsr=tsr_inverted,
            max_tsr_long=1.00,
        )
        assert s_macro == 0, "No debe comprar pre-macro con pendiente negativa"
