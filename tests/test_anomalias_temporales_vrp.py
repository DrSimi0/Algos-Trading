"""
tests/test_anomalias_temporales_vrp.py
Suite exhaustiva de pruebas unitarias, matemáticas, microestructurales y de estrés
para la Estrategia F9: Anomalías Temporales y Variance Risk Premium (VRP).

Cubre:
1. Fórmulas matemáticas puras de VRP, IV, RV, TSR y Z-score (Sección 3.1).
2. Compuerta de régimen estructural S_VRP in {-1, 0, +1} y asimetría de pánico.
3. Deriva nocturna condicional (Boyarchenko et al., 2023) y salidas (08:30 EST y SL 0.50%).
4. Deriva pre-macro informada (Kurov et al., 2019) y salida forzosa T-3m (SL 10 pts).
5. Cumplimiento absoluto del embargo de noticias Prop Trading [T-2m, T+2m].
6. Cierre universal CME previo al mantenimiento (16:45 EST).
7. Dimensionamiento de posición, partición fraccionaria de DLL y conmutación Estándar/Micro.
8. Simulación End-to-End con cuenta de fondeo (DLL, MTDD, comisiones y slippage de futuros CME).
"""

import math
from datetime import time, datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.strategies.anomalias_temporales_vrp import (
    TemporalAnomaliesVRPStrategy,
    TemporalAnomaliesVRPBacktester,
    VRPPropAccount,
    VRPTradeRecord,
    VRPExitReason,
    VRPRegimeState,
    SubModelType,
    calculate_implied_variance,
    calculate_daily_realized_variance,
    calculate_annualized_realized_variance,
    calculate_vrp,
    calculate_z_score_vrp,
    calculate_term_structure_ratio,
    evaluate_vrp_regime_gate,
    calculate_intraday_return,
    evaluate_overnight_signal,
    calculate_window_vwap,
    evaluate_pre_macro_signal,
    calculate_overnight_sizing,
    calculate_pre_macro_sizing,
)
from src.core.futures_specs import get_spec


# ==============================================================================
# 1. PRUEBAS DE FORMULACIÓN MATEMÁTICA PURA (SECCIÓN 3.1)
# ==============================================================================

class TestVRPMathematicalFormulas:
    """Verifica las ecuaciones matemáticas exactas formuladas en F9."""

    def test_implied_variance_formula(self):
        """IV_t = (VIX_t / 100)^2."""
        # VIX = 20.0 => (0.20)^2 = 0.04
        assert pytest.approx(calculate_implied_variance(20.0), abs=1e-8) == 0.04
        # VIX = 15.0 => (0.15)^2 = 0.0225
        assert pytest.approx(calculate_implied_variance(15.0), abs=1e-8) == 0.0225

        with pytest.raises(ValueError):
            calculate_implied_variance(-5.0)

    def test_daily_realized_variance(self):
        """RV_t^{(d)} = sum r_{5m}^2."""
        returns = [0.001, -0.002, 0.0015, -0.0005]
        expected = sum(r**2 for r in returns)
        assert pytest.approx(calculate_daily_realized_variance(returns), abs=1e-10) == expected
        assert calculate_daily_realized_variance([]) == 0.0

    def test_annualized_realized_variance(self):
        """RV_t^ann = (252 / K) * sum_{k=0}^{K-1} RV_{t-k}^{(d)} con K = 22."""
        # 22 días con RV diaria de 0.0001
        daily_rvs = [0.0001] * 22
        expected = 252.0 * 0.0001  # 0.0252
        calc = calculate_annualized_realized_variance(daily_rvs, k=22)
        assert pytest.approx(calc, abs=1e-8) == expected

    def test_vrp_formula(self):
        """VRP_t = IV_t - RV_t^ann."""
        iv = 0.0400
        rv_ann = 0.0250
        assert pytest.approx(calculate_vrp(iv, rv_ann), abs=1e-8) == 0.0150

    def test_z_score_vrp(self):
        """Z_VRP = (VRP - mu) / sigma."""
        history = [0.01, 0.02, 0.015, 0.018, 0.012]
        curr = 0.03
        all_vals = history + [curr]
        mu = np.mean(all_vals)
        sigma = np.std(all_vals, ddof=1)
        expected = (curr - mu) / sigma

        calc = calculate_z_score_vrp(curr, history, lookback=60)
        assert pytest.approx(calc, abs=1e-6) == expected

    def test_term_structure_ratio(self):
        """TSR_t = VIX_t / VIX3M_t."""
        # Contango: VIX=16, VIX3M=18 => TSR = 16/18 ~ 0.8888
        assert pytest.approx(calculate_term_structure_ratio(16.0, 18.0), abs=1e-4) == 0.8889
        # Backwardation: VIX=25, VIX3M=22 => TSR = 25/22 ~ 1.1364
        assert pytest.approx(calculate_term_structure_ratio(25.0, 22.0), abs=1e-4) == 1.1364

        with pytest.raises(ValueError):
            calculate_term_structure_ratio(20.0, 0.0)

    def test_vrp_regime_gate(self):
        """
        S_VRP:
        +1: Z_VRP >= 1.50 and TSR <= 0.95 (Complacencia / Favorable Long Bias)
        -1: Z_VRP <= -1.20 or TSR >= 1.00 (Backwardation / Bloqueo Total)
         0: Neutral
        """
        # Caso +1: Alta prima de riesgo y curva en contango
        assert evaluate_vrp_regime_gate(z_vrp=1.80, tsr=0.90) == 1
        # Caso -1 por inversión de curva (TSR >= 1.00) aunque Z_VRP sea positivo
        assert evaluate_vrp_regime_gate(z_vrp=1.80, tsr=1.05) == -1
        # Caso -1 por colapso de VRP (Z_VRP <= -1.20) aunque TSR esté bajo
        assert evaluate_vrp_regime_gate(z_vrp=-1.50, tsr=0.90) == -1
        # Caso 0: Parámetros normales
        assert evaluate_vrp_regime_gate(z_vrp=0.50, tsr=0.92) == 0
        assert evaluate_vrp_regime_gate(z_vrp=-0.50, tsr=0.94) == 0

    def test_intraday_return_formula(self):
        """R_{intraday, t-1} = ln( P(16:00) / P(09:30) )."""
        p_930 = 5000.0
        p_1600 = 4950.0  # Caída de 1%
        expected = math.log(4950.0 / 5000.0)
        calc = calculate_intraday_return(p_930, p_1600)
        assert pytest.approx(calc, abs=1e-8) == expected
        assert calc < 0

        with pytest.raises(ValueError):
            calculate_intraday_return(-100.0, 5000.0)


# ==============================================================================
# 2. PRUEBAS DE SUB-MODELOS: OVERNIGHT DRIFT Y PRE-MACRO DRIFT (SECCIÓN 3.2 Y 3.3)
# ==============================================================================

class TestSubModelsLogic:
    """Valida la activación y filtros de Overnight y Pre-Macro."""

    def test_overnight_signal_trigger_and_panic_veto(self):
        """
        S_overnight = I( R_{intraday, t-1} <= -0.0075 ) * I( S_VRP != -1 ).
        """
        # Caída de 0.80% (<= -0.0075) con régimen neutral (S_VRP = 0) => +1
        assert evaluate_overnight_signal(r_intraday_prev=-0.0080, s_vrp=0, theta_drop=0.0075) == 1
        # Caída de 1.20% con régimen complaciente (S_VRP = +1) => +1
        assert evaluate_overnight_signal(r_intraday_prev=-0.0120, s_vrp=1, theta_drop=0.0075) == 1
        # Caída leve de 0.50% (> -0.0075) => 0 (No cumple umbral theta_drop)
        assert evaluate_overnight_signal(r_intraday_prev=-0.0050, s_vrp=1, theta_drop=0.0075) == 0
        # Fuerte caída de 2.0% pero S_VRP = -1 (Backwardation / Pánico) => 0 (VETO ABSOLUTO)
        assert evaluate_overnight_signal(r_intraday_prev=-0.0200, s_vrp=-1, theta_drop=0.0075) == 0

    def test_window_vwap_calculation(self):
        """VWAP = sum(P * V) / sum(V)."""
        prices = [5000.0, 5005.0, 5010.0]
        volumes = [100.0, 200.0, 300.0]
        expected = (5000 * 100 + 5005 * 200 + 5010 * 300) / 600.0
        assert pytest.approx(calculate_window_vwap(prices, volumes), abs=1e-8) == expected

    def test_pre_macro_signal_trigger_and_veto(self):
        """
        S_macro:
        +1: Z_OFI >= 2.0 and P > VWAP and S_VRP != -1
        -1: Z_OFI <= -2.0 and P < VWAP and S_VRP != +1
         0: otros
        """
        vwap = 5000.0

        # Long pre-macro válido
        assert evaluate_pre_macro_signal(z_ofi_pre=2.5, current_price=5005.0, vwap_prior=vwap, s_vrp=0) == 1
        # Long pre-macro vetado por pánico (S_VRP = -1)
        assert evaluate_pre_macro_signal(z_ofi_pre=2.5, current_price=5005.0, vwap_prior=vwap, s_vrp=-1) == 0
        # Long no cumple precio superior a VWAP
        assert evaluate_pre_macro_signal(z_ofi_pre=2.5, current_price=4995.0, vwap_prior=vwap, s_vrp=0) == 0

        # Short pre-macro válido
        assert evaluate_pre_macro_signal(z_ofi_pre=-2.5, current_price=4995.0, vwap_prior=vwap, s_vrp=0) == -1
        # Short pre-macro vetado por complacencia alcista fuerte (S_VRP = +1)
        assert evaluate_pre_macro_signal(z_ofi_pre=-2.5, current_price=4995.0, vwap_prior=vwap, s_vrp=1) == 0
        # Short no cumple precio inferior a VWAP
        assert evaluate_pre_macro_signal(z_ofi_pre=-2.5, current_price=5005.0, vwap_prior=vwap, s_vrp=0) == 0

        # Flujo débil (|Z_OFI| < 2.0)
        assert evaluate_pre_macro_signal(z_ofi_pre=1.2, current_price=5005.0, vwap_prior=vwap, s_vrp=0) == 0


# ==============================================================================
# 3. PRUEBAS DE GESTIÓN DE RIESGO PROP TRADING Y SIZING (SECCIÓN 5)
# ==============================================================================

class TestPropRiskManagementAndSizing:
    """Valida partición de DLL, colchón MTDD y conmutación automática Estándar a Micro."""

    def test_dll_partition_overnight_vs_rth(self):
        """Delta DLL_t^nocturno = 0.25 * Delta DLL, Delta DLL_t^RTH = 0.75 * Delta DLL."""
        account = VRPPropAccount(dll=1000.0, mtdd=2000.0)
        assert account.delta_dll == 1000.0
        assert account.delta_dll_overnight == 250.0
        assert account.delta_dll_rth == 750.0

        # Si ya se perdió $400 en el día:
        account.daily_realized_loss = 400.0
        assert account.delta_dll == 600.0
        assert account.delta_dll_overnight == 150.0
        assert account.delta_dll_rth == 450.0

    def test_overnight_sizing_standard_and_micro_switch(self):
        """
        N_ON = min( floor( min(DLL_on, 0.05 * Cushion) / Risk_unit ), N_max )
        """
        m_es = 50.0
        m_mes = 5.0
        p_entry = 5000.0
        delta_sl = 0.0050  # 25 puntos = $1,250 en ES, $125 en MES

        # Presupuesto amplio: $3,000 en colchón y $2,000 en DLL
        # Riesgo unitario ES: 0.0050 * 5000 * 50 = $1,250
        n_contracts, c_type = calculate_overnight_sizing(
            budget_daily_on=2000.0,
            budget_cushion_on=2500.0,
            p_entry=p_entry,
            multiplier_std=m_es,
            multiplier_micro=m_mes,
            delta_sl_on=delta_sl,
            n_max_std=2,
        )
        assert c_type == "STD"
        assert n_contracts == 1  # 2000 // 1250 = 1

        # Presupuesto restringido a $250 (partición nocturna típica de $1000 DLL):
        # En ES requiere $1,250 -> Insuficiente para Estándar (N_std = 0)
        # En MES requiere $125 -> Alcanza para 2 microcontratos ($250 // $125 = 2)
        n_contracts_micro, c_type_micro = calculate_overnight_sizing(
            budget_daily_on=250.0,
            budget_cushion_on=100.0,  # Acotado por 5% de MTDD ($2000 * 0.05 = $100)
            p_entry=p_entry,
            multiplier_std=m_es,
            multiplier_micro=m_mes,
            delta_sl_on=delta_sl,
            n_max_std=2,
            n_max_micro=20,
        )
        # $100 // $125 = 0 -> Si el colchón es $100, no alcanza ni para 1 micro
        assert n_contracts_micro == 0
        assert c_type_micro == "NONE"

        # Con colchón suficiente ($150):
        n_c, c_t = calculate_overnight_sizing(
            budget_daily_on=250.0,
            budget_cushion_on=150.0,
            p_entry=p_entry,
            multiplier_std=m_es,
            multiplier_micro=m_mes,
            delta_sl_on=delta_sl,
        )
        assert c_t == "MICRO"
        assert n_c == 1  # 150 // 125 = 1 MES

    def test_pre_macro_sizing_standard_and_micro(self):
        """
        N_macro = min( floor( min( 0.15 * DLL_RTH, 0.05 * Cushion ) / (Delta P_stop * M) ), N_max )
        """
        m_es = 50.0
        m_mes = 5.0
        delta_p_stop = 10.0  # 10 pts * 50 = $500 en ES, $50 en MES

        # Caso Estándar: Presupuesto $1,000
        n_std, c_std = calculate_pre_macro_sizing(
            budget_daily_macro=1000.0,
            budget_cushion_macro=1000.0,
            delta_p_stop=delta_p_stop,
            multiplier_std=m_es,
            multiplier_micro=m_mes,
            n_max_std=2,
        )
        assert c_std == "STD"
        assert n_std == 2

        # Caso Micro: Presupuesto $150 (insuficiente para $500 en ES)
        n_mic, c_mic = calculate_pre_macro_sizing(
            budget_daily_macro=150.0,
            budget_cushion_macro=150.0,
            delta_p_stop=delta_p_stop,
            multiplier_std=m_es,
            multiplier_micro=m_mes,
            n_max_std=2,
        )
        assert c_mic == "MICRO"
        assert n_mic == 3  # 150 // 50 = 3 MES


# ==============================================================================
# 4. PRUEBAS DE SALIDAS, EMBARGO Y REGLAS DE TIEMPO (SECCIÓN 4)
# ==============================================================================

class TestExitsAndEmbargoCompliance:
    """Verifica salidas deterministas, Stop Loss y regla de embargo de Prop Trading."""

    def test_overnight_deterministic_exit_at_830(self):
        """Posición nocturna se liquida obligatoriamente a las 08:30:00 EST."""
        strat = TemporalAnomaliesVRPStrategy()
        # Crear datos de prueba sintéticos: día previo con caída > 0.75%, abre a las 01:30
        timestamps = pd.date_range("2026-10-15 00:00:00", "2026-10-15 10:00:00", freq="5min", tz="America/New_York")
        df = pd.DataFrame({
            "Open": [5000.0] * len(timestamps),
            "High": [5005.0] * len(timestamps),
            "Low": [4995.0] * len(timestamps),
            "Close": [5000.0] * len(timestamps),
            "Volume": [1000.0] * len(timestamps),
        }, index=timestamps)

        backtester = TemporalAnomaliesVRPBacktester(strategy=strat)
        # Mock de trade abierto a las 01:30
        trade = VRPTradeRecord(
            trade_id=1,
            date="2026-10-15",
            submodel=SubModelType.OVERNIGHT.value,
            direction=1,
            contracts=1,
            contract_type="STD",
            symbol="ES",
            entry_time="2026-10-15 01:30:00-04:00",
            entry_price=5000.0,
            stop_loss_price=4975.0,  # 0.50% SL
        )

        # Barra de las 08:30:00 debe gatillar DETERMINISTIC_TIME_EXIT
        ts_830 = pd.Timestamp("2026-10-15 08:30:00-04:00")
        backtester._close_trade(
            trade=trade,
            exit_price=5010.0,
            exit_time=ts_830,
            exit_reason=VRPExitReason.DETERMINISTIC_TIME_EXIT,
            slippage_ticks=1.0,
        )
        assert trade.exit_reason == VRPExitReason.DETERMINISTIC_TIME_EXIT
        assert trade.exit_time == str(ts_830)
        assert trade.net_pnl_usd > 0  # 10 pts * 50 = $500 - comisiones/slippage

    def test_pre_macro_exit_t_minus_3_and_embargo_isolation(self):
        """
        Posición pre-macro iniciada en T-15m se liquida forzosamente en T-3m,
        garantizando cuenta plana durante el embargo [T-2m, T+2m].
        """
        macro_event_dt = pd.Timestamp("2026-10-15 08:30:00-04:00")
        strat = TemporalAnomaliesVRPStrategy(params={
            "macro_schedule": [macro_event_dt],
            "delta_p_stop_macro": 10.0,
        })
        backtester = TemporalAnomaliesVRPBacktester(strategy=strat, account_size=50000.0)

        # Crear barras de 5 minutos: 08:00 a 09:00
        # Ventana previa [08:00, 08:15) en 5000.0 (VWAP = 5000.0)
        # Ventana pre-macro [08:15, 08:27] en 5005.0 (P_t > VWAP)
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
        # Salida determinista forzosa en T-3m (08:27:00 EST)
        expected_exit_time = str(macro_event_dt - timedelta(minutes=3))
        assert t["exit_time"] == expected_exit_time
        assert t["exit_reason"] == VRPExitReason.DETERMINISTIC_TIME_EXIT.value
        # Verificar que la salida precede al embargo (08:28:00) por al menos 60 segundos
        exit_dt = pd.Timestamp(t["exit_time"])
        embargo_start = macro_event_dt - timedelta(minutes=2)
        assert exit_dt <= embargo_start - timedelta(seconds=60)


# ==============================================================================
# 5. PRUEBAS END-TO-END DE BACKTEST (SECCIÓN 6)
# ==============================================================================

class TestEndToEndBacktestExecution:
    """Verifica la ejecución determinista del backtest completo."""

    def test_backtest_with_synthetic_overnight_drift(self):
        """Simula una sesión con caída previa RTH y posterior rebote nocturno."""
        strat = TemporalAnomaliesVRPStrategy(params={
            "theta_drop": 0.0075,
            "delta_sl_on": 0.0050,
        })
        backtester = TemporalAnomaliesVRPBacktester(
            strategy=strat,
            account_size=50000.0,
            daily_loss_limit=1000.0,
            max_trailing_loss=3000.0,
        )

        # Generar 2 días completos de barras de 5m (288 barras por día = 576 barras)
        # Día 1: cae de 5000 a 4950 (-1.0% en RTH)
        # Día 2: abre a las 01:30 en 4950 y sube a 4975 a las 08:30 (+0.50% nocturno)
        ts_d1 = pd.date_range("2026-10-14 00:00:00", "2026-10-14 23:55:00", freq="5min", tz="America/New_York")
        ts_d2 = pd.date_range("2026-10-15 00:00:00", "2026-10-15 23:55:00", freq="5min", tz="America/New_York")
        all_ts = ts_d1.append(ts_d2)

        prices = []
        for ts in all_ts:
            if ts.date() == ts_d1[0].date():
                if time(9, 30) <= ts.time() <= time(16, 0):
                    # Caída gradual de 5000 a 4950
                    frac = (ts - ts.replace(hour=9, minute=30)).total_seconds() / (6.5 * 3600)
                    prices.append(5000.0 - (50.0 * frac))
                else:
                    prices.append(5000.0 if ts.time() < time(9, 30) else 4950.0)
            else:
                if time(1, 30) <= ts.time() <= time(8, 30):
                    # Subida gradual de 4950 a 4980
                    frac = (ts - ts.replace(hour=1, minute=30)).total_seconds() / (7.0 * 3600)
                    prices.append(4950.0 + (30.0 * frac))
                else:
                    prices.append(4980.0 if ts.time() > time(8, 30) else 4950.0)

        df = pd.DataFrame({
            "Open": prices,
            "High": [p + 0.50 for p in prices],
            "Low": [p - 0.50 for p in prices],
            "Close": prices,
            "Volume": [1000.0] * len(prices),
        }, index=all_ts)

        results = backtester.run(df)
        trades = results["trades"]
        metrics = results["metrics"]
        prop_eval = results["prop_evaluation"]

        # Debe haber ejecutado exactamente 1 trade nocturno
        assert len(trades) == 1
        t = trades.iloc[0]
        assert t["submodel"] == SubModelType.OVERNIGHT.value
        assert t["direction"] == 1
        assert t["exit_reason"] == VRPExitReason.DETERMINISTIC_TIME_EXIT.value
        assert t["net_pnl_usd"] > 0
        assert prop_eval["status"] in ["PASS", "ACTIVE"]
        assert not prop_eval["is_breached"]


# ==============================================================================
# 6. PRUEBAS DE CASOS EXTREMOS, ROBUSTEZ Y ESTRÉS
# ==============================================================================

class TestEdgeCasesAndStressValidations:
    """Valida resiliencia ante datos anómalos, ceros, límites de riesgo y condiciones límite."""

    def test_missing_columns_validation(self):
        """Verifica que un DataFrame incompleto lance ValueError."""
        strat = TemporalAnomaliesVRPStrategy()
        bad_df = pd.DataFrame({"Close": [5000.0, 5010.0]}, index=pd.date_range("2026-10-15 09:30", periods=2, freq="5min"))
        with pytest.raises(ValueError, match="Faltan columnas requeridas"):
            strat.validate_data(bad_df)

    def test_zero_volume_vwap_fallback(self):
        """VWAP con volumen cero no divide por cero, retorna la media de precios."""
        prices = [5000.0, 5010.0, 5020.0]
        volumes = [0.0, 0.0, 0.0]
        vwap = calculate_window_vwap(prices, volumes)
        assert pytest.approx(vwap, abs=1e-8) == 5010.0
        assert calculate_window_vwap([], []) == 0.0

    def test_z_score_epsilon_guard_constant_history(self):
        """Z-score sobre una serie con varianza cero no produce ZeroDivisionError gracias a epsilon guard."""
        history = [0.01] * 50
        curr = 0.01
        z = calculate_z_score_vrp(curr, history, lookback=60, eps=1e-6)
        assert pytest.approx(z, abs=1e-5) == 0.0

    def test_overnight_stop_loss_trigger(self):
        """Si durante la noche el precio cae >= 0.50%, se activa el Stop Loss de inmediato."""
        strat = TemporalAnomaliesVRPStrategy(params={"delta_sl_on": 0.0050})
        backtester = TemporalAnomaliesVRPBacktester(strategy=strat, account_size=50000.0, max_trailing_loss=3000.0)

        # Día 1: caída RTH para habilitar overnight
        ts_d1 = pd.date_range("2026-10-14 00:00:00", "2026-10-14 23:55:00", freq="5min", tz="America/New_York")
        ts_d2 = pd.date_range("2026-10-15 00:00:00", "2026-10-15 10:00:00", freq="5min", tz="America/New_York")
        all_ts = ts_d1.append(ts_d2)

        prices = []
        for ts in all_ts:
            if ts.date() == ts_d1[0].date():
                if time(9, 30) <= ts.time() <= time(16, 0):
                    frac = (ts - ts.replace(hour=9, minute=30)).total_seconds() / (6.5 * 3600)
                    prices.append(5000.0 - (50.0 * frac))  # Cae a 4950
                else:
                    prices.append(5000.0 if ts.time() < time(9, 30) else 4950.0)
            else:
                # A las 01:30 entra en 4950. A las 03:00 cae violentamente a 4900 (-1.0% > 0.50% SL)
                if ts.time() < time(1, 30):
                    prices.append(4950.0)
                elif time(1, 30) <= ts.time() < time(3, 0):
                    prices.append(4950.0)
                elif ts.time() >= time(3, 0):
                    prices.append(4900.0)

        df = pd.DataFrame({
            "Open": prices,
            "High": prices,
            "Low": [p - 1.0 for p in prices],
            "Close": prices,
            "Volume": [1000.0] * len(prices),
        }, index=all_ts)

        results = backtester.run(df)
        trades = results["trades"]
        assert len(trades) == 1
        t = trades.iloc[0]
        assert t["exit_reason"] == VRPExitReason.STOP_LOSS.value
        assert t["gross_pnl_usd"] < 0

    def test_pre_macro_short_and_deterministic_t_minus_3_exit(self):
        """Simula una orden corta pre-macro ejecutada a T-15m y cerrada exactamente en T-3m."""
        macro_time = "2026-10-15 08:30:00"
        strat = TemporalAnomaliesVRPStrategy(params={
            "macro_schedule": [macro_time],
            "delta_p_stop_macro": 10.0,
        })
        backtester = TemporalAnomaliesVRPBacktester(strategy=strat, account_size=50000.0, max_trailing_loss=3000.0)

        ts = pd.date_range("2026-10-15 07:00:00", "2026-10-15 09:00:00", freq="1min", tz="America/New_York")
        prices = []
        z_ofis = []
        for t in ts:
            if t < pd.Timestamp("2026-10-15 08:15:00", tz="America/New_York"):
                prices.append(5000.0)
                z_ofis.append(0.0)
            elif pd.Timestamp("2026-10-15 08:15:00", tz="America/New_York") <= t < pd.Timestamp("2026-10-15 08:27:00", tz="America/New_York"):
                prices.append(4990.0)  # Cae por debajo de VWAP
                z_ofis.append(-2.5)    # Fuerte flujo vendedor
            else:
                prices.append(4980.0)
                z_ofis.append(0.0)

        df = pd.DataFrame({
            "Open": prices,
            "High": [p + 0.25 for p in prices],
            "Low": [p - 0.25 for p in prices],
            "Close": prices,
            "Volume": [2000.0] * len(prices),
            "Z_OFI": z_ofis,
        }, index=ts)

        results = backtester.run(df)
        trades = results["trades"]
        assert len(trades) == 1
        t = trades.iloc[0]
        assert t["submodel"] == SubModelType.PRE_MACRO.value
        assert t["direction"] == -1  # Short
        assert t["exit_reason"] == VRPExitReason.DETERMINISTIC_TIME_EXIT.value
        assert t["net_pnl_usd"] > 0

    def test_account_breach_blocks_subsequent_operations(self):
        """Verifica que una cuenta quebrada (DLL violado) impida nuevas operaciones."""
        strat = TemporalAnomaliesVRPStrategy()
        account = VRPPropAccount(dll=500.0, mtdd=2000.0)
        account.update_balance(-600.0)
        assert account.is_breached
        assert "Daily Loss Limit" in account.breach_reason

    def test_prop_account_custom_capital_initialization(self):
        """Verifica que un capital inicial arbitrario inicialice correctamente equity y peak_equity."""
        account = VRPPropAccount(initial_capital=100000.0, dll=2500.0, mtdd=5000.0)
        assert account.initial_capital == 100000.0
        assert account.equity == 100000.0
        assert account.peak_equity == 100000.0

    def test_gross_pnl_friction_accounting_identity(self):
        """Verifica que para cualquier trade cerrado se cumpla: gross_pnl_usd - friction_usd == net_pnl_usd."""
        strat = TemporalAnomaliesVRPStrategy()
        backtester = TemporalAnomaliesVRPBacktester(strategy=strat)
        trade = VRPTradeRecord(
            trade_id=1,
            date="2026-10-15",
            submodel=SubModelType.OVERNIGHT.value,
            direction=1,
            contracts=2,
            contract_type="STD",
            symbol="ES",
            entry_time="2026-10-15 01:30:00-04:00",
            entry_price=5000.25,
            raw_entry_price=5000.0,
            stop_loss_price=4975.0,
        )
        backtester._close_trade(
            trade=trade,
            exit_price=5010.0,
            exit_time=pd.Timestamp("2026-10-15 08:30:00-04:00"),
            exit_reason=VRPExitReason.DETERMINISTIC_TIME_EXIT,
            slippage_ticks=1.0,
        )
        assert pytest.approx(trade.friction_usd, abs=0.01) == trade.commissions_usd + trade.slippage_usd
        assert pytest.approx(trade.gross_pnl_usd - trade.friction_usd, abs=0.01) == trade.net_pnl_usd

    def test_overnight_exit_before_macro_embargo_at_830(self):
        """
        Si hay un anuncio macroeconómico a las 08:30 EST, la posición nocturna
        debe liquidarse antes de entrar al embargo [08:28, 08:32].
        """
        macro_time = pd.Timestamp("2026-10-15 08:30:00-04:00")
        strat = TemporalAnomaliesVRPStrategy(params={
            "macro_schedule": [macro_time],
            "theta_drop": 0.0075,
        })
        backtester = TemporalAnomaliesVRPBacktester(strategy=strat, account_size=100000.0, max_trailing_loss=5000.0)

        ts_d1 = pd.date_range("2026-10-14 00:00:00", "2026-10-14 23:55:00", freq="5min", tz="America/New_York")
        ts_d2 = pd.date_range("2026-10-15 00:00:00", "2026-10-15 10:00:00", freq="5min", tz="America/New_York")
        all_ts = ts_d1.append(ts_d2)

        prices = []
        for ts in all_ts:
            if ts.date() == ts_d1[0].date():
                if time(9, 30) <= ts.time() <= time(16, 0):
                    frac = (ts - ts.replace(hour=9, minute=30)).total_seconds() / (6.5 * 3600)
                    prices.append(5000.0 - (50.0 * frac))
                else:
                    prices.append(5000.0 if ts.time() < time(9, 30) else 4950.0)
            else:
                prices.append(4960.0)

        df = pd.DataFrame({
            "Open": prices,
            "High": prices,
            "Low": [p - 0.5 for p in prices],
            "Close": prices,
            "Volume": [1000.0] * len(prices),
        }, index=all_ts)

        results = backtester.run(df)
        trades = results["trades"]
        assert len(trades) >= 1
        t_on = trades[trades["submodel"] == SubModelType.OVERNIGHT.value].iloc[0]
        # Debe salir antes de 08:28:00
        exit_dt = pd.Timestamp(t_on["exit_time"])
        assert exit_dt <= pd.Timestamp("2026-10-15 08:27:00-04:00")
        assert t_on["exit_reason"] in [VRPExitReason.EMBARGO_FLAT.value, VRPExitReason.DETERMINISTIC_TIME_EXIT.value]

    def test_generate_signals_pre_macro_not_shadowed_by_overnight(self):
        """Verifica que un anuncio a las 08:30 genere señal pre-macro y no quede eclipsado por overnight."""
        macro_dt = pd.Timestamp("2026-10-15 08:30:00-04:00")
        strat = TemporalAnomaliesVRPStrategy(params={
            "macro_schedule": [macro_dt],
        })

        ts = pd.date_range("2026-10-15 08:00:00", "2026-10-15 08:35:00", freq="5min", tz="America/New_York")
        prices = [5000.0 if t < pd.Timestamp("2026-10-15 08:15:00", tz="America/New_York") else 5005.0 for t in ts]
        df = pd.DataFrame({
            "Open": prices,
            "High": [p + 2.0 for p in prices],
            "Low": [p - 2.0 for p in prices],
            "Close": prices,
            "Volume": [1000.0] * len(ts),
            "Z_OFI": [2.5] * len(ts),
        }, index=ts)

        signals = strat.generate_signals(df)
        # A las 08:15:00 (T-15m) debe tener señal pre-macro +1.0
        assert signals.loc["2026-10-15 08:15:00-04:00"] == 1.0
        # A las 08:20:00 debe tener señal pre-macro +1.0
        assert signals.loc["2026-10-15 08:20:00-04:00"] == 1.0
        # A las 08:30:00 (dentro del embargo [08:28, 08:32]) debe ser 0.0
        assert signals.loc["2026-10-15 08:30:00-04:00"] == 0.0

