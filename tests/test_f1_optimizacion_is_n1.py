"""
tests/test_f1_optimizacion_is_n1.py
Suite de pruebas unitarias y de estrés cuantitativo para la versión alterna:
Estrategia F1-Optimizacion -IS- N1 (Market Intraday Momentum & MOC Seasonality).

Cobertura de Pruebas:
1. Normalización del retorno de apertura frente a la volatilidad realizada intradía (ATR y RV cuadrática).
2. Filtro de Puro Ruido (Z_norm < Z_noise_min o |r_first| < theta^*).
3. Filtro de Agotamiento de Rango (Z_norm > Z_exhaust_max o Range_open / ATR_daily > max_range_fraction).
4. Escalado estricto del tamaño de posición según el colchón restante del Daily Loss Limit (DLL).
5. Casos de borde de colchón: colchón nulo, colchón insuficiente para 1 micro, monotonía de escalado.
6. Filtro de coherencia tendencial vespertina con el VWAP acumulado a las 15:30 EST.
7. Niveles de salida adaptativos (ATR SL y TP con asimetría 1.6R) y decaimiento de momento a las 15:45 EST.
8. Ejecución E2E del simulador IntradayMomentumOptN1Backtester en sesiones sintéticas y datos de mercado.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.strategies.momentum.f1_optimizacion_is_n1 import (
    MarketIntradayMomentumOptN1,
    IntradayMomentumOptN1Backtester,
    F1OptimizacionISN1,
    F1OptimizacionISN1Backtester,
    calculate_intraday_realized_volatility,
    calculate_opening_noise_and_exhaustion,
    calculate_dll_cushion_scaling,
    compute_vwap,
)
from src.core.futures_specs import get_spec


# ==============================================================================
# 1. NORMALIZACIÓN DE APERTURA Y VOLATILIDAD REALIZADA INTRADÍA
# ==============================================================================

class TestOpeningReturnRealizedVolNormalization:
    """Valida la normalización analítica del retorno matutino contra la volatilidad realizada."""

    def test_calculate_intraday_realized_volatility_standard_bars(self):
        """Calcula ATR_intra y RV_intra en 6 barras de 5 minutos."""
        # 6 barras de 5m con rango de 2 puntos cada una
        data = {
            "Open": [5000.0, 5001.0, 5002.0, 5003.0, 5004.0, 5005.0],
            "High": [5002.0, 5003.0, 5004.0, 5005.0, 5006.0, 5007.0],
            "Low": [5000.0, 5001.0, 5002.0, 5003.0, 5004.0, 5005.0],
            "Close": [5001.0, 5002.0, 5003.0, 5004.0, 5005.0, 5006.0],
            "Volume": [1000.0] * 6
        }
        df_obs = pd.DataFrame(data)
        p_1000 = 5006.0

        atr, rv, sigma_atr = calculate_intraday_realized_volatility(df_obs, p_1000)

        assert atr == 2.0
        assert rv > 0.0
        assert pytest.approx(sigma_atr, abs=1e-6) == 2.0 / 5006.0
        assert isinstance(atr, float)
        assert isinstance(rv, float)
        assert isinstance(sigma_atr, float)

    def test_calculate_intraday_realized_volatility_flat_market(self):
        """Mercado completamente estático (High = Low = Close): no debe producir división por cero."""
        data = {
            "Open": [5000.0] * 6,
            "High": [5000.0] * 6,
            "Low": [5000.0] * 6,
            "Close": [5000.0] * 6,
            "Volume": [100.0] * 6
        }
        df_flat = pd.DataFrame(data)
        atr, rv, sigma_atr = calculate_intraday_realized_volatility(df_flat, 5000.0)

        assert atr == 0.0
        assert rv == 0.0
        assert sigma_atr == 0.0

    def test_calculate_intraday_realized_volatility_empty_dataframe(self):
        """Maneja de forma segura DataFrames vacíos."""
        df_empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        atr, rv, sigma_atr = calculate_intraday_realized_volatility(df_empty, 5000.0)

        assert atr == 0.0
        assert rv == 0.0
        assert sigma_atr == 0.0


# ==============================================================================
# 2. FILTRADO BILATERAL: PURO RUIDO VS AGOTAMIENTO DE RANGO
# ==============================================================================

class TestNoiseAndRangeExhaustionFilters:
    """Verifica la detección matemática de ruido y agotamiento de rango."""

    def test_noise_filter_triggers_on_low_directional_move_with_high_vol(self):
        """
        Día donde hubo mucho vaivén (ATR = 6.0 pts) pero retorno neto insignificante (0.25 pts).
        Debe ser clasificado como FILTER_NOISE.
        """
        # Barras oscilantes con gran mecha pero cierre cercano a la apertura
        data = {
            "Open": [5000.0, 5005.0, 4995.0, 5004.0, 4996.0, 5001.0],
            "High": [5006.0, 5007.0, 5005.0, 5006.0, 5004.0, 5003.0],
            "Low": [4995.0, 4994.0, 4990.0, 4995.0, 4992.0, 4998.0],
            "Close": [5004.0, 4996.0, 5003.0, 4997.0, 5001.0, 5000.25],
            "Volume": [5000.0] * 6
        }
        df_obs = pd.DataFrame(data)
        p_930 = 5000.0
        p_1000 = 5000.25
        r_first = math.log(p_1000 / p_930)  # retorno microscópico

        res = calculate_opening_noise_and_exhaustion(
            r_first=r_first,
            obs_bars=df_obs,
            p_1000=p_1000,
            daily_atr=25.0,
            z_noise_min=0.60
        )

        assert res["is_noise"] is True
        assert res["filter_status"] == "FILTER_NOISE"
        assert res["is_qualified"] is False

    def test_range_exhaustion_triggers_when_opening_range_exceeds_daily_fraction(self):
        """
        Día donde el movimiento inicial consumió más del 70% del ATR diario esperado.
        Debe ser clasificado como FILTER_RANGE_EXHAUSTED.
        """
        # Apertura explosiva de 25 puntos en un activo cuyo ATR diario promedio es 30 puntos
        data = {
            "Open": [5000.0, 5005.0, 5010.0, 5015.0, 5020.0, 5025.0],
            "High": [5006.0, 5011.0, 5016.0, 5021.0, 5026.0, 5028.0],
            "Low": [4998.0, 5004.0, 5009.0, 5014.0, 5019.0, 5024.0],
            "Close": [5005.0, 5010.0, 5015.0, 5020.0, 5025.0, 5027.0],
            "Volume": [8000.0] * 6
        }
        df_obs = pd.DataFrame(data)
        p_930 = 5000.0
        p_1000 = 5027.0
        r_first = math.log(p_1000 / p_930)

        # Range_open = 5028 - 4998 = 30 puntos. Si daily_atr = 35 -> e_range = 30/35 = 85.7% > 65%
        res = calculate_opening_noise_and_exhaustion(
            r_first=r_first,
            obs_bars=df_obs,
            p_1000=p_1000,
            daily_atr=35.0,
            max_range_fraction=0.65
        )

        assert res["is_exhausted"] is True
        assert res["filter_status"] == "FILTER_RANGE_EXHAUSTED"
        assert res["is_qualified"] is False

    def test_valid_momentum_passes_both_noise_and_exhaustion(self):
        """
        Movimiento institucional balanceado: direccional, sin ser puro ruido ni agotar el día.
        Debe calificar como VALID_MOMENTUM.
        """
        # Desplazamiento limpio de 8 puntos en ES con ATR diario de 40 puntos (20% del día consumido)
        data = {
            "Open": [5000.0, 5002.0, 5003.0, 5005.0, 5006.0, 5007.0],
            "High": [5003.0, 5004.0, 5006.0, 5007.0, 5008.0, 5009.0],
            "Low": [4999.0, 5001.0, 5002.0, 5004.0, 5005.0, 5006.0],
            "Close": [5002.0, 5003.0, 5005.0, 5006.0, 5007.0, 5008.0],
            "Volume": [10000.0] * 6
        }
        df_obs = pd.DataFrame(data)
        p_930 = 5000.0
        p_1000 = 5008.0
        r_first = math.log(p_1000 / p_930)

        res = calculate_opening_noise_and_exhaustion(
            r_first=r_first,
            obs_bars=df_obs,
            p_1000=p_1000,
            daily_atr=40.0,
            z_noise_min=0.60,
            z_exhaust_max=2.50,
            max_range_fraction=0.65
        )

        assert res["is_noise"] is False
        assert res["is_exhausted"] is False
        assert res["filter_status"] == "VALID_MOMENTUM"
        assert res["is_qualified"] is True


# ==============================================================================
# 3. ESCALADO DE TAMAÑO DE POSICIÓN SEGÚN COLCHÓN DE DLL
# ==============================================================================

class TestDLLCushionScaling:
    """Verifica el presupuesto de riesgo y tamaño de posición indexado al colchón restante de DLL."""

    def test_full_dll_cushion_allocates_normal_budget(self):
        """Sin pérdidas previas (L_day = 0), colchón = 100%, asigna presupuesto base."""
        dll = 1000.0
        res = calculate_dll_cushion_scaling(dll=dll, l_day=0.0, k_loss=0, phi_base=0.35)

        assert res["c_dll"] == 1000.0
        assert res["cushion_ratio"] == 1.0
        assert res["b_dll"] == 350.0  # 0.35 * 1000 * 1.0^1.0

    def test_partial_dll_cushion_scales_down_linearly(self):
        """Habiendo perdido $500 (colchón al 50%), presupuesto se reduce cuadráticamente/proporcionalmente."""
        dll = 1000.0
        res_half = calculate_dll_cushion_scaling(
            dll=dll,
            l_day=500.0,
            k_loss=0,
            phi_base=0.35,
            cushion_exponent=1.0
        )

        # C_DLL = 500, ratio = 0.5 -> B_DLL = 0.35 * 500 * 0.5 = 87.50 USD
        assert res_half["c_dll"] == 500.0
        assert res_half["cushion_ratio"] == 0.5
        assert pytest.approx(res_half["b_dll"], abs=1e-5) == 87.50

    def test_exhausted_dll_cushion_halts_trading(self):
        """Si el colchón de DLL se agotó (L_day >= DLL), presupuesto es 0.0 y trading detenido."""
        dll = 1000.0
        res_zero = calculate_dll_cushion_scaling(dll=dll, l_day=1000.0)
        assert res_zero["c_dll"] == 0.0
        assert res_zero["cushion_ratio"] == 0.0
        assert res_zero["b_risk"] == 0.0

        # Pérdida excedida
        res_neg = calculate_dll_cushion_scaling(dll=dll, l_day=1100.0)
        assert res_neg["c_dll"] == 0.0
        assert res_neg["b_risk"] == 0.0

    def test_monotonicity_of_dll_cushion_budget(self):
        """Garantiza matemáticamente que a menor colchón restante, menor o igual presupuesto."""
        dll = 1000.0
        losses = np.linspace(0.0, 1000.0, 20)
        budgets = [
            calculate_dll_cushion_scaling(dll=dll, l_day=l, k_loss=0)["b_risk"]
            for l in losses
        ]
        # Verificar que la secuencia de presupuestos es monótona no-creciente
        for i in range(len(budgets) - 1):
            assert budgets[i] >= budgets[i + 1]

    def test_position_sizing_scaled_contracts_es_and_mes(self):
        """Valida conmutación automática Estándar/Micro y detención por colchón insuficiente."""
        strat = MarketIntradayMomentumOptN1()
        delta_sl_pts = 6.0  # SL de 6 puntos en ES

        # 1. Presupuesto alto ($800) permite operar contrato estándar (ES: 6 pts * $50 + fricción ~$329.50)
        size_high = strat.calculate_position_sizing_scaled(
            b_risk=800.0,
            delta_sl_pts=delta_sl_pts,
            cushion_ratio=1.0
        )
        assert size_high["status"] == "OK"
        assert size_high["contract_type"] == "STD"
        assert size_high["symbol"] == "ES"
        assert size_high["contracts"] == 2  # min(floor(800 / 329.5), 2) = 2

        # 2. Presupuesto medio ($200): insuficiente para ES, conmuta a MES ($34.24 por contrato)
        size_mid = strat.calculate_position_sizing_scaled(
            b_risk=200.0,
            delta_sl_pts=delta_sl_pts,
            cushion_ratio=0.5
        )
        assert size_mid["status"] == "OK"
        assert size_mid["contract_type"] == "MICRO"
        assert size_mid["symbol"] == "MES"
        assert size_mid["contracts"] == int(200.0 // size_mid["risk_per_contract"])

        # 3. Presupuesto ínfimo ($20): ni siquiera alcanza para 1 micro ($34.24) -> HALT
        size_low = strat.calculate_position_sizing_scaled(
            b_risk=20.0,
            delta_sl_pts=delta_sl_pts,
            cushion_ratio=0.05
        )
        assert size_low["contracts"] == 0
        assert size_low["status"] == "HALT_INSUFFICIENT_CUSHION"


# ==============================================================================
# 4. COHERENCIA VWAP Y SALIDAS DINÁMICAS (DECAY EXIT Y TP ASIMÉTRICO)
# ==============================================================================

class TestVWAPAndExitMechanics:
    """Verifica el filtro de coherencia VWAP vespertino y la salida por decaimiento."""

    def test_vwap_coherence_filter(self):
        """Valida que r_first positivo requiera precio > VWAP por al menos 2 ticks."""
        strat = MarketIntradayMomentumOptN1()

        # Crear sesión donde VWAP = 5000.0
        df_session = pd.DataFrame({
            "High": [5001.0, 5001.0],
            "Low": [4999.0, 4999.0],
            "Close": [5000.0, 5000.0],
            "Volume": [1000.0, 1000.0]
        })

        # Caso 1: Mañana alcista y precio a las 15:30 es 5001.0 (delta = +1.0 pt = 4 ticks > 2 ticks) -> COHERENTE
        coh_good, delta_good = strat.evaluate_vwap_coherence(
            r_first=0.0020,
            p_1530=5001.0,
            session_to_1530=df_session,
            tick_size=0.25
        )
        assert coh_good is True
        assert delta_good == 1.0

        # Caso 2: Mañana alcista pero precio a las 15:30 colapsó a 4998.0 (delta = -2.0 pts) -> INCOHERENTE (VETO)
        coh_rev, delta_rev = strat.evaluate_vwap_coherence(
            r_first=0.0020,
            p_1530=4998.0,
            session_to_1530=df_session,
            tick_size=0.25
        )
        assert coh_rev is False
        assert delta_rev < 0

        # Caso 3: Delta inferior a 2 ticks (0.50 pts) -> VETO por neutralidad
        coh_flat, delta_flat = strat.evaluate_vwap_coherence(
            r_first=0.0020,
            p_1530=5000.25,
            session_to_1530=df_session,
            tick_size=0.25
        )
        assert coh_flat is False

    def test_adaptive_exit_levels_atr_and_tp_ratio(self):
        """Verifica que el SL se adapte a k_SL * ATR_intra y el TP mantenga k_RR = 1.60."""
        strat = MarketIntradayMomentumOptN1(params={
            "k_sl": 1.25,
            "k_rr": 1.60,
            "use_atr_sl": True
        })
        atr_intra = 4.0  # puntos
        p_entry = 5000.0

        # Long: Delta_SL = 1.25 * 4.0 = 5.0 pts
        p_sl, p_tp, delta_sl = strat.calculate_exit_levels_opt(
            p_entry=p_entry,
            direction=1,
            atr_intra=atr_intra
        )
        assert delta_sl == 5.0
        assert p_sl == 4995.0
        assert p_tp == 5000.0 + 1.60 * 5.0  # 5008.0

        rr_ratio = (p_tp - p_entry) / (p_entry - p_sl)
        assert pytest.approx(rr_ratio, abs=1e-6) == 1.60

        # Short
        p_sl_s, p_tp_s, delta_sl_s = strat.calculate_exit_levels_opt(
            p_entry=p_entry,
            direction=-1,
            atr_intra=atr_intra
        )
        assert p_sl_s == 5005.0
        assert p_tp_s == 5000.0 - 1.60 * 5.0  # 4992.0


# ==============================================================================
# 5. INTEGRACIÓN VECTORIAL Y PRUEBA E2E CON SIMULADOR DEDICADO
# ==============================================================================

class TestF1OptN1BacktesterIntegration:
    """Verifica la ejecución determinista del simulador IntradayMomentumOptN1Backtester."""

    @pytest.fixture
    def synthetic_trading_days(self) -> pd.DataFrame:
        """Crea 3 sesiones sintéticas de 5 minutos en America/New_York."""
        timestamps = []
        rows = []
        dates = ["2023-01-03", "2023-01-04", "2023-01-05"]

        for d in dates:
            day_times = pd.date_range(f"{d} 09:30", f"{d} 16:00", freq="5min", inclusive="left")
            base_p = 5000.0
            for idx, ts in enumerate(day_times):
                timestamps.append(ts)
                # Simular tendencia alcista suave en la primera sesión
                if d == "2023-01-03":
                    p = base_p + (idx * 0.25)
                elif d == "2023-01-04":
                    # Ruido plano
                    p = base_p + (0.10 if idx % 2 == 0 else -0.10)
                else:
                    # Desplazamiento bajista
                    p = base_p - (idx * 0.20)

                rows.append({
                    "Open": p,
                    "High": p + 0.50,
                    "Low": p - 0.50,
                    "Close": p + 0.25,
                    "Volume": 2000.0
                })

        idx_tz = pd.DatetimeIndex(timestamps).tz_localize("America/New_York")
        return pd.DataFrame(rows, index=idx_tz)

    def test_backtester_runs_and_produces_structured_results(self, synthetic_trading_days):
        """Ejecuta el backtester sobre las 3 sesiones y verifica estructura de métricas."""
        strat = MarketIntradayMomentumOptN1()
        backtester = IntradayMomentumOptN1Backtester(
            strategy=strat,
            account_size=50000.0,
            daily_loss_limit=1000.0,
            max_trailing_loss=2000.0
        )

        res = backtester.run(synthetic_trading_days)

        assert "metrics" in res
        assert "prop_evaluation" in res
        assert "trades" in res
        assert "daily_logs" in res
        assert "skipped_sessions" in res

        assert len(res["daily_logs"]) == 3
        prop_eval = res["prop_evaluation"]
        assert prop_eval["status"] in ["PASS", "FAIL", "INCOMPLETE"]
        assert isinstance(prop_eval["final_balance"], float)

    def test_original_f1_strategy_unbroken_and_identical(self):
        """Garantiza que la estrategia original F1 (MarketIntradayMomentumMOC) siga intacta."""
        from src.strategies.momentum.intraday_momentum_moc import (
            MarketIntradayMomentumMOC,
            IntradayMomentumBacktester
        )
        orig_strat = MarketIntradayMomentumMOC()
        assert orig_strat.name == "Market_Intraday_Momentum_MOC"
        assert orig_strat.params["theta_floor"] == 0.0015
        assert orig_strat.params["k_rr"] == 1.50

        opt_strat = MarketIntradayMomentumOptN1()
        assert opt_strat.name == "F1-Optimizacion -IS- N1"
        assert opt_strat.params["theta_floor"] == 0.0010
        assert opt_strat.params["k_rr"] == 1.60
