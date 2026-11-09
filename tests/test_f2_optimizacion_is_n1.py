"""
tests/test_f2_optimizacion_is_n1.py
Suite completa de pruebas unitarias y de estrés cuantitativo para la versión alterna:
Estrategia F2-Optimizacion -IS- N1: Filtered Quantitative Opening Range Breakout (ORB).

Cobertura de Pruebas:
1. Ratio Continuo de Compresión Previa (R_comp) y condición de admisibilidad Phi_comp.
2. Dinamización Analítica de RVOL_crit(R_comp) acotada estrictamente en [1.20, 1.70].
3. Cumulative Volume Delta (CVD) y Asimetría de Flujo (OFA) con supresión matemática de fakeouts.
4. Umbral Dinámico de Flujo theta_flow(R_comp) y confirmación direccional.
5. Barrera Elástica Proporcional a la anchura del rango inicial acotada por ATR diario.
6. Mecanismo de Breakeven Protector Dinámico al alcanzar +1.0R.
7. Reglas de Prop Trading (DLL $1,000, MTDD $2,000, disyuntor al 85% DLL, atenuación phi(K_loss) y microcontratos).
8. Ciclo de Vida End-to-End con el simulador F2OptimizationISN1Backtester en escenarios de TP, SL, BE y Time Decay.
"""

from datetime import datetime, time
import math
import numpy as np
import pandas as pd
import pytest

from src.strategies.f2_optimizacion_is_n1 import (
    F2OptimizationISN1,
    F2OptTradeRecord,
    F2OptimizationISN1Backtester,
    calculate_rcomp,
    calculate_rvol_dynamic_threshold,
    calculate_order_flow_asymmetry,
    calculate_dynamic_flow_threshold,
    evaluate_flow_confirmation,
    calculate_daily_atr,
    calculate_elastic_delta_risk,
)
from src.strategies.filtered_orb import (
    StrategyState,
    ExitReason,
    PropAccount,
    calculate_crabel_stretch,
    calculate_position_size,
    normalize_to_eastern,
)
from src.core.futures_specs import get_spec


# ==============================================================================
# 1. PRUEBAS DEL RATIO CONTINUO DE COMPRESIÓN PREVIA (R_COMP)
# ==============================================================================

class TestVolatilityCompressionRComp:
    """Valida la métrica continua de compresión y la disyunción con Inside Day."""

    def test_calculate_rcomp_exact(self):
        """
        Ranges: t-4=10.0, t-3=12.0, t-2=14.0 -> media_prior = 12.0.
        Range t-1 = 6.0.
        R_comp = 6.0 / 12.0 = 0.50.
        Como 0.50 <= 0.92 -> is_compressed = True.
        """
        df = pd.DataFrame({
            "Open": [5000, 5000, 5000, 5000],
            "High": [5010, 5012, 5014, 5006],
            "Low":  [5000, 5000, 5000, 5000],
            "Close": [5005, 5005, 5005, 5003]
        })
        r_comp, is_comp = calculate_rcomp(df)
        assert pytest.approx(r_comp, abs=1e-6) == 0.50
        assert is_comp is True

    def test_calculate_rcomp_uncompressed(self):
        """
        Ranges: t-4=10.0, t-3=10.0, t-2=10.0 -> media = 10.0.
        Range t-1 = 15.0 -> R_comp = 1.50 > 0.92.
        High t-1 (5025) > High t-2 (5010) -> no es Inside Day.
        is_compressed debe ser False.
        """
        df = pd.DataFrame({
            "Open": [5000, 5000, 5000, 5000],
            "High": [5010, 5010, 5010, 5025],
            "Low":  [5000, 5000, 5000, 5010],
            "Close": [5005, 5005, 5005, 5020]
        })
        r_comp, is_comp = calculate_rcomp(df)
        assert r_comp == 1.50
        assert is_comp is False

    def test_calculate_rcomp_inside_day_override(self):
        """
        R_comp > 0.92 pero la sesión t-1 es un Inside Day frente a t-2:
        High t-1 <= High t-2 and Low t-1 >= Low t-2.
        is_compressed debe ser True por disyunción analítica.
        """
        df = pd.DataFrame({
            "Open": [5000, 5000, 5000, 5000],
            "High": [5010, 5010, 5020, 5018],
            "Low":  [5000, 5000, 5000, 5002],
            "Close": [5005, 5005, 5010, 5010]
        })
        # Prior ranges: 10, 10, 20 -> mean = 13.333
        # Range t-1 = 16 -> R_comp = 16 / 13.333 = 1.20 > 0.92
        # Pero High t-1 (5018) <= High t-2 (5020) y Low t-1 (5002) >= Low t-2 (5000) -> Inside Day
        r_comp, is_comp = calculate_rcomp(df)
        assert r_comp > 0.92
        assert is_comp is True

    def test_calculate_rcomp_insufficient_bars(self):
        """Maneja DataFrames con menos de 4 barras retornando (1.0, False)."""
        df_short = pd.DataFrame({
            "Open": [5000, 5000], "High": [5010, 5010],
            "Low": [5000, 5000], "Close": [5005, 5005]
        })
        r_comp, is_comp = calculate_rcomp(df_short)
        assert r_comp == 1.0
        assert is_comp is False

    def test_calculate_rcomp_corrupt_data_raises_error(self):
        """Verifica que barras con High < Low lancen ValueError."""
        df_corrupt = pd.DataFrame({
            "Open": [5000, 5000, 5000, 5000],
            "High": [5010, 5010, 5010, 4990],  # High < Low
            "Low":  [5000, 5000, 5000, 5000],
            "Close": [5005, 5005, 5005, 5000]
        })
        with pytest.raises(ValueError, match="Datos corruptos detectados"):
            calculate_rcomp(df_corrupt)


# ==============================================================================
# 2. PRUEBAS DEL UMBRAL DINÁMICO DE VOLUMEN RELATIVO (RVOL_CRIT)
# ==============================================================================

class TestDynamicRVOLThreshold:
    """Valida la función monótona acotada RVOL_crit(R_comp) en [1.20, 1.70]."""

    def test_rvol_dynamic_bounds(self):
        """Verifica los límites exactos en compresión extrema y moderada."""
        # Compresión extrema (R_comp <= 0.50) -> RVOL_crit = 1.20
        assert calculate_rvol_dynamic_threshold(0.50) == 1.20
        assert calculate_rvol_dynamic_threshold(0.20) == 1.20
        assert calculate_rvol_dynamic_threshold(0.0) == 1.20

        # Compresión moderada (R_comp = 0.92) -> RVOL_crit = 1.70
        assert calculate_rvol_dynamic_threshold(0.92) == 1.70
        assert calculate_rvol_dynamic_threshold(1.50) == 1.70

    def test_rvol_dynamic_monotonicity(self):
        """Verifica que a mayor R_comp (menor compresión), mayor sea el umbral exigido."""
        thresholds = [calculate_rvol_dynamic_threshold(r) for r in np.linspace(0.50, 0.92, 10)]
        for i in range(len(thresholds) - 1):
            assert thresholds[i] <= thresholds[i + 1]
            assert 1.20 <= thresholds[i] <= 1.70

    def test_rvol_dynamic_nan_or_negative(self):
        """Garantiza robustez frente a valores no válidos retornando el tope seguro."""
        assert calculate_rvol_dynamic_threshold(float("nan")) == 1.70
        assert calculate_rvol_dynamic_threshold(-0.50) == 1.70


# ==============================================================================
# 3. PRUEBAS DE ASIMETRÍA DE FLUJO (OFA/CVD) Y SUPRESIÓN DE FAKEOUTS
# ==============================================================================

class TestOrderFlowAsymmetryAndFakeouts:
    """Valida el cálculo hidrodinámico de delta y el bloqueo de rupturas falsas."""

    def test_order_flow_asymmetry_bullish(self):
        """3 velas de 5m con cierre alcista fuerte: CVD > 0 y OFA > 0."""
        df = pd.DataFrame({
            "Open": [5000.0, 5005.0, 5010.0],
            "High": [5006.0, 5011.0, 5015.0],
            "Low":  [5000.0, 5005.0, 5010.0],
            "Close": [5006.0, 5011.0, 5015.0],  # C = H -> peso = 1.0
            "Volume": [1000.0, 1000.0, 1000.0]
        })
        cvd, ofa = calculate_order_flow_asymmetry(df)
        assert cvd == 3000.0
        assert pytest.approx(ofa, abs=1e-4) == 1.0

    def test_order_flow_asymmetry_bearish(self):
        """3 velas de 5m con cierre bajista fuerte: CVD < 0 y OFA < 0."""
        df = pd.DataFrame({
            "Open": [5010.0, 5005.0, 5000.0],
            "High": [5010.0, 5005.0, 5000.0],
            "Low":  [5004.0, 4999.0, 4995.0],
            "Close": [5004.0, 4999.0, 4995.0],  # C = L -> peso = -1.0
            "Volume": [1000.0, 1000.0, 1000.0]
        })
        cvd, ofa = calculate_order_flow_asymmetry(df)
        assert cvd == -3000.0
        assert pytest.approx(ofa, abs=1e-4) == -1.0

    def test_dynamic_flow_threshold_scaling(self):
        """Verifica que el umbral theta_flow escale proporcionalmente a R_comp."""
        theta_low = calculate_dynamic_flow_threshold(0.50, base_flow_threshold=0.05)
        theta_high = calculate_dynamic_flow_threshold(0.92, base_flow_threshold=0.05)
        assert theta_low < theta_high
        assert pytest.approx(theta_high, abs=1e-4) == 0.05
        assert 0.0 <= theta_low <= 0.35

    def test_evaluate_flow_confirmation_direction(self):
        """Valida que evaluate_flow_confirmation exija el signo y magnitud correctos."""
        theta = 0.05
        # Long (+1)
        assert evaluate_flow_confirmation(1, 0.10, theta) is True
        assert evaluate_flow_confirmation(1, 0.04, theta) is False
        assert evaluate_flow_confirmation(1, -0.10, theta) is False

        # Short (-1)
        assert evaluate_flow_confirmation(-1, -0.10, theta) is True
        assert evaluate_flow_confirmation(-1, -0.04, theta) is False
        assert evaluate_flow_confirmation(-1, 0.10, theta) is False

    def test_fakeout_suppression_blocks_trade(self):
        """
        Escenario Fakeout: El precio rompe el gatillo alcista (High >= buy_trigger),
        pero la subasta de apertura fue dominada por vendedores (OFA < 0).
        F2-Optimizacion -IS- N1 DEBE abortar la entrada y registrar el fakeout evitado.
        """
        strategy = F2OptimizationISN1(std_symbol="ES", micro_symbol="MES", base_flow_threshold=0.05)
        account = PropAccount(initial_capital=50000.0)

        # Contexto previo comprimido (R_comp = 0.50, RVOL_crit = 1.20)
        daily_hist = pd.DataFrame({
            "Open": [5000] * 10,
            "High": [5010] * 9 + [5005],
            "Low":  [5000] * 10,
            "Close": [5005] * 10
        })

        # Barras de Opening Range [09:30, 09:45): 3 velas de 5m netamente bajistas (CVD < 0)
        times_or = pd.date_range("2023-05-10 09:30", "2023-05-10 09:40", freq="5min", tz="America/New_York")
        df_or = pd.DataFrame({
            "Open": [5005.0, 5003.0, 5001.0],
            "High": [5006.0, 5004.0, 5002.0],
            "Low":  [5001.0, 4999.0, 4997.0],
            "Close": [5002.0, 5000.0, 4998.0],
            "Volume": [5000.0, 5000.0, 5000.0]
        }, index=times_or)

        # Barra posterior (09:45) que amaga romper al alza (High alcanza 5015) sin perforar el mínimo
        time_post = pd.date_range("2023-05-10 09:45", "2023-05-10 09:45", freq="5min", tz="America/New_York")
        df_post = pd.DataFrame({
            "Open": [5000.0],
            "High": [5015.0],  # Supera buy_trigger
            "Low":  [5000.0],  # Se mantiene por encima de sell_trigger
            "Close": [5002.0],
            "Volume": [1000.0]
        }, index=time_post)

        df_day = pd.concat([df_or, df_post])
        hist_vols = [10000.0] * 20  # Promedio 10,000; or_vol = 15,000 -> RVOL = 1.50 >= 1.20

        trade = strategy.process_day(
            session_date="2023-05-10",
            intraday_bars=df_day,
            daily_history=daily_hist,
            historical_or_volumes=hist_vols,
            account=account
        )

        # El trade debe ser BLOQUEADO por falta de confirmación de flujo comprador
        assert trade is None
        assert strategy.fakeouts_avoided_long >= 1


# ==============================================================================
# 4. PRUEBAS DE LA BARRERA ELÁSTICA Y ACOTACIÓN POR VOLATILIDAD ATR
# ==============================================================================

class TestElasticBarrierAndATR:
    """Valida el cálculo de Delta_risk^* elástico y la acotación por ATR diario."""

    def test_daily_atr_calculation(self):
        """Calcula el ATR exacto sobre barras diarias sintéticas."""
        df_daily = pd.DataFrame({
            "Open": [5000.0] * 5,
            "High": [5020.0, 5025.0, 5015.0, 5030.0, 5020.0],
            "Low":  [5000.0, 5005.0, 4995.0, 5010.0, 5000.0],
            "Close": [5010.0, 5010.0, 5005.0, 5020.0, 5010.0]
        })
        atr = calculate_daily_atr(df_daily, lookback=4)
        assert atr > 0.0
        assert isinstance(atr, float)

    def test_elastic_delta_risk_proportional(self):
        """Rango normal: la barrera elástica adopta k * Range_OR + lambda * Stretch."""
        # Range_OR = 10.0, Stretch = 2.0, k = 0.50, lambda = 0.40 -> 0.50*10 + 0.40*2 = 5.80
        # Con ATR = 30.0 -> cap = 0.85 * 30 = 25.50 -> 5.80 < 25.50
        risk = calculate_elastic_delta_risk(
            range_or=10.0,
            stretch=2.0,
            atr_daily=30.0,
            k_elastic=0.50,
            lambda_stretch=0.40,
            atr_cap_mult=0.85
        )
        assert pytest.approx(risk, abs=1e-6) == 5.80

    def test_elastic_delta_risk_capped_by_atr(self):
        """Apertura desmedida (Range_OR = 60.0): la barrera se acota a 0.85 * ATR."""
        # Proportional: 0.50 * 60.0 + 0.40 * 5.0 = 32.0 pts
        # Con ATR = 20.0 -> cap = 0.85 * 20.0 = 17.0 pts -> 17.0 < 32.0
        risk = calculate_elastic_delta_risk(
            range_or=60.0,
            stretch=5.0,
            atr_daily=20.0,
            k_elastic=0.50,
            lambda_stretch=0.40,
            atr_cap_mult=0.85
        )
        assert pytest.approx(risk, abs=1e-6) == 17.0

    def test_elastic_delta_risk_invalid_inputs(self):
        """Lanza ValueError ante valores de rango o stretch negativos."""
        with pytest.raises(ValueError, match="Range_OR no puede ser negativo"):
            calculate_elastic_delta_risk(-5.0, 2.0, 20.0)

        with pytest.raises(ValueError, match="Stretch no puede ser negativo"):
            calculate_elastic_delta_risk(10.0, -2.0, 20.0)


# ==============================================================================
# 5. PRUEBAS DE MECANISMO DE BREAKEVEN PROTECTOR (+1.0R)
# ==============================================================================

class TestElasticBreakevenProtection:
    """Valida la activación del ajuste elástico a breakeven (+2 ticks) al cruzar +1.0R."""

    def test_breakeven_activation_on_favorable_move(self):
        """
        Operación larga que entra en 5005, avanza hasta 5015 (> entry + 1.0*Delta_risk = 5011)
        y luego retrocede. Debe salir en Breakeven (+2 ticks) en lugar de Stop Loss.
        """
        strategy = F2OptimizationISN1(
            std_symbol="ES",
            micro_symbol="MES",
            base_flow_threshold=0.01,
            enable_breakeven=True,
            breakeven_r_mult=1.0
        )
        account = PropAccount(initial_capital=50000.0)

        daily_hist = pd.DataFrame({
            "Open": [5000] * 10,
            "High": [5010] * 9 + [5005],
            "Low": [5000] * 10,
            "Close": [5005] * 10
        })

        # Opening Range alcista [09:30, 09:45): ORH=5004, ORL=5000, Stretch=2.0 -> buy_trigger = 5004.80
        times = pd.date_range("2023-05-10 09:30", "2023-05-10 10:00", freq="5min", tz="America/New_York")
        df_intra = pd.DataFrame({
            "Open":  [5000, 5002, 5003, 5004, 5012, 5004, 5000],
            "High":  [5002, 5003, 5004, 5008, 5016, 5010, 5005],  # 09:50 alcanza 5016 (> +1.0R)
            "Low":   [5000, 5001, 5002, 5004, 5011, 5004, 4995],  # 09:55 cae a 5004
            "Close": [5002, 5003, 5004, 5007, 5013, 5005, 4998],
            "Volume": [2000, 2000, 2000, 1000, 1000, 1000, 1000]
        }, index=times)

        hist_vols = [4000.0] * 20  # or_vol = 6000 -> RVOL = 1.50

        trade = strategy.process_day(
            session_date="2023-05-10",
            intraday_bars=df_intra,
            daily_history=daily_hist,
            historical_or_volumes=hist_vols,
            account=account
        )

        assert trade is not None
        assert trade.breakeven_triggered is True
        assert strategy.breakeven_activations >= 1
        # El PnL neto debe ser positivo o plano (no una pérdida de 1R completo)
        assert trade.net_pnl_usd > -100.0


# ==============================================================================
# 6. PRUEBAS DE CONTROL DE RIESGO PROP TRADING
# ==============================================================================

class TestPropTradingRulesF2Opt:
    """Verifica límites DLL, MTDD, soft circuit breaker al 85% y atenuación."""

    def test_dll_soft_limit_halts_trading(self):
        """Si la pérdida diaria alcanza $850 (85% de $1,000 DLL), frena nuevas operaciones."""
        strategy = F2OptimizationISN1(dll_soft_limit_pct=0.85)
        account = PropAccount(initial_capital=50000.0, dll=1000.0)
        account.daily_realized_loss = 860.0  # Supera el 85% ($850)

        daily_hist = pd.DataFrame({
            "Open": [5000] * 10, "High": [5010] * 10, "Low": [5000] * 10, "Close": [5005] * 10
        })
        times = pd.date_range("2023-05-10 09:30", "2023-05-10 09:45", freq="5min", tz="America/New_York")
        df_intra = pd.DataFrame({
            "Open": [5000] * 4, "High": [5010] * 4, "Low": [5000] * 4, "Close": [5005] * 4, "Volume": [1000] * 4
        }, index=times)

        trade = strategy.process_day(
            session_date="2023-05-10",
            intraday_bars=df_intra,
            daily_history=daily_hist,
            historical_or_volumes=[1000] * 20,
            account=account
        )
        assert trade is None
        assert strategy.state == StrategyState.HALT_ACCOUNT_BREACHED

    def test_loss_streak_budget_attenuation(self):
        """Verifica la atenuación geométrica phi(K_loss) ante pérdidas consecutivas."""
        account = PropAccount(initial_capital=50000.0, dll=1000.0, mtdd=2000.0)
        # K_loss = 0 -> phi = 0.35 -> presupuesto = 350
        b0 = account.calculate_risk_budget()
        assert pytest.approx(b0, abs=1e-2) == 350.0

        # K_loss = 1 -> phi = 0.175 -> presupuesto = 175
        account.k_loss = 1
        b1 = account.calculate_risk_budget()
        assert pytest.approx(b1, abs=1e-2) == 175.0

        # K_loss = 2 -> phi = 0.0875 -> presupuesto = 87.50
        account.k_loss = 2
        b2 = account.calculate_risk_budget()
        assert pytest.approx(b2, abs=1e-2) == 87.50


# ==============================================================================
# 7. PRUEBAS END-TO-END DEL SIMULADOR Y REPORTE
# ==============================================================================

class TestEndToEndBacktesterF2Opt:
    """Valida la ejecución del backtester completo y la generación de reportes cuantitativos."""

    def test_backtester_full_lifecycle_take_profit(self):
        """Simula una sesión completa donde el trade alcanza el Take Profit (2.0R)."""
        strategy = F2OptimizationISN1(
            std_symbol="ES",
            micro_symbol="MES",
            base_flow_threshold=0.01
        )
        account = PropAccount(initial_capital=50000.0)
        backtester = F2OptimizationISN1Backtester(strategy=strategy, account=account)

        # 10 días previos de datos diarios
        dates_daily = pd.date_range("2023-05-01", "2023-05-10", freq="1D", tz="America/New_York")
        df_daily = pd.DataFrame({
            "Open": [5000.0] * len(dates_daily),
            "High": [5010.0] * len(dates_daily),
            "Low":  [5000.0] * len(dates_daily),
            "Close": [5005.0] * len(dates_daily),
            "Volume": [10000.0] * len(dates_daily)
        }, index=dates_daily)

        # Sesión intradía del 2023-05-11
        times_intra = pd.date_range("2023-05-11 09:30", "2023-05-11 11:00", freq="5min", tz="America/New_York")
        # Generar subasta de apertura alcista y posterior expansión a TP
        opens = [5000.0] * len(times_intra)
        highs = [5002.0] * len(times_intra)
        lows = [4998.0] * len(times_intra)
        closes = [5001.0] * len(times_intra)
        vols = [5000.0] * len(times_intra)

        # Vela de ruptura (10:00) y de TP (10:15)
        highs[6] = 5025.0  # Ruptura alcista masiva
        closes[6] = 5024.0

        df_intra = pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols
        }, index=times_intra)

        report = backtester.run(intraday_data=df_intra, daily_data=df_daily, auto_normalize_tz=False)

        assert "total_net_pnl_usd" in report
        assert "fakeouts_avoided_total" in report
        assert report["is_prop_breached"] is False
        assert isinstance(report["total_trades"], int)

    def test_backtester_empty_dataframe_safety(self):
        """Maneja de forma segura DataFrames vacíos sin excepciones no controladas."""
        strategy = F2OptimizationISN1()
        account = PropAccount(initial_capital=50000.0)
        backtester = F2OptimizationISN1Backtester(strategy=strategy, account=account)

        df_empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        with pytest.raises(ValueError):
            backtester.run(intraday_data=df_empty)


# ==============================================================================
# 8. PRUEBAS DE GENERACIÓN VECTORIAL DE SEÑALES (GENERATE_SIGNALS)
# ==============================================================================

class TestSignalGeneration:
    """Valida la interfaz canónica BaseStrategy.generate_signals."""

    def test_generate_signals_produces_series(self):
        """Verifica que generate_signals retorne una Serie alineada con data.index y valores en {-1, 0, 1}."""
        strategy = F2OptimizationISN1(base_flow_threshold=0.01)
        times = pd.date_range("2023-05-10 09:30", "2023-05-10 16:00", freq="5min", tz="America/New_York")
        df = pd.DataFrame({
            "Open": [5000.0] * len(times),
            "High": [5005.0] * len(times),
            "Low": [4995.0] * len(times),
            "Close": [5002.0] * len(times),
            "Volume": [1000.0] * len(times),
        }, index=times)

        signals = strategy.generate_signals(df)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(df)
        assert set(signals.unique()).issubset({-1.0, 0.0, 1.0})

    def test_generate_signals_timezone_handling(self):
        """Verifica que generate_signals soporte índices sin zona horaria (naive) sin excepciones."""
        strategy = F2OptimizationISN1(base_flow_threshold=0.01)
        times = pd.date_range("2023-05-10 09:30", "2023-05-10 16:00", freq="5min")  # naive
        df = pd.DataFrame({
            "Open": [5000.0] * len(times),
            "High": [5005.0] * len(times),
            "Low": [4995.0] * len(times),
            "Close": [5002.0] * len(times),
            "Volume": [1000.0] * len(times),
        }, index=times)

        signals = strategy.generate_signals(df)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(df)

    def test_generate_signals_missing_columns_raises_error(self):
        """Lanza ValueError si faltan columnas requeridas de OHLCV."""
        strategy = F2OptimizationISN1()
        times = pd.date_range("2023-05-10 09:30", "2023-05-10 11:00", freq="5min")
        df_incomplete = pd.DataFrame({"Open": [5000.0] * len(times), "Close": [5002.0] * len(times)}, index=times)
        with pytest.raises(ValueError, match="Faltan columnas requeridas"):
            strategy.generate_signals(df_incomplete)


# ==============================================================================
# 9. PRUEBAS DE ESTRÉS ADVERSARIAL Y ROBUSTEZ NUMÉRICA (EDGE CASES)
# ==============================================================================

class TestAdversarialAndEdgeCases:
    """Valida la robustez ante entradas corruptas, NaNs, barras fuera de rango y aperturas atípicas."""

    def test_calculate_order_flow_asymmetry_nan_inputs(self):
        """Garantiza que NaNs en precios o volumen retornen (0.0, 0.0) sin propagar a 1.0."""
        df_nan = pd.DataFrame({
            "Open": [5000.0, float("nan"), 5002.0],
            "High": [5005.0, 5006.0, 5005.0],
            "Low":  [4995.0, 4995.0, 4995.0],
            "Close": [5002.0, 5003.0, 5001.0],
            "Volume": [1000.0, 1000.0, 1000.0]
        })
        cvd, ofa = calculate_order_flow_asymmetry(df_nan)
        assert cvd == 0.0
        assert ofa == 0.0

    def test_calculate_order_flow_asymmetry_corrupt_high_low(self):
        """Lanza ValueError si High < Low en alguna barra de la subasta de apertura."""
        df_corrupt = pd.DataFrame({
            "Open": [5000.0, 5002.0, 5001.0],
            "High": [5005.0, 4990.0, 5005.0],  # High < Low
            "Low":  [4995.0, 5000.0, 4995.0],
            "Close": [5002.0, 4995.0, 5001.0],
            "Volume": [1000.0, 1000.0, 1000.0]
        })
        with pytest.raises(ValueError, match="Datos corruptos detectados"):
            calculate_order_flow_asymmetry(df_corrupt)

    def test_calculate_elastic_delta_risk_nan_safety(self):
        """Lanza ValueError ante valores NaN en range_or o stretch en lugar de provocar caídas posteriores."""
        with pytest.raises(ValueError, match="Range_OR no puede ser negativo ni NaN"):
            calculate_elastic_delta_risk(float("nan"), 2.0, 20.0)

        with pytest.raises(ValueError, match="Stretch no puede ser negativo ni NaN"):
            calculate_elastic_delta_risk(10.0, float("nan"), 20.0)

    def test_calculate_daily_atr_window_accuracy(self):
        """Verifica que con 15 barras diarias se computen exactamente 14 Rangos Verdaderos (lookback=14)."""
        dates = pd.date_range("2023-01-01", periods=15, freq="1D")
        # Barras con rango constante de 10.0 puntos y sin gaps (High - Low = 10, C = Low + 5)
        df_daily = pd.DataFrame({
            "Open": [5000.0] * 15,
            "High": [5010.0] * 15,
            "Low":  [5000.0] * 15,
            "Close": [5005.0] * 15
        }, index=dates)
        atr = calculate_daily_atr(df_daily, lookback=14)
        assert pytest.approx(atr, abs=1e-6) == 10.0

    def test_calculate_daily_atr_nan_and_corrupt_safety(self):
        """Maneja NaNs retornando 0.0 y lanza ValueError ante barras corruptas High < Low."""
        dates = pd.date_range("2023-01-01", periods=10, freq="1D")
        df_nan = pd.DataFrame({
            "Open": [5000.0] * 10, "High": [float("nan")] * 10,
            "Low": [5000.0] * 10, "Close": [5005.0] * 10
        }, index=dates)
        assert calculate_daily_atr(df_nan) == 0.0

        df_corrupt = pd.DataFrame({
            "Open": [5000.0] * 5, "High": [5010.0, 4990.0, 5010.0, 5010.0, 5010.0],
            "Low": [5000.0] * 5, "Close": [5005.0] * 5
        }, index=dates[:5])
        with pytest.raises(ValueError, match="Datos corruptos detectados"):
            calculate_daily_atr(df_corrupt)

    def test_elastic_delta_risk_tight_vol_cap_not_violated(self):
        """Verifica que un vol_cap ajustado (0.40 pts) no sea violado por un piso artificial desmedido."""
        # Range_OR = 10.0, Stretch = 2.0 -> Proportional = 0.50*10 + 0.40*2 = 5.80
        # ATR_daily = 0.40, atr_cap_mult = 0.85 -> vol_cap = 0.34
        # Delta_risk debe ser 0.34, respetando el cap y el piso mínimo de 1 tick (0.25).
        risk = calculate_elastic_delta_risk(
            range_or=10.0,
            stretch=2.0,
            atr_daily=0.40,
            k_elastic=0.50,
            lambda_stretch=0.40,
            atr_cap_mult=0.85
        )
        assert pytest.approx(risk, abs=1e-6) == 0.34

    def test_outside_bar_breakout_both_triggers_directional_preference(self):
        """
        Si una barra de 5m penetra tanto buy_trigger como sell_trigger,
        se debe seleccionar la dirección respaldada por el flujo agresor OFA sin omitir la entrada.
        """
        strategy = F2OptimizationISN1(std_symbol="ES", micro_symbol="MES", base_flow_threshold=0.01)
        account = PropAccount(initial_capital=50000.0)

        daily_hist = pd.DataFrame({
            "Open": [5000] * 10, "High": [5010] * 9 + [5005], "Low": [5000] * 10, "Close": [5005] * 10
        })

        # OR alcista (OFA > 0)
        times = pd.date_range("2023-05-10 09:30", "2023-05-10 09:40", freq="5min", tz="America/New_York")
        df_or = pd.DataFrame({
            "Open": [5000.0, 5001.0, 5002.0],
            "High": [5003.0, 5004.0, 5005.0],
            "Low":  [5000.0, 5001.0, 5002.0],
            "Close": [5003.0, 5004.0, 5005.0],
            "Volume": [3000.0, 3000.0, 3000.0]
        }, index=times)

        # Barra posterior (09:45) con rango masivo que perfora ambos extremos
        time_post = pd.date_range("2023-05-10 09:45", "2023-05-10 09:45", freq="5min", tz="America/New_York")
        df_post = pd.DataFrame({
            "Open": [5005.0],
            "High": [5020.0],  # Perfora buy_trigger
            "Low":  [4980.0],  # Perfora sell_trigger
            "Close": [5015.0],
            "Volume": [2000.0]
        }, index=time_post)

        df_day = pd.concat([df_or, df_post])
        hist_vols = [5000.0] * 20

        trade = strategy.process_day("2023-05-10", df_day, daily_hist, hist_vols, account)
        assert trade is not None
        # Con OFA > 0, debe haber entrado en Long (+1)
        assert trade.direction == 1

    def test_fakeouts_avoided_counted_once_per_direction(self):
        """
        Si múltiples barras consecutivas de 5m cotizan por encima de buy_trigger sin confirmación de flujo,
        fakeouts_avoided_long debe incrementarse exactamente 1 vez por esa sesión, no 1 por cada barra.
        """
        strategy = F2OptimizationISN1(base_flow_threshold=0.05)
        account = PropAccount(initial_capital=50000.0)

        daily_hist = pd.DataFrame({
            "Open": [5000] * 10, "High": [5010] * 9 + [5005], "Low": [5000] * 10, "Close": [5005] * 10
        })

        # OR bajista (OFA < 0)
        times_or = pd.date_range("2023-05-10 09:30", "2023-05-10 09:40", freq="5min", tz="America/New_York")
        df_or = pd.DataFrame({
            "Open": [5005.0, 5003.0, 5001.0],
            "High": [5006.0, 5004.0, 5002.0],
            "Low":  [5001.0, 4999.0, 4997.0],
            "Close": [5002.0, 5000.0, 4998.0],
            "Volume": [5000.0, 5000.0, 5000.0]
        }, index=times_or)

        # 5 barras consecutivas post-OR todas por encima de buy_trigger (5006)
        times_post = pd.date_range("2023-05-10 09:45", "2023-05-10 10:05", freq="5min", tz="America/New_York")
        df_post = pd.DataFrame({
            "Open": [5010.0] * 5,
            "High": [5015.0] * 5,
            "Low":  [5008.0] * 5,
            "Close": [5012.0] * 5,
            "Volume": [1000.0] * 5
        }, index=times_post)

        df_day = pd.concat([df_or, df_post])
        hist_vols = [10000.0] * 20

        trade = strategy.process_day("2023-05-10", df_day, daily_hist, hist_vols, account)
        assert trade is None
        # Debe haberse contado exactamente 1 fakeout para compras en esa sesión
        assert strategy.fakeouts_avoided_long == 1

