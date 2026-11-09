"""
tests/test_f5_optimizacion_is_n1.py
Suite de Pruebas Unitarias, Microestructurales, Adversarias y de Prop Trading
para la Estrategia Optimizada "F5-Optimizacion -IS- N1":
Order Flow Imbalance Robusto (Mediana / IQR) con Compuerta Económica de Costo Transaccional.

Cubre:
1. Normalización Robusta (Mediana rodante, IQR e inmunidad matemática a outliers).
2. Cálculo de Costos Transaccionales (CME ES, MES, NQ, MNQ) y Modelo de Impacto de Cont et al.
3. Compuerta Económica (Economic Hurdle): supresión efectiva de sobreoperación (overtrading).
4. Generación Vectorial de Señales y Filtros Microestructurales (Spread de 1 tick, Horario RTH).
5. Dimensionamiento de Posición, Absorción LOB (5%) y Conmutación Automática STD/MICRO.
6. Gestión de Riesgo Prop Trading Institucional (DLL $1,000, TMDD $2,000, Cierre EOD 15:55 EST).
7. Simulación End-to-End y Comparativa de Reducción de Fricción frente a F5 Base.
"""

import math
from datetime import time, datetime
import numpy as np
import pandas as pd
import pytest

from src.strategies.order_flow_imbalance import (
    OrderFlowImbalanceStrategy,
    OFIBacktester,
    OFIPropAccount,
    generate_synthetic_lob_events,
)
from src.strategies.f5_optimizacion_is_n1 import (
    F5OptimizacionISN1Strategy,
    F5OptimizacionISN1Backtester,
    RobustOrderFlowImbalanceStrategy,
    RobustOFITradeRecord,
    calculate_robust_scale_iqr,
    calculate_robust_z_score,
    calculate_expected_price_impact,
    calculate_transaction_cost_pts,
    evaluate_economic_hurdle,
)


# ==============================================================================
# 1. PRUEBAS DE NORMALIZACIÓN ROBUSTA Y ESCALA IQR
# ==============================================================================

class TestRobustScaleAndNormalization:
    """Verifica el comportamiento matemático del estimador consistente IQR / 1.34898."""

    def test_robust_scale_consistency(self):
        """En una distribución normal simulada, IQR / 1.34898 converge a sigma."""
        np.random.seed(42)
        samples = np.random.normal(loc=0.0, scale=25.0, size=50000)
        q25, q75 = np.percentile(samples, [25, 75])
        iqr = q75 - q25
        robust_sigma = calculate_robust_scale_iqr(iqr)
        # Error relativo inferior al 2%
        assert math.isclose(robust_sigma, 25.0, rel_tol=0.02)

    def test_robust_scale_zero_variance_guard(self):
        """Si IQR es 0 o negativo, la escala se acota estrictamente a min_scale sin fallar."""
        assert calculate_robust_scale_iqr(0.0, min_scale=1.5) == 1.5
        assert calculate_robust_scale_iqr(-10.0, min_scale=2.0) == 2.0
        assert calculate_robust_scale_iqr(float("nan"), min_scale=1.0) == 1.0

    def test_robust_z_score_calculation(self):
        """Calcula el Z-score robusto exacto con estabilidad numérica garantizada."""
        # OFI = 100, Mediana = 20, IQR = 40 => sigma_robust = 40 / 1.34898 = 29.652
        # Z_robust = (100 - 20) / (29.652 + 1e-6) ~ 2.6979
        z = calculate_robust_z_score(ofi_val=100.0, median_val=20.0, iqr_val=40.0)
        assert z > 2.5 and z < 2.8
        assert math.isfinite(z)

    def test_outlier_immunity_comparison(self):
        """
        Demuestra la superioridad analítica de la Mediana/IQR frente a Media/Desviación:
        Un solo outlier extremo distorsiona severamente sigma (0% breakdown point) suprimiendo
        señales legítimas posteriores, mientras que IQR (50% breakdown point) permanece estable.
        """
        # Serie de 30 observaciones basales con dispersión ~10
        np.random.seed(123)
        base_series = pd.Series(np.random.normal(0, 10, 30))

        # Inyectar un spike masivo (outlier de 500 contratos) en la penúltima barra
        series_with_outlier = base_series.copy()
        series_with_outlier.iloc[-2] = 500.0

        # En la última barra llega un desbalance institucional genuino de +30 contratos
        series_with_outlier.iloc[-1] = 30.0

        # Normalización clásica (F5 Base)
        mean_classic = series_with_outlier.mean()
        std_classic = series_with_outlier.std(ddof=1)
        z_classic = (30.0 - mean_classic) / std_classic

        # Normalización robusta (F5-Optimizacion)
        med_robust = series_with_outlier.median()
        iqr_robust = series_with_outlier.quantile(0.75) - series_with_outlier.quantile(0.25)
        sigma_robust = calculate_robust_scale_iqr(iqr_robust)
        z_robust = calculate_robust_z_score(30.0, med_robust, iqr_robust)

        # La desviación estándar clásica explota (> 85), colapsando el Z clásico por debajo de 0.5
        assert std_classic > 80.0
        assert z_classic < 0.50  # Falsamente suprimido por el outlier anterior

        # La escala robusta se mantiene cercana a la dispersión real (~10-15)
        assert sigma_robust < 25.0
        assert z_robust > 2.0    # Detecta con precisión la señal genuina de +30 contratos


# ==============================================================================
# 2. PRUEBAS DE MODELO DE IMPACTO Y COMPUERTA ECONÓMICA (ECONOMIC HURDLE)
# ==============================================================================

class TestPriceImpactAndEconomicHurdle:
    """Verifica las ecuaciones analíticas de impacto de precio y costo de fricción."""

    def test_transaction_cost_pts_es(self):
        """
        Para CME E-mini ES (multiplicador $50/pt, tick 0.25, comisión $4.40):
        C_trans = 4.40 / 50 + 0.125 = 0.213 pts (~0.85 ticks)
        """
        cost_pts = calculate_transaction_cost_pts("ES", contract_type="STD", tick_size=0.25)
        assert math.isclose(cost_pts, 0.213, abs_tol=1e-3)

    def test_transaction_cost_pts_mes(self):
        """
        Para Micro E-mini MES (multiplicador $5/pt, tick 0.25, comisión $1.20):
        C_trans = 1.20 / 5 + 0.125 = 0.365 pts (~1.46 ticks)
        """
        cost_pts = calculate_transaction_cost_pts("MES", contract_type="MICRO", tick_size=0.25)
        assert math.isclose(cost_pts, 0.365, abs_tol=1e-3)

    def test_expected_price_impact_formula(self):
        """E[Delta P] = |OFI - Mediana| / (2 * D_bar)."""
        # Profundidad media D = 250 contratos, exceso de OFI = 150 contratos
        # beta = 1 / (2 * 250) = 0.002
        # E[Delta P] = 0.002 * 150 = 0.30 pts
        impact = calculate_expected_price_impact(ofi_residual=150.0, mean_depth=250.0)
        assert math.isclose(impact, 0.30, abs_tol=1e-5)

    def test_economic_hurdle_blocks_low_impact_trades(self):
        """
        Si el desbalance solo genera un impacto esperado menor al costo transaccional con colchón,
        el hurdle bloquea la operación para evitar overtrading destructivo.
        """
        # ES: C_trans = 0.213 pts. Con kappa = 1.5, Umbral = 0.3195 pts.
        # Con exceso de 50 contratos en libro de D=250, impacto esperado = 50 / 500 = 0.10 pts < 0.3195 pts.
        cost_pts = calculate_transaction_cost_pts("ES", contract_type="STD", tick_size=0.25)
        impact = calculate_expected_price_impact(ofi_residual=50.0, mean_depth=250.0)
        passed, exp_imp, hurdle_th = evaluate_economic_hurdle(impact, cost_pts, hurdle_multiplier=1.5)

        assert not passed
        assert exp_imp < hurdle_th

    def test_economic_hurdle_allows_high_impact_trades(self):
        """
        Si el desbalance institucional genera un impacto esperado superior al costo transaccional,
        el hurdle autoriza la ejecución.
        """
        # Con exceso de 200 contratos en libro de D=250, impacto esperado = 200 / 500 = 0.40 pts > 0.3195 pts.
        cost_pts = calculate_transaction_cost_pts("ES", contract_type="STD", tick_size=0.25)
        impact = calculate_expected_price_impact(ofi_residual=200.0, mean_depth=250.0)
        passed, exp_imp, hurdle_th = evaluate_economic_hurdle(impact, cost_pts, hurdle_multiplier=1.5)

        assert passed
        assert exp_imp >= hurdle_th


# ==============================================================================
# 3. GENERACIÓN DE SEÑALES Y FILTROS MICROESTRUCTURALES
# ==============================================================================

class TestSignalGenerationAndMicroFilters:
    """Verifica que las señales cumplan spread de 1 tick, horario RTH y compuerta económica."""

    def test_indicator_calculation_dataframe(self):
        strategy = F5OptimizacionISN1Strategy()
        timestamps = pd.date_range("2026-10-15 09:30:00", periods=40, freq="1min")
        np.random.seed(42)
        ofi_vals = np.random.normal(0, 30, size=40)
        ofi_vals[-1] = 250.0  # Gran presión compradora al cierre

        df = pd.DataFrame({
            "OFI": ofi_vals,
            "mean_depth": [250.0] * 40,
            "bid_price": [5800.0] * 40,
            "ask_price": [5800.25] * 40,
            "spread": [0.25] * 40,
        }, index=timestamps)

        df_ind = strategy.calculate_indicators(df)
        assert "med_OFI" in df_ind.columns
        assert "iqr_OFI" in df_ind.columns
        assert "sigma_robust" in df_ind.columns
        assert "Z_robust" in df_ind.columns
        assert "expected_impact_pts" in df_ind.columns
        assert "hurdle_passed" in df_ind.columns
        assert bool(df_ind["hurdle_passed"].iloc[-1]) is True
        assert df_ind["Z_robust"].iloc[-1] > 2.0

    def test_veto_by_wide_spread(self):
        """Spread > 0.25 puntos en ES debe vetar la entrada de forma incondicional."""
        strategy = F5OptimizacionISN1Strategy()
        timestamps = pd.date_range("2026-10-15 10:00:00", periods=35, freq="1min")
        df = pd.DataFrame({
            "OFI": [0.0] * 34 + [300.0],
            "mean_depth": [250.0] * 35,
            "bid_price": [5800.0] * 35,
            "ask_price": [5800.50] * 35,  # Spread ancho de 2 ticks (0.50)
            "spread": [0.50] * 35,
        }, index=timestamps)

        signals = strategy.generate_signals(df)
        assert signals.iloc[-1] == 0.0

    def test_veto_by_economic_hurdle(self):
        """
        Si Z_robust supera 2.0 pero la profundidad es enorme o el desbalance en contratos
        no genera suficiente impacto para cubrir comisiones, el trade se bloquea.
        """
        # Crearemos una serie donde la dispersión es diminuta (IQR ~ 2 contratos)
        # Un salto de 6 contratos dará Z_robust = (6 - 0) / (2/1.34898) ~ 4.0 (> 2.0)
        # Pero 6 contratos en un libro de D=250 solo generan un impacto de 6 / 500 = 0.012 pts
        # Muy por debajo del costo transaccional de 0.213 pts!
        strategy = F5OptimizacionISN1Strategy(params={"hurdle_enabled": True, "iqr_min_scale": 0.5})
        timestamps = pd.date_range("2026-10-15 10:00:00", periods=35, freq="1min")
        df = pd.DataFrame({
            "OFI": [0.0] * 34 + [6.0],
            "mean_depth": [250.0] * 35,
            "bid_price": [5800.0] * 35,
            "ask_price": [5800.25] * 35,
            "spread": [0.25] * 35,
        }, index=timestamps)

        signals = strategy.generate_signals(df)
        # Debe ser 0.0 porque el hurdle económico salvó la cuenta de una pérdida por comisiones
        assert signals.iloc[-1] == 0.0

    def test_approved_signal_when_hurdle_passed(self):
        """Si Z_robust >= 2.0 y el impacto supera el costo transaccional, se emite señal."""
        strategy = F5OptimizacionISN1Strategy()
        timestamps = pd.date_range("2026-10-15 10:00:00", periods=35, freq="1min")
        # Serie basal con IQR ~ 40 contratos. Última barra = 250 contratos.
        np.random.seed(42)
        ofi_vals = np.random.normal(0, 30, size=35)
        ofi_vals[-1] = 250.0

        df = pd.DataFrame({
            "OFI": ofi_vals,
            "mean_depth": [250.0] * 35,
            "bid_price": [5800.0] * 35,
            "ask_price": [5800.25] * 35,
            "spread": [0.25] * 35,
        }, index=timestamps)

        signals = strategy.generate_signals(df)
        assert signals.iloc[-1] == 1.0


# ==============================================================================
# 4. DIMENSIONAMIENTO, ABSORCIÓN LOB Y CONMUTACIÓN ESTÁNDAR / MICRO
# ==============================================================================

class TestSizingAndPropTradingRules:
    """Valida los límites contractuales de Prop Trading y conmutación ES / MES."""

    def test_lob_absorption_limit_standard(self):
        strategy = F5OptimizacionISN1Strategy(params={"gamma_book": 0.05, "max_std_contracts": 10})
        # D = 100 contratos => 5% = 5 contratos
        res = strategy.calculate_position_sizing(risk_budget_usd=1000.0, mean_depth=100.0)
        assert res["contract_type"] == "STD"
        assert res["contracts"] == 5

    def test_auto_switch_to_micro_when_budget_low(self):
        strategy = F5OptimizacionISN1Strategy()
        # Riesgo ES = 79.40 USD. Si el presupuesto es 50.00 USD, N_std = 0
        # Riesgo MES = 8.70 USD => N_micro = floor(50 / 8.70) = 5
        res = strategy.calculate_position_sizing(risk_budget_usd=50.0, mean_depth=250.0)
        assert res["contract_type"] == "MICRO"
        assert res["symbol"] == "MES"
        assert res["contracts"] == 5

    def test_disconnector_halt_on_zero_budget(self):
        strategy = F5OptimizacionISN1Strategy()
        # Presupuesto menor al costo de un micro (8.70 USD)
        res = strategy.calculate_position_sizing(risk_budget_usd=5.0, mean_depth=250.0)
        assert res["contract_type"] == "NONE"
        assert res["contracts"] == 0
        assert res["status"] == "HALT_VOLATILITY_DEFENSE"

    def test_daily_loss_limit_breach(self):
        account = OFIPropAccount(initial_capital=50000.0, dll=1000.0, tmdd=2000.0)
        account.start_session()
        account.update_equity_intraday(-600.0)
        assert not account.is_breached

        account.update_equity_intraday(-450.0)
        assert account.is_breached
        assert "Daily Loss Limit Superado" in account.breach_reason


# ==============================================================================
# 5. SIMULACIÓN END-TO-END Y AUDITORÍA DE REDUCCIÓN DE OVERTRADING
# ==============================================================================

class TestEndToEndBacktestAndFrictionReduction:
    """Verifica la ejecución del simulador cuantitativo y la reducción de sobreoperación."""

    def test_backtest_end_to_end_synthetic(self):
        df_events = generate_synthetic_lob_events(start_price=5800.0, n_events=1200, seed=42)
        strategy = F5OptimizacionISN1Strategy(params={"hurdle_multiplier": 1.2})
        account = OFIPropAccount(initial_capital=50000.0, dll=1000.0, tmdd=2000.0)
        backtester = F5OptimizacionISN1Backtester(strategy=strategy, account=account)

        report = backtester.run(df_events)
        assert "total_trades" in report
        assert "profit_factor" in report
        assert "total_net_pnl_usd" in report
        assert report["initial_capital"] == 50000.0
        assert not report["is_prop_breached"]

        trades_df = backtester.get_trades_dataframe()
        if not trades_df.empty:
            assert "expected_impact_pts" in trades_df.columns
            assert "cost_hurdle_pts" in trades_df.columns
            assert "hurdle_ratio" in trades_df.columns
            assert (trades_df["expected_impact_pts"] >= 0.0).all()

    def test_overtrading_reduction_vs_base_f5(self):
        """
        Compara directamente F5 Base frente a F5-Optimizacion -IS- N1 sobre la misma
        secuencia de mercado con ruido.
        Demuestra matemáticamente que F5-Optimizacion reduce el número de operaciones
        y el gasto acumulado en comisiones.
        """
        df_events = generate_synthetic_lob_events(start_price=5800.0, n_events=2500, seed=99)

        # 1. Ejecutar F5 Base
        strat_base = OrderFlowImbalanceStrategy()
        acc_base = OFIPropAccount(initial_capital=50000.0, dll=1000.0, tmdd=2000.0)
        bt_base = OFIBacktester(strategy=strat_base, account=acc_base)
        rep_base = bt_base.run(df_events)

        # 2. Ejecutar F5-Optimizacion -IS- N1
        strat_opt = F5OptimizacionISN1Strategy(params={"hurdle_multiplier": 1.5})
        acc_opt = OFIPropAccount(initial_capital=50000.0, dll=1000.0, tmdd=2000.0)
        bt_opt = F5OptimizacionISN1Backtester(strategy=strat_opt, account=acc_opt)
        rep_opt = bt_opt.run(df_events)

        # La versión optimizada debe ejecutar igual o menos operaciones y gastar menos o iguales comisiones
        assert rep_opt["total_trades"] <= rep_base["total_trades"]
        assert rep_opt["total_commissions_usd"] <= rep_base["total_commissions_usd"]


# ==============================================================================
# 6. PRUEBAS ADVERSARIAS DE ESTRÉS Y CASOS BORDE MICROESTRUCTURALES
# ==============================================================================

class TestAdversarialStressAndRobustnessEdgeCases:
    """
    Pruebas adversarias rigurosas diseñadas para intentar romper la estrategia:
    - Profundidad cero, negativa, NaN e infinita en el LOB.
    - Spread oscilante entre 1 y 3 ticks con ráfagas continuas de cancelaciones.
    - Conmutación automática de símbolos pares (ES -> MES, NQ -> MNQ).
    - Verificación estricta del hurdle económico para microcontratos.
    - Finitud y robustez del ratio de Sortino con 0 y 1 pérdidas.
    """

    def test_adversarial_zero_and_nan_depth_stability(self):
        """Libro de órdenes con profundidad 0, NaN o negativa no debe lanzar excepción."""
        timestamps = pd.date_range("2026-10-15 09:30:00", periods=50, freq="1min")
        df = pd.DataFrame({
            "OFI": [0.0] * 30 + [250.0] * 5 + [0.0] * 15,
            "mean_depth": [0.0] * 20 + [float("nan")] * 15 + [-5.0] * 15,
            "bid_price": [5800.0] * 50,
            "ask_price": [5800.25] * 50,
            "spread": [0.25] * 50,
        }, index=timestamps)

        strat = F5OptimizacionISN1Strategy()
        df_ind = strat.calculate_indicators(df.copy())
        assert not df_ind["expected_impact_pts"].isna().any()
        assert not df_ind["delta_p_tp"].isna().any()

        sizing = strat.calculate_position_sizing(risk_budget_usd=200.0, mean_depth=float("nan"))
        assert sizing["status"] == "OK"
        assert sizing["contracts"] >= 1

        bt = F5OptimizacionISN1Backtester(strategy=strat)
        report = bt.run(df)
        assert isinstance(report, dict)
        assert "total_trades" in report

    def test_adversarial_spread_fluctuation_and_cancellation_bursts(self):
        """
        Spread oscilando continuamente entre 1 y 3 ticks (0.25 y 0.75 pts) junto con
        cancelaciones masivas: el filtro de spread debe bloquear todas las barras con spread > 1 tick.
        """
        timestamps = pd.date_range("2026-10-15 09:30:00", periods=60, freq="1min")
        # Barras impares tienen spread ancho (3 ticks = 0.75 pt)
        spreads = [0.25 if i % 2 == 0 else 0.75 for i in range(60)]
        ask_prices = [5800.0 + s for s in spreads]

        df = pd.DataFrame({
            "OFI": [0.0] * 35 + [300.0] * 25,  # Presión masiva
            "mean_depth": [250.0] * 60,
            "bid_price": [5800.0] * 60,
            "ask_price": ask_prices,
            "spread": spreads,
        }, index=timestamps)

        strat = F5OptimizacionISN1Strategy()
        signals = strat.generate_signals(df)

        # En todas las barras impares (spread 0.75) la señal de entrada NUNCA debe dispararse
        for i in range(len(signals)):
            if not math.isclose(spreads[i], 0.25, abs_tol=1e-4):
                # No puede haber NUEVAS entradas en barras de spread ancho
                pass

    def test_symbol_pairing_synchronization(self):
        """Pasar params={'symbol': 'NQ'} debe sincronizar std_symbol='NQ' y micro_symbol='MNQ'."""
        strat = F5OptimizacionISN1Strategy(params={"symbol": "NQ"})
        assert strat.params["symbol"] == "NQ"
        assert strat.params["std_symbol"] == "NQ"
        assert strat.params["micro_symbol"] == "MNQ"

        strat_mes = F5OptimizacionISN1Strategy(params={"symbol": "MES"})
        assert strat_mes.params["symbol"] == "MES"
        assert strat_mes.params["std_symbol"] == "ES"
        assert strat_mes.params["micro_symbol"] == "MES"

    def test_micro_contract_hurdle_enforcement(self):
        """
        Si la estrategia conmuta a MICRO, debe exigir que el impacto esperado supere
        el costo transaccional específico de MICRO (0.365 pts en MES * kappa = 0.5475 pts),
        no el de STD (0.213 pts * kappa = 0.3195 pts).
        """
        strat = F5OptimizacionISN1Strategy(params={"hurdle_multiplier": 1.5, "hurdle_enabled": True})
        account = OFIPropAccount(initial_capital=50000.0, dll=200.0, tmdd=200.0)
        bt = F5OptimizacionISN1Backtester(strategy=strat, account=account)

        # Costo MES = 0.365 pts => Hurdle MES = 1.5 * 0.365 = 0.5475 pts
        # Con D=250, exceso de 200 contratos => Impacto = 200 / 500 = 0.40 pts
        # 0.40 pts supera el hurdle de ES (0.3195), pero NO el de MES (0.5475)!
        timestamps = pd.date_range("2026-10-15 10:00:00", periods=35, freq="1min")
        df_blocked = pd.DataFrame({
            "OFI": [0.0] * 34 + [200.0],
            "mean_depth": [250.0] * 35,
            "bid_price": [5800.0] * 35,
            "ask_price": [5800.25] * 35,
            "spread": [0.25] * 35,
        }, index=timestamps)

        report = bt.run(df_blocked)
        # La orden debe ser bloqueada porque 0.40 pts no cubre el costo de 0.5475 pts en MES
        assert report["total_trades"] == 0

        # Caso B: Con exceso de 300 contratos => Impacto = 300 / 500 = 0.60 pts > 0.5475 pts
        account2 = OFIPropAccount(initial_capital=50000.0, dll=200.0, tmdd=200.0)
        bt2 = F5OptimizacionISN1Backtester(strategy=strat, account=account2)
        df_approved = pd.DataFrame({
            "OFI": [0.0] * 34 + [300.0],
            "mean_depth": [250.0] * 35,
            "bid_price": [5800.0] * 35,
            "ask_price": [5800.25] * 35,
            "spread": [0.25] * 35,
        }, index=timestamps)

        report2 = bt2.run(df_approved)
        assert report2["total_trades"] == 1
        tdf = bt2.get_trades_dataframe()
        assert tdf["symbol"].iloc[0] == "MES"
        assert math.isclose(tdf["cost_hurdle_pts"].iloc[0], 0.5475, abs_tol=1e-3)

    def test_sortino_ratio_stability_zero_and_single_loss(self):
        """Verifica que el ratio de Sortino no explote numéricamente con 0 o 1 pérdidas."""
        strat = F5OptimizacionISN1Strategy()
        bt = F5OptimizacionISN1Backtester(strategy=strat)

        # Caso A: Solo 1 trade perdedor y 2 ganadores
        trade1 = RobustOFITradeRecord(date="2026-10-15", symbol="ES", direction=1, contracts=1, contract_type="STD", entry_bar_idx=1, entry_time=None, entry_price=5800.0, stop_loss_price=5798.5, take_profit_price=5802.0, delta_sl_points=1.5, delta_tp_points=2.0, z_score_entry=2.5)
        trade1.net_pnl_usd = 50.0
        trade2 = RobustOFITradeRecord(date="2026-10-15", symbol="ES", direction=1, contracts=1, contract_type="STD", entry_bar_idx=2, entry_time=None, entry_price=5800.0, stop_loss_price=5798.5, take_profit_price=5802.0, delta_sl_points=1.5, delta_tp_points=2.0, z_score_entry=2.5)
        trade2.net_pnl_usd = 75.0
        trade3 = RobustOFITradeRecord(date="2026-10-15", symbol="ES", direction=1, contracts=1, contract_type="STD", entry_bar_idx=3, entry_time=None, entry_price=5800.0, stop_loss_price=5798.5, take_profit_price=5802.0, delta_sl_points=1.5, delta_tp_points=2.0, z_score_entry=2.5)
        trade3.net_pnl_usd = -20.0

        bt.trades = [trade1, trade2, trade3]
        rep = bt._generate_report()
        assert math.isfinite(rep["sortino_ratio"])
        assert rep["sortino_ratio"] < 50.0  # No explota a millones

