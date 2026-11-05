"""
tests/test_order_flow_imbalance.py
Suite de Pruebas Unitarias, Microestructurales y de Estrés para la Estrategia F5:
Order Flow Imbalance de Nivel 1 (Cont, Kukanov & Stoikov 2014).

Cubre:
1. Clasificación analítica exhaustiva de eventos de LOB (RUB-F5-01).
2. Normalización móvil e invarianza estacional (Z-Score con epsilon guard, RUB-F5-02).
3. Filtro determinista de spread de 1 tick (RUB-F5-03).
4. Salidas microestructurales dinámicas: Take-Profit, Stop-Loss y Flujo (RUB-F5-04).
5. Finitud temporal: Decaimiento informativo H_max y Cierre EOD 15:55 EST (RUB-F5-05).
6. Capacidad de absorción del LOB y límite de autolesión (RUB-F5-06).
7. Dimensionamiento de posición, conmutación automática Estándar/Micro y disyuntor (RUB-F5-07).
8. Simulación End-to-End con auditoría de Prop Trading (DLL y Trailing Max Drawdown).
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
    OFITradeRecord,
    OFIExitReason,
    calculate_eb,
    calculate_ea,
    calculate_en,
    compute_ofi_from_events,
    generate_synthetic_lob_events,
)
from src.core.futures_specs import get_spec


# ==============================================================================
# 1. PRUEBAS DE MICROESTRUCTURA DE LOB: CASOS DE EVENTO (RUB-F5-01)
# ==============================================================================

class TestLOBEventFormulas:
    """Valida la descomposición particionada exhaustiva de Cont et al. (2014)."""

    def test_bid_tick_up(self):
        # P_n^b > P_{n-1}^b => e_n^b = +q_n^b
        assert calculate_eb(5800.25, 15, 5800.00, 20) == 15.0

    def test_bid_same_price_increase_and_decrease(self):
        # P_n^b == P_{n-1}^b => e_n^b = q_n^b - q_{n-1}^b
        assert calculate_eb(5800.00, 30, 5800.00, 20) == 10.0  # Inserción pasiva neta
        assert calculate_eb(5800.00, 12, 5800.00, 20) == -8.0  # Cancelación / Agresión pasiva

    def test_bid_tick_down(self):
        # P_n^b < P_{n-1}^b => e_n^b = -q_{n-1}^b
        assert calculate_eb(5799.75, 25, 5800.00, 20) == -20.0

    def test_ask_tick_up(self):
        # P_n^a > P_{n-1}^a => e_n^a = -q_{n-1}^a
        assert calculate_ea(5800.50, 15, 5800.25, 25) == -25.0

    def test_ask_same_price_increase_and_decrease(self):
        # P_n^a == P_{n-1}^a => e_n^a = q_n^a - q_{n-1}^a
        assert calculate_ea(5800.25, 35, 5800.25, 25) == 10.0  # Inserción vendedora
        assert calculate_ea(5800.25, 10, 5800.25, 25) == -15.0 # Cancelación / Agresión compradora

    def test_ask_tick_down(self):
        # P_n^a < P_{n-1}^a => e_n^a = -q_n^a
        assert calculate_ea(5800.00, 18, 5800.25, 25) == -18.0

    def test_net_imbalance_sign_consistency(self):
        # e_n = e_n^b - e_n^a
        # Compras agresivas / inserción de compras: eb > 0, ea < 0 => e_n fuertemente positivo
        en_bull = calculate_en(eb=20.0, ea=-15.0)
        assert en_bull == 35.0 > 0

        # Ventas agresivas / inserción de ventas: eb < 0, ea > 0 => e_n fuertemente negativo
        en_bear = calculate_en(eb=-20.0, ea=15.0)
        assert en_bear == -35.0 < 0

        # Libro balanceado
        assert calculate_en(eb=10.0, ea=10.0) == 0.0

    def test_compute_ofi_from_synthetic_events(self):
        df_events = generate_synthetic_lob_events(start_price=5800.0, n_events=500, seed=123)
        agg = compute_ofi_from_events(df_events, freq="1min")

        assert not agg.empty
        assert "OFI" in agg.columns
        assert "mean_depth" in agg.columns
        assert "bid_price" in agg.columns
        assert "ask_price" in agg.columns
        assert "spread" in agg.columns
        assert (agg["spread"] >= 0.25).all()
        assert (agg["mean_depth"] > 0).all()


# ==============================================================================
# 2. NORMALIZACIÓN Z-SCORE E INVARIANZA ESTACIONAL (RUB-F5-02)
# ==============================================================================

class TestDynamicZScoreNormalization:
    """Verifica el cálculo de mu, sigma y Z-score con guarda de estabilidad."""

    def test_z_score_basic_calculation(self):
        strategy = OrderFlowImbalanceStrategy()
        # Creamos una serie de 35 barras de 1 minuto
        timestamps = pd.date_range("2026-10-15 09:30:00", periods=35, freq="1min")
        np.random.seed(42)
        ofi_vals = np.random.normal(0, 50, size=35)
        # Inyectar un pico significativo en la barra 34
        ofi_vals[-1] = 200.0

        df = pd.DataFrame({
            "OFI": ofi_vals,
            "mean_depth": [250.0] * 35,
            "bid_price": [5800.0] * 35,
            "ask_price": [5800.25] * 35,
            "spread": [0.25] * 35,
        }, index=timestamps)

        df_ind = strategy.calculate_indicators(df)
        assert not df_ind["Z_OFI"].isna().any()
        # En la barra 34 (índice -1), el Z-score debe ser fuertemente positivo
        assert df_ind["Z_OFI"].iloc[-1] > 2.0

    def test_z_score_epsilon_guard_zero_variance(self):
        strategy = OrderFlowImbalanceStrategy()
        timestamps = pd.date_range("2026-10-15 09:30:00", periods=35, freq="1min")
        # Serie con varianza cero (todos los valores idénticos a 100.0)
        df = pd.DataFrame({
            "OFI": [100.0] * 35,
            "mean_depth": [250.0] * 35,
            "bid_price": [5800.0] * 35,
            "ask_price": [5800.25] * 35,
            "spread": [0.25] * 35,
        }, index=timestamps)

        df_ind = strategy.calculate_indicators(df)
        z_series = df_ind["Z_OFI"].dropna()
        # No debe haber NaN, inf ni divisiones por cero
        assert np.isfinite(z_series).all()
        assert (z_series == 0.0).all()

    def test_price_impact_coefficient(self):
        strategy = OrderFlowImbalanceStrategy()
        timestamps = pd.date_range("2026-10-15 09:30:00", periods=35, freq="1min")
        # ES: profundidad media 250 -> beta = 1 / (2 * 250) = 0.002 pts/contrato
        df_es = pd.DataFrame({
            "OFI": [10.0] * 35,
            "mean_depth": [250.0] * 35,
            "bid_price": [5800.0] * 35,
            "ask_price": [5800.25] * 35,
            "spread": [0.25] * 35,
        }, index=timestamps)
        df_es = strategy.calculate_indicators(df_es)
        assert df_es["beta_OFI"].iloc[-1] == pytest.approx(1.0 / (2.0 * 250.0))

        # NQ: profundidad media 30 -> beta = 1 / (2 * 30) = 0.01666 pts/contrato
        df_nq = pd.DataFrame({
            "OFI": [10.0] * 35,
            "mean_depth": [30.0] * 35,
            "bid_price": [20000.0] * 35,
            "ask_price": [20000.25] * 35,
            "spread": [0.25] * 35,
        }, index=timestamps)
        df_nq = strategy.calculate_indicators(df_nq)
        assert df_nq["beta_OFI"].iloc[-1] == pytest.approx(1.0 / (2.0 * 30.0))
        assert df_nq["beta_OFI"].iloc[-1] > df_es["beta_OFI"].iloc[-1]


# ==============================================================================
# 3. FILTRO DE SPREAD ESTRICTO (RUB-F5-03)
# ==============================================================================

class TestStrictSpreadFilter:
    """Verifica que el sistema vete entradas deterministamente si el spread supera 1 tick."""

    def test_spread_filter_blocks_entry_when_spread_widens(self):
        strategy = OrderFlowImbalanceStrategy(params={"theta_enter": 2.00, "tick_size": 0.25})
        timestamps = pd.date_range("2026-10-15 09:45:00", periods=35, freq="1min")

        # Configurar un Z-score gigante (> 2.0) pero con spread = 2 ticks (0.50 pt)
        ofi_vals = [0.0] * 34 + [500.0]
        spreads = [0.25] * 34 + [0.50]  # Spread ensanchado en la barra de señal
        asks = [5800.25] * 34 + [5800.50]
        bids = [5800.00] * 35

        df = pd.DataFrame({
            "OFI": ofi_vals,
            "mean_depth": [250.0] * 35,
            "bid_price": bids,
            "ask_price": asks,
            "spread": spreads,
        }, index=timestamps)

        signals = strategy.generate_signals(df)
        # La última barra debe tener señal 0.0 debido al veto del spread
        assert signals.iloc[-1] == 0.0

    def test_spread_filter_permits_entry_when_spread_is_one_tick(self):
        strategy = OrderFlowImbalanceStrategy(params={"theta_enter": 2.00, "tick_size": 0.25})
        timestamps = pd.date_range("2026-10-15 09:45:00", periods=35, freq="1min")

        ofi_vals = [0.0] * 34 + [500.0]
        spreads = [0.25] * 35  # Spread óptimo de 1 tick
        bids = [5800.00] * 35
        asks = [5800.25] * 35

        df = pd.DataFrame({
            "OFI": ofi_vals,
            "mean_depth": [250.0] * 35,
            "bid_price": bids,
            "ask_price": asks,
            "spread": spreads,
        }, index=timestamps)

        signals = strategy.generate_signals(df)
        # Debe disparar Long (+1.0)
        assert signals.iloc[-1] == 1.0


# ==============================================================================
# 4. SALIDAS MICROESTRUCTURALES Y PROTECCIÓN DE FLUJO (RUB-F5-04 & RUB-F5-05)
# ==============================================================================

class TestExitsAndTimeDecay:
    """Verifica salidas por TP, SL, Reversión de Flujo y Time-Decay H_max."""

    def test_take_profit_by_price(self):
        strategy = OrderFlowImbalanceStrategy(params={"theta_enter": 2.0, "tick_size": 0.25})
        # 30 barras para warmup + barra 31 entrada + barra 32 subida de 5 ticks (1.25 pt)
        timestamps = pd.date_range("2026-10-15 09:45:00", periods=35, freq="1min")

        ofi = [0.0] * 30 + [200.0, 150.0, 100.0, 50.0, 0.0]
        bids = [5800.00] * 31 + [5801.50, 5801.50, 5801.50, 5801.50]
        asks = [b + 0.25 for b in bids]

        df = pd.DataFrame({
            "OFI": ofi,
            "mean_depth": [250.0] * 35,
            "bid_price": bids,
            "ask_price": asks,
            "spread": [0.25] * 35,
        }, index=timestamps)

        backtester = OFIBacktester(strategy=strategy)
        report = backtester.run(df)
        assert report["total_trades"] >= 1
        trade = backtester.trades[0]
        assert trade.exit_reason == OFIExitReason.TAKE_PROFIT_PRICE
        assert trade.net_pnl_usd > 0
        assert trade.exit_price == pytest.approx(trade.take_profit_price)

    def test_stop_loss_by_hard_price_stop(self):
        strategy = OrderFlowImbalanceStrategy(params={"theta_enter": 2.0, "tick_size": 0.25})
        timestamps = pd.date_range("2026-10-15 09:45:00", periods=35, freq="1min")

        # Entrada en barra 30 con Long, y desplome de 7 ticks (1.75 pts) en barra 31 sin reversión de Z
        ofi = [0.0] * 30 + [300.0, 50.0, 0.0, 0.0, 0.0]
        bids = [5800.00] * 31 + [5798.25, 5798.00, 5798.00, 5798.00]
        asks = [b + 0.25 for b in bids]

        df = pd.DataFrame({
            "OFI": ofi,
            "mean_depth": [250.0] * 35,
            "bid_price": bids,
            "ask_price": asks,
            "spread": [0.25] * 35,
        }, index=timestamps)

        backtester = OFIBacktester(strategy=strategy)
        report = backtester.run(df)
        assert report["total_trades"] >= 1
        trade = backtester.trades[0]
        assert trade.exit_reason == OFIExitReason.STOP_LOSS_PRICE
        assert trade.net_pnl_usd < 0
        assert trade.exit_price == pytest.approx(trade.stop_loss_price)

    def test_flow_reversal_exit_long(self):
        strategy = OrderFlowImbalanceStrategy(params={"theta_enter": 2.0, "theta_reversal": -1.25})
        timestamps = pd.date_range("2026-10-15 09:45:00", periods=35, freq="1min")

        # Entrada en barra 30 (Z > 2.0) y en barra 31 inyección masiva vendedora (Z <= -1.25)
        # pero precio solo cae 1 tick (0.25 pt, lejos del stop rígido de 6 ticks)
        ofi = [0.0] * 30 + [250.0, -200.0, 0.0, 0.0, 0.0]
        bids = [5800.00] * 31 + [5799.75, 5799.75, 5799.75, 5799.75]
        asks = [b + 0.25 for b in bids]

        df = pd.DataFrame({
            "OFI": ofi,
            "mean_depth": [250.0] * 35,
            "bid_price": bids,
            "ask_price": asks,
            "spread": [0.25] * 35,
        }, index=timestamps)

        backtester = OFIBacktester(strategy=strategy)
        report = backtester.run(df)
        assert report["total_trades"] >= 1
        trade = backtester.trades[0]
        assert trade.exit_reason == OFIExitReason.FLOW_REVERSAL
        assert trade.exit_price == pytest.approx(5799.75)

    def test_flow_exhaustion_exit_long(self):
        strategy = OrderFlowImbalanceStrategy(params={"theta_enter": 2.0, "theta_exhaust": 0.50})
        timestamps = pd.date_range("2026-10-15 09:45:00", periods=35, freq="1min")

        # Entrada Long en barra 30; en barra 31 precio sube +0.50 pt (2 ticks, menos que TP de 4 ticks)
        # pero el flujo se normaliza a Z <= 0.50 (asegurando beneficio)
        ofi = [0.0] * 30 + [250.0, 10.0, 0.0, 0.0, 0.0]
        bids = [5800.00] * 31 + [5800.75, 5800.75, 5800.75, 5800.75]
        asks = [b + 0.25 for b in bids]

        df = pd.DataFrame({
            "OFI": ofi,
            "mean_depth": [250.0] * 35,
            "bid_price": bids,
            "ask_price": asks,
            "spread": [0.25] * 35,
        }, index=timestamps)

        backtester = OFIBacktester(strategy=strategy)
        report = backtester.run(df)
        assert report["total_trades"] >= 1
        trade = backtester.trades[0]
        assert trade.exit_reason == OFIExitReason.FLOW_EXHAUSTION
        assert trade.net_pnl_usd > 0

    def test_short_position_lifecycle(self):
        strategy = OrderFlowImbalanceStrategy(params={"theta_enter": 2.0, "tick_size": 0.25})
        timestamps = pd.date_range("2026-10-15 09:45:00", periods=35, freq="1min")

        # Disparo Short en barra 30 (Z < -2.0) y caída de precio para TP
        ofi = [0.0] * 30 + [-250.0, -150.0, -100.0, 0.0, 0.0]
        bids = [5800.00] * 31 + [5798.50, 5798.50, 5798.50, 5798.50]
        asks = [b + 0.25 for b in bids]

        df = pd.DataFrame({
            "OFI": ofi,
            "mean_depth": [250.0] * 35,
            "bid_price": bids,
            "ask_price": asks,
            "spread": [0.25] * 35,
        }, index=timestamps)

        backtester = OFIBacktester(strategy=strategy)
        report = backtester.run(df)
        assert report["total_trades"] >= 1
        trade = backtester.trades[0]
        assert trade.direction == -1
        assert trade.exit_reason == OFIExitReason.TAKE_PROFIT_PRICE
        assert trade.net_pnl_usd > 0

    def test_time_decay_exit_after_h_max(self):
        strategy = OrderFlowImbalanceStrategy(params={"theta_enter": 2.0, "h_max": 10})
        # 30 barras de warmup + 1 barra de entrada + 12 barras planas
        timestamps = pd.date_range("2026-10-15 09:45:00", periods=45, freq="1min")

        ofi = [0.0] * 30 + [250.0] + [10.0] * 14
        bids = [5800.00] * 45
        asks = [5800.25] * 45

        df = pd.DataFrame({
            "OFI": ofi,
            "mean_depth": [250.0] * 45,
            "bid_price": bids,
            "ask_price": asks,
            "spread": [0.25] * 45,
        }, index=timestamps)

        backtester = OFIBacktester(strategy=strategy)
        report = backtester.run(df)
        assert report["total_trades"] >= 1
        trade = backtester.trades[0]
        assert trade.holding_bars <= 10

    def test_overnight_flat_rule_at_1555_est(self):
        # Usamos ventana w=5 y theta_enter=1.5 para este test de cierre EOD
        strategy = OrderFlowImbalanceStrategy(params={
            "window_w": 5,
            "theta_enter": 1.5,
            "exec_end_time": time(15, 55),
        })
        # Velas alrededor de las 15:55 EST
        timestamps = pd.date_range("2026-10-15 15:45:00", periods=15, freq="1min")

        ofi = [0.0] * 5 + [300.0] + [50.0] * 9  # Entrada en 15:50
        bids = [5800.00] * 15
        asks = [5800.25] * 15

        df = pd.DataFrame({
            "OFI": ofi,
            "mean_depth": [250.0] * 15,
            "bid_price": bids,
            "ask_price": asks,
            "spread": [0.25] * 15,
        }, index=timestamps)

        backtester = OFIBacktester(strategy=strategy)
        report = backtester.run(df)
        assert report["total_trades"] >= 1
        trade = backtester.trades[-1]
        assert trade.exit_reason == OFIExitReason.OVERNIGHT_FLAT

    def test_strategy_signals_and_backtester_alignment(self):
        strategy = OrderFlowImbalanceStrategy(params={"theta_enter": 2.0, "tick_size": 0.25})
        timestamps = pd.date_range("2026-10-15 09:45:00", periods=35, freq="1min")

        ofi = [0.0] * 30 + [250.0, 150.0, 100.0, 50.0, 0.0]
        bids = [5800.00] * 31 + [5801.50, 5801.50, 5801.50, 5801.50]
        asks = [b + 0.25 for b in bids]

        df = pd.DataFrame({
            "OFI": ofi,
            "mean_depth": [250.0] * 35,
            "bid_price": bids,
            "ask_price": asks,
            "spread": [0.25] * 35,
        }, index=timestamps)

        signals = strategy.generate_signals(df)
        backtester = OFIBacktester(strategy=strategy)
        report = backtester.run(df)

        assert report["total_trades"] == 1
        trade = backtester.trades[0]
        # Durante la barra de entrada, la señal debe ser activa (+1.0)
        assert signals.iloc[trade.entry_bar_idx] == 1.0
        # Tras el cierre del trade, la señal debe volver a ser neutra (0.0)
        assert signals.iloc[trade.exit_bar_idx] == 0.0

    def test_resting_stop_loss_fill_price_not_polluted_by_bar_close(self):
        strategy = OrderFlowImbalanceStrategy(params={"theta_enter": 2.0, "tick_size": 0.25, "sl_max_ticks": 6})
        timestamps = pd.date_range("2026-10-15 09:45:00", periods=35, freq="1min")

        # Entrada Long a Ask = 5800.25 -> Stop loss resting = 5800.25 - 1.50 = 5798.75
        # En la siguiente barra, el precio cae violentamente a 5750.00 (-50 puntos)
        ofi = [0.0] * 30 + [250.0, 0.0, 0.0, 0.0, 0.0]
        bids = [5800.00] * 31 + [5750.00] * 4
        asks = [b + 0.25 for b in bids]

        df = pd.DataFrame({
            "OFI": ofi,
            "mean_depth": [250.0] * 35,
            "bid_price": bids,
            "ask_price": asks,
            "spread": [0.25] * 35,
        }, index=timestamps)

        backtester = OFIBacktester(strategy=strategy)
        report = backtester.run(df)
        assert report["total_trades"] == 1
        trade = backtester.trades[0]
        assert trade.exit_reason == OFIExitReason.STOP_LOSS_PRICE
        # La orden resting en CME debe llenar a 5798.75, NO a 5750.00
        assert trade.exit_price == pytest.approx(5798.75)
        # Riesgo monetario acotado a 1.50 puntos
        pts_loss = trade.entry_price - trade.exit_price
        assert pts_loss == pytest.approx(1.50)

    def test_synthetic_lob_e2e_trading(self):
        # Generar secuencia L1 sintética y verificar que genera operaciones en el backtest
        df_events = generate_synthetic_lob_events(start_price=5800.0, n_events=5000, seed=42)
        strategy = OrderFlowImbalanceStrategy()
        backtester = OFIBacktester(strategy=strategy)
        report = backtester.run(df_events)
        assert report["total_trades"] >= 1
        assert not backtester.account.is_breached


# ==============================================================================
# 5. CAPACIDAD DE ABSORCIÓN DEL LOB Y SIZING PROP TRADING (RUB-F5-06 & RUB-F5-07)
# ==============================================================================

class TestAbsorptionCapacityAndSizing:
    """Verifica límites de autolesión, riesgo monetario y conmutación a micros."""

    def test_lob_absorption_capacity(self):
        strategy = OrderFlowImbalanceStrategy()
        # ES: D = 250 -> 5% = 12 contratos
        sizing_es = strategy.calculate_position_sizing(risk_budget_usd=2000.0, mean_depth=250.0)
        assert sizing_es["contracts"] <= 12

        # NQ: D = 30 -> 5% = 1 contrato
        strategy_nq = OrderFlowImbalanceStrategy(params={"symbol": "NQ", "std_symbol": "NQ", "micro_symbol": "MNQ"})
        sizing_nq = strategy_nq.calculate_position_sizing(risk_budget_usd=2000.0, mean_depth=30.0)
        assert sizing_nq["contracts"] <= 1

    def test_prop_budget_formula(self):
        account = OFIPropAccount(initial_capital=50000.0, dll=1000.0, tmdd=2000.0)
        # Margen intacto: 20% de 1000 = 200, 15% de 2000 = 300 => min(200, 300) = 200 USD
        budget = account.calculate_risk_budget(phi_dll=0.20, phi_tmdd=0.15)
        assert budget == pytest.approx(200.0)

        # Si ya se perdieron 600 USD en el día: margen diario = 400 => 20% = 80 USD
        account.daily_realized_loss = 600.0
        budget_lower = account.calculate_risk_budget(phi_dll=0.20, phi_tmdd=0.15)
        assert budget_lower == pytest.approx(80.0)

    def test_standard_to_micro_switching(self):
        strategy = OrderFlowImbalanceStrategy(params={"std_symbol": "ES", "micro_symbol": "MES"})
        # Riesgo por contrato ES: (1.50 * 50) + 4.40 = 79.40 USD
        # Si el presupuesto es 200 USD: 200 / 79.40 = 2 contratos ES
        sizing_std = strategy.calculate_position_sizing(risk_budget_usd=200.0, mean_depth=250.0)
        assert sizing_std["contract_type"] == "STD"
        assert sizing_std["symbol"] == "ES"
        assert sizing_std["contracts"] == 2

        # Si el presupuesto se reduce a 50 USD (< 79.40 USD):
        # Conmuta a MES: riesgo MES = (1.50 * 5) + 1.20 = 8.70 USD
        # 50 / 8.70 = 5 contratos MES
        sizing_micro = strategy.calculate_position_sizing(risk_budget_usd=50.0, mean_depth=250.0)
        assert sizing_micro["contract_type"] == "MICRO"
        assert sizing_micro["symbol"] == "MES"
        assert sizing_micro["contracts"] == 5

    def test_volatility_circuit_breaker(self):
        strategy = OrderFlowImbalanceStrategy(params={"std_symbol": "ES", "micro_symbol": "MES"})
        # Presupuesto minúsculo: 5 USD (< 8.70 USD)
        sizing_halt = strategy.calculate_position_sizing(risk_budget_usd=5.0, mean_depth=250.0)
        assert sizing_halt["contracts"] == 0
        assert sizing_halt["status"] == "HALT_VOLATILITY_DEFENSE"


# ==============================================================================
# 6. VERIFICACIÓN E2E DE AUDITORÍA Y LÍMITES DE PROP TRADING
# ==============================================================================

class TestPropTradingAuditE2E:
    """Verifica que el backtester respete incondicionalmente el DLL y el Trailing Drawdown."""

    def test_daily_loss_limit_breach(self):
        strategy = OrderFlowImbalanceStrategy(params={
            "theta_enter": 2.0,
            "tick_size": 0.25,
            "phi_dll": 1.0,  # Permite asignar el presupuesto completo para probar el corte de DLL
        })
        account = OFIPropAccount(initial_capital=50000.0, dll=30.0, tmdd=2000.0)
        backtester = OFIBacktester(strategy=strategy, account=account)

        # Generar un trade con gap adverso que exceda el stop y rompa el DLL de $30
        timestamps = pd.date_range("2026-10-15 09:45:00", periods=35, freq="1min")
        ofi = [0.0] * 30 + [300.0, 0.0, 0.0, 0.0, 0.0]
        # Gap down de 5 puntos en la apertura de la barra 31
        opens = [5800.00] * 31 + [5795.00] * 4
        bids = [5800.00] * 31 + [5795.00] * 4
        asks = [b + 0.25 for b in bids]

        df = pd.DataFrame({
            "Open": opens,
            "High": opens,
            "Low": bids,
            "Close": bids,
            "OFI": ofi,
            "mean_depth": [250.0] * 35,
            "bid_price": bids,
            "ask_price": asks,
            "spread": [0.25] * 35,
        }, index=timestamps)

        report = backtester.run(df)
        assert report["is_prop_breached"] is True
        assert "Daily Loss Limit Superado" in report["breach_reason"]

    def test_trailing_drawdown_floor_locking(self):
        account = OFIPropAccount(initial_capital=50000.0, tmdd=2000.0, lock_floor_at_start=True)
        assert account.trailing_floor == 48000.0

        # Ganancia lleva equity a 51,500
        account.update_equity_intraday(+1500.0)
        assert account.equity == 51500.0
        # Peak = 51500 -> floor = min(51500 - 2000, 50000) = 49500
        assert account.trailing_floor == 49500.0

        # Ganancia lleva equity a 53,000
        account.update_equity_intraday(+1500.0)
        assert account.equity == 53000.0
        # Con lock_floor_at_start=True, el piso nunca sube más allá de initial_capital (50,000)
        assert account.trailing_floor == 50000.0


# ==============================================================================
# 7. CALIBRACIÓN DE SIMULACIÓN LOB SINTÉTICO Y ADVERTENCIAS DE INFRAESTRUCTURA
# ==============================================================================

class TestCalibratedSyntheticLOBAndInfrastructureWarnings:
    """
    Verifica la calibración del simulador de vaciado de colas (Cont & de Larrard 2013),
    la correlación de impacto de precio (Cont et al. 2014), la eficacia teórica de la
    estrategia F5 y la emisión de advertencias ante datos agregados de 5m.
    """

    def test_calibrated_synthetic_lob_price_impact_correlation(self):
        """Verifica que OFI acumulado tenga fuerte correlación positiva con Delta P."""
        df_events = generate_synthetic_lob_events(
            start_price=5800.0,
            n_events=10000,
            tick_size=0.25,
            mean_depth=250.0,
            seed=42,
        )
        agg = compute_ofi_from_events(df_events, freq="1min")
        assert len(agg) >= 30
        delta_p = agg["Close"].diff()
        valid = agg["OFI"].iloc[1:]
        delta_valid = delta_p.iloc[1:]
        corr = valid.corr(delta_valid)
        # La correlación teórica de Cont et al. debe ser positiva y estadísticamente robusta
        assert corr > 0.50, f"Correlación observada insuficientemente predictiva: {corr:.4f}"

    def test_synthetic_lob_queue_depletion_tick_jumps(self):
        """Verifica que los saltos de cotización respeten la granularidad discreta de 0.25 pts."""
        df_events = generate_synthetic_lob_events(
            start_price=5800.0,
            n_events=3000,
            tick_size=0.25,
            mean_depth=250.0,
            seed=100,
        )
        bids = df_events["bid_price"]
        asks = df_events["ask_price"]
        spread = asks - bids

        # El spread siempre debe ser >= 0.25 pts
        assert (spread >= 0.25).all()
        # Los precios deben ser múltiplos exactos de 0.25
        assert (np.remainder(np.round(bids * 100), 25) == 0).all()
        assert (np.remainder(np.round(asks * 100), 25) == 0).all()
        # Las profundidades deben ser estrictamente positivas
        assert (df_events["bid_size"] > 0).all()
        assert (df_events["ask_size"] > 0).all()

    def test_ohlcv_proxy_lag_warning(self):
        """Verifica que prepare_data emita una advertencia explícita ante datos OHLCV sin L1."""
        strategy = OrderFlowImbalanceStrategy()
        timestamps = pd.date_range("2026-10-15 09:30:00", periods=50, freq="5min")
        ohlcv_df = pd.DataFrame({
            "Open": [5800.0] * 50,
            "High": [5802.0] * 50,
            "Low": [5799.0] * 50,
            "Close": [5801.0] * 50,
            "Volume": [1000] * 50,
        }, index=timestamps)

        with pytest.warns(UserWarning, match="ADVERTENCIA DE MICROESTRUCTURA"):
            prepared = strategy.prepare_data(ohlcv_df)

        assert "OFI" in prepared.columns
        assert "mean_depth" in prepared.columns

    def test_calibrated_synthetic_lob_backtest_efficacy(self):
        """Verifica que sobre el LOB sintético calibrado F5 alcance alta eficacia teórica."""
        df_events = generate_synthetic_lob_events(
            start_price=5800.0,
            n_events=20000,
            tick_size=0.25,
            mean_depth=250.0,
            seed=42,
        )
        strategy = OrderFlowImbalanceStrategy(params={"symbol": "ES", "std_symbol": "ES", "micro_symbol": "MES"})
        account = OFIPropAccount()
        backtester = OFIBacktester(strategy=strategy, account=account)
        report = backtester.run(df_events)

        assert report["total_trades"] >= 2
        # La eficacia teórica en un LOB calibrado sin retardo de 5m supera el 50% de Win Rate
        assert report["win_rate_pct"] >= 50.0
        assert report["total_net_pnl_usd"] > 0.0
        assert not report["is_prop_breached"]
        assert report["breach_reason"] == "COMPLIANT (Ningún límite violado)"
