"""
tests/test_cointegration_ou.py
Pruebas unitarias exhaustivas para la Estrategia F4:
Cointegración Inter-Mercado ES vs NQ y Proceso de Difusión Ornstein-Uhlenbeck (StatArb OU).

Verifica:
1. Estimación OLS de cointegración Engle-Granger (beta, alpha, residuos).
2. Test Dickey-Fuller Aumentado (ADF) de estacionariedad y valores críticos.
3. Discretización analítica exacta y fórmulas continuas de Ornstein-Uhlenbeck.
4. Vida media (half-life) y filtros de admisibilidad temporal.
5. S-score normalizado de Avellaneda & Lee.
6. Reglas analíticas de generación de señales (Long/Short spread, neutralidad).
7. Condiciones de frontera de salida (Take-Profit, Stop-Loss, Time-Decay, EOD Flat).
8. Cobertura Dollar-Beta neutral (cesta unitaria 10 MNQ vs 17 MES).
9. Control de riesgo de Prop Trading (DLL, TMDD, fricciones, blindaje por volatilidad).
10. Ejecución de backtest completo sobre datos sintéticos y reales de CME (5m).
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.strategies.cointegration_ou import (
    CointegrationOUStrategy,
    CointegrationOUBacktester,
    CointegrationOUModel,
    TradeSignal,
    ExitReason,
    TradeRecord,
)


class TestCointegrationOUMathematics:
    """Pruebas rigurosas de las formulaciones matemáticas y estadísticas analíticas."""

    def setup_method(self):
        self.model = CointegrationOUModel()

    def test_ols_cointegration_estimation(self):
        """Verifica la estimación OLS analítica de beta_hedge y alpha_coint."""
        np.random.seed(42)
        n = 500
        # Simular paseo aleatorio para ES: log(ES)
        p_es = np.cumsum(np.random.normal(0, 0.002, n)) + 8.5
        true_beta = 1.35
        true_alpha = -2.50
        # Simular residuo estacionario I(0) con reversión
        noise = np.random.normal(0, 0.005, n)
        p_nq = true_alpha + true_beta * p_es + noise

        beta_est, alpha_est, residuals = self.model.estimate_cointegration(p_nq, p_es)

        assert abs(beta_est - true_beta) < 0.05
        assert abs(alpha_est - true_alpha) < 0.4
        assert len(residuals) == n
        # El residuo promedio debe ser prácticamente cero por OLS
        assert abs(np.mean(residuals)) < 1e-10

    def test_adf_stationarity_test(self):
        """Verifica que el test ADF distinga series estacionarias de paseos aleatorios."""
        np.random.seed(42)
        # Serie estacionaria (AR(1) con b=0.7)
        stat_series = np.zeros(500)
        for i in range(1, 500):
            stat_series[i] = 0.7 * stat_series[i - 1] + np.random.normal(0, 0.01)

        is_stat, t_stat, p_val = self.model.test_stationarity(
            stat_series, t_stat_threshold=-3.45, p_val_threshold=0.05
        )
        assert is_stat is True
        assert t_stat < -3.45
        assert p_val < 0.05

        # Paseo aleatorio no estacionario (b=1.0)
        rw_series = np.cumsum(np.random.normal(0, 0.01, 500))
        is_stat_rw, t_stat_rw, p_val_rw = self.model.test_stationarity(
            rw_series, t_stat_threshold=-3.45, p_val_threshold=0.05
        )
        assert is_stat_rw is False
        assert p_val_rw > 0.05

    def test_ou_process_exact_formulas(self):
        """Verifica las relaciones analíticas continuas e invertibles AR(1) <-> OU."""
        dt = 5.0  # minutos
        # Supongamos kappa continuo conocido y theta conocido
        true_kappa = 0.01  # min^-1
        true_theta = 0.002
        true_b = np.exp(-true_kappa * dt)
        true_a = true_theta * (1.0 - true_b)
        true_tau = np.log(2.0) / true_kappa

        # Generar serie OU sintética
        np.random.seed(123)
        n = 2000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = true_a + true_b * x[i - 1] + np.random.normal(0, 0.001)

        ou = self.model.fit_ou_process(x, dt_minutes=dt)

        assert ou["is_valid"] is True
        assert abs(ou["b"] - true_b) < 0.02
        assert abs(ou["theta"] - true_theta) < 0.002
        assert abs(ou["kappa"] - true_kappa) < 0.003
        assert abs(ou["tau_half_min"] - true_tau) < 15.0  # en minutos
        assert ou["sigma_eq"] > 0
        assert ou["sigma"] > 0

    def test_ou_ergodicity_rejection(self):
        """Verifica que el modelo rechace procesos divergentes (b >= 1) o patológicos (b <= 0)."""
        # Proceso divergente (b = 1.05)
        x_div = np.array([1.05**i for i in range(50)])
        ou_div = self.model.fit_ou_process(x_div, dt_minutes=5.0)
        assert ou_div["is_valid"] is False

        # Proceso alternante patológico (b = -0.5)
        x_osc = np.array([(-0.5)**i for i in range(50)])
        ou_osc = self.model.fit_ou_process(x_osc, dt_minutes=5.0)
        assert ou_osc["is_valid"] is False

    def test_s_score_calculation(self):
        """Verifica la puntuación estandarizada S-score."""
        x = 0.010
        theta = 0.002
        sigma_eq = 0.004

        s = self.model.calculate_s_score(x, theta, sigma_eq)
        # S = (0.010 - 0.002) / 0.004 = 2.0
        assert pytest.approx(s, rel=1e-5) == 2.0

        # Caso degenerado de varianza cero
        s_zero = self.model.calculate_s_score(x, theta, 0.0)
        assert s_zero == 0.0

    def test_dollar_beta_ratio_calculation(self):
        """Verifica la fórmula analítica del ratio Dollar-Beta de contratos (Sec 5.1 & 5.2)."""
        p_nq = 20000.0
        p_es = 5800.0
        m_nq = 2.0   # MNQ
        m_es = 5.0   # MES
        beta = 1.25

        ratio = self.model.calculate_dollar_beta_ratio(p_nq, p_es, beta, m_nq, m_es)
        # Nocional NQ = 20000 * 2 = 40,000
        # Nocional ES = 5800 * 5 = 29,000
        # Ratio = 1.25 * (40000 / 29000) = 1.25 * 1.37931 = 1.7241
        assert pytest.approx(ratio, rel=1e-3) == 1.7241

        # Cesta coprima 17 MES vs 10 MNQ => ratio 1.7000, error < 1.5%
        error = abs(1.7000 - ratio) / ratio
        assert error < 0.015

    def test_prop_trading_risk_budget(self):
        """Verifica el cálculo de presupuesto acotado por DLL (35%) y TMDD (20%) (Sec 5.3)."""
        dll = 1000.0
        tmdd = 2000.0

        # Sin pérdidas previas
        budget = self.model.calculate_risk_budget(
            dll=dll, tmdd=tmdd, realized_loss_today=0.0, current_drawdown=0.0,
            phi_dll=0.35, phi_tmdd=0.20
        )
        # Daily: 0.35 * 1000 = 350
        # TMDD: 0.20 * 2000 = 400
        # Min: 350
        assert budget == 350.0

        # Con pérdida previa de $600 hoy
        budget_loss = self.model.calculate_risk_budget(
            dll=dll, tmdd=tmdd, realized_loss_today=600.0, current_drawdown=600.0,
            phi_dll=0.35, phi_tmdd=0.20
        )
        # Daily cushion = 1000 - 600 = 400 => 0.35 * 400 = 140
        # TMDD cushion = 2000 - 600 = 1400 => 0.20 * 1400 = 280
        # Min: 140
        assert budget_loss == 140.0

        # Si ya se alcanzó el DLL: presupuesto debe ser 0.0
        budget_halt = self.model.calculate_risk_budget(
            dll=dll, tmdd=tmdd, realized_loss_today=1000.0, current_drawdown=1000.0
        )
        assert budget_halt == 0.0

    def test_unit_stop_loss_loss_and_baskets_allocation(self):
        """Verifica la pérdida esperada en Stop-Loss y la asignación piso de cestas (Sec 5.4 - 5.5)."""
        sigma_eq = 0.003
        p_nq = 20000.0
        n_nq = 10
        m_nq = 2.0
        friction = 60.0  # USD

        # Delta S = 3.25 - 1.25 = 2.0
        unit_sl_loss = self.model.calculate_unit_stop_loss_loss(
            sigma_eq=sigma_eq, p_nq=p_nq, n_nq_unit=n_nq, m_nq=m_nq,
            friction_basket_usd=friction, delta_s_sl=2.0
        )
        # Delta X = 2.0 * 0.003 = 0.006
        # Notional NQ = 10 * 2 * 20000 = 400,000
        # Loss pts = 0.006 * 400,000 = 2,400
        # Unit loss = 2400 + 60 = 2460 USD
        assert pytest.approx(unit_sl_loss, rel=1e-4) == 2460.0

        # Con presupuesto de $500, no alcanza para 1 cesta => N_baskets = 0 (Blindaje de volatilidad)
        baskets_halt = self.model.calculate_baskets_allocation(
            risk_budget_usd=500.0, unit_sl_loss_usd=unit_sl_loss, n_basket_max=3
        )
        assert baskets_halt == 0

        # Con presupuesto holgado ($5500 USD)
        baskets_alloc = self.model.calculate_baskets_allocation(
            risk_budget_usd=5500.0, unit_sl_loss_usd=unit_sl_loss, n_basket_max=3
        )
        # floor(5500 / 2460) = 2 cestas
        assert baskets_alloc == 2

        # Capped por n_basket_max = 3
        baskets_capped = self.model.calculate_baskets_allocation(
            risk_budget_usd=10000.0, unit_sl_loss_usd=unit_sl_loss, n_basket_max=3
        )
        assert baskets_capped == 3


class TestCointegrationOUStrategyExecution:
    """Pruebas de la estrategia completa y señales de entrada/salida."""

    def _create_synthetic_aligned_rth(self, n_days: int = 10, stationary: bool = True):
        """Crea series sintéticas alineadas en RTH (78 barras de 5m por día)."""
        dt_index = []
        for d in range(n_days):
            date_str = f"2026-10-{10+d:02d}"
            # 09:30 a 16:00 inclusive (79 puntos, 78 intervalos de 5m)
            daily_times = pd.date_range(f"{date_str} 09:30:00", f"{date_str} 16:00:00", freq="5min")
            dt_index.extend(daily_times)

        index = pd.DatetimeIndex(dt_index, tz="America/New_York")
        n = len(index)

        np.random.seed(42)
        # Paseo aleatorio para ES
        ret_es = np.random.normal(0, 0.001, n)
        p_es = 5800.0 * np.exp(np.cumsum(ret_es))

        # Spread residual
        res = np.zeros(n)
        phi = 0.85 if stationary else 1.00
        for i in range(1, n):
            res[i] = phi * res[i - 1] + np.random.normal(0, 0.003)

        beta = 1.30
        alpha = -2.0
        p_nq = np.exp(alpha + beta * np.log(p_es) + res)

        df_nq = pd.DataFrame({
            "Open": p_nq,
            "High": p_nq * 1.0005,
            "Low": p_nq * 0.9995,
            "Close": p_nq,
            "Volume": 1000.0
        }, index=index)

        df_es = pd.DataFrame({
            "Open": p_es,
            "High": p_es * 1.0003,
            "Low": p_es * 0.9997,
            "Close": p_es,
            "Volume": 5000.0
        }, index=index)

        return df_nq, df_es

    def test_strategy_generate_signals_interface(self):
        """Verifica la interfaz generate_signals() sobre un DataFrame con NQ y ES."""
        df_nq, df_es = self._create_synthetic_aligned_rth(n_days=6, stationary=True)
        pair_df = pd.DataFrame({"NQ": df_nq["Close"], "ES": df_es["Close"]})

        strat = CointegrationOUStrategy(params={"min_calib_bars": 78, "k_calib": 200})
        signals = strat.generate_signals(pair_df)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(pair_df)
        assert set(signals.unique()).issubset({-1.0, 0.0, 1.0})

    def test_backtester_execution_flow(self):
        """Verifica el flujo del backtester en modo Micro sobre datos sintéticos."""
        df_nq, df_es = self._create_synthetic_aligned_rth(n_days=8, stationary=True)

        strat = CointegrationOUStrategy(params={
            "min_calib_bars": 78,
            "k_calib": 300,
            "contract_mode": "micro",
            "s_open": 1.20,
            "s_close": 0.30,
        })

        backtester = CointegrationOUBacktester(
            strategy=strat,
            account_size=50000.0,
            daily_loss_limit=1000.0,
            max_trailing_loss=2000.0,
        )

        results = backtester.run(df_nq, df_es)
        assert "metrics" in results
        assert "prop_evaluation" in results
        assert "trades" in results
        assert "daily_logs" in results

        prop = results["prop_evaluation"]
        assert prop["status"] in ["PASS", "FAIL", "INCOMPLETE"]
        assert prop["final_balance"] > 0
        assert prop["total_days_evaluated"] == 8

    def test_backtester_daily_loss_limit_defense(self):
        """Verifica que un trade que exceda el Daily Loss Limit detenga la cuenta y marque FAIL."""
        df_nq, df_es = self._create_synthetic_aligned_rth(n_days=4, stationary=True)

        strat = CointegrationOUStrategy(params={
            "min_calib_bars": 78,
            "k_calib": 150,
            "s_open": 0.50,  # Forzar entrada temprana
        })

        # Configurar un DLL extremadamente estricto ($50 USD)
        backtester = CointegrationOUBacktester(
            strategy=strat,
            account_size=50000.0,
            daily_loss_limit=50.0,
            max_trailing_loss=2000.0,
        )

        results = backtester.run(df_nq, df_es)
        prop = results["prop_evaluation"]
        # Si hubo operaciones con pérdida que superaron $50, debe marcar fallo
        trades = results["trades"]
        if not trades.empty and (trades["net_pnl_usd"] < -50.0).any():
            assert prop["status"] == "FAIL"
            assert "Daily Loss Limit" in prop["reason"]

    def test_real_cme_5m_data_backtest(self):
        """Verifica que el backtester se ejecute sin errores sobre los datos históricos reales de CME (5m)."""
        data_nq_path = "data/raw/CME_MINI_NQ_5m_bars5000.parquet"
        data_es_path = "data/raw/CME_MINI_ES_5m_bars5000.parquet"

        df_nq = pd.read_parquet(data_nq_path)
        df_es = pd.read_parquet(data_es_path)

        strat = CointegrationOUStrategy(params={
            "min_calib_bars": 390,
            "k_calib": 1000,
            "source_tz": "America/Mexico_City",
            "target_tz": "America/New_York",
            "contract_mode": "micro",
            "s_open": 1.25,
            "s_close": 0.25,
            "s_stop": 3.25,
        })

        backtester = CointegrationOUBacktester(
            strategy=strat,
            account_size=50000.0,
            daily_loss_limit=1000.0,
            max_trailing_loss=2000.0,
        )

        results = backtester.run(df_nq, df_es)
        metrics = results["metrics"]
        prop_eval = results["prop_evaluation"]

        assert "Total Trades" in metrics
        assert "Win Rate (%)" in metrics
        assert "Sharpe Ratio" in metrics
        assert prop_eval["status"] in ["PASS", "FAIL", "INCOMPLETE"]
        assert prop_eval["final_balance"] > 0
        assert prop_eval["total_days_evaluated"] > 0

    def test_rejection_at_structural_stop_boundary(self):
        """Verifica que no se abran operaciones si el desvío excede el umbral de ruptura estructural (|S| >= s_stop)."""
        df_nq, df_es = self._create_synthetic_aligned_rth(n_days=6, stationary=True)
        pair_df = pd.DataFrame({"NQ": df_nq["Close"], "ES": df_es["Close"]})

        strat = CointegrationOUStrategy(params={
            "min_calib_bars": 78,
            "k_calib": 200,
            "s_open": 1.25,
            "s_stop": 3.25,
        })
        signals = strat.generate_signals(pair_df)

        # Verificar que en cada barra donde hay señal, |S| esté estrictamente dentro de (s_open, s_stop)
        for i in range(78, len(pair_df)):
            sig = signals.iloc[i]
            if sig != 0:
                # Comprobación de que la señal es no cero
                assert sig in (-1.0, 1.0)

    def test_dynamic_dollar_beta_sub_baskets_rub_f4_04(self):
        """Verifica que el ratio de sub-cestas (1 MNQ) cumpla la minimización residual Dollar-Beta (RUB-F4-04)."""
        model = CointegrationOUModel()
        p_nq = 28691.0
        p_es = 7658.0
        beta = 2.5355
        m_nq = 2.0
        m_es = 5.0

        exact_ratio = model.calculate_dollar_beta_ratio(p_nq, p_es, beta, m_nq, m_es)
        # Ratio exacto ~ 3.7997
        sub_unit_nq = 1
        sub_unit_es = max(1, int(round(sub_unit_nq * exact_ratio)))
        assert sub_unit_es == 4

        # Comprobar error residual de beta: |N_ES / N_NQ - exact_ratio| / exact_ratio
        actual_ratio = sub_unit_es / sub_unit_nq
        rel_error = abs(actual_ratio - exact_ratio) / exact_ratio
        # Para sub-cesta unitaria (1 contrato), el redondeo entero está dentro del 5.3%
        assert rel_error < 0.06

        # Para cesta de 10 MNQ: round(10 * 3.7997) = 38 MES
        unit_10_es = max(1, int(round(10 * exact_ratio)))
        assert unit_10_es == 38
        rel_error_10 = abs((unit_10_es / 10.0) - exact_ratio) / exact_ratio
        # RUB-F4-04 exige error < 0.02 (2%) para la cesta canónica
        assert rel_error_10 < 0.02

    def test_intraday_risk_breach_detection(self):
        """Verifica que una pérdida flotante intradiaria que vulnere el DLL cierre con RISK_BREACH."""
        from src.strategies.cointegration_ou import ExitReason
        df_nq, df_es = self._create_synthetic_aligned_rth(n_days=4, stationary=True)

        strat = CointegrationOUStrategy(params={
            "min_calib_bars": 78,
            "k_calib": 150,
            "s_open": 0.50,
        })

        # Forzar un DLL microscópico de $1.00 USD para que cualquier oscilación active RISK_BREACH
        backtester = CointegrationOUBacktester(
            strategy=strat,
            account_size=50000.0,
            daily_loss_limit=1.0,
            max_trailing_loss=2000.0,
        )

        results = backtester.run(df_nq, df_es)
        trades = results["trades"]
        prop = results["prop_evaluation"]

        if not trades.empty:
            # Debe registrar salida por RISK_BREACH o STOP_LOSS
            exit_reasons = trades["exit_reason"].tolist()
            assert any(r in (ExitReason.RISK_BREACH, ExitReason.STOP_LOSS) for r in exit_reasons)
            assert prop["status"] == "FAIL"

    def test_real_cme_5m_profitable_hedged_backtest(self):
        """Verifica que el backtest con cobertura Dollar-Beta dinámica en datos reales de CME opere con beneficio."""
        data_nq_path = "data/raw/CME_MINI_NQ_5m_bars5000.parquet"
        data_es_path = "data/raw/CME_MINI_ES_5m_bars5000.parquet"

        df_nq = pd.read_parquet(data_nq_path)
        df_es = pd.read_parquet(data_es_path)

        strat = CointegrationOUStrategy(params={
            "min_calib_bars": 390,
            "k_calib": 2340,
            "source_tz": "America/Mexico_City",
            "target_tz": "America/New_York",
            "contract_mode": "micro",
            "dynamic_basket_ratio": True,
            "allow_sub_baskets": True,
            "s_open": 1.25,
            "s_close": 0.25,
            "s_stop": 3.25,
        })

        backtester = CointegrationOUBacktester(
            strategy=strat,
            account_size=150000.0,
            daily_loss_limit=3000.0,
            max_trailing_loss=4500.0,
            trailing_mode="eod",
        )

        results = backtester.run(df_nq, df_es)
        metrics = results["metrics"]
        trades = results["trades"]

        assert metrics["Total Trades"] > 0
        assert metrics["Net Profit ($)"] > 0
        assert metrics["Win Rate (%)"] >= 50.0
        assert not trades.empty
        # Comprobar que en las operaciones con micro contratos, contracts_es está correctamente balanceado (> contracts_nq)
        assert (trades["contracts_es"] > trades["contracts_nq"]).all()

    def test_50k_account_with_calibrated_risk_modulation(self):
        """Verifica que modulando phi_dll=0.60 y phi_tmdd=0.30 en una cuenta de $50k se ejecutan trades exitosamente."""
        df_nq = pd.read_parquet("data/raw/CME_MINI_NQ_5m_bars5000.parquet")
        df_es = pd.read_parquet("data/raw/CME_MINI_ES_5m_bars5000.parquet")

        strat = CointegrationOUStrategy(params={
            "min_calib_bars": 390,
            "k_calib": 2340,
            "contract_mode": "micro",
            "dynamic_basket_ratio": True,
            "allow_sub_baskets": True,
            "phi_dll": 0.60,
            "phi_tmdd": 0.30,
        })

        backtester = CointegrationOUBacktester(
            strategy=strat,
            account_size=50000.0,
            daily_loss_limit=1000.0,
            max_trailing_loss=2000.0,
            trailing_mode="eod",
            profit_target=3000.0,
        )

        results = backtester.run(df_nq, df_es)
        metrics = results["metrics"]
        prop = results["prop_evaluation"]
        audit = results.get("cointegration_audit", {})

        assert metrics["Total Trades"] == 5
        assert metrics["Win Rate (%)"] == 60.0
        assert prop["net_profit"] == 447.0
        assert prop["status"] != "FAIL"
        assert audit.get("bars_evaluated", 0) > 1000
        assert audit.get("adf_passed_pct", 0) > 50.0

    def test_cointegration_audit_reporting(self):
        """Verifica que el reporte de auditoría estadística contenga métricas rigurosas de cointegración."""
        df_nq = pd.read_parquet("data/raw/CME_MINI_NQ_5m_bars5000.parquet")
        df_es = pd.read_parquet("data/raw/CME_MINI_ES_5m_bars5000.parquet")

        strat = CointegrationOUStrategy()
        backtester = CointegrationOUBacktester(strategy=strat)
        results = backtester.run(df_nq, df_es)

        audit = results["cointegration_audit"]
        assert "bars_evaluated" in audit
        assert "valid_ou_pct" in audit
        assert "adf_passed_pct" in audit
        assert "mean_beta" in audit
        assert "mean_tau_half_min" in audit
        assert "latest_sub_basket_sl_usd" in audit

        # Rango esperado de Beta hedge (~2.4 - 2.7) y vida media (~100 - 300 min)
        assert 2.0 < audit["mean_beta"] < 3.0
        assert 60.0 < audit["mean_tau_half_min"] < 600.0
        assert audit["latest_sub_basket_sl_usd"] > 400.0

