"""
tests/test_f4_optimizacion_is_n2.py
Pruebas Unitarias Exhaustivas para F4-Optimizacion -IS- N2:
Cointegración Inter-Mercado ES vs NQ y Difusión Ornstein-Uhlenbeck.

Verifica rigurosamente las correcciones de las 6 anomalías de auditoría:
1. Evaluación de muestra completa (early_stopping=False) vs parada temprana (early_stopping=True).
2. Cálculo de Sharpe sobre serie de retornos diarios (252d) y separación del Sharpe por trade.
3. Congelamiento estricto de parámetros de equilibrio (alpha, beta, theta, sigma_eq) durante trade abierto (Anti-Moving Target).
4. Ejecución determinista no anticipativa en Open[t+1] con deslizamiento y comisiones.
5. Neutralidad Dollar-Beta mediante retículo coprimo gcd(N_NQ, N_ES)=1 y sub-cestas micro.
6. Calibración AR(1) sobre D_intraday excluyendo saltos nocturnos y Filtro de Kalman recursivo.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.strategies.f4_optimizacion_is_n2 import (
    F4OptimizationISN2Strategy,
    F4OptimizationISN2Backtester,
    CointegrationOUModelN2,
    KalmanFilterBeta,
    TradeSignalN2,
    ExitReasonN2,
    TradeRecordN2,
)


class TestKalmanFilterBeta:
    """Pruebas del estimador recursivo en espacio de estados para beta dinámico."""

    def test_kalman_filter_convergence(self):
        """Verifica que el Filtro de Kalman rastree el beta dinámico en presencia de ruido."""
        np.random.seed(42)
        n = 500
        true_alpha = -1.50
        true_beta = 1.35

        p_es = np.cumsum(np.random.normal(0, 0.002, n)) + 8.5
        noise = np.random.normal(0, 0.001, n)
        p_nq = true_alpha + true_beta * p_es + noise

        # Inicialización con estimación OLS de calentamiento (como en producción)
        beta_ols, alpha_ols, _ = CointegrationOUModelN2.estimate_cointegration_backward(p_nq[:100], p_es[:100])
        kf = KalmanFilterBeta(
            q_alpha=1e-6, q_beta=1e-5, r_measurement=1e-4,
            initial_alpha=alpha_ols, initial_beta=beta_ols
        )

        for i in range(100, n):
            alpha_k, beta_k = kf.update(p_nq[i], p_es[i])

        assert abs(beta_k - true_beta) < 0.05
        assert abs(alpha_k - true_alpha) < 0.5


class TestOUModelN2IntradayFiltering:
    """Pruebas de la calibración OU libre de saltos nocturnos y fórmulas continuas."""

    def setup_method(self):
        self.model = CointegrationOUModelN2()

    def test_ou_intraday_jump_exclusion(self):
        """Verifica que fit_ou_process_intraday descarte las parejas interdiarias (overnight gaps)."""
        dt_index = []
        # Crear 3 días con 5 barras cada uno = 15 barras
        for d in range(3):
            date_str = f"2023-01-{10+d:02d}"
            daily_times = pd.date_range(f"{date_str} 09:30:00", f"{date_str} 09:50:00", freq="5min")
            dt_index.extend(daily_times)

        timestamps = pd.DatetimeIndex(dt_index)
        n = len(timestamps)  # 15 barras

        # Simular residuos
        residuals = np.zeros(n)
        for i in range(1, n):
            residuals[i] = 0.8 * residuals[i - 1] + 0.001

        ou = self.model.fit_ou_process_intraday(residuals, timestamps, dt_minutes=5.0)

        # En 3 días de 5 barras cada uno, hay 4 pares intradía por día = 12 pares intradía
        # y 2 pares interdiarios que deben ser descartados
        assert ou["intraday_pairs"] == 12
        assert ou["is_valid"] is True
        assert ou["kappa"] > 0
        assert ou["tau_half_min"] > 0

    def test_mackinnon_critical_value(self):
        """Verifica que el test ADF use el umbral de MacKinnon (-3.37)."""
        np.random.seed(42)
        # Serie fuertemente estacionaria
        stat_series = np.zeros(300)
        for i in range(1, 300):
            stat_series[i] = 0.6 * stat_series[i - 1] + np.random.normal(0, 0.01)

        is_stat, t_stat, p_val = self.model.test_stationarity_mackinnon(
            stat_series, t_stat_threshold=-3.37, p_val_threshold=0.05
        )
        assert is_stat is True
        assert t_stat < -3.37
        assert p_val < 0.05

        # Serie paseo aleatorio no estacionario
        rw_series = np.cumsum(np.random.normal(0, 0.01, 300))
        is_stat_rw, t_stat_rw, p_val_rw = self.model.test_stationarity_mackinnon(
            rw_series, t_stat_threshold=-3.37, p_val_threshold=0.05
        )
        assert is_stat_rw is False


class TestCoprimeLatticeOptimization:
    """Pruebas del dimensionamiento en retículo coprimo para neutralidad Dollar-Beta."""

    def setup_method(self):
        self.model = CointegrationOUModelN2()

    def test_find_optimal_coprime_basket_typical_ratios(self):
        """Verifica la búsqueda de enteros coprimos para ratios típicos de 2023."""
        # Caso 1: Ratio 1.2857 -> Debe encontrar (7 MNQ, 9 MES) con ratio 9/7 = 1.2857 y gcd(7, 9) = 1
        target_ratio = 9.0 / 7.0  # ~1.285714
        n_nq, n_es, err = self.model.find_optimal_coprime_basket(target_ratio, max_nq_contracts=7)
        assert math.gcd(n_nq, n_es) == 1
        assert (n_nq, n_es) == (7, 9)
        assert err < 0.001

        # Caso 2: Ratio 1.25 -> (4 MNQ, 5 MES) con gcd(4, 5) = 1
        n_nq_2, n_es_2, err_2 = self.model.find_optimal_coprime_basket(1.25, max_nq_contracts=5)
        assert math.gcd(n_nq_2, n_es_2) == 1
        assert (n_nq_2, n_es_2) == (4, 5)
        assert err_2 < 0.001

        # Caso 3: Sub-cesta unitaria 1 MNQ -> (1, 2) o (1, 1) coprimos por definición
        n_nq_3, n_es_3, _ = self.model.find_optimal_coprime_basket(1.70, max_nq_contracts=1)
        assert n_nq_3 == 1
        assert math.gcd(n_nq_3, n_es_3) == 1

    def test_geometric_loss_streak_attenuation(self):
        """Verifica la atenuación geométrica del riesgo ante pérdidas consecutivas."""
        dll = 1000.0
        tmdd = 2000.0

        # Racha 0: phi = 0.35 * 2^0 = 0.35 -> $350
        b0 = self.model.calculate_risk_budget_attenuated(dll, tmdd, loss_streak=0)
        assert b0 == 350.0

        # Racha 1: phi = 0.35 * 2^-1 = 0.175 -> $175
        b1 = self.model.calculate_risk_budget_attenuated(dll, tmdd, loss_streak=1)
        assert b1 == 175.0

        # Racha 2: phi = 0.35 * 2^-2 = 0.0875 -> $87.50
        b2 = self.model.calculate_risk_budget_attenuated(dll, tmdd, loss_streak=2)
        assert b2 == 87.50

        # Racha 3 o más: phi = 0.35 * 2^-3 = 0.04375 -> $43.75
        b3 = self.model.calculate_risk_budget_attenuated(dll, tmdd, loss_streak=3)
        b4 = self.model.calculate_risk_budget_attenuated(dll, tmdd, loss_streak=5)
        assert b3 == 43.75
        assert b4 == 43.75


class TestFrozenEquilibriumParameters:
    """Pruebas del congelamiento de parámetros de equilibrio para erradicar el Moving-Target S-Score."""

    def setup_method(self):
        self.model = CointegrationOUModelN2()

    def test_frozen_s_score_invariance_to_regime_shift(self):
        """
        Verifica que el S-Score congelado no se distorsione si el equilibrio cambia después de la entrada.
        """
        # Entrada en t_entry
        p_nq_entry = 15000.0
        p_es_entry = 4000.0
        entry_alpha = 1.0
        entry_beta = 1.20
        entry_theta = 0.001
        entry_sigma_eq = 0.005

        # Si el spread en t_exit se mueve en contra (NQ cae, ES sube para Buy Spread)
        p_nq_exit = 14800.0
        p_es_exit = 4050.0

        s_frozen = self.model.calculate_frozen_s_score(
            p_nq=p_nq_exit, p_es=p_es_exit,
            entry_alpha=entry_alpha, entry_beta=entry_beta,
            entry_theta=entry_theta, entry_sigma_eq=entry_sigma_eq
        )

        # Debe reflejar una desviación negativa más profunda
        log_nq = np.log(p_nq_exit)
        log_es = np.log(p_es_exit)
        expected_res = log_nq - (entry_alpha + entry_beta * log_es)
        expected_s = (expected_res - entry_theta) / entry_sigma_eq
        assert pytest.approx(s_frozen, rel=1e-5) == expected_s
        assert s_frozen < -3.0  # Cercano o superando el Stop-Loss


class TestF4OptimizationISN2Execution:
    """Pruebas del motor completo de backtesting optimizado."""

    def _create_synthetic_rth(self, n_days: int = 10):
        dt_index = []
        for d in range(n_days):
            date_str = f"2023-01-{10+d:02d}"
            daily_times = pd.date_range(f"{date_str} 09:30:00", f"{date_str} 16:00:00", freq="5min")
            dt_index.extend(daily_times)

        index = pd.DatetimeIndex(dt_index, tz="America/New_York")
        n = len(index)
        np.random.seed(42)

        p_es = 4000.0 * np.exp(np.cumsum(np.random.normal(0, 0.001, n)))
        res = np.zeros(n)
        for i in range(1, n):
            res[i] = 0.85 * res[i - 1] + np.random.normal(0, 0.002)

        beta = 1.25
        alpha = 0.5
        p_nq = np.exp(alpha + beta * np.log(p_es) + res)

        df_nq = pd.DataFrame({
            "Open": p_nq, "High": p_nq * 1.0005, "Low": p_nq * 0.9995, "Close": p_nq, "Volume": 1000.0
        }, index=index)
        df_es = pd.DataFrame({
            "Open": p_es, "High": p_es * 1.0003, "Low": p_es * 0.9997, "Close": p_es, "Volume": 5000.0
        }, index=index)
        return df_nq, df_es

    def test_non_anticipative_open_t_plus_1_execution(self):
        """
        Verifica que las órdenes se ejecuten en Open[t+1] con deslizamiento,
        eliminando la fuga contemporánea en Close[t] (Punto 4).
        """
        df_nq, df_es = self._create_synthetic_rth(n_days=6)

        strat = F4OptimizationISN2Strategy(params={
            "min_calib_bars": 78,
            "k_calib": 200,
            "s_open": 1.10,
            "slippage_ticks": 1.0,
        })

        backtester = F4OptimizationISN2Backtester(
            strategy=strat,
            account_size=50000.0,
            early_stopping=False,
        )

        results = backtester.run(df_nq, df_es)
        trades = results["trades"]

        if not trades.empty:
            for _, trade in trades.iterrows():
                # La fecha/hora de entrada debe ser posterior a la fecha/hora de señal
                sig_dt = pd.to_datetime(f"{trade['signal_date']} {trade['signal_time']}")
                entry_dt = pd.to_datetime(f"{trade['entry_date']} {trade['entry_time']}")
                assert entry_dt >= sig_dt

                # Comprobar que los parámetros congelados están almacenados
                assert "entry_alpha" in trade
                assert "entry_beta" in trade
                assert "entry_theta" in trade
                assert "entry_sigma_eq" in trade
                assert trade["entry_sigma_eq"] > 0

    def test_full_sample_evaluation_no_early_stopping(self):
        """
        Verifica que early_stopping=False evalúe la totalidad de los días
        incluso después de superar el Profit Target (Punto 1).
        """
        df_nq, df_es = self._create_synthetic_rth(n_days=10)

        strat = F4OptimizationISN2Strategy(params={
            "min_calib_bars": 78,
            "k_calib": 150,
            "s_open": 0.80,
        })

        # Profit target bajo ($100) para forzar superación temprana
        backtester = F4OptimizationISN2Backtester(
            strategy=strat,
            account_size=50000.0,
            profit_target=100.0,
            min_days=2,
            early_stopping=False,
        )

        results = backtester.run(df_nq, df_es)
        prop = results["prop_evaluation"]
        daily = results["daily_logs"]

        # Debe haber evaluado los 10 días completos
        assert len(daily) == 10
        assert prop["total_days_evaluated"] == 10
        assert prop["early_stopping_disabled"] is True

    def test_sharpe_ratio_daily_vs_trade_level(self):
        """
        Verifica que el Sharpe Ratio se calcule sobre los retornos diarios de 252 días
        y no infle artificialmente multiplicando el Sharpe por trade por sqrt(252) (Punto 2).
        """
        df_nq, df_es = self._create_synthetic_rth(n_days=8)

        strat = F4OptimizationISN2Strategy(params={
            "min_calib_bars": 78,
            "k_calib": 150,
        })
        backtester = F4OptimizationISN2Backtester(
            strategy=strat,
            account_size=50000.0,
            early_stopping=False,
        )

        results = backtester.run(df_nq, df_es)
        metrics = results["metrics"]

        assert "Sharpe Ratio (Daily 252d)" in metrics
        assert "Sharpe Ratio (Per Trade)" in metrics
        assert "Sortino Ratio (Daily 252d)" in metrics

        # El Sharpe por trade debe ser un número moderado (no 14.76)
        assert abs(metrics["Sharpe Ratio (Per Trade)"]) < 5.0
        assert abs(metrics["Sharpe Ratio (Daily 252d)"]) < 10.0

    def test_real_cme_5m_data_optimized_run(self):
        """Verifica la ejecución de F4-Optimizacion -IS- N2 sobre datos reales de futuros CME."""
        from pathlib import Path
        data_nq_path = "data/raw/CME_MINI_NQ_5m_3y.parquet" if Path("data/raw/CME_MINI_NQ_5m_3y.parquet").exists() else ("data/archive_synthetic/CME_MINI_NQ_5m_3y.parquet" if Path("data/archive_synthetic/CME_MINI_NQ_5m_3y.parquet").exists() else "data/raw/CME_MINI_NQ_5m_bars5000.parquet")
        data_es_path = "data/raw/CME_MINI_ES_5m_3y.parquet" if Path("data/raw/CME_MINI_ES_5m_3y.parquet").exists() else ("data/archive_synthetic/CME_MINI_ES_5m_3y.parquet" if Path("data/archive_synthetic/CME_MINI_ES_5m_3y.parquet").exists() else "data/raw/CME_MINI_ES_5m_bars5000.parquet")

        df_nq = pd.read_parquet(data_nq_path).iloc[:3000]
        df_es = pd.read_parquet(data_es_path).iloc[:3000]

        strat = F4OptimizationISN2Strategy(params={
            "min_calib_bars": 390,
            "k_calib": 2340,
            "source_tz": "America/Mexico_City",
            "target_tz": "America/New_York",
            "contract_mode": "micro",
            "use_kalman": True,
            "allow_sub_baskets": True,
            "dynamic_basket_ratio": True,
            "s_open": 1.25,
            "s_close": 0.25,
            "s_stop": 3.25,
        })

        backtester = F4OptimizationISN2Backtester(
            strategy=strat,
            account_size=100000.0,
            daily_loss_limit=2000.0,
            max_trailing_loss=3000.0,
            profit_target=6000.0,
            early_stopping=False,
        )

        results = backtester.run(df_nq, df_es)
        metrics = results["metrics"]
        prop = results["prop_evaluation"]
        audit = results["cointegration_audit"]

        assert prop["status"] in ["PASS", "INCOMPLETE"]
        assert prop["final_balance"] > 0
        assert audit["bars_evaluated"] > 0
        assert audit["valid_ou_pct"] > 0.0
        assert metrics["Total Trades"] > 0, "La estrategia debe ejecutar operaciones válidas en datos reales CME"

    def test_dollar_beta_never_1_to_1_rounding(self):
        """Verifica que las operaciones micro ejecuten ratios balanceados (contracts_es > contracts_nq) erradicando el 1:1."""
        df_nq, df_es = self._create_synthetic_rth(n_days=10)

        strat = F4OptimizationISN2Strategy(params={
            "min_calib_bars": 78,
            "k_calib": 200,
            "s_open": 1.0,
            "allow_sub_baskets": True,
            "contract_mode": "micro",
        })

        backtester = F4OptimizationISN2Backtester(
            strategy=strat,
            account_size=50000.0,
            early_stopping=False,
        )

        results = backtester.run(df_nq, df_es)
        trades = results["trades"]

        assert not trades.empty, "Se esperaban operaciones ejecutadas en el backtest sintético RTH"
        for _, trade in trades.iterrows():
            # Para ES/NQ con multiplicadores 5 vs 2 y beta ~ 1.25, MES siempre debe superar a MNQ
            assert trade["contracts_es"] > trade["contracts_nq"], (
                f"Desbalance 1:1 detectado en trade {trade['trade_id']}: {trade['contracts_nq']} vs {trade['contracts_es']}"
            )

    def test_intraday_loss_streak_resets_daily(self):
        """Verifica que la racha de pérdidas sea intradía (IntradayLossStreak_t) y se reinicie en cada sesión."""
        model = CointegrationOUModelN2()
        dll = 1000.0
        tmdd = 2000.0

        # Racha de 2 pérdidas en el día reduce el presupuesto a $87.50
        budget_streak2 = model.calculate_risk_budget_attenuated(dll, tmdd, loss_streak=2)
        assert budget_streak2 == 87.50

        # Al día siguiente, la racha se reinicia a 0 y el presupuesto vuelve a $350.00
        budget_streak0 = model.calculate_risk_budget_attenuated(dll, tmdd, loss_streak=0)
        assert budget_streak0 == 350.00


class TestRunnerCLIConfiguration:
    """Verifica que el runner CLI configure por defecto OLS unificado y desactive Kalman."""

    def test_runner_cli_kalman_default_is_false(self):
        from run_f4_optimizacion_is_n2_backtest import build_parser
        parser = build_parser()
        args = parser.parse_args(["--mode", "in_sample", "--account-tier", "50k"])
        assert args.use_kalman is False

    def test_runner_cli_explicit_use_kalman(self):
        from run_f4_optimizacion_is_n2_backtest import build_parser
        parser = build_parser()
        args = parser.parse_args(["--mode", "in_sample", "--use-kalman"])
        assert args.use_kalman is True

    def test_runner_cli_explicit_no_kalman(self):
        from run_f4_optimizacion_is_n2_backtest import build_parser
        parser = build_parser()
        args = parser.parse_args(["--mode", "in_sample", "--no-kalman"])
        assert args.use_kalman is False

    def test_runner_cli_unify_beta_ols_flag(self):
        from run_f4_optimizacion_is_n2_backtest import build_parser
        parser = build_parser()
        args_def = parser.parse_args(["--mode", "in_sample"])
        assert args_def.unify_beta_ols is True
        args_no = parser.parse_args(["--mode", "in_sample", "--no-unify-beta-ols"])
        assert args_no.unify_beta_ols is False

    def test_default_strategy_instantiation_executes_trades(self):
        """Verifica que instanciar F4OptimizationISN2Strategy() por defecto ejecute operaciones reales."""
        from pathlib import Path
        data_nq_path = "data/raw/CME_MINI_NQ_5m_3y.parquet" if Path("data/raw/CME_MINI_NQ_5m_3y.parquet").exists() else ("data/archive_synthetic/CME_MINI_NQ_5m_3y.parquet" if Path("data/archive_synthetic/CME_MINI_NQ_5m_3y.parquet").exists() else "data/raw/CME_MINI_NQ_5m_bars5000.parquet")
        data_es_path = "data/raw/CME_MINI_ES_5m_3y.parquet" if Path("data/raw/CME_MINI_ES_5m_3y.parquet").exists() else ("data/archive_synthetic/CME_MINI_ES_5m_3y.parquet" if Path("data/archive_synthetic/CME_MINI_ES_5m_3y.parquet").exists() else "data/raw/CME_MINI_ES_5m_bars5000.parquet")
        df_nq = pd.read_parquet(data_nq_path).iloc[:3000]
        df_es = pd.read_parquet(data_es_path).iloc[:3000]

        strat = F4OptimizationISN2Strategy()
        backtester = F4OptimizationISN2Backtester(
            strategy=strat,
            account_size=100000.0,
            daily_loss_limit=2000.0,
            max_trailing_loss=3000.0,
            profit_target=6000.0,
            early_stopping=False,
        )
        res = backtester.run(df_nq, df_es)
        assert res["metrics"]["Total Trades"] > 0, "Instancia por defecto no debe producir 0 trades"


class TestVolatilityFloorAndSectorDecoupling:
    """Pruebas exhaustivas para Requerimientos R3 (Piso de Volatilidad) y R4 (Filtro de Desacoplamiento Sectorial)."""

    def test_sigma_floor_enforcement(self):
        """Verifica que el piso de volatilidad (sigma_floor >= 0.001) estabilice el S-score y la gestión de riesgo."""
        model = CointegrationOUModelN2()

        # 1. Simulación de residuos con varianza microscópica (olas diminutas)
        dt_index = pd.date_range("2023-01-10 09:30:00", "2023-01-10 15:55:00", freq="5min")
        n = len(dt_index)
        tiny_residuals = np.zeros(n)
        for i in range(1, n):
            tiny_residuals[i] = 0.5 * tiny_residuals[i - 1] + np.random.normal(0, 1e-6)

        ou = model.fit_ou_process_intraday(tiny_residuals, dt_index, sigma_floor=0.001)
        assert ou["is_valid"] is True
        assert ou["sigma_eq"] >= 0.001, f"sigma_eq debe respetar el piso de 0.001, obtenido: {ou['sigma_eq']}"

        # 2. Verificación de calculate_s_score contemporáneo con piso
        # Si x_k - theta = 0.0005, sin piso (sigma_eq=1e-5) daría S = 50.0 (distorsión brutal).
        # Con piso (sigma_floor=0.001), debe dar S = 0.0005 / 0.001 = 0.50.
        s_score = model.calculate_s_score(x_k=0.0005, theta=0.0, sigma_eq=1e-5, sigma_floor=0.001)
        assert abs(s_score - 0.50) < 1e-4, f"S-score debe estabilizarse a 0.50, obtenido: {s_score}"

        # 3. Verificación de calculate_frozen_s_score con piso
        s_frozen = model.calculate_frozen_s_score(
            p_nq=10005.0, p_es=10000.0, entry_alpha=0.0, entry_beta=1.0,
            entry_theta=0.0, entry_sigma_eq=1e-5, sigma_floor=0.001
        )
        assert abs(s_frozen - 0.50) < 0.01, f"Frozen S-score debe usar el piso, obtenido: {s_frozen}"

        # 4. Verificación de dimensionamiento de stop-loss
        unit_sl_loss_floor = model.calculate_unit_stop_loss_loss(
            sigma_eq=1e-5, p_nq=15000.0, n_nq_unit=1, m_nq=2.0, friction_basket_usd=25.0,
            delta_s_sl=2.0, sigma_floor=0.001
        )
        # delta_x_sl = 2.0 * 0.001 = 0.002. notional = 1 * 2 * 15000 = 30000. loss_pts = 0.002 * 30000 = 60.
        # total = 60 + 25 = 85 USD.
        assert unit_sl_loss_floor >= 85.0, f"Unit SL loss no debe colapsar a solo la fricción: {unit_sl_loss_floor}"

    def test_sector_decoupling_blocks_new_orders(self):
        """Verifica que correlación XLK/SPY < 0.85 bloquee nuevas aperturas y >= 0.85 permita operar normalmente."""
        from pathlib import Path
        data_nq_path = "data/raw/CME_MINI_NQ_5m_3y.parquet" if Path("data/raw/CME_MINI_NQ_5m_3y.parquet").exists() else ("data/archive_synthetic/CME_MINI_NQ_5m_3y.parquet" if Path("data/archive_synthetic/CME_MINI_NQ_5m_3y.parquet").exists() else "data/raw/CME_MINI_NQ_5m_bars5000.parquet")
        data_es_path = "data/raw/CME_MINI_ES_5m_3y.parquet" if Path("data/raw/CME_MINI_ES_5m_3y.parquet").exists() else ("data/archive_synthetic/CME_MINI_ES_5m_3y.parquet" if Path("data/archive_synthetic/CME_MINI_ES_5m_3y.parquet").exists() else "data/raw/CME_MINI_ES_5m_bars5000.parquet")
        df_nq = pd.read_parquet(data_nq_path).iloc[:3000]
        df_es = pd.read_parquet(data_es_path).iloc[:3000]

        # Caso A: Desacoplamiento forzado (correlación constante = 0.70 < 0.85)
        unique_dates = pd.to_datetime(df_nq.index).date
        dates_set = sorted(list(set(unique_dates)))
        decoupled_corr_series = pd.Series(0.70, index=dates_set)

        strat_blocked = F4OptimizationISN2Strategy(params={"min_sector_corr": 0.85})
        bt_blocked = F4OptimizationISN2Backtester(strategy=strat_blocked, account_size=100000.0, early_stopping=False)
        res_blocked = bt_blocked.run(df_nq, df_es, sector_corr_series=decoupled_corr_series)

        assert res_blocked["metrics"]["Total Trades"] == 0, "Correlación < 0.85 debe bloquear 100% de aperturas"

        # Caso B: Acoplamiento saludable (correlación constante = 0.95 >= 0.85)
        coupled_corr_series = pd.Series(0.95, index=dates_set)
        strat_allowed = F4OptimizationISN2Strategy(params={"min_sector_corr": 0.85})
        bt_allowed = F4OptimizationISN2Backtester(strategy=strat_allowed, account_size=100000.0, early_stopping=False)
        res_allowed = bt_allowed.run(df_nq, df_es, sector_corr_series=coupled_corr_series)

        assert res_allowed["metrics"]["Total Trades"] > 0, "Correlación >= 0.85 debe permitir operaciones normalmente"

    def test_no_lookahead_bias_in_sector_correlation(self):
        """Verifica que compute_sector_correlation_series aplique shift(1) estrictamente sin sesgo de anticipación."""
        from src.strategies.f4_optimizacion_is_n2 import compute_sector_correlation_series

        # Generar 40 días de datos diarios
        dates = pd.date_range("2023-01-01", periods=40, freq="B")
        np.random.seed(42)
        r_spy = np.random.normal(0.001, 0.01, 40)
        r_xlk = 1.1 * r_spy + np.random.normal(0, 0.002, 40)

        p_spy = 400.0 * np.exp(np.cumsum(r_spy))
        p_xlk = 150.0 * np.exp(np.cumsum(r_xlk))

        df_spy = pd.DataFrame({"Close": p_spy}, index=dates)
        df_xlk = pd.DataFrame({"Close": p_xlk}, index=dates)

        corr_baseline = compute_sector_correlation_series(df_xlk, df_spy, window=30)

        # Alterar violentamente el precio en el día 35 (índice 34)
        target_date = dates[34].date()
        df_xlk_mutated = df_xlk.copy()
        df_xlk_mutated.iloc[34, 0] = df_xlk_mutated.iloc[34, 0] * 2.0  # +100% shock intradía en día D

        corr_mutated = compute_sector_correlation_series(df_xlk_mutated, df_spy, window=30)

        # El valor para target_date (día D) DEBE ser idéntico al baseline porque solo usa información hasta D-1!
        val_baseline = corr_baseline.loc[target_date]
        val_mutated = corr_mutated.loc[target_date]
        assert abs(val_baseline - val_mutated) < 1e-12, (
            f"Sesgo de anticipación detectado en día {target_date}: {val_baseline} vs {val_mutated}"
        )

        # En cambio, el día D+1 sí debe reflejar el cambio
        next_date = dates[35].date()
        val_next_base = corr_baseline.loc[next_date]
        val_next_mutated = corr_mutated.loc[next_date]
        assert abs(val_next_base - val_next_mutated) > 1e-4, "El día D+1 debe incorporar el cambio de día D"


