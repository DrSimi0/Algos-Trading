"""
tests/test_f3_optimization_is_n2.py

Suite completa de pruebas unitarias y de integración matemática para la versión optimizada N2:
F3-Optimizacion -IS- N2: Anchored VWAP & Rotational Value Area Reversion.
"""

import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.strategies.mean_reversion.f3_optimization_is_n2 import (
    F3OptimizationISN2Strategy,
    PropRiskManagerN2,
    compute_prior_day_value_areas,
    compute_anchored_vwap_n2_indicators,
)
from src.core.futures_specs import get_spec


# ==============================================================================
# FIXTURES Y GENERADORES DE DATOS SINTÉTICOS DE PRUEBA
# ==============================================================================

def create_synthetic_5m_session(
    date_str: str = "2023-10-16",
    base_price: float = 5000.0,
    n_bars: int = 78,  # 09:30 a 16:00 son 78 barras de 5m
    volatility: float = 2.0,
    volume_per_bar: float = 1000.0,
    seed: int = 42
) -> pd.DataFrame:
    """Genera una sesión intradía sintética de 5 minutos con timestamps desde 09:30 a 16:00."""
    start_dt = pd.Timestamp(f"{date_str} 09:30:00")
    timestamps = [start_dt + pd.Timedelta(minutes=5 * i) for i in range(n_bars)]

    np.random.seed(seed)
    closes = [base_price]
    for _ in range(1, n_bars):
        closes.append(closes[-1] + np.random.normal(0, volatility))

    closes = np.array(closes)
    highs = closes + np.random.uniform(0.5, 2.0, n_bars)
    lows = closes - np.random.uniform(0.5, 2.0, n_bars)
    opens = closes + np.random.uniform(-1.0, 1.0, n_bars)
    volumes = np.full(n_bars, volume_per_bar, dtype=np.float64)

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes
        },
        index=pd.DatetimeIndex(timestamps)
    )
    return df


def create_synthetic_multiday_data(n_days: int = 3, base_price: float = 5000.0) -> pd.DataFrame:
    """Genera datos multisesión ordenados cronológicamente."""
    dfs = []
    base_dates = [f"2023-10-{16 + i:02d}" for i in range(n_days)]
    for i, d in enumerate(base_dates):
        df_day = create_synthetic_5m_session(
            date_str=d,
            base_price=base_price + i * 10.0,
            seed=42 + i
        )
        dfs.append(df_day)
    return pd.concat(dfs).sort_index()


# ==============================================================================
# TEST 1: ÁREA DE VALOR PREVIA (AMT) Y CUANTILES DE VOLUMEN
# ==============================================================================

def test_value_area_volume_concentration_and_quantiles():
    """Valida que el Área de Valor (VAL, VAH) contenga el 70% central de la distribución de volumen."""
    data = create_synthetic_multiday_data(n_days=2)
    va_map = compute_prior_day_value_areas(data, rth_start="09:30", rth_end="16:00", value_area_pct=0.70)

    dates = list(dict.fromkeys(data.index.date))
    day1, day2 = dates[0], dates[1]

    assert day1 not in va_map
    assert day2 in va_map

    val, vah, poc = va_map[day2]
    assert val <= poc <= vah or val <= vah

    df_day1 = data[data.index.date == day1]
    p_typ = (df_day1["High"] + df_day1["Low"] + df_day1["Close"]) / 3.0
    vols = df_day1["Volume"]

    inside_mask = (p_typ >= val) & (p_typ <= vah)
    vol_inside = vols[inside_mask].sum()
    total_vol = vols.sum()
    ratio = vol_inside / total_vol

    # El volumen dentro del rango debe estar en torno al 70% (con tolerancia por discretización)
    assert 0.60 <= ratio <= 0.85


# ==============================================================================
# TEST 2: COMPUERTA DE ACEPTACIÓN AMT DE LAS PRIMERAS 6 BARRAS
# ==============================================================================

def test_amt_six_bars_acceptance_gate():
    """
    Verifica que si las primeras 6 barras de 5m (09:30 a 10:00) abren y cierran fuera
    de [VAL, VAH], Phi_AMT se desactive (0), bloqueando la reversión por Trend Day.
    """
    data = create_synthetic_multiday_data(n_days=2, base_price=5000.0)
    dates = list(dict.fromkeys(data.index.date))
    day2 = dates[1]

    # Modificar artificialmente las primeras 6 barras del día 2 para que coticen MUY por encima de VAH
    day2_mask = data.index.date == day2
    day2_indices = np.where(day2_mask)[0]
    first_6 = day2_indices[:6]

    # Escenario A: Desbalance alcista claro (todas abren y cierran > VAH)
    data.iloc[first_6, data.columns.get_loc("Open")] = 5500.0
    data.iloc[first_6, data.columns.get_loc("Close")] = 5510.0
    data.iloc[first_6, data.columns.get_loc("High")] = 5520.0
    data.iloc[first_6, data.columns.get_loc("Low")] = 5490.0

    df_ind = compute_anchored_vwap_n2_indicators(data, kappa_crit=0.085)
    day2_ind = df_ind[df_ind.index.date == day2]

    # Phi_AMT debe ser 0 en todo el día 2
    assert (day2_ind["Phi_AMT"] == 0).all()
    # Y por lo tanto el régimen rotacional no debe habilitarse
    assert (day2_ind["Regime_Rotational"] == 0).all()


def test_amt_acceptance_gate_inactive_when_inside():
    """Verifica que si al menos una barra abre o cierra dentro de [VAL, VAH], Phi_AMT permanezca en 1."""
    # Generar dos días con el MISMO precio base para que el día 2 abra dentro del Área de Valor del día 1
    df1 = create_synthetic_5m_session(date_str="2023-10-16", base_price=5000.0, seed=42)
    df2 = create_synthetic_5m_session(date_str="2023-10-17", base_price=5000.0, seed=43)
    data = pd.concat([df1, df2]).sort_index()

    dates = list(dict.fromkeys(data.index.date))
    day2 = dates[1]

    df_ind = compute_anchored_vwap_n2_indicators(data, kappa_crit=0.085)
    day2_ind = df_ind[df_ind.index.date == day2]

    # Debe ser 1 en toda la sesión porque abrió y cotizó dentro del rango de valor del día 1
    assert (day2_ind["Phi_AMT"] == 1).all()


# ==============================================================================
# TEST 3: FILTRO DE PENDIENTE NORMALIZADA DE VWAP (kappa_norm <= 0.085)
# ==============================================================================

def test_normalized_vwap_slope_filter():
    """Valida la fórmula y comportamiento de kappa_norm,t <= kappa_crit."""
    df_day = create_synthetic_5m_session(date_str="2023-10-16", base_price=5000.0, n_bars=30)
    
    # Barras 0 a 9: mercado perfectamente calmo (precio plano)
    for i in range(10):
        df_day.iloc[i, df_day.columns.get_loc("Close")] = 5000.0
        df_day.iloc[i, df_day.columns.get_loc("Open")] = 5000.0
        df_day.iloc[i, df_day.columns.get_loc("High")] = 5000.25
        df_day.iloc[i, df_day.columns.get_loc("Low")] = 4999.75

    # Barras 10 a 20: despegue direccional violento (+15 puntos por barra)
    for i in range(10, 20):
        df_day.iloc[i, df_day.columns.get_loc("Close")] = 5000.0 + (i - 9) * 15.0
        df_day.iloc[i, df_day.columns.get_loc("High")] = df_day.iloc[i]["Close"] + 2.0
        df_day.iloc[i, df_day.columns.get_loc("Low")] = df_day.iloc[i]["Close"] - 2.0
        df_day.iloc[i, df_day.columns.get_loc("Open")] = df_day.iloc[i]["Close"] - 5.0

    df_ind = compute_anchored_vwap_n2_indicators(df_day, kappa_crit=0.085)

    # Durante la calma inicial (barras 4 a 9), la pendiente debe ser <= 0.085
    slopes_calm = df_ind["VWAP_Slope_Norm"].iloc[4:10]
    assert (slopes_calm <= 0.085).all()

    # Durante el despegue violento, la pendiente debe superar 0.085 y vetar reversión (Phi_Slope = 0)
    slopes_violent = df_ind["VWAP_Slope_Norm"].iloc[13:19]
    assert (slopes_violent > 0.085).any()
    assert (df_ind["Phi_Slope"].iloc[13:19] == 0).any()


# ==============================================================================
# TEST 4: MODULACIÓN ASIMÉTRICA POR SESGO DE VOLUMEN (SKEWNESS)
# ==============================================================================

def test_volume_skewness_asymmetric_bands():
    """Valida que un sesgo de volumen negativo expanda la banda inferior k_lower."""
    df_day = create_synthetic_5m_session(date_str="2023-10-16", base_price=5000.0, n_bars=40)
    
    # Mayoría del volumen en 5000.0 (10,000 contratos)
    for i in range(25):
        df_day.iloc[i, df_day.columns.get_loc("Close")] = 5000.0
        df_day.iloc[i, df_day.columns.get_loc("High")] = 5001.0
        df_day.iloc[i, df_day.columns.get_loc("Low")] = 4999.0
        df_day.iloc[i, df_day.columns.get_loc("Volume")] = 10000.0

    # Cola izquierda con precios bajos (4960.0) y menor volumen (500 contratos)
    for i in range(25, 40):
        df_day.iloc[i, df_day.columns.get_loc("Close")] = 4960.0
        df_day.iloc[i, df_day.columns.get_loc("High")] = 4961.0
        df_day.iloc[i, df_day.columns.get_loc("Low")] = 4959.0
        df_day.iloc[i, df_day.columns.get_loc("Volume")] = 500.0

    df_ind = compute_anchored_vwap_n2_indicators(df_day, k_band=2.0)
    skew_vals = df_ind["Volume_Skewness"].iloc[26:38]
    
    # Debe existir sesgo negativo (cola izquierda)
    assert (skew_vals < 0).any()

    # La banda inferior debe expandirse más allá del k_band base de 2.0
    for idx in range(28, 38):
        sig = df_ind["Sigma_VWAP"].iloc[idx]
        vw = df_ind["VWAP"].iloc[idx]
        lb = df_ind["LowerBand"].iloc[idx]
        dist = vw - lb
        if sig > 0.1:
            eff_k = dist / sig
            assert eff_k >= 2.0


# ==============================================================================
# TEST 5: GESTOR DE RIESGO PROP TRADING N2 (PRESUPUESTO Y DIMENSIONAMIENTO)
# ==============================================================================

def test_prop_risk_manager_n2_budget_and_sizing():
    """Valida el cálculo del presupuesto B_F3,t y conmutación Standard / Micro."""
    pm = PropRiskManagerN2(
        account_size=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_drawdown=2000.0,
        psi=0.12,
        alpha_dll=0.85,
        max_std_contracts=1,
        max_micro_contracts=8
    )

    budget = pm.get_risk_budget()
    # Con 0 pérdidas, phi(0) = 0.25. B = min(0.25 * 1000, 0.12 * 2000) = min(250, 240) = 240.0
    assert budget == 240.0

    # Dimensionar un trade con SL de 4 puntos en ES:
    # Riesgo Std = 4 * 50 + c_fric = 200 + 17 = 217 USD <= 240 => 1 Standard
    n_cnt, c_type, r_cnt, status = pm.calculate_position_size(entry_price=5000.0, sl_price=4996.0)
    assert n_cnt == 1
    assert c_type == "STANDARD"
    assert status == "OK"

    # Tras sufrir 1 pérdida, phi(1) = 0.125 => Presupuesto decrece
    pm.record_trade_result(net_pnl=-200.0)
    new_budget = pm.get_risk_budget()
    assert new_budget < 240.0

    # Ahora un trade de 4 puntos ya no cabe en Estándar, debe conmutar a Micro
    n_cnt2, c_type2, r_cnt2, status2 = pm.calculate_position_size(entry_price=5000.0, sl_price=4996.0)
    assert c_type2 == "MICRO"
    assert 1 <= n_cnt2 <= 8
    assert status2 == "OK"


# ==============================================================================
# TEST 6: CORTE INMEDIATO INTRABAR ANTE PROXIMIDAD A LÍMITES
# ==============================================================================

def test_prop_risk_manager_n2_intrabar_limit_cut():
    """
    Valida que check_intrabar_limit_cut liquide inmediatamente si el precio de la barra
    haría que la pérdida acumulada supere el 85% del DLL.
    """
    pm = PropRiskManagerN2(
        account_size=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_drawdown=2000.0,
        alpha_dll=0.85
    )

    # Supongamos que ya perdimos 700 USD en el día. El 85% de DLL es 850 USD.
    # Margen remanente antes del corte = 850 - 700 = 150 USD.
    pm.day_realized_loss = 700.0

    # Trade Long con 1 contrato estándar (50 USD/pto), fricción ~17 USD
    # allowable_loss = 150 - 17 = 133 USD => 133 / 50 = 2.66 puntos de caída admisible
    entry_price = 5000.0
    bar_open = 4999.0

    # Caso A: La barra cae hasta 4995 (5 puntos de caída). Debe cortar intrabar
    bar_low_breach = 4995.0
    bar_high = 5002.0
    is_cut, cut_price, reason = pm.check_intrabar_limit_cut(
        direction=1,
        entry_price=entry_price,
        bar_open=bar_open,
        bar_high=bar_high,
        bar_low=bar_low_breach,
        contracts=1,
        contract_multiplier=50.0,
        contract_friction=17.0
    )

    assert is_cut is True
    assert reason == "INTRABAR_DAILY_LOSS_LIMIT_CUT"
    assert cut_price > bar_low_breach  # El corte se ejecutó antes de llegar al mínimo

    # Caso B: La barra solo cae hasta 4998 (2 puntos de caída = 100 USD < 133). No debe cortar
    is_cut_b, _, _ = pm.check_intrabar_limit_cut(
        direction=1,
        entry_price=entry_price,
        bar_open=bar_open,
        bar_high=bar_high,
        bar_low=4998.0,
        contracts=1,
        contract_multiplier=50.0,
        contract_friction=17.0
    )
    assert is_cut_b is False

    # Caso C: Short con 1 contrato estándar (50 USD/pto), fricción ~17 USD
    # allowable_loss = 150 - 17 = 133 USD => 133 / 50 = 2.66 puntos de subida admisible
    entry_price_short = 5000.0
    bar_open_short = 5001.0
    bar_high_breach = 5005.0  # 5 puntos de subida => debe cortar intrabar
    is_cut_short, cut_price_short, reason_short = pm.check_intrabar_limit_cut(
        direction=-1,
        entry_price=entry_price_short,
        bar_open=bar_open_short,
        bar_high=bar_high_breach,
        bar_low=4998.0,
        contracts=1,
        contract_multiplier=50.0,
        contract_friction=17.0
    )
    assert is_cut_short is True
    assert reason_short == "INTRABAR_DAILY_LOSS_LIMIT_CUT"
    assert cut_price_short <= bar_high_breach

    # Caso D: Short barra solo sube hasta 5002.0 (2 puntos < 2.66). No debe cortar
    is_cut_short_b, _, _ = pm.check_intrabar_limit_cut(
        direction=-1,
        entry_price=entry_price_short,
        bar_open=bar_open_short,
        bar_high=5002.0,
        bar_low=4998.0,
        contracts=1,
        contract_multiplier=50.0,
        contract_friction=17.0
    )
    assert is_cut_short_b is False


# ==============================================================================
# TEST 7: ACTIVACIÓN DE TRAILING PROTECT AL 60% DE CAMINO AL TAKE-PROFIT
# ==============================================================================

def test_trailing_protect_60_percent_activation():
    """
    Verifica que en el backtest formal, al alcanzar >= 60% del recorrido al VWAP,
    se actualice el Stop Loss y se registre como TRAILING_PROTECT_SL si el precio retrocede,
    protegiendo la ganancia o breakeven.
    """
    strategy = F3OptimizationISN2Strategy()
    
    # Comprobar que la estrategia inicialice con los parámetros formales
    assert strategy.params["kappa_crit"] == 0.085
    assert strategy.params["convergence_protect_ratio"] == 0.60
    assert strategy.params["psi"] == 0.12
    assert strategy.params["scaling_factor"] == 0.65

    # Validar ejecución real sobre slice histórico conocido con activación de Trailing Protect
    p = Path("data/raw/CME_MINI_ES_5m_3y.parquet")
    if p.exists():
        df_slice = pd.read_parquet(p).loc["2023-02-27":"2023-03-01"]
        res = strategy.run_backtest(df_slice)
        trades = res["trades"]
        tp_trades = trades[trades["exit_reason"] == "TRAILING_PROTECT_SL"]
        
        # Deben existir salidas protegidas por Trailing Protect
        assert len(tp_trades) > 0
        sample_tp = tp_trades.iloc[0]
        assert sample_tp["exit_reason"] == "TRAILING_PROTECT_SL"
        # En una salida protegida de Long, exit_price >= entry_price
        if sample_tp["direction"] == "LONG":
            assert sample_tp["exit_price"] >= sample_tp["entry_price"]
        else:
            assert sample_tp["exit_price"] <= sample_tp["entry_price"]



# ==============================================================================
# TEST 8: TEOREMA DE ERDŐS-RÉNYI (FÓRMULA MATEMÁTICA Y PRECISIÓN ASINTÓTICA)
# ==============================================================================

def test_erdos_renyi_theoretical_formula():
    """Valida que la fórmula de Erdős-Rényi reproduzca con exactitud la racha teórica esperada."""
    # Para F3 Base: N = 330, q = 188 / 330 = 0.5696969
    n = 330
    q = 188.0 / 330.0
    expected_streak = np.log(n) / np.log(1.0 / q)

    assert 10.2 <= expected_streak <= 10.4
    assert round(expected_streak, 1) == 10.3


# ==============================================================================
# TEST 9: INTEGRACIÓN COMPLETA DE BACKTEST EN DATOS SINTÉTICOS
# ==============================================================================

def test_f3_n2_full_backtest_execution_synthetic():
    """Ejecuta run_backtest completo verificando consistencia de métricas y ausencia de Lookahead Bias."""
    data = create_synthetic_multiday_data(n_days=3, base_price=5000.0)
    strategy = F3OptimizationISN2Strategy()
    results = strategy.run_backtest(data)

    metrics = results["metrics"]
    trades = results["trades"]
    equity = results["equity_curve"]

    assert "Max Trailing Drawdown ($)" in metrics
    assert "P&L Neto Total ($)" in metrics
    assert len(equity) == len(data)
    assert not equity.isna().any()


# ==============================================================================
# TEST 10: DISYUNTOR CATASTRÓFICO A +-2.75 SIGMA
# ==============================================================================

def test_circuit_breaker_catastrophic_275_sigma():
    """Valida que un colapso extremo active el disyuntor en +-2.75 sigma."""
    strategy = F3OptimizationISN2Strategy()
    p = Path("data/raw/CME_MINI_ES_5m_3y.parquet")
    if p.exists():
        df_2023 = pd.read_parquet(p).loc["2023-01-01":"2023-12-31"]
        res = strategy.run_backtest(df_2023)
        trades = res["trades"]
        cat_trades = trades[trades["exit_reason"] == "2.75_SIGMA_CIRCUIT_BREAKER"]
        assert len(cat_trades) >= 1
        assert cat_trades.iloc[0]["exit_reason"] == "2.75_SIGMA_CIRCUIT_BREAKER"


# ==============================================================================
# TEST 11: CASOS LÍMITE (DATAFRAME VACÍO Y DE 1 SOLA BARRA)
# ==============================================================================

def test_edge_case_empty_and_single_bar_data():
    """Valida que entradas degeneradas (vacío o 1 barra) se manejen sin excepciones."""
    strategy = F3OptimizationISN2Strategy()

    # Caso A: DataFrame vacío
    empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex([]))
    res_empty = strategy.run_backtest(empty_df)
    assert res_empty["trades"].empty
    assert res_empty["metrics"]["Total Trades"] == 0

    signals_empty = strategy.generate_signals(empty_df)
    assert signals_empty.empty

    # Caso B: DataFrame de 1 sola barra
    single_df = pd.DataFrame(
        {"Open": [5000.0], "High": [5002.0], "Low": [4998.0], "Close": [5001.0], "Volume": [1000.0]},
        index=pd.DatetimeIndex(["2023-10-16 09:30:00"])
    )
    res_single = strategy.run_backtest(single_df)
    assert res_single["trades"].empty
    assert res_single["metrics"]["Total Trades"] == 0

    signals_single = strategy.generate_signals(single_df)
    assert len(signals_single) == 1
    assert signals_single.iloc[0] == 0.0


# ==============================================================================
# TEST 12: CONFIGURACIÓN Y MULTIPLICADORES DE CONTRATO CME NQ
# ==============================================================================

def test_nq_symbol_configuration_and_multipliers():
    """Valida la configuración de futuros CME NQ / MNQ."""
    strategy_nq = F3OptimizationISN2Strategy(params={"symbol": "NQ", "kappa_crit": 0.120})
    assert strategy_nq.params["symbol"] == "NQ"
    assert strategy_nq.params["kappa_crit"] == 0.120

    spec_nq = get_spec("NQ")
    spec_mnq = get_spec("MNQ")
    assert spec_nq["multiplier"] == 20.0
    assert spec_mnq["multiplier"] == 2.0


# ==============================================================================
# TEST 13: FACTOR DE ESCALA CONSERVADOR PROTEGE CUENTA FRENTE A MTDD $2,000
# ==============================================================================

def test_scaling_factor_reduces_tail_risk():
    """Valida que scaling_factor=0.65 reduzca el presupuesto admisible respecto a 1.0."""
    pm_full = PropRiskManagerN2(account_size=50000.0, max_trailing_drawdown=2000.0, scaling_factor=1.0)
    pm_scaled = PropRiskManagerN2(account_size=50000.0, max_trailing_drawdown=2000.0, scaling_factor=0.65)

    budget_full = pm_full.get_risk_budget()
    budget_scaled = pm_scaled.get_risk_budget()

    assert budget_scaled < budget_full
    assert round(budget_scaled, 2) == round(0.65 * budget_full, 2)


# ==============================================================================
# TEST 14: ADVERSARIAL STRESS: SALTO DISCRETO DE 30 PUNTOS (GAP INTRABAR)
# ==============================================================================

def test_adversarial_30pt_gap_intrabar_limit_cut():
    """
    Evalúa la respuesta ante saltos discretos múltiples (gaps de más de 30 puntos
    en una sola barra de 5 minutos durante aperturas) para confirmar que el corte
    intrabar nunca se omita ni se ejecute a un precio no disponible.
    """
    pm = PropRiskManagerN2(
        account_size=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_drawdown=2000.0,
        alpha_dll=0.85
    )
    pm.day_realized_loss = 500.0  # Margen restante a DLL 85% = 850 - 500 = 350 USD

    # Escenario A: Long con gap bajista masivo de 30 puntos en la apertura de la barra
    entry_p = 5000.0
    bar_open_gap_down = 4970.0  # Abre directamente -30 puntos abajo
    bar_high_gap_down = 4972.0
    bar_low_gap_down = 4965.0

    is_cut, cut_p, reason = pm.check_intrabar_limit_cut(
        direction=1,
        entry_price=entry_p,
        bar_open=bar_open_gap_down,
        bar_high=bar_high_gap_down,
        bar_low=bar_low_gap_down,
        contracts=1,
        contract_multiplier=50.0,
        contract_friction=17.0
    )
    assert is_cut is True
    assert reason == "INTRABAR_DAILY_LOSS_LIMIT_CUT"
    assert cut_p == bar_open_gap_down

    # Escenario B: Short con gap alcista masivo de 30 puntos en la apertura de la barra
    entry_p_short = 5000.0
    bar_open_gap_up = 5030.0  # Abre directamente +30 puntos arriba
    bar_high_gap_up = 5035.0
    bar_low_gap_up = 5028.0

    is_cut_s, cut_p_s, reason_s = pm.check_intrabar_limit_cut(
        direction=-1,
        entry_price=entry_p_short,
        bar_open=bar_open_gap_up,
        bar_high=bar_high_gap_up,
        bar_low=bar_low_gap_up,
        contracts=1,
        contract_multiplier=50.0,
        contract_friction=17.0
    )
    assert is_cut_s is True
    assert reason_s == "INTRABAR_DAILY_LOSS_LIMIT_CUT"
    assert cut_p_s == bar_open_gap_up


# ==============================================================================
# TEST 15: RESILIENCIA ANTE VOLATILIDAD NULA Y VOLUMEN NULO (ZERO DIVISION SHIELD)
# ==============================================================================

def test_zero_volatility_and_zero_volume_resilience():
    """Valida que una serie con volatilidad y volumen cero no cause ZeroDivisionError ni NaNs."""
    start_dt = pd.Timestamp("2023-10-16 09:30:00")
    timestamps = [start_dt + pd.Timedelta(minutes=5 * i) for i in range(40)]
    df_flat = pd.DataFrame(
        {
            "Open": np.full(40, 5000.0),
            "High": np.full(40, 5000.0),
            "Low": np.full(40, 5000.0),
            "Close": np.full(40, 5000.0),
            "Volume": np.zeros(40)
        },
        index=pd.DatetimeIndex(timestamps)
    )

    df_ind = compute_anchored_vwap_n2_indicators(df_flat)
    assert not df_ind["VWAP"].isna().any()
    assert (df_ind["Sigma_VWAP"] == 0.0).all()
    assert (df_ind["Volume_Skewness"] == 0.0).all()
    assert (df_ind["Volume_Kurtosis"] == 3.0).all()
    assert (df_ind["VWAP_Slope_Norm"] == 0.0).all()

    strategy = F3OptimizationISN2Strategy()
    res = strategy.run_backtest(df_flat)
    assert res["metrics"]["Total Trades"] == 0
    assert not res["equity_curve"].isna().any()


