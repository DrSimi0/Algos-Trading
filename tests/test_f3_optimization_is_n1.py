"""
tests/test_f3_optimization_is_n1.py

Suite completa de pruebas unitarias y de integración matemática para la versión alterna optimizada:
F3-Optimizacion -IS- N1: Anchored VWAP & Rotational Value Area Reversion.
"""

import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.strategies.mean_reversion.f3_optimization_is_n1 import (
    F3OptimizationISN1Strategy,
    PropRiskManagerOptimized,
    compute_anchored_vwap_optimized_bands,
    compute_prior_day_value_areas,
    calculate_adx,
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
# TIER 1: ÁREA DE VALOR PREVIA (AMT) Y PERFIL DE VOLUMEN
# ==============================================================================

def test_value_area_volume_concentration_and_quantiles():
    """Valida que el Área de Valor (VAL, VAH) contenga el 70% central de la distribución de volumen."""
    data = create_synthetic_multiday_data(n_days=2)
    va_map = compute_prior_day_value_areas(data, rth_start="09:30", rth_end="16:00", value_area_pct=0.70)

    # El día 2 (segundo día) debe tener Área de Valor calculada a partir del día 1
    dates = list(dict.fromkeys(data.index.date))
    day1, day2 = dates[0], dates[1]

    assert day1 not in va_map  # El primer día no tiene día previo
    assert day2 in va_map

    val, vah, poc = va_map[day2]
    assert val <= poc <= vah or val <= vah

    # Verificar que el rango [val, vah] contenga aproximadamente el 70% del volumen del día 1
    df_day1 = data[data.index.date == day1]
    p_typ = (df_day1["High"] + df_day1["Low"] + df_day1["Close"]) / 3.0
    vols = df_day1["Volume"]

    in_va_mask = (p_typ >= val) & (p_typ <= vah)
    vol_in_va = vols[in_va_mask].sum()
    total_vol = vols.sum()
    va_ratio = vol_in_va / total_vol

    # La aproximación por cuantiles (15th a 85th) debe contener alrededor de 70% del volumen
    assert 0.65 <= va_ratio <= 0.85


def test_poc_highest_volume_detection():
    """Valida que el Point of Control (POC) coincida con el precio de mayor volumen."""
    timestamps = [pd.Timestamp(f"2023-10-16 09:30:00") + pd.Timedelta(minutes=5 * i) for i in range(10)]
    prices = [5000.0, 5002.0, 5005.0, 5003.0, 5001.0, 5000.0, 5004.0, 5002.0, 5006.0, 5005.0]
    # Volumen con un pico deliberado en 5005.0 (índice 2)
    volumes = [100.0, 150.0, 5000.0, 200.0, 100.0, 120.0, 80.0, 150.0, 90.0, 110.0]

    df = pd.DataFrame({
        "Open": prices,
        "High": prices,
        "Low": prices,
        "Close": prices,
        "Volume": volumes
    }, index=pd.DatetimeIndex(timestamps))

    # Añadir un segundo día para que el mapeo entregue el POC
    df_day2 = df.copy()
    df_day2.index = df_day2.index + pd.Timedelta(days=1)
    df_combined = pd.concat([df, df_day2])

    va_map = compute_prior_day_value_areas(df_combined, rth_start="09:30", rth_end="16:00")
    day2_date = df_day2.index[0].date()

    val, vah, poc = va_map[day2_date]
    assert np.isclose(poc, 5005.0)


# ==============================================================================
# TIER 2: AVWAP, SESGO (SKEWNESS), CURTOSIS Y BANDAS ASIMÉTRICAS
# ==============================================================================

def test_typical_price_and_vwap_exactness():
    """Valida la fórmula de precio típico y VWAP ponderado por volumen."""
    data = create_synthetic_5m_session(n_bars=10, volume_per_bar=100.0)
    df = compute_anchored_vwap_optimized_bands(data, anchor_time="09:30")

    expected_ptyp = (data["High"] + data["Low"] + data["Close"]) / 3.0
    np.testing.assert_allclose(df["P_typ"].values, expected_ptyp.values, rtol=1e-6)

    expected_vwap = expected_ptyp.expanding().mean().values
    np.testing.assert_allclose(df["VWAP"].values, expected_vwap, rtol=1e-5)


def test_volume_skewness_and_asymmetric_bands():
    """Valida que distribuciones con sesgo asimétrico expandan la banda opuesta."""
    # Simular una distribución con cola larga hacia abajo (masa de volumen arriba en 5000, cola delgada en 4900 => Skewness negativo)
    n_bars = 20
    timestamps = [pd.Timestamp(f"2023-10-16 09:30:00") + pd.Timedelta(minutes=5 * i) for i in range(n_bars)]
    closes = np.linspace(5000.0, 4900.0, n_bars)  # Caída con cola
    highs = closes + 1.0
    lows = closes - 1.0
    opens = closes
    volumes = np.linspace(2000.0, 100.0, n_bars)  # Mayor volumen arriba en 5000, poco volumen abajo

    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes
    }, index=pd.DatetimeIndex(timestamps))

    res = compute_anchored_vwap_optimized_bands(df, anchor_time="09:30", k_band=2.0)

    # Con cola larga hacia abajo, el sesgo debe ser negativo (Skew < 0)
    last_skew = res["Volume_Skewness"].iloc[-1]
    assert last_skew < 0.0

    # Banda inferior debe expandirse más allá de 2.0 sigma debido a skew < 0
    vwap_val = res["VWAP"].iloc[-1]
    sig_val = res["Sigma_VWAP"].iloc[-1]
    lower_band = res["LowerBand"].iloc[-1]

    eff_k_lower = (vwap_val - lower_band) / sig_val
    assert eff_k_lower > 2.0  # Se expandió automáticamente por la asimetría negativa


def test_catastrophic_275_sigma_band():
    """Valida que las bandas catastróficas estén fijadas estrictamente en 2.75 sigma."""
    data = create_synthetic_5m_session(n_bars=25)
    res = compute_anchored_vwap_optimized_bands(data, k_catastrophic=2.75)

    for i in range(5, len(res)):
        vw = res["VWAP"].iloc[i]
        sig = res["Sigma_VWAP"].iloc[i]
        assert np.isclose(res["UpperCatastrophic"].iloc[i], vw + 2.75 * sig)
        assert np.isclose(res["LowerCatastrophic"].iloc[i], vw - 2.75 * sig)


# ==============================================================================
# TIER 3: PENDIENTE NORMALIZADA DEL VWAP Y COMPUERTA AMT
# ==============================================================================

def test_normalized_vwap_slope_calculation():
    """Valida que kappa_norm sea |Delta_VWAP| / (3 * sigma)."""
    data = create_synthetic_5m_session(n_bars=15)
    df = compute_anchored_vwap_optimized_bands(data, slope_lookback_bars=3)

    # Las primeras 3 barras ancladas deben tener pendiente normalizada 0.0
    assert df["VWAP_Slope_Norm"].iloc[0] == 0.0
    assert df["VWAP_Slope_Norm"].iloc[1] == 0.0
    assert df["VWAP_Slope_Norm"].iloc[2] == 0.0

    # En la barra 3: |vwap[3] - vwap[0]| / (3 * max(sigma, 0.50))
    d_vw = abs(df["VWAP"].iloc[3] - df["VWAP"].iloc[0])
    expected_norm = d_vw / (3.0 * max(df["Sigma_VWAP"].iloc[3], 0.50))
    assert np.isclose(df["VWAP_Slope_Norm"].iloc[3], expected_norm, rtol=1e-5)


def test_outside_value_area_acceptance_gate():
    """Valida que si las primeras 6 barras abren y cierran fuera del VA previo, Phi_AMT sea 0."""
    # Día 1: precios centrados en 5000 (VA en ~4995 - ~5005)
    df1 = create_synthetic_5m_session(date_str="2023-10-16", base_price=5000.0, n_bars=78)

    # Día 2: Gap masivo a 5100 (muy por encima de VAH=5005), todas las barras arriba
    df2 = create_synthetic_5m_session(date_str="2023-10-17", base_price=5100.0, n_bars=78)

    df_combined = pd.concat([df1, df2]).sort_index()
    res = compute_anchored_vwap_optimized_bands(df_combined, anchor_time="09:30")

    # En el día 2, las barras deben tener Phi_AMT == 0 (veto por Outside Value Acceptance)
    day2_mask = res.index.date == datetime.date(2023, 10, 17)
    assert (res.loc[day2_mask, "Phi_AMT"] == 0).all()
    # Y el régimen rotacional debe ser 0 durante todo ese día
    assert (res.loc[day2_mask, "Regime_Rotational"] == 0).all()


# ==============================================================================
# TIER 4: CONTROL DE RIESGO PROP TRADING (PropRiskManagerOptimized)
# ==============================================================================

def test_alpha_dll_circuit_breaker():
    """Valida que alcanzar alpha_DLL * DLL (ej. 850 de 1000) detenga la operativa diaria."""
    rm = PropRiskManagerOptimized(
        account_size=50000.0,
        daily_loss_limit=1000.0,
        alpha_dll=0.85
    )

    # Pérdida de $850 (85% de $1000)
    rm.record_trade_result(-850.0)
    assert rm.is_halted_day is True
    assert rm.get_risk_budget() == 0.0

    cnt, _, _, status = rm.calculate_position_size(entry_price=5000.0, sl_price=4995.0)
    assert cnt == 0
    assert status == "DAILY_LOSS_LIMIT_REACHED"


def test_max_intraday_consecutive_losses():
    """Valida que registrar 2 pérdidas consecutivas en el mismo día bloquee nuevas operaciones."""
    rm = PropRiskManagerOptimized(
        account_size=50000.0,
        daily_loss_limit=1000.0,
        max_intraday_consecutive_losses=2
    )

    # Pérdida 1: -$50
    rm.record_trade_result(-50.0)
    assert rm.is_halted_day is False
    assert rm.intraday_consecutive_losses == 1

    # Pérdida 2: -$50 => 2 pérdidas consecutivas en el día
    rm.record_trade_result(-50.0)
    assert rm.is_halted_day is True
    assert rm.intraday_consecutive_losses == 2

    cnt, _, _, status = rm.calculate_position_size(entry_price=5000.0, sl_price=4995.0)
    assert cnt == 0
    assert status == "MAX_INTRADAY_LOSSES_REACHED"


def test_standard_and_micro_contract_caps():
    """Valida que los contratos estándar no excedan max_std_contracts y los micros no excedan max_micro."""
    rm = PropRiskManagerOptimized(
        account_size=50000.0,
        daily_loss_limit=2000.0,
        max_trailing_drawdown=5000.0,
        max_std_contracts=2,
        max_micro_contracts=10,
        symbol="ES",
        std_multiplier=50.0,
        micro_multiplier=5.0
    )
    # Con un stop diminuto de 0.50 puntos, el presupuesto admitiría muchos contratos estándar
    # Pero debe acotarse estrictamente a 2 contratos estándar
    cnt, c_type, _, status = rm.calculate_position_size(entry_price=5000.0, sl_price=4999.50)
    assert c_type == "STANDARD"
    assert cnt <= 2
    assert status == "OK"


# ==============================================================================
# TIER 5: ESTRATEGIA, SALIDAS Y BACKTEST COMPLETO
# ==============================================================================

def test_strategy_signals_generation_structure():
    """Valida que generate_signals retorne una serie coherente con valores en {-1, 0, 1}."""
    data = create_synthetic_multiday_data(n_days=3)
    strat = F3OptimizationISN1Strategy()
    signals = strat.generate_signals(data)

    assert len(signals) == len(data)
    assert isinstance(signals, pd.Series)
    assert set(signals.unique()).issubset({-1.0, 0.0, 1.0})


def test_forced_flat_at_1515_est():
    """Valida que cualquier posición abierta sea liquidada forzosamente a las 15:15 EST."""
    data = create_synthetic_5m_session(date_str="2023-10-16", n_bars=78)
    strat = F3OptimizationISN1Strategy()
    signals = strat.generate_signals(data)

    times = [t.time() for t in data.index]
    flat_time = datetime.time(15, 15)

    for i, t in enumerate(times):
        if t >= flat_time:
            assert signals.iloc[i] == 0.0


def test_end_to_end_backtest_execution():
    """Ejecuta un backtest completo sobre datos sintéticos y valida la estructura del reporte."""
    data = create_synthetic_multiday_data(n_days=4)
    strat = F3OptimizationISN1Strategy()
    results = strat.run_backtest(data)

    assert "metrics" in results
    assert "trades" in results
    assert "equity_curve" in results
    assert "data" in results

    metrics = results["metrics"]
    assert "Capital Inicial ($)" in metrics
    assert "P&L Neto Total ($)" in metrics
    assert "Max Trailing Drawdown ($)" in metrics
    assert "Win Rate (%)" in metrics
    assert "Profit Factor" in metrics

    assert len(results["equity_curve"]) == len(data)


def test_floating_drawdown_account_halt():
    """Valida que update_floating_equity detenga la cuenta inmediatamente si el drawdown flotante alcanza MTDD."""
    rm = PropRiskManagerOptimized(
        account_size=50000.0,
        max_trailing_drawdown=2000.0
    )
    # HWM sube a 52000
    rm.update_floating_equity(52000.0)
    assert rm.peak_equity == 52000.0
    assert rm.is_halted_account is False

    # Drawdown flotante cae a 49950 (drawdown = 2050 > 2000)
    rm.update_floating_equity(49950.0)
    assert rm.is_halted_account is True
    assert rm.get_risk_budget() == 0.0

    cnt, _, _, status = rm.calculate_position_size(5000.0, 4995.0)
    assert cnt == 0
    assert status == "ACCOUNT_TRAILING_DD_BREACH"


def test_volume_kurtosis_filter_in_regime():
    """Valida que una distribución platykúrtica (bimodal/dispersa) inhabilite el régimen rotacional."""
    data = create_synthetic_5m_session(n_bars=30)
    # Con min_kurtosis alto (ej. 10.0), el régimen debe inhibirse
    df_strict = compute_anchored_vwap_optimized_bands(data, min_kurtosis=10.0)
    assert (df_strict["Phi_Kurtosis"] == 0).any()

    # Con min_kurtosis bajo (ej. 1.0), el filtro de curtosis autoriza
    df_permissive = compute_anchored_vwap_optimized_bands(data, min_kurtosis=1.0)
    assert (df_permissive["Phi_Kurtosis"] == 1).all()


def test_min_distance_to_vwap_prevents_low_reward_entries():
    """Valida que señales a menos de 0.8 sigma de VWAP sean inhibidas para proteger ratio beneficio/fricción."""
    data = create_synthetic_5m_session(n_bars=20)
    # Estrategia con min_dist muy exigente (2.0 sigma) no debe entrar
    strat_strict = F3OptimizationISN1Strategy(params={"min_dist_to_vwap_ratio": 3.0})
    sigs_strict = strat_strict.generate_signals(data)
    assert (sigs_strict == 0.0).all()

