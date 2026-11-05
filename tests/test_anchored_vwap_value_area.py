"""
tests/test_anchored_vwap_value_area.py

Suite completa de pruebas unitarias y de integración para la Estrategia F3:
Anchored VWAP & Rotational Value Area Reversion.
"""

import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.strategies.mean_reversion.anchored_vwap_value_area import (
    AnchoredVWAPValueAreaStrategy,
    PropRiskManager,
    compute_anchored_vwap_bands,
    calculate_adx,
)
from src.core.futures_specs import get_spec


# ==============================================================================
# FIXTURES Y GENERADORES DE DATOS SINTÉTICOS DE PRUEBA
# ==============================================================================

def create_synthetic_5m_session(
    date_str: str = "2026-10-14",
    base_price: float = 5000.0,
    n_bars: int = 78,  # 09:30 a 16:00 son 78 barras de 5m
    volatility: float = 2.0,
    volume_per_bar: float = 1000.0
) -> pd.DataFrame:
    """Genera una sesión intradía sintética de 5 minutos con timestamps desde 09:30 a 16:00."""
    start_dt = pd.Timestamp(f"{date_str} 09:30:00")
    timestamps = [start_dt + pd.Timedelta(minutes=5 * i) for i in range(n_bars)]
    
    np.random.seed(42)
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


# ==============================================================================
# TIER 1: VERIFICACIÓN MATEMÁTICA Y DE INDICADORES
# ==============================================================================

def test_typical_price_and_vwap_exactness():
    """Valida la fórmula de precio típico y VWAP ponderado por volumen."""
    data = create_synthetic_5m_session(n_bars=10, volume_per_bar=100.0)
    df = compute_anchored_vwap_bands(data, anchor_time="09:30")

    # Pi_typ = (H + L + C) / 3
    expected_ptyp = (data["High"] + data["Low"] + data["Close"]) / 3.0
    np.testing.assert_allclose(df["P_typ"].values, expected_ptyp.values, rtol=1e-6)

    # Para volumen constante V, VWAP es la media acumulada de P_typ
    expected_vwap = expected_ptyp.expanding().mean().values
    np.testing.assert_allclose(df["VWAP"].values, expected_vwap, rtol=1e-5)


def test_volume_weighted_standard_deviation_and_bands():
    """Valida el cálculo de la varianza ponderada, sigma y las bandas +-2sigma y +-3sigma."""
    data = create_synthetic_5m_session(n_bars=20, volume_per_bar=200.0)
    df = compute_anchored_vwap_bands(data, anchor_time="09:30", k_band=2.0, k_catastrophic=3.0)

    for i in range(len(df)):
        vwap_val = df["VWAP"].iloc[i]
        sig_val = df["Sigma_VWAP"].iloc[i]
        assert sig_val >= 0.0

        # Upper = VWAP + 2 * sigma
        assert np.isclose(df["UpperBand"].iloc[i], vwap_val + 2.0 * sig_val)
        # Lower = VWAP - 2 * sigma
        assert np.isclose(df["LowerBand"].iloc[i], vwap_val - 2.0 * sig_val)
        # Upper Catastrophic = VWAP + 3 * sigma
        assert np.isclose(df["UpperCatastrophic"].iloc[i], vwap_val + 3.0 * sig_val)
        # Lower Catastrophic = VWAP - 3 * sigma
        assert np.isclose(df["LowerCatastrophic"].iloc[i], vwap_val - 3.0 * sig_val)


def test_vwap_slope_calculation():
    """Valida que kappa_slope sea (VWAP_t - VWAP_t-3) / (3 * Delta_t)."""
    data = create_synthetic_5m_session(n_bars=15)
    df = compute_anchored_vwap_bands(data, anchor_time="09:30", bar_minutes=5.0)

    # Las primeras 3 barras ancladas deben tener pendiente 0.0
    assert df["VWAP_Slope"].iloc[0] == 0.0
    assert df["VWAP_Slope"].iloc[1] == 0.0
    assert df["VWAP_Slope"].iloc[2] == 0.0

    # Desde la barra 3 (índice 3), slope = (vwap[3] - vwap[0]) / 15.0
    expected_slope_3 = (df["VWAP"].iloc[3] - df["VWAP"].iloc[0]) / 15.0
    assert np.isclose(df["VWAP_Slope"].iloc[3], expected_slope_3, rtol=1e-5)


def test_adx_indicator_ranges():
    """Valida que el ADX esté acotado en [0, 100] y no sea nulo."""
    data = create_synthetic_5m_session(n_bars=50)
    adx = calculate_adx(data, period=14)

    assert len(adx) == len(data)
    assert not adx.isna().any()
    assert (adx >= 0.0).all()
    assert (adx <= 100.0).all()


def test_regime_rotational_gate():
    """Valida que el filtro de régimen rotacional se active sólo cuando ADX < 25 y pendiente <= delta."""
    data = create_synthetic_5m_session(n_bars=40)
    df = compute_anchored_vwap_bands(data, anchor_time="09:30", delta_max_slope=0.15)

    for i in range(len(df)):
        adx_val = df["ADX"].iloc[i]
        slope_val = abs(df["VWAP_Slope"].iloc[i])
        regime = df["Regime_Rotational"].iloc[i]

        if adx_val < 25.0 and slope_val <= 0.15:
            assert regime == 1
        else:
            assert regime == 0


# ==============================================================================
# TIER 2: CONTROL DE RIESGO PROP TRADING (PropRiskManager)
# ==============================================================================

def test_phi_k_loss_streak_decay():
    """Valida la fórmula de decaimiento por rachas de pérdidas: phi(K) = 0.30 * 2^(-min(K, 3))."""
    rm = PropRiskManager(account_size=50000.0, daily_loss_limit=1000.0, max_trailing_drawdown=2000.0)

    # K = 0 => 0.30
    assert np.isclose(rm.phi_k_loss(), 0.30)

    # K = 1 => 0.15
    rm.consecutive_losses = 1
    assert np.isclose(rm.phi_k_loss(), 0.15)

    # K = 2 => 0.075
    rm.consecutive_losses = 2
    assert np.isclose(rm.phi_k_loss(), 0.075)

    # K = 3 => 0.0375
    rm.consecutive_losses = 3
    assert np.isclose(rm.phi_k_loss(), 0.0375)

    # K >= 4 se satura en 3 => 0.0375
    rm.consecutive_losses = 5
    assert np.isclose(rm.phi_k_loss(), 0.0375)


def test_risk_budget_formula():
    """Valida que B_F3 sea el mínimo entre el margen diario y el margen de trailing drawdown."""
    rm = PropRiskManager(
        account_size=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_drawdown=2000.0,
        psi=0.20
    )

    # Inicialmente: L_day = 0, DD = 0
    # rem_daily = 1000 => phi(0)*1000 = 300.0
    # rem_trailing = 2000 => psi*2000 = 400.0
    # B = min(300, 400) = 300.0
    assert np.isclose(rm.get_risk_budget(), 300.0)

    # Si hay drawdown de $1000 => rem_trailing = 1000 => psi*1000 = 200.0
    # B = min(300, 200) = 200.0
    rm.current_equity = 49000.0
    assert np.isclose(rm.get_risk_budget(), 200.0)


def test_standard_to_micro_contract_switching():
    """Valida la conmutación automática de contratos estándar a micro según presupuesto."""
    rm = PropRiskManager(
        account_size=50000.0,
        daily_loss_limit=1000.0,
        max_trailing_drawdown=2000.0,
        symbol="ES",
        std_multiplier=50.0,
        micro_multiplier=5.0,
        std_commission_rt=4.50,
        micro_commission_rt=1.24
    )
    # Presupuesto inicial = 300.0
    # Caso 1: stop de 3 puntos en ES => Riesgo Std = 3*50 + 29.50 = 179.50.
    # 300 / 179.50 = 1 contrato estándar.
    cnt, c_type, r_cnt, status = rm.calculate_position_size(entry_price=5000.0, sl_price=4997.0)
    assert c_type == "STANDARD"
    assert cnt == 1
    assert status == "OK"

    # Caso 2: stop de 8 puntos en ES => Riesgo Std = 8*50 + 29.50 = 429.50 > 300.0 => 0 estándar.
    # Conmuta a Micro (MES): Riesgo Micro = 8*5 + 2*0.25*5 + 1.24 = 40 + 2.5 + 1.24 = 43.74.
    # Micro contracts = floor(300 / 43.74) = 6 micro-contratos.
    cnt, c_type, r_cnt, status = rm.calculate_position_size(entry_price=5000.0, sl_price=4992.0)
    assert c_type == "MICRO"
    assert cnt == 6
    assert status == "OK"


def test_volatility_defense_circuit_breaker():
    """Valida que si ni siquiera 1 micro-contrato entra en presupuesto, se active HALT_VOLATILITY_DEFENSE."""
    rm = PropRiskManager(
        account_size=50000.0,
        daily_loss_limit=100.0,  # Muy pequeño
        max_trailing_drawdown=200.0,
        consecutive_losses=3,    # phi = 0.0375 => budget = 0.0375 * 100 = 3.75
        symbol="ES"
    )
    # Stop de 5 puntos en MES: riesgo micro = 5*5 + 3.74 = 28.74 > 3.75
    cnt, c_type, r_cnt, status = rm.calculate_position_size(entry_price=5000.0, sl_price=4995.0)
    assert cnt == 0
    assert c_type == "NONE"
    assert status == "HALT_VOLATILITY_DEFENSE"


def test_daily_loss_limit_and_trailing_dd_lockout():
    """Valida que alcanzar el DLL o el MTDD bloquee toda nueva asignación de riesgo."""
    rm = PropRiskManager(account_size=50000.0, daily_loss_limit=1000.0, max_trailing_drawdown=2000.0)

    # Simular pérdida de $1000 en el día
    rm.record_trade_result(-1000.0)
    assert rm.is_halted_day is True
    assert rm.get_risk_budget() == 0.0

    cnt, _, _, status = rm.calculate_position_size(entry_price=5000.0, sl_price=4998.0)
    assert cnt == 0
    assert status == "DAILY_LOSS_LIMIT_REACHED"

    # Simular reinicio de nuevo día: el día se resetea pero el trailing DD persiste
    rm.reset_day_if_new(datetime.date(2026, 10, 15))
    assert rm.is_halted_day is False
    # Equity ahora es 49000, DD = 1000 < 2000 => puede operar
    assert rm.get_risk_budget() > 0.0

    # Simular pérdida adicional de $1100 => Total DD = 2100 > MTDD 2000
    rm.record_trade_result(-1100.0)
    assert rm.is_halted_account is True
    assert rm.get_risk_budget() == 0.0

    cnt, _, _, status = rm.calculate_position_size(entry_price=5000.0, sl_price=4998.0)
    assert cnt == 0
    assert status == "ACCOUNT_TRAILING_DD_BREACH"


def test_daily_loss_limit_net_pnl_recovery():
    """Valida que trades ganadores compensen pérdidas diarias y eviten bloqueos falsos."""
    rm = PropRiskManager(account_size=50000.0, daily_loss_limit=1000.0, max_trailing_drawdown=2000.0)

    # Pérdida de $600
    rm.record_trade_result(-600.0)
    assert rm.day_realized_loss == 600.0
    assert rm.is_halted_day is False

    # Ganancia posterior de $800 => día en neto +$200
    rm.record_trade_result(800.0)
    assert rm.day_realized_loss == 0.0
    assert rm.is_halted_day is False
    # Presupuesto diario restante debe ser el 100% de DLL ($1000)
    assert rm.get_risk_budget() > 0.0


# ==============================================================================
# TIER 3: LÓGICA DE ENTRADA, SALIDA Y TIEMPO DE LA ESTRATEGIA
# ==============================================================================

def test_time_window_restrictions():
    """Valida que no se generen entradas en W_init (< 10:30) ni tras W_inhibit (>= 15:00)."""
    # Crear un día con precios planos
    data = create_synthetic_5m_session(n_bars=78)
    strat = AnchoredVWAPValueAreaStrategy()
    signals = strat.generate_signals(data)

    # Verificar que entre 09:30 y 10:25 la posición sea 0.0
    for idx, t in enumerate(signals.index):
        if t.time() < datetime.time(10, 30):
            assert signals.iloc[idx] == 0.0
        elif t.time() >= datetime.time(15, 15):
            assert signals.iloc[idx] == 0.0


def test_two_stage_reversion_trigger_and_exits():
    """
    Crea un escenario controlado donde:
    1. En t-1 el precio perfora la banda inferior.
    2. En t el precio cierra dentro de la banda => Se dispara señal Long.
    3. En t+k el precio toca el VWAP => Se ejecuta Dynamic Take Profit.
    """
    timestamps = [pd.Timestamp("2026-10-14 09:30:00") + pd.Timedelta(minutes=5 * i) for i in range(40)]
    pattern = [3.0, -3.0, 2.0, -2.0, 3.0, -3.0] * 7
    ptyp = [5000.0 + pattern[i] for i in range(40)]
    highs = [p + 2.0 for p in ptyp]
    lows = [p - 2.0 for p in ptyp]
    closes = [p for p in ptyp]
    opens = [p for p in ptyp]
    volumes = [1000.0] * 40

    # Barra 13 (10:35): perfora la banda inferior (~4994.5)
    lows[13] = 4990.0
    closes[13] = 4992.0
    opens[13] = 4996.0
    highs[13] = 4996.0

    # Barra 14 (10:40): rebote y cierre por encima de la banda inferior
    opens[14] = 4992.0
    lows[14] = 4992.0
    closes[14] = 4997.0
    highs[14] = 4997.0

    # Barra 15 (10:45): alcanza y supera el VWAP (~5000)
    opens[15] = 4997.0
    lows[15] = 4997.0
    highs[15] = 5003.0
    closes[15] = 5001.0

    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(timestamps)
    )

    strat = AnchoredVWAPValueAreaStrategy(params={"symbol": "ES"})
    results = strat.run_backtest(df)

    trades = results["trades"]
    assert len(trades) >= 1
    first_trade = trades.iloc[0]
    assert first_trade["direction"] == "LONG"
    assert first_trade["exit_reason"] == "TAKE_PROFIT_VWAP"


def test_circuit_breaker_3sigma_exit():
    """Valida la salida de emergencia por disyuntor catastrófico si el precio rompe +-3 sigma."""
    start_dt = pd.Timestamp("2026-10-14 09:30:00")
    timestamps = [start_dt + pd.Timedelta(minutes=5 * i) for i in range(40)]

    opens = [5000.0] * 40
    highs = [5002.0] * 40
    lows = [4998.0] * 40
    closes = [5000.0] * 40
    volumes = [1000.0] * 40

    # Forzar entrada Long en índice 13-14 con mecha profunda
    lows[13] = 4950.0
    closes[13] = 4982.0
    lows[14] = 4985.0
    closes[14] = 4995.0

    # En índice 15 el precio cae a 4955 (rompe 3-sigma ~4968, antes de tocar sl_price 4949)
    lows[15] = 4955.0
    closes[15] = 4956.0

    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(timestamps)
    )

    strat = AnchoredVWAPValueAreaStrategy(params={
        "symbol": "ES",
        "adx_threshold": 50.0,
        "daily_loss_limit": 5000.0,
        "max_trailing_drawdown": 10000.0
    })
    results = strat.run_backtest(df)

    trades = results["trades"]
    assert len(trades) >= 1
    # Debe haber salido estrictamente por Disyuntor 3-sigma
    assert trades.iloc[0]["exit_reason"] == "3_SIGMA_CIRCUIT_BREAKER"


def test_stop_loss_technical_exit():
    """Valida la salida por Stop Loss técnico cuando la distancia del stop es más ajustada que 3-sigma."""
    start_dt = pd.Timestamp("2026-10-14 09:30:00")
    timestamps = [start_dt + pd.Timedelta(minutes=5 * i) for i in range(40)]

    pattern = [10.0, -10.0] * 20
    opens = [5000.0 + p for p in pattern]
    highs = [p + 2.0 for p in opens]
    lows = [p - 2.0 for p in opens]
    closes = [p for p in opens]
    volumes = [1000.0] * 40

    # Entrada Long en 13-14 con dispersión previa suficiente
    lows[13] = 4975.0
    closes[13] = 4980.0
    lows[14] = 4980.0
    closes[14] = 4990.0

    # En 15 cae a 4972 (toca SL en 4974.0 antes de alcanzar 3-sigma en ~4968)
    lows[15] = 4972.0
    closes[15] = 4973.0

    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(timestamps)
    )

    strat = AnchoredVWAPValueAreaStrategy(params={"symbol": "ES", "adx_threshold": 50.0})
    results = strat.run_backtest(df)

    trades = results["trades"]
    assert len(trades) >= 1
    assert trades.iloc[0]["exit_reason"] == "STOP_LOSS"


def test_no_entry_when_overshooting_vwap():
    """Valida que una reversión Long NO ingrese si la barra de confirmación ya cerró por encima del VWAP."""
    start_dt = pd.Timestamp("2026-10-14 09:30:00")
    timestamps = [start_dt + pd.Timedelta(minutes=5 * i) for i in range(40)]

    opens = [5000.0] * 40
    highs = [5002.0] * 40
    lows = [4998.0] * 40
    closes = [5000.0] * 40
    volumes = [1000.0] * 40

    # En 13 perfora la banda inferior (~4994)
    lows[13] = 4990.0
    closes[13] = 4992.0

    # En 14 rebota con tanta fuerza que cierra en 5005 (por encima de VWAP ~5000)
    lows[14] = 4992.0
    closes[14] = 5005.0
    highs[14] = 5006.0

    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(timestamps)
    )

    strat = AnchoredVWAPValueAreaStrategy(params={"symbol": "ES", "adx_threshold": 50.0})
    signals = strat.generate_signals(df)

    # No debe disparar señal Long en índice 14 porque ya superó el objetivo VWAP
    assert signals.iloc[14] == 0.0


def test_realistic_gap_fill_on_stop():
    """Valida que si una barra abre por debajo del Stop Loss, la ejecución tome el precio de Open (deslizamiento de gap)."""
    start_dt = pd.Timestamp("2026-10-14 09:30:00")
    timestamps = [start_dt + pd.Timedelta(minutes=5 * i) for i in range(40)]

    pattern = [10.0, -10.0] * 20
    opens = [5000.0 + p for p in pattern]
    highs = [p + 2.0 for p in opens]
    lows = [p - 2.0 for p in opens]
    closes = [p for p in opens]
    volumes = [1000.0] * 40

    lows[13] = 4975.0
    closes[13] = 4980.0
    lows[14] = 4980.0
    closes[14] = 4990.0  # SL queda en 4974.0

    # En índice 15 el mercado abre con un gap a 4970.0 (por debajo de SL 4974.0 y arriba de 3-sigma 4968.0)
    opens[15] = 4970.0
    lows[15] = 4969.0
    closes[15] = 4969.5

    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(timestamps)
    )

    strat = AnchoredVWAPValueAreaStrategy(params={"symbol": "ES", "adx_threshold": 50.0})
    results = strat.run_backtest(df)

    trades = results["trades"]
    assert len(trades) >= 1
    # La salida no debe ser el precio artificial de SL (4974.0), sino el precio real del Gap en Open (4970.0)
    assert trades.iloc[0]["exit_price"] == 4970.0
    assert trades.iloc[0]["exit_reason"] == "STOP_LOSS"


# ==============================================================================
# TIER 4: PRUEBAS E2E CON DATOS HISTÓRICOS REALES DE MERCADO
# ==============================================================================

def test_e2e_cme_es_5m_dataset():
    """Ejecuta el backtest completo sobre el dataset real de ES 5m existente en el repositorio."""
    data_path = Path("data/raw/CME_MINI_ES_5m_bars5000.csv")
    if not data_path.exists():
        pytest.skip("Dataset CME_MINI_ES_5m_bars5000.csv no encontrado.")

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    strat = AnchoredVWAPValueAreaStrategy(params={"symbol": "ES"})
    results = strat.run_backtest(df)

    metrics = results["metrics"]
    trades = results["trades"]
    equity = results["equity_curve"]

    assert isinstance(metrics, dict)
    assert "Capital Final ($)" in metrics
    assert "P&L Neto Total ($)" in metrics
    assert "Max Trailing Drawdown ($)" in metrics
    assert len(equity) == len(df)
    assert metrics["Capital Final ($)"] > 0.0


def test_e2e_cme_nq_5m_dataset():
    """Ejecuta el backtest completo sobre el dataset real de NQ 5m con k_band=2.5 y delta_slope=0.45."""
    data_path = Path("data/raw/CME_MINI_NQ_5m_bars5000.csv")
    if not data_path.exists():
        pytest.skip("Dataset CME_MINI_NQ_5m_bars5000.csv no encontrado.")

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    strat = AnchoredVWAPValueAreaStrategy(params={"symbol": "NQ"})
    assert strat.params["k_band"] == 2.5
    assert strat.params["delta_max_slope"] == 0.45

    results = strat.run_backtest(df)
    metrics = results["metrics"]
    assert metrics["Capital Final ($)"] > 0.0


def test_signal_mutual_exclusion_and_consistency():
    """Valida la garantía analítica de exclusión mutua S_Long * S_Short == 0 ante barras de volatilidad anómala."""
    start_dt = pd.Timestamp("2026-10-14 09:30:00")
    timestamps = [start_dt + pd.Timedelta(minutes=5 * i) for i in range(40)]

    opens = [5000.0] * 40
    highs = [5002.0] * 40
    lows = [4998.0] * 40
    closes = [5000.0] * 40
    volumes = [1000.0] * 40

    # Barra 13 es una barra descomunal que penetra simultáneamente banda superior e inferior
    highs[13] = 5050.0
    lows[13] = 4950.0
    closes[13] = 5000.0

    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(timestamps)
    )

    strat = AnchoredVWAPValueAreaStrategy(params={"symbol": "ES", "adx_threshold": 50.0})
    signals = strat.generate_signals(df)

    # En la barra 14 no debe generarse posición contraria o ambigua
    assert signals.iloc[14] == 0.0
