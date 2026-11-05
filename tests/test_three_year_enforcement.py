"""
tests/test_three_year_enforcement.py

Batería de pruebas unitarias y de integración para la arquitectura de 3 años:
1. Partición estricta de 3 años por calendario (2023 = IS, 2024 = OOS, 2025 = Holdout).
2. Seguridad de bóvedas: Prohibición estricta de acceso a OOS y Holdout sin autorización explícita.
3. Prevención de fuga de información (Data Leakage) con clip_to_partition.
4. Ejecución por defecto forzada en In-Sample en todos los runners de estrategias.
"""

import sys
import subprocess
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.core.data_split import (
    DataSplitManager,
    SecurityGateError,
    check_partition_authorization,
    clip_to_partition,
)


def create_dummy_3y_series():
    """Genera 3 años completos de datos diarios sintéticos (2023-2025)."""
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D", tz="America/New_York")
    df = pd.DataFrame(
        {
            "Open": np.linspace(100, 150, len(dates)),
            "High": np.linspace(101, 152, len(dates)),
            "Low": np.linspace(99, 149, len(dates)),
            "Close": np.linspace(100.5, 151, len(dates)),
            "Volume": 1000,
        },
        index=dates,
    )
    return df


def test_data_split_manager_split_years():
    """Verifica que split_years crea 3 universos disjuntos y no superpuestos por año calendario."""
    df = create_dummy_3y_series()
    mgr = DataSplitManager(df, split_years=[2023, 2024, 2025])

    is_df = mgr.get_in_sample()
    oos_df = mgr.get_out_of_sample(authorized=True, reason="Unit test OOS")
    holdout_df = mgr.get_holdout(authorized=True, reason="Unit test Holdout")

    # Verificar años
    assert (is_df.index.year == 2023).all()
    assert (oos_df.index.year == 2024).all()
    assert (holdout_df.index.year == 2025).all()

    # Verificar ausencia total de solapamiento temporal
    assert is_df.index.max() < oos_df.index.min()
    assert oos_df.index.max() < holdout_df.index.min()

    # Verificar cobertura completa
    assert len(is_df) + len(oos_df) + len(holdout_df) == len(df)


def test_security_gate_unauthorized_oos_raises_security_gate_error():
    """Verifica que acceder a OOS o Holdout sin autorización lanza SecurityGateError."""
    df = create_dummy_3y_series()
    mgr = DataSplitManager(df, split_years=[2023, 2024, 2025])

    # In-Sample siempre debe permitirse
    assert len(mgr.get_in_sample()) > 0

    # OOS no autorizado debe fallar
    with pytest.raises(SecurityGateError):
        mgr.get_out_of_sample(authorized=False)

    # Holdout no autorizado debe fallar
    with pytest.raises(SecurityGateError):
        mgr.get_holdout(authorized=False)


def test_check_partition_authorization_cli():
    """Verifica la función check_partition_authorization que protege el CLI."""
    # In-Sample debe pasar sin flags
    check_partition_authorization("in_sample", approve_oos=False, approve_holdout=False)

    # OOS sin flag debe salir con código 1
    with pytest.raises(SystemExit) as exc_info:
        check_partition_authorization("out_of_sample", approve_oos=False, approve_holdout=False)
    assert exc_info.value.code == 1

    # OOS con flag debe pasar
    check_partition_authorization("out_of_sample", approve_oos=True, approve_holdout=False)

    # Holdout sin flag debe salir con código 1
    with pytest.raises(SystemExit) as exc_info:
        check_partition_authorization("holdout", approve_oos=False, approve_holdout=False)
    assert exc_info.value.code == 1

    # Holdout con flag debe pasar
    check_partition_authorization("holdout", approve_oos=False, approve_holdout=True)


def test_clip_to_partition_anti_leakage():
    """Verifica que clip_to_partition recorta datos de orden superior sin fuga al futuro."""
    daily_dates = pd.date_range("2023-01-01", "2025-12-31", freq="D", tz="America/New_York")
    df_daily = pd.DataFrame({"Close": np.arange(len(daily_dates))}, index=daily_dates)

    # Simular partición activa de 2023
    intra_dates = pd.date_range("2023-01-02 09:30", "2023-12-29 15:55", freq="5min", tz="America/New_York")
    df_intra = pd.DataFrame({"Close": np.arange(len(intra_dates))}, index=intra_dates)

    df_clipped = clip_to_partition(df_daily, df_intra)

    # No debe tener fechas de 2024 ni 2025
    assert df_clipped.index.max().year == 2023
    assert (df_clipped.index.year == 2023).all()
    assert len(df_clipped) < len(df_daily)


@pytest.mark.parametrize("script_name", [
    "run_f1_backtest.py",
    "run_orb_backtest.py",
    "run_f3_backtest.py",
    "run_cointegration_ou_backtest.py",
    "run_meta_labeling_backtest.py",
    "run_f9_backtest.py",
    "run_har_rv_j_backtest.py",
    "run_ofi_backtest.py",
])
def test_runners_reject_unauthorized_oos(script_name):
    """Verifica que todos los runners de estrategias rechacen el acceso a OOS sin autorización."""
    cmd = [sys.executable, script_name, "--mode", "out_of_sample"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 1
    assert "ERROR DE SEGURIDAD" in result.stdout or "ERROR DE SEGURIDAD" in result.stderr
