"""
tests/test_vault_security.py
Suite de Pruebas Unitarias y de Integración para la Bóveda Criptográfica y Seguridad Antifugas:
1. Verificación de barrera física: pd.read_parquet falla sobre archivos .enc de la bóveda.
2. Integridad bit a bit: Cifrado y descifrado en RAM idéntico al original.
3. Soporte de contenedores multi-activo (diccionarios de DataFrames en RAM).
4. Tripwire / Honeypot: Clave incorrecta dispara VaultTripwireTriggered y alerta inmutable.
5. Cero huella en disco: Prohibición total de archivos temporales en disco.
6. Truncamiento estricto de data/raw/: Contiene única y exclusivamente el año 2023 (20,280 barras).
7. Purga de reportes: reports/ no contiene archivos residuales de OOS o Holdout.
8. Validación del motor Walk-Forward Matrix (WFM 2D) con WFE, C_temp y MSI.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.core.vault_security import (
    encrypt_dataset_to_vault,
    decrypt_vault_to_memory,
    generate_master_vault_key,
    is_vault_file,
    plant_canary_trap,
    record_tripwire_event,
    VaultTripwireTriggered,
    VaultSecurityError,
    DEFAULT_TRIPWIRE_LOG,
)
from src.core.multi_year_wfm import MultiYearWalkForwardMatrix


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def dummy_timeseries():
    """Genera una serie de tiempo sintética con DatetimeIndex."""
    dates = pd.date_range("2024-01-01 09:30", periods=100, freq="5min", tz="America/New_York")
    df = pd.DataFrame({
        "Open": np.linspace(5000, 5020, 100),
        "High": np.linspace(5005, 5025, 100),
        "Low": np.linspace(4995, 5015, 100),
        "Close": np.linspace(5002, 5022, 100),
        "Volume": np.random.randint(100, 1000, 100),
    }, index=dates)
    return df


@pytest.fixture
def master_key():
    return generate_master_vault_key()


# ==============================================================================
# PRUEBAS DE SEGURIDAD Y BÓVEDA
# ==============================================================================

def test_vault_unreadable_by_pandas(tmp_path, dummy_timeseries, master_key):
    """Verifica que pd.read_parquet falle de forma irrecuperable sobre un archivo cifrado."""
    enc_path = tmp_path / "protected.enc"
    encrypt_dataset_to_vault(dummy_timeseries, enc_path, master_key)

    assert enc_path.exists()
    assert is_vault_file(enc_path) is True

    # pd.read_parquet no debe poder interpretarlo
    with pytest.raises(Exception) as exc_info:
        pd.read_parquet(enc_path)

    # El error debe ser de formato parquet inválido (no magic PAR1)
    err_msg = str(exc_info.value).lower()
    assert any(x in err_msg for x in ["magic", "not a valid", "corrupt", "bytes", "format", "parquet", "failed"])


def test_vault_encryption_decryption_bit_identical(tmp_path, dummy_timeseries, master_key):
    """Verifica que el ciclo cifrado -> descifrado en RAM conserve la integridad bit a bit."""
    enc_path = tmp_path / "sample.enc"
    encrypt_dataset_to_vault(dummy_timeseries, enc_path, master_key)

    # Desencriptar en RAM
    decrypted_df = decrypt_vault_to_memory(enc_path, master_key)

    assert isinstance(decrypted_df, pd.DataFrame)
    assert len(decrypted_df) == len(dummy_timeseries)
    assert list(decrypted_df.columns) == list(dummy_timeseries.columns)
    assert dummy_timeseries.equals(decrypted_df)


def test_vault_dictionary_containers(tmp_path, dummy_timeseries, master_key):
    """Verifica que múltiples DataFrames se empaqueten y desempaqueten en memoria sin colisiones."""
    df_es = dummy_timeseries.copy()
    df_nq = dummy_timeseries.copy() * 3.5

    data_container = {"ES": df_es, "NQ": df_nq}
    enc_path = tmp_path / "multi_asset.enc"
    encrypt_dataset_to_vault(data_container, enc_path, master_key)

    # 1. Recuperar diccionario completo
    restored_dict = decrypt_vault_to_memory(enc_path, master_key)
    assert isinstance(restored_dict, dict)
    assert set(restored_dict.keys()) == {"ES", "NQ"}
    assert restored_dict["ES"].equals(df_es)
    assert restored_dict["NQ"].equals(df_nq)

    # 2. Recuperar un símbolo específico
    restored_es = decrypt_vault_to_memory(enc_path, master_key, symbol="ES")
    assert isinstance(restored_es, pd.DataFrame)
    assert restored_es.equals(df_es)


def test_vault_wrong_key_triggers_tripwire(tmp_path, dummy_timeseries, master_key):
    """Verifica que una clave errónea dispare VaultTripwireTriggered y escriba en el log."""
    enc_path = tmp_path / "vault_trip.enc"
    encrypt_dataset_to_vault(dummy_timeseries, enc_path, master_key)

    bad_key = generate_master_vault_key()
    assert bad_key != master_key

    with pytest.raises(VaultTripwireTriggered) as exc_info:
        decrypt_vault_to_memory(enc_path, bad_key)

    assert "TRIPWIRE ACTIVADO" in str(exc_info.value)
    assert DEFAULT_TRIPWIRE_LOG.exists()


def test_no_temporary_files_created_on_disk(tmp_path, dummy_timeseries, master_key):
    """Verifica que la desencriptación ocurra estrictamente en RAM sin escribir temporales."""
    enc_path = tmp_path / "secure_ram.enc"
    encrypt_dataset_to_vault(dummy_timeseries, enc_path, master_key)

    files_before = set(tmp_path.iterdir())
    dec_df = decrypt_vault_to_memory(enc_path, master_key)
    files_after = set(tmp_path.iterdir())

    # No deben existir nuevos archivos temporales en el directorio
    assert files_before == files_after
    assert len(dec_df) > 0


def test_raw_data_strictly_truncated_to_2023():
    """Verifica que data/raw contenga EXCLUSIVAMENTE el año 2023 (20,280 barras)."""
    p_es = Path("data/raw/CME_MINI_ES_5m_3y.parquet")
    p_nq = Path("data/raw/CME_MINI_NQ_5m_3y.parquet")

    assert p_es.exists(), "Falta CME_MINI_ES_5m_3y.parquet en data/raw"
    assert p_nq.exists(), "Falta CME_MINI_NQ_5m_3y.parquet en data/raw"

    df_es = pd.read_parquet(p_es)
    df_nq = pd.read_parquet(p_nq)

    # Exactamente 20,280 barras
    assert len(df_es) == 20280, f"ES debe tener exactamente 20,280 barras, tiene {len(df_es)}"
    assert len(df_nq) == 20280, f"NQ debe tener exactamente 20,280 barras, tiene {len(df_nq)}"

    # Exclusivamente el año 2023
    assert set(df_es.index.year.unique()) == {2023}
    assert set(df_nq.index.year.unique()) == {2023}


def test_reports_purged_from_oos_and_holdout():
    """Verifica que en reports/ no quede ningún archivo con nombres OOS o Holdout."""
    rep_dir = Path("reports")
    assert rep_dir.exists()

    leak_files = [
        f.name for f in rep_dir.glob("*")
        if f.is_file() and any(x in f.name.lower() for x in ["oos", "holdout"])
    ]
    assert len(leak_files) == 0, f"Fuga de reportes detectada en reports/: {leak_files}"

    # reports/vault debe contener los archivos purgados
    vault_rep = rep_dir / "vault"
    assert vault_rep.exists()
    assert len(list(vault_rep.glob("*"))) >= 10


def test_canary_trap_present():
    """Verifica que el canary trap de la bóveda esté plantado."""
    canary = plant_canary_trap()
    assert canary.exists()
    assert is_vault_file(canary) is True


# ==============================================================================
# PRUEBAS DEL MOTOR WALK-FORWARD MATRIX (WFM 2D)
# ==============================================================================

def test_multi_year_wfm_metrics_calculation():
    """Verifica el cálculo de WFE, Consistencia Temporal y Matrix Stability Index."""
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="2D", tz="UTC")
    pnl = [150.0 if i % 4 != 0 else -80.0 for i in range(len(dates))]
    trades = pd.DataFrame({"exit_time": dates, "net_pnl_usd": pnl})

    wfm = MultiYearWalkForwardMatrix(
        train_windows=[60, 90],
        test_windows=[20, 30],
        mtdd_limit=2000.0,
        dll_limit=1000.0,
        initial_capital=50000.0,
    )
    res = wfm.evaluate_trades(trades)

    assert "matrix_passed" in res
    assert "mean_wfe_return" in res
    assert "matrix_stability_index_msi" in res
    assert "mean_c_temp" in res
    assert res["mean_wfe_return"] > 0.0
    assert 0.0 <= res["matrix_stability_index_msi"] <= 1.0
    assert res["mean_c_temp"] >= 70.0
    assert res["total_dll_violations"] == 0
    assert res["total_mtdd_violations"] == 0


def test_vault_empty_or_none_or_malformed_key_triggers_tripwire(tmp_path, dummy_timeseries, master_key):
    """Verifica que claves vacías, None o tipos no válidos disparen VaultTripwireTriggered inmediatamente."""
    enc_path = tmp_path / "key_edge_cases.enc"
    encrypt_dataset_to_vault(dummy_timeseries, enc_path, master_key)

    # 1. Clave None
    with pytest.raises(VaultTripwireTriggered):
        decrypt_vault_to_memory(enc_path, None)

    # 2. Clave vacía o solo espacios
    with pytest.raises(VaultTripwireTriggered):
        decrypt_vault_to_memory(enc_path, "")

    with pytest.raises(VaultTripwireTriggered):
        decrypt_vault_to_memory(enc_path, "   ")

    # 3. Tipo inválido (e.g. número entero)
    with pytest.raises(VaultTripwireTriggered):
        decrypt_vault_to_memory(enc_path, 12345)


def test_canary_trap_access_triggers_specific_honeypot_alert(tmp_path, master_key):
    """Verifica que acceder directamente al Canary Trap active el honeypot tripwire."""
    canary = plant_canary_trap(vault_dir=tmp_path)
    assert canary.exists()

    with pytest.raises(VaultTripwireTriggered) as exc_info:
        decrypt_vault_to_memory(canary, master_key)

    assert "HONEYPOT CANARY ACTIVADO" in str(exc_info.value)


def test_wfm_drawdown_measured_from_initial_capital():
    """
    Verifica que el drawdown en WFM se mida con respecto al capital inicial
    y no empiece en 0 si la primera operación de una ventana de prueba es una pérdida.
    """
    dates = pd.date_range("2023-01-01", "2023-06-30", freq="2D", tz="UTC")
    # Primera operación del periodo de prueba OOS (marzo) tiene pérdida de -$2,500
    pnl = [-2500.0 if d == pd.Timestamp("2023-03-04", tz="UTC") else 100.0 for d in dates]
    trades = pd.DataFrame({"exit_time": dates, "net_pnl_usd": pnl})

    wfm = MultiYearWalkForwardMatrix(
        train_windows=[60],
        test_windows=[20],
        mtdd_limit=2000.0,
        dll_limit=1000.0,
        initial_capital=50000.0,
    )
    res = wfm.evaluate_trades(trades)

    # El peor drawdown debe ser de al menos 2,500 y detectar la violación de MTDD
    assert res["worst_matrix_drawdown_usd"] >= 2500.0
    assert res["total_mtdd_violations"] > 0
    assert res["matrix_passed"] is False


def test_wfm_anchored_mode_evaluation():
    """Verifica la evaluación en modo anclado (anchored=True) de Walk-Forward Matrix."""
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="2D", tz="UTC")
    pnl = [150.0 if i % 3 != 0 else -60.0 for i in range(len(dates))]
    trades = pd.DataFrame({"exit_time": dates, "net_pnl_usd": pnl})

    wfm = MultiYearWalkForwardMatrix(
        train_windows=[60, 90],
        test_windows=[30],
        mtdd_limit=2000.0,
        dll_limit=1000.0,
        initial_capital=50000.0,
        anchored=True,
    )
    res = wfm.evaluate_trades(trades)

    assert "matrix_passed" in res
    assert res["mean_wfe_return"] > 0.0
    assert res["mean_c_temp"] >= 70.0


def test_datasplit_vaulted_oos_records_tripwire(dummy_timeseries):
    """Verifica que un intento no autorizado a datos vaciados/en bóveda registre evento en tripwire."""
    from src.core.data_split import DataSplitManager, SecurityGateError
    from src.core.vault_security import DEFAULT_TRIPWIRE_LOG

    # Dataframe solo 2023
    dates_2023 = pd.date_range("2023-01-01", "2023-12-31", freq="1D", tz="America/New_York")
    df_2023 = pd.DataFrame({"Close": np.linspace(100, 200, len(dates_2023))}, index=dates_2023)

    mgr = DataSplitManager(df_2023, split_years=[2023, 2024, 2025])

    # OOS no autorizado
    with pytest.raises(SecurityGateError):
        mgr.get_out_of_sample(authorized=False)

    assert DEFAULT_TRIPWIRE_LOG.exists()

