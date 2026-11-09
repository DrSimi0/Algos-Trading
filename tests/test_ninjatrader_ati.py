"""
tests/test_ninjatrader_ati.py
Pruebas unitarias para el conector NinjaTrader ATI y el Live Data Buffer.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.execution.ninjatrader_ati import NinjaTraderATIClient
from src.execution.data_receiver import LiveDataBuffer


def test_ati_order_generation_and_file_write(tmp_path):
    client = NinjaTraderATIClient(account="Sim101", mode="file", incoming_dir=str(tmp_path))
    
    # 1. Enviar orden Market
    ok = client.place_market_order("MNQ", "BUY", 1, order_id="TEST_001")
    assert ok is True

    files = list(tmp_path.glob("order_*.txt"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8").strip()
    assert content.startswith("PLACE;Sim101;MNQ;BUY;1;MARKET;;;DAY;;TEST_001")

    # 2. Cerrar posición
    ok_close = client.close_position("MNQ")
    assert ok_close is True
    files = list(tmp_path.glob("order_*.txt"))
    assert len(files) == 2


def test_data_receiver_buffer():
    buffer = LiveDataBuffer()
    ts = pd.Timestamp("2026-11-06 10:00:00", tz="America/New_York")
    
    buffer.append_bar("ES", ts, 5800.0, 5810.0, 5795.0, 5805.0, 1000)
    buffer.append_bar("NQ", ts, 20500.0, 20550.0, 20480.0, 20520.0, 1500)

    df_x, df_y = buffer.get_aligned_data()
    assert len(df_x) == 1
    assert len(df_y) == 1
    assert float(df_x["close"].iloc[0]) == 5805.0
    assert float(df_y["close"].iloc[0]) == 20520.0
