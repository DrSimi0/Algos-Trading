"""
scripts/example_futures_backtest.py
Ejemplo práctico de backtesting cuantitativo sobre futuros de ES (S&P 500)
utilizando los datos extraídos de TradingView y el BacktestEngine del proyecto.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
from src.core.data_loader import DataLoader
from src.core.base_strategy import BaseStrategy
from src.backtest.engine import BacktestEngine


class MovingAverageCross(BaseStrategy):
    """
    Estrategia de cruce de medias móviles para futuros:
    - Compra (1.0) cuando SMA corta > SMA larga
    - Cierre / Fuera (0.0) cuando SMA corta <= SMA larga
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data)
        fast_period = self.params.get("fast_period", 20)
        slow_period = self.params.get("slow_period", 50)

        fast_sma = data["Close"].rolling(window=fast_period).mean()
        slow_sma = data["Close"].rolling(window=slow_period).mean()

        signals = pd.Series(0.0, index=data.index)
        signals[fast_sma > slow_sma] = 1.0
        return signals


def main():
    print("=" * 60)
    print(" EJEMPLO DE BACKTESTING SOBRE FUTUROS: E-MINI S&P 500 (ES)")
    print("=" * 60)

    loader = DataLoader(cache_dir="data/raw")
    # Cargar datos de 1 hora de ES (5000 barras)
    df = loader.fetch_tradingview(
        symbol="ES",
        exchange="CME_MINI",
        interval="1h",
        n_bars=5000,
        fut_contract=1,
        use_cache=True
    )

    print(f"Datos cargados: {len(df)} barras horarias desde {df.index.min()} hasta {df.index.max()}")

    strategy = MovingAverageCross(
        name="SMA_Cross_ES_1H",
        params={"fast_period": 20, "slow_period": 50}
    )
    signals = strategy.generate_signals(df)

    engine = BacktestEngine(
        commission_bps=1.0,     # Fricción para futuros
        slippage_bps=1.0,
        periods_per_year=252 * 23  # Velas horarias de futuros CME (~23h por día)
    )

    results = engine.run(data=df, signals=signals, initial_capital=50000.0)

    print("\n--- RESULTADOS DEL BACKTEST ---")
    metrics = results.get("metrics", {})
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f" {k:<25}: {v:,.4f}")
        else:
            print(f" {k:<25}: {v}")

    print("=" * 60)


if __name__ == "__main__":
    main()
