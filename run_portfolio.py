"""
run_portfolio.py
Ejecutable principal del Portfolio Manager.
Demuestra la carga de datos, ejecución de estrategias individuales,
auditoría de riesgo y combinación multi-estrategia en el portafolio.
"""

import sys
import pandas as pd
from src.core.data_loader import DataLoader
from src.strategies.momentum.sma_crossover import SMACrossoverStrategy
from src.strategies.mean_reversion.rsi_reversion import RSIReversionStrategy
from src.backtest.engine import BacktestEngine
from src.portfolio.allocator import PortfolioAllocator
from src.portfolio.risk_manager import RiskManager


def main():
    print("=" * 70)
    print("    PORTFOLIO MANAGER - SISTEMA MULTI-ESTRATEGIA CUANTITATIVO")
    print("=" * 70)

    # 1. Cargar Datos Históricos
    symbol = "SPY"
    start_date = "2021-01-01"
    end_date = "2023-12-31"

    print(f"\n[1] Cargando datos de mercado para {symbol} ({start_date} a {end_date})...")
    loader = DataLoader()
    try:
        data = loader.fetch_data(symbol=symbol, start_date=start_date, end_date=end_date)
        print(f"    OK: {len(data)} velas diarias cargadas.")
    except Exception as e:
        print(f"    Error descargando datos online: {e}")
        print("    Generando datos sintéticos para demostración...")
        dates = pd.date_range(start_date, end_date, freq="B")
        import numpy as np
        np.random.seed(42)
        prices = 380.0 + np.cumsum(np.random.randn(len(dates)) * 2.0)
        data = pd.DataFrame({
            "Open": prices,
            "High": prices + 1.5,
            "Low": prices - 1.5,
            "Close": prices,
            "Volume": 1000000.0
        }, index=dates)

    # 2. Instanciar Estrategias
    print("\n[2] Inicializando estrategias aprobadas...")
    strat_trend = SMACrossoverStrategy(
        name="SMA_Trend_SPY",
        params={"fast_window": 20, "slow_window": 50, "allow_short": False}
    )
    strat_reversion = RSIReversionStrategy(
        name="RSI_Reversion_SPY",
        params={"period": 14, "oversold": 30.0, "overbought": 70.0}
    )

    # 3. Generar Señales y Backtest Individual
    print("\n[3] Ejecutando simulación de cada estrategia...")
    engine = BacktestEngine(commission_bps=5.0, slippage_bps=5.0)

    sig_trend = strat_trend.generate_signals(data)
    res_trend = engine.run(data, sig_trend, initial_capital=10000.0)

    sig_reversion = strat_reversion.generate_signals(data)
    res_reversion = engine.run(data, sig_reversion, initial_capital=10000.0)

    # Mostrar métricas individuales
    def print_metrics(name, m):
        print(f"\n--- Estrategia: {name} ---")
        for k, v in m.items():
            print(f"  {k:25}: {v}")

    print_metrics(strat_trend.name, res_trend["metrics"])
    print_metrics(strat_reversion.name, res_reversion["metrics"])

    # 4. Combinar en Portafolio Multi-Estrategia
    print("\n[4] Consolidando Portafolio con Asignación por Riesgo (Inverse Volatility)...")
    returns_matrix = pd.DataFrame({
        strat_trend.name: res_trend["data"]["strategy_return"],
        strat_reversion.name: res_reversion["data"]["strategy_return"]
    })

    allocator = PortfolioAllocator(method="inverse_volatility")
    portfolio_res = allocator.combine(returns_matrix, initial_capital=20000.0)

    print("\n--- Ponderaciones Asignadas ---")
    for s_name, w in portfolio_res["weights"].items():
        print(f"  {s_name:25}: {w:.1%}")

    print("\n--- Métricas del Portafolio Consolidado ---")
    for k, v in portfolio_res["metrics"].items():
        print(f"  {k:25}: {v}")

    # 5. Auditoría del Risk Manager
    print("\n[5] Auditoría de Riesgo del Manager...")
    risk_mgr = RiskManager()
    corr_warnings = risk_mgr.audit_correlations(returns_matrix)
    if corr_warnings:
        for w in corr_warnings:
            print(f"  [ALERTA CORRELACIÓN] {w['warning']}")
    else:
        print("  OK: Correlaciones dentro del rango seguro (diversificación efectiva).")

    dd_status = risk_mgr.check_drawdown_limit(portfolio_res["metrics"]["Max Drawdown (%)"] / 100.0)
    print(f"  Estado de Drawdown: {dd_status['action']} (Max DD: {dd_status['current_drawdown_pct']}%, Límite: {dd_status['max_drawdown_limit_pct']}%)")

    print("\n" + "=" * 70)
    print("  EJECUCIÓN DEL PORTAFOLIO FINALIZADA CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    main()
