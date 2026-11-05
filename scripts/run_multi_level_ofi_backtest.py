"""
scripts/run_multi_level_ofi_backtest.py
Script ejecutable para correr el backtest cuantitativo de la Estrategia F6:
Multi-Level Order Flow Imbalance (Xu, Gould & Howison, 2019).

Uso:
    python scripts/run_multi_level_ofi_backtest.py --symbol ES --bars 500
    python scripts/run_multi_level_ofi_backtest.py --symbol NQ --account-size 50000 --dll 1000 --mtdd 2000
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.strategies.multi_level_ofi import (
    MultiLevelOFIStrategy,
    MultiLevelOFIBacktester,
    calculate_spatial_betas,
)
from src.core.futures_specs import get_spec


def generate_synthetic_multilevel_lob(
    symbol: str = "ES",
    n_bars: int = 400,
    levels: int = 5,
    seed: int = 42,
    base_price: float = 5000.0
) -> pd.DataFrame:
    """
    Genera un conjunto de datos sintético de alta fidelidad que simula
    el libro de órdenes profundo (MDP 3.0 L = 5) de CME Globex durante el RTH.
    Incluye dinámicas de acumulación genuina, episodios de spoofing y movimientos de precio.
    """
    np.random.seed(seed)
    spec = get_spec(symbol)
    tick = spec["tick"]

    # Generar timestamps regulares de 1 minuto durante horario RTH (09:30 a 16:00 EST)
    start_dt = datetime(2026, 10, 19, 9, 30)
    timestamps = []
    curr = start_dt

    for _ in range(n_bars):
        if curr.time() >= datetime.strptime("16:00", "%H:%M").time():
            # Siguiente día
            curr = (curr + timedelta(days=1)).replace(hour=9, minute=30)
        timestamps.append(curr)
        curr += timedelta(minutes=1)

    data_dict = {}
    current_mid = base_price

    # Generar series de niveles de precio y profundidades
    # En niveles 1..L:
    # bid_price_l = mid - 0.5 * tick - (l-1)*tick
    # ask_price_l = mid + 0.5 * tick + (l-1)*tick
    rows = []
    
    # Pre-generar regímenes: acumulaciones institucionales vs spoofing con persistencia temporal
    current_regime = "normal"
    regime_bars_left = 0
    intensity = 1.0

    for i in range(n_bars):
        if regime_bars_left <= 0:
            current_regime = np.random.choice(
                ["normal", "long_push", "spoof_long", "short_push", "spoof_short"],
                p=[0.70, 0.10, 0.05, 0.10, 0.05]
            )
            regime_bars_left = np.random.randint(4, 9) if current_regime != "normal" else np.random.randint(5, 15)
            intensity = 1.0
        else:
            regime_bars_left -= 1
            intensity = max(0.3, intensity * 0.85)  # decaimiento gradual del impulso institucional

        regime = current_regime
        
        if regime == "long_push":
            drift = tick * np.random.choice([0.0, 1.0, 2.0], p=[0.2, 0.5, 0.3])
            current_mid += drift
            ofi_1 = int(np.random.randint(220, 480) * intensity)
            ofi_deep = [int((ofi_1 * (0.8 ** l) + np.random.randint(20, 60)) * intensity) for l in range(1, levels)]
        elif regime == "spoof_long":
            current_mid += tick * np.random.choice([0.0, -1.0])
            ofi_1 = int(np.random.randint(200, 350) * intensity)
            ofi_deep = [-int(np.random.randint(100, 250) * intensity) for _ in range(1, levels)]
        elif regime == "short_push":
            drift = tick * np.random.choice([0.0, 1.0, 2.0], p=[0.2, 0.5, 0.3])
            current_mid -= drift
            ofi_1 = -int(np.random.randint(220, 480) * intensity)
            ofi_deep = [-int((abs(ofi_1) * (0.8 ** l) + np.random.randint(20, 60)) * intensity) for l in range(1, levels)]
        elif regime == "spoof_short":
            current_mid += tick * np.random.choice([0.0, 1.0])
            ofi_1 = -int(np.random.randint(200, 350) * intensity)
            ofi_deep = [int(np.random.randint(100, 250) * intensity) for _ in range(1, levels)]
        else:
            current_mid += tick * np.random.choice([-1.0, 0.0, 1.0], p=[0.30, 0.40, 0.30])
            ofi_1 = np.random.randint(-50, 50)
            ofi_deep = [np.random.randint(-30, 30) for _ in range(1, levels)]


        mid_rounded = round(current_mid / tick) * tick
        best_bid = mid_rounded - (tick / 2.0)
        best_ask = mid_rounded + (tick / 2.0)

        row = {
            "Open": best_bid,
            "High": best_ask + tick,
            "Low": best_bid - tick,
            "Close": mid_rounded,
            "ofi_1": float(ofi_1)
        }
        for l in range(2, levels + 1):
            row[f"ofi_{l}"] = float(ofi_deep[l - 2])

        # Cotizaciones LOB explícitas para cálculo de profundidad y absorción
        for l in range(1, levels + 1):
            row[f"bid_price_{l}"] = best_bid - (l - 1) * tick
            row[f"ask_price_{l}"] = best_ask + (l - 1) * tick
            row[f"bid_size_{l}"] = float(np.random.randint(100, 400))
            row[f"ask_size_{l}"] = float(np.random.randint(100, 400))

        rows.append(row)

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps))
    return df


def print_table_header(title: str, width: int = 78):
    print("\n" + "=" * width)
    print(f" {title:^{width-2}} ")
    print("=" * width)


def main():
    parser = argparse.ArgumentParser(description="Backtest Cuantitativo: Estrategia F6 Multi-Level OFI")
    parser.add_argument("--symbol", type=str, default="ES", help="Instrumento objetivo (ES, MES, NQ, MNQ)")
    parser.add_argument("--bars", type=int, default=500, help="Número de barras de 1 minuto a simular")
    parser.add_argument("--account-size", type=float, default=50000.0, help="Capital inicial ($)")
    parser.add_argument("--dll", type=float, default=1000.0, help="Daily Loss Limit contractual ($)")
    parser.add_argument("--mtdd", type=float, default=2000.0, help="Maximum Trailing Drawdown ($)")
    parser.add_argument("--alpha-decay", type=float, default=1.0, help="Exponente de decaimiento espacial de impacto")
    parser.add_argument("--weighting-mode", type=str, default="power_law", choices=["power_law", "pca"], help="Método de ponderación: power_law o pca")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad del entorno estocástico")
    parser.add_argument("--output-csv", type=str, default=None, help="Ruta para exportar registro de operaciones en CSV")

    args = parser.parse_args()

    symbol = args.symbol.upper()
    print_table_header(f"CONFIGURACIÓN DE LA ESTRATEGIA F6: MULTI-LEVEL OFI ({symbol})")
    print(f" Instrumento Objetivo        : {symbol} (CME Globex MDP 3.0)")
    print(f" Capital Inicial             : ${args.account_size:,.2f}")
    print(f" Límite Pérdida Diaria (DLL) : ${args.dll:,.2f}")
    print(f" Trailing Drawdown (MTDD)    : ${args.mtdd:,.2f}")
    print(f" Ponderación Multi-Nivel     : {args.weighting_mode.upper()}")
    print(f" Exponente Decaimiento Ley   : alpha = {args.alpha_decay:.2f}")
    print(f" Número de Barras de 1-Min   : {args.bars}")

    # 1. Generar / Cargar Datos
    print("\n[1/3] Generando datos de microestructura LOB a profundidad L = 5...")
    df = generate_synthetic_multilevel_lob(
        symbol=symbol,
        n_bars=args.bars,
        levels=5,
        seed=args.seed
    )
    print(f" Dataset generado: {len(df)} barras desde {df.index[0]} hasta {df.index[-1]}.")

    # 2. Configurar e Inicializar Backtest
    print("\n[2/3] Ejecutando simulación microestructural y reglas de Prop Trading...")
    strategy = MultiLevelOFIStrategy(
        params={
            "symbol": symbol,
            "levels": 5,
            "alpha_decay": args.alpha_decay,
            "weighting_mode": args.weighting_mode,
            "dll": args.dll,
            "mtdd": args.mtdd,
            "theta_entry": 2.25,
            "phi_min": 0.40,
        }
    )

    backtester = MultiLevelOFIBacktester(
        strategy=strategy,
        account_size=args.account_size,
        daily_loss_limit=args.dll,
        max_trailing_loss=args.mtdd,
        symbol=symbol
    )

    results = backtester.run(df)
    metrics = results["metrics"]
    trades = results["trades"]

    # 3. Mostrar Registro de Operaciones
    print_table_header("REGISTRO AUDITADO DE TRADES EJECUTADOS")
    if len(trades) == 0:
        print(" No se ejecutaron operaciones en el periodo analizado.")
    else:
        print(f" {'ID':<4} | {'Hora Entrada':<16} | {'Dir':<5} | {'Contr':<6} | {'Precio In':<10} | {'Precio Out':<10} | {'PnL Neto ($)':<12} | {'Motivo Salida':<20}")
        print("-" * 96)
        for t in trades:
            dir_str = "LONG" if t.direction == 1 else "SHORT"
            sym_str = f"{t.contracts}x{t.symbol}"
            print(f" {t.trade_id:<4} | {t.entry_time[11:19]:<16} | {dir_str:<5} | {sym_str:<6} | {t.entry_price:<10.2f} | {t.exit_price:<10.2f} | ${t.net_pnl_usd:>10.2f} | {t.exit_reason.value if t.exit_reason else 'OPEN':<20}")

    # 4. Mostrar Resumen de Rendimiento y Evaluación Prop Firm
    print_table_header("EVALUACIÓN CUANTITATIVA Y CONTROL DE RIESGO PROP TRADING")
    print(f" Total Operaciones           : {metrics['total_trades']}")
    print(f" Operaciones Ganadoras       : {metrics.get('winning_trades', 0)}")
    print(f" Operaciones Perdedoras      : {metrics.get('losing_trades', 0)}")
    print(f" Tasa de Acierto (Win Rate)  : {metrics['win_rate_pct']:.2f}%")
    print(f" Factor de Beneficio (PF)    : {metrics['profit_factor']:.2f}")
    print(f" PnL Neto Total              : ${metrics['net_pnl_usd']:,.2f}")
    print(f" Equity Final                : ${metrics['final_equity']:,.2f}")
    print(f" Drawdown Máximo Registrado  : ${metrics['max_drawdown_usd']:,.2f}")
    print(f" Sharpe Ratio                : {metrics['sharpe_ratio']:.2f}")
    print(f" Sortino Ratio               : {metrics['sortino_ratio']:.2f}")
    print(f" Veredicto Prop Firm         : [{metrics['prop_status']}]")
    if metrics.get("breach_reason"):
        print(f" Causa de Infracción         : {metrics['breach_reason']}")
    print("=" * 78 + "\n")

    # Exportar si se solicitó
    if args.output_csv and len(trades) > 0:
        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        trades_df.to_csv(args.output_csv, index=False)
        print(f" [OK] Registro de operaciones guardado exitosamente en: {args.output_csv}\n")


if __name__ == "__main__":
    main()
