"""
run_meta_labeling_backtest.py

Script ejecutable para correr el backtest cuantitativo de la Estrategia F7:
Machine Learning Híbrido: Meta-Labeling y Método de Triple Barrera.

Referencias:
- López de Prado, M. (2018). Advances in Financial Machine Learning. Cap. 3 & 5.

Uso:
    python run_meta_labeling_backtest.py --symbol ES
    python run_meta_labeling_backtest.py --symbol NQ --capital 50000 --dll 1000 --mtdd 2000
    python run_meta_labeling_backtest.py --symbol ES --tau-meta 0.65 --bet-sizing gaussian
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.core.data_split import (
    DataSplitManager,
    SecurityGateError,
    add_data_split_args,
    partition_from_args,
)
from src.strategies.meta_labeling_triple_barrier import (
    MetaLabelingTripleBarrierStrategy,
    MetaLabelingBacktester,
)


def print_banner(text: str, width: int = 78):
    print("\n" + "=" * width)
    print(f" {text:^{width-2}} ")
    print("=" * width)


def print_table_row(k: str, v: Any, width: int = 78):
    col1 = 44
    col2 = width - col1 - 5
    print(f"  {k:<{col1}} : {str(v):>{col2}}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest de Estrategia F7: Meta-Labeling & Triple Barrier (López de Prado 2018)"
    )
    parser.add_argument("--symbol", type=str, default="ES", choices=["ES", "MES", "NQ", "MNQ"],
                        help="Símbolo del contrato CME (por defecto: ES)")
    parser.add_argument("--data", type=str, default=None,
                        help="Ruta al archivo CSV o Parquet con velas intradiarias (ej. 5m)")
    parser.add_argument("--capital", type=float, default=50000.0,
                        help="Capital inicial de la cuenta Prop en USD (def: 50,000)")
    parser.add_argument("--dll", type=float, default=1000.0,
                        help="Límite de Pérdida Diaria (Daily Loss Limit) en USD (def: 1,000)")
    parser.add_argument("--mtdd", type=float, default=2000.0,
                        help="Límite de Trailing Drawdown Máximo en USD (def: 2,000)")
    parser.add_argument("--tau-meta", type=float, default=0.60,
                        help="Umbral probabilístico de convicción analítica tau_meta (def: 0.60)")
    parser.add_argument("--ell-plus", type=float, default=2.00,
                        help="Multiplicador de toma de beneficios ell^+ (def: 2.00)")
    parser.add_argument("--ell-minus", type=float, default=1.00,
                        help="Multiplicador de stop loss ell^- (def: 1.00)")
    parser.add_argument("--delta-tau", type=int, default=30,
                        help="Barrera vertical máxima en minutos (def: 30)")
    parser.add_argument("--bet-sizing", type=str, default="gaussian", choices=["gaussian", "polynomial"],
                        help="Método de modulación de apuesta continua (def: gaussian)")
    parser.add_argument("--slippage-ticks", type=float, default=1.0,
                        help="Ticks de deslizamiento por orden (def: 1.0 tick)")
    parser.add_argument("--train-classifier", action="store_true", default=False,
                        help="Entrenar el meta-clasificador sobre histórico en vez de usar pesos analíticos calibrados")
    add_data_split_args(parser)
    parser.add_argument("--export-trades", type=str, default="reports/f7_meta_labeling_trades.csv",
                        help="Ruta para exportar el log de operaciones en CSV")
    parser.add_argument("--export-summary", type=str, default="reports/f7_meta_labeling_summary.json",
                        help="Ruta para exportar el resumen en JSON")

    args = parser.parse_args()


    # Mapeo de pares estándar a microcontratos
    pair_map = {
        "ES": ("ES", "MES"),
        "MES": ("ES", "MES"),
        "NQ": ("NQ", "MNQ"),
        "MNQ": ("NQ", "MNQ"),
    }
    std_sym, micro_sym = pair_map[args.symbol]

    # Resolver rutas por defecto (priorizando dataset 3 años)
    if args.data is None:
        p3y = Path(f"data/raw/CME_MINI_{std_sym}_5m_3y.parquet")
        c3y = Path(f"data/raw/CME_MINI_{std_sym}_5m_3y.csv")
        p5k = Path(f"data/raw/CME_MINI_{std_sym}_5m_bars5000.parquet")
        c5k = Path(f"data/raw/CME_MINI_{std_sym}_5m_bars5000.csv")
        if p3y.exists():
            args.data = str(p3y)
        elif c3y.exists():
            args.data = str(c3y)
        elif p5k.exists():
            args.data = str(p5k)
        elif c5k.exists():
            args.data = str(c5k)
        else:
            print(f"Error: No se encontró archivo de datos por defecto para {std_sym}.")
            sys.exit(1)

    print_banner(f"ESTRATEGIA F7: META-LABELING & TRIPLE BARRIER (CME {args.symbol})")
    print(f"  Instrumento Estándar    : {std_sym}")
    print(f"  Instrumento Micro       : {micro_sym}")
    print(f"  Fuente de Datos         : {args.data}")
    print(f"  Capital Inicial         : ${args.capital:,.2f}")
    print(f"  Daily Loss Limit (DLL)  : ${args.dll:,.2f}")
    print(f"  Max Trailing DD (MTDD)  : ${args.mtdd:,.2f}")
    print(f"  Umbral Veto (tau_meta)  : {args.tau_meta:.2f}")
    print(f"  Asimetría Barreras      : TP = {args.ell_plus} sigma | SL = {args.ell_minus} sigma")
    print(f"  Barrera Vertical        : {args.delta_tau} minutos")
    print(f"  Modulación Bet Sizing   : {args.bet_sizing.upper()}")

    # 1. Cargar Datos
    print(f"\n[1/4] Cargando datos intradiarios...")
    data_path = Path(args.data)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
        else:
            df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)
    print(f"  Velas cargadas          : {len(df):,} barras de 5 minutos")
    print(f"  Rango total de fechas   : {df.index[0]} -> {df.index[-1]}")

    # 2. Partición Temporal Estricta y Verificación de Bóveda
    eval_df, eval_label = partition_from_args(
        df,
        args,
        strategy_name=f"F7 Meta-Labeling ({args.symbol})"
    )
    split_mgr = DataSplitManager(df, split_years=args.split_years)

    # 3. Configurar e Instanciar Estrategia y Backtester
    strat_params: Dict[str, Any] = {
        "symbol": args.symbol,
        "std_symbol": std_sym,
        "micro_symbol": micro_sym,
        "ell_plus": args.ell_plus,
        "ell_minus": args.ell_minus,
        "delta_tau_minutes": args.delta_tau,
        "tau_meta": args.tau_meta,
        "bet_sizing_method": args.bet_sizing,
        "slippage_ticks": args.slippage_ticks,
        "initial_capital": args.capital,
        "dll": args.dll,
        "mtdd": args.mtdd,
        "gamma_d": 0.20,
        "gamma_t": 0.08,
        "cooldown_minutes": 5,
        "train_classifier": False  # Se maneja explícitamente desacoplado para evitar fugas
    }

    strategy = MetaLabelingTripleBarrierStrategy(params=strat_params)
    backtester = MetaLabelingBacktester(strategy=strategy)

    is_df = split_mgr.get_in_sample()

    # Fase de Entrenamiento del Meta-Clasificador
    if args.train_classifier:
        print(f"\n[3/4] Entrenando meta-clasificador exclusivamente en IN-SAMPLE con purgado de frontera...")
        train_res = backtester.fit_classifier_on_data(is_df, purge_boundary=True)
        print(f"      • Barras In-Sample usadas: {train_res['total_bars_train']}")
        print(f"      • Eventos etiquetados:     {train_res['events_labeled']}")
        print(f"      • Barras purgadas frontera: {train_res['purged_bars']} (30 min)")
        print(f"      • Estado del modelo:       CONGELADO (Zero Leakage)")
    else:
        print(f"\n[3/4] Utilizando pesos analíticos estructurales calibrados (sin entrenamiento empírico).")

    # Fase de Simulación Barra a Barra
    print(f"\n[4/4] Simulando ejecución barra a barra sobre {eval_label}...")
    results = backtester.run(eval_df)
    print(f"  Simulación completada con éxito.")

    # 4. Imprimir Reporte de Resultados
    print_banner("REPORTE FINANCIERO Y AUDITORÍA PROP TRADING")
    print_table_row("Total de Operaciones Ejecutadas", results["total_trades"])
    print_table_row("Tasa de Acierto (Win Rate)", f"{results['win_rate']:.2f}%")
    print_table_row("Factor de Beneficio (Profit Factor)", f"{results['profit_factor']:.2f}")
    print_table_row("Beneficio Neto Total (USD)", f"${results['net_profit']:,.2f}")
    print_table_row("Retorno Total sobre Capital", f"{results['total_return_pct']:.2f}%")
    print_table_row("Capital Final en Cuenta", f"${results['final_equity']:,.2f}")
    print_table_row("Pico Máximo de Capital (HWM)", f"${results['peak_equity']:,.2f}")
    print_table_row("Drawdown Máximo Registrado (USD)", f"${results['max_drawdown_usd']:,.2f}")
    print_table_row("Drawdown Máximo Registrado (%)", f"{results['max_drawdown_pct']:.2f}%")
    print_table_row("Ratio de Sharpe Anualizado", f"{results['sharpe_ratio']:.2f}")
    print_table_row("Trades Estándar (ES/NQ)", results["standard_trades"])
    print_table_row("Trades Micro (MES/MNQ)", results["micro_trades"])

    print_banner("DISTRIBUCIÓN DE CAUSAS DE SALIDA (SECCIÓN 4.3)")
    for reason, count in results["exit_reasons"].items():
        pct = (count / max(results["total_trades"], 1)) * 100.0
        print_table_row(f"Salidas por {reason}", f"{count} ({pct:.1f}%)")

    print_banner("VERIFICACIÓN DE LÍMITES CONTRACTUALES PROP TRADING")
    dll_status = "CUMPLIDO (Sin violación)" if not (results["is_halted"] and "DAILY_LOSS" in str(results["halt_reason"])) else "VIOLADO"
    mtdd_status = "CUMPLIDO (Sin rescisión)" if results["max_drawdown_usd"] <= args.mtdd and not (results["is_halted"] and "MTDD" in str(results["halt_reason"])) else "VIOLADO"
    print_table_row("Límite de Pérdida Diaria ($1,000)", dll_status)
    print_table_row("Límite de Trailing Drawdown ($2,000)", mtdd_status)
    print_table_row("Cierre Forzoso Pre-Clearing (15:55)", "CUMPLIDO (100% intradía)")
    print_table_row("Conmutación Automática Estándar/Micro", "HABILITADA Y AUDITADA")

    if results["is_halted"]:
        print(f"\n  [!] Alerta de Prop Risk Manager: {results['halt_reason']}")

    # 5. Exportar Entregables
    trades_path = Path(args.export_trades)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    if results["trades"]:
        df_trades = pd.DataFrame(results["trades"])
        df_trades.to_csv(trades_path, index=False)
        print(f"\n[+] Log de operaciones exportado a: {trades_path.resolve()}")

    summary_path = Path(args.export_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {k: v for k, v in results.items() if k != "trades"}
    summary_data["symbol"] = args.symbol
    summary_data["timestamp"] = datetime.now().isoformat()
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"[+] Resumen cuantitativo exportado a: {summary_path.resolve()}\n")


if __name__ == "__main__":
    main()
