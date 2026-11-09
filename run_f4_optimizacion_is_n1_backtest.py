"""
run_f4_optimizacion_is_n1_backtest.py
Script ejecutable para correr el backtest cuantitativo de F4-Optimizacion -IS- N1:
Cointegración Inter-Mercado ES vs NQ y Proceso de Difusión Ornstein-Uhlenbeck.

Corrige las 6 anomalías de auditoría:
1. Sin parada prematura: Evalúa la muestra anual completa (260 días / 20,280 barras).
2. Sharpe y Sortino anualizados sobre retornos diarios (252 días).
3. Parámetros de equilibrio (alpha, beta, theta, sigma_eq) congelados al abrir trade.
4. Ejecución no anticipativa estricta en Open[t+1] con deslizamiento y comisiones CME.
5. Neutralidad Dollar-Beta mediante retículo coprimo y sub-cestas micro balanceadas.
6. Calibración libre de saltos nocturnos (D_intraday) y Filtro de Kalman dinámico.

Uso:
    python run_f4_optimizacion_is_n1_backtest.py --mode in_sample --account-tier 50k
    python run_f4_optimizacion_is_n1_backtest.py --mode in_sample --account-tier 100k
    python run_f4_optimizacion_is_n1_backtest.py --data-nq data/raw/CME_MINI_NQ_5m_bars5000.parquet --data-es data/raw/CME_MINI_ES_5m_bars5000.parquet
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict
import pandas as pd

from src.strategies.f4_optimizacion_is_n1 import (
    F4OptimizationISN1Strategy,
    F4OptimizationISN1Backtester,
    CointegrationOUModelN1,
)
from src.core.data_split import add_data_split_args, partition_from_args, clip_to_partition


def print_banner(text: str, width: int = 78):
    print("\n" + "=" * width)
    print(f" {text:^{width-2}} ")
    print("=" * width)


def print_table_row(k: str, v: Any, width: int = 78):
    col1 = 44
    col2 = width - col1 - 5
    print(f"  {k:<{col1}} : {str(v):>{col2}}")


def load_market_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Formato no soportado ({path.suffix}). Use .parquet o .csv")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    required_cols = ["Open", "High", "Low", "Close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"El archivo {file_path} carece de columnas requeridas: {missing}")

    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest Cuantitativo de F4-Optimizacion -IS- N1: Cointegración ES/NQ y Difusión OU"
    )
    def_nq = "data/raw/CME_MINI_NQ_2021_2023_5m.parquet" if Path("data/raw/CME_MINI_NQ_2021_2023_5m.parquet").exists() else "data/raw/CME_MINI_NQ_5m_3y.parquet"
    def_es = "data/raw/CME_MINI_ES_2021_2023_5m.parquet" if Path("data/raw/CME_MINI_ES_2021_2023_5m.parquet").exists() else "data/raw/CME_MINI_ES_5m_3y.parquet"
    def_xlk = "data/raw/XLK_1d_2021-01-01_2023-12-31.parquet"
    def_spy = "data/raw/SPY_1d_2021-01-01_2023-12-31.parquet"

    parser.add_argument("--data-nq", type=str, default=def_nq, help="Ruta a datos de NQ/MNQ")
    parser.add_argument("--data-es", type=str, default=def_es, help="Ruta a datos de ES/MES")
    parser.add_argument("--data-xlk", type=str, default=def_xlk, help="Ruta a datos diarios de XLK para filtro sectorial")
    parser.add_argument("--data-spy", type=str, default=def_spy, help="Ruta a datos diarios de SPY para filtro sectorial")
    parser.add_argument("--account-tier", type=str, choices=["50k", "100k", "150k", "custom"], default="50k",
                        help="Preset de cuenta de Prop Trading: 50k, 100k, 150k o custom (def: 50k)")
    parser.add_argument("--capital", type=float, default=50000.0, help="Capital inicial en USD")
    parser.add_argument("--dll", type=float, default=1000.0, help="Daily Loss Limit en USD")
    parser.add_argument("--mtdd", type=float, default=2000.0, help="Trailing Drawdown Máximo en USD")
    parser.add_argument("--profit-target", type=float, default=3000.0, help="Objetivo de beneficio en USD")
    parser.add_argument("--trailing-mode", type=str, choices=["eod", "intraday"], default="eod")
    parser.add_argument("--early-stopping", action="store_true", default=False,
                        help="Si se activa, detiene la simulación apenas alcanza el profit target (sesgo antiguo). Por defecto False.")
    parser.add_argument("--contracts", type=str, choices=["micro", "standard"], default="micro",
                        help="Contratos: 'micro' (MNQ/MES) o 'standard' (NQ/ES)")
    parser.add_argument("--s-open", type=float, default=1.25, help="Umbral S-score de apertura (def: 1.25)")
    parser.add_argument("--s-close", type=float, default=0.25, help="Umbral S-score de Take-Profit (def: 0.25)")
    parser.add_argument("--s-stop", type=float, default=3.25, help="Umbral S-score de Stop-Loss (def: 3.25)")
    parser.add_argument("--k-calib", type=int, default=2340, help="Ventana rodante hacia atrás en barras (def: 2,340)")
    parser.add_argument("--min-calib", type=int, default=390, help="Barras mínimas de calibración (def: 390)")
    parser.add_argument("--use-kalman", dest="use_kalman", action="store_true", default=False, help="Activar Filtro de Kalman recursivo")
    parser.add_argument("--no-kalman", dest="use_kalman", action="store_false", help="Desactivar Filtro de Kalman (usar OLS)")
    parser.add_argument("--unify-beta-ols", dest="unify_beta_ols", action="store_true", default=True, help="Unificar beta con OLS rodante en toda la tubería (def: True)")
    parser.add_argument("--no-unify-beta-ols", dest="unify_beta_ols", action="store_false", help="Usar beta del Filtro de Kalman en dimensionamiento de cobertura")
    parser.add_argument("--max-coprime-nq", type=int, default=7, help="Máximo de contratos NQ en retículo coprimo")
    parser.add_argument("--n-basket-max", type=int, default=3, help="Máximo de cestas por trade (def: 3)")
    parser.add_argument("--allow-sub-baskets", action="store_true", default=True)
    parser.add_argument("--dynamic-ratio", action="store_true", default=True)
    parser.add_argument("--phi-dll", type=float, default=None)
    parser.add_argument("--psi-tmdd", type=float, default=None)
    parser.add_argument("--export-trades", type=str, default="reports/f4_optimizacion_is_n1_trades.csv")
    parser.add_argument("--export-summary", type=str, default="reports/f4_optimizacion_is_n1_summary.json")
    add_data_split_args(parser, mode_arg_name="--mode")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    TIER_PRESETS = {
        "50k": {
            "capital": 50000.0, "dll": 1000.0, "mtdd": 2000.0, "profit_target": 3000.0,
            "phi_dll": 0.60, "psi_tmdd": 0.30, "desc": "Cuenta $50k Prop Trading (60% DLL, 30% TMDD - 50K_ADAPTED)"
        },
        "100k": {
            "capital": 100000.0, "dll": 2000.0, "mtdd": 3000.0, "profit_target": 6000.0,
            "phi_dll": 0.35, "psi_tmdd": 0.20, "desc": "Cuenta $100k Prop Trading Estándar"
        },
        "150k": {
            "capital": 150000.0, "dll": 3000.0, "mtdd": 4500.0, "profit_target": 9000.0,
            "phi_dll": 0.35, "psi_tmdd": 0.20, "desc": "Cuenta $150k Institucional"
        },
    }

    if args.account_tier in TIER_PRESETS:
        p = TIER_PRESETS[args.account_tier]
        if args.capital == 50000.0 and args.account_tier != "50k":
            args.capital = p["capital"]
        if args.dll == 1000.0 and args.account_tier != "50k":
            args.dll = p["dll"]
        if args.mtdd == 2000.0 and args.account_tier != "50k":
            args.mtdd = p["mtdd"]
        if args.profit_target == 3000.0 and args.account_tier != "50k":
            args.profit_target = p["profit_target"]
        if args.phi_dll is None:
            args.phi_dll = p["phi_dll"]
        if args.psi_tmdd is None:
            args.psi_tmdd = p["psi_tmdd"]

    if args.phi_dll is None:
        args.phi_dll = 0.60
    if args.psi_tmdd is None:
        args.psi_tmdd = 0.30

    print_banner("F4-OPTIMIZACION -IS- N1: COINTEGRACIÓN ES/NQ & DIFUSIÓN OU")
    print("  Versión Optimizada Auditada: Corrección de las 6 Anomalías Cuantitativas")
    print(f"  Modo Contractual    : {args.contracts.upper()} (Sub-cestas coprimas Dollar-Beta neutral)")
    print(f"  Filtro de Kalman    : {'ACTIVO (beta dinámico)' if args.use_kalman else 'DESACTIVADO (OLS rodante)'}")
    print(f"  Ejecución Órdenes   : ESTRICTAMENTE NO ANTICIPATIVA EN Open[t+1]")
    print(f"  Parámetros Entrada  : CONGELADOS durante trade abierto (Anti-Moving Target)")
    print(f"  Evaluación Muestral : {'MUESTRA COMPLETA (Sin parada prematura)' if not args.early_stopping else 'PARADA TEMPRANA (Legacy)'}")
    print(f"  Capital Inicial     : ${args.capital:,.2f} USD")
    print(f"  Límites de Riesgo   : DLL=${args.dll:,.2f} | TMDD=${args.mtdd:,.2f} | PT=${args.profit_target:,.2f}")

    print("\n[1/3] Cargando y sincronizando datos de futuros CME...")
    df_nq = load_market_data(args.data_nq)
    df_es = load_market_data(args.data_es)

    df_nq, part_label = partition_from_args(df_nq, args, "F4_Opt_NQ")
    df_es, _ = partition_from_args(df_es, args, "F4_Opt_ES")
    common_idx = df_nq.index.intersection(df_es.index)
    df_nq = df_nq.loc[common_idx]
    df_es = df_es.loc[common_idx]

    print(f"  Partición: {part_label} ({len(df_nq):,} barras sincronizadas de 5m)")

    df_xlk = None
    df_spy = None
    if args.data_xlk and Path(args.data_xlk).exists() and args.data_spy and Path(args.data_spy).exists():
        df_xlk = load_market_data(args.data_xlk)
        df_spy = load_market_data(args.data_spy)
        df_xlk = clip_to_partition(df_xlk, df_nq)
        df_spy = clip_to_partition(df_spy, df_nq)
        print(f"  Filtro Sectorial    : XLK ({len(df_xlk)} barras) y SPY ({len(df_spy)} barras) sincronizados")

    print("\n[2/3] Configurando estrategia F4-Optimizacion -IS- N1...")
    strat_params = {
        "contract_mode": args.contracts,
        "use_kalman": args.use_kalman,
        "unify_beta_ols": args.unify_beta_ols,
        "max_coprime_nq": args.max_coprime_nq,
        "allow_sub_baskets": args.allow_sub_baskets,
        "dynamic_basket_ratio": args.dynamic_ratio,
        "s_open": args.s_open,
        "s_close": args.s_close,
        "s_stop": args.s_stop,
        "k_calib": args.k_calib,
        "min_calib_bars": args.min_calib,
        "phi_dll": args.phi_dll,
        "psi_tmdd": args.psi_tmdd,
        "survival_floor": 105.0,
        "time_decay_multiplier": 3.5,
        "n_basket_max": args.n_basket_max,
        "source_tz": "America/Mexico_City",
        "target_tz": "America/New_York",
    }
    strategy = F4OptimizationISN1Strategy(params=strat_params)

    backtester = F4OptimizationISN1Backtester(
        strategy=strategy,
        account_size=args.capital,
        daily_loss_limit=args.dll,
        max_trailing_loss=args.mtdd,
        trailing_mode=args.trailing_mode,
        profit_target=args.profit_target,
        early_stopping=args.early_stopping,
    )

    print("\n[3/3] Ejecutando simulación cuantitativa optimizada...")
    results = backtester.run(df_nq=df_nq, df_es=df_es, df_xlk=df_xlk, df_spy=df_spy)

    metrics = results["metrics"]
    prop_eval = results["prop_evaluation"]
    trades_df = results["trades"]
    audit = results.get("cointegration_audit", {})

    print_banner("EVALUACIÓN PROP TRADING (F4-OPTIMIZACION -IS- N1)")
    print_table_row("Estado de Evaluación", prop_eval["status"])
    print_table_row("Razón del Veredicto", prop_eval["reason"])
    if prop_eval.get("target_achieved_date"):
        print_table_row("Fecha Alcance Target", prop_eval["target_achieved_date"])
    print_table_row("Balance Final", f"${prop_eval['final_balance']:,.2f}")
    print_table_row("Beneficio Neto ($)", f"${prop_eval['net_profit']:,.2f}")
    print_table_row("Pico Máximo de Capital", f"${prop_eval['peak_equity']:,.2f}")
    print_table_row("Piso Final Trailing DD", f"${prop_eval['final_floor']:,.2f}")
    print_table_row("Sesiones RTH Evaluadas", prop_eval["total_days_evaluated"])

    print_banner("MÉTRICAS CUANTITATIVAS REALISTAS Y AUDITADAS")
    print_table_row("Total de Operaciones", metrics["Total Trades"])
    print_table_row("Operaciones Ganadoras", metrics.get("Winning Trades", 0))
    print_table_row("Operaciones Perdedoras", metrics.get("Losing Trades", 0))
    print_table_row("Tasa de Acierto (Win Rate)", f"{metrics['Win Rate (%)']:.2f}%")
    print_table_row("Factor de Beneficio (Profit Factor)", metrics["Profit Factor"])
    print_table_row("Retorno Total Acumulado", f"{metrics['Total Return (%)']:.2f}%")
    print_table_row("Retorno Anualizado Estimado", f"{metrics.get('Annualized Return (%)', 0.0):.2f}%")
    print_table_row("Drawdown Máximo Flotante", f"${metrics['Max Drawdown ($)']:,.2f}")
    print_table_row("Ratio Sharpe Anualizado (252d)", f"{metrics['Sharpe Ratio (Daily 252d)']:.2f}")
    print_table_row("Ratio Sharpe por Trade (sin inflar)", f"{metrics['Sharpe Ratio (Per Trade)']:.4f}")
    print_table_row("Ratio Sortino Anualizado (252d)", f"{metrics['Sortino Ratio (Daily 252d)']:.2f}")

    if audit:
        print_banner("AUDITORÍA DE COINTEGRACIÓN Y DIFUSIÓN ORNSTEIN-UHLENBECK")
        print_table_row("Barras Evaluadas", f"{audit['bars_evaluated']:,}")
        print_table_row("Aprobación OU Válido", f"{audit['valid_ou_pct']:.1f}%")
        print_table_row("Aprobación MacKinnon ADF", f"{audit['adf_passed_pct']:.1f}%")
        print_table_row("Vida Media en Rango (Tau)", f"{audit['tau_passed_pct']:.1f}%")
        print_table_row("Beta Promedio (Beta_hedge)", f"{audit['mean_beta']:.4f}")
        print_table_row("Vida Media Promedio (Tau)", f"{audit['mean_tau_half_min']:.1f} min")
        print_table_row("Volatilidad Equilibrio (Sigma_eq)", f"{audit['mean_sigma_eq']:.6f}")

    if "Exit Reasons" in metrics and metrics["Exit Reasons"]:
        print_banner("DISTRIBUCIÓN DE SALIDAS (CON PARÁMETROS CONGELADOS)")
        for reason, count in metrics["Exit Reasons"].items():
            print_table_row(f"Razón: {reason}", count)

    if not trades_df.empty:
        print_banner("MUESTRA DE OPERACIONES EJECUTADAS EN OPEN[t+1]")
        display_cols = ["trade_id", "entry_date", "entry_time", "signal", "contracts_nq", "contracts_es", "entry_price_nq", "entry_price_es", "exit_reason", "friction_usd", "net_pnl_usd"]
        available_cols = [c for c in display_cols if c in trades_df.columns]
        print(trades_df[available_cols].head(10).to_string(index=False))
        if len(trades_df) > 10:
            print(f"  ... y {len(trades_df) - 10} operaciones más.")

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not trades_df.empty:
        trades_df.to_csv(args.export_trades, index=False)
        print(f"\n[+] Operaciones exportadas a: {args.export_trades}")

    summary_data = {
        "strategy": "F4-Optimizacion -IS- N1",
        "parameters": strat_params,
        "prop_evaluation": prop_eval,
        "metrics": metrics,
        "cointegration_audit": audit,
    }
    with open(args.export_summary, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"[+] Resumen cuantitativo exportado a: {args.export_summary}\n")


if __name__ == "__main__":
    main()
