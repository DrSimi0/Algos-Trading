"""
run_cointegration_ou_backtest.py
Script ejecutable para correr el backtest cuantitativo de la Estrategia F4:
Cointegración Inter-Mercado ES vs NQ y Proceso de Difusión Ornstein-Uhlenbeck (StatArb OU).

Uso:
    python run_cointegration_ou_backtest.py
    python run_cointegration_ou_backtest.py --mode micro --capital 50000 --dll 1000 --mtdd 2000
    python run_cointegration_ou_backtest.py --data-nq data/raw/CME_MINI_NQ_5m_bars5000.parquet --data-es data/raw/CME_MINI_ES_5m_bars5000.parquet
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict
import pandas as pd

from src.strategies.cointegration_ou import (
    CointegrationOUStrategy,
    CointegrationOUBacktester,
    CointegrationOUModel,
)
from src.core.data_split import add_data_split_args, partition_from_args


def print_banner(text: str, width: int = 78):
    """Imprime un banner destacado en la consola."""
    print("\n" + "=" * width)
    print(f" {text:^{width-2}} ")
    print("=" * width)


def print_table_row(k: str, v: Any, width: int = 78):
    """Imprime una fila clave-valor formateada."""
    col1 = 44
    col2 = width - col1 - 5
    print(f"  {k:<{col1}} : {str(v):>{col2}}")


def load_market_data(file_path: str) -> pd.DataFrame:
    """Carga datos de mercado desde CSV o Parquet con validación de columnas."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {file_path}")

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
        raise ValueError(f"El archivo {file_path} carece de las columnas requeridas: {missing}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Cuantitativo de Estrategia F4: Cointegración ES/NQ y Difusión Ornstein-Uhlenbeck"
    )
    # Rutas de datos (prioriza 3 años)
    def_nq = "data/raw/CME_MINI_NQ_5m_3y.parquet" if Path("data/raw/CME_MINI_NQ_5m_3y.parquet").exists() else "data/raw/CME_MINI_NQ_5m_bars5000.parquet"
    def_es = "data/raw/CME_MINI_ES_5m_3y.parquet" if Path("data/raw/CME_MINI_ES_5m_3y.parquet").exists() else "data/raw/CME_MINI_ES_5m_bars5000.parquet"
    parser.add_argument(
        "--data-nq", type=str, default=def_nq,
        help="Ruta al archivo Parquet o CSV con velas de 5m de NQ/MNQ"
    )
    parser.add_argument(
        "--data-es", type=str, default=def_es,
        help="Ruta al archivo Parquet o CSV con velas de 5m de ES/MES"
    )
    # Parámetros de cuenta y Prop Trading
    parser.add_argument("--account-tier", type=str, choices=["50k", "100k", "150k", "custom"], default="custom",
                        help="Preset de cuenta de Prop Trading: '50k', '100k', '150k' o 'custom' (def: custom)")
    parser.add_argument("--capital", type=float, default=50000.0,
                        help="Capital inicial de la cuenta en USD (def: 50,000)")
    parser.add_argument("--dll", type=float, default=1000.0,
                        help="Límite de Pérdida Diaria (Daily Loss Limit) en USD (def: 1,000)")
    parser.add_argument("--mtdd", type=float, default=2000.0,
                        help="Límite de Trailing Drawdown Máximo en USD (def: 2,000)")
    parser.add_argument("--profit-target", type=float, default=3000.0,
                        help="Objetivo de beneficio de evaluación en USD (def: 3,000)")
    parser.add_argument("--trailing-mode", type=str, choices=["eod", "intraday"], default="eod",
                        help="Modo de cálculo del Trailing Drawdown (def: eod)")
    # Parámetros de la estrategia OU (renombrado a --contracts para resolver colisión CLI)
    parser.add_argument("--contracts", "--contract-mode", dest="contracts", type=str, choices=["micro", "standard"], default="micro",
                        help="Contratos a operar: 'micro' (MNQ/MES) o 'standard' (NQ/ES) (def: micro)")
    parser.add_argument("--s-open", type=float, default=1.25,
                        help="Umbral S-score de apertura de posición (def: 1.25)")
    parser.add_argument("--s-close", type=float, default=0.25,
                        help="Umbral S-score de salida por Take-Profit (def: 0.25)")
    parser.add_argument("--s-stop", type=float, default=3.25,
                        help="Umbral S-score de salida por Stop-Loss estructural (def: 3.25)")
    parser.add_argument("--k-calib", type=int, default=2340,
                        help="Ventana máxima de calibración histórica en barras (def: 2,340)")
    parser.add_argument("--min-calib", type=int, default=390,
                        help="Mínimo de barras históricas para iniciar calibración (def: 390)")
    parser.add_argument("--n-basket-max", type=int, default=3,
                        help="Número máximo de cestas permitidas por trade (def: 3)")
    parser.add_argument("--allow-sub-baskets", action="store_true", default=True,
                        help="Permite sub-cestas (1 MNQ vs exact MES) cuando el presupuesto es insuficiente para 10 MNQ")
    parser.add_argument("--dynamic-ratio", action="store_true", default=True,
                        help="Calcula dinámicamente el ratio de cobertura Dollar-Beta (RUB-F4-04)")
    parser.add_argument("--phi-dll", type=float, default=0.35,
                        help="Fracción del límite de pérdida diaria arriesgable por trade (def: 0.35)")
    parser.add_argument("--phi-tmdd", type=float, default=None,
                        help="Fracción del colchón de Trailing Drawdown arriesgable por trade (def: según tier o 0.20)")
    # Exportación
    parser.add_argument("--export-trades", type=str, default="reports/cointegration_ou_trades.csv",
                        help="Ruta para exportar el registro de operaciones CSV")
    parser.add_argument("--export-summary", type=str, default="reports/cointegration_ou_summary.json",
                        help="Ruta para exportar el resumen de métricas JSON")
    add_data_split_args(parser, mode_arg_name="--mode")

    args = parser.parse_args()

    # Pre-configuración según Account Tier (si fue especificado)
    TIER_PRESETS = {
        "50k": {
            "capital": 50000.0,
            "dll": 1000.0,
            "mtdd": 2000.0,
            "profit_target": 3000.0,
            "phi_dll": 0.60,
            "phi_tmdd": 0.30,
            "label": "50K_ADAPTED",
            "desc": "Cuenta $50k (Calibrada con phi_dll=0.60 y phi_tmdd=0.30 para sub-cesta micro)"
        },
        "100k": {
            "capital": 100000.0,
            "dll": 2000.0,
            "mtdd": 3000.0,
            "profit_target": 6000.0,
            "phi_dll": 0.35,
            "phi_tmdd": 0.20,
            "label": "100K_STANDARD",
            "desc": "Cuenta $100k Estándar (Parámetros conservadores de Prop Firm)"
        },
        "150k": {
            "capital": 150000.0,
            "dll": 3000.0,
            "mtdd": 4500.0,
            "profit_target": 9000.0,
            "phi_dll": 0.35,
            "phi_tmdd": 0.20,
            "label": "150K_INSTITUTIONAL",
            "desc": "Cuenta $150k Institucional (Parámetros conservadores de Prop Firm)"
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
        if args.phi_dll == 0.35:
            args.phi_dll = p["phi_dll"]
        if args.phi_tmdd is None:
            args.phi_tmdd = p["phi_tmdd"]

    if args.phi_tmdd is None:
        args.phi_tmdd = 0.20

    print_banner("ESTRATEGIA F4: COINTEGRACIÓN ES/NQ & PROCESO ORNSTEIN-UHLENBECK")
    print("  Fundamento Académico: Avellaneda & Lee (2010), Hasbrouck (2003)")
    print(f"  Modo Contractual    : {args.contracts.upper()} ({'10 MNQ vs MES dinámico' if args.contracts == 'micro' else '1 NQ vs ES dinámico'})")
    if args.account_tier in TIER_PRESETS:
        print(f"  Preset de Cuenta    : {TIER_PRESETS[args.account_tier]['desc']}")
    print(f"  Capital Inicial     : ${args.capital:,.2f} USD")
    print(f"  Límite Pérdida Día  : ${args.dll:,.2f} USD (Fracción riesgosa por trade phi_dll: {args.phi_dll:.2f})")
    print(f"  Trailing Drawdown   : ${args.mtdd:,.2f} USD (Modo: {args.trailing_mode} | phi_tmdd: {args.phi_tmdd:.2f})")
    print(f"  Umbrales S-Score    : Entry={args.s_open:.2f} | TP={args.s_close:.2f} | SL={args.s_stop:.2f}")

    print("\n[1/3] Cargando series históricas de futuros CME...")
    df_nq = load_market_data(args.data_nq)
    df_es = load_market_data(args.data_es)
    
    # Partición temporal sincronizada y obligatoria
    df_nq, part_label = partition_from_args(df_nq, args, "F4_Cointegration_NQ")
    df_es, _ = partition_from_args(df_es, args, "F4_Cointegration_ES")
    common_idx = df_nq.index.intersection(df_es.index)
    df_nq = df_nq.loc[common_idx]
    df_es = df_es.loc[common_idx]
    
    print(f"  NQ: {len(df_nq):,} barras sincronizadas desde {df_nq.index[0]} hasta {df_nq.index[-1]}")
    print(f"  ES: {len(df_es):,} barras sincronizadas desde {df_es.index[0]} hasta {df_es.index[-1]}")

    print("\n[2/3] Configurando estrategia y motor de backtesting...")
    strat_params = {
        "contract_mode": args.contracts,
        "s_open": args.s_open,
        "s_close": args.s_close,
        "s_stop": args.s_stop,
        "k_calib": args.k_calib,
        "min_calib_bars": args.min_calib,
        "n_basket_max": args.n_basket_max,
        "allow_sub_baskets": args.allow_sub_baskets,
        "dynamic_basket_ratio": args.dynamic_ratio,
        "phi_dll": args.phi_dll,
        "phi_tmdd": args.phi_tmdd,
        "source_tz": "America/Mexico_City",
        "target_tz": "America/New_York",
    }
    strategy = CointegrationOUStrategy(params=strat_params)

    backtester = CointegrationOUBacktester(
        strategy=strategy,
        account_size=args.capital,
        daily_loss_limit=args.dll,
        max_trailing_loss=args.mtdd,
        trailing_mode=args.trailing_mode,
        profit_target=args.profit_target,
    )

    print("\n[3/3] Ejecutando simulación de arbitraje estadístico...")
    results = backtester.run(df_nq=df_nq, df_es=df_es)

    metrics = results["metrics"]
    prop_eval = results["prop_evaluation"]
    trades_df = results["trades"]
    audit = results.get("cointegration_audit", {})

    # Imprimir auditoría cuantitativa de cointegración y difusión OU
    if audit:
        print_banner("AUDITORÍA DE COINTEGRACIÓN Y DIFUSIÓN ORNSTEIN-UHLENBECK")
        print_table_row("Barras RTH Evaluadas (post-calibración)", f"{audit['bars_evaluated']:,}")
        print_table_row("Modelos OU Válidos y Ergódicos", f"{audit['valid_ou_count']:,} ({audit['valid_ou_pct']:.1f}%)")
        print_table_row("Aprobación Test Dickey-Fuller (ADF)", f"{audit['adf_passed_count']:,} ({audit['adf_passed_pct']:.1f}%)")
        print_table_row("Vida Media Reversión en Rango (Tau)", f"{audit['tau_passed_count']:,} ({audit['tau_passed_pct']:.1f}%)")
        print_table_row("Condiciones Simultáneas Cumplidas", f"{audit['both_passed_count']:,} ({audit['both_passed_pct']:.1f}%)")
        print_table_row("Beta Cobertura Promedio (Beta_hedge)", f"{audit['mean_beta']:.4f} [min: {audit['min_beta']:.4f}, max: {audit['max_beta']:.4f}]")
        print_table_row("Vida Media Promedio (Tau 1/2)", f"{audit['mean_tau_half_min']:.1f} min ({audit['mean_tau_half_days']:.2f} sesiones)")
        print_table_row("Volatilidad de Equilibrio (Sigma_eq)", f"{audit['mean_sigma_eq']:.6f}")
        print_table_row("Stop-Loss Monetario Sub-Cesta (1 MNQ)", f"${audit['latest_sub_basket_sl_usd']:,.2f} USD")
        print_table_row("Stop-Loss Cesta Estándar (10 MNQ)", f"${audit['latest_standard_basket_sl_usd']:,.2f} USD")

    # Imprimir resumen de Prop Trading
    print_banner("EVALUACIÓN DE CUENTA DE PROP TRADING")
    print_table_row("Estado de Evaluación", prop_eval["status"])
    print_table_row("Razón del Veredicto", prop_eval["reason"])
    print_table_row("Balance Final", f"${prop_eval['final_balance']:,.2f}")
    print_table_row("Beneficio Neto ($)", f"${prop_eval['net_profit']:,.2f}")
    print_table_row("Pico Máximo de Capital", f"${prop_eval['peak_equity']:,.2f}")
    print_table_row("Piso Final de Trailing DD", f"${prop_eval['final_floor']:,.2f}")
    print_table_row("Sesiones RTH Evaluadas", prop_eval["total_days_evaluated"])

    # Imprimir métricas cuantitativas
    print_banner("MÉTRICAS FINANCIERAS Y DE RENDIMIENTO")
    print_table_row("Total de Operaciones", metrics["Total Trades"])
    print_table_row("Operaciones Ganadoras", metrics.get("Winning Trades", 0))
    print_table_row("Operaciones Perdedoras", metrics.get("Losing Trades", 0))
    print_table_row("Tasa de Acierto (Win Rate)", f"{metrics['Win Rate (%)']:.2f}%")
    print_table_row("Factor de Beneficio (Profit Factor)", metrics["Profit Factor"])
    print_table_row("Retorno Total Acumulado", f"{metrics['Total Return (%)']:.2f}%")
    print_table_row("Drawdown Máximo Flotante", f"${metrics['Max Drawdown ($)']:,.2f}")
    print_table_row("Ratio Sharpe Anualizado", f"{metrics['Sharpe Ratio']:.2f}")
    print_table_row("Ratio Sortino Anualizado", f"{metrics['Sortino Ratio']:.2f}")

    if "Exit Reasons" in metrics and metrics["Exit Reasons"]:
        print_banner("DISTRIBUCIÓN DE SALIDAS DE POSICIÓN")
        for reason, count in metrics["Exit Reasons"].items():
            print_table_row(f"Razón: {reason}", count)

    if not trades_df.empty:
        print_banner("DETALLE DE OPERACIONES EJECUTADAS")
        display_cols = ["trade_id", "entry_time", "signal", "contracts_nq", "contracts_es", "exit_reason", "friction_usd", "net_pnl_usd"]
        available_cols = [c for c in display_cols if c in trades_df.columns]
        print(trades_df[available_cols].to_string(index=False))
    elif metrics["Total Trades"] == 0:
        sl_sub = audit.get("latest_sub_basket_sl_usd", 529.34)
        budget_daily = args.dll * args.phi_dll
        budget_tmdd = args.mtdd * args.phi_tmdd
        eff_budget = min(budget_daily, budget_tmdd)
        print("\n" + "!" * 78)
        print("  AVISO DE CONTROL DE RIESGO (DISYUNTOR DE VOLATILIDAD ACTIVO - SEC 5.5)")
        print(f"  No se abrieron operaciones porque el riesgo monetario del Stop-Loss (${sl_sub:,.2f} USD)")
        print(f"  superó el presupuesto permitido por trade:")
        print(f"    - Presupuesto por Límite Diario   (phi_dll  * DLL)  : ${budget_daily:,.2f} USD")
        print(f"    - Presupuesto por Trailing DD     (phi_tmdd * TMDD) : ${budget_tmdd:,.2f} USD")
        print(f"    - Presupuesto Efectivo [min(Daily, TMDD)]           : ${eff_budget:,.2f} USD")
        print("  Para habilitar operaciones de forma segura según el capital de la cuenta:")
        print("    - En cuenta de $50k:  python run_cointegration_ou_backtest.py --account-tier 50k")
        print("                          (o: --phi-dll 0.60 --phi-tmdd 0.30)")
        print("    - En cuenta de $100k: python run_cointegration_ou_backtest.py --account-tier 100k")
        print("    - En cuenta de $150k: python run_cointegration_ou_backtest.py --account-tier 150k")
        print("!" * 78)

    # Exportar resultados
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not trades_df.empty:
        trades_df.to_csv(args.export_trades, index=False)
        print(f"\n[+] Registro de operaciones exportado a: {args.export_trades}")

    summary_data = {
        "strategy": "F4_Cointegracion_ES_NQ_OU",
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
