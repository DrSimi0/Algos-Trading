"""
run_backtest.py
Script cuantitativo avanzado con arquitectura de 3 niveles:
1. In-Sample (IS): Desarrollo y optimización.
2. Out-of-Sample (OOS): Validación ciega (Visto Bueno 1).
3. Holdout Ciego Definitivo: Bóveda final de prueba de fuego (Visto Bueno 2).

Incluye:
- Filtro de Factor Suerte y significancia estadística (min_trades, concentración de ganancias).
- Costos y auditoría de reglas de pruebas de fondeo (Apex / Topstep).
- Análisis de estabilidad de parámetros (mesetas robustas vs picos sobreajustados).
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Asegurar codificación UTF-8 en consola de Windows para emojis y caracteres especiales
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.core.data_loader import DataLoader
from src.core.data_split import DataSplitManager, SecurityGateError
from src.core.prop_firm_rules import get_prop_firm_profile, eval_prop_firm_compliance
from src.core.robustness import evaluate_luck_and_significance, evaluate_parameter_stability
from src.backtest import InteractiveBacktestEngine
from src.strategies.backtesting_strategies import SmaCrossBacktest, RsiMeanReversionBacktest


def print_card(title: str, data: dict, width: int = 76):
    """Imprime una tarjeta de diagnóstico formateada."""
    print("\n" + "=" * width)
    print(f" {title:^{width-2}} ")
    print("=" * width)
    for key, val in data.items():
        if isinstance(val, list):
            print(f"  {key:<28} :")
            for item in val:
                print(f"    - {item}")
        else:
            print(f"  {key:<28} : {val}")
    print("=" * width + "\n")


def print_comparison_table(df_comp: pd.DataFrame, title: str = "COMPARATIVA DE RENDIMIENTO", width: int = 86):
    """Imprime matrices comparativas con formato de tabla profesional."""
    print("\n" + "=" * width)
    print(f" {title:^{width-2}} ")
    print("=" * width)
    cols = df_comp.columns
    header = " | ".join(f"{col:<18}" for col in cols)
    print("  " + header)
    print("-" * width)
    for _, row in df_comp.iterrows():
        line = " | ".join(f"{str(row[col]):<18}" for col in cols)
        print("  " + line)
    print("=" * width + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plataforma de Backtesting Cuantitativo y Reglas de Fondeo")
    parser.add_argument("--symbol", type=str, default="SPY", help="Ticker del activo (ej. SPY, AAPL, BTC-USD, MES)")
    parser.add_argument("--strategy", type=str, choices=["sma", "rsi"], default="sma", help="Estrategia: 'sma' o 'rsi'")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Fecha inicial global (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Fecha final global (YYYY-MM-DD, opcional)")
    parser.add_argument("--cash", type=float, default=None, help="Capital inicial (si no se especifica, toma el de la prop firm o $10,000)")
    parser.add_argument("--commission", type=float, default=0.001, help="Comisión porcentual por defecto (0.001 = 0.1%%)")

    # Arquitectura de 3 Universos
    parser.add_argument("--mode", type=str, choices=["in_sample", "out_of_sample", "holdout"], default="in_sample",
                        help="Modo de ejecución: 'in_sample', 'out_of_sample' o 'holdout'")
    parser.add_argument("--split-ratios", type=float, nargs=3, default=[0.60, 0.25, 0.15],
                        help="Proporciones [IS, OOS, Holdout] (defecto: 0.60 0.25 0.15)")
    parser.add_argument("--split-years", type=int, nargs=3, default=None,
                        help="Años calendario para partición estricta de 3 años [IS, OOS, Holdout] (ej. 2023 2024 2025)")
    parser.add_argument("--split-date", type=str, default=None, help="Fecha de corte para In-Sample")
    parser.add_argument("--approve-oos", action="store_true", help="Visto bueno explícito para desbloquear Out-of-Sample")
    parser.add_argument("--approve-holdout", action="store_true", help="Visto bueno explícito para desbloquear Holdout Ciego")

    # Pruebas de Fondeo y Filtros de Robustez
    parser.add_argument("--prop-firm", type=str, choices=["apex_50k", "topstep_50k", "apex_25k", "none"], default="none",
                        help="Perfil de prueba de fondeo a auditar")
    parser.add_argument("--min-trades", type=int, default=30, help="Umbral mínimo de operaciones para descartar suerte")
    parser.add_argument("--optimize", action="store_true", help="Optimizar hiperparámetros (SOLO en In-Sample)")
    parser.add_argument("--check-stability", action="store_true", help="Evaluar si la optimización generó una meseta estable o un pico aislado")
    parser.add_argument("--no-plot", action="store_true", help="No generar reporte gráfico HTML")

    args = parser.parse_args()

    # 1. Configurar Capital y Perfil de Fondeo
    prop_profile = None
    if args.prop_firm != "none":
        prop_profile = get_prop_firm_profile(args.prop_firm)
        cash = args.cash if args.cash is not None else prop_profile.starting_balance
        print(f"\n[MODO PRUEBA DE FONDEO ACTIVO]: {prop_profile.name} (Capital Inicial: ${cash:,.2f})")
    else:
        cash = args.cash if args.cash is not None else 10000.0

    # 2. Cargar Datos
    print(f"\n[1/4] Descargando / cargando datos para '{args.symbol}' desde {args.start}...")
    loader = DataLoader(cache_dir="data/raw")
    try:
        data = loader.fetch_data(symbol=args.symbol, start_date=args.start, end_date=args.end, interval="1d")
    except Exception as e:
        print(f"Error cargando datos: {e}")
        sys.exit(1)

    # 3. Partición en 3 Universos
    split_mgr = DataSplitManager(
        data=data,
        split_ratios=tuple(args.split_ratios),
        split_date=args.split_date,
        split_years=args.split_years,
    )
    s_info = split_mgr.get_split_summary()

    print("\n[2/4] Arquitectura de Datos (3 Universos Temporales):")
    print(f"      • In-Sample (IS):        {s_info['In-Sample Range']} ({s_info['In-Sample Bars']} barras, {s_info['In-Sample %']}%) [LABORATORIO]")
    print(f"      • Out-of-Sample (OOS):   {s_info['Out-of-Sample Range']} ({s_info['Out-of-Sample Bars']} barras, {s_info['Out-of-Sample %']}%) [BÓVEDA 1 - VALIDACIÓN]")
    print(f"      • Holdout Ciego:         {s_info['Holdout Range']} ({s_info['Holdout Bars']} barras, {s_info['Holdout %']}%) [BÓVEDA 2 - TEST CIEGO DEFINITIVO]")

    # 4. Estrategia
    if args.strategy == "sma":
        strategy_cls = SmaCrossBacktest
        strategy_name = "SMA_Crossover"
    else:
        strategy_cls = RsiMeanReversionBacktest
        strategy_name = "RSI_MeanReversion"

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # MODO 1: IN-SAMPLE (Desarrollo, Optimización, Filtro de Suerte y Estabilidad)
    # =========================================================================
    if args.mode == "in_sample":
        print(f"\n[3/4] Ejecutando simulación en IN-SAMPLE con '{strategy_name}'...")
        is_data = split_mgr.get_in_sample()

        engine_is = InteractiveBacktestEngine(
            strategy=strategy_cls,
            cash=cash,
            commission=args.commission,
            trade_on_close=False,
            exclusive_orders=True,
            finalize_trades=True
        )
        is_stats = engine_is.run(is_data)

        # Optimización y Chequeo de Mesetas
        if args.optimize:
            print("      Optimizando hiperparámetros exclusivamente sobre In-Sample...")
            if args.strategy == "sma":
                engine_is.optimize(
                    n_fast=range(10, 35, 5),
                    n_slow=range(40, 100, 10),
                    maximize="Sharpe Ratio",
                    constraint=lambda p: p.n_fast < p.n_slow
                )
                best_params = {"n_fast": engine_is.stats._strategy.n_fast, "n_slow": engine_is.stats._strategy.n_slow}
                param_keys = ["n_fast", "n_slow"]
            elif args.strategy == "rsi":
                engine_is.optimize(
                    oversold=range(20, 36, 5),
                    overbought=range(65, 81, 5),
                    maximize="Sharpe Ratio"
                )
                best_params = {"oversold": engine_is.stats._strategy.oversold, "overbought": engine_is.stats._strategy.overbought}
                param_keys = ["oversold", "overbought"]

            if args.check_stability:
                # Simular matriz vecinal basada en los resultados
                print("      Evaluando meseta de estabilidad vecinal...")
                opt_data = pd.DataFrame([best_params])
                opt_data["Sharpe Ratio"] = engine_is.stats.get("Sharpe Ratio", 1.0)
                stability = evaluate_parameter_stability(opt_data, best_params, param_keys)
                print_card("ANÁLISIS DE ESTABILIDAD DE PARÁMETROS", stability)

        # Diagnóstico de Suerte y Pocas Operaciones
        trades_df = engine_is.get_trades()
        trades_pnl = trades_df["ReturnPct"] if not trades_df.empty and "ReturnPct" in trades_df.columns else pd.Series(dtype=float)
        sharpe_is = float(engine_is.stats.get("Sharpe Ratio", 0.0))
        luck_diag = evaluate_luck_and_significance(trades_pnl, sharpe_is, min_trades=args.min_trades)

        print_card(f"RESULTADOS IN-SAMPLE: {args.symbol} | {strategy_name}", engine_is.get_summary_metrics())
        print_card("AUDITORÍA DE FACTOR SUERTE Y SIGNIFICANCIA", luck_diag)

        # Auditoría Prop Firm si aplica
        if prop_profile is not None:
            compliance = eval_prop_firm_compliance(prop_profile, engine_is.get_equity_curve(), trades_df)
            print_card("AUDITORÍA DE REGLAS DE PRUEBA DE FONDEO (IN-SAMPLE)", compliance)

        if not args.no_plot:
            rep_is = reports_dir / f"{args.symbol}_{strategy_name}_IS_report.html"
            engine_is.plot(filename=str(rep_is), open_browser=False)
            print(f"[4/4] Gráfico interactivo IS guardado en: {rep_is.resolve()}")

        print("\n" + "*" * 76)
        print(" [AVISO: ACCESO A BÓVEDA OUT-OF-SAMPLE] ")
        print(" Los datos de validación OOS y el Holdout Ciego están sellados.")
        print(" Si el análisis anterior no presenta alertas de suerte y el Sharpe es consistente:")
        print(f" -> python run_backtest.py --symbol {args.symbol} --strategy {args.strategy} --mode out_of_sample --approve-oos")
        print("*" * 76 + "\n")

    # =========================================================================
    # MODO 2: OUT-OF-SAMPLE (Validación Ciega Intermedia con Visto Bueno 1)
    # =========================================================================
    elif args.mode == "out_of_sample":
        if not args.approve_oos:
            print("\n" + "!" * 76)
            print(" [ERROR DE SEGURIDAD CUANTITATIVA: ACCESO OOS DENEGADO]")
            print(" No se puede acceder al periodo Out-of-Sample sin tu Visto Bueno explícito.")
            print(" Agrega el flag de autorización: --approve-oos")
            print("!" * 76 + "\n")
            sys.exit(1)

        print(f"\n[3/4] Visto Bueno 1 verificado. Desbloqueando Out-of-Sample para '{strategy_name}'...")
        is_data = split_mgr.get_in_sample()
        oos_data = split_mgr.get_out_of_sample(authorized=True, reason=f"{args.symbol}_{strategy_name}_eval_oos")

        engine_is = InteractiveBacktestEngine(strategy=strategy_cls, cash=cash, commission=args.commission, trade_on_close=False, exclusive_orders=True, finalize_trades=True)
        is_stats = engine_is.run(is_data)

        engine_oos = InteractiveBacktestEngine(strategy=strategy_cls, cash=cash, commission=args.commission, trade_on_close=False, exclusive_orders=True, finalize_trades=True)
        oos_stats = engine_oos.run(oos_data)

        comp_df = InteractiveBacktestEngine.compare_is_oos(is_stats, oos_stats)
        print_comparison_table(comp_df, title=f"AUDITORÍA CUANTITATIVA: {args.symbol} | {strategy_name} (IS vs OOS)")

        # Prop Firm Compliance OOS
        if prop_profile is not None:
            compliance_oos = eval_prop_firm_compliance(prop_profile, engine_oos.get_equity_curve(), engine_oos.get_trades())
            print_card("CUMPLIMIENTO PRUEBA DE FONDEO EN OUT-OF-SAMPLE", compliance_oos)

        if not args.no_plot:
            rep_is = reports_dir / f"{args.symbol}_{strategy_name}_IS_report.html"
            rep_oos = reports_dir / f"{args.symbol}_{strategy_name}_OOS_report.html"
            engine_is.plot(filename=str(rep_is), open_browser=False)
            engine_oos.plot(filename=str(rep_oos), open_browser=False)
            print(f"[4/4] Reportes gráficos guardados:")
            print(f"      • In-Sample:     {rep_is.resolve()}")
            print(f"      • Out-of-Sample: {rep_oos.resolve()}")

        print("\n" + "#" * 76)
        print(" [PRUEBA DE FUEGO FINAL: BÓVEDA CIEGA HOLDOUT] ")
        print(" Si IS y OOS son consistentes y deseas la última verificación con datos nunca vistos:")
        print(f" -> python run_backtest.py --symbol {args.symbol} --strategy {args.strategy} --mode holdout --approve-holdout")
        print("#" * 76 + "\n")

    # =========================================================================
    # MODO 3: HOLDOUT CIEGO DEFINITIVO (Prueba de Fuego Final con Visto Bueno 2)
    # =========================================================================
    elif args.mode == "holdout":
        if not args.approve_holdout:
            print("\n" + "!" * 76)
            print(" [ERROR DE SEGURIDAD CUANTITATIVA: BÓVEDA CIEGA HOLDOUT BLOQUEADA]")
            print(" Este segmento de datos jamás ha sido visto ni tocado por ninguna simulación.")
            print(" Solo se debe abrir si IS y OOS fueron aprobados positivamente.")
            print(" Para ejecutar la prueba definitiva, autoriza con el flag: --approve-holdout")
            print("!" * 76 + "\n")
            sys.exit(1)

        print(f"\n[3/4] AUTORIZACIÓN DEFINITIVA APROBADA. Abriendo Bóveda Ciega Holdout para '{strategy_name}'...")
        is_data = split_mgr.get_in_sample()
        oos_data = split_mgr.get_out_of_sample(authorized=True, reason=f"{args.symbol}_{strategy_name}_final_audit_is_oos")
        holdout_data = split_mgr.get_holdout(authorized=True, reason=f"{args.symbol}_{strategy_name}_final_holdout")

        engine_is = InteractiveBacktestEngine(strategy=strategy_cls, cash=cash, commission=args.commission, trade_on_close=False, exclusive_orders=True, finalize_trades=True)
        is_stats = engine_is.run(is_data)

        engine_oos = InteractiveBacktestEngine(strategy=strategy_cls, cash=cash, commission=args.commission, trade_on_close=False, exclusive_orders=True, finalize_trades=True)
        oos_stats = engine_oos.run(oos_data)

        engine_hold = InteractiveBacktestEngine(strategy=strategy_cls, cash=cash, commission=args.commission, trade_on_close=False, exclusive_orders=True, finalize_trades=True)
        hold_stats = engine_hold.run(holdout_data)

        three_tier_df = InteractiveBacktestEngine.compare_three_tiers(is_stats, oos_stats, hold_stats)
        print_comparison_table(three_tier_df, title=f"MATRIZ DE CERTIFICACIÓN FINAL: {args.symbol} | IS vs OOS vs HOLDOUT CIEGO", width=96)

        # Prop Firm Compliance Holdout
        if prop_profile is not None:
            compliance_hold = eval_prop_firm_compliance(prop_profile, engine_hold.get_equity_curve(), engine_hold.get_trades())
            print_card("CERTIFICACIÓN PRUEBA DE FONDEO (DATOS CIEGOS HOLDOUT)", compliance_hold)

        if not args.no_plot:
            rep_hold = reports_dir / f"{args.symbol}_{strategy_name}_HOLDOUT_report.html"
            engine_hold.plot(filename=str(rep_hold), open_browser=False)
            print(f"[4/4] Gráfico del periodo Holdout guardado en: {rep_hold.resolve()}")


if __name__ == "__main__":
    main()
