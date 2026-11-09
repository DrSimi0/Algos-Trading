"""
run_f3_optimization_n2_backtest.py
Script CLI ejecutable para correr el backtest formal de la versión optimizada N2:
F3-Optimizacion -IS- N2: Anchored VWAP & Rotational Value Area Reversion

Cumple con:
- Microestructura CME Globex (ES, MES, NQ, MNQ).
- Comisiones oficiales CME y deslizamiento realista (2 ticks bid-ask).
- Reglas estrictas de Proprietary Trading (Apex, Topstep, Lucid, MyFundedFutures):
  Daily Loss Limit (DLL), Trailing Drawdown flotante, conmutación Estándar / Micro.
- Innovaciones N2:
  1. Pendiente Normalizada de VWAP ultra-sensible (kappa_norm <= 0.085 para ES, 0.120 para NQ).
  2. Compuerta AMT de 6 barras (09:30 - 10:00 EST abren y cierran fuera de [VAL, VAH]).
  3. Bandas asimétricas moduladas por Volume Skewness.
  4. Trailing Protect dinámico al 60% de camino al Take-Profit.
  5. Disyuntor catastrófico a +-2.75 sigma.
  6. Corte Inmediato Intrabar ante proximidad a límites de Prop Trading (85% de DLL o MTDD).
- Motor de Simulación de Monte Carlo (2,500 iteraciones por Reshuffling y Bootstrapping)
  con validación formal de la Ley de Erdős-Rényi y cálculo de tasas de supervivencia.

Uso:
    python run_f3_optimization_n2_backtest.py --symbol ES --mode in_sample --run-mc
    python run_f3_optimization_n2_backtest.py --symbol ES --mode in_sample --compare-base --run-mc
    python run_f3_optimization_n2_backtest.py --symbol ES --mode in_sample --compare-n1 --run-mc
"""

import argparse
import sys
from pathlib import Path
import datetime
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.strategies.mean_reversion.f3_optimization_is_n2 import (
    F3OptimizationISN2Strategy,
    PropRiskManagerN2
)
from src.strategies.mean_reversion.f3_optimization_is_n1 import (
    F3OptimizationISN1Strategy
)
from src.strategies.mean_reversion.anchored_vwap_value_area import (
    AnchoredVWAPValueAreaStrategy
)
from src.core.futures_specs import get_spec
from src.core.data_split import add_data_split_args, partition_from_args


def load_dataset(symbol: str, data_path: str = None) -> pd.DataFrame:
    """Carga los datos de 5 minutos desde archivo local o ruta especificada."""
    sym = symbol.upper()
    if data_path:
        path = Path(data_path)
    else:
        candidates = [
            Path(f"data/raw/CME_MINI_{sym}_2021_2023_5m.parquet"),
            Path(f"data/raw/CME_MINI_{sym}_2021_2023_5m.csv"),
            Path(f"data/raw/CME_MINI_{sym}_5m_3y.parquet"),
            Path(f"data/raw/CME_MINI_{sym}_5m_3y.csv"),
            Path(f"data/raw/CME_MINI_{sym}_5m_bars5000.csv"),
            Path(f"data/raw/CME_MINI_{sym}_5m_bars5000.parquet"),
            Path(f"data/raw/CME_MINI_ES_2021_2023_5m.parquet") if "ES" in sym else Path(f"data/raw/CME_MINI_NQ_2021_2023_5m.parquet"),
            Path(f"data/raw/CME_MINI_ES_5m_3y.parquet") if "ES" in sym else Path(f"data/raw/CME_MINI_NQ_5m_3y.parquet"),
            Path(f"data/raw/CME_MINI_ES_5m_bars5000.csv") if "ES" in sym else Path(f"data/raw/CME_MINI_NQ_5m_bars5000.csv")
        ]
        path = None
        for cand in candidates:
            if cand.exists():
                path = cand
                break

    if path is None or not path.exists():
        raise FileNotFoundError(
            f"No se encontró archivo de datos para {sym}. Especifique --data-path explícitamente."
        )

    print(f"[*] Cargando dataset desde: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida: {col} en el dataset.")

    df = df[required].copy().sort_index()
    df.dropna(inplace=True)
    return df


def print_banner():
    banner = """
========================================================================================
         ESTRATEGIA F3-OPTIMIZACION -IS- N2: REINGENIERIA ANALITICA Y MONTE CARLO
    Fundamento: Steidlmayer (1986), Madhavan (2002), Gatheral (2010), Erdos-Renyi (1970)
    Innovaciones: Pendiente Normalizada (kappa <= 0.085), Compuerta AMT 6 Barras,
                  Trailing Protect al 60% y Corte Inmediato Intrabar de Limites Prop
    Marco de Riesgo: Proprietary Trading (Apex / Topstep / Lucid / MyFundedFutures)
========================================================================================
"""
    print(banner)


def compute_consecutive_losses(trades_pnl: np.ndarray) -> Tuple[int, float]:
    """Calcula la racha máxima observada de pérdidas consecutivas y la duración media."""
    if len(trades_pnl) == 0:
        return 0, 0.0

    is_loss = (trades_pnl < 0).astype(int)
    streaks = []
    current_streak = 0

    for val in is_loss:
        if val == 1:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)

    max_streak = max(streaks) if len(streaks) > 0 else 0
    avg_streak = float(np.mean(streaks)) if len(streaks) > 0 else 0.0
    return max_streak, avg_streak


def run_monte_carlo_engine(
    trades_pnl: np.ndarray,
    account_size: float = 50000.0,
    profit_target: float = 3000.0,
    mtdd_limits: List[float] = [2000.0, 2500.0, 3000.0],
    n_iterations: int = 2500,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Ejecuta el motor de simulación de Monte Carlo bajo dos modalidades:
      1. Reshuffling (Permutación sin reemplazo - secuencia de trades).
      2. Bootstrapping (Remuestreo con reemplazo).
    
    Evalúa drawdown mediano, P95, CVaR, riesgo de ruina y tasa de aprobación limpia.
    """
    np.random.seed(seed)
    n_trades = len(trades_pnl)
    if n_trades < 10:
        return {}

    n_losses = int(np.sum(trades_pnl < 0))
    q_rate = n_losses / n_trades
    erdos_renyi_expected = (np.log(n_trades) / np.log(1.0 / q_rate)) if (0 < q_rate < 1.0) else 0.0
    obs_max_streak, obs_avg_streak = compute_consecutive_losses(trades_pnl)

    results = {
        "N": n_trades,
        "n_losses": n_losses,
        "q_rate": q_rate,
        "erdos_renyi_expected": erdos_renyi_expected,
        "obs_max_streak": obs_max_streak,
        "obs_avg_streak": obs_avg_streak,
        "modes": {}
    }

    modes = ["Reshuffling (Permutación)", "Bootstrapping (Reemplazo)"]
    for mode in modes:
        max_dds = np.zeros(n_iterations, dtype=np.float64)
        ruin_counts = {lim: 0 for lim in mtdd_limits}
        clean_pass_counts = {lim: 0 for lim in mtdd_limits}
        max_streaks = np.zeros(n_iterations, dtype=int)

        for it in range(n_iterations):
            if "Permutación" in mode:
                perm_pnl = np.random.permutation(trades_pnl)
            else:
                perm_pnl = np.random.choice(trades_pnl, size=n_trades, replace=True)

            it_max_s, _ = compute_consecutive_losses(perm_pnl)
            max_streaks[it] = it_max_s

            cum_equity = account_size + np.cumsum(perm_pnl)
            all_equity = np.insert(cum_equity, 0, account_size)
            peak_equity = np.maximum.accumulate(all_equity)
            dd_curve = peak_equity - all_equity
            it_max_dd = np.max(dd_curve)
            max_dds[it] = it_max_dd

            # Evaluar aprobación limpia vs ruina
            for lim in mtdd_limits:
                # Buscar primer momento de cruce
                breach_idx = np.where(dd_curve >= lim)[0]
                target_idx = np.where((all_equity - account_size) >= profit_target)[0]

                has_breach = len(breach_idx) > 0
                has_target = len(target_idx) > 0

                if has_breach:
                    ruin_counts[lim] += 1

                if has_target and not has_breach:
                    clean_pass_counts[lim] += 1
                elif has_target and has_breach:
                    if target_idx[0] < breach_idx[0]:
                        clean_pass_counts[lim] += 1

        p50 = float(np.percentile(max_dds, 50))
        p90 = float(np.percentile(max_dds, 90))
        p95 = float(np.percentile(max_dds, 95))
        p99 = float(np.percentile(max_dds, 99))
        mean_dd = float(np.mean(max_dds))
        worst_5_pct = max_dds[max_dds >= p95]
        cvar_95 = float(np.mean(worst_5_pct)) if len(worst_5_pct) > 0 else p95

        mode_stats = {
            "mean_dd": mean_dd,
            "median_dd": p50,
            "p90_dd": p90,
            "p95_dd": p95,
            "p99_dd": p99,
            "cvar_95": cvar_95,
            "worst_dd": float(np.max(max_dds)),
            "avg_max_streak": float(np.mean(max_streaks)),
            "ruin_rates": {lim: (ruin_counts[lim] / n_iterations) * 100.0 for lim in mtdd_limits},
            "survival_rates": {lim: (1.0 - ruin_counts[lim] / n_iterations) * 100.0 for lim in mtdd_limits},
            "clean_pass_rates": {lim: (clean_pass_counts[lim] / n_iterations) * 100.0 for lim in mtdd_limits}
        }
        results["modes"][mode] = mode_stats

    return results


def print_monte_carlo_table(mc_results: Dict[str, Any], title: str = "AUDITORÍA ESTOCÁSTICA DE MONTE CARLO"):
    """Imprime el informe detallado de Monte Carlo y el contraste con Erdős-Rényi."""
    if not mc_results:
        print("[!] No hay suficientes datos para simulación de Monte Carlo.")
        return

    print("\n" + "=" * 90)
    print(f" {title:^88} ")
    print("=" * 90)
    print(f"  Muestra de Operaciones (N)              : {mc_results['N']} trades")
    print(f"  Operaciones Perdedoras (n_losses)       : {mc_results['n_losses']} trades")
    print(f"  Tasa de Pérdida Basal (q)               : {mc_results['q_rate'] * 100.0:.2f}%")
    print(f"  Racha Máxima Observada en Calendario    : {mc_results['obs_max_streak']} pérdidas consecutivas")
    print(f"  Racha Esperada Teorema Erdos-Renyi     : {mc_results['erdos_renyi_expected']:.1f} pérdidas consecutivas (E[L_max] = ln(N)/ln(1/q))")
    print("-" * 90)

    for mode_name, stats in mc_results["modes"].items():
        print(f"\n  >>> MODALIDAD: {mode_name.upper()} (2,500 ITERACIONES)")
        print(f"  {'Drawdown Medio':<35} : ${stats['mean_dd']:,.2f} USD")
        print(f"  {'Drawdown Mediano (P50)':<35} : ${stats['median_dd']:,.2f} USD")
        print(f"  {'Drawdown P90':<35} : ${stats['p90_dd']:,.2f} USD")
        print(f"  {'Drawdown en Riesgo (P95)':<35} : ${stats['p95_dd']:,.2f} USD")
        print(f"  {'Drawdown Extremo (P99)':<35} : ${stats['p99_dd']:,.2f} USD")
        print(f"  {'CVaR 95% (Media Peor 5%)':<35} : ${stats['cvar_95']:,.2f} USD")
        print(f"  {'Peor Drawdown Absoluto Simulado':<35} : ${stats['worst_dd']:,.2f} USD")
        print(f"  {'Racha Media de Pérdidas Simulada':<35} : {stats['avg_max_streak']:.1f} pérdidas seguidas")
        print("  " + "-" * 75)
        print(f"  {'Límite Trailing DD':<22} | {'Riesgo Ruina':<16} | {'Supervivencia':<16} | {'Aprobación Limpia':<16}")
        print("  " + "-" * 75)
        for lim in [2000.0, 2500.0, 3000.0]:
            r_rate = stats["ruin_rates"][lim]
            s_rate = stats["survival_rates"][lim]
            cp_rate = stats["clean_pass_rates"][lim]
            status_tag = "[APROBADO]" if s_rate >= 95.0 else ("[PRECAUCIÓN]" if s_rate >= 90.0 else "[INVIABLE]")
            print(f"  MTDD ${lim:>6,.0f} USD       | {r_rate:>13.2f}% | {s_rate:>13.2f}% | {cp_rate:>14.2f}%  {status_tag}")
        print("  " + "-" * 75)
    print("=" * 90 + "\n")


def print_metrics_table(metrics: dict, title: str = "RESULTADOS DEL BACKTEST"):
    print("\n" + "=" * 68)
    print(f" {title:^66} ")
    print("=" * 68)
    for key, val in metrics.items():
        if key == "Desglose de Salidas":
            print(f"  {key:<34} :")
            if isinstance(val, dict) and val:
                for r_name, r_count in val.items():
                    print(f"    - {r_name:<30} : {r_count}")
            else:
                print("    - Ninguna")
        else:
            print(f"  {key:<34} : {val}")
    print("=" * 68 + "\n")


def print_comparison_table(metrics_base: dict, metrics_opt: dict, label_base: str, label_opt: str, symbol: str):
    print("\n" + "=" * 84)
    print(f" {'COMPARATIVA CUANTITATIVA: ' + label_base + ' vs ' + label_opt + ' (' + symbol + ')':^82} ")
    print("=" * 84)
    header = f"  {'Métrica':<32} | {label_base:<18} | {label_opt:<18} | {'Delta':<10}"
    print(header)
    print("-" * 84)

    keys = [
        ("P&L Neto Total ($)", "$", True),
        ("Retorno Total (%)", "%", True),
        ("Max Trailing Drawdown ($)", "$", False),
        ("Max Trailing Drawdown (%)", "%", False),
        ("Total Trades", "", False),
        ("Win Rate (%)", "%", True),
        ("Profit Factor", "", True),
        ("Trade Promedio ($)", "$", True)
    ]

    for k, unit, _ in keys:
        v_base = metrics_base.get(k, "N/A")
        v_opt = metrics_opt.get(k, "N/A")
        if isinstance(v_base, (int, float)) and isinstance(v_opt, (int, float)):
            diff = v_opt - v_base
            diff_str = f"{diff:+.2f}{unit}"
            print(f"  {k:<32} | {v_base:>15.2f}{unit} | {v_opt:>15.2f}{unit} | {diff_str:>10}")
        else:
            print(f"  {k:<32} | {str(v_base):>16} | {str(v_opt):>16} | {'-':>10}")
    print("=" * 84 + "\n")


def print_trades_table(df_trades: pd.DataFrame, max_trades: int = 15):
    if df_trades.empty:
        print("[*] No se ejecutaron operaciones en el período.")
        return

    print("\n" + "=" * 120)
    print(f" {'REGISTRO DE OPERACIONES AUDITADAS (ÚLTIMAS ' + str(min(len(df_trades), max_trades)) + ')':^118} ")
    print("=" * 120)
    headers = f"  {'Entrada':<16} | {'Salida':<16} | {'Tipo':<5} | {'Cnt':<3} | {'Precio In':<10} | {'Precio Out':<10} | {'Net PnL ($)':<12} | {'Razón Salida':<26}"
    print(headers)
    print("-" * 120)

    tail_trades = df_trades.tail(max_trades)
    for _, tr in tail_trades.iterrows():
        t_in = str(tr['entry_time'])[:16]
        t_out = str(tr['exit_time'])[:16]
        d = tr['direction']
        cnt = f"{tr['contracts']} {tr['contract_type'][:3]}"
        p_in = f"{tr['entry_price']:.2f}"
        p_out = f"{tr['exit_price']:.2f}"
        net = f"${tr['net_pnl']:+,.2f}"
        reason = str(tr['exit_reason'])
        print(f"  {t_in:<16} | {t_out:<16} | {d:<5} | {cnt:<7} | {p_in:<10} | {p_out:<10} | {net:<12} | {reason:<26}")
    print("=" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Cuantitativo Formal - Estrategia F3-Optimizacion -IS- N2 con Simulación de Monte Carlo"
    )
    parser.add_argument("--symbol", type=str, default="ES", choices=["ES", "MES", "NQ", "MNQ"],
                        help="Símbolo del contrato CME Globex (ES, MES, NQ, MNQ)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Ruta al archivo CSV o Parquet con velas de 5 minutos")
    parser.add_argument("--account-size", type=float, default=50000.0,
                        help="Tamaño de cuenta de Prop Trading en USD (ej. 50,000)")
    parser.add_argument("--dll", type=float, default=1000.0,
                        help="Daily Loss Limit (DLL) en USD")
    parser.add_argument("--mtdd", type=float, default=2000.0,
                        help="Maximum Trailing Drawdown (MTDD) en USD")
    parser.add_argument("--k-band", type=float, default=None,
                        help="Múltiplo de desviación estándar para entrada (2.20 para ES, 2.5 para NQ)")
    parser.add_argument("--kappa-crit", type=float, default=None,
                        help="Umbral crítico de pendiente normalizada de VWAP (0.085 para ES, 0.120 para NQ)")
    parser.add_argument("--convergence-protect-ratio", type=float, default=0.50,
                        help="Ratio de convergencia hacia VWAP para proteger (0.50 calibrado [default], 0.60 base)")
    parser.add_argument("--scaling-factor", type=float, default=0.65,
                        help="Factor de escala global de dimensionamiento (0.65 = Conservador MTDD $2,000 [default], 0.75 = MTDD $2,500, 1.0 = MTDD $3,000)")
    parser.add_argument("--anchor-time", type=str, default="09:30",
                        help="Hora de anclaje de la sesión regular (HH:MM)")
    parser.add_argument("--run-mc", action="store_true",
                        help="Ejecutar simulación estocástica de Monte Carlo completa")
    parser.add_argument("--mc-iterations", type=int, default=2500,
                        help="Número de iteraciones de Monte Carlo (por defecto 2,500)")
    parser.add_argument("--compare-base", action="store_true",
                        help="Ejecutar y contrastar contra la estrategia base F3")
    parser.add_argument("--compare-n1", action="store_true",
                        help="Ejecutar y contrastar contra la versión optimizada F3 N1")
    parser.add_argument("--print-trades", action="store_true",
                        help="Imprimir tabla detallada de operaciones ejecutadas")
    parser.add_argument("--output-csv", type=str, default="reports/f3_optimization_n2_trades.csv",
                        help="Ruta para exportar el log de operaciones a CSV (por defecto: reports/f3_optimization_n2_trades.csv)")
    add_data_split_args(parser)

    args = parser.parse_args()

    print_banner()

    # 1. Cargar datos
    data = load_dataset(symbol=args.symbol, data_path=args.data_path)
    data, part_label = partition_from_args(data, args, "F3_Optimization_IS_N2")
    print(f"[*] Rango de Datos: {data.index[0]} a {data.index[-1]} ({len(data):,} barras de 5m)")

    # 2. Configurar hiperparámetros de N2
    k_band = args.k_band
    if k_band is None:
        k_band = 2.5 if args.symbol.upper() in ["NQ", "MNQ"] else 2.20

    kappa_crit = args.kappa_crit
    if kappa_crit is None:
        kappa_crit = 0.120 if args.symbol.upper() in ["NQ", "MNQ"] else 0.085

    n2_params = {
        "symbol": args.symbol,
        "anchor_time": args.anchor_time,
        "window_init_end": "10:15",
        "window_revers_end": "15:00",
        "window_inhibit_end": "15:15",
        "k_band": k_band,
        "k_catastrophic": 2.75,
        "kappa_crit": kappa_crit,
        "slope_lookback_bars": 3,
        "max_vol_outside_ratio": 0.55,
        "min_kurtosis": 1.7,
        "min_dist_to_vwap_ratio": 0.80,
        "k_buffer_ticks": 3,
        "convergence_protect_ratio": args.convergence_protect_ratio,
        "account_size": args.account_size,
        "daily_loss_limit": args.dll,
        "max_trailing_drawdown": args.mtdd,
        "psi": 0.12,
        "max_std_contracts": 1,
        "max_micro_contracts": 8,
        "alpha_dll": 0.85,
        "max_intraday_consecutive_losses": 2,
        "scaling_factor": args.scaling_factor,
        "bar_minutes": 5.0
    }

    n2_strategy = F3OptimizationISN2Strategy(params=n2_params)

    # 3. Ejecutar Backtest de F3 N2
    print(f"[*] Ejecutando backtest de F3-Optimizacion -IS- N2 con Prop Risk Management...")
    results_n2 = n2_strategy.run_backtest(data)
    metrics_n2 = results_n2["metrics"]
    trades_n2_df = results_n2["trades"]

    print_metrics_table(metrics_n2, title=f"RESULTADOS F3-OPTIMIZACION -IS- N2 - {args.symbol.upper()} (5m)")

    # 4. Comparativa con F3 Base
    if args.compare_base:
        print("[*] Ejecutando backtest de Estrategia Base F3 para comparación...")
        base_params = {
            "symbol": args.symbol,
            "anchor_time": args.anchor_time,
            "k_band": k_band,
            "k_catastrophic": 3.0,
            "delta_max_slope": 0.45 if args.symbol.upper() in ["NQ", "MNQ"] else 0.15,
            "adx_threshold": 25.0,
            "account_size": args.account_size,
            "daily_loss_limit": args.dll,
            "max_trailing_drawdown": args.mtdd,
            "psi": 0.20,
            "k_buffer_ticks": 4,
            "bar_minutes": 5.0
        }
        base_strategy = AnchoredVWAPValueAreaStrategy(params=base_params)
        results_base = base_strategy.run_backtest(data)
        metrics_base = results_base["metrics"]
        print_comparison_table(metrics_base, metrics_n2, "F3 Base", "F3-N2 Optimizada", symbol=args.symbol.upper())

    # 5. Comparativa con F3 N1
    if args.compare_n1:
        print("[*] Ejecutando backtest de F3-Optimizacion -IS- N1 para comparación...")
        n1_params = {
            "symbol": args.symbol,
            "anchor_time": args.anchor_time,
            "k_band": k_band,
            "k_catastrophic": 2.75,
            "kappa_crit": 2.2,
            "account_size": args.account_size,
            "daily_loss_limit": args.dll,
            "max_trailing_drawdown": args.mtdd,
            "psi": 0.20,
            "k_buffer_ticks": 3,
            "bar_minutes": 5.0
        }
        n1_strategy = F3OptimizationISN1Strategy(params=n1_params)
        results_n1 = n1_strategy.run_backtest(data)
        metrics_n1 = results_n1["metrics"]
        print_comparison_table(metrics_n1, metrics_n2, "F3-N1 Opt", "F3-N2 Optimizada", symbol=args.symbol.upper())

    # 6. Simulación de Monte Carlo
    if args.run_mc:
        print(f"\n[*] Ejecutando simulación de Monte Carlo ({args.mc_iterations:,} iteraciones)...")
        trades_pnl = trades_n2_df["net_pnl"].values if not trades_n2_df.empty else np.array([])
        mc_results = run_monte_carlo_engine(
            trades_pnl=trades_pnl,
            account_size=args.account_size,
            profit_target=3000.0,
            mtdd_limits=[2000.0, 2500.0, 3000.0],
            n_iterations=args.mc_iterations,
            seed=42
        )
        print_monte_carlo_table(mc_results, title=f"MONTE CARLO (2,500 ITERACIONES) - F3-OPTIMIZACION -IS- N2 ({args.symbol.upper()})")

    if args.print_trades or len(trades_n2_df) > 0:
        print_trades_table(trades_n2_df)

    if args.output_csv:
        out_p = Path(args.output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        trades_n2_df.to_csv(out_p, index=False)
        print(f"[*] Registro de operaciones exportado exitosamente a: {out_p.resolve()}")

    trailing_dd = metrics_n2["Max Trailing Drawdown ($)"]
    if trailing_dd > args.mtdd:
        print(f"\n[!] ALERTA: Trailing Drawdown (${trailing_dd:,.2f}) excedió el MTDD (${args.mtdd:,.2f}).")
    else:
        print(f"\n[OK] Cumplimiento de Riesgo Prop Trading: Trailing DD dentro del límite admisible (${trailing_dd:,.2f} <= ${args.mtdd:,.2f}).")


if __name__ == "__main__":
    main()
