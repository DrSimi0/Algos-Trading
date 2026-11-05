"""
src/core/prop_firm_rules.py
Modelado de costos, restricciones y reglas de auditoría para Pruebas de Fondeo (Prop Trading Firms).
Compatible con empresas líderes de futuros (Apex Trader Funding, Topstep, MyFundedFutures, etc.).
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd


@dataclass
class PropFirmProfile:
    """Perfil de reglas y costos para cuentas de fondeo."""
    name: str
    starting_balance: float
    profit_target: float
    max_trailing_drawdown: float
    daily_loss_limit: Optional[float] = None
    consistency_rule_pct: Optional[float] = 30.0  # Ningún día > 30% de las ganancias totales
    commission_rt: float = 1.24                  # Dólares por contrato ida y vuelta (Micro CME)
    slippage_ticks: int = 1                       # Deslizamiento mínimo en órdenes a mercado


PROFILES: Dict[str, PropFirmProfile] = {
    "apex_50k": PropFirmProfile(
        name="Apex 50k Full Trailing",
        starting_balance=50000.0,
        profit_target=3000.0,
        max_trailing_drawdown=2500.0,
        daily_loss_limit=None,            # Apex no tiene Daily Loss Limit formal
        consistency_rule_pct=30.0,        # Regla de consistencia del 30% para retiros
        commission_rt=1.24,               # Micro futuros MES / MNQ
        slippage_ticks=1
    ),
    "topstep_50k": PropFirmProfile(
        name="Topstep 50k Express",
        starting_balance=50000.0,
        profit_target=3000.0,
        max_trailing_drawdown=2000.0,     # Max Loss limit
        daily_loss_limit=1000.0,          # Límite estricto de pérdida diaria
        consistency_rule_pct=50.0,        # Regla de consistencia del 50%
        commission_rt=1.24,
        slippage_ticks=1
    ),
    "apex_25k": PropFirmProfile(
        name="Apex 25k Micro",
        starting_balance=25000.0,
        profit_target=1500.0,
        max_trailing_drawdown=1500.0,
        daily_loss_limit=None,
        consistency_rule_pct=30.0,
        commission_rt=1.24,
        slippage_ticks=1
    ),
}


def get_prop_firm_profile(name: str) -> PropFirmProfile:
    """Obtiene un perfil preconfigurado de prueba de fondeo por nombre."""
    key = name.lower().strip()
    if key not in PROFILES:
        raise ValueError(f"Perfil de fondeo '{name}' no reconocido. Disponibles: {list(PROFILES.keys())}")
    return PROFILES[key]


def eval_prop_firm_compliance(
    profile: PropFirmProfile,
    equity_curve: pd.Series,
    trades_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Audita una serie temporal de balance respecto a las reglas de la prueba de fondeo.
    
    Args:
        profile: Reglas y límites de la firma de fondeo.
        equity_curve: Serie temporal de balance del portafolio.
        trades_df: DataFrame con las operaciones cerradas (opcional).
        
    Returns:
        Diccionario con estado de aprobación, violaciones y métricas de consistencia.
    """
    if equity_curve.empty:
        return {"status": "FAIL", "reason": "Curva de balance vacía"}

    # 1. Trailing Drawdown desde el pico histórico
    peak = equity_curve.cummax()
    drawdown_dollars = peak - equity_curve
    max_drawdown_dollars = drawdown_dollars.max()
    trailing_dd_breach = max_drawdown_dollars >= profile.max_trailing_drawdown

    # 2. Daily Loss Limit (DLL)
    daily_equity = equity_curve.resample("1D").last().dropna()
    daily_pnl = daily_equity.diff()
    max_daily_loss = abs(daily_pnl.min()) if len(daily_pnl) > 0 and daily_pnl.min() < 0 else 0.0
    daily_loss_breach = False
    if profile.daily_loss_limit is not None:
        daily_loss_breach = max_daily_loss >= profile.daily_loss_limit

    # 3. Objetivo de Ganancias (Profit Target)
    net_profit = equity_curve.iloc[-1] - profile.starting_balance
    target_reached = net_profit >= profile.profit_target

    # 4. Regla de Consistencia
    # Ningún día individual puede representar más de X% de las ganancias totales
    consistency_breach = False
    max_day_profit_pct = 0.0
    if len(daily_pnl) > 0:
        winning_days = daily_pnl[daily_pnl > 0]
        total_gains = winning_days.sum()
        if total_gains > 0:
            best_day = winning_days.max()
            max_day_profit_pct = round((best_day / total_gains) * 100, 2)
            if profile.consistency_rule_pct is not None and max_day_profit_pct > profile.consistency_rule_pct:
                consistency_breach = True

    # 5. Veredicto Final
    violations = []
    if trailing_dd_breach:
        violations.append(
            f"Trailing Drawdown excedido (${max_drawdown_dollars:,.2f} >= ${profile.max_trailing_drawdown:,.2f})"
        )
    if daily_loss_breach:
        violations.append(
            f"Daily Loss Limit excedido (${max_daily_loss:,.2f} >= ${profile.daily_loss_limit:,.2f})"
        )
    if consistency_breach:
        violations.append(
            f"Regla de consistencia violada (Mejor día generó {max_day_profit_pct}% > {profile.consistency_rule_pct}%)"
        )

    passed = bool(target_reached and not trailing_dd_breach and not daily_loss_breach and not consistency_breach)

    status_str = "🟢 PASÓ LA PRUEBA" if passed else ("🔴 CUENTA QUEMADA" if (trailing_dd_breach or daily_loss_breach) else "🟡 EN PROGRESO")

    return {
        "Perfil Fondeo": profile.name,
        "Estado": status_str,
        "Aprobada": passed,
        "Objetivo Alcanzado": f"{'SÍ' if target_reached else 'NO'} (${net_profit:+,.2f} / ${profile.profit_target:,.2f})",
        "Max Trailing DD ($)": f"${max_drawdown_dollars:,.2f} (Límite: ${profile.max_trailing_drawdown:,.2f})",
        "Max Daily Loss ($)": f"${max_daily_loss:,.2f}" + (f" (Límite: ${profile.daily_loss_limit:,.2f})" if profile.daily_loss_limit else " (Sin límite DLL)"),
        "Día de Mayor Ganancia (%)": f"{max_day_profit_pct}%" + (f" (Límite consistencia: {profile.consistency_rule_pct}%)" if profile.consistency_rule_pct else ""),
        "Violaciones": violations if violations else ["Ninguna"]
    }
