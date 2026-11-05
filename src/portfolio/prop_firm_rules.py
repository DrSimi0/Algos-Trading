"""
src/portfolio/prop_firm_rules.py
Simulador de evaluaciones de cuentas de fondeo (prop firms) de futuros.

Reglas modeladas:
- Profit target: ganancia necesaria para pasar.
- Max Loss trailing (EOD o intradía): el "piso" sube con el máximo saldo.
    * EOD: el piso solo se actualiza con el saldo al cierre de cada día.
    * Intradía: se actualiza con el máximo saldo en cualquier barra.
  La cuenta falla si el saldo toca el piso en CUALQUIER barra.
- Daily Loss Limit: hard = falla la cuenta; soft = se detiene el día.
- Consistencia: el mejor día no puede superar X% de la ganancia total.

IMPORTANTE: los perfiles son aproximados (nov-2026). Verifica siempre las reglas
vigentes en la web oficial de cada firma.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd


@dataclass
class PropFirmProfile:
    name: str
    account_size: float
    profit_target: float
    max_loss: float
    trailing: str = "eod"            # 'eod' | 'intraday' | 'none'
    lock_floor_at_start: bool = True # el piso no sube más allá del balance inicial
    daily_loss_limit: Optional[float] = None
    daily_loss_hard: bool = False    # True = viola cuenta; False = detiene día
    consistency_pct: Optional[float] = None # ej. 0.50 = max 50% en un día
    min_days: int = 1
    commission_rt_micro: float = 1.24 # comisión round-trip por contrato micro


PROFILES: Dict[str, PropFirmProfile] = {
    "topstep_50k": PropFirmProfile(
        name="Topstep 50K Combine",
        account_size=50000.0,
        profit_target=3000.0,
        max_loss=2000.0,
        trailing="eod",
        daily_loss_limit=1000.0,
        daily_loss_hard=False,
        consistency_pct=0.5,
        min_days=2,
        commission_rt_micro=1.4,
    ),
    "lucid_50k": PropFirmProfile(
        name="Lucid 50K",
        account_size=50000.0,
        profit_target=3000.0,
        max_loss=2000.0,
        trailing="eod",
        daily_loss_limit=None,
        consistency_pct=0.5,
        min_days=1,
        commission_rt_micro=1.0,
    ),
}


class PropFirmSimulator:
    def __init__(self, profile: PropFirmProfile):
        self.p = profile

    def evaluate(self, bar_pnl: pd.Series) -> Dict[str, Any]:
        """
        Simula UNA evaluación a partir del P&L en dólares por barra
        (columna 'pnl' de BacktestEngine.run_futures).
        """
        p = self.p
        start = p.account_size
        balance = start
        peak = start
        floor = start - p.max_loss
        day_results: List[float] = []

        for day, day_pnl in bar_pnl.groupby(bar_pnl.index.normalize()):
            day_start = balance
            for ts, x in day_pnl.items():
                balance += x
                if p.trailing == "intraday" and balance > peak:
                    peak = balance
                    floor = self._new_floor(peak)
                if balance <= floor:
                    return self._result("FAIL", "Max Loss (trailing)", day, balance, day_results)
                if p.daily_loss_limit is not None and (balance - day_start) <= -p.daily_loss_limit:
                    if p.daily_loss_hard:
                        return self._result("FAIL", "Daily Loss Limit", day, balance, day_results)
                    else:
                        break

            day_results.append(balance - day_start)

            if p.trailing == "eod" and balance > peak:
                peak = balance
                floor = self._new_floor(peak)

            profit = balance - start
            if profit >= p.profit_target and len(day_results) >= p.min_days:
                if p.consistency_pct is None or max(day_results) <= p.consistency_pct * profit:
                    return self._result("PASS", None, day, balance, day_results)

        return self._result("INCOMPLETE", "No alcanzó el objetivo", None, balance, day_results)

    def pass_probability(self, bar_pnl: pd.Series, max_days: int) -> Dict[str, Any]:
        """
        Corre la evaluación empezando en cada día del histórico (ventanas móviles
        de hasta `max_days` días). Las ventanas se solapan: es una estimación, no
        observaciones independientes.
        """
        days = bar_pnl.index.normalize().unique()
        counts = {"PASS": 0, "FAIL": 0, "INCOMPLETE": 0}
        days_to_pass: List[int] = []
        n_windows = max(len(days) - max_days + 1, 1)

        for i in range(n_windows):
            window_days = days[i : i + max_days]
            mask = bar_pnl.index.normalize().isin(window_days)
            res = self.evaluate(bar_pnl[mask])
            counts[res["status"]] += 1
            if res["status"] == "PASS":
                days_to_pass.append(res["days_traded"])

        total = sum(counts.values())
        return {
            "windows": total,
            "pass_rate (%)": round(100 * counts["PASS"] / total, 1),
            "fail_rate (%)": round(100 * counts["FAIL"] / total, 1),
            "incomplete_rate (%)": round(100 * counts["INCOMPLETE"] / total, 1),
            "median_days_to_pass": (
                float(pd.Series(days_to_pass).median()) if days_to_pass else None
            ),
        }

    def _new_floor(self, peak: float) -> float:
        floor = peak - self.p.max_loss
        if self.p.lock_floor_at_start:
            floor = min(floor, self.p.account_size)
        return floor

    def _result(
        self,
        status: str,
        reason: Optional[str],
        day: Any,
        balance: float,
        day_results: List[float],
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "reason": reason,
            "date": day,
            "final_balance": round(balance, 2),
            "profit": round(balance - self.p.account_size, 2),
            "days_traded": len(day_results) + (1 if status == "FAIL" else 0),
        }
