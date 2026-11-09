"""
src/core/data_split.py
Gestor estricto de partición temporal en 3 niveles:
1. In-Sample (IS): Laboratorio para diseño, filtrado y optimización.
2. Out-of-Sample (OOS): Bóveda intermedia para validación ciega (Visto Bueno 1).
3. Holdout (Bóveda Ciega Definitiva): Datos nunca vistos, desbloqueables solo si IS y OOS son aprobados (Visto Bueno 2).
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple, Sequence
import pandas as pd

# Esquema canónico de 3 años: Año 1 In-Sample, Año 2 Out-of-Sample, Año 3 Holdout Ciego
DEFAULT_SPLIT_YEARS = (2023, 2024, 2025)


class SecurityGateError(PermissionError):
    """Excepción lanzada cuando se intenta acceder a periodos protegidos sin autorización."""
    pass


class DataSplitManager:
    """
    Controla la división y el acceso a los datos de mercado en tres universos temporales:
    - In-Sample (IS): Libre acceso para experimentación.
    - Out-of-Sample (OOS): Sellado bajo cuarentena de Nivel 1.
    - Holdout Ciego: Sellado bajo cuarentena de Nivel 2 (Prueba Definitiva).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        split_ratios: Tuple[float, float, float] = (0.60, 0.25, 0.15),
        split_ratio: Optional[float] = None,
        split_date: Optional[Union[str, pd.Timestamp]] = None,
        oos_date: Optional[Union[str, pd.Timestamp]] = None,
        holdout_date: Optional[Union[str, pd.Timestamp]] = None,
        is_start: Optional[str] = None,
        is_end: Optional[str] = None,
        oos_start: Optional[str] = None,
        oos_end: Optional[str] = None,
        holdout_start: Optional[str] = None,
        holdout_end: Optional[str] = None,
        split_years: Optional[Sequence[int]] = None,
        audit_log_path: Union[str, Path] = "data/audit_oos_runs.log"
    ):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("El DataFrame debe estar indexado con DatetimeIndex.")
        if data.empty:
            raise ValueError("El DataFrame no puede estar vacío.")

        self.data = data.sort_index().copy()
        self.split_years = list(split_years) if split_years else None
        if self.split_years is not None and len(self.split_years) != 3:
            raise ValueError("split_years debe contener exactamente 3 años [IS, OOS, Holdout].")
        
        # Soporte para split_ratio legacy (2 niveles adaptado a 3 niveles)
        if split_ratio is not None:
            r_is = split_ratio
            rem = 1.0 - r_is
            self.split_ratios = (r_is, rem * 0.625, rem * 0.375)
        else:
            self.split_ratios = split_ratios

        # Todas las fechas de corte adoptan la zona horaria del índice (evita tz-naive vs tz-aware)
        self.split_date = self._to_ts(split_date)
        self.oos_date = self._to_ts(oos_date)
        self.holdout_date = self._to_ts(holdout_date)

        self.is_start = self._to_ts(is_start)
        self.is_end = self._to_ts(is_end)
        self.oos_start = self._to_ts(oos_start)
        self.oos_end = self._to_ts(oos_end)
        self.holdout_start = self._to_ts(holdout_start)
        self.holdout_end = self._to_ts(holdout_end)
        self.audit_log_path = Path(audit_log_path)

        self._in_sample, self._out_of_sample, self._holdout = self._compute_three_way_split()
        self.validate_no_leakage()

    def _to_ts(self, value: Optional[Union[str, pd.Timestamp]]) -> Optional[pd.Timestamp]:
        """Convierte una fecha a Timestamp alineado con la zona horaria del índice."""
        if value is None:
            return None
        ts = pd.Timestamp(value)
        tz = self.data.index.tz
        if tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        elif tz is None and ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        elif tz is not None and ts.tzinfo is not None:
            ts = ts.tz_convert(tz)
        return ts

    def _compute_three_way_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Calcula la partición en 3 universos cronológicos sin solapamiento."""
        # Opción 0: Años calendario completos (ej. [2023, 2024, 2025])
        if self.split_years is not None:
            years = self.data.index.year
            y_is, y_oos, y_hold = self.split_years
            # En series históricas consolidadas (ej. CME Globex 2021-2023), el periodo In-Sample
            # cubre todos los años históricos hasta y_is inclusive (2021, 2022, 2023), sin descartar
            # los años previos 2021 y 2022. OOS (y_oos) y Holdout (y_hold) permanecen estrictamente confinados.
            df_is = self.data[years <= y_is]
            df_oos = self.data[years == y_oos]
            df_holdout = self.data[years == y_hold]
            if df_is.empty:
                available = sorted(set(years))
                raise ValueError(
                    f"No hay datos para el periodo In-Sample (<= {y_is}). Años disponibles en el dataset: {available}. "
                    "Importe datos históricos reales de mercado CME Globex (ver data/raw/CME_MINI_ES_2021_2023_5m.parquet)."
                )
            return df_is, df_oos, df_holdout

        # Opción 1: Rangos explícitos
        if self.is_end is not None and self.oos_start is not None and self.holdout_start is not None:
            df_is = self.data.loc[self.is_start:self.is_end]
            df_oos = self.data.loc[self.oos_start:self.oos_end]
            df_holdout = self.data.loc[self.holdout_start:self.holdout_end]
            return df_is, df_oos, df_holdout

        # Opción 2: Fechas de corte explícitas (split_date / oos_date / holdout_date)
        if self.split_date is not None and self.holdout_date is not None:
            df_is = self.data[self.data.index < self.split_date]
            df_oos = self.data[(self.data.index >= self.split_date) & (self.data.index < self.holdout_date)]
            df_holdout = self.data[self.data.index >= self.holdout_date]
            return df_is, df_oos, df_holdout

        # Si solo se pasa split_date (compatibilidad hacia atrás):
        # dividimos lo posterior a split_date en 60% OOS y 40% Holdout
        if self.split_date is not None:
            df_is = self.data[self.data.index < self.split_date]
            after_split = self.data[self.data.index >= self.split_date]
            if len(after_split) < 2:
                raise ValueError("No hay suficientes datos después de split_date para crear OOS y Holdout.")
            oos_cutoff = int(len(after_split) * 0.65)
            df_oos = after_split.iloc[:oos_cutoff]
            df_holdout = after_split.iloc[oos_cutoff:]
            return df_is, df_oos, df_holdout

        # Opción 3: Proporciones de 3 niveles (ej. 60% IS, 25% OOS, 15% Holdout)
        r_is, r_oos, r_holdout = self.split_ratios
        if not abs((r_is + r_oos + r_holdout) - 1.0) < 1e-4:
            raise ValueError(f"Los split_ratios deben sumar 1.0 (recibido: {self.split_ratios}).")

        n_total = len(self.data)
        idx_is = int(n_total * r_is)
        idx_oos = int(n_total * (r_is + r_oos))

        df_is = self.data.iloc[:idx_is]
        df_oos = self.data.iloc[idx_is:idx_oos]
        df_holdout = self.data.iloc[idx_oos:]
        return df_is, df_oos, df_holdout

    def validate_no_leakage(self) -> None:
        """Verifica que no exista ninguna fuga ni solapamiento entre IS, OOS y Holdout."""
        if self._in_sample.empty:
            raise ValueError("El periodo In-Sample está vacío.")

        if not self._out_of_sample.empty:
            max_is = self._in_sample.index.max()
            min_oos = self._out_of_sample.index.min()
            if max_is >= min_oos:
                raise ValueError(f"Fuga temporal IS -> OOS detectada: {max_is} >= {min_oos}")
            if len(self._in_sample.index.intersection(self._out_of_sample.index)) > 0:
                raise ValueError("Fuga detectada: Intersección no vacía entre IS y OOS.")

        if not self._holdout.empty and not self._out_of_sample.empty:
            max_oos = self._out_of_sample.index.max()
            min_holdout = self._holdout.index.min()
            if max_oos >= min_holdout:
                raise ValueError(f"Fuga temporal OOS -> Holdout detectada: {max_oos} >= {min_holdout}")
            if len(self._out_of_sample.index.intersection(self._holdout.index)) > 0:
                raise ValueError("Fuga detectada: Intersección no vacía entre OOS y Holdout.")

    def get_in_sample(self) -> pd.DataFrame:
        """Retorna el subconjunto In-Sample libre para desarrollo y optimización."""
        return self._in_sample.copy()

    def get_out_of_sample(self, authorized: bool = False, reason: str = "") -> pd.DataFrame:
        """
        Retorna el subconjunto Out-of-Sample (Nivel 1 de cuarentena).
        Requiere autorización explícita tras auditar In-Sample.
        """
        if not authorized:
            range_info = (
                f"{self._out_of_sample.index[0].date()} a {self._out_of_sample.index[-1].date()} ({len(self._out_of_sample)} barras)"
                if not self._out_of_sample.empty else "Sellado en bóveda criptográfica data/vault/oos_2024_5m.enc"
            )
            try:
                from src.core.vault_security import record_tripwire_event
                record_tripwire_event("UNAUTHORIZED_OOS_ACCESS_ATTEMPT", f"Intento no autorizado a OOS ({range_info})")
            except Exception:
                pass
            raise SecurityGateError(
                "\n[ACCESO BLOQUEADO - CUARENTENA NIVEL 1: OUT-OF-SAMPLE]\n"
                "El periodo Out-of-Sample está protegido para prevenir sobreajuste.\n"
                "Para desbloquearlo, debes otorgar tu Visto Bueno explícito (--approve-oos).\n"
                f"Rango OOS: {range_info}."
            )

        if self._out_of_sample.empty:
            try:
                from src.core.vault_security import record_tripwire_event
                record_tripwire_event("VAULTED_OOS_BLOCKED", "Intento de leer OOS cuando los datos están protegidos en bóveda")
            except Exception:
                pass
            raise SecurityGateError(
                "\n[ACCESO BLOQUEADO - DATOS PROTEGIDOS EN BÓVEDA CRIPTOGRÁFICA]\n"
                "Los datos de 2024 (Out-of-Sample) están físicamente encriptados en data/vault/oos_2024_5m.enc.\n"
                "Para desbloquearlos en RAM, utiliza el runner del orquestador: run_multi_year_robustness_audit.py."
            )

        self._log_audit_access("OOS_UNLOCKED", reason, self._out_of_sample)
        return self._out_of_sample.copy()

    def get_holdout(self, authorized: bool = False, reason: str = "") -> pd.DataFrame:
        """
        Retorna el subconjunto Holdout Ciego Definitivo (Nivel 2 de cuarentena).
        Solo debe abrirse como prueba de fuego final tras verificar que IS y OOS son consistentes.
        """
        if not authorized:
            range_info = (
                f"{self._holdout.index[0].date()} a {self._holdout.index[-1].date()} ({len(self._holdout)} barras)"
                if not self._holdout.empty else "Sellado en bóveda criptográfica data/vault/holdout_2025_5m.enc"
            )
            try:
                from src.core.vault_security import record_tripwire_event
                record_tripwire_event("UNAUTHORIZED_HOLDOUT_ACCESS_ATTEMPT", f"Intento no autorizado a Holdout ({range_info})")
            except Exception:
                pass
            raise SecurityGateError(
                "\n[ACCESO BLOQUEADO - CUARENTENA NIVEL 2: BÓVEDA CIEGA HOLDOUT]\n"
                "Este conjunto de datos jamás ha sido visto por la estrategia.\n"
                "Solo debe abrirse si In-Sample y Out-of-Sample han resultado positivos y aprobados.\n"
                "Para desbloquearlo, debes otorgar la autorización definitiva (--approve-holdout).\n"
                f"Rango Holdout: {range_info}."
            )

        if self._holdout.empty:
            try:
                from src.core.vault_security import record_tripwire_event
                record_tripwire_event("VAULTED_HOLDOUT_BLOCKED", "Intento de leer Holdout cuando los datos están protegidos en bóveda")
            except Exception:
                pass
            raise SecurityGateError(
                "\n[ACCESO BLOQUEADO - DATOS PROTEGIDOS EN BÓVEDA CRIPTOGRÁFICA]\n"
                "Los datos de 2025 (Holdout) están físicamente encriptados en data/vault/holdout_2025_5m.enc.\n"
                "Para desbloquearlos en RAM, utiliza el runner del orquestador: run_multi_year_robustness_audit.py."
            )

        self._log_audit_access("HOLDOUT_FINAL_UNLOCKED", reason, self._holdout)
        return self._holdout.copy()

    def _log_audit_access(self, event_type: str, reason: str, df: pd.DataFrame) -> None:
        """Registra el acceso en un log inmutable para trazabilidad."""
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().isoformat()
            if not df.empty:
                r_range = f"{df.index[0].date()} a {df.index[-1].date()} ({len(df)} bars)"
            else:
                r_range = "0 bars (encrypted in vault)"
            log_entry = f"[{timestamp}] {event_type} | Reason: '{reason or 'N/A'}' | Range: {r_range}\n"
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception:
            pass

    def get_split_summary(self) -> Dict[str, Any]:
        """Retorna el resumen cuantitativo de las 3 particiones."""
        n_total = len(self.data)
        n_is = len(self._in_sample)
        n_oos = len(self._out_of_sample)
        n_hold = len(self._holdout)

        summary = {
            "Total Bars": n_total,
            "Total Range": f"{self.data.index[0].date()} a {self.data.index[-1].date()}",
            "In-Sample Range": f"{self._in_sample.index[0].date()} a {self._in_sample.index[-1].date()}",
            "In-Sample Bars": n_is,
            "In-Sample %": round((n_is / n_total) * 100, 2),
        }
        if n_oos > 0:
            summary["Out-of-Sample Range"] = f"{self._out_of_sample.index[0].date()} a {self._out_of_sample.index[-1].date()}"
            summary["Out-of-Sample Bars"] = n_oos
            summary["Out-of-Sample %"] = round((n_oos / n_total) * 100, 2)
        else:
            summary["Out-of-Sample"] = "PROTEGIDO EN BÓVEDA (data/vault/oos_2024_5m.enc)"

        if n_hold > 0:
            summary["Holdout Range"] = f"{self._holdout.index[0].date()} a {self._holdout.index[-1].date()}"
            summary["Holdout Bars"] = n_hold
            summary["Holdout %"] = round((n_hold / n_total) * 100, 2)
        else:
            summary["Holdout"] = "PROTEGIDO EN BÓVEDA (data/vault/holdout_2025_5m.enc)"

        return summary


# =============================================================================
# Helpers universales para runners (forzado de In-Sample por defecto)
# =============================================================================

def add_data_split_args(parser: argparse.ArgumentParser, mode_arg_name: str = "--mode") -> None:
    """Añade los argumentos estandarizados de partición y cuarentena a un runner."""
    parser.add_argument(
        mode_arg_name,
        dest="split_mode",
        type=str,
        choices=["in_sample", "out_of_sample", "holdout"],
        default="in_sample",
        help="Partición temporal: 'in_sample' (por defecto), 'out_of_sample' o 'holdout'",
    )
    parser.add_argument("--approve-oos", action="store_true", default=False,
                        help="Visto bueno explícito para desbloquear el Año 2 (Out-of-Sample)")
    parser.add_argument("--approve-holdout", action="store_true", default=False,
                        help="Visto bueno explícito para desbloquear el Año 3 (Holdout Ciego)")
    parser.add_argument("--split-years", type=int, nargs=3, default=list(DEFAULT_SPLIT_YEARS),
                        help="Años calendario [IS OOS Holdout] (defecto: 2023 2024 2025). En series consolidadas multi-anuales (ej: 2021-2023), In-Sample abarca todos los años históricos hasta IS inclusive.")


def _deny(title: str, detail: str, flag: str) -> None:
    print("\n" + "!" * 80)
    print(f" [ERROR DE SEGURIDAD CUANTITATIVA: {title}]")
    print(f" {detail}")
    print(f" Para continuar, revisa primero tus resultados y agrega explícitamente: {flag}")
    print("!" * 80 + "\n")
    sys.exit(1)


def check_partition_authorization(mode: str, approve_oos: bool, approve_holdout: bool) -> None:
    """Falla rápido (exit 1) si se pide OOS/Holdout sin autorización. Llamar ANTES de cargar datos."""
    if mode == "out_of_sample" and not approve_oos:
        _deny("ACCESO OUT-OF-SAMPLE DENEGADO",
              "El Año 2 (Out-of-Sample) está sellado hasta que apruebes el In-Sample.", "--approve-oos")
    if mode == "holdout" and not approve_holdout:
        _deny("ACCESO A HOLDOUT DENEGADO",
              "El Año 3 (Holdout Ciego) solo se abre tras aprobar In-Sample y Out-of-Sample.",
              "--approve-holdout")


def extract_partition_data(
    split_mgr: "DataSplitManager",
    mode: str,
    approve_oos: bool,
    approve_holdout: bool,
    strategy_name: str,
) -> Tuple[pd.DataFrame, str]:
    """Devuelve (DataFrame de la partición activa, etiqueta) validando las barreras de seguridad."""
    check_partition_authorization(mode, approve_oos, approve_holdout)
    if mode == "in_sample":
        df = split_mgr.get_in_sample()
        return df, f"IN-SAMPLE (Laboratorio) | {df.index[0]} -> {df.index[-1]} | {len(df):,} barras"
    if mode == "out_of_sample":
        df = split_mgr.get_out_of_sample(authorized=True, reason=f"{strategy_name}_oos")
        return df, f"OUT-OF-SAMPLE (Bóveda 1) | {df.index[0]} -> {df.index[-1]} | {len(df):,} barras"
    if mode == "holdout":
        df = split_mgr.get_holdout(authorized=True, reason=f"{strategy_name}_holdout")
        return df, f"HOLDOUT CIEGO (Bóveda 2) | {df.index[0]} -> {df.index[-1]} | {len(df):,} barras"
    raise ValueError(f"Modo de partición no reconocido: {mode}")


def partition_from_args(df: pd.DataFrame, args: argparse.Namespace, strategy_name: str) -> Tuple[pd.DataFrame, str]:
    """Atajo para runners: construye el DataSplitManager por años y extrae la partición activa."""
    check_partition_authorization(args.split_mode, args.approve_oos, args.approve_holdout)
    split_mgr = DataSplitManager(df, split_years=args.split_years)
    df_part, label = extract_partition_data(
        split_mgr, args.split_mode, args.approve_oos, args.approve_holdout, strategy_name
    )
    print(f"\n  [PARTICIÓN ACTIVA] {label}")
    if args.split_mode == "in_sample":
        print("  [CUARENTENA] Año 2 (OOS) y Año 3 (Holdout) sellados. Usa --mode out_of_sample --approve-oos tras aprobar IS.")
    return df_part, label


def clip_to_partition(df_other: pd.DataFrame, df_active: pd.DataFrame) -> pd.DataFrame:
    """Recorta un DataFrame auxiliar (ej. diario o par) para que nunca exceda la fecha final de la partición activa."""
    end = df_active.index.max()
    idx = df_other.index
    if idx.tz is None and end.tzinfo is not None:
        end = end.tz_convert(None)
    elif idx.tz is not None and end.tzinfo is None:
        end = end.tz_localize(idx.tz)
    elif idx.tz is not None:
        end = end.tz_convert(idx.tz)
    # Para datos diarios, comparar contra el final del día de la última barra activa
    return df_other[idx <= end.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)] \
        if (idx.normalize() == idx).all() else df_other[idx <= end]
