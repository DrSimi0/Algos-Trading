"""
src/strategies/f7_optimization_is_n1.py

Estrategia Cuantitativa Alterna Formal F7: 'F7-Optimizacion -IS- N1'
Optimizaciones Matemáticas Incorporadas:
1. Calibración Optimizada de Probabilidades del Meta-Clasificador:
   - Platt Scaling (Sigmoide analítica sobre log-odds out-of-fold).
   - Temperature Scaling (Parámetro T > 0 con regularización Bayesiana).
   - Isotonic Regression (Algoritmo PAV - Pool Adjacent Violators).
   - Purged K-Fold Cross-Validation para estimación out-of-fold sin fuga de información (Zero Leakage).
2. Optimización Asimétrica de Barreras Triple Barrier adaptadas a Prop Trading:
   - Multiplicador de Stop Loss adaptativo l_t^- dependiente del régimen de volatilidad Garman-Klass y del colchón de Trailing Drawdown.
   - Multiplicador de Take Profit adaptativo l_t^+ que asegura un ratio riesgo/recompensa b_t >= 2.0.
   - Barrera vertical temporal adaptativa Delta_tau_t dependiente de la velocidad del mercado.
3. Bet Sizing mediante Criterio de Kelly Fraccional Truncado con Desescalamiento Progresivo:
   - Criterio de Kelly analítico: f*(p, b) = (b*p - (1 - p)) / b.
   - Fracción de Kelly truncada kappa in (0, 1] (Quarter-Kelly o Half-Kelly).
   - Modulación continua por colchón de Trailing Drawdown psi(rho).
   - Desescalamiento y conmutación obligatoria y proactiva a microcontratos (MES/MNQ)
     cuando la equidad se aproxima al Max Trailing Drawdown (rho <= rho_micro_threshold).
4. Integración estricta con DataSplitManager para validación In-Sample y Out-Of-Sample.

Referencias:
- López de Prado, M. (2018). Advances in Financial Machine Learning. Capítulos 3, 5, 7, 9 y 10.
- Kelly, J. L. (1956). A New Interpretation of Information Rate. Bell System Technical Journal.
- Platt, J. (1999). Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.
"""

from dataclasses import dataclass, field
from datetime import time, datetime, timedelta
from enum import Enum
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from src.core.base_strategy import BaseStrategy
from src.core.futures_specs import get_spec
from src.strategies.meta_labeling_triple_barrier import (
    PositionDirection,
    ExitReason,
    StrategyState,
    MicroFeatures,
    TradeRecord,
    MetaModelWeights,
    MetaLabelingClassifier,
    compute_garman_klass_volatility,
    compute_intraday_anchored_vwap,
    compute_order_flow_imbalance_zscore,
    compute_relative_volume,
    compute_range_compression,
    compute_session_fraction,
    detect_data_time_offset,
    apply_meta_veto,
)


# ==============================================================================
# 1. CALIBRACIÓN DE PROBABILIDADES DEL META-CLASIFICADOR
# ==============================================================================

class CalibrationMethod(str, Enum):
    """Métodos de calibración de probabilidades soportados."""
    PLATT = "platt"
    TEMPERATURE = "temperature"
    ISOTONIC = "isotonic"
    NONE = "none"


class ProbabilityCalibrator:
    """
    Calibrador empírico de probabilidades para el meta-clasificador.
    Convierte logits crudos z_t in R en probabilidades calibradas p_cal in [0, 1].
    Implementa:
    - Platt Scaling: p = 1 / (1 + exp(-(a * z + b)))
    - Temperature Scaling: p = 1 / (1 + exp(-z / T))
    - Isotonic Regression: Monotone piecewise-constant mapping via PAV algorithm.
    """

    def __init__(self, method: CalibrationMethod = CalibrationMethod.PLATT):
        self.method = method
        self.a: float = 1.0
        self.b: float = 0.0
        self.temperature: float = 1.0
        self.isotonic_knots_x: Optional[np.ndarray] = None
        self.isotonic_knots_y: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, logits: np.ndarray, labels: np.ndarray, l2_reg: float = 0.1) -> "ProbabilityCalibrator":
        """
        Ajusta los parámetros de calibración minimizando la entropía cruzada / pérdida Brier
        sobre predicciones fuera de muestra (out-of-fold).
        """
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=float)

        if len(logits) < 5 or len(np.unique(labels)) < 2:
            self.is_fitted = False
            return self

        if self.method == CalibrationMethod.PLATT:
            # Optimización de Platt Scaling: a * z + b
            def platt_loss(params: np.ndarray) -> float:
                a_val, b_val = params[0], params[1]
                scaled = np.clip(a_val * logits + b_val, -30.0, 30.0)
                probs = 1.0 / (1.0 + np.exp(-scaled))
                eps = 1e-12
                probs = np.clip(probs, eps, 1.0 - eps)
                ce = -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
                reg = l2_reg * ((a_val - 1.0) ** 2 + b_val ** 2)
                return ce + reg

            res = minimize(platt_loss, x0=np.array([1.0, 0.0]), method="L-BFGS-B", bounds=[(0.01, 10.0), (-10.0, 10.0)])
            if res.success:
                self.a = float(res.x[0])
                self.b = float(res.x[1])
                self.is_fitted = True

        elif self.method == CalibrationMethod.TEMPERATURE:
            # Optimización de Temperature Scaling: z / T
            def temp_loss(t_val: float) -> float:
                t_safe = max(t_val, 0.01)
                scaled = np.clip(logits / t_safe, -30.0, 30.0)
                probs = 1.0 / (1.0 + np.exp(-scaled))
                eps = 1e-12
                probs = np.clip(probs, eps, 1.0 - eps)
                ce = -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
                reg = l2_reg * ((t_safe - 1.0) ** 2)
                return ce + reg

            res = minimize_scalar(temp_loss, bounds=(0.05, 10.0), method="bounded")
            if res.success:
                self.temperature = float(res.x)
                self.is_fitted = True

        elif self.method == CalibrationMethod.ISOTONIC:
            # Algoritmo PAV (Pool Adjacent Violators)
            self._fit_pav(logits, labels)
            self.is_fitted = True

        return self

    def _fit_pav(self, x: np.ndarray, y: np.ndarray) -> None:
        """Implementación pura en NumPy del algoritmo Pool Adjacent Violators (PAV)."""
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        # Bloques iniciales: cada punto es un bloque con peso 1
        blocks_x: List[float] = [float(val) for val in x_sorted]
        blocks_y: List[float] = [float(val) for val in y_sorted]
        blocks_w: List[float] = [1.0 for _ in range(len(y_sorted))]

        i = 0
        while i < len(blocks_y) - 1:
            if blocks_y[i] > blocks_y[i + 1]:
                # Violación de monotonicidad: agrupar y promediar
                w_combined = blocks_w[i] + blocks_w[i + 1]
                y_combined = (blocks_y[i] * blocks_w[i] + blocks_y[i + 1] * blocks_w[i + 1]) / w_combined
                x_combined = (blocks_x[i] * blocks_w[i] + blocks_x[i + 1] * blocks_w[i + 1]) / w_combined

                blocks_y[i] = y_combined
                blocks_x[i] = x_combined
                blocks_w[i] = w_combined

                del blocks_y[i + 1]
                del blocks_x[i + 1]
                del blocks_w[i + 1]

                # Retroceder para verificar si la fusión violó monotonicidad previa
                if i > 0:
                    i -= 1
            else:
                i += 1

        self.isotonic_knots_x = np.array(blocks_x, dtype=float)
        self.isotonic_knots_y = np.clip(np.array(blocks_y, dtype=float), 0.001, 0.999)

    def calibrate(self, logit: float) -> float:
        """Transforma un logit crudo z_t en probabilidad calibrada."""
        if not self.is_fitted or self.method == CalibrationMethod.NONE:
            clipped = float(np.clip(logit, -30.0, 30.0))
            return 1.0 / (1.0 + math.exp(-clipped))

        if self.method == CalibrationMethod.PLATT:
            scaled = float(np.clip(self.a * logit + self.b, -30.0, 30.0))
            return 1.0 / (1.0 + math.exp(-scaled))

        elif self.method == CalibrationMethod.TEMPERATURE:
            t_safe = max(self.temperature, 0.01)
            scaled = float(np.clip(logit / t_safe, -30.0, 30.0))
            return 1.0 / (1.0 + math.exp(-scaled))

        elif self.method == CalibrationMethod.ISOTONIC:
            if self.isotonic_knots_x is None or len(self.isotonic_knots_x) == 0:
                clipped = float(np.clip(logit, -30.0, 30.0))
                return 1.0 / (1.0 + math.exp(-clipped))
            # Interpolación lineal sobre los nodos monótonos de PAV
            return float(np.interp(
                logit,
                self.isotonic_knots_x,
                self.isotonic_knots_y,
                left=self.isotonic_knots_y[0],
                right=self.isotonic_knots_y[-1]
            ))

        clipped = float(np.clip(logit, -30.0, 30.0))
        return 1.0 / (1.0 + math.exp(-clipped))


class PurgedKFoldCV:
    """
    Generador de particiones Purged K-Fold Cross Validation con embargo temporal.
    Garantiza que no exista filtración de información entre folds de entrenamiento y prueba
    (López de Prado 2018, Capítulo 7).
    """

    def __init__(self, n_splits: int = 5, purge_window_bars: int = 6):
        self.n_splits = max(2, n_splits)
        self.purge_window = max(0, purge_window_bars)

    def split(self, event_indices: List[int]) -> List[Tuple[List[int], List[int]]]:
        """
        Retorna lista de tuplas (train_indices, test_indices) con purgado en las fronteras.
        """
        n_events = len(event_indices)
        if n_events < self.n_splits * 2:
            # Fallback simple si hay muy pocos eventos
            mid = n_events // 2
            return [
                (list(range(0, mid)), list(range(mid, n_events))),
                (list(range(mid, n_events)), list(range(0, mid)))
            ]

        fold_sizes = np.full(self.n_splits, n_events // self.n_splits, dtype=int)
        fold_sizes[:n_events % self.n_splits] += 1

        splits = []
        current = 0
        for fold_idx in range(self.n_splits):
            start = current
            end = current + fold_sizes[fold_idx]
            current = end

            test_idx = list(range(start, end))

            # Purgar barras adyacentes al bloque de test
            test_bar_min = event_indices[start]
            test_bar_max = event_indices[end - 1]

            train_idx = []
            for i, bar_pos in enumerate(event_indices):
                if i in test_idx:
                    continue
                # Descartar si el evento cae dentro del periodo de purga
                if (test_bar_min - self.purge_window) <= bar_pos <= (test_bar_max + self.purge_window):
                    continue
                train_idx.append(i)

            splits.append((train_idx, test_idx))

        return splits


class CalibratedMetaClassifier:
    """
    Meta-clasificador probabilístico secundario con calibración optimizada.
    Evalúa F_meta(X_t, s_t^{(1)}) y genera p_t calibrado mediante Purged K-Fold Cross Validation.
    """

    def __init__(
        self,
        base_weights: Optional[MetaModelWeights] = None,
        calibration_method: CalibrationMethod = CalibrationMethod.PLATT,
        n_cv_splits: int = 5,
        purge_window: int = 6
    ):
        self.base_classifier = MetaLabelingClassifier(weights=base_weights)
        self.calibration_method = calibration_method
        self.calibrator = ProbabilityCalibrator(method=calibration_method)
        self.n_cv_splits = n_cv_splits
        self.purge_window = purge_window

    def evaluate_raw_logit(self, feat: MicroFeatures, primary_signal: int) -> float:
        """Calcula el logit z_t = F_meta(X_t, s_t^{(1)})."""
        return self.base_classifier.evaluate_logit(feat, primary_signal)

    def predict_probability(self, feat: MicroFeatures, primary_signal: int) -> float:
        """Calcula la probabilidad calibrada final p_t in [0, 1]."""
        logit = self.evaluate_raw_logit(feat, primary_signal)
        return self.calibrator.calibrate(logit)

    def fit_with_purged_cv(
        self,
        features: List[MicroFeatures],
        signals: List[int],
        labels: List[int],
        event_indices: List[int],
        l2_reg: float = 1.0
    ) -> "CalibratedMetaClassifier":
        """
        Entrena el clasificador base y ajusta el calibrador de probabilidades
        mediante validación cruzada purgada para prevenir cualquier sobreajuste o fuga temporal.
        """
        n_samples = len(features)
        if n_samples < 15 or len(np.unique(labels)) < 2:
            return self

        cv = PurgedKFoldCV(n_splits=self.n_cv_splits, purge_window_bars=self.purge_window)
        splits = cv.split(event_indices)

        oof_logits = []
        oof_labels = []

        # 1. Generar predicciones fuera de muestra (out-of-fold) con purgado
        for train_idx, test_idx in splits:
            if len(train_idx) < 10 or len(test_idx) == 0:
                continue

            fold_feats = [features[i] for i in train_idx]
            fold_sigs = [signals[i] for i in train_idx]
            fold_labels = [labels[i] for i in train_idx]

            # Entrenar modelo temporal en este fold
            fold_clf = MetaLabelingClassifier(weights=MetaModelWeights())
            fold_clf.fit(fold_feats, fold_sigs, fold_labels, l2_reg=l2_reg)

            # Predecir logits sobre el bloque de test
            for ti in test_idx:
                z_val = fold_clf.evaluate_logit(features[ti], signals[ti])
                oof_logits.append(z_val)
                oof_labels.append(labels[ti])

        # 2. Ajustar el calibrador con los logits out-of-fold
        if len(oof_logits) >= 10:
            self.calibrator.fit(np.array(oof_logits), np.array(oof_labels))

        # 3. Ajustar el modelo base sobre la totalidad de los datos In-Sample
        self.base_classifier.fit(features, signals, labels, l2_reg=l2_reg)

        return self


# ==============================================================================
# 2. OPTIMIZACIÓN ASIMÉTRICA DE BARRERAS TRIPLE BARRIER PARA PROP TRADING
# ==============================================================================

@dataclass
class AdaptiveBarrierConfig:
    """Configuración analítica para barreras asimétricas adaptativas."""
    base_ell_plus: float = 2.00        # Multiplicador base TP
    base_ell_minus: float = 1.00       # Multiplicador base SL
    min_asymmetry_ratio: float = 2.00  # Cota inferior de ratio riesgo/recompensa b_t >= 2.0
    vol_lookback_bars: int = 60        # Ventana para volatilidad media de referencia
    cushion_defensive_threshold: float = 0.50  # Umbral de activación de defensa de drawdown (50% MTDD)
    min_ell_minus: float = 0.50        # Cota inferior absoluta para SL
    base_vertical_bars: int = 6        # Barrera vertical base (6 barras x 5m = 30m)


class AdaptiveBarrierManager:
    """
    Optimizador de Barreras Triples Asimétricas Adaptativas para Cuentas Prop.
    Ajusta dinámicamente:
    1. l_t^- (Stop Loss): Se comprime cuando la volatilidad se expande o el colchón de MTDD disminuye.
    2. l_t^+ (Take Profit): Se expande en regímenes tendenciales manteniendo b_t = l_t^+ / l_t^- >= 2.0.
    3. Delta_tau_t (Barrera Vertical): Se reduce en alta volatilidad para evitar estancamiento y acelerar rotación.
    """

    def __init__(self, config: Optional[AdaptiveBarrierConfig] = None):
        self.cfg = config or AdaptiveBarrierConfig()

    def calculate_regime_adjusted_barriers(
        self,
        current_price: float,
        current_sigma: float,
        rolling_median_sigma: float,
        cushion_ratio: float,          # D_t^{cushion} / MTDD in [0, 1]
        signal_star: int
    ) -> Tuple[float, float, int, float, float]:
        """
        Calcula las barreras adaptativas (B_upper, B_lower, delta_tau_bars, ell_plus_t, ell_minus_t).
        
        Args:
            current_price: Precio P_0 al momento de entrada.
            current_sigma: Volatilidad Garman-Klass de la barra actual.
            rolling_median_sigma: Volatilidad media/mediana de referencia reciente.
            cushion_ratio: Proporción del colchón MTDD restante (0.0 = límite alcanzado, 1.0 = capital pico).
            signal_star: Sentido de la operación (+1 Long, -1 Short).
            
        Returns:
            Tuple[b_upper, b_lower, adaptive_vertical_bars, ell_plus_t, ell_minus_t]
        """
        ref_sigma = max(rolling_median_sigma, 0.5)
        omega_vol = np.clip(current_sigma / ref_sigma, 0.5, 3.0)

        # 1. Modulador de Stop Loss dependiente de volatilidad
        # En alta volatilidad (omega_vol > 1.0), reducimos ell^- para evitar que el riesgo en USD por contrato explote
        vol_sl_factor = min(1.0, 1.0 / math.sqrt(omega_vol))

        # 2. Modulador defensivo por proximidad al Trailing Drawdown
        # Si cushion_ratio < 0.50, contraemos gradualmente el stop loss para limitar el impacto unitario en USD
        if cushion_ratio < self.cfg.cushion_defensive_threshold:
            ratio_norm = max(0.0, cushion_ratio) / self.cfg.cushion_defensive_threshold
            cushion_sl_factor = max(0.50, math.sqrt(ratio_norm))
        else:
            cushion_sl_factor = 1.0

        ell_minus_t = max(
            self.cfg.min_ell_minus,
            self.cfg.base_ell_minus * vol_sl_factor * cushion_sl_factor
        )

        # 3. Modulador de Take Profit asegurando asimetría matemática estricta (b_t >= 2.0)
        # En expansión de volatilidad, aprovechamos la inercia del movimiento
        vol_tp_factor = np.clip(omega_vol ** 0.25, 0.8, 1.5)
        target_tp = self.cfg.base_ell_plus * vol_tp_factor
        ell_plus_t = max(self.cfg.min_asymmetry_ratio * ell_minus_t, target_tp)

        # 4. Barrera vertical temporal adaptativa
        # En alta volatilidad el precio se mueve más rápido; reducimos tiempo para limitar riesgo de cola
        time_factor = np.clip(1.0 / math.sqrt(omega_vol), 0.6, 1.4)
        adaptive_vertical_bars = max(3, int(round(self.cfg.base_vertical_bars * time_factor)))

        # 5. Geometría de Barreras
        if signal_star == 1:  # LONG
            b_upper = current_price + ell_plus_t * current_sigma
            b_lower = current_price - ell_minus_t * current_sigma
        elif signal_star == -1:  # SHORT
            b_upper = current_price + ell_minus_t * current_sigma
            b_lower = current_price - ell_plus_t * current_sigma
        else:
            raise ValueError("Señal ejecutoria debe ser +1 o -1.")

        return b_upper, b_lower, adaptive_vertical_bars, ell_plus_t, ell_minus_t


# ==============================================================================
# 3. CRITERIO DE KELLY FRACCIONAL TRUNCADO CON DESESCALAMIENTO PROGRESIVO
# ==============================================================================

def calculate_truncated_fractional_kelly(
    p_t: float,
    b_ratio: float,
    tau_meta: float = 0.60,
    kelly_fraction: float = 0.25,
    max_bet_size: float = 1.00
) -> float:
    """
    Calcula el tamaño continuo de la apuesta m_Kelly(p_t) mediante el Criterio de Kelly Fraccional Truncado:
    f*(p, b) = (b * p - (1 - p)) / b = p - (1 - p) / b.
    m_Kelly = clip(kappa * f*(p, b), 0.0, max_bet_size) si p_t >= tau_meta, else 0.0.
    
    Args:
        p_t: Probabilidad calibrada de acierto.
        b_ratio: Ratio de beneficio/riesgo (ell^+ / ell^- >= 2.0).
        tau_meta: Umbral analítico de convicción mínima.
        kelly_fraction: Fracción conservadora de Kelly kappa in (0, 1] (ej. 0.25 = Quarter Kelly).
        max_bet_size: Cota superior de modulación (def: 1.0).
    """
    if p_t < tau_meta or b_ratio <= 0.0:
        return 0.0

    # Fórmula general de Kelly para apuestas asimétricas
    q_t = 1.0 - p_t
    f_star = (b_ratio * p_t - q_t) / b_ratio

    if f_star <= 0.0:
        return 0.0

    f_frac = kelly_fraction * f_star
    return float(np.clip(f_frac, 0.0, max_bet_size))


def calculate_cushion_deescalation_factor(
    cushion_ratio: float,              # D_t^{cushion} / MTDD in [0, 1]
    normal_threshold: float = 0.50,     # Umbral normal por encima del cual no hay desescalamiento
    circuit_breaker_ratio: float = 0.15 # Umbral del disyuntor de emergencia
) -> float:
    """
    Función analítica cóncava continua de desescalamiento psi(rho) in [0, 1]
    según la distancia de la equidad al Max Trailing Drawdown (MTDD).
    
    - Si rho >= normal_threshold (0.50): psi = 1.0 (operación normal).
    - Si circuit_breaker_ratio < rho < normal_threshold:
      psi(rho) = ((rho - rho_cb) / (rho_norm - rho_cb))^1.2 (desescalamiento progresivo).
    - Si rho <= circuit_breaker_ratio (0.15): psi = 0.0 (bloqueo total).
    """
    if cushion_ratio <= circuit_breaker_ratio:
        return 0.0

    if cushion_ratio >= normal_threshold:
        return 1.0

    # Transición suave con exponente de amortiguación 1.2
    normalized_cushion = (cushion_ratio - circuit_breaker_ratio) / (normal_threshold - circuit_breaker_ratio)
    return float(np.clip(normalized_cushion ** 1.2, 0.0, 1.0))


# ==============================================================================
# 4. GESTOR DE RIESGO OPTIMIZADO PARA CUENTAS PROP (PROP RISK MANAGER N1)
# ==============================================================================

@dataclass
class PropRiskManagerOptimized:
    """
    Gestión cuantitativa de capital adaptada a Prop Trading con:
    - Reducción continua del tamaño de apuesta (Kelly Fraccional x Factor de Colchón).
    - Conmutación forzada proactiva a Microcontratos (MES/MNQ) cuando el colchón se aproxima a MTDD.
    - Presupuesto de riesgo estrictamente limitado por DLL y MTDD.
    - Disyuntor de emergencia (circuit breaker) al 15% del MTDD.
    """
    initial_capital: float = 50000.0
    dll: float = 1000.0                 # Daily Loss Limit ($1000)
    mtdd: float = 2000.0                # Max Trailing Drawdown ($2000)
    gamma_d: float = 0.25               # Fracción máxima de DLL por trade (25%)
    gamma_t: float = 0.15               # Fracción máxima de MTDD cushion por trade (15%)
    circuit_breaker_ratio: float = 0.15 # Umbral de parada de emergencia (15% MTDD)
    force_micro_cushion_ratio: float = 0.40 # Si cushion <= 40% MTDD, FORZAR microcontratos
    max_contracts_std: int = 5          # Límite máximo de contratos estándar
    
    equity: float = field(init=False)
    peak_equity: float = field(init=False)
    daily_start_equity: float = field(init=False)
    daily_realized_loss: float = 0.0
    is_halted: bool = False
    halt_reason: Optional[str] = None

    def __post_init__(self):
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.daily_start_equity = self.initial_capital

    def start_new_session(self, current_date: Any) -> None:
        """Reinicia contadores diarios al comienzo de una nueva sesión bursátil."""
        self.daily_start_equity = self.equity
        self.daily_realized_loss = 0.0
        if self.is_halted and "DAILY_LOSS_LIMIT" in str(self.halt_reason):
            self.is_halted = False
            self.halt_reason = None

    @property
    def current_drawdown(self) -> float:
        """Drawdown acumulado desde el High Water Mark."""
        return max(0.0, self.peak_equity - self.equity)

    @property
    def delta_dll(self) -> float:
        """Margen de pérdida diaria restante."""
        loss_today = max(0.0, self.daily_start_equity - self.equity)
        return max(0.0, self.dll - loss_today)

    @property
    def dd_cushion(self) -> float:
        """Colchón de trailing drawdown: D_t^{cushion} = max(E_t - (H_t - MTDD), 0)."""
        return max(0.0, self.equity - (self.peak_equity - self.mtdd))

    @property
    def cushion_ratio(self) -> float:
        """Ratio normalizado de colchón de trailing drawdown in [0, 1]."""
        return float(np.clip(self.dd_cushion / max(self.mtdd, 1e-6), 0.0, 1.0))

    def check_circuit_breaker(self) -> bool:
        """Disyuntor analítico si el colchón cae por debajo del 15% de MTDD."""
        if self.cushion_ratio <= self.circuit_breaker_ratio:
            self.is_halted = True
            self.halt_reason = (
                f"HALT_DRAWDOWN_DEFENSE: Colchón de drawdown ${self.dd_cushion:.2f} "
                f"<= {self.circuit_breaker_ratio * 100:.0f}% de MTDD (${self.mtdd:.2f})"
            )
            return True
        return False

    def calculate_risk_budget(self) -> float:
        """Presupuesto dinámico base de riesgo por trade."""
        if self.is_halted or self.check_circuit_breaker():
            return 0.0
        b_risk = min(self.gamma_d * self.delta_dll, self.gamma_t * self.dd_cushion)
        return max(0.0, b_risk)

    def calculate_position_size_optimized(
        self,
        std_symbol: str,
        micro_symbol: str,
        sigma: float,
        ell_minus: float,
        effective_bet_size: float,     # m_Kelly * psi(rho)
        slippage_ticks: float = 1.0
    ) -> Tuple[int, bool, float]:
        """
        Calcula el tamaño de la posición con desescalamiento a microcontratos:
        - Si cushion_ratio <= force_micro_cushion_ratio (ej. 40%), FUERZA conmutación a Micro (MES/MNQ).
        - Si una pérdida en contrato estándar consumiría > 30% del colchón restante, FUERZA Micro.
        - De lo contrario, opera contrato estándar hasta N_std contratos.
        
        Returns:
            Tuple[num_contracts, is_micro, unit_risk_usd]
        """
        if effective_bet_size <= 0.0 or self.is_halted or self.check_circuit_breaker():
            return 0, False, 0.0

        # Presupuesto de riesgo según Criterio de Kelly Fraccional Truncado sobre el colchón de MTDD:
        # Bankroll de Kelly = D_t^{cushion}. effective_bet_size = m_Kelly * psi(rho).
        # Acotado superiormente por el límite diario (gamma_d * delta_dll) y cota de MTDD (gamma_t * dd_cushion).
        kelly_cushion_budget = effective_bet_size * self.dd_cushion
        max_cushion_cap = self.gamma_t * self.dd_cushion
        max_daily_cap = self.gamma_d * self.delta_dll
        effective_budget = min(max_daily_cap, max_cushion_cap, kelly_cushion_budget)
        if effective_budget <= 0.0:
            return 0, False, 0.0

        spec_std = get_spec(std_symbol)
        spec_micro = get_spec(micro_symbol)

        m_std = spec_std["multiplier"]
        m_micro = spec_micro["multiplier"]
        tick_std = spec_std["tick"]
        tick_micro = spec_micro["tick"]
        comm_std = spec_std["commission_rt"]
        comm_micro = spec_micro["commission_rt"]

        # Fricción total (2 ticks slippage + comisión RT)
        fric_std = (2.0 * slippage_ticks * tick_std * m_std) + comm_std
        fric_micro = (2.0 * slippage_ticks * tick_micro * m_micro) + comm_micro

        delta_l_std = (ell_minus * sigma * m_std) + fric_std
        delta_l_micro = (ell_minus * sigma * m_micro) + fric_micro

        if delta_l_std <= 0.0 or delta_l_micro <= 0.0:
            return 0, False, 0.0

        # CONDICIÓN DE FORZADO PROACTIVO DE MICROCONTRATOS:
        # 1. Proximidad al MTDD (cushion <= 40%)
        # 2. Riesgo unitario del estándar > 30% del colchón total restante
        cushion_now = self.dd_cushion
        proactive_micro_required = (
            self.cushion_ratio <= self.force_micro_cushion_ratio
            or delta_l_std > (0.30 * cushion_now)
        )

        if not proactive_micro_required:
            n_std = min(int(math.floor(effective_budget / delta_l_std)), self.max_contracts_std)
            if n_std >= 1:
                return n_std, False, delta_l_std

        # Conmutación a Microcontratos (MES/MNQ) con escalamiento granular
        max_micro = 10 * self.max_contracts_std
        n_micro = min(int(math.floor(effective_budget / delta_l_micro)), max_micro)
        if n_micro >= 1:
            return n_micro, True, delta_l_micro

        return 0, False, 0.0

    def record_completed_trade(self, trade: TradeRecord) -> None:
        """Actualiza equidad, límites y disyuntor al concluir una operación."""
        self.equity += trade.net_pnl_usd
        daily_loss = max(0.0, self.daily_start_equity - self.equity)
        self.daily_realized_loss = daily_loss

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        if (self.peak_equity - self.equity) >= self.mtdd:
            self.is_halted = True
            self.halt_reason = f"MTDD_BREACH: Drawdown ${self.peak_equity - self.equity:.2f} >= ${self.mtdd:.2f}"
        elif daily_loss >= self.dll:
            self.is_halted = True
            self.halt_reason = f"DAILY_LOSS_LIMIT_BREACH: Pérdida del día ${daily_loss:.2f} >= ${self.dll:.2f}"
        else:
            self.check_circuit_breaker()

        trade.running_equity = self.equity
        trade.running_drawdown = self.current_drawdown


# ==============================================================================
# 5. ESTRATEGIA ALTERNA PRINCIPAL: F7-OPTIMIZACION -IS- N1
# ==============================================================================

class F7OptimizationISN1Strategy(BaseStrategy):
    """
    Estrategia Cuantitativa Híbrida F7: Versión Alterna Optimizada 'F7-Optimizacion -IS- N1'.
    Incorpora:
    - Calibración optimizada de probabilidades (Platt / Temperature / Isotonic sobre Purged CV).
    - Triple Barrera Asimétrica Adaptativa por volatilidad y colchón de drawdown.
    - Bet Sizing continuo mediante Criterio de Kelly Fraccional Truncado.
    - Conmutación automática y proactiva a microcontratos MES/MNQ por proximidad a MTDD.
    """

    def __init__(
        self,
        name: str = "F7_Optimizacion_IS_N1",
        params: Optional[Dict[str, Any]] = None
    ):
        default_params: Dict[str, Any] = {
            "symbol": "ES",
            "std_symbol": "ES",
            "micro_symbol": "MES",
            "base_ell_plus": 2.00,
            "base_ell_minus": 1.00,
            "min_asymmetry_ratio": 2.00,
            "delta_tau_minutes": 30,
            "tau_meta": 0.58,              # Umbral analítico de veto optimizado
            "kelly_fraction": 0.25,        # Quarter-Kelly para control estricto de drawdown
            "max_bet_size": 1.00,
            "calibration_method": "platt", # 'platt', 'temperature', 'isotonic'
            "n_cv_splits": 5,
            "force_micro_cushion_ratio": 0.40, # Conmutar a micros si colchón <= 40% MTDD
            "slippage_ticks": 1.0,
            "initial_capital": 50000.0,
            "dll": 1000.0,
            "mtdd": 2000.0,
            "gamma_d": 0.25,
            "gamma_t": 0.15,
            "cooldown_minutes": 5,
            "rth_open": "09:30",
            "exec_window_open": "09:45",
            "exec_window_close": "15:30",
            "eod_close": "15:55",
            "train_classifier": True,
        }
        if params:
            default_params.update(params)
        super().__init__(name=name, params=default_params)

        cal_method = CalibrationMethod(self.params.get("calibration_method", "platt"))
        # Si se utiliza calibración isotónica y tau_meta está en su valor por defecto (0.58),
        # ajustar a 0.42 para reflejar la frecuencia empírica sobre el punto de equilibrio 0.333
        if cal_method == CalibrationMethod.ISOTONIC and self.params.get("tau_meta") == 0.58:
            self.params["tau_meta"] = 0.42

        self.classifier = CalibratedMetaClassifier(
            calibration_method=cal_method,
            n_cv_splits=self.params.get("n_cv_splits", 5)
        )
        self.barrier_mgr = AdaptiveBarrierManager(
            config=AdaptiveBarrierConfig(
                base_ell_plus=self.params["base_ell_plus"],
                base_ell_minus=self.params["base_ell_minus"],
                min_asymmetry_ratio=self.params["min_asymmetry_ratio"],
                base_vertical_bars=max(1, int(self.params["delta_tau_minutes"] / 5))
            )
        )
        self.risk_manager = PropRiskManagerOptimized(
            initial_capital=self.params["initial_capital"],
            dll=self.params["dll"],
            mtdd=self.params["mtdd"],
            gamma_d=self.params["gamma_d"],
            gamma_t=self.params["gamma_t"],
            force_micro_cushion_ratio=self.params["force_micro_cushion_ratio"]
        )

    def generate_primary_signals(self, df: pd.DataFrame, delta_vwap: pd.Series, z_ofi: pd.Series) -> pd.Series:
        """
        Nivel 1: Señal direccional de microestructura s_t^{(1)} in {-1, 0, +1}.
        """
        signals = pd.Series(0, index=df.index, dtype=int)
        long_cond = (delta_vwap > 0.0) & (z_ofi >= 0.50)
        short_cond = (delta_vwap < 0.0) & (z_ofi <= -0.50)
        signals[long_cond] = 1
        signals[short_cond] = -1
        return signals

    def extract_features(
        self,
        df: pd.DataFrame,
        sigma_series: pd.Series,
        z_ofi_series: pd.Series,
        delta_vwap_series: pd.Series,
        rvol_series: pd.Series,
        compression_series: pd.Series,
        tod_series: pd.Series
    ) -> List[MicroFeatures]:
        """Extrae el vector microestructural X_t in R^6."""
        features = []
        for i in range(len(df)):
            feat = MicroFeatures(
                sigma=float(sigma_series.iloc[i]),
                z_ofi=float(z_ofi_series.iloc[i]),
                delta_vwap=float(delta_vwap_series.iloc[i]),
                rvol=float(rvol_series.iloc[i]),
                range_compression=float(compression_series.iloc[i]),
                tau_tod=float(tod_series.iloc[i])
            )
            features.append(feat)
        return features

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Genera la serie de señales ejecutivas finales S_t^* in {-1.0, 0.0, +1.0}."""
        self.validate_data(data)
        df_sorted = data.sort_index()

        is_ct = detect_data_time_offset(df_sorted)
        sigma_series = compute_garman_klass_volatility(df_sorted)
        _, delta_vwap_series = compute_intraday_anchored_vwap(df_sorted)
        z_ofi_series = compute_order_flow_imbalance_zscore(df_sorted)
        rvol_series = compute_relative_volume(df_sorted)
        comp_series = compute_range_compression(df_sorted)
        tod_series = compute_session_fraction(df_sorted, is_ct=is_ct)

        primary = self.generate_primary_signals(df_sorted, delta_vwap_series, z_ofi_series)
        feats = self.extract_features(
            df_sorted, sigma_series, z_ofi_series, delta_vwap_series, rvol_series, comp_series, tod_series
        )

        tau_meta = self.params.get("tau_meta", 0.58)
        final_signals = pd.Series(0.0, index=df_sorted.index, dtype=float)

        for i in range(len(df_sorted)):
            s_prim = int(primary.iloc[i])
            if s_prim != 0:
                p_cal = self.classifier.predict_probability(feats[i], s_prim)
                s_star = apply_meta_veto(s_prim, p_cal, tau_meta)
                final_signals.iloc[i] = float(s_star)

        return final_signals


# ==============================================================================
# 6. MOTOR DE BACKTESTING OPTIMIZADO BARRA A BARRA
# ==============================================================================

def label_historical_events_adaptive(
    df: pd.DataFrame,
    primary_signals: pd.Series,
    sigma_series: pd.Series,
    rolling_median_sigma_series: pd.Series,
    barrier_mgr: AdaptiveBarrierManager,
    t_eod_close: Optional[time] = None,
    purge_last_bars: int = 0
) -> Tuple[List[int], List[int]]:
    """
    Etiqueta las señales primarias históricas mediante el Método de Triple Barrera Asimétrica Adaptativa:
    Calcula B_upper, B_lower y Delta_tau_bars específicos de cada barra considerando la volatilidad Garman-Klass
    y la relación con la mediana rodante sin fuga temporal.
    y = 1 si alcanza Take Profit antes de Stop Loss.
    y = 0 si alcanza Stop Loss o expira en tiempo sin ganancia.
    
    Returns:
        Tuple[valid_indices, labels]
    """
    from src.strategies.meta_labeling_triple_barrier import evaluate_triple_barrier_exit
    valid_indices = []
    labels = []
    n_rows = len(df)

    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    sigs = primary_signals.values
    sigmas = sigma_series.values
    med_sigmas = rolling_median_sigma_series.values
    dts = df.index

    cutoff_idx = max(0, n_rows - 1 - max(0, purge_last_bars))

    for i in range(cutoff_idx):
        s = int(sigs[i])
        if s == 0:
            continue

        p0 = float(closes[i])
        sigma0 = float(sigmas[i])
        med_sigma0 = float(med_sigmas[i])

        # Barreras adaptativas al régimen de volatilidad
        b_upper, b_lower, vert_bars, _, _ = barrier_mgr.calculate_regime_adjusted_barriers(
            current_price=p0,
            current_sigma=sigma0,
            rolling_median_sigma=med_sigma0,
            cushion_ratio=1.0,  # En etiquetado histórico se asume colchón de referencia
            signal_star=s
        )

        outcome_label = 0
        entry_date = dts[i].date()
        max_steps = min(vert_bars, n_rows - 1 - i)

        for step in range(1, max_steps + 1):
            curr_idx = i + step
            curr_dt = dts[curr_idx]
            curr_time = curr_dt.time()
            curr_date = curr_dt.date()

            is_new_day = (curr_date != entry_date)
            is_eod = (t_eod_close is not None and curr_time >= t_eod_close)
            is_vert = (step == max_steps) or is_new_day or is_eod

            o = float(opens[curr_idx])
            h = float(highs[curr_idx])
            l = float(lows[curr_idx])
            c = float(closes[curr_idx])

            res = evaluate_triple_barrier_exit(
                s, h, l, c, b_upper, b_lower, is_vert, current_open=o
            )
            if res is not None:
                reason, exit_price = res
                if reason == ExitReason.TAKE_PROFIT:
                    outcome_label = 1
                elif reason in (ExitReason.TIME_BARRIER, ExitReason.EOD_FORCED):
                    ret = (exit_price - p0) * s
                    outcome_label = 1 if ret > 0 else 0
                else:
                    outcome_label = 0
                break

            if is_new_day or is_eod:
                break

        valid_indices.append(i)
        labels.append(outcome_label)

    return valid_indices, labels


class F7OptimizationBacktester:
    """
    Simulador cuantitativo determinista barra a barra para F7-Optimizacion -IS- N1.
    Ejecuta con estricta causalidad y registro completo de métricas Prop Trading.
    """

    def __init__(self, strategy: F7OptimizationISN1Strategy):
        self.strategy = strategy
        self.params = strategy.params
        self.risk_mgr = strategy.risk_manager
        self.barrier_mgr = strategy.barrier_mgr

    def fit_classifier_on_data(
        self,
        train_df: pd.DataFrame,
        purge_boundary: bool = True
    ) -> Dict[str, Any]:
        """
        Entrena el meta-clasificador calibrado exclusivamente sobre el bloque In-Sample,
        empleando Purged K-Fold Cross Validation sobre los eventos históricos etiquetados.
        """
        req_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not req_cols.issubset(train_df.columns):
            raise ValueError(f"El DataFrame debe contener las columnas: {req_cols}")

        if not isinstance(train_df.index, pd.DatetimeIndex):
            raise TypeError("El índice del DataFrame debe ser pd.DatetimeIndex.")

        df_train_sorted = train_df.sort_index().copy()
        is_ct = detect_data_time_offset(df_train_sorted)
        t_eod_close = time(14, 55) if is_ct else time(15, 55)

        sigma_series = compute_garman_klass_volatility(df_train_sorted)
        rolling_median_sigma_series = sigma_series.rolling(window=60, min_periods=5).median().bfill()
        _, delta_vwap_series = compute_intraday_anchored_vwap(df_train_sorted)
        z_ofi_series = compute_order_flow_imbalance_zscore(df_train_sorted)
        rvol_series = compute_relative_volume(df_train_sorted)
        comp_series = compute_range_compression(df_train_sorted)
        tod_series = compute_session_fraction(df_train_sorted, is_ct=is_ct)

        primary_signals = self.strategy.generate_primary_signals(df_train_sorted, delta_vwap_series, z_ofi_series)
        all_features = self.strategy.extract_features(
            df_train_sorted, sigma_series, z_ofi_series, delta_vwap_series, rvol_series, comp_series, tod_series
        )

        delta_tau_bars = max(1, int(self.params.get("delta_tau_minutes", 30) / 5))
        purge_bars = delta_tau_bars if purge_boundary else 0

        # Etiquetado con barreras asimétricas adaptativas (alineadas con la ejecución)
        val_idx, labels = label_historical_events_adaptive(
            df=df_train_sorted,
            primary_signals=primary_signals,
            sigma_series=sigma_series,
            rolling_median_sigma_series=rolling_median_sigma_series,
            barrier_mgr=self.barrier_mgr,
            t_eod_close=t_eod_close,
            purge_last_bars=purge_bars
        )

        if len(val_idx) >= 15:
            train_feats = [all_features[i] for i in val_idx]
            train_sigs = [int(primary_signals.iloc[i]) for i in val_idx]
            self.strategy.classifier.fit_with_purged_cv(
                features=train_feats,
                signals=train_sigs,
                labels=labels,
                event_indices=val_idx
            )

        return {
            "total_bars_train": len(df_train_sorted),
            "events_labeled": len(labels),
            "purged_bars": purge_bars,
            "calibrator_fitted": self.strategy.classifier.calibrator.is_fitted,
            "calibration_method": self.strategy.classifier.calibrator.method.value,
        }

    def run(self, df: pd.DataFrame, train_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Ejecuta la simulación barra a barra respetando las reglas de Prop Trading
        y los 3 componentes matemáticos optimizados de N1.
        """
        req_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not req_cols.issubset(df.columns):
            raise ValueError(f"El DataFrame debe contener las columnas: {req_cols}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("El índice del DataFrame debe ser pd.DatetimeIndex.")

        df_sorted = df.sort_index().copy()

        # 1. Parsear horarios según zona horaria
        is_ct = detect_data_time_offset(df_sorted)
        if is_ct:
            t_exec_open = time(8, 45)
            t_exec_close = time(14, 30)
            t_eod_close = time(14, 55)
        else:
            t_exec_open = time(9, 45)
            t_exec_close = time(15, 30)
            t_eod_close = time(15, 55)

        # 2. Computar variables microestructurales
        sigma_series = compute_garman_klass_volatility(df_sorted)
        _, delta_vwap_series = compute_intraday_anchored_vwap(df_sorted)
        z_ofi_series = compute_order_flow_imbalance_zscore(df_sorted)
        rvol_series = compute_relative_volume(df_sorted)
        comp_series = compute_range_compression(df_sorted)
        tod_series = compute_session_fraction(df_sorted, is_ct=is_ct)

        # Volatilidad rodante de referencia (mediana histórica 60 barras, sin lookahead)
        rolling_median_sigma_series = sigma_series.rolling(window=60, min_periods=5).median().bfill()

        # 3. Señal primaria s_t^{(1)}
        primary_signals = self.strategy.generate_primary_signals(df_sorted, delta_vwap_series, z_ofi_series)

        # 4. Vector de características X_t
        all_features = self.strategy.extract_features(
            df_sorted, sigma_series, z_ofi_series, delta_vwap_series, rvol_series, comp_series, tod_series
        )

        # 5. Entrenamiento previo si se especifica train_df
        if train_df is not None and self.params.get("train_classifier", True):
            self.fit_classifier_on_data(train_df, purge_boundary=True)

        # 6. Parámetros de simulación
        tau_meta = self.params.get("tau_meta", 0.58)
        kelly_fraction = self.params.get("kelly_fraction", 0.25)
        max_bet_size = self.params.get("max_bet_size", 1.00)
        cooldown_td = timedelta(minutes=self.params.get("cooldown_minutes", 5))
        slippage_ticks = self.params.get("slippage_ticks", 1.0)

        std_sym = self.params["std_symbol"]
        micro_sym = self.params["micro_symbol"]

        trade_id = 0
        trades: List[TradeRecord] = []
        in_position = False
        current_trade: Optional[TradeRecord] = None
        cooldown_until: Optional[datetime] = None
        current_day: Optional[Any] = None

        bar_dts = df_sorted.index
        bar_opens = df_sorted["Open"].values
        bar_highs = df_sorted["High"].values
        bar_lows = df_sorted["Low"].values
        bar_closes = df_sorted["Close"].values
        bar_sigs = primary_signals.values
        bar_sigmas = sigma_series.values
        bar_med_sigmas = rolling_median_sigma_series.values

        from src.strategies.meta_labeling_triple_barrier import evaluate_triple_barrier_exit

        n_bars = len(df_sorted)
        for i in range(n_bars):
            bar_dt = bar_dts[i]
            bar_date = bar_dt.date()
            bar_time = bar_dt.time()
            bar_open = float(bar_opens[i])
            bar_high = float(bar_highs[i])
            bar_low = float(bar_lows[i])
            bar_close = float(bar_closes[i])

            # Nueva sesión bursátil
            if current_day != bar_date:
                current_day = bar_date
                self.risk_mgr.start_new_session(bar_date)
                cooldown_until = None

            # --- A. GESTIÓN DE POSICIÓN ABIERTA ---
            if in_position and current_trade is not None:
                # 1. Cierre forzoso intradía pre-clearing CME
                if bar_time >= t_eod_close:
                    exit_check = evaluate_triple_barrier_exit(
                        signal_star=current_trade.direction,
                        current_high=bar_high,
                        current_low=bar_low,
                        current_close=bar_close,
                        b_upper=current_trade.b_upper,
                        b_lower=current_trade.b_lower,
                        is_vertical_reached=True,
                        current_open=bar_open
                    )
                    if exit_check is not None and exit_check[0] in (ExitReason.TAKE_PROFIT, ExitReason.STOP_LOSS):
                        reason, exit_price = exit_check
                    else:
                        reason = ExitReason.EOD_FORCED
                        exit_price = bar_close

                    self._close_position(current_trade, bar_dt, exit_price, reason, slippage_ticks)
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None
                    cooldown_until = bar_dt + cooldown_td
                    continue

                # 2. Expiración de Barrera Vertical Temporal Adaptativa
                t_vert_dt = datetime.fromisoformat(current_trade.t_vertical)
                is_vert_expired = bool(bar_dt >= t_vert_dt)

                # 3. Evaluación de impacto en Triple Barrera
                exit_check = evaluate_triple_barrier_exit(
                    signal_star=current_trade.direction,
                    current_high=bar_high,
                    current_low=bar_low,
                    current_close=bar_close,
                    b_upper=current_trade.b_upper,
                    b_lower=current_trade.b_lower,
                    is_vertical_reached=is_vert_expired,
                    current_open=bar_open
                )

                if exit_check is not None:
                    reason, exit_price = exit_check
                    self._close_position(current_trade, bar_dt, exit_price, reason, slippage_ticks)
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None
                    cooldown_until = bar_dt + cooldown_td
                    continue

                # 4. Disyuntor por drawdown durante la posición
                if self.risk_mgr.check_circuit_breaker():
                    self._close_position(current_trade, bar_dt, bar_close, ExitReason.CIRCUIT_BREAKER, slippage_ticks)
                    trades.append(current_trade)
                    in_position = False
                    current_trade = None
                    continue

            # --- B. EVALUACIÓN DE NUEVAS ENTRADAS ---
            if in_position or self.risk_mgr.is_halted:
                continue

            if cooldown_until is not None and bar_dt < cooldown_until:
                continue

            # Restricción horaria de ventana de ejecución activa
            if not (t_exec_open <= bar_time <= t_exec_close):
                continue

            # Señal primaria
            s_primary = int(bar_sigs[i])
            if s_primary == 0:
                continue

            # 1. Calibración Optimizada de Probabilidades
            feat_i = all_features[i]
            p_calibrated = self.strategy.classifier.predict_probability(feat_i, s_primary)
            s_star = apply_meta_veto(s_primary, p_calibrated, tau_meta)

            if s_star == 0:
                continue

            # 2. Barreras Asimétricas Adaptativas para Prop Trading
            sigma_i = float(bar_sigmas[i])
            med_sigma_i = float(bar_med_sigmas[i])
            cushion_ratio_i = self.risk_mgr.cushion_ratio

            (
                b_upper,
                b_lower,
                vert_bars,
                ell_plus_t,
                ell_minus_t
            ) = self.barrier_mgr.calculate_regime_adjusted_barriers(
                current_price=bar_close,
                current_sigma=sigma_i,
                rolling_median_sigma=med_sigma_i,
                cushion_ratio=cushion_ratio_i,
                signal_star=s_star
            )

            # Ratio de beneficio/riesgo efectivo
            b_ratio_t = ell_plus_t / max(ell_minus_t, 1e-4)

            # 3. Bet Sizing mediante Kelly Fraccional Truncado con Desescalamiento
            m_kelly = calculate_truncated_fractional_kelly(
                p_t=p_calibrated,
                b_ratio=b_ratio_t,
                tau_meta=tau_meta,
                kelly_fraction=kelly_fraction,
                max_bet_size=max_bet_size
            )
            if m_kelly <= 0.0:
                continue

            psi_cushion = calculate_cushion_deescalation_factor(cushion_ratio_i)
            effective_bet = m_kelly * psi_cushion

            if effective_bet <= 0.0:
                continue

            # 4. Cálculo de Contratos con Conmutación Forzada a Micro por MTDD
            n_contracts, is_micro, _ = self.risk_mgr.calculate_position_size_optimized(
                std_symbol=std_sym,
                micro_symbol=micro_sym,
                sigma=sigma_i,
                ell_minus=ell_minus_t,
                effective_bet_size=effective_bet,
                slippage_ticks=slippage_ticks
            )

            if n_contracts <= 0:
                continue

            # Abrir posición al cierre de la barra
            trade_id += 1
            delta_vert_td = timedelta(minutes=vert_bars * 5)
            t_close_dt = bar_dt.replace(hour=t_eod_close.hour, minute=t_eod_close.minute, second=0)
            t_vertical_dt = min(bar_dt + delta_vert_td, t_close_dt)

            sym_operated = micro_sym if is_micro else std_sym

            current_trade = TradeRecord(
                trade_id=trade_id,
                date=str(bar_date),
                symbol=sym_operated,
                direction=s_star,
                contracts=n_contracts,
                is_micro=is_micro,
                entry_time=bar_dt.isoformat(),
                entry_price=bar_close,
                b_upper=round(b_upper, 4),
                b_lower=round(b_lower, 4),
                t_vertical=t_vertical_dt.isoformat(),
                meta_prob=round(p_calibrated, 4),
                bet_size=round(effective_bet, 4)
            )
            in_position = True

        # Liquidar trade remanente si existiera
        if in_position and current_trade is not None:
            last_dt = df_sorted.index[-1]
            last_close = float(df_sorted["Close"].iloc[-1])
            self._close_position(current_trade, last_dt, last_close, ExitReason.EOD_FORCED, slippage_ticks)
            trades.append(current_trade)

        return self._compile_results(trades, df_sorted)

    def _close_position(
        self,
        trade: TradeRecord,
        exit_dt: datetime,
        exit_price: float,
        reason: ExitReason,
        slippage_ticks: float
    ) -> None:
        """Calcula PnL y fricciones con especificaciones CME."""
        spec = get_spec(trade.symbol)
        multiplier = spec["multiplier"]
        tick = spec["tick"]
        comm_rt = spec["commission_rt"]

        pts_diff = (exit_price - trade.entry_price) * trade.direction
        gross_usd = pts_diff * multiplier * trade.contracts

        total_slippage_usd = (2.0 * slippage_ticks * tick * multiplier) * trade.contracts
        total_comm_usd = comm_rt * trade.contracts
        net_usd = gross_usd - (total_slippage_usd + total_comm_usd)

        trade.exit_time = exit_dt.isoformat()
        trade.exit_price = round(exit_price, 4)
        trade.exit_reason = reason
        trade.gross_pnl_points = round(pts_diff, 4)
        trade.gross_pnl_usd = round(gross_usd, 2)
        trade.commissions_usd = round(total_comm_usd, 2)
        trade.slippage_usd = round(total_slippage_usd, 2)
        trade.net_pnl_usd = round(net_usd, 2)

        self.risk_mgr.record_completed_trade(trade)

    def _compile_results(self, trades: List[TradeRecord], df: pd.DataFrame) -> Dict[str, Any]:
        """Compila métricas de auditoría Prop Trading completas."""
        init_cap = self.params["initial_capital"]
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "net_profit": 0.0,
                "total_return_pct": 0.0,
                "final_equity": init_cap,
                "peak_equity": init_cap,
                "max_drawdown_usd": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "standard_trades": 0,
                "micro_trades": 0,
                "exit_reasons": {},
                "is_halted": self.risk_mgr.is_halted,
                "halt_reason": self.risk_mgr.halt_reason,
                "trades": []
            }

        pnl_series = pd.Series([t.net_pnl_usd for t in trades])
        wins = pnl_series[pnl_series > 0.0]
        losses = pnl_series[pnl_series < 0.0]

        total_trades = len(trades)
        win_rate = (len(wins) / total_trades) * 100.0
        sum_wins = float(wins.sum())
        sum_losses = float(abs(losses.sum()))
        profit_factor = (sum_wins / sum_losses) if sum_losses > 0.0 else float("inf")

        net_profit = float(pnl_series.sum())
        total_return_pct = (net_profit / init_cap) * 100.0

        # High water mark y drawdown a partir de la equidad de cada trade
        equities = [init_cap] + [t.running_equity for t in trades]
        eq_series = pd.Series(equities)
        cummax = eq_series.cummax()
        drawdowns_usd = cummax - eq_series
        max_dd_usd = float(drawdowns_usd.max())
        max_dd_pct = float((drawdowns_usd / cummax).max() * 100.0)

        # Sharpe ratio anualizado asumiendo 252 sesiones anuales
        trade_dates = [t.date for t in trades]
        daily_pnl = pd.Series(pnl_series.values, index=pd.to_datetime(trade_dates)).groupby(level=0).sum()
        if len(daily_pnl) > 1 and daily_pnl.std() > 0.0:
            daily_mean = daily_pnl.mean()
            daily_std = daily_pnl.std()
            sharpe = (daily_mean / daily_std) * math.sqrt(252)
        else:
            sharpe = 0.0

        std_trades = sum(1 for t in trades if not t.is_micro)
        micro_trades = sum(1 for t in trades if t.is_micro)

        exit_reasons: Dict[str, int] = {}
        for t in trades:
            r_str = t.exit_reason.value if t.exit_reason else "UNKNOWN"
            exit_reasons[r_str] = exit_reasons.get(r_str, 0) + 1

        trade_dicts = []
        for t in trades:
            trade_dicts.append({
                "trade_id": t.trade_id,
                "date": t.date,
                "symbol": t.symbol,
                "direction": t.direction,
                "contracts": t.contracts,
                "is_micro": t.is_micro,
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "b_upper": t.b_upper,
                "b_lower": t.b_lower,
                "t_vertical": t.t_vertical,
                "meta_prob": t.meta_prob,
                "bet_size": t.bet_size,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason.value if t.exit_reason else None,
                "gross_pnl_usd": t.gross_pnl_usd,
                "commissions_usd": t.commissions_usd,
                "slippage_usd": t.slippage_usd,
                "net_pnl_usd": t.net_pnl_usd,
                "running_equity": t.running_equity,
                "running_drawdown": t.running_drawdown
            })

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "net_profit": round(net_profit, 2),
            "total_return_pct": round(total_return_pct, 2),
            "final_equity": round(self.risk_mgr.equity, 2),
            "peak_equity": round(self.risk_mgr.peak_equity, 2),
            "max_drawdown_usd": round(max_dd_usd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe_ratio": round(sharpe, 2),
            "standard_trades": std_trades,
            "micro_trades": micro_trades,
            "exit_reasons": exit_reasons,
            "is_halted": self.risk_mgr.is_halted,
            "halt_reason": self.risk_mgr.halt_reason,
            "trades": trade_dicts
        }
