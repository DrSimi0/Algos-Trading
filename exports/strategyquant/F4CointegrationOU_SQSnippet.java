package com.strategyquant.extend.Indicators;

import com.strategyquant.lib.indicator.Indicator;
import com.strategyquant.lib.data.DataBlock;
import com.strategyquant.lib.data.DataList;

/**
 * StrategyQuant X - Custom Snippet: F4 Ornstein-Uhlenbeck S-Score Indicator
 * 
 * Este indicador calcula el S-Score de reversión a la media bajo el modelo
 * Ornstein-Uhlenbeck (OU) para StrategyQuant X (Code Editor -> New Indicator / Snippet).
 * 
 * Se aplica sobre el instrumento sintético SPREAD_NQ_ES (o cualquier spread de pares).
 * 
 * Fórmula:
 *   dx_t = kappa * (theta - x_{t-1}) * dt + sigma * dz
 *   S-Score = (Spread_t - theta) / sigma_eq
 *   donde sigma_eq = sigma / sqrt(2 * kappa)
 */
public class F4CointegrationOU_SQSnippet {

    // Parámetros configurables
    private int lookbackBars = 60;        // Ventana de calibración (ej. 60 barras de 5 min)
    private double dt = 1.0 / 252.0;       // Intervalo de tiempo base
    private double entryThreshold = 1.25;  // Umbral de entrada (|S-Score| >= 1.25)
    private double exitThreshold = 0.50;   // Umbral de toma de ganancias (|S-Score| <= 0.50)
    private double stopThreshold = 3.00;   // Umbral de Stop Loss de seguridad (|S-Score| >= 3.00)
    private double sigmaFloor = 0.001;     // Guardarraíl: Piso de volatilidad mínima (sigma_floor >= 0.001)

    /**
     * Calcula el S-Score de Ornstein-Uhlenbeck en la barra actual.
     * 
     * @param spreadPrices Array con la serie de precios de cierre del Spread
     * @param currentIndex Índice de la barra actual
     * @return S-Score normalizado. Valores típicos: entre -3.0 y +3.0
     */
    public double calculateSScore(double[] spreadPrices, int currentIndex) {
        if (currentIndex < lookbackBars || spreadPrices.length <= currentIndex) {
            return 0.0;
        }

        // 1. Regresión lineal autoregresiva AR(1): x_t = a * x_{t-1} + b + epsilon
        double sumX = 0.0;
        double sumY = 0.0;
        double sumXY = 0.0;
        double sumX2 = 0.0;
        int n = lookbackBars;
        int start = currentIndex - lookbackBars + 1;

        for (int i = 0; i < n; i++) {
            double x_prev = spreadPrices[start + i - 1];
            double x_curr = spreadPrices[start + i];
            sumX += x_prev;
            sumY += x_curr;
            sumXY += x_prev * x_curr;
            sumX2 += x_prev * x_prev;
        }

        double denominator = (n * sumX2 - sumX * sumX);
        if (Math.abs(denominator) < 1e-12) {
            return 0.0;
        }

        double a = (n * sumXY - sumX * sumY) / denominator;
        double b = (sumY - a * sumX) / n;

        // Condición de reversión a la media: 0 < a < 1
        if (a <= 0.0001 || a >= 0.9999) {
            // No hay reversión a la media estable
            return 0.0;
        }

        // 2. Parámetros Ornstein-Uhlenbeck
        double kappa = -Math.log(a) / dt;
        double theta = b / (1.0 - a);

        // Varianza de los residuos
        double sumResid2 = 0.0;
        for (int i = 0; i < n; i++) {
            double x_prev = spreadPrices[start + i - 1];
            double x_curr = spreadPrices[start + i];
            double pred = a * x_prev + b;
            double res = x_curr - pred;
            sumResid2 += res * res;
        }
        double varResidual = sumResid2 / (n - 2);
        if (varResidual <= 0.0) {
            return 0.0;
        }

        double sigma = Math.sqrt(varResidual * 2.0 * kappa / (1.0 - Math.pow(a, 2.0)));
        double sigma_eq_raw = Math.sqrt(varResidual / (1.0 - Math.pow(a, 2.0)));

        // Guardarraíl: Imponer piso de volatilidad para evitar división por micro-varianza
        double sigma_eq = Math.max(sigma_eq_raw, sigmaFloor);

        if (sigma_eq <= 1e-8) {
            return 0.0;
        }

        // 3. S-Score actual
        double currentSpread = spreadPrices[currentIndex];
        double sScore = (currentSpread - theta) / sigma_eq;

        return sScore;
    }

    /**
     * Señal de trading generada:
     *  +1 : COMPRAR Spread (S-Score <= -entryThreshold)
     *  -1 : VENDER Spread (S-Score >= +entryThreshold)
     *   0 : CERRAR posición o mantener neutral
     */
    public int getTradingSignal(double[] spreadPrices, int currentIndex, int currentPosition) {
        double sScore = calculateSScore(spreadPrices, currentIndex);

        // Señal de Entrada
        if (currentPosition == 0) {
            if (sScore <= -entryThreshold && sScore > -stopThreshold) {
                return 1; // Long Spread
            } else if (sScore >= entryThreshold && sScore < stopThreshold) {
                return -1; // Short Spread
            }
        } 
        // Señal de Salida (Take Profit o Stop Loss por divergencia)
        else if (currentPosition == 1) { // Estábamos Comprados
            if (sScore >= -exitThreshold || sScore <= -stopThreshold) {
                return 0; // Cerrar posición Long
            }
        } else if (currentPosition == -1) { // Estábamos Vendidos
            if (sScore <= exitThreshold || sScore >= stopThreshold) {
                return 0; // Cerrar posición Short
            }
        }

        return currentPosition; // Mantener estado
    }
}
