import numpy as np
import pandas as pd

class PHMMetrics:
    """Clase especializada en el cálculo y análisis de métricas asimétricas

    para problemas de Prognostics and Health Management (PHM), orientada
    al conjunto de datos C-MAPSS de la NASA.
    """

    @staticmethod
    def nasa_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Calcula el NASA Score (función de penalización asimétrica)

        para un conjunto de predicciones de RUL.

        Parámetros:
        -----------
        y_true : np.ndarray
            Vectores con los valores reales de RUL.
        y_pred : np.ndarray
            Vectores con los valores predichos de RUL.

        Retorna:
        --------
        float : El Score acumulado de la NASA (S).
        """
        # Convertir a arreglos de numpy y aplanar
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

        # d > 0 significa sobreestimación (Peligroso)
        # d < 0 significa subestimación (Seguro pero costoso)
        d = y_pred - y_true

        # Aplicar las funciones exponenciales asimétricas según el signo de d
        # s_i = exp(-d/13) - 1  si d < 0
        # s_i = exp(d/10) - 1   si d >= 0
        penalizaciones = np.where(
            d < 0,
            np.exp(-d / 13.0) - 1.0,
            np.exp(d / 10.0) - 1.0,
        )

        # Retornar la suma total acumulada (S)
        return float(np.sum(penalizaciones))

    @classmethod
    def mean_nasa_score(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Calcula el NASA Score promedio por observación (motor/ciclo)."""
        total_score = cls.nasa_score(y_true, y_pred)
        return total_score / len(y_true)

    @classmethod
    def analyze_predictions(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, any]:
        """Realiza un diagnóstico detallado del comportamiento del modelo

        separando el análisis del NASA Score por tipo de error.
        """
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        d = y_pred - y_true

        sobreestimaciones = d[d > 0]
        subestimaciones = d[d < 0]

        score_sobre = np.sum(np.exp(sobreestimaciones / 10.0) - 1.0)
        score_sub = np.sum(np.exp(-subestimaciones / 13.0) - 1.0)
        total_score = score_sobre + score_sub

        return {
            "total_nasa_score": float(total_score),
            "mean_nasa_score": float(total_score / len(y_true)),
            "overestimation_score_pct": (
                float(score_sobre / total_score * 100)
                if total_score > 0
                else 0.0
            ),
            "underestimation_score_pct": (
                float(score_sub / total_score * 100)
                if total_score > 0
                else 0.0
            ),
            "overestimation_count": int(len(sobreestimaciones)),
            "underestimation_count": int(len(subestimaciones)),
            "max_overestimation_cycles": (
                float(sobreestimaciones.max())
                if len(sobreestimaciones) > 0
                else 0.0
            ),
            "max_underestimation_cycles": (
                float(np.abs(subestimaciones.min()))
                if len(subestimaciones) > 0
                else 0.0
            ),
        }
