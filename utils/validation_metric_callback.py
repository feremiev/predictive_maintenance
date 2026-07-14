from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from tensorflow import keras


class ValidationMetricCallback(keras.callbacks.Callback):
    """
    Calculate a prediction-based metric on the complete validation set
    after every training epoch.

    The metric function must accept:

        metric_function(y_true, y_pred) -> float

    The calculated value is added to the Keras logs, allowing callbacks
    such as EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint to
    monitor it.
    """

    def __init__(
        self,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
        metric_function: Callable[
            [np.ndarray, np.ndarray],
            float,
        ],
        metric_name: str,
        batch_size: Optional[int] = None,
        verbose: int = 1,
    ) -> None:
        super().__init__()

        if not callable(metric_function):
            raise TypeError("metric_function must be callable.")

        if not metric_name or not metric_name.strip():
            raise ValueError("metric_name cannot be empty.")

        self.X_validation = X_validation
        self.y_validation = np.asarray(
            y_validation,
            dtype=np.float64,
        ).reshape(-1)

        self.metric_function = metric_function
        self.metric_name = metric_name.strip()
        self.batch_size = batch_size
        self.verbose = int(verbose)

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[dict[str, float]] = None,
    ) -> None:
        if logs is None:
            return

        predictions = self.model.predict(
            self.X_validation,
            batch_size=self.batch_size,
            verbose=0,
        ).reshape(-1)

        if len(predictions) != len(self.y_validation):
            raise ValueError(
                "The number of validation predictions does not match "
                "the number of validation targets."
            )

        metric_value = self.metric_function(
            self.y_validation,
            predictions,
        )

        log_name = f"val_{self.metric_name}"
        logs[log_name] = float(metric_value)

        if self.verbose:
            print(
                f" — {log_name}: {float(metric_value):.6f}"
            )
