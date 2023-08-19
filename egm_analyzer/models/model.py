from typing import Protocol

import numpy as np


class PredictionModel(Protocol):
    def predict(self, signal: np.ndarray) -> np.ndarray:
        ...
