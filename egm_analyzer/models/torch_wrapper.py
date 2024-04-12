from typing import Literal
from pathlib import Path

import numpy as np
import torch


class TorchModelWrapper(object):
    def __init__(self, model_path: Path, device: Literal["cuda", "cpu"] | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = device
        self._model = torch.jit.load(model_path)
        self._model.eval().to(self._device)

    def predict(self, signal: np.ndarray) -> np.ndarray:
        data = torch.from_numpy(signal.astype(np.float32)).to(self._device)

        with torch.no_grad():
            prediction = self._model(data).detach().cpu().numpy()

        return prediction
