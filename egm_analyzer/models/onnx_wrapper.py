from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxModelWrapper(object):
    def __init__(self, model_path: Path, providers: list | None = None) -> None:
        if providers is None:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.ort_input_name = self.session.get_inputs()[0].name

    def predict(self, signal: np.ndarray) -> np.ndarray:
        ort_input = {self.ort_input_name: signal}
        prediction, *__ = self.session.run(None, ort_input)

        return prediction  # type: ignore
