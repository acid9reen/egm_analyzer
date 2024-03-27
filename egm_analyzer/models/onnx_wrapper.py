from pathlib import Path

import numpy as np
import onnxruntime as ort


def get_providers(gpu_mem_limit: int) -> list:
    providers = [
        (
            'CUDAExecutionProvider',
            {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': gpu_mem_limit * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            },
        ),
        'CPUExecutionProvider',
    ]

    return providers


class OnnxModelWrapper(object):
    def __init__(self, model_path: Path, providers: list | None = None) -> None:
        if providers is None:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.ort_input_name = self.session.get_inputs()[0].name

    def predict(self, signal: np.ndarray) -> np.ndarray:
        ort_input = {self.ort_input_name: signal.astype(np.float32)}
        prediction, *__ = self.session.run(None, ort_input)

        return prediction  # type: ignore
