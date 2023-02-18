from typing import Generator

import numpy as np

from egm_analyzer.models.model import PredictionModel


seconds = int


def batcher(
        signal: np.ndarray,
        batch_size: int,
        step: int,
) -> Generator[np.ndarray, None, None]:
    batch = []
    num_batches = len(signal) // step

    for batch_index in range(num_batches):
        start_index = batch_index * step
        stop_index = start_index + step

        batch.append(signal[start_index:stop_index])

        if len(batch) % batch_size == 0:
            yield np.array(batch)

    yield np.array(batch)


class SignalProcessor(object):
    def __init__(
            self,
            model: PredictionModel,
            batch_size: int,
            signal_frequency: int = 5_000,
            signal_length: seconds = 2,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._step = signal_frequency * signal_length

    def process(self, signal: np.ndarray) -> None:
        for channel_index, channel in enumerate(signal):
            for batch in batcher(channel, self._batch_size, self._step):
                prediction_batch = self._model.predict(batch)
                _ = prediction_batch.flatten()
