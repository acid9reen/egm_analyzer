from typing import Generator

import numpy as np
from tqdm import tqdm
from egmlib.preprocess import highpass_filter, moving_avg_filter, scale

from egm_analyzer.models.model import PredictionModel
from egm_analyzer.pred_processor import Compressor
from egm_analyzer.types import Peak


seconds = int


def batcher(
        signal: np.ndarray,
        batch_size: int,
        length: int,
        intersection_length: int = 500,
) -> Generator[np.ndarray, None, None]:
    signal_copy = signal.copy()
    signal_copy = highpass_filter(signal_copy, 5000, order=2, critical_frequency=250)
    signal_copy = moving_avg_filter(signal_copy, size=3)

    step = length - intersection_length

    batch = []
    _, remaining = divmod(len(signal_copy), step)

    for step_index in range(step, len(signal_copy), step):
        stop_index = step_index + intersection_length
        start_index = stop_index - length

        signal_cutout = scale(signal_copy[start_index:stop_index])

        batch.append([signal_cutout])

        if len(batch) % batch_size == 0:
            try:
                yield np.array(batch)
            finally:
                batch = []

    # Get the remaining part of the signal
    if remaining != 0:
        batch.append([scale(signal_copy[-length:])])

    # Exit on empty batch
    if not batch:
        return

    yield np.array(batch)


class SignalProcessor(object):
    def __init__(
            self,
            model: PredictionModel,
            batch_size: int,
            compressor: Compressor,
            signal_frequency: int = 5_000,
            signal_length: seconds = 2,
            threshold: float = 0.5,
            intersection: int = 500,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._step = signal_frequency * signal_length
        self._threshold = threshold
        self._intersection = intersection
        self._compressor = compressor

    def process(self, signal: np.ndarray) -> list[list[Peak]]:
        result: list[list[Peak]] = []

        for channel in tqdm(signal, total=len(signal), colour='green'):
            channel_predictions_batches: list[np.ndarray] = []

            for batch in batcher(channel, self._batch_size, self._step, self._intersection):
                prediction_batch = self._model.predict(batch)
                channel_predictions_batches.append(prediction_batch.squeeze())

            channel_predictions = np.vstack(channel_predictions_batches)

            accumulated_step = 0
            prediction_indexes = []
            for batch in channel_predictions[:-1]:
                indexes, *__ = np.nonzero(batch > self._threshold)
                indexes += accumulated_step
                prediction_indexes.append(indexes)
                accumulated_step += (self._step - self._intersection)

            last_indexes, *__ = np.nonzero(channel_predictions[-1] > self._threshold)
            last_indexes += len(channel) - self._step
            prediction_indexes.append(last_indexes)

            channel_prediction = np.sort(np.hstack(prediction_indexes))
            channel_result = self._compressor.compress(channel_prediction, channel)
            result.append(channel_result)

        return result
