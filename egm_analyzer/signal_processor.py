import csv
from pathlib import Path
from typing import Generator

import numpy as np
from tqdm import tqdm

from egm_analyzer.models.model import PredictionModel
from egm_analyzer.pred_processor import Compressor


seconds = int


def batcher(
        signal: np.ndarray,
        batch_size: int,
        step: int,
        intersection_length: int = 500,
) -> Generator[np.ndarray, None, None]:
    batch = []
    num_batches, remaining = divmod(len(signal), step - intersection_length)

    for batch_index in range(num_batches):
        start_index = batch_index * (step - intersection_length)
        stop_index = start_index + step

        batch.append([signal[start_index:stop_index]])

        if len(batch) % batch_size == 0:
            try:
                yield np.array(batch)
            finally:
                batch = []

    # Get the remaining part of the signal
    if remaining != 0:
        batch.append([signal[-step:]])

    # Exit on empty batch
    if not batch:
        return

    yield np.array(batch)


class SignalProcessor(object):
    def __init__(
            self,
            model: PredictionModel,
            batch_size: int,
            output_file: Path,
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
        self._output_file = output_file
        self._intersection = intersection
        self._compressor = compressor

    def _write_result(self, result: list[int]) -> None:
        with open(self._output_file, 'a', newline='') as out:
            csv_writer = csv.writer(out, dialect='excel')
            csv_writer.writerow(result)

    def process(self, signal: np.ndarray) -> None:
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
            last_indexes += len(channel) - 1 - self._step
            prediction_indexes.append(last_indexes)

            channel_prediction = np.hstack(prediction_indexes)
            result = self._compressor.compress(channel_prediction, channel)
            self._write_result(result)
