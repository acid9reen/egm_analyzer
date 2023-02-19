from typing import Callable

import numpy as np


def mean_compressor(pred: np.ndarray) -> int:
    return len(pred) // 2


def max_compressor(pred: np.ndarray) -> int:
    return np.argmax(pred)


def compress(
        pred: np.ndarray,
        treshold: float,
        skew: int,
        compressor: Callable[[np.ndarray], int] = max_compressor,
) -> np.ndarray:

    peaks_indexes = []
    flag = False

    for index, elem in enumerate(pred):

        if (elem > treshold) and (not flag):
            index_start = index
            flag = True

        if (elem < treshold) and (flag):
            index_end = index
            flag = False

            realtive_peak_index = compressor(pred[index_start:index_end])
            peaks_indexes.append(index_start + skew + realtive_peak_index)

    if flag:
        realtive_peak_index = compressor(pred[index_start: len(pred)])
        peaks_indexes.append(index_start + realtive_peak_index)

    return peaks_indexes

# TODO fix close preds in different samples
