import numpy as np


def first_derivative(func: np.ndarray) -> float:

    maxi_left = np.max(func[0], func[1])
    mini_right = np.min(func[3], func[4])

    return mini_right - maxi_left


def compress(
        pred: np.ndarray,
        signal: np.ndarray,
        window_size: int,
) -> list[int]:

    pred = np.sort(pred)

    peaks_indexes = []
    group_start = pred[0]
    founded_end = False

    for index, elem in enumerate(pred[2:-2]):
        founded_end = False

        if (pred[index + 1] - elem > window_size):
            group_end = elem

            if group_end - group_start > 5:

                min_derivative = 1_000_000_000
                min_derivative_idx = 0

                for idx in range(group_start - 2, group_end + 2):
                    derivative = first_derivative(signal[idx:idx + 5])
                    if derivative < min_derivative:
                        min_derivative = derivative
                        min_derivative_idx = idx + 2

                group_start = pred[index + 1]
                founded_end = True
                peaks_indexes.append(min_derivative_idx)
            else:
                group_start = pred[index + 1]

    if not founded_end:

        group_end = min(pred[-1], len(signal) - 2)

        if group_end - group_start > 5:

            min_derivative = 1_000_000_000
            min_derivative_idx = 0

            for idx in range(group_start - 2, group_end + 2):
                derivative = first_derivative(signal[idx:idx + 5])
                if derivative < min_derivative:
                    min_derivative = derivative
                    min_derivative_idx = idx + 2

                peaks_indexes.append(min_derivative_idx)

    return peaks_indexes
