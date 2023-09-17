import numpy as np
from scipy.interpolate import CubicSpline

from egm_analyzer.types import Hz
from egm_analyzer.types import Index
from egm_analyzer.types import Peak
from egm_analyzer.types import Stage


class Compressor:
    def __init__(
            self,
            window_size: int = 100,
            signal_frequency: Hz = 5000,
            target_frequency: Hz = 20_000,
    ) -> None:

        self._window_size = window_size
        self._signal_to_target_frequency = target_frequency / signal_frequency

    def find_relative_minimum_derivative_index(
            self,
            signal_cutout: np.ndarray,
    ) -> Index | None:
        if len(signal_cutout) < 5:
            return None

        indexes = range(len(signal_cutout))
        cs = CubicSpline(indexes, signal_cutout)

        high_res_indexes = np.linspace(
            0,
            len(signal_cutout) - 1,
            int(self._signal_to_target_frequency * (len(signal_cutout) - 1)),
            endpoint=False,
        )

        abscis_min_der_index = int(np.argmin(cs(high_res_indexes, 1)))

        return abscis_min_der_index / self._signal_to_target_frequency

    def compress(self, preds: np.ndarray, signal: np.ndarray) -> list[Peak]:
        if len(preds) < 1:
            return []

        preds = np.sort(preds)

        peaks_indexes: list[Peak] = []
        group_start: int = int(preds[0])
        group_end = group_start

        for pred in preds:
            if pred - group_start < self._window_size:
                group_end = pred
                continue

            relative_minimum_derivative_index = (
                self.find_relative_minimum_derivative_index(
                    signal[group_start:min(group_end + 1, len(signal))],
                )
            )

            if relative_minimum_derivative_index is not None:
                peaks_indexes.append(
                    Peak(
                        position=relative_minimum_derivative_index + group_start,
                        creation_stage=Stage.PEAK_SEARCH,
                        search_segment=(group_start, group_end),
                    ),
                )

            group_start = pred
            group_end = pred

        relative_minimum_derivative_index = (
            self.find_relative_minimum_derivative_index(
                signal[group_start:min(group_end + 1, len(signal))],
            )
        )

        if relative_minimum_derivative_index is not None:
            peaks_indexes.append(
                Peak(
                    position=relative_minimum_derivative_index + group_start,
                    creation_stage=Stage.PEAK_SEARCH,
                    search_segment=(group_start, group_end),
                ),
            )

        return peaks_indexes
