from typing import Protocol

import numpy as np


class Derivative(Protocol):
    @property
    def kernel_size(self) -> int:
        ...

    def calculate_approximation(self, signal_cutout: np.ndarray) -> float:
        ...


class ThreePointDerivative:
    @property
    def kernel_size(self) -> int:
        return 3

    def calculate_approximation(self, signal_cutout: np.ndarray) -> float:
        return signal_cutout[2] - signal_cutout[0]


class FivePointDerivative:
    @property
    def kernel_size(self) -> int:
        return 5

    def calculate_approximation(self, signal_cutout: np.ndarray) -> float:
        maxi_left = max(signal_cutout[0], signal_cutout[1])
        mini_right = min(signal_cutout[3], signal_cutout[4])

        return mini_right - maxi_left


class Compressor:
    def __init__(
            self,
            derivative_calculator: Derivative = ThreePointDerivative(),
            window_size: int = 100,
            frequency: int = 5000,
    ) -> None:
        self._derivative = derivative_calculator
        self._window_size = window_size
        self._padding_size = self._derivative.kernel_size // 2
        self._indices_to_microseconds_constant = int(1 / frequency * 1e6)

    def find_relative_minimum_derivative_index(
            self,
            signal_cutout: np.ndarray,
    ) -> int | None:
        if len(signal_cutout) < self._derivative.kernel_size:
            return None

        minimum_derivative = self._derivative.calculate_approximation(
            signal_cutout[:self._derivative.kernel_size],
        )
        minimum_derivative_index = self._padding_size
        start_index = minimum_derivative_index + 1

        for point_index in range(start_index, len(signal_cutout) - self._padding_size):
            derivative = self._derivative.calculate_approximation(
                signal_cutout[
                    point_index - self._padding_size:point_index + self._padding_size + 1
                ],
            )

            if derivative <= minimum_derivative:
                minimum_derivative = derivative
                minimum_derivative_index = point_index

        return minimum_derivative_index

    def compress(self, preds: np.ndarray, signal: np.ndarray) -> list[int]:
        if len(preds) < 1:
            return []

        preds = np.sort(preds)

        peaks_indexes = []
        group_start = preds[0]
        group_end = group_start

        for pred in preds:
            if pred - group_start < self._window_size:
                group_end = pred
                continue

            relative_minimum_derivative_index = (
                self.find_relative_minimum_derivative_index(
                    signal[max(group_start - self._padding_size, 0):group_end + self._padding_size],
                )
            )

            if relative_minimum_derivative_index is not None:
                peaks_indexes.append(
                    (
                        relative_minimum_derivative_index + group_start - self._padding_size
                    ) * self._indices_to_microseconds_constant,
                )

            group_start = pred
            group_end = pred

        relative_minimum_derivative_index = (
            self.find_relative_minimum_derivative_index(
                signal[max(group_start - self._padding_size, 0):group_end + self._padding_size],
            )
        )

        if relative_minimum_derivative_index is not None:
            peaks_indexes.append(
                (
                    relative_minimum_derivative_index + group_start - self._padding_size
                ) * self._indices_to_microseconds_constant,
            )

        return peaks_indexes
