import statistics
from typing import NamedTuple


class Peak(NamedTuple):
    peak_index: float
    channel: int


def labels_to_pairs(labels: list[list[float]]) -> tuple[list[Peak], set[int]]:
    peaks = []
    empty_channels = set()
    channels_lengths = []

    for channel_index, channel in enumerate(labels):
        transformed_channel = map(lambda index: Peak(index, channel_index), channel)
        peaks.extend(transformed_channel)
        channels_lengths.append(len(channel))

    mean_length = statistics.mean(channels_lengths)
    empty_channels = {
        channel_num for channel_num, channel_length in enumerate(channels_lengths)
        if channel_length < mean_length
    }

    return sorted(peaks, key=lambda peak: peak.peak_index), empty_channels


def fix_peaks(
    peaks: list[Peak],
    empty_channels: set[int],
    window_size: float = 20,
) -> list[list[float]]:
    result: list[list[float]] = [[] for __ in range(64)]
    threshold = (64 - len(empty_channels)) // 2

    pivot = peaks[0].peak_index
    window_frame: list[Peak] = []

    for peak in peaks:
        if (index := peak.peak_index) < pivot + window_size:
            window_frame.append(peak)
            continue

        if len(window_frame) > threshold:
            channel_index = {peak.channel: peak.peak_index for peak in window_frame}

            for i in range(len(result)):
                if i in empty_channels:
                    continue

                if (found_peak := channel_index.get(i)) is not None:
                    result[i].append(found_peak)

        pivot = index
        window_frame: list[Peak] = []

    return result


def postprocess_predictions(
        labels: list[list[float]],
        window_size: float,
) -> list[list[float]]:
    peaks, empty_channels = labels_to_pairs(labels)
    result = fix_peaks(peaks, empty_channels, window_size)

    return result
