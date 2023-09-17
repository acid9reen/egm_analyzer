import statistics
from typing import NamedTuple

from egm_analyzer.types import Index
from egm_analyzer.types import Peak


class PeakWithChannel(NamedTuple):
    peak: Peak
    channel: int


def labels_to_pairs(labels: list[list[Peak]]) -> tuple[list[PeakWithChannel], set[int]]:
    peaks: list[PeakWithChannel] = []
    empty_channels = set()
    channels_lengths = []

    for channel_index, channel in enumerate(labels):
        transformed_channel = map(lambda p: PeakWithChannel(p, channel_index), channel)
        peaks.extend(transformed_channel)
        channels_lengths.append(len(channel))

    mean_length = statistics.mean(channels_lengths)
    empty_channels = {
        channel_num for channel_num, channel_length in enumerate(channels_lengths)
        if channel_length < mean_length
    }

    return sorted(peaks, key=lambda peak: peak.peak.position), empty_channels


def fix_peaks(
    peaks: list[PeakWithChannel],
    empty_channels: set[int],
    window_size: Index = 200,
) -> list[list[Peak]]:
    result: list[list[Peak]] = [[] for __ in range(64)]
    threshold = (64 - len(empty_channels)) // 2

    pivot = peaks[0].peak.position
    window_frame: list[PeakWithChannel] = []

    for peak in peaks:
        if (index := peak.peak.position) < pivot + window_size:
            window_frame.append(peak)
            continue

        if len(window_frame) > threshold:
            channel_to_peak = {peak.channel: peak.peak for peak in window_frame}

            for i in range(len(result)):
                if i in empty_channels:
                    continue

                if (found_peak := channel_to_peak.get(i)) is not None:
                    result[i].append(found_peak)

        pivot = index
        window_frame = []

    return result


def postprocess_predictions(
        labels: list[list[Peak]],
        window_size: Index,
) -> list[list[Peak]]:
    peaks, empty_channels = labels_to_pairs(labels)
    result = fix_peaks(peaks, empty_channels, window_size)

    return result
