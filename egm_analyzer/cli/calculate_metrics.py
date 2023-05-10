import argparse
import csv
import json
from functools import reduce
from itertools import cycle
from itertools import starmap
from itertools import tee
from pathlib import Path


class CalculateMetricsNamespace(argparse.Namespace):
    predictions_filepath: Path
    ground_truth_filepath: Path
    num_workers: int
    output_file: Path
    window_size: int


def parse_args() -> CalculateMetricsNamespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'predictions_filepath',
        type=Path,
        help='Path to .csv file (tmp one)',
    )
    parser.add_argument(
        'ground_truth_filepath',
        type=Path,
        help='Path to ground truth labels .json file',
    )
    parser.add_argument(
        '-n',
        '--num_workers',
        type=int,
        default=1,
        help='Number of workers for metrics calculation',
    )
    parser.add_argument(
        '-o',
        '--output_file',
        type=Path,
        help='Output file to store information about prediction errors',
    )
    parser.add_argument(
        '-s',
        '--window_size',
        type=int,
        default=3,
        help='Length of window size for verification',
    )

    return parser.parse_args(namespace=CalculateMetricsNamespace())


def read_ground_truth(filepath: Path) -> list[set[int]]:
    with open(filepath, 'r') as input_:
        labels: list[list[int]] = json.load(input_)

    return [set(channel) for channel in labels]


def read_predictions(filepath: Path) -> list[set[int]]:
    labels: list[set[int]] = []

    with open(filepath, 'r') as input_:
        csv_reader = csv.reader(input_, dialect='excel')

        for row in csv_reader:
            channel = set(map(lambda x: round(float(x) / 200), row))
            labels.append(channel)

    return labels


def calculate_tp_fp(
        ground_truth: set[int],
        predictions: set[int],
        window_size: int,
) -> tuple[int, int]:
    offset = window_size // 2
    tp, fp = 0, 0

    for target_index in predictions:
        matched = any(
            map(
                lambda index: index in ground_truth,
                range(target_index - offset, target_index + offset + 1),
            ),
        )

        tp += int(matched)
        fp += int(not matched)

    return tp, fp


def calculate_fn(
        ground_truth: set[int],
        predictions: set[int],
        window_size: int,
) -> int:
    offset = window_size // 2
    fn = 0

    for target_index in ground_truth:
        matched = any(
            map(
                lambda index: index in predictions,
                range(target_index - offset, target_index + offset + 1),
            ),
        )

        fn += int(not matched)

    return fn


def elementwise_sum(
        first: tuple[float, ...],
        second: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple([sum(x) for x in zip(first, second)])


def main() -> int:
    args = parse_args()
    ground_truth = read_ground_truth(args.ground_truth_filepath)
    predictions = read_predictions(args.predictions_filepath)

    total_tp, total_fp, total_fn = 0, 0, 0

    metrics_args = zip(
        ground_truth,
        predictions,
        cycle([args.window_size]),
    )

    tp_fp_args, fn_args = tee(metrics_args, 2)

    total_tp, total_fp = reduce(elementwise_sum, starmap(calculate_tp_fp, tp_fp_args))
    total_fn = reduce(lambda x, y: x + y, starmap(calculate_fn, fn_args))

    precision = total_tp / denominator if (denominator := total_tp + total_fp) != 0 else 0
    recall = total_tp / denominator if (denominator := total_tp + total_fn) != 0 else 0

    print(f'{total_tp=}, {total_fp=}, {total_fn=}', sep=', ')
    print(f'Precision = {precision:.5f}')
    print(f'Recall = {recall:.5f}')
    print(f'F1_score = {(2 * precision * recall) / (precision + recall):.5f}')

    return 0
