import argparse
import json
from collections import defaultdict
from itertools import count
from itertools import repeat
from itertools import starmap
from itertools import tee
from pathlib import Path

from egm_analyzer.types import InferenceResult


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
    with open(filepath, 'r') as input_:
        labels = InferenceResult(**json.load(input_))

    return [set(map(lambda x: round(x.position), channel)) for channel in labels.peaks]


def calculate_tp_fp(
        ground_truth: set[int],
        predictions: set[int],
        window_size: int,
        channel: int,
) -> tuple[int, int, dict[int, list[int]]]:
    offset = window_size // 2
    tp, fp = 0, 0
    errors: dict[int, list[int]] = defaultdict(list)

    for target_index in predictions:
        matched = any(
            map(
                lambda index: index in ground_truth,
                range(target_index - offset, target_index + offset + 1),
            ),
        )

        if matched:
            tp += int(matched)
            continue

        fp += int(not matched)
        errors[channel].append(target_index)

    errors[channel].sort()

    return tp, fp, errors


def calculate_fn(
        ground_truth: set[int],
        predictions: set[int],
        window_size: int,
        channel: int,
) -> tuple[int, dict[int, list[int]]]:
    offset = window_size // 2
    fn = 0
    errors: dict[int, list[int]] = defaultdict(list)

    for target_index in ground_truth:
        matched = any(
            map(
                lambda index: index in predictions,
                range(target_index - offset, target_index + offset + 1),
            ),
        )

        if matched:
            continue

        fn += 1
        errors[channel].append(target_index)

    return fn, errors


def elementwise_sum(
        first: tuple[float, ...],
        second: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple([sum(x) for x in zip(first, second, strict=True)])


def main() -> int:
    args = parse_args()
    ground_truth = read_ground_truth(args.ground_truth_filepath)
    predictions = read_predictions(args.predictions_filepath)

    metrics_args = zip(
        ground_truth,
        predictions,
        repeat(args.window_size),
        count(0, 1),
    )
    tp_fp_args, fn_args = tee(metrics_args, 2)

    tp_fp_res = starmap(calculate_tp_fp, tp_fp_args)
    total_tp, total_fp, fp_errors = 0, 0, {}
    for tp, fp, errors in tp_fp_res:
        total_tp += tp
        total_fp += fp
        fp_errors |= errors

    fn_res = starmap(calculate_fn, fn_args)

    total_fn, fn_errors = 0, {}
    for fn, errors in fn_res:
        total_fn += fn
        fn_errors |= errors

    errors = {'fp': fp_errors, 'fn': fn_errors}

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as out:
        json.dump(errors, out)

    precision = total_tp / denominator if (denominator := total_tp + total_fp) != 0 else 0
    recall = total_tp / denominator if (denominator := total_tp + total_fn) != 0 else 0

    try:
        f1_score = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0

    print(f'{total_tp=}, {total_fp=}, {total_fn=}', sep=', ')
    print(f'Precision = {precision:.5f}')
    print(f'Recall = {recall:.5f}')
    print(f'F1_score = {f1_score:.5f}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
