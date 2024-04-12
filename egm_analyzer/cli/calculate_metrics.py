import argparse
import json
from itertools import count
from itertools import repeat
from itertools import starmap
from itertools import tee
from pathlib import Path

from egm_analyzer.types import Error
from egm_analyzer.types import ErrorType
from egm_analyzer.types import InferenceResult
from egm_analyzer.types import Metrics
from egm_analyzer.types import MetricsMeta
from egm_analyzer.types import MetricsResult


class CalculateMetricsNamespace(argparse.Namespace):
    inference_result_filepath: Path
    ground_truth_filepath: Path
    num_workers: int
    output_folder: Path
    window_size: int


def parse_args() -> CalculateMetricsNamespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'inference_result_filepath',
        type=Path,
        help='Path to .json inference result file',
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
        default=4,
        help='Number of workers for metrics calculation',
    )
    parser.add_argument(
        '-o',
        '--output_folder',
        type=Path,
        default=Path('./out/metrics_results'),
        help='Output folder save metrics result into',
    )
    parser.add_argument(
        '-s',
        '--window_size',
        type=int,
        default=7,
        help='Length of window size for verification',
    )

    return parser.parse_args(namespace=CalculateMetricsNamespace())


def read_ground_truth(filepath: Path) -> list[set[int]]:
    with open(filepath, 'r') as input_:
        labels: list[list[int]] = json.load(input_)

    return [set(map(lambda x: round(x), channel)) for channel in labels]


def read_predictions(filepath: Path) -> list[set[int]]:
    with open(filepath, 'r') as input_:
        labels = InferenceResult(**json.load(input_))

    return [set(map(lambda x: round(x.position), channel)) for channel in labels.peaks]


def calculate_tp_fp(
        ground_truth: set[int],
        predictions: set[int],
        window_size: int,
        channel: int,
) -> tuple[int, int, list[Error]]:
    offset = window_size // 2
    tp, fp = 0, 0
    errors: list[Error] = []

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
        errors.append(
            Error(
                position=target_index,
                channel=channel,
                error_type=ErrorType.FALSE_POSITIVE,
            ),
        )

    return tp, fp, errors


def calculate_fn(
        ground_truth: set[int],
        predictions: set[int],
        window_size: int,
        channel: int,
) -> tuple[int, list[Error]]:
    offset = window_size // 2
    fn = 0
    errors: list[Error] = []

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
        errors.append(
            Error(
                position=target_index,
                channel=channel,
                error_type=ErrorType.FALSE_NEGATIVE,
            ),
        )

    return fn, errors


def elementwise_sum(
        first: tuple[float, ...],
        second: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple([sum(x) for x in zip(first, second, strict=True)])


def main() -> int:
    args = parse_args()
    ground_truth = read_ground_truth(args.ground_truth_filepath)
    predictions = read_predictions(args.inference_result_filepath)

    metrics_args = zip(
        ground_truth,
        predictions,
        repeat(args.window_size),
        count(0, 1),
    )
    tp_fp_args, fn_args = tee(metrics_args, 2)

    tp_fp_res = starmap(calculate_tp_fp, tp_fp_args)

    total_errors: list[Error] = []

    total_tp, total_fp = 0, 0
    for tp, fp, errors in tp_fp_res:
        total_tp += tp
        total_fp += fp
        total_errors.extend(errors)

    fn_res = starmap(calculate_fn, fn_args)

    total_fn = 0
    for fn, errors in fn_res:
        total_fn += fn
        total_errors.extend(errors)

    precision = total_tp / denominator if (denominator := total_tp + total_fp) != 0 else 0
    recall = total_tp / denominator if (denominator := total_tp + total_fn) != 0 else 0

    try:
        f1_score = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0

    metrics_result = MetricsResult(
        errors=total_errors,
        metrics=Metrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
        ),
        meta=MetricsMeta(
            inference_result_path=args.inference_result_filepath.resolve().as_posix(),
            ground_truth_path=args.ground_truth_filepath.resolve().as_posix(),
        ),
    )

    output_filename = '_'.join([
        args.inference_result_filepath.stem,
        'metrics',
    ]) + args.inference_result_filepath.suffix

    args.output_folder.mkdir(parents=True, exist_ok=True)
    with open(args.output_folder / output_filename, 'w') as out:
        json.dump(metrics_result.model_dump(), out)

    print(f'{total_tp=}, {total_fp=}, {total_fn=}')
    print(f'Precision = {precision:.5f}')
    print(f'Recall = {recall:.5f}')
    print(f'F1_score = {f1_score:.5f}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
