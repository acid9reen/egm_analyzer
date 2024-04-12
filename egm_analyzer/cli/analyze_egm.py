import argparse
import csv
import json
import multiprocessing as mp
from itertools import zip_longest
from pathlib import Path

import numpy as np

from egm_analyzer.pred_processor import Compressor
from egm_analyzer.predictions_postprocess import postprocess_predictions
from egm_analyzer.signal_processor import SignalProcessor
from egm_analyzer.types import Gb
from egm_analyzer.types import InferenceMeta
from egm_analyzer.types import InferenceResult
from egm_analyzer.types import Peak


class EGMAnalyzerNamespace(argparse.Namespace):
    batch_size: int
    model_path: Path
    num_workers: int
    output_folder: Path
    signal_path: Path
    threshold: float
    gpu_mem_limit: Gb


def parse_args() -> EGMAnalyzerNamespace:
    parser = argparse.ArgumentParser(
        description='Program to analyze electrogram for heartbeats',
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        help='Batch size for predictor',
        default=1,
    )
    parser.add_argument(
        '-w', '--num_workers',
        type=int,
        help='Number of workers for multiprocessing',
        default=mp.cpu_count() // 2,
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        help='Threshold value for model activation',
        default=0.5,
    )
    parser.add_argument(
        '-m', '--model_path',
        type=Path,
        help='Path to .onnx model',
        required=True,
    )
    parser.add_argument(
        '-s', '--signal_path',
        type=Path,
        help='Path to target signal',
        required=True,
    )
    parser.add_argument(
        '-f', '--output_folder',
        help='Output folder',
        type=Path,
        default=Path('./out'),
    )
    parser.add_argument(
        '--gpu_mem_limit',
        type=Gb,
        help='GPU memory limit in GB',
        default=8,
    )

    return parser.parse_args(namespace=EGMAnalyzerNamespace())


def main() -> int:
    args = parse_args()

    prediction_filename = args.signal_path.stem + '.json'
    prediction_filepath = args.output_folder / prediction_filename

    output_filename = args.signal_path.stem + '.csv'
    output_filepath = args.output_folder / output_filename

    match model_type := args.model_path.suffix:
        case ".pt":
            try:
                from egm_analyzer.models.torch_wrapper import TorchModelWrapper
            except ImportError as e:
                raise ImportError(
                    """
                    To use .pt models you need to install torch oprional
                    dependencies (via `[torch]`)
                    """
                ) from e
            predictor = TorchModelWrapper(args.model_path)
        case ".onnx":
            from egm_analyzer.models.onnx_wrapper import OnnxModelWrapper
            from egm_analyzer.models.onnx_wrapper import get_providers
            predictor = OnnxModelWrapper(
                args.model_path, providers=get_providers(args.gpu_mem_limit)
            )
        case _:
            raise ValueError(f"Unknown model type `{model_type}`")

    compressor = Compressor(target_frequency=20_000)
    signal_processor = SignalProcessor(
        predictor,
        args.batch_size,
        compressor,
        threshold=args.threshold,
    )

    signal = np.load(args.signal_path, mmap_mode='r')
    peaks = signal_processor.process(signal)
    # peaks = postprocess_predictions(peaks, 200)

    # Do not forget to make output folder
    args.output_folder.mkdir(exist_ok=True, parents=True)

    result = InferenceResult(
        peaks=peaks,
        meta=InferenceMeta(
            threshold=args.threshold,
            path_to_model=args.model_path.resolve().as_posix(),
            path_to_signal=args.signal_path.resolve().as_posix(),
        ),
    )

    # Save prediction file
    with open(prediction_filepath, 'w') as out:
        json.dump(result.model_dump(), out)

    with open(output_filepath, 'w', newline='') as out:
        csv_writer = csv.writer(out, dialect='excel')
        for row in zip_longest(*peaks, fillvalue=''):
            csv_writer.writerow(
                map(lambda p: p.position * 200 if isinstance(p, Peak) else p, row),
            )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
