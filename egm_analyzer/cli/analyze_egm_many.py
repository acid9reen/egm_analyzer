import argparse
import csv
import json
import multiprocessing as mp
from itertools import zip_longest
from pathlib import Path

import numpy as np
from scipy.signal import decimate

from egm_analyzer.models.onnx_wrapper import OnnxModelWrapper
from egm_analyzer.pred_processor import Compressor
from egm_analyzer.predictions_postprocess import postprocess_predictions
from egm_analyzer.signal_processor import SignalProcessor
from egm_analyzer.types import Gb
from egm_analyzer.types import Index


class EGMAnalyzerNamespace(argparse.Namespace):
    batch_size: int
    model_path: Path
    num_workers: int
    output_folder: Path
    signals_folder: Path
    threshold: float
    gpu_mem_limit: Gb
    q: int


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
        '-f', '--signals_folder',
        type=Path,
        help='Path folders with signals',
        required=True,
    )
    parser.add_argument(
        '-o', '--output_folder',
        help='Output folder',
        default=Path('./out'),
    )
    parser.add_argument(
        '--gpu_mem_limit',
        type=Gb,
        help='GPU memory limit in GB',
        default=8,
    )
    parser.add_argument(
        '-q', '--q',
        type=int,
        help='q for decimate',
        default=4,
    )

    return parser.parse_args(namespace=EGMAnalyzerNamespace())


def main() -> int:
    args = parse_args()

    providers = [
        (
            'CUDAExecutionProvider',
            {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': args.gpu_mem_limit * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            },
        ),
        'CPUExecutionProvider',
    ]

    predictor = OnnxModelWrapper(args.model_path, providers=providers)
    compressor = Compressor(target_frequency=20_000)
    signal_processor = SignalProcessor(
        predictor,
        args.batch_size,
        compressor,
        threshold=args.threshold,
    )

    signal_paths = args.signals_folder.glob('*.npy')

    for signal_path in signal_paths:
        prediction_filename = signal_path.stem + '.json'
        prediction_filepath = args.output_folder / prediction_filename

        output_filename = signal_path.stem + '.csv'
        output_filepath = args.output_folder / output_filename

        signal = np.load(signal_path, mmap_mode='r')
        signal = decimate(signal, args.q)
        result = signal_processor.process(signal)
        result = postprocess_predictions(result, 100_000)

        # Do not forget to make output folder
        args.output_folder.mkdir(exist_ok=True, parents=True)

        # Save prediction file
        with open(prediction_filepath, 'w') as out:
            indexes: list[list[Index]] = []

            for channel in result:
                tmp = []
                for number in channel:
                    tmp.append(number / 200)
                indexes.append(tmp)

            json.dump(indexes, out)

        with open(output_filepath, 'w', newline='') as out:
            csv_writer = csv.writer(out, dialect='excel')
            csv_writer.writerows(zip_longest(*result, fillvalue=''))

    return 0
