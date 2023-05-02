import argparse
import csv
import multiprocessing as mp
import os
from itertools import zip_longest
from pathlib import Path

import numpy as np

from egm_analyzer.models.onnx_wrapper import OnnxModelWrapper
from egm_analyzer.pred_processor import Compressor
from egm_analyzer.signal_processor import SignalProcessor


class EGMAnalyzerNamespace(argparse.Namespace):
    batch_size: int
    model_path: Path
    num_workers: int
    output_file: Path
    signal_path: Path
    threshold: float
    remove_temp_file: bool
    gpu_mem_limit: int


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
        '-o', '--output_filepath',
        type=Path,
        help='Path to output file',
        default='./out',
    )
    parser.add_argument(
        '--remove_temp_file',
        action='store_true',
        help='Remove intermediate .csv file',
    )
    parser.add_argument(
        '--gpu_mem_limit',
        type=int,
        help='GPU memory limit in GB',
        default=8,
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

    temp_filename = '.temp_' + args.output_filepath.name
    temp_filepath = args.output_filepath.parent / temp_filename

    predictor = OnnxModelWrapper(args.model_path, providers=providers)
    compressor = Compressor()
    signal_processor = SignalProcessor(
        predictor,
        args.batch_size,
        compressor,
        threshold=args.threshold,
    )

    signal = np.load(args.signal_path, mmap_mode='r')
    result = signal_processor.process(signal)

    with open(temp_filepath, 'w', newline='') as out:
        csv_writer = csv.writer(out, dialect='excel')
        for channel in result:
            csv_writer.writerow(channel)

    with open(temp_filepath, 'r', newline='') as temp, \
         open(args.output_filepath, 'w', newline='') as out:

        csv_reader = csv.reader(temp, dialect='excel')
        csv_writer = csv.writer(out, dialect='excel')

        csv_writer.writerows(zip_longest(*csv_reader, fillvalue=''))

    if args.remove_temp_file:
        os.remove(temp_filepath)

    return 0
