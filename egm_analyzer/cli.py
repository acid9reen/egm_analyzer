import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

from egm_analyzer.models.onnx_wrapper import OnnxModelWrapper
from egm_analyzer.signal_processor import SignalProcessor


class EGMAnalyzerNamespace(argparse.Namespace):
    batch_size: int
    model_path: Path
    num_workers: int
    output_file: Path
    signal_path: Path
    threshold: float


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

    return parser.parse_args(namespace=EGMAnalyzerNamespace())


def main() -> int:
    args = parse_args()

    providers = [
        (
            'CUDAExecutionProvider',
            {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            },
        ),
        'CPUExecutionProvider',
    ]

    predictor = OnnxModelWrapper(args.model_path, providers=providers)
    signal_processor = SignalProcessor(
        predictor,
        args.batch_size,
        args.output_filepath,
        threshold=args.threshold,
    )

    signal = np.load(args.signal_path, mmap_mode='r')
    signal_processor.process(signal)

    return 0
