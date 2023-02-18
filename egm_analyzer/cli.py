import argparse
import multiprocessing as mp
from pathlib import Path


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
        desc='Batch size for predictor',
        default=1,
    )
    parser.add_argument(
        '-w', '--num_workers',
        type=int,
        desc='Number of workers for multiprocessing',
        default=mp.cpu_count() // 2,
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        desc='Threshold value for model activation',
        default=0.5,
    )
    parser.add_argument(
        '-m', '--model_path',
        type=Path,
        desc='Path to .onnx model',
        required=True,
    )
    parser.add_argument(
        '-s', '--signal_path',
        type=Path,
        desc='Path to target signal',
        required=True,
    )
    parser.add_argument(
        '-o', '--output_filepath',
        type=Path,
        desc='Path to output file',
        default='./out',
    )

    return parser.parse_args(namespace=EGMAnalyzerNamespace())


def main() -> int:
    _ = parse_args()

    return 0
