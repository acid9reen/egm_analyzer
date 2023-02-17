import argparse


class EGMAnalyzerNamespace(argparse.Namespace):
    ...


def parse_args() -> EGMAnalyzerNamespace:
    parser = argparse.ArgumentParser(
        description='Program to analyze electrogram for heartbeats',
    )

    return parser.parse_args(namespace=EGMAnalyzerNamespace())


def main() -> int:
    _ = parse_args()

    return 0
