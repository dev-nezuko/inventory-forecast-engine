"""
DeepAR (GluonTS) training stub. Not yet implemented.
Run from repo root: python -m src.models.train_deepar_gluonts --horizon 12
"""
from __future__ import annotations

import argparse


def main() -> None:
    raise NotImplementedError("train_deepar_gluonts is not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()
    main()
