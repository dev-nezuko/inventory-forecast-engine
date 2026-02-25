"""
LightGBM quantile training stub. Not yet implemented.
Run from repo root: python -m src.models.train_lgb_quantile --quantiles 0.1 0.5 0.9 --horizon 12
"""
from __future__ import annotations

import argparse


def main() -> None:
    raise NotImplementedError("train_lgb_quantile is not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantiles", nargs="+", default=["0.1", "0.5", "0.9"])
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()
    main()
