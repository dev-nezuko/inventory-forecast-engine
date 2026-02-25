"""
Generate the synthetic demo dataset in examples/sample_data.

Every customer, SKU and transaction is invented. Demand per SKU x region follows
a level, a mild trend, yearly seasonality and Poisson noise, so the trainer has
enough weekly history for a time-based holdout.

Usage: python -m src.synthetic --out examples/sample_data --weeks 104 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SKUS = {"SKU-ANCHOR": 2.5, "SKU-BRACKET": 6.0, "SKU-CLAMP": 3.75, "SKU-DOWEL": 0.9}
REGIONS = {"NE": 1.0, "SE": 0.8, "MW": 1.2, "W": 0.9}
AGE_GROUPS = ["18-34", "35-54", "55+"]
INCOME_BINS = ["<35k", "35-50k", "50-75k", "75k+"]


def generate(
    weeks: int = 104,
    n_customers: int = 400,
    panel_share: float = 0.6,
    seed: int = 7,
    start: str = "2024-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (credit, panel) DataFrames with the columns ingest expects."""
    rng = np.random.default_rng(seed)
    regions = list(REGIONS)

    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(1, n_customers + 1)],
            "region": rng.choice(regions, size=n_customers),
        }
    )

    week_starts = pd.date_range(start, periods=weeks, freq="W-MON")
    rows = []
    for s_idx, (sku, price) in enumerate(SKUS.items()):
        for region, scale in REGIONS.items():
            pool = customers.loc[
                customers["region"] == region, "customer_id"
            ].to_numpy()
            level = 8 + 4 * s_idx
            for w, week_start in enumerate(week_starts):
                season = 1 + 0.3 * np.sin(2 * np.pi * (w + 4 * s_idx) / 52)
                trend = 1 + 0.004 * w
                units = rng.poisson(level * scale * season * trend)
                remaining = units
                while remaining > 0:
                    qty = int(min(remaining, rng.integers(1, 5)))
                    remaining -= qty
                    day = week_start + pd.Timedelta(days=int(rng.integers(0, 7)))
                    rows.append((rng.choice(pool), day, sku, region, price, qty))

    credit = pd.DataFrame(
        rows, columns=["customer_id", "txn_date", "sku", "region", "price", "quantity"]
    )
    credit = credit.sort_values("txn_date", kind="stable").reset_index(drop=True)
    credit.insert(0, "txn_id", np.arange(1, len(credit) + 1))
    credit["txn_date"] = credit["txn_date"].dt.strftime("%Y-%m-%d")
    credit["amount"] = (credit["price"] * credit["quantity"]).round(2)
    credit = credit[
        [
            "txn_id",
            "customer_id",
            "txn_date",
            "sku",
            "region",
            "amount",
            "price",
            "quantity",
        ]
    ]

    members = customers.sample(frac=panel_share, random_state=seed).sort_values(
        "customer_id"
    )
    panel = pd.DataFrame(
        {
            "panel_id": [f"P{i:04d}" for i in range(1, len(members) + 1)],
            "customer_id": members["customer_id"].to_numpy(),
            "age_group": rng.choice(AGE_GROUPS, size=len(members), p=[0.3, 0.45, 0.25]),
            "region": members["region"].to_numpy(),
            "income_bin": rng.choice(INCOME_BINS, size=len(members)),
            "household_size": rng.integers(1, 6, size=len(members)),
        }
    )
    return credit, panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", type=Path, default=Path("examples/sample_data"))
    parser.add_argument("--weeks", type=int, default=104)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    credit, panel = generate(weeks=args.weeks, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    credit.to_csv(args.out / "credit.csv", index=False)
    panel.to_csv(args.out / "panel.csv", index=False)
    print(
        f"Wrote {len(credit)} transactions and {len(panel)} panel members to {args.out}"
    )


if __name__ == "__main__":
    main()
