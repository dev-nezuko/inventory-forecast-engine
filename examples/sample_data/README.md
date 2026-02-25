# Synthetic demonstration data

Everything in this folder is invented by `python -m src.synthetic` (seed 7): 400
made-up customers (`C0001`...), 240 of them in the panel, and two years of
weekly demand (2024-01-01 to 2025-12-28) for four SKUs in four regions, shaped
by a level, a slight trend, yearly seasonality and Poisson noise. No real
person, household or transaction is represented.

| File | Produced by |
| --- | --- |
| `credit.csv`, `panel.csv` | `src.synthetic` |
| `landing/credit_txn.parquet`, `landing/panel.parquet` | `src.ingest.ingest` |
| `landing/linked_panel_credit.parquet` | `src.linking.link_panel_credit` (pseudonyms from a throwaway key) |
| `landing/panel_weighted.parquet` | `src.weighting.raking` (toy targets) |
| `features/features.parquet` | `src.features.build_features` |
| `features/forecast_gbm.parquet`, `features/gbm_backtest.json` | `src.models.train_gbm_quantile --horizon 12 --holdout 12 --data-label synthetic` |
| `features/forecast_ensemble.parquet` | `src.ensemble.ensemble_and_reconcile` (GBM only, so the same values) |

Input columns:

- `credit.csv`: `txn_id, customer_id, txn_date, sku, region, amount, price, quantity`
- `panel.csv`: `panel_id, customer_id, age_group, region, income_bin, household_size`

Running the pipeline with this folder as `DATA_DIR` overwrites these files. Point
`DATA_DIR` at a copy for experiments you want to keep apart.

The linked table keeps only the HMAC pseudonym, but `credit.csv`, `panel.csv` and
the landing files still hold the raw customer IDs; pseudonymisation is not
anonymisation. Never replace these examples with real transaction or household
data in a public repository.
