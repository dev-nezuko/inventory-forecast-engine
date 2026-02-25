# Demonstration data and artifacts

This folder includes example CSV input and precomputed Parquet files for inspecting the pipeline and serving layers. The supplied forecasts are demonstration artifacts, not a reproduced training result: all three model-training entry points are currently stubs.

Input columns:

- `credit.csv`: `txn_id, customer_id, txn_date, sku, region, amount, price, quantity`
- `panel.csv`: `panel_id, customer_id, age_group, region, income_bin, household_size`

`landing/` contains intermediate tables; `features/` contains aggregated features and forecast-shaped artifacts. Running ingestion, linkage, or feature commands with this folder as `DATA_DIR` can overwrite them. Use another data directory for experiments you want to keep separate.

Customer identifiers are retained in source/landing files. Hashed linkage keys do not make the whole dataset anonymous. Never replace these examples with real transaction or household data in a public repository.
