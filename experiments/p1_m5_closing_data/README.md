# P1-M5 Closing Data Audit

This is a **supplementary**, post-lock reporting utility; it does not propose or
train any model, change the frozen study manifest, extend its evaluation sample,
or read any forbidden-date cache. The formal holdout acquisition remains in
`tasks.stochopia_p1.holdout_data`, using the exact frozen M5 task code.

## Reproduce the development data summary

Use Python 3.12 with Stochopia's locked dependencies and both repositories on
`PYTHONPATH`. Paths below are examples of explicitly supplied private roots, not
hard-coded locations. Run from a directory where the `experiments` namespace of
both repositories is importable.

```sh
python -m experiments.p1_m5_closing_data development \
  --market "$M5_MARKET_ROOT" \
  --previous "$M2_RECONCILIATION_JSON" \
  --output "$PRIVATE_ROOT/development-data-reconciliation.json"
```

The utility reads only the 36 training and 24 development dates registered in the
market root (every date must precede or equal 2017-04-13). Prior stock-day
reconciliation rows are reused only when all three projected raw table paths
refer to the same underlying files. Missing rows are reconstructed using the
registered SZSE replay logic, with sequence IDs verified against the existing
cache. It does not save or replace any day cache. Both registered session windows
are checked using timing derived from the training-only injection protocol;
training anchors are descriptive rather than formal rollout evidence. The
one-level matching rate means both bid and ask prices matched at the aligned
snapshot. Both it and ten-level log-depth MAE aggregate across all observed quote
snapshots, weighted by stock-day snapshot counts.

## After freeze and holdout acquisition only

Run the frozen study's original holdout acquisition first. Only then run:

```sh
python -m experiments.p1_m5_closing_data enrich-receipt \
  --artifact "$M5_ARTIFACT_ROOT" \
  --development-file "$PRIVATE_ROOT/development-data-reconciliation.json"
```

The utility validates the M5 frozen manifest, all 90 development trial slots,
source revision, development summary, and registered 20-day holdout receipt
**before** calling any remote inventory API. It lists the pinned Hugging Face
repository tree for 2017-06-01 onward. The output counts complete date bundles
(quote, order, and trade Parquet files), reports partial days separately, and
explicitly identifies that market-wide date coverage cannot establish stock-wise
presence. It does not mount or read any later market contents. The optional
`--footers` additionally uses strictly bounded metadata-only HTTP byte ranges for
Parquet footers and trailers (never data pages), inspecting only `wind_code`
row-group min/max, null counts and row counts. Matching endpoints can establish
stock presence, but a stock in the interior of a bound or an unavailable footer
remains unresolved. A stock-day must be confirmed in all three tables; lower and
upper bounds are reported if any remain unresolved. Installed `huggingface_hub`,
`requests`, and PyArrow plus existing authorized Hugging Face access are needed
for the optional remote operation; CI tests use synthetic metadata and do not
contact the service.

The receipt is updated atomically under the existing acquisition lock. Original
holdout receipt evidence remains unchanged; it gains the reporting fields:

- `development_reconciliation`: 120 train/dev stock-day rows.
- `development_anchors`: 240 session checks (`split`, `valid`, `spread_ticks`,
  and reason), with no private sequence ID.
- `split_summary`: six split/stock summary rows with `trading_days`, `n_events`,
  `n_snapshots`, `valid_bilateral_windows`, `planned_windows`,
  `touch_match_rate`, and `top10_log_vol_mae`.
- `summary_method`: weighting and training-anchor interpretation.
- `remote_availability`: pinned revision, timestamp, market-wide date/month and
  table counts, complete/partial dates, and per-stock coverage/status. Per-stock
  exact `trade_day_files` or `months` can be null; metadata-based bounds, if
  measured, are represented separately. No claim about valid positive-spread
  sessions, suspension or universe membership is inferred from file listing.

Raw Level-2 files, caches, provider credentials and original exchange sequence
IDs stay outside the repositories.
