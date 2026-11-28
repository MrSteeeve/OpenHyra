# P2-M4a SZSE Level-2 Q_rec admission scan, train-only Z stratification, and 256-step permutation nulls (2026-11-28)

This sealed bundle records the zero-API (`$0`), zero-training P2-M4a data-quality and stratification audit on Shenzhen Stock Exchange (SZSE) Level-2 tick-by-tick data (`venvoo/china-a-share-l2-level2-limit-order-book-tick-data`, pinned revision `1faacb93f0c07e72f49659c6879008f96782c238`):

1. **Stage 1 (5-day fast `Q_rec` gate across 24 SZSE candidate stocks)**: projected HTTP Range scan on `train_dates[:5]` (`20170601–20170607`) across Main Board (`000xxx.SZ`), SME (`002xxx.SZ`), and ChiNext (`300xxx.SZ`), spanning low/mid/high price tiers (`top1_quote_match_rate >= 0.95`, `invalid_spread_rate <= 0.01`, zero suspensions or full-day limit lockups).
2. **Stage 2 (train-only orthogonal `4 + 6 + 1` stock pool lock & 48-day slice seal)**: strictly prior to opening `dev_dates` (`12` days) or `holdout_test_dates` (`20` days), deterministic rank-quadrant selection on `16` `train_dates` (`20170601–20170622`) locks `4` `exploration_pool` stocks (`000725.SZ`, `002024.SZ`, `002304.SZ`, `000661.SZ`), `6` `unseen_validation_pool` stocks (`000001.SZ`, `000651.SZ`, `002230.SZ`, `002475.SZ`, `002821.SZ`, `000858.SZ`), and `1` `low_qrec_control` stock (`300001.SZ`), sealing `SHA256` fingerprints for all `11 × 48 = 528` external `.npz` slices alongside an offline covariate drift audit (`drift_audit.json`, never read by discovery loops).
3. **Task 3 (500-draw permutation null calibration)**: 500 independent random trade-sign permutations and 10ms-clock Poisson/exponential + global wait-shuffle draws per stock on 256-event training windows, verifying exact agreement with closed-form finite-window expectations in `discovery_loop.finite_window_nulls`.

No raw `.parquet` or derived `.npz` market files are stored in Git. No `P2-M4b` API experiments were started.

## Baseline commits

- Stochopia [`2533584477cda21b24c3f2b7972df741d9762c04`](https://github.com/MrSteeeve/Stochopia/commit/2533584477cda21b24c3f2b7972df741d9762c04)
- OpenHyra [`5fce212aecca40d8fec5cccdc5ef23f4fd811d47`](https://github.com/MrSteeeve/OpenHyra/commit/5fce212aecca40d8fec5cccdc5ef23f4fd811d47)

| Path | Evidence |
| --- | --- |
| `candidate_qrec_scan.csv` | Complete 24-stock Stage-1 5-day and Stage-2 16-day `Q_rec` admission/rejection audit table |
| `locked_m4_manifest.json` | Preregistered 48+8 calendar, locked `4 + 6 + 1` roster, 9-metric `Z_train` (256-window & session), and all `528` `.npz` slice `SHA256` fingerprints |
| `szse_permutation_nulls.json` | 500-draw empirical permutation nulls vs analytical finite-window expectations across all 11 locked stocks |
| `drift_audit.json` | Offline `compute_property_drift(Z_train, Z_holdout)` audit across 20 independent holdout days (never exposed to discovery prompts) |

Run `shasum -a 256 -c SHA256SUMS` in this directory to verify every sealed file. Regenerate the four-table report without API calls or model fitting via:

```bash
python -m experiments.szse_m4a_report artifacts/p2-m4a-szse-stratification /tmp/p2-m4a-report.md
```
