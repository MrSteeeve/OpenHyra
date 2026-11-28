# P2-M3.2 open-catalog isomorphic control and Loop 2 single-factor transplants (2026-11-28)

This sealed bundle records the P2-M3.2 two-loop synthetic verification under the simplified open-catalog world-model Harness:
1. **Loop 1 (`R = 2`, 12 trajectories, 72 formal API calls + 1 connectivity check)**: isomorphic three-arm (`d_only`, `true_z_d`, `shuffled_z_d`) open-catalog discovery across `env_alpha` (`W6_thin_hawkes`) and `env_beta` (`W2_thick_long`), followed by delayed 12-day independent holdout verification after global freeze.
2. **Loop 2 (0 API calls, 12 frozen claims across 2 single-factor world pairs)**: zero-API counterfactual transplantation of all 12 promoted trajectory winners into matched `P2-M2` single-factor control worlds (`memory`: `w1_thick_short <-> w2_thick_long`; `thinness`: `w1_thick_short <-> w3_thin_short`), testing each candidate's ex ante declared `(weaker_when_z_metric, weaker_when_z_direction, target_metric)` failure boundary.

It contains synthetic data receipts only, no private market data or model credentials. No `P2-M4` real-market experiment was started.

## Locked research runtime

- Stochopia [`66aa6003ee26c167159c7866bb52fb209204eca9`](https://github.com/MrSteeeve/Stochopia/commit/66aa6003ee26c167159c7866bb52fb209204eca9)
- OpenHyra [`aac75e6add6ec072ebacc29ceb55185cb67cb0ff`](https://github.com/MrSteeeve/OpenHyra/commit/aac75e6add6ec072ebacc29ceb55185cb67cb0ff)

| Path | Evidence |
| --- | --- |
| `registration.json`, `baseline.json`, `baseline_calibration_audit.json` | Preregistered v3.2 protocol (`PROMPT_CAP = 4500`, 11 target metrics, tightened `viability_recovery`, Z boundary schema); identical B0/data/checkpoint hashes to M3 and M3.1 |
| `initial_control_audit.json`, `first_round_sampling_audit.json` | Isomorphic non-Z first-round prompts within each replicate and distinct first replies/sources across replicates |
| `handshake.json`, `api_calls.jsonl` | Exactly 1 connectivity check and 72/72 formal request reservations and completions with complete token metering |
| `trajectories/rep_0`, `rep_1` | All 12 six-round tracks, preregistered prompts/predictions, raw replies, candidate sources, gate errors, unified diffs, paired development deltas |
| `freeze.json`, `env_*-holdout-receipt.json`, `holdout.json`, `study_summary.json` | 12/12 non-B0 trajectory promotions, 71 unique holdout evaluations reusing development checkpoints, R=2 group summaries, and Regret |
| `transplant-freeze.json`, `transplants/` | Sealed Loop 2 roster, frozen sources, per-pair single-factor holdout contrasts (`memory.json`, `thinness.json`), and automated boundary verdicts (`BOUNDARY_CONFIRMED = 4`, `BOUNDARY_REFUTED = 0`, `INCONCLUSIVE = 8`) |

Run `shasum -a 256 -c SHA256SUMS` in this directory to verify every sealed file. Regenerate the five-table report without API calls or model fitting via:

```bash
python -m experiments.open_catalog_report artifacts/p2-m32-open-catalog-control /tmp/p2-m32-report.md
```
