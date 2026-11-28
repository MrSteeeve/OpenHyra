# P2-M3.1 calibrated control and c06 ablation (2026-11-28)

This sealed bundle records the three symmetric Harness calibrations, three independent searches per environment/arm, and the retrospective registered c06 ablation. It contains synthetic data receipts only, no private market data or model credentials. The formal outcome is INCONCLUSIVE under the preregistered joint criterion. No P2-M4 experiment was started.

## Locked research runtime

- Stochopia [`4a659298c12c4703ec5b764fd2472e4871607529`](https://github.com/MrSteeeve/Stochopia/commit/4a659298c12c4703ec5b764fd2472e4871607529)
- OpenHyra [`d28a1eb46920b1a4da2d5abbe66d8949831207dc`](https://github.com/MrSteeeve/OpenHyra/commit/d28a1eb46920b1a4da2d5abbe66d8949831207dc)

| Path | Evidence |
| --- | --- |
| `registration.json`, `baseline.json`, `baseline_calibration_audit.json` | Fixed protocol, same B0/data/checkpoint hashes and prior metrics as M3 |
| `initial_control_audit.json`, `first_round_sampling_audit.json` | Identical non-Z first-round prompts and distinct actual first replies/sources within each of six cells |
| `handshake.json`, `api_calls.jsonl` | Exactly one connectivity check and 108 formal request reservations and terminal receipts, with actual usage |
| `trajectories/rep_0`, `rep_1`, `rep_2` | All 18 six-round tracks, preregistered prompts/predictions, raw replies, candidate sources, gate errors, unified diffs, full paired development results |
| `freeze.json`, `env_*-holdout-receipt.json`, `holdout.json` | Final selections and authorized candidate digests; 101 unique environment/source holdout evaluations with original weights |
| `study_summary.json` | Three-track descriptive statistics, common-day paired bootstrap, cross-arm comparisons, Regret, and offline prior posterior |
| `ablation/` | Exact 12 fits, source variants, data checksums, individual seed and physical-day observations, paired contrasts and factorial-style interaction |

The holdout dates match the earlier M3 registration. They were unavailable to the M3.1 proposal phases and materialized only after all 18 tracks froze, but are not a new dataset relative to historical M3. The ablation is retrospective; A00's topology differs from the 52+8 variants, so the specified interaction is a configuration contrast, not a pure two-operator causal effect.

Run `shasum -a 256 -c SHA256SUMS` here to verify every evidence file. The [six-table report](https://github.com/MrSteeeve/Stochopia/blob/main/docs/research/P2-M3.1-Calibrated-Control-and-Ablation.md) can be regenerated without API or training using `python -m experiments.calibrated_control_report <this-artifact> <fresh-report.md>` from OpenHyra with the sibling repositories on PYTHONPATH. The live runner and zero-API ablation commands are implemented in `experiments/run_isomorphic_control.py --num-replicates 3` and `experiments/synthetic_tau_z/ablate_m3_winner.py`, respectively. For any new research run use separate output and cache locations; do not overwrite this bundle. Raw synthetic arrays and checkpoints remain in the private work/cache directory, with provenance and SHA256 receipts recorded here.
