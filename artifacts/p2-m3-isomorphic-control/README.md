# P2-M3 isomorphic information-control study (2026-11-28)

This is the sealed M3-only evidence for six anonymous research trajectories, each of six rounds, using GPT-6.2 Sol. 36 formal API requests plus one preflight were recorded. The study used public synthetic events only (no private market data). The numerical and scientific decisions were frozen before opening the independent holdout; no candidate was refit on holdout. The study conclusion is PARTIAL, not a demonstrated general causal advantage of True-Z.

## Registered runtime revisions

- Stochopia [`e53ab181c14ac79308c4bf7014b007417bac9c5b`](https://github.com/MrSteeeve/Stochopia/commit/e53ab181c14ac79308c4bf7014b007417bac9c5b)
- OpenHyra [`c5318228c7ff58984081368cd7e08ca2c50cfca2`](https://github.com/MrSteeeve/OpenHyra/commit/c5318228c7ff58984081368cd7e08ca2c50cfca2)

| File | Contents |
| --- | --- |
| `registration.json` | Code, model, fit, data, arm, call, prompt, selection and statistical thresholds |
| `baseline.json` | Shared source, checkpoint hashes, train-only Z, development D, data receipts |
| `initial_control_audit.json` | First round non-Z prompt hash, card set and token counts |
| `handshake.json`, `api_calls.jsonl` | One preflight and all 36 reservations/terminal provider receipts, raw response and actual token usage |
| `trajectories/` | Per-arm source, preregistered request/prediction, raw assistant reply, unified diff, complete results and folded-lineage ledger |
| `freeze.json` | Six final selections and approved Oracle source digests, sealed before holdout generation |
| `env_alpha-holdout-receipt.json`, `env_beta-holdout-receipt.json` | Independent day calendars and hashes tied to the frozen source roster |
| `holdout.json` | All 21 unique approved environment/model evaluations and original B0 paired effects |
| `study_summary.json` | Descriptive alignment, exact paired cross-arm effects, common-panel Regret, total usage and elapsed time |

Check all evidence files from this directory with `shasum -a 256 -c SHA256SUMS`. The local private cache of synthetic arrays and weights is not published; hashes and deterministic generation/fit configurations are public. The generated [five-table report](https://github.com/MrSteeeve/Stochopia/blob/main/docs/research/P2-M3-Isomorphic-Control-Study.md) can be reproduced offline from these recorded results using `python -m experiments.isomorphic_report <artifact-dir> <fresh-report.md>` from OpenHyra with the sibling repositories on PYTHONPATH. That command makes no API request and trains no model. The live discovery command is registered in `experiments/run_isomorphic_control.py`; repeating it would consume new API calls and is not part of this sealed run.
