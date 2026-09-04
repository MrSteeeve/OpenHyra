# Feedback-loop four-arm ablation

This is a synthetic protocol test, not a Bermudan result or a claim of general algorithmic superiority.
Budget: 5 seeds × 8 rounds × 4 candidates × 4 arms; every candidate records two objective calls.
The adaptive arms use the same AlgorithmDiscoveryLoop round barrier; all events from one round carry the post-round state hash.
Score-only and static-directional packets are retained in records but are not applied to acquisition state.

| arm | candidates | mean best score | mean simple regret | mean final state cells |
|---|---:|---:|---:|---:|
| score_only | 160 | 0.2357221282697101 | 0.07919964608029853 | 0.0 |
| static_directional | 160 | 0.2993356040522713 | 0.015586170297737345 | 0.0 |
| adaptive_directional | 160 | 0.3149217743500086 | 0.0 | 48.0 |
| adaptive_matched_control | 160 | 0.3149217743500086 | 0.0 | 48.0 |

Interpretation: lower simple regret is better. The synthetic objective is deliberately finite and seed-dependent; use this artifact to verify protocol wiring and equal-budget reporting, not as evidence for a financial or mathematical claim.

Raw data: `records.jsonl`, per-run discovery ledgers under `runs/`, `summary.json`, and `summary.tsv`.
