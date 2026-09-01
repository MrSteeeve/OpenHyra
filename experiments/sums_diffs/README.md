# SimpleTES protocol trial runs

Five independent one-proposal smoke trials were run with Codex CLI,
`gpt-5.6-sol`, and trial seeds 1 through 5. Every trial starts from the official
17-element SimpleTES initial construction and uses an isolated Experience Bank.

## Result

- Valid proposal rate: 5/5.
- Crash, timeout, and protocol-violation counts: 0/0/0.
- Strict improvement rate: 1/5.
- Median best-after-one: 1.0597930945472454.
- Best: trial 04 at 1.0734301120997747.
- Four of five candidate canonical hashes equal the seed hash.

The best candidate is `[2,3,5,6,7,10,14,15,18,22,23,26,30,31,33,34,35]`,
with `|A|=17`, `|A+A|=67`, and `|A-A|=61`. The trusted evaluator recomputed
the score from the snapshotted artifact.

Each trial directory contains `manifest.json`, `records.jsonl`, Context analysis,
candidate source/artifacts, and `summary.tsv`. `trial_summary.tsv` is the compact
cross-trial view. `final_evaluator_verification.json` independently re-evaluates
all five saved artifacts with the final evaluator while preserving each original
manifest and its execution-time evaluator hash.
