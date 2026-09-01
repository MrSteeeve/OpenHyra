# Trial 04 continuation: 50 additional iterations

This experiment continued from `sol_0001`, the trial 04 construction with
`C(A)=1.0734301120997747`, under `simpletes-sums-diffs-v1`. Both the Context
Agent and Proposal Agent used Codex CLI with `gpt-5.6-sol`; scoring was the
local trusted exact evaluator.

## Execution phases

| Iterations | Workers | Max in flight | Trial seeds | Feedback mode | Checkpoint |
|---:|---:|---:|---:|---|---|
| 1–20 | 2 | 4 | 4001–4020 | bounded asynchronous | `trial_04_plus20` |
| 21 | 2 | 4 | 5021 | interrupted Codex/MCP failure | retained in EB |
| 22–50 | 1 | 1 | 5022–5050 | strict sequential feedback | `trial_04_plus50` |

For every strict-feedback iteration 22–50, the Context metadata contains the
immediately preceding iteration's committed solution id. The EB versions are
consecutive from 23 through 51, and every analysis is linked to its resulting
record.

## Result

- Additional iterations: 50.
- Status: 44 `ok`, 6 `crash`, 0 `timeout`, 0 `violation`.
- Valid-result rate: 88%.
- Distinct canonical hashes among valid continuation results: 11.
- All 46 scored bundle records (seed, original trial 04, and 44 continuation
  results) were independently recomputed with the final evaluator: 46/46 match.
- Best score: `1.0946069773879967` at iteration 35 (`sol_0036`).
- Improvement over the trial 04 parent: `+0.0211768652882220` (1.9728%).
- The run closed 29.64% of the trial04-to-1.144887 reference gap.

## Improvement path

| Iteration | Record | Parent | n | sums | diffs | span | Exact C(A) |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | `sol_0001` | `sol_0000` | 17 | 67 | 61 | 33 | 1.0734301120997747 |
| 11 | `sol_0012` | `sol_0001` | 277 | 4489 | 3657 | 2244 | 1.0794405274361936 |
| 22 | `sol_0023` | `sol_0012` | 277 | 3763 | 3079 | 1881 | 1.0832986300172944 |
| 28 | `sol_0029` | `sol_0023` | 473 | 6899 | 5605 | 3449 | 1.0840172410969450 |
| 34 | `sol_0035` | `sol_0029` | 83 | 315 | 281 | 186 | 1.0936585916425763 |
| 35 | `sol_0036` | `sol_0030` | 66 | 265 | 235 | 158 | **1.0946069773879967** |

The strongest qualitative transition was from tensor/product constructions to
a small dense-core/asymmetric-fringe construction at iterations 34–35.

## Best construction

```text
[0,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,
 73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,
 94,95,96,97,98,99,100,101,102,103,104,105,132,133,134,137,139,
 143,147,151,155,156,157,158]
```

- Canonical set hash:
  `2bc478af5f16d4c38b117028f1e094cd6fb4691499cd2429b8199cb579c9bc63`
- Evaluated artifact SHA-256:
  `d461ad316ef0efe7b2b6671d9121d38a116755c8be90682e5ddf6c0c7c1c6a0e`
- Saved artifact: `solutions/sol_0036/solution.json`.

## Failures

One crash was a candidate assertion error (iteration 14). Five were Codex CLI
failures, predominantly MCP startup/shutdown errors. No scored candidate hit the
180-second task timeout or violated the frozen-file protocol.
