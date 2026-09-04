# Bermudan Python search: continuation-policy protocols v1

Status: candidate-facing specification for the two non-MLP runners registered
by `bermudan_python_search`.

The candidate still writes Python in `train.py`.  Python is the search surface:
it may estimate parameters, choose a rule, or compile a symbolic expression.
The trusted evaluator only consumes the data artifact described by
`manifest.json`; it owns simulation, discounting, stopping, scoring, and the
private primal--dual audit.  The historical MLP wire format is specified in
[`../bermudan_optimal_stopping/POLICY_ARTIFACT_SPEC_V1.md`](../bermudan_optimal_stopping/POLICY_ARTIFACT_SPEC_V1.md).

## 1. Shared manifest shape

Both protocols use exactly these six top-level fields.  Unknown fields are not
an extension mechanism and are rejected by the loader.

```json
{
  "schema": "continuation-linear.v1",
  "runner_type": "linear",
  "inference_config": {
    "input_dim": "n_assets",
    "output_dim": 1,
    "output_clip": [-1000000.0, 1000000.0]
  },
  "output_semantics": "discounted_continuation_value_t0",
  "normalization": "per_step",
  "weight_pattern": "step_{:03d}.npy"
}
```

The expression variant changes only the protocol-specific fields shown below:

```json
{
  "schema": "continuation-expression.v1",
  "runner_type": "expression",
  "inference_config": {
    "input_dim": "n_assets",
    "output_dim": 1,
    "output_clip": [-1000000.0, 1000000.0]
  },
  "output_semantics": "discounted_continuation_value_t0",
  "normalization": "none",
  "weight_pattern": "step_{:03d}.json"
}
```

`input_dim` may instead be a fixed integer in the trusted range.  `output_dim`
and `output_clip` have the exact values above; they are not tunable model
hyperparameters.  For `T = len(exercise_times)`, there is one artifact step for
each non-terminal time, so every protocol emits exactly `T - 1` step files.

## 2. Linear protocol (`continuation-linear.v1`)

### Artifact tree

The per-instance output directory contains exactly:

```text
normalization.json
step_000.npy
step_001.npy
...
step_{T-2:03d}.npy
```

`normalization.json` has one record per step:

```json
{
  "steps": [
    {"mean": [100.0, 95.0], "scale": [12.0, 11.0]},
    {"mean": [101.0, 96.0], "scale": [13.0, 12.0]}
  ]
}
```

For resolved dimension `d`, each `step_XXX.npy` is one native-endian,
C-contiguous `float64` vector of length `d + 1`:

```text
[coefficient_0, coefficient_1, ..., coefficient_{d-1}, bias]
```

The trusted runner computes, in order,

```text
x_i = (states - mean_i) / scale_i
C_i = dot(coefficients_i, x_i) + bias_i
```

and clips `C_i` to the protocol interval.  `C_i` is already a time-zero
discounted currency value.  In particular, `payoffs.npy` is already discounted
by the evaluator; do not multiply a fitted target by a second discount factor.

### Minimal executable `train.py`

This is a deliberately simple smoke baseline: it fits a per-step affine model
to the largest future discounted payoff.  It is a valid artifact, not a claim
that this target is the best stopping algorithm.  It uses only the four files
that the training bridge exposes.

```python
import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", required=True, type=int)
args = parser.parse_args()

inp = Path(args.input)
out = Path(args.output)
paths = np.load(inp / "training_paths.npy", allow_pickle=False).astype(np.float64)
payoffs = np.load(inp / "payoffs.npy", allow_pickle=False).astype(np.float64)
steps = paths.shape[1] - 1
d = paths.shape[2]
normalization = []

for i in range(steps):
    states = paths[:, i, :]
    mean = states.mean(axis=0)
    scale = states.std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    x = (states - mean) / scale

    # `payoffs` is already t0-discounted.  This is only a compact smoke target.
    target = np.max(payoffs[:, i + 1 :], axis=1)
    design = np.column_stack((x, np.ones(len(x))))
    theta, *_ = np.linalg.lstsq(design, target, rcond=None)
    flat = np.asarray(theta, dtype=np.float64)
    np.save(out / f"step_{i:03d}.npy", flat, allow_pickle=False)
    normalization.append({"mean": mean.tolist(), "scale": scale.tolist()})

(out / "normalization.json").write_text(
    json.dumps({"steps": normalization}, sort_keys=True, separators=(",", ":")),
    encoding="utf-8",
)
```

The manifest accompanying this file is the linear example in Section 1.  A
candidate may replace the target and regression procedure, but must retain the
same output file shapes and units.

## 3. Expression protocol (`continuation-expression.v1`)

### Artifact tree

The required files are one JSON AST per non-terminal time:

```text
step_000.json
step_001.json
...
step_{T-2:03d}.json
```

`normalization.json` is optional.  If it is present, it has the same `steps`
schema as the linear protocol and is validated for exactly `T - 1` entries.  A
manifest with `normalization: "none"` should normally omit it.  A manifest with
`normalization: "per_step"` should normally emit it; the loader also accepts a
missing file so a hand-written pure rule can remain raw-state based.

Each step file is either the AST object itself (recommended) or exactly one
wrapper object of the form `{"expression": AST}`.  No Python source is parsed
by the trusted runner.

### AST grammar

Every node is a JSON object with the fields implied by its operation.  The
whole tree is bounded to at most 128 nodes and depth 8; constants are finite
numbers in `[-10, 10]`; `spot.asset` is an integer in `[0, 3]` and must exist in
the current instance.  Outputs are finite and clipped at the runner boundary.

Leaves:

| AST object | Meaning before the final unit conversion |
| --- | --- |
| `{"op":"constant","value": c}` | finite scalar `c` |
| `{"op":"time"}` | `exercise_time / maturity` |
| `{"op":"time_to_maturity"}` | `1 - exercise_time / maturity` |
| `{"op":"spot","asset": j}` | `S[j] / strike` |
| `{"op":"mean_spot"}` | mean spot divided by strike |
| `{"op":"max_spot"}` | maximum spot divided by strike |
| `{"op":"min_spot"}` | minimum spot divided by strike |
| `{"op":"basket_spot"}` | weighted basket spot divided by strike |
| `{"op":"underlying"}` | contract underlying divided by strike |
| `{"op":"intrinsic"}` | current contract payoff divided by strike |

Unary nodes have the form `{"op": OP, "arg": EXPR}` and support:
`abs`, `square`, `cube`, `sqrt_abs`, `log1p_abs`,
`exp_neg_abs`, and `reciprocal_one_plus_abs`.

Binary nodes have the form
`{"op": OP, "left": EXPR, "right": EXPR}` and support:
`add`, `subtract`, `multiply`, `divide_safe`, `minimum`, and `maximum`.
`divide_safe` replaces a denominator with magnitude below `1e-8` by a signed
`1e-8`.  There is intentionally no `stop`, `exercise`, price, or metric
operation: the evaluator compares the returned continuation value with the
discounted immediate payoff and makes the stopping decision.

### Units and discounting

The expression AST deliberately follows the legacy Feature IR convention.  Its
finance-aware terminals are in strike-normalized units at the current exercise
date.  After evaluating the AST, the trusted expression runner performs this
single conversion:

```text
continuation_t0 = AST_value * strike * exp(-rate * exercise_time)
```

The evaluator then compares `continuation_t0` with its own time-zero
discounted payoff.  The candidate must not multiply by strike or apply this
discount a second time.  `time` and `time_to_maturity` are dimensionless.

`normalization: "per_step"` normalizes the state once before the expression is
evaluated, so every finance-aware terminal (including `underlying` and
`intrinsic`) consumes the normalized state; operators then combine those
terminal/constant values.  Use `normalization: "none"` for a pure logic rule
that intends to inspect raw contract spots and payoff units.

### Minimal executable `train.py`

This pure logic candidate emits the same bounded AST at every non-terminal
time.  It is intentionally simple and demonstrates that the candidate writes
Python while the trusted runner executes only the resulting typed tree.

```python
import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", required=True, type=int)
args = parser.parse_args()

inp = Path(args.input)
out = Path(args.output)
paths = np.load(inp / "training_paths.npy", allow_pickle=False)
steps = paths.shape[1] - 1

rule = {
    "op": "add",
    "left": {"op": "intrinsic"},
    "right": {
        "op": "multiply",
        "left": {"op": "underlying"},
        "right": {"op": "time_to_maturity"},
    },
}
for i in range(steps):
    (out / f"step_{i:03d}.json").write_text(
        json.dumps(rule, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
```

The accompanying expression manifest is the expression example in Section 1;
because it declares `normalization: "none"`, this program correctly emits no
`normalization.json`.

## 4. Shared boundaries

The two runners are selected only by their registered manifest schema.  A
candidate cannot introduce a new runner by adding a manifest field.  For every
instance/repeat the evaluator:

1. supplies only `training_paths.npy`, `payoffs.npy`, `discount_factors.npy`,
   and `instance.json` to `train.py`;
2. loads and validates the data-only artifact in a fresh cell;
3. freezes the runner and applies it to independent pricing paths; and
4. owns payoff, discounting, exercise, public paired scoring, and private audit.

Training source is still ordinary Python and may implement a neural network,
linear regression, or symbolic search.  The artifact boundary is what keeps
the financial truth and stopping rule evaluator-owned.
