# Task: Bermudan Python search — train a continuation policy

Submit an `AlgorithmBundle` whose `train.py` constructs a bounded,
data-only continuation-policy artifact for one Bermudan contract at a time.
The trusted evaluator keeps the financial model and the stopping rule fixed;
the candidate searches the training algorithm and artifact parameters.

This task is an additive Python track.  The historical
`bermudan_optimal_stopping` Feature IR task remains unchanged and continues to
use its own `feature_program.json` seed and protocol.

## Running a task

```bash
python3 harness.py --task bermudan_python_search --run-id python-v1 --v5 --init
python3 harness.py --task bermudan_python_search --run-id python-v1 --v5 --iterations 4
python3 harness.py --task bermudan_python_search --run-id python-v1 --v5 --final-audit
python3 harness.py --task bermudan_python_search --run-id python-v1 --v5 --export-bundle artifacts/python-v1
```

The private audit is a separate one-shot operation.  The harness freezes the
selected public candidates before drawing its private seed; private results
are not returned to proposal or repair prompts.

## Candidate bundle and source boundary

The task protocol is `bermudan-lsmc-algorithm-bundle.v1` and the candidate mode
is `algorithm_bundle`.  Exactly these source files are admitted:

* `train.py` — the candidate entrypoint, run once for every instance/repeat;
* `manifest.json` — a closed declaration of the trusted artifact protocol.

`solve.sh` is task-owned and frozen.  It only copies `manifest.json` to the
required `solution.json` transport artifact.  Generated weights are written
to an evaluator-created output directory during each training cell and are
never part of the submitted source bundle.

The entrypoint is invoked as:

```text
python3 train.py --input INPUT_DIR --output OUTPUT_DIR --seed INTEGER
```

`INPUT_DIR` is read-only and contains exactly four files:

| File | Shape/content |
| --- | --- |
| `training_paths.npy` | positive float64 states `(n_paths, n_times, n_assets)` |
| `payoffs.npy` | evaluator-computed, time-zero-discounted payoffs with the same first two dimensions |
| `discount_factors.npy` | float64 discount factors for the exercise grid |
| `instance.json` | public contract parameters and `openhyra-bermudan-training-instance.v1` schema |

The candidate receives no pricing paths, hidden paths, dual-inner draws,
evaluation request, reference prices, or other candidate outputs.  The seed
is a deterministic cell seed supplied by the trusted evaluator; it is not a
permission to choose evaluator randomness.

## Artifact contract

The seed uses the MLP continuation protocol.  The manifest is exactly:

```json
{
  "schema": "openhyra-policy-spec.v1",
  "runner_type": "mlp",
  "inference_config": {
    "input_dim": "n_assets",
    "layers": [16, 16],
    "activation": "tanh",
    "output_dim": 1,
    "output_clip": [-1000000.0, 1000000.0]
  },
  "output_semantics": "discounted_continuation_value_t0",
  "normalization": "per_step",
  "weight_pattern": "step_{:03d}.npy"
}
```

Other registered candidates may use `continuation-linear.v1` or
`continuation-expression.v1` if their manifest and files pass the same trusted
loader.  Unknown manifest fields, runner types, operations, dimensions, and
artifact files are rejected.

The MLP and linear runners emit continuation values directly in time-zero
discounted currency units.  The expression runner keeps the legacy Feature IR
terminal convention: its AST is evaluated in strike-normalized units at the
current exercise date, and the trusted runner then multiplies by
`strike * exp(-rate * exercise_time)` before the evaluator compares it with a
discounted payoff.  This conversion is part of `continuation-expression.v1`;
an expression must not apply that final conversion a second time.

For `continuation-expression.v1`, `normalization=per_step` normalizes the
state once before the expression is evaluated, so every finance-aware
terminal (including `underlying` and `intrinsic`) consumes the normalized
state; operators then combine those terminal/constant values.  Use
`normalization=none` when a pure logic rule is intended to refer to raw
contract spots and payoff units.

For the MLP protocol, `OUTPUT_DIR` must contain exactly:

* `normalization.json`, with one `{ "mean": [...], "scale": [...] }` record
  for every non-terminal exercise time;
* `step_000.npy`, `step_001.npy`, …, one canonical native float64 vector per
  non-terminal time.  Each vector concatenates every dense layer's weights in
  row-major `(output, input)` order followed by that layer's bias.

The trusted loader checks shape, dtype, finite values, file identity, total
artifact size, and the manifest's protocol limits.  Continuation outputs are
clipped to the protocol range at the runner boundary.  A candidate must not
write a price, stopping decision, confidence interval, path, seed, or metric
file as an artifact.

### Registered pure-logic protocols

Python remains the search surface for non-MLP candidates: `train.py` may fit an
affine model or choose/compile a typed symbolic rule, then emit only the files
below.  The complete, copyable manifests, output trees, executable `train.py`
examples, AST grammar, and unit equations are in
[`POLICY_PROTOCOLS_V1.md`](POLICY_PROTOCOLS_V1.md).

| Protocol | Manifest `schema` / `runner_type` | Per-instance files | Returned value units |
| --- | --- | --- | --- |
| Linear | `continuation-linear.v1` / `linear` | `normalization.json` and `step_{:03d}.npy` | t0-discounted currency |
| Expression | `continuation-expression.v1` / `expression` | `step_{:03d}.json`; optional `normalization.json` | AST strike-normalized current-date units, converted once by the trusted runner to t0 currency |

For the linear runner, each step vector is `[coefficients..., bias]` and is
applied after its per-step state normalization.  For the expression runner,
each step JSON is a bounded AST built from the documented terminals/operators;
there is no `stop` operation.  Finance-aware terminals are divided by strike at
the current exercise date, and the trusted runner applies exactly
`strike * exp(-rate * exercise_time)` once.  Do not apply that conversion in
`train.py`.  With `normalization=per_step`, normalization happens before every
expression terminal; use `normalization=none` for rules that need raw contract
spots/payoffs.

## Financial protocol and scoring

The evaluator fixes correlated risk-neutral geometric Brownian motion,
exercise dates, strike, rates, dividends, volatilities, correlation, payoff,
discounting, and all path budgets.  It fits the candidate on the supplied
training paths, freezes the resulting runner, and then applies it causally to
an independent pricing sample.  At each non-terminal date the trusted
evaluator compares the discounted immediate payoff with the candidate's
continuation value; exercise and maturity settlement are evaluator-owned.

Public search uses the frozen public suite and paired common-random-number
pricing paths against the legacy Ridge LSMC feature baseline.  The score is
the 95% lower confidence bound of the equally weighted,
strike-normalized candidate-minus-baseline lower-bound improvement across
instance/repeat cells.  Higher is better.  A finite score is evidence for
this fixed development suite only.

The private audit derives a hidden multi-product suite from a sealed fresh
seed.  Policy-fit, pricing, dual-outer, and nested-inner streams are
domain-separated.  The evaluator constructs its own conditionally centered
martingale and reports the normalized primal-dual confidence gap (plus its
Q90 term); lower is better.  Raw bound-order reversals are retained as
diagnostics and are never repaired by candidate claims.

Training telemetry (wall time, memory, input and artifact hashes, and status)
is recorded by the trusted supervisor outside the candidate artifact.  It is
not read back as policy data.

## Resource and integrity limits

Each training cell has a bounded wall-time, CPU, memory, file-size, and output
budget from `task.json`.  The source tree is sealed before execution and
checked against the editable-file allowlist.  Candidate code runs in the
training sandbox with no network or host-credential access.  The evaluator
loads only validated data artifacts and never imports candidate modules.

These controls provide reproducible execution and an auditable provenance
chain.  They do not make a public score a theorem, a production price, or a
claim of universal algorithmic superiority.

## Design guidance

Good first experiments keep the protocol unchanged and make one focused
mechanism instantiation per candidate: a small architecture adjustment, a
deterministic normalization rule, a backward cash-flow target, or a regularized
optimizer.  Fit only from the
provided training bundle, preserve finite outputs, and check public gains over
repeated cells before considering a private audit.  Compare MLP, linear, and
typed-expression runners only as separate, explicitly declared artifact
protocols; do not smuggle a new runner through `manifest.json` fields.

## Open algorithm-design portfolio

This task also exposes a small mechanism portfolio in `task.json` under
`mechanism_design`.  The Context Agent may propose several hypotheses in one
round; each hypothesis names a mechanism, a predicted slice-level effect, a
failure condition, and a matched control.  Proposal Agents receive the
portfolio with their candidate-slot identity and may implement or extend a
different mechanism per slot.  This is intentionally an open proposal space,
while execution remains inside the registered artifact protocols and the
trusted evaluator.  A `PROPOSAL.md` note records the selected mechanism and
its hypothesis; the note is descriptive metadata, not evaluator evidence.
The task's `matched_control.enabled` flag is on by default for this track so a
later experiment planner can pair guided and control candidates without
changing the candidate artifact protocol.

## Directional feedback and recursive state

Every public evaluation also emits an `openhyra-feedback-packet.v1` sidecar.
The primary score is unchanged.  The packet separates evaluator-owned
`observed` values from provisional `recommendation` fields, preserves explicit
`not_observed` markers, and adds per-instance/domain-slice effects such as
payoff family, dimension, moneyness, volatility, correlation, and exercise-grid
class.  A deterministic reducer maintains mechanism-by-slice counts, means,
standard errors, confidence intervals, and positive-effect probability.

The state changes only after all candidates and matched controls in a Context
round have finished.  The next Context therefore sees one coherent state
version rather than a completion-order-dependent partial round.  This works
with or without V5; V5 additionally persists the packet/object references and
uses the state alongside island and behavior retrieval.  Private-audit packets
are retained only for audit provenance and never enter the Context-facing
state or a later proposal.
