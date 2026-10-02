# Bermudan Python program protocol v1

Status: candidate-facing contract for `bermudan_python_search`.

This task evaluates complete Python algorithms. It does not select a trusted
Linear, Expression, or MLP runner. Those representations remain historical
baselines in `bermudan_optimal_stopping`; they are not the admitted search
space of this task.

## Source tree

Each candidate contains a bounded, recursively hashed source tree:

```text
algorithm.py
manifest.json
helpers/*.py   (optional; arbitrary relative depth)
```

`algorithm.py` owns training, representation, model state, inference, and the
continuation or stopping rule. Helper `.py` modules and `.json`/`.toml`
configuration files may be imported from the candidate tree. The harness and
trusted evaluator reject symlinks and unsupported extensions, cap the tree at
64 files, and include every accepted file in the source digest and provenance.
It may define arbitrary finite functions,
classes, loops, numerical procedures, symbolic procedures, internal searches,
or state machines in that file. Training may write any model tree beneath the
evaluator-provided output directory.

The manifest is exactly one of:

```json
{"schema":"openhyra-python-program.v1","interface":"continuation"}
```

```json
{"schema":"openhyra-python-program.v1","interface":"decision"}
```

The manifest declares an interaction interface, not an algorithm family.

## Fit command

The evaluator invokes:

```text
python3 algorithm.py fit --input INPUT_DIR --output MODEL_DIR --seed INTEGER
```

The program must expose the command body as:

```python
def fit(input_dir, output_dir, seed):
    ...
```

`INPUT_DIR` contains:

- `training_paths.npy`: evaluator-generated training paths;
- `payoffs.npy`: evaluator-computed discounted immediate payoffs;
- `discount_factors.npy`: discount factors on the exercise grid;
- `instance.json`: contract and exercise-grid parameters.

The candidate decides how these observations are transformed, fitted, stored,
or searched. There is no required optimizer, loss, feature grammar, network,
regression family, or model-file schema.

## Predict command

At every non-terminal exercise time the evaluator invokes:

```text
python3 algorithm.py predict --model MODEL_DIR --input QUERY_DIR --output RESULT_DIR
```

The program must expose the command body as:

```python
def predict(model_dir, input_dir, output_dir):
    ...
```

`QUERY_DIR` contains:

- `history.npy`: the path prefix through the current exercise time;
- `states.npy`: the current state batch;
- `immediate_payoffs.npy`: evaluator-computed current discounted payoffs;
- `request.json`: the current time index and instance description.

The program writes one NumPy array to `RESULT_DIR/predictions.npy` with the
same leading batch shape as `states.npy`:

- `continuation`: finite real continuation values in discounted payoff units;
- `decision`: booleans or exact zero/one values, where true means exercise.

## Search operations

The search system may create a candidate through:

- whole-program generation from a blank fit/predict scaffold;
- whole-program or subsystem rewrite by an LLM;
- executable AST mutation inside algorithmic functions;
- two-parent program composition with an evolvable prediction combiner.

These are generation operations, not registered solution families. The final
candidate is always the resulting Python program, and measured evaluator
outcomes determine later parent selection.

For the workshop protocol, the harness normalizes Context aliases to four
canonical operator identities: `whole_program_restart`, `ast_mutation`,
`ast_crossover`, and `subsystem_rewrite`. The normalized identity, operator
detail, source-tree digest, parent lineage, matched arm, and candidate seed are
written beside the evaluator result, together with the Proposal layer's actual
operator materialization receipt. A control arm retains the same baseline
source and request budget so the measured contrast is reviewable.

Context-generated hypotheses are required to serialize one of those four
names. For compatibility with an archived packet that omitted an operator, the
Context parser infers it from the declared family and intervention scope before
the Harness builds a proposal slot; task-declared legacy directions retain
their original aliases. After a round, Context reads the bounded
`research/prediction_table.json` projection, while the append-only ledger stays
the lossless provenance record.

For every Python candidate the evaluator can run an independent same-seed fit
and a causal lookahead check. The replay compares the opaque model digest and a
fresh prediction; the lookahead changes only the suffix after the supplied
history and requires identical predictions. Both probes are diagnostic
evidence and never modify the primary score. Per-cell records also include
the exact training-path and discounted-payoff hashes, an evaluator target
stream hash, the actual (folded) fit seed, separate simulator path seed, model-file digests,
and wall time. The candidate's
internal continuation targets remain outside the evaluator observation
boundary.

The workshop bundle includes `manifest.json` containing runner/evaluator and
candidate-source digests, the request matrix, and a numerical replay command.
`artifact_hashes.json` identifies the exported evidence files. The frozen source,
requests and programs reconstruct the numerical results without new model
calls. Backend generation is not guaranteed to reproduce the same text; actual
responses and all failed attempts are retained instead.

## Evidence boundary

Acceptance by this protocol proves only that the candidate is an executable
algorithm evaluated on the configured Bermudan task. It does not prove that
the program is novel, superior outside the measured suite, or mathematically
correct in general. Those claims require repeated search runs, matched
controls, held-out evaluation, and inspection of the discovered mechanism.

Research mode never retries a denied native sandbox without isolation. Probe
and replay durations are measured costs, not zero-filled deterministic fields.
The live workshop runner uses common public requests across all candidates in
a seed. Paired pathwise deltas provide covariance-aware slice standard errors;
when these samples are absent, a labelled conservative SE-sum bound is used.
The Context prediction table joins guided/control results on the declared slice
and keeps this diagnostic separate from the fixed primary score.
