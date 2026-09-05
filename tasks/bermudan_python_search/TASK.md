# Task: Bermudan whole-program algorithm search

This track searches complete Python algorithms. A candidate is not a choice
among registered continuation runners: it owns training, representation,
model state, inference, and either the continuation rule or the stopping
decision itself.

The financial problem remains evaluator-owned so independently generated
programs can be compared on the same paths and objective. Reusing the
Bermudan benchmark does not make it a new mathematical task; the research
question here is whether program-level search discovers useful algorithmic
structure beyond the historical bounded artifact track.

## Candidate program

The editable candidate consists of a bounded source tree:

- `algorithm.py`: an arbitrary finite Python program;
- `manifest.json`: the two-field interface declaration;
- additional `.py`, `.json`, or `.toml` helper/configuration files at any
  relative depth (up to 64 files and the task source-byte limit). Symlinks,
  undeclared binary extensions, and hidden runtime directories are rejected.
- `solve.sh` remains task-owned and copies the manifest to
  `solution.json` for the harness transport.

The complete command and data contract is specified in
[`PYTHON_PROGRAM_PROTOCOL_V1.md`](PYTHON_PROGRAM_PROTOCOL_V1.md).

The manifest is one of:

```json
{"schema":"openhyra-python-program.v1","interface":"continuation"}
```

```json
{"schema":"openhyra-python-program.v1","interface":"decision"}
```

There is no model-family, layer, feature-grammar, optimizer, loss, or artifact
file menu in this declaration. `manifest.json` remains the only candidate
declaration; the source-tree policy is task-owned and independently checked by
the harness and trusted evaluator.

## Fit lifecycle

For each evaluator-owned instance and seed:

```text
python3 algorithm.py fit --input INPUT_DIR --output MODEL_DIR --seed INTEGER
```

`INPUT_DIR` contains:

| File | Content |
| --- | --- |
| `training_paths.npy` | causal training paths `(n_paths, n_times, n_assets)` |
| `payoffs.npy` | evaluator-computed discounted payoffs |
| `discount_factors.npy` | discount factors on the exercise grid |
| `instance.json` | contract and exercise-grid parameters |

The program may write any model tree beneath `MODEL_DIR`. It may implement
new features, objectives, iterative solvers, symbolic or numerical programs,
ensembles, state machines, internal search, or other finite algorithms.
The CLI should delegate to top-level `fit(...)` and `predict(...)`
functions; this is the structural composition hook, not a restriction on the
algorithm inside those functions.

## Predict lifecycle

At each non-terminal exercise time the evaluator invokes:

```text
python3 algorithm.py predict --model MODEL_DIR --input QUERY_DIR --output RESULT_DIR
```

`QUERY_DIR` contains `history.npy`, `states.npy`,
`immediate_payoffs.npy`, and `request.json`. History stops at the current
exercise time. The program writes exactly one vector to
`RESULT_DIR/predictions.npy`:

- `continuation`: finite discounted continuation values;
- `decision`: booleans or zero/one values indicating exercise now.

The evaluator applies those predictions on independent paths and owns the
reported score. Direct-decision programs supply only their lower-bound policy;
the hidden upper-bound diagnostic comes from a separate evaluator-owned value
approximation rather than pretending the candidate implemented continuation.

## Search semantics

The Context Agent may propose several unrelated mechanisms in one round.
Proposal may rewrite a whole program, mutate executable AST structure,
combine two parent programs, or restart from a blank implementation. The
Experience Bank stores program source, measured outcomes, and full parent
lineage. Scores and slice feedback guide which structures are revised,
combined, or abandoned in later rounds.

The old feature, affine, expression, and small-network implementations remain
useful baselines. They no longer define the admitted search space.

## Running

```bash
python3 harness.py --task bermudan_python_search --run-id program-v1 --v5 --init
python3 harness.py --task bermudan_python_search --run-id program-v1 --v5 --iterations 4
python3 harness.py --task bermudan_python_search --run-id program-v1 --v5 --final-audit
```

Passing this task demonstrates that OpenHyra can generate and evaluate whole
programs under a fixed mathematical verifier. It does not by itself establish
that a scientifically novel algorithm was discovered; that claim still
requires matched controls, multiple seeds, held-out evidence, and inspection
of the resulting mechanism.
