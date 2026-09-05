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

## Workshop evidence loop

The reproducible pilot in
`experiments/run_bermudan_workshop.py` freezes three seeds and two rounds for
both a feedback-conditioned Context-to-Proposal path and a direct-generation
path. Each round runs two guided/control pairs. A pair shares the baseline
parent reference, candidate seed, evaluator request, path budget, and timeout;
the control keeps the baseline source. The executed operator is recorded as
one of `whole_program_restart`, `ast_mutation`, `ast_crossover`, or
`subsystem_rewrite`. After the public rounds, the best successful guided
candidate for every mode/seed receives one private hidden audit.

`research/prediction_ledger.jsonl` is the append-only harness ledger for a live
run. It joins the hypothesis, falsifier, target slice, operator, source digest,
parent lineage, evaluator effect and standard error, failure reason, cost, and
the next action. At the next Context barrier, the bounded tail of its
`prediction_table.json` projection is injected into the Context prompt; its
schema, row count, and digest are retained in Context metadata. For an open
Python hypothesis, Context JSON carries one of the four canonical operators;
when an older packet omits that field, the parser derives the canonical value
from the declared family/scope before Harness dispatch. The workshop bundle
additionally records each candidate's
source tree, evaluator model-file digest, training path/payoff hashes,
evaluator target-stream hash, train seed, and fit wall time. Candidate-owned
continuation targets are marked unobserved rather than inferred from those
input hashes.

Independent validation has two explicit probes for Python programs: a fresh
same-seed fit with model/prediction digest comparison, and a lookahead probe
that changes only the future suffix after a fixed history. These are evidence
fields and do not enter the primary score. `research_mode` is likewise a
provenance bit for the managed-host sandbox fallback; it cannot change the
evaluator's paths, payoff, stopping application, or score arithmetic.

The pre-registered complete-program controls under
`research_candidates/` include linear/ridge, PCA, gated ridge, residual
hybrid, a real NumPy MLP, direct decisions, and repeated policy iteration.
They are measured source artifacts for comparison, not a closed menu for the
open track. The pilot establishes an auditable evaluator-guided open-program
loop; it does not establish an LLM search effect, novelty, or out-of-suite
mathematical superiority.

The bundle's `manifest.json` records the runner and evaluator source digests,
candidate-family digests, request matrix, artifact hashes, and the exact rebuild
command. It makes the reported ledger and summary reconstructible from the
checked-in source tree.

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
