# Four-family, 20-seed-block sweep

This batch evaluates frozen guided/control source snapshots from the
five-round pilot. It is a seed-robustness experiment, not 80 fresh LLM
search trajectories and not a claim that the four families are already
causally identical in parent lineage.

The canonical snapshots were chosen from the pilot's public results (the
inductive-bias guided snapshot is `sol_0020`), so the estimates are conditional
post-selection seed robustness, not an unbiased confirmatory test of family
discovery. No LLM calls or private hidden-audit calls are made in this batch.

Planned blocks: 20 seeds × 4 families × 2 arms = 160 candidate evaluations.
Each evaluator call uses all four public contract instances × 2 repeats.
The same master seed is reused across families for blocked CRN pairing.

| family | mechanism | complete pairs | guided mean vs Ridge | seed-level transfer mean | nominal 95% CI | positive rate |
|---|---|---:|---:|---:|---|---:|
| representation | `residual_continuation_v1` | 20/20 | -0.0024285185767936716 | 6.92577064974267e-05 | [-0.00017425098588867263, 0.00031276639888352606] | 0.45 |
| estimation | `cross_fitted_targets` | 20/20 | -0.004226534762402379 | -0.00172875847911128 | [-0.0021461737125425856, -0.0013113432456799748] | 0.0 |
| optimization | `boundary_weighted_loss_v1` | 20/20 | -0.005448867712329877 | -0.0029510914290387796 | [-0.0032440661022965236, -0.0026581167557810357] | 0.0 |
| inductive_bias | `symbolic_affine_switch` | 20/20 | -0.00024748841667219374 | 0.002036704056770213 | [0.0018714500475265439, 0.002201958066013882] | 1.0 |

Interpretation rules:
- The primary independent unit is the master-seed block; evaluator cells and repeats are nested.
- `paired_lower_bound_lcb` is higher-is-better and is relative to the fixed Ridge evaluator reference. The seed-level means above are point improvements, not averaged LCBs.
- Controls and guided arms inherit source snapshots from different pilot parent lineages in the inductive-bias family; compare within family first.
- The current evaluator request seed changes training paths, pricing paths, and derived candidate seeds together; this is a run-level seed sweep, not isolated training-noise ablation.
- Some frozen snapshots retain their own candidate initialization seed (for example, representation `sol_0004` and inductive-bias control `sol_0019`), while the affine guided runner is deterministic. This is another reason to treat the sweep as snapshot robustness rather than a pure initialization ablation.
- The affine guided arm uses the `continuation-linear.v1` runner and is materially faster than the MLP controls in wall-clock time; the pair is therefore not an equal-wall-time or equal-FLOP comparison.
- The three MLP control snapshots used for representation, estimation, and optimization are semantically the same uniform-MSE baseline despite different bundle hashes; they are repeated local controls, not three independent baseline methods.
- The evaluator-reported `deterministic_reproduction_passed=true` flag is metadata from the shared evaluator; this batch did not perform a separate duplicate-request replay for every record, so it is not independent replay evidence.
- No private audit is run in this batch. Hidden audit should be stratified by family and include Ridge/control before making a generalization claim.
- Failures and incomplete pairs remain in `records.jsonl` and the planned denominator; no failed observation is silently dropped.

Detailed data: `records.jsonl`, `pairs.jsonl`, `summary.json`, and `summary.tsv`.
For publication portability, absolute checkout prefixes in the `records.jsonl`
source-path fields were normalized to repository-relative paths; numerical and
evaluator fields are unchanged.
