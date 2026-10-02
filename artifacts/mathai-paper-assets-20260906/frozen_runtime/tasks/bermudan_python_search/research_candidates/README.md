# Pre-registered complete-program families

These directories are executable program candidates used for the Bermudan
workshop control grid.  Every directory contains an independent
`algorithm.py` and a two-field `manifest.json`; the evaluator sees only the
fit/predict protocol and never selects a registered model family.  The family
names make the comparison legible and provide matched-budget controls.  They
are not a closed search menu: Context→Proposal may mutate, cross, rewrite, or
replace any complete program and may create files outside this set subject to
the source-tree contract.

| family | interface | intended mechanism |
| --- | --- | --- |
| `linear_ridge` | continuation | standardized log-state ridge baseline |
| `pca_ridge` | continuation | learned low-rank state representation + ridge |
| `gated_ridge` | continuation | two ridge policies selected by a state gate |
| `residual_hybrid` | continuation | ridge plus a trained one-hidden-layer MLP residual |
| `mlp` | continuation | one-hidden-layer NumPy network trained by gradient descent |
| `direct_decision` | decision | evaluator-applied exercise decision threshold |
| `policy_iteration` | continuation | repeated backward Bellman policy sweeps |

The files are complete finite algorithms rather than labels for evaluator
implementations.  Their measured source digests, per-instance fit records,
path/payoff/target hashes, model hashes, seeds, and wall time belong in the
workshop ledger.  Historical artifacts and hand-written feature IR remain
separate baselines.
