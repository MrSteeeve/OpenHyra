# OpenHyra Bermudan 5-round experiment

Run: `mathai-bermudan-5r-codex-20260903`

Backend: Codex / `gpt-5.6-luna`

Task: `bermudan_python_search` (`bermudan-lsmc-algorithm-bundle.v1`)

Configuration: 5 Context rounds, 4 candidates per round, 10 guided/control pairs, V5 enabled, `trial_seed=1729`.

## Execution outcome

- 20/20 candidate attempts reached the trusted evaluator (`status=ok`).
- All candidates were finite; no evaluator failures or V5 sync errors were observed. An independent duplicate replay was not run (`deterministic_reproduction_passed` is `not_observed` in the evaluator record), so this run does not establish deterministic reproduction.
- The fixed Ridge reference is score `0`; the seeded MLP (`sol_0000`) scored `-0.003420352499`.
- Best public candidate: `sol_0020`, `symbolic_affine_switch`, score `-0.000614566387`.
- The run then completed the one-shot hidden audit; hidden winner was also `sol_0020`.

## Public search by round

`paired_lower_bound_lcb` is higher-is-better.

| round | best candidate | mechanism | score | change vs seed |
|---:|---|---|---:|---:|
| 0 | sol_0004 | residual_continuation_v1 | -0.003231424 | +0.000188928 |
| 1 | sol_0006 | robust_target_loss_v1 control | -0.003420352 | +0.000000000 |
| 2 | sol_0010 | residual_continuation | -0.003420352 | +0.000000000 |
| 3 | sol_0013 | time_conditioned_representation | -0.002549520 | +0.000870832 |
| 4 | sol_0020 | symbolic_affine_switch | -0.000614566 | +0.002805786 |

For `sol_0020`, the evaluator reports mean paired improvement `+0.000156991`, aggregate SE `0.000393659`, and nominal 95% interval `[-0.000614581, +0.000928562]`. Its public LCB therefore remains below zero; this is not a statistically established win over Ridge.

## Matched-control contrasts

| round | mechanism | guided score | control score | guided-control gain |
|---:|---|---:|---:|---:|
| 0 | boundary_weighted_loss_v1 | -0.006270228 | -0.003420352 | -0.002849875 |
| 0 | residual_continuation_v1 | -0.003231424 | -0.003420352 | +0.000188928 |
| 1 | robust_target_loss_v1 | -0.004868452 | -0.003420352 | -0.001448099 |
| 1 | cross_fitted_targets | -0.004807684 | -0.003420352 | -0.001387332 |
| 2 | boundary_weighted_loss* | -0.006515130 | -0.005031261 | -0.001483869 |
| 2 | residual_continuation | -0.003420352 | -0.003420352 | +0.000000000 |
| 3 | time_conditioned_representation | -0.002549520 | -0.002689980 | +0.000140460 |
| 3 | robust_target_loss | -0.004109867 | -0.004410934 | +0.000301068 |
| 4 | fold_ensemble | -0.003568452 | -0.003571844 | +0.000003392 |
| 4 | symbolic_affine_switch | -0.000614566 | -0.003185386 | **+0.002570820** |

The automatic `transfer_supported` threshold is `gain > 0.01`, so all ten rows are labeled `transfer_refuted`; that label is not a significance test. The field named `transfer_gain_standard_error` is currently a relative-gain diagnostic ratio, not an estimator standard error. The symbolic-affine gain is a parent-frontier-local contrast (its common parent is `sol_0004`), not a direct Ridge comparison. The round-2 boundary row is also excluded from causal interpretation because its slot metadata and sealed implementation disagree.

`*` The round-2 boundary control's slot metadata says boundary control, but its `PROPOSAL.md` and source implement cross-fitting. Exclude that row from a clean boundary causal claim.

## Behavior and hidden audit

The public winner's early-exercise rate mean is `0.230652` (cross-instance dispersion `0.055578`; per-instance repeat standard deviations are `0.035400`, `0.009521`, `0.005371`, and `0.018311`), versus `0.3600` for its matched MLP control. Its per-instance rates are basket `0.1692`, max-call `0.3142`, ATM put `0.2456`, high-vol put `0.1936`; all public cells are finite and reproducible.

Hidden audit (`bermudan-python-hidden-v1`, 3 products x 2 repeats, lower-is-better confidence gap):

| candidate | runner | hidden score | mean gap | q90 gap | raw bound order |
|---|---|---:|---:|---:|---|
| sol_0020 | linear | 0.013324509 | 0.009253118 | 0.016285565 | one diagnostic reversal |
| sol_0013 | MLP | 0.056369435 | 0.042052525 | 0.057267637 | all six cells ordered |
| sol_0014 | MLP | 0.066942222 | 0.049777083 | 0.068660556 | all six cells ordered |

This is a ranking among the frozen top three, not a hidden comparison against Ridge or a hidden matched control. `sol_0020` has one hidden cell with a negative raw primal-dual gap (`-0.00043413`); the audit retained it as a diagnostic and did not treat it as a theorem or universal proof.

## Interpretation

The five rounds demonstrate a functioning multi-structure proposal and evaluation loop. The strongest hypothesis is a holdout-gated affine/linear continuation rule: it is much better than its local MLP control and has the lowest hidden confidence gap among the audited finalists. The symbolic-affine direction was available in the task's open mechanism portfolio, so the defensible claim is that the agent selected and concretized a new model family, not that it invented it from an empty space. Its even/odd gate is an internal training-time deterministic gate, not independent validation. The evidence is still exploratory because the public LCB is below zero, the hidden audit has no Ridge/control arm for this mechanism, hidden evaluation remains within the same Bermudan task family, compute/seed equality is declared rather than runtime-measured (the linear arm is substantially faster in wall time), one pair has a slot/source mismatch, and `workers=2` allowed up to three Context rounds to be generated concurrently. A serial, budget-matched, multi-seed rerun plus cross-task transfer is required before claiming causal algorithmic superiority; no formal proof was produced.

Raw data are in `records.jsonl`, `research/matched_controls.jsonl`, `analyses/`, `v5/`, `final_audit.json`, and `summary.tsv` in this bundle.
