# Repeated seed-anchoring study — 2026-09-06

## Question and fixed design

Does showing the seed source, separately from explicitly asking for nonlocal exploration, change an LLM research agent's ability to generate executable changes of method?

- Models: GPT-5.5, GPT-5.6 Luna, Terra, Sol. Separate Context and Proposal calls use the same model, with medium reasoning effort. No model substitutions.
- Ten paired numerical seeds: 101 through 110. These control environment, training, and evaluation randomness; the CLI does not expose an LLM sampling seed. Each generation is a fresh session, not a deterministic LLM replay.
- Four conditions: source visible/hidden crossed with generic/nonlocal exploration instructions. All use exactly three alternatives and the same plan schema, contract, whole-program restart, and evaluator budget.
- Twenty rounds per model × seed × condition: 3,200 scheduled candidate attempts, at most 6,400 research-agent calls (preflight/recovery costs reported separately).
- Fixed Ridge parent; each condition consumes its own complete public history, including failures and nonwinners, through an identical compact projection. No cross-arm or held-out feedback. This measures feedback-conditioned exploration around a fixed initial algorithm; it does not test evolving parent selection.
- Counterbalanced arm order and deterministic model-order randomization. All 160 attempts of a round finish before the next round. Eight concurrent candidate processes, serial calls within an attempt. Context 180 seconds, Proposal 300 seconds, no candidate repair/retry. Fit 256 paths and 30 seconds, NumPy + standard library only.
- Baseline for every numerical seed. Public evaluation: two instances, 1,024 pricing paths, 128 outer paths × eight inner paths. Held-out evaluation of baseline and every publicly executable candidate begins only after all public rounds finish, with fresh numerical seeds and independent validation.
- Freeze runtime, task contract, seeds, model IDs, limits, schedule, stopping policy, and hashes before generation. Each record links its plan, packet, previous record, protocol, and candidate source. Keep original pilot unchanged.

## Outcomes and inference

Primary research endpoint, after mechanism audit: proportion of independent model × seed × condition trajectories with at least one executable substantive transition by round 20. Supporting outcomes: first transition round (unsuccessful trajectories censored), attempted transitions, distinct verified mechanisms, and stage-specific failure rates. Do not equate self-reported family, different source hash, or a restart operator label with method transition.

Audit learning objective, representation, fitted state, and the actual prediction path. Separate local Ridge refinement, structural change within linear prediction, different deployed method, hybrid/fallback, and uncertain. A neural network name alone is insufficient. Any claim of behavioral change requires common-query execution evidence, not source appearance alone. Review uncertain cases explicitly; retain an audit trail and blind condition/model metadata when feasible. Mechanism labels are not assigned automatically by this runner.

Independent unit is the numerical-seed trajectory (ten per model/condition), not each of its 20 correlated candidates. Report paired seed-level contrasts for source exposure and exploration instruction, interaction, uncertainty, and model-specific results. Small n supports effect estimates and exploratory inference, not a universal creativity claim. Model-to-model differences are descriptive comparisons of these served systems and budgets. Equal time/call caps do not mean equal token cost, FLOPs, or model capacity.

Payoff, confidence-gap score, and uncertainty are separate secondary outcomes. Public score improvement alone is not a payoff gain; retain failure/timeouts in denominators and report actual cost. A successful transition with worse payoff still supports an exploration outcome, not algorithmic superiority. If a method repeats twenty times, repeated source rewrites do not establish twenty innovations.

The explicit-exploration factor bundles mechanism reasoning, analogy boundaries, and a request to select a nonlocal plan. Its effect cannot separately identify analogy, taste, or reasoning quality. Source hiding removes both an anchor and information; a positive effect is an effect of source availability, not proof of a unique psychological mechanism. Four cells isolate these two operational treatments, not every possible cause of seed anchoring.

## Stopping and operational policy

Fixed 20-round horizon; no outcome-dependent early stopping. Save complete-round checkpoints, including rounds 1/5/10/20. Partial data must be marked interim, with all scheduled/started/completed counts. Workshop cutoff may use only disclosed complete paired blocks, without claiming the full run finished.

Account exhaustion, unavailable models, transport failure, source drift, or worker crash pause the supervisor. They are infrastructure events, not negative algorithm results. Preserve all calls and unfinished artifacts; no automatic reroll of an interrupted attempt. Completed records are reused only after provenance verification. Recovery needs a documented amendment, and its cost and selection implications must be reported. No usage-reset credits are redeemed automatically.

The trusted evaluator and Seatbelt protections remain unchanged. Isolated generation workspaces are outside the project instruction tree; prompt-only no-file/no-network instructions are not a confidentiality guarantee. Logs allow auditing instruction violations.

## Additional evidence, before adding more sophistication

1. Inspect the entire plan → source → fit → prediction chain to locate where transitions disappear.
2. Treat ten paired seeds as replication, rather than inflating n with 3,200 dependent candidates.
3. If cross-method proposals become executable but do not improve payoff, report this as exploration/selection separation.
4. A later predeclared no-feedback condition would isolate the contribution of accumulating evidence; it is not tested by these four cells.
5. A later resource-budget sensitivity study could test whether the 30-second NumPy-only evaluator favors Ridge. The present run holds that constraint fixed and cannot rule out this cause.
