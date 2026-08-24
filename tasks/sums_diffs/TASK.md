# Task: sums_diffs — maximize the sum-vs-difference exponent C(A)

Construct a finite set A of distinct integers maximizing

    C(A) = log(|A+A| / |A|) / log(|A-A| / |A|)

where A+A = {a+b : a,b in A} and A-A = {a-b : a,b in A}. HIGHER IS BETTER.
For "most" sets C(A) < 1 (differences outnumber sums since addition commutes);
sum-dominant constructions push it above 1. For historical context, SimpleTES
scored approximately 1.144887 under a size-bounded protocol; it is not a
same-protocol reference for the current task.

## Protocol

- You may ONLY modify `solver.py`. It must write `solution.json` in its own
  directory containing `{"A": [list of integers]}`. It may attach the optional
  research object described below.
- Constraints (checked by a trusted evaluator outside your working directory):
  A is interpreted as a finite set (duplicate values are removed), |A| >= 2
  after deduplication, and -1000000 <= a <= 1000000. There is no fixed upper
  bound on |A|.
- The evaluator recomputes |A+A| and |A-A| by exact set enumeration. Your
  reported numbers are not accepted — only `A` and the optional strict
  `research` object are allowed. Artifact size, evaluator time, and evaluator
  memory limits remain authoritative, so scale constructions only when exact
  verification fits those budgets.
- `solver.py` has a hard 180-second timeout. Finish safely before the limit so
  `solution.json` is always complete.
- Python standard library + numpy only. No network access.

## Optional research artifact

Finite search should act as a probe for reusable structure. When an experiment
supports a concrete mechanism, `solution.json` may use the single current
schema below. There are no `v1`/`v2` variants or compatibility aliases:

```json
{
  "A": [0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29],
  "research": {
    "schema": "openhyra-research",
    "scope": "current_task",
    "hypothesis": "A carry-free positional product preserves the digit-level sum and difference counts.",
    "construction": {
      "schema": "openhyra-digit-product",
      "base": 60,
      "digits": [0, 1, 2, 4, 5, 9, 12, 13, 14, 16, 17, 21, 24, 25, 26, 28, 29],
      "levels": 2,
      "check_levels": [1, 2],
      "obligations": [
        {
          "id": "sum_product",
          "type": "product_formula",
          "operation": "sum"
        },
        {
          "id": "diff_product",
          "type": "product_formula",
          "operation": "difference"
        },
        {
          "id": "sum_no_carry",
          "type": "sum_no_carry"
        },
        {
          "id": "signed_bound",
          "type": "signed_digit_bound"
        }
      ]
    },
    "claims": [
      {
        "id": "G1",
        "template": "supporting_lemma",
        "statement": "The checked positional products obey both digitwise product formulas.",
        "depends_on": [],
        "obligation_ids": ["sum_product", "diff_product"]
      }
    ],
    "falsification_plan": "The next exact check that could refute the claim.",
    "proof_sketch": "Optional unverified reasoning."
  }
}
```

Research fields are intentionally narrow:

- `scope` is `current_task` or `all_finite_integer_sets`.
- `construction` is a typed `openhyra-digit-product` object. It defines
  \(A_l=\{\sum_{i=0}^{l-1}d_i b^i:d_i\in D\}\), declares bounded
  `check_levels`, and may request only the allowlisted obligations
  `level_counts`, `product_formula`, `sum_no_carry`, and
  `signed_digit_bound`. The trusted evaluator generates each checked level and
  recomputes its sumset and difference-set counts.
- Claim templates are `observation`, `finite_witness`, `supporting_lemma`,
  `universal_upper_bound`, `approximating_family`, `supremum_eq`, or
  `nonattainment`.
- Every claim explicitly lists `depends_on` and `obligation_ids`. The four
  formal templates also require a normalized rational `target` object with
  integer `numerator` and positive `denominator`.
- `certificates`, when present, remain limited to standalone
  `modular_sum_diff` finite facts.
- Unknown fields, invalid dependencies, and dependency cycles are rejected.
  A false obligation or certificate does not erase the numerical result.

The trusted evaluator writes a separate `evidence.json`. An obligation advances
only to `bounded_checked` after recomputation. If it fails, the ledger records
`refuted`, including the smallest checked counterexample that was found. A
claim records whether its linked obligations were checked or refuted, but the
link itself is not a trusted implication rule: the claim remains `unverified`
until a typed formal proof is accepted. Passing every requested finite check is
still not an asymptotic proof. Natural-language `statement`, `hypothesis`, and
`proof_sketch` fields remain unverified narrative.

## Optional formalization

A formal claim supplies a candidate-discovered rational target, for example:

```json
{
  "id": "U",
  "template": "universal_upper_bound",
  "statement": "Candidate explanation; the trusted theorem type comes from the template and target.",
  "depends_on": [],
  "obligation_ids": [],
  "target": {"numerator": 3, "denominator": 2}
}
```

Each entry in `research.formalization.proofs` contains only a claim ID and an
inline Lean proof term:

```json
{
  "schema": "openhyra-lean4-request",
  "target": "lean4",
  "proofs": [
    {"claim_id": "U", "term": "by\n  ..."}
  ]
}
```

The candidate does not control imports, declaration names, theorem types,
verdicts, or the axiom allowlist. A parent-owned wrapper binds each proof term
to the corresponding trusted type. A claim becomes `formal_checked` only after
an isolated runner compiles the wrapper, then starts a separate audit process
that imports the compiled module and runs the parent-owned declaration and
axiom checks. Candidate compiler output is not an audit channel. The runner
must rematerialize the trusted audit source, expose compile outputs read-only,
attest that output was not truncated, and bind the actual Lean binary,
toolchain, Mathlib commit/tree, and immutable environment. Compilation failure
rejects the proof artifact; it does not by itself refute the mathematical
claim. Inline comments, unbalanced wrapper delimiters, proof holes, declaration
commands, initializers, and direct metaprogram commands are rejected before
Lean runs.

Trusted output states are deliberately distinct:

- linked finite evidence is `bounded_checked`, `contains_refutation`, or
  `not_linked`; it never promotes natural-language claim text;
- a claim is `unverified` until the proof gate promotes its typed theorem to
  `formal_checked`;
- a formal request is `unavailable`, `rejected`, `verified`, or
  `infrastructure_error` after the parent gate runs;
- if typed theorems verify while the same record still contains a trusted
  finite refutation, the aggregate state is
  `formal_checked_with_refutation`, not clean proof completion.

The proof-completion gate requires all four templates below to be
`formal_checked` **in one record at the same normalized rational target**, and
that record must contain zero trusted claim, obligation, and certificate
refutations:

1. `universal_upper_bound`: every admissible finite set has \(C(A)<q\);
2. `approximating_family`: finite sets approach \(q\) arbitrarily closely;
3. `supremum_eq`: \(q\) is the least upper bound;
4. `nonattainment`: no admissible finite set attains \(q\).

If no isolated formal runner is configured, a submitted proof is recorded as
`unavailable`; it never becomes `formal_checked`, and the proof-completion gate
fails closed. The repository defines and validates the runner protocol but does
not currently bundle a production isolated runner or an offline pinned Mathlib
environment. This protocol does not claim that OpenHyra has independently
discovered or proved \(q=2\). A theoretical target must never be emitted as the
numerical score of a finite candidate.
