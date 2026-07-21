# Task: sums_diffs — maximize the sum-vs-difference exponent C(A)

Construct a finite set A of distinct integers maximizing

    C(A) = log(|A+A| / |A|) / log(|A-A| / |A|)

where A+A = {a+b : a,b in A} and A-A = {a-b : a,b in A}. HIGHER IS BETTER.
For "most" sets C(A) < 1 (differences outnumber sums since addition commutes);
sum-dominant constructions push it above 1. The reference result under this
exact public protocol is SimpleTES at approximately 1.144887.

## Protocol

- You may ONLY modify `solver.py`. It must write `solution.json` in its own
  directory containing `{"A": [list of integers]}`.
- Constraints (checked by a trusted evaluator outside your working directory):
  A is interpreted as a set (duplicate values are removed), 2 <= |A| <= 512
  after deduplication, and -1000000 <= a <= 1000000.
- The evaluator recomputes |A+A| and |A-A| by exact set enumeration. Your
  reported numbers are ignored — only the emitted set matters.
- `solver.py` has a hard 180-second timeout. Finish safely before the limit so
  `solution.json` is always complete.
- Python standard library + numpy only. No network access.
