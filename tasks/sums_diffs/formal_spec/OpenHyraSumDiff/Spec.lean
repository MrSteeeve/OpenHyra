import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Finset.Pointwise
import Mathlib.Data.Int.Basic
import Mathlib.Order.ConditionallyCompleteLattice.Basic

/-!
Trusted theorem targets for OpenHyra's sum/difference research loop.

The candidate never edits this file. It discovers a rational target `q` and
submits proof terms for one or more of the propositions below. The parent
verifier generates declarations with these exact types before invoking Lean.
-/

set_option autoImplicit false

open scoped Pointwise

namespace OpenHyraSumDiff

noncomputable def sigma (A : Finset ℤ) : ℝ :=
  ((A + A).card : ℝ) / (A.card : ℝ)

noncomputable def delta (A : Finset ℤ) : ℝ :=
  ((A - A).card : ℝ) / (A.card : ℝ)

noncomputable def growthExponent (A : Finset ℤ) : ℝ :=
  Real.log (sigma A) / Real.log (delta A)

def admissibleExponents : Set ℝ :=
  {x | ∃ A : Finset ℤ, 2 ≤ A.card ∧ growthExponent A = x}

/-- Every admissible finite set has exponent strictly below `q`. -/
def UniversalUpperBoundAt (q : ℝ) : Prop :=
  ∀ A : Finset ℤ, 2 ≤ A.card → growthExponent A < q

/-- Admissible finite sets approach `q` from below arbitrarily closely. -/
def ApproximatingAt (q : ℝ) : Prop :=
  ∀ ε : ℝ, 0 < ε →
    ∃ A : Finset ℤ, 2 ≤ A.card ∧
      q - ε < growthExponent A ∧ growthExponent A < q

/-- `q` is a least upper bound of all admissible exponents. -/
def SupremumAt (q : ℝ) : Prop :=
  IsLUB admissibleExponents q

/-- No admissible finite set attains `q`. -/
def NonattainedAt (q : ℝ) : Prop :=
  ∀ A : Finset ℤ, 2 ≤ A.card → growthExponent A ≠ q

end OpenHyraSumDiff
