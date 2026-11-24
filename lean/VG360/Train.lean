import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import VG360.Clifford

/-!
# Training-side guarantees (training notebook, `vg360-rules::validate`)

* **Folding.** A Clifford convolution layer equals the real convolution with folded weights
  `W[k, j] = Σᵢ w[i] · M[i, j, k]` — the exported ONNX graph computes the trained layer.
* **MCC.** The metric every model and rule is gated on: perfect = 1, inverted = −1, invariant under
  swapping the positive / negative class, and always in [−1, 1]. Same zero-denominator convention as
  `vg360_rules::validate::mcc` and scikit-learn (0).
* **SAM.** The inner step `ε = ρ g / ‖g‖` lies on the ρ-sphere and maximizes the first-order loss
  increase `⟪g, e⟫` over the ρ-ball — the perturbation SAM descends from is the worst case.
-/
namespace VG360.Train

open VG360.Clifford

/-! ## Clifford convolution folding -/

/-- the Cayley tensor of Cl(2,0), as the notebook builds it (`_cayley`) -/
def cayley {R : Type*} [Ring R] (i j k : Fin 4) : R := if gpIdx i j = k then (gpSgn i j : R) else 0

/-- geometric product of multivectors over any commutative ring -/
def gp {R : Type*} [CommRing R] (w x : Fin 4 → R) (k : Fin 4) : R := ∑ i, ∑ j, w i * cayley i j k * x j

/-- folded real weight (`einsum("ocixy,ijk->okcjxy")` for one channel pair and one tap) -/
def fold {R : Type*} [CommRing R] (w : Fin 4 → R) (k j : Fin 4) : R := ∑ i, w i * cayley i j k

/-- the folded real layer computes the geometric product of the trained multivector weight -/
theorem fold_correct {R : Type*} [CommRing R] (w x : Fin 4 → R) (k : Fin 4) :
    gp w x k = ∑ j, fold w k j * x j := by
  unfold gp fold
  rw [Finset.sum_comm]
  simp [Finset.sum_mul]

/-- a convolution is a sum of such products over input channels and taps, so folding commutes with it -/
theorem fold_conv {R : Type*} [CommRing R] {ι : Type*} (s : Finset ι) (w x : ι → Fin 4 → R) (k : Fin 4) :
    ∑ t ∈ s, gp (w t) (x t) k = ∑ t ∈ s, ∑ j, fold (w t) k j * x t j := by
  simp [fold_correct]

/-! ## Matthews correlation coefficient -/

/-- MCC from a confusion matrix; 0 when a marginal is empty (sklearn / `validate::mcc`) -/
noncomputable def mcc (tp fp fn_ tn : ℝ) : ℝ :=
  if (tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_) = 0 then 0
  else (tp * tn - fp * fn_) / Real.sqrt ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_))

theorem mcc_perfect (tp tn : ℝ) (htp : 0 < tp) (htn : 0 < tn) : mcc tp 0 0 tn = 1 := by
  have h : (tp + 0) * (tp + 0) * (tn + 0) * (tn + 0) = (tp * tn) ^ 2 := by ring
  have hpos : 0 < tp * tn := mul_pos htp htn
  unfold mcc
  rw [h]
  split_ifs with hd
  · exact absurd hd (by positivity)
  · rw [Real.sqrt_sq hpos.le]
    field_simp
    ring

theorem mcc_inverted (fp fn_ : ℝ) (hfp : 0 < fp) (hfn : 0 < fn_) : mcc 0 fp fn_ 0 = -1 := by
  have h : (0 + fp) * (0 + fn_) * (0 + fp) * (0 + fn_) = (fp * fn_) ^ 2 := by ring
  have hpos : 0 < fp * fn_ := mul_pos hfp hfn
  unfold mcc
  rw [h]
  split_ifs with hd
  · exact absurd hd (by positivity)
  · rw [Real.sqrt_sq hpos.le]
    field_simp
    ring

/-- relabelling (positive ↔ negative) does not change MCC -/
theorem mcc_swap (tp fp fn_ tn : ℝ) : mcc tp fp fn_ tn = mcc tn fn_ fp tp := by
  unfold mcc
  have hd : (tn + fn_) * (tn + fp) * (tp + fn_) * (tp + fp) = (tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_) := by ring
  have hn : tn * tp - fn_ * fp = tp * tn - fp * fn_ := by ring
  rw [hd, hn]

/-- the squared numerator never exceeds the denominator -/
theorem mcc_num_sq_le (tp fp fn_ tn : ℝ) (h1 : 0 ≤ tp) (h2 : 0 ≤ fp) (h3 : 0 ≤ fn_) (h4 : 0 ≤ tn) :
    (tp * tn - fp * fn_) ^ 2 ≤ (tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_) := by
  have key : (tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_) - (tp * tn - fp * fn_) ^ 2
      = (tp * tn + fp * fn_) * (tp * fn_ + fp * tn) + (tp * tn + fp * fn_) * (tp * fp + fn_ * tn)
        + (tp * fn_ + fp * tn) * (tp * fp + fn_ * tn) + 4 * (tp * tn * fp * fn_) := by ring
  have : 0 ≤ (tp * tn + fp * fn_) * (tp * fn_ + fp * tn) + (tp * tn + fp * fn_) * (tp * fp + fn_ * tn)
        + (tp * fn_ + fp * tn) * (tp * fp + fn_ * tn) + 4 * (tp * tn * fp * fn_) := by positivity
  linarith

/-- MCC ∈ [−1, 1] for any confusion matrix -/
theorem mcc_abs_le_one (tp fp fn_ tn : ℝ) (h1 : 0 ≤ tp) (h2 : 0 ≤ fp) (h3 : 0 ≤ fn_) (h4 : 0 ≤ tn) :
    |mcc tp fp fn_ tn| ≤ 1 := by
  unfold mcc
  split_ifs with hd
  · simp
  · have hpos : 0 < Real.sqrt ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)) :=
      Real.sqrt_pos.mpr (lt_of_le_of_ne (by positivity) (Ne.symm hd))
    rw [abs_div, abs_of_pos hpos, div_le_one hpos]
    exact Real.abs_le_sqrt (mcc_num_sq_le tp fp fn_ tn h1 h2 h3 h4)

/-! ## Sharpness-aware minimization -/

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- SAM's perturbation `ε = ρ g / ‖g‖` has norm exactly ρ -/
theorem sam_eps_norm (g : E) (hg : g ≠ 0) (ρ : ℝ) (hρ : 0 ≤ ρ) : ‖(ρ / ‖g‖) • g‖ = ρ := by
  have hn : ‖g‖ ≠ 0 := norm_ne_zero_iff.mpr hg
  rw [norm_smul, Real.norm_eq_abs, abs_div, abs_of_nonneg hρ, abs_norm, div_mul_cancel₀ _ hn]

/-- …and it maximizes the linearized loss increase `⟪g, e⟫` over the ball `‖e‖ ≤ ρ` -/
theorem sam_eps_maximizes (g e : E) (hg : g ≠ 0) (ρ : ℝ) (he : ‖e‖ ≤ ρ) :
    inner ℝ g e ≤ inner ℝ g ((ρ / ‖g‖) • g) := by
  have hn : ‖g‖ ≠ 0 := norm_ne_zero_iff.mpr hg
  rw [real_inner_smul_right, real_inner_self_eq_norm_sq]
  calc inner ℝ g e ≤ ‖g‖ * ‖e‖ := real_inner_le_norm g e
    _ ≤ ‖g‖ * ρ := mul_le_mul_of_nonneg_left he (norm_nonneg g)
    _ = ρ / ‖g‖ * ‖g‖ ^ 2 := by field_simp

end VG360.Train
