import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring


/-!
# Detector fine-tune facts (training notebook §10, `vg360-infer::yolox`)

* IoU of two boxes with intersection `i` and areas `a`, `b` lies in [0, 1] and is 1 for a box with itself;
  so the YOLOX IoU loss `1 − IoU²` lies in [0, 1] and is 0 exactly for a perfect box.
* Letterboxing scales by `r > 0`: labels mapped into the 416² tensor and detections mapped back are exact
  inverses (`vg360-infer` divides by the same `r` the notebook multiplies by).
* Cell MCC is `VG360.Train.mcc` on the per-cell confusion matrix, so `mcc_abs_le_one` and `mcc_swap` apply.
-/
namespace VG360.Detect

/-- intersection over union from the intersection area and the two box areas -/
noncomputable def iou (i a b : ℝ) : ℝ := i / (a + b - i)

theorem iou_nonneg (i a b : ℝ) (hi : 0 ≤ i) (hu : 0 < a + b - i) : 0 ≤ iou i a b := by
  unfold iou; positivity

/-- the intersection is no larger than either box, so IoU ≤ 1 -/
theorem iou_le_one (i a b : ℝ) (hia : i ≤ a) (hib : i ≤ b) (hu : 0 < a + b - i) : iou i a b ≤ 1 := by
  unfold iou
  rw [div_le_one hu]
  linarith

theorem iou_self (a : ℝ) (ha : 0 < a) : iou a a a = 1 := by
  unfold iou
  have : a + a - a = a := by ring
  rw [this, div_self (ne_of_gt ha)]

/-- the IoU loss term of YOLOX is in [0, 1] … -/
theorem iou_loss_mem (u : ℝ) (h0 : 0 ≤ u) (h1 : u ≤ 1) : 0 ≤ 1 - u ^ 2 ∧ 1 - u ^ 2 ≤ 1 := by
  constructor <;> nlinarith

/-- … and vanishes exactly at IoU = 1 (for IoU ≥ 0) -/
theorem iou_loss_zero (u : ℝ) (h0 : 0 ≤ u) : 1 - u ^ 2 = 0 ↔ u = 1 := by
  constructor
  · intro h
    have : (u - 1) * (u + 1) = 0 := by linarith [h]
    rcases mul_eq_zero.mp this with h' | h'
    · linarith
    · linarith
  · intro h; subst h; ring

/-- letterbox: frame → tensor coordinates (× r) and back (÷ r) are inverse -/
theorem letterbox_roundtrip (x r : ℝ) (hr : 0 < r) : x * r / r = x := by
  field_simp

/-- … and preserve the order of box corners, so a box stays a box -/
theorem letterbox_monotone (x y r : ℝ) (hr : 0 < r) (h : x ≤ y) : x * r ≤ y * r := by
  exact mul_le_mul_of_nonneg_right h hr.le

end VG360.Detect
