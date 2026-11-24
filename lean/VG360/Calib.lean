import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.LinearCombination

/-!
# Auto-calibration (`vg360_geom::autocal`, notebook §3)

* **Focal length from two vanishing points.** A camera-frame direction `d` (z forward) vanishes at
  `p + f·(dₓ/d_z, d_y/d_z)`. For two orthogonal directions, `(v₁ − p)·(v₂ − p) = −f²` — the formula
  `from_vanishing_points` inverts.
* **Mount height from people.** Feet and head estimates scale linearly with the mount height `h`, so the head
  height seen with a 1 m probe camera, `m`, gives `h = A / m` for people of height `A`.
-/
namespace VG360.Calib

/-- vanishing point offsets from the principal point, for focal length `f` -/
noncomputable def vp (f dx dy dz : ℝ) : ℝ × ℝ := (f * dx / dz, f * dy / dz)

theorem focal_from_orthogonal_vps (f ax ay az bx by_ bz : ℝ) (ha : az ≠ 0) (hb : bz ≠ 0)
    (horth : ax * bx + ay * by_ + az * bz = 0) :
    (vp f ax ay az).1 * (vp f bx by_ bz).1 + (vp f ax ay az).2 * (vp f bx by_ bz).2 = -f ^ 2 := by
  unfold vp
  simp only
  field_simp
  linear_combination f ^ 2 * horth

/-- head height for a camera at height `h`: the feet ray from (·,·,h) with direction `w` meets the deck at
    parameter `h / (−w₂)`; the head ray `w'` meets the vertical plane through the feet (normal `n`) at
    `s = (h / −w₂)(n·w)/(n·w')`, so the head is at `h + s·w'₂`. Directions do not depend on `h`. -/
noncomputable def headZ (h w2 nw nw' w2' : ℝ) : ℝ := h + (h / (-w2)) * nw / nw' * w2'

theorem head_height_scales (h w2 nw nw' w2' : ℝ) : headZ h w2 nw nw' w2' = h * headZ 1 w2 nw nw' w2' := by
  unfold headZ
  ring

/-- with `m` the head height at a 1 m probe height, `h = A / m` puts that person's head at exactly `A` -/
theorem height_from_people (A w2 nw nw' w2' : ℝ) (hm : headZ 1 w2 nw nw' w2' ≠ 0) :
    headZ (A / headZ 1 w2 nw nw' w2') w2 nw nw' w2' = A := by
  rw [head_height_scales]
  field_simp

end VG360.Calib
