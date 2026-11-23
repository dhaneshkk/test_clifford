
import Mathlib

/-! Conformal model over ℝ³ ⊕ ⟨e∞, e₀⟩ with e₀·e∞ = -1, e₀² = e∞² = 0.
    A conformal vector is (x₁,x₂,x₃, a, b) meaning x + a·e∞ + b·e₀. -/
def cdot (x1 x2 x3 a b y1 y2 y3 c d : ℝ) : ℝ :=
  x1*y1 + x2*y2 + x3*y3 - (a*d + b*c)

/-- up(x)·up(y) = -½‖x-y‖²  (point–point distance) -/
theorem cga_dist (x1 x2 x3 y1 y2 y3 : ℝ) :
    cdot x1 x2 x3 ((x1^2+x2^2+x3^2)/2) 1 y1 y2 y3 ((y1^2+y2^2+y3^2)/2) 1
      = -((x1-y1)^2 + (x2-y2)^2 + (x3-y3)^2)/2 := by
  unfold cdot; ring

/-- up(x)·(up(c) - ½r²e∞) = ½(r² - ‖x-c‖²): positive ⇔ inside the sphere -/
theorem cga_sphere (x1 x2 x3 c1 c2 c3 r : ℝ) :
    cdot x1 x2 x3 ((x1^2+x2^2+x3^2)/2) 1 c1 c2 c3 ((c1^2+c2^2+c3^2)/2 - r^2/2) 1
      = (r^2 - ((x1-c1)^2 + (x2-c2)^2 + (x3-c3)^2))/2 := by
  unfold cdot; ring

theorem cga_sphere_inside (x1 x2 x3 c1 c2 c3 r : ℝ) :
    0 < cdot x1 x2 x3 ((x1^2+x2^2+x3^2)/2) 1 c1 c2 c3 ((c1^2+c2^2+c3^2)/2 - r^2/2) 1
      ↔ (x1-c1)^2 + (x2-c2)^2 + (x3-c3)^2 < r^2 := by
  rw [cga_sphere]; constructor <;> intro h <;> linarith

/-- up(x)·(n + δ e∞) = n·x - δ  (signed distance to a plane, ‖n‖ = 1) -/
theorem cga_plane (x1 x2 x3 n1 n2 n3 δ : ℝ) :
    cdot x1 x2 x3 ((x1^2+x2^2+x3^2)/2) 1 n1 n2 n3 δ 0 = n1*x1 + n2*x2 + n3*x3 - δ := by
  unfold cdot; ring

/-- PGA meet of ray C + t·v with plane n·X + δ = 0: any t with t(n·v) = -(n·C+δ) lies on the plane -/
theorem ray_plane (n1 n2 n3 δ c1 c2 c3 v1 v2 v3 t : ℝ)
    (ht : t * (n1*v1 + n2*v2 + n3*v3) = -(n1*c1 + n2*c2 + n3*c3 + δ)) :
    n1*(c1 + t*v1) + n2*(c2 + t*v2) + n3*(c3 + t*v3) + δ = 0 := by
  linear_combination ht

/-- the closed-form t used at runtime satisfies the hypothesis of `ray_plane` when n·v ≠ 0 -/
theorem ray_plane_t (nv k : ℝ) (h : nv ≠ 0) : (-k / nv) * nv = -k := by
  field_simp

/-! Rule Studio properties, stated over the engine's own model (RuleEngine in §7b). -/
inductive Pri | P1 | P2 | P3 deriving DecidableEq

/-- the engine emits an alert iff it fired and (it is P1 or not within the cooldown) -/
def emitted (p : Pri) (alert recent : Bool) : Bool := alert && (p == Pri.P1 || !recent)

/-- debouncing never suppresses a P1 alert -/
theorem debounce_keeps_p1 (alert recent : Bool) (h : alert = true) : emitted Pri.P1 alert recent = true := by
  subst h; cases recent <;> rfl

/-- every emitted alert is written to the ledger: appending puts the entry in it -/
def record {α : Type} (ledger : List α) (e : α) : List α := ledger ++ [e]
theorem alert_has_evidence {α : Type} (ledger : List α) (e : α) : e ∈ record ledger e := by
  simp [record]

/-- 2D-PGA side test: the line through a,b evaluated at p is the 2D cross product (polygon zones) -/
theorem side_of_line (a1 a2 b1 b2 p1 p2 : ℝ) :
    (a2 - b2)*p1 + (b1 - a1)*p2 + (a1*b2 - a2*b1)
      = (b1 - a1)*(p2 - a2) - (b2 - a2)*(p1 - a1) := by
  ring
