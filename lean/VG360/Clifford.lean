/-!
# Cl(2,0) product table of the PPE classifier's Clifford convolutions

`CliffordConv2d` (training notebook) folds multivector weights into a real convolution with the
tensor `M[i, j, gpIdx i j] = gpSgn i j` over the blades `1, e₁, e₂, e₁₂` (bitmask order 0..3).
The notebook asserts that its table equals `gpIdx` / `gpSgn` below; this file proves that the
table is the geometric product of Cl(2,0): associative, `e₁² = e₂² = 1`, `e₁e₂ = −e₂e₁`, `e₁₂² = −1`.
Only core Lean is needed (no Mathlib).
-/
namespace VG360.Clifford

/-- blade index of `bᵢ bⱼ` (bitmask xor) -/
def gpIdx : Fin 4 → Fin 4 → Fin 4
  | 0, j => j
  | i, 0 => i
  | 1, 1 => 0 | 1, 2 => 3 | 1, 3 => 2
  | 2, 1 => 3 | 2, 2 => 0 | 2, 3 => 1
  | 3, 1 => 2 | 3, 2 => 1 | 3, 3 => 0
  | _, _ => 0

/-- sign of `bᵢ bⱼ` -/
def gpSgn : Fin 4 → Fin 4 → Int
  | 2, 1 => -1 | 2, 3 => -1
  | 3, 1 => -1 | 3, 3 => -1
  | _, _ => 1

/-- the table as nested lists, exactly as the notebook prints `CL2.gp_idx` / `CL2.gp_sgn` -/
def idxTable : List (List Nat) := (List.finRange 4).map fun i => (List.finRange 4).map fun j => (gpIdx i j).val
def sgnTable : List (List Int) := (List.finRange 4).map fun i => (List.finRange 4).map fun j => gpSgn i j

theorem idxTable_eq : idxTable = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]] := by decide
theorem sgnTable_eq : sgnTable = [[1, 1, 1, 1], [1, 1, 1, 1], [1, -1, 1, -1], [1, -1, 1, -1]] := by decide

/-- blades multiply associatively, signs included: (bᵢbⱼ)bₖ = bᵢ(bⱼbₖ) -/
theorem gp_assoc : ∀ i j k : Fin 4,
    gpIdx (gpIdx i j) k = gpIdx i (gpIdx j k) ∧
    gpSgn i j * gpSgn (gpIdx i j) k = gpSgn j k * gpSgn i (gpIdx j k) := by decide

/-- 1 is the identity -/
theorem gp_one : ∀ i : Fin 4, gpIdx 0 i = i ∧ gpIdx i 0 = i ∧ gpSgn 0 i = 1 ∧ gpSgn i 0 = 1 := by decide

/-- Euclidean signature: e₁² = e₂² = 1 -/
theorem e1_sq : gpIdx 1 1 = 0 ∧ gpSgn 1 1 = 1 := by decide
theorem e2_sq : gpIdx 2 2 = 0 ∧ gpSgn 2 2 = 1 := by decide

/-- e₁e₂ = e₁₂ = −e₂e₁ -/
theorem e1e2_anticomm : gpIdx 1 2 = 3 ∧ gpIdx 2 1 = 3 ∧ gpSgn 1 2 = - gpSgn 2 1 := by decide

/-- the bivector squares to −1 (rotation generator) -/
theorem e12_sq : gpIdx 3 3 = 0 ∧ gpSgn 3 3 = -1 := by decide

/-- every row and column is a permutation: each product blade is hit exactly once -/
theorem gp_latin : ∀ i k : Fin 4, ∃ j : Fin 4, gpIdx i j = k := by decide

end VG360.Clifford
