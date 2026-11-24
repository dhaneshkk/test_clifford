//! VisionGuard360 core, ported from the notebook (`notebooks/vg360_clifford_safety_cpu.ipynb`).
//!
//! The Clifford-algebra derivations live in `lean/CGA.lean`; at runtime they reduce to the
//! closed forms below (ray–plane meet = ground homography, CGA inner product = signed
//! sphere test, 2-D PGA join = side-of-line test). Golden tests check them against values
//! exported by the notebook.
pub mod autocal;
pub mod camera;
pub mod ledger;
pub mod metrics;
pub mod rules;
pub mod twin;
pub mod zones;
