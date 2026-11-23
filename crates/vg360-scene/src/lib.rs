//! VisionGuard360 per-camera scene analysis, ported from the notebook's §4, §5 and §7b.
//!
//! * [`zones`] — CGA sphere / half-space and PGA floor-polygon clearances.
//! * [`pose`] — 3-D body pose by analysis-by-synthesis (19 cuboid hypotheses), rotor tilt.
//! * [`track`] — SE(2) motor tracker (closed form), Hungarian assignment.
//! * [`scene`] — boxes → deck geometry → tracks → zone clearances → fall / intrusion events.
//! * [`rules`] — Rule Studio engine with gap tolerance, priority cooldowns and the SHA-256 ledger.
//!
//! Golden test (`tests/golden.rs`): the notebook's 60-s simulated site (600 frames, 8 018 boxes) replayed
//! frame by frame must give the same tracks, geometry, events, rule alerts and ledger hashes.
pub mod pose;
pub mod rules;
pub mod scene;
pub mod track;
pub mod zones;
