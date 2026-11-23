//! Constant-velocity tracker on SE(2) (notebook §5 `Track` / `Tracker`).
//!
//! The notebook keeps each track as a Cl(2,0,1) motor M = T(p)·R(θ) and a translational
//! velocity generator B = log T(v). Because a foot point observes position only, the measurement
//! motor shares the predicted heading, so every motor product in the update reduces to closed form:
//!   predict:  exp(B·Δt)·M        = T(p + vΔt)·R(θ)
//!   correct:  exp(β·log(Mz·M̃p))·Mp = T(p̂ + β(z − p̂))·R(θ)
//! which is what is computed here. Assignment: optimal (Hungarian) on predicted-position distance.
use serde::Serialize;

#[derive(Clone, Copy, Debug)]
pub struct TrackerCfg { pub gate_m: f64, pub max_age_s: f64, pub blend: f64, pub vel_alpha: f64 }

impl Default for TrackerCfg {
    fn default() -> Self { Self { gate_m: 2.5, max_age_s: 1.5, blend: 0.6, vel_alpha: 0.4 } }
}

#[derive(Clone, Debug, Default)]
pub struct FallState {
    /// (t, head_z, tilt, down)
    pub hist: Vec<(f64, f64, f64, bool)>,
    pub fall_latched: bool,
    pub in_zone_since: Vec<(String, f64)>,
    pub zone_fired: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct TrackState { pub down: bool, pub fallen: bool, pub head_z: f64, pub tilt: f64, pub inside: Vec<(String, bool)> }

#[derive(Clone, Debug)]
pub struct Track {
    pub id: u64,
    pub pos: [f64; 2],
    pub heading: f64,
    pub vel: [f64; 2],
    pub t: f64,
    pub hits: u32,
    pub fall: FallState,
    pub state: TrackState,
}

impl Track {
    pub fn predict(&self, t: f64) -> [f64; 2] {
        [self.pos[0] + self.vel[0] * (t - self.t), self.pos[1] + self.vel[1] * (t - self.t)]
    }

    fn update(&mut self, z: [f64; 2], t: f64, cfg: &TrackerCfg) {
        let dt = (t - self.t).max(1e-3);
        let p = self.predict(t);
        let mut hd = self.heading;
        let pn = [p[0] + cfg.blend * (z[0] - p[0]), p[1] + cfg.blend * (z[1] - p[1])];
        let vel = [(pn[0] - self.pos[0]) / dt, (pn[1] - self.pos[1]) / dt];
        if vel[0].hypot(vel[1]) > 0.3 {
            let d = (vel[1].atan2(vel[0]) - hd + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
            let lim = 20f64.to_radians();
            hd += d.clamp(-lim, lim);
        }
        self.pos = pn;
        self.heading = hd;
        self.vel = [cfg.vel_alpha * vel[0] + (1.0 - cfg.vel_alpha) * self.vel[0],
                    cfg.vel_alpha * vel[1] + (1.0 - cfg.vel_alpha) * self.vel[1]];
        self.t = t;
        self.hits += 1;
    }
}

/// Minimum-cost assignment (Hungarian / Kuhn–Munkres with potentials) for an n×m cost matrix.
/// Returns (row, col) pairs covering min(n, m) rows/cols, rows ascending (as scipy's `linear_sum_assignment`).
pub fn assign(cost: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let n = cost.len();
    if n == 0 { return vec![]; }
    let m = cost[0].len();
    if m == 0 { return vec![]; }
    let transpose = n > m;
    let (rows, cols) = if transpose { (m, n) } else { (n, m) };
    let c = |i: usize, j: usize| if transpose { cost[j][i] } else { cost[i][j] };
    // rows ≤ cols; 1-based arrays as in the classic e-maxx formulation
    let (mut u, mut v) = (vec![0.0; rows + 1], vec![0.0; cols + 1]);
    let (mut p, mut way) = (vec![0usize; cols + 1], vec![0usize; cols + 1]);
    for i in 1..=rows {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; cols + 1];
        let mut used = vec![false; cols + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let (mut delta, mut j1) = (f64::INFINITY, 0);
            for j in 1..=cols {
                if !used[j] {
                    let cur = c(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] { minv[j] = cur; way[j] = j0; }
                    if minv[j] < delta { delta = minv[j]; j1 = j; }
                }
            }
            for j in 0..=cols {
                if used[j] { u[p[j]] += delta; v[j] -= delta; } else { minv[j] -= delta; }
            }
            j0 = j1;
            if p[j0] == 0 { break; }
        }
        loop { let j1 = way[j0]; p[j0] = p[j1]; j0 = j1; if j0 == 0 { break; } }
    }
    let mut out: Vec<(usize, usize)> = (1..=cols).filter(|&j| p[j] != 0)
        .map(|j| if transpose { (j - 1, p[j] - 1) } else { (p[j] - 1, j - 1) }).collect();
    out.sort();
    out
}

#[derive(Debug, Default)]
pub struct Tracker { pub cfg: TrackerCfg, pub tracks: Vec<Track>, next_id: u64 }

impl Tracker {
    pub fn new(cfg: TrackerCfg) -> Self { Self { cfg, tracks: vec![], next_id: 1 } }

    /// Feed one frame of foot points; returns the track id for each measurement.
    pub fn step(&mut self, meas: &[[f64; 2]], t: f64) -> Vec<u64> {
        let mut ids = vec![0u64; meas.len()];
        if !self.tracks.is_empty() && !meas.is_empty() {
            let pred: Vec<[f64; 2]> = self.tracks.iter().map(|tr| tr.predict(t)).collect();
            let cost: Vec<Vec<f64>> = pred.iter().map(|p| meas.iter().map(|z| (p[0] - z[0]).hypot(p[1] - z[1])).collect()).collect();
            for (i, j) in assign(&cost) {
                if cost[i][j] < self.cfg.gate_m {
                    let cfg = self.cfg;
                    self.tracks[i].update(meas[j], t, &cfg);
                    ids[j] = self.tracks[i].id;
                }
            }
        }
        for (j, id) in ids.iter_mut().enumerate() {
            if *id == 0 {
                let tr = Track { id: self.next_id, pos: meas[j], heading: 0.0, vel: [0.0; 2], t, hits: 1,
                                 fall: FallState::default(), state: TrackState::default() };
                self.next_id += 1;
                *id = tr.id;
                self.tracks.push(tr);
            }
        }
        let age = self.cfg.max_age_s;
        self.tracks.retain(|tr| t - tr.t <= age);
        ids
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut Track> { self.tracks.iter_mut().find(|t| t.id == id) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hungarian_beats_greedy() {
        // greedy would take (0,0)=1 then (1,1)=10 → 11; optimum is (0,1)+(1,0) = 2+2 = 4
        let c = vec![vec![1.0, 2.0], vec![2.0, 10.0]];
        assert_eq!(assign(&c), vec![(0, 1), (1, 0)]);
    }
    #[test]
    fn hungarian_rectangular() {
        let c = vec![vec![5.0, 1.0, 9.0]];
        assert_eq!(assign(&c), vec![(0, 1)]);
        let c = vec![vec![5.0], vec![1.0], vec![9.0]];
        assert_eq!(assign(&c), vec![(1, 0)]);
    }
    #[test]
    fn constant_velocity_walker_keeps_one_id() {
        let mut tr = Tracker::new(TrackerCfg::default());
        let mut ids = vec![];
        for k in 0..50 { let t = k as f64 * 0.1; ids.extend(tr.step(&[[0.1 * k as f64, 5.0]], t)); }
        assert!(ids.iter().all(|&i| i == 1));
        assert!((tr.tracks[0].vel[0] - 1.0).abs() < 0.05);
    }
}
