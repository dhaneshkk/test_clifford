//! One camera's scene (notebook `box_geometry` / `finish_recs` / `update_rules` / `analyse_boxes`):
//! person boxes → feet on the deck (PGA ray ∧ plane) and head height → fitted 3-D pose → track →
//! zone clearances → fall and intrusion events.
use serde::Serialize;
use vg360_geom::autocal::{feet_on_deck, person_vertical};
use vg360_geom::camera::Camera;

use crate::pose::fit_pose;
use crate::track::{TrackState, Tracker, TrackerCfg};
use crate::zones::{Zone, zone_hits};

#[derive(Clone, Copy, Debug)]
pub struct FallCfg { pub fall_head_z: f64, pub fall_tilt_deg: f64, pub fall_sustain_s: f64, pub stand_head_z: f64, pub intrusion_sustain_s: f64 }

impl Default for FallCfg {
    fn default() -> Self { Self { fall_head_z: 0.8, fall_tilt_deg: 60.0, fall_sustain_s: 1.0, stand_head_z: 1.2, intrusion_sustain_s: 0.3 } }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Fall { t: f64, track: u64, xy: [f64; 2], head_z_m: f64, tilt_deg: f64 },
    Intrusion { t: f64, zone: String, track: u64, clearance_m: f64 },
    Rule { t: f64, rule: String, track: String },
}

#[derive(Clone, Debug, Serialize)]
pub struct Rec {
    pub bbox: [f64; 4],
    pub foot: [f64; 2],
    pub head_z: f64,
    pub tilt: f64,
    pub pose_iou: f64,
    pub track: u64,
    pub zones: Vec<(String, f64)>,
    pub events: Vec<Event>,
    pub state: TrackState,
}

/// Python's round(x, n) for display values (half-to-even on the binary value, like the notebook).
fn round_n(x: f64, n: i32) -> f64 {
    let s = 10f64.powi(n);
    let y = x * s;
    let r = y.round();
    let r = if (y - y.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - y.signum() } else { r };
    r / s
}

/// One box seen by `cam` → (feet origin, head height, tilt, pose IoU); `None` if it cannot stand on the deck.
/// Fisheye cameras call this with the pinhole camera of the virtual view the box was found in.
pub fn box_geometry(cam: &Camera, b: [f64; 4]) -> Option<([f64; 2], f64, f64, f64)> {
    let c = cam.centre();
    // fit_pose is anchored on the box-bottom-centre ray (its hull rendering is self-consistent with it)
    let ft = feet_on_deck(cam, (b[0] + b[2]) / 2.0, b[3])?;
    // head z only: feet and head on one vertical straddling the box centre column (see `person_vertical`)
    let mut hz = person_vertical(cam, b).map_or(0.0, |(_, z, _)| z);
    let fwd = { let r = cam.rotation(); [r[0][2], r[1][2], r[2][2]] };
    if (ft[0] - c[0]) * fwd[0] + (ft[1] - c[1]) * fwd[1] + (ft[2] - c[2]) * fwd[2] <= 0.0 { return None; }
    let pose = fit_pose(cam, b, [ft[0], ft[1]]);
    if pose.tilt_deg > 1.0 { hz = pose.top_z; }
    Some((pose.origin, hz, pose.tilt_deg, pose.iou))
}

pub struct Scene {
    pub cam: Camera,
    pub zones: Vec<Zone>,
    pub tracker: Tracker,
    pub fall: FallCfg,
    /// safety events in order (also receives rule events via [`Scene::log_rule`])
    pub events: Vec<Event>,
}

impl Scene {
    pub fn new(cam: Camera, zones: Vec<Zone>) -> Self {
        Self { cam, zones, tracker: Tracker::new(TrackerCfg::default()), fall: FallCfg::default(), events: vec![] }
    }

    /// One box → (feet origin, head height, tilt, pose IoU); `None` if it cannot stand on the deck.
    pub fn box_geometry(&self, b: [f64; 4]) -> Option<([f64; 2], f64, f64, f64)> { box_geometry(&self.cam, b) }

    fn recent_fall_near(&self, t: f64, xy: [f64; 2]) -> bool {
        let n = self.events.len();
        self.events[n.saturating_sub(200)..].iter().rev().any(|e| matches!(e,
            Event::Fall { t: te, xy: p, .. } if t - te <= 12.0 && (p[0] - xy[0]).hypot(p[1] - xy[1]) < 2.0))
    }

    fn update_rules(&mut self, id: u64, t: f64, head_z: f64, tilt: f64, hits: &[(String, f64)]) -> Vec<Event> {
        let cfg = self.fall;
        let down = head_z < cfg.fall_head_z && tilt > cfg.fall_tilt_deg;
        let (latched_now, pos) = {
            let tr = self.tracker.get_mut(id).expect("track");
            let f = &mut tr.fall;
            f.hist.push((t, head_z, tilt, down));
            f.hist.retain(|h| t - h.0 <= 6.0);
            let stood = f.hist.iter().any(|h| t - h.0 <= 3.0 + cfg.fall_sustain_s && h.1 > cfg.stand_head_z);
            let mut down_for = 0.0;
            for h in f.hist.iter().rev() { if !h.3 { break; } down_for = t - h.0; }
            let fire = down && down_for >= cfg.fall_sustain_s && stood && !f.fall_latched;
            if fire { f.fall_latched = true; }
            (fire, tr.pos)
        };
        let mut out = Vec::new();
        if latched_now {
            let xy = [round_n(pos[0], 2), round_n(pos[1], 2)];
            if !self.recent_fall_near(t, xy) {
                let e = Event::Fall { t: round_n(t, 2), track: id, xy, head_z_m: round_n(head_z, 2), tilt_deg: round_n(tilt, 1) };
                self.events.push(e.clone());
                out.push(e);
            }
        }
        let tr = self.tracker.get_mut(id).expect("track");
        let f = &mut tr.fall;
        if f.fall_latched && f.hist.iter().filter(|h| t - h.0 <= 1.0 && h.1 > cfg.stand_head_z).count() >= 3 {
            f.fall_latched = false;
        }
        let mut intrusions = Vec::new();
        for (zn, clr) in hits {
            if *clr < 0.0 {
                let since = match f.in_zone_since.iter().find(|(z, _)| z == zn) { Some(s) => s.1, None => { f.in_zone_since.push((zn.clone(), t)); t } };
                if t - since >= cfg.intrusion_sustain_s && !f.zone_fired.contains(zn) {
                    f.zone_fired.push(zn.clone());
                    intrusions.push(Event::Intrusion { t: round_n(t, 2), zone: zn.clone(), track: id, clearance_m: round_n(*clr, 2) });
                }
            } else {
                f.in_zone_since.retain(|(z, _)| z != zn);
                f.zone_fired.retain(|z| z != zn);
            }
        }
        tr.state = TrackState { down, fallen: f.fall_latched, head_z, tilt, inside: hits.iter().map(|(z, c)| (z.clone(), *c < 0.0)).collect() };
        self.events.extend(intrusions.iter().cloned());
        out.extend(intrusions);
        out
    }

    /// All person boxes of one frame → one record per box (`None` where the box cannot be on the deck).
    pub fn analyse(&mut self, boxes: &[[f64; 4]], t: f64) -> Vec<Option<Rec>> {
        let geo: Vec<Option<([f64; 2], f64, f64, f64)>> = boxes.iter().map(|b| self.box_geometry(*b)).collect();
        self.analyse_geo(boxes, &geo, t)
    }

    /// Like [`Scene::analyse`] with the per-box geometry already computed (fisheye: each box through its own
    /// virtual view's camera). `boxes` are only carried into the records.
    pub fn analyse_geo(&mut self, boxes: &[[f64; 4]], geo: &[Option<([f64; 2], f64, f64, f64)>], t: f64) -> Vec<Option<Rec>> {
        let valid: Vec<usize> = (0..boxes.len()).filter(|&i| geo[i].is_some()).collect();
        let feet: Vec<[f64; 2]> = valid.iter().map(|&i| geo[i].unwrap().0).collect();
        let ids = self.tracker.step(&feet, t);
        let mut out: Vec<Option<Rec>> = vec![None; boxes.len()];
        for (k, &i) in valid.iter().enumerate() {
            let (foot, head_z, tilt, iou) = geo[i].unwrap();
            let body = [[foot[0], foot[1], 0.1], [foot[0], foot[1], 0.9], [foot[0], foot[1], head_z.max(0.2)]];
            let zones = zone_hits(&self.zones, &body, t);
            let events = self.update_rules(ids[k], t, head_z, tilt, &zones);
            let state = self.tracker.get_mut(ids[k]).unwrap().state.clone();
            out[i] = Some(Rec { bbox: boxes[i], foot, head_z, tilt, pose_iou: iou, track: ids[k], zones, events, state });
        }
        out
    }

    /// Rule alerts share the event log with geometry events (the notebook's EVENT_LOG).
    pub fn log_rule(&mut self, t: f64, rule: &str, track: &str) {
        self.events.push(Event::Rule { t: round_n(t, 2), rule: rule.into(), track: track.into() });
    }
}
