//! Replays the notebook's §8 simulated site (scripts/export_golden_scene.py → testdata/golden/scene.json):
//! same boxes in, and the Rust scene must produce the same track ids, geometry, zone clearances, fall /
//! intrusion events, rule alerts and — byte for byte — the same SHA-256 evidence ledger.
use serde::Deserialize;
use std::collections::BTreeMap;
use vg360_geom::{camera::Camera, ledger::Entry};
use vg360_scene::{rules::{Ctx, RuleEngine, State, default_rules}, scene::{Event, Scene}, zones::default_zones};

#[derive(Deserialize)] struct CamG { f: f64, cx: f64, cy: f64, h: f64, pitch: f64, yaw: f64 }
#[derive(Deserialize)] struct RecG { foot: [f64; 2], head_z: f64, tilt: f64, track: u64, zones: BTreeMap<String, f64>, down: bool, fallen: bool, events: Vec<serde_json::Value> }
#[derive(Deserialize)] struct CtxG { suspended: bool, pipe_handling: bool }
#[derive(Deserialize)] struct FrameG { boxes: Vec<[f64; 4]>, recs: Vec<Option<RecG>>, ctx: Option<CtxG> }
#[derive(Deserialize)] struct StepG { key: u64, fired: Vec<String> }
#[derive(Deserialize)] struct Golden { camera: CamG, dt: f64, frames: Vec<FrameG>, rule_steps: Vec<StepG>, ledger: Vec<Entry> }

fn close(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() <= tol * (1.0 + b.abs()) }

#[test]
fn simulated_site_replays_identically() {
    let g: Golden = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/golden/scene.json")).unwrap()).unwrap();
    let c = &g.camera;
    let cam = Camera { f_px: c.f, cx: c.cx, cy: c.cy, height_m: c.h, pitch_deg: c.pitch, yaw_deg: c.yaw, roll_deg: 0.0, x0: 0.0, y0: 0.0 };
    let mut scene = Scene::new(cam, default_zones());
    let mut engine = RuleEngine::new(default_rules(1.0 / g.dt), 0.5);
    let (mut n_rec, mut n_events, mut steps) = (0usize, 0usize, g.rule_steps.iter());
    let mut worst = [0f64; 4]; // foot, head_z, tilt, zone clearance
    for (k, f) in g.frames.iter().enumerate() {
        let t = k as f64 * g.dt;
        let recs = scene.analyse(&f.boxes, t);
        assert_eq!(recs.len(), f.recs.len());
        for (i, (r, w)) in recs.iter().zip(&f.recs).enumerate() {
            match (r, w) {
                (None, None) => {}
                (Some(r), Some(w)) => {
                    n_rec += 1;
                    let ctx = format!("frame {k} box {i}");
                    assert_eq!(r.track, w.track, "{ctx}: track id");
                    worst[0] = worst[0].max((r.foot[0] - w.foot[0]).abs().max((r.foot[1] - w.foot[1]).abs()));
                    worst[1] = worst[1].max((r.head_z - w.head_z).abs());
                    worst[2] = worst[2].max((r.tilt - w.tilt).abs());
                    assert!(close(r.foot[0], w.foot[0], 1e-5) && close(r.foot[1], w.foot[1], 1e-5), "{ctx}: foot {:?} vs {:?}", r.foot, w.foot);
                    assert!(close(r.head_z, w.head_z, 1e-5), "{ctx}: head_z {} vs {}", r.head_z, w.head_z);
                    assert!(close(r.tilt, w.tilt, 1e-5), "{ctx}: tilt {} vs {}", r.tilt, w.tilt);
                    for (z, v) in &r.zones {
                        worst[3] = worst[3].max((v - w.zones[z]).abs());
                        assert!(close(*v, w.zones[z], 1e-5), "{ctx}: zone {z} {v} vs {}", w.zones[z]);
                    }
                    assert_eq!((r.state.down, r.state.fallen), (w.down, w.fallen), "{ctx}: down/fallen");
                    assert_eq!(r.events.len(), w.events.len(), "{ctx}: events {:?} vs {:?}", r.events, w.events);
                    for (e, we) in r.events.iter().zip(&w.events) {
                        n_events += 1;
                        match e {
                            Event::Fall { track, .. } => assert!(we["type"] == "fall" && we["track"] == *track, "{ctx}"),
                            Event::Intrusion { zone, track, .. } => assert!(we["type"] == "intrusion" && we["zone"] == zone.as_str() && we["track"] == *track, "{ctx}"),
                            Event::Rule { .. } => unreachable!(),
                        }
                    }
                }
                _ => panic!("frame {k} box {i}: on-deck mismatch {:?} vs {}", r.as_ref().map(|r| r.foot), w.is_some()),
            }
        }
        // rule engine, per valid record in box order, exactly as the notebook's ENG_P
        let ctx = f.ctx.as_ref().map(|c| Ctx { suspended: c.suspended, pipe_handling: c.pipe_handling })
            .unwrap_or(Ctx { suspended: (t % 40.0) < 26.0, pipe_handling: (t % 30.0) < 18.0 });
        let clip = format!("sim-{}", (t * (1.0 / g.dt)) as i64);
        for r in recs.iter().flatten() {
            let s = State { clear: r.zones.clone(), down: r.state.down, p_helmet: None };
            let fired = engine.step(&r.track.to_string(), &s, &ctx, t, g.dt, &clip);
            for a in &fired { scene.log_rule(t, &a.rule, &a.subject); }
            let w = steps.next().expect("rule step");
            assert_eq!(w.key, r.track);
            assert_eq!(fired.iter().map(|a| a.rule.clone()).collect::<Vec<_>>(), w.fired, "frame {k} track {}", r.track);
        }
    }
    assert!(steps.next().is_none());
    assert_eq!(engine.ledger.entries, g.ledger, "ledger differs");
    eprintln!("{} frames · {n_rec} person records · {n_events} fall/intrusion events · {} rule alerts · ledger identical (head {}…)",
        g.frames.len(), engine.ledger.entries.len(), &engine.ledger.entries.last().unwrap().hash[..12]);
    eprintln!("max |Δ| vs Python: foot {:.2e} m · head {:.2e} m · tilt {:.2e}° · clearance {:.2e} m", worst[0], worst[1], worst[2], worst[3]);
}
