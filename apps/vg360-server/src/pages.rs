use std::collections::BTreeMap;

use topcoat::{
    Result,
    context::{Cx, context},
    router::{error::not_found, href, page, path_param},
    view::{View, class, view},
};
use vg360_geom::ledger::Verify;
use vg360_guard::{config as gc, policy::Status};
use vg360_pipeline::Mode;

use crate::state::{Cam, Lens, Site};
use crate::ui::{Trusted, guard_tone};

fn verify_text(v: &Verify) -> (bool, String) {
    match v {
        Verify::Ok { entries, head } => (true, format!("{entries} entries · head {}…", &head[..12])),
        Verify::Broken { index, reason } => (false, format!("broken at entry {index}: {reason}")),
    }
}

fn pct(x: f64) -> String { format!("{:.1}%", x * 100.0) }

fn lens_label(l: Lens) -> &'static str {
    match l { Lens::Pinhole => "Rectilinear", Lens::Fisheye => "Fisheye 190°" }
}

fn alert_counts(site: &Site) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for e in &site.ledger.entries { *m.entry(e.rule.clone()).or_insert(0) += 1; }
    m
}

// ───────────────────────────── overview ─────────────────────────────

#[page("/")]
pub async fn overview(cx: &Cx) -> Result<impl View> {
    let site = context::<Site>(cx);
    let (ok, vtxt) = verify_text(&site.ledger_check);
    let alarms: Vec<&Cam> = site.cams.iter().filter(|c| c.guard != Status::Ok).collect();
    let armed = site.iogp.armed();
    let feed: Vec<_> = site.ledger.entries.iter().rev().take(12).collect();
    let counts = alert_counts(site);
    Ok(view! {
        <section class="kpis">
            <div class="kpi"><b>(site.cams.len())</b><span>"cameras"</span><em>"5 rectilinear · 1 fisheye"</em></div>
            <div class=(class!("kpi", "bad" if !alarms.is_empty()))>
                <b>(alarms.len())</b><span>"camera-guard alarms"</span>
                <em>for c in &alarms { (c.id) " " (c.guard.label()) }</em>
            </div>
            <div class="kpi"><b>(site.ledger.entries.len())</b><span>"alerts in evidence ledger"</span>
                <em>for (k, n) in &counts { (k) " " (n) " · " }</em></div>
            <div class=(class!("kpi", "ok" if ok, "bad" if !ok))>
                <b>if ok { "✓" } else { "✗" }</b><span>"ledger verified in Rust"</span><em>(vtxt)</em>
            </div>
            <div class=(class!("kpi", "bad" if site.iogp.draft))><b>(armed) "/" (site.iogp.rules.len())</b><span>"IOGP rules armed"</span>
                <em>if site.iogp.draft { "draft config — approve in iogp-editor" } else { "approved and MCC ≥ floor" }</em></div>
        </section>
        if let Some(c) = alarms.first() {
            <a class="banner bad" href=(href!(camera, id: c.id))>
                <strong>(c.id) " · " (c.guard.label())</strong>
                if c.mode == Mode::Compensated {
                    " — analytics continue through shift compensation; please re-aim the camera. "
                } else {
                    " — analytics paused on this camera (ground calibration no longer valid). "
                }
                (c.guidance().join(" · "))
                <span class="go">"Open guidance →"</span>
            </a>
        }
        <div class="cols">
            <section class="grid">
                for c in &site.cams {
                    let tone = guard_tone(c.guard);
                    <a class=(class!("card", "cam", "alarm" if c.guard != Status::Ok)) href=(href!(camera, id: c.id))>
                        <div class="thumb">
                            <img src=(c.frame) alt=(c.name) loading="lazy">
                            <span class=(class!("pill", tone))>(c.guard.label())</span>
                            <span class="lens">(lens_label(c.lens))</span>
                        </div>
                        <div class="meta">
                            <div><strong>(c.id)</strong> " " (c.name)</div>
                            <div class="sub">(format!("{:.0} fps · {:.0} ms/frame · h {:.0} m · pitch {:.0}°", c.fps, c.infer_ms, c.height_m, c.pitch_deg))</div>
                            <div class="chips">for r in c.rules { <span class="chip">(*r)</span> }</div>
                        </div>
                    </a>
                }
            </section>
            <aside class="card feed">
                <h3>"Alert feed" <small>"latest 12 · hash-chained"</small></h3>
                <ol>
                    for e in feed {
                        <li>
                            <span class="t">(e.time.clone()) "s"</span>
                            <span class=(class!("chip", "r"))>(e.rule.clone())</span>
                            <span class="who">(e.who.clone())</span>
                            <code>(e.hash[..10].to_string())</code>
                        </li>
                    }
                </ol>
                <a class="more" href=(href!(rules))>"All evidence →"</a>
            </aside>
        </div>
    })
}

// ───────────────────────────── camera ─────────────────────────────

path_param!(id);

#[page("/cameras/{id}")]
pub async fn camera(cx: &Cx) -> Result<impl View> {
    let site = context::<Site>(cx);
    let c = site.cam(id(cx)).ok_or_else(not_found)?;
    let tone = guard_tone(c.guard);
    let overlay = c.ref_quad.map(|q| {
        let pts: Vec<String> = q.iter().map(|p| format!("{:.1},{:.1}", p[0], p[1])).collect();
        let (cx0, cy0) = (q.iter().map(|p| p[0]).sum::<f64>() / 4.0, q.iter().map(|p| p[1]).sum::<f64>() / 4.0);
        Trusted(format!(
            r##"<svg class="ov" viewBox="0 0 1280 720" preserveAspectRatio="none">
<polygon points="{}" class="refbox"/>
<line x1="640" y1="360" x2="{cx0:.0}" y2="{cy0:.0}" class="arrow" marker-end="url(#ah)"/>
<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#38bdf8"/></marker></defs>
<text x="{:.0}" y="{:.0}" class="reflabel">correct view is here</text></svg>"##,
            pts.join(" "), q[0][0] + 14.0, q[0][1].max(0.0) + 34.0))
    });
    let guidance = c.guidance();
    let cam_rules: Vec<_> = c.rules.iter().filter_map(|r| site.iogp.get(r).map(|rr| (*r, rr))).collect();
    let fisheye = matches!(c.lens, Lens::Fisheye);
    Ok(view! {
        <div class="crumbs"><a href=(href!(overview))>"Overview"</a> " / " (c.id)</div>
        <h1>(c.id) " · " (c.name) <span class=(class!("pill", tone))>(c.guard.label())</span></h1>
        <div class="cols wide">
            <section class="card frame">
                <div class="stage">
                    <img src=(c.frame) alt=(c.name)>
                    if let Some(o) = overlay { (o) }
                    if let Some(r) = c.reference {
                        <figure class="inset"><img src=(r) alt="reference view"><figcaption>"reference view"</figcaption></figure>
                    }
                </div>
                <div class="caption">
                    if c.guard == Status::Ok {
                        "Live analytics overlay: ground-plane boxes, zones, track IDs (rendered by the inference worker)."
                    } else {
                        "Camera guard overlay: blue box = where the reference view now sits in the live image. Follow the arrow until the box fills the frame."
                    }
                </div>
            </section>
            <aside class="stack">
                <section class=(class!("card", "guard", tone))>
                    <h3>"Camera guard" <small>"FAST/BRIEF · Hamming · MSAC — pure Rust"</small></h3>
                    <div class="metrics">
                        <div><span>"shift x"</span><b>(pct(c.pose.dx))</b></div>
                        <div><span>"shift y"</span><b>(pct(c.pose.dy))</b></div>
                        <div><span>"rotation"</span><b>(format!("{:.1}°", c.pose.angle))</b></div>
                        <div><span>"zoom"</span><b>(format!("{:.1}%", (c.pose.scale - 1.0) * 100.0 + 0.0).replace("-0.0", "0.0"))</b></div>
                        <div><span>"corner shift"</span><b>(pct(c.pose.corner))</b></div>
                        <div><span>"inliers"</span><b>(c.inliers)</b></div>
                    </div>
                    if guidance.is_empty() {
                        <p class="okmsg">"Position correct — within " (pct(gc::OK_SHIFT)) " shift, " (gc::OK_ROT) "° rotation."</p>
                    } else {
                        <ol class="guide">for g in &guidance { <li>(g.clone())</li> }</ol>
                        if c.mode == Mode::Compensated {
                            <p class="note okline">"Analytics continue: the move is small (≤ 8 % shift, ≤ 5° rotation, ≥ 60 matched points), so detections are mapped back through the measured homography H⁻¹ onto the original deck calibration. Results from this camera carry "
                                <code>"mode: Compensated"</code> ". The alarm stays until the view is re-aimed and aligned for " (gc::OK_HOLD) " s."</p>
                        } else {
                            <p class="note">"Analytics paused: the move is too large to compensate (or the view is blocked / lost). Rules on this camera are suspended until the view is aligned for " (gc::OK_HOLD) " s."</p>
                        }
                        <div class="btns"><button disabled=(true)>"Re-capture reference"</button><button disabled=(true)>"Recalibrate deck points"</button></div>
                    }
                </section>
                <section class="card">
                    <h3>"Ground calibration"</h3>
                    <dl>
                        <dt>"analytics"</dt><dd>(match c.mode { Mode::Aligned => "running", Mode::Compensated => "running · shift-compensated", Mode::Paused => "paused" })</dd>
                        <dt>"method"</dt><dd>(c.calib_method)</dd>
                        <dt>"mount height"</dt><dd>(format!("{:.1} m", c.height_m))</dd>
                        <dt>"pitch"</dt><dd>(format!("{:.0}°", c.pitch_deg))</dd>
                        <dt>"field of view"</dt><dd>(format!("{:.0}°", c.hfov_deg))</dd>
                        <dt>"deck RMS"</dt><dd>(format!("{:.3} m", c.calib_rms_m))</dd>
                        <dt>"lens"</dt><dd>(lens_label(c.lens))</dd>
                    </dl>
                    if fisheye {
                        <p class="note">"Fisheye: detection runs on 3 upright virtual pinhole views; ground points use the Kannala–Brandt ray directly (error 0.005 m at 1 m → 0.039 m at 30 m in the notebook self-test)."</p>
                    }
                </section>
                <section class="card">
                    <h3>"Rules on this camera"</h3>
                    <ul class="rl">
                        for (rid, r) in cam_rules {
                            let pass = r.armed && c.mode != Mode::Paused;
                            <li>
                                <span class="chip">(rid)</span>
                                <span>(r.alert.clone())</span>
                                <span class=(class!("pill", "ok" if pass, "warn" if !pass))>
                                    if c.mode == Mode::Paused { "paused" } else if !r.armed { "shadow" } else if c.mode == Mode::Compensated { "armed · compensated" } else { "armed" }
                                </span>
                            </li>
                        }
                    </ul>
                </section>
            </aside>
        </div>
    })
}

// ───────────────────────────── rules ─────────────────────────────

#[page("/rules")]
pub async fn rules(cx: &Cx) -> Result<impl View> {
    let site = context::<Site>(cx);
    let (ok, vtxt) = verify_text(&site.ledger_check);
    let ir = &site.iogp;
    let vesdb = site.vesdb.as_ref().map(|v| (v.base.clone(), v.checkpoint().ok()));
    Ok(view! {
        <h1>"Rules & evidence"</h1>
        <p class="lede">"Rules are written in controlled English and approved (author + approver) in "<b>"iogp-editor"</b>
            "; it compiles them with APE to first-order logic and a detector configuration. Each camera worker evaluates that configuration with "
            <b>"iogp-runtime"</b>". A rule sounds an alarm only when it is approved "<em>"and"</em>" its validated MCC clears the floor for its severity (critical 0.85 · high 0.75 · medium/low 0.60); otherwise it runs in shadow mode."</p>
        if ir.draft {
            <div class="banner warn"><div><strong>"Draft configuration"</strong>" — "(ir.config_path.clone())" was compiled by iogp-editor but no rule has been approved yet, so every rule runs in shadow mode. Approve the rules in iogp-editor (four-eyes sign-off), export "<code>"config.json"</code>" and set "<code>"VG360_RULES_CONFIG"</code>"."</div></div>
        }
        <section class="card">
            <h3>"IOGP rules" <small>(format!("config SHA-256 {}…", &ir.sha256[..16]))</small></h3>
            <table class="rules">
                <thead><tr><th>"Rule"</th><th>"Controlled English (iogp-editor)"</th><th>"Severity"</th><th>"MCC vs floor"</th><th>"TP / FP / FN"</th><th>"Approval"</th><th>"State"</th></tr></thead>
                <tbody>
                    for r in &ir.rules {
                        let m = r.mcc.unwrap_or(0.0);
                        let cm = r.cm.unwrap_or([0; 4]);
                        let pass = r.mcc.is_some_and(|m| m >= r.floor);
                        <tr>
                            <td><span class="chip">(r.id.clone())</span> <b>(r.alert.clone())</b>
                                <div class="sub">if r.lsr > 0 { (format!("IOGP Life-Saving Rule {}", r.lsr)) } else { "site rule (not an IOGP LSR)" }</div></td>
                            <td><div>(r.text.clone())</div>
                                <ul class="readback">for l in r.readback.iter().skip(1) { <li>(l.clone())</li> }</ul>
                                <code class="fol">(r.fol.clone())</code></td>
                            <td>(r.severity.clone().unwrap_or_default()) <div class="sub">(r.tier.clone())</div></td>
                            <td class="bar">
                                <div class="track">
                                    <div class=(class!("fill", "ok" if pass, "warn" if !pass)) style=(format!("width:{:.1}%", m.max(0.0) * 100.0))></div>
                                    <div class="floor" style=(format!("left:{:.1}%", r.floor * 100.0))></div>
                                </div>
                                <div class="sub">(format!("{m:.3} / floor {:.2}", r.floor))</div>
                                <div class="sub">(r.validated_on.chars().take(48).collect::<String>())</div>
                            </td>
                            <td class="num">(format!("{} / {} / {}", cm[0], cm[1], cm[2]))</td>
                            <td>if let Some(a) = &r.approved_by { <span class="pill ok">"approved"</span><div class="sub">(a.clone())</div> } else { <span class="pill warn">"draft"</span> }</td>
                            <td><span class=(class!("pill", "ok" if r.armed, "warn" if !r.armed))>if r.armed { "armed" } else { "shadow" }</span></td>
                        </tr>
                    }
                </tbody>
            </table>
        </section>
        <div class="cols">
            <section class="card embed">
                <h3>"Rule Studio" <small>"MCC gates, formal logic and the evidence chain — live from this site"</small></h3>
                <iframe src="/rulestudio/index.html" title="VG360 Rule Studio" loading="lazy"></iframe>
            </section>
            <aside class="stack">
                <section class=(class!("card", "verify", "ok" if vesdb.as_ref().is_some_and(|v| v.1.is_some())))>
                    <h3>"Evidence log" <small>"vesdb · RFC 9162 Merkle log"</small></h3>
                    if let Some((base, cp)) = &vesdb {
                        <p class="big">if cp.is_some() { "✓ witness-cosigned" } else { "… waiting for a checkpoint" }</p>
                        <p>(base.clone())</p>
                        if let Some(cp) = cp { <p class="note">(format!("checkpoint size {} · {}", cp["size"], cp["origin"].as_str().unwrap_or("")))</p> }
                    } else {
                        <p class="big">"not connected"</p>
                        <p class="note">"Set "<code>"VG360_VESDB_URL"</code>", "<code>"VG360_VESDB_WRITE_TOKEN"</code>" and "<code>"VG360_VESDB_AUDIT_TOKEN"</code>". Every raised alert is then written as an incident ("<code>"hazard-detected"</code>" + "<code>"alert-raised"</code>", frame SHA-256, ruleset and model hashes) with a signed receipt; bundles verify offline with "<code>"vesdb-verify"</code>"."</p>
                    }
                </section>
                <section class=(class!("card", "verify", "ok" if ok, "bad" if !ok))>
                    <h3>"Notebook ledger" <small>"Python → Rust interop"</small></h3>
                    <p class="big">if ok { "✓ chain intact" } else { "✗ chain broken" }</p>
                    <p>(vtxt)</p>
                    <p class="note">"The simulated site's hash chain written by the notebook, re-verified in Rust on load."</p>
                </section>
            </aside>
        </div>
    })
}

// ───────────────────────────── twin ─────────────────────────────

#[page("/twin")]
pub async fn twin(cx: &Cx) -> Result<impl View> {
    let site = context::<Site>(cx);
    let tw = site.twin.read().unwrap().clone(); // snapshot: the lock must not be held across rendering
    Ok(view! {
        <h1>"Site twin"</h1>
        <p class="lede">"Build the rig from standard components, place cameras and see their coverage and expected alerts. "<b>"Apply to site"</b>" sends the twin to the console: every camera becomes a calibrated camera model for its worker (heading converted to a compass bearing, focal length from FOV and resolution, height above the deck it watches) and is recorded as a "<code>"calibration-set"</code>" event in the evidence log."</p>
        <section class="card embed tall">
            <iframe src="/twin/index.html" title="VG360 Rig Twin" loading="lazy"></iframe>
        </section>
        if let Some(t) = tw.as_ref() {
            <section class="card">
                <h3>"Applied to site" <small>(format!("{} · {} · {} cameras · {} zones · {} rule switches on", t.source, t.applied_at, t.cameras.len(), t.zones, t.rules_on))</small></h3>
                <table class="mini">
                    <thead><tr><th>"Camera"</th><th>"Position (m)"</th><th>"Height above deck"</th><th>"Compass yaw"</th><th>"Pitch"</th><th>"Focal length"</th><th>"Resolution"</th><th>"Check vs twin"</th></tr></thead>
                    <tbody>
                        for c in &t.cameras {
                            let rms = c.rms_px();
                            <tr><td><b>(c.id.clone())</b></td>
                                <td class="num">(format!("{:.0}, {:.0}", c.camera.x0, c.camera.y0))</td>
                                <td class="num">(format!("{:.1} m", c.camera.height_m)) if c.surf > 0.0 { <span class="sub">(format!(" (deck at {:.0} m)", c.surf))</span> }</td>
                                <td class="num">(format!("{:.0}°", c.camera.yaw_deg))</td>
                                <td class="num">(format!("{:.0}°", c.camera.pitch_deg))</td>
                                <td class="num">(format!("{:.0} px", c.camera.f_px))</td>
                                <td class="num">(format!("{}×{}", c.width, c.height))</td>
                                <td>if let Some(r) = rms { <span class=(class!("pill", "ok" if r < 1.0, "warn" if r >= 1.0))>(format!("{r:.2} px · {} pts", c.points.len()))</span> } else { <span class="sub">"no deck points in view"</span> }</td>
                            </tr>
                        }
                    </tbody>
                </table>
                <p class="note">"“Check vs twin” projects the twin's calibration points through the runtime camera model and compares them with the pixels the twin itself predicts (the twin rounds to whole pixels)."</p>
            </section>
        }
    })
}

// ───────────────────────────── training ─────────────────────────────

#[page("/training")]
pub async fn training(cx: &Cx) -> Result<impl View> {
    let site = context::<Site>(cx);
    let r = &site.report;
    let f = |p: &str| r.pointer(p).and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
    let (fp32, int8) = (f("/det_ms_fp32"), f("/det_ms_int8"));
    let ppe_mcc = f("/ppe/mcc");
    let steps = [
        ("1", "Data", "Site clips → frames → labels (CVAT export); synthetic deck crops for PPE smoke tests."),
        ("2", "PPE classifier", "Cl(2,0) Clifford conv, trained with SAM (ρ = 0.05), per-crop standardisation; folded to plain conv for export."),
        ("3", "Detector", "YOLOX-Nano fine-tune with SAM; 3×3 tiling at 1080p; int8 static QDQ only if it is faster on the target CPU."),
        ("4", "Validate", "MCC per event on held-out clips; per-rule MCC vs priority floor written to rule_studio_export.json."),
        ("5", "Export", "ONNX (opset 13) + golden vectors for the Rust crates; SHA-256 manifest."),
        ("6", "Serve", "Rust: vg360-infer (ONNX Runtime CPU) → vg360-geom → rules → ledger → this console."),
    ];
    Ok(view! {
        <h1>"Models & training"</h1>
        <p class="lede">"Training stays in Python (single-cell Colab notebook); the Rust runtime only loads the exported ONNX files and checks them against the notebook's golden vectors."</p>
        <div class="cols">
            <section class="card">
                <h3>"Model registry"</h3>
                <table class="rules">
                    <thead><tr><th>"Model"</th><th>"Format"</th><th>"CPU latency"</th><th>"Quality"</th><th>"Status"</th></tr></thead>
                    <tbody>
                        <tr><td><b>"YOLOX-Nano"</b><div class="sub">"person detector, 416²"</div></td><td>"ONNX fp32"</td>
                            <td class="num">(format!("{fp32:.1} ms"))</td><td>"COCO weights · not yet fine-tuned on site clips"</td><td><span class="pill ok">"serving"</span></td></tr>
                        <tr><td><b>"YOLOX-Nano"</b><div class="sub">"static QDQ int8"</div></td><td>"ONNX int8"</td>
                            <td class="num">(format!("{int8:.1} ms"))</td><td>"same detections on test frame (3 / 3)"</td><td><span class="pill warn">"rejected — slower"</span></td></tr>
                        <tr><td><b>"PPE Clifford-conv"</b><div class="sub">(format!("{} Clifford params → {} folded", r.pointer("/ppe/conv_params_clifford").unwrap_or(&serde_json::Value::Null), r.pointer("/ppe/conv_params_folded_real").unwrap_or(&serde_json::Value::Null)))</div></td>
                            <td>"ONNX fp32"</td><td class="num">(format!("{:.1} ms / 8 crops", f("/ppe/onnx_ms_per_8_crops")))</td>
                            <td>(format!("MCC {ppe_mcc:.3} on synthetic crops"))</td><td><span class="pill warn">"needs real crops"</span></td></tr>
                    </tbody>
                </table>
                <p class="note">"Clifford conv is not yet shown to beat a plain CNN of equal size; an A/B on real site crops is part of the plan."</p>
            </section>
            <aside class="card">
                <h3>"Notebook pipeline" <small>"notebooks/vg360_train.ipynb"</small></h3>
                <ol class="steps">
                    for (n, t, d) in steps { <li><span class="n">(n)</span><div><b>(t)</b><p>(d)</p></div></li> }
                </ol>
            </aside>
        </div>
        <section class="card">
            <h3>"Golden tests (Python-exported values checked in Rust)"</h3>
            <table class="mini">
                <thead><tr><th>"Test"</th><th>"Checks"</th></tr></thead>
                <tbody>
                    <tr><td><code>"golden::notebook_ledger_verifies_in_rust"</code></td><td>"SHA-256 chain written by Python re-verifies byte-for-byte"</td></tr>
                    <tr><td><code>"golden::notebook_mcc_matches_confusion_matrix"</code></td><td>"Rust MCC = sklearn MCC for every rule; floor matches priority"</td></tr>
                    <tr><td><code>"metrics::notebook_ppe_mcc"</code></td><td>"PPE confusion matrix → 0.8200540091801047"</td></tr>
                    <tr><td><code>"camera::round_trip"</code></td><td>"deck point → pixel → deck point with roll and yaw"</td></tr>
                    <tr><td><code>"policy::moved_needs_hold_then_clears_after_ok_hold"</code></td><td>"camera-guard hold times ported from camera_guard"</td></tr>
                    <tr><td><code>"guard::bumped_camera_is_moved_with_correct_guidance"</code></td><td>"real frames, no OpenCV: known +70 px / −25 px / 3.2° bump recovered (dx 5.47 %, dy −1.98 %, −3.22°)"</td></tr>
                    <tr><td><code>"guard::people_moving_is_not_a_camera_move"</code></td><td>"workers walked around, camera fixed → aligned"</td></tr>
                    <tr><td><code>"infer::detections_and_ppe_match_python_within_1px"</code></td><td>"YOLOX + 3×3 tiles + PPE vs onnxruntime/cv2: boxes within 0.125 px"</td></tr>
                    <tr><td><code>"scene::simulated_site_replays_identically"</code></td><td>"600 frames, 8 018 boxes: same tracks, geometry, events, rule alerts, identical ledger"</td></tr>
                    <tr><td><code>"calib::recovers_notebook_selftest_camera"</code></td><td>"8 deck points, ±0.3 px noise → f, height, pitch, yaw, roll (0.23 px, 0.012 m RMS)"</td></tr>
                </tbody>
            </table>
        </section>
    })
}

// ───────────────────────────── settings ─────────────────────────────

#[page("/settings")]
pub async fn settings(cx: &Cx) -> Result<impl View> {
    let site = context::<Site>(cx);
    let th = [
        ("MOVE_SHIFT", pct(gc::MOVE_SHIFT)), ("MOVE_ROT", format!("{}°", gc::MOVE_ROT)), ("MOVE_SCALE", pct(gc::MOVE_SCALE)),
        ("OK_SHIFT", pct(gc::OK_SHIFT)), ("OK_ROT", format!("{}°", gc::OK_ROT)), ("MOVED_HOLD", format!("{} s", gc::MOVED_HOLD)),
        ("BLOCKED_HOLD", format!("{} s", gc::BLOCKED_HOLD)), ("LOST_HOLD", format!("{} s", gc::LOST_HOLD)), ("MIN_INLIERS", gc::MIN_INLIERS.to_string()),
    ];
    Ok(view! {
        <h1>"Settings"</h1>
        <div class="cols">
            <section class="card">
                <h3>"Camera streams"</h3>
                <p class="note">"Stream URLs (with credentials) are never stored in the repository or shown here. Each camera reads its RTSP URL from an environment variable or the host's secret store."</p>
                <table class="mini">
                    <thead><tr><th>"Camera"</th><th>"Environment variable"</th><th>"Set on this host"</th></tr></thead>
                    <tbody>
                        for c in &site.cams {
                            let set = std::env::var(c.env_var()).is_ok();
                            <tr><td>(c.id) " · " (c.name)</td><td><code>(c.env_var())</code></td>
                                <td><span class=(class!("pill", "ok" if set, "warn" if !set))>if set { "yes" } else { "no — sample frames" }</span></td></tr>
                        }
                    </tbody>
                </table>
            </section>
            <aside class="card">
                <h3>"Camera-guard thresholds" <small>"vg360-guard::config"</small></h3>
                <dl>for (k, v) in th { <dt><code>(k)</code></dt><dd>(v)</dd> }</dl>
                <h3>"Endpoints"</h3>
                <dl>
                    <dt><code>"POST /v1/chat/completions"</code></dt><dd>"OpenAI-compatible ops assistant"</dd>
                    <dt><code>"GET /v1/models"</code></dt><dd>"model list"</dd>
                    <dt><code>"GET /v1/rules"</code></dt><dd>"rules + MCC gates"</dd>
                    <dt><code>"GET /v1/ledger"</code></dt><dd>"evidence chain + verification"</dd>
                    <dt><code>"GET /v1/cameras"</code></dt><dd>"camera + guard status"</dd>
                </dl>
            </aside>
        </div>
    })
}
