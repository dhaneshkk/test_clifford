use std::collections::BTreeMap;

use topcoat::{
    Result,
    context::{Cx, context},
    router::{error::not_found, href, page, path_param},
    view::{View, class, view},
};
use vg360_geom::ledger::Verify;
use vg360_guard::{config as gc, policy::Status};

use crate::state::{Cam, Lens, Site};
use crate::ui::{Trusted, guard_tone};

fn verify_text(v: &Verify) -> (bool, String) {
    match v {
        Verify::Ok { entries, head } => (true, format!("{entries} entries · head {}…", &head[..12])),
        Verify::Broken { index, reason } => (false, format!("broken at entry {index}: {reason}")),
    }
}

fn pct(x: f64) -> String { format!("{:.1}%", x * 100.0) }


fn alert_counts(site: &Site) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for e in &site.ledger.entries { *m.entry(e.rule.clone()).or_insert(0) += 1; }
    m
}

// ───────────────────────────── camera view model ─────────────────────────────

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PoseView { pub dx: f64, pub dy: f64, pub angle: f64, pub scale: f64, pub corner: f64 }

/// Everything the camera page and the overview need about one camera, whether it is one of the static
/// sample cameras or a live worker (built defensively from the live snapshot: missing fields get defaults).
#[derive(Clone, Debug, Default)]
pub struct CamView {
    pub id: String,
    pub name: String,
    pub is_sample: bool,
    pub status_label: String,
    /// "ok" | "warn" | "bad"
    pub tone: &'static str,
    pub alarm: bool,
    /// "Aligned" | "Compensated" | "Paused"
    pub mode: String,
    pub pose: Option<PoseView>,
    pub inliers: Option<i64>,
    pub guidance: Vec<String>,
    pub warning_level: String,
    pub warning_text: String,
    pub ref_quad: Option<[[f64; 2]; 4]>,
    pub w: f64,
    pub h: f64,
    pub frame_url: String,
    pub reference_url: Option<String>,
    pub height_m: Option<f64>,
    pub pitch_deg: Option<f64>,
    pub hfov_deg: Option<f64>,
    pub fisheye: bool,
    pub method: String,
    pub rms_m: Option<f64>,
    pub rules: Vec<String>,
    pub auto_height: Option<String>,
    pub pending: Option<String>,
    pub error: Option<String>,
    pub sub: String,
}

fn tone_of(status: &str) -> &'static str {
    match status { "Ok" => "ok", "Realigning" => "warn", "Moved" | "Blocked" | "ViewLost" => "bad", _ => "warn" }
}

fn status_text(status: &str) -> String {
    match status {
        "Ok" => Status::Ok.label(), "Moved" => Status::Moved.label(), "Blocked" => Status::Blocked.label(),
        "ViewLost" => Status::ViewLost.label(), "Realigning" => Status::Realigning.label(), _ => "STARTING",
    }.to_string()
}

fn lens_text(fisheye: bool) -> &'static str { if fisheye { "Fisheye 190°" } else { "Rectilinear" } }

fn num(v: &serde_json::Value, k: &str) -> Option<f64> { v.get(k).and_then(|x| x.as_f64()) }

impl CamView {
    pub fn from_sample(c: &Cam) -> Self {
        let fisheye = matches!(c.lens, Lens::Fisheye);
        let alarm = c.guard != Status::Ok;
        CamView {
            id: c.id.into(), name: c.name.into(), is_sample: true, status_label: c.guard.label().into(), tone: guard_tone(c.guard), alarm,
            mode: format!("{:?}", c.mode),
            pose: Some(PoseView { dx: c.pose.dx, dy: c.pose.dy, angle: c.pose.angle, scale: c.pose.scale, corner: c.pose.corner }),
            inliers: Some(c.inliers as i64), guidance: c.guidance(),
            warning_level: if alarm { "alarm" } else { "ok" }.into(), warning_text: c.guard.label().into(),
            ref_quad: c.ref_quad, w: 1280.0, h: 720.0, frame_url: c.frame.into(), reference_url: c.reference.map(Into::into),
            height_m: Some(c.height_m), pitch_deg: Some(c.pitch_deg), hfov_deg: Some(c.hfov_deg), fisheye,
            method: c.calib_method.into(), rms_m: Some(c.calib_rms_m), rules: c.rules.iter().map(|r| r.to_string()).collect(),
            auto_height: None, pending: None, error: None,
            sub: format!("{:.0} fps · {:.0} ms/frame · h {:.0} m · pitch {:.0}°", c.fps, c.infer_ms, c.height_m, c.pitch_deg),
        }
    }

    /// From one entry of `Live::snapshot()["cameras"]`. Every field except `id` is optional.
    pub fn from_live(c: &serde_json::Value) -> Self {
        let id = c["id"].as_str().unwrap_or("").to_string();
        let l = &c["latest"];
        let status = l["status"].as_str().unwrap_or("");
        let error = c["error"].as_str().map(str::to_string);
        let fisheye = c["fisheye"].as_bool().unwrap_or(false);
        let pt = |v: &serde_json::Value| -> Option<PoseView> {
            v.is_object().then(|| PoseView { dx: num(v, "dx").unwrap_or(0.0), dy: num(v, "dy").unwrap_or(0.0), angle: num(v, "angle").unwrap_or(0.0),
                                             scale: num(v, "scale").unwrap_or(1.0), corner: num(v, "corner").unwrap_or(0.0) })
        };
        let pose = pt(&l["pose"]).or_else(|| pt(&l["smoothed_pose"]));
        let ref_quad = l["ref_box"].as_array().filter(|a| a.len() == 4).and_then(|a| {
            let mut q = [[0.0; 2]; 4];
            for (i, p) in a.iter().enumerate() { q[i] = [p.get(0)?.as_f64()?, p.get(1)?.as_f64()?]; }
            Some(q)
        });
        let level = l["warning"]["level"].as_str().unwrap_or("ok").to_string();
        let alarm = level == "alarm" || (!status.is_empty() && status != "Ok");
        let guidance: Vec<String> = l["guidance"].as_array().into_iter().flatten().filter_map(|g| g.as_str().map(str::to_string)).collect();
        let status_label = l["status_label"].as_str().map(str::to_string).unwrap_or_else(|| status_text(status));
        let tone = if status.is_empty() { if error.is_some() { "bad" } else { "warn" } } else { tone_of(status) };
        let tone = if level == "alarm" && tone == "ok" { "bad" } else { tone };
        let mode = l["mode"].as_str().unwrap_or("Aligned").to_string();
        let ah = &l["auto_height"];
        let auto_height = ah.is_object().then(|| {
            let st = ah["state"].as_str().unwrap_or("?");
            match (num(ah, "value"), ah["samples"].as_u64()) {
                (Some(v), Some(n)) => format!("{st} · {v:.1} m ({n} samples)"),
                (Some(v), None) => format!("{st} · {v:.1} m"),
                (None, Some(n)) => format!("{st} ({n} samples)"),
                _ => st.to_string(),
            }
        });
        let pending = l["pending"].as_object().map(|p| format!("{} for {:.1} s", p.get("status").and_then(|s| s.as_str()).unwrap_or("?"),
                                                            p.get("elapsed_s").and_then(|s| s.as_f64()).unwrap_or(0.0)));
        let height_m = num(l, "height_m");
        CamView {
            name: c["label"].as_str().unwrap_or("").to_string(), is_sample: false, status_label, tone, alarm, mode,
            pose, inliers: l["inliers"].as_i64(), guidance, warning_level: level, warning_text: l["warning"]["text"].as_str().unwrap_or("").to_string(),
            ref_quad, w: c["w"].as_f64().filter(|w| *w > 0.0).unwrap_or(1280.0), h: c["h"].as_f64().filter(|h| *h > 0.0).unwrap_or(720.0),
            frame_url: format!("/api/live/{id}/frame.jpg"), reference_url: None,
            height_m, fisheye, method: "live calibration".into(), auto_height, pending,
            sub: format!("{} · {:.0} fps · {} frames{}", c["source"].as_str().unwrap_or("live"), c["fps"].as_f64().unwrap_or(0.0), c["frames"].as_u64().unwrap_or(0),
                         height_m.map(|h| format!(" · h {h:.1} m")).unwrap_or_default()),
            error, id, ..Default::default()
        }
    }

    /// Fill calibration values and rule ids that the live snapshot does not carry.
    pub fn with_site(mut self, site: &Site) -> Self {
        if self.is_sample { return self; }
        if let Some(t) = site.twin.read().unwrap().as_ref() {
            if let Some(tc) = t.cameras.iter().find(|c| c.id.eq_ignore_ascii_case(&self.id)) {
                self.pitch_deg = Some(tc.camera.pitch_deg);
                if tc.lens.is_none() && tc.width > 0 && tc.camera.f_px > 0.0 {
                    self.hfov_deg = Some(2.0 * (tc.width as f64 / 2.0 / tc.camera.f_px).atan().to_degrees());
                }
                if self.height_m.is_none() { self.height_m = Some(tc.camera.height_m); }
                self.method = "Rig Twin".into();
            }
        }
        self.rules = site.iogp.rules.iter().map(|r| r.id.clone()).collect();
        self
    }
}

/// Live cameras from the snapshot.
pub fn live_views(site: &Site) -> Vec<CamView> {
    site.live.snapshot()["cameras"].as_array().into_iter().flatten().map(|c| CamView::from_live(c).with_site(site)).collect()
}

/// Cameras the alarm count / banner / header pill are built from: live cameras when any exist
/// (samples are excluded), otherwise the static samples. Returns (alarmed cameras, live cameras present).
pub fn alarm_cams(site: &Site) -> (Vec<CamView>, bool) {
    let lv = live_views(site);
    if lv.is_empty() {
        (site.cams.iter().map(CamView::from_sample).filter(|c| c.alarm).collect(), false)
    } else {
        (lv.into_iter().filter(|c| c.alarm).collect(), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn live_full() {
        let v = CamView::from_live(&json!({"id":"CAM-02","label":"Deck","source":"rtsp","w":640,"h":360,"fisheye":false,"fps":9.5,"frames":42,
            "latest":{"status":"Moved","mode":"Compensated","pose":{"dx":0.05,"dy":-0.02,"angle":1.5,"scale":1.01,"corner":0.06},"inliers":88,
            "ref_box":[[0,0],[10,0],[10,10],[0,10]],"guidance":["move left"],"pending":{"status":"Moved","elapsed_s":1.5},
            "height_m":11.0,"auto_height":{"state":"estimating","value":10.5,"samples":7},"warning":{"level":"alarm","text":"camera moved"}}}));
        assert!(v.alarm && v.tone == "bad" && !v.is_sample);
        assert_eq!(v.status_label, "CAMERA MOVED");
        assert_eq!(v.mode, "Compensated");
        assert_eq!(v.inliers, Some(88));
        assert_eq!(v.ref_quad.unwrap()[1], [10.0, 0.0]);
        assert_eq!((v.w, v.h), (640.0, 360.0));
        assert_eq!(v.frame_url, "/api/live/CAM-02/frame.jpg");
        assert_eq!(v.warning_text, "camera moved");
        assert!(v.auto_height.unwrap().contains("10.5"));
        assert!(v.pending.unwrap().contains("1.5"));
    }

    #[test]
    fn live_missing_fields() {
        let v = CamView::from_live(&json!({"id":"X"}));
        assert!(!v.alarm && v.tone == "warn");
        assert_eq!(v.status_label, "STARTING");
        assert_eq!((v.w, v.h), (1280.0, 720.0));
        assert!(v.pose.is_none() && v.ref_quad.is_none() && v.guidance.is_empty() && v.auto_height.is_none());
        assert_eq!(v.mode, "Aligned");
        let e = CamView::from_live(&json!({"id":"X","error":"no stream","latest":{"status":"Ok","warning":{"level":"alarm","text":"t"}}}));
        assert!(e.alarm && e.tone == "bad");
        let s = CamView::from_live(&json!({"id":"X","latest":{"smoothed_pose":{"dx":0.1},"ref_box":[[1,2]]}}));
        assert_eq!(s.pose.unwrap().dx, 0.1);
        assert!(s.ref_quad.is_none());
    }
}

// ───────────────────────────── overview ─────────────────────────────

#[page("/")]
pub async fn overview(cx: &Cx) -> Result<impl View> {
    let site = context::<Site>(cx);
    let (ok, vtxt) = verify_text(&site.ledger_check);
    let (alarms, live_present) = alarm_cams(site);
    let mut cards = live_views(site);
    cards.extend(site.cams.iter().map(CamView::from_sample));
    let n_live = cards.iter().filter(|c| !c.is_sample).count();
    let armed = site.iogp.armed();
    let feed: Vec<_> = site.ledger.entries.iter().rev().take(12).collect();
    let counts = alert_counts(site);
    let lsnap = site.live.snapshot();
    let live_cams = lsnap["cameras"].as_array().map_or(0, |c| c.len());
    let live_alerts = lsnap["alerts"].as_array().map_or(0, |a| a.len());
    Ok(view! {
        if live_cams > 0 {
            <a class="banner live" href=(href!(live))><strong>"● " (live_cams) " live stream(s)"</strong>
                " — " (live_alerts) " alert(s) since start (armed sound, shadow logged)." <span class="go">"Open live console →"</span></a>
        }
        <section class="kpis">
            <div class="kpi"><b>(cards.len())</b><span>"cameras"</span><em>if live_present { (n_live) " live · " (site.cams.len()) " sample" } else { "5 rectilinear · 1 fisheye (sample data)" }</em></div>
            <div class=(class!("kpi", "bad" if !alarms.is_empty()))>
                <b>(alarms.len())</b><span>if live_present { "camera-guard alarms (live)" } else { "camera-guard alarms (sample data)" }</span>
                <em>for c in &alarms { (c.id.clone()) " " (c.status_label.clone()) " · " }</em>
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
            <a class="banner bad" href=(format!("/cameras/{}", c.id))>
                <strong>(c.id.clone()) " · " (c.status_label.clone())</strong>
                if c.is_sample {
                    if c.mode == "Compensated" {
                        " — analytics continue through shift compensation; please re-aim the camera. "
                    } else {
                        " — analytics paused on this camera (ground calibration no longer valid). "
                    }
                } else {
                    " — " (c.warning_text.clone()) " "
                }
                (c.guidance.join(" · "))
                <span class="go">"Open guidance →"</span>
            </a>
        }
        <div class="cols">
            <section class="grid">
                for c in &cards {
                    <a class=(class!("card", "cam", "alarm" if c.alarm && (!c.is_sample || !live_present))) href=(format!("/cameras/{}", c.id))>
                        <div class="thumb">
                            <img src=(c.frame_url.clone()) alt=(c.name.clone()) loading="lazy">
                            <span class=(class!("pill", c.tone))>(c.status_label.clone())</span>
                            <span class="lens">if c.is_sample && live_present { "sample · " } (lens_text(c.fisheye))</span>
                        </div>
                        <div class="meta">
                            <div><strong>(c.id.clone())</strong> " " (c.name.clone()) if c.is_sample && live_present { " " <span class="chip">"sample"</span> }</div>
                            <div class="sub">(c.sub.clone())</div>
                            <div class="chips">for r in &c.rules { <span class="chip">(r.clone())</span> }</div>
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
    let want = id(cx);
    let v = live_views(site).into_iter().find(|c| c.id.eq_ignore_ascii_case(&want))
        .or_else(|| site.cam(&want).map(CamView::from_sample))
        .ok_or_else(not_found)?;
    let overlay = v.ref_quad.map(|q| {
        let pts: Vec<String> = q.iter().map(|p| format!("{:.1},{:.1}", p[0], p[1])).collect();
        let (cx0, cy0) = (q.iter().map(|p| p[0]).sum::<f64>() / 4.0, q.iter().map(|p| p[1]).sum::<f64>() / 4.0);
        Trusted(format!(
            r##"<svg class="ov" viewBox="0 0 {:.0} {:.0}" preserveAspectRatio="none">
<polygon points="{}" class="refbox"/>
<line x1="{:.0}" y1="{:.0}" x2="{cx0:.0}" y2="{cy0:.0}" class="arrow" marker-end="url(#ah)"/>
<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#38bdf8"/></marker></defs>
<text x="{:.0}" y="{:.0}" class="reflabel">correct view is here</text></svg>"##,
            v.w, v.h, pts.join(" "), v.w / 2.0, v.h / 2.0, q[0][0] + 14.0, q[0][1].max(0.0) + 34.0))
    });
    let refresh = (!v.is_sample).then(|| Trusted("<script>setTimeout(function(){location.reload()},2000)</script>".into()));
    let cam_rules: Vec<_> = v.rules.iter().filter_map(|r| site.iogp.get(r).map(|rr| (r.clone(), rr))).collect();
    let paused = v.mode == "Paused";
    let compensated = v.mode == "Compensated";
    // "position correct" only when there is a guard measurement and no warning at all (a starting or
    // errored live camera has neither, and must not claim its position is correct)
    let ok = !v.alarm && v.warning_level == "ok" && (v.is_sample || v.pose.is_some());
    let rereference = format!("VG360_CAM_{}_REREFERENCE", v.id.replace('-', "_"));
    let rereference_file = format!("state/reference/{}.png", v.id);
    Ok(view! {
        <div class="crumbs"><a href=(href!(overview))>"Overview"</a> " / " (v.id.clone())</div>
        <h1>(v.id.clone()) " · " (v.name.clone()) <span class=(class!("pill", v.tone))>(v.status_label.clone())</span>
            if v.is_sample { " " <span class="chip">"sample"</span> } else { " " <span class="chip">"live"</span> }</h1>
        if let Some(e) = &v.error { <div class="banner bad"><strong>"stream error: "</strong>(e.clone())</div> }
        <div class="cols wide">
            <section class="card frame">
                <div class="stage">
                    <img src=(v.frame_url.clone()) alt=(v.name.clone())>
                    if let Some(o) = overlay { (o) }
                    if let Some(r) = &v.reference_url {
                        <figure class="inset"><img src=(r.clone()) alt="reference view"><figcaption>"reference view"</figcaption></figure>
                    }
                </div>
                <div class="caption">
                    if ok {
                        "Live analytics overlay: ground-plane boxes, zones, track IDs (rendered by the inference worker)."
                    } else {
                        "Camera guard overlay: blue box = where the reference view now sits in the live image. Follow the arrow until the box fills the frame."
                    }
                </div>
            </section>
            <aside class="stack">
                <section class=(class!("card", "guard", v.tone))>
                    <h3>"Camera guard" <small>"FAST/BRIEF · Hamming · MSAC — pure Rust"</small></h3>
                    if let Some(p) = &v.pose {
                        <div class="metrics">
                            <div><span>"shift x"</span><b>(pct(p.dx))</b></div>
                            <div><span>"shift y"</span><b>(pct(p.dy))</b></div>
                            <div><span>"rotation"</span><b>(format!("{:.1}°", p.angle))</b></div>
                            <div><span>"zoom"</span><b>(format!("{:.1}%", (p.scale - 1.0) * 100.0 + 0.0).replace("-0.0", "0.0"))</b></div>
                            <div><span>"corner shift"</span><b>(pct(p.corner))</b></div>
                            <div><span>"inliers"</span><b>(v.inliers.map_or("—".to_string(), |n| n.to_string()))</b></div>
                        </div>
                    } else {
                        <p class="note">"No guard measurement yet (the worker is starting, or no pose was reported for this frame)."</p>
                    }
                    if let Some(p) = &v.pending { <p class="note">"Pending change: " (p.clone())</p> }
                    if !v.warning_text.is_empty() && !v.is_sample { <p class=(class!("note", "okline" if v.warning_level == "ok"))>(v.warning_text.clone())</p> }
                    if ok && v.guidance.is_empty() {
                        <p class="okmsg">"Position correct — within " (pct(gc::OK_SHIFT)) " shift, " (gc::OK_ROT) "° rotation."</p>
                    } else {
                        <ol class="guide">for g in &v.guidance { <li>(g.clone())</li> }</ol>
                        if compensated {
                            <p class="note okline">"Analytics continue: the move is small (≤ 8 % shift, ≤ 5° rotation, ≥ 60 matched points), so detections are mapped back through the measured homography H⁻¹ onto the original deck calibration. Results from this camera carry "
                                <code>"mode: Compensated"</code> ". The alarm stays until the view is re-aimed and aligned for " (gc::OK_HOLD) " s."</p>
                        } else if paused && v.error.is_none() {
                            <p class="note">"Analytics paused: the move is too large to compensate (or the view is blocked / lost). Rules on this camera are suspended until the view is aligned for " (gc::OK_HOLD) " s."</p>
                        }
                        <p class="note"><b>"Re-baseline: "</b>"If the camera was deliberately re-mounted, delete " <code>(rereference_file.clone())</code> " (or set " <code>(rereference.clone()) "=1"</code>
                            ") and restart; recalibrate with " <code>"POST /api/calibrate"</code> " or the Rig Twin."</p>
                    }
                </section>
                <section class="card">
                    <h3>"Ground calibration"</h3>
                    <dl>
                        <dt>"analytics"</dt><dd>(if paused { "paused" } else if compensated { "running · shift-compensated" } else { "running" })</dd>
                        <dt>"method"</dt><dd>(v.method.clone())</dd>
                        <dt>"mount height"</dt><dd>(v.height_m.map_or("not set".to_string(), |h| format!("{h:.1} m")))</dd>
                        if let Some(a) = &v.auto_height { <dt>"auto-height"</dt><dd>(a.clone())</dd> }
                        <dt>"pitch"</dt><dd>(v.pitch_deg.map_or("—".to_string(), |x| format!("{x:.0}°")))</dd>
                        <dt>"field of view"</dt><dd>(v.hfov_deg.map_or("—".to_string(), |x| format!("{x:.0}°")))</dd>
                        if let Some(r) = v.rms_m { <dt>"deck RMS"</dt><dd>(format!("{r:.3} m"))</dd> }
                        <dt>"lens"</dt><dd>(lens_text(v.fisheye))</dd>
                    </dl>
                    if v.fisheye {
                        <p class="note">"Fisheye: detection runs on 3 upright virtual pinhole views; ground points use the Kannala–Brandt ray directly (error 0.005 m at 1 m → 0.039 m at 30 m in the notebook self-test)."</p>
                    }
                </section>
                <section class="card">
                    <h3>"Rules on this camera"</h3>
                    <ul class="rl">
                        for (rid, r) in cam_rules {
                            let pass = r.armed && !paused;
                            <li>
                                <span class="chip">(rid)</span>
                                <span>(r.alert.clone())</span>
                                <span class=(class!("pill", "ok" if pass, "warn" if !pass))>
                                    if paused { "paused" } else if !r.armed { "shadow" } else if compensated { "armed · compensated" } else { "armed" }
                                </span>
                            </li>
                        }
                    </ul>
                </section>
            </aside>
        </div>
        if let Some(r) = refresh { (r) }
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
                                <td>if let Some(r) = rms { <span class=(class!("pill", "ok" if r < 1.0, "warn" if r >= 1.0))>(format!("{r:.2} px · {} pts", c.points.len()))</span> } else { <span class="sub">if c.lens.is_some() { "not computed for fisheye" } else { "no deck points in view" }</span> }</td>
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
    let card = site.ppe_card.as_ref().ok();
    let cj = move |p: &str| card.and_then(|c| c.json.pointer(p).cloned()).unwrap_or(serde_json::Value::Null);
    let ppe_quality = match card {
        Some(c) => format!("MCC {:.3} [{:.3}, {:.3}] on {} (n={})", c.test_mcc().unwrap_or(f64::NAN), c.mcc_ci95().map_or(f64::NAN, |x| x.0),
                           c.mcc_ci95().map_or(f64::NAN, |x| x.1), if c.real_data() { "real crops" } else { "synthetic crops" }, cj("/data/n_test")),
        None => "no verified model card".into(),
    };
    let baseline = card.map(|_| format!("plain CNN ({} conv weights, SAM): MCC {:.3} [{:.3}, {:.3}]", cj("/baseline_cnn/conv_params"),
        cj("/baseline_cnn/test_mcc").as_f64().unwrap_or(f64::NAN), cj("/baseline_cnn/test_mcc_ci95/0").as_f64().unwrap_or(f64::NAN),
        cj("/baseline_cnn/test_mcc_ci95/1").as_f64().unwrap_or(f64::NAN)));
    let steps = [
        ("1", "Data", "Site footage → audit + crop harvest (§9) → labels; synthetic crops only as a smoke test."),
        ("2", "PPE classifier", "Cl(2,0) Clifford conv, trained with SAM (ρ = 0.05), per-crop standardisation; folded to plain conv for export."),
        ("3", "Detector", "§10: YOLOX-Nano (official layout, COCO weights) fine-tuned with SAM on YOLO-format site labels; cell MCC; drop-in ONNX + card (VG360_DETECTOR)."),
        ("4", "Validate", "Test MCC with bootstrap 95 % CI, plain-CNN baseline, sharpness at the SAM point; model selection on val MCC."),
        ("5", "Export", "Fold → ONNX (opset 17), parity checks, model card with SHA-256, golden crops for the Rust tests; Lean facts checked in CI."),
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
                        <tr><td><b>"YOLOX-Nano"</b><div class="sub">"person detector, 416² · COCO"</div></td><td>"ONNX fp32"</td>
                            <td class="num">(format!("{fp32:.1} ms"))</td><td>"stock weights · on real drill-floor footage (RIG 2004) finds 1 of 2–3 workers at 0.35"</td>
                            <td>if matches!(site.detector.1, Ok(None)) { <span class="pill warn">"serving · needs site fine-tune"</span> } else { <span class="pill">"standby"</span> }</td></tr>
                        match &site.detector.1 {
                            Ok(Some(c)) => {
                                <tr><td><b>"YOLOX-Nano fine-tuned"</b><div class="sub">(format!("SAM ρ {} · {}", c.json["training"]["rho"], c.sha256()[..12].to_string()))</div></td><td>"ONNX fp32"</td><td class="num">"—"</td>
                                    <td>(format!("cell MCC {:.3} [{:.3}, {:.3}] (stock {:.3}) · AP50 {:.3} on {}", c.json["metrics"]["test"]["cell_mcc"].as_f64().unwrap_or(f64::NAN),
                                        c.json["metrics"]["test"]["cell_mcc_ci95"][0].as_f64().unwrap_or(f64::NAN), c.json["metrics"]["test"]["cell_mcc_ci95"][1].as_f64().unwrap_or(f64::NAN),
                                        c.json["metrics"]["test_before_finetune"]["cell_mcc"].as_f64().unwrap_or(f64::NAN), c.json["metrics"]["test"]["ap50"].as_f64().unwrap_or(f64::NAN), c.data_source()))</td>
                                    <td><span class=(class!("pill", "ok" if c.real_data(), "warn" if !c.real_data()))>if c.real_data() { "serving" } else { "serving · smoke model (test only)" }</span></td></tr>
                            }
                            Err(e) => { <tr><td><b>"YOLOX-Nano fine-tuned"</b></td><td colspan="3">(e.clone())</td><td><span class="pill bad">"refused"</span></td></tr> }
                            Ok(None) => {}
                        }
                        <tr><td><b>"YOLOX-Nano"</b><div class="sub">"static QDQ int8"</div></td><td>"ONNX int8"</td>
                            <td class="num">(format!("{int8:.1} ms"))</td><td>"same detections on test frame (3 / 3)"</td><td><span class="pill warn">"rejected — slower"</span></td></tr>
                        <tr><td><b>"PPE Clifford-conv"</b><div class="sub">(format!("{} Clifford weights → {} folded · SAM ρ {}", cj("/conv_params/clifford_trainable"), cj("/conv_params/folded_real"), cj("/training/rho")))</div></td>
                            <td>"ONNX fp32"</td><td class="num">(format!("{:.1} ms / 8 crops", cj("/parity/onnx_ms_per_8_crops").as_f64().unwrap_or(f64::NAN)))</td>
                            <td>(ppe_quality)</td>
                            <td>if card.is_some_and(|c| c.real_data() && c.meets_floor()) { <span class="pill ok">"serving"</span> }
                                else if card.is_some() { <span class="pill warn">"serving · shadow (below floor / synthetic)"</span> }
                                else { <span class="pill bad">"disabled"</span> }</td></tr>
                    </tbody>
                </table>
                <h3>"PPE model card" <small>"models/ppe_clifford_folded.card.json · SHA-256 verified at start-up"</small></h3>
                match &site.ppe_card {
                    Ok(c) => {
                        <dl>
                            <dt>"sha256"</dt><dd><code>(c.sha256()[..16].to_string()) "…"</code></dd>
                            <dt>"data"</dt><dd>(c.data_source().to_string())</dd>
                            <dt>"selection"</dt><dd>(format!("best val MCC (epoch {})", cj("/metrics/best_epoch")))</dd>
                            <dt>"sharpness"</dt><dd>(format!("test loss +{:.4} at the SAM point (ρ {})", cj("/metrics/sharpness/increase").as_f64().unwrap_or(f64::NAN), cj("/training/rho")))</dd>
                            <dt>"parity"</dt><dd>(format!("torch ≡ folded ≡ onnxruntime (max |Δ| {:.1e})", cj("/parity/max_abs_torch_vs_onnxruntime").as_f64().unwrap_or(f64::NAN)))</dd>
                            <dt>"baseline"</dt><dd>(baseline.clone().unwrap_or_default())</dd>
                            <dt>"gate"</dt><dd>(format!("{} → floor {} · {}", cj("/gate/severity").as_str().unwrap_or(""), cj("/gate/mcc_floor"), if c.meets_floor() { "meets" } else { "below: rule stays shadow" }))</dd>
                        </dl>
                    }
                    Err(e) => { <p class="note">"PPE classifier disabled: " (e.clone())</p> }
                }
            </section>
            <aside class="card">
                <h3>"Notebook pipeline" <small>"notebooks/vg360_train_cpu.ipynb"</small></h3>
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
                    <tr><td><code>"golden::ppe_model_matches_its_card"</code></td><td>"ONNX SHA-256 = model card = golden file; card MCC inside its bootstrap CI"</td></tr>
                    <tr><td><code>"golden::ppe_tract_matches_notebook_on_golden_crops"</code></td><td>"16 notebook test crops: P(helmet) within 1e-4 of onnxruntime (measured 2e-6)"</td></tr>
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
    let twin_cams: Vec<(String, String, String, bool, bool, bool)> = site.twin.read().unwrap().as_ref().map(|t| t.cameras.iter().map(|c| {
        let env = format!("VG360_CAM_{}_URL", c.id.replace('-', "_"));
        let ah = std::env::var(format!("VG360_CAM_{}_AUTO_HEIGHT", c.id.replace('-', "_"))).is_ok_and(|v| v == "1");
        let rr = std::env::var(format!("VG360_CAM_{}_REREFERENCE", c.id.replace('-', "_"))).is_ok_and(|v| v == "1");
        (c.id.clone(), format!("{} · {}×{}", if c.lens.is_some() { "fisheye" } else { "pinhole" }, c.width, c.height), env.clone(), std::env::var(&env).is_ok(), ah, rr)
    }).collect()).unwrap_or_default();
    Ok(view! {
        <h1>"Settings"</h1>
        <div class="cols">
            <section class="card">
                <h3>"Camera streams" <small>"cameras of the applied Rig Twin"</small></h3>
                <p class="note">"Stream URLs (with credentials) are never stored in the repository or shown here. A live worker starts for every twin camera whose "
                    <code>"VG360_CAM_<ID>_URL"</code> " is set on this host; " <code>"VG360_CAM_<ID>_AUTO_HEIGHT=1"</code> " estimates its mount height from standing workers; " <code>"VG360_CAM_<ID>_REREFERENCE=1"</code> " re-captures the reference view at the next start (use after a deliberate re-mount, then unset it)."</p>
                <table class="mini">
                    <thead><tr><th>"Camera"</th><th>"Lens · resolution"</th><th>"Environment variable"</th><th>"Stream"</th><th>"Auto-height"</th><th>"Re-reference"</th></tr></thead>
                    <tbody>
                        for c in &twin_cams {
                            <tr><td>(c.0.clone())</td><td>(c.1.clone())</td><td><code>(c.2.clone())</code></td>
                                <td><span class=(class!("pill", "ok" if c.3, "warn" if !c.3))>if c.3 { "set · live" } else { "not set" }</span></td>
                                <td>if c.4 { <span class="pill ok">"on"</span> } else { "—" }</td>
                                <td>if c.5 { <span class="pill warn">"set — unset after restart"</span> } else { "—" }</td></tr>
                        }
                    </tbody>
                </table>
                if twin_cams.is_empty() { <p class="note">"No Rig Twin applied — open Site twin and press Apply to site."</p> }
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
                    <dt><code>"GET /api/live/events"</code></dt><dd>"live frames and alerts (SSE)"</dd>
                    <dt><code>"POST /api/signals"</code></dt><dd>"crane load, pipe handling"</dd>
                    <dt><code>"POST /api/calibrate"</code></dt><dd>"vanishing points + people → FOV, pitch, roll, height"</dd>
                </dl>
            </aside>
        </div>
    })
}

/// Live console: every running camera worker, its latest frame and detections, and the live alert feed
/// (server-sent events from `/api/live/events`; see `public/live.js`).
#[page("/live")]
pub async fn live(cx: &Cx) -> Result<impl View> {
    let site = context::<Site>(cx);
    let snap = site.live.snapshot();
    let cams: Vec<(String, String, u64, u64, Option<String>)> = snap["cameras"].as_array().into_iter().flatten()
        .map(|c| (c["id"].as_str().unwrap_or("").to_string(), c["label"].as_str().unwrap_or("").to_string(),
                  c["w"].as_u64().unwrap_or(1280), c["h"].as_u64().unwrap_or(720), c["error"].as_str().map(|e| format!("error: {e}")))).collect();
    let sig = site.live.signals.read().unwrap().clone();
    Ok(view! {
        <h1>"Live analytics" <span class="pill" id="live-state">"connecting…"</span></h1>
        if cams.is_empty() {
            <section class="card"><p class="note">"No camera streams are running. Set "
                <code>"VG360_CAM_<ID>_URL"</code> " for cameras of the applied Rig Twin (RTSP, credentials stay in the environment), or "
                <code>"VG360_DEMO=1"</code> " to replay the simulated rig clip as a camera."</p></section>
        }
        <div class="cols wide">
            <section class="livegrid">
                for (id, label, w, h, err) in &cams {
                    <figure class="card frame live" data-cam=(id.clone())>
                        <div class="stage">
                            <img class="lv" alt=(label.clone()) src=(format!("/api/live/{id}/frame.jpg")) style=(format!("aspect-ratio: {w} / {h}"))>
                            <svg class="ov" viewBox=(format!("0 0 {w} {h}")) preserveAspectRatio="none"></svg>
                            <span class="pill livebadge">"LIVE"</span>
                        </div>
                        <figcaption class="caption"><strong>(id.clone())</strong> " " (label.clone()) " · " <span class=(class!("stat", "bad" if err.is_some()))>(err.clone().unwrap_or_else(|| "starting… (connecting to the stream)".into()))</span></figcaption>
                    </figure>
                }
            </section>
            <aside class="stack">
                <section class="card">
                    <h3>"Site signals" <small>"from the rig control system — gate the rules"</small></h3>
                    <form id="signals" class="sig">
                        <label><input type="checkbox" name="suspended_load" checked=(sig.suspended_load)>" suspended load (crane)"</label>
                        <label><input type="checkbox" name="pipe-handling" checked=(sig.active.iter().any(|a| a == "pipe-handling"))>" pipe handling"</label>
                    </form>
                </section>
                <section class="card feed">
                    <h3>"Live alerts" <small>"armed sound · shadow are logged only"</small></h3>
                    <ol id="live-alerts"></ol>
                </section>
            </aside>
        </div>
        <script src="/live.js"></script>
    })
}
