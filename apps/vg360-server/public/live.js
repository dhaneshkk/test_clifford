// VisionGuard360 live console: subscribes to /api/live/events (server-sent events) and updates
// each camera tile (latest JPEG + detection overlay) and the live alert feed. No framework.
(() => {
  const st = document.getElementById("live-state");
  const feed = document.getElementById("live-alerts");
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const tiles = {};
  const tstate = {}; // per camera: {signal (no-signal text | null), warning, guidance}
  document.querySelectorAll("[data-cam]").forEach((el) => {
    tiles[el.dataset.cam] = el;
    const cap = el.querySelector(".caption");
    if (cap) { const b = document.createElement("div"); b.className = "guardwarn"; b.hidden = true; cap.before(b); }
  });
  const stateOf = (cam) => (tstate[cam] = tstate[cam] || { signal: null, warning: null, guidance: [] });

  // Warning banner above the caption: "No signal" (alarm) wins over the guard warning level.
  function renderBanner(cam) {
    const el = tiles[cam]; if (!el) return;
    const b = el.querySelector(".guardwarn"); if (!b) return;
    const s = stateOf(cam);
    let level = null, text = "", guidance = [];
    if (s.signal) { level = "alarm"; text = `No signal: ${s.signal}`; }
    else if (s.warning && (s.warning.level === "warn" || s.warning.level === "alarm")) {
      level = s.warning.level; text = s.warning.text || ""; guidance = Array.isArray(s.guidance) ? s.guidance : [];
    }
    b.className = "guardwarn" + (level ? " " + level : "");
    b.hidden = !level;
    b.replaceChildren();
    if (!level) return;
    const p = document.createElement("div"); p.textContent = text; b.append(p);
    if (guidance.length) {
      const ul = document.createElement("ul");
      for (const g of guidance) { const li = document.createElement("li"); li.textContent = String(g); ul.append(li); }
      b.append(ul);
    }
  }

  function frame(m) {
    const l = m.latest; if (!l) return;
    const el = tiles[l.camera]; if (!el) return;
    const img = el.querySelector("img.lv");
    img.src = `/api/live/${encodeURIComponent(l.camera)}/frame.jpg?t=${l.t}`;
    const svg = el.querySelector("svg.ov");
    let g = "";
    if (l.ref_box && l.mode !== "Aligned") g += `<polygon class="refbox" points="${l.ref_box.map((p) => p.join(",")).join(" ")}"/>`;
    for (const p of l.people) {
      const [x1, y1, x2, y2] = p.box;
      const sw = Math.max(3, svg.viewBox.baseVal.width / 400);
      const cls = p.down ? "pb down" : (p.inside && p.inside.length ? "pb in" : (p.helmet != null && p.helmet < 0.5 ? "pb noppe" : "pb"));
      const tag = [p.track != null ? `#${p.track}` : "", p.down ? "DOWN" : "", (p.inside || []).join(" "),
                   p.helmet != null ? `helmet ${(p.helmet * 100).toFixed(0)}%` : ""].filter(Boolean).join(" · ");
      g += `<rect class="${cls}" style="stroke-width:${sw}" x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}"/><text class="pt" style="font-size:${sw * 9}px" x="${x1}" y="${Math.max(14, y1 - 5)}">${esc(tag)}</text>`;
    }
    svg.innerHTML = g;
    const s = stateOf(l.camera);
    s.signal = m.error ? String(m.error) : null; // a frame without an error clears "No signal"
    s.warning = l.warning || null;
    s.guidance = l.guidance || [];
    renderBanner(l.camera);
    const stat = el.querySelector(".stat");
    const guarded = !!l.status && l.status !== "Ok";
    stat.classList.toggle("bad", !!(m.error || guarded));
    const parts = [`${m.fps.toFixed(1)} fps`, `${l.ms.toFixed(0)} ms`, `${l.people.length} in view`, l.mode];
    if (guarded) parts.push(`guard ${l.status_label || l.status}`);
    if (l.pending) parts.push("confirming…");
    const ah = l.auto_height;
    if (ah && ah.state === "estimating") parts.push(`auto-height ${ah.samples || 0} samples`);
    else if (ah && ah.state === "failed") parts.push("auto-height failed");
    parts.push(l.calibrating ? "estimating mount height (rules off)" : `h ${l.height_m.toFixed(1)} m`);
    if (typeof m.stale_s === "number" && m.stale_s > 5) parts.push(`stale ${Math.round(m.stale_s)} s`);
    stat.textContent = m.error ? `error: ${m.error}` : parts.join(" · ");
  }

  function alert(a) {
    if (!feed) return;
    const li = document.createElement("li");
    li.className = a.armed ? "armed" : "shadow";
    li.innerHTML = `<span class="t">${a.t.toFixed(1)}s</span><span class="chip r">${esc(a.rule)}</span>` +
      `<span class="who">${esc(a.camera)} ${esc(a.track)} — ${esc(a.kind)} ${esc(a.alert)}${a.mode ? " · " + esc(a.mode) : ""}</span>` +
      `<span class="pill ${a.armed ? "bad" : "warn"}">${a.armed ? "armed" : "shadow"}</span>` +
      (a.incident && a.snapshot_sha256 ? ` <a href="/api/evidence/${encodeURIComponent(a.incident)}/snapshot.jpg" target="_blank">snapshot</a> <code>${esc(a.snapshot_sha256.slice(0, 10))}</code>` : "");
    feed.prepend(li);
    while (feed.children.length > 40) feed.lastChild.remove();
  }

  function error(m) {
    const el = tiles[m.camera]; if (!el) return;
    const s = el.querySelector(".stat");
    s.textContent = `error: ${m.error}`; s.classList.add("bad");
    stateOf(m.camera).signal = String(m.error ?? "unknown");
    renderBanner(m.camera);
  }

  // guard {camera, status, status_label, mode, warning, guidance}: non-fatal camera guard state.
  function onGuard(m) {
    const el = tiles[m.camera]; if (!el) return;
    const s = stateOf(m.camera);
    s.warning = m.warning || null;
    s.guidance = m.guidance || [];
    renderBanner(m.camera);
    el.querySelector(".stat").classList.toggle("bad", (!!m.status && m.status !== "Ok") || !!s.signal);
  }

  const es = new EventSource("/api/live/events");
  es.addEventListener("hello", (e) => {
    const s = JSON.parse(e.data);
    if (st) { st.textContent = s.cameras.length ? "streaming" : "idle"; st.className = "pill " + (s.cameras.length ? "ok" : "warn"); }
    if (feed) { feed.innerHTML = ""; s.alerts.slice().reverse().forEach(alert); }
    s.cameras.forEach((c) => {
      if (c.latest) frame(c);
      if (c.error) error({ camera: c.id, error: c.error });
      else { stateOf(c.id).signal = null; renderBanner(c.id); }
    });
  });
  es.addEventListener("frame", (e) => frame(JSON.parse(e.data)));
  es.addEventListener("alert", (e) => alert(JSON.parse(e.data)));
  es.addEventListener("camera-error", (e) => error(JSON.parse(e.data)));
  es.addEventListener("guard", (e) => onGuard(JSON.parse(e.data)));
  es.onerror = () => { if (st) { st.textContent = "reconnecting…"; st.className = "pill warn"; } };

  const form = document.getElementById("signals");
  if (form) form.addEventListener("change", () => {
    const active = form.elements["pipe-handling"].checked ? ["pipe-handling"] : [];
    fetch("/api/signals", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ suspended_load: form.elements["suspended_load"].checked, active }) });
  });
})();
