// VisionGuard360 live console: subscribes to /api/live/events (server-sent events) and updates
// each camera tile (latest JPEG + detection overlay) and the live alert feed. No framework.
(() => {
  const st = document.getElementById("live-state");
  const feed = document.getElementById("live-alerts");
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const tiles = {};
  document.querySelectorAll("[data-cam]").forEach((el) => (tiles[el.dataset.cam] = el));

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
    el.querySelector(".stat").textContent =
      `${m.fps.toFixed(1)} fps · ${l.ms.toFixed(0)} ms · ${l.people.length} in view · ${l.mode}${l.status !== "Ok" ? " · guard " + l.status : ""}` +
      (l.calibrating ? " · estimating mount height (rules off)" : ` · h ${l.height_m.toFixed(1)} m`);
  }

  function alert(a) {
    if (!feed) return;
    const li = document.createElement("li");
    li.className = a.armed ? "armed" : "shadow";
    li.innerHTML = `<span class="t">${a.t.toFixed(1)}s</span><span class="chip r">${esc(a.rule)}</span>` +
      `<span class="who">${esc(a.camera)} ${esc(a.track)} — ${esc(a.kind)} ${esc(a.alert)}</span>` +
      `<span class="pill ${a.armed ? "bad" : "warn"}">${a.armed ? "armed" : "shadow"}</span>` +
      (a.incident && a.snapshot_sha256 ? ` <a href="/api/evidence/${encodeURIComponent(a.incident)}/snapshot.jpg" target="_blank">snapshot</a> <code>${esc(a.snapshot_sha256.slice(0, 10))}</code>` : "");
    feed.prepend(li);
    while (feed.children.length > 40) feed.lastChild.remove();
  }

  const es = new EventSource("/api/live/events");
  es.addEventListener("hello", (e) => {
    const s = JSON.parse(e.data);
    if (st) { st.textContent = s.cameras.length ? "streaming" : "idle"; st.className = "pill " + (s.cameras.length ? "ok" : "warn"); }
    if (feed) { feed.innerHTML = ""; s.alerts.slice().reverse().forEach(alert); }
    s.cameras.forEach((c) => c.latest && frame(c));
  });
  es.addEventListener("frame", (e) => frame(JSON.parse(e.data)));
  es.addEventListener("alert", (e) => alert(JSON.parse(e.data)));
  es.onerror = () => { if (st) { st.textContent = "reconnecting…"; st.className = "pill warn"; } };

  const form = document.getElementById("signals");
  if (form) form.addEventListener("change", () => {
    const active = form.elements["pipe-handling"].checked ? ["pipe-handling"] : [];
    fetch("/api/signals", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ suspended_load: form.elements["suspended_load"].checked, active }) });
  });
})();
