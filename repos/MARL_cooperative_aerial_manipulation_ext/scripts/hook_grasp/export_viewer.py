"""Export a hook-grasp rollout (``<tag>_rollout.npz`` from ``play.py``) as a self-contained interactive 3D viewer.

    python scripts/hook_grasp/export_viewer.py videos/final_rollout.npz [--out videos/final_viewer.html]

Open the resulting ``<tag>_viewer.html`` in any web browser (three.js is loaded from a CDN, so it needs internet).
Mouse: left-drag rotate, wheel zoom, right-drag pan. Keys: space play/pause, left/right arrows step (shift = x10).
Needs only numpy (works inside the Isaac Lab container; ``play.py`` calls it after every rollout).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_rollout import hook_bar_distance, wall_polygons  # noqa: E402  (numpy-only helpers)


def _r(x, nd=4):
    return np.round(np.asarray(x, dtype=np.float64), nd).tolist()


def export(npz_path, out_html=None):
    """Write the viewer next to the npz (``<tag>_viewer.html``) or to ``out_html``. Returns the path."""
    d = np.load(npz_path, allow_pickle=False)
    T = d["drone_pos"].shape[0]
    hx, hy, H = float(d["basket_half_x"]), float(d["basket_half_y"]), float(d["basket_wall_height"])
    half_len = float(d["bar_half_len"])
    hook, bar_c, bar_a = d["hook_pos"], d["bar_center"], d["bar_axis"]
    walls = np.stack([wall_polygons(d["basket_pos"][k], d["basket_quat"][k], hx, hy, H) for k in range(T)])
    data = {
        "T": T,
        "dt": float(d["dt"]),
        "checkpoint": str(d["checkpoint"]) if "checkpoint" in d else "",
        "seed": int(d["seed"]) if "seed" in d else None,
        "drone": _r(d["drone_pos"]),
        "rotors": _r(d["rotor_pos"]),
        "wire": _r(d["wire_pos"]),
        "hook": _r(hook),
        "walls": _r(walls, 3),
        "bar0": _r(bar_c - half_len * bar_a),
        "bar1": _r(bar_c + half_len * bar_a),
        "attached": d["attached"].astype(int).tolist(),
        "caught": d["newly_caught"].astype(int).tolist(),
        "done": d["done"].astype(int).tolist(),
        "dist": _r(hook_bar_distance(hook, bar_c, bar_a, half_len)),
        "act": _r(np.linalg.norm(d["action"], axis=-1), 3),
        "reward": _r(d["reward"], 3),
    }
    if out_html is None:
        base = npz_path[:-4]
        base = base[: -len("_rollout")] if base.endswith("_rollout") else base
        out_html = base + "_viewer.html"
    title = os.path.basename(data["checkpoint"]) or os.path.basename(npz_path)
    html = TEMPLATE.replace("__TITLE__", title).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(out_html, "w") as f:
        f.write(html)
    print(f"[viewer] wrote {out_html} ({T} steps, {os.path.getsize(out_html) / 1e6:.1f} MB)")
    return out_html


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hook-grasp viewer</title>
<style>
  :root { --bg:#f6f6f4; --panel:#ffffff; --fg:#1d1d1f; --muted:#6b6b70; --line:#d9d9de; --accent:#1f77b4;
          --ok:#2ca02c; --bar:#d62728; --scene:#eef0f3; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
          --bg:#16171a; --panel:#1f2024; --fg:#ececf0; --muted:#9a9aa3; --line:#34353b; --accent:#5aa9e6;
          --ok:#4cc35a; --bar:#ef5350; --scene:#202227; } }
  :root[data-theme="dark"] { --bg:#16171a; --panel:#1f2024; --fg:#ececf0; --muted:#9a9aa3; --line:#34353b;
          --accent:#5aa9e6; --ok:#4cc35a; --bar:#ef5350; --scene:#202227; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--fg);
               font:14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
  #app { display:grid; grid-template-rows: auto 1fr auto auto; height:100vh; max-width:100vw; }
  header { display:flex; flex-wrap:wrap; gap:4px 16px; align-items:baseline; padding:8px 16px;
           border-bottom:1px solid var(--line); }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  header .sub { color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
  #view { position:relative; min-height:240px; }
  #view canvas { display:block; width:100%; height:100%; }
  #hud { position:absolute; top:10px; left:10px; background:color-mix(in srgb, var(--panel) 88%, transparent);
         border:1px solid var(--line); border-radius:8px; padding:8px 10px; font-size:12px;
         font-variant-numeric: tabular-nums; pointer-events:none; min-width:170px; }
  #hud .state { font-weight:700; font-size:13px; }
  #hud .state.on { color:var(--ok); }
  #help { position:absolute; bottom:8px; right:10px; color:var(--muted); font-size:11px; text-align:right; }
  #controls { display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; padding:8px 16px;
              border-top:1px solid var(--line); background:var(--panel); }
  #controls button { font:inherit; min-width:36px; height:32px; border-radius:6px; border:1px solid var(--line);
                     background:var(--bg); color:var(--fg); cursor:pointer; }
  #controls button:hover { border-color:var(--accent); }
  #controls label { display:flex; gap:6px; align-items:center; color:var(--muted); font-size:12px; }
  #controls select { font:inherit; height:30px; border-radius:6px; border:1px solid var(--line);
                     background:var(--bg); color:var(--fg); }
  #slider { flex:1 1 220px; min-width:160px; accent-color:var(--accent); }
  #time { font-variant-numeric: tabular-nums; min-width:120px; }
  #plots { width:100%; height:150px; display:block; background:var(--panel); border-top:1px solid var(--line);
           cursor:pointer; }
  @media (max-width: 640px) { #plots { height:120px; } #help { display:none; } }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>Hook-grasp rollout</h1>
    <span class="sub" id="meta"></span>
  </header>
  <div id="view">
    <div id="hud">
      <div class="state" id="h-state">free</div>
      <div id="h-time"></div>
      <div id="h-dist"></div>
      <div id="h-act"></div>
      <div id="h-rew"></div>
    </div>
    <div id="help">left-drag: rotate &middot; wheel: zoom &middot; right-drag: pan<br>
      space: play/pause &middot; &larr;/&rarr;: &plusmn;1 step (Shift: &plusmn;10)</div>
  </div>
  <div id="controls">
    <button id="b-back" title="-1 step (←)">&#9664;&#9646;</button>
    <button id="b-play" title="Play/pause (space)">&#9654;</button>
    <button id="b-fwd" title="+1 step (→)">&#9646;&#9654;</button>
    <input type="range" id="slider" min="0" value="0" step="1">
    <span id="time"></span>
    <label>speed
      <select id="speed">
        <option value="0.1">0.1&times;</option><option value="0.25">0.25&times;</option>
        <option value="0.5">0.5&times;</option><option value="1" selected>1&times;</option>
        <option value="2">2&times;</option><option value="4">4&times;</option>
      </select></label>
    <label><input type="checkbox" id="loop" checked> loop</label>
    <label><input type="checkbox" id="follow" checked> follow drone</label>
    <label>trail
      <select id="trail">
        <option value="0">0</option><option value="50">0.5 s</option>
        <option value="100" selected>1 s</option><option value="300">3 s</option>
      </select></label>
    <button id="b-reset" title="Reset camera">&#8634;</button>
  </div>
  <canvas id="plots"></canvas>
</div>

<script type="importmap">
{ "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/" } }
</script>
<script id="rollout-data" type="application/json">__DATA__</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const D = JSON.parse(document.getElementById("rollout-data").textContent);
const T = D.T, dt = D.dt;
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
document.getElementById("meta").textContent =
  `__TITLE__ · ${T} steps · ${(T * dt).toFixed(2)} s · seed ${D.seed ?? "?"}`;

// ---------------------------------------------------------------- scene
const view = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
view.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.up.set(0, 0, 1);                       // z up, like Isaac
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xffffff, 0x888888, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(2, -3, 5); scene.add(sun);
const grid = new THREE.GridHelper(6, 30, 0x999999, 0xbbbbbb);
grid.rotation.x = Math.PI / 2; grid.material.transparent = true; grid.material.opacity = 0.35; scene.add(grid);
scene.add(new THREE.AxesHelper(0.3));

const V = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const UP = new THREE.Vector3(0, 1, 0);
function stick(color, radius) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 1, 10),
                           new THREE.MeshStandardMaterial({ color, roughness: 0.6 }));
  scene.add(m); return m;
}
function placeStick(m, a, b) {
  const pa = V(a), pb = V(b), dir = pb.clone().sub(pa), len = dir.length();
  m.visible = len > 1e-6;
  if (!m.visible) return;
  m.position.copy(pa).add(pb).multiplyScalar(0.5);
  m.scale.set(1, len, 1);
  m.quaternion.setFromUnitVectors(UP, dir.normalize());
}
function ball(color, r) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 12),
                           new THREE.MeshStandardMaterial({ color, roughness: 0.5 }));
  scene.add(m); return m;
}

// basket walls (4 quads) + edges
const wallGeom = new THREE.BufferGeometry();
wallGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(4 * 6 * 3), 3));
const walls = new THREE.Mesh(wallGeom, new THREE.MeshStandardMaterial({
  color: 0xc8a165, transparent: true, opacity: 0.28, side: THREE.DoubleSide, depthWrite: false }));
scene.add(walls);
const edgeGeom = new THREE.BufferGeometry();
edgeGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(4 * 8 * 3), 3));
const edges = new THREE.LineSegments(edgeGeom, new THREE.LineBasicMaterial({ color: 0x8a6d3b }));
scene.add(edges);

const bar = stick(css("--bar") || "#d62728", 0.012);
const DRONE = css("--accent") || "#1f77b4";
const arms = [0, 1, 2, 3].map(() => stick(DRONE, 0.008));
const rotors = [0, 1, 2, 3].map(() => {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.004, 20),
    new THREE.MeshStandardMaterial({ color: DRONE, transparent: true, opacity: 0.55 }));
  m.rotation.x = Math.PI / 2; scene.add(m); return m;
});
const body = ball(DRONE, 0.025);
const nW = D.wire[0].length;
const wireSegs = Array.from({ length: nW + 1 }, () => stick(css("--muted") || "#6b6b70", 0.0035));
const hookBall = ball(0x7f7f7f, 0.022);
const FREE = new THREE.Color(0x7f7f7f), ATT = new THREE.Color(css("--ok") || "#2ca02c");

const MAXTRAIL = 400;
const trailGeom = new THREE.BufferGeometry();
trailGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(MAXTRAIL * 3), 3));
const trail = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: 0x2ca02c, transparent: true, opacity: 0.7 }));
scene.add(trail);

// episode start of each step (trail must not cross resets; done[k] => state at k is already reset)
const epStart = new Int32Array(T);
for (let k = 1; k < T; k++) epStart[k] = D.done[k] ? k : epStart[k - 1];

// ---------------------------------------------------------------- state / UI
let k = 0, playing = false, acc = 0;
const slider = document.getElementById("slider"); slider.max = T - 1;
const bPlay = document.getElementById("b-play");
const speedSel = document.getElementById("speed"), loopCb = document.getElementById("loop");
const followCb = document.getElementById("follow"), trailSel = document.getElementById("trail");
const lastDrone = new THREE.Vector3();

function resetCamera() {
  const c = V(D.drone[k]).add(V(D.bar0[k])).multiplyScalar(0.5);
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(1.6, -1.9, 1.0));
  lastDrone.copy(V(D.drone[k]));
  controls.update();
}

function setStep(n) {
  k = Math.max(0, Math.min(T - 1, n));
  // walls
  const wp = wallGeom.attributes.position.array, ep = edgeGeom.attributes.position.array;
  let i = 0, j = 0;
  for (const q of D.walls[k]) {
    for (const idx of [0, 1, 2, 0, 2, 3]) { wp[i++] = q[idx][0]; wp[i++] = q[idx][1]; wp[i++] = q[idx][2]; }
    for (const [a, b] of [[0, 1], [1, 2], [2, 3], [3, 0]]) {
      ep[j++] = q[a][0]; ep[j++] = q[a][1]; ep[j++] = q[a][2]; ep[j++] = q[b][0]; ep[j++] = q[b][1]; ep[j++] = q[b][2];
    }
  }
  wallGeom.attributes.position.needsUpdate = true; wallGeom.computeVertexNormals();
  edgeGeom.attributes.position.needsUpdate = true;
  placeStick(bar, D.bar0[k], D.bar1[k]);
  // drone
  const dpos = D.drone[k];
  body.position.copy(V(dpos));
  D.rotors[k].forEach((r, m) => { placeStick(arms[m], dpos, r); rotors[m].position.copy(V(r)); });
  // wire + hook
  const chain = [dpos, ...D.wire[k], D.hook[k]];
  for (let s = 0; s < chain.length - 1; s++) placeStick(wireSegs[s], chain[s], chain[s + 1]);
  hookBall.position.copy(V(D.hook[k]));
  const att = D.attached[k] === 1;
  hookBall.material.color.copy(att ? ATT : FREE);
  // trail
  const len = Math.min(+trailSel.value, MAXTRAIL - 1), s0 = Math.max(epStart[k], k - len);
  const tp = trailGeom.attributes.position.array; let n2 = 0;
  for (let s = s0; s <= k; s++) { const h = D.hook[s]; tp[n2++] = h[0]; tp[n2++] = h[1]; tp[n2++] = h[2]; }
  trailGeom.setDrawRange(0, k - s0 + 1); trailGeom.attributes.position.needsUpdate = true;
  trail.material.color.copy(att ? ATT : FREE);
  // follow: translate camera + target with the drone, keep the user's viewing angle
  const dv = V(dpos);
  if (followCb.checked) {
    const delta = dv.clone().sub(lastDrone);
    if (delta.length() < 1.0) { camera.position.add(delta); controls.target.add(delta); }
  }
  lastDrone.copy(dv);
  // HUD
  const st = document.getElementById("h-state");
  st.textContent = att ? "ATTACHED" : "free"; st.classList.toggle("on", att);
  document.getElementById("h-time").textContent = `t = ${(k * dt).toFixed(2)} s · step ${k}`;
  document.getElementById("h-dist").textContent = `hook–bar: ${D.dist[k].toFixed(3)} m`;
  document.getElementById("h-act").textContent = `|action|: ${D.act[k].toFixed(2)}`;
  document.getElementById("h-rew").textContent = `reward: ${D.reward[k].toFixed(3)}`;
  document.getElementById("time").textContent = `${(k * dt).toFixed(2)} / ${((T - 1) * dt).toFixed(2)} s`;
  slider.value = k;
  drawCursor();
}

function setPlaying(p) { playing = p; bPlay.innerHTML = p ? "&#10074;&#10074;" : "&#9654;"; acc = 0; }
bPlay.onclick = () => { if (!playing && k >= T - 1) setStep(0); setPlaying(!playing); };
document.getElementById("b-back").onclick = () => { setPlaying(false); setStep(k - 1); };
document.getElementById("b-fwd").onclick = () => { setPlaying(false); setStep(k + 1); };
document.getElementById("b-reset").onclick = resetCamera;
slider.oninput = () => setStep(+slider.value);
trailSel.onchange = () => setStep(k);
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); bPlay.onclick(); }
  else if (e.code === "ArrowRight") { e.preventDefault(); setPlaying(false); setStep(k + (e.shiftKey ? 10 : 1)); }
  else if (e.code === "ArrowLeft") { e.preventDefault(); setPlaying(false); setStep(k - (e.shiftKey ? 10 : 1)); }
});

// ---------------------------------------------------------------- plots
const pc = document.getElementById("plots"), pctx = pc.getContext("2d");
let plotBg = null;
const PAD_L = 92, PAD_R = 12;
const series = [
  { name: "hook–bar [m]", y: D.dist, color: () => css("--bar") },
  { name: "attached", y: D.attached, color: () => css("--ok"), fill: true, max: 1 },
  { name: "|action|", y: D.act, color: () => css("--accent") },
];
const xOf = (s, w) => PAD_L + (s / Math.max(1, T - 1)) * (w - PAD_L - PAD_R);
function drawPlots() {
  const dpr = window.devicePixelRatio || 1, w = pc.clientWidth, h = pc.clientHeight;
  pc.width = w * dpr; pc.height = h * dpr; pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  pctx.clearRect(0, 0, w, h);
  const rowH = h / series.length;
  pctx.font = "11px system-ui, sans-serif";
  series.forEach((sr, r) => {
    const top = r * rowH + 6, bot = (r + 1) * rowH - 6;
    const ymax = sr.max ?? Math.max(1e-6, ...sr.y);
    pctx.fillStyle = css("--muted"); pctx.textBaseline = "middle";
    pctx.fillText(sr.name, 8, (top + bot) / 2);
    pctx.strokeStyle = css("--line"); pctx.lineWidth = 1;
    pctx.beginPath(); pctx.moveTo(PAD_L, bot + 0.5); pctx.lineTo(w - PAD_R, bot + 0.5); pctx.stroke();
    pctx.strokeStyle = sr.color(); pctx.fillStyle = sr.color(); pctx.lineWidth = 1.4;
    pctx.beginPath();
    for (let s = 0; s < T; s++) {
      const x = xOf(s, w), y = bot - (sr.y[s] / ymax) * (bot - top);
      s ? pctx.lineTo(x, y) : pctx.moveTo(x, y);
    }
    if (sr.fill) { pctx.lineTo(xOf(T - 1, w), bot); pctx.lineTo(xOf(0, w), bot); pctx.globalAlpha = 0.45; pctx.fill(); pctx.globalAlpha = 1; }
    else pctx.stroke();
  });
  pctx.setLineDash([3, 3]);
  for (let s = 0; s < T; s++) {
    if (!D.caught[s] && !D.done[s]) continue;
    pctx.strokeStyle = D.caught[s] ? css("--ok") : css("--fg");
    pctx.beginPath(); pctx.moveTo(xOf(s, w), 0); pctx.lineTo(xOf(s, w), h); pctx.stroke();
  }
  pctx.setLineDash([]);
  plotBg = pctx.getImageData(0, 0, pc.width, pc.height);
}
function drawCursor() {
  if (!plotBg) return;
  pctx.putImageData(plotBg, 0, 0);
  const x = xOf(k, pc.clientWidth);
  pctx.strokeStyle = css("--fg"); pctx.lineWidth = 1.5;
  pctx.beginPath(); pctx.moveTo(x, 0); pctx.lineTo(x, pc.clientHeight); pctx.stroke();
}
function seekFromPlot(e) {
  const r = pc.getBoundingClientRect(), w = r.width;
  const s = Math.round(((e.clientX - r.left - PAD_L) / (w - PAD_L - PAD_R)) * (T - 1));
  setStep(s);
}
let dragging = false;
pc.addEventListener("pointerdown", (e) => { dragging = true; setPlaying(false); seekFromPlot(e); });
window.addEventListener("pointermove", (e) => { if (dragging) seekFromPlot(e); });
window.addEventListener("pointerup", () => { dragging = false; });

// ---------------------------------------------------------------- loop
function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix();
  drawPlots(); drawCursor();
}
window.addEventListener("resize", resize);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { applyTheme(); resize(); });
function applyTheme() { scene.background = new THREE.Color(css("--scene") || "#eef0f3"); }

let last = performance.now();
function tick(now) {
  const el = (now - last) / 1000; last = now;
  if (playing) {
    acc += (el * +speedSel.value) / dt;
    const n = Math.floor(acc);
    if (n > 0) {
      acc -= n;
      let nk = k + n;
      if (nk >= T) { if (loopCb.checked) nk = nk % T; else { nk = T - 1; setPlaying(false); } }
      setStep(nk);
    }
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
applyTheme();
setStep(0);
resetCamera();
resize();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", help="<tag>_rollout.npz written by play.py")
    parser.add_argument("--out", default=None, help="Output .html (default: <tag>_viewer.html next to the npz).")
    args = parser.parse_args()
    export(args.npz, args.out)


if __name__ == "__main__":
    main()
