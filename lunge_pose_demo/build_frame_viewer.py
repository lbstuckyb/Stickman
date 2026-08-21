"""Run RF-DETR keypoint preview on one image and write:

  - <label>_keypoints_out.jpg   annotated frame (skeleton + 1-sigma uncertainty ellipses)
  - <label>_keypoints_out.json  normalized payload — { people: [{ score, box, keypoints }] }
  - <label>_keypoints_viewer.html   standalone interactive skeleton reconstruction
                                     (points + lines only, drawn from the JSON above —
                                     no photo underneath)

Usage:
    ../.venv/bin/python build_frame_viewer.py lunge_frame.jpg
    ../.venv/bin/python build_frame_viewer.py lunge_frame.jpg --label lunge --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root, for rfdetr_pose

from rfdetr_pose import (KEYPOINT_NAMES, SKELETON, PoseModel,  # noqa: E402
                         add_device_args, announce_device)
from test_rfdetr_keypoints import annotate  # noqa: E402


def build_html(data_json: str) -> str:
    return r"""<title>Keypoints</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root{
    --bg:#12161b; --surface:#1a2028; --surface-2:#212933; --border:#2b3440; --border-soft:#232b35;
    --text:#e8edf1; --text-dim:#8d99a6; --text-faint:#5c6875;
    --teal:#4fd1c5; --teal-dim:#2f6b66; --amber:#f2b544; --amber-dim:#7a5c26; --warn:#e2574c;
    --stage-bg:#0d1116; --grid:#1b222a;
  }
  *{box-sizing:border-box;}
  body{ margin:0; background:var(--bg); color:var(--text); font-family:'IBM Plex Sans', system-ui, sans-serif; -webkit-font-smoothing:antialiased; }
  .wrap{ max-width:1400px; margin:0 auto; padding:2.5rem 2rem 4rem; }
  header{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-end; gap:1.5rem; margin-bottom:1.75rem; padding-bottom:1.5rem; border-bottom:1px solid var(--border-soft); }
  .eyebrow{ font-family:'IBM Plex Mono', monospace; font-size:0.72rem; letter-spacing:0.14em; text-transform:uppercase; color:var(--teal); margin:0 0 0.5rem; }
  h1{ font-family:'Sora', sans-serif; font-weight:700; font-size:clamp(1.6rem, 2.4vw, 2.15rem); margin:0; letter-spacing:-0.01em; text-wrap:balance; }
  h1 span{ color:var(--text-dim); font-weight:500; }
  .stat-strip{ display:flex; gap:0; flex-wrap:wrap; background:var(--surface); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
  .stat{ padding:0.65rem 1.1rem; border-right:1px solid var(--border); min-width:6.5rem; }
  .stat:last-child{ border-right:none; }
  .stat-label{ font-family:'IBM Plex Mono', monospace; font-size:0.62rem; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-faint); margin-bottom:0.2rem; }
  .stat-value{ font-family:'IBM Plex Mono', monospace; font-size:1.05rem; font-weight:500; color:var(--text); font-variant-numeric:tabular-nums; }
  .stat-value.accent{ color:var(--amber); }
  main{ display:grid; grid-template-columns:minmax(0, 1.55fr) minmax(320px, 1fr); gap:1.5rem; align-items:start; }
  @media (max-width: 920px){ main{ grid-template-columns:1fr; } }
  .panel{ background:var(--surface); border:1px solid var(--border); border-radius:12px; }
  .stage-panel{ overflow:hidden; }
  .stage-head{ display:flex; justify-content:space-between; align-items:center; padding:0.9rem 1.15rem; border-bottom:1px solid var(--border-soft); }
  .stage-head h2, .table-head h2{ font-family:'Sora', sans-serif; font-size:0.92rem; font-weight:600; margin:0; }
  .legend{ display:flex; gap:1rem; font-family:'IBM Plex Mono', monospace; font-size:0.7rem; color:var(--text-dim); }
  .legend-item{ display:flex; align-items:center; gap:0.4rem; }
  .swatch{ width:0.85rem; height:0.15rem; border-radius:1px; }
  .swatch.dot{ width:0.5rem; height:0.5rem; border-radius:50%; }
  .stage{ position:relative; width:100%;
    background: linear-gradient(var(--grid) 1px, transparent 1px) 0 0 / 100% 12.5%,
                linear-gradient(90deg, var(--grid) 1px, transparent 1px) 0 0 / 12.5% 100%,
                var(--stage-bg); }
  .stage svg{ position:absolute; inset:0; width:100%; height:100%; }
  .kp-line{ stroke:var(--teal); stroke-linecap:round; opacity:0.85; transition:opacity 0.15s; }
  .kp-line.dim{ opacity:0.12; }
  .kp-dot{ fill:var(--amber); stroke:#0b0e11; cursor:pointer; transition:opacity 0.15s; }
  .kp-dot.dim{ opacity:0.18; }
  .kp-dot.active{ fill:#fff; }
  .kp-ring{ fill:none; stroke:var(--amber); opacity:0.55; pointer-events:none; transition:opacity 0.15s; }
  .kp-ring.dim{ opacity:0.08; }
  .kp-label{ font-family:'IBM Plex Mono', monospace; fill:var(--text); paint-order:stroke; stroke:#0b0e11; pointer-events:none; opacity:0; transition:opacity 0.12s; }
  .kp-label.show{ opacity:1; }
  .box-rect{ fill:none; stroke:var(--text-faint); stroke-dasharray:4 3; opacity:0.6; }
  .box-score{ font-family:'IBM Plex Mono', monospace; fill:var(--text-dim); }
  .controls{ display:flex; flex-wrap:wrap; gap:1.5rem; align-items:center; padding:0.85rem 1.15rem; border-top:1px solid var(--border-soft); font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:var(--text-dim); }
  .control{ display:flex; align-items:center; gap:0.6rem; }
  .control label{ white-space:nowrap; }
  .control input[type=range]{ width:120px; accent-color:var(--amber); }
  .control .val{ color:var(--amber); min-width:2.6em; font-variant-numeric:tabular-nums; }
  .toggle{ display:inline-flex; align-items:center; gap:0.45rem; cursor:pointer; user-select:none; }
  .toggle input{ accent-color:var(--teal); }
  .table-panel{ display:flex; flex-direction:column; }
  .table-head{ display:flex; justify-content:space-between; align-items:center; padding:0.9rem 1.15rem; border-bottom:1px solid var(--border-soft); }
  .table-head .count{ font-family:'IBM Plex Mono', monospace; font-size:0.7rem; color:var(--text-faint); }
  .joint-list{ display:flex; flex-direction:column; }
  .joint-row{ display:grid; grid-template-columns:1fr auto; gap:0.15rem 0.75rem; padding:0.55rem 1.15rem; border-bottom:1px solid var(--border-soft); cursor:pointer; transition:background 0.12s; }
  .joint-row:last-child{ border-bottom:none; }
  .joint-row:hover, .joint-row.active{ background:var(--surface-2); }
  .joint-name{ font-size:0.82rem; font-weight:500; text-transform:capitalize; color:var(--text); }
  .joint-coords{ font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:var(--text-faint); font-variant-numeric:tabular-nums; text-align:right; }
  .joint-meta{ grid-column:1 / -1; display:flex; align-items:center; gap:0.6rem; }
  .conf-track{ flex:1; height:4px; background:var(--border); border-radius:2px; overflow:hidden; }
  .conf-fill{ height:100%; background:var(--teal); border-radius:2px; }
  .conf-fill.low{ background:var(--warn); }
  .conf-num{ font-family:'IBM Plex Mono', monospace; font-size:0.7rem; color:var(--text-dim); font-variant-numeric:tabular-nums; min-width:2.9em; text-align:right; }
  .badge{ font-family:'IBM Plex Mono', monospace; font-size:0.62rem; letter-spacing:0.04em; text-transform:uppercase; padding:0.1rem 0.4rem; border-radius:3px; border:1px solid var(--border); color:var(--text-faint); }
  .badge.occ{ color:var(--warn); border-color:#5a2f2b; }
  footer{ margin-top:1.75rem; padding-top:1.25rem; border-top:1px solid var(--border-soft); font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:var(--text-faint); line-height:1.7; }
  footer code{ background:var(--surface-2); padding:0.05rem 0.35rem; border-radius:3px; color:var(--text-dim); }
  ::selection{ background:var(--teal-dim); color:#fff; }
</style>

<div class="wrap">
  <header>
    <div>
      <p class="eyebrow">RF-DETR Keypoint Preview &middot; COCO-17</p>
      <h1>Skeleton rebuilt from raw output</h1>
    </div>
    <div class="stat-strip">
      <div class="stat"><div class="stat-label">People</div><div class="stat-value" id="stat-people">&mdash;</div></div>
      <div class="stat"><div class="stat-label">Detection score</div><div class="stat-value accent" id="stat-score">&mdash;</div></div>
      <div class="stat"><div class="stat-label">Inference</div><div class="stat-value" id="stat-latency">&mdash;</div></div>
      <div class="stat"><div class="stat-label">Resolution</div><div class="stat-value" id="stat-res">&mdash;</div></div>
      <div class="stat"><div class="stat-label">Device</div><div class="stat-value" id="stat-device">&mdash;</div></div>
    </div>
  </header>
  <main>
    <div class="panel stage-panel">
      <div class="stage-head">
        <h2>Skeleton reconstruction</h2>
        <div class="legend">
          <div class="legend-item"><span class="swatch" style="background:var(--teal)"></span>skeleton</div>
          <div class="legend-item"><span class="swatch dot" style="background:var(--amber)"></span>joint</div>
        </div>
      </div>
      <div class="stage" id="stage"><svg id="overlay" preserveAspectRatio="xMidYMid meet"></svg></div>
      <div class="controls">
        <div class="control">
          <label for="conf-threshold">confidence &ge;</label>
          <input type="range" id="conf-threshold" min="0" max="1" step="0.01" value="0.3" />
          <span class="val" id="conf-threshold-val">0.30</span>
        </div>
        <label class="toggle"><input type="checkbox" id="toggle-labels" /> joint labels</label>
        <label class="toggle"><input type="checkbox" id="toggle-box" checked /> person box</label>
      </div>
    </div>
    <div class="panel table-panel">
      <div class="table-head"><h2>Joints</h2><span class="count" id="joint-count">&mdash;</span></div>
      <div class="joint-list" id="joint-list"></div>
    </div>
  </main>
  <footer>
    drawn from <code>{ people: [{ score, box, keypoints: [{ x, y, v, occluded }] }] }</code> only &mdash;
    no video frame, just the (x, y) pairs and the fixed COCO-17 skeleton edge list, denormalized to
    source pixels and reprojected onto the grid below. <code>v</code> is per-joint &ldquo;findable&rdquo;
    confidence, not IoU. hover or tap a joint to isolate it.
  </footer>
</div>

<script>
const DATA = __DATA_JSON__;
const svg = document.getElementById('overlay');
const jointList = document.getElementById('joint-list');
const thresholdInput = document.getElementById('conf-threshold');
const thresholdVal = document.getElementById('conf-threshold-val');
const toggleLabels = document.getElementById('toggle-labels');
const toggleBox = document.getElementById('toggle-box');
const jointCount = document.getElementById('joint-count');

const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs){ const e = document.createElementNS(NS, tag); for(const k in attrs) e.setAttribute(k, attrs[k]); return e; }

document.getElementById('stat-people').textContent = '1';
document.getElementById('stat-score').textContent = DATA.score.toFixed(2);
document.getElementById('stat-latency').textContent = Math.round(DATA.meta) + ' ms';
document.getElementById('stat-res').textContent = DATA.resolution + 'px';
document.getElementById('stat-device').textContent = DATA.device;

const xs = DATA.keypoints.map(k => k.x), ys = DATA.keypoints.map(k => k.y);
const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
const padX = (maxX - minX) * 0.22, padY = (maxY - minY) * 0.12;
const vbX = minX - padX, vbY = minY - padY, vbW = (maxX - minX) + padX * 2, vbH = (maxY - minY) + padY * 2;
svg.setAttribute('viewBox', `${vbX} ${vbY} ${vbW} ${vbH}`);
document.getElementById('stage').style.aspectRatio = `${vbW} / ${vbH}`;

const unit = Math.max(vbW, vbH) / 90;
const dotR = unit * 0.9, ringR = unit * 2.3, lineW = unit * 0.42;
const labelSize = unit * 1.4, boxStroke = unit * 0.17, scoreSize = unit * 1.5;

const [bx1, by1, bx2, by2] = DATA.box;
const boxRect = el('rect', { class: 'box-rect', x: bx1 * DATA.image.w, y: by1 * DATA.image.h, width: (bx2 - bx1) * DATA.image.w, height: (by2 - by1) * DATA.image.h, 'stroke-width': boxStroke });
svg.appendChild(boxRect);
const boxScore = el('text', { class: 'box-score', x: bx1 * DATA.image.w, y: by1 * DATA.image.h - unit * 0.5, 'font-size': scoreSize });
boxScore.textContent = DATA.score.toFixed(2);
svg.appendChild(boxScore);

const lineEls = DATA.skeleton.map(([a, b]) => {
  const ka = DATA.keypoints[a], kb = DATA.keypoints[b];
  const line = el('line', { class: 'kp-line', x1: ka.x, y1: ka.y, x2: kb.x, y2: kb.y, 'stroke-width': lineW });
  svg.appendChild(line);
  return { line, a, b };
});

const jointEls = DATA.keypoints.map((k, i) => {
  const ring = el('circle', { class: 'kp-ring', cx: k.x, cy: k.y, r: ringR, 'stroke-width': lineW * 0.85 });
  const dot = el('circle', { class: 'kp-dot', cx: k.x, cy: k.y, r: dotR, 'stroke-width': boxStroke * 3 });
  const label = el('text', { class: 'kp-label', x: k.x + dotR * 1.8, y: k.y - dotR * 1.6, 'font-size': labelSize, 'stroke-width': labelSize * 0.24 });
  label.textContent = k.label;
  svg.appendChild(ring); svg.appendChild(dot); svg.appendChild(label);
  return { ring, dot, label, k, i };
});

const rows = DATA.keypoints.map((k, i) => {
  const row = document.createElement('div'); row.className = 'joint-row';
  const name = document.createElement('div'); name.className = 'joint-name'; name.textContent = k.label;
  const coords = document.createElement('div'); coords.className = 'joint-coords'; coords.textContent = `${k.x.toFixed(0)}, ${k.y.toFixed(0)}px`;
  const meta = document.createElement('div'); meta.className = 'joint-meta';
  const track = document.createElement('div'); track.className = 'conf-track';
  const fill = document.createElement('div'); fill.className = 'conf-fill' + (k.conf < 0.3 ? ' low' : ''); fill.style.width = (k.conf * 100).toFixed(0) + '%';
  track.appendChild(fill);
  const num = document.createElement('div'); num.className = 'conf-num'; num.textContent = k.conf.toFixed(2);
  const badge = document.createElement('div'); badge.className = 'badge' + (k.occluded ? ' occ' : ''); badge.textContent = k.occluded ? 'occluded' : 'visible';
  meta.appendChild(track); meta.appendChild(num); meta.appendChild(badge);
  row.appendChild(name); row.appendChild(coords); row.appendChild(meta);
  row.addEventListener('mouseenter', () => setActive(i));
  row.addEventListener('mouseleave', () => setActive(null));
  jointList.appendChild(row);
  return row;
});

jointEls.forEach(({ dot }, i) => {
  dot.addEventListener('mouseenter', () => setActive(i));
  dot.addEventListener('mouseleave', () => setActive(null));
});

function setActive(i){
  jointEls.forEach((j, idx) => j.dot.classList.toggle('active', idx === i));
  rows.forEach((r, idx) => r.classList.toggle('active', idx === i));
  if(i !== null) rows[i].scrollIntoView({ block: 'nearest' });
}

function applyThreshold(){
  const t = parseFloat(thresholdInput.value);
  thresholdVal.textContent = t.toFixed(2);
  let visibleCount = 0;
  jointEls.forEach(({ ring, dot, label, k }) => {
    const pass = k.conf >= t;
    if(pass) visibleCount++;
    ring.classList.toggle('dim', !pass);
    dot.classList.toggle('dim', !pass);
    label.classList.toggle('show', toggleLabels.checked && pass);
  });
  lineEls.forEach(({ line, a, b }) => {
    line.classList.toggle('dim', !(DATA.keypoints[a].conf >= t && DATA.keypoints[b].conf >= t));
  });
  rows.forEach((row, i) => { row.style.opacity = DATA.keypoints[i].conf >= t ? '1' : '0.35'; });
  jointCount.textContent = `${visibleCount} of ${DATA.keypoints.length}`;
}

thresholdInput.addEventListener('input', applyThreshold);
toggleLabels.addEventListener('change', applyThreshold);
toggleBox.addEventListener('change', () => {
  boxRect.style.display = toggleBox.checked ? '' : 'none';
  boxScore.style.display = toggleBox.checked ? '' : 'none';
});
applyThreshold();
</script>
""".replace("__DATA_JSON__", data_json)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--label", default=None, help="output filename prefix (default: image stem)")
    add_device_args(ap)
    args = ap.parse_args()

    image_path = Path(args.image)
    label = args.label or image_path.stem
    out_dir = image_path.parent

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print(f"FAIL: cannot read {image_path}")
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]

    announce_device(args.device)
    model = PoseModel(resolution=args.resolution, device=args.device,
                      half=args.half, optimize=args.optimize)
    kp, _ = model.predict(rgb, threshold=args.threshold)
    if len(kp) == 0:
        print("FAIL: no people detected")
        return 1

    out_jpg = out_dir / f"{label}_keypoints_out.jpg"
    cv2.imwrite(str(out_jpg), annotate(bgr, kp))

    payload = model.predict_payload(rgb, threshold=args.threshold, width=w, height=h)
    out_json = out_dir / f"{label}_keypoints_out.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    best = max(payload["people"], key=lambda p: p["score"])
    viewer_data = {
        "image": {"w": w, "h": h},
        "keypoints": [
            {
                "name": name, "label": name.replace("_", " "),
                "x": k["x"] * w, "y": k["y"] * h,
                "conf": k["v"], "occluded": k["occluded"],
            }
            for name, k in zip(KEYPOINT_NAMES, best["keypoints"])
        ],
        "skeleton": SKELETON,
        "box": best["box"],
        "score": best["score"],
        "meta": payload["inference_ms"],
        "resolution": payload["resolution"],
        "device": payload["device"],
    }
    out_html = out_dir / f"{label}_keypoints_viewer.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(build_html(json.dumps(viewer_data)))

    print(f"wrote {out_jpg}")
    print(f"wrote {out_json}")
    print(f"wrote {out_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
