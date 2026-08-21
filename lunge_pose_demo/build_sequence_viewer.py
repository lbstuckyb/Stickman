"""Run RF-DETR keypoint preview on every frame of a video and write:

  - <label>_all_frames.json          raw per-frame payloads (normalized 0..1 coords)
  - <label>_keypoints_scrubber.html  standalone frame-by-frame skeleton viewer
                                      (arrow keys / scrubber / play — points + lines only,
                                      no video frame embedded)

This decodes and runs inference on every frame, so it's slow: at 384px on CPU,
budget roughly 0.7-1s/frame (see the README table for other resolutions).

Usage:
    ../.venv/bin/python build_sequence_viewer.py lunge.webm
    ../.venv/bin/python build_sequence_viewer.py lunge.webm --resolution 384 --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root, for rfdetr_pose

from rfdetr_pose import (KEYPOINT_NAMES, SKELETON, PoseModel,  # noqa: E402
                         add_device_args, announce_device)


def build_html(data_json: str) -> str:
    return r"""<title>Keypoints — All Frames</title>
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
  .stat{ padding:0.65rem 1.1rem; border-right:1px solid var(--border); min-width:5.5rem; }
  .stat:last-child{ border-right:none; }
  .stat-label{ font-family:'IBM Plex Mono', monospace; font-size:0.62rem; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-faint); margin-bottom:0.2rem; }
  .stat-value{ font-family:'IBM Plex Mono', monospace; font-size:1.05rem; font-weight:500; color:var(--text); font-variant-numeric:tabular-nums; }
  .stat-value.accent{ color:var(--amber); }
  main{ display:grid; grid-template-columns:minmax(0, 1.55fr) minmax(320px, 1fr); gap:1.5rem; align-items:start; }
  @media (max-width: 920px){ main{ grid-template-columns:1fr; } }
  .panel{ background:var(--surface); border:1px solid var(--border); border-radius:12px; }
  .stage-panel{ overflow:hidden; }
  .stage-head{ display:flex; justify-content:space-between; align-items:center; padding:0.9rem 1.15rem; border-bottom:1px solid var(--border-soft); gap:1rem; flex-wrap:wrap; }
  .stage-head h2, .table-head h2{ font-family:'Sora', sans-serif; font-size:0.92rem; font-weight:600; margin:0; }
  .frame-readout{ display:flex; align-items:baseline; gap:0.85rem; font-family:'IBM Plex Mono', monospace; font-size:0.75rem; color:var(--text-dim); }
  .frame-readout b{ font-size:0.98rem; color:var(--text); font-weight:500; font-variant-numeric:tabular-nums; }
  .legend{ display:flex; gap:1rem; font-family:'IBM Plex Mono', monospace; font-size:0.7rem; color:var(--text-dim); }
  .legend-item{ display:flex; align-items:center; gap:0.4rem; }
  .swatch{ width:0.85rem; height:0.15rem; border-radius:1px; }
  .swatch.dot{ width:0.5rem; height:0.5rem; border-radius:50%; }
  .stage{ position:relative; width:100%;
    background: linear-gradient(var(--grid) 1px, transparent 1px) 0 0 / 100% 12.5%,
                linear-gradient(90deg, var(--grid) 1px, transparent 1px) 0 0 / 12.5% 100%,
                var(--stage-bg); }
  .stage svg{ position:absolute; inset:0; width:100%; height:100%; }
  .no-detect{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-family:'IBM Plex Mono', monospace; font-size:0.8rem; letter-spacing:0.04em; color:var(--warn); opacity:0; pointer-events:none; transition:opacity 0.1s; }
  .no-detect.show{ opacity:0.75; }
  .kp-line{ stroke:var(--teal); stroke-linecap:round; opacity:0.85; }
  .kp-line.dim{ opacity:0.12; }
  .kp-dot{ fill:var(--amber); stroke:#0b0e11; cursor:pointer; }
  .kp-dot.dim{ opacity:0.18; }
  .kp-dot.active{ fill:#fff; }
  .kp-ring{ fill:none; stroke:var(--amber); opacity:0.55; pointer-events:none; }
  .kp-ring.dim{ opacity:0.08; }
  .kp-label{ font-family:'IBM Plex Mono', monospace; fill:var(--text); paint-order:stroke; stroke:#0b0e11; pointer-events:none; opacity:0; }
  .kp-label.show{ opacity:1; }
  .box-rect{ fill:none; stroke:var(--text-faint); stroke-dasharray:4 3; opacity:0.6; }
  .box-score{ font-family:'IBM Plex Mono', monospace; fill:var(--text-dim); }
  .scrub-row{ display:flex; align-items:center; gap:0.75rem; padding:0.85rem 1.15rem; border-top:1px solid var(--border-soft); }
  .transport-btn{ display:flex; align-items:center; justify-content:center; width:2.1rem; height:2.1rem; border-radius:7px; border:1px solid var(--border); background:var(--surface-2); color:var(--text); cursor:pointer; flex:none; transition:background 0.12s, border-color 0.12s; }
  .transport-btn:hover{ background:#283241; border-color:#3a4655; }
  .transport-btn:active{ background:#1b222a; }
  .transport-btn svg{ width:0.9rem; height:0.9rem; fill:currentColor; }
  .transport-btn.play{ color:var(--amber); }
  input[type=range].scrub{ flex:1; -webkit-appearance:none; appearance:none; height:4px; border-radius:2px; background:var(--border); accent-color:var(--teal); }
  .frame-num{ font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:var(--text-faint); font-variant-numeric:tabular-nums; min-width:5.2em; text-align:right; }
  .controls{ display:flex; flex-wrap:wrap; gap:1.5rem; align-items:center; padding:0.15rem 1.15rem 0.85rem; font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:var(--text-dim); }
  .control{ display:flex; align-items:center; gap:0.6rem; }
  .control label{ white-space:nowrap; }
  .control input[type=range]{ width:110px; accent-color:var(--amber); }
  .control .val{ color:var(--amber); min-width:2.6em; font-variant-numeric:tabular-nums; }
  .toggle{ display:inline-flex; align-items:center; gap:0.45rem; cursor:pointer; user-select:none; }
  .toggle input{ accent-color:var(--teal); }
  .hint{ margin-left:auto; color:var(--text-faint); }
  .hint kbd{ font-family:'IBM Plex Mono', monospace; background:var(--surface-2); border:1px solid var(--border); border-radius:3px; padding:0.02rem 0.35rem; color:var(--text-dim); }
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
  .empty-state{ padding:2.5rem 1.15rem; text-align:center; font-family:'IBM Plex Mono', monospace; font-size:0.78rem; color:var(--text-faint); }
  footer{ margin-top:1.75rem; padding-top:1.25rem; border-top:1px solid var(--border-soft); font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:var(--text-faint); line-height:1.7; }
  footer code{ background:var(--surface-2); padding:0.05rem 0.35rem; border-radius:3px; color:var(--text-dim); }
  ::selection{ background:var(--teal-dim); color:#fff; }
</style>

<div class="wrap">
  <header>
    <div>
      <p class="eyebrow">RF-DETR Keypoint Preview &middot; COCO-17 &middot; all frames</p>
      <h1>Skeleton across every frame</h1>
    </div>
    <div class="stat-strip">
      <div class="stat"><div class="stat-label">Frames</div><div class="stat-value" id="stat-frames">&mdash;</div></div>
      <div class="stat"><div class="stat-label">Detected</div><div class="stat-value accent" id="stat-detected">&mdash;</div></div>
      <div class="stat"><div class="stat-label">Source fps</div><div class="stat-value" id="stat-fps">&mdash;</div></div>
      <div class="stat"><div class="stat-label">Resolution</div><div class="stat-value" id="stat-res">&mdash;</div></div>
      <div class="stat"><div class="stat-label">Device</div><div class="stat-value" id="stat-device">&mdash;</div></div>
    </div>
  </header>
  <main>
    <div class="panel stage-panel">
      <div class="stage-head">
        <h2>Skeleton reconstruction</h2>
        <div class="frame-readout">
          <span>frame <b id="ro-frame">0</b></span>
          <span>t=<b id="ro-t">0.00</b>s</span>
          <span>score <b id="ro-score">&mdash;</b></span>
          <span>infer <b id="ro-ms">&mdash;</b>ms</span>
        </div>
        <div class="legend">
          <div class="legend-item"><span class="swatch" style="background:var(--teal)"></span>skeleton</div>
          <div class="legend-item"><span class="swatch dot" style="background:var(--amber)"></span>joint</div>
        </div>
      </div>
      <div class="stage" id="stage">
        <svg id="overlay" preserveAspectRatio="xMidYMid meet"></svg>
        <div class="no-detect" id="no-detect">no detection above threshold</div>
      </div>
      <div class="scrub-row">
        <button class="transport-btn" id="btn-prev" title="Previous frame (&larr;)"><svg viewBox="0 0 24 24"><path d="M17 4 7 12l10 8z"/></svg></button>
        <button class="transport-btn play" id="btn-play" title="Play / pause (space)">
          <svg id="icon-play" viewBox="0 0 24 24"><path d="M6 4l14 8-14 8z"/></svg>
          <svg id="icon-pause" viewBox="0 0 24 24" style="display:none"><path d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg>
        </button>
        <button class="transport-btn" id="btn-next" title="Next frame (&rarr;)"><svg viewBox="0 0 24 24"><path d="M7 4l10 8-10 8z"/></svg></button>
        <input type="range" class="scrub" id="scrub" min="0" max="0" step="1" value="0" />
        <div class="frame-num" id="frame-num">0 / 0</div>
      </div>
      <div class="controls">
        <div class="control">
          <label for="conf-threshold">confidence &ge;</label>
          <input type="range" id="conf-threshold" min="0" max="1" step="0.01" value="0.3" />
          <span class="val" id="conf-threshold-val">0.30</span>
        </div>
        <label class="toggle"><input type="checkbox" id="toggle-labels" /> joint labels</label>
        <label class="toggle"><input type="checkbox" id="toggle-box" checked /> person box</label>
        <span class="hint"><kbd>&larr;</kbd> <kbd>&rarr;</kbd> step &middot; <kbd>space</kbd> play</span>
      </div>
    </div>
    <div class="panel table-panel">
      <div class="table-head"><h2>Joints</h2><span class="count" id="joint-count">&mdash;</span></div>
      <div class="joint-list" id="joint-list"></div>
    </div>
  </main>
  <footer>
    every frame decoded from the source video and run through the same <code>RFDETRKeypointPreview</code>
    call as the single-frame viewer &mdash; no tracking or interpolation between frames, no video
    underneath here either. each step is an independent detection:
    <code>{ score, box, keypoints: [{ x, y, v, occluded }] }</code> per person, drawn against a
    viewBox held fixed across the whole sequence so the figure doesn't rescale as it moves.
  </footer>
</div>

<script>
const DATA = __DATA_JSON__;
const NAMES = DATA.names, SKELETON = DATA.skeleton;
const [VBX, VBY, VBW, VBH] = DATA.viewBox;

const svg = document.getElementById('overlay');
const stage = document.getElementById('stage');
const jointList = document.getElementById('joint-list');
const noDetect = document.getElementById('no-detect');
const scrub = document.getElementById('scrub');
const frameNum = document.getElementById('frame-num');
const thresholdInput = document.getElementById('conf-threshold');
const thresholdVal = document.getElementById('conf-threshold-val');
const toggleLabels = document.getElementById('toggle-labels');
const toggleBox = document.getElementById('toggle-box');
const jointCount = document.getElementById('joint-count');
const btnPrev = document.getElementById('btn-prev');
const btnNext = document.getElementById('btn-next');
const btnPlay = document.getElementById('btn-play');
const iconPlay = document.getElementById('icon-play');
const iconPause = document.getElementById('icon-pause');

const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs){ const e = document.createElementNS(NS, tag); for(const k in attrs) e.setAttribute(k, attrs[k]); return e; }

document.getElementById('stat-frames').textContent = DATA.video.n_frames;
document.getElementById('stat-detected').textContent = `${DATA.detected} / ${DATA.video.n_frames}`;
document.getElementById('stat-fps').textContent = DATA.video.fps.toFixed(1);
document.getElementById('stat-res').textContent = DATA.resolution + 'px';
document.getElementById('stat-device').textContent = DATA.device;

svg.setAttribute('viewBox', `${VBX} ${VBY} ${VBW} ${VBH}`);
stage.style.aspectRatio = `${VBW} / ${VBH}`;

const unit = Math.max(VBW, VBH) / 90;
const dotR = unit * 0.9, ringR = unit * 2.3, lineW = unit * 0.42;
const labelSize = unit * 1.4, boxStroke = unit * 0.17, scoreSize = unit * 1.5;

const maxPeople = DATA.frames.reduce((m, f) => Math.max(m, f[3].length), 1);
const pools = [];
for(let p = 0; p < maxPeople; p++){
  const lines = SKELETON.map(() => { const line = el('line', { class: 'kp-line', 'stroke-width': lineW }); svg.appendChild(line); return line; });
  const joints = NAMES.map(() => {
    const ring = el('circle', { class: 'kp-ring', r: ringR, 'stroke-width': lineW * 0.85 });
    const dot = el('circle', { class: 'kp-dot', r: dotR, 'stroke-width': boxStroke * 3 });
    const label = el('text', { class: 'kp-label', 'font-size': labelSize, 'stroke-width': labelSize * 0.24 });
    svg.appendChild(ring); svg.appendChild(dot); svg.appendChild(label);
    return { ring, dot, label };
  });
  const boxRect = el('rect', { class: 'box-rect', 'stroke-width': boxStroke });
  const boxScore = el('text', { class: 'box-score', 'font-size': scoreSize });
  svg.appendChild(boxRect); svg.appendChild(boxScore);
  pools.push({ lines, joints, boxRect, boxScore });
}

let current = 0, activeJoint = null, playing = false, playTimer = null;

function hidePool(pool){
  pool.lines.forEach(l => l.style.display = 'none');
  pool.joints.forEach(j => { j.ring.style.display = 'none'; j.dot.style.display = 'none'; j.label.style.display = 'none'; });
  pool.boxRect.style.display = 'none'; pool.boxScore.style.display = 'none';
}

function render(i){
  current = Math.max(0, Math.min(DATA.frames.length - 1, i));
  const [idx, t, ms, people] = DATA.frames[current];

  document.getElementById('ro-frame').textContent = idx;
  document.getElementById('ro-t').textContent = t.toFixed(2);
  document.getElementById('ro-score').textContent = people.length ? people[0][0].toFixed(2) : '—';
  document.getElementById('ro-ms').textContent = ms.toFixed(0);
  scrub.value = current;
  frameNum.textContent = `${current + 1} / ${DATA.frames.length}`;
  noDetect.classList.toggle('show', people.length === 0);

  const t_ = parseFloat(thresholdInput.value);
  let visCount = 0, totalCount = 0;

  pools.forEach((pool, p) => {
    const person = people[p];
    if(!person){ hidePool(pool); return; }
    const [score, box, joints] = person;

    pool.boxRect.style.display = toggleBox.checked ? '' : 'none';
    pool.boxScore.style.display = toggleBox.checked ? '' : 'none';
    pool.boxRect.setAttribute('x', box[0]); pool.boxRect.setAttribute('y', box[1]);
    pool.boxRect.setAttribute('width', box[2] - box[0]); pool.boxRect.setAttribute('height', box[3] - box[1]);
    pool.boxScore.setAttribute('x', box[0]); pool.boxScore.setAttribute('y', box[1] - unit * 0.5);
    pool.boxScore.textContent = score.toFixed(2);

    joints.forEach(([x, y, v, occ], j) => {
      totalCount++;
      const { ring, dot, label } = pool.joints[j];
      const pass = v >= t_;
      if(pass) visCount++;
      [ring, dot, label].forEach(node => node.style.display = '');
      ring.setAttribute('cx', x); ring.setAttribute('cy', y);
      dot.setAttribute('cx', x); dot.setAttribute('cy', y);
      label.setAttribute('x', x + dotR * 1.8); label.setAttribute('y', y - dotR * 1.6);
      label.textContent = NAMES[j].replace(/_/g, ' ');
      ring.classList.toggle('dim', !pass);
      dot.classList.toggle('dim', !pass);
      dot.classList.toggle('active', p === 0 && j === activeJoint);
      label.classList.toggle('show', toggleLabels.checked && pass);
    });

    SKELETON.forEach(([a, b], li) => {
      const line = pool.lines[li];
      const ja = joints[a], jb = joints[b];
      line.style.display = '';
      line.setAttribute('x1', ja[0]); line.setAttribute('y1', ja[1]);
      line.setAttribute('x2', jb[0]); line.setAttribute('y2', jb[1]);
      line.classList.toggle('dim', !(ja[2] >= t_ && jb[2] >= t_));
    });
  });

  jointCount.textContent = totalCount ? `${visCount} of ${totalCount}` : '—';
  renderTable(people[0]);
}

function renderTable(person){
  jointList.innerHTML = '';
  if(!person){
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = 'no person above the detection threshold on this frame';
    jointList.appendChild(empty);
    return;
  }
  const [, , joints] = person;
  joints.forEach(([x, y, v, occ], i) => {
    const row = document.createElement('div'); row.className = 'joint-row';
    const name = document.createElement('div'); name.className = 'joint-name'; name.textContent = NAMES[i].replace(/_/g, ' ');
    const coords = document.createElement('div'); coords.className = 'joint-coords'; coords.textContent = `${x.toFixed(0)}, ${y.toFixed(0)}px`;
    const meta = document.createElement('div'); meta.className = 'joint-meta';
    const track = document.createElement('div'); track.className = 'conf-track';
    const fill = document.createElement('div'); fill.className = 'conf-fill' + (v < 0.3 ? ' low' : ''); fill.style.width = (v * 100).toFixed(0) + '%';
    track.appendChild(fill);
    const num = document.createElement('div'); num.className = 'conf-num'; num.textContent = v.toFixed(2);
    const badge = document.createElement('div'); badge.className = 'badge' + (occ ? ' occ' : ''); badge.textContent = occ ? 'occluded' : 'visible';
    meta.appendChild(track); meta.appendChild(num); meta.appendChild(badge);
    row.appendChild(name); row.appendChild(coords); row.appendChild(meta);
    row.addEventListener('mouseenter', () => { activeJoint = i; render(current); });
    row.addEventListener('mouseleave', () => { activeJoint = null; render(current); });
    jointList.appendChild(row);
  });
}

function step(delta){ pause(); render(current + delta); }
function play(){
  playing = true; iconPlay.style.display = 'none'; iconPause.style.display = '';
  const interval = Math.max(1000 / DATA.video.fps, 16);
  playTimer = setInterval(() => { let next = current + 1; if(next >= DATA.frames.length) next = 0; render(next); }, interval);
}
function pause(){ playing = false; iconPlay.style.display = ''; iconPause.style.display = 'none'; clearInterval(playTimer); }

scrub.max = DATA.frames.length - 1;
scrub.addEventListener('input', () => { pause(); render(parseInt(scrub.value, 10)); });
btnPrev.addEventListener('click', () => step(-1));
btnNext.addEventListener('click', () => step(1));
btnPlay.addEventListener('click', () => playing ? pause() : play());
thresholdInput.addEventListener('input', () => { thresholdVal.textContent = parseFloat(thresholdInput.value).toFixed(2); render(current); });
toggleLabels.addEventListener('change', () => render(current));
toggleBox.addEventListener('change', () => render(current));

document.addEventListener('keydown', (e) => {
  if(e.key === 'ArrowLeft'){ e.preventDefault(); step(-1); }
  else if(e.key === 'ArrowRight'){ e.preventDefault(); step(1); }
  else if(e.key === ' '){ e.preventDefault(); playing ? pause() : play(); }
});

render(0);
</script>
""".replace("__DATA_JSON__", data_json)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--resolution", type=int, default=384)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--label", default=None, help="output filename prefix (default: video stem)")
    add_device_args(ap)
    args = ap.parse_args()

    video_path = Path(args.video)
    label = args.label or video_path.stem
    out_dir = video_path.parent

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {w}x{h} @ {fps:.2f}fps, ~{n_frames} frames")

    announce_device(args.device)
    model = PoseModel(resolution=args.resolution, device=args.device,
                      half=args.half, optimize=args.optimize)
    print(f"model: RFDETRKeypointPreview @ {model.resolution}px on {model.device}")

    raw_frames = []
    idx = 0
    t0 = time.perf_counter()
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        payload = model.predict_payload(rgb, threshold=args.threshold, width=w, height=h)
        raw_frames.append({"i": idx, "t": round(idx / fps, 3), "ms": payload["inference_ms"], "people": payload["people"]})
        idx += 1
        if idx % 20 == 0:
            rate = idx / (time.perf_counter() - t0)
            print(f"  {idx}/{n_frames}  ({rate:.1f} fps, eta {(n_frames - idx) / rate:.0f}s)")
    cap.release()
    print(f"done: {idx} frames in {time.perf_counter() - t0:.1f}s")

    raw_out = {
        "video": {"w": w, "h": h, "fps": fps, "n_frames": idx},
        "resolution": model.resolution, "device": model.device, "threshold": args.threshold,
        "keypoint_names": KEYPOINT_NAMES, "skeleton": SKELETON, "frames": raw_frames,
    }
    raw_path = out_dir / f"{label}_all_frames.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_out, f)
    print(f"wrote {raw_path}")

    # compact per-frame arrays + a viewBox fixed across the whole sequence
    frames_out = []
    gx0 = gy0 = float("inf")
    gx1 = gy1 = float("-inf")
    detected = 0
    for fr in raw_frames:
        people_out = []
        for p in fr["people"]:
            joints = []
            for k in p["keypoints"]:
                x, y = k["x"] * w, k["y"] * h
                joints.append([round(x, 1), round(y, 1), round(k["v"], 3), k["occluded"]])
                if k["v"] >= 0.3:
                    gx0, gy0 = min(gx0, x), min(gy0, y)
                    gx1, gy1 = max(gx1, x), max(gy1, y)
            box = p["box"]
            people_out.append([round(p["score"], 3), [round(box[0] * w, 1), round(box[1] * h, 1),
                                                        round(box[2] * w, 1), round(box[3] * h, 1)], joints])
        if people_out:
            detected += 1
        frames_out.append([fr["i"], fr["t"], round(fr["ms"], 1), people_out])

    pad_x, pad_y = (gx1 - gx0) * 0.18, (gy1 - gy0) * 0.08
    view_box = [gx0 - pad_x, gy0 - pad_y, (gx1 - gx0) + pad_x * 2, (gy1 - gy0) + pad_y * 2]

    data = {
        "video": raw_out["video"], "resolution": model.resolution, "device": model.device,
        "names": KEYPOINT_NAMES, "skeleton": SKELETON,
        "viewBox": [round(v, 1) for v in view_box], "detected": detected, "frames": frames_out,
    }

    out_html = out_dir / f"{label}_keypoints_scrubber.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(build_html(json.dumps(data, separators=(",", ":"))))
    print(f"wrote {out_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
