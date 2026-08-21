# Lunge pose demo

Everything from one session of poking at `RFDETRKeypointPreview` (see `../README.md` for what
that model is) using `../lunge.webm` as the source clip. Two questions got answered here: what
the model's raw output looks like on one frame, and what it looks like across a whole clip.
The third page is not a viewer at all: it's a small arena game where the measured lunge *is* the
attack animation, driven by the same keypoints rather than by hand-authored keyframes.

## Files

| file | what it is | made by |
| --- | --- | --- |
| `lunge.webm` | source clip (moved here from the repo root) | — |
| `lunge_frame.jpg` | one frame pulled from the clip at t=3.5s | `ffmpeg -ss 3.5 -i lunge.webm -frames:v 1 ...` |
| `lunge_keypoints_out.jpg` | that frame annotated with skeleton + 1σ uncertainty ellipses | `../test_rfdetr_keypoints.py` |
| `lunge_keypoints_out.json` | raw model output for that frame — `{ people: [{ score, box, keypoints }] }` | `../test_rfdetr_keypoints.py` |
| `lunge_all_frames.json` | raw model output for **all 264 frames** of the clip | `build_sequence_viewer.py` |
| `lunge_keypoints_viewer.html` | interactive single-frame skeleton, rebuilt from the JSON only — no photo | `build_frame_viewer.py` |
| `lunge_keypoints_scrubber.html` | interactive all-frames skeleton, step/scrub/play through all 264 detections | `build_sequence_viewer.py` |
| `lunge_game.html` | **playable**: stick figure that *is* the captured skeleton — move, jump, and lunge at drifting targets | `build_game.py` |
| `build_frame_viewer.py` | reusable: image → annotated jpg + json + interactive viewer html | — |
| `build_sequence_viewer.py` | reusable: video → raw json + interactive scrubber html | — |
| `build_game.py` | reusable: per-frame keypoint json → clips + game html (no model inference) | — |
| `serve_game.py` | local server: serves `lunge_game.html`, runs RF-DETR on recorded/uploaded video for the in-page recorder, reads/writes `custom_moves.json` | — |
| `custom_moves.json` | recorded moves saved from the page (gitignored, written at runtime) | `serve_game.py` |

The three `build_*.py` scripts are the actual source of the three HTML pages — everything in them
(palette, layout, the "no video underneath, just the raw points and the fixed skeleton edge
list" framing) is inlined in the `.html` files already, so the HTML files stand alone. The
scripts exist so this is reproducible with a different image/video/threshold rather than a
one-off.

`build_game.py` is the odd one out: it never touches the model. It reads
`lunge_all_frames.json` and cuts three clips out of the track — `idle` (source frames 79–106,
the ready stance held before the lunge, ping-ponged), `lunge` (frames 106–134, a short hold at
the deepest pose, then 154–188, with a strike window over the deep extension), and `walk`.

The character **is the measured skeleton**. A clip frame is the **15 joint positions** the
detector reported, hip-centred and multiplied by one scale factor (`0.4466`, sized so the
subject stands 249 px tall in the arena); the lunge also carries a **root track** — how far the
hips actually travelled, frame by frame — so pressing `F` drives the figure 203 px forward and
steps it back to within 13 px of where it started, the way it happens on video. Nothing is
rebuilt from angles at runtime, so the limbs foreshorten exactly as they do in the source and
the body drops as far as the body drops. Two things that need saying:

- The **walk is procedural** and labelled as such in the emitted JSON: there is no walk cycle
  anywhere in the clip, the subject just stands, lunges, and stands again. It is authored in
  angles off the measured rest pose (torso, head and arm carriage measured; the ready stance's
  split legs averaged into one straight leg to swing around) and solved onto the subject's own
  measured bone lengths, so nothing changes shape when the state changes.
- Frames 161–162 carry a **duplicate detection** whose higher-scoring copy has left and right
  swapped. The builder follows one subject across frames — each frame keeps the candidate
  closest to the frame before it, and flips back a detection whose mirror image is closer —
  instead of taking `max(score)` per frame, which used to snap the limbs mid-recovery.

## Opening this again later

**All three HTML files are fully self-contained** — the keypoint data is inlined as JSON in a
`<script>` tag, there's no separate server or image to load. Just double-click any of the
`.html` files (or drag one into a browser tab / `open lunge_game.html`). The only external
call is a Google Fonts stylesheet; without internet it falls back to the system sans-serif and
still works fine.

**They're also published as Claude Artifacts**, private to this account:
- Single frame: https://claude.ai/code/artifact/4ac7ff38-8545-4378-8354-ffae7ed5dddf
- All frames (scrubber): https://claude.ai/code/artifact/e652cbfe-7d60-4157-85d5-59e6daca349a

Those links work from any browser, any machine, whether or not this repo or terminal session is
still around — Anthropic hosts them, not this machine. Find them again anytime via
[claude.ai/code/artifacts](https://claude.ai/code/artifacts) if you lose the links.

## Regenerating

From the repo root, using the project's existing `.venv` (see `../README.md` for setup):

```bash
.venv/bin/python lunge_pose_demo/build_frame_viewer.py lunge_pose_demo/lunge_frame.jpg
.venv/bin/python lunge_pose_demo/build_sequence_viewer.py lunge_pose_demo/lunge.webm
.venv/bin/python lunge_pose_demo/build_game.py
```

On Windows use `.venv\Scripts\python` in place of `.venv/bin/python` (and `python` for
`python3`); see `../README.md` for the CUDA install and the `--device` / `--half` flags, which
the first two accept (`build_game.py` never touches the model).

`build_frame_viewer.py` takes a few seconds. `build_sequence_viewer.py` re-runs inference on
every frame — at the default 384px on CPU that's roughly 3.5 minutes for this clip (~0.8s/frame);
on a GPU with FP16 it is seconds. `--label` keeps a GPU re-run from clobbering the committed
CPU output, so the two can be compared:

```bash
.venv/bin/python lunge_pose_demo/build_sequence_viewer.py lunge_pose_demo/lunge.webm --label lunge_gpu
```

If OpenCV opens the `.webm` but reports zero frames (its bundled FFmpeg has to decode VP9),
transcode to mp4 first, or decode with the `av` package that is already in the venv.
Both viewers also accept `--threshold` and `--resolution`.

`build_game.py` runs in well under a second and needs nothing from the venv — plain `python3`
works, since it only reads the JSON the sequence builder already wrote. Its clip windows are
flags, so they can be retuned without editing code:

```bash
python3 lunge_pose_demo/build_game.py --idle 79:106 --lunge 106:134,154:188 --speed 1.4
```

`--hold` sets how many frames the deepest pose is held for, `--smooth` the moving-average window
over the joint tracks, `--scale` overrides the automatic source-px → game-px factor (the figure
resizes; the arena tuning doesn't), and `--out` sets the output path. The `--idle` window does
double duty: it is the standing loop *and* the window the scale and the bone lengths are
measured on, so it wants a stretch where the subject is still and square to the lens. It prints
the scale, the lunge's measured travel, and each clip's frame count, duration and source window
on the way out.

### Playing it

`←`/`→` (or `A`/`D`) move, `Space` jumps, `F` performs the captured lunge, `C` performs your own
recorded move (once you've recorded one — see below), `P` pauses and `R` restarts. The hitbox is
read off the measured pose — circles at the lead ankle and lead wrist, live only during the
move's strike window — and it sweeps forward with the step-out, putting the lead hand about
350 px ahead of where you pressed. Scoring means being in range *and* timing the press.
Hits build a combo; letting a target drift off the far edge resets it. Rounds last 60 seconds.

### Recording your own move

Opening `lunge_game.html` directly (`file://`) still plays `F` and any move already baked in or
cached in the browser, but recording a new one needs a local model to analyze it against:

```bash
.venv/bin/python lunge_pose_demo/serve_game.py --resolution 384
# Windows:  .venv\Scripts\python lunge_pose_demo\serve_game.py --resolution 384
```

The startup banner and `http://127.0.0.1:8000/health` both report which device and precision
the model ended up on — on a GPU box expect `cuda` and `fp16`, and treat `cpu` as an install
problem rather than a slow machine. Then open `http://127.0.0.1:8000/` and click
**Record / upload a move**. Record up to 4 seconds
from a webcam (with a live MediaPipe skeleton overlay — preview only, not what gets measured) or
upload a video and pick a 4-second window from it. Each frame is sent to the same RF-DETR model,
threshold and tracker the lunge itself was built from; the result is trimmed and previewed before
saving.

What's measured, exactly as reported by the model, no different from the lunge: every joint
position in every frame, the strike window (auto-detected from where a lead wrist or ankle
reaches furthest, adjustable by hand in the trim step), and the hips' own frame-by-frame travel.
What's altered is proportions only — each bone (spine, upper arm, forearm, thigh, shin, ...) is
scaled by one constant, `lunge subject's bone length ÷ your bone length`, so your move plays out
on the lunge character's body. Direction and foreshortening are never touched: a limb measured at
60% projection in the recording still draws at 60%. Nothing is solved from angles, and the
retargeting has to run before the strike window and facing direction are picked — differently
proportioned bones can genuinely change which way a reach points, so both are read off the same
retargeted frames rather than off the raw recording.

Saving writes to `custom_moves.json` next to `serve_game.py` (the move's real home — it survives
a cleared browser or a different machine) and mirrors to the browser's `localStorage` as an
offline fallback; either way `C` works immediately, no reload. To ship a move with no server at
all, bake it into the HTML:

```bash
python3 lunge_pose_demo/build_game.py --custom lunge_pose_demo/custom_moves.json
```
