# Stickman Challenge — the game

A small arena game where the measured lunge *is* the attack animation, driven by real captured
keypoints rather than by hand-authored keyframes. See [`../README.md`](../README.md) for setup
and the Windows GPU quickstart, and [`../pipeline/PIPELINE.md`](../pipeline/PIPELINE.md) for how
`data/lunge_all_frames.json` was captured in the first place.

## Files

| file | what it is | made by |
| --- | --- | --- |
| `lunge_game.html` | **playable**: stick figure that *is* the captured skeleton — move, jump, and lunge at drifting targets | `build_game.py` |
| `build_game.py` | reusable: per-frame keypoint json → clips + game html (no model inference) | — |
| `serve_game.py` | local server: serves `lunge_game.html`, runs RF-DETR on recorded/uploaded video for the in-page recorder, reads/writes `custom_moves.json` | — |
| `data/lunge.webm` | source clip for the captured lunge | — |
| `data/lunge_all_frames.json` | raw model output for all 264 frames of `lunge.webm` | `../pipeline/build_sequence_viewer.py` |
| `custom_moves.json` | recorded moves saved from the page (gitignored, written at runtime) | `serve_game.py` |

`build_game.py` is the source of `lunge_game.html` — everything in it (palette, layout, the rig)
is inlined in the HTML already, so the HTML file stands alone. The script exists so this is
reproducible with different clip windows or a different capture.

`build_game.py` never touches the model. It reads `data/lunge_all_frames.json` and cuts three
clips out of the track — `idle` (source frames 79–106, the ready stance held before the lunge,
ping-ponged), `lunge` (frames 106–134, a short hold at the deepest pose, then 154–188, with a
strike window over the deep extension), and `walk`.

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
  instead of taking `max(score)` per frame, which used to snap the limbs mid-recovery. See
  [`../pipeline/PIPELINE.md`](../pipeline/PIPELINE.md) for the full write-up.

## Opening this again later

**`lunge_game.html` is fully self-contained** — the keypoint data is inlined as JSON in a
`<script>` tag, there's no separate server or image to load. Just double-click the file (or
drag it into a browser tab / `open game/lunge_game.html`). The only external call is a Google
Fonts stylesheet; without internet it falls back to the system sans-serif and still works fine.

## Regenerating

From the repo root, using the project's existing `.venv` (see [`../README.md`](../README.md)
for setup):

```bash
.venv/bin/python game/build_game.py
```

On Windows use `.venv\Scripts\python` in place of `.venv/bin/python`; see
[`../README.md`](../README.md) for the CUDA install and the `--device` / `--half` flags (not
needed here — `build_game.py` never touches the model).

`build_game.py` runs in well under a second and needs nothing from the venv — plain `python3`
works, since it only reads the JSON the sequence builder already wrote. Its clip windows are
flags, so they can be retuned without editing code:

```bash
python3 game/build_game.py --idle 79:106 --lunge 106:134,154:188 --speed 1.4
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
.venv/bin/python game/serve_game.py --resolution 384
# Windows:  .venv\Scripts\python game\serve_game.py --resolution 384
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
python3 game/build_game.py --custom game/custom_moves.json
```
