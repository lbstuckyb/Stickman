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

`build_frame_viewer.py` takes a few seconds. `build_sequence_viewer.py` re-runs inference on
every frame — at the default 384px on CPU that's roughly 3.5 minutes for this clip (~0.8s/frame).
Both accept `--threshold` and `--resolution`; `build_sequence_viewer.py` also takes `--label` to
change the output filename prefix if you point it at a different video.

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

`←`/`→` (or `A`/`D`) move, `Space` jumps, `F` performs the captured lunge, `P` pauses and `R`
restarts. The hitbox is read off the measured pose — circles at the lead ankle and lead wrist,
live only during the lunge's strike window — and it sweeps forward with the step-out, putting
the lead hand about 350 px ahead of where you pressed. Scoring means being in range *and*
timing the press.
Hits build a combo; letting a target drift off the far edge resets it. Rounds last 60 seconds.
