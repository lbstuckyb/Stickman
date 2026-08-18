# Lunge pose demo

Everything from one session of poking at `RFDETRKeypointPreview` (see `../README.md` for what
that model is) using `../lunge.webm` as the source clip. Two questions got answered here: what
the model's raw output looks like on one frame, and what it looks like across a whole clip.

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
| `build_frame_viewer.py` | reusable: image → annotated jpg + json + interactive viewer html | — |
| `build_sequence_viewer.py` | reusable: video → raw json + interactive scrubber html | — |

The two `build_*.py` scripts are the actual source of the two HTML viewers — everything in them
(palette, layout, the "no video underneath, just the raw points and the fixed skeleton edge
list" framing) is inlined in the `.html` files already, so the HTML files stand alone. The
scripts exist so this is reproducible with a different image/video/threshold rather than a
one-off.

## Opening this again later

**The two HTML files are fully self-contained** — the keypoint data is inlined as JSON in a
`<script>` tag, there's no separate server or image to load. Just double-click either `.html`
file (or drag it into a browser tab / `open lunge_keypoints_scrubber.html`). The only external
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
```

`build_frame_viewer.py` takes a few seconds. `build_sequence_viewer.py` re-runs inference on
every frame — at the default 384px on CPU that's roughly 3.5 minutes for this clip (~0.8s/frame).
Both accept `--threshold` and `--resolution`; `build_sequence_viewer.py` also takes `--label` to
change the output filename prefix if you point it at a different video.
