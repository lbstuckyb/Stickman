# RF-DETR keypoints → stick rig

`pose-stick-figure.html` drives a fixed-proportion stick rig from pose landmarks. It
originally read those landmarks from MediaPipe's `PoseLandmarker` running in the browser.
This adds a second build that reads them from
[RF-DETR Keypoint Preview](https://blog.roboflow.com/real-time-keypoint-detection-with-rf-detr/)
instead.

RF-DETR is a PyTorch model, so it can't run in the page. The model lives in a small local
server; the page keeps the camera, the retargeting, and the rig.

| file | what it is |
| --- | --- |
| `pose-stick-figure.html` | the original, unchanged — MediaPipe, 33 landmarks, all in-browser |
| `pose-stick-figure-rfdetr.html` | same rig, COCO-17 joints, keypoints fetched from the server |
| `rfdetr_pose.py` | model wrapper: load once, predict, normalize to JSON |
| `serve_rfdetr.py` | localhost server — serves the page, answers `POST /keypoints` |
| `test_rfdetr_keypoints.py` | smoke test + latency benchmark, writes an annotated image |
| `test_retarget.mjs` | runs the page's own `readAngles()`/`solve()` over a saved payload (`payload.json`) |
| `test_mirror.mjs` | guards the left/right mirroring — raise an arm, the rig raises the matching one |
| `lunge_pose_demo/` | standalone HTML viewers of raw keypoint output (single frame + full-clip scrubber) — see its own README |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install rfdetr supervision opencv-python-headless
```

The 156 MB checkpoint downloads itself on first use, to `~/.roboflow/models/`.

## Test

```bash
.venv/bin/python test_rfdetr_keypoints.py sample_people.jpg --runs 5
```

Prints every joint with its confidence, visible/occluded flag, and positional sigma, then
writes `rfdetr_keypoints_out.jpg` (skeleton + 1σ uncertainty ellipse per joint) and
`rfdetr_keypoints_out.json`.

```bash
node test_retarget.mjs payload.json
```

Checks the browser side without a browser: it pulls `readAngles()` and `solve()` out of the
HTML and runs them over a saved server payload, so a wrong COCO index shows up as a spine
that doesn't point up rather than as a weird-looking puppet.

## Live demo

```bash
.venv/bin/python serve_rfdetr.py --resolution 384    # then open http://127.0.0.1:8000/
```

Pick **Camera** or a video file. The header shows real inference latency and achieved fps.
The **Send** control sets the width frames are downscaled to before upload — it trims
upload and JPEG cost, separate from `--resolution`, which is what the model actually runs at.

## Notes

- **Resolution must be divisible by 24** for this checkpoint (`patch_size 12 × 2 windows`),
  not the 56 that other RF-DETR variants want. 576 is the default; 384 and 288 are the
  useful faster steps.
- **Feed it RGB.** OpenCV hands you BGR; passing that through costs a little confidence and
  shifts joints by a few pixels. `serve_rfdetr.py` and the test both convert.
- **COCO-17 has no head-top point**, so the rig takes head direction from the ear midpoint
  and falls back to the nose. Everything else the rig needs maps 1:1 from COCO.
- **Mirroring is two operations.** Flipping landmark x is only half of it — each joint is
  still labelled with the body side it came from, so the right-arm chain starts at the rig's
  right shoulder and reaches the wrong way, dragging the arm across the chest. `mirrorLandmarks()`
  flips x *and* swaps the paired indices. `test_mirror.mjs` pins this down.
- **Uncertainty is the interesting output.** Each joint comes with a 2×2 covariance, so a
  joint the model is unsure about is a wide ellipse rather than a confidently wrong point.
  The rig only uses the confidence scalar today; the covariance is a better smoothing input
  if you want to go further.

## Measured here (CPU, no GPU in this box)

| resolution | median latency | people found in `sample_people.jpg` |
| --- | --- | --- |
| 288 | 322 ms | 3 |
| 384 | 408 ms | 3 |
| 480 | 600 ms | 3 |
| 576 | 880 ms | 3 |

For contrast, the blog quotes 9.8 ms at 576×576 on a T4 with TensorRT FP16, at 71.8 keypoint
AP on COCO val2017. The gap here is entirely CPU-vs-accelerator: same weights, no TensorRT,
no FP16. On CPU the live demo runs at roughly 2–3 fps, which the One Euro filter smooths into
something watchable but not real time.
