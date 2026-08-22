# RF-DETR keypoints → stick rig

Where the [Stickman Challenge](../README.md) game started: a fixed-proportion stick rig driven
from pose landmarks. `pose-stick-figure.html` originally read those landmarks from MediaPipe's
`PoseLandmarker` running in the browser. This directory adds a second build that reads them from
[RF-DETR Keypoint Preview](https://blog.roboflow.com/real-time-keypoint-detection-with-rf-detr/)
instead — the same retargeting math (`readAngles()`/`solve()`) was later ported into
[`../game/build_game.py`](../game/build_game.py) to drive the game's figure.

RF-DETR is a PyTorch model, so it can't run in the page. The model lives in a small local
server; the page keeps the camera, the retargeting, and the rig.

| file | what it is |
| --- | --- |
| `pose-stick-figure.html` | the original, unchanged — MediaPipe, 33 landmarks, all in-browser |
| `pose-stick-figure-rfdetr.html` | same rig, COCO-17 joints, keypoints fetched from the server |
| `serve_rfdetr.py` | localhost server — serves the page, answers `POST /keypoints` |
| `test_rfdetr_keypoints.py` | smoke test + latency benchmark, writes an annotated image |
| `test_retarget.mjs` | runs the page's own `readAngles()`/`solve()` over a saved payload (`payload.json`) |
| `test_mirror.mjs` | guards the left/right mirroring — raise an arm, the rig raises the matching one |
| `sample_people.jpg`, `payload.json` | fixtures for the two tests above |
| `rfdetr_keypoints_out.jpg` / `.json` | output of `test_rfdetr_keypoints.py` on `sample_people.jpg` |

The model wrapper (`PoseModel`, device/precision flags, `annotate()`) lives in
[`../rfdetr_pose.py`](../rfdetr_pose.py), shared with `pipeline/` and `game/`. See
[`../README.md`](../README.md) for setup, the Windows GPU quickstart, and the device flags every
entry point here takes.

## Test

```bash
.venv/bin/python research/test_rfdetr_keypoints.py research/sample_people.jpg --runs 5
```

Prints every joint with its confidence, visible/occluded flag, and positional sigma, then
writes `rfdetr_keypoints_out.jpg` (skeleton + 1σ uncertainty ellipse per joint) and
`rfdetr_keypoints_out.json` next to it.

```bash
node research/test_retarget.mjs research/payload.json
```

Checks the browser side without a browser: it pulls `readAngles()` and `solve()` out of the
HTML and runs them over a saved server payload, so a wrong COCO index shows up as a spine
that doesn't point up rather than as a weird-looking puppet.

## Live demo

```bash
.venv/bin/python research/serve_rfdetr.py --resolution 384    # then open http://127.0.0.1:8000/
```

Pick **Camera** or a video file. The header shows real inference latency and achieved fps.
The **Send** control sets the width frames are downscaled to before upload — it trims
upload and JPEG cost, separate from `--resolution`, which is what the model actually runs at.

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

(Those counts were taken with the package versions of the day; a later `rfdetr` finds 2 people
in `sample_people.jpg` at 288px, not 3. The latencies still hold.)

## Measured on the RTX 4070 Ti

To fill in from the Windows box — `--runs 20` at each resolution, FP16 and FP32:

| resolution | CPU | GPU FP32 | GPU FP16 | GPU FP16 + `--optimize` |
| --- | --- | --- | --- | --- |
| 288 | 322 ms | | | |
| 384 | 408 ms | | | |
| 576 | 880 ms | | | |

Step 5 of the [Windows quickstart](../README.md#windows-quickstart-gpu) is the FP16 column.
For the FP32 one:

```bat
.venv\Scripts\python research\test_rfdetr_keypoints.py research\sample_people.jpg --runs 20 --resolution 384 --no-half --out research\gpu_fp32.jpg
```

`--out` also names the JSON beside it, so an FP16 and an FP32 run can be diffed rather than
overwriting each other. Expect them to agree to about a pixel; the per-joint covariance
ellipses in the annotated JPEG are the output most likely to look wrong if FP16 misbehaves.
