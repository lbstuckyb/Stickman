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
| `check_gpu.py` | torch-only GPU check — run it first on a new machine, before anything loads the model |
| `test_retarget.mjs` | runs the page's own `readAngles()`/`solve()` over a saved payload (`payload.json`) |
| `test_mirror.mjs` | guards the left/right mirroring — raise an arm, the rig raises the matching one |
| `lunge_pose_demo/` | standalone HTML viewers of raw keypoint output (single frame + full-clip scrubber) — see its own README |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install rfdetr supervision opencv-python-headless
```

The 156 MB checkpoint downloads itself on first use, to `~/.roboflow/models/`.

### Windows, with a GPU

Install order matters. On Windows, pip's default index serves a **CPU-only** torch, and that
wheel imports and runs perfectly well — just thirty times slower, with nothing in the output to
say why. So install torch from the CUDA index *first*, then let `rfdetr` install on top of it:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
.venv\Scripts\python -m pip install rfdetr supervision opencv-python-headless
.venv\Scripts\python check_gpu.py
```

The `cu129` tag moves — take the current one from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). The check that
matters is `torch.version.cuda is not None`, which is exactly what `check_gpu.py` reports; it
also times an FP32 and an FP16 matmul so you can see the tensor cores actually working. Don't
copy a `.venv` across machines — `pyvenv.cfg` hardcodes the interpreter path.

Every command below is written for POSIX; on Windows swap `.venv/bin/python` for
`.venv\Scripts\python`.

### Device and precision flags

Every entry point (`test_rfdetr_keypoints.py`, `serve_rfdetr.py`, `lunge_pose_demo/serve_game.py`,
`lunge_pose_demo/build_frame_viewer.py`, `lunge_pose_demo/build_sequence_viewer.py`) takes the
same device flags:

| flag | default | what it does |
| --- | --- | --- |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:N`. `auto` picks CUDA when it's there |
| `--half` / `--no-half` | on under CUDA | FP16 tensor cores. Ignored on CPU, where FP16 is slower |
| `--optimize` | off | also `torch.jit.trace` the model — slow first call, faster after |

`--half` is where most of the GPU win is: it routes the model through rfdetr's
`inference(dtype=torch.float16)`, which is what Roboflow's own "~8x on a T4" note refers to.
Under CUDA the wrapper also turns on `cudnn.benchmark` (the input shape is fixed at startup) and
TF32 matmuls. `--optimize` is `torch.jit.trace`, **not** `torch.compile`, so it needs no C++
toolchain on Windows. Every startup banner and every `/health` response reports the device, the
precision and the GPU name, so a silent CPU fallback is visible without benchmarking for it.

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

(Those counts were taken with the package versions of the day; a later `rfdetr` finds 2 people
in `sample_people.jpg` at 288px, not 3. The latencies still hold.)

## Measured on the RTX 4070 Ti

To fill in from the Windows box — `--runs 20` at each resolution, FP16 and FP32:

| resolution | CPU | GPU FP32 | GPU FP16 | GPU FP16 + `--optimize` |
| --- | --- | --- | --- | --- |
| 288 | 322 ms | | | |
| 384 | 408 ms | | | |
| 576 | 880 ms | | | |

```bash
.venv\Scripts\python check_gpu.py
.venv\Scripts\python test_rfdetr_keypoints.py sample_people.jpg --runs 20 --resolution 384
.venv\Scripts\python test_rfdetr_keypoints.py sample_people.jpg --runs 20 --resolution 384 --no-half --out gpu_fp32.jpg
```

`--out` also names the JSON beside it, so an FP16 and an FP32 run can be diffed rather than
overwriting each other. Expect them to agree to about a pixel; the per-joint covariance
ellipses in the annotated JPEG are the output most likely to look wrong if FP16 misbehaves.
