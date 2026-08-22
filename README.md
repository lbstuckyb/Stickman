# Stickman Challenge

A browser arena game whose stick figure *is* a real captured motion clip: every walk, jump
and lunge is driven by joint positions [RF-DETR Keypoint Preview](https://blog.roboflow.com/real-time-keypoint-detection-with-rf-detr/)
measured off actual video, not hand-authored keyframes. RF-DETR is a PyTorch model, so it
can't run in the page — a small local server runs it, and the page keeps the camera, the
retargeting, and the rig.

```bash
.venv/bin/python game/serve_game.py --resolution 384
# then open http://127.0.0.1:8000/
```

## Layout

| dir | what's in it |
| --- | --- |
| [`game/`](game/README.md) | **the playable thing.** Serve it, play it, record your own moves. |
| [`pipeline/`](pipeline/PIPELINE.md) | the capture pipeline that turned `lunge.webm` into `game/data/lunge_all_frames.json` — the raw single-frame and full-clip keypoint viewers, and the write-up of every step. |
| [`research/`](research/README.md) | where this started: the RF-DETR-vs-MediaPipe stick-rig comparison that the game's retargeting math (`readAngles()`/`solve()`) was ported from. |

Root-level: `rfdetr_pose.py` (the model wrapper shared by all three), `check_gpu.py`
(torch-only GPU check), `requirements.txt`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The 156 MB checkpoint downloads itself on first use, to `~/.roboflow/models/`. Commands below
are POSIX; on Windows swap `.venv/bin/python` for `.venv\Scripts\python`, and start from the
quickstart rather than from here. **Run everything from the repo root** — every script here
defaults to repo-root-relative paths.

## Windows quickstart (GPU)

Written against the RTX 4070 Ti box, but nothing in it is specific to that card. Run everything
from the repo root — several scripts default to relative paths — and call
`.venv\Scripts\python` directly rather than activating the venv; that works identically in
`cmd` and PowerShell.

**1. Prerequisites.** An NVIDIA driver (`nvidia-smi` should print the card — the CUDA Toolkit is
*not* needed, the torch wheels bundle their own), Python 3.12 (`py -0` lists what you have;
3.10–3.13 all work), and git.

**2. Get the code.**

```bat
git clone git@github.com:lbstuckyb/Stickman.git
cd Stickman
git checkout measured-lunge-replay
```

**3. Create the venv and install — order matters.**

```bat
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
.venv\Scripts\python -m pip install -r requirements.txt
```

Torch has to come first, and from that index. Plain `pip install torch` on Windows resolves a
**CPU-only** wheel that imports and runs perfectly well — just thirty times slower, with nothing
in the output to say why. `requirements.txt` is no defence: it pins a version, not the index the
wheel came from, and both `rfdetr` and the pin are satisfied either way. The `cu129` tag moves;
take the current one from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), any cu12x/cu13x
build supports Ada. Don't copy a `.venv` from another machine — `pyvenv.cfg` hardcodes the
interpreter path.

**4. Confirm the GPU before anything slow.**

```bat
.venv\Scripts\python check_gpu.py
```

Want: `NVIDIA GeForce RTX 4070 Ti, 12.0 GB, compute 8.9`, a non-null CUDA build, and FP16 several
times faster than FP32. `FAIL` names which of the two problems you have. This imports torch and
nothing else — no rfdetr, no checkpoint — so a bad install fails in two seconds.

**5. First real run** (this is what downloads the 156 MB checkpoint, to
`%USERPROFILE%\.roboflow\models\`).

```bat
.venv\Scripts\python research\test_rfdetr_keypoints.py research\sample_people.jpg --runs 20 --resolution 384
```

Want `RFDETRKeypointPreview @ 384px on cuda (fp16)` and `PASS`. Against the CPU baseline of
408 ms, that number is the whole point of the exercise. For the table below, repeat at 288 and
576, and add `--no-half` and `--optimize` runs.

**6. The game and the move recorder.**

```bat
.venv\Scripts\python game\serve_game.py --resolution 576
```

576 is the checkpoint's native resolution — CPU couldn't afford it, the GPU can. The banner must
say `cuda` and `fp16`; `cpu` there is an install problem, not a slow machine (`/health` reports
the same thing). Open `http://127.0.0.1:8000/`, allow the camera, press `F` for the baked lunge,
then **Record / upload a move**, and `C` plays yours back. Windows will raise a firewall prompt —
the server binds `127.0.0.1` only, so declining public access is correct. To bake a recorded move
into the standalone HTML:

```bat
.venv\Scripts\python game\build_game.py --custom game\custom_moves.json
```

**7. Re-process the video on the GPU.**

```bat
.venv\Scripts\python pipeline\build_sequence_viewer.py game\data\lunge.webm --label lunge_gpu
```

264 frames: ~3.5 minutes on CPU, seconds here. `--label` keeps the committed CPU output intact
for comparison. If OpenCV opens the file but reports zero frames, its bundled FFmpeg didn't get
VP9 — transcode to mp4 first, or decode with the `av` package already in the venv.

Optionally `node research\test_retarget.mjs research\payload.json` and `node research\test_mirror.mjs`,
which check the browser-side math without a browser.

### When it goes wrong

| symptom | cause |
| --- | --- |
| banner says `cpu`, `check_gpu.py` reports a CPU-only build | plain-PyPI torch — reinstall with `--force-reinstall` from the CUDA index |
| `check_gpu.py` reports a CUDA build but no visible GPU | driver problem, not Python — start with `nvidia-smi` |
| `port 8000 in use` | `research/serve_rfdetr.py` and `game/serve_game.py` share the default; pass `--port 8001` |
| `cannot read sample_people.jpg` / `cannot read lunge_frame.jpg` | run from the repo root; the scripts use repo-root-relative defaults |

`--device cpu` on any command forces the old path, for a same-machine A/B.

## Device and precision flags

Every entry point (`research/test_rfdetr_keypoints.py`, `research/serve_rfdetr.py`,
`game/serve_game.py`, `pipeline/build_frame_viewer.py`, `pipeline/build_sequence_viewer.py`)
takes the same device flags:

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

## Notes

- **Resolution must be divisible by 24** for this checkpoint (`patch_size 12 × 2 windows`),
  not the 56 that other RF-DETR variants want. 576 is the default; 384 and 288 are the
  useful faster steps.
- **Feed it RGB.** OpenCV hands you BGR; passing that through costs a little confidence and
  shifts joints by a few pixels. Every server and script here converts.
- **COCO-17 has no head-top point**, so the rig takes head direction from the ear midpoint
  and falls back to the nose.
- **Mirroring is two operations.** Flipping landmark x is only half of it — each joint is
  still labelled with the body side it came from, so the right-arm chain starts at the rig's
  right shoulder and reaches the wrong way, dragging the arm across the chest. See
  [`research/README.md`](research/README.md) for the details and the guard test.
- **Uncertainty is the interesting output.** Each joint comes with a 2×2 covariance, so a
  joint the model is unsure about is a wide ellipse rather than a confidently wrong point —
  see the annotated images in `research/` and `pipeline/`.
