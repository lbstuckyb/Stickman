"""RF-DETR keypoint preview — model wrapper shared by the test script and the server.

The model is https://blog.roboflow.com/real-time-keypoint-detection-with-rf-detr/
It returns a supervision `KeyPoints` object:

    xy                       (n, 17, 2)  pixel coords in the source image
    keypoint_confidence      (n, 17)     per-joint "findable" confidence
    visible                  (n, 17)     per-joint visible (True) vs occluded (False)
    detection_confidence     (n,)        per-person score
    data["xyxy"]             (n, 4)      person box
    data["covariance"]       (n, 17, 2, 2)  per-joint 2-D uncertainty
    data["source_shape"]     (n, 2)      (height, width) of the source image
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

# COCO-17, the schema the preview checkpoint is trained on.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton used for drawing and for the stick rig, as index pairs.
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
]


@dataclass
class Timing:
    load_s: float
    warmup_s: float


CPU_WHEEL_FIX = (
    "this torch has no CUDA build. Reinstall from the PyTorch CUDA index:\n"
    "    pip install --force-reinstall torch torchvision "
    "--index-url https://download.pytorch.org/whl/cu129\n"
    "    (take the current cuXXX tag from https://pytorch.org/get-started/locally/)"
)


def resolve_device(requested: str | None = "auto") -> str:
    """'auto' picks the best accelerator present; anything else is validated and passed through."""
    import torch

    if requested in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    return str(torch.device(requested))     # raises on nonsense like "gpu"


def device_report(device: str | None = None) -> dict:
    """What torch we are on and what it can see.

    `cpu_only_wheel` is the one that matters on Windows: pip's default index serves a
    CPU-only torch there, and that wheel imports and runs perfectly well - just thirty times
    slower, with nothing in the output to say why. It reports `torch.version.cuda is None`.
    """
    import torch

    dev = torch.device(device) if device else torch.device(resolve_device("auto"))
    report = {
        "device": str(dev),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cpu_only_wheel": torch.version.cuda is None,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": None,
    }
    if report["cuda_available"]:
        index = dev.index if (dev.type == "cuda" and dev.index is not None) else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        report.update({
            "gpu": props.name,
            "capability": f"{props.major}.{props.minor}",
            "vram_gb": round(props.total_memory / 1024 ** 3, 1),
            "cudnn": torch.backends.cudnn.version(),
        })
    return report


def no_cuda_explanation(report: dict) -> str:
    """Why is there no GPU? The two answers need different fixes, and on Windows the first
    one is by far the more common - a CPU-only wheel runs fine, just thirty times slower."""
    if report["cpu_only_wheel"]:
        return CPU_WHEEL_FIX
    return ("torch has a CUDA build but no GPU is visible - check the NVIDIA driver "
            "with nvidia-smi.")


def format_device_report(report: dict) -> str:
    """One ASCII line for a startup banner - Windows consoles are not reliably UTF-8. It leads
    with the device actually in use, so a GPU sitting idle next to a CPU run reads as such."""
    build = f"CUDA {report['torch_cuda_build']}" if report["torch_cuda_build"] else "CPU-only build"
    torch_bit = f"torch {report['torch']} ({build})"
    if report["gpu"] is None:
        return f"{report['device']} - no GPU visible, {torch_bit}"
    gpu = (f"{report['gpu']}, {report['vram_gb']} GB, compute {report['capability']}, "
           f"cuDNN {report['cudnn']}")
    if report["device"].startswith("cuda"):
        return f"{report['device']} - {gpu}, {torch_bit}"
    return f"{report['device']} - not using the GPU present ({gpu}), {torch_bit}"


def announce_device(requested: str = "auto", label: str = "device") -> dict:
    """Print the device line every entry point starts with, and flag the two ways a GPU box
    silently ends up on the CPU. Returns the report so callers can reuse it (e.g. /health)."""
    report = device_report(resolve_device(requested))
    print(f"{label:<11}: {format_device_report(report)}")
    if requested != "cpu" and not report["cuda_available"]:
        print(f"WARNING    : {no_cuda_explanation(report)}")
    return report


def add_device_args(ap) -> None:
    """The device/precision flags every entry point takes, so they stay identical."""
    ap.add_argument("--device", default="auto",
                    help="auto (default), cpu, cuda, or cuda:N")
    ap.add_argument("--half", dest="half", action="store_true", default=None,
                    help="FP16 tensor cores - the default on CUDA, never used on CPU")
    ap.add_argument("--no-half", dest="half", action="store_false",
                    help="force FP32 on the GPU (slower; use it to compare against CPU output)")
    ap.add_argument("--optimize", action="store_true",
                    help="also torch.jit.trace the model - slow first call, faster afterwards")


class PoseModel:
    """Thin wrapper: load once, predict on numpy/PIL/path, emit normalized JSON."""

    def __init__(self, resolution: int | None = None, device: str | None = "auto",
                 half: bool | None = None, optimize: bool = False):
        import torch
        from rfdetr import RFDETRKeypointPreview  # imported late; torch load is slow

        dev = resolve_device(device)
        cuda = dev.startswith("cuda")
        if cuda and not torch.cuda.is_available():
            raise RuntimeError(f"asked for {dev}: "
                               f"{no_cuda_explanation(device_report('cpu'))}")

        # FP16 is most of the win on tensor cores and a loss on CPU, so it follows the
        # device unless asked otherwise.
        if half is None:
            half = cuda
        elif half and not cuda:
            print(f"warning    : ignoring --half on {dev}; FP16 is slower than FP32 there")
            half = False

        if cuda:
            # The input shape never changes after startup, so cuDNN's autotune pays for
            # itself on the first frame; TF32 is free accuracy-for-speed on the FP32 path.
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

        t0 = time.perf_counter()
        kwargs: dict = {"device": dev}
        if resolution is not None:
            kwargs["resolution"] = resolution
        self.model = RFDETRKeypointPreview(**kwargs)
        if half or optimize:
            # rfdetr renamed optimize_for_inference() -> inference() in 1.9 and drops the old
            # name in 1.11, so bind whichever exists. `compile` here is torch.jit.trace, not
            # torch.compile: no C++ toolchain needed, which is what makes it usable on Windows.
            tune = getattr(self.model, "inference", None) or self.model.optimize_for_inference
            tune(compile=optimize, batch_size=1,
                 dtype=torch.float16 if half else torch.float32)
        load_s = time.perf_counter() - t0

        self.resolution = getattr(self.model.model, "resolution", resolution or 576)
        self.device = str(getattr(self.model.model, "device", dev))
        self.half = bool(half)
        self.optimize = bool(optimize)
        self.report = device_report(self.device)
        self.gpu = self.report["gpu"]
        self._torch = torch
        self._sync = cuda

        # First call builds kernels/graphs; keep it out of the measured numbers.
        t0 = time.perf_counter()
        self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8), threshold=0.9)
        if self._sync:
            torch.cuda.synchronize()
        warmup_s = time.perf_counter() - t0
        self.timing = Timing(load_s=load_s, warmup_s=warmup_s)

    def summary(self) -> str:
        """`RFDETRKeypointPreview @ 384px on cuda (fp16)` - what every entry point prints."""
        flags = "fp16" if self.half else "fp32"
        if self.optimize:
            flags += ", jit"
        return f"RFDETRKeypointPreview @ {self.resolution}px on {self.device} ({flags})"

    def predict(self, image, threshold: float = 0.5):
        """Returns (KeyPoints, inference_ms). numpy input must be RGB, not cv2's BGR -
        feeding BGR still works but costs a little accuracy and shifts joints by a few px."""
        t0 = time.perf_counter()
        kp = self.model.predict(image, threshold=threshold, include_source_image=False)
        if self._sync:
            self._torch.cuda.synchronize()   # CUDA kernels are async; time them, not the launch
        return kp, (time.perf_counter() - t0) * 1000.0

    def predict_payload(self, image, threshold: float = 0.5, width: int | None = None,
                        height: int | None = None) -> dict:
        """Normalized (0..1) keypoints, people sorted by score - what the browser eats."""
        kp, ms = self.predict(image, threshold=threshold)
        if width is None or height is None:
            if len(kp) and "source_shape" in kp.data:
                height, width = (int(v) for v in kp.data["source_shape"][0])
            else:
                height, width = image.shape[:2] if hasattr(image, "shape") else (1, 1)

        people = []
        for i in range(len(kp)):
            xy = kp.xy[i]
            conf = kp.keypoint_confidence[i]
            vis = kp.visible[i] if kp.visible is not None else np.ones(len(xy), dtype=bool)
            people.append({
                "score": float(kp.detection_confidence[i]),
                "box": [float(v) for v in kp.data["xyxy"][i] / np.array([width, height, width, height])],
                "keypoints": [
                    {
                        "x": float(x) / width,
                        "y": float(y) / height,
                        "v": float(c),                 # findable confidence
                        "occluded": not bool(vv),      # visible flag from the model
                    }
                    for (x, y), c, vv in zip(xy, conf, vis)
                ],
            })
        people.sort(key=lambda p: p["score"], reverse=True)
        return {"people": people, "inference_ms": round(ms, 1),
                "resolution": self.resolution, "device": self.device,
                "half": self.half, "gpu": self.gpu}


def covariance_ellipse(cov: np.ndarray, n_sigma: float = 1.0):
    """(cov 2x2) -> (half_axis_a, half_axis_b, angle_degrees) for drawing."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-6, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    a, b = n_sigma * np.sqrt(vals)
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    return float(a), float(b), angle


_TEAL = (197, 209, 79)      # BGR
_AMBER = (68, 181, 242)
_DIM = (150, 130, 108)


def annotate(image: np.ndarray, kp) -> np.ndarray:
    """Draw the skeleton + per-joint 1-sigma uncertainty ellipse over a BGR frame."""
    out = image.copy()
    for i in range(len(kp)):
        xy = kp.xy[i].astype(int)
        conf = kp.keypoint_confidence[i]
        vis = kp.visible[i] if kp.visible is not None else np.ones(len(xy), bool)
        cov = kp.data["covariance"][i]

        x1, y1, x2, y2 = kp.data["xyxy"][i].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), _DIM, 1)
        cv2.putText(out, f"{kp.detection_confidence[i]:.2f}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _DIM, 1, cv2.LINE_AA)

        for a, b in SKELETON:
            if conf[a] < 0.3 or conf[b] < 0.3:
                continue
            cv2.line(out, tuple(xy[a]), tuple(xy[b]), _TEAL, 2, cv2.LINE_AA)

        for j, (x, y) in enumerate(xy):
            if conf[j] < 0.3:
                continue
            # 1-sigma ellipse: thick where the model calls the joint visible,
            # thin where it thinks the joint is occluded but still findable.
            ax_a, ax_b, angle = covariance_ellipse(cov[j], n_sigma=1.0)
            cv2.ellipse(out, (int(x), int(y)), (max(1, int(ax_a)), max(1, int(ax_b))),
                        angle, 0, 360, _AMBER, 2 if vis[j] else 1, cv2.LINE_AA)
            cv2.circle(out, (int(x), int(y)), 3, _AMBER, -1, cv2.LINE_AA)
    return out
