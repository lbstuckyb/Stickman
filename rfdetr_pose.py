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


class PoseModel:
    """Thin wrapper: load once, predict on numpy/PIL/path, emit normalized JSON."""

    def __init__(self, resolution: int | None = None, device: str | None = None,
                 half: bool = False, optimize: bool = False):
        from rfdetr import RFDETRKeypointPreview  # imported late; torch load is slow

        t0 = time.perf_counter()
        kwargs: dict = {}
        if resolution is not None:
            kwargs["resolution"] = resolution
        if device is not None:
            kwargs["device"] = device
        self.model = RFDETRKeypointPreview(**kwargs)
        if optimize:
            import torch
            self.model.optimize_for_inference(dtype=torch.float16 if half else torch.float32)
        load_s = time.perf_counter() - t0

        self.resolution = getattr(self.model.model, "resolution", resolution or 576)
        self.device = str(getattr(self.model.model, "device", "cpu"))

        # First call builds kernels/graphs; keep it out of the measured numbers.
        t0 = time.perf_counter()
        self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8), threshold=0.9)
        warmup_s = time.perf_counter() - t0
        self.timing = Timing(load_s=load_s, warmup_s=warmup_s)

    def predict(self, image, threshold: float = 0.5):
        """Returns (KeyPoints, inference_ms). numpy input must be RGB, not cv2's BGR —
        feeding BGR still works but costs a little accuracy and shifts joints by a few px."""
        t0 = time.perf_counter()
        kp = self.model.predict(image, threshold=threshold, include_source_image=False)
        return kp, (time.perf_counter() - t0) * 1000.0

    def predict_payload(self, image, threshold: float = 0.5, width: int | None = None,
                        height: int | None = None) -> dict:
        """Normalized (0..1) keypoints, people sorted by score — what the browser eats."""
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
                "resolution": self.resolution, "device": self.device}


def covariance_ellipse(cov: np.ndarray, n_sigma: float = 1.0):
    """(cov 2x2) -> (half_axis_a, half_axis_b, angle_degrees) for drawing."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-6, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    a, b = n_sigma * np.sqrt(vals)
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    return float(a), float(b), angle
