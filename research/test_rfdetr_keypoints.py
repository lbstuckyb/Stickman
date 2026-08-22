"""Smoke test + benchmark for RF-DETR Keypoint Preview.

    .venv/bin/python research/test_rfdetr_keypoints.py [image.jpg] [--runs 5] [--resolution 576]

Checks the model loads, returns well-formed COCO-17 keypoints, and writes an
annotated image showing each joint's predicted uncertainty ellipse (the thing
that makes this model different from a heatmap pose model: it regresses a
full 2-D Gaussian per keypoint, not a single point).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root, for rfdetr_pose

from rfdetr_pose import (KEYPOINT_NAMES, PoseModel, add_device_args,  # noqa: E402
                         announce_device, annotate)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=str(HERE / "sample_people.jpg"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--resolution", type=int, default=None,
                    help="inference resolution; the preview checkpoint's default is 576")
    ap.add_argument("--out", default=str(HERE / "rfdetr_keypoints_out.jpg"))
    add_device_args(ap)
    args = ap.parse_args()
    out_json = Path(args.out).with_suffix(".json")   # so fp16/fp32 runs do not overwrite

    image = cv2.imread(args.image)
    if image is None:
        print(f"FAIL: cannot read {args.image}")
        return 1
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # the model wants RGB; cv2 hands back BGR
    h, w = image.shape[:2]
    print(f"image      : {args.image} ({w}x{h})")

    announce_device(args.device)
    model = PoseModel(resolution=args.resolution, device=args.device,
                      half=args.half, optimize=args.optimize)
    print(f"model      : {model.summary()}")
    print(f"load       : {model.timing.load_s:.1f}s   warmup: {model.timing.warmup_s*1000:.0f}ms")

    kp, _ = model.predict(rgb, threshold=args.threshold)
    n = len(kp)
    print(f"people     : {n} above threshold {args.threshold}")

    failures = []
    if n == 0:
        failures.append("no people detected")
    else:
        if kp.xy.shape[1:] != (17, 2):
            failures.append(f"expected (17,2) keypoints per person, got {kp.xy.shape[1:]}")
        if kp.data["covariance"].shape[1:] != (17, 2, 2):
            failures.append(f"expected per-keypoint 2x2 covariance, got {kp.data['covariance'].shape[1:]}")
        inside = ((kp.xy[..., 0] >= -w * 0.1) & (kp.xy[..., 0] <= w * 1.1) &
                  (kp.xy[..., 1] >= -h * 0.1) & (kp.xy[..., 1] <= h * 1.1))
        if not inside.all():
            failures.append(f"{(~inside).sum()} keypoints fall outside the image")
        if not (0 <= kp.detection_confidence.min() <= kp.detection_confidence.max() <= 1):
            failures.append("detection confidence outside [0,1]")

    if n:
        best = int(np.argmax(kp.detection_confidence))
        print(f"\nbest person (score {kp.detection_confidence[best]:.3f}):")
        conf = kp.keypoint_confidence[best]
        vis = kp.visible[best] if kp.visible is not None else np.ones(17, bool)
        for j, name in enumerate(KEYPOINT_NAMES):
            x, y = kp.xy[best][j]
            sig = np.sqrt(np.diag(kp.data["covariance"][best][j]))
            print(f"  {name:<15} ({x:7.1f},{y:7.1f})  conf {conf[j]:.2f}  "
                  f"{'visible ' if vis[j] else 'occluded'}  sigma {sig[0]:5.1f},{sig[1]:5.1f}px")

    # Latency. CPU here; the blog's 9.8 ms is a T4 with TensorRT FP16 at 576px.
    times = []
    for _ in range(args.runs):
        _, ms = model.predict(rgb, threshold=args.threshold)
        times.append(ms)
    print(f"\nlatency    : median {statistics.median(times):.0f} ms  "
          f"min {min(times):.0f}  max {max(times):.0f}  (n={args.runs}, {model.device})")

    payload = model.predict_payload(rgb, threshold=args.threshold, width=w, height=h)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    for p in payload["people"]:
        for k in p["keypoints"]:
            if not (-0.2 <= k["x"] <= 1.2 and -0.2 <= k["y"] <= 1.2):
                failures.append("normalized payload coordinate out of range")
                break

    cv2.imwrite(args.out, annotate(image, kp))
    print(f"wrote      : {args.out}, {out_json}")

    if failures:
        print("\nFAIL:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
