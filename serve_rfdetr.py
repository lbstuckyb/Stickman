"""Local RF-DETR keypoint server for pose-stick-figure-rfdetr.html.

    .venv/bin/python serve_rfdetr.py [--resolution 384] [--port 8000]
    then open http://127.0.0.1:8000/

The page posts a JPEG frame to /keypoints and gets normalized COCO-17 keypoints
back. RF-DETR is a PyTorch model, so it cannot run in the browser the way the
MediaPipe build of this page did — this process is the model, the page is just
the camera and the rig.

Inference is serialized behind a lock: one frame at a time, newest wins. The
page never queues more than one request, so a slow reply costs frame rate, not
latency.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from rfdetr_pose import PoseModel, add_device_args, announce_device

ROOT = Path(__file__).parent
PAGE = "pose-stick-figure-rfdetr.html"
MAX_BODY = 8 * 1024 * 1024

model: PoseModel | None = None
model_lock = threading.Lock()
threshold = 0.5


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", f"/{PAGE}"):
            page = ROOT / PAGE
            if not page.exists():
                return self._json(404, {"error": f"{PAGE} not found next to this script"})
            return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
        if path == "/health":
            return self._json(200, {
                "model": "RFDETRKeypointPreview",
                "resolution": model.resolution,
                "device": model.device,
                "half": model.half,
                "optimize": model.optimize,
                "gpu": model.gpu,
                "torch": model.report["torch"],
                "torch_cuda_build": model.report["torch_cuda_build"],
                "threshold": threshold,
            })
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/keypoints":
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return self._json(400, {"error": "bad Content-Length"})
        if length <= 0 or length > MAX_BODY:
            return self._json(413, {"error": "empty or oversized body"})

        raw = self.rfile.read(length)
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return self._json(400, {"error": "could not decode image"})
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        try:
            with model_lock:      # the model is not re-entrant; keep it to one frame
                payload = model.predict_payload(rgb, threshold=threshold, width=w, height=h)
        except Exception as e:    # noqa: BLE001 - the page shows whatever went wrong
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})

        payload["frame"] = {"width": w, "height": h}
        return self._json(200, payload)

    def log_message(self, fmt, *args):
        pass   # the per-frame access log would drown everything else


def main():
    global model, threshold
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--resolution", type=int, default=384,
                    help="inference resolution, must be divisible by 24 "
                         "(576 is the checkpoint default and most accurate; "
                         "384 or 288 trade accuracy for CPU frame rate)")
    ap.add_argument("--threshold", type=float, default=0.5)
    add_device_args(ap)
    args = ap.parse_args()

    if args.resolution % 24:
        ap.error(f"--resolution must be divisible by 24 (got {args.resolution})")

    threshold = args.threshold
    announce_device(args.device)
    print(f"loading RFDETRKeypointPreview @ {args.resolution}px ...")
    model = PoseModel(resolution=args.resolution, device=args.device,
                      half=args.half, optimize=args.optimize)
    print(f"ready      : {model.summary()} - load {model.timing.load_s:.1f}s, "
          f"warmup {model.timing.warmup_s*1000:.0f}ms")
    print(f"open http://{args.host}:{args.port}/")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
