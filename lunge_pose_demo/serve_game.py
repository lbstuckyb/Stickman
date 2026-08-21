"""Local server for lunge_game.html's recorder: analyzes a captured move with
RF-DETR and turns it into a clip in exactly the shape the baked-in lunge uses.

    .venv/bin/python lunge_pose_demo/serve_game.py [--resolution 384] [--port 8000]
    then open http://127.0.0.1:8000/

Modelled on ../serve_rfdetr.py. /keypoints has the same semantics as that
server's route — one JPEG in, one {people:[...]} payload out — copied
verbatim. The new part is /clip: it imports build_game.py as a module and
calls its build_custom_clip(), so a recorded move goes through literally the
same track_subject -> point_tracks -> smooth_points -> pose_at pipeline that
produced the lunge, retargeted onto the lunge subject's own measured
proportions. Inference stays behind model_lock, one frame at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from rfdetr_pose import PoseModel  # noqa: E402
import build_game as bg  # noqa: E402

PAGE = "lunge_game.html"
MOVES_PATH = HERE / "custom_moves.json"
MAX_BODY = 8 * 1024 * 1024
DEFAULT_IDLE = (79, 106)  # same default window build_game.py measures the lunge rig on

model: PoseModel | None = None
model_lock = threading.Lock()
threshold = 0.5
dst_rig: dict | None = None   # the lunge subject's own bone lengths, unscaled
dst_k: float = 1.0            # source-px -> game-px, same factor the lunge was built at
dst_fps: float = 30.0
moves_lock = threading.Lock()


def _load_lunge_rig():
    """The lunge's own rig and scale — the retarget target for every recorded move.

    `k` has to match build_game.main()'s own scale factor exactly, or a custom
    move would draw the figure at a different size than the idle/walk/lunge
    clips it shares the arena with; that comes from measure_rig()'s idle-window
    medians, same as main(). But that idle stance under-measures a bone that
    only reaches its true length during the lunge itself — the forearm, held
    bent in the idle window, projects near-full length during the strike — so
    the retarget *target* proportions use measure_rig_dynamic() over the whole
    clip instead: the same p90-of-projected-length method a recording is
    measured with, so retargeting the lunge subject onto their own proportions
    is close to the identity it should be.
    """
    data_path = HERE / "lunge_all_frames.json"
    src = json.loads(data_path.read_text())
    frames, vid = src["frames"], src["video"]
    tracked = bg.track_subject(frames, vid["w"], vid["h"])
    pts = bg.smooth_points(bg.point_tracks(tracked), 3)
    k = bg.FIGURE_H / bg.measure_rig(pts, tracked, *DEFAULT_IDLE)["height"]
    rig = bg.measure_rig_dynamic(pts, tracked, 0, len(frames) - 1)
    return rig, k, vid["fps"]


def _read_body(handler) -> bytes:
    length = int(handler.headers.get("Content-Length", 0))
    if length <= 0 or length > MAX_BODY:
        raise ValueError("empty or oversized body")
    return handler.rfile.read(length)


def _load_moves() -> dict:
    if not MOVES_PATH.exists():
        return {"moves": {}, "current": None}
    return json.loads(MOVES_PATH.read_text())


def _save_moves(store: dict) -> None:
    MOVES_PATH.write_text(json.dumps(store, separators=(",", ":")))


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

    # ------------------------------------------------------------------ GET

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", f"/{PAGE}"):
            page = HERE / PAGE
            if not page.exists():
                return self._json(404, {"error": f"{PAGE} not found next to this script — "
                                                   f"run build_game.py first"})
            return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
        if path == "/health":
            return self._json(200, {
                "model": "RFDETRKeypointPreview",
                "resolution": model.resolution,
                "device": model.device,
                "threshold": threshold,
                "rig": dst_rig,
                "scale": dst_k,
            })
        if path == "/moves":
            with moves_lock:
                return self._json(200, _load_moves())
        return self._json(404, {"error": "not found"})

    # ----------------------------------------------------------------- POST

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/keypoints":
            return self._handle_keypoints()
        if path == "/clip":
            return self._handle_clip()
        if path == "/moves":
            return self._handle_save_move()
        return self._json(404, {"error": "not found"})

    def _handle_keypoints(self):
        try:
            raw = _read_body(self)
        except ValueError as e:
            return self._json(413, {"error": str(e)})

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

    def _handle_clip(self):
        try:
            raw = _read_body(self)
        except ValueError as e:
            return self._json(413, {"error": str(e)})
        try:
            body = json.loads(raw)
            frames = body["frames"]
            w, h = int(body["w"]), int(body["h"])
            fps = float(body.get("fps", dst_fps))
            smooth = int(body.get("smooth", 3))
            flip = body.get("flip", None)  # null -> auto-detect
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            return self._json(400, {"error": f"bad request: {e}"})
        if not frames:
            return self._json(400, {"error": "no frames"})

        try:
            clip = bg.build_custom_clip(frames, w, h, fps, dst_rig, dst_k,
                                        smooth=smooth, flip=flip)
        except Exception as e:  # noqa: BLE001 - surfaced to the page, not fatal to the server
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})

        diagnostics = [{"detected": bool(f.get("people")),
                        "score": max((p["score"] for p in f.get("people") or []), default=None)}
                       for f in frames]
        return self._json(200, {"clip": clip, "flip": clip["flip"], "diagnostics": diagnostics})

    def _handle_save_move(self):
        try:
            raw = _read_body(self)
        except ValueError as e:
            return self._json(413, {"error": str(e)})
        try:
            body = json.loads(raw)
            name = str(body["name"]).strip()
            clip = body["clip"]
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            return self._json(400, {"error": f"bad request: {e}"})
        if not name:
            return self._json(400, {"error": "name required"})

        with moves_lock:
            store = _load_moves()
            store["moves"][name] = {"clip": clip}
            store["current"] = name
            _save_moves(store)
        return self._json(200, {"ok": True, "current": name})

    # --------------------------------------------------------------- DELETE

    def do_DELETE(self):
        path = self.path.split("?", 1)[0]
        if not path.startswith("/moves/"):
            return self._json(404, {"error": "not found"})
        name = path[len("/moves/"):]
        with moves_lock:
            store = _load_moves()
            existed = store["moves"].pop(name, None) is not None
            if store.get("current") == name:
                store["current"] = next(iter(store["moves"]), None)
            _save_moves(store)
        return self._json(200, {"ok": True, "existed": existed, "current": store.get("current")})

    def log_message(self, fmt, *args):
        pass   # the per-frame access log would drown everything else


def main():
    global model, threshold, dst_rig, dst_k, dst_fps
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--resolution", type=int, default=384,
                    help="inference resolution, must be divisible by 24 "
                         "(576 is the checkpoint default and most accurate; "
                         "384 or 288 trade accuracy for CPU frame rate)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--optimize", action="store_true",
                    help="torch.compile the model — slow first call, faster afterwards")
    args = ap.parse_args()

    if args.resolution % 24:
        ap.error(f"--resolution must be divisible by 24 (got {args.resolution})")

    threshold = args.threshold
    dst_rig, dst_k, dst_fps = _load_lunge_rig()
    print(f"lunge rig loaded: height {dst_rig['height'] * dst_k:.0f}px, scale {dst_k:.4f}")

    print(f"loading RFDETRKeypointPreview @ {args.resolution}px ...")
    model = PoseModel(resolution=args.resolution, optimize=args.optimize)
    print(f"ready on {model.device} — load {model.timing.load_s:.1f}s, "
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
