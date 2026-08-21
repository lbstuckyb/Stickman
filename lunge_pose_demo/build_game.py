"""Turn the captured RF-DETR lunge into a playable arena game.

Pure JSON in, HTML out: this reads the keypoints `build_sequence_viewer.py` already
wrote (`lunge_all_frames.json`) and emits a single self-contained page,
`lunge_game.html`, where a stick figure driven by those measurements walks, jumps,
and — on `F` — performs the lunge that was actually captured on video.

The character *is* the measured skeleton. Every frame of every captured clip is the
15 joint positions the detector reported, hip-centred and multiplied by a single
scale factor that sizes the subject into the arena, plus a root offset carrying how
far the hips actually travelled across the floor. Nothing is rebuilt from angles at
runtime, so the lunge covers the ground it covers on video, drops as far as the body
drops, and its limbs foreshorten the way the real ones do. Angles survive only
inside the clip that had to be invented — the walk — and the kinematics that turns
them into points uses the subject's own measured bone lengths, so nothing changes
shape when the state changes.

Usage:
    python3 build_game.py
    python3 build_game.py --idle 79:106 --lunge 106:134,154:188 --speed 1.4
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# COCO-17 indices — the same map as pose-stick-figure-rfdetr.html:158.
L = {
    "nose": 0, "eyeL": 1, "eyeR": 2, "earL": 3, "earR": 4,
    "shL": 5, "shR": 6, "elL": 7, "elR": 8, "wrL": 9, "wrR": 10,
    "hipL": 11, "hipR": 12, "knL": 13, "knR": 14, "anL": 15, "anR": 16,
}
# The drawn figure, and the storage format: every clip frame is these 15 points,
# in this order, as x/y pairs relative to the hip midpoint, in game pixels.
POINTS = [
    "hipC", "hipL", "hipR", "shC", "shL", "shR", "headC",
    "elL", "wrL", "elR", "wrR", "knL", "anL", "knR", "anR",
]
# The walk is authored in angles before it is turned into points — same 10
# channels readAngles() uses in ../pose-stick-figure-rfdetr.html.
CHANNELS = [
    "spine", "head",
    "armLU", "armLF", "armRU", "armRF",
    "legLU", "legLS", "legRU", "legRS",
]
# Mirror pairs, for spotting a left/right-swapped detection.
LATERAL = [("eyeL", "eyeR"), ("earL", "earR"), ("shL", "shR"), ("elL", "elR"),
           ("wrL", "wrR"), ("hipL", "hipR"), ("knL", "knR"), ("anL", "anR")]

MIN_VIS = 0.4        # same visibility gate the viewer uses
WALK_CYCLE_S = 0.62  # nominal seconds per procedural stride
FIGURE_H = 249.0     # game px, head to heel — what the old fixed rig drew at
MAX_GAP = 5          # frames without a detection before the tracker restarts


def _mid(a, b):
    return {"x": (a["x"] + b["x"]) / 2, "y": (a["y"] + b["y"]) / 2, "v": min(a["v"], b["v"])}


def _ok(p):
    return p is not None and p["v"] >= MIN_VIS


# --------------------------------------------------------------------------
# 1. Following one person through the clip
# --------------------------------------------------------------------------

def _to_px(kp, w, h):
    """Normalized keypoints -> source pixels, so every distance below is real."""
    return [{"x": p["x"] * w, "y": p["y"] * h, "v": p["v"]} for p in kp]


def _mirror(kp):
    out = list(kp)
    for a, b in LATERAL:
        out[L[a]], out[L[b]] = out[L[b]], out[L[a]]
    return out


def _drift(a, b):
    """Mean joint-to-joint distance between two detections, over what both see."""
    s, n = 0.0, 0
    for pa, pb in zip(a, b):
        if pa["v"] >= MIN_VIS and pb["v"] >= MIN_VIS:
            s += math.hypot(pa["x"] - pb["x"], pa["y"] - pb["y"])
            n += 1
    return s / n if n else math.inf


def track_subject(frames, w, h):
    """One detection per source frame, in pixels, following the same person.

    Taking `max(score)` independently per frame is what dropped a left/right
    *mirrored* duplicate into the middle of the recovery at frame 162 — hips,
    knees and ankles all swapped, which the smoothing then smeared over its
    neighbours as a limb snap. Each frame instead keeps whichever candidate sits
    closest to the frame before it, and a detection whose mirror image is closer
    still gets flipped back (a lone detection can be mirrored too).
    """
    out, prev, gap = [], None, 0
    for fr in frames:
        people = fr.get("people") or []
        if not people:
            out.append(None)
            gap += 1
            if gap > MAX_GAP:
                prev = None
            continue
        cands = [_to_px(p["keypoints"], w, h) for p in people]
        if prev is None:
            kp = cands[max(range(len(people)), key=lambda i: people[i]["score"])]
        else:
            kp = min(cands, key=lambda c: _drift(c, prev))
            flipped = _mirror(kp)
            if _drift(flipped, prev) < _drift(kp, prev):
                kp = flipped
        out.append(kp)
        prev, gap = kp, 0
    return out


def rig_points(kp):
    """The 15 drawn joints, in source pixels, from one tracked detection."""
    g = lambda name: kp[L[name]]
    shC = _mid(g("shL"), g("shR"))
    # COCO-17 has no point above the ears, so the ear midpoint carries the head,
    # falling back to the nose when the far ear isn't seen (most of the lunge).
    ears = _mid(g("earL"), g("earR"))
    pts = {"hipC": _mid(g("hipL"), g("hipR")), "shC": shC,
           "headC": ears if _ok(ears) else g("nose")}
    for name in ("hipL", "hipR", "shL", "shR", "elL", "wrL", "elR", "wrR",
                 "knL", "anL", "knR", "anR"):
        pts[name] = g(name)
    return pts


def point_tracks(tracked):
    """Per-joint x/y track over the whole clip, one entry per source frame.

    A joint the model can't see holds its previous position rather than snapping
    — the same rule retarget() applies live in the viewer, moved from angles to
    points.
    """
    tracks = {p: [] for p in POINTS}
    last, seeded = {p: (0.0, 0.0) for p in POINTS}, False
    for kp in tracked:
        if kp is not None:
            pts = rig_points(kp)
            for name in POINTS:
                q = pts[name]
                if _ok(q) or not seeded:
                    last[name] = (q["x"], q["y"])
            seeded = True
        for name in POINTS:
            tracks[name].append(last[name])
    return tracks


def smooth_points(tracks, window):
    """Centred moving average per coordinate, edges clamped.

    Points need none of the unwrap machinery angles did: there is no ±π seam to
    smear, so this is the whole of it.
    """
    if window <= 1:
        return {name: list(seq) for name, seq in tracks.items()}
    half = window // 2
    out = {}
    for name, seq in tracks.items():
        run = []
        for i in range(len(seq)):
            lo, hi = max(0, i - half), min(len(seq), i + half + 1)
            n = hi - lo
            run.append((sum(p[0] for p in seq[lo:hi]) / n,
                        sum(p[1] for p in seq[lo:hi]) / n))
        out[name] = run
    return out


def _seg(tracks, a, b, lo, hi):
    return [math.hypot(tracks[a][i][0] - tracks[b][i][0],
                       tracks[a][i][1] - tracks[b][i][1]) for i in range(lo, hi + 1)]


def measure_rig(tracks, tracked, lo, hi):
    """The subject's own bone lengths, in source pixels, over the idle window.

    Medians, because a projected bone breathes as the body turns; the standing
    window is where the subject is squarest to the lens and least foreshortened.
    """
    med = lambda *pairs: statistics.median(
        [v for a, b in pairs for v in _seg(tracks, a, b, lo, hi)])
    rig = {
        "spine": med(("hipC", "shC")),
        "neck": med(("shC", "headC")),
        "thigh": med(("hipL", "knL"), ("hipR", "knR")),
        "shin": med(("knL", "anL"), ("knR", "anR")),
        "upperArm": med(("shL", "elL"), ("shR", "elR")),
        "forearm": med(("elL", "wrL"), ("elR", "wrR")),
        "pelvisHalf": med(("hipL", "hipR")) / 2,
        "shoulderHalf": med(("shL", "shR")) / 2,
    }
    # The head circle is drawn to wrap the ears with a little margin — the only
    # part of the figure COCO-17 gives no extent for.
    ears = [math.hypot(kp[L["earL"]]["x"] - kp[L["earR"]]["x"],
                       kp[L["earL"]]["y"] - kp[L["earR"]]["y"])
            for kp in tracked[lo:hi + 1]
            if kp is not None and _ok(kp[L["earL"]]) and _ok(kp[L["earR"]])]
    rig["headR"] = 0.7 * statistics.median(ears) if ears else 0.42 * rig["neck"]
    # Standing height, head to heel: what the scale factor is calibrated against.
    rig["height"] = rig["neck"] + rig["spine"] + rig["thigh"] + rig["shin"]
    return rig


# --------------------------------------------------------------------------
# 1b. Retargeting a recorded move onto the lunge subject's proportions
# --------------------------------------------------------------------------

def _percentile(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = (p / 100) * (len(s) - 1)
    lo, frac = int(math.floor(idx)), idx - int(math.floor(idx))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - frac) + s[hi] * frac


def measure_rig_dynamic(tracks, tracked, lo, hi):
    """Like measure_rig(), but a high percentile of each bone length instead of
    the median.

    measure_rig()'s medians assume a still window to measure from; a 4-second
    action recording has no guaranteed still stretch, and the median of a bone
    that spends most of the clip foreshortened would under-measure it. The
    bone's true length is the extreme its projection reaches toward, so this
    takes the 90th percentile over the whole window instead.
    """
    p90 = lambda *pairs: _percentile(
        [v for a, b in pairs for v in _seg(tracks, a, b, lo, hi)], 90)
    rig = {
        "spine": p90(("hipC", "shC")),
        "neck": p90(("shC", "headC")),
        "thigh": p90(("hipL", "knL"), ("hipR", "knR")),
        "shin": p90(("knL", "anL"), ("knR", "anR")),
        "upperArm": p90(("shL", "elL"), ("shR", "elR")),
        "forearm": p90(("elL", "wrL"), ("elR", "wrR")),
        "pelvisHalf": p90(("hipL", "hipR")) / 2,
        "shoulderHalf": p90(("shL", "shR")) / 2,
    }
    ears = [math.hypot(kp[L["earL"]]["x"] - kp[L["earR"]]["x"],
                       kp[L["earL"]]["y"] - kp[L["earR"]]["y"])
            for kp in tracked[lo:hi + 1]
            if kp is not None and _ok(kp[L["earL"]]) and _ok(kp[L["earR"]])]
    rig["headR"] = 0.7 * (_percentile(ears, 90) if ears else 0.6 * rig["neck"])
    rig["height"] = rig["neck"] + rig["spine"] + rig["thigh"] + rig["shin"]
    return rig


# Walked outward from hipC. Each entry is (parent, child, bone-name-in-rig).
_HIERARCHY = [
    ("hipC", "hipL", "pelvisHalf"), ("hipC", "hipR", "pelvisHalf"),
    ("hipC", "shC", "spine"),
    ("shC", "shL", "shoulderHalf"), ("shC", "shR", "shoulderHalf"),
    ("shC", "headC", "neck"),
    ("shL", "elL", "upperArm"), ("elL", "wrL", "forearm"),
    ("shR", "elR", "upperArm"), ("elR", "wrR", "forearm"),
    ("hipL", "knL", "thigh"), ("knL", "anL", "shin"),
    ("hipR", "knR", "thigh"), ("knR", "anR", "shin"),
]


def retarget_ratio(frame_pts, src_rig, dst_rig):
    """One frame of measured points -> the same points on the lunge subject's
    proportions.

    Forcing every bone to the target's exact length would keep direction but
    throw away projection: a limb foreshortened to 60% in the recording would
    pop out to full length and stop looking measured. Instead each bone's
    offset from its parent joint is scaled by one constant, dst/src rest
    length, walking outward from hipC. Every measured direction and every
    measured projection survives; only the bone's own scale changes.
    """
    out = {"hipC": frame_pts["hipC"]}
    for parent, child, bone in _HIERARCHY:
        ox, oy = out[parent]                  # where the parent landed after rescaling
        px, py = frame_pts[parent]            # where it was measured — the bone vector
        cx, cy = frame_pts[child]             # is child - parent in the *original* frame
        ratio = (dst_rig[bone] / src_rig[bone]) if src_rig[bone] else 1.0
        out[child] = (ox + (cx - px) * ratio, oy + (cy - py) * ratio)
    return out


def auto_flip(tracks, lo, hi):
    """True if the recording's forward reach points at +x and should be mirrored.

    The captured clips face -x by convention (toWorld() in the template flips
    on that). Forward is wherever a lead wrist or ankle reaches furthest from
    the hips — the same reach_joints signal auto_strike() measures — so the
    two stay consistent: whichever way auto_flip mirrors the clip, auto_strike
    finds its peak on the -x side. A strike can happen with little net hip
    travel (a reach more than a step), so hip excursion alone is not a
    reliable proxy for it and is kept only as a fallback for a subject who
    barely reaches at all, then head lean relative to the hips after that.
    """
    reach_names = ("wrL", "wrR", "anL", "anR")
    best = 0.0
    for i in range(lo, hi + 1):
        hip_x = tracks["hipC"][i][0]
        for name in reach_names:
            d = tracks[name][i][0] - hip_x
            if abs(d) > abs(best):
                best = d
    if abs(best) > 5:
        return best > 0
    x0 = tracks["hipC"][lo][0]
    excursion = max((tracks["hipC"][i][0] - x0 for i in range(lo, hi + 1)), key=abs)
    if abs(excursion) > 5:
        return excursion > 0
    lean = sum(tracks["headC"][i][0] - tracks["hipC"][i][0] for i in range(lo, hi + 1))
    return lean > 0


_REACH_JOINTS = [POINTS.index(n) for n in ("wrL", "wrR", "anL", "anR")]


def auto_strike(clip_frames):
    """[lo, hi] clip-frame span where forward reach is within ~85% of its peak.

    Frames are already hip-relative (pose_at()'s output), and the clip faces
    -x, so "forward" is the most negative x any lead wrist or ankle reaches.
    Same [lo, hi] contract as CLIPS.lunge's `active`.
    """
    reach = [max(-fr[2 * i] for i in _REACH_JOINTS) for fr in clip_frames]
    peak_i = max(range(len(reach)), key=lambda i: reach[i])
    peak = reach[peak_i]
    if peak <= 0:
        return [0, len(clip_frames) - 1]
    thresh = 0.85 * peak
    lo = hi = peak_i
    while lo > 0 and reach[lo - 1] >= thresh:
        lo -= 1
    while hi < len(reach) - 1 and reach[hi + 1] >= thresh:
        hi += 1
    return [lo, hi]


def build_custom_clip(payload_frames, w, h, fps, dst_rig, k, smooth=3, flip=None):
    """A recorded move -> a clip in exactly the shape the other clips use.

    Runs the same track_subject -> point_tracks -> smooth_points pipeline the
    lunge itself went through, measures the recording's own bone lengths with
    measure_rig_dynamic(), retargets every frame onto `dst_rig` (the lunge
    subject's proportions) with retarget_ratio(), then hands off to pose_at() -
    the same function that flattens every other clip. No angle is ever solved.

    The flip decision is made *after* retargeting, not before: each bone is
    rescaled by its own dst/src ratio, and when the recorded subject's
    proportions differ a lot from the lunge subject's, that can change which
    direction the net reach from the hips actually points. auto_flip and
    auto_strike need to agree on the same geometry, so both read the
    retargeted frames. Negating x before or after retarget_ratio's linear,
    hip-relative scaling gives the same hip-centered result either way, so
    flipping post-retarget costs nothing.
    """
    n = len(payload_frames)
    tracked = track_subject(payload_frames, w, h)
    raw_pts = point_tracks(tracked)
    pts = smooth_points(raw_pts, smooth)

    src_rig = measure_rig_dynamic(pts, tracked, 0, n - 1)

    retgt = {name: [] for name in POINTS}
    for i in range(n):
        frame_pts = {name: pts[name][i] for name in POINTS}
        placed = retarget_ratio(frame_pts, src_rig, dst_rig)
        for name in POINTS:
            retgt[name].append(placed[name])

    if flip is None:
        flip = auto_flip(retgt, 0, n - 1)
    if flip:
        retgt = {name: [(-x, y) for x, y in seq] for name, seq in retgt.items()}

    frames = [pose_at(retgt, i, k) for i in range(n)]
    # retarget_ratio() anchors the walk at hipC and never rescales it, so it's
    # still in the recording's own pixel space — unlike every other joint, which
    # comes out rescaled bone-by-bone onto dst_rig. Multiplying that raw delta by
    # `k` (calibrated for the lunge video's pixel space) would compare two
    # different cameras' pixel spaces as if they were the same one, so the hip's
    # own travel needs the same src->dst rescale the bones got, via one overall
    # body-size ratio.
    body_ratio = dst_rig["height"] / src_rig["height"] if src_rig["height"] else 1.0
    hip0 = retgt["hipC"][0][0]
    root = [round((retgt["hipC"][i][0] - hip0) * body_ratio * k, 1) for i in range(n)]
    active = auto_strike(frames)
    detected = sum(1 for f in payload_frames if f.get("people"))

    return {
        "fps": round(fps, 3),
        "frames": frames,
        "root": root,
        "active": active,
        "flip": bool(flip),
        "source": f"recorded, {n} frames, {detected} detected",
        "note": f"recorded move &mdash; {n} frames captured at {fps:.1f}fps, {detected} with a "
                f"detection, retargeted onto the lunge subject's proportions with one ratio per "
                f"bone. Nothing is solved from angles.",
        "measured": True,
    }


# --------------------------------------------------------------------------
# 2. Keypoints -> angles  (only the procedural walk still needs these)
# --------------------------------------------------------------------------

def _ang(a, b):
    return math.atan2(b["y"] - a["y"], b["x"] - a["x"])


def read_angles(kp):
    """kp: 17 dicts in source pixels. Returns (angles, confident).

    Port of readAngles(), pose-stick-figure-rfdetr.html:241. Pixels rather than
    normalized coordinates, so the angles are measured in real proportions.
    """
    g = lambda name: kp[L[name]]
    pts = rig_points(kp)
    hips, shs, head = pts["hipC"], pts["shC"], pts["headC"]

    raw = {
        "spine": _ang(hips, shs),
        "head": _ang(shs, head),
        "armLU": _ang(g("shL"), g("elL")), "armLF": _ang(g("elL"), g("wrL")),
        "armRU": _ang(g("shR"), g("elR")), "armRF": _ang(g("elR"), g("wrR")),
        "legLU": _ang(g("hipL"), g("knL")), "legLS": _ang(g("knL"), g("anL")),
        "legRU": _ang(g("hipR"), g("knR")), "legRS": _ang(g("knR"), g("anR")),
    }
    conf = {
        "spine": _ok(hips) and _ok(shs), "head": _ok(shs) and _ok(head),
        "armLU": _ok(g("shL")) and _ok(g("elL")), "armLF": _ok(g("elL")) and _ok(g("wrL")),
        "armRU": _ok(g("shR")) and _ok(g("elR")), "armRF": _ok(g("elR")) and _ok(g("wrR")),
        "legLU": _ok(g("hipL")) and _ok(g("knL")), "legLS": _ok(g("knL")) and _ok(g("anL")),
        "legRU": _ok(g("hipR")) and _ok(g("knR")), "legRS": _ok(g("knR")) and _ok(g("anR")),
    }
    return raw, conf


def extract_channels(tracked):
    """Per-channel angle track over the whole clip, one value per source frame.

    A channel the model isn't confident about holds its previous angle instead of
    snapping — the same rule retarget() applies live in the viewer.
    """
    tracks = {c: [] for c in CHANNELS}
    knee = []  # right-knee interior angle, used to find the deepest lunge frame
    last = {c: 0.0 for c in CHANNELS}
    for kp in tracked:
        if kp is None:
            for c in CHANNELS:
                tracks[c].append(last[c])
            knee.append(180.0)
            continue
        raw, conf = read_angles(kp)
        for c in CHANNELS:
            if conf[c]:
                last[c] = raw[c]
            tracks[c].append(last[c])
        knee.append(_interior(kp, "hipR", "knR", "anR"))
    return tracks, knee


def _interior(kp, a, b, c):
    """Interior angle at joint b, in degrees (180 = straight)."""
    pa, pb, pc = (kp[L[n]] for n in (a, b, c))
    v1 = (pa["x"] - pb["x"], pa["y"] - pb["y"])
    v2 = (pc["x"] - pb["x"], pc["y"] - pb["y"])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def circular_mean(tracks, lo, hi):
    out = {}
    for c in CHANNELS:
        s = sum(math.sin(a) for a in tracks[c][lo:hi])
        k = sum(math.cos(a) for a in tracks[c][lo:hi])
        out[c] = math.atan2(s, k)
    return out


# --------------------------------------------------------------------------
# 3. Clips
# --------------------------------------------------------------------------

def pose_at(tracks, i, k):
    """One clip frame: 15 hip-relative x/y pairs in game pixels, flattened."""
    hx, hy = tracks["hipC"][i]
    out = []
    for name in POINTS:
        x, y = tracks[name][i]
        out += [round((x - hx) * k, 1), round((y - hy) * k, 1)]
    return out


def solve_points(pose, rig, k):
    """Forward kinematics on the subject's measured bones -> a clip frame.

    This used to run in the browser on a fixed rig, once per drawn frame; the
    measured clips have no use for it any more. It stays here so the one clip
    that *is* authored in angles — the walk — comes out in the same 15-point
    format, built at the same proportions as the captured frames.
    """
    step = lambda pt, a, ln: (pt[0] + math.cos(a) * ln, pt[1] + math.sin(a) * ln)
    off = step  # same walk, used for the lateral pelvis/shoulder offsets
    hipC = (0.0, 0.0)
    perp = pose["spine"] + math.pi / 2

    j = {"hipC": hipC}
    j["hipL"] = off(hipC, perp, rig["pelvisHalf"])
    j["hipR"] = off(hipC, perp, -rig["pelvisHalf"])
    j["shC"] = step(hipC, pose["spine"], rig["spine"])
    j["shL"] = off(j["shC"], perp, rig["shoulderHalf"])
    j["shR"] = off(j["shC"], perp, -rig["shoulderHalf"])
    j["headC"] = step(j["shC"], pose["head"], rig["neck"])
    j["elL"] = step(j["shL"], pose["armLU"], rig["upperArm"])
    j["wrL"] = step(j["elL"], pose["armLF"], rig["forearm"])
    j["elR"] = step(j["shR"], pose["armRU"], rig["upperArm"])
    j["wrR"] = step(j["elR"], pose["armRF"], rig["forearm"])
    j["knL"] = step(j["hipL"], pose["legLU"], rig["thigh"])
    j["anL"] = step(j["knL"], pose["legLS"], rig["shin"])
    j["knR"] = step(j["hipR"], pose["legRU"], rig["thigh"])
    j["anR"] = step(j["knR"], pose["legRS"], rig["shin"])

    out = []
    for name in POINTS:
        out += [round(j[name][0] * k, 1), round(j[name][1] * k, 1)]
    return out


LEG_CHANNELS = ["legLU", "legLS", "legRU", "legRS"]


def neutral_legs(rest):
    """The rest pose with its four leg channels averaged into one straight leg.

    The idle window is the subject's ready stance: knees bent, feet split, one of
    them angled toward the lens. Swinging a walk cycle around *that* gives a limp
    with a knee that folds the wrong way. Averaging the four channels leaves a
    single straight leg at the stance's own lean to swing both legs around; the
    torso, head and arm carriage stay measured, and so do the bone lengths, so
    the walking figure is still the same body at nearly the same height.
    """
    out = dict(rest)
    m = math.atan2(sum(math.sin(rest[c]) for c in LEG_CHANNELS),
                   sum(math.cos(rest[c]) for c in LEG_CHANNELS))
    for c in LEG_CHANNELS:
        out[c] = m
    return out


def build_walk(rest, n_frames, rig, k):
    """Procedural — there is no walk cycle anywhere in the source clip.

    The subject stands, lunges, and stands again, so a walk has to be invented.
    It is built off the measured rest pose (so the torso, head carriage and arm
    hang are still the captured ones) with sinusoidal swing on the four leg
    channels and a counter-swing on the upper arms, then solved onto the
    subject's own bone lengths. The emitted clip is tagged "procedural" rather
    than dressed up as measurement.

    The character faces -x (that is the direction the subject lunges in the
    video), so a forward swing raises the thigh angle toward pi.
    """
    swing = 0.62      # thigh amplitude, radians
    bend = 0.85       # extra knee flexion through the swing phase
    arm = 0.42        # upper-arm counter-swing
    fore = 0.20       # forearm follow
    bob_amp = 3.2     # px of lift at mid-stance, applied on top of the ground clamp

    frames, bob = [], []
    for i in range(n_frames):
        p = 2 * math.pi * i / n_frames
        q = p + math.pi  # opposite leg
        pose = dict(rest)
        pose["legRU"] = rest["legRU"] + swing * math.sin(p)
        pose["legLU"] = rest["legLU"] + swing * math.sin(q)
        pose["legRS"] = rest["legRS"] + swing * math.sin(p) - bend * max(0.0, math.sin(p + 0.9))
        pose["legLS"] = rest["legLS"] + swing * math.sin(q) - bend * max(0.0, math.sin(q + 0.9))
        # arms swing against the same-side leg
        pose["armRU"] = rest["armRU"] - arm * math.sin(p)
        pose["armLU"] = rest["armLU"] - arm * math.sin(q)
        pose["armRF"] = rest["armRF"] - fore * math.sin(p)
        pose["armLF"] = rest["armLF"] - fore * math.sin(q)
        pose["spine"] = rest["spine"] + 0.03 * math.sin(2 * p)
        frames.append(solve_points(pose, rig, k))
        # two bob cycles per stride: lifted at mid-stance, level at foot contact
        bob.append(round(-bob_amp * (0.5 - 0.5 * math.cos(2 * p)), 2))
    return frames, bob


def build_tuck(rest, rig, k):
    """The airborne pose, emitted once: knees up, arms up, off the rest pose.

    The runtime lerps toward it by how close the jump is to its apex, instead of
    doing angle arithmetic per frame.
    """
    pose = dict(rest)
    pose["legRU"] += 0.55
    pose["legLU"] += 0.33
    pose["legRS"] -= 0.95
    pose["legLS"] -= 0.62
    pose["armRU"] -= 0.75
    pose["armLU"] -= 0.55
    pose["armRF"] -= 0.45
    pose["armLF"] -= 0.30
    return solve_points(pose, rig, k)


def parse_range(text, what):
    try:
        lo, hi = (int(v) for v in text.split(":"))
    except ValueError:
        raise SystemExit(f"--{what}: expected LO:HI, got {text!r}")
    if hi <= lo:
        raise SystemExit(f"--{what}: {lo}:{hi} is empty")
    return lo, hi


# --------------------------------------------------------------------------
# 4. The page
# --------------------------------------------------------------------------

def build_html(data_json: str) -> str:
    return r"""<title>Lunge Arena</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root{
    --bg:#12161b; --surface:#1a2028; --surface-2:#212933; --border:#2b3440; --border-soft:#232b35;
    --text:#e8edf1; --text-dim:#8d99a6; --text-faint:#5c6875;
    --teal:#4fd1c5; --teal-dim:#2f6b66; --amber:#f2b544; --amber-dim:#7a5c26; --warn:#e2574c;
    --stage-bg:#0d1116; --grid:#1b222a;
  }
  *{box-sizing:border-box;}
  body{ margin:0; background:var(--bg); color:var(--text); font-family:'IBM Plex Sans', system-ui, sans-serif; -webkit-font-smoothing:antialiased; }
  .wrap{ max-width:1400px; margin:0 auto; padding:2.5rem 2rem 4rem; }
  header{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-end; gap:1.5rem; margin-bottom:1.75rem; padding-bottom:1.5rem; border-bottom:1px solid var(--border-soft); }
  .eyebrow{ font-family:'IBM Plex Mono', monospace; font-size:0.72rem; letter-spacing:0.14em; text-transform:uppercase; color:var(--teal); margin:0 0 0.5rem; }
  h1{ font-family:'Sora', sans-serif; font-weight:700; font-size:clamp(1.6rem, 2.4vw, 2.15rem); margin:0; letter-spacing:-0.01em; text-wrap:balance; }
  h1 span{ color:var(--text-dim); font-weight:500; }
  .stat-strip{ display:flex; gap:0; flex-wrap:wrap; background:var(--surface); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
  .stat{ padding:0.65rem 1.1rem; border-right:1px solid var(--border); min-width:5.5rem; }
  .stat:last-child{ border-right:none; }
  .stat-label{ font-family:'IBM Plex Mono', monospace; font-size:0.62rem; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-faint); margin-bottom:0.2rem; }
  .stat-value{ font-family:'IBM Plex Mono', monospace; font-size:1.05rem; font-weight:500; color:var(--text); font-variant-numeric:tabular-nums; }
  .stat-value.accent{ color:var(--amber); }
  .stat-value.teal{ color:var(--teal); }
  .stat-value.low{ color:var(--warn); }
  main{ display:grid; grid-template-columns:minmax(0, 1.55fr) minmax(320px, 1fr); gap:1.5rem; align-items:start; }
  @media (max-width: 920px){ main{ grid-template-columns:1fr; } }
  .panel{ background:var(--surface); border:1px solid var(--border); border-radius:12px; }
  .stage-panel{ overflow:hidden; }
  .stage-head{ display:flex; justify-content:space-between; align-items:center; padding:0.9rem 1.15rem; border-bottom:1px solid var(--border-soft); gap:1rem; flex-wrap:wrap; }
  .stage-head h2, .table-head h2{ font-family:'Sora', sans-serif; font-size:0.92rem; font-weight:600; margin:0; }
  .frame-readout{ display:flex; align-items:baseline; gap:0.85rem; font-family:'IBM Plex Mono', monospace; font-size:0.75rem; color:var(--text-dim); }
  .frame-readout b{ font-size:0.9rem; color:var(--text); font-weight:500; font-variant-numeric:tabular-nums; }
  .strike-badge{ font-family:'IBM Plex Mono', monospace; font-size:0.62rem; letter-spacing:0.08em; text-transform:uppercase; padding:0.12rem 0.45rem; border-radius:3px; border:1px solid var(--border); color:var(--text-faint); transition:color 0.08s, border-color 0.08s, background 0.08s; }
  .strike-badge.on{ color:#12161b; background:var(--amber); border-color:var(--amber); }
  .stage{ position:relative; width:100%; background:var(--stage-bg); line-height:0; }
  .stage canvas{ display:block; width:100%; height:auto; touch-action:none; }
  .hintbar{ display:flex; flex-wrap:wrap; gap:1.1rem; align-items:center; padding:0.8rem 1.15rem; border-top:1px solid var(--border-soft); font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:var(--text-dim); }
  .hintbar .k{ display:flex; align-items:center; gap:0.4rem; }
  kbd{ font-family:'IBM Plex Mono', monospace; background:var(--surface-2); border:1px solid var(--border); border-bottom-width:2px; border-radius:4px; padding:0.05rem 0.4rem; color:var(--text); font-size:0.7rem; }
  .hintbar .spacer{ margin-left:auto; color:var(--text-faint); }
  .side{ display:flex; flex-direction:column; gap:1.5rem; }
  .table-head{ display:flex; justify-content:space-between; align-items:center; padding:0.9rem 1.15rem; border-bottom:1px solid var(--border-soft); }
  .table-head .count{ font-family:'IBM Plex Mono', monospace; font-size:0.7rem; color:var(--text-faint); }
  .clip-row{ display:grid; grid-template-columns:1fr auto; gap:0.2rem 0.75rem; padding:0.7rem 1.15rem; border-bottom:1px solid var(--border-soft); transition:background 0.12s; }
  .clip-row:last-child{ border-bottom:none; }
  .clip-row.active{ background:var(--surface-2); }
  .clip-name{ font-size:0.82rem; font-weight:600; color:var(--text); display:flex; align-items:center; gap:0.5rem; }
  .clip-name .dot{ width:0.45rem; height:0.45rem; border-radius:50%; background:var(--border); }
  .clip-row.active .clip-name .dot{ background:var(--teal); box-shadow:0 0 0 3px rgba(79,209,197,0.15); }
  .clip-frames{ font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:var(--text-faint); font-variant-numeric:tabular-nums; text-align:right; }
  .clip-src{ grid-column:1 / -1; font-family:'IBM Plex Mono', monospace; font-size:0.68rem; color:var(--text-dim); line-height:1.5; }
  .clip-src em{ color:var(--amber); font-style:normal; }
  .badge{ font-family:'IBM Plex Mono', monospace; font-size:0.6rem; letter-spacing:0.04em; text-transform:uppercase; padding:0.08rem 0.35rem; border-radius:3px; border:1px solid var(--border); color:var(--text-faint); }
  .badge.proc{ color:var(--amber); border-color:var(--amber-dim); }
  .badge.meas{ color:var(--teal); border-color:var(--teal-dim); }
  .note{ padding:0.9rem 1.15rem; font-size:0.78rem; line-height:1.65; color:var(--text-dim); border-top:1px solid var(--border-soft); }
  .note b{ color:var(--text); font-weight:600; }
  footer{ margin-top:1.75rem; padding-top:1.25rem; border-top:1px solid var(--border-soft); font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:var(--text-faint); line-height:1.7; }
  footer code{ background:var(--surface-2); padding:0.05rem 0.35rem; border-radius:3px; color:var(--text-dim); }
  ::selection{ background:var(--teal-dim); color:#fff; }

  .rec-open-btn{ font-family:'IBM Plex Mono', monospace; font-size:0.68rem; letter-spacing:0.04em; text-transform:uppercase; color:var(--teal); background:var(--surface-2); border:1px solid var(--border); border-radius:6px; padding:0.45rem 0.8rem; cursor:pointer; }
  .rec-open-btn:hover{ border-color:var(--teal-dim); }
  .rec-open-btn:disabled{ color:var(--text-faint); cursor:not-allowed; border-color:var(--border); }
  .rec-modal{ position:absolute; inset:0; background:rgba(13,17,22,0.92); display:flex; align-items:center; justify-content:center; z-index:10; padding:1.25rem; line-height:1.5; }
  .rec-modal[hidden]{ display:none; }
  .rec-panel{ background:var(--surface); border:1px solid var(--border); border-radius:12px; width:min(880px, 100%); max-height:100%; overflow:auto; display:flex; flex-direction:column; }
  .rec-head{ display:flex; justify-content:space-between; align-items:center; padding:0.85rem 1.1rem; border-bottom:1px solid var(--border-soft); }
  .rec-head h3{ font-family:'Sora', sans-serif; font-size:0.92rem; margin:0; font-weight:600; }
  .rec-close{ background:none; border:none; color:var(--text-dim); font-size:1.3rem; line-height:1; cursor:pointer; padding:0.1rem 0.5rem; }
  .rec-close:hover{ color:var(--text); }
  .rec-body{ padding:1.1rem; display:flex; flex-direction:column; gap:0.85rem; }
  .rec-note{ font-size:0.75rem; color:var(--text-dim); line-height:1.55; margin:0; }
  .rec-note b{ color:var(--teal); font-weight:600; }
  .rec-server-note{ font-size:0.75rem; color:var(--warn); margin:0; }
  .rec-capture-stage{ position:relative; width:100%; max-width:480px; aspect-ratio:4/3; background:#000; border-radius:8px; overflow:hidden; margin:0 auto; }
  .rec-capture-stage video{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; transform:scaleX(-1); }
  .rec-capture-stage canvas{ position:absolute; inset:0; width:100%; height:100%; transform:scaleX(-1); }
  .rec-countdown{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-family:'Sora', sans-serif; font-weight:700; font-size:4.5rem; color:#fff; text-shadow:0 2px 14px rgba(0,0,0,0.65); pointer-events:none; }
  .rec-timebar{ position:absolute; left:0; right:0; bottom:0; height:5px; background:rgba(255,255,255,0.15); }
  .rec-timebar i{ display:block; height:100%; width:0%; background:var(--warn); }
  .rec-controls{ display:flex; gap:0.55rem; flex-wrap:wrap; align-items:center; justify-content:center; }
  .rec-body button{ font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:var(--text); background:var(--surface-2); border:1px solid var(--border); border-radius:6px; padding:0.5rem 0.85rem; cursor:pointer; }
  .rec-body button:hover:not(:disabled){ border-color:var(--teal-dim); }
  .rec-body button:disabled{ color:var(--text-faint); cursor:not-allowed; }
  .rec-body button.primary{ color:#12161b; background:var(--teal); border-color:var(--teal); font-weight:600; }
  .rec-body input[type=text]{ font-family:'IBM Plex Mono', monospace; font-size:0.75rem; color:var(--text); background:var(--surface-2); border:1px solid var(--border); border-radius:6px; padding:0.5rem 0.7rem; width:9rem; }
  .rec-body input[type=range]{ accent-color:var(--teal); }
  .rec-body label{ font-size:0.72rem; color:var(--text-dim); display:flex; align-items:center; gap:0.35rem; }
  .rec-progress{ height:8px; border-radius:4px; background:var(--surface-2); overflow:hidden; }
  .rec-progress i{ display:block; height:100%; width:0%; background:var(--teal); transition:width 0.15s; }
  .rec-progress-text{ font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:var(--text-dim); }
  .rec-trim-wrap{ display:flex; gap:1.15rem; align-items:flex-start; flex-wrap:wrap; }
  .rec-preview-box{ width:230px; flex:none; }
  .rec-preview-box canvas{ width:100%; aspect-ratio:960/540; background:var(--stage-bg); border-radius:8px; border:1px solid var(--border); display:block; }
  .rec-trim-col{ flex:1; min-width:280px; display:flex; flex-direction:column; gap:0.55rem; }
  .rec-ticks{ display:flex; height:20px; border-radius:4px; overflow:hidden; border:1px solid var(--border); }
  .rec-ticks i{ flex:1; background:var(--teal-dim); }
  .rec-ticks i.miss{ background:var(--border-soft); }
  .rec-range{ position:relative; height:32px; background:var(--surface-2); border-radius:6px; }
  .rec-range .fill, .rec-range .strike-fill{ position:absolute; top:0; bottom:0; pointer-events:none; }
  .rec-range .fill{ background:rgba(79,209,197,0.16); border-radius:6px; }
  .rec-range .strike-fill{ background:rgba(242,181,68,0.38); }
  .rec-handle{ position:absolute; top:0; bottom:0; width:10px; margin-left:-5px; cursor:ew-resize; border-radius:3px; touch-action:none; }
  .rec-handle.io{ background:var(--teal); }
  .rec-handle.strike{ background:var(--amber); width:8px; margin-left:-4px; }
  .rec-trim-meta{ font-family:'IBM Plex Mono', monospace; font-size:0.68rem; color:var(--text-dim); display:flex; gap:1rem; flex-wrap:wrap; justify-content:space-between; }
  .rec-save-row{ display:flex; gap:0.55rem; align-items:center; flex-wrap:wrap; }
</style>

<div class="wrap">
  <header>
    <div>
      <p class="eyebrow">RF-DETR keypoints &middot; playable</p>
      <h1>Lunge Arena <span>&mdash; the captured lunge is the attack</span></h1>
    </div>
    <div class="stat-strip">
      <div class="stat"><div class="stat-label">Score</div><div class="stat-value accent" id="statScore">0</div></div>
      <div class="stat"><div class="stat-label">Combo</div><div class="stat-value teal" id="statCombo">&times;1</div></div>
      <div class="stat"><div class="stat-label">Time</div><div class="stat-value" id="statTime">60.0</div></div>
      <div class="stat"><div class="stat-label">Best</div><div class="stat-value" id="statBest">0</div></div>
    </div>
  </header>

  <main>
    <section class="panel stage-panel">
      <div class="stage-head">
        <h2>Arena</h2>
        <div class="frame-readout">
          <span>state <b id="roState">idle</b></span>
          <span>clip frame <b id="roFrame">&mdash;</b></span>
          <span class="strike-badge" id="roStrike">strike</span>
        </div>
        <button id="recOpen" class="rec-open-btn">Record / upload a move</button>
        <span id="recOpenNote" class="rec-progress-text" hidden></span>
      </div>
      <div class="stage">
        <canvas id="cv"></canvas>
        <div id="recModal" class="rec-modal" hidden>
          <div class="rec-panel">
            <div class="rec-head">
              <h3 id="recTitle">Record a move</h3>
              <button id="recClose" class="rec-close" aria-label="close">&times;</button>
            </div>
            <div class="rec-body">
              <p id="recServerNote" class="rec-server-note" hidden></p>

              <section id="recStageCapture" class="rec-stage">
                <p class="rec-note">Live skeleton is <b>preview only</b> &mdash; the move itself
                  is measured by RF-DETR on the server, frame by frame, once you record.</p>
                <div class="rec-capture-stage">
                  <video id="recVideo" playsinline muted autoplay></video>
                  <canvas id="recOverlay"></canvas>
                  <div id="recCountdown" class="rec-countdown"></div>
                  <div class="rec-timebar"><i id="recTimeFill"></i></div>
                </div>
                <div class="rec-controls">
                  <button id="recStartCam">Start camera</button>
                  <button id="recGo" disabled>Record (4s)</button>
                  <button id="recUploadBtn">Upload a video&hellip;</button>
                  <input type="file" id="recFileInput" accept="video/*" hidden>
                </div>
                <div id="recUploadPanel" class="rec-controls" hidden>
                  <input type="range" id="recUploadStart" min="0" max="0" step="0.05" value="0" style="flex:1">
                  <span id="recUploadWindow" class="rec-progress-text"></span>
                  <button id="recUploadUse" class="primary">Use this 4s window</button>
                </div>
              </section>

              <section id="recStageAnalyze" class="rec-stage" hidden>
                <p class="rec-note">Sending each frame to RF-DETR &mdash; one at a time, the same
                  way the arena's own clips were measured. Roughly 0.35s/frame on CPU.</p>
                <div class="rec-progress"><i id="recProgFill"></i></div>
                <div id="recProgText" class="rec-progress-text">&mdash;</div>
              </section>

              <section id="recStageTrim" class="rec-stage" hidden>
                <div class="rec-trim-wrap">
                  <div class="rec-preview-box">
                    <canvas id="recPreview" width="960" height="540"></canvas>
                    <div class="rec-trim-meta">
                      <span id="recPreviewFrames">&mdash;</span>
                      <label><input type="checkbox" id="recFlip"> flip facing</label>
                    </div>
                  </div>
                  <div class="rec-trim-col">
                    <div id="recTicks" class="rec-ticks"></div>
                    <div id="recRange" class="rec-range">
                      <div class="fill" id="recFill"></div>
                      <div class="strike-fill" id="recStrikeFill"></div>
                      <div class="rec-handle io" id="recInHandle"></div>
                      <div class="rec-handle strike" id="recStrikeLoHandle"></div>
                      <div class="rec-handle strike" id="recStrikeHiHandle"></div>
                      <div class="rec-handle io" id="recOutHandle"></div>
                    </div>
                    <div class="rec-trim-meta">
                      <span id="recWindowMeta">&mdash;</span>
                      <span id="recStrikeMeta">&mdash;</span>
                    </div>
                    <div class="rec-save-row">
                      <input type="text" id="recMoveName" value="custom" maxlength="24">
                      <button id="recSave" class="primary">Use this move</button>
                      <button id="recRetry">Record again</button>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          </div>
        </div>
      </div>
      <div class="hintbar">
        <span class="k"><kbd>&larr;</kbd><kbd>&rarr;</kbd> move</span>
        <span class="k"><kbd>Space</kbd> jump</span>
        <span class="k"><kbd>F</kbd> lunge</span>
        <span class="k"><kbd>C</kbd> custom move</span>
        <span class="k"><kbd>P</kbd> pause</span>
        <span class="k"><kbd>R</kbd> restart</span>
        <span class="spacer">hit targets with the lead foot or hand, mid-lunge</span>
      </div>
    </section>

    <aside class="side">
      <section class="panel">
        <div class="table-head"><h2>Clips driving the figure</h2><span class="count" id="clipCount"></span></div>
        <div id="clipList"></div>
        <div class="note">
          Every frame above is <b>15 joint positions</b> &mdash; where a COCO-17 detection
          actually put the hips, knees, ankles, shoulders, elbows, wrists and head. They are
          hip-centred and multiplied by one scale factor that sizes the subject into this
          arena; nothing is rebuilt at runtime. So the lunge covers the ground the subject
          covered, drops as far as the body dropped, and foreshortens the way the real limbs
          did. Only the walk is invented &mdash; and it is solved onto the same measured
          bone lengths, so nothing changes shape between states.
        </div>
      </section>
      <section class="panel">
        <div class="table-head"><h2>How scoring works</h2></div>
        <div class="note" style="border-top:none;">
          The hitbox is read off the <b>measured pose</b>, not off the sprite: circles at the
          lead ankle and lead wrist, live only during the lunge's strike window. The strike
          sweeps forward with the step-out, so range is what the subject's own reach makes it.
          You have to be in range <em>and</em> time the press. Each hit raises the combo;
          letting a target cross the far edge resets it to &times;1.
        </div>
      </section>
    </aside>
  </main>

  <footer id="foot"></footer>
</div>

<script>
const DATA = __DATA_JSON__;
const PT = DATA.points, CLIPS = DATA.clips, RIG = DATA.rig, TUCK = DATA.tuck;
const NP = PT.length, IDX = {};
PT.forEach((name, i) => IDX[name] = i);

/* ---------------------------------------------------------------
   1. A pose is 15 measured joint positions
      Every number below came off a detection: hip-centred x/y pairs in
      game pixels, already scaled by build_game.py so the subject stands
      the right height in this arena. There is no rig here and nothing is
      solved at runtime — the shape on screen is the shape the model saw,
      foreshortening and all. `root` carries how far the hips travelled.
   --------------------------------------------------------------- */
const lerp = (a,b,t) => a + (b-a)*t;
const clamp = (v,lo,hi) => v < lo ? lo : v > hi ? hi : v;

function newPose(){ return { pts:new Float64Array(NP*2), bob:0, root:0 }; }

/* Hip-relative pose -> named joints, hips dropped at world y `oy`. x stays
   local: facing is a canvas flip, see toWorld() and drawFigure(). */
function place(pose, oy){
  const j = {}, a = pose.pts;
  for(let i = 0; i < NP; i++) j[PT[i]] = { x:a[2*i], y:oy + a[2*i+1] };
  return j;
}

/* ---------------------------------------------------------------
   2. Clip sampling
   --------------------------------------------------------------- */
function clipDur(c){ return c.frames.length / c.fps; }

/* Float frame index at time t, honouring the clip's loop mode. */
function frameIndex(c, t){
  const n = c.frames.length;
  let f = t * c.fps;
  if(c.pingpong){
    const span = 2*(n-1);
    f = ((f % span) + span) % span;
    if(f > n-1) f = span - f;                 // reflect: the loop has no seam
  } else if(c.loop){
    f = ((f % n) + n) % n;
  } else {
    f = clamp(f, 0, n-1);
  }
  return f;
}

function samplePose(c, t, out){
  const n = c.frames.length, f = frameIndex(c, t);
  const i0 = Math.floor(f);
  const i1 = c.loop ? (i0+1) % n : Math.min(i0+1, n-1);
  const u = f - i0, a = c.frames[i0], b = c.frames[i1];
  out = out || newPose();
  for(let k = 0; k < NP*2; k++) out.pts[k] = lerp(a[k], b[k], u);
  out.bob  = c.bob  ? lerp(c.bob[i0],  c.bob[i1],  u) : 0;
  out.root = c.root ? lerp(c.root[i0], c.root[i1], u) : 0;
  return out;
}
function copyPose(p, out){
  out = out || newPose();
  out.pts.set(p.pts); out.bob = p.bob; out.root = p.root;
  return out;
}
function blendPose(a, b, t, out){
  out = out || newPose();
  for(let k = 0; k < NP*2; k++) out.pts[k] = lerp(a.pts[k], b.pts[k], t);
  out.bob = lerp(a.bob, b.bob, t);
  out.root = lerp(a.root, b.root, t);
  return out;
}

/* ---------------------------------------------------------------
   3. World
   --------------------------------------------------------------- */
const W = 960, H = 540, GROUND = 466;
const STROKE = RIG.stroke;          // line weights only — joints arrive pre-scaled
const ACCEL = 1500, MAXV = 275, FRICTION = 9, AIR_FRICTION = 1.4;
const GRAV = 2100, JUMPV = -760;
const STRIDE = 132;                 // px of travel per walk cycle — kills foot slide
const FADE = 0.10;                  // state cross-fade, seconds
const LUNGE_COOLDOWN = 0.18;
const ROUND = 60;
const FOOT_R = 40, HAND_R = 32;

/* One-shot attacks: 'lunge' always exists; 'custom' exists only once a move has
   been recorded or baked in. Both are driven by the same state machine below. */
function oneShot(state){ return CLIPS[state]; }
function activeT(c){ return [c.active[0] / c.fps, c.active[1] / c.fps]; }
function isAttacking(p){ return p.state === 'lunge' || p.state === 'custom'; }

const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let dpr = 1, viewScale = 1;
function fit(){
  const r = cv.getBoundingClientRect();
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  viewScale = (r.width || W) / W;
  cv.width  = Math.max(1, Math.round(W * viewScale * dpr));
  cv.height = Math.max(1, Math.round(H * viewScale * dpr));
  cv.style.height = (H * viewScale) + 'px';
}
new ResizeObserver(fit).observe(cv);
fit();

let game, keys = new Set(), paused = false, over = false, started = false;
let best = 0;

function reset(){
  game = {
    p: { x: W*0.5, y: GROUND-150, vx:0, vy:0, dir:1, grounded:true,
         state:'idle', t:0, walkPhase:0, cool:0, hits:new Set(),
         pose:null, rootAnchorX: W*0.5 },
    fade: { from:null, t:1 },
    targets: [], parts: [], pops: [],
    score: 0, combo: 0, time: ROUND, spawn: 0.6, spawned: 0, hits: 0, misses: 0
  };
  over = false; started = false; paused = false;
}
reset();

/* --- pose for the current state ------------------------------------- */
const scratch = newPose(), poseBuf = newPose();
function statePose(p, out){
  if(p.state === 'lunge')       return samplePose(CLIPS.lunge, p.t, out);
  if(p.state === 'custom')      return samplePose(CLIPS.custom, p.t, out);
  if(p.state === 'walk')        return samplePose(CLIPS.walk, p.walkPhase * clipDur(CLIPS.walk), out);
  if(p.state === 'jump'){
    const base = samplePose(CLIPS.idle, p.t, out);
    /* One emitted tuck pose — knees up, arms up — lerped in by how close the
       jump is to its apex, where |vy| is small. */
    const k = clamp(1 - Math.abs(p.vy)/Math.abs(JUMPV), 0.3, 1);
    for(let i = 0; i < NP*2; i++) base.pts[i] = lerp(base.pts[i], TUCK[i], k);
    base.bob = 0; base.root = 0;
    return base;
  }
  return samplePose(CLIPS.idle, p.t, out);
}

function setState(s){
  const p = game.p;
  if(p.state === s) return;
  game.fade.from = copyPose(p.pose || statePose(p, newPose()));
  game.fade.t = 0;
  p.state = s;
  p.t = 0;
  if(s === 'lunge' || s === 'custom'){ p.hits.clear(); p.rootAnchorX = p.x; }
}

/* Hip-to-lowest-foot distance for a pose: the figure stands on its feet
   rather than on an arbitrary baseline, so the lunge's drop is free. With
   measured shapes this lands within ~13 px of the hip height the video
   itself recorded, all the way through the move. */
function stanceOffset(pose){
  return Math.max(pose.pts[IDX.anL*2+1], pose.pts[IDX.anR*2+1]);
}
/* Local rig x -> world x. The clips face -x (that is the direction the
   subject lunges in the video), so facing right is a canvas flip. */
function toWorld(pt, p){ return { x: p.x + (p.dir === 1 ? -pt.x : pt.x), y: pt.y }; }

/* ---------------------------------------------------------------
   4. Simulation
   --------------------------------------------------------------- */
function update(dt){
  const p = game.p;
  p.cool = Math.max(0, p.cool - dt);
  p.t += dt;
  if(game.fade.t < 1) game.fade.t = Math.min(1, game.fade.t + dt/FADE);

  const left  = keys.has('ArrowLeft')  || keys.has('KeyA');
  const right = keys.has('ArrowRight') || keys.has('KeyD');
  const locked = isAttacking(p);                // no steering mid-attack

  let ax = 0;
  if(!locked){ if(left) ax -= ACCEL; if(right) ax += ACCEL; }
  if(ax !== 0) p.dir = ax > 0 ? 1 : -1;
  p.vx += ax*dt;
  if(ax === 0) p.vx -= p.vx * Math.min(1, (p.grounded ? FRICTION : AIR_FRICTION) * dt);
  if(locked)   p.vx -= p.vx * Math.min(1, 14 * dt);
  p.vx = clamp(p.vx, -MAXV, MAXV);
  p.x = clamp(p.x + p.vx*dt, 60, W-60);

  if(!p.grounded){ p.vy += GRAV*dt; p.y += p.vy*dt; }

  if(p.state === 'walk' || (p.state === 'idle' && Math.abs(p.vx) > 4))
    p.walkPhase += (Math.abs(p.vx)/STRIDE) * dt;

  // state machine
  if(isAttacking(p)){
    if(p.t >= clipDur(oneShot(p.state))){ p.cool = LUNGE_COOLDOWN; setState(p.grounded ? 'idle' : 'jump'); }
  } else if(!p.grounded){
    if(p.state !== 'jump') setState('jump');
  } else {
    setState(Math.abs(p.vx) > 18 ? 'walk' : 'idle');
  }

  // pose, then let the feet decide where the hips sit
  const raw = statePose(p, scratch);
  p.pose = game.fade.t < 1 ? blendPose(game.fade.from, raw, game.fade.t, poseBuf)
                           : copyPose(raw, poseBuf);
  /* The lunge drives its own horizontal travel: `root` is how far the hips
     moved on video, in game px, so the step-out covers real ground instead of
     happening on the spot. The clips face -x, so facing right flips the sign. */
  if(isAttacking(p))
    p.x = clamp(p.rootAnchorX + (p.dir === 1 ? -p.pose.root : p.pose.root), 60, W-60);
  const stance = stanceOffset(p.pose);
  if(p.grounded){
    p.y = GROUND - stance + (p.pose.bob || 0);
  } else if(p.vy >= 0 && p.y + stance >= GROUND){
    p.y = GROUND - stance; p.vy = 0; p.grounded = true;
    burst(p.x, GROUND, 10, '#4fd1c5', 0.5);
    if(p.state === 'jump') setState(Math.abs(p.vx) > 18 ? 'walk' : 'idle');
  }

  if(started && !over){
    game.time -= dt;
    if(game.time <= 0){
      game.time = 0; over = true;
      best = Math.max(best, game.score);
      try{ localStorage.setItem('lungeArenaBest', String(best)); }catch(e){}
    }
  }

  updateTargets(dt);
  if(isAttacking(p)){
    const [t0, t1] = activeT(oneShot(p.state));
    if(p.t >= t0 && p.t <= t1) strike();
  }

  for(let i = game.parts.length-1; i >= 0; i--){
    const q = game.parts[i];
    q.vy += 900*dt; q.x += q.vx*dt; q.y += q.vy*dt; q.life -= dt;
    if(q.life <= 0) game.parts.splice(i,1);
  }
  for(let i = game.pops.length-1; i >= 0; i--){
    const q = game.pops[i]; q.y -= 34*dt; q.life -= dt;
    if(q.life <= 0) game.pops.splice(i,1);
  }
}

function updateTargets(dt){
  if(!over && started){
    game.spawn -= dt;
    if(game.spawn <= 0){
      spawnTarget();
      const heat = 1 - clamp(game.time/ROUND, 0, 1);       // slowly busier
      game.spawn = lerp(1.35, 0.62, heat) * (0.8 + Math.random()*0.5);
    }
  }
  for(let i = game.targets.length-1; i >= 0; i--){
    const t = game.targets[i];
    t.x += t.vx*dt; t.phase += dt*t.wob;
    t.y = t.y0 + Math.sin(t.phase)*t.amp;
    if(t.x < -60 || t.x > W+60){
      game.targets.splice(i,1);
      if(!over){ game.misses++; if(game.combo > 1) pop(clamp(t.x,40,W-40), t.y0, 'combo lost', '#e2574c'); game.combo = 0; }
    }
  }
}

function spawnTarget(){
  const fromLeft = Math.random() < 0.5;
  const y0 = lerp(GROUND-38, GROUND-232, Math.random());
  const speed = lerp(58, 132, Math.random());
  game.targets.push({
    x: fromLeft ? -40 : W+40, y: y0, y0,
    vx: fromLeft ? speed : -speed,
    r: lerp(15, 21, Math.random()),
    phase: Math.random()*6.28, wob: lerp(0.9, 2.1, Math.random()), amp: lerp(6, 26, Math.random()),
    id: game.spawned++
  });
}

/* Hitbox comes off the measured pose: whichever ankle and wrist are furthest
   forward for the current facing, live only inside the clip's strike window. */
function strike(){
  const p = game.p;
  const j = place(p.pose, p.y);
  const foot = pickLead(j.anL, j.anR, p), hand = pickLead(j.wrL, j.wrR, p);
  for(const t of game.targets){
    if(p.hits.has(t.id)) continue;
    if(hit(foot, t, FOOT_R) || hit(hand, t, HAND_R)){
      p.hits.add(t.id);
      t.dead = true;
      game.combo++; game.hits++;
      const mult = Math.min(game.combo, 9);
      game.score += 100*mult;
      burst(t.x, t.y, 22, '#f2b544', 0.75);
      pop(t.x, t.y, '+' + (100*mult) + (mult > 1 ? '  x' + mult : ''), '#f2b544');
    }
  }
  game.targets = game.targets.filter(t => !t.dead);
}
function pickLead(a, b, p){
  const wa = toWorld(a, p), wb = toWorld(b, p);
  return (p.dir*wa.x > p.dir*wb.x) ? wa : wb;
}
function hit(pt, t, r){
  const dx = pt.x - t.x, dy = pt.y - t.y;
  return dx*dx + dy*dy <= (r + t.r)*(r + t.r);
}
function burst(x, y, n, col, life){
  for(let i = 0; i < n; i++){
    const a = Math.random()*6.283, sp = lerp(40, 300, Math.random());
    game.parts.push({x, y, vx:Math.cos(a)*sp, vy:Math.sin(a)*sp - 60, life:life*(0.5+Math.random()), col});
  }
}
function pop(x, y, text, col){ game.pops.push({x, y, text, col, life:0.9}); }

/* ---------------------------------------------------------------
   5. Drawing
   --------------------------------------------------------------- */
function draw(){
  ctx.setTransform(viewScale*dpr, 0, 0, viewScale*dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d1116'; ctx.fillRect(0, 0, W, H);

  // grid, matching the scrubber's stage
  ctx.strokeStyle = '#1b222a'; ctx.lineWidth = 1;
  ctx.beginPath();
  for(let x = 0; x <= W; x += 60){ ctx.moveTo(x+0.5, 0); ctx.lineTo(x+0.5, H); }
  for(let y = 0; y <= H; y += 60){ ctx.moveTo(0, y+0.5); ctx.lineTo(W, y+0.5); }
  ctx.stroke();

  // ground
  ctx.strokeStyle = '#2f6b66'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(0, GROUND+0.5); ctx.lineTo(W, GROUND+0.5); ctx.stroke();
  ctx.fillStyle = 'rgba(79,209,197,0.045)'; ctx.fillRect(0, GROUND+1, W, H-GROUND-1);

  const p = game.p;
  let active = false;
  if(isAttacking(p)){
    const [t0, t1] = activeT(oneShot(p.state));
    active = p.t >= t0 && p.t <= t1;
  }

  for(const t of game.targets) drawTarget(t);
  drawFigure(ctx, p, active);

  for(const q of game.parts){
    ctx.globalAlpha = clamp(q.life, 0, 1);
    ctx.fillStyle = q.col;
    ctx.fillRect(q.x-1.6, q.y-1.6, 3.2, 3.2);
  }
  ctx.globalAlpha = 1;

  ctx.textAlign = 'center'; ctx.font = "500 15px 'IBM Plex Mono', monospace";
  for(const q of game.pops){
    ctx.globalAlpha = clamp(q.life/0.9, 0, 1);
    ctx.fillStyle = q.col;
    ctx.fillText(q.text, q.x, q.y);
  }
  ctx.globalAlpha = 1;

  drawOverlays();
}

function drawTarget(t){
  ctx.save();
  ctx.strokeStyle = '#f2b544'; ctx.fillStyle = 'rgba(242,181,68,0.10)';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(t.x, t.y, t.r, 0, 6.2832); ctx.fill(); ctx.stroke();
  ctx.globalAlpha = 0.55;
  ctx.beginPath(); ctx.arc(t.x, t.y, t.r*0.42, 0, 6.2832); ctx.stroke();
  ctx.globalAlpha = 0.22;
  ctx.beginPath(); ctx.moveTo(t.x - Math.sign(t.vx)*t.r, t.y); ctx.lineTo(t.x - Math.sign(t.vx)*(t.r+26), t.y); ctx.stroke();
  ctx.restore();
}

function drawFigure(c, p, active){
  if(!p.pose) return;
  const j = place(p.pose, p.y);
  const s = STROKE;

  // shadow, sized by height off the ground
  const lowest = Math.max(j.anL.y, j.anR.y);
  const air = clamp(1 - (GROUND - lowest)/220, 0.18, 1);
  c.save();
  c.globalAlpha = 0.28*air; c.fillStyle = '#000';
  c.beginPath(); c.ellipse(p.x, GROUND+3, 42*air, 6*air, 0, 0, 6.2832); c.fill();
  c.restore();

  c.save();
  c.translate(p.x, 0);
  c.scale(p.dir === 1 ? -1 : 1, 1);      // facing is a canvas flip, nothing else

  const line = (a,b,w) => { c.lineWidth = w*s; c.beginPath(); c.moveTo(a.x,a.y); c.lineTo(b.x,b.y); c.stroke(); };
  c.lineCap = 'round';
  c.strokeStyle = '#4fd1c5';

  line(j.hipL, j.hipR, 5); line(j.hipC, j.shC, 7); line(j.shL, j.shR, 5);
  line(j.shL, j.elL, 5); line(j.elL, j.wrL, 4);
  line(j.shR, j.elR, 5); line(j.elR, j.wrR, 4);
  line(j.hipL, j.knL, 6); line(j.knL, j.anL, 5);
  line(j.hipR, j.knR, 6); line(j.knR, j.anR, 5);

  c.lineWidth = 5*s;
  c.beginPath(); c.arc(j.headC.x, j.headC.y, RIG.headR, 0, 6.2832); c.stroke();
  line(j.shC, j.headC, 4);

  c.fillStyle = '#1d2938';
  for(const k of ['elL','elR','wrL','wrR','knL','knR','anL','anR','shC','hipC']){
    c.beginPath(); c.arc(j[k].x, j[k].y, 3.2*s, 0, 6.2832); c.fill();
    c.lineWidth = 2*s; c.stroke();
  }
  c.restore();

  if(active){
    const foot = pickLead(j.anL, j.anR, p), hand = pickLead(j.wrL, j.wrR, p);
    c.save();
    c.strokeStyle = '#f2b544'; c.fillStyle = 'rgba(242,181,68,0.10)'; c.lineWidth = 2;
    for(const [pt, r] of [[foot, FOOT_R], [hand, HAND_R]]){
      c.beginPath(); c.arc(pt.x, pt.y, r, 0, 6.2832); c.fill(); c.stroke();
    }
    c.restore();
  }
}

function drawOverlays(){
  const msg = over ? 'ROUND OVER'
            : paused ? 'PAUSED'
            : !started ? 'PRESS ANY KEY' : null;
  if(!msg) return;
  ctx.save();
  ctx.fillStyle = 'rgba(13,17,22,0.78)'; ctx.fillRect(0, 0, W, H);
  ctx.textAlign = 'center';
  ctx.fillStyle = over ? '#f2b544' : '#e8edf1';
  ctx.font = "700 38px 'Sora', system-ui, sans-serif";
  ctx.fillText(msg, W/2, H/2 - 12);
  ctx.font = "400 15px 'IBM Plex Mono', monospace";
  ctx.fillStyle = '#8d99a6';
  const sub = over ? game.score + ' points  ' + game.hits + ' hits  ' + game.misses + ' missed   —   R to play again'
            : paused ? 'P to resume'
            : 'move with the arrow keys, jump with space, lunge with F';
  ctx.fillText(sub, W/2, H/2 + 24);
  ctx.restore();
}

/* ---------------------------------------------------------------
   6. HUD
   --------------------------------------------------------------- */
const el = id => document.getElementById(id);
const hud = { score:el('statScore'), combo:el('statCombo'), time:el('statTime'), best:el('statBest'),
              state:el('roState'), frame:el('roFrame'), strike:el('roStrike') };
let hudCache = {};
function setText(node, key, value){
  if(hudCache[key] === value) return;
  hudCache[key] = value; node.textContent = value;
}
function drawHud(){
  const p = game.p;
  setText(hud.score, 'score', String(game.score));
  setText(hud.combo, 'combo', '×' + Math.max(1, Math.min(game.combo, 9)));
  setText(hud.time,  'time',  game.time.toFixed(1));
  setText(hud.best,  'best',  String(best));
  hud.time.classList.toggle('low', game.time <= 10);
  setText(hud.state, 'state', p.state);
  const attacking = isAttacking(p);
  const c = attacking ? oneShot(p.state) : null;
  setText(hud.frame, 'frame', attacking
    ? (Math.floor(frameIndex(c, p.t)) + 1) + '/' + c.frames.length
    : '—');
  let active = false;
  if(attacking){ const [t0, t1] = activeT(c); active = p.t >= t0 && p.t <= t1; }
  hud.strike.classList.toggle('on', active);
  for(const row of clipRows) row.el.classList.toggle('active', row.key === p.state);
}

const clipRows = [];
function buildClipList(){
  const list = el('clipList');
  list.innerHTML = '';
  clipRows.length = 0;
  let total = 0;
  for(const key of Object.keys(CLIPS)){
    const c = CLIPS[key];
    total += c.frames.length;
    const row = document.createElement('div');
    row.className = 'clip-row';
    const measured = c.source !== 'procedural';
    row.innerHTML =
      '<div class="clip-name"><span class="dot"></span>' + key +
        ' <span class="badge ' + (measured ? 'meas' : 'proc') + '">' + (measured ? 'measured' : 'procedural') + '</span></div>' +
      '<div class="clip-frames">' + c.frames.length + ' fr &middot; ' + clipDur(c).toFixed(2) + 's</div>' +
      '<div class="clip-src">' + c.note + '</div>';
    list.appendChild(row);
    clipRows.push({key, el:row});
  }
  el('clipCount').textContent = total + ' frames total';
  el('foot').innerHTML =
    'Built by <code>build_game.py</code> from <code>lunge_all_frames.json</code> &mdash; ' +
    DATA.meta.detected + ' detections across ' + DATA.meta.n_frames + ' source frames at ' +
    DATA.meta.fps.toFixed(2) + ' fps. Joint positions are the measured ones, scaled by ' +
    DATA.meta.scale.toFixed(4) + '; drawing ported from <code>pose-stick-figure-rfdetr.html</code>. ' +
    'Self-contained: the only external request is the Google Fonts stylesheet.';
}
buildClipList();

/* ---------------------------------------------------------------
   7. Input + loop
   --------------------------------------------------------------- */
const HELD = new Set(['ArrowLeft','ArrowRight','KeyA','KeyD','Space','ArrowUp','ArrowDown']);
addEventListener('keydown', e => {
  if(HELD.has(e.code)) e.preventDefault();
  if(e.repeat) return;
  if(!started && !over){ started = true; }
  keys.add(e.code);
  if(e.code === 'KeyR'){ const b = best; reset(); best = b; return; }
  if(e.code === 'KeyP'){ if(!over) paused = !paused; return; }
  if(over) return;
  if(paused) return;
  const p = game.p;
  if((e.code === 'Space' || e.code === 'ArrowUp') && p.grounded && !isAttacking(p)){
    p.vy = JUMPV; p.grounded = false; setState('jump');
    burst(p.x, GROUND, 8, '#4fd1c5', 0.35);
  }
  if(e.code === 'KeyF' && !isAttacking(p) && p.cool <= 0){ setState('lunge'); }
  if(e.code === 'KeyC' && !isAttacking(p) && p.cool <= 0){
    if(CLIPS.custom) setState('custom');
    else pop(p.x, p.y - 140, 'record a move first', '#8d99a6');
  }
});
addEventListener('keyup', e => keys.delete(e.code));
addEventListener('blur', () => { keys.clear(); if(!over) paused = true; });

try{ best = parseInt(localStorage.getItem('lungeArenaBest') || '0', 10) || 0; }catch(e){}

let last = performance.now();
function frame(now){
  const dt = Math.min(0.05, (now - last)/1000);
  last = now;
  if(!paused && !over) update(dt);
  draw();
  drawHud();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>

<script>
/* ---------------------------------------------------------------
   Recorder: capture or upload a move, analyze it on serve_game.py
   with the exact same pipeline that produced the lunge, trim it,
   and bind it to CLIPS.custom. Everything below reuses the arena's
   own globals (CLIPS, place, drawFigure, samplePose, stanceOffset,
   clipDur, frameIndex, W, H, GROUND, clamp, el, buildClipList) —
   classic <script> tags share one top-level scope, so no wiring is
   needed to reach them.
   --------------------------------------------------------------- */
const REC = {
  stream:null, frameW:640, frameH:480,
  clipResp:null, n:0, inF:0, outF:0, strikeLo:0, strikeHi:0, fps:30,
  baseFlip:false, userFlip:false,
};
const sleep = ms => new Promise(res => setTimeout(res, ms));
const clampInt = (v,lo,hi) => v < lo ? lo : v > hi ? hi : v;

function mirrorClipFrames(frames){
  return frames.map(fr => fr.map((v,i) => i % 2 === 0 ? -v : v));
}

/* --- server availability -------------------------------------- */
async function checkServer(){
  try{ const r = await fetch('/health', {cache:'no-store'}); return r.ok; }
  catch(e){ return false; }
}

/* --- modal open/close ------------------------------------------- */
function showRecStage(name){
  el('recStageCapture').hidden = name !== 'capture';
  el('recStageAnalyze').hidden = name !== 'analyze';
  el('recStageTrim').hidden    = name !== 'trim';
  el('recTitle').textContent = name === 'capture' ? 'Record a move'
                              : name === 'analyze' ? 'Analyzing…' : 'Trim and preview';
}
function stopCameraStream(){
  if(REC.stream){ REC.stream.getTracks().forEach(t => t.stop()); REC.stream = null; }
}
function resetCaptureUI(){
  el('recGo').disabled = true;
  el('recStartCam').disabled = false;
  el('recUploadPanel').hidden = true;
  el('recTimeFill').style.width = '0%';
  el('recCountdown').textContent = '';
  const video = el('recVideo');
  video.pause(); video.removeAttribute('src'); video.srcObject = null; video.style.transform = '';
  stopOverlayLoop();
}
function openRecModal(){
  el('recModal').hidden = false;
  showRecStage('capture');
}
function closeRecModal(){
  el('recModal').hidden = true;
  stopOverlayLoop();
  stopCameraStream();
  resetCaptureUI();
}
el('recOpen').addEventListener('click', () => { if(!el('recOpen').disabled) openRecModal(); });
el('recClose').addEventListener('click', closeRecModal);

/* --- live MediaPipe overlay: preview only ------------------------
   The move itself never comes from this — it's RF-DETR, on the
   server, once you record. If the CDN is unreachable this silently
   drops out and recording still works. */
const REC_L = { nose:0, earL:7, earR:8, shL:11, shR:12, elL:13, elR:14,
                wrL:15, wrR:16, hipL:23, hipR:24, knL:25, knR:26, anL:27, anR:28 };
const REC_EDGES = [
  [REC_L.shL,REC_L.shR],[REC_L.shL,REC_L.elL],[REC_L.elL,REC_L.wrL],
  [REC_L.shR,REC_L.elR],[REC_L.elR,REC_L.wrR],
  [REC_L.shL,REC_L.hipL],[REC_L.shR,REC_L.hipR],[REC_L.hipL,REC_L.hipR],
  [REC_L.hipL,REC_L.knL],[REC_L.knL,REC_L.anL],[REC_L.hipR,REC_L.knR],[REC_L.knR,REC_L.anR],
];
let recLandmarker = null, recOverlayRunning = false;
async function ensureLandmarker(){
  if(recLandmarker) return recLandmarker;
  const mod = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14');
  const files = await mod.FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
  recLandmarker = await mod.PoseLandmarker.createFromOptions(files, {
    baseOptions:{ modelAssetPath:'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
                 delegate:'GPU' },
    runningMode:'VIDEO', numPoses:1,
  });
  return recLandmarker;
}
function drawOverlaySkeleton(c2, lm, w, h){
  c2.strokeStyle = '#4fd1c5'; c2.fillStyle = '#4fd1c5'; c2.lineWidth = 2.5; c2.lineCap = 'round';
  c2.beginPath();
  for(const [a,b] of REC_EDGES){
    const pa = lm[a], pb = lm[b];
    if(!pa || !pb) continue;
    if((pa.visibility !== undefined && pa.visibility < 0.4) ||
       (pb.visibility !== undefined && pb.visibility < 0.4)) continue;
    c2.moveTo(pa.x*w, pa.y*h); c2.lineTo(pb.x*w, pb.y*h);
  }
  c2.stroke();
  for(const i of Object.values(REC_L)){
    const p = lm[i];
    if(!p || (p.visibility !== undefined && p.visibility < 0.4)) continue;
    c2.beginPath(); c2.arc(p.x*w, p.y*h, 3, 0, 6.2832); c2.fill();
  }
}
function startOverlayLoop(){
  if(recOverlayRunning) return;
  recOverlayRunning = true;
  const video = el('recVideo'), cv2 = el('recOverlay'), c2 = cv2.getContext('2d');
  let lastVT = -1;
  async function tick(){
    if(!recOverlayRunning) return;
    const r = cv2.getBoundingClientRect(), d = Math.min(devicePixelRatio || 1, 2);
    cv2.width = Math.max(1, r.width*d); cv2.height = Math.max(1, r.height*d);
    c2.setTransform(d,0,0,d,0,0);
    c2.clearRect(0, 0, r.width, r.height);
    try{
      const lm = await ensureLandmarker();
      if(video.readyState >= 2 && video.currentTime !== lastVT){
        lastVT = video.currentTime;
        const res = lm.detectForVideo(video, performance.now());
        if(res.landmarks && res.landmarks[0]) drawOverlaySkeleton(c2, res.landmarks[0], r.width, r.height);
      }
    }catch(e){ /* CDN or model unavailable — overlay just doesn't draw */ }
    if(recOverlayRunning) requestAnimationFrame(tick);
  }
  tick();
}
function stopOverlayLoop(){ recOverlayRunning = false; }

/* --- capture: camera --------------------------------------------- */
el('recStartCam').addEventListener('click', async () => {
  try{
    const stream = await navigator.mediaDevices.getUserMedia({video:{width:640, height:480}});
    REC.stream = stream;
    const video = el('recVideo');
    video.srcObject = stream;
    video.style.transform = '';
    await video.play();
    el('recGo').disabled = false;
    el('recStartCam').disabled = true;
    startOverlayLoop();
  }catch(e){
    el('recServerNote').hidden = false;
    el('recServerNote').textContent = 'camera unavailable: ' + e.message;
  }
});
el('recGo').addEventListener('click', () => {
  el('recGo').disabled = true;
  runCountdownThenCapture();
});
async function runCountdownThenCapture(){
  const cd = el('recCountdown');
  for(const label of ['3','2','1','GO!']){ cd.textContent = label; await sleep(600); }
  cd.textContent = '';
  await captureFrames(el('recVideo'), null);
}

/* --- capture: upload ---------------------------------------------- */
el('recUploadBtn').addEventListener('click', () => el('recFileInput').click());
el('recFileInput').addEventListener('change', async () => {
  const f = el('recFileInput').files && el('recFileInput').files[0];
  if(!f) return;
  stopCameraStream(); stopOverlayLoop();
  const video = el('recVideo');
  video.srcObject = null;
  video.style.transform = 'none';
  video.src = URL.createObjectURL(f);
  video.loop = true;
  await new Promise(res => { video.onloadedmetadata = res; });
  const maxStart = Math.max(0, video.duration - 4);
  const startInput = el('recUploadStart');
  startInput.max = maxStart.toFixed(2); startInput.value = '0';
  updateUploadWindowLabel();
  el('recUploadPanel').hidden = false;
  el('recGo').disabled = true;
  video.currentTime = 0;
  await video.play().catch(() => {});
});
function updateUploadWindowLabel(){
  const s = +el('recUploadStart').value, video = el('recVideo');
  el('recUploadWindow').textContent =
    s.toFixed(1) + 's – ' + (s+4).toFixed(1) + 's  /  ' + video.duration.toFixed(1) + 's';
}
el('recUploadStart').addEventListener('input', () => {
  updateUploadWindowLabel();
  el('recVideo').currentTime = +el('recUploadStart').value;
});
el('recUploadUse').addEventListener('click', async () => {
  const video = el('recVideo'), startT = +el('recUploadStart').value;
  el('recUploadPanel').hidden = true;
  video.loop = false;
  await captureFrames(video, startT);
});
function seekTo(video, t){
  return new Promise(res => {
    const onSeeked = () => { video.removeEventListener('seeked', onSeeked); res(); };
    video.addEventListener('seeked', onSeeked);
    video.currentTime = t;
  });
}

/* --- capture -> blobs, live from the camera or seeked from a file - */
async function captureFrames(video, uploadStartT){
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.min(1, 640 / Math.max(vw, vh));
  REC.frameW = Math.max(1, Math.round(vw*scale));
  REC.frameH = Math.max(1, Math.round(vh*scale));
  const grabCv = document.createElement('canvas');
  grabCv.width = REC.frameW; grabCv.height = REC.frameH;
  const grabCtx = grabCv.getContext('2d', {willReadFrequently:false});
  const blobs = [];
  const FPS = 30, CAP_MS = 4000, MAX_FRAMES = 120;

  if(uploadStartT === null){
    /* live: aim for a steady 30fps clock, hard-stopped at 4.0s. The pacer
       only holds if drawImage+toBlob fits in 33ms; on a slow machine it
       falls behind and the real rate drops. The clip's fps has to be the
       rate the frames were actually taken at, not the rate we asked for,
       or the move replays faster than it happened. */
    const t0 = performance.now();
    let nextT = t0;
    while(performance.now() - t0 < CAP_MS && blobs.length < MAX_FRAMES){
      grabCtx.drawImage(video, 0, 0, grabCv.width, grabCv.height);
      blobs.push(await new Promise(res => grabCv.toBlob(res, 'image/jpeg', 0.8)));
      el('recTimeFill').style.width = Math.min(100, 100*(performance.now()-t0)/CAP_MS) + '%';
      nextT += 1000/FPS;
      const wait = nextT - performance.now();
      if(wait > 0) await sleep(wait);
    }
    const elapsed = (performance.now() - t0) / 1000;
    REC.fps = elapsed > 0 ? blobs.length / elapsed : FPS;
  }else{
    // upload: pull frames by seeking through the chosen window
    for(let i = 0; i < MAX_FRAMES; i++){
      const t = uploadStartT + i/FPS;
      if(t > video.duration) break;
      await seekTo(video, t);
      grabCtx.drawImage(video, 0, 0, grabCv.width, grabCv.height);
      blobs.push(await new Promise(res => grabCv.toBlob(res, 'image/jpeg', 0.8)));
      el('recTimeFill').style.width = Math.min(100, 100*(i+1)/(FPS*4)) + '%';
    }
    REC.fps = FPS;   // seeked to exactly i/FPS, so this one is exact
  }
  el('recTimeFill').style.width = '100%';
  stopOverlayLoop();
  const payloadFrames = await analyzeFrames(blobs);
  await finishAnalysis(payloadFrames);
}

/* --- analyze: one frame in flight at a time, same as the live pump  */
async function postKeypoints(blob){
  const r = await fetch('/keypoints', {method:'POST', headers:{'Content-Type':'image/jpeg'}, body:blob});
  if(!r.ok) throw new Error('server ' + r.status);
  return r.json();
}
async function analyzeFrames(blobs){
  showRecStage('analyze');
  el('recProgFill').style.width = '0%';
  const results = [];
  const t0 = performance.now();
  for(let i = 0; i < blobs.length; i++){
    let payload;
    try{ payload = await postKeypoints(blobs[i]); }
    catch(e){ payload = {people:[]}; }
    results.push(payload);
    const elapsed = (performance.now() - t0) / 1000;
    const eta = Math.max(0, (elapsed/(i+1)) * (blobs.length-i-1));
    el('recProgFill').style.width = (100*(i+1)/blobs.length) + '%';
    el('recProgText').textContent = 'frame ' + (i+1) + ' / ' + blobs.length + '  —  ETA ' + eta.toFixed(0) + 's';
  }
  return results;
}
async function finishAnalysis(payloadFrames){
  el('recProgText').textContent = 'building the clip…';
  try{
    const r = await fetch('/clip', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ frames: payloadFrames, w: REC.frameW, h: REC.frameH, fps: REC.fps, smooth: 3 }),
    });
    if(!r.ok) throw new Error('server ' + r.status);
    const resp = await r.json();
    openTrimStage(resp);
  }catch(e){
    el('recProgText').textContent = 'error building the clip: ' + e.message;
  }
}

/* --- trim + preview ------------------------------------------------ */
let recPreviewRunning = false;
function openTrimStage(resp){
  REC.clipResp = resp;
  REC.n = resp.clip.frames.length;
  REC.inF = 0; REC.outF = REC.n - 1;
  REC.strikeLo = resp.clip.active[0]; REC.strikeHi = resp.clip.active[1];
  REC.baseFlip = resp.flip; REC.userFlip = resp.flip;
  el('recFlip').checked = resp.flip;
  el('recMoveName').value = 'custom';
  renderTicks(resp.diagnostics);
  layoutHandles();
  showRecStage('trim');
  if(!recPreviewRunning) startPreviewLoop();
}
function renderTicks(diag){
  const wrap = el('recTicks');
  wrap.innerHTML = '';
  for(const d of diag){
    const i = document.createElement('i');
    if(!d.detected) i.className = 'miss';
    wrap.appendChild(i);
  }
}
function pct(i){ return (100 * i / Math.max(1, REC.n - 1)) + '%'; }
function layoutHandles(){
  const span = Math.max(1, REC.n - 1);
  el('recInHandle').style.left = pct(REC.inF);
  el('recOutHandle').style.left = pct(REC.outF);
  el('recStrikeLoHandle').style.left = pct(REC.strikeLo);
  el('recStrikeHiHandle').style.left = pct(REC.strikeHi);
  el('recFill').style.left = pct(REC.inF);
  el('recFill').style.width = (100*(REC.outF-REC.inF)/span) + '%';
  el('recStrikeFill').style.left = pct(REC.strikeLo);
  el('recStrikeFill').style.width = (100*(REC.strikeHi-REC.strikeLo)/span) + '%';
  const fps = REC.clipResp.clip.fps;
  el('recWindowMeta').textContent = 'window ' + (REC.inF+1) + '–' + (REC.outF+1) + ' / ' + REC.n +
    '  (' + ((REC.outF-REC.inF+1)/fps).toFixed(2) + 's)';
  el('recStrikeMeta').textContent = 'strike ' + (REC.strikeLo+1) + '–' + (REC.strikeHi+1);
  el('recPreviewFrames').textContent = (REC.outF-REC.inF+1) + ' frames selected';
}
function applyHandleMove(kind, i){
  if(kind === 'in') REC.inF = clampInt(i, 0, REC.outF - 1);
  else if(kind === 'out') REC.outF = clampInt(i, REC.inF + 1, REC.n - 1);
  else if(kind === 'strikeLo') REC.strikeLo = clampInt(i, REC.inF, REC.strikeHi);
  else if(kind === 'strikeHi') REC.strikeHi = clampInt(i, REC.strikeLo, REC.outF);
}
function wireHandle(handleId, kind){
  const handle = el(handleId);
  handle.addEventListener('pointerdown', e => {
    handle.setPointerCapture(e.pointerId);
    const move = ev => {
      const rect = el('recRange').getBoundingClientRect();
      const frac = clamp((ev.clientX - rect.left) / rect.width, 0, 1);
      applyHandleMove(kind, Math.round(frac * (REC.n - 1)));
      layoutHandles();
    };
    const up = () => {
      handle.releasePointerCapture(e.pointerId);
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', up);
    };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', up);
  });
}
wireHandle('recInHandle', 'in');
wireHandle('recOutHandle', 'out');
wireHandle('recStrikeLoHandle', 'strikeLo');
wireHandle('recStrikeHiHandle', 'strikeHi');
el('recFlip').addEventListener('change', () => { REC.userFlip = el('recFlip').checked; });

function startPreviewLoop(){
  recPreviewRunning = true;
  const cv2 = el('recPreview'), c2 = cv2.getContext('2d');
  let last = performance.now(), t = 0;
  const pp = { x: W/2, y: 0, dir: 1, pose: null };
  function tick(now){
    if(el('recStageTrim').hidden){ recPreviewRunning = false; return; }
    const dt = Math.min(0.05, (now-last)/1000); last = now;
    const frames = REC.userFlip === REC.baseFlip ? REC.clipResp.clip.frames : mirrorClipFrames(REC.clipResp.clip.frames);
    const sliced = { fps: REC.clipResp.clip.fps, loop: true, frames: frames.slice(REC.inF, REC.outF + 1) };
    t = (t + dt) % Math.max(1/60, clipDur(sliced));
    const pose = samplePose(sliced, t);
    pp.pose = pose;
    pp.y = GROUND - stanceOffset(pose);
    const localLo = clampInt(REC.strikeLo - REC.inF, 0, sliced.frames.length - 1);
    const localHi = clampInt(REC.strikeHi - REC.inF, 0, sliced.frames.length - 1);
    const fi = Math.round(frameIndex(sliced, t));
    const active = fi >= localLo && fi <= localHi;

    c2.setTransform(1,0,0,1,0,0);
    c2.clearRect(0, 0, W, H);
    c2.fillStyle = '#0d1116'; c2.fillRect(0, 0, W, H);
    c2.strokeStyle = '#2f6b66'; c2.lineWidth = 2;
    c2.beginPath(); c2.moveTo(0, GROUND+0.5); c2.lineTo(W, GROUND+0.5); c2.stroke();
    drawFigure(c2, pp, active);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function buildFinalClip(){
  const src = REC.clipResp.clip;
  const frames0 = REC.userFlip === REC.baseFlip ? src.frames : mirrorClipFrames(src.frames);
  const root0 = REC.userFlip === REC.baseFlip ? src.root : src.root.map(v => -v);
  const lo = REC.inF, hi = REC.outF;
  const frames = frames0.slice(lo, hi + 1);
  const rootWindow = root0.slice(lo, hi + 1);
  const r0 = rootWindow[0];
  const root = rootWindow.map(v => Math.round((v - r0) * 10) / 10);
  const active = [clampInt(REC.strikeLo - lo, 0, frames.length - 1),
                  clampInt(REC.strikeHi - lo, 0, frames.length - 1)];
  return { fps: src.fps, frames, root, active, source: src.source, note: src.note, measured: true };
}
async function saveMove(){
  const clip = buildFinalClip();
  const name = el('recMoveName').value.trim() || 'custom';
  try{
    await fetch('/moves', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name, clip}),
    });
  }catch(e){ /* server may be gone; the move still lives in localStorage below */ }
  try{ localStorage.setItem('lungeArenaCustomMove', JSON.stringify(clip)); }catch(e){}
  CLIPS.custom = clip;
  buildClipList();
  closeRecModal();
}
el('recSave').addEventListener('click', async () => {
  el('recSave').disabled = true;
  try{ await saveMove(); }
  finally{ el('recSave').disabled = false; }
});
el('recRetry').addEventListener('click', () => {
  resetCaptureUI();
  showRecStage('capture');
});

/* --- boot: server check, and a move already saved on disk/localStorage ---
   The server's custom_moves.json is the move's real home — it survives a
   cleared browser profile or a different browser entirely. localStorage is
   only the fallback for when the server isn't running. */
(async function initRecorder(){
  const ok = await checkServer();
  if(!CLIPS.custom){
    let loaded = false;
    if(ok){
      try{
        const r = await fetch('/moves', {cache:'no-store'});
        if(r.ok){
          const store = await r.json();
          const cur = store.current && store.moves[store.current];
          if(cur){ CLIPS.custom = cur.clip; buildClipList(); loaded = true; }
        }
      }catch(e){}
    }
    if(!loaded){
      try{
        const raw = localStorage.getItem('lungeArenaCustomMove');
        if(raw){ CLIPS.custom = JSON.parse(raw); buildClipList(); }
      }catch(e){}
    }
  }
  if(!ok){
    el('recOpen').disabled = true;
    el('recOpen').title = 'run python3 lunge_pose_demo/serve_game.py to record';
    el('recOpenNote').hidden = false;
    el('recOpenNote').textContent = 'run serve_game.py to record';
  }
})();
</script>
""".replace("__DATA_JSON__", data_json)


# --------------------------------------------------------------------------
# 5. Driver
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(HERE / "lunge_all_frames.json"),
                    help="per-frame keypoint JSON from build_sequence_viewer.py")
    ap.add_argument("--idle", default="79:106", metavar="LO:HI",
                    help="source frames for the idle loop (ping-ponged at runtime); also "
                         "the window the scale and the bone lengths are measured on")
    ap.add_argument("--lunge", default="106:134,154:188", metavar="LO:HI,LO:HI",
                    help="source frames for the lunge: strike segment,recovery segment")
    ap.add_argument("--hold", type=int, default=4, help="frames held at the deepest pose")
    ap.add_argument("--speed", type=float, default=1.4, help="playback rate of the captured clips")
    ap.add_argument("--walk-frames", type=int, default=24, help="length of the procedural walk cycle")
    ap.add_argument("--smooth", type=int, default=3, help="moving-average window, in frames")
    ap.add_argument("--scale", type=float, default=None,
                    help="source px -> game px (default: auto, sizing the subject to "
                         f"{FIGURE_H:.0f} px tall)")
    ap.add_argument("--out", default=str(HERE / "lunge_game.html"))
    ap.add_argument("--custom", default=None, metavar="custom_moves.json",
                    help="bake a recorded move (written by serve_game.py) into the page as "
                         "CLIPS.custom, so it plays with no server and no browser storage")
    args = ap.parse_args()

    src = json.loads(Path(args.data).read_text(encoding="utf-8"))
    frames = src["frames"]
    vid = src["video"]
    fps = vid["fps"]
    detected = sum(1 for f in frames if f.get("people"))
    print(f"read {Path(args.data).name}: {len(frames)} frames, {detected} with a detection, "
          f"{vid['w']}x{vid['h']} @ {fps:.2f}fps")

    tracked = track_subject(frames, vid["w"], vid["h"])
    multi = sum(1 for f in frames if len(f.get("people") or []) > 1)
    print(f"tracked one subject across {sum(1 for k in tracked if k):d} detections "
          f"({multi} frames offered more than one)")

    raw_pts = point_tracks(tracked)
    pts = smooth_points(raw_pts, args.smooth)
    print(f"points: {len(POINTS)} joints x {len(frames)} frames, "
          f"smoothed over {args.smooth} frames")

    idle_lo, idle_hi = parse_range(args.idle, "idle")
    try:
        strike_txt, return_txt = args.lunge.split(",")
    except ValueError:
        raise SystemExit(f"--lunge: expected LO:HI,LO:HI, got {args.lunge!r}")
    s_lo, s_hi = parse_range(strike_txt, "lunge")
    r_lo, r_hi = parse_range(return_txt, "lunge")

    n = len(frames)
    for name, hi in (("idle", idle_hi), ("lunge", max(s_hi, r_hi))):
        if hi > n - 1:
            raise SystemExit(f"--{name}: frame {hi} is past the end of the clip ({n} frames)")

    # --- scale: the subject, measured standing, sized into the arena ---
    rig = measure_rig(pts, tracked, idle_lo, idle_hi)
    k = args.scale if args.scale else FIGURE_H / rig["height"]
    bones = ", ".join(f"{b} {rig[b] * k:.1f}" for b in
                      ("spine", "thigh", "shin", "upperArm", "forearm"))
    how = ("(--scale)" if args.scale else
           f"= {FIGURE_H:.0f} px figure / {rig['height']:.1f} px standing height")
    print(f"scale: {k:.4f} {how}, drawing the subject {rig['height'] * k:.0f} px tall  "
          f"[{bones}]")

    # Angles are still how the deepest lunge frame is found and how the two
    # invented clips are authored; they are no longer what anything is stored as.
    angles, knee = extract_channels(tracked)

    clip_fps = fps * args.speed

    # --- idle: a quiet standing window, ping-ponged so the loop has no seam ---
    # This window is the ready stance the subject holds in the seconds before the
    # lunge, and it ends on the lunge's first frame: same stance, same distance
    # from the lens, so pressing F never pops. The root is dropped — a standing
    # window still wanders a little, and ping-ponging that wander would slide the
    # character sideways. The limb sway is measured; the drift is not kept.
    idle_frames = [pose_at(pts, i, k) for i in range(idle_lo, idle_hi + 1)]

    # --- lunge: step out + strike, a short hold at the deepest pose, recovery ---
    # `root` rides along frame for frame: how far the hips had travelled from
    # where the move started, so the clip carries its own displacement. It
    # freezes through the hold, and the 134 -> 154 cut is displacement-neutral
    # (the hips sit at 800 and 799 source px).
    deepest = min(range(s_hi, r_lo + 1), key=lambda i: knee[i])
    hip0 = pts["hipC"][s_lo][0]
    src_index = list(range(s_lo, s_hi + 1)) + [deepest] * args.hold + list(range(r_lo, r_hi + 1))
    hold_start = s_hi - s_lo + 1
    hold_end = hold_start + args.hold - 1
    return_start = hold_end + 1
    lunge_frames = [pose_at(pts, i, k) for i in src_index]
    lunge_root = [round((pts["hipC"][i][0] - hip0) * k, 1) for i in src_index]

    def to_clip_index(src_i):
        """Source frame number -> index in the assembled lunge clip."""
        if src_i <= s_hi:
            return clamp_int(src_i - s_lo, 0, hold_start - 1)
        if src_i < r_lo:
            return hold_end                      # inside the compressed hold
        return clamp_int(return_start + (src_i - r_lo), 0, len(lunge_frames) - 1)

    # the strike window: from the moment the lead leg is committed through the
    # deep extension, i.e. the span where the reach is actually out there
    active_src_lo = s_lo + int(round((s_hi - s_lo) * 0.62))
    active_src_hi = min(r_lo + 2, r_hi)
    active = [to_clip_index(active_src_lo), to_clip_index(active_src_hi)]

    out_px = min(lunge_root)
    print(f"lunge travel: {out_px:.0f} px out, {lunge_root[-1] - out_px:+.0f} px back, "
          f"net {lunge_root[-1]:+.0f} px  (arena is 960 px wide)")

    # --- walk + jump tuck: procedural, off the measured rest pose and bones ---
    rest = neutral_legs(circular_mean(angles, idle_lo, idle_hi + 1))
    walk_frames, walk_bob = build_walk(rest, args.walk_frames, rig, k)
    tuck = build_tuck(rest, rig, k)

    clips = {
        "idle": {
            "fps": round(clip_fps, 3), "pingpong": True,
            "source": f"frames {idle_lo}-{idle_hi}",
            "note": f"source frames <em>{idle_lo}&ndash;{idle_hi}</em>, the ready stance held "
                    f"in the seconds before the lunge &mdash; played out and back so the loop "
                    f"has no seam, and it ends where the lunge begins",
            "frames": idle_frames,
        },
        "lunge": {
            "fps": round(clip_fps, 3), "active": active,
            "source": f"frames {s_lo}-{s_hi} + {args.hold}-frame hold + {r_lo}-{r_hi}",
            "note": f"source frames <em>{s_lo}&ndash;{s_hi}</em> (step out and strike), "
                    f"{args.hold} frames held at the deepest pose (frame {deepest}), then "
                    f"<em>{r_lo}&ndash;{r_hi}</em> (both recovery steps). The hips carry their "
                    f"measured travel with them: <em>{out_px:.0f} px</em> out, "
                    f"<em>{lunge_root[-1] - out_px:+.0f} px</em> back. Strike window: clip "
                    f"frames {active[0] + 1}&ndash;{active[1] + 1}.",
            "frames": lunge_frames, "root": lunge_root,
        },
        "walk": {
            "fps": round(args.walk_frames / WALK_CYCLE_S, 3), "loop": True,
            "source": "procedural",
            "note": "there is no walk cycle anywhere in the source clip &mdash; the subject "
                    "stands, lunges, stands. This one is generated from the measured rest "
                    "pose &mdash; torso, head and arm carriage measured, the ready stance's "
                    "split legs averaged into one straight leg to swing around: sinusoidal "
                    "swing on the leg channels, counter-swing on the arms, solved onto the "
                    "subject's own bone lengths.",
            "frames": walk_frames, "bob": walk_bob,
        },
    }

    if args.custom:
        moves = json.loads(Path(args.custom).read_text(encoding="utf-8"))
        current = moves.get("current")
        entry = moves.get("moves", {}).get(current) if current else None
        if entry is None:
            print(f"--custom: {args.custom} has no current move, skipping", file=sys.stderr)
        else:
            clips["custom"] = entry["clip"]
            print(f"baked custom move {current!r} into the page as CLIPS.custom")

    data = {
        "points": POINTS,
        "rig": {
            "headR": round(rig["headR"] * k, 2),
            # Stroke weights track the figure's size, so --scale resizes the
            # whole drawing rather than fattening a smaller stick figure.
            "stroke": round(0.86 * rig["height"] * k / FIGURE_H, 3),
            # The retarget target for a recorded move: the lunge subject's own
            # bone lengths, in game px. serve_game.py measures the same rig
            # independently from lunge_all_frames.json; this copy is for the
            # page and for anyone rebuilding a custom clip by hand.
            "bones": {b: round(rig[b] * k, 2) for b in
                      ("spine", "neck", "thigh", "shin", "upperArm", "forearm",
                       "pelvisHalf", "shoulderHalf")},
        },
        "tuck": tuck,
        "meta": {
            "source": Path(args.data).name,
            "fps": fps, "n_frames": len(frames), "detected": detected,
            "speed": args.speed, "smooth": args.smooth, "scale": round(k, 5),
        },
        "clips": clips,
    }

    for name in ("idle", "lunge", "walk"):
        c = clips[name]
        extra = ""
        if name == "lunge":
            extra = (f", strike window frames {c['active'][0] + 1}-{c['active'][1] + 1} "
                     f"({(c['active'][1] - c['active'][0] + 1) / c['fps']:.2f}s)")
        print(f"clip {name:<5} {len(c['frames']):>3} frames @ {c['fps']:>5.1f}fps = "
              f"{len(c['frames']) / c['fps']:.2f}s  [{c['source']}]{extra}")

    flat = [v for c in clips.values() for fr in c["frames"] for v in fr] + list(tuck)
    if not all(math.isfinite(v) for v in flat):
        raise SystemExit("non-finite coordinate in the emitted clips — refusing to write")

    out = Path(args.out)
    payload = json.dumps(data, separators=(",", ":"))
    out.write_text(build_html(payload), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {len(payload) / 1024:.0f} KB of clip data)")
    return 0


def clamp_int(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


if __name__ == "__main__":
    sys.exit(main())
