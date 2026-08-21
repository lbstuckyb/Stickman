# From video to playable lunge: how the capture became the attack animation

Full pipeline, start to finish, in the order Claude actually built it across two sessions
(initial commit `7b27b78`, then a rework in `eb508f8`).

## Step 1 — Get one frame's worth of keypoints out of the model

`test_rfdetr_keypoints.py` is a smoke test: it loads a single frame (`lunge_frame.jpg`, pulled
from the video at t=3.5s with `ffmpeg`), runs it through `RFDETRKeypointPreview` (a model that
returns COCO-17 keypoints — nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles —
each with an x/y position **and** a 2-D uncertainty ellipse, not just a single point), and
writes `lunge_keypoints_out.json` in the shape `{ people: [{ score, box, keypoints: [{x, y, v}, ...] }] }`,
with `x`/`y` normalized 0–1 and `v` a visibility/confidence score. This step never touches the
game — it's just "does the model work, what does its output look like."

## Step 2 — Run the model over every frame of the video

`build_sequence_viewer.py lunge.webm` opens the clip with OpenCV, decodes it frame by frame,
runs the same model on **all 264 frames**, and writes `lunge_all_frames.json`: one entry per
frame (`{i, t, people: [...]}`), same per-person keypoint shape as step 1. This is the slow step
— roughly 0.7–1s/frame on CPU — and it's the only step that runs the neural network. Every later
step is pure math over this one file; the model is never invoked again.

## Step 3 — Follow one person across frames, not just "best score per frame" (`build_game.py:96` `track_subject`)

Some frames had more than one candidate detection. The naive approach — pick whichever detection
scored highest each frame — silently picked a **left/right-mirrored duplicate** at frame 162,
which snapped the limbs mid-recovery when smoothed. Claude replaced this with a tracker that,
for each frame, keeps whichever candidate sits closest (by mean joint distance) to the previous
frame's accepted pose, and flips a candidate left-right if its mirror image sits *even closer*
to the previous frame. This produces one clean, continuous detection per frame.

## Step 4 — Reduce 17 raw points to the 15 the figure actually draws (`build_game.py:128` `rig_points`)

COCO-17 has no single "head" point, so the figure's head is the ear midpoint (falling back to
the nose when only one ear is visible). Left/right hip and shoulder midpoints are computed for
the spine line. That leaves 15 named joints: `hipC, hipL, hipR, shC, shL, shR, headC, elL, wrL,
elR, wrR, knL, anL, knR, anR`.

## Step 5 — Build a continuous track per joint, holding position through occlusion (`build_game.py:143` `point_tracks`)

For each of the 15 joints, across all 264 frames: if the model saw that joint clearly this
frame, record it; if not (occluded, or no detection at all that frame), **hold the last known
position** rather than snapping to zero or interpolating blindly. Then a small centered
moving-average (window of 3 frames, `--smooth`) is applied per coordinate to remove jitter — the
only smoothing done anywhere in the pipeline.

## Step 6 — Measure the subject's own body, and the video→game scale factor (`build_game.py:191` `measure_rig`)

Over source frames 79–106 (the "ready stance" the subject holds just before lunging — chosen
specifically because that's where they're standing still and square to the camera), Claude took
the **median** bone length for spine, neck, thigh, shin, upper arm, forearm, half-pelvis,
half-shoulder-width, and head radius, straight off the measured points. Standing height
(neck+spine+thigh+shin) came out such that one scale factor, `k = 249 / height ≈ 0.4466`, makes
the figure 249 px tall in the game canvas. Every point from here on gets multiplied by this one
constant — nothing is resized differently per limb.

## Step 7 — Cut three clips out of the one continuous track (`build_game.py` `main()`, ~line 1959 onward)

- **idle**: source frames 79–106, ping-ponged (played forward then backward) so the loop has no
  seam, and chosen so it ends on the exact frame the lunge begins on — this is also the window
  step 6 measured from.
- **lunge**: frames 106–134 (step-out + strike), then the single deepest-knee-bend frame held
  for 4 extra frames (`--hold`), then frames 154–188 (both recovery steps back to standing).
  This is the actual captured attack.
- **walk**: the *only* invented clip — there is no walk cycle anywhere in the source video (the
  subject only stands, lunges, and stands again). It's built by taking the measured rest pose,
  averaging the ready stance's two different leg angles into one straight leg, and swinging that
  procedurally with sine waves, then solving it back into (x,y) points using the **subject's own
  measured bone lengths** from step 6 — so it's fake motion but on a real body.

## Step 8 — Store each lunge frame as 15 hip-relative points, plus how far the hips actually moved (`build_game.py:496` `pose_at`, and the `lunge_root` list around line 1979)

For every frame kept in the lunge clip, each of the 15 joints is stored as
`(x - hipX, y - hipY) * k` — i.e., position relative to the hip center, in game pixels.
Separately, a **root track** records the hip's absolute x position that frame, relative to
where the lunge started, in game px: this is the "how far the body actually traveled" number
(printed as e.g. "203 px out, 216 px back"). This is the detail the commit message calls out as
the fix: earlier, the game pinned the hips to the player's on-screen x and rebuilt the legs from
angles on a fixed-length rig, so the 203 px of real forward travel from the video was silently
thrown away and the recovered stick figure wasn't wearing the subject's real proportions.
Storing raw measured points + a separate root offset keeps both of those honest.

## Step 9 — Bake everything into one self-contained HTML file (`build_game.py:634` `build_html`, and `main()`)

`build_game.py` assembles a Python dict — `points` (the 15 names), `clips` (idle/lunge/walk,
each with its frame array, fps, loop mode, and for lunge, `root` and the `active` strike-window
frame range), `rig` (stroke widths etc.), `tuck` (the one-off jump pose) — dumps it to JSON, and
substitutes it into a `<script>const DATA = __DATA_JSON__;</script>` tag inside a single HTML
template. The result, `lunge_game.html`, has no server dependency and no separate data file:
opening it in any browser (even `file://`) loads the whole captured motion inline.

## Step 10 — What happens in the browser when you press F (`lunge_game.html`, inlined from `build_game.py:1402` and the runtime code around lines 900–1180)

1. `KeyF` fires → if the player isn't already mid-attack and the cooldown has expired,
   `setState('lunge')` runs: it resets the clip clock `p.t = 0` and anchors
   `p.rootAnchorX = p.x` (remembers where on screen the lunge started).
2. Every simulation tick, `statePose()` calls `samplePose(CLIPS.lunge, p.t, ...)`, which
   converts elapsed time to a frame index at the clip's fps (`speed=1.4`× the source video's
   frame rate) and **linearly interpolates between the two nearest stored frames** — this is the
   only "reconstruction" happening; the shape itself is never rebuilt from angles.
3. `place()` takes those 15 hip-relative points and adds the world hip position back — nothing
   about limb shape changes, only where the whole rig sits.
4. The x position comes from the root track: `p.x = rootAnchorX + (±)pose.root`, so the
   character actually walks the 203 px the subject walked on video, then the recovery steps
   carry it back to 13 px from its start — the same net drift the source clip has.
5. The y position comes from `stanceOffset()`: the lowest of the two ankle y-values in the
   current pose decides how far the hips sit off the ground line, frame by frame — so the body's
   drop into the lunge is exactly whatever the measured knee-bend produces, not a hand-tuned dip.
6. During the clip's stored `active` frame range (the "strike window," auto-computed as the span
   where a lead wrist/ankle sits within 85% of its deepest reach), `strike()` reads the live lead
   ankle and lead wrist positions straight off the same measured pose and checks them against
   target hitboxes — so the hitbox literally is the captured hand/foot position, sweeping
   forward exactly as far as the real lunge reached.
7. After the clip's duration elapses, cooldown starts and the state machine drops back to `idle`
   or `jump`.

## Summary

End to end: **video → per-frame model inference → tracked/cleaned point clouds → measured bone
lengths and a single scale factor → three sliced clips (two measured, one procedural-on-measured-bones)
→ baked as inline JSON → replayed in the browser as literal interpolated point positions**, with
pressing `F` doing nothing more clever than "start sampling the `lunge` clip from t=0 and drive
world position off its stored root/stance offsets."
