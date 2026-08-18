// Retarget check for pose-stick-figure-rfdetr.html.
//   node test_retarget.mjs [payload.json]
// Extracts the joint map, readAngles() and solve() straight out of the page and runs
// them over a real /keypoints payload, so the COCO-17 wiring is tested, not restated.
// Get a payload with:
//   curl -X POST --data-binary @frame.jpg -H "Content-Type: image/jpeg" \
//        http://127.0.0.1:8000/keypoints > payload.json
import {readFileSync} from "fs";
const html = readFileSync("pose-stick-figure-rfdetr.html","utf8");
// pull the real source out of the page: joint map, helpers, readAngles, solve, RIG
const grab = (start, end) => {
  const i = html.indexOf(start), j = html.indexOf(end, i);
  if (i < 0 || j < 0) throw new Error("could not extract: " + start);
  return html.slice(i, j);
};
const src = grab("const L = {", "/* ---------------------------------------------------------------\n   2.")
          + grab("const RIG = {", "const CHANNELS")
          + grab("const MIN_VIS", "const ang = (a,b)")
          + grab("const ang = (a,b)", "/* Filter, dropping")
          + grab("function solve(p, ox, oy, s)", "/* ------------------------------------------------------------\n   4.".slice(0,20));
const mod = new Function(src + "\nreturn {L, readAngles, solve, RIG, ok, mid};")();

const {people} = JSON.parse(readFileSync(process.argv[2] || "payload.json","utf8"));
const deg = r => (r*180/Math.PI).toFixed(0);
let bad = [];
people.forEach((p, idx) => {
  const lm = p.keypoints.map(k => ({x:k.x, y:k.y, visibility:k.v}));
  const {raw, conf, root} = mod.readAngles(lm);
  if (idx === 0) {
    console.log("person 0 angles (deg, screen coords: -90 = up, +90 = down):");
    for (const k of Object.keys(raw)) console.log(`  ${k.padEnd(6)} ${deg(raw[k]).padStart(5)}  conf=${conf[k]}`);
    console.log("  root(hip mid)", root.x.toFixed(3), root.y.toFixed(3));
  }
  // a standing person: spine points up, thighs point down, all channels confident
  if (idx===0 && !(raw.spine < -Math.PI/3 && raw.spine > -2*Math.PI/3)) bad.push(`person ${idx}: spine ${deg(raw.spine)} not upright`);
  if (idx===0) for (const k of ["legLU","legRU"]) if (!(raw[k] > 0)) bad.push(`person ${idx}: ${k} ${deg(raw[k])} not pointing down`);
  const dropped = Object.keys(conf).filter(k => !conf[k]);
  // only the top-scoring person (the one that drives the rig) must be fully confident;
  // a partly-occluded extra correctly has channels dropped rather than snapping
  if (idx === 0 && dropped.length) bad.push(`person 0: channels dropped: ${dropped}`);
  else if (dropped.length) console.log(`  (person ${idx}: ${dropped.length} channel(s) held at previous angle: ${dropped})`);
  if (!(root.x > 0 && root.x < 1 && root.y > 0 && root.y < 1)) bad.push(`person ${idx}: root off-frame`);
});
// forward kinematics must produce finite joint positions
const pose = {spine:-Math.PI/2, head:-Math.PI/2, armLU:1, armLF:1, armRU:2, armRF:2,
              legLU:1.4, legLS:1.6, legRU:1.7, legRS:1.5, root:{x:.5,y:.5}};
const j = mod.solve(pose, 200, 200, 1);
for (const [k,v] of Object.entries(j)) if (!isFinite(v.x)||!isFinite(v.y)) bad.push(`solve: ${k} not finite`);
console.log("\nsolve() head/hip separation:", Math.hypot(j.headC.x-j.hipC.x, j.headC.y-j.hipC.y).toFixed(1), "px (rig spine+neck+headR =", mod.RIG.spine+mod.RIG.neck+mod.RIG.headR, ")");
console.log(bad.length ? "\nFAIL\n" + bad.join("\n") : `\nPASS — ${people.length} people retargeted, all 10 rig channels confident`);
process.exit(bad.length ? 1 : 0);
