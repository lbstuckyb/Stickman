// Mirror check for pose-stick-figure-rfdetr.html.
//   node research/test_mirror.mjs
// Builds a person facing the camera with their RIGHT arm held straight out to
// the side, pushes it through the page's own mirrorLandmarks() / readAngles() /
// solve(), and asserts the rig's arm ends up out to the side rather than folded
// across the chest. Also runs the old x-flip-only path to show what it did.

import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const HERE = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(HERE, "pose-stick-figure-rfdetr.html"), "utf8");
const grab = (start, end) => {
  const i = html.indexOf(start), j = html.indexOf(end, i);
  if (i < 0 || j < 0) throw new Error("could not extract: " + start);
  return html.slice(i, j);
};
const src = grab("const L = {", "/* ---------------------------------------------------------------\n   2.")
          + grab("const RIG = {", "const CHANNELS")
          + grab("const MIN_VIS", "const ang = (a,b)")
          + grab("const ang = (a,b)", "/* Filter, dropping")
          + grab("function solve(p, ox, oy, s)", "/* ---");
const { L, mirrorLandmarks, readAngles, solve } =
  new Function(src + "\nreturn {L, mirrorLandmarks, readAngles, solve};")();

/* A person facing the camera, right arm straight out to their side.
   Un-mirrored camera frame: the subject's right side sits at LOW x. */
function armOutRight() {
  const lm = Array.from({ length: 17 }, () => ({ x: .5, y: .5, visibility: 0 }));
  const set = (i, x, y) => lm[i] = { x, y, visibility: 1 };
  set(L.nose, .50, .18);  set(L.earL, .53, .18);  set(L.earR, .47, .18);
  set(L.shL, .57, .30);   set(L.shR, .43, .30);          // shoulders
  set(L.elL, .58, .42);   set(L.elR, .31, .30);          // right elbow: out, level
  set(L.wrL, .59, .53);   set(L.wrR, .20, .30);          // right wrist: further out
  set(L.hipL, .55, .55);  set(L.hipR, .45, .55);
  set(L.knL, .55, .72);   set(L.knR, .45, .72);
  set(L.anL, .55, .88);   set(L.anR, .45, .88);
  return lm;
}
const flipOnly = lm => lm.map(p => ({ ...p, x: 1 - p.x }));   // the old, broken mirror

const deg = r => r * 180 / Math.PI;
const bad = [];

function run(label, lm, expectSideways) {
  const { raw } = readAngles(lm);
  // Drive the rig from these angles and read where the wrists land.
  const pose = { ...raw, spine: raw.spine, root: { x: .5, y: .5 } };
  const j = solve(pose, 200, 300, 1);
  const outR = j.wrR.x - j.shR.x;        // +ve = wrist right of that shoulder
  const outL = j.wrL.x - j.shL.x;
  // solve() draws the rig facing the viewer: its L shoulder sits at screen right,
  // its R shoulder at screen left. So an outward L wrist is further right than its
  // shoulder, and an outward R wrist is further left. Positive spread = outward.
  const raised = Math.abs(outR) > Math.abs(outL) ? "R" : "L";
  const spread = raised === "R" ? j.shR.x - j.wrR.x : j.wrL.x - j.shL.x;
  console.log(`${label}`);
  console.log(`  raised arm lands on rig side : ${raised}`);
  console.log(`  upper-arm angle              : ${deg(raised === "R" ? raw.armRU : raw.armLU).toFixed(0)}°`);
  console.log(`  wrist offset from shoulder   : ${(raised === "R" ? outR : outL).toFixed(1)} px`);
  console.log(`  reaching outward             : ${spread > 0 ? "yes" : "no — across the chest"}`);
  if (expectSideways && spread <= 0) bad.push(`${label}: raised arm crosses the chest instead of extending outward`);
  if (expectSideways && spread > 0 && spread < 30) bad.push(`${label}: arm barely extends (${Math.abs(spread).toFixed(0)} px)`);
  return { j, raw };
}

const lm = armOutRight();

console.log("=== old behaviour: flip x only (no index swap) ===");
const oldRun = run("flip-only", flipOnly(lm), false);
{ const j0 = oldRun.j, sp = j0.shR.x - j0.wrR.x;
  if (sp > 0) bad.push("flip-only no longer reproduces the bug — this guard is stale"); }

console.log("\n=== current behaviour: mirrorLandmarks() ===");
const { j } = run("mirrored", mirrorLandmarks(lm), true);

/* The subject raised the arm on their right, which in a mirrored view appears on
   the RIGHT of the screen. The rig should extend an arm to the right of centre. */
const centre = (j.shL.x + j.shR.x) / 2;
const far = Math.abs(j.wrL.x - centre) > Math.abs(j.wrR.x - centre) ? j.wrL : j.wrR;
console.log(`\nextended wrist is ${(far.x - centre).toFixed(0)} px from centre ` +
            `(${far.x > centre ? "screen right" : "screen left"}) — mirrored video shows it screen right`);
if (far.x <= centre) bad.push("extended arm is on the wrong side of the screen for a mirrored view");

// And with mirroring off, the same pose must reach out on the opposite side.
console.log("\n=== mirror off ===");
const { j: j2 } = run("unmirrored", lm, true);
const centre2 = (j2.shL.x + j2.shR.x) / 2;
const far2 = Math.abs(j2.wrL.x - centre2) > Math.abs(j2.wrR.x - centre2) ? j2.wrL : j2.wrR;
if (far2.x >= centre2) bad.push("unmirrored: extended arm should be on screen left");

console.log(bad.length ? "\nFAIL\n" + bad.join("\n") : "\nPASS");
process.exit(bad.length ? 1 : 0);
