// tests/rotation-v6.mjs — FIT-V6 TASK G2: arc-built F1 hairpin templates,
// clipped-Track validation, dense-spline clearance and feasibility gates.
// Plain Node, guarded CommonJS exports, no npm deps — same harness shape as
// tests/rotation-v5.mjs / track-clipped.mjs.
//
// Run:            node tests/rotation-v6.mjs
// Capture goldens: node tests/rotation-v6.mjs --capture
//
// Everything is derived from the approved plans
// (.opencode/plans/fit-v6-clipped-walls.md + fit-v5-f1-circuits.md) and from
// the frozen spike data, NOT from the implementation under test:
//   * the six arc-built F1 hairpin templates are the byte-frozen 91-96 point arrays
//     from /tmp/opencode/fitv6-archtemplates/templates_hairpin.json
//     (C1,C2,C3,B2,B3,A2), start-vertex aligned; G1's 30 raw-template wall-byte
//     goldens stay pinned to the ORIGINAL 21-23pt arrays (G1_TEMPLATES below);
//   * the halfWidth ladder is 50 - 0.25 d;
//   * the wall-byte oracle is G1's byte-frozen leftWall|rightWall|segments
//     serialization; G1's 30 pinned hashes are embedded below;
//   * the apex metric is the minimum control-polygon circumradius over a
//     three-step window (plan note: metric not numerically pinned).

import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const ROOT = fileURLToPath(new URL('..', import.meta.url));
const { Track, canonicalJSON, cyrb53 } = require('../js/track.js');
const rotationMod = require('../js/rotation.js');
const { Rotation } = rotationMod;
const CAPTURE = process.argv.includes('--capture');

// ---------------------------------------------------------------------------
// Harness — one line per test, never throws out of a test, exit 1 on failure.
// ---------------------------------------------------------------------------
let passed = 0;
const failures = [];
async function test(name, fn) {
  try {
    await fn();
    passed += 1;
    console.log('  ok   - ' + name);
    if (CAPTURE) return;
  } catch (err) {
    failures.push({ name, err });
    const detail = String((err && err.stack) || err)
      .split('\n').map((line) => '         ' + line).join('\n');
    console.log('  FAIL - ' + name + '\n' + detail);
  }
}
function assert(cond, msg) {
  if (!cond) throw new Error('assert failed: ' + msg);
}
function assertValidationError(fn, msg) {
  let threw = null;
  try { fn(); } catch (err) { threw = err; }
  assert(threw !== null, msg + ' - expected ValidationError, nothing thrown');
  assert(threw.name === 'ValidationError', msg + ' - expected ValidationError, got ' + threw.name + ': ' + threw.message);
}

// ---------------------------------------------------------------------------
// Independent spec helpers.
// ---------------------------------------------------------------------------
const TAU = Math.PI * 2;
const T32 = 1 / 4294967296;
const CENTER = { x: 550, y: 325 };
const round3 = (v) => Number(v.toFixed(3));
const DIFFICULTIES = [0, 25, 50, 75, 100];
function halfWidthOf(r) { return Number((50 - 0.25 * Math.min(r, 100)).toFixed(9)); }
function hashOf(points, halfWidth) { return cyrb53(canonicalJSON({ points, halfWidth })); }
function templateIndex(seed, r) { return ((seed ^ Math.imul(r + 1, 0x9E3779B1)) >>> 0) % 6; }
function refMulberry(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  };
}
function orient(ax, ay, bx, by, cx, cy) { return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax); }
function segCross(ax, ay, bx, by, cx, cy, dx, dy) {
  const o1 = orient(ax, ay, bx, by, cx, cy), o2 = orient(ax, ay, bx, by, dx, dy);
  const o3 = orient(cx, cy, dx, dy, ax, ay), o4 = orient(cx, cy, dx, dy, bx, by);
  return ((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) && ((o3 > 0 && o4 < 0) || (o4 > 0 && o3 < 0));
}
function polylineSimple(points) {
  const n = points.length;
  for (let i = 0; i < n; i++) for (let j = i + 2; j < n; j++) {
    if (i === 0 && j === n - 1) continue;
    if (segCross(points[i].x, points[i].y, points[(i + 1) % n].x, points[(i + 1) % n].y,
                 points[j].x, points[j].y, points[(j + 1) % n].x, points[(j + 1) % n].y)) return false;
  }
  return true;
}
function circumRadius(a, b, c) {
  const ab = Math.hypot(b.x - a.x, b.y - a.y), bc = Math.hypot(c.x - b.x, c.y - b.y);
  const ca = Math.hypot(a.x - c.x, a.y - c.y);
  const cr = Math.abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
  return cr < 1e-12 ? Infinity : ab * bc * ca / (2 * cr);
}
function apexRadius(points, w = 3) {
  const n = points.length;
  let m = Infinity;
  for (let i = 0; i < n; i++) {
    m = Math.min(m, circumRadius(points[(i - w + n) % n], points[i], points[(i + w) % n]));
  }
  return m;
}
function tanDeviation(points) {
  const n = points.length, a = points[n - 1], b = points[0], c = points[1];
  const v1x = b.x - a.x, v1y = b.y - a.y, v2x = c.x - b.x, v2y = c.y - b.y;
  return Math.atan2(Math.abs(v1x * v2y - v1y * v2x), v1x * v2x + v1y * v2y) * 180 / Math.PI;
}
// G1 byte-frozen wall serialization (verbatim from tests/track-clipped.mjs).
function wallHash(track) {
  const encPt = (p) => Number(p.x.toFixed(9)).toFixed(9) + ',' + Number(p.y.toFixed(9)).toFixed(9);
  const encWall = (w) => w.map(encPt).join(';');
  const encSeg = (s) => Number(s.x1.toFixed(9)).toFixed(9) + ',' + Number(s.y1.toFixed(9)).toFixed(9) + ',' +
    Number(s.x2.toFixed(9)).toFixed(9) + ',' + Number(s.y2.toFixed(9)).toFixed(9) + ',' + s.side;
  return cyrb53(encWall(track.leftWall) + '|' + encWall(track.rightWall) + '|' + track.segments.map(encSeg).join(';'));
}

// The six arc-built F1 hairpin templates (byte-frozen, start-vertex aligned;
// identical to rotation.js V5_TEMPLATES) — 91-96 points, one >=150deg U-turn each.
const TEMPLATES = [
[[307.838,127.95],[325.044,127.95],[362.897,127.95],[446.175,127.95],[773.785,127.95],[857.063,127.95],[894.917,127.95],[912.123,127.95],[926.461,127.95],[936.728,128.46],[946.893,129.985],[956.857,132.511],[966.522,136.012],[975.792,140.454],[984.575,145.793],[992.787,151.977],[1000.344,158.944],[1007.174,166.627],[1013.208,174.948],[1018.388,183.827],[1022.662,193.175],[1025.988,202.902],[1028.334,212.91],[1029.675,223.101],[1030,233.375],[1029.305,243.631],[1027.596,253.767],[1024.891,263.684],[1021.216,273.283],[1016.607,282.471],[1011.11,291.157],[1004.779,299.256],[997.676,306.686],[989.872,313.376],[981.443,319.259],[972.472,324.278],[963.048,328.382],[953.263,331.532],[939.414,335.243],[922.794,339.697],[886.23,349.494],[805.79,371.048],[628.821,418.466],[579.72,431.623],[402.751,479.042],[322.311,500.596],[285.747,510.393],[269.127,514.846],[255.277,518.557],[245.567,520.666],[235.699,521.834],[225.764,522.05],[215.855,521.314],[206.062,519.631],[196.475,517.017],[187.182,513.497],[178.27,509.102],[169.821,503.873],[161.911,497.858],[150.927,488.642],[117.975,460.992],[106.991,451.775],[99.624,444.959],[92.952,437.461],[87.036,429.352],[81.934,420.709],[77.692,411.612],[74.35,402.147],[71.941,392.404],[70.486,382.473],[70,372.447],[70,358.109],[70,315.093],[70,300.755],[70.499,290.605],[71.99,280.552],[74.459,270.694],[77.883,261.126],[82.228,251.939],[87.452,243.222],[93.506,235.06],[100.331,227.53],[110.47,217.391],[140.886,186.975],[151.025,176.836],[159.08,169.583],[167.849,163.212],[177.237,157.792],[187.139,153.383],[197.448,150.034],[211.298,146.323],[252.847,135.189],[266.697,131.478],[275.517,129.523],[284.474,128.344],[293.499,127.95]],
[[366.404,80.546],[387.299,80.546],[433.267,80.546],[534.398,80.546],[718.878,80.546],[820.009,80.546],[865.977,80.546],[886.872,80.546],[904.284,80.546],[916.805,81.171],[929.202,83.039],[941.351,86.133],[953.132,90.421],[964.427,95.86],[975.124,102.397],[985.118,109.967],[994.308,118.494],[1002.603,127.894],[1009.922,138.073],[1016.19,148.93],[1021.346,160.358],[1025.339,172.241],[1028.129,184.464],[1029.687,196.903],[1030,209.436],[1029.063,221.938],[1026.886,234.284],[1023.491,246.352],[1018.91,258.023],[1013.191,269.179],[1006.389,279.71],[998.573,289.511],[989.819,298.486],[980.215,306.545],[969.857,313.607],[958.847,319.603],[947.295,324.472],[930.933,330.428],[911.298,337.574],[868.102,353.296],[773.07,387.885],[468.945,498.577],[373.913,533.166],[330.717,548.888],[311.083,556.035],[294.72,561.99],[282.487,565.749],[269.936,568.246],[257.195,569.454],[244.398,569.361],[231.676,567.968],[219.162,565.289],[206.985,561.352],[195.271,556.198],[184.141,549.88],[173.712,542.463],[164.09,534.025],[155.376,524.653],[144.183,511.314],[110.606,471.298],[99.414,457.96],[91.954,448.128],[85.495,437.613],[80.097,426.515],[75.813,414.941],[72.684,403.003],[70.74,390.816],[70,378.498],[70.471,366.166],[71.989,348.82],[76.541,296.782],[78.059,279.436],[79.773,267.016],[82.715,254.83],[86.855,242.997],[92.154,231.634],[98.557,220.856],[106.001,210.769],[114.413,201.473],[126.726,189.161],[163.663,152.224],[175.975,139.912],[185.757,131.104],[196.406,123.367],[207.806,116.785],[219.831,111.432],[232.349,107.364],[249.168,102.857],[299.625,89.337],[316.444,84.831],[327.155,82.456],[338.032,81.024],[348.992,80.546]],
[[456.712,70],[479.616,70],[530.007,70],[745.472,70],[795.862,70],[818.767,70],[837.854,70],[851.643,70.691],[865.294,72.759],[878.669,76.181],[891.635,80.924],[904.062,86.94],[915.824,94.169],[926.805,102.538],[936.893,111.964],[945.987,122.351],[953.997,133.596],[960.843,145.586],[966.454,158.201],[970.775,171.313],[973.764,184.792],[975.389,198.503],[975.634,212.307],[974.497,226.066],[971.99,239.643],[968.138,252.901],[962.978,265.707],[956.564,277.932],[948.958,289.455],[940.238,300.159],[930.492,309.937],[919.816,318.692],[908.318,326.334],[896.113,332.788],[878.814,340.855],[858.055,350.535],[812.386,371.831],[711.915,418.681],[577.778,481.23],[477.306,528.081],[431.637,549.377],[410.878,559.057],[393.58,567.123],[380.408,572.457],[366.757,576.407],[352.772,578.93],[338.601,580],[324.396,579.605],[310.307,577.751],[296.483,574.455],[283.073,569.754],[270.217,563.697],[258.054,556.349],[246.711,547.788],[232.09,535.519],[188.225,498.712],[173.603,486.443],[163.89,477.464],[155.082,467.595],[147.26,456.928],[140.497,445.56],[134.854,433.596],[130.384,421.146],[127.128,408.325],[125.116,395.252],[124.366,382.045],[124.885,368.827],[126.549,349.813],[131.539,292.769],[133.203,273.755],[135.042,260.352],[138.186,247.194],[142.604,234.408],[148.254,222.116],[155.082,210.438],[163.022,199.485],[171.997,189.362],[181.921,180.169],[196.542,167.9],[240.407,131.093],[255.029,118.824],[266.553,110.14],[278.921,102.708],[291.998,96.61],[305.641,91.912],[319.701,88.667],[338.498,85.352],[394.89,75.409],[413.687,72.094],[425.61,70.525],[437.625,70]],
[[320.219,89.775],[339.906,89.775],[383.218,89.775],[478.505,89.775],[736.855,89.775],[832.142,89.775],[875.455,89.775],[895.142,89.775],[911.548,89.775],[923.346,90.364],[935.026,92.125],[946.473,95.04],[957.573,99.08],[968.216,104.205],[978.295,110.364],[987.711,117.497],[996.37,125.531],[1004.186,134.388],[1011.082,143.979],[1016.988,154.208],[1021.846,164.975],[1025.608,176.173],[1028.237,187.689],[1029.706,199.41],[1030,211.218],[1029.117,222.998],[1027.066,234.63],[1023.867,246.001],[1019.551,256.997],[1014.162,267.509],[1007.754,277.431],[1000.389,286.667],[992.141,295.123],[983.092,302.716],[973.332,309.37],[962.958,315.019],[952.074,319.607],[936.657,325.219],[918.157,331.952],[877.457,346.766],[787.916,379.356],[474.575,493.403],[385.035,525.993],[344.334,540.806],[325.834,547.54],[310.418,553.151],[299.559,556.523],[288.428,558.838],[277.126,560.076],[265.757,560.225],[254.426,559.283],[243.238,557.26],[232.295,554.174],[221.698,550.053],[211.545,544.935],[201.93,538.868],[188.491,529.458],[148.173,501.228],[134.734,491.817],[125.548,484.689],[117.104,476.694],[109.485,467.91],[102.764,458.421],[97.005,448.319],[92.264,437.702],[88.586,426.671],[86.008,415.332],[83.159,399.176],[74.612,350.705],[71.763,334.548],[70.294,322.827],[70,311.019],[70.883,299.239],[72.934,287.606],[76.133,276.235],[80.449,265.24],[85.838,254.728],[94.041,240.52],[118.65,197.896],[126.853,183.688],[133.406,173.565],[140.953,164.161],[149.417,155.573],[158.711,147.889],[168.737,141.19],[179.392,135.545],[194.261,128.611],[238.868,107.81],[253.737,100.877],[265.725,96.064],[278.167,92.584],[290.913,90.48],[303.812,89.775]],
[[522.448,70],[545.422,70],[595.965,70],[693.434,70],[743.977,70],[766.951,70],[786.096,70],[799.995,70.7],[813.754,72.794],[827.232,76.261],[840.293,81.064],[852.806,87.157],[864.642,94.476],[875.683,102.948],[885.817,112.487],[894.94,122.996],[902.96,134.369],[909.797,146.491],[915.381,159.239],[919.655,172.483],[922.576,186.09],[924.114,199.922],[924.254,213.838],[922.994,227.697],[920.348,241.36],[916.342,254.688],[911.016,267.546],[904.425,279.803],[896.635,291.335],[887.725,302.026],[877.786,311.767],[866.918,320.459],[855.231,328.016],[838.651,337.588],[818.755,349.075],[774.983,374.347],[678.686,429.944],[627.364,459.575],[531.066,515.172],[487.295,540.444],[467.398,551.931],[450.818,561.503],[438.431,567.846],[425.469,572.912],[412.063,576.649],[398.35,579.02],[384.468,580],[370.557,579.58],[356.759,577.763],[343.214,574.569],[330.059,570.029],[317.426,564.19],[305.444,557.111],[294.235,548.863],[283.912,539.53],[270.374,525.992],[229.761,485.379],[216.223,471.842],[207.302,462.02],[199.364,451.389],[192.481,440.047],[186.717,428.096],[182.125,415.648],[178.747,402.818],[176.614,389.722],[175.746,376.483],[176.151,363.221],[177.826,350.059],[181.15,331.205],[191.124,274.642],[194.448,255.788],[197.457,242.557],[201.749,229.684],[207.281,217.294],[214.002,205.506],[221.845,194.433],[230.736,184.183],[240.589,174.853],[251.309,166.534],[266.992,155.553],[314.04,122.61],[329.723,111.628],[342.525,103.668],[356.118,97.148],[370.339,92.146],[385.021,88.723],[403.875,85.399],[460.438,75.425],[479.292,72.101],[491.252,70.526],[503.303,70]],
[[396.413,70],[418.782,70],[467.994,70],[576.261,70],[685.106,70],[793.373,70],[842.585,70],[864.954,70],[883.595,70],[896.833,70.652],[909.942,72.603],[922.796,75.834],[935.27,80.312],[947.244,85.996],[958.6,92.829],[969.23,100.746],[979.029,109.67],[987.904,119.514],[995.768,130.182],[1002.544,141.573],[1008.168,153.574],[1012.584,166.071],[1015.751,178.941],[1017.636,192.06],[1018.223,205.301],[1017.504,218.535],[1015.488,231.634],[1012.194,244.472],[1007.653,256.924],[1001.91,268.869],[995.02,280.191],[987.05,290.781],[978.078,300.536],[968.189,309.361],[957.482,317.172],[946.057,323.891],[934.028,329.455],[916.744,336.438],[896.004,344.818],[850.375,363.253],[749.992,403.811],[522.192,495.848],[421.809,536.405],[376.181,554.84],[355.44,563.22],[338.157,570.203],[325.653,574.553],[312.783,577.653],[299.67,579.475],[286.442,580],[273.226,579.223],[260.151,577.152],[247.342,573.807],[234.923,569.22],[223.015,563.436],[211.733,556.51],[201.186,548.509],[186.906,536.527],[144.066,500.58],[129.787,488.598],[119.844,479.362],[110.881,469.172],[102.99,458.132],[96.251,446.354],[90.731,433.956],[86.488,421.067],[83.563,407.815],[81.988,394.336],[81.777,380.767],[82.428,362.138],[84.38,306.249],[85.03,287.619],[86.139,274.454],[88.532,261.461],[92.188,248.765],[97.07,236.488],[103.132,224.749],[110.316,213.661],[118.552,203.33],[127.76,193.856],[141.394,181.143],[182.293,143.004],[195.926,130.291],[205.973,121.801],[216.799,114.332],[228.304,107.955],[240.375,102.731],[252.898,98.711],[270.904,93.886],[324.922,79.412],[342.927,74.587],[354.394,72.045],[366.038,70.512],[377.772,70]]
];

// G1 wall-byte goldens are pinned to the ORIGINAL 21-23pt raw templates (a
// track.js regression corpus owned by tests/track-clipped.mjs). Keep an
// independent frozen copy here so G2-14 keeps verifying that agreement; the
// generator contract itself uses TEMPLATES above.
const G1_TEMPLATES = [
[[735.681,462.089],[781.5,381.5],[827.8,302.3],[876,220.3],[777,206.4],[683.4,188.2],[589.2,180.7],[497.5,160.2],[400.7,146.3],[307.9,134.2],[216.2,119.5],[121.5,107.6],[281.7,105.1],[435.9,109.4],[597,107.1],[726.9,107.7],[835.7,108.9],[911.6,243.8],[978.5,372.8],[833.6,461],[688.6,544.9]],
  [[684.906,181.99],[759.3,259.4],[836.2,331.6],[843.4,442.1],[840.9,539.3],[767.8,537.3],[707,537.6],[582.1,557.4],[529.3,491.2],[496.2,445.8],[459.5,369.3],[416.6,331],[384.5,272.3],[338.9,206.3],[307.6,149.4],[256.6,98],[350.1,109.1],[416.2,112.7],[485.3,99.3],[552.5,102.2],[599,92.6]],
  [[622.818,358.304],[639,289.9],[557.5,243.5],[479.7,193.7],[399.5,144.1],[319.4,97.6],[420.4,94],[521.2,96.3],[622.2,97.9],[702.3,96.6],[778.4,95.2],[777.5,155.8],[780.6,221.4],[779.1,283.8],[775.7,348.7],[729,400.9],[671.3,452.1],[623.7,505.7],[568.8,556],[587.3,492.7],[607.3,423.9]],
  [[184.973,180.789],[296.8,201],[411.2,210.1],[531.5,207.2],[652.3,186.3],[743.6,228.9],[876.8,302.9],[991.9,140.3],[952.2,492.5],[900.6,509.7],[828.1,494],[706.3,487.6],[624.9,356.4],[478.2,334.1],[355.6,356.7],[257.3,395.3],[152.7,425],[128.2,379.8],[148.1,289.1],[109.1,209.7],[108.1,161.6],[120.3,169.1]],
  [[240.524,270.508],[308,234.9],[416.4,192.5],[533.5,103.8],[581.2,130],[589.7,104.3],[720.8,190.9],[823.7,193.6],[956.4,277.6],[971.1,385.1],[923.5,424.6],[885.3,334.3],[825.1,376.2],[734.6,519.4],[679.9,546.2],[555,494.9],[479.2,379.7],[315.2,367.5],[228.9,333.4],[171.5,322.9],[128.9,287.2],[172.7,306.3]],
  [[749.274,500.728],[669.3,566.5],[540.3,521.6],[456.8,487.1],[373.6,461.3],[273.2,317.1],[253.1,403.2],[197.1,290.6],[139,207],[81,111.3],[248,83.5],[410.3,96.3],[528.2,175.1],[597.3,268.5],[676.5,332.1],[762.1,336],[847.2,316.9],[933.4,324.3],[1019,341.3],[929.9,340.4],[843.3,423.4]],
];
const toPoints = (t) => t.map((p) => ({ x: p[0], y: p[1] }));

// G1's 30 pinned wall-byte hashes (t:d -> hash), from tests/track-clipped.mjs.
const G1_WALL_GOLDEN = {
  '0:0':'0e6a97540a6dd38e','0:25':'de42fa98c2c64918','0:50':'1a685dc268e02a0b','0:75':'43605d99713edcfd','0:100':'a078e56245542179',
  '1:0':'f015f7de5e35e200','1:25':'2c6d73e76d2be2b9','1:50':'01556caea2886cc6','1:75':'88d939b238d0eb11','1:100':'7ca3dc5ca08dc4d7',
  '2:0':'97751c04a53432f0','2:25':'9833011691c19d9d','2:50':'bd6415c5e558411f','2:75':'b3a748583b425eb5','2:100':'40351f4a084640cb',
  '3:0':'09c0d063d1692d64','3:25':'b68dba1ca0357f3e','3:50':'79285343e785c04c','3:75':'00e4ab0eec88ae4b','3:100':'a52bd0b380198e55',
  '4:0':'9a1c63581e799f9f','4:25':'73cda34b8d2138ab','4:50':'179f868acef77115','4:75':'5a892fdbeef83499','4:100':'be92bb929ed2dc1e',
  '5:0':'c87754b400b247e8','5:25':'b226d790ec65585e','5:50':'af0daa150a391e0c','5:75':'69982f7a68cd8a3d','5:100':'87301e3ccfd23d21',
};

// Pinned generator golden vectors — regenerated for the new V2 / new ladder.
// (seed -> { r -> trackHash }); filled once from `node ... --capture`.
const GOLDEN_V2 = {"1":{"1":"f73efc03c72ad1fe","2":"c53b4eb9edab5861","3":"e8c41d16f85651a4","4":"d9d3198e66a4ba3a","5":"3b04f483f08fae13","6":"18bbd6b646a80751","7":"4d38536d57ec9f50","8":"08a17dc010e602c8","9":"f3d001e17ae1cfe0","10":"879d9b5f9907e9a2","11":"9e79a48d906b54a8","12":"7392974530ac71f3","13":"fe83d3df49bc5276","14":"e029c4240d7c77ff","15":"a01b59968b4c16e7","16":"bdb6880448b2e436","17":"d731a7fae0d3edda","18":"d570e4a90e914f22","19":"be039b507141cc52","20":"2090a3013f7d91fb","21":"6040023b4957b673","22":"d4d7d653844a3267","23":"da8d1138039526e4","24":"810336449435cc2f","25":"f5aff0bfed6a43c6","26":"60fb3bb796ff2f40","27":"cf1030719a40ef23","28":"b9e97caa6f1ce300","29":"394251cada601254","30":"98aeb9f57acb8c7e","31":"f26fe7c12f14fb1b","32":"6de14077cab175bb","33":"14e36254fd7c3abf","34":"8642597ff164930e","35":"41321675d819d708","36":"07dc660298c58723","37":"907744a3f2dde81b","38":"7e876cd8e6f7a935","39":"28f78f5d0bd2d1b5","40":"479d25e3c3923c31","41":"021f5e506c8256aa","42":"4bcd3c0f33502d60","43":"f8bbe4b8b07643cd","44":"5be2d731a9464139","45":"ab23e6e90d8cfa73","46":"c47f24792af9a108","47":"880b7adf51195bfb","48":"590942e74b882821","49":"ebb9659310e818af","50":"0cc946187de240e7","51":"4d2bd75896a14f3a","52":"091593acf4735d1e","53":"595f77ff3fa98496","54":"684f7119bc477df5","55":"fd66df7abd45a9de","56":"a531c430a61087ee","57":"72608b2e51761d51","58":"656102fda94b33e3","59":"98f65bcbf91f58b5","60":"f434502896c4b4d9","61":"7b8b0a67b21bebdc","62":"926b1b14797b0839","63":"49d1914317f92bb0","64":"5618683527be35aa","65":"5aebbe49cb1e9c6a","66":"7ec8eefa0b5fd620","67":"373cf08ba41f622c","68":"b5a210cfc0cdecbe","69":"a199383da2eaf5ad","70":"572107db9b690d3f","71":"2e602d6096e3ce1c","72":"9e7ca8ef2d1dcc99","73":"11e6f457033d7aad","74":"6beae88bc4441fb7","75":"3ad08bdebc20a925","76":"dd7563e2704d1c80","77":"e1dc5d130662449e","78":"f5ecabda3efa0e28","79":"ae3fa60ab89e65e3","80":"dde656a3946a8026","81":"2bbc3392f746171d","82":"9fb3d1983c203da6","83":"591a7f3491aa0c70","84":"67183e11e7403030","85":"86b1a057da7ad9ef","86":"bc9b371b2e176501","87":"5af666204d8d0041","88":"75d71e52dbd38a45","89":"b2a39eec2f74fec8","90":"81bb0ba670e77d71","91":"ea220291e4d044ec","92":"edeca5bb28cb925f","93":"fa9c8496d1776008","94":"28927f0ea363b9d1","95":"f8332abe5eacde58","96":"552249045859f059","97":"5f6253658c7eca2d","98":"0cb061751bb3e3fe","99":"84efa5da34c93c98"},"7":{"1":"b17c4aaa972f97b5","2":"53390c603ce379c9","3":"865f851e6d61a85b","4":"be41ac70d045b205","5":"3b04f483f08fae13","6":"18bbd6b646a80751","7":"4d38536d57ec9f50","8":"08a17dc010e602c8","9":"cded31d39c66b212","10":"0432f7627254397e","11":"221a870fdcfe95c6","12":"a1521e9ef976117c","13":"fe83d3df49bc5276","14":"e029c4240d7c77ff","15":"a01b59968b4c16e7","16":"bdb6880448b2e436","17":"54dde53fd6f82f5c","18":"54586d4a4f4b57a4","19":"216650d1c86c925c","20":"9d2d315ee0e7c03d","21":"6040023b4957b673","22":"d4d7d653844a3267","23":"da8d1138039526e4","24":"810336449435cc2f","25":"536666b210a4cf60","26":"5c5cf1554491e35c","27":"62c6a935011be049","28":"4447a35db39116f9","29":"394251cada601254","30":"98aeb9f57acb8c7e","31":"f26fe7c12f14fb1b","32":"6de14077cab175bb","33":"f73f5940638d1670","34":"be0039e2e674052a","35":"5ca1f8015037d800","36":"22ebfd02bdabf7b1","37":"907744a3f2dde81b","38":"7e876cd8e6f7a935","39":"28f78f5d0bd2d1b5","40":"479d25e3c3923c31","41":"54de76181942e74b","42":"3905150e99443309","43":"b2150db23eb570ec","44":"836a806c75bd8147","45":"ab23e6e90d8cfa73","46":"c47f24792af9a108","47":"880b7adf51195bfb","48":"590942e74b882821","49":"0227bbd0f66287d7","50":"85bb0635ec01a8b4","51":"5f9bbf92cb51f2bb","52":"3b321bdb9a46ffad","53":"595f77ff3fa98496","54":"684f7119bc477df5","55":"fd66df7abd45a9de","56":"a531c430a61087ee","57":"0b9b2ec2afe2802e","58":"9d380b6fef00c224","59":"7c4c8d4063bdee61","60":"313e1a726a643dfe","61":"7b8b0a67b21bebdc","62":"926b1b14797b0839","63":"49d1914317f92bb0","64":"5618683527be35aa","65":"d1c1538d70d7d709","66":"0509e2b9431682f9","67":"b05f419c7a1417cc","68":"9a29e3123343c7db","69":"a199383da2eaf5ad","70":"572107db9b690d3f","71":"2e602d6096e3ce1c","72":"9e7ca8ef2d1dcc99","73":"b4ee6453d027a0c0","74":"43fa6157af25819e","75":"a335d1c3c8c5e604","76":"52a8e21971ef6324","77":"e1dc5d130662449e","78":"f5ecabda3efa0e28","79":"ae3fa60ab89e65e3","80":"dde656a3946a8026","81":"ce081078ae2d2256","82":"17bdfe68ee136112","83":"116b7a94feec54e2","84":"40a31edce5f42ac5","85":"86b1a057da7ad9ef","86":"bc9b371b2e176501","87":"5af666204d8d0041","88":"75d71e52dbd38a45","89":"bc94259bc2cda362","90":"af2ed2ca5019d4d6","91":"b39749d6f26c7bd0","92":"f2a9c8606b8996c0","93":"fa9c8496d1776008","94":"28927f0ea363b9d1","95":"f8332abe5eacde58","96":"552249045859f059","97":"a32b559bd0d80cff","98":"e723d05837738947","99":"f8e73fb2f48a5fd7"},"42":{"1":"b4b5ef22970acbcc","2":"88c775a4373cf64c","3":"818d71f984eea273","4":"55f16d2794b70d16","5":"cc94784565f764c9","6":"c69b8e5f922b1125","7":"290784948495218f","8":"b6805f5cc74a21a5","9":"647804426859ffee","10":"7d65b858bae5b2a6","11":"176c6768b58d6bb2","12":"980e8ca4d80d8f24","13":"13980a9b9d1c31aa","14":"4a8d38564d0c9302","15":"463014318f539292","16":"53e83bb5143ebd86","17":"327108699447064f","18":"27d6777db0dfc2a8","19":"ba2f85ecc7982774","20":"312be8314e5916e1","21":"c33c06a864b08465","22":"ee99abb903a5af41","23":"111cb2e761b252a5","24":"7b9a85cdf3e01543","25":"4b729fe7108bbd82","26":"cebdeb09af5f38a6","27":"63669039997232f3","28":"ff3ca973ea1f5a07","29":"1fcb1fd2103a41dd","30":"704f88a9c3d91dc7","31":"d6d44484ac976c80","32":"643fe06c368f37d2","33":"b32da6371c57d1e0","34":"02e0d59f87bd853c","35":"ba5e2bb2bb4d561b","36":"efeed4e7fdef1b7a","37":"8dec8a796a965801","38":"736022afe1e0a15f","39":"4fceab314038f223","40":"3709eb51f57bb065","41":"3a2b678b215954f7","42":"0e89369eb40a3642","43":"6289b6c27ea72366","44":"1d6ca80ea695bc16","45":"bdcefc9499ff281e","46":"8b18f203a2050aee","47":"96a26b00be6d7f97","48":"29f5963becc3ad91","49":"da81c2f5adb8f3fd","50":"c3f8481ca452f88a","51":"84905c0b438e953b","52":"53124261b8263495","53":"2642094ec8c74793","54":"b5da636d746926c1","55":"2a6a71a7b293947b","56":"1ad48be86e300525","57":"0b7de2a2e64f0ab4","58":"022a9567d140290e","59":"53f7a7dd614505c1","60":"ea5b6853b5d9ae51","61":"18a6f22273603cf1","62":"699aa4b07091cc56","63":"923eed4b33f85e1e","64":"2d89766c4f2a949b","65":"7afdaf56ac5e9435","66":"6d8c7d2f7d62db73","67":"8e11053229938468","68":"41491de691839cda","69":"3b8098c3422e1f42","70":"33e81c1006c31071","71":"33fe0fa0cdfb97f1","72":"2b03e362234229b0","73":"80c36a56c9138028","74":"6efe99a2f3f95740","75":"d1bd24d1aa7fcfa5","76":"b2617a4d518535cc","77":"df345c94d6f2fe41","78":"e6d72d156e4d9b7e","79":"164e3e631bc30bb0","80":"63d193d1734fa062","81":"e61b504af57579cd","82":"acbea7dc0b830f4a","83":"d1cd1dd4665ec0a9","84":"e8688ac4db006a25","85":"612f76481d3b1c5b","86":"1b7e7b624d130d9c","87":"fb0d209ecb9a0cf9","88":"6293eaf93905fdcf","89":"c48f4b43117dd137","90":"6167c93e458d4edd","91":"29ebe41b7ee4ba9f","92":"6fddbee38621b22f","93":"93bb50213dfabd0a","94":"beb009b76f3ce223","95":"9a8b9bd58fdda68d","96":"3285476811486179","97":"069a01bb85102118","98":"64043c77abcd3440","99":"87910c7712957782"},"2654435761":{"1":"780b1d507bbc2111","2":"9039fd17618a1c91","3":"b200f6630ad3cd74","4":"29edce1edea85680","5":"b893e44464d41ed0","6":"d08a0b1429b9c466","7":"ec8dca653e99f0a9","8":"cd82a068fd277ebe","9":"ad18be0735bd1cbf","10":"dbce1303c99c7f57","11":"ee8ff23fb246f869","12":"d5af7e02d0aa4f1d","13":"f2bb17bcd6e41c48","14":"59b2e2975b5d06aa","15":"c5f5880d6635e2dc","16":"a8f4e28deebe75e4","17":"749925609622ed21","18":"28fc6f372af2d58f","19":"dae6c99423aa3a31","20":"0cd43b103aa4d84c","21":"42db2517f3fc855b","22":"fcafddc5ec5cabdd","23":"33d12aafdd14ee8f","24":"54512fbfd530a23b","25":"9325ad192aac825b","26":"5f545b85fb8000f8","27":"b18f78845514a730","28":"7de246a4227f83ea","29":"405c4f32b5729783","30":"e0ee53dcc9326b9d","31":"f9e6b30d5b0bd73d","32":"9b33c7bad71f2516","33":"84507e2619a8e6aa","34":"07348ac0776dd120","35":"480beb00d4f74081","36":"4601c9e4fd59d2bd","37":"ba0d130c189a3f26","38":"90a53cd50c51bdaa","39":"7dbe59276d7d7efe","40":"bb197cd0d19edfac","41":"496b335217559b4c","42":"4f8e9f2d3d2dd154","43":"1893efc5d8828580","44":"4166ba140fcdec10","45":"6c6a64477cd8aebb","46":"072dd03234e1f6ea","47":"38b7ed7de957885f","48":"c498929156127d7e","49":"8b38033fd303056c","50":"d7e4643201dde542","51":"2923b66ffd0706db","52":"401915178c94c01a","53":"6d2010447190d885","54":"5364ef8acfd4fd1d","55":"a3ef409c08e394df","56":"f06d44f773bbe318","57":"3367dfeef7115579","58":"0f49498d8ba6c4bd","59":"1b33aa182c9bf178","60":"6de574e102d6d8bc","61":"70d7eb466a86951e","62":"d711e0099b41bd98","63":"f701bbffbcc4b0f4","64":"0295442622cb2104","65":"8eda3ccc6f73522a","66":"83388ee5ccde29e6","67":"ae1f036855d4ec12","68":"483885909e7e40aa","69":"233fa68448abdf1e","70":"839bac1c066d7fac","71":"b45b4c89c9140c71","72":"a80be6d2947bcb86","73":"224ff984993d5f4a","74":"9a4a45e2b5559883","75":"87ada9156c7c5ef7","76":"5649baa12c5d8e77","77":"0cf8b35c0167a8bd","78":"05ffa423e15e8640","79":"17a73d93002c3152","80":"688bf4e5672a3650","81":"3a9bbdfd159297b6","82":"bfe6937e4dee8c90","83":"23402f995ede05fb","84":"f5ab0490693aaadf","85":"52821ef0045f599c","86":"3728f7b3aa69ee2a","87":"83510bf213670f01","88":"d559a002aa58bdda","89":"fa314b812ddf3d5f","90":"03f37eb6d2bbdecf","91":"1134294275c6136e","92":"d3e7d4eec88c1985","93":"6d46ba5c33bb8992","94":"3a7f195a7689cc21","95":"048cb3a6abe1601d","96":"6511b1ee30782883","97":"35b8a25779a69249","98":"8a53bcba38f9d38f","99":"b21e7c685f18976a"},"4294967295":{"1":"1d492f904ced3b79","2":"0b1390463c46d354","3":"f257ccbcaf4b8074","4":"bbd71f4ff463dea5","5":"619236d60ddcc14e","6":"6662107e17e0b4dd","7":"f0f8f41b08a4580f","8":"f44067f3ee076177","9":"d964b0ecbecc6a84","10":"46b794c4370b4f7e","11":"8da9576efe948448","12":"6869e478c43d1b14","13":"252b148d4a4c6f79","14":"83255e25aaa5230c","15":"0b4cd4fc90d140a0","16":"fc0515550e5518d3","17":"24167f36089fc197","18":"d5e83406129b3784","19":"17c03f97aefd5b4e","20":"169b5e1aa2763fed","21":"751f9493d53a4102","22":"408c9cb9104489a2","23":"0c745c77ada77c8b","24":"2a60da1066010aa5","25":"b76c65fac95df255","26":"558b53ff1842cc05","27":"1e7b491c23f6325b","28":"302a93ab4dc96848","29":"bccb3cf4eafc7493","30":"a358a5432734cb93","31":"d1ee269749b22005","32":"e06dd80067cae65f","33":"35722d1108e01f47","34":"8a270712ede90e76","35":"3b57c78e44b8e11f","36":"d76e456c3238d93f","37":"fa936fdd1c4dda54","38":"f0dc0eaac51c909d","39":"80c19752c3f02684","40":"4882e4fc17fe4047","41":"01394f943c1c696e","42":"08d8fc626c96fe71","43":"4cd839dc4477cfd3","44":"0b123260345e5f22","45":"97600dc066cab1b1","46":"5fc582bfbd7175f3","47":"d007dde0a897fdc1","48":"a03f24a623acb2aa","49":"dfb7796027fd824a","50":"e039226d58a3b55f","51":"1c76e37a37142085","52":"defb5a807a321b79","53":"d72c36f23817ee98","54":"29039ae7e7e45041","55":"21c85ee049be40be","56":"1c2e4184353f709a","57":"8107e4f031418e2e","58":"e7c803ae736d823f","59":"414e18cd3ae73432","60":"0e66f711966c8a0c","61":"4183623aebd881e6","62":"9aef7fe23f7e559a","63":"8eca1e1ba03a4906","64":"11c33132f1a5bd06","65":"b75b588eb8620b67","66":"d20bf4637a724637","67":"688c0b2ae7c497a6","68":"a787981aa5e54f35","69":"8164733570f2acf6","70":"0ae6bbc6a1bfc789","71":"946de460fa0cce0a","72":"6d0050fa9430c1ab","73":"7f9dea65fdf59f09","74":"28dd05aa98dbd082","75":"b24bcca0439eb343","76":"0dde1fbe5828f93a","77":"e022f699d921c7fd","78":"95152e00d5106df2","79":"74a3b5f1c6b9dc23","80":"68409394093cfbe1","81":"23d2e40f42b14e68","82":"c2f7a0e1f5057873","83":"3a0de8e533ce3725","84":"6873108de6321ea6","85":"1036d183146fba54","86":"135f4ac16dc29a56","87":"5c13d2d63fa7fbbd","88":"50243755f1150e23","89":"e1884e75bbcfae47","90":"3e39e6a3e68a8dfc","91":"43dd0ffec0ddd3f4","92":"a5764e9b8a32037f","93":"d1be1f5ffbb0d190","94":"725a5ce4d575450c","95":"47aa3349fa3da0f9","96":"613d76184d4f928a","97":"aec90ecbaa7a9d8d","98":"b91b1de168d7a4e3","99":"bc8d3b1dce9499e6"}};
const GOLDEN_V1 = {"1":{"1":"caab32e6e15bd6d7","2":"67139ff7454b023a","25":"ce99f9bbe31bdf57","50":"1399522a57b967e3","75":"8799a4612380ad70","99":"b47f7b90209dde22"},"7":{"1":"4b0be48602f947a6","2":"5ae4082f8fe6dcaf","25":"3e7995b5171a733b","50":"9a690fe519ab276c","75":"2af740fde4f64741","99":"d750408ca6076b25"},"42":{"1":"d92a6a7015f6f943","2":"c9fce9eecf6d3380","25":"8f4d111ab4502cd2","50":"620ad1dff6229e60","75":"1b0f6b4e48dd7c77","99":"ba0f424463ca7f0c"},"2654435761":{"1":"78e3494955a5699d","2":"c66395ab07eb3d46","25":"399bd0612d4331c7","50":"4e8de91ec5b6a3eb","75":"a1dea1f1af4f4136","99":"a3e7c3250c12ed0f"},"4294967295":{"1":"99de0fe90dd709ae","2":"f027c4cf4f5ae6be","25":"2f303cf03183ff9d","50":"5223dede9a221269","75":"6c0518e28a943e95","99":"eb8f63e7c0556139"}};
const GOLDEN_WALL_V2 = {"1":{"1":"2a4db31785a71a82","50":"40d67fb0f19b32b5","99":"90e4b2bcfc56af27"},"42":{"1":"a3d3993c20ad5695","50":"6c8fd3409e49efde","99":"0ada6039c2e781b6"},"2654435761":{"1":"93327441a75f79b9","50":"c0960c12274e89b3","99":"e449ab8a172ac7d7"}};
const CAPTURE_SEEDS = [1, 7, 42, 2654435761, 4294967295];
const CAPTURE_R = [1, 2, 25, 50, 75, 99];
const WALL_CAPTURE_SEEDS = [1, 42, 2654435761];
const WALL_CAPTURE_R = [1, 50, 99];

console.log('car_racer rotation V6 suite (FIT-V6 TASK G2)');
console.log('-------------------------------------------');

// ---------------------------------------------------------------------------
await test('G2-1 halfWidth ladder is 50 - 0.25 d (50..25) for both versions', () => {
  for (const d of DIFFICULTIES) assert(Rotation.halfWidthForDifficulty(d) === halfWidthOf(d),
    'ladder d=' + d + ' got ' + Rotation.halfWidthForDifficulty(d) + ' want ' + halfWidthOf(d));
  assert(Rotation.halfWidthForDifficulty(0) === 50, 'hw(0)=50');
  assert(Rotation.halfWidthForDifficulty(100) === 25, 'hw(100)=25');
  for (const r of CAPTURE_R) {
    const v2 = Rotation.resolveGeneratedTrack(1, r, 2);
    const v1 = Rotation.resolveGeneratedTrack(1, r, 1);
    assert(v2.halfWidth === halfWidthOf(r), 'V2 hw r=' + r);
    assert(v1.halfWidth === halfWidthOf(r), 'V1 hw r=' + r);
  }
});

await test('G2-2 V1 generator literals unchanged (14 points, radii, fallback, ladders)', () => {
  assert(Rotation.TRACK_POINTS === 14, 'TRACK_POINTS 14');
  assert(Rotation.TRACK_MAX_RADIUS === 245, 'TRACK_MAX_RADIUS 245');
  assert(Rotation.TRACK_FALLBACK_RADIUS === 170, 'TRACK_FALLBACK_RADIUS 170');
  assert(Rotation.TRACK_MAX_REJECTS === 200, 'TRACK_MAX_REJECTS 200');
  for (const d of DIFFICULTIES) {
    assert(Rotation.harmonicsForDifficulty(d) === 2 + Math.floor(d / 25), 'harmonics d=' + d);
    assert(Rotation.amplitudeForDifficulty(d) === Number((0.08 + 0.34 * (d / 100)).toFixed(9)), 'amplitude d=' + d);
  }
});

await test('G2-3 templates are the arc-built F1 hairpin arrays (C1,C2,C3,B2,B3,A2)', () => {
  assert(Rotation.V5_TEMPLATES.length === 6, 'six templates');
  for (let i = 0; i < 6; i++) {
    const got = Rotation.V5_TEMPLATES[i];
    const want = TEMPLATES[i];
    assert(got.length === want.length && got.length >= 91 && got.length <= 96, 'template ' + i + ' has 91-96 points');
    for (let j = 0; j < want.length; j++) {
      assert(got[j][0] === want[j][0] && got[j][1] === want[j][1], 'template ' + i + ' point ' + j + ' matches spike array');
    }
  }
});

await test('G2-4 legacyV2Hash fully removed (Rotation + CommonJS + source)', () => {
  assert(Rotation.legacyV2Hash === undefined, 'Rotation.legacyV2Hash absent');
  assert(rotationMod.legacyV2Hash === undefined, 'CommonJS legacyV2Hash absent');
  const src = readFileSync(ROOT + 'js/rotation.js', 'utf8');
  assert(!/legacyV2Hash/.test(src), 'no legacyV2Hash reference in rotation.js');
  assert(!/legacyGenerateTrackV2/.test(src), 'no legacyGenerateTrackV2 in rotation.js');
});

await test('G2-5 two-mode version parsers unchanged', () => {
  assert(Rotation.parseGeneratorVersion() === 1, 'absent -> 1');
  assert(Rotation.parseGeneratorVersion(null) === 1, 'null -> 1');
  assert(Rotation.parseGeneratorVersion(1) === 1 && Rotation.parseGeneratorVersion('1') === 1, '1 / "1"');
  assert(Rotation.parseGeneratorVersion(2) === 2 && Rotation.parseGeneratorVersion('2') === 2, '2 / "2"');
  for (const bad of [0, 3, '', ' ', 'x', 2.5]) assertValidationError(() => Rotation.parseGeneratorVersion(bad), 'bad ' + JSON.stringify(bad));
  assert(Rotation.requireGeneratorVersion(2) === 2, 'require 2');
  for (const bad of [undefined, null, '']) assertValidationError(() => Rotation.requireGeneratorVersion(bad), 'required ' + JSON.stringify(bad));
});

await test('G2-6 version dispatch canonical; no inline version checks', () => {
  const src = readFileSync(ROOT + 'js/rotation.js', 'utf8');
  // No inline `=== 2` / `|| 1` version interpretation outside the parser.
  const inline = src.split('\n').filter((l) => /version\s*===\s*[12]|version\s*==\s*[12]|\|\|\s*1\b/.test(l) &&
    !/^\s*\/\//.test(l) &&
    !/function parseGeneratorVersion|raw === 1|raw === 2|raw === '1'|raw === '2'|version === undefined/.test(l));
  assert(inline.length === 0, 'no inline version checks: ' + inline.join(' | '));
  assertValidationError(() => Rotation.resolveGeneratedTrack(1, 1, 3), 'version 3 rejected');
  assertValidationError(() => Rotation.resolveGeneratedTrack(1, 1, 1.5), 'version 1.5 rejected');
});

await test('G2-7 five resolver sites pass an explicit version', () => {
  const files = ['js/main.js', 'js/trainer.js', 'js/train-worker.js', 'js/db.js', 'js/db-session.js'];
  let count = 0;
  for (const f of files) {
    const s = readFileSync(ROOT + f, 'utf8');
    const re = /(?:R|Rotation|requireRotation\(\))\.resolveGeneratedTrack\s*\([^)]*\)/g;
    let m;
    while ((m = re.exec(s))) {
      // every production call must carry a third argument (explicit version)
      const args = m[0].slice(m[0].indexOf('(') + 1, -1);
      const parts = args.split(',').map((x) => x.trim()).filter(Boolean);
      if (parts.length >= 3) count++;
      else assert(false, f + ': resolver call lacks explicit version: ' + m[0]);
    }
  }
  assert(count >= 5, 'found ' + count + ' explicit-version resolver sites');
});

await test('G2-8 apex-radius and arm-separation helpers exported', () => {
  assert(typeof Rotation.v6ApexRadius === 'function', 'v6ApexRadius exported');
  assert(typeof Rotation.v6ArmSeparation === 'function', 'v6ArmSeparation exported');
  for (let t = 0; t < 6; t++) {
    const pts = toPoints(TEMPLATES[t]);
    const R = Rotation.v6ApexRadius(pts);
    assert(R >= 75, 'template ' + t + ' apex radius ' + R + ' >= 75');
    assert(Rotation.v6ArmSeparation(pts) > 2 * 50 + 20, 'template ' + t + ' arm separation');
    assert(Rotation.v6ArmSeparation(pts) === 2 * R, 'arm separation = 2R');
  }
});

await test('G2-9 golden vectors: 5 seeds x r=1..99 = 495 V2 hashes', () => {
  for (const s of CAPTURE_SEEDS) for (let r = 1; r <= 99; r++) {
    const got = Rotation.resolveGeneratedTrack(s, r, 2).trackHash;
    if (CAPTURE) {
      (GOLDEN_V2[s] || (GOLDEN_V2[s] = {}))[r] = got;
      continue;
    }
    assert(GOLDEN_V2[s] && GOLDEN_V2[s][r], 'golden exists seed=' + s + ' r=' + r);
    assert(got === GOLDEN_V2[s][r], 'V2 hash seed=' + s + ' r=' + r + ' got ' + got + ' want ' + GOLDEN_V2[s][r]);
  }
});

await test('G2-10 V1 golden vectors regenerated against new ladder', () => {
  for (const s of CAPTURE_SEEDS) for (const r of CAPTURE_R) {
    const got = Rotation.resolveGeneratedTrack(s, r, 1).trackHash;
    if (CAPTURE) { (GOLDEN_V1[s] || (GOLDEN_V1[s] = {}))[r] = got; continue; }
    assert(GOLDEN_V1[s] && GOLDEN_V1[s][r], 'V1 golden exists seed=' + s + ' r=' + r);
    assert(got === GOLDEN_V1[s][r], 'V1 hash seed=' + s + ' r=' + r);
  }
});

await test('G2-11 never throws across 400+ seeds x r=1..99 (V2)', () => {
  // 400 seeds spread across the whole uint32 space (not just 0..399) plus
  // explicit edge/regression seeds that historically broke the fallback.
  const seeds = [];
  for (let i = 0; i < 400; i++) seeds.push((Math.imul(i + 1, 0x9E3779B1)) >>> 0);
  for (const s of [0, 1, 3669806527, 90802087, 441399996, 4294967295]) seeds.push(s);
  let n = 0;
  for (const s of seeds) for (let r = 1; r <= 99; r++) {
    let out;
    try { out = Rotation.resolveGeneratedTrack(s, r, 2); } catch (err) {
      assert(false, 'threw seed=' + s + ' r=' + r + ': ' + err.message);
    }
    assert(Array.isArray(out.controlPoints) && out.controlPoints.length >= 14, 'points r=' + r);
    n++;
  }
  assert(n === seeds.length * 99, 'covered ' + n + ' cases');
});

await test('G2-12 determinism: same (seed,r,version) yields identical bytes', () => {
  for (const s of [0, 1, 42, 2654435761, 4294967295]) for (const r of [1, 25, 50, 99]) {
    const a = Rotation.resolveGeneratedTrack(s, r, 2);
    const b = Rotation.resolveGeneratedTrack(s, r, 2);
    assert(hashOf(a.controlPoints, a.halfWidth) === hashOf(b.controlPoints, b.halfWidth), 'deterministic ' + s + ':' + r);
    assert(a.trackHash === b.trackHash, 'hash deterministic ' + s + ':' + r);
  }
});

await test('G2-13 apex/feasibility gates hold across d in {0,25,50,75,100}', () => {
  for (let t = 0; t < 6; t++) {
    for (const d of DIFFICULTIES) {
      const hw = halfWidthOf(d);
      // pick a seed/r that maps to template t
      let seed = 0, r = 1;
      while (templateIndex(seed, r) !== t && seed < 500) seed++;
      assert(templateIndex(seed, r) === t, 'found seed for template ' + t);
      const out = Rotation.resolveGeneratedTrack(seed, r, 2);
      const R = Rotation.v6ApexRadius(out.controlPoints);
      assert(R >= 75, 't' + t + ' d' + d + ' apex ' + R + ' >= 75');
      assert(2 * R > 2 * hw + 20, 't' + t + ' d' + d + ' arm separation');
      assert(tanDeviation(out.controlPoints) <= 1.0, 't' + t + ' d' + d + ' tangent <= 1deg');
      assert(polylineSimple(out.controlPoints), 't' + t + ' d' + d + ' control polygon simple');
    }
  }
});

await test('G2-14 wall-byte golden agreement with G1 (30 vectors)', () => {
  for (let t = 0; t < 6; t++) for (const d of DIFFICULTIES) {
    const track = new Track(toPoints(G1_TEMPLATES[t]), halfWidthOf(d));
    const got = wallHash(track);
    const key = t + ':' + d;
    assert(got === G1_WALL_GOLDEN[key], 'wall hash ' + key + ' got ' + got + ' want ' + G1_WALL_GOLDEN[key]);
  }
});

await test('G2-15 generated tracks produce canonical clipped walls (no crossings, finite)', () => {
  for (const s of [0, 1, 42, 2654435761]) for (const r of [1, 25, 50, 99]) {
    const out = Rotation.resolveGeneratedTrack(s, r, 2);
    const track = new Track(out.controlPoints, out.halfWidth);
    assert(track.leftWall.length === track.rightWall.length, 'equal wall lengths ' + s + ':' + r);
    assert(track.segments.length === 2 * track.leftWall.length, 'segments from walls ' + s + ':' + r);
    for (const w of [track.leftWall, track.rightWall]) {
      assert(polylineSimple(w), 'clipped wall simple ' + s + ':' + r);
      for (const p of w) assert(Number.isFinite(p.x) && Number.isFinite(p.y), 'wall finite ' + s + ':' + r);
    }
  }
});

await test('G2-16 generated wall-byte hashes pinned (G1 wallHash on resolved tracks)', () => {
  for (const s of WALL_CAPTURE_SEEDS) for (const r of WALL_CAPTURE_R) {
    const out = Rotation.resolveGeneratedTrack(s, r, 2);
    const got = wallHash(new Track(out.controlPoints, out.halfWidth));
    if (CAPTURE) { (GOLDEN_WALL_V2[s] || (GOLDEN_WALL_V2[s] = {}))[r] = got; continue; }
    assert(GOLDEN_WALL_V2[s] && GOLDEN_WALL_V2[s][r], 'wall golden exists ' + s + ':' + r);
    assert(got === GOLDEN_WALL_V2[s][r], 'wall hash ' + s + ':' + r + ' got ' + got + ' want ' + GOLDEN_WALL_V2[s][r]);
  }
});

await test('G2-18 pairwise-distinct hashes across r=1..99 (caller-side gate)', () => {
  for (const s of CAPTURE_SEEDS) {
    const seen = new Set();
    for (let r = 1; r <= 99; r++) {
      const h = Rotation.resolveGeneratedTrack(s, r, 2).trackHash;
      assert(!seen.has(h), 'duplicate hash seed=' + s + ' r=' + r + ' hash=' + h);
      seen.add(h);
    }
    assert(seen.size === 99, 'seed ' + s + ' produced 99 distinct hashes');
  }
});

await test('G2-17 200-attempt cap and full-redraw machinery preserved (source)', () => {
  const src = readFileSync(ROOT + 'js/rotation.js', 'utf8');
  assert(/V5_MAX_ATTEMPTS\s*=\s*200/.test(src), '200-attempt cap');
  assert(/attempt < V5_MAX_ATTEMPTS/.test(src), 'attempt loop');
  assert(/0x9E3779B1/.test(src), 'template permutation constant');
  assert(/0\.10/.test(src), 'theta scale 0.10');
  assert(/v5TangentWithinPreFilter/.test(src), 'tangent pre-filter present');
});

// ---------------------------------------------------------------------------
// FIT-V6 cycle-3 S1 — genuine hairpins + dense-spline clipped-wall clearance.
// ---------------------------------------------------------------------------
// Largest continuous same-sign deflection over the closed control polygon.
// On a densely sampled circular hairpin this accumulates the arc's turn; any
// run >= 150deg is a genuine U-turn (not a single sparse vertex).
function hairpinDeg(points) {
  const n = points.length;
  let best = 0, run = 0, sign = 0;
  for (let i = 0; i < 2 * n; i++) {
    const a = points[(i - 1 + n) % n], b = points[i % n], c = points[(i + 1) % n];
    const e1x = b.x - a.x, e1y = b.y - a.y, e2x = c.x - b.x, e2y = c.y - b.y;
    const d = Math.atan2(e1x * e2y - e1y * e2x, e1x * e2x + e1y * e2y) * 180 / Math.PI;
    const sg = Math.sign(d);
    if (sg === 0) continue;
    if (sg !== sign) { sign = sg; run = 0; }
    run += Math.abs(d);
    if (run > best) best = run;
  }
  return best;
}
function segDist2(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1, L2 = dx * dx + dy * dy;
  let t = L2 > 0 ? ((px - x1) * dx + (py - y1) * dy) / L2 : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}
function exactClearance(track) {
  let min = Infinity;
  for (const s of track.samples) for (const g of track.segments) {
    const d = segDist2(s.x, s.y, g.x1, g.y1, g.x2, g.y2);
    if (d < min) min = d;
  }
  return min;
}

await test('G2-19 every template carries a genuine >=150-degree U-turn hairpin', () => {
  for (let i = 0; i < 6; i++) {
    const hp = hairpinDeg(toPoints(TEMPLATES[i]));
    assert(hp >= 150, 'template ' + i + ' hairpin ' + hp.toFixed(1) + ' >= 150');
    assert(TEMPLATES[i].length >= 91 && TEMPLATES[i].length <= 96, 'template ' + i + ' has 91-96 points');
  }
});

await test('G2-20 dense-spline clearance: CAR_RADIUS+4=18px to CLIPPED walls at hw 50/49.75/37.25/25, Car at every sample, 0 crashes', () => {
  const { Car } = require('../js/car.js');
  for (let t = 0; t < 6; t++) {
    for (const hw of [50, 49.75, 37.25, 25]) {
      const tr = new Track(toPoints(TEMPLATES[t]), hw);
      const cl = exactClearance(tr);
      assert(cl >= 18, 't' + t + ' hw' + hw + ' clearance ' + cl.toFixed(3) + ' >= 18');
      let crashes = 0;
      for (const s of tr.samples) {
        const car = new Car(tr, s.x, s.y, s.tangent);
        car.update(1 / 120, { gas: 0, steer: 0 });
        if (car.crashed) crashes++;
      }
      assert(crashes === 0, 't' + t + ' hw' + hw + ' car-at-sample crashes ' + crashes);
    }
  }
});

await test('G2-21 clearance gate fails closed on the old sharp (cusp) template', () => {
  assert(typeof Rotation.v5Valid === 'function', 'v5Valid exported as a test seam');
  assert(typeof Rotation.v6TrackClearance === 'function', 'v6TrackClearance exported');
  assert(typeof Rotation.v6SplineRadius === 'function', 'v6SplineRadius exported');
  const cusp = toPoints(G1_TEMPLATES[0]);
  const tr = new Track(cusp, 50);
  assert(Rotation.v6TrackClearance(tr) < 18, 'old sharp template clearance < 18');
  assert(Rotation.v5Valid(cusp, 50) === false, 'v5Valid rejects the cusp candidate (fail closed)');
});

// ---------------------------------------------------------------------------
if (CAPTURE) {
  console.log('\n--- GOLDEN_V2 ---');
  console.log(JSON.stringify(GOLDEN_V2));
  console.log('--- GOLDEN_V1 ---');
  console.log(JSON.stringify(GOLDEN_V1));
  console.log('--- GOLDEN_WALL_V2 ---');
  console.log(JSON.stringify(GOLDEN_WALL_V2));
  process.exit(0);
}
console.log('\ntests passed: ' + passed + '/' + (passed + failures.length));
if (failures.length) {
  console.log('FAILURES: ' + failures.length);
  for (const f of failures) console.log('  - ' + f.name + ': ' + String(f.err.message || f.err));
  process.exit(1);
}
