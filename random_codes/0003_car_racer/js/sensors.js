// js/sensors.js — 5-ray sensor fan for the 0003_car_racer NN drivers.
//
// Layout (EXACT, ordered — see plan SENSOR_ANGLES / CONSTANTS):
//     SENSOR_ANGLES_DEG = [-70, -35, 0, +35, +70]   (degrees)
//   Sensor i ray direction = car.heading + degToRad(SENSOR_ANGLES_DEG[i])
//   (canvas convention: y-down, positive angle = clockwise; steer +1 = right)
//   Sensor origin = car.driverSeat()  (this arcade model: the car centre —
//   documented in car.js).
//
// read() → {
//   norm: Float32Array(5)  — distance / SENSOR_MAX, clamped to [0,1];
//                            1 when the ray hits nothing within SENSOR_MAX
//   px:   plain array(5)   — raw raycast distance in px, capped at maxDist
// }
// Distances are measured against the track's wall segments via the grid
// raycaster — exactly what the NN sees at inference time.
//
// Boundary semantics: a ray hitting a wall at exactly t == SENSOR_MAX →
// norm === 1.0, indistinguishable from no-hit (measure-zero, benign); same
// for t == 0.
//
// Plain classic script; browser globals on globalThis; guarded CommonJS
// export at the bottom for node tests.

'use strict';

// Shared ValidationError — see nn.js for the rationale (script order safety).
if (typeof globalThis.ValidationError !== 'function') {
  globalThis.ValidationError = class ValidationError extends Error {
    constructor(message) {
      super(message);
      this.name = 'ValidationError';
    }
  };
}
var ValidationError = globalThis.ValidationError;

var SENSOR_MAX = 300;                       // px — NN-input normaliser
var SENSOR_ANGLES_DEG = [-70, -35, 0, 35, 70];
var SENSOR_ANGLES_RAD = SENSOR_ANGLES_DEG.map(function (deg) { return (deg * Math.PI) / 180; });
var SENSOR_COUNT = SENSOR_ANGLES_DEG.length; // 5

// ---------------------------------------------------------------------------
// SensorFan(car, track, maxDist = SENSOR_MAX)
//   car:   a Car instance — heading + driverSeat() are read on every read()
//   track: a Track instance — raycast() is used for the measurements
//   maxDist: ray length cap in px (defaults to SENSOR_MAX)
// ---------------------------------------------------------------------------
function SensorFan(car, track, maxDist) {
  if (!car || typeof car.driverSeat !== 'function' || !Number.isFinite(car.heading)) {
    throw new ValidationError('SensorFan requires a Car-like object (driverSeat(), heading)');
  }
  if (!track || typeof track.raycast !== 'function') {
    throw new ValidationError('SensorFan requires a Track-like object (raycast())');
  }
  var md = maxDist === undefined ? SENSOR_MAX : Number(maxDist);
  if (!Number.isFinite(md) || md <= 0) {
    throw new ValidationError('SensorFan maxDist must be a finite number > 0, got ' + String(maxDist));
  }
  this.car = car;
  this.track = track;
  this.maxDist = md;
}

// read() — cast all five rays. No allocations besides the small result object.
SensorFan.prototype.read = function read() {
  var origin = this.car.driverSeat();
  var heading = this.car.heading;
  var maxDist = this.maxDist;
  var ray = this.track.raycast;

  var norm = new Float32Array(SENSOR_COUNT);
  var px = new Array(SENSOR_COUNT);
  for (var i = 0; i < SENSOR_COUNT; i++) {
    var d = ray.call(this.track, origin.x, origin.y, heading + SENSOR_ANGLES_RAD[i], maxDist);
    // raycast caps at maxDist by contract; the min() guard is belt-and-braces
    // in case a custom maxDist is ever passed.
    if (!(d >= 0)) d = maxDist;             // NaN/∞ defence → "no hit"
    if (d > maxDist) d = maxDist;
    px[i] = d;
    norm[i] = d / maxDist;                  // ∈ [0,1]; 1 == no hit
  }
  return { norm: norm, px: px };
};

// Browser-global exposure + guarded CommonJS export (node tests).
if (typeof globalThis !== 'undefined') {
  globalThis.SensorFan = SensorFan;
  globalThis.SENSOR_MAX = SENSOR_MAX;
  globalThis.SENSOR_ANGLES_DEG = SENSOR_ANGLES_DEG;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    SensorFan: SensorFan,
    ValidationError: ValidationError,
    SENSOR_MAX: SENSOR_MAX,
    SENSOR_ANGLES_DEG: SENSOR_ANGLES_DEG,
  };
}
