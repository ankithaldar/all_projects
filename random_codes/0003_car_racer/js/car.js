// js/car.js — Arcade car physics for the 0003_car_racer game.
//
// PHYSICS MODEL (arcade, documented — not a simulation of a real vehicle):
//
//   Inputs:  { gas ∈ [−1,1], steer ∈ [−1,1] }   (values are clamped; NaN/∞ → 0)
//     gas > 0  → throttle. Force fades as speed approaches MAX_SPEED:
//                 F = gas * ENGINE_ACCEL * (1 − speed / MAX_SPEED)
//                (at speed == MAX_SPEED the throttle force is exactly 0)
//                The fade is clamped to [0, 1.25]. The 1.25 ceiling exists
//                for REVERSE motion: at full reverse (speed = −150) the raw
//                fade would be 1.25, i.e. the engine pulls at 1.25× the
//                forward force until speed rises back toward 0 — a deliberate
//                reverse-boost so the car can leave a dead standstill in
//                reverse with real authority. Forward motion never exceeds a
//                fade of 1 (speed ≥ 0 ⇒ fade ≤ 1).
//     gas < 0  → brake. Deceleration = |gas| * BRAKE_ACCEL applied opposite
//                the current motion; once stopped it acts as REVERSE throttle,
//                limited to REVERSE_MAX_SPEED (with the same (1 − s/max) fade).
//   Drag (always, opposes motion):  a_drag = DRAG * speed  (linear in speed)
//   Rolling resistance:              a_roll = ROLLING_RESISTANCE (opposes sign of motion; zeroes tiny speeds)
//
//   Top speed: equilibrium of gas*ENGINE*(1 − s/MAX_SPEED) = DRAG*s + ROLLING.
//   With ENGINE_ACCEL=550, MAX_SPEED=600, DRAG=0.05, ROLLING=2:
//     550 − (550/600)·s = 0.05·s + 2  ⇒  s ≈ (550 − 2) / 0.9667 ≈ 567 px/s.
//   "Near MAX_SPEED" (~94 %) is by design; MAX_SPEED is a hard clamp, not a
//   guaranteed equilibrium. Speed is clamped to [−REVERSE_MAX_SPEED, MAX_SPEED]
//   after every integration step.
//
//   Steering: steerAngle = steer * maxSteerAngle(speed), where maxSteerAngle
//   falls linearly from MAX_STEER_LOW (0.6 rad @ 0 speed) to MAX_STEER_HIGH
//   (0.18 rad @ MAX_SPEED, proportional to |speed|). yaw rate via the bicycle
//   model:  heading += (speed / WHEEL_BASE) * tan(steerAngle) * dt
//   (y-down canvas ⇒ positive angle = clockwise ⇒ steer +1 = RIGHT = clockwise
//    heading increase — matches the plan's convention).
//
//   Position: x += cos(heading) * speed * dt ; y += sin(heading) * speed * dt
//   Odometer: odometer += speed * dt  (net signed px — the odometer component
//     of fitness; fitness may be signed; NO monotonicity claim of any kind)
//   Collision: the car is a CIRCLE of radius CAR_RADIUS (14 px). If any wall
//   segment comes within < CAR_RADIUS of the car centre (segment-point
//   distance), the car CRASHES: crashed = true, speed = 0. A crashed car is
//   frozen (update() is a no-op) until reset() is called.
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

// Track's segment-point distance helper is REQUIRED for crash detection.
// Script order Track → Car is guaranteed in the browser (plan: Script order),
// and the node test loader requires track.js before car.js. There is NO
// duplicated collision math anywhere: the shared helper lives on Track.
// Failing loudly at load time beats a half-working car discovered mid-run.
var segmentPointDistanceSquared =
  (typeof globalThis !== 'undefined' && globalThis.Track && globalThis.Track.segmentPointDistanceSquared) ||
  null;
if (typeof segmentPointDistanceSquared !== 'function') {
  throw new ValidationError('car.js requires track.js to load first (global Track.segmentPointDistanceSquared is missing)');
}

// ---------------------------------------------------------------------------
// Constants — single source of truth for car physics.
// MAX_SPEED must match the track design budget (see plan CONSTANTS).
// ---------------------------------------------------------------------------
var MAX_SPEED = 600;              // px/s — hard clamp, and the NN-input normaliser
var REVERSE_MAX_SPEED = 150;      // px/s — reverse gear limit
var ENGINE_ACCEL = 550;           // px/s² — throttle force at 0 speed (before fade)
var BRAKE_ACCEL = 600;            // px/s² — brake / reverse force
var DRAG = 0.05;                  // 1/s — linear drag coefficient (a = DRAG * speed)
var ROLLING_RESISTANCE = 2;       // px/s² — constant rolling resistance
var WHEEL_BASE = 48;              // px — bicycle-model wheelbase
var CAR_RADIUS = 14;              // px — collision circle (covers body ~30x18)
var MAX_STEER_LOW = 0.6;          // rad — max steer angle at standstill
var MAX_STEER_HIGH = 0.18;        // rad — max steer angle at MAX_SPEED

function clampNumber(value, lo, hi, fallback) {
  var v = Number(value);
  if (!Number.isFinite(v)) v = fallback;
  return v < lo ? lo : v > hi ? hi : v;
}

// ---------------------------------------------------------------------------
// Car — one driver's vehicle.
//
// constructor(track, x, y, heading)
//   track:   a Track instance (wall segments are read from .segments; startPose
//            from .startPose()); the track is stored by reference, never mutated.
//   x, y:    spawn position in px
//   heading: spawn heading in radians (canvas convention: + = clockwise)
//
// State (public, read-only by convention):
//   x, y, heading (rad), speed (px/s, signed: < 0 means reversing),
//   odometer (px, NET SIGNED distance — Σ speed·dt; may be negative),
//   crashed (bool)
// ---------------------------------------------------------------------------
function Car(track, x, y, heading) {
  if (!track || !Array.isArray(track.segments) || typeof track.startPose !== 'function') {
    throw new ValidationError('Car requires a Track-like object (segments[], startPose())');
  }
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(heading)) {
    throw new ValidationError('Car spawn requires finite x, y, heading');
  }
  this.track = track;
  this.spawnX = x;
  this.spawnY = y;
  this.spawnHeading = heading;
  this.x = x;
  this.y = y;
  this.heading = heading;
  this.speed = 0;      // signed px/s; negative = reversing
  this.odometer = 0;   // px, NET signed distance (Σ speed·dt) — may be negative (odometer component of fitness)
  this.crashed = false;
}

// update(dt, inputs) — advance the physics by dt seconds.
//   dt: seconds; non-finite or <= 0 → no-op (documented, defensive).
//   inputs: { gas ∈ [−1,1], steer ∈ [−1,1] }; missing/NaN → 0.
Car.prototype.update = function update(dt, inputs) {
  if (!Number.isFinite(dt) || dt <= 0) return; // dead time is not simulated
  if (this.crashed) return;                    // wrecked cars do not move

  var gas = clampNumber(inputs && inputs.gas, -1, 1, 0);
  var steer = clampNumber(inputs && inputs.steer, -1, 1, 0);

  // ---- longitudinal: throttle / brake / reverse ---------------------------
  if (gas >= 0) {
    // Throttle force fades linearly as speed approaches MAX_SPEED.
    // Clamped to [0, 1.25] — see the PHYSICS MODEL header for the 1.25
    // reverse-boost ceiling this permits at full reverse.
    var fade = clampNumber(1 - this.speed / MAX_SPEED, 0, 1.25, 0);
    this.speed += gas * ENGINE_ACCEL * fade * dt;
  } else {
    var brake = -gas * BRAKE_ACCEL; // |gas| * BRAKE_ACCEL, always positive
    if (this.speed > 0) {
      // Braking against forward motion — never crosses zero in one tick.
      this.speed = Math.max(0, this.speed - brake * dt);
    } else {
      // From standstill (or while reversing): reverse throttle with fade.
      var reverseFade = Math.max(0, 1 - Math.abs(this.speed) / REVERSE_MAX_SPEED);
      this.speed = Math.max(-REVERSE_MAX_SPEED, this.speed - brake * reverseFade * dt);
    }
  }

  // ---- drag + rolling resistance (always oppose motion) -------------------
  this.speed -= DRAG * this.speed * dt;
  if (this.speed !== 0) {
    var roll = ROLLING_RESISTANCE * dt;
    if (Math.abs(this.speed) <= roll) {
      this.speed = 0; // stop completely rather than oscillate around 0
    } else {
      this.speed -= Math.sign(this.speed) * roll;
    }
  }

  // ---- hard speed clamp ----------------------------------------------------
  this.speed = clampNumber(this.speed, -REVERSE_MAX_SPEED, MAX_SPEED, 0);

  // ---- steering (speed-dependent) ------------------------------------------
  var speedRatio = Math.min(1, Math.abs(this.speed) / MAX_SPEED);
  var maxSteer = MAX_STEER_LOW + (MAX_STEER_HIGH - MAX_STEER_LOW) * speedRatio;
  var steerAngle = steer * maxSteer;
  // Bicycle model yaw rate. Positive speed swings the heading forward;
  // while reversing, tan(steerAngle) flips the effect via speed's sign.
  this.heading += (this.speed / WHEEL_BASE) * Math.tan(steerAngle) * dt;

  // ---- integrate position + odometer ----------------------------------------
  this.x += Math.cos(this.heading) * this.speed * dt;
  this.y += Math.sin(this.heading) * this.speed * dt;
  this.odometer += this.speed * dt; // net px (signed) — may decrease under reverse

  // ---- collision: circle vs every wall segment ------------------------------
  // O(segments) per tick. Segment count is ~2× track samples (≈ 1050 for the
  // default circuit) — acceptable for 20 cars @ 120 Hz, and the uniform grid
  // is reserved for raycasting where it pays off.
  var radiusSq = CAR_RADIUS * CAR_RADIUS;
  var segs = this.track.segments;
  for (var i = 0; i < segs.length; i++) {
    var seg = segs[i];
    if (segmentPointDistanceSquared(seg.x1, seg.y1, seg.x2, seg.y2, this.x, this.y) < radiusSq) {
      this.crashed = true;
      this.speed = 0;
      return;
    }
  }
};

// reset() — teleport back to the track's start pose, clear all dynamics.
Car.prototype.reset = function reset() {
  var pose = this.track.startPose();
  this.x = pose.x;
  this.y = pose.y;
  this.heading = pose.heading;
  this.speed = 0;
  this.odometer = 0;
  this.crashed = false;
};

// driverSeat() → { x, y } of the driver's eyes.
// This arcade model has NO seat offset — the seat is the car's collision
// circle centre (documented simplification; sensors fan out from here).
Car.prototype.driverSeat = function driverSeat() {
  return { x: this.x, y: this.y };
};

// draw(ctx) — simple readable rendering: rotated red body (rounded rect),
// windshield, heading tick. No gradients, no layering tricks.
Car.prototype.draw = function draw(ctx) {
  if (!ctx || typeof ctx.save !== 'function' || typeof ctx.translate !== 'function') {
    throw new ValidationError('Car.draw requires a canvas 2D-like context');
  }
  var L = 30, W = 18, R = 6; // body length/width/corner radius (px)

  ctx.save();
  ctx.translate(this.x, this.y);
  ctx.rotate(this.heading);

  // Body — manual rounded-rect path (works on every canvas implementation).
  var x = -L / 2, y = -W / 2;
  ctx.beginPath();
  ctx.moveTo(x + R, y);
  ctx.lineTo(x + L - R, y);
  ctx.arcTo(x + L, y, x + L, y + R, R);
  ctx.lineTo(x + L, y + W - R);
  ctx.arcTo(x + L, y + W, x + L - R, y + W, R);
  ctx.lineTo(x + R, y + W);
  ctx.arcTo(x, y + W, x, y + W - R, R);
  ctx.lineTo(x, y + R);
  ctx.arcTo(x, y, x + R, y, R);
  ctx.closePath();
  ctx.fillStyle = this.crashed ? '#6c757d' : '#d62828';
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#7f1d1d';
  ctx.stroke();

  // Windshield — lighter rounded slab toward the FRONT (+x local).
  ctx.beginPath();
  ctx.fillStyle = '#a9d6e5';
  ctx.fillRect(L * 0.05, -W * 0.32, L * 0.24, W * 0.64);

  // Heading tick — thin yellow line from the nose.
  ctx.beginPath();
  ctx.moveTo(L / 2, 0);
  ctx.lineTo(L / 2 + 7, 0);
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#ffd60a';
  ctx.stroke();

  ctx.restore();
};

// Browser-global exposure + guarded CommonJS export (node tests).
if (typeof globalThis !== 'undefined') {
  globalThis.Car = Car;
  globalThis.MAX_SPEED = MAX_SPEED;
  globalThis.REVERSE_MAX_SPEED = REVERSE_MAX_SPEED;
  globalThis.ENGINE_ACCEL = ENGINE_ACCEL;
  globalThis.BRAKE_ACCEL = BRAKE_ACCEL;
  globalThis.DRAG = DRAG;
  globalThis.ROLLING_RESISTANCE = ROLLING_RESISTANCE;
  globalThis.WHEEL_BASE = WHEEL_BASE;
  globalThis.CAR_RADIUS = CAR_RADIUS;
  globalThis.MAX_STEER_LOW = MAX_STEER_LOW;
  globalThis.MAX_STEER_HIGH = MAX_STEER_HIGH;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    Car: Car,
    ValidationError: ValidationError,
    MAX_SPEED: MAX_SPEED,
    REVERSE_MAX_SPEED: REVERSE_MAX_SPEED,
    ENGINE_ACCEL: ENGINE_ACCEL,
    BRAKE_ACCEL: BRAKE_ACCEL,
    DRAG: DRAG,
    ROLLING_RESISTANCE: ROLLING_RESISTANCE,
    WHEEL_BASE: WHEEL_BASE,
    CAR_RADIUS: CAR_RADIUS,
    MAX_STEER_LOW: MAX_STEER_LOW,
    MAX_STEER_HIGH: MAX_STEER_HIGH,
  };
}
