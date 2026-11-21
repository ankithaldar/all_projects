// js/spectate.js — Spectate: main-thread live eval of the same lineage spec
// for the 0003_car_racer browser shell. Plain classic script; exposes the
// spectate globals js/main.js wires up (classic-script global functions).
//
// WHAT THIS FILE OWNS
//   * spawnSpectateGeneration(evo) — one DriverSim per population net (the
//     SAME sim the headless eval path and replay run), sharing the active
//     track; the golden-angle hue assignment (SPECTATE_HUE_STEP / spectateHue).
//   * startSpectate / stopSpectate — the Spectate toggle (armed state, toggle
//     text), the resolveLineage()-gated starts.
//   * The per-tick DriverSim stepping happens in js/main.js's rAF loop
//     (frame()); finalizeSpectate() drains the finished cars into results,
//     records the evaluation (Evolution + elitism) and logs through the SAME
//     db.logGeneration() transaction with the shared buildLogGenerationPayload
//     builder. nextSpectateGeneration() spawns generation N+1 from the
//     elitism-mutated best driver (the "Next generation" button).
//   * drawSpectateFrame() — the 20-car renderer (golden-angle hues, index
//     tags, done-gray with reduced alpha).
//
// Load position (LOAD-BEARING): AFTER js/main-db.js and BEFORE js/db-session.js,
// js/editor.js and js/main.js. `state` lives in js/main.js; everything else
// this file calls (buildLogGenerationPayload, pushMiniLog, onPersistRequested,
// …) is a global from js/main.js, resolved at call time.

'use strict';

/* ============================================================
   11. Spectate — main-thread live eval of the same lineage spec
   ============================================================ */

var SPECTATE_HUE_STEP = 137.508; // golden angle — 20 maximally distinct hues

function spectateHue(index) {
  return (index * SPECTATE_HUE_STEP) % 360;
}

// spawnSpectateGeneration(evo) — one DriverSim per population member (the
// SAME sim the headless eval path and replay run), evaluated on the ROTATION
// track for the generation (currentTrainingTrackFor — the rotation block's
// Track, cached per block, byte-identical to the worker's), shared by all 20
// cars. The block's track is remembered on state.spectateTrack so the render
// path draws EXACTLY what the cars run on (never the ambient editor/Manual
// track; the pin for rotation 0).
function spawnSpectateGeneration(evo) {
  var track = currentTrainingTrackFor(evo.generation);
  state.spectateTrack = track;
  var sims = [];
  for (var i = 0; i < evo.population.length; i++) {
    sims.push({ sim: new globalThis.DriverSim(track, evo.population[i]), index: i });
  }
  return sims;
}

function startSpectate() {
  var lin = resolveLineage();
  if (!lin) return;
  startRotationSession(lin); // FIT-V3 — pin the session to the lineage's seed + pin track; reset the block cache
  var evo;
  try {
    evo = lin.kind === 'fresh'
      ? new globalThis.Evolution(lin.seed)
      : globalThis.Evolution.resumeFromBest(lin.bestWeights, lin.generation, lin.rngState);
    if (lin.kind === 'fresh') evo.randomPopulation();
  } catch (err) {
    showBanner('Spectate refused — ' + errMsg(err));
    return;
  }
  var cars = spawnSpectateGeneration(evo);
  state.spectate = {
    evo: evo,
    cars: cars,
    running: true,
    waitingForNext: false,
    generation: evo.generation,
  };
  state.spectateAcc = 0;
  el('spectate-summary').hidden = true;
  el('btn-next-generation').hidden = true;
  el('stat-generation').textContent = String(evo.generation);
  var rotationElStart = el('stat-rotation');
  if (rotationElStart) rotationElStart.textContent = rotationLabelFor(evo.generation);
  setPersistPill('idle', 'unknown');
  updateTrainUi();
  drawTrainStatus('Spectating generation ' + evo.generation + ' — 20 drivers live', state.spectateTrack);
}

function stopSpectate() {
  if (state.spectate) {
    state.spectate.running = false;
    // The toggle click means "I'm done with this spectate" — disarm an armed
    // (waiting-for-next) generation too, so destructive paths unblock.
    state.spectate.waitingForNext = false;
  }
  updateTrainUi(); // re-renders the toggle text, Next-generation button, armed state
}

function finalizeSpectate() {
  var spec = state.spectate;
  spec.running = false;
  spec.waitingForNext = true;
  var results = [];
  var sum = 0, worst = Infinity, maxAlive = 0;
  for (var i = 0; i < spec.cars.length; i++) {
    var s = spec.cars[i];
    var distance = s.sim.fitness(); // bonus-inclusive fitness (net signed + lap bonus; may be negative)
    var topSpeed = s.sim.topSpeed();
    var timeAlive = s.sim.timeAlive;
    results.push({
      distance: distance,
      topSpeed: topSpeed,
      timeAlive: timeAlive,
      crashed: s.sim.crashed,
      stuck: s.sim.stuck,
    });
    sum += distance;
    if (distance < worst) worst = distance;
    if (timeAlive > maxAlive) maxAlive = timeAlive;
  }
  var bestIndex;
  try {
    bestIndex = spec.evo.recordEvaluation(results);
  } catch (err) {
    spec.waitingForNext = false; // the catch path must not leave a dead Next button armed
    showBanner('Spectate logging failed — ' + errMsg(err));
    updateTrainUi();
    return;
  }
  var best = results[bestIndex];
  var avg = sum / results.length;

  el('stat-generation').textContent = String(spec.generation);
  var rotationElFinish = el('stat-rotation');
  if (rotationElFinish) rotationElFinish.textContent = rotationLabelFor(spec.generation);
  el('stat-best').textContent = formatPx(best.distance);
  el('stat-avg').textContent = formatPx(avg);
  el('stat-worst').textContent = formatPx(worst);
  el('stat-sim-seconds').textContent = formatSeconds(maxAlive);
  pushMiniLog('gen ' + spec.generation + ' · best ' + formatPx(best.distance) + ' · avg ' + formatPx(avg) + ' · worst ' + formatPx(worst) + ' · best #' + bestIndex);

  if (!state.sessionLoggingDisabled && state.dbAvailable && state.db) {
    try {
      var payload = buildLogGenerationPayload({
        generation: spec.generation,
        results: results,
        bestIndex: bestIndex,
        rngState: spec.evo.rngState(),
        bestWeights: spec.evo.population[bestIndex].getWeights(),
      });
      state.db.logGeneration(payload);
      onPersistRequested(payload); // same persist semantics as headless
      var replayMax = el('replay-generation');
      replayMax.max = String(Math.max(Number(replayMax.max) || 1, spec.generation));
      refreshReplayBounds(); // generation committed — unlock replay
      updateResumeInputBounds();
    } catch (err) {
      if (isLineageIntegrity(err)) {
        state.haltedLineage = true;
        showBanner('LINEAGE INTEGRITY ERROR — RUN HALTED — committed lineage untouched');
        updateTrainUi(); // reflect the halt — Start/Spectate/Resume disabled until reload
        return;
      }
      state.sessionLoggingDisabled = true;
      state.lastFailureGeneration = spec.generation;
      setLoggingIndicator(true);
      setPersistPill('skipped (logging disabled)', 'warn');
      showBanner('GENERATION ' + spec.generation + ' NOT LOGGED — this tab\'s committed prefix is intact through generation ' + (spec.generation - 1) + ' — LOGGING DISABLED THIS SESSION — TRAINING CONTINUES WITHOUT LOGGING — resume later to recompute generation ' + spec.generation + ' identically (the lineage guard requires the same track)');
    }
  }

  el('sum-line').textContent = 'gen ' + spec.generation + ' · best ' + formatPx(best.distance) + ' (#' + bestIndex + ') · avg ' + formatPx(avg) + ' · worst ' + formatPx(worst);
  el('spectate-summary').hidden = false;
  el('btn-next-generation').hidden = false;
  updateTrainUi();
}

function nextSpectateGeneration() {
  var spec = state.spectate;
  if (!spec || !spec.waitingForNext) return;
  try {
    spec.evo.spawnNextGeneration();
  } catch (err) {
    showBanner('Could not spawn generation ' + (spec.generation + 1) + ' — ' + errMsg(err));
    return;
  }
  spec.generation = spec.evo.generation;
  spec.cars = spawnSpectateGeneration(spec.evo);
  spec.waitingForNext = false;
  spec.running = true;
  el('spectate-summary').hidden = true;
  el('btn-next-generation').hidden = true;
  el('stat-generation').textContent = String(spec.generation);
  var rotationElNext = el('stat-rotation');
  if (rotationElNext) rotationElNext.textContent = rotationLabelFor(spec.generation);
  setPersistPill('idle', 'unknown');
  updateTrainUi();
  drawTrainStatus('Spectating generation ' + spec.generation + ' — 20 drivers live', state.spectateTrack);
}

function drawSpectateFrame() {
  var canvas = el('train-canvas');
  var ctx = canvasContext(canvas);
  var spectateBlockTrack = state.spectateTrack || state.activeTrack;
  if (!ctx || !spectateBlockTrack || !state.spectate) return;
  var spec = state.spectate;
  drawGrass(ctx);
  spectateBlockTrack.draw(ctx);
  for (var i = 0; i < spec.cars.length; i++) {
    var s = spec.cars[i];
    var car = s.sim.car;
    var hue = spectateHue(i);
    var done = s.sim.outcome !== null;
    ctx.globalAlpha = done ? 0.55 : 1;
    ctx.fillStyle = done ? '#6c757d' : 'hsl(' + hue.toFixed(1) + ', 75%, 58%)';
    drawCarGlyph(ctx, car.x, car.y, car.heading);
    ctx.globalAlpha = 1;
    // Driver index tag
    ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
    ctx.fillRect(car.x - 9, car.y + 12, 18, 13);
    ctx.fillStyle = '#fff';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(String(i), car.x, car.y + 22);
  }
  ctx.save();
  ctx.fillStyle = 'rgba(10, 12, 14, 0.55)';
  ctx.fillRect(0, 0, CANVAS_W, 34);
  ctx.fillStyle = '#ffd60a';
  ctx.font = '600 14px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('spectating generation ' + spec.generation + ' · ' + spec.cars.filter(function (s) { return s.sim.outcome === null; }).length + ' cars running', 12, 22);
  ctx.restore();
}

