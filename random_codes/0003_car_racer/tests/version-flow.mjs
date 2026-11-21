// tests/version-flow.mjs — FIT-V4 / TASK C: version plumbing across
// main.js → trainer.js (fallback core) → train-worker.js (Worker core).
//
// Proves the canonical integer generator version reaches every one of the
// FIVE resolver call sites (db.js sites are Task B's, gated statically here),
// that all FOUR message handlers + Trainer.start refuse a missing/empty
// version instead of defaulting, that an async Worker failure replays the
// handshake WITH the version, that both cores cross a V2 rotation boundary
// using the V2 resolver hash, and that the spectate path (main-thread
// resolver through startRotationSession → currentTrainingTrackFor) carries
// the same version.
//
// Style mirrors tests/smoke.mjs: createRequire of the classic scripts, a tiny
// leg harness, assert/assertValidationError, waitFor, one line per leg, exit 1
// on any failure. Run with:  node tests/version-flow.mjs

import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const JS_DIR = fileURLToPath(new URL('../js/', import.meta.url));

// ---- load the real production modules (same order as smoke.mjs) ------------
require(path.join(JS_DIR, 'driver-sim.js'));
require(path.join(JS_DIR, 'nn.js'));
const { Track, DEFAULT_CONTROL_POINTS } = require(path.join(JS_DIR, 'track.js'));
require(path.join(JS_DIR, 'car.js'));
require(path.join(JS_DIR, 'sensors.js'));
const { Evolution } = require(path.join(JS_DIR, 'evolution.js'));
const { NeuralNetwork, mulberry32, ValidationError } = require(path.join(JS_DIR, 'nn.js'));
const { Rotation } = require(path.join(JS_DIR, 'rotation.js'));
const { Trainer, createFallbackCore } = require(path.join(JS_DIR, 'trainer.js'));

// ---- tiny harness (smoke.mjs style) ---------------------------------------
let legsPassed = 0;
const failures = [];

async function leg(name, fn) {
  try {
    await fn();
    legsPassed += 1;
    console.log('  ok   — ' + name);
  } catch (err) {
    failures.push({ name, err });
    const detail = String((err && err.stack) || err)
      .split('\n')
      .map((line) => '         ' + line)
      .join('\n');
    console.log('  FAIL — ' + name + '\n' + detail);
  }
}

function assert(condition, message) {
  if (!condition) throw new Error('assert failed: ' + message);
}

function assertValidationError(fn, message) {
  let threw = null;
  try {
    fn();
  } catch (err) {
    threw = err;
  }
  assert(threw !== null, message + ' — expected ValidationError, but nothing was thrown');
  assert(
    threw.name === 'ValidationError',
    message + ' — expected ValidationError, got ' + threw.name + ': ' + threw.message
  );
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function waitFor(condition, label, timeoutMs) {
  const t0 = Date.now();
  const limit = typeof timeoutMs === 'number' ? timeoutMs : 60000;
  while (Date.now() - t0 < limit) {
    if (condition()) return;
    await sleep(2);
  }
  throw new Error('timeout waiting for ' + label);
}

// Require the Task-A parser interface; never redefine it locally.
function requireVersionApi() {
  if (!Rotation || typeof Rotation.requireGeneratorVersion !== 'function') {
    throw new Error('Rotation.requireGeneratorVersion is missing — TASK A has not landed');
  }
  return Rotation;
}

// ---- Worker harness: run the REAL js/train-worker.js in-process ------------
// (same stubbed global surface as smoke.mjs leg 15; importScripts list is the
// fixed literal seven-script list, untouched). Kept installed for the process
// so the worker's `post()` (a live global lookup) keeps capturing messages.
const workerMsgs = [];
globalThis.self = globalThis;
globalThis.postMessage = (msg) => workerMsgs.push(msg);
globalThis.importScripts = (...urls) => {
  for (const u of urls) {
    require(path.join(JS_DIR, u.replace(/^\.\//, '')));
  }
};
require(path.join(JS_DIR, 'train-worker.js'));
function sendWorker(msg) {
  globalThis.onmessage({ data: msg });
}

// ---- small FakeWorker for Trainer handshake/replay legs -------------------
function makeFakeWorker() {
  const instances = [];
  class FakeWorker {
    constructor(url) {
      this.url = url;
      this.onmessage = null;
      this.onerror = null;
      this.terminated = false;
      this.messages = [];
      instances.push(this);
    }
    postMessage(msg) { this.messages.push(msg); }
    terminate() { this.terminated = true; this.messages.push({ type: '__terminated__' }); }
    emit(type, extra) { if (this.onmessage) this.onmessage({ data: Object.assign({ type }, extra || {}) }); }
    error(message) { if (this.onerror) this.onerror({ message }); }
  }
  return { FakeWorker, instances };
}

// ---- static resolver-call-site scan ---------------------------------------
function extractCallArgs(src, from) {
  let depth = 1;
  let i = from;
  let quote = null;
  const args = [];
  let start = i;
  while (i < src.length && depth > 0) {
    const ch = src[i];
    if (quote) {
      if (ch === '\\') { i += 2; continue; }
      if (ch === quote) quote = null;
      i += 1;
      continue;
    }
    if (ch === "'" || ch === '"' || ch === '`') { quote = ch; i += 1; continue; }
    if (ch === '(' || ch === '[' || ch === '{') depth += 1;
    else if (ch === ')' || ch === ']' || ch === '}') {
      depth -= 1;
      if (depth === 0) { args.push(src.slice(start, i).trim()); return args; }
    } else if (ch === ',' && depth === 1) {
      args.push(src.slice(start, i).trim());
      start = i + 1;
    }
    i += 1;
  }
  args.push(src.slice(start).trim());
  return args;
}

function scanResolverSites() {
  const sites = [];
  const CALL_RE = /(?:Rotation|R)\.resolveGeneratedTrack\s*\(/g;
  for (const file of fs.readdirSync(JS_DIR).filter((f) => f.endsWith('.js')).sort()) {
    const src = fs.readFileSync(path.join(JS_DIR, file), 'utf8');
    src.split(/\r?\n/).forEach((line, idx) => {
      const trimmed = line.trim();
      if (trimmed.startsWith('//') || trimmed.startsWith('*') || trimmed.startsWith('/*')) return;
      CALL_RE.lastIndex = 0;
      let m;
      while ((m = CALL_RE.exec(line)) !== null) {
        const openParen = m.index + m[0].length - 1;
        sites.push({ file, line: idx + 1, args: extractCallArgs(line, openParen + 1) });
      }
    });
  }
  return sites;
}

// =============================================================================
// V1 — all five resolver call sites carry an explicit integer version
// =============================================================================
// The plan pins exactly FIVE resolver call sites: db.js:651 pre-materialization
// + db.js:2334 boundary re-derivation (Task B), and main.js / trainer.js /
// train-worker.js (Task C). Every one must pass an explicit canonical version.
const EXPECTED_TOTAL_SITES = 5;
const EXPECTED_BY_FILE = { 'db.js': 2, 'main.js': 1, 'trainer.js': 1, 'train-worker.js': 1 };

await leg('V1 — exactly five resolveGeneratedTrack call sites, each with an explicit version argument (static gate)', () => {
  const sites = scanResolverSites();
  assert(sites.length === EXPECTED_TOTAL_SITES, 'expected ' + EXPECTED_TOTAL_SITES + ' resolver call sites in js/, got ' + sites.length + ': ' + sites.map((s) => s.file + ':' + s.line).join(', '));

  const byFile = {};
  for (const s of sites) byFile[s.file] = (byFile[s.file] || 0) + 1;
  for (const file of Object.keys(EXPECTED_BY_FILE)) {
    assert(byFile[file] === EXPECTED_BY_FILE[file], file + ' must hold exactly ' + EXPECTED_BY_FILE[file] + ' resolver site(s), got ' + (byFile[file] || 0));
  }

  for (const s of sites) {
    assert(s.args.length >= 3, s.file + ':' + s.line + ' — resolver call must pass an explicit 3rd (version) argument, got args ' + JSON.stringify(s.args));
    assert(/version/i.test(s.args[2]), s.file + ':' + s.line + ' — 3rd argument must be the canonical version value, got ' + JSON.stringify(s.args[2]));
  }
});

// The literal plan grep-gate (definition is a bare function declaration, not
// matched by the (Rotation|R). resolver pattern; comments are skipped above).
await leg('V1b — literal grep gate: grep -rnE "(Rotation|R)\\.resolveGeneratedTrack\\(" js/ = 5 lines, each with a version', () => {
  const out = execFileSync('grep', ['-rnE', '(Rotation|R)\\.resolveGeneratedTrack\\(', JS_DIR], { encoding: 'utf8' });
  const lines = out.split('\n').filter((l) => l.trim().length > 0);
  assert(lines.length === EXPECTED_TOTAL_SITES, 'grep gate: expected ' + EXPECTED_TOTAL_SITES + ' resolver call lines, got ' + lines.length + '\n' + out);
  for (const l of lines) {
    assert(/version/i.test(l.slice(l.indexOf('resolveGeneratedTrack('))), 'grep gate: line lacks a version argument: ' + l);
  }
});

// =============================================================================
// V2 — strict validation: missing/empty version never defaults
// =============================================================================
await leg('V2 — Worker init/resume handlers refuse missing/empty generatorVersion', async () => {
  requireVersionApi();
  const base = {
    type: 'init',
    controlPoints: JSON.parse(JSON.stringify(DEFAULT_CONTROL_POINTS)),
    halfWidth: 70,
    seed: 4242,
    rotationSeed: 4242,
  };
  const weights = new NeuralNetwork(mulberry32(1)).getWeights();
  const rngState = new Evolution(4242).rngState();
  const bad = [undefined, null, ''];
  for (const v of bad) {
    workerMsgs.length = 0;
    sendWorker(Object.assign({}, base, v === undefined ? {} : { generatorVersion: v }));
    assert(workerMsgs.some((m) => m.type === 'error' && /version/i.test(m.message)), 'Worker init refused generatorVersion=' + JSON.stringify(v) + ' with an error mentioning version, got ' + JSON.stringify(workerMsgs));
    workerMsgs.length = 0;
    sendWorker({
      type: 'resume',
      controlPoints: base.controlPoints,
      halfWidth: base.halfWidth,
      bestWeights: weights,
      rngState,
      generation: 1,
      rotationSeed: base.rotationSeed,
    });
    assert(workerMsgs.some((m) => m.type === 'error' && /version/i.test(m.message)), 'Worker resume refused missing generatorVersion with an error mentioning version, got ' + JSON.stringify(workerMsgs));
  }
});

await leg('V2b — MessageChannel fallback handlers refuse missing/empty generatorVersion', () => {
  requireVersionApi();
  const core = createFallbackCore();
  const msgs = [];
  const send = (m) => core(m, (out) => msgs.push(out));
  const base = {
    type: 'init',
    controlPoints: JSON.parse(JSON.stringify(DEFAULT_CONTROL_POINTS)),
    halfWidth: 70,
    seed: 4242,
    rotationSeed: 4242,
  };
  const weights = new NeuralNetwork(mulberry32(1)).getWeights();
  const rngState = new Evolution(4242).rngState();
  for (const v of [undefined, null, '']) {
    msgs.length = 0;
    send(Object.assign({}, base, v === undefined ? {} : { generatorVersion: v }));
    assert(msgs.some((m) => m.type === 'error' && /version/i.test(m.message)), 'fallback init refused generatorVersion=' + JSON.stringify(v) + ', got ' + JSON.stringify(msgs));
    msgs.length = 0;
    send({
      type: 'resume',
      controlPoints: base.controlPoints,
      halfWidth: base.halfWidth,
      bestWeights: weights,
      rngState,
      generation: 1,
      rotationSeed: base.rotationSeed,
    });
    assert(msgs.some((m) => m.type === 'error' && /version/i.test(m.message)), 'fallback resume refused missing generatorVersion, got ' + JSON.stringify(msgs));
  }
});

await leg('V2c — Trainer.start refuses a lineage with missing/empty generatorVersion', () => {
  requireVersionApi();
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);
  for (const v of [undefined, null, '']) {
    const lineage = { seed: 4242, rotationSeed: 4242 };
    if (v !== undefined) lineage.generatorVersion = v;
    const tr = new Trainer(track, { lineage });
    let threw = null;
    try { tr.start(1); } catch (err) { threw = err; }
    assert(threw !== null, 'Trainer.start refused generatorVersion=' + JSON.stringify(v) + ' (never defaulted)');
    assert(threw.name === 'ValidationError', 'Trainer.start threw ValidationError for generatorVersion=' + JSON.stringify(v) + ', got ' + threw.name);
    assert(/version/i.test(threw.message), 'Trainer.start error mentions the version field, got: ' + threw.message);
  }
});

// =============================================================================
// V3 — Worker core crosses a V2 rotation boundary (gen 50 → 51)
// =============================================================================
const DRIVE_SEED = 1234;
async function driveCrossBoundary(send, msgs) {
  const weights = new NeuralNetwork(mulberry32(1)).getWeights();
  const rngState = new Evolution(4242).rngState();
  msgs.length = 0;
  send({
    type: 'resume',
    controlPoints: JSON.parse(JSON.stringify(DEFAULT_CONTROL_POINTS)),
    halfWidth: 70,
    bestWeights: weights,
    rngState,
    generation: 50,
    rotationSeed: DRIVE_SEED,
    generatorVersion: 2,
  });
  send({ type: 'start', maxGenerations: 51 });
  await waitFor(() => msgs.some((m) => m.type === 'generation' && m.payload.generation === 51), 'generation 51 payload across the V2 boundary', 60000);
  return msgs.find((m) => m.type === 'generation' && m.payload.generation === 51).payload;
}

let workerV2Payload = null;
await leg('V3 — Worker core: generation 51 rides the V2 resolver track, not V1', async () => {
  const Rotation = requireVersionApi();
  workerV2Payload = await driveCrossBoundary(sendWorker, workerMsgs);
  const v2 = Rotation.resolveGeneratedTrack(DRIVE_SEED, 1, 2).trackHash;
  const v1 = Rotation.resolveGeneratedTrack(DRIVE_SEED, 1, 1).trackHash;
  assert(workerV2Payload.rotationIndex === 1, 'Worker gen 51 is rotation 1, got ' + workerV2Payload.rotationIndex);
  assert(workerV2Payload.trackHash === v2, 'Worker gen 51 trackHash === resolveGeneratedTrack(seed,1,2).trackHash, got ' + workerV2Payload.trackHash + ' vs ' + v2);
  assert(v2 !== v1, 'V2 and V1 resolver hashes differ at r=1 (V2 generator landed)');
  assert(workerV2Payload.trackHash !== v1, 'Worker gen 51 trackHash is NOT the V1 hash ' + v1);
});

// =============================================================================
// V4 — fallback core crosses the SAME V2 boundary, byte-identical to Worker
// =============================================================================
await leg('V4 — fallback core: generation 51 rides the V2 resolver track, byte-identical to Worker', async () => {
  const Rotation = requireVersionApi();
  const core = createFallbackCore();
  const msgs = [];
  const payload = await driveCrossBoundary((m) => core(m, (out) => msgs.push(out)), msgs);
  const v2 = Rotation.resolveGeneratedTrack(DRIVE_SEED, 1, 2).trackHash;
  assert(payload.rotationIndex === 1, 'fallback gen 51 is rotation 1');
  assert(payload.trackHash === v2, 'fallback gen 51 trackHash === V2 resolver hash, got ' + payload.trackHash + ' vs ' + v2);
  assert(JSON.stringify(payload) === JSON.stringify(workerV2Payload), 'Worker and fallback gen 51 payloads are byte-identical across the V2 boundary');
});

// =============================================================================
// V5 — replay handshake carries the version after an async Worker failure
// =============================================================================
await leg('V5 — async pre-first-commit Worker failure replays the handshake WITH generatorVersion', async () => {
  requireVersionApi();
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);
  const { FakeWorker, instances } = makeFakeWorker();
  const outbound = [];
  const tr = new Trainer(track, {
    lineage: { seed: 4242, rotationSeed: 4242, generatorVersion: 2 },
    workerFactory: (url) => new FakeWorker(url),
    onBanner() {},
  });
  const origPost = tr._postToBackend.bind(tr);
  tr._postToBackend = (msg) => { outbound.push(msg); return origPost(msg); };
  tr.start(1);
  assert(tr._lastStart && tr._lastStart.handshake.generatorVersion === 2, 'Trainer._lastStart.handshake carries generatorVersion 2');
  assert(tr._lastStart && tr._lastStart.shared.generatorVersion === 2, 'Trainer._lastStart.shared carries generatorVersion 2');
  assert(outbound[0].type === 'init' && outbound[0].generatorVersion === 2, 'initial init handshake carries generatorVersion 2');
  instances[0].error('boom: simulated pre-first-commit worker failure');
  await waitFor(() => outbound.filter((m) => m.type === 'init').length === 2, 'replayed init handshake', 10000);
  assert(outbound[2].type === 'init', 'the replayed handshake is the 3rd outbound message');
  assert(outbound[2].generatorVersion === 2, 'replayed init handshake carries generatorVersion 2');
  assert(JSON.stringify(outbound[0]) === JSON.stringify(outbound[2]), 'replayed init handshake is byte-identical to the original');
  tr.stop();
});

// =============================================================================
// V6 — spectate path: main-thread resolver carries state.generatorVersion === 2
// =============================================================================
function loadMainContext() {
  const sandbox = {
    console, setTimeout, clearTimeout, setInterval, clearInterval,
    Track, Rotation, DEFAULT_CONTROL_POINTS, NeuralNetwork, mulberry32, Evolution,
    DriverSim: class StubDriverSim { constructor(track) { this.track = track; } },
    document: {
      getElementById() { return { textContent: '', hidden: false, value: '', checked: false, style: {} }; },
      querySelectorAll() { return []; },
      createElement() { return { style: {}, setAttribute() {}, appendChild() {} }; },
      body: { appendChild() {} },
    },
    requestAnimationFrame() {},
    crypto: globalThis.crypto,
    // Db/DbSession intentionally absent ⇒ boot contract fails ⇒ boot() skipped;
    // renderBootError uses only the stubbed document above.
  };
  const context = vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(JS_DIR, 'main.js'), 'utf8'), context, { filename: 'main.js' });
  vm.runInContext(fs.readFileSync(path.join(JS_DIR, 'spectate.js'), 'utf8'), context, { filename: 'spectate.js' });
  return sandbox;
}

await leg('V6 — spectate: state.generatorVersion===2 and spectate track hash === V2 resolver hash at r>=1', () => {
  const Rotation = requireVersionApi();
  const main = loadMainContext();
  const pin = new Track(DEFAULT_CONTROL_POINTS, 70);
  main.startRotationSession({ rotationSeed: DRIVE_SEED, track: pin, generatorVersion: 2 });
  assert(main.state.generatorVersion === 2, 'startRotationSession set state.generatorVersion === 2, got ' + JSON.stringify(main.state.generatorVersion));

  const sims = main.spawnSpectateGeneration({ generation: 51, population: [{}] });
  assert(Array.isArray(sims) && sims.length === 1, 'spawnSpectateGeneration produced one sim');
  const spectateHash = main.state.spectateTrack.hash();
  const v2 = Rotation.resolveGeneratedTrack(DRIVE_SEED, 1, 2).trackHash;
  assert(spectateHash === v2, 'spectate track hash === resolveGeneratedTrack(seed,1,2).trackHash, got ' + spectateHash + ' vs ' + v2);

  const sameBlock = main.currentTrainingTrackFor(52);
  assert(sameBlock.hash() === v2, 'the same rotation block reuses the V2 track at generation 52');
});

// =============================================================================
console.log('');
if (failures.length === 0) {
  console.log('VERSION-FLOW OK — ' + legsPassed + ' legs passed');
} else {
  console.log('VERSION-FLOW FAIL — ' + failures.length + ' leg(s) failed, ' + legsPassed + ' passed');
  for (const f of failures) console.log('  - ' + f.name);
}
// A trailing MessageChannel port from the replay leg can keep node's event
// loop alive; flush stdout/stderr and exit with the leg-driven status.
process.exit(failures.length === 0 ? 0 : 1);
