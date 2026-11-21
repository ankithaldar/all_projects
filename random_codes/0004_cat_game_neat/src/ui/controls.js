// Validated simulation controls. Every value is checked by validate.js before
// it reaches a callback or getConfig(); invalid input falls back to defaults.
// This is the ONLY site that wires the start/pause/reset buttons.
import { validateControl } from '../validate.js';
import { capString } from './format.js';
import { getActiveLogger } from './logger.js';

const DEFAULTS = { speed: '1x', seed: 1, preset: 'reuse', popSize: 20, generations: 100 };

function byId(id) {
  return document.getElementById(id);
}
function valueOf(id) {
  const el = byId(id);
  return el ? String(el.value == null ? '' : el.value).trim() : '';
}
function validated(name, value, fallback) {
  try {
    return validateControl(name, value);
  } catch (_err) {
    return fallback;
  }
}

export function createControls(callbacks = {}) {
  // Status node created with the DOM API and appended next to Start.
  const status = document.createElement('p');
  status.className = 'controls-status';
  status.setAttribute('role', 'status');
  const anchor = byId('btn-start');
  const host = (anchor && anchor.parentNode) || document.body;
  host.appendChild(status);

  function read() {
    const speed = validated('speed', valueOf('speed'), DEFAULTS.speed);
    const preset = validated('preset', valueOf('preset'), DEFAULTS.preset);
    const seedRaw = parseInt(valueOf('seed'), 10);
    const seed = validated('seed', Number.isNaN(seedRaw) ? null : seedRaw, DEFAULTS.seed);
    const popRaw = parseInt(valueOf('pop-size'), 10);
    const popSize = validated('pop-size', Number.isNaN(popRaw) ? null : popRaw, DEFAULTS.popSize);
    const genRaw = parseInt(valueOf('generations'), 10);
    const generations = validated('generations', Number.isNaN(genRaw) ? null : genRaw,
      DEFAULTS.generations);
    return { speed, seed, preset, popSize, generations };
  }

  function invoke(name) {
    const fn = callbacks[name];
    if (typeof fn === 'function') fn(read());
  }
  function bind(id, name) {
    const btn = byId(id);
    if (btn && btn.addEventListener) btn.addEventListener('click', () => invoke(name));
  }
  bind('btn-start', 'onStart');
  bind('btn-pause', 'onPause');
  bind('btn-reset', 'onReset');

  // Exactly ONE #preset change listener: a preset change re-prefills the
  // Initial state panel via main's onPresetChange callback.
  const presetEl = byId('preset');
  if (presetEl && presetEl.addEventListener) {
    presetEl.addEventListener('change', () => invoke('onPresetChange'));
  }

  // Logger filter/clear wiring — single owner, no duplicate listeners elsewhere.
  const logger = () => callbacks.logger || getActiveLogger();
  const filterEl = byId('log-filter');
  if (filterEl && filterEl.addEventListener) {
    filterEl.addEventListener('input', () => {
      const log = logger();
      if (log && typeof log.setFilter === 'function') {
        log.setFilter(String(filterEl.value == null ? '' : filterEl.value).trim() || null);
      }
    });
  }
  const clearEl = byId('log-clear');
  if (clearEl && clearEl.addEventListener) {
    clearEl.addEventListener('click', () => {
      const log = logger();
      if (log && typeof log.clear === 'function') log.clear();
      if (filterEl) filterEl.value = '';
    });
  }

  return {
    getConfig: read,
    setStatus(s) {
      status.textContent = capString(s == null ? '' : s);
    },
  };
}
