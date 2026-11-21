// Renders the full static preview into any page carrying the DOM registry.
// Uses only the production UI modules + placeholder data — no sim/config.
import { createDag } from './dag.js';
import { createHud } from './hud.js';
import { createLogger } from './logger.js';
import { createControls } from './controls.js';
import { createResults } from './results.js';
import {
  placeholderSnapshot, placeholderEvents,
  placeholderResult, placeholderTelemetry,
} from './mock-data.js';

const $ = (id) => document.getElementById(id);

export function wireTabs() {
  const tabs = [['tab-train', 'panel-train'], ['tab-logger', 'panel-logger'], ['tab-results', 'panel-results']];
  for (const [tabId, panelId] of tabs) {
    const tab = $(tabId);
    if (!tab) continue;
    tab.addEventListener('click', () => {
      for (const [tId, pId] of tabs) {
        const t = $(tId), p = $(pId);
        if (!t || !p) continue;
        const active = tId === tabId;
        t.classList.toggle('is-active', active);
        t.setAttribute('aria-selected', active ? 'true' : 'false');
        p.hidden = !active;
      }
    });
  }
}

export function renderMock() {
  wireTabs();
  const snapshot = placeholderSnapshot();
  const controls = createControls({
    onStart() { controls.setStatus('PREVIEW — start requires the full simulation build'); },
    onPause() { controls.setStatus('PREVIEW — paused'); },
    onReset() { controls.setStatus('PREVIEW — reset'); },
  });
  createDag($('dag-svg')).update(snapshot);
  createHud({
    stashEl: $('stash-list'), coinEl: $('coin-counter'), spentEl: $('coin-spent'),
  }).update(snapshot);
  createLogger($('log-list')).append(placeholderEvents());
  const results = createResults($('results-body'));
  results.show(placeholderResult());
  results.showGenomeTelemetry(placeholderTelemetry());
  controls.setStatus('PREVIEW — static placeholder data');
}
