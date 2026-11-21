// Bounded event logger: 5,000-entry ring buffer + capped DOM rows.
// Rows are built with createElement + textContent only — never innerHTML.
import { LOG_CAP, capString } from './format.js';

const DOM_CAP = 600;

// The most recently created logger. controls.js uses this to wire #log-filter
// and #log-clear without needing an extra argument threaded through main.js.
let activeLogger = null;
export function getActiveLogger() { return activeLogger; }

export function createLogger(listEl) {
  let buffer = [];   // [{event, el, show}]
  let filter = null;

  function matches(event) {
    if (!filter) return true;
    if (typeof filter === 'function') return !!filter(event);
    if (typeof filter === 'string') {
      const q = filter.toLowerCase();
      return String(event.type || '').toLowerCase().includes(q)
        || String(event.item || '').toLowerCase().includes(q);
    }
    if (typeof filter === 'object') {
      if (filter.type && event.type !== filter.type) return false;
      if (filter.item && event.item !== filter.item) return false;
      return true;
    }
    return true;
  }

  function makeRow(event) {
    const row = document.createElement('div');
    row.className = 'log-row';
    row.setAttribute('data-type', capString(event.type || ''));
    row.setAttribute('data-item', capString(event.item || ''));
    row.setAttribute('title', capString(
      `tick ${capString(event.tick)} | type ${capString(event.type)} | item ${capString(event.item)} | ` +
      `batch ${capString(event.batch)} | delta ${capString(event.delta)} | reason ${capString(event.reason)}`));
    const cols = [
      ['tick', event.tick], ['type', event.type], ['item', event.item],
      ['batch', event.batch], ['delta', event.delta], ['reason', event.reason],
    ];
    for (const [key, value] of cols) {
      const span = document.createElement('span');
      span.className = `log-col log-${key}`;
      span.textContent = capString(value == null ? '' : value);
      row.appendChild(span);
    }
    return row;
  }

  function autoScroll() {
    listEl.scrollTop = listEl.scrollHeight;
  }

  function rebuild() {
    listEl.textContent = '';
    const visible = buffer.filter((b) => b.show).slice(-DOM_CAP);
    for (const b of visible) listEl.appendChild(b.el);
    autoScroll();
  }

  function append(events) {
    const arr = (Array.isArray(events) ? events : [events])
      .filter((e) => e && typeof e === 'object');
    for (const ev of arr) {
      const entry = { event: ev, el: makeRow(ev), show: matches(ev) };
      buffer.push(entry);
      if (entry.show) listEl.appendChild(entry.el);
    }
    while (buffer.length > LOG_CAP) {
      const dropped = buffer.shift();
      if (dropped.el.parentNode) dropped.el.parentNode.removeChild(dropped.el);
    }
    while (listEl.children.length > DOM_CAP) {
      listEl.removeChild(listEl.firstChild);
    }
    autoScroll();
  }

  function clear() {
    buffer = [];
    listEl.textContent = '';
  }

  function setFilter(f) {
    filter = typeof f === 'function' || typeof f === 'string' || (f && typeof f === 'object')
      ? f : null;
    for (const b of buffer) b.show = matches(b.event);
    rebuild();
  }

  const api = { append, clear, setFilter };
  activeLogger = api;
  return api;
}
