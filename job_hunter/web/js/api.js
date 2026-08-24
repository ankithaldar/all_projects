// Job Hunter shared API helpers and tiny DOM utilities.

const jh = {
  async get(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return res.json();
  },
  async send(path, method, body) {
    const res = await fetch(path, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return res.status === 204 ? null : res.json();
  },
  post(path, body) { return jh.send(path, 'POST', body ?? {}); },
  put(path, body) { return jh.send(path, 'PUT', body); },
  patch(path, body) { return jh.send(path, 'PATCH', body); },
  del(path) { return jh.send(path, 'DELETE'); },

  qs(params) {
    const clean = Object.entries(params).filter(([, v]) => v !== '' && v != null);
    return new URLSearchParams(Object.fromEntries(clean)).toString();
  },

  el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') node.className = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else if (v != null) node.setAttribute(k, v);
    }
    for (const child of children.flat()) {
      if (child == null) continue;
      node.append(child.nodeType ? child : document.createTextNode(child));
    }
    return node;
  },

  toast(message, isError = false) {
    const box = document.getElementById('jh-toast') || (() => {
      const t = jh.el('div', { id: 'jh-toast' });
      document.body.append(t);
      return t;
    })();
    box.textContent = message;
    box.className = `toast show${isError ? ' error' : ''}`;
    setTimeout(() => { box.className = 'toast'; }, 3200);
  },

  watchRun(runId, onEvent, onDone) {
    const es = new EventSource(`/api/runs/${runId}/stream`);
    es.addEventListener('log', (e) => onEvent(JSON.parse(e.data)));
    es.addEventListener('done', (e) => { onDone(JSON.parse(e.data)); es.close(); });
    es.onerror = () => { es.close(); onDone({ status: 'disconnected', stats: '{}' }); };
    return es;
  },

  scoreColor(score) {
    if (score >= 75) return 'good';
    if (score >= 55) return 'ok';
    return 'weak';
  },
};

window.jh = jh;
