// Job Hunter shared API helpers and tiny DOM utilities.

// Token is stored only in sessionStorage (never localStorage) so it
// does not persist across browser sessions. Loopback-only deployment
// keeps the bearer token on the same origin.
function storedToken() {
  try { return sessionStorage.getItem('jh_token') || ''; }
  catch { return ''; }
}

function authHeaders(extra = {}) {
  const token = storedToken();
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}

async function fetchWithAuth(path, options = {}, retried = false) {
  const opts = { ...options, headers: authHeaders(options.headers || {}) };
  let res = await fetch(path, opts);
  if (res.status === 401 && !retried) {
    const entered = prompt('Enter JH_API_TOKEN');
    if (entered) {
      try { sessionStorage.setItem('jh_token', entered); } catch {}
      res = await fetchWithAuth(path, options, true);
    }
  }
  return res;
}

const jh = {
  async get(path) {
    const res = await fetchWithAuth(path);
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return res.json();
  },
  async send(path, method, body) {
    const res = await fetchWithAuth(path, {
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

  authHeaders,

  qs(params) {
    const clean = Object.entries(params).filter(([, v]) => v !== '' && v != null);
    return new URLSearchParams(Object.fromEntries(clean)).toString();
  },

  el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') node.className = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else if (k === 'href') {
        const value = String(v);
        // Reject control chars (incl. whitespace) anywhere in the
        // href before the allowlist below, so smuggled characters
        // cannot bypass the scheme/path checks.
        if (!/[\u0000-\u0020]/.test(value)) {
          if (/^https?:\/\//i.test(value)) {
            node.setAttribute('href', value);
            node.setAttribute('rel', 'noopener noreferrer');
          } else if (/^\/$|^\/(?![\/\\])/.test(value) || /^#/.test(value) || /^\./.test(value)) {
            node.setAttribute('href', value);
          }
        }
        // else omit: javascript:, data:, vbscript:, //evil.com, etc.
      } else if (k === 'rel') {
        node.setAttribute('rel', String(v).includes('noopener') ? 'noopener noreferrer' : v);
      } else if (v != null) node.setAttribute(k, v);
    }
    for (const child of children.flat()) {
      if (child == null) continue;
      node.append(child.nodeType ? child : document.createTextNode(child));
    }
    return node;
  },

  toast(message, isError = false) {
    const box = document.getElementById('jh-toast') || (() => {
      const t = jh.el('div', { id: 'jh-toast', role: 'status', 'aria-live': 'polite' });
      document.body.append(t);
      return t;
    })();
    box.textContent = message;
    box.className = `toast show${isError ? ' error' : ''}`;
    setTimeout(() => { box.className = 'toast'; }, 3200);
  },

  // Fetch-stream SSE: the native event stream cannot send
  // Authorization headers, so the stream is fetched with the
  // bearer token instead of putting the token in the URL.
  // watchRun(runId, onEvent, onDone) returns a handle with close().
  watchRun(runId, onEvent, onDone) {
    const controller = new AbortController();
    const es = { close() { controller.abort(); } };
    const parse = (data) => {
      try { return JSON.parse(data); } catch { return null; }
    };
    (async () => {
      try {
        const res = await fetchWithAuth(`/api/runs/${runId}/stream`, {
          signal: controller.signal,
        });
        if (!res.ok || !res.body) {
          es.close();
          onDone({ status: 'disconnected', stats: '{}' });
          return;
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        const dispatch = (raw) => {
          let eventName = 'message';
          const dataLines = [];
          for (const line of raw.split('\n')) {
            if (line.startsWith('event:')) eventName = line.slice(6).trim();
            else if (line.startsWith('data:')) {
              const payload = line.slice(5);
              dataLines.push(payload.startsWith(' ') ? payload.slice(1) : payload);
            }
          }
          const data = dataLines.join('\n');
          if (!data && eventName === 'message') return true;
          if (eventName === 'log') {
            const v = parse(data);
            if (v) onEvent(v);
          } else if (eventName === 'done') {
            const v = parse(data) || { status: 'done' };
            es.close();
            onDone(v);
            return false;
          } else if (eventName === 'timeout') {
            es.close();
            onDone({ status: 'timeout' });
            return false;
          }
          return true;
        };
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            onDone({ status: 'disconnected', stats: '{}' });
            return;
          }
          buffer += decoder.decode(value, { stream: true });
          let boundary = buffer.indexOf('\n\n');
          while (boundary !== -1) {
            const frame = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);
            if (!dispatch(frame)) return;
            boundary = buffer.indexOf('\n\n');
          }
        }
      } catch (err) {
        if (err && err.name === 'AbortError') return;
        try { es.close(); } catch {}
        onDone({ status: 'disconnected', stats: '{}' });
      }
    })();
    return es;
  },

  scoreColor(score) {
    if (score >= 75) return 'good';
    if (score >= 55) return 'ok';
    return 'weak';
  },
};

window.jh = jh;
