async function loadStats() {
  try {
    const response = await fetch('/api/stats');
    if (!response.ok) throw new Error(String(response.status));
    const stats = await response.json();
    renderSummary(stats);
    renderStatuses(stats.statuses || []);
    renderProviders(stats.per_provider || []);
    renderSessions(stats.per_session || []);
    renderRecent(stats.recent_calls || []);
  } catch (err) {
    const el = document.getElementById('summary-cards');
    el.replaceChildren();
    const cardEl = document.createElement('div');
    cardEl.className = 'card';
    cardEl.textContent = 'Error loading stats: ' + (err && err.message ? err.message : err);
    el.append(cardEl);
  }
}

function makeCard(label, value) {
  const cardEl = document.createElement('div');
  cardEl.className = 'card';
  const labelEl = document.createElement('div');
  labelEl.textContent = label;
  const valueEl = document.createElement('div');
  valueEl.className = 'value';
  valueEl.textContent = String(value);
  cardEl.append(labelEl, valueEl);
  return cardEl;
}

function renderSummary(stats) {
  const summary = stats.summary || {};
  const el = document.getElementById('summary-cards');
  el.replaceChildren(
    makeCard('Total Calls', summary.total_calls || 0),
    makeCard('Avg Latency (ms)', Number(summary.avg_latency_ms || 0).toFixed(2)),
    makeCard('Input Tokens', summary.input_tokens || 0),
    makeCard('Output Tokens', summary.output_tokens || 0),
    makeCard('Total Cost', Number(summary.total_cost || 0).toFixed(6)),
    makeCard('Error Rate', (Number(stats.error_rate || 0) * 100).toFixed(2) + '%')
  );
}

function renderTable(el, columns, rows) {
  el.replaceChildren();
  if (!rows || !rows.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.textContent = 'No data';
    tr.append(td);
    el.append(tr);
    return;
  }
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  for (const column of columns) {
    const th = document.createElement('th');
    th.setAttribute('scope', 'col');
    th.textContent = column.label;
    headRow.append(th);
  }
  thead.append(headRow);
  const tbody = document.createElement('tbody');
  for (const row of rows) {
    const tr = document.createElement('tr');
    for (const column of columns) {
      const td = document.createElement('td');
      const value = row[column.key];
      td.textContent = value === null || value === undefined ? '' : String(value);
      tr.append(td);
    }
    tbody.append(tr);
  }
  el.append(thead, tbody);
}

function renderStatuses(rows) {
  renderTable(document.getElementById('statuses'),
    [{key: 'status', label: 'Status'}, {key: 'count', label: 'Count'}], rows);
}
function renderProviders(rows) {
  renderTable(document.getElementById('providers'),
    [{key: 'provider', label: 'Provider'}, {key: 'calls', label: 'Calls'},
     {key: 'avg_latency_ms', label: 'Avg Latency ms'},
     {key: 'input_tokens', label: 'Input Tokens'},
     {key: 'output_tokens', label: 'Output Tokens'},
     {key: 'total_cost', label: 'Cost'}], rows);
}
function renderSessions(rows) {
  renderTable(document.getElementById('sessions'),
    [{key: 'session_id', label: 'Session'}, {key: 'calls', label: 'Calls'},
     {key: 'avg_latency_ms', label: 'Avg Latency ms'},
     {key: 'input_tokens', label: 'Input Tokens'},
     {key: 'output_tokens', label: 'Output Tokens'},
     {key: 'total_cost', label: 'Cost'}], rows);
}
function renderRecent(rows) {
  renderTable(document.getElementById('recent'),
    [{key: 'timestamp', label: 'Timestamp'}, {key: 'provider', label: 'Provider'},
     {key: 'model_used', label: 'Model'}, {key: 'status', label: 'Status'},
     {key: 'latency_ms', label: 'Latency ms'},
     {key: 'input_tokens', label: 'Input Tokens'},
     {key: 'output_tokens', label: 'Output Tokens'},
     {key: 'cost', label: 'Cost'}], rows);
}

loadStats();
setInterval(loadStats, 10000);
