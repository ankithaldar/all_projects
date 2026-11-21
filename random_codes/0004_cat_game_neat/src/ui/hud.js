// Top-left stash panel + live coin counter and running totalSpent.
import { ITEMS } from './item-registry.js';
import { capString } from './format.js';

export function createHud({ stashEl, coinEl, spentEl }) {
  const rows = new Map();
  let prevCounts = null;
  let prevCoins = null;
  let prevSpent = null;

  for (const it of ITEMS) {
    const row = document.createElement('div');
    row.className = 'stash-row';
    row.setAttribute('data-item-id', capString(it.id));
    row.setAttribute('data-tier', String(it.tier));

    const name = document.createElement('span');
    name.className = 'stash-name';
    name.textContent = capString(it.name || it.id);
    const count = document.createElement('span');
    count.className = 'stash-count';
    count.textContent = '0';
    const ticker = document.createElement('span');
    ticker.className = 'stash-change';
    ticker.textContent = '';

    row.appendChild(name);
    row.appendChild(ticker);
    row.appendChild(count);
    stashEl.appendChild(row);
    row.setAttribute('title', capString(
      `${it.name || it.id} — tier ${it.tier} · current stash 0 units.`));
    rows.set(it.id, { row, count, name: it.name || it.id, tier: it.tier });
  }

  coinEl.setAttribute('title', capString(
    'Coin balance — spendable coins: +210 per tick, minus each accepted batch cost.'));
  spentEl.setAttribute('title', capString(
    'totalSpent — cumulative coins spent on batches (explicit accumulator, not initial - final balance).'));
  coinEl.textContent = '';
  const coinTicker = document.createElement('span');
  coinTicker.className = 'coin-ticker';
  coinTicker.textContent = '';
  coinEl.appendChild(coinTicker);

  function update(snapshot) {
    if (!snapshot) return;
    const stash = snapshot.stash || {};
    const counts = {};
    for (const id of rows.keys()) {
      counts[id] = Number.isFinite(stash[id]) ? Math.max(0, Math.floor(stash[id])) : 0;
      const refs = rows.get(id);
      refs.count.textContent = String(counts[id]);
      refs.row.setAttribute('title', capString(
        `${refs.name} — tier ${refs.tier} · current stash ${counts[id]} units. Counts update every display snapshot.`));
      const prev = prevCounts ? prevCounts[id] : null;
      refs.row.classList.remove('changed-up', 'changed-down');
      if (prev !== null && counts[id] !== prev) {
        refs.row.classList.add(counts[id] > prev ? 'changed-up' : 'changed-down');
      }
    }
    prevCounts = counts;

    const coins = Number.isFinite(snapshot.coins) ? Math.max(0, Math.floor(snapshot.coins)) : 0;
    const spent = Number.isFinite(snapshot.totalSpent) ? Math.max(0, Math.floor(snapshot.totalSpent)) : 0;
    // Keep a stable numeric node first, ticker second.
    if (!coinEl._num) {
      coinEl._num = document.createElement('span');
      coinEl._num.className = 'coin-value';
      coinEl.insertBefore(coinEl._num, coinTicker);
    }
    coinEl._num.textContent = String(coins);

    if (prevCoins !== null && prevSpent !== null) {
      const income = (coins - prevCoins) + (spent - prevSpent);
      const cost = spent - prevSpent;
      const parts = [];
      if (income > 0) parts.push(`+${income}`);
      if (cost > 0) parts.push(`-${cost}`);
      coinTicker.textContent = capString(parts.join(' / '));
    }
    prevCoins = coins;
    prevSpent = spent;
    spentEl.textContent = String(spent);
  }

  return { update };
}
