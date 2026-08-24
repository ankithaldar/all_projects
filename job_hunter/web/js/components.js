// Job Hunter shared renderers.

const components = {
  nav(active) {
    const links = [
      ['/', 'Dashboard'], ['/recommendations.html', 'Recommendations'],
      ['/companies.html', 'Companies'], ['/jobs.html', 'Jobs'],
      ['/profile.html', 'Profile'], ['/runs.html', 'Runs'],
      ['/settings.html', 'Settings'],
    ];
    return jh.el('nav', { class: 'nav' },
      jh.el('a', { class: 'brand', href: '/' }, '🎯 Job Hunter'),
      ...links.map(([href, label]) =>
        jh.el('a', { href, class: active === href ? 'active' : '' }, label)));
  },

  kpi(label, value) {
    return jh.el('div', { class: 'kpi' },
      jh.el('div', { class: 'kpi-value' }, String(value)),
      jh.el('div', { class: 'kpi-label' }, label));
  },

  scoreBadge(score) {
    const s = Number(score ?? 0);
    return jh.el('span', { class: `score ${jh.scoreColor(s)}` }, s.toFixed(1));
  },

  breakdown(bars = {}) {
    const wrap = jh.el('div', { class: 'bars' });
    for (const [name, value] of Object.entries(bars)) {
      const pct = Math.round(Number(value) * 100);
      wrap.append(jh.el('div', { class: 'bar-row' },
        jh.el('span', { class: 'bar-label' }, name),
        jh.el('div', { class: 'bar-track' },
          jh.el('div', { class: `bar-fill ${jh.scoreColor(pct)}`, style: `width:${pct}%` })),
        jh.el('span', { class: 'bar-val' }, `${pct}%`)));
    }
    return wrap;
  },

  recCard(rec, onAction) {
    const gateFails = Array.isArray(rec.gate_failures) ? rec.gate_failures : [];
    const card = jh.el('div', { class: `card rec${rec.status !== 'new' ? ' reviewed' : ''}` },
      jh.el('div', { class: 'card-head' },
        this.scoreBadge(rec.total_score),
        jh.el('a', { class: 'title', href: rec.url, target: '_blank', rel: 'noopener' }, rec.title || '(untitled)'),
        jh.el('span', { class: 'company' }, rec.company_name || ''),
        jh.el('span', { class: 'badge' }, rec.vertical || 'unknown')),
      jh.el('div', { class: 'meta' },
        [rec.city, rec.work_mode,
          rec.salary_min_lpa && rec.salary_max_lpa
            ? `${rec.salary_min_lpa}–${rec.salary_max_lpa} LPA` : null,
          rec.posted_at ? new Date(rec.posted_at).toLocaleDateString() : null]
          .filter(Boolean).join(' · ')),
      rec.rationale ? jh.el('p', { class: 'rationale' }, rec.rationale) : null,
      this.breakdown(rec.score_breakdown || {}),
      gateFails.length ? jh.el('div', { class: 'gates' }, 'Gated: ' + gateFails.join(', ')) : null,
      jh.el('div', { class: 'actions' },
        jh.el('button', { class: 'btn save', onclick: () => onAction('saved', rec) },
          rec.status === 'saved' ? '★ Saved' : '☆ Save'),
        jh.el('button', { class: 'btn dismiss', onclick: () => onAction('dismissed', rec) }, '✕ Dismiss')));
    return card;
  },

  table(headers, rows, rowRenderer) {
    return jh.el('table', {},
      jh.el('thead', {}, jh.el('tr', {}, headers.map(h => jh.el('th', {}, h)))),
      jh.el('tbody', {}, rows.map(r => {
        const cells = rowRenderer(r);
        return jh.el('tr', {}, cells.map(c => jh.el('td', {}, c)));
      })));
  },
};

window.components = components;
