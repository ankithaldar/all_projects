"use strict";
/* DOM-less logic + contract tests for web/app.js (no browser).
 * Run from repo root: node forecast_ml_framework/web/test_app.logic.js
 * Requires ./app.js to export pure helpers under Node (see app.js bottom). */
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const app = require("./app.js");
const dir = __dirname;
const read = (f) => fs.readFileSync(path.join(dir, f), "utf8");
let passed = 0;
function ok(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`ok - ${name}`);
  } catch (err) {
    console.error(`FAIL - ${name}: ${err.message}`);
    process.exitCode = 1;
  }
}
ok("I1: nextSortDir captures same-col before assignment (switch -> asc)", () => {
  assert.strictEqual(app.nextSortDir("roc_auc", 1, "ece"), 1, "switch column must start asc");
  assert.strictEqual(app.nextSortDir("roc_auc", -1, "ece"), 1, "switch column must start asc even from desc");
  assert.strictEqual(app.nextSortDir("roc_auc", 1, "roc_auc"), -1, "same column asc toggles desc");
  assert.strictEqual(app.nextSortDir("roc_auc", -1, "roc_auc"), 1, "same column desc toggles asc");
});
ok("I6: selectBuckets falls back to by_bucket/by_decile aliases", () => {
  const aliasOnly = { by_bucket: { north: { accuracy: 0.9 } }, by_decile: { "0": { accuracy: 0.99 } } };
  assert.deepStrictEqual(app.selectBuckets(aliasOnly, "segment"), { north: { accuracy: 0.9 } });
  assert.deepStrictEqual(app.selectBuckets(aliasOnly, "decile"), { "0": { accuracy: 0.99 } });
  const both = { buckets: { segment: { east: { accuracy: 0.8 } } }, by_bucket: { north: { accuracy: 0.9 } } };
  assert.deepStrictEqual(app.selectBuckets(both, "segment"), { east: { accuracy: 0.8 } }, "buckets preferred over alias");
  assert.deepStrictEqual(app.selectBuckets({}, "segment"), {}, "missing buckets -> {}");
  assert.deepStrictEqual(app.selectBuckets(null, "segment"), {}, "null data -> {}");
});
ok("I8: shapBarWidth coerces Number and clamps [0,100]", () => {
  assert.strictEqual(app.shapBarWidth(0.5, 1), 50);
  assert.strictEqual(app.shapBarWidth("0.25", 1), 25, "numeric string coerced");
  assert.strictEqual(app.shapBarWidth("abc", 1), 0, "non-numeric -> 0");
  assert.strictEqual(app.shapBarWidth(-5, 10), 0, "negative -> 0");
  assert.strictEqual(app.shapBarWidth(10, 1), 100, "over max -> 100");
  assert.strictEqual(app.shapBarWidth(1, 0), 0, "zero max -> 0 (no NaN)");
});
ok("I10: selectKpi picks global vs regression_global by task", () => {
  const cls = { task: "classification", global: { roc_auc: 0.9 } };
  assert.deepStrictEqual(app.selectKpi(cls).source, { roc_auc: 0.9 });
  assert.deepStrictEqual(app.selectKpi(cls).keys, app.CONFIG.kpiKeys);
  const reg = { task: "regression", regression_global: { mae: 1, rmse: 2, r2: 0.8, mape: 0.05, smape: 0.04 } };
  assert.deepStrictEqual(app.selectKpi(reg).source, { mae: 1, rmse: 2, r2: 0.8, mape: 0.05, smape: 0.04 });
  assert.deepStrictEqual(app.selectKpi(reg).keys, ["mae", "rmse", "r2", "mape", "smape"]);
  assert.deepStrictEqual(app.selectKpi({ task: "regression", global: { mae: 3 } }).source, { mae: 3 }, "regression fallback to global");
  assert.deepStrictEqual(app.selectKpi({ meta: { task: "regression" }, regression_global: { mae: 1 } }).source, { mae: 1 }, "meta.task fallback");
});
ok("F-10: safeChartFile allowlist (bare or charts/ prefix, no scheme/slash)", () => {
  assert.strictEqual(app.safeChartFile("charts/roc.png"), "charts/roc.png");
  assert.strictEqual(app.safeChartFile("roc.png"), "roc.png");
  assert.strictEqual(app.safeChartFile("https://evil.example/x.png"), "");
  assert.strictEqual(app.safeChartFile("//evil.example/x.png"), "");
  assert.strictEqual(app.safeChartFile("/etc/passwd"), "");
  assert.strictEqual(app.safeChartFile("../secret.png"), "");
  assert.strictEqual(app.safeChartFile("charts/../secret.png"), "");
  assert.strictEqual(app.safeChartFile("javascript:alert(1)"), "");
  assert.strictEqual(app.safeChartFile("charts\\roc.png"), "");
  assert.strictEqual(app.safeChartFile(""), "");
  assert.strictEqual(app.safeChartFile(null), "");
});
ok("esc coverage kept (XSS regression guard)", () => {
  assert.strictEqual(app.esc('&<>"\''), "&amp;&lt;&gt;&quot;&#39;");
  assert.strictEqual(app.fmt(0.912), "0.912");
});
ok("I2/I4: app.js renders aria-sort + real button headers with focus restore", () => {
  const js = read("app.js");
  assert.ok(js.includes("aria-sort="), "aria-sort must be written on th");
  assert.ok(js.includes("ascending") && js.includes("descending") && js.includes('"none"'), "aria-sort values present");
  assert.ok(js.includes('<button type="button" data-col='), "real button with data-col");
  assert.ok(js.includes("focusCol") && js.includes(".focus()"), "focus restore to equivalent header");
  assert.ok(!js.includes('tabindex="0"') && !js.includes("tabindex=\"0\""), "th must not keep tabindex=0");
  const order = js.indexOf("safeChartFile");
  assert.ok(order !== -1 && js.indexOf("img.src") > order, "allowlist must run before img.src");
});
ok("I3: sort-dir button exposes state via aria-pressed, no stale aria-label", () => {
  const html = read("index.html");
  const js = read("app.js");
  assert.ok(!html.includes("Toggle sort direction"), "stale static aria-label removed");
  assert.ok(html.includes('id="sort-dir"') && html.includes("aria-pressed="), "aria-pressed on sort-dir in HTML");
  assert.ok(js.includes('setAttribute("aria-pressed"') || js.includes("setAttribute('aria-pressed'"), "aria-pressed updated on toggle");
});
ok("I5/I9: styles.css focus + minmax + overflow-wrap + aspect-ratio", () => {
  const css = read("styles.css");
  assert.ok(css.includes("th>button:focus-visible"), "th>button:focus-visible outline rule");
  assert.ok(css.includes("minmax(0,1fr)"), "minmax(0,1fr) in shap grid");
  assert.ok(css.includes("overflow-wrap:anywhere"), "overflow-wrap:anywhere for long names");
  assert.ok(css.includes("aspect-ratio:3/2"), "figure img aspect-ratio 3/2 (900x600 samples)");
});
ok("README documents regression KPIs + button headers", () => {
  const md = read("README.md");
  assert.ok(md.includes("regression_global"), "README mentions regression_global");
  assert.ok(md.includes("smape"), "README lists smape");
  assert.ok(/button.*header|header.*button/i.test(md), "README documents button headers");
});
console.log(`\n${passed} passed, exit code ${process.exitCode || 0}`);
