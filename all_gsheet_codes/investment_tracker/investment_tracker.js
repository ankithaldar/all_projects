/** @OnlyCurrentDoc */

const CONFIG = Object.freeze({
  IBJA_URL: 'https://ibjarates.com/',
  US_MARKET: Object.freeze({
    URL: 'https://goldprice.org/gold-price-usa.html',
    OUNCE_TO_GRAM: 31.1034768, // TROY OUNCE
    GOLD_XPATH: '/html/body/main/div[2]/div/div/div[2]/div/article/div/div[1]/div[1]/div[2]/div/div/div/div/div/div[1]/div[1]/span',
    SILVER_XPATH: '/html/body/main/div[2]/div/div/div[2]/div/article/div/div[1]/div[2]/div/div/div/div/div/div/div[1]/div[1]/span',
  }),
  EMAIL_TO: 'ankithaldar@gmail.com',
  IBJA: Object.freeze({
    GOLD_ROW: 1,
    GOLD_AM_COL: 1,
    GOLD_PM_COL: 2,
    SILVER_ROW: 6,
    SILVER_AM_COL: 1,
    SILVER_PM_COL: 2,
    DIVISOR_GOLD: 10, // req #6: per 10g -> per gram
    DIVISOR_SILVER: 1000, // req #7: per 1000g/kg -> per gram
  }),
  SHEETS: Object.freeze({
    INVESTMENT: 'Investment Tracker',
    GOLD_LOG: 'gold_price_log',
    SILVER_LOG: 'silver_price_log',
    LOG_SHEET: 'Execution_Logs',
  }),
  RANGES: Object.freeze({
    GOLD_PRICE_CELL: 'M3',
    MF_URL_COL: 18, // R
    MF_NAV_COL: 21, // U
    MF_START_ROW: 5,
  }),
  CACHE_TTL: 21600, // 6 hrs
});

let LOGGER = null;
let _TZ = null;
const getTz_ = () => _TZ || (_TZ = Session.getScriptTimeZone());
const NAV_RE = /\b\d{2,}\.\d{4}\b/;

class ExecutionLogger_ {
  constructor() {
    this.entries = [];
    this.startTime = new Date();
    this.hasErrors = false;
  }
  _add(l, m, d) {
    this.entries.push({
      ts: new Date(),
      level: l,
      msg: m,
      data: d ? JSON.stringify(d).slice(0, 2000) : ''
    });
    if (l === 'ERROR') this.hasErrors = true;
    console.log(`[${l}] ${m}`);
  }
  info(m, d) {
    this._add('INFO', m, d);
  }
  warn(m, d) {
    this._add('WARN', m, d);
  }
  error(m, d) {
    this._add('ERROR', m, d);
  }
  debug(m, d) {
    this._add('DEBUG', m, d);
  }
  durationMs() {
    return new Date() - this.startTime;
  }
  toPlainText() {
    return `Start: ${this.startTime}\nDuration: ${this.durationMs()}ms\n\n` + this.entries.map(e =>
      `[${e.level}] ${e.ts.toISOString()} ${e.msg} ${e.data}`).join('\n');
  }
  toHtml() {
    const rows = this.entries.map(e =>
      `<tr><td>${e.ts.toLocaleTimeString()}</td><td>${e.level}</td><td>${e.msg}</td><td style="font-family:monospace;font-size:11px;word-break:break-all;">${e.data}</td></tr>`
      ).join('');
    return `<h2>Tracker Log - ${this.hasErrors? 'FAILED' : 'SUCCESS'}</h2><p>Duration: ${this.durationMs()}ms</p><table border=1 cellpadding=6><tr><th>Time</th><th>Level</th><th>Message</th><th>Data</th></tr>${rows}</table>`;
  }
}

function onOpen() {
  SpreadsheetApp.getUi().createMenu('Investments').addItem('Update All Now', 'updateAllInvestments').addToUi();
}

function updateAllInvestments() {
  LOGGER = new ExecutionLogger_();
  LOGGER.info('=== START updateAllInvestments ===');
  _TZ = getTz_();
  const lock = LockService.getDocumentLock();
  if (!lock.tryLock(10000)) {
    LOGGER.warn('Could not obtain lock - another run in progress');
    sendLogEmail_(LOGGER);
    return;
  }
  try {
    const ss = SpreadsheetApp.getActiveSpreadsheet();

    // 1. IBJA Gold + Silver
    const metals = getIBJAMetalsPerGram_();
    LOGGER.info('Metals per-gram ready', metals);

    // 2. US Market Gold + Silver
    const usMetals = getUSMetalsPerGram_();
    LOGGER.info('US metals per-gram ready', usMetals);

    // 3. Gold log + Investment Tracker
    if (metals.goldPerGram) {
      const invSheet = ss.getSheetByName(CONFIG.SHEETS.INVESTMENT);
      if (invSheet) {
        invSheet.getRange(CONFIG.RANGES.GOLD_PRICE_CELL).setValue(metals.goldPerGram);
        LOGGER.info('Gold updated in Investment Tracker', {
          cell: CONFIG.RANGES.GOLD_PRICE_CELL,
          perGram: metals.goldPerGram,
          raw: metals.goldRaw,
          source: metals.goldSource
        });
      }
      const goldLogSheet = ss.getSheetByName(CONFIG.SHEETS.GOLD_LOG);
      updatePriceLogBatchIndependent_(goldLogSheet, metals.goldPerGram, 'GOLD', usMetals.goldPerGram);
    } else {
      LOGGER.warn('Gold per gram is null - skipping log');
    }

    // 4. Silver log only (not in Investment Tracker)
    if (metals.silverPerGram) {
      const silverLogSheet = ss.getSheetByName(CONFIG.SHEETS.SILVER_LOG);
      updatePriceLogBatchIndependent_(silverLogSheet, metals.silverPerGram, 'SILVER', usMetals.silverPerGram);
      LOGGER.info('Silver updated ONLY in silver_price_log', {
        perGram: metals.silverPerGram,
        raw: metals.silverRaw,
        source: metals.silverSource
      });
    } else {
      LOGGER.warn('Silver per gram is null - skipping log');
    }

    // 5. MF NAVs
    updateAllMFNavsBatch_(ss);
    LOGGER.info('=== SUCCESS ===');

  } catch (e) {
    LOGGER.error('FATAL', {
      msg: e.message,
      stack: e.stack
    });
  } finally {
    lock.releaseLock();
    writeLogToSheet_(LOGGER);
    sendLogEmail_(LOGGER);
  }
}

// -------------------- IBJA --------------------
function getIBJAMetalsPerGram_() {
  const cache = CacheService.getScriptCache();
  // 1. Try small JSON cache first
  try {
    const cachedJson = cache ? cache.get('ibja_metals_json') : null;
    if (cachedJson) {
      const parsed = JSON.parse(cachedJson);
      LOGGER.info('IBJA metals FROM CACHE', parsed);
      return parsed;
    }
  } catch (e) {
    LOGGER.warn('IBJA cache read failed: ' + e.message);
  }

  // 2. Live fetch (no large cache put)
  const resp = UrlFetchApp.fetch(CONFIG.IBJA_URL, {
    muteHttpExceptions: true,
    headers: {
      'User-Agent': 'Mozilla/5.0'
    }
  });
  const code = resp.getResponseCode();
  const html = resp.getContentText();
  LOGGER.info('IBJA fetched LIVE', {
    code,
    len: html.length
  });

  if (code !== 200) throw new Error('IBJA HTTP ' + code);

  const raw = extractByXPathIndexes_(html);
  const goldChosen = choosePMorAM_(raw.goldPM, raw.goldAM);
  const silverChosen = choosePMorAM_(raw.silverPM, raw.silverAM);

  LOGGER.info('PM>AM chosen', {
    goldPM: raw.goldPM,
    goldAM: raw.goldAM,
    chosenGold: goldChosen.value,
    silverPM: raw.silverPM,
    silverAM: raw.silverAM,
    chosenSilver: silverChosen.value
  });

  const result = {
    goldRaw: goldChosen.value,
    goldSource: goldChosen.source,
    silverRaw: silverChosen.value,
    silverSource: silverChosen.source,
    goldPerGram: goldChosen.value ? Number(goldChosen.value) / CONFIG.IBJA.DIVISOR_GOLD : null,
    silverPerGram: silverChosen.value ? Number(silverChosen.value) / CONFIG.IBJA.DIVISOR_SILVER : null,
  };

  // 3. Cache small JSON (<1KB) not full HTML
  try {
    if (cache) cache.put('ibja_metals_json', JSON.stringify(result), CONFIG.CACHE_TTL);
  } catch (e) {
    LOGGER.warn('IBJA cache write failed: ' + e.message);
  }
  return result;
}

function extractByXPathIndexes_(html) {
  const tableMatch = html.match(/<table[^>]*>[\s\S]*?999[\s\S]*?<\/table>/i);
  const tableHtml = tableMatch ? tableMatch[0] : html;
  const rows = [];
  const re = /<tr[^>]*>([\s\S]*?)<\/tr>/gi;
  let m;
  while ((m = re.exec(tableHtml)) !== null) rows.push(m[1]);

  let goldRow = rows[CONFIG.IBJA.GOLD_ROW] || '';
  let silverRow = rows[CONFIG.IBJA.SILVER_ROW] || '';

  if (!/999/.test(goldRow)) goldRow = rows.find(r => /Gold.*999/i.test(r)) || '';
  if (!/Silver/i.test(silverRow)) silverRow = rows.find(r => /Silver/i.test(r)) || '';

  const goldCells = parseTdValues_(goldRow);
  const silverCells = parseTdValues_(silverRow);

  return {
    goldAM: goldCells[CONFIG.IBJA.GOLD_AM_COL] || '',
    goldPM: goldCells[CONFIG.IBJA.GOLD_PM_COL] || '',
    silverAM: silverCells[CONFIG.IBJA.SILVER_AM_COL] || '',
    silverPM: silverCells[CONFIG.IBJA.SILVER_PM_COL] || '',
  };
}

function parseTdValues_(rowHtml) {
  if (!rowHtml) return [];
  const out = [];
  const tdRe = /<td[^>]*>([\s\S]*?)<\/td>/gi;
  let m;
  while ((m = tdRe.exec(rowHtml)) !== null) {
    let td = m[1];
    const span = td.match(/<span[^>]*>([\s\S]*?)<\/span>/i);
    if (span) td = span[1];
    out.push(td.replace(/<[^>]+>/g, '').replace(/[^0-9.,]/g, '').replace(/,/g, '').trim());
  }
  return out;
}

function choosePMorAM_(pm, am) {
  if (pm && pm !== '' && pm !== '0') return {
    value: pm,
    source: 'PM'
  };
  return {
    value: am,
    source: 'AM'
  };
}

// -------------------- US MARKET --------------------
function getUSMetalsPerGram_() {
  const cache = CacheService.getScriptCache();
  try {
    const cachedJson = cache ? cache.get('us_metals_json') : null;
    if (cachedJson) {
      const parsed = JSON.parse(cachedJson);
      LOGGER.info('US metals FROM CACHE', parsed);
      return parsed;
    }
  } catch (e) {}

  let goldRawOunce = null,
    silverRawOunce = null;
  let goldPerGram = null,
    silverPerGram = null;

  // Source 1: gold-api.com - reliable, free, no Cloudflare
  try {
    const opts = {
      muteHttpExceptions: true,
      headers: {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0'
      }
    };
    const gResp = UrlFetchApp.fetch('https://api.gold-api.com/price/XAU', opts);
    const sResp = UrlFetchApp.fetch('https://api.gold-api.com/price/XAG', opts);

    if (gResp.getResponseCode() === 200 && sResp.getResponseCode() === 200) {
      const gJson = JSON.parse(gResp.getContentText());
      const sJson = JSON.parse(sResp.getContentText());
      goldRawOunce = (gJson.price || gJson.Price || '').toString();
      silverRawOunce = (sJson.price || sJson.Price || '').toString();
      if (goldRawOunce) goldPerGram = Number(goldRawOunce) / CONFIG.US_MARKET.OUNCE_TO_GRAM;
      if (silverRawOunce) silverPerGram = Number(silverRawOunce) / CONFIG.US_MARKET.OUNCE_TO_GRAM;
      LOGGER.info('US metals from gold-api.com', {
        goldRawOunce,
        silverRawOunce,
        goldPerGram,
        silverPerGram
      });
    } else {
      LOGGER.warn(`gold-api.com HTTP ${gResp.getResponseCode()}/${sResp.getResponseCode()}`);
    }
  } catch (e) {
    LOGGER.warn('gold-api.com failed: ' + e.message);
  }

  // Source 2: Fallback to original HTML with real browser headers (only if API failed)
  if (!goldPerGram && !silverPerGram) {
    try {
      const resp = UrlFetchApp.fetch(CONFIG.US_MARKET.URL, {
        muteHttpExceptions: true,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
          'Accept-Language': 'en-US,en;q=0.9',
          'Referer': 'https://www.google.com/',
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        }
      });
      const code = resp.getResponseCode();
      if (code === 200) {
        const raw = extractUSPrices_(resp.getContentText());
        if (raw.gold) {
          goldRawOunce = raw.gold;
          goldPerGram = Number(raw.gold) / CONFIG.US_MARKET.OUNCE_TO_GRAM;
        }
        if (raw.silver) {
          silverRawOunce = raw.silver;
          silverPerGram = Number(raw.silver) / CONFIG.US_MARKET.OUNCE_TO_GRAM;
        }
        LOGGER.info('US metals from HTML fallback', raw);
      } else {
        LOGGER.warn(`US HTML fallback HTTP ${code} - will continue with IBJA only`);
      }
    } catch (e) {
      LOGGER.warn('US HTML fallback failed: ' + e.message);
    }
  }

  const result = {
    goldRawOunce,
    silverRawOunce,
    goldPerGram,
    silverPerGram
  };

  if (!goldPerGram && !silverPerGram) {
    LOGGER.warn('US market unavailable - IBJA will still be logged', result);
  }

  try {
    if (cache) cache.put('us_metals_json', JSON.stringify(result), CONFIG.CACHE_TTL);
  } catch (e) {
    LOGGER.warn('US cache write failed: ' + e.message);
  }

  return result;
}

function extractUSPrices_(html) {
  let gold = '',
    silver = '';
  const clean = v => v ? v.replace(/,/g, '').replace(/[^0-9.]/g, '').trim() : '';

  const g1 = html.match(/Gold[^$]{0,800}?\$?\s*([\d,]+\.\d{1,4})/i);
  const s1 = html.match(/Silver[^$]{0,800}?\$?\s*([\d,]+\.\d{1,4})/i);
  if (g1) gold = g1[1];
  if (s1) silver = s1[1];

  if (!gold || !silver) {
    const article = (html.match(/<article[\s\S]*?>([\s\S]*?)<\/article>/i) || [])[1] || html;
    const spans = [...article.matchAll(/<span[^>]*>\s*\$?\s*([\d,]+\.\d+)\s*<\/span>/gi)].map(m => m[1]);
    const nums = spans.map(r => ({
      raw: r,
      val: parseFloat(r.replace(/,/g, ''))
    })).filter(o => !isNaN(o.val));
    if (!gold) {
      const cand = nums.find(o => o.val > 1000 && o.val < 10000);
      if (cand) gold = cand.raw;
    }
    if (!silver) {
      const cand = nums.find(o => o.val >= 10 && o.val < 500);
      if (cand) silver = cand.raw;
      else if (nums.length >= 2) silver = nums[1].raw;
    }
  }

  if (!gold || !silver) {
    const all = [...html.matchAll(/\$\s*([\d,]+\.\d{2})/g)].map(m => m[1]);
    if (all[0] && !gold) gold = all[0];
    if (all[1] && !silver) silver = all[1];
  }

  return {
    gold: clean(gold),
    silver: clean(silver)
  };
}

function extractUSPrices_(html) {
  let gold = '',
    silver = '';
  const clean = v => v ? v.replace(/,/g, '').replace(/[^0-9.]/g, '').trim() : '';

  const g1 = html.match(/Gold[^$]{0,800}?\$?\s*([\d,]+\.\d{1,4})/i);
  const s1 = html.match(/Silver[^$]{0,800}?\$?\s*([\d,]+\.\d{1,4})/i);
  if (g1) gold = g1[1];
  if (s1) silver = s1[1];

  if (!gold || !silver) {
    const article = (html.match(/<article[\s\S]*?>([\s\S]*?)<\/article>/i) || [])[1] || html;
    const spans = [...article.matchAll(/<span[^>]*>\s*\$?\s*([\d,]+\.\d+)\s*<\/span>/gi)].map(m => m[1]);
    const nums = spans.map(r => ({
      raw: r,
      val: parseFloat(r.replace(/,/g, ''))
    })).filter(o => !isNaN(o.val));
    if (!gold) {
      const cand = nums.find(o => o.val > 1000 && o.val < 10000);
      if (cand) gold = cand.raw;
    }
    if (!silver) {
      const cand = nums.find(o => o.val >= 10 && o.val < 500 && String(o.val) !== gold);
      if (cand) silver = cand.raw;
      else if (nums.length >= 2) silver = nums[1].raw;
    }
  }

  if (!gold || !silver) {
    const all = [...html.matchAll(/\$\s*([\d,]+\.\d{2})/g)].map(m => m[1]);
    if (all[0] && !gold) gold = all[0];
    if (all[1] && !silver) silver = all[1];
  }

  return {
    gold: clean(gold),
    silver: clean(silver)
  };
}

// -------------------- LOG UPDATER --------------------
function updatePriceLogBatchIndependent_(sheet, perGramPrice, metalLabel, usPerGramPrice) {
  if (!sheet) {
    LOGGER.error(`Sheet not found for ${metalLabel}`);
    return;
  }

  const lRow = sheet.getLastRow();
  LOGGER.debug(`Checking last row for ${metalLabel}`, {
    sheet: sheet.getName(),
    lRow
  });
  const usVal = (usPerGramPrice != null && !isNaN(usPerGramPrice)) ? usPerGramPrice : '';

  if (lRow === 0) {
    sheet.getRange(1, 1, 1, 4).setValues([
      ['Date', 'Day', 'Price Per Gram', metalLabel === 'GOLD' ? 'US Gold Price/g' : 'US Silver Price/g']
    ]);
    sheet.getRange(2, 1, 1, 4).setValues([
      [new Date(), Utilities.formatDate(new Date(), getTz_(), 'EEE'), perGramPrice, usVal]
    ]);
    LOGGER.info(`${metalLabel} log initialized`, {
      sheet: sheet.getName()
    });
    createOrModifyChart_(sheet, 2);
    return;
  }

  if (!sheet.getRange(1, 4).getValue()) {
    sheet.getRange(1, 4).setValue(metalLabel === 'GOLD' ? 'US Gold Price/g' : 'US Silver Price/g');
  }

  const lastDateVal = sheet.getRange(lRow, 1).getValue();
  const lastDateStr = lastDateVal ? new Date(lastDateVal).toDateString() : '';
  const todayStr = new Date().toDateString();

  if (lastDateStr === todayStr) {
    if (usVal === '') sheet.getRange(lRow, 3).setValue(perGramPrice);
    else sheet.getRange(lRow, 3, 1, 2).setValues([
      [perGramPrice, usVal]
    ]);
    LOGGER.info(`${metalLabel} log same day overwrite`, {
      sheet: sheet.getName(),
      row: lRow,
      perGram: perGramPrice,
      usPerGram: usVal
    });
    createOrModifyChart_(sheet, lRow);
  } else {
    sheet.insertRowAfter(lRow);
    const dayName = Utilities.formatDate(new Date(), getTz_(), 'EEE');
    sheet.getRange(lRow + 1, 1, 1, 4).setValues([
      [new Date(), dayName, perGramPrice, usVal]
    ]);
    LOGGER.info(`${metalLabel} log new row inserted`, {
      sheet: sheet.getName(),
      row: lRow + 1,
      perGram: perGramPrice,
      usPerGram: usVal
    });
    createOrModifyChart_(sheet, lRow + 1);
  }
}

function createOrModifyChart_(sheet, lRow) {
  const rowCount = lRow || sheet.getLastRow();
  if (rowCount < 2) return;
  const old = sheet.getCharts()[0];
  if (old) sheet.removeChart(old);
  const chart = sheet.newChart()
    .setChartType(Charts.ChartType.LINE)
    .addRange(sheet.getRange('A1:A' + rowCount))
    .addRange(sheet.getRange('C1:C' + rowCount))
    .setPosition(1, 6, 101, 0)
    .setOption('title', sheet.getName() + ' (per gram)')
    .setOption('height', 900).setOption('width', 1600).build();
  sheet.insertChart(chart);
}

function updateAllMFNavsBatch_(ss) {
  const sheet = ss.getSheetByName(CONFIG.SHEETS.INVESTMENT);
  if (!sheet) {
    LOGGER.warn('Investment sheet not found for MF');
    return;
  }
  const numRows = sheet.getLastRow() - CONFIG.RANGES.MF_START_ROW + 1;
  if (numRows <= 0) return;
  const rich = sheet.getRange(CONFIG.RANGES.MF_START_ROW, CONFIG.RANGES.MF_URL_COL, numRows, 1).getRichTextValues();
  const reqs = [];
  const idxMap = [];
  for (let i = 0; i < numRows; i++) {
    let u = '';
    try {
      u = rich[i][0]?.getLinkUrl() || '';
    } catch (_) {}
    if (u) {
      reqs.push({
        url: u,
        muteHttpExceptions: true
      });
      idxMap.push(i);
    }
  }
  if (!reqs.length) {
    LOGGER.info('No MF URLs found');
    return;
  }
  const resps = UrlFetchApp.fetchAll(reqs);
  const out = new Array(numRows).fill(0).map(() => ['']);
  for (let r = 0; r < resps.length; r++) {
    const txt = resps[r].getContentText().substring(0, 3000);
    const m = txt.match(NAV_RE);
    out[idxMap[r]][0] = m ? m[0] : '';
  }
  sheet.getRange(CONFIG.RANGES.MF_START_ROW, CONFIG.RANGES.MF_NAV_COL, numRows, 1).setValues(out);
  LOGGER.info('MF NAVs updated', {
    count: out.length,
    fetched: reqs.length
  });
}

function sendLogEmail_(logger) {
  try {
    MailApp.sendEmail({
      to: CONFIG.EMAIL_TO,
      subject: `[Tracker] ${logger.hasErrors? 'FAILED' : 'SUCCESS'} ${new Date().toLocaleString()}`,
      body: logger.toPlainText(),
      htmlBody: logger.toHtml()
    });
    LOGGER.info('Log email sent', {
      to: CONFIG.EMAIL_TO
    });
  } catch (e) {
    console.error('Email failed: ' + e.message);
  }
}

function writeLogToSheet_(logger) {
  try {
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    let sh = ss.getSheetByName(CONFIG.SHEETS.LOG_SHEET);
    if (!sh) sh = ss.insertSheet(CONFIG.SHEETS.LOG_SHEET);
    sh.appendRow([new Date(), logger.hasErrors ? 'FAILED' : 'SUCCESS', logger.durationMs(), logger.toPlainText().slice(
      0, 40000)]);
  } catch (e) {
    console.error('Log sheet write failed: ' + e.message);
  }
}
