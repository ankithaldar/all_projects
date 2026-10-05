/**
 * =============================================================================
 * errors.js  -  Water billing: error handling and preflight checks.
 * =============================================================================
 *
 * Five small pieces of cross-cutting machinery, used by 01_make_unpivot.js:
 *
 *   check_apps_script_api_()  Preflight. Asserts every Apps Script method this
 *                             project calls actually exists, BEFORE any work is
 *                             done. Two of the worst bugs in this pipeline were
 *                             "TypeError: copy.setPosition is not a function" and
 *                             "anchor.getPosition is not a function" - both from
 *                             methods that do not exist. This turns that class of
 *                             failure into one clear up-front message naming what
 *                             is missing, instead of a crash halfway through a
 *                             run that has already modified the workbook.
 *
 *   require_sheets_(names)    Fail fast when a required sheet is missing, and
 *                             list all of them at once rather than the first.
 *
 *   run_stage_(label, fn)     Runs one pipeline stage, logging start / ok / fail
 *                             with elapsed milliseconds, and attaches the stack
 *                             on failure. Re-throws so a single top-level handler
 *                             alerts once instead of every stage alerting.
 *
 *   handle_errors_(label, fn) Top-level handler: turns any uncaught error into a
 *                             readable alert plus a full trace in the execution
 *                             log, instead of the raw Apps Script error dialog.
 *
 *   retry_service_(label, fn) Retries transient Spreadsheets service failures
 *                             ("timed out", rate limits, backend errors) with a
 *                             linear backoff. Errors that are not transient are
 *                             re-thrown immediately, so real bugs are not
 *                             hidden behind retries.
 *
 * DEPENDENCIES
 *   Uses log_() and fail_() from 01_make_unpivot.js - all files in a bound Apps
 *   Script project share one global scope, so nothing needs importing. Keeping
 *   the logger here rather than duplicating it means the execution log has one
 *   format for every stage.
 * =============================================================================
 */


/**
 * Apps Script methods this project depends on, by object. Verified against the
 * Google Apps Script reference; keep in sync with every call site.
 */
const REQUIRED_API = Object.freeze({
  SpreadsheetApp: ['getActiveSpreadsheet', 'getUi'],
  Session: ['getScriptTimeZone'],
  Utilities: ['formatDate', 'sleep'],
});

/** Spreadsheet methods used: sheet lookup, tab order, active sheet. */
const REQUIRED_SPREADSHEET_API = Object.freeze([
  'getSheetByName', 'getSheets', 'getActiveSheet', 'setActiveSheet', 'moveActiveSheet',
]);

/** Sheet methods used: ranges, row surgery, copying, visibility, identity. */
const REQUIRED_SHEET_API = Object.freeze([
  'getRange', 'getMaxRows', 'getMaxColumns', 'getFilter', 'insertRowAfter', 'deleteRows',
  'setName', 'copyTo', 'isSheetHidden', 'hideSheet', 'getSheetId', 'getParent',
  'getName',
]);

/** Range methods used: reading, writing, formulas, clearing, filtering, fill. */
const REQUIRED_RANGE_API = Object.freeze([
  'getValue', 'getValues', 'getFormulas', 'setValues', 'setFormulas', 'clearContent',
  'createFilter', 'autoFill', 'getA1Notation', 'getNumColumns',
]);

/** Filter methods used: removing the previous run's filter. */
const REQUIRED_FILTER_API = Object.freeze(['remove', 'getRange']);

/**
 * Message fragments that mark a Spreadsheets failure as worth retrying.
 * Everything else is treated as a real bug and re-thrown at once.
 */
const TRANSIENT_ERROR_PATTERNS = Object.freeze([
  /timed out/i,
  /rate limit/i,
  /quota/i,
  /backend error/i,
  /internal error/i,
  /try again/i,
  /resource exhausted/i,
]);


/**
 * Preflight: assert every Apps Script method and enum this project uses exists.
 *
 * Cheap - a handful of typeof checks - and it runs before the pipeline touches
 * the workbook, so a missing method is reported once instead of aborting a run
 * that has already created sheets and cleared rows.
 *
 * @return {boolean} true when everything required is present.
 */
function check_apps_script_api_() {
  const missing = [];

  for (const [obj_name, methods] of Object.entries(REQUIRED_API)) {
    const obj = globalThis[obj_name];
    if (!obj) {
      missing.push(`${obj_name} (object not available)`);
      continue;
    }
    for (const method of methods) {
      if (typeof obj[method] !== 'function') missing.push(`${obj_name}.${method}`);
    }
  }

  // Enum used by the formula fill: SpreadsheetApp.AutoFillSeries.DEFAULT_SERIES.
  const series = globalThis.SpreadsheetApp && globalThis.SpreadsheetApp.AutoFillSeries;
  if (!series || series.DEFAULT_SERIES === undefined) {
    missing.push('SpreadsheetApp.AutoFillSeries.DEFAULT_SERIES');
  }

  // The remaining objects only exist once a spreadsheet is open, so they are
  // probed through the live workbook rather than the global scope.
  try {
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const sheet = ss.getSheets()[0];
    const range = sheet.getRange('A1:A1');   // read only, nothing is written

    for (const m of REQUIRED_SPREADSHEET_API) {
      if (typeof ss[m] !== 'function') missing.push(`Spreadsheet.${m}`);
    }
    for (const m of REQUIRED_SHEET_API) {
      if (typeof sheet[m] !== 'function') missing.push(`Sheet.${m}`);
    }
    for (const m of REQUIRED_RANGE_API) {
      if (typeof range[m] !== 'function') missing.push(`Range.${m}`);
    }

    // Filter is probed only when one already exists. Creating one to test it
    // would leave a filter on the user's sheet, and createFilter() throws when
    // the range overlaps an existing filter - so the probe could itself fail on
    // a perfectly healthy workbook.
    const existing_filter = sheet.getFilter();
    if (existing_filter) {
      for (const m of REQUIRED_FILTER_API) {
        if (typeof existing_filter[m] !== 'function') missing.push(`Filter.${m}`);
      }
    } else {
      log_("INFO", `check_apps_script_api_: no filter on '${sheet.getName()}' to ` +
                   `probe, so the Filter checks were skipped`);
    }
  } catch (e) {
    missing.push(`could not probe the live workbook: ${e}`);
  }

  if (!missing.length) {
    log_("INFO", "check_apps_script_api_: all required Apps Script methods are present");
    return true;
  }

  fail_("Error: this Apps Script runtime is missing methods the script needs, " +
        "so it cannot run safely:\n\n  " + missing.join("\n  ") +
        "\n\nCheck the Apps Script runtime / V8 setting, or update the " +
        "REQUIRED_*_API lists if the API has legitimately changed.");
  return false;
}


/**
 * Fail fast when required sheets are missing, reporting all of them at once.
 *
 * @param {Array<string>} names Sheet names the pipeline cannot work without.
 * @return {boolean} true when every sheet is present.
 */
function require_sheets_(names) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const missing = names.filter(name => !ss.getSheetByName(name));

  if (!missing.length) {
    log_("INFO", `require_sheets_: all ${names.length} required sheet(s) present ` +
                 `(${names.join(', ')})`);
    return true;
  }

  fail_("Error: missing required sheet(s): " + missing.join(', ') +
        "\n\nCreate them and try again, or use Water > Check setup for details.");
  return false;
}


/**
 * Normalise anything thrown into a readable one-line description plus its trace.
 * @param {*} e Thrown value.
 * @return {{message: string, stack: string, is_timeout: boolean}} Parsed error.
 */
function describe_error_(e) {
  const message = (e && e.message) ? String(e.message) : String(e);
  const stack = (e && e.stack) ? String(e.stack) : '(no stack)';
  return {
    message: message,
    stack: stack,
    is_timeout: TRANSIENT_ERROR_PATTERNS.some(re => re.test(message)),
  };
}


/**
 * Run one pipeline stage with timing and error capture on the log.
 *
 * Re-throws on failure so that handle_errors_() alerts the user once, rather
 * than every stage popping its own dialog.
 *
 * @param {string} label Short stage name, e.g. "create month sheets".
 * @param {function(): *} fn Stage to run.
 * @return {*} Whatever fn returns.
 * @throws {*} Whatever fn throws, after logging it with a stack.
 */
function run_stage_(label, fn) {
  const started = Date.now();
  log_("INFO", `=== stage: ${label} - start`);
  try {
    const value = fn();
    log_("INFO", `=== stage: ${label} - ok (${Date.now() - started} ms)`);
    return value;
  } catch (e) {
    const info = describe_error_(e);
    log_("ERROR", `=== stage: ${label} - FAILED after ${Date.now() - started} ms | ` +
                  `${info.message}`);
    log_("ERROR", `=== stage: ${label} - stack:\n${info.stack}`);
    throw e;
  }
}


/**
 * Top-level handler: run fn, and convert any uncaught error into a readable
 * alert plus a full trace in the execution log.
 *
 * The raw Apps Script error dialog names a line number and nothing else, which
 * is why two API mistakes here went unnoticed for several runs.
 *
 * @param {string} label What was being attempted, e.g. "the water billing pipeline".
 * @param {function(): *} fn Work to run.
 * @return {*} fn's value, or null when it threw.
 */
function handle_errors_(label, fn) {
  try {
    return fn();
  } catch (e) {
    const info = describe_error_(e);
    log_("ERROR", `${label} FAILED | ${info.message}`);
    log_("ERROR", `${label} stack:\n${info.stack}`);
    try {
      SpreadsheetApp.getUi().alert(
        `${label} failed.\n\n${info.message}\n\n` +
        `The full trace is in the execution log:\n` +
        `Apps Script editor > Executions > this run > Logs.`
      );
    } catch (ui_error) {
      // No UI available (e.g. a trigger without authorisation): the log is all
      // we get, and throwing here would mask the original error.
      log_("ERROR", `${label}: could not show an alert (${ui_error}); see the log.`);
    }
    return null;
  }
}


/**
 * Run fn, retrying transient Spreadsheets service failures with a linear backoff.
 *
 * Transient means the service said "try again": timeouts, rate limits, backend
 * errors. Anything else is a bug and is re-thrown on the first attempt, so a
 * typo cannot hide behind five retries.
 *
 * @param {string} label What is being attempted, for the log.
 * @param {function(): *} fn Operation to run.
 * @param {{attempts: (number|undefined), delay_ms: (number|undefined)}} [opts]
 *     attempts - total tries including the first (default 3).
 *     delay_ms - base backoff, multiplied by the attempt number (default 2000).
 * @return {*} fn's value.
 * @throws {*} fn's last error when every attempt fails, or immediately for a
 *     non-transient error.
 */
function retry_service_(label, fn, opts) {
  const attempts = (opts && opts.attempts) || 3;
  const delay_ms = (opts && opts.delay_ms) || 2000;

  let last_error;
  for (let i = 1; i <= attempts; i++) {
    try {
      const value = fn();
      if (i > 1) log_("INFO", `${label}: succeeded on attempt ${i}/${attempts}`);
      return value;
    } catch (e) {
      last_error = e;
      const info = describe_error_(e);

      if (!info.is_timeout) {
        log_("ERROR", `${label}: not a transient error, not retrying | ${info.message}`);
        throw e;
      }
      if (i === attempts) {
        log_("ERROR", `${label}: giving up after ${attempts} attempt(s) | ${info.message}`);
        throw e;
      }

      const wait = delay_ms * i;
      log_("WARN", `${label}: attempt ${i}/${attempts} failed | ${info.message} | ` +
                   `retrying in ${wait} ms`);
      // sleep() is verified by check_apps_script_api_(); guarded here so the
      // retry logic itself can never be the thing that breaks.
      if (typeof Utilities.sleep === 'function') Utilities.sleep(wait);
    }
  }

  throw last_error;   // unreachable: the loop either returns or throws
}
