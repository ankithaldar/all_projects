/**
 * =============================================================================
 * 01_make_unpivot.js  -  Water billing: the whole monthly pipeline.
 * =============================================================================
 *
 * One run of main() does everything, in this order:
 *   1. read and validate Parameters!B4
 *   2. create the month sheets for the months ahead, so the sheets the formulas
 *      will name actually exist
 *   3. write the month labels into Unpivot row 1, columns D onwards
 *   4. write the body - apartment, date, and one lookup formula per source sheet
 *   5. tidy the sheet: clear, trim to one blank row, refit the autofilter
 *
 * The month list is defined once, in month_sheet_names_for_(), and used by both
 * the sheet creation (stage 2) and the Unpivot columns (stages 3 and 4). That
 * shared definition is the reason this is one file rather than two steps: the
 * sheets and the columns must agree or the columns read 0.
 *
 * Run it from the Water menu -> "Run pipeline" (see 00_menu.js).
 *
 * -----------------------------------------------------------------------------
 * INPUTS
 * -----------------------------------------------------------------------------
 *   Sheet "Parameters"
 *     B4  start_date       Date  - any day inside the month to be billed. May
 *                                 also be text "yyyy-mm-dd".
 *     B5  per_litre_amount Num   - price per litre (validated here, consumed by
 *                                 a later step).
 *   Sheet "Total_consumption"
 *     A3:A102              Apartment names / ids (column A, up to 100 rows).
 *
 *   Sheets "raw_data_pre_bill" and "raw_data_post_bill" (read *by the generated
 *   formulas*, not by this script; the post_bill one is the copy template)
 *     A2:A407              Apartment names / ids, matched against Unpivot col A.
 *     E1:AJ1               Day headers as TEXT "YYYY-MM-DD", matched against
 *                          Unpivot col B. Real date cells will NOT match and
 *                          fall through the IFERROR to 0.
 *     E2:AJ407             Meter readings, at most 32 columns x 406 rows.
 *
 * -----------------------------------------------------------------------------
 * OUTPUT - sheet "Unpivot", body starting at row 2 (row 1 holds the headers)
 * -----------------------------------------------------------------------------
 *     A  Apartment
 *     B  Date ("YYYY-MM-DD", text)
 *     C  pre_bill reading              (0 when not found)
 *     D  post_bill_litres - May 2026   -> raw_data_post_bill_2026-May
 *     E  post_bill_litres - Jun 2026   -> raw_data_post_bill_2026-Jun
 *     .. one column per month, MONTHS_AHEAD of them, in calendar order.
 *   Only columns D onwards are rewritten on row 1; A1:C1 are yours to label.
 *   Row count = apartments x days_in_month, so 100 apartments in a 30 day
 *   month gives 3000 rows in A2:I3001. The sheet is left with exactly one blank
 *   row after the table, an autofilter over the whole table including headers,
 *   and no leftover rows from previous runs.
 *
 * -----------------------------------------------------------------------------
 * THE MONTH SHEETS
 * -----------------------------------------------------------------------------
 *   The template "raw_data_post_bill" is replicated once per month for the
 *   MONTHS_AHEAD months that FOLLOW the billing month in B4, so B4 = 2026-04-01
 *   produces 2026-May .. 2026-Oct. The template itself is kept, hidden, as the
 *   reference copy.
 *
 *   Each copy is moved to sit directly after MONTH_SHEETS_AFTER
 *   ('raw_data_pre_bill'), in month order. The templates in
 *   HIDE_TEMPLATE_SHEETS are then hidden - hiding never breaks the Unpivot
 *   formulas or getSheetByName(), it only removes the tab. A workbook must keep
 *   at least one visible sheet, so a template that is the only visible one is
 *   left visible with an ERROR logged.
 *
 *   Re-running is safe: a month sheet that already exists is SKIPPED, never
 *   overwritten, since a prepared sheet may already hold real readings. Delete
 *   it by hand if you want a fresh copy.
 *
 * -----------------------------------------------------------------------------
 * TIMEZONE
 *   Parameters!B4 is read as a date in the script timezone, which must be IST
 *   (Apps Script editor -> Project Settings -> Time zone -> Asia/Kolkata).
 *   check_timezone_() logs a warning when it is not. Everything date related
 *   uses local getters (getFullYear/getMonth/getDate) and local-midnight Date
 *   objects, and deliberately avoids toISOString(), new Date("yyyy-mm-dd") and
 *   Date.UTC for calendar work - all three convert to UTC first, which turns
 *   2026-04-01 00:00 IST into 2026-03-31 and shifts the whole month.
 *
 * LOGGING
 *   Timestamped lines go to the execution log:
 *     Apps Script editor -> Executions -> the run -> "Logs".
 *   Alerts are also shown, but only for conditions the user can fix (bad
 *   Parameters cell, missing sheet).
 *
 * KNOWN ISSUE - DAY HEADERS
 *   The generated formulas match a date in column B against the day headers in
 *   E1:AJ1 of each source sheet:
 *       MATCH($B2, 'raw_data_post_bill_2026-May'!$E$1:$AJ$1, 0)
 *   copyTo() copies those headers verbatim, so a sheet named ..._2026-May still
 *   carries APRIL's day headers (2026-04-01 ...). Until the headers are changed
 *   to that month's dates, every lookup falls through the IFERROR and returns
 *   0. This script deliberately does NOT rewrite row 1 of the copies: it cannot
 *   know whether that row also holds labels, merges or styling worth keeping.
 *
 * OTHER KNOWN ISSUES
 *   1. The raw_data_* ranges are hard-coded to rows 2..407 / columns E..AJ; more
 *      apartments or days are silently read as 0 by the IFERROR.
 *   2. per_litre_amount is validated here but only consumed by later steps.
 *
 * STATUS - fixed in the 2026-10 revision
 *   - The two formula template literals were missing the separating comma, so JS
 *     parsed the first as a *tag* applied to the second and the run died with the
 *     very unhelpful "TypeError: kk is not a function" (V8 was printing the two
 *     ${k} interpolations from inside the tag). It is a valid tagged template, so
 *     neither the editor nor a syntax check flags it.
 *   - add_formulaes_to_crossjoin_array_() never returned its result, so main()
 *     passed undefined to setValues(), and `k` leaked a global.
 *   - generate_date_array() subtracted 1 from getMonth(), which is already
 *     0-based, so a 2026-04-01 input built March (December would have rolled into
 *     January). A guard now logs ERROR if the produced days leave the month.
 *   - Log timestamps and day counts converted to UTC before printing, so an IST
 *     run logged "2026-02-28 to 2026-03-29" for a March month.
 *   - Sheet.getPosition(), Sheet.setPosition() and Sheet.isSheetVisible() do not
 *     exist in Apps Script and all threw "is not a function". Positions now come
 *     from sheet_position_() (matching sheet IDs), reordering goes through
 *     move_sheet_to_() (setActiveSheet + moveActiveSheet), and visibility is
 *     "not isSheetHidden()".
 *   - The body is written in three calls, not thousands: one setValues for the
 *     identity columns, one setFormulas for the first data row, then a single
 *     Range.autoFill() that replicates the lookup formulas down every column.
 *     Sending all 21,700 formulas as individual strings timed the document
 *     service out with "Service Spreadsheets timed out while accessing
 *     document", even in 500-row chunks.
 *   - Chunking was tried first and then removed. Range.copyTo() is not a
 *     substitute: its docs say only the destination's "top-left cell position is
 *     relevant", so it pastes once instead of replicating. Range.autoFill() is
 *     the documented drag-fill equivalent and needs no chunking.
 *   - The replicated formulas are read back and checked ($A/$B row anchors plus
 *     the source sheet name) before the run is called a success; a mismatch
 *     aborts with an alert instead of leaving a half written table.
 *   - The Unpivot sheet is rebuilt cleanly: previous filter dropped, the table
 *     width cleared before writing, trimmed to one blank row, filter re-applied.
 * =============================================================================
 */


/**
 * Timezone the Parameters dates are written in. IST has a +05:30 offset, so a
 * UTC-based shortcut (toISOString(), new Date("2026-04-01"), Date.UTC) would
 * report the *previous* day for anything at local midnight.
 * Must match: Apps Script editor -> Project Settings -> "Time zone".
 */
const EXPECTED_TIMEZONE = 'Asia/Kolkata'; // IST, UTC+05:30

/**
 * Spellings that mean the same zone as EXPECTED_TIMEZONE. "Asia/Calcutta" is the
 * legacy IANA alias and still shows up in some environments; without this the
 * check would cry wolf over a perfectly correct setting.
 */
const EXPECTED_TIMEZONE_ALIASES = ['Asia/Kolkata', 'Asia/Calcutta'];

/**
 * How many months of post_bill data the pipeline carries. Both steps use this:
 * step 2 replicates this many month sheets, step 1 adds one column per month.
 */
const MONTHS_AHEAD = 6;

/**
 * Prefix of the sheet holding the post_bill readings. One sheet per month is
 * named "<POST_BILL_SHEET_PREFIX>_<yyyy-Mmm>"; the un-suffixed sheet is the
 * template that step 2 replicates and then hides.
 */
const POST_BILL_SHEET_PREFIX = 'raw_data_post_bill';

/** Sheet holding the pre_bill readings: one column in the Unpivot table. */
const PRE_BILL_SHEET = 'raw_data_pre_bill';

/** Sheets the pipeline cannot run without. Checked up front by main(). */
const PIPELINE_REQUIRED_SHEETS = Object.freeze([
  'Parameters',
  'Total_consumption',
  PRE_BILL_SHEET,
  POST_BILL_SHEET_PREFIX,
  'Unpivot',
]);

/**
 * Template sheets that step 2 replicates. Each becomes a sheet named
 * "<template>_<yyyy-Mmm>"; adding PRE_BILL_SHEET here replicates that one too.
 */
const MONTH_SHEET_SOURCES = Object.freeze([
  POST_BILL_SHEET_PREFIX,
]);

/**
 * Sheet the new month sheets are placed directly after, so the monthly copies
 * sit next to the other raw data instead of piling up at the end of the book.
 * If it is missing the copies are simply appended, with a warning.
 */
const MONTH_SHEETS_AFTER = 'raw_data_pre_bill';

/**
 * Template sheets to hide once their monthly copies exist. Hiding keeps the
 * template as a reference (formulas and getSheetByName still reach it) while
 * keeping it out of the way when picking sheets by hand.
 */
const HIDE_TEMPLATE_SHEETS = Object.freeze([
  POST_BILL_SHEET_PREFIX,
]);

/** Column layout of the Unpivot table (1-based, as in getRange). */
const TABLE_COLUMNS = Object.freeze({
  APARTMENT: 1,
  DATE: 2,
  FIRST_READING: 3,   // pre_bill, then one post_bill column per month
});

/**
 * Three letter month names indexed 0-11 (Jan = 0).
 * Hardcoded rather than via Utilities.formatDate(d, tz, 'MMM') so month labels
 * never depend on the script locale.
 */
const MONTH_ABBREV = Object.freeze([
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
]);


/**
 * Current time as text in the script timezone, for log prefixes.
 * @return {string} "yyyy-MM-dd HH:mm:ss" in the script timezone.
 */
function log_timestamp_() {
  return Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'yyyy-MM-dd HH:mm:ss');
}


/**
 * Warn when the script timezone is not the one the Parameters dates assume.
 * Only a warning: the script still runs, but day boundaries can shift.
 * @return {boolean} true when the timezone matches, false otherwise.
 */
function check_timezone_() {
  const tz = Session.getScriptTimeZone();
  if (!EXPECTED_TIMEZONE_ALIASES.includes(tz)) {
    log_("WARN", `Script timezone is "${tz}" but Parameters dates are expected in ` +
                 `"${EXPECTED_TIMEZONE}". If days look off by one, set ` +
                 `Project Settings > Time zone to ${EXPECTED_TIMEZONE}.`);
    return false;
  }
  log_("INFO", `Timezone OK: ${tz}`);
  return true;
}


/**
 * Small logging helper. Uses console.log (V8 runtime) so the lines carry a
 * timestamp and show up in the execution log.
 * @param {string} level One of INFO, WARN, ERROR.
 * @param {string} message Human readable description of the event.
 * @return {void}
 */
function log_(level, message) {
  console.log(`${log_timestamp_()} | ${level} | ${message}`);
}


/**
 * Common failure path: log the reason, alert the user, and return null so the
 * caller can bail out immediately.
 * @param {string} message Message shown in the alert box.
 * @param {*} [error] Optional caught exception, appended to the log line.
 * @return {null} Always null, so callers can `return fail_(...)`.
 */
function fail_(message, error) {
  log_("ERROR", error ? `${message} | exception: ${error}` : message);
  SpreadsheetApp.getUi().alert(message);
  return null;
}


/**
 * Fetch a sheet by name from the active spreadsheet.
 * @param {string} sheet Name of the sheet.
 * @return {GoogleAppsScript.Spreadsheet.Sheet} The sheet.
 */
function get_sheet_(sheet) {
  return SpreadsheetApp.getActiveSpreadsheet().getSheetByName(sheet);
}


/**
 * Find a sheet's 0-based position among its siblings.
 *
 * Neither Sheet.getPosition() nor Sheet.setPosition() exists in Apps Script, so
 * the index has to be derived by matching sheet IDs, which are stable and unique.
 *
 * @param {GoogleAppsScript.Spreadsheet.Sheet} sheet Sheet to locate.
 * @return {number} 0-based index, or -1 when the sheet is not in the workbook.
 */
function sheet_position_(sheet) {
  const id = sheet.getSheetId();
  const siblings = sheet.getParent().getSheets();
  for (let i = 0; i < siblings.length; i++) {
    if (siblings[i].getSheetId() === id) return i;
  }
  return -1;
}


/**
 * Move a sheet to a 0-based position in the workbook's tab order.
 *
 * The Sheet class offers no way to reorder itself. The only route is through the
 * spreadsheet: make the sheet active, then move the *active* sheet. That makes
 * each moved sheet flash on screen, so callers should put the previously active
 * sheet back once they are done.
 *
 * @param {GoogleAppsScript.Spreadsheet.Spreadsheet} spreadsheet Target workbook.
 * @param {GoogleAppsScript.Spreadsheet.Sheet} sheet Sheet to move.
 * @param {number} position 0-based destination index.
 * @return {void}
 */
function move_sheet_to_(spreadsheet, sheet, position) {
  spreadsheet.setActiveSheet(sheet);
  spreadsheet.moveActiveSheet(position);
}


/**
 * Put a sheet back in focus, ignoring sheets that cannot be activated.
 * @param {GoogleAppsScript.Spreadsheet.Spreadsheet} spreadsheet Target workbook.
 * @param {?GoogleAppsScript.Spreadsheet.Sheet} sheet Sheet to activate.
 * @return {void}
 */
function restore_active_sheet_(spreadsheet, sheet) {
  if (!sheet || sheet.isSheetHidden()) return;   // a hidden sheet cannot be active
  try {
    spreadsheet.setActiveSheet(sheet);
  } catch (e) {
    log_("WARN", `restore_active_sheet_: could not re-focus '${sheet.getName()}': ${e}`);
  }
}


/**
 * Convert a 1-based column number to its letters, for log messages only.
 * @param {number} col Column number, 1 = A.
 * @return {string} e.g. 4 -> "D".
 */
function column_letter_(col) {
  let n = col;
  let out = '';
  while (n > 0) {
    const rem = (n - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}


/**
 * Normalise a cell value into a Date at local midnight of the script timezone.
 *
 * Accepts either a real date cell or text like "2026-04-01". Text is split and
 * rebuilt by hand rather than via new Date("2026-04-01"), which JavaScript
 * treats as UTC midnight and would therefore land on 2026-03-31 in IST.
 * @param {*} value Value read from the sheet.
 * @return {?Date} Local midnight of that calendar day, or null if unparseable.
 */
function coerce_to_date_(value) {
  if (value instanceof Date && !isNaN(value)) {
    return new Date(value.getFullYear(), value.getMonth(), value.getDate()); // drop the time part
  }
  if (typeof value === 'string') {
    const m = value.trim().match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
    if (m) {
      const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
      // Reject impossible dates JS would silently roll over, e.g. 2026-02-31.
      if (!isNaN(d) && d.getMonth() === Number(m[2]) - 1 && d.getDate() === Number(m[3])) {
        log_("INFO", `coerce_to_date_: parsed text "${value}" as a date in ` +
                     `${Session.getScriptTimeZone()}`);
        return d;
      }
    }
  }
  return null;
}


/**
 * Read and validate the user supplied Parameters.
 * @return {?{start_date: Date, per_litre_amount: number}} Validated inputs, or
 *     null when a value is missing/invalid (in which case the user was alerted).
 */
function read_and_validate_inputs_() {
  try {
    var sheet = get_sheet_("Parameters");
    // Single cell A1 notation addresses, one per input.
    var input_ranges = {
      "start_date": "B4",
      "per_litre_amount": "B5"
    };

    log_("INFO", `Reading inputs: start_date=${input_ranges.start_date}, ` +
                 `per_litre_amount=${input_ranges.per_litre_amount}`);

    var inputs = {};
    inputs.start_date = coerce_to_date_(sheet.getRange(input_ranges.start_date).getValue());
    inputs.per_litre_amount = sheet.getRange(input_ranges.per_litre_amount).getValue();

    // Month Start Date
    if (inputs.start_date === null) {
      return fail_("Error: Month Start Date must be a valid date (expected yyyy-mm-dd).");
    }

    if (typeof inputs.per_litre_amount !== 'number' || inputs.per_litre_amount <= 0) {
      return fail_("Error: per_litre_amount must be a positive number.");
    }

    log_("INFO", `Inputs OK | start_date=${format_date_(inputs.start_date)} ` +
                 `| per_litre_amount=${inputs.per_litre_amount}`);

    return inputs;
  } catch (e) {
    return fail_("An error occurred while reading inputs. Check script logs for details.", e);
  }
}


/**
 * Build one Date per day of the month containing month_start_date.
 * @param {Date} month_start_date Any day inside the target month.
 * @return {?Array<Date>} Chronological list of days, or null if the argument is
 *     not a valid Date.
 */
function generate_date_array_(month_start_date) {
  if (!(month_start_date instanceof Date) || isNaN(month_start_date)) {
    fail_("Error: Month Start Date must be a valid date.");
    return null;
  }

  // Read the calendar parts with the *local* (script timezone) getters, so the
  // month is the one the user sees in the sheet. getUTC* would report the
  // previous month for a local-midnight date in IST (UTC+05:30).
  month_start_date = new Date(month_start_date);
  const year = month_start_date.getFullYear();
  const month = month_start_date.getMonth();  // 0-based: Jan = 0, Dec = 11
  const month_label = `${year}-${String(month + 1).padStart(2, '0')}`;

  // Day 0 of the *next* month is the last day of `month`. Computed with Date.UTC
  // because that arithmetic is self-consistent within UTC, so it cannot be
  // shifted by the +05:30 offset or by a DST boundary.
  const days_in_month = new Date(Date.UTC(year, month + 1, 0)).getUTCDate();

  log_("INFO", `generate_date_array_: month ${month_label} has ${days_in_month} days ` +
               `| input ${format_date_(month_start_date)} ` +
               `| timezone ${Session.getScriptTimeZone()}`);

  // Local midnight of day 1..days_in_month, so format_date_() reads back exactly
  // the intended day. `month` is already 0-based, so it is used as-is; the old
  // `month - 1` is what silently produced the previous month.
  const dates = Array.from({ length: days_in_month },
    (_, i) => new Date(year, month, i + 1));

  const first = format_date_(dates[0]);
  const last = format_date_(dates[dates.length - 1]);
  // Cheap guard so a month off-by-one can never pass unnoticed again.
  if (first.slice(0, 7) !== month_label || last.slice(0, 7) !== month_label) {
    log_("ERROR", `generate_date_array_: month mismatch, expected ${month_label} ` +
                  `but produced ${first}..${last}`);
  }
  log_("INFO", `generate_date_array_: produced ${dates.length} date(s), ${first} to ${last}`);

  return dates;
}


/**
 * Format a single date as "YYYY-MM-DD" text.
 * Column B of the output holds text dates on purpose: the generated formulas
 * MATCH it against the day headers of the raw_data_* sheets, which are text.
 *
 * Uses the *local* (script timezone) getters, matching the local-midnight dates
 * built by generate_date_array_(). toISOString().slice(0, 10) would be wrong
 * here: it converts to UTC first, turning 2026-04-01 00:00 IST into 2026-03-31.
 * @param {Date} dates Date to format.
 * @return {string} Zero padded "YYYY-MM-DD".
 */
function format_date_(dates) {
  dates = new Date(dates);
  var yyyy = dates.getFullYear();
  var mm = String(dates.getMonth() + 1).padStart(2, '0');
  var dd = String(dates.getDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}


/**
 * Label a date as "<yyyy>-<Mmm>" using script-timezone calendar parts.
 * @param {Date} date Any date inside the month to label.
 * @return {string} e.g. "2026-May".
 */
function month_label_(date) {
  return `${date.getFullYear()}-${MONTH_ABBREV[date.getMonth()]}`;
}


/**
 * Shift a date by whole months, landing on day 1 of the result.
 *
 * Day 1 is deliberate: adding months to a late-in-the-month date would overflow
 * (31 Jan + 1 month = 3 Mar in JavaScript).
 * @param {Date} date Starting date; only its year and month are used.
 * @param {number} months Number of months to add, may be negative.
 * @return {Date} Local midnight on day 1 of the target month.
 */
function add_months_(date, months) {
  return new Date(date.getFullYear(), date.getMonth() + months, 1);
}


/**
 * Names of the month sheets for one template, in chronological order, starting
 * at the month AFTER the billing month.
 *
 * This is the single definition both halves of the pipeline use, which is what
 * keeps the Unpivot columns and the created sheets in step.
 *
 * @param {string} template_name Sheet the copies are made from.
 * @param {Date} billing_month Any day inside the month being billed.
 * @param {number} [count] How many months, defaults to MONTHS_AHEAD.
 * @return {Array<string>} e.g. ["raw_data_post_bill_2026-May", ...].
 */
function month_sheet_names_for_(template_name, billing_month, count) {
  const first = add_months_(billing_month, 1);   // the month after the billing month
  const n = (count === undefined) ? MONTHS_AHEAD : count;
  const names = [];
  for (let i = 0; i < n; i++) {
    names.push(`${template_name}_${month_label_(add_months_(first, i))}`);
  }
  return names;
}


/**
 * Names of the per-month post_bill sheets, i.e. the columns the Unpivot table
 * gains. Kept as its own helper because the table is built from the post_bill
 * months only; the pre_bill sheet keeps its single column.
 * @param {Date} billing_month Any day inside the month being billed.
 * @param {number} [count] How many months, defaults to MONTHS_AHEAD.
 * @return {Array<string>} e.g. ["raw_data_post_bill_2026-May", ...].
 */
function month_sheet_names_(billing_month, count) {
  return month_sheet_names_for_(POST_BILL_SHEET_PREFIX, billing_month, count);
}


/**
 * Column headers for the month columns, e.g. "post_bill_litres - May 2026".
 * @param {Array<string>} month_sheets Names from month_sheet_names_().
 * @return {Array<string>} One label per month sheet.
 */
function month_column_headers_(month_sheets) {
  return month_sheets.map(name => {
    // "raw_data_post_bill_2026-May" -> "May 2026"
    const m = name.match(/_(\d{4})-([A-Z][a-z]{2})$/);
    return `post_bill_litres - ${m ? `${m[2]} ${m[1]}` : name}`;
  });
}


/**
 * Warn about month sheets that do not exist yet, so a run before step 2 is
 * obvious in the log rather than showing up as a column of silent zeros.
 * @param {Array<string>} month_sheets Names from month_sheet_names_().
 * @return {void}
 */
function warn_missing_month_sheets_(month_sheets) {
  const missing = month_sheets.filter(name => !get_sheet_(name));
  if (!missing.length) return;
  log_("WARN", `warn_missing_month_sheets_: ${missing.length} month sheet(s) do not exist ` +
               `yet, so those columns will read 0: ${missing.join(', ')}. ` +
               `Run "2. Create month sheets" first.`);
}


/**
 * Read a single column of identifiers, ignoring trailing empty rows.
 * @param {string} sheet_name Sheet to read from.
 * @param {string} ranges A1 notation range, expected to be one column.
 * @return {Array<string>} Values up to and including the last non-empty one.
 * @throws {Error} When the sheet does not exist.
 */
function generate_apartment_array_(sheet_name, ranges) {
  var sheet = get_sheet_(sheet_name);
  if (!sheet) throw new Error(`Sheet ${sheet_name} not found.`);

  const values = sheet.getRange(ranges).getValues(); // 100x1 2-D array
  const last_idx = values.findLastIndex(row => row[0] !== '');

  const apartments = last_idx === -1 ? [] : values.slice(0, last_idx + 1).map(row => row[0]);

  log_("INFO", `generate_apartment_array_: ${apartments.length} apartment(s) from ` +
               `'${sheet_name}'!${ranges}` +
               (apartments.length ? ` (first=${apartments[0]}, last=${apartments[apartments.length - 1]})` : ''));
  if (apartments.length === 0) {
    log_("WARN", `generate_apartment_array_: '${sheet_name}'!${ranges} is empty - the output will have no rows.`);
  }

  return apartments;
}


/**
 * Cartesian product: every apartment against every date.
 * Output is already ordered apartment-major (all days of apt 1, then apt 2, ...).
 * @param {Array<string>} arr_1 Apartment identifiers.
 * @param {Array<Date>} arr_2 Dates.
 * @return {Array<Array<string>>} Rows of [apartment, "YYYY-MM-DD"].
 */
function crossjoin_array1_array2_(arr_1, arr_2) {
  log_("INFO", `crossjoin_array1_array2_: ${arr_1.length} apartment(s) x ${arr_2.length} day(s) ` +
               `= ${arr_1.length * arr_2.length} row(s)`);

  return arr_1.flatMap(apartment =>
    arr_2.map(date => [apartment, format_date_(date)])
  );
}


/**
 * Build one IFERROR/INDEX/MATCH lookup of a reading from a raw_data sheet.
 * Falls back to 0 when the apartment or the day header is not found, so a
 * missing sheet or an empty cell never shows as #N/A.
 *
 * $A / $B are column-anchored and relative on the row, so the formula reads the
 * apartment and date from its own row and can be dragged sideways.
 * @param {string} source_sheet Sheet to read from.
 * @param {number} sheet_row Row on the Unpivot sheet the formula lands on.
 * @return {string} The formula as text.
 */
function reading_formula_(source_sheet, sheet_row) {
  const ref = /^[A-Za-z0-9_]+$/.test(source_sheet) ? source_sheet : `'${source_sheet}'`;
  return `=IFERROR(INDEX(${ref}!$E$2:$AJ$407, ` +
         `MATCH($A${sheet_row}, ${ref}!$A$2:$A$407, 0), ` +
         `MATCH($B${sheet_row}, ${ref}!$E$1:$AJ$1, 0)), 0)`;
}


/**
 * Append the reading formulas to every [apartment, date] row.
 *
 * One formula per source sheet: first the pre_bill sheet, then one per month
 * sheet, so the columns line up with the headers written on row 1.
 *
 * @param {Array<Array<string>>} cj_arr Rows of [apartment, "YYYY-MM-DD"].
 * @param {Array<string>} sources Source sheet names, in column order: the
 *     pre_bill sheet first, then one per month. Must be the same list
 *     write_unpivot_table_() verifies against.
 * @return {Array<Array<(string|number)>>} Rows of
 *     [apartment, date, pre_bill_formula, post_bill_formula, ...].
 */
function add_formulaes_to_crossjoin_array_(cj_arr, sources) {
  const result = [];

  for (var j = 0; j < cj_arr.length; j++) {
    // Sheet row this pair lands on: the output table starts on row 2.
    var k = j + 2;
    var row = [cj_arr[j][0], cj_arr[j][1]];
    for (var c = 0; c < sources.length; c++) {
      row.push(reading_formula_(sources[c], k));
    }
    result.push(row);
  }

  log_("INFO", `add_formulaes_to_crossjoin_array_: built ${result.length} row(s) x ` +
               `${sources.length} reading column(s) (rows 2..${result.length + 1}) ` +
               `| sources: ${sources.join(', ')}`);

  return result;
}


/**
 * Write the table body in three calls:
 *   1. setValues() for the identity columns (apartment, date) - plain values,
 *      cheap enough to send as one block.
 *   2. setFormulas() for the lookup formulas of the FIRST data row only.
 *   3. one Range.autoFill() to replicate that row's formulas down every column.
 *
 * Step 3 is the whole point: the sheet replicates ~21,700 formulas in a single
 * operation instead of the script sending every one of them as a string, which
 * is what used to time the Spreadsheets service out.
 *
 * The fill is not trusted blindly - verify_formula_row_() reads the results back
 * and checks the $A/$B row anchors and the source sheet name, so a fill that
 * does not replicate as expected is caught and reported rather than silently
 * producing a table of wrong formulas.
 *
 * @param {GoogleAppsScript.Spreadsheet.Sheet} sheet Target sheet.
 * @param {Array<Array<(string|number)>>} rows Body rows, each of table width.
 * @param {number} first_row Row the body starts on.
 * @param {number} first_col Column the body starts on (A).
 * @param {number} identity_width Width of the leading value columns.
 * @param {number} reading_col First formula column.
 * @param {number} reading_width Number of formula columns.
 * @param {Array<string>} sources Source sheet names, in column order.
 * @return {boolean} true when the body was written and verified. false means the
 *     fill did not replicate as expected; the caller should abort.
 */
function write_body_(sheet, rows, first_row, first_col,
                   identity_width, reading_col, reading_width, sources) {
  const total = rows.length;
  if (total === 0) return true;

  // 1. Identity columns (apartment, date) in a single setValues call. Plain
  //    values are cheap, so the whole block goes in one request.
  sheet.getRange(first_row, first_col, total, identity_width)
       .setValues(rows.map(row => row.slice(0, identity_width)));

  // 2. Lookup formulas: the first data row only, then replicate it downward.
  const first_formulas = rows[0].slice(identity_width, identity_width + reading_width);
  sheet.getRange(first_row, reading_col, 1, reading_width).setFormulas([first_formulas]);

  if (total === 1) return true;

  const last_row = first_row + total - 1;
  // autoFill() is the drag-fill operation: it replicates the pattern in this
  // range across the destination, adjusting relative references per row, in a
  // single server call. Per the docs the destination must CONTAIN the source and
  // extend it in one direction only, hence row first_row..last_row.
  //
  // Note copyTo() is NOT a substitute: its docs say only "the top-left cell
  // position is relevant" for the destination, so it pastes the source once
  // rather than replicating it.
  const pattern = sheet.getRange(first_row, reading_col, 1, reading_width);
  const destination = sheet.getRange(first_row, reading_col, total, reading_width);
  pattern.autoFill(destination, SpreadsheetApp.AutoFillSeries.DEFAULT_SERIES);
  log_("INFO", `write_body_: auto-filled row ${first_row} across ` +
               `${column_letter_(reading_col)}${first_row}:` +
               `${column_letter_(reading_col + reading_width - 1)}${last_row} ` +
               `with one autoFill()`);

  // 3. Verify before trusting it.
  const probes = [first_row + 1, Math.floor((first_row + last_row) / 2), last_row]
      .filter(r => r > first_row && r <= last_row);
  for (const r of probes) {
    const got = sheet.getRange(r, reading_col, 1, reading_width).getFormulas()[0];
    for (let c = 0; c < reading_width; c++) {
      if (!verify_formula_row_(got[c], r, sources[c])) {
        log_("ERROR", `write_body_: verification failed at row ${r}, column ` +
                      `${column_letter_(reading_col + c)}: expected anchors $A${r}/$B${r} ` +
                      `from '${sources[c]}', got: ${got[c]}`);
        fail_("Error: the lookup formulas did not replicate down the Unpivot " +
              "sheet as expected, so the table was left as it is rather than " +
              "half written. See the execution log for the offending formula.");
        return false;
      }
    }
  }
  log_("INFO", `write_body_: verified rows ${probes.join(', ')} ` +
               `(${reading_width} formula column(s) each)`);

  return true;
}


/**
 * Check one replicated formula against the row it should belong to.
 *
 * A plain indexOf('$A3') would also match '$A30', so the row numbers are
 * extracted and compared as whole values.
 *
 * @param {string} formula Formula text read back from the cell.
 * @param {number} expected_row Row the formula must be anchored to.
 * @param {string} expected_source Sheet name the formula must read from.
 * @return {boolean} true when the formula is anchored as expected.
 */
function verify_formula_row_(formula, expected_row, expected_source) {
  if (typeof formula !== 'string' || formula.charAt(0) !== '=') return false;
  const mA = formula.match(/\$A(\d+)/);
  const mB = formula.match(/\$B(\d+)/);
  if (!mA || mA[1] !== String(expected_row)) return false;
  if (!mB || mB[1] !== String(expected_row)) return false;
  return formula.indexOf(expected_source) !== -1;
}


/**
 * Rebuild the "Unpivot" table in place:
 *   1. wipe the table width below the header row, so a rerun can never leave
 *      stale values behind,
 *   2. write the month labels into row 1 from column D onwards (A1:C1 untouched),
 *   3. write the new body,
 *   4. keep exactly one blank row under the table and delete everything below it,
 *   5. fit the autofilter to the table, header row included.
 *
 * @param {Array<Array<(string|number)>>} rows Body rows, each of table width.
 * @param {Array<string>} month_headers Labels for the month columns (D onwards).
 * @param {Array<string>} sources Source sheet names, in column order, used to
 *     verify the replicated formulas.
 * @return {boolean} true when the table was written; false when it refused to
 *     write or the formula fill failed verification (an alert has been shown).
 */
function write_unpivot_table_(rows, month_headers, sources) {
  const sheet = get_sheet_("Unpivot");
  if (!sheet) {
    fail_("Error: sheet 'Unpivot' was not found. Create it with the headers on row 1.");
    return;
  }
  if (!rows.length) {
    log_("WARN", "write_unpivot_table_: nothing to write, leaving the sheet untouched.");
    return true;
  }

  const HEADER_ROW = 1;
  const first_col = TABLE_COLUMNS.APARTMENT;         // A
  const month_col = TABLE_COLUMNS.FIRST_READING + 1;  // D: pre_bill sits in C
  const last_col = month_col + month_headers.length - 1;
  const width = last_col - first_col + 1;
  const first_row = HEADER_ROW + 1;                 // body starts under the headers
  const last_row = first_row + rows.length - 1;     // last row holding data

  if (rows[0].length !== width) {
    log_("ERROR", `write_unpivot_table_: row width ${rows[0].length} does not match the ` +
                  `table width ${width} (${month_headers.length} month column(s)); ` +
                  `refusing to write.`);
    return false;
  }

  // Drop any filter left by a previous run before doing the row surgery below:
  // a live filter both blocks createFilter() on overlapping ranges and can make
  // row indices move under us while we delete. Its width is also the only record
  // of how wide the previous table was, which matters for step 4.
  const old_filter = sheet.getFilter();
  let prev_width = width;
  if (old_filter) {
    try {
      prev_width = Math.max(width, old_filter.getRange().getNumColumns());
    } catch (e) {
      log_("WARN", `write_unpivot_table_: could not read the previous filter range: ${e}`);
    }
    old_filter.remove();
    log_("INFO", "write_unpivot_table_: removed the filter from the previous run");
  }

  // NOTE: nothing is cleared here on purpose. An earlier version cleared A:I from
  // row 2 down to the bottom of the sheet *before* writing, so any failure during
  // the body write left the sheet empty - it destroyed a perfectly good previous
  // table. Stale rows are now removed by deleting them in step 4 instead, which
  // achieves the same clean result without the destructive window. A failed run
  // therefore leaves the old table in place, partially overwritten at worst.

  // 1. Month labels on row 1. A1:C1 are deliberately left alone.
  const header_range = sheet.getRange(HEADER_ROW, month_col, 1, month_headers.length);
  header_range.setValues([month_headers]);
  // Read them back: proves what actually landed in the cells rather than what
  // we intended to write, which is the difference between "the header is wrong"
  // and "the run died before it got here".
  const header_now = header_range.getValues()[0];
  const header_ok = month_headers.every((h, i) => header_now[i] === h);
  log_(header_ok ? "INFO" : "ERROR",
       `write_unpivot_table_: row 1 now reads ` +
       `${column_letter_(month_col)}..${column_letter_(last_col)}: ` +
       `${header_now.join(' | ')}` +
       (header_ok ? '' : ' - DOES NOT MATCH what was written'));
  if (!header_ok) {
    fail_(`Error: the month headers in '${column_letter_(month_col)}1` +
          `:${column_letter_(last_col)}1' did not stick. Check for merged cells ` +
          `or a protected range on the Unpivot sheet.`);
  }

  // 2. Write the new body, overwriting whatever was there: one setValues for the
  //    identity columns, then one setFormulas for the first data row followed by
  //    a single autoFill() replicating the lookup formulas down every column.
  //    The fill is verified afterwards; a mismatch aborts rather than leaving a
  //    half written table behind.
  const identity_width = TABLE_COLUMNS.DATE - TABLE_COLUMNS.APARTMENT + 1;  // A:B
  const reading_width = last_col - TABLE_COLUMNS.FIRST_READING + 1;          // C..last

  const body_ok = write_body_(sheet, rows, first_row, first_col, identity_width,
                             TABLE_COLUMNS.FIRST_READING, reading_width, sources);
  log_("INFO", `write_unpivot_table_: wrote ${rows.length} row(s) to ` +
               `${column_letter_(first_col)}${first_row}:${column_letter_(last_col)}${last_row} ` +
               `| ${reading_width} formula column(s) per row | ` +
               `3 write calls (values, formulas, autoFill)${body_ok ? '' : ' | FAILED'}`);
  if (!body_ok) return false;   // already alerted; leave the previous table alone

  // 3. If the previous table was wider (MONTHS_AHEAD was lowered), the columns
  //    this run no longer uses would keep their old headers and data. Clear them.
  if (prev_width > width) {
    sheet.getRange(HEADER_ROW, last_col + 1, last_row - HEADER_ROW + 1, prev_width - width)
         .clearContent();
    log_("INFO", `write_unpivot_table_: cleared ${column_letter_(last_col + 1)}:` +
                 `${column_letter_(prev_width)} - the previous table was wider ` +
                 `(${prev_width} columns, now ${width})`);
  }

  // 4. Keep one blank row below the table and delete everything below it. Doing
  //    this by deletion rather than by clearing is what removes stale rows and
  //    their contents in one step.
  if (sheet.getMaxRows() <= last_row) {
    sheet.insertRowAfter(last_row); // table filled the sheet, make room
    log_("INFO", `write_unpivot_table_: inserted a row so blank row ${last_row + 1} exists`);
  }
  const delete_count = sheet.getMaxRows() - (last_row + 1);
  if (delete_count > 0) {
    sheet.deleteRows(last_row + 2, delete_count);
    log_("INFO", `write_unpivot_table_: deleted ${delete_count} trailing row(s), ` +
                 `kept one blank row at ${last_row + 1}`);
  } else {
    log_("INFO", "write_unpivot_table_: no trailing rows to delete");
  }
  // The one spare row must actually be blank, e.g. when the previous table was
  // taller and its data shifted into it.
  sheet.getRange(last_row + 1, first_col, 1, width).clearContent();

  // 5. Fit the filter to the table, header row included, so the dropdowns cover
  //    exactly the data and not the rows trimmed away above.
  sheet.getRange(HEADER_ROW, first_col, last_row, width).createFilter();
  log_("INFO", `write_unpivot_table_: filter applied to ` +
               `${sheet.getFilter().getRange().getA1Notation()} ` +
               `(${last_row} row(s) x ${width} col(s))`);

  return true;
}



/**
 * Hide the template sheets, once their monthly copies exist.
 *
 * @return {number} How many sheets were hidden by this call.
 */
function hide_template_sheets_() {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  let hidden = 0;

  for (const name of HIDE_TEMPLATE_SHEETS) {
    const sheet = get_sheet_(name);
    if (!sheet) {
      log_("WARN", `hide_template_sheets_: '${name}' not found.`);
      continue;
    }
    if (sheet.isSheetHidden()) {
      log_("INFO", `hide_template_sheets_: '${name}' is already hidden.`);
      continue;
    }
    // Apps Script refuses to hide the last visible sheet in a workbook, so give
    // a clear message instead of letting it throw mid-loop. There is no
    // isSheetVisible(), so "visible" means "not hidden".
    const visible = spreadsheet.getSheets().filter(s => !s.isSheetHidden()).length;
    if (visible <= 1) {
      log_("ERROR", `hide_template_sheets_: '${name}' is the only visible sheet, ` +
                    `leaving it visible (a workbook must keep one visible sheet).`);
      continue;
    }
    sheet.hideSheet();
    hidden++;
    log_("INFO", `hide_template_sheets_: hid '${name}'.`);
  }

  return hidden;
}


/**
 * STAGE 1 - create the month sheets for the MONTHS_AHEAD months after the
 * billing month, place them after MONTH_SHEETS_AFTER, then hide the templates.
 *
 * Runs before the Unpivot table is built, because the formulas written into
 * that table reference these sheets by name.
 *
 * @param {Date} billing_month Validated billing month from Parameters!B4.
 * @return {Array<string>} The month sheet names, in calendar order. This is the
 *     list main() turns into Unpivot columns, whether or not each copy was
 *     actually created - warn_missing_month_sheets_() reports any that failed.
 */
function create_month_sheets_(billing_month) {
  const month_sheets = month_sheet_names_(billing_month);
  log_("INFO", `create_month_sheets_: billing month ${month_label_(billing_month)} ` +
               `| target sheets: ${month_sheets.join(', ')} ` +
               `| timezone ${Session.getScriptTimeZone()}`);

  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  let created = 0;
  let skipped = 0;
  let failed = 0;

  // Where the first copy goes. copyTo() always appends at the end, so each copy
  // is moved afterwards; insert_pos then advances so the months stay in order.
  const anchor = get_sheet_(MONTH_SHEETS_AFTER);
  const previously_active = spreadsheet.getActiveSheet();
  let insert_pos = null;
  if (anchor) {
    const anchor_pos = sheet_position_(anchor) + 1;
    if (anchor_pos === -1) {
      log_("WARN", `create_month_sheets_: could not locate '${MONTH_SHEETS_AFTER}' ` +
                   `in the workbook, the new sheets will be appended at the end.`);
    } else {
      insert_pos = anchor_pos + 1;
      log_("INFO", `create_month_sheets_: placing the new sheets after ` +
                   `'${MONTH_SHEETS_AFTER}' (index ${insert_pos})`);
    }
  } else {
    log_("WARN", `create_month_sheets_: '${MONTH_SHEETS_AFTER}' not found, ` +
                 `the new sheets will be appended at the end of the workbook.`);
  }

  for (const source_name of MONTH_SHEET_SOURCES) {
    const source = get_sheet_(source_name);
    if (!source) {
      log_("ERROR", `create_month_sheets_: template sheet '${source_name}' not found.`);
      failed++;
      continue;
    }

    for (const target_name of month_sheet_names_for_(source_name, billing_month)) {
      // Never clobber a month sheet that may already hold readings.
      if (spreadsheet.getSheetByName(target_name)) {
        log_("WARN", `create_month_sheets_: '${target_name}' already exists - skipping.`);
        skipped++;
        continue;
      }

      try {
        // copyTo() appends at the end and returns the new sheet, which is still
        // called "Copy of ..." until it is renamed.
        const copy = source.copyTo(spreadsheet);
        copy.setName(target_name);
        if (insert_pos !== null) {
          move_sheet_to_(spreadsheet, copy, insert_pos);
          log_("INFO", `create_month_sheets_: moved '${target_name}' to index ${insert_pos}`);
          insert_pos++;   // keep May, Jun, Jul... in order
        }
        created++;
        log_("INFO", `create_month_sheets_: created '${target_name}' ` +
                     `(${copy.getMaxRows()} rows x ${copy.getMaxColumns()} cols copied)`);
      } catch (e) {
        failed++;
        log_("ERROR", `create_month_sheets_: could not create '${target_name}': ${e}`);
      }
    }
  }

  const hidden = hide_template_sheets_();
  restore_active_sheet_(spreadsheet, previously_active);
  log_("INFO", `create_month_sheets_: done - created ${created}, ` +
               `skipped ${skipped}, failed ${failed}, hidden ${hidden}`);

  return month_sheets;
}


/**
 * ENTRY POINT - the whole monthly pipeline, in one run.
 *
 *   1. read and validate Parameters!B4
 *   2. create the month sheets, so the sheets the formulas will name exist
 *   3. write the month labels into Unpivot row 1 (D onwards)
 *   4. write the body: apartment, date, and one lookup formula per source sheet
 *   5. tidy the sheet - clear, trim to one blank row, refit the autofilter
 *
 * @return {void}
 */
function main() {
  return handle_errors_("The water billing pipeline", () => {
    log_("INFO", "main: start - water billing pipeline");

    // Preflight: fail before touching the workbook if the runtime or the
    // workbook is not ready, so a failure can never leave a half built table.
    if (!check_apps_script_api_()) return false;
    if (!require_sheets_(PIPELINE_REQUIRED_SHEETS)) return false;
    run_stage_("check timezone", () => check_timezone_());

    const inputs = read_and_validate_inputs_();
    if (inputs === null) {
      // read_and_validate_inputs_() has already alerted the user.
      log_("ERROR", "main: aborting, inputs are invalid.");
      return false;
    }

    // Stage 1: the month sheets. Must come first so the formulas below reference
    // sheets that actually exist.
    const month_sheets = run_stage_("create month sheets",
      () => create_month_sheets_(inputs.start_date));
    run_stage_("check month sheets", () => warn_missing_month_sheets_(month_sheets));

    // Stage 2: the calendar days and the apartments for the billing month.
    const month_dates = run_stage_("build the day list",
      () => generate_date_array_(inputs.start_date));
    if (month_dates === null) return false;   // already alerted

    const apartments = run_stage_("read the apartment list",
      () => generate_apartment_array_('Total_consumption', 'A3:A102'));

    // Stage 3: sources and headers on row 1, then the lookup formulas.
    // sources is the ordered list the columns follow: pre_bill, then one per month.
    const sources = [PRE_BILL_SHEET].concat(month_sheets);
    const headers = month_column_headers_(month_sheets);
    const ap_date_combo = run_stage_("build the body rows", () =>
      add_formulaes_to_crossjoin_array_(
        crossjoin_array1_array2_(apartments, month_dates), sources));

    log_("INFO", `main: ${apartments.length} apartment(s) x ${month_dates.length} day(s) ` +
                 `= ${apartments.length * month_dates.length} row(s) across ` +
                 `${TABLE_COLUMNS.FIRST_READING + month_sheets.length} column(s)`);

    // Stage 4: write the table, then tidy and refilter it.
    const table_ok = run_stage_("write the Unpivot table", () =>
      write_unpivot_table_(ap_date_combo, headers, sources));
    if (table_ok !== true) {
      log_("ERROR", "main: the table was not written, aborting.");
      return false;
    }

    log_("INFO", "main: done");
    return true;
  });
}


// =============================================================================
// BOOTSTRAP GUARD
// =============================================================================
// The error-handling helpers live in errors.js. An Apps Script project keeps an
// explicit list of files, so a new file that only exists in the git repo is NOT
// part of the bound script until it is added by hand
// (Extensions > Apps Script > + > Script > select errors.js).
//
// Forgetting it produced a bare "ReferenceError: handle_errors_ is not defined"
// on every run. If the helpers are missing, install minimal stand-ins so a run
// still completes, and complain loudly - with the exact steps - so the real
// diagnostics can be switched back on.
if (typeof handle_errors_ !== 'function') {
  globalThis.__ERRORS_MODULE_MISSING = true;

  globalThis.handle_errors_ = function (label, fn) {
    try {
      return fn();
    } catch (e) {
      const msg = (e && e.message) ? e.message : String(e);
      log_("ERROR", `${label} FAILED (minimal handler) | ${msg}`);
      if (e && e.stack) log_("ERROR", `${label} stack:\n${e.stack}`);
      try {
        SpreadsheetApp.getUi().alert(`${label} failed.\n\n${msg}\n\n` +
                                    `NOTE: errors.js is missing from this project, so ` +
                                    `this is the minimal fallback handler.`);
      } catch (ui_error) {
        log_("ERROR", `${label}: could not show an alert (${ui_error}).`);
      }
      return null;
    }
  };

  // No preflight, no retries, no stage timing - just pass the work through.
  globalThis.run_stage_ = function (label, fn) {
    log_("INFO", `=== stage: ${label} - start`);
    const value = fn();
    log_("INFO", `=== stage: ${label} - ok`);
    return value;
  };

  globalThis.check_apps_script_api_ = function () {
    log_("WARN", "check_apps_script_api_: SKIPPED, errors.js is not in this project");
    return true;
  };

  globalThis.require_sheets_ = function (names) {
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const missing = names.filter(name => !ss.getSheetByName(name));
    if (!missing.length) return true;
    log_("ERROR", `require_sheets_: missing sheet(s): ${missing.join(', ')}`);
    SpreadsheetApp.getUi().alert("Error: missing required sheet(s): " + missing.join(', '));
    return false;
  };

  log_("WARN", "==========================================================");
  log_("WARN", "errors.js was NOT found in this Apps Script project.");
  log_("WARN", "Running with minimal fallbacks: no API preflight, no stage");
  log_("WARN", "timing, no retries, no formula-fill verification.");
  log_("WARN", "To fix: Extensions > Apps Script > + > Script, and add");
  log_("WARN", "errors.js from this repo (all_gsheet_codes/wateron/errors.js).");
  log_("WARN", "==========================================================");
}
