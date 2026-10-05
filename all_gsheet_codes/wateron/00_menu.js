/**
 * =============================================================================
 * 00_menu.js  -  Water billing: adds the "Water" menu to the spreadsheet.
 * =============================================================================
 *
 * PURPOSE
 *   onOpen() is a *simple trigger*: Apps Script runs it automatically every time
 *   someone opens the spreadsheet, before any menu exists. It builds the custom
 *   "Water" menu so nobody has to open the Apps Script editor and remember the
 *   name of the entry point (main).
 *
 * THE MENU
 *   Water
 *     Run pipeline            -> main(), in 01_make_unpivot.js. Creates the
 *                                month sheets, writes the Unpivot headers and
 *                                the lookup formulas, then tidies the sheet.
 *     -------------------------
 *     Check setup
 *     About this menu
 *
 * WHY THIS IS A SEPARATE FILE
 *   01_make_unpivot.js holds the actual work (the unpivot table and the month
 *   sheet preparation). It stays free of UI wiring; this file is the only place
 *   that talks to the menu, so new steps are registered from a single spot.
 *
 *   Menu items must point at top-level functions *by name*, so the wrappers
 *   below exist purely to give each action a friendly label plus before/after
 *   feedback.
 *
 * INSTALLATION
 *   Paste this file next to 01_make_unpivot.js in the same bound script project
 *   (Extensions > Apps Script). The menu shows up after reopening the file.
 *
 * AUTHORISATION
 *   - onOpen() itself needs no authorisation (simple triggers may only use
 *     services that are safe for unverified code, and SpreadsheetApp.getUi() is).
 *   - Running "Build Unpivot table" DOES need authorisation, because setValues()
 *     modifies the spreadsheet. The first run pops the usual "Google hasn't
 *     verified this app" consent screen; click Advanced -> Go to (project).
 *     After that the menu item works silently.
 *
 * NOTES
 *   - Every menu item must point at a top-level function *by name*, so the
 *     wrappers below exist purely to give each action a friendly label plus
 *     before/after feedback.
 *   - A custom menu cannot be edited by the user; to change it you must change
 *     this file and reopen the spreadsheet.
 *   - Only one onOpen() may exist in a project. If another file in this project
 *     ever adds its own, delete or merge it - the last definition loaded wins
 *     and the other menu silently disappears.
 * =============================================================================
 */

/** Name of the custom menu shown in the toolbar. */
const WATER_MENU_TITLE = 'WaterOn';

/**
 * Simple trigger - fires every time the spreadsheet is opened.
 *
 * Kept deliberately cheap and defensive: it runs on every open, so it must never
 * throw. A failing onOpen silently leaves the user with no menu and no clue.
 * @return {void}
 */
function onOpen() {
  try {
    SpreadsheetApp.getUi()
      .createMenu(WATER_MENU_TITLE)
      .addItem('Run pipeline', 'menu_build_unpivot')
      .addSeparator()
      .addItem('Check setup', 'menu_check_setup')
      .addItem('About this menu', 'menu_about')
      .addToUi();
  } catch (e) {
    // Nothing sensible to show the user here (no menu exists yet), and throwing
    // would be worse than a missing menu - the editor still allows manual runs.
    console.log(`${new Date().toISOString()} | ERROR | onOpen failed to build the ` +
                `"${WATER_MENU_TITLE}" menu: ${e}`);
  }
}


/**
 * Menu action: rebuild the Unpivot table by delegating to main().
 *
 * Failures are reported with an alert rather than the raw Apps Script error
 * dialog, and the stack is pointed at in the execution log.
 * @return {void}
 */
function menu_build_unpivot() {
  const ui = SpreadsheetApp.getUi();
  ui.alert('Running the water billing pipeline, please wait...');

  // main() runs inside handle_errors_(), which alerts on failure itself and
  // returns false, so this wrapper only has to report success.
  if (main() === true) {
    ui.alert('Done.\n\nOpen the "Unpivot" sheet to see the table, and check the\n' +
             'new raw_data_post_bill_<month> sheets are there.');
  }
}


function menu_about() {
  SpreadsheetApp.getUi().alert(
    'WATER MENU\n\n' +
    '"Run pipeline" does the whole monthly job in one go:\n\n' +
    '  1. Creates the month sheets for the months after Parameters!B4,\n' +
    '     e.g. raw_data_post_bill_2026-May .. _2026-Oct, placed after\n' +
    '     raw_data_pre_bill. Existing ones are skipped, never overwritten.\n' +
    '  2. Writes the month labels into Unpivot row 1 (columns D onwards).\n' +
    '  3. Writes one row per apartment per day, with a lookup formula\n' +
    '     per source sheet: C = pre_bill, D.. = each month of post_bill.\n' +
    '  4. Tidies the sheet - one blank row below, autofilter fitted.\n\n' +
    'It reads Parameters!B4 (month to bill) and Parameters!B5 (price per\n' +
    'litre, used by later steps).\n\n' +
    'IMPORTANT: the new month sheets keep the template day headers, so\n' +
    'update E1:AJ1 of each to that month\'s dates, otherwise the lookups\n' +
    'return 0.\n\n' +
    'Logs: Apps Script editor > Executions > <run> > Logs.\n' +
    'The menu itself is rebuilt every time this file is reopened.'
  );
}


/**
 * Read a cell and describe its current value for the setup report.
 * @param {GoogleAppsScript.Spreadsheet.Spreadsheet} ss Active spreadsheet.
 * @param {string} sheet_name Sheet to read from.
 * @param {string} a1 A1 notation address of the cell.
 * @return {string} Short human readable description, e.g. "empty" or "2026-01-01".
 */
function describe_cell_(ss, sheet_name, a1) {
  const sheet = ss.getSheetByName(sheet_name);
  if (!sheet) return '(sheet not found)';

  const value = sheet.getRange(a1).getValue();
  if (value === '' || value === null) return 'EMPTY - set this';
  if (value instanceof Date) {
    // Local getters, not toISOString(): for a date stored at local midnight in
    // IST, toISOString() converts to UTC first and would report the day before.
    const pad = n => String(n).padStart(2, '0');
    return `${value.getFullYear()}-${pad(value.getMonth() + 1)}-${pad(value.getDate())}`;
  }
  if (typeof value === 'number') return String(value);
  return String(value);
}
