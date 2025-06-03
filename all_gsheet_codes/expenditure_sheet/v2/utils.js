// ---------------------------------------------------------------------------------------------------------------------
/**
 * Deletes sheets from the active spreadsheet based on the provided list of sheet names.
 *
 * Iterates through each name in the `sheet_names` array. If a sheet with the given name
 * exists, it will be deleted. If not, a message will be logged indicating the sheet was not found.
 *
 * Args:
 *   sheet_names (string[]): An array of sheet names to delete from the active spreadsheet.
 *
 * Returns:
 *   void
 */
function utils_delete_sheets_by_names(sheet_names) {
  const ss = CONFIG.sheet_cache;

  sheet_names.forEach(name => {
    const sheet = ss.getSheetByName(name);
    if (sheet) {
      ss.deleteSheet(sheet);
      if (CONFIG.is_print) console.log(`${new Date().toISOString()} | Deleted sheet: ${name}`);
    } else {
      if (CONFIG.is_print) console.log(`${new Date().toISOString()} | Sheet "${name}" not found.`);
    }
  });
};
// ---------------------------------------------------------------------------------------------------------------------
/**
 * Unhides a hidden sheet in the active spreadsheet by its name.
 *
 * If the sheet exists and is hidden, it will be unhidden. If the sheet does not exist,
 * or is already visible, a corresponding message will be logged
 *
 * Args:
 *   sheet_name (string[]): An array of sheet names to unhide.
 *
 * Returns:
 *   void
 */
function utils_unhide_sheets_by_name(sheet_names) {
  sheet_names.forEach(name => {
    const sheet = CONFIG.sheet_cache.getSheetByName(name);
    if (sheet && sheet.isSheetHidden()) {
      sheet.showSheet();
      if (CONFIG.is_print) console.log(`${new Date().toISOString()} | Unhide sheet: ${name}`);
    } else if (!sheet) {
      if (CONFIG.is_print) console.log(`${new Date().toISOString()} | Sheet "${name}" not found.`);
    } else {
      if (CONFIG.is_print) console.log(`${new Date().toISOString()} | Sheet "${name}" is already visible.`);
    }
  });
};
// ---------------------------------------------------------------------------------------------------------------------
/**
 * Hides a visible sheet in the active spreadsheet by its name.
 *
 * If the sheet exists and is visible, it will be hidden. If the sheet does not exist,
 * or is already hidden, a corresponding message will be logged
 *
 * Args:
 *   sheet_name (string[]): An array of sheet names Hide.
 *
 * Returns:
 *   void
 */
function utils_hide_sheets_by_name(sheet_names) {
  sheet_names.forEach(name => {
    const sheet = CONFIG.sheet_cache.getSheetByName(name);
    if (sheet && !sheet.isSheetHidden()) {
      sheet.hideSheet();
      if (CONFIG.is_print) console.log(`${new Date().toISOString()} | Hide sheet: ${name}`);
    } else if (!sheet) {
      if (CONFIG.is_print) console.log(`${new Date().toISOString()} | Sheet "${name}" not found.`);
    } else {
      if (CONFIG.is_print) console.log(`${new Date().toISOString()} | Sheet "${name}" is already hidden.`);
    }
  });
};
// ---------------------------------------------------------------------------------------------------------------------
/**
 * Duplicates a template sheet multiple times with specified names.
 * Skips creation if a sheet with the target name already exists.
 *
 * @param {string} template_sheet_name - Name of the template sheet to duplicate
 * @param {string[]} sheet_names - Array of names for the new sheets
 * @throws {Error} If template sheet is not found
 * @return {void}
 */
function utils_duplicate_sheets_by_names(template_sheet_name, sheet_names) {
  const ss = CONFIG.sheet_cache;
  const template_sheet = ss.getSheetByName(template_sheet_name);

  if (!template_sheet) {
    throw new Error(`Template sheet "${template_sheet_name}" not found`);
  }

  sheet_names.forEach(name => {
    if (!ss.getSheetByName(name)) {
      const new_sheet = template_sheet.copyTo(ss);
      new_sheet.setName(name);
      ss.setActiveSheet(new_sheet);
      ss.moveActiveSheet(ss.getNumSheets());
      if (CONFIG.is_print) console.log(`${new Date().toISOString()} | Duplicated ${template_sheet_name} to ${name}`);
    } else {
      if (CONFIG.is_print) console.log(`${new Date().toISOString()} | Sheet ${name} already exists, skipping`);
    }
  });
};
// ---------------------------------------------------------------------------------------------------------------------
