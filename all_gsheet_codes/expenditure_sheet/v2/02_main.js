// ---------------------------------------------------------------------------------------------------------------------
/**
 * Creates a new expense sheet for the entire year.
 *
 * This function generates new monthly sheets and marks important days
 * in each sheet based on configuration values.
 */
function make_year_expense_sheet() {
  // Make new sheets for the year
  make_new_sheets();

  // Mark various Days of the month
  mark_various_days();
}

// ---------------------------------------------------------------------------------------------------------------------
/**
 * Deletes sheets for months that haven't occurred yet in the current year.
 *
 * Useful when stopping the tracking mid-year. Also unhides a previously hidden template sheet.
 */

function delete_sheets_mid_year() {
  utils_delete_sheets_by_names([
    ...Object.keys(CONFIG.month_days).slice(new Date().getMonth() + 1).map(month => month)
  ]);

  // Unhide all hidden template sheets
  utils_unhide_sheets_by_name([CONFIG.hidden_tabs[0]]);
};

// ---------------------------------------------------------------------------------------------------------------------
/**
 * Deletes all monthly and bank-related sheets, including credit card statements.
 *
 * After deletion, it unhides all sheets listed in `hidden_tabs`.
 */

function delete_all_created_sheets() {

  // Delete all sheets
  utils_delete_sheets_by_names([
    ...Object.keys(CONFIG.month_days),
    ...CONFIG.banks.map(bank => `BankStatement - ${bank}`),
    'CCstatement'
  ]);

  // Unhide all hidden template sheets
  utils_unhide_sheets_by_name(CONFIG.hidden_tabs);

};
// ---------------------------------------------------------------------------------------------------------------------
