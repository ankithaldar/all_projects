/** @OnlyCurrentDoc */


// Define Global Variables
// inputs
let
  // Specify current year
  years = "2025",

  // Current bank names with accounts
  banks = ['HDFC', 'SBI'],

  // bank in which salary is credited to
  salary_bank = "HDFC",

  salary_amount = 000000,
  meal_card_amount = 000000,
  maid_salary = 000000,
  car_cleaning = 000000,
  home_emi = 000000,

  old_sheet_link = '',

  // Debugger Options
  if_debug = false,

  // months array
  mons = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
  days_in_mons = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],

  // sheets to hide and show
  hidden_sheets = ['Template', 'BankStatement', 'CCStatement-Template'],

  card_map = {
    "CC IndusInd 0596": { "bill_date":  3, "sheet_cell": "$T$25", "repayment_days": 20 },
    "CC SC 8148":       { "bill_date":  8, "sheet_cell": "$T$26", "repayment_days": 22 },
    "CC Axis 6599":     { "bill_date": 11, "sheet_cell": "$T$27", "repayment_days": 19 },
    "CC ICICI 7007":    { "bill_date": 14, "sheet_cell": "$T$28", "repayment_days": 18 },
    "CC One 0531":      { "bill_date": 14, "sheet_cell": "$T$29", "repayment_days": 17 },
    "CC Axis 7878":     { "bill_date": 21, "sheet_cell": "$T$30", "repayment_days": 21 },
    "CC Axis 7175":     { "bill_date": 21, "sheet_cell": "$T$31", "repayment_days": 21 },
    "CC ICICI 8019":    { "bill_date": 28, "sheet_cell": "$T$32", "repayment_days": 18 }
  }
;

// ------------------------------------------------------
// Main Functions Start

function make_year_expense_sheet() {
  // Leap year check and manipulate days in feb
  days = leap_year_check_and_manipulate(years, days_in_mons);

  // Make new sheets for the year
  make_new_sheets(days)

  // Mark various Days of the month
  mark_various_days(days);
};


function delete_sheets_mid_year() {
  var today = new Date();
  for (var i = today.getMonth() + 1; i < mons.length; i++) {
    delete_individual_sheet(mons[i]);
  }

  // UnHide template after the delete
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  ss = ss.setActiveSheet(ss.getSheetByName(hidden_sheets[0])).showSheet();
};

function delete_all_created_sheets() {

  for (var i = 0; i < mons.length; i++) {
    delete_individual_sheet(mons[i]);
  }

  // Delete bank sheets
  for (var i = 0; i < banks.length; i++) {
    delete_individual_sheet('BankStatement - ' + banks[i])
  }

  delete_individual_sheet('CCstatement');

  // UnHide template after the delete
  for (var i = 0; i < hidden_sheets.length; i++) {
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    ss = ss.setActiveSheet(ss.getSheetByName(hidden_sheets[i])).showSheet();
  }
};
