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
  maid_salary = 000000,
  car_cleaning = 000000,
  home_emi = 000000,

  old_sheet_link = ''

  // months array
  mons = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
  days_in_mons = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],

  // sheets to hide and show
  hidden_sheets = ['Template', 'BankStatement', 'CCStatement-Template'],

  card_map = {
    "CC IndusInd 0596": { "bill_date":  3, "sheet_cell": "$M$28", "repayment_days": 20 },
    "CC SC 8148":       { "bill_date":  8, "sheet_cell": "$M$29", "repayment_days": 22 },
    "CC ICICI 7007":    { "bill_date": 14, "sheet_cell": "$M$30", "repayment_days": 18 },
    "CC One 0531":      { "bill_date": 14, "sheet_cell": "$M$31", "repayment_days": 17 },
    "CC Citi 7878":     { "bill_date": 21, "sheet_cell": "$M$32", "repayment_days": 21 },
    "CC Citi 7175":     { "bill_date": 21, "sheet_cell": "$M$33", "repayment_days": 21 },
    "CC ICICI 8019":    { "bill_date": 28, "sheet_cell": "$M$34", "repayment_days": 18 }
  }
;

// ------------------------------------------------------
// Main Functions Start

function make_year_expense_sheet() {
  // Create bank statements
  create_bank_statement();

  // Create CC Statement sheet
  create_cc_statement();

  // Leap year check and manipulate days in feb
  days = leap_year_check_and_manipulate(years, days_in_mons);

  // Create Month templates
  create_month_templates(days);

  // Mark various Days of the month
  mark_various_days(days);
};


function delete_sheets_mid_year() {
  var today = new Date();
  for (var i = today.getMonth() + 1; i < mons.length; i++) {
    delete_individual_sheet(mons[i])
  }

  // UnHide template after the delete
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  ss = ss.setActiveSheet(ss.getSheetByName(hidden_sheets[0])).showSheet();
};

function delete_all_created_sheets() {

  for (var i = 0; i < mons.length; i++) {
    delete_individual_sheet(mons[i])
  }

  // Delete bank sheets
  for (var i = 0; i < banks.length; i++) {
    delete_individual_sheet('BankStatement - ' + banks[i])
  }

  delete_individual_sheet('CCstatement')

  // UnHide template after the delete
  for (var i = 0; i < hidden_sheets.length; i++) {
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    ss = ss.setActiveSheet(ss.getSheetByName(hidden_sheets[i])).showSheet();
  }

};
