function create_bank_statement() {
  // Duplicating bank sheets
  for (var i = 0; i < banks.length; i++) {
    duplicate_individual_sheet(hidden_sheets[1], hidden_sheets[1] + ' - ' + banks[i])
  }
  // Hide the bank statement sheet
  // hide_sheets(hidden_sheets[1]);
};

function create_cc_statement() {
  duplicate_individual_sheet(hidden_sheets[2], 'CCStatement');
};

// -----------------------------------------------------------------------------


function create_month_templates(days) {
  // get today date
  var today = get_date_today();

  for (var i = today.getMonth() + (!mns_check_if_jan_sheet_exists()) ? 0 : 1; i < mons.length; i++) {
    duplicate_individual_sheet(hidden_sheets[0], mons[i])

    mns_set_formulaes_for_the_month(mons[i]);

    if (i > 0) {
      mns_last_month_cash_at_hand(mons[i], mons[i - 1]);
      mns_mark_currency_notes(mons[i], mons[i - 1]);
      mns_mark_credit_cards(mons[i], mons[i - 1]);
      mns_mark_last_month_wallet_carry_overs(mons[i], mons[i - 1]);
      mns_set_first_date_of_month(mons[i], mons[i - 1], days[i - 1]);
    } else if (i == 0) {
      mns_set_first_date_of_jan(mons[i]);
    }

    mns_clear_last_rows_of_dates_from_each_sheet(mons[i], days[i])
  }

  hide_sheets(hidden_sheets);
  // mns_mark_bank_statement_first_day();
  // mns_mark_cc_statement_first_day();
  // mns_mark_imp_events_first_day();
  mns_move_required_sheets();

  mns_mark_from_last_year();

};

// -----------------------------------------------------------------------------

function mns_check_if_jan_sheet_exists() {
  var
    ss = SpreadsheetApp.getActiveSpreadsheet(),
    itt = ss.getSheetByName(mons[0])
    ;

  return itt
};

function mns_set_formulaes_for_the_month(sheet) {
  var ss = get_sheet(sheet);

  mns_define_average_expenditure_over_the_month(ss);
  mns_add_bank_sums_for_cash_online(ss);
  mns_update_cc_formulae(ss);

};

function mns_define_average_expenditure_over_the_month(sheet) {
  // Committed AverageExp
  var total_cell = "K3"; // cell from which the formulae is to be calculated
  var place_cell = "K4"; // cell in which the formulae is to be set
  sheet.getRange(place_cell).setFormula("=ROUND(" + total_cell + "/IFS(AND(TODAY()>=A2, TODAY()<=EOMONTH(A2,0)), DAY(TODAY()), TODAY()>=EOMONTH(A2,0), DAY(EOMONTH(A2,0)), TRUE, 1), 2)");

  // Remaining AvgExp / day
  var total_cell = "K8"; // cell from which the formulae is to be calculated
  var place_cell = "K10"; // cell in which the formulae is to be set
  sheet.getRange(place_cell).setFormula("=ROUND(" + total_cell +"/IFS(AND(TODAY()>=A2, TODAY()<=EOMONTH(A2,0)), DAY(EOMONTH(A2, 0)) - DAY(TODAY()) + 1, TODAY()>=EOMONTH(A2,0), DAY(EOMONTH(A2, 0)), TRUE, 1), 2)");

  // My Expenditure Per Day
  var total_cell = "K12"; // cell from which the formulae is to be calculated
  var place_cell = "K13"; // cell in which the formulae is to be set
  sheet.getRange(place_cell).setFormula("=ROUND(" + total_cell + "/IFS(AND(TODAY()>=A2, TODAY()<=EOMONTH(A2,0)), DAY(EOMONTH(A2, 0)) - DAY(TODAY()) + 1, TODAY()>=EOMONTH(A2,0), DAY(EOMONTH(A2, 0)), TRUE, 1), 2)");
};

function mns_add_bank_sums_for_cash_online(sheet) {
  // j=9 since [Bank Transactions] start from row 9
  var start_row_num = 9;
  var end_row_num = 10;
  var entry_name_column = "R";
  for (var j = start_row_num; j <= end_row_num; j++) {
    var formula_withdraw = "";
    var formula_deposit = "";

    for (var k = 0; k < banks.length; k++) {
      formula_withdraw = formula_withdraw + "SUMIFS('BankStatement - " + banks[k] + "'!$F:$F, 'BankStatement - " + banks[k] + "'!$B:$B, TEXT($A$2, \"MMMM\"), 'BankStatement - " + banks[k] + "'!$C:$C, $" + entry_name_column + j + ")" + (k < (banks.length - 1) ? " + " : "");
      formula_deposit = formula_deposit + "SUMIFS('BankStatement - " + banks[k] + "'!$G:$G, 'BankStatement - " + banks[k] + "'!$B:$B, TEXT($A$2, \"MMMM\"), 'BankStatement - " + banks[k] + "'!$C:$C, $" + entry_name_column + j + ")" + (k < (banks.length - 1) ? " + " : "");
    }
    sheet.getRange("S" + j).setFormula("=(" + formula_withdraw + ") - (" + formula_deposit + ")");
  }
};

function mns_update_cc_formulae(sheet) {
  // addded 1 to for loop for meal card tracking
  var start_row_num = 25;
  var end_row_num = Object.keys(card_map).length + start_row_num + 1;

  var card_name_column = "R";
  var card_bill_date_column = "U";

  for (var j = start_row_num; j < end_row_num; j++) {

    sheet.getRange("S" + j).setFormula("=SUMIFS('CCStatement'!$F:$F, 'CCStatement'!$C:$C, $" + card_name_column + j + ", 'CCStatement'!$B:$B, TEXT($A$2, \"MMM-YY\"))")


    if (j < end_row_num - 1) {
      // Mapped to analytics sheet
      // in analytics sheet the dates are in order and start from row 4
      sheet.getRange(card_bill_date_column + j).setFormula("='Analytics'!G" + (j-21))
    } else {
      sheet.getRange(card_bill_date_column + j).setFormula("=DAY(EOMONTH($A$2, 0))")
    }

    sheet.getRange("T" + j).setFormula("=ROUND(SUMIFS('CCStatement'!$F:$F, 'CCStatement'!$C:$C, $" + card_name_column + j + ", 'CCStatement'!$A:$A, \"<\"& DATE(YEAR($A$2), MONTH($A$2), $" + card_bill_date_column + j + ")) - SUMIFS('CCStatement'!$G:$G, 'CCStatement'!$C:$C, $" + card_name_column + j + ", 'CCStatement'!$A:$A, \"<\"& DATE(YEAR($A$2), MONTH($A$2), $" + card_bill_date_column + j + ")), 2)")
  }

};

function mns_last_month_cash_at_hand(sheet, prev_sheet) {
  var ss = get_sheet(sheet);

  ss.getRange("S6").setFormula("=" + prev_sheet + "!S2");
};

function mns_mark_currency_notes(sheet, prev_sheet) {
  var ss = get_sheet(sheet);

  var start_row_num = 4;
  var end_row_num = 15;
  var place_cell="X";

  for (var j = start_row_num; j <= end_row_num; j++) {
    ss.getRange(place_cell + j).setFormula("=" + prev_sheet + "!" + place_cell + j);
  }
};

function mns_mark_credit_cards(sheet, prev_sheet){
  // Mark Credit Cards
  var ss = get_sheet(sheet);

  var start_row_num = 25;
  var end_row_num = Object.keys(card_map).length + start_row_num + 1;
  var place_cell = "R";

  for (var j = start_row_num; j <= end_row_num; j++) {
    ss.getRange(place_cell + j).setFormula("=" + prev_sheet + "!" + place_cell + j);
  }
};

function mns_mark_last_month_wallet_carry_overs(sheet, prev_sheet) {
  var ss = get_sheet(sheet);

  var start_row_num = 13;
  var end_row_num = start_row_num + 10;
  var name_cell = "R";
  var balance_cell = "U";

  for (var j = start_row_num; j < end_row_num; j++) {
    ss.getRange(name_cell + j).setFormula("=" + prev_sheet + "!" + name_cell + j);
    ss.getRange(balance_cell + j).setFormula("=" + prev_sheet + "!T" + j);

    // ss.getRange("S" + j).setFormula("=L" + (j + 10));
    // ss.getRange("T" + j).setFormula("=" + prev_sheet + "!N" + (j + 10));
  }
};

function mns_set_first_date_of_month(sheet, prev_sheet, days_in_last_mon) {
  var ss = get_sheet(sheet);
  ss.getRange("A2").setFormula("=" + prev_sheet + "!A" + (days_in_last_mon + 1) + " + 1");
};

function mns_set_first_date_of_jan(sheet) {
  var ss = get_sheet(sheet);
  ss.getRange("A2").setValue("1/1/" + years);
};

function mns_clear_last_rows_of_dates_from_each_sheet(sheet, days_in_this_mon) {
  var ss = get_sheet(sheet);
  if ((days_in_this_mon + 2) <= 32) {
    ss.getRange("A" + (days_in_this_mon + 2) + ":A32").clearContent();
  }
  // Add Maid Salary in Month Sheets
  add_pay_salary = {
    "Maid Monthly": maid_salary,
    "Car Washing Monthly": car_cleaning
  };

  var i = 1;

  for (var keys in add_pay_salary) {
    ss.getRange("B" + (days_in_this_mon + i)).setValue(keys);
    ss.getRange("C" + (days_in_this_mon + i)).setValue(1);
    ss.getRange("D" + (days_in_this_mon + i)).setValue(add_pay_salary[keys]);
    ss.getRange("H" + (days_in_this_mon + i)).setValue("Home");
    i++;
  }
};

function mns_mark_bank_statement_first_day() {
  for (var i = 0; i < banks.length; i++) {
    var ss = get_sheet(hidden_sheets[1] + " - " + banks[i]);
    ss.getRange("A2").setFormula("=Jan!A2")
  }
};

function mns_mark_cc_statement_first_day() {
  var ss = get_sheet("CCStatement");
  ss.getRange("A2").setFormula("=Jan!A2 - 31");
};

function mns_mark_imp_events_first_day() {
  var ss = get_sheet("ImpEvents");
  ss.getRange("A2").setFormula("=Jan!A2");
};

function mns_mark_from_last_year() {
  // fill continuation of bank statement
  for (var i = 0; i < banks.length; i++) {
    var ss = get_sheet(hidden_sheets[1] + ' - ' + banks[i]);
    ss.getRange("O1").setFormula('=IMPORTRANGE("' + old_sheet_link + '", "' + hidden_sheets[1] + ' - ' + banks[i] + '!P1")');
  }

  // fill Jan Last Month Cash in hand
  var ss = get_sheet("Jan");
  // after 2025 comment this line and unconnect the next
  ss.getRange("S6").setFormula('=IMPORTRANGE("' + old_sheet_link + '", "Dec!M20")');
  // ss.getRange("S6").setFormula('=IMPORTRANGE("' + old_sheet_link + '", "Dec!S2")');

  // fill Jan Denomination
  for (i = 4; i <= 15; i++) {
    // after 2025 comment this line and unconnect the next
    ss.getRange('Jan!X' + i).setFormula('=IMPORTRANGE("' + old_sheet_link + '", "Dec!P' + (i - 2) + '")');
    // ss.getRange('Jan!X' + i).setFormula('=IMPORTRANGE("' + old_sheet_link + '", "Dec!X' + i + '")');
  }

  // fill Jan Wallet Balances
  // for (i = 13; i < 13 + 10; i++) {
  //   var ss = get_sheet("Jan");
  //   ss.getRange('Jan!X' + i).setFormula('=IMPORTRANGE("' + old_sheet_link + '", "Dec!X' + i + '")');
  // }

};

function mns_move_required_sheets() {

  var ss = SpreadsheetApp.getActiveSpreadsheet();
  ss.setActiveSheet(ss.getSheetByName("TiT_MoM"));
  ss.moveActiveSheet(ss.getNumSheets());

  ss.setActiveSheet(ss.getSheetByName("ImpEvents"));
  ss.moveActiveSheet(ss.getNumSheets());

  ss.setActiveSheet(ss.getSheetByName("Analytics"));
  ss.moveActiveSheet(ss.getNumSheets());
};

// -----------------------------------------------------------------------------
