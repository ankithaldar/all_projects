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
  var today = mns_get_date_today();

  for (var i = today.getMonth() + (!mns_check_if_jan_sheet_exists()) ? 0 : 1; i < mons.length; i++) {
    duplicate_individual_sheet(hidden_sheets[0], mons[i])

    mns_set_formulaes_for_the_month(mons[i]);

    if (i > 0) {
      mns_last_month_cash_at_hand(mons[i], mons[i - 1]);
      mns_mark_currency_notes(mons[i], mons[i - 1]);
      mns_mark_last_month_app_carry_overs(mons[i], mons[i - 1]);
      mns_set_first_date_of_month(mons[i], mons[i - 1], days[i - 1]);
    } else if (i == 0) {
      mns_set_first_date_of_jan(mons[i]);
    }

    mns_clear_last_rows_of_dates_from_each_sheet(mons[i], days[i])
  }

  hide_sheets(hidden_sheets);
  mns_mark_bank_statement_first_day();
  mns_mark_cc_statement_first_day();
  mns_mark_imp_events_first_day();
  mns_move_required_sheets();

  mns_mark_from_last_year();

};

// -----------------------------------------------------------------------------

function get_sheet(sheet) {
  return SpreadsheetApp.getActiveSpreadsheet().getSheetByName(sheet);
}

function mns_get_date_today() {
  return new Date();
};

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
  sheet.getRange("J4").setFormula("=ROUND(J3/IFS(AND(TODAY()>=A2, TODAY()<=EOMONTH(A2,0)), DAY(TODAY()), TODAY()>=EOMONTH(A2,0), DAY(EOMONTH(A2,0)), TRUE, 1), 2)");
  sheet.getRange("J25").setFormula("=ROUND(J18/IFS(AND(TODAY()>=A2, TODAY()<=EOMONTH(A2,0)), DAY(EOMONTH(A2, 0)) - DAY(TODAY()) + 1, TODAY()>=EOMONTH(A2,0), DAY(EOMONTH(A2, 0)), TRUE, 1), 2)");
  sheet.getRange("J26").setFormula("=ROUND(J18/IFS(AND(TODAY()>=A2, TODAY()<=EOMONTH(A2,0)), DAY(EOMONTH(A2, 0)) - DAY(TODAY() + K26), TODAY()>=EOMONTH(A2,0), DAY(EOMONTH(A2, 0)), TRUE, 1), 2)");

};

function mns_add_bank_sums_for_cash_online(sheet) {
  for (var j = 2; j <= 3; j++) {
    var formula_withdraw = "";
    var formula_deposit = "";

    for (var k = 0; k < banks.length; k++) {
      formula_withdraw = formula_withdraw + "SUMIFS('BankStatement - " + banks[k] + "'!$F:$F, 'BankStatement - " + banks[k] + "'!$B:$B, TEXT($A$2, \"MMMM\"), 'BankStatement - " + banks[k] + "'!$C:$C, $L" + j + ")" + (k < (banks.length - 1) ? " + " : "");
      formula_deposit = formula_deposit + "SUMIFS('BankStatement - " + banks[k] + "'!$G:$G, 'BankStatement - " + banks[k] + "'!$B:$B, TEXT($A$2, \"MMMM\"), 'BankStatement - " + banks[k] + "'!$C:$C, $L" + j + ")" + (k < (banks.length - 1) ? " + " : "");
    }
    sheet.getRange("M" + j).setFormula("=(" + formula_withdraw + ") - (" + formula_deposit + ")");
  }
};

function mns_update_cc_formulae(sheet) {
  for (var j = 4; j < Object.keys(card_map).length + 4; j++) {
    sheet.getRange("M" + j).setFormula("=SUMIFS('CCStatement'!$F:$F, 'CCStatement'!$C:$C, $L" + j + ", 'CCStatement'!$B:$B, TEXT($A$2, \"MMM-YY\"))")

    sheet.getRange("M" + (j + 24)).setFormula("=ROUND(SUMIFS('CCStatement'!$F:$F, 'CCStatement'!$C:$C, $L" + (j + 24) + ", 'CCStatement'!$A:$A, \"<\"& DATE(YEAR($A$2), MONTH($A$2), $N" + (j + 24) + ")) - SUMIFS('CCStatement'!$G:$G, 'CCStatement'!$C:$C, $L" + (j + 24) + ", 'CCStatement'!$A:$A, \"<\"& DATE(YEAR($A$2), MONTH($A$2), $N" + (j + 24) + ")), 2)")
  }

}

function mns_last_month_cash_at_hand(sheet, prev_sheet) {
  var ss = get_sheet(sheet);
  ss.getRange("M18").setFormula("=" + prev_sheet + "!M20");
};

function mns_mark_currency_notes(sheet, prev_sheet) {
  var ss = get_sheet(sheet);
  for (var j = 2; j < 15; j++) {
    ss.getRange("P" + j).setFormula("=" + prev_sheet + "!P" + j);
  }

  // Mark Credit Cards
  for (var j = 4; j <= 11; j++) {
    ss.getRange("L" + j).setFormula("=" + prev_sheet + "!L" + j);
  }
};

function mns_mark_last_month_app_carry_overs(sheet, prev_sheet) {
  var ss = get_sheet(sheet);
  for (var j = 2; j < 6; j++) {
    ss.getRange("L" + (j + 10)).setFormula("=" + prev_sheet + "!L" + (j + 10));
    ss.getRange("S" + j).setFormula("=L" + (j + 10));
    ss.getRange("T" + j).setFormula("=" + prev_sheet + "!N" + (j + 10));
  }
}

function mns_set_first_date_of_month(sheet, prev_sheet, days_in_last_mon) {
  var ss = get_sheet(sheet);
  ss.getRange("A2").setFormula("=" + prev_sheet + "!A" + (days_in_last_mon + 1) + " + 1");
};

function mns_set_first_date_of_jan(sheet) {
  var ss = get_sheet(sheet);
  ss.getRange("A2").setValue("1/1/" + years);
}

function mns_clear_last_rows_of_dates_from_each_sheet(sheet, days_in_this_mon) {
  var ss = get_sheet(sheet);
  if ((days_in_this_mon + 2) <= 32) {
    ss.getRange("A" + (days_in_this_mon + 2) + ":A32").clearContent();
  }
  // Add Maid Salary in Month Sheets
  add_pay_salary = {
    "Maid Salary": maid_salary,
    "Car Washing Salary": car_cleaning
  };

  var i = 1;

  for (var keys in add_pay_salary) {
    ss.getRange("B" + (days_in_this_mon + i)).setValue(keys);
    ss.getRange("C" + (days_in_this_mon + i)).setValue(1);
    ss.getRange("D" + (days_in_this_mon + i)).setValue(add_pay_salary[keys]);
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
}

function mns_mark_from_last_year() {
  // fill continuation of bank statement
  for (var i = 0; i < banks.length; i++) {
    var ss = get_sheet(hidden_sheets[1] + ' - ' + banks[i]);
    ss.getRange("O1").setFormula('=IMPORTRANGE("' + old_sheet_link + '", "' + hidden_sheets[1] + ' - ' + banks[i] + '!P1")');
  }

  // fill Jan Last Month Cash in hand
  var ss = get_sheet("Jan");
  ss.getRange("M18").setFormula('=IMPORTRANGE("' + old_sheet_link + '", "Dec!M20")');


  for (i = 2; i <= 14; i++) {
    var ss = get_sheet("Jan");
    ss.getRange('Jan!P' + i).setFormula('=IMPORTRANGE("' + old_sheet_link + '", "Dec!P' + i + '")');
  }

};

function mns_move_required_sheets() {

  var ss = SpreadsheetApp.getActiveSpreadsheet();
  ss.setActiveSheet(ss.getSheetByName("TiT_MoM"));
  ss.moveActiveSheet(ss.getNumSheets());

  ss.setActiveSheet(ss.getSheetByName("ImpEvents"));
  ss.moveActiveSheet(ss.getNumSheets());
}

// -----------------------------------------------------------------------------
