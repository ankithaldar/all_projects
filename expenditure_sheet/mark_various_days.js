function mark_various_days(days) {
  var
    days = leap_year_check_and_manipulate(years, days_in_mons),
    date_object = mvd_flag_dates(mvd_create_date_arrays(), days)
    ;

  //mark dates in the calendar
  mvd_mark_dates_in_calendar(date_object);

};

// -----------------------------------------------------------------------------
function mvd_create_date_arrays() {
  // create array for each of the date types
  var date_object = {
    'maid_salary': new Array(),
    'car_salary': new Array(),
    'home_emi': new Array(),
    'empty_array': new Array()
  }

  for (let key of Object.keys(card_map)) {
    date_object[key] = new Array();
  }

  return date_object;
};

function mvd_flag_dates(date_object, days) {

  for (var i = 0; i < days.length; i++) {

    var dt = new Date(years, i + 1, 0);

    for (let key of Object.keys(date_object)) {

      if (key === 'maid_salary') {
        if (dt.getDay() === 0) {
          var new_dt = new Date(years, i, dt.getDate() - 2);
          date_object[key].push(new_dt);
        } else if (dt.getDay() === 6) {
          var new_dt = new Date(years, i, dt.getDate() - 1);
          date_object[key].push(new_dt);
        } else {
          date_object[key].push(dt);
        }
      } else if (key === 'car_salary') {
        date_object[key].push(dt);
      } else if (key === 'home_emi') {
        var new_dt = new Date(years, i, 5);
        date_object[key].push(new_dt);
      } else if (key === 'empty_array') {
        // need empty array to handle last day houserent orruring twice in line 85 this script
        // pass
      } else {
        var
          bill_date = new Date(years, i, card_map[key]['bill_date']),
          payment_date = new Date(bill_date.setDate(bill_date.getDate() + 20 - 1));

        date_object[key].push(payment_date);
      }
    }
  }

  return date_object;
};

function mvd_mark_dates_in_calendar(date_object) {
  var
    ss_b = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("BankStatement - " + salary_bank),
    ss_cc = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("CCStatement"),
    row_cc = 33,
    row_b = 2,
    month = 0,
    i = 0
    ;

  var bank_sheet_last_row = get_last_row(ss_b);

  while (row_b <= bank_sheet_last_row) {
    var sheet_dt = new Date(ss_b.getRange("A" + row_b).getValue()).toString();

    for (let key of Object.keys(date_object)) {
      var arr_dt = new Date(date_object[key][0]).toString();

      if (sheet_dt === arr_dt) {
        // solution hardcoded here
        rows = mvd_mark_date(key, row_b, row_cc, ss_b, ss_cc, month)
        row_b = rows[0];
        row_cc = rows[1];
        date_object[key].shift();
        i++;
        if (i % Object.keys(date_object).length === 0) {
          month++;
        }
      }
    }
    row_b++;
    row_cc++;
    bank_sheet_last_row = get_last_row(ss_b);
  }
};

function get_last_row(sheet) {
  return sheet.getLastRow();
};

function mvd_mark_date(key_name, bank_row, cc_row, sheet_bank, sheet_cc, month) {
  // check key
  if (key_name == 'maid_salary') {
    bank_row = fill_bank_sheet('Salary', salary_amount, 0, "", bank_row, sheet_bank);
    bank_row = fill_bank_sheet('Maid Salaries', "", maid_salary, "Cash In [Bank]", bank_row, sheet_bank);

  } else if (key_name == 'car_salary') {
    bank_row = fill_bank_sheet('Car Cleaning', "", car_cleaning, "Online", bank_row, sheet_bank);

  } else if (key_name == 'home_emi') {
    bank_row = fill_bank_sheet('House EMI', "", home_emi, "Online", bank_row, sheet_bank);

  } else if (key_name == 'empty_array') {
    // need empty array to handle last day houserent orruring twice in line 85 this script
    // pass
  } else {
    cc_row = fill_cc_sheet(key_name, cc_row, sheet_cc, mons[month]);
    bank_row = fill_bank_sheet(key_name + " - Repayment", "", "", "", bank_row, sheet_bank, cc_row);
  }

  // return last filled row to avoid overlaps
  return [bank_row, cc_row];
};

function fill_bank_sheet(reason, deposit, withdraw, particulars, bank_row, sheet_bank, refer_row_cc) {
  if (check_bank_sheet_row_empty(bank_row, sheet_bank)) {
    sheet_bank.getRange(bank_row, 3).setValue(particulars);
    sheet_bank.getRange(bank_row, 5).setValue(reason);
    if (typeof withdraw === 'number') {
      sheet_bank.getRange(bank_row, 6).setValue(withdraw);
    } else if (typeof withdraw === 'string') {
      // setting withdraw for credit card only
      sheet_bank.getRange(bank_row, 6).setFormula("=IF(INT(CCStatement!$G$" + refer_row_cc + ") >= 0, INT(CCStatement!$G$" + refer_row_cc + "), 0)");
    }
    sheet_bank.getRange(bank_row, 7).setValue(deposit);
  } else {
    bank_row = add_rows_to_bank_sheet(sheet_bank, bank_row);
    bank_row = fill_bank_sheet(reason, deposit, withdraw, particulars, bank_row, sheet_bank, refer_row_cc);
  }
  return bank_row;
};

function fill_cc_sheet(card_num, cc_row, sheet_cc, month_name) {
  if (check_cc_sheet_row_empty(cc_row, sheet_cc)) {
    sheet_cc.getRange(cc_row, 3).setValue(card_num);
    sheet_cc.getRange(cc_row, 5).setValue(card_num + " - Repayment");
    sheet_cc.getRange(cc_row, 7).setFormula("=INT(" + month_name + "!" + card_map[card_num]['sheet_cell'] + ")");
  } else {
    cc_row = add_rows_to_cc_sheet(sheet_cc, cc_row);
    cc_row = fill_cc_sheet(card_num, cc_row, sheet_cc, month_name);
  }
  return cc_row;
};

function check_bank_sheet_row_empty(bank_row, sheet_bank) {
  if (sheet_bank.getRange(bank_row, 3).isBlank() && sheet_bank.getRange(bank_row, 5).isBlank() && sheet_bank.getRange(bank_row, 6).isBlank() && sheet_bank.getRange(bank_row, 7).isBlank()) {
    return true;
  } else {
    return false;
  }
};

function check_cc_sheet_row_empty(cc_row, sheet_cc) {
  if (sheet_cc.getRange(cc_row, 3).isBlank() && sheet_cc.getRange(cc_row, 5).isBlank() && sheet_cc.getRange(cc_row, 6).isBlank() && sheet_cc.getRange(cc_row, 7).isBlank()) {
    return true;
  } else {
    return false;
  }
};


function add_rows_to_bank_sheet(sheet_bank, bank_row) {
  sheet_bank.insertRowAfter(bank_row); bank_row++;

  sheet_bank.getRange("A" + bank_row).setFormula("=A" + (bank_row - 1));
  sheet_bank.getRange("B" + bank_row).setFormula("=TEXT(A" + bank_row + ", \"mmmm\")");
  sheet_bank.getRange("H" + bank_row).setFormula("=H" + (bank_row - 1) + "-F" + bank_row + "+G" + bank_row);
  sheet_bank.getRange("M" + bank_row).setFormula("=IF(AND($A" + bank_row + " < TODAY(), ISBLANK($D" + bank_row + ")), 1, 0)");

  return bank_row;
};

function add_rows_to_cc_sheet(sheet_cc, cc_row) {
  sheet_cc.insertRowAfter(cc_row); cc_row++;

  sheet_cc.getRange("A" + cc_row).setFormula("=A" + (cc_row - 1));
  sheet_cc.getRange("B" + cc_row).setFormula("=TEXT(A" + cc_row + ", \"MMM-YY\")");
  sheet_cc.getRange("L" + cc_row).setFormula("=IF(AND($A" + cc_row + " < TODAY(), ISBLANK($D" + cc_row + ")), 1, 0)");

  return cc_row;
};
// -----------------------------------------------------------------------------
