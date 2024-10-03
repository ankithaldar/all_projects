// globals

let
  bank_rows_added = 0,
  cc_rows_added = 0
  ;

// -----------------------------------------------------------------------------
function mark_various_days(days) {
  var date_object = mvd_flag_dates(mvd_create_date_arrays(), days_in_mons);

  //mark dates in the calendar
  mvd_mark_dates_in_calendar(date_object);

};

// -----------------------------------------------------------------------------
function mvd_create_date_arrays() {
  let
    date_object = new Object(),
    current_date = new Date(parseInt(years), 0, 1),
    end_date = new Date(parseInt(years) + 1, 0, 0);

  while (current_date <= end_date) {
    date_object[current_date] = new Array();
    current_date.setDate(current_date.getDate() + 1);
  }

  return date_object

}

function mvd_add_days(date, days) {
  var result = new Date(date);
  result.setDate(result.getDate() + days);
  return result;
}

function get_last_row(sheet) {
  return sheet.getLastRow();
};

function mvd_flag_dates(date_object, days) {
  for (var i = 0; i < days.length; i++) {

    let
      last_working_day = new Date(years, i + 1, 0),
      last_month_day = new Date(years, i + 1, 0),
      home_emi_day = new Date(years, i, 2),
      maintanance_payment_day = new Date(years, i, 9),
      internet_payment_day = new Date(years, i, 14),
      electricity_payment_day = new Date(years, i, 15)
      ;


    // Get last working day of month for Monthly Salary
    if (last_working_day.getDay() === 0) {
      last_working_day = new Date(years, i, last_working_day.getDate() - 2)
    } else if (last_working_day.getDay() === 6) {
      last_working_day = new Date(years, i, last_working_day.getDate() - 1)
    }

    // Set Salary Date
    date_object[last_working_day.toString()].push('Salary - ' + last_working_day.toLocaleString('en-EN', { month: 'long' }));

    // Maid Salaries
    date_object[last_month_day.toString()].push('Maid Monthly - ' + last_month_day.toLocaleString('en-EN', { month: 'long' }));
    date_object[last_month_day.toString()].push('Car Washing Monthly - ' + last_month_day.toLocaleString('en-EN', { month: 'long' }));

    // Home EMI
    date_object[home_emi_day.toString()].push('Home EMI 1 - Bajaj Housing Finance Ltd.');

    // Maintenance Payment
    date_object[maintanance_payment_day.toString()].push('SLS Springs Maintinance + Water Charges - MyGate - ' + maintanance_payment_day.toLocaleString('en-EN', { month: 'long' }));

    // Internet Bill Payment
    date_object[internet_payment_day.toString()].push('Internet Bill Payment - ' + internet_payment_day.toLocaleString('en-EN', { month: 'long' }));

    // Electricity Bill Payment
    date_object[electricity_payment_day.toString()].push('Electricity Bill Payment - ' + electricity_payment_day.toLocaleString('en-EN', { month: 'long' }));

    for (let card_nums of Object.keys(card_map)) {
      let
        repay_date_handler = {
          get: function (target, name) {
            return target.hasOwnProperty(name) ? target[name] : 20;
          }
        },
        bill_date = new Date(years, i, card_map[card_nums]['bill_date']),
        p = new Proxy(card_map[card_nums], repay_date_handler);

      if (mvd_add_days(bill_date, p.repayment_days - 1).toString() in date_object) {
        date_object[mvd_add_days(bill_date, p.repayment_days - 1).toString()].push(card_nums + ' - Repayment - ' + bill_date.toLocaleString('en-EN', { month: 'long' }));
      }
    }
  }

  let date_req_object = new Object();
  for (var keys in date_object) {
    if (date_object[keys].length != 0) {
      date_req_object[keys] = date_object[keys]
    }
  }
  return date_req_object;
}

function mvd_mark_dates_in_calendar(date_object) {
  var
    ss_b = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("BankStatement - " + salary_bank),
    ss_cc = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("CCStatement"),

    bank_sheet_rows = new Object(),
    cc_sheet_rows = new Object(),
    year_start_date = new Date(years, 0, 1)
    ;


  for (var keys in date_object) {
    let date_diff = Math.floor(Math.abs(new Date(keys) - year_start_date) / (1000 * 60 * 60 * 24));
    bank_sheet_rows[keys] = date_diff + 2;
    cc_sheet_rows[keys] = date_diff + 33
  }

  for (var keys in date_object) {
    for (var i = 0; i < date_object[keys].length; i++) {
      updated_in = mvd_mark_date(date_object[keys][i], bank_sheet_rows[keys] + bank_rows_added, cc_sheet_rows[keys] + cc_rows_added, ss_b, ss_cc)
    }
  }
};

function mvd_mark_date(key_name, bank_row, cc_row, sheet_bank, sheet_cc) {
  if (key_name.includes('Salary')) {
    // Fill this in particular day of Bank sheet
    mvd_fill_bank_sheet('', key_name, 'd', salary_amount, bank_row, sheet_bank)
    return 'bank'
  }

  if (key_name.includes('Maid Monthly')) {
    // Fill this in particular day of Bank sheet
    mvd_fill_bank_sheet('Online', key_name, 'w', maid_salary, bank_row, sheet_bank)
    return 'bank'
  }

  if (key_name.includes('Car Washing Monthly')) {
    // Fill this in particular day of Bank sheet
    mvd_fill_bank_sheet('Online', key_name, 'w', car_cleaning, bank_row, sheet_bank)
    return 'bank'
  }

  if (key_name.includes('EMI')) {
    // Fill this in particular day of Bank sheet
    mvd_fill_bank_sheet('', key_name, 'w', home_emi, bank_row, sheet_bank)
    return 'bank'
  }

  if (key_name.includes('Maintinance')) {
    // Fill this in particular day of Bank sheet
    month_name = key_name.split(' - ')[2].substring(0, 3)
    mvd_fill_bank_sheet('Online', key_name, 'w', '=' + month_name + '!$F$10', bank_row, sheet_bank)
    return 'bank'
  }

  if (key_name.includes('Electricity')) {
    // Fill this in particular day of Bank sheet
    month_name = key_name.split(' - ')[1].substring(0, 3)
    mvd_fill_bank_sheet('Online', key_name, 'w', '=' + month_name + '!$F$16', bank_row, sheet_bank)
    return 'bank'
  }

  if (key_name.includes('Internet')) {
    // Fill this in particular day of CC sheet
    cc_card_num = 'CC One 0531'
    month_name = key_name.split(' - ')[1].substring(0, 3)
    mvd_fill_cc_sheet(cc_card_num, key_name, 'd', '=' + month_name + '!$F$15', cc_row, sheet_cc)
    return 'cc'
  }

  if (key_name.includes('CC')) {
    // Fill this in particular day of CC sheet
    cc_card_num = key_name.split(' - ')[0]
    month_name = key_name.split(' - ')[2].substring(0, 3)
    total_formulae = "=MAX(INT(" + month_name + "!" + card_map[cc_card_num]['sheet_cell'] + "), 0)"
    mvd_fill_cc_sheet(cc_card_num, key_name, 'c', total_formulae, cc_row, sheet_cc)
    mvd_fill_bank_sheet('', key_name, 'w', total_formulae, bank_row, sheet_bank)
    return 'both'
  }

}


function mvd_fill_bank_sheet(particulars, reason, txn_type, amount, row_num, sheet) {
  if (mvd_check_bank_sheet_row_empty(row_num, sheet)) {
    sheet.getRange(row_num, 3).setValue(particulars);
    sheet.getRange(row_num, 5).setValue(reason);
    if (txn_type == 'w') {
      if (typeof amount === 'number') {
        sheet.getRange(row_num, 6).setValue(amount);
      }
      if (typeof amount === 'string') {
        sheet.getRange(row_num, 6).setFormula(amount);
      }
    } else if (txn_type == 'd') {
      sheet.getRange(row_num, 7).setValue(amount);
    }
  } else {
    row_num = mvd_add_rows_to_bank_sheet(sheet, row_num);
    mvd_fill_bank_sheet(particulars, reason, txn_type, amount, row_num, sheet)
  }
}


function mvd_fill_cc_sheet(cc_card_num, reason, txn_type, amount, row_num, sheet) {
  if (mvd_check_cc_sheet_row_empty(row_num, sheet)) {
    sheet.getRange(row_num, 3).setValue(cc_card_num);
    sheet.getRange(row_num, 5).setValue(reason);
    if (txn_type == 'c') {
      if (typeof amount === 'string') {
        sheet.getRange(row_num, 7).setFormula(amount);
      }
    }
    if (txn_type == 'd') {
      if (typeof amount === 'string') {
        sheet.getRange(row_num, 6).setFormula(amount);
      }
    }
  } else {
    row_num = mvd_add_rows_to_cc_sheet(sheet, row_num);
    mvd_fill_cc_sheet(cc_card_num, reason, txn_type, amount, row_num, sheet)
  }
}


function mvd_check_bank_sheet_row_empty(bank_row, sheet_bank) {
  if (sheet_bank.getRange(bank_row, 3).isBlank() && sheet_bank.getRange(bank_row, 5).isBlank() && sheet_bank.getRange(bank_row, 6).isBlank() && sheet_bank.getRange(bank_row, 7).isBlank()) {
    return true;
  } else {
    return false;
  }
};

function mvd_check_cc_sheet_row_empty(cc_row, sheet_cc) {
  if (sheet_cc.getRange(cc_row, 3).isBlank() && sheet_cc.getRange(cc_row, 5).isBlank() && sheet_cc.getRange(cc_row, 6).isBlank() && sheet_cc.getRange(cc_row, 7).isBlank()) {
    return true;
  } else {
    return false;
  }
};


function mvd_add_rows_to_bank_sheet(sheet, row_num) {
  sheet.insertRowAfter(row_num); row_num++;

  bank_rows_added++;

  sheet.getRange("A" + row_num).setFormula("=A" + (row_num - 1));
  sheet.getRange("B" + row_num).setFormula("=TEXT(A" + row_num + ", \"mmmm\")");
  sheet.getRange("H" + row_num).setFormula("=H" + (row_num - 1) + "-F" + row_num + "+G" + row_num);
  sheet.getRange("M" + row_num).setFormula("=IF(AND($A" + row_num + " < TODAY(), ISBLANK($D" + row_num + ")), 1, 0)");

  return row_num;
};

function mvd_add_rows_to_cc_sheet(sheet, row_num) {
  sheet.insertRowAfter(row_num); row_num++;

  cc_rows_added++;

  sheet.getRange("A" + row_num).setFormula("=A" + (row_num - 1));
  sheet.getRange("B" + row_num).setFormula("=TEXT(A" + row_num + ", \"MMM-YY\")");
  sheet.getRange("L" + row_num).setFormula("=IF(AND($A" + row_num + " < TODAY(), ISBLANK($D" + row_num + ")), 1, 0)");

  return row_num;
};
