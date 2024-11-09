// globals for mark_various_days

let
  bank_rows_added = 0,
  cc_rows_added = 0
;

// -----------------------------------------------------------------------------
function mark_various_days(days) {
  //mark dates in the calendar
  mvd_mark_dates_in_calendar(mvd_flag_dates(days_in_mons));
};

// -----------------------------------------------------------------------------
function mvd_fill_date_object(date_object, key_name, fill_value) {
  // if the date doesnot exist in the object
  if (!(key_name in date_object)) {
    date_object[key_name.toString()] = new Array();
  }
  date_object[key_name.toString()].push(fill_value);

  return date_object;
};


function mvd_sort_object(date_object) {
  const sortedKeys = Object.keys(date_object).map(key => new Date(key)).sort((a, b) => a - b);
  const sortedObj = new Object();

  for (var i = 0; i <= sortedKeys.length; i++) {
    if (new Date(sortedKeys[i]).getFullYear() === new Date(parseInt(years) + 1, 0, 0).getFullYear()) {
      sortedObj[sortedKeys[i].toString()] = date_object[sortedKeys[i].toString()];
    }
  }
  return sortedObj;
};


function mvd_flag_dates(days) {
  let date_object = new Object();

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
      last_working_day = new Date(years, i, last_working_day.getDate() - 2);
    } else if (last_working_day.getDay() === 6) {
      last_working_day = new Date(years, i, last_working_day.getDate() - 1);
    }

    // Set Salary Date
    date_object = mvd_fill_date_object(date_object, last_working_day, 'Salary - ' + last_working_day.toLocaleString('en-EN', { month: 'long' }));

    // Set Meal Card Refill Date
    date_object = mvd_fill_date_object(date_object, last_working_day, 'Pluxee Meal Card Refill - ' + last_working_day.toLocaleString('en-EN', { month: 'long' }));

    // Maid Salaries
    date_object = mvd_fill_date_object(date_object, last_month_day, 'Maid Monthly - ' + last_month_day.toLocaleString('en-EN', { month: 'long' }));
    date_object = mvd_fill_date_object(date_object, last_month_day, 'Car Washing Monthly - ' + last_month_day.toLocaleString('en-EN', { month: 'long' }));

    // Home EMI
    date_object = mvd_fill_date_object(date_object, home_emi_day, 'Home EMI 1 - Bajaj Housing Finance Ltd.');

    // Maintenance Payment
    date_object = mvd_fill_date_object(date_object, maintanance_payment_day, 'SLS Springs Maintinance + Water Charges - MyGate - ' + maintanance_payment_day.toLocaleString('en-EN', { month: 'long' }));

    // Internet Bill Payment
    date_object = mvd_fill_date_object(date_object, internet_payment_day, 'Internet Bill Payment - ' + internet_payment_day.toLocaleString('en-EN', { month: 'long' }));

    // Electricity Bill Payment
    date_object = mvd_fill_date_object(date_object, electricity_payment_day, 'Electricity Bill Payment - ' + electricity_payment_day.toLocaleString('en-EN', { month: 'long' }));

    for (let card_nums of Object.keys(card_map)) {
      let
        repay_date_handler = {
          get: function (target, name) {
            return target.hasOwnProperty(name) ? target[name] : 20;
          }
        },
        bill_date = new Date(years, i, card_map[card_nums]['bill_date']),
        p = new Proxy(card_map[card_nums], repay_date_handler);
      ;

      date_object = mvd_fill_date_object(date_object, add_days(bill_date, p.repayment_days - 1), card_nums + ' - Repayment - ' + bill_date.toLocaleString('en-EN', { month: 'long' }));
    }
  }
  return mvd_sort_object(date_object);
};


function mvd_mark_dates_in_calendar(date_object) {
  var
    ss_b = get_sheet("BankStatement - " + salary_bank),
    ss_cc = get_sheet("CCStatement")
    ;


  for (var keys in date_object) {
    let
      date_diff = Math.floor(Math.abs(new Date(keys) - new Date(years, 0, 1)) / (1000 * 60 * 60 * 24)),
      bank_sheet_rows = date_diff + 2,
      cc_sheet_rows = date_diff + 33

    for (var i = 0; i < date_object[keys].length; i++) {
      mvd_mark_date(date_object[keys][i], bank_sheet_rows + bank_rows_added, cc_sheet_rows + cc_rows_added, ss_b, ss_cc)

      // printing each entry
      if (if_debug) {
        console.log(
          "marking: " + date_object[keys][i] + " | " + keys + " | Bank Sheet Date: " + ss_b.getRange("A" + (bank_sheet_rows + bank_rows_added)).getValue() + " | CC Sheet Date: " + ss_cc.getRange("A" + (cc_sheet_rows + cc_rows_added)).getValue()
        )
      }
    }
  }
};


function mvd_mark_date(key_name, bank_row, cc_row, sheet_bank, sheet_cc) {
  if (key_name.includes('Salary')) {
    // Fill this in particular day of Bank sheet
    mvd_fill_bank_sheet('', key_name, 'd', salary_amount, bank_row, sheet_bank)
  }

  if (key_name.includes('Maid Monthly')) {
    // Fill this in particular day of Bank sheet
    mvd_fill_bank_sheet('Online', key_name, 'w', '=' + get_mon_3_char(key_name) + '!$F$' + (last_date_of_month(get_mon_3_char(key_name)) + 1), bank_row, sheet_bank)
  }

  if (key_name.includes('Car Washing Monthly')) {
    // Fill this in particular day of Bank sheet
    mvd_fill_bank_sheet('Online', key_name, 'w', '=' + get_mon_3_char(key_name) + '!$F$' + (last_date_of_month(get_mon_3_char(key_name)) + 2), bank_row, sheet_bank)
  }

  if (key_name.includes('EMI')) {
    // Fill this in particular day of Bank sheet
    mvd_fill_bank_sheet('', key_name, 'w', home_emi, bank_row, sheet_bank)
  }

  if (key_name.includes('Maintinance')) {
    // Fill this in particular day of Bank sheet
    month_name = key_name.split(' - ')[2].substring(0, 3)
    mvd_fill_bank_sheet('Online', key_name, 'w', '=' + month_name + '!$F$10', bank_row, sheet_bank)
  }

  if (key_name.includes('Electricity')) {
    // Fill this in particular day of Bank sheet
    mvd_fill_bank_sheet('Online', key_name, 'w', '=' + get_mon_3_char(key_name) + '!$F$16', bank_row, sheet_bank)
  }

  if (key_name.includes('Internet')) {
    // Fill this in particular day of CC sheet
    cc_card_num = 'CC One 0531'
    mvd_fill_cc_sheet(cc_card_num, key_name, 'd', '=' + get_mon_3_char(key_name) + '!$F$15', cc_row, sheet_cc)
  }

  if (key_name.includes('Pluxee')) {
    // Fill this in particular day of CC sheet
    cc_card_num = 'CC Pluxee 6314'
    mvd_fill_cc_sheet(cc_card_num, key_name, 'c', meal_card_amount, cc_row, sheet_cc)
  }

  if (key_name.includes('CC')) {
    // Fill this in particular day of CC sheet
    cc_card_num = key_name.split(' - ')[0]
    month_name = key_name.split(' - ')[2].substring(0, 3)
    total_formulae = "=MAX(" + month_name + "!" + card_map[cc_card_num]['sheet_cell'] + ", 0)"
    mvd_fill_cc_sheet(cc_card_num, key_name, 'c', total_formulae, cc_row, sheet_cc)
    mvd_fill_bank_sheet('', key_name, 'w', total_formulae, bank_row, sheet_bank)
  }
};


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
};


function mvd_fill_cc_sheet(cc_card_num, reason, txn_type, amount, row_num, sheet) {
  if (mvd_check_cc_sheet_row_empty(row_num, sheet)) {
    sheet.getRange(row_num, 3).setValue(cc_card_num);
    sheet.getRange(row_num, 5).setValue(reason);
    if (txn_type == 'c') {
      if (typeof amount === 'string') {
        sheet.getRange(row_num, 7).setFormula(amount);
      }
      if (typeof amount === 'number') {
        sheet.getRange(row_num, 7).setValue(amount);
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
};


function mvd_check_bank_sheet_row_empty(row_num, sheet) {
  if (sheet.getRange(row_num, 3).isBlank() && sheet.getRange(row_num, 5).isBlank() && sheet.getRange(row_num, 6).isBlank() && sheet.getRange(row_num, 7).isBlank()) {
    return true;
  } else {
    return false;
  }
};

function mvd_check_cc_sheet_row_empty(row_num, sheet) {
  if (sheet.getRange(row_num, 3).isBlank() && sheet.getRange(row_num, 5).isBlank() && sheet.getRange(row_num, 6).isBlank() && sheet.getRange(row_num, 7).isBlank()) {
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
