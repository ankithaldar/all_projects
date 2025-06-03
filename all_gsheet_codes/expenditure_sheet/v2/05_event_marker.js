// ---------------------------------------------------------------------------------------------------------------------
function mark_various_days() {
  var final_tabs = em_flag_dates()
  Object.keys(final_tabs).forEach(type => {
    em_push_to_tab(final_tabs[type], type);
  });
};

function get_last_row(sheet) {
  return sheet.getLastRow();
};

function em_push_to_tab(flat_tabs, type) {
  if (type === 'bank') {
    var tab = `BankStatement - ${CONFIG.salary_bank}`;
    var range = `A2:M${flat_tabs.length + 1}`;
  }
  if (type === 'ccard') {
    var tab = 'CCStatement';
    var range = `A2:L${flat_tabs.length + 1}`
  }

  var tab_cache = CONFIG.sheet_cache.getSheetByName(tab);
  var last_row = get_last_row(tab_cache);
  if (last_row < flat_tabs.length) tab_cache.insertRowsAfter(last_row, flat_tabs.length - last_row + 1)
  tab_cache.getRange(range).setValues(flat_tabs);
}
// ---------------------------------------------------------------------------------------------------------------------
function em_create_all_dates_in_objects() {
  var b_i = 2, c_j = 2;
  var bank_tab = new Object(), cc_tab = new Object();

  for (let d = new Date(`${CONFIG.years - 1}-12-01`); d <= new Date(`${CONFIG.years}-12-31`); d.setDate(d.getDate() + 1)) {
    if (d >= new Date(`${CONFIG.years}-01-01`)) bank_tab[format_date(d)] = new Array();
    cc_tab[format_date(d)] = new Array();
  }
  return {
    'bank_tab': bank_tab,
    'cc_tab': cc_tab
  };
};
function format_date(dates) {
  dates = new Date(dates)
  var yyyy = dates.getFullYear();
  var mm = String(dates.getMonth() + 1).padStart(2, '0');
  var dd = String(dates.getDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
};
function add_days(date, days) {
  var result = new Date(date);
  result.setDate(result.getDate() + days);
  return result;
};
// ---------------------------------------------------------------------------------------------------------------------
function get_last_working_day_for_month(dates) {
  if (dates.getDay() === 0) {
    dates = add_days(dates, - 2);
  } else if (dates.getDay() === 6) {
    dates = add_days(dates, - 1);
  }
  return dates;
};
function amount_bank_cc_formula(reason, current_row, last_digit_round = false) {
  var formula = `SUMIFS(${get_month_abbr(reason)}!$T:$T, ${get_month_abbr(reason)}!$R:$R, TRIM(SPLIT($E${current_row}, "-")))`;
  if (last_digit_round) {
    formula = `ROUND(${formula}, 0)`;
  }
  return `=MAX(${formula}, 0)`;
}
// ---------------------------------------------------------------------------------------------------------------------

function em_flag_dates() {
  var tabs = em_create_all_dates_in_objects();
  var bank_tab = tabs.bank_tab;
  var cc_tab = tabs.cc_tab;

  Object.keys(CONFIG.month_days).forEach((mon, k) => {
    // bank based payments
    bank_tab = em_flag_salary(mon, k, bank_tab);
    bank_tab = em_flag_end_month_salary(mon, k, bank_tab);
    bank_tab = em_flag_home_loan_emi_payment(mon, k, bank_tab);
    bank_tab = em_flag_flat_maintanance_payment(mon, k, bank_tab);
    // bank_tab = em_flag_flat_electricity_payment(mon, k, bank_tab);
    bank_tab = em_flag_personal_loan_adhira_appa(mon, k, bank_tab);
    bank_tab = em_flag_earning_adhira_appa(mon, k, bank_tab);

    // cc based_payments
    cc_tab = em_flag_flat_electricity_payment(mon, k, cc_tab);
    cc_tab = em_flag_internet_payment(mon, k, cc_tab);
    cc_tab = em_flag_meal_card(mon, k, cc_tab);

    // Credit card payments
    tabs = em_flag_cc_card_payments(mon, k, bank_tab, cc_tab);
    bank_tab = tabs.bank_tab, cc_tab = tabs.cc_tab;
  });

  // Credit card payments for December prev Year
  tabs = em_flag_cc_card_payments('Dec', 11, bank_tab, cc_tab, -1);
  bank_tab = tabs.bank_tab, cc_tab = tabs.cc_tab;


  // flatting the tabs
  var bank_flat_tab = flatten_bank_tab_to_array(bank_tab);
  var cc_flat_tab = flatten_cc_tab_to_array(cc_tab);

  return {
    'bank': bank_flat_tab,
    'ccard': cc_flat_tab
  }
};
// ---------------------------------------------------------------------------------------------------------------------

function em_flag_salary(mon, k, bank_tab) {
  var last_working_day = get_last_working_day_for_month(new Date(CONFIG.years, k + 1, 0));
  var events = bank_tab[format_date(last_working_day)];

  events.push(['', `Salary - ${last_working_day.toLocaleString('en-EN', { month: 'long' })}`, 'd', CONFIG.salary_amount]);

  bank_tab[format_date(last_working_day)] = events;

  return bank_tab;

};

function em_flag_end_month_salary(mon, k, bank_tab) {
  var last_month_day = new Date(CONFIG.years, k + 1, 0);
  var events = bank_tab[format_date(last_month_day)], i = 1;

  Object.keys(CONFIG.month_end_salary).forEach(salary => {
    events.push(['Online', `${salary} - ${last_month_day.toLocaleString('en-EN', { month: 'long' })}`, 'w', `=${mon}!$F$${CONFIG.month_days[mon] + i}`]);
    i++;
  });

  bank_tab[format_date(last_month_day)] = events;

  return bank_tab;

};

function em_flag_home_loan_emi_payment(mon, k, bank_tab) {
  var home_emi_day = new Date(CONFIG.years, k, 2);
  var events = bank_tab[format_date(home_emi_day)];

  events.push(['Online', `Peaches - Home EMI 1 - Bajaj Housing Finance Ltd. Part 01 - ${home_emi_day.toLocaleString('en-EN', { month: 'long' })}`, 'd', CONFIG.home_emi / 2]);
  events.push(['Online', `Home EMI 1 - Bajaj Housing Finance Ltd. Part 01 - ${home_emi_day.toLocaleString('en-EN', { month: 'long' })}`, 'w', CONFIG.home_emi]);
  events.push(['Online', `Pet - SIP <> - ${home_emi_day.toLocaleString('en-EN', { month: 'long' })}`, 'w', 3000]);

  bank_tab[format_date(home_emi_day)] = events;

  return bank_tab;

};

function em_flag_personal_loan_adhira_appa(mon, k, bank_tab) {
  var personal_loan_emi_day = new Date(CONFIG.years, k, 4);
  var events = bank_tab[format_date(personal_loan_emi_day)];

  events.push(['', `Adhira & Appa EMI - IndusInd Bank - ${personal_loan_emi_day.toLocaleString('en-EN', { month: 'long' })}`, 'w', CONFIG.a_a_emi]);
  events.push(['', `Adhira & Appa EMI - Axis Bank - Peaches - ${personal_loan_emi_day.toLocaleString('en-EN', { month: 'long' })}`, 'w', 21242]);

  bank_tab[format_date(personal_loan_emi_day)] = events;

  return bank_tab;
};

function em_flag_earning_adhira_appa(mon, k, bank_tab) {
  var a_a_earning_day = new Date(CONFIG.years, k, 15);
  var events = bank_tab[format_date(a_a_earning_day)];

  events.push(['', `Adhira & Appa Earning - ${a_a_earning_day.toLocaleString('en-EN', { month: 'long' })}`, 'd', '117600']);

  bank_tab[format_date(a_a_earning_day)] = events;

  return bank_tab;
}

function em_flag_flat_maintanance_payment(mon, k, bank_tab) {
  var maintanance_payment_day = new Date(CONFIG.years, k, 11);
  var events = bank_tab[format_date(maintanance_payment_day)];

  events.push(['Online', `SLS Springs Maintinance + Water Charges - MyGate - ${maintanance_payment_day.toLocaleString('en-EN', { month: 'long' })}`, 'w', `=${mon}!$F$12`]);

  bank_tab[format_date(maintanance_payment_day)] = events;

  return bank_tab;

};

function em_flag_flat_electricity_payment(mon, k, bank_tab) {
  var electricity_payment_day = new Date(CONFIG.years, k, 15);
  var events = bank_tab[format_date(electricity_payment_day)];

  events.push(['CC Cred Flash', `Electricity Bill Payment - ${electricity_payment_day.toLocaleString('en-EN', { month: 'long' })}`, 'd', `=${mon}!$F$16`]);

  bank_tab[format_date(electricity_payment_day)] = events;

  return bank_tab;

};

function em_flag_internet_payment(mon, k, cc_tab) {
  var internet_payment_day = new Date(CONFIG.years, k, 14);
  var events = cc_tab[format_date(internet_payment_day)];

  events.push(['CC One 0531', `Internet Bill Payment - ${internet_payment_day.toLocaleString('en-EN', { month: 'long' })}`, 'd', `=${mon}!$F$15`]);

  cc_tab[format_date(internet_payment_day)] = events;

  return cc_tab;
};

function em_flag_meal_card(mon, k, cc_tab) {
  var last_working_day = get_last_working_day_for_month(new Date(CONFIG.years, k + 1, 0));
  var events = cc_tab[format_date(last_working_day)];

  events.push([CONFIG.meal_card, `Pluxee Meal Card Refill - ${last_working_day.toLocaleString('en-EN', { month: 'long' })}`, 'c', CONFIG.meal_card_amount]);

  cc_tab[format_date(last_working_day)] = events;

  return cc_tab;
};

function em_flag_cc_card_payments(mon, k, bank_tab, cc_tab, prev_year = 0) {
  Object.keys(CONFIG.card_map).forEach(cc_num => {
    var bill_date = new Date(CONFIG.years + prev_year, k, CONFIG.card_map[cc_num]['bill_date']);
    var repayment_date = add_days(bill_date, CONFIG.card_map[cc_num]['repayment_days'] - 1);

    if (repayment_date.getFullYear() <= CONFIG.years) {
      if (repayment_date.getFullYear() == CONFIG.years) {
        var events = bank_tab[format_date(repayment_date)];
        events.push(['', `${cc_num} - Repayment - ${bill_date.toLocaleString('en-EN', { month: 'long' })}`, 'w', '']);
        bank_tab[format_date(repayment_date)] = events;
      }

      // bill date for previous year
      if (prev_year === -1) {
        var events = cc_tab[format_date(bill_date)];
        events.push([cc_num, `${cc_num} - Bill - ${bill_date.toLocaleString('en-EN', { month: 'long' })}`, 'd', '']);
        cc_tab[format_date(bill_date)] = events;
      }

      var events = cc_tab[format_date(repayment_date)];
      events.push([cc_num, `${cc_num} - Repayment - ${bill_date.toLocaleString('en-EN', { month: 'long' })}`, 'c', '']);
      cc_tab[format_date(repayment_date)] = events;
    }
  });

  return {
    'bank_tab': bank_tab,
    'cc_tab': cc_tab
  };
};
// ---------------------------------------------------------------------------------------------------------------------

function get_month_abbr(reason) {
  if (!reason) return '';
  const match_regex = new RegExp(`(${Object.keys(CONFIG.month_days).join('|')})`, 'i');
  const match = reason.match(match_regex);
  if (match) return match[1];
  return '';
}

function flatten_bank_tab_to_array(bank_tab) {
  let current_row = 2, prev_date_row = 2;
  const bank_tab_flat = new Array();

  for (let i = 0; i < Object.keys(bank_tab).length; i++) {
    const date = Object.keys(bank_tab)[i];
    const events = bank_tab[date];

    const rows_to_generate = events.length <= 1 ? 1 : events.length;

    for (let num_events_rows = 0; num_events_rows < rows_to_generate; num_events_rows++) {
      var withdraw = '', deposit = '';

      var particulars = events[num_events_rows]?.[0] || '';
      var reason = events[num_events_rows]?.[1] || '';
      var w_d = events[num_events_rows]?.[2] || '';
      var amount = events[num_events_rows]?.[3] || '';

      var cc_num = reason.split(' - ', 1);

      amount = (reason.startsWith('CC') & amount === '') ? amount_bank_cc_formula(reason, current_row, CONFIG.card_map[cc_num]['last_digit_round']) : amount;

      switch (w_d) {
        case 'w':
          withdraw = amount;
          break;
        case 'd':
          deposit = amount;
          break;
        default:
          break;
      }

      let date_formula;
      if (current_row === 2) {
        date_formula = `=IFERROR(Jan!$A$2, Template!$A$2)`;
      } else if (num_events_rows === 0) {
        date_formula = `=A${prev_date_row} + 1`;
      } else {
        date_formula = `=A${current_row - 1}`;
      }

      var month = `=TEXT(A${current_row}, "MMMM")`;

      var balance = current_row === 2 ? `=O1 - F2 + G2` : `=H${current_row - 1} - F${current_row} + G${current_row}`;
      var filter_4 = `=IF(AND($A${current_row} < EOMONTH(TODAY(), -1) + 1), 1, 0)`;
      var filter_5 = `=IF(AND($A${current_row} < TODAY(), ISBLANK($D${current_row}), ISBLANK($E${current_row})), 1, 0)`;

      bank_tab_flat.push([date_formula, month, particulars, '', reason, withdraw, deposit, balance, '', '', '', filter_4, filter_5])
      current_row++;
    }
    prev_date_row = current_row - rows_to_generate;
  }
  return bank_tab_flat;
};

function flatten_cc_tab_to_array(cc_tab) {
  let current_row = 2, prev_date_row = 2;
  const cc_tab_flat = new Array();

  for (let i = 0; i < Object.keys(cc_tab).length; i++) {
    const date = Object.keys(cc_tab)[i];
    const events = cc_tab[date];

    const rows_to_generate = events.length <= 1 ? 1 : events.length;

    for (let num_events_rows = 0; num_events_rows < rows_to_generate; num_events_rows++) {
      var debit = '', credit = '';

      var cc_num = events[num_events_rows]?.[0] || '';
      var reason = events[num_events_rows]?.[1] || '';
      var d_c = events[num_events_rows]?.[2] || '';
      var amount = events[num_events_rows]?.[3] || '';

      amount = (reason.startsWith('CC') & amount === '') ? amount_bank_cc_formula(reason, current_row, CONFIG.card_map[cc_num]['last_digit_round']) : amount;

      switch (d_c) {
        case 'c':
          credit = amount;
          break;
        case 'd':
          debit = amount;
          break;
        default:
          break;
      }

      let date_formula;
      if (current_row === 2) {
        date_formula = `=IFERROR(Jan!$A$2, Template!$A$2) - 31`;
      } else if (num_events_rows === 0) {
        date_formula = `=A${prev_date_row} + 1`;
      } else {
        date_formula = `=A${current_row - 1}`;
      }

      var month = `=TEXT(A${current_row}, "MMM-YY")`;
      var filter_4 = `=IF(AND($A${current_row} < EOMONTH(TODAY(), -1) + 1), 1, 0)`;
      var filter_5 = `=IF(AND($A${current_row} < TODAY(), ISBLANK($D${current_row}), ISBLANK($E${current_row})), 1, 0)`;

      cc_tab_flat.push([date_formula, month, cc_num, '', reason, debit, credit, '', '', '', filter_4, filter_5])
      current_row++;
    }
    prev_date_row = current_row - rows_to_generate;
  }
  return cc_tab_flat;
}
