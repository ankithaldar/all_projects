// ---------------------------------------------------------------------------------------------------------------------
function mfm_cache_tab_bank_formula() {
  var start_row_num = 9, end_row_num = 10, entry_name_column = 'R';

  const bank_formulas_for_tab = [];
  for (let row = start_row_num; row <= end_row_num; row++) {
    let formula_withdraw = '';
    let formula_deposit = '';

    CONFIG.banks.forEach((bank, k) => {
      const common_criteria = `'BankStatement - ${bank}'!$B:$B, TEXT($A$2, "MMMM"), 'BankStatement - ${bank}'!$C:$C, $${entry_name_column}${row}`;
      formula_withdraw += `SUMIFS('BankStatement - ${bank}'!$F:$F, ${common_criteria})`;
      formula_deposit += `SUMIFS('BankStatement - ${bank}'!$G:$G, ${common_criteria})`;

      if (k < CONFIG.banks.length - 1) {
        formula_withdraw += ' + ';
        formula_deposit += ' + ';
      }
    });
    bank_formulas_for_tab.push([`=(${formula_withdraw}) - (${formula_deposit})`]);
  }
  return bank_formulas_for_tab;
};

function mfm_cache_tab_cc_formula() {
  var end_row_num = CONFIG.card_start_row + [...Object.keys(CONFIG.card_map), CONFIG.meal_card].length - 1;
  var card_name_column = `R`, card_bill_date_column = `U`;

  const cc_formulas_for_tab = [];
  for (let row = CONFIG.card_start_row; row <= end_row_num; row++) {
    const expense = `=SUMIFS('CCStatement'!$F:$F, 'CCStatement'!$C:$C, $${card_name_column}${row}, 'CCStatement'!$B:$B, TEXT($A$2, "MMM-YY"))`;
    const bill_amt = `=ROUND(SUMIFS('CCStatement'!$F:$F, 'CCStatement'!$C:$C, $${card_name_column}${row}, 'CCStatement'!$A:$A, "<" & DATE(YEAR($A$2), MONTH($A$2), $${card_bill_date_column}${row})) - SUMIFS('CCStatement'!$G:$G, 'CCStatement'!$C:$C, $${card_name_column}${row}, 'CCStatement'!$A:$A, "<" & DATE(YEAR($A$2), MONTH($A$2), $${card_bill_date_column}${row})), 2)`;

    cc_formulas_for_tab.push([expense, bill_amt]);
  }
  return cc_formulas_for_tab;
};

// ---------------------------------------------------------------------------------------------------------------------
// Create formulas for each month sheet and push into a object
function mfm_cache_all_month_formulas(bank_formula, cc_formula) {
  var tab_formulas = new Object();
  Object.keys(CONFIG.month_days).forEach(mon => {

    tab_formulas[mon] = {
      'bank_txn': {
        'range': 'S9:S10',
        'formula': bank_formula
      },
      'cc_formula': {
        'range': `R${CONFIG.card_start_row}:T${CONFIG.card_start_row + [...Object.keys(CONFIG.card_map), CONFIG.meal_card].length - 1}`,
        'formula': mfm_create_cc_formula_for_tab(mon, cc_formula)
      },
      'wallet': {
        'range': `R13:U22`,
        'formula': mfm_create_wallet_formula_for_tab(mon)
      },
      'lm_cash': {
        'range': `S6`,
        'formula': [[check_if_month_jan(mon) ? `=IMPORTRANGE("${CONFIG.old_sheet_link}", "Dec!$S2")` : `=${get_previous_month_initials(mon)}!$S2`]]
      },
      'denomination': {
        'range': `X4:X15`,
        'formula': mfm_create_denom_formula_for_tab(mon)
      },
      'end_month': {
        'range': `B${CONFIG.month_days[mon] + 1}:H${CONFIG.month_days[mon] + Object.keys(CONFIG.month_end_salary).length}`,
        'formula': mfm_create_end_month_formula_for_tab(mon)
      },
      'first_day': {
        'range': `A2`,
        'formula': [[check_if_month_jan(mon) ? `${mon} 01, ${CONFIG.years}` : `=${get_previous_month_initials(mon)}!A${CONFIG.month_days[get_previous_month_initials(mon)] + 1} + 1`]]
      },
      'expense_list': {
        'range': `M2:M53`,
        'formula': Array.from({ length: 52 }, (_, i) => [check_if_month_jan(mon) ? `=IMPORTRANGE("${CONFIG.old_sheet_link}", "Dec!$M${i + 2}")` : `=${get_previous_month_initials(mon)}!$M${i + 2}`]
        )
      }
    }
  });

  return tab_formulas;
};

function get_previous_month_initials(mon) {
  var all_months = Object.keys(CONFIG.month_days);
  return all_months[all_months.indexOf(mon) - 1];
};

function check_if_month_jan(mon) {
  return mon === 'Jan';
};

function mfm_create_cc_formula_for_tab(mon, cc_formula) {
  var formula = new Array(), map_analytics_tab = 4;

  for (let i = 0; i < cc_formula.length; i++) {
    var cc_name = i === 0 && check_if_month_jan(mon) ? CONFIG.meal_card : i > 0 && check_if_month_jan(mon) ? `=Analytics!H${map_analytics_tab + i - 1}` : `=${get_previous_month_initials(mon)}!R${CONFIG.card_start_row + i}`;
    formula.push([cc_name, cc_formula[i][0], cc_formula[i][1]]);
  }
  return formula;
};


function mfm_create_wallet_formula_for_tab(mon) {
  var formula = new Array();
  for (let i = 13; i <= 22; i++) {
    formula.push([
      check_if_month_jan(mon) ? `=IMPORTRANGE("${CONFIG.old_sheet_link}", "Dec!$R${i}")` : `=${get_previous_month_initials(mon)}!$R${i}`,
      '',
      `=U${i}-S${i}`,
      check_if_month_jan(mon) ? `=IMPORTRANGE("${CONFIG.old_sheet_link}", "Dec!$U${i}")` : `=${get_previous_month_initials(mon)}!$T${i}`
    ]);
  }
  return formula;
};


function mfm_create_denom_formula_for_tab(mon) {
  var formula = new Array();
  for (let i = 4; i <= 15; i++) {
    formula.push([check_if_month_jan(mon) ? `=IMPORTRANGE("${CONFIG.old_sheet_link}", "Dec!$X${i}")` : `=${get_previous_month_initials(mon)}!$X${i}`]);
  }
  return formula;
};


function mfm_create_end_month_formula_for_tab(mon) {
  var formula = new Array(), m_days = CONFIG.month_days[mon] + 1;

  for (let i = 0; i < Object.keys(CONFIG.month_end_salary).length; i++) {
    const item = Object.keys(CONFIG.month_end_salary)[i];
    const rate = CONFIG.month_end_salary[item];
    const total = `=C${m_days + i} * D${m_days + i}`;
    const paid = `=IF(E${m_days + i}=0, 0, ROUND(E${m_days + i} - G${m_days + i}, 2))`;

    formula.push([item, 1, rate, total, paid, '', 'Home']);
  }
  return formula;
};
