// ---------------------------------------------------------------------------------------------------------------------
function make_new_sheets() {
  mns_create_bank_statement();

  mns_create_cc_statement();

  mns_create_month_templates(mfm_cache_all_month_formulas(mfm_cache_tab_bank_formula(), mfm_cache_tab_cc_formula()));
};

function mns_create_bank_statement() {
  utils_duplicate_sheets_by_names(
    CONFIG.hidden_tabs[1],
    CONFIG.banks.map(bank => `${CONFIG.hidden_tabs[1]} - ${bank}`)
  );

  CONFIG.banks.forEach(bank => {
    var tab = CONFIG.sheet_cache.getSheetByName(`${CONFIG.hidden_tabs[1]} - ${bank}`)
    tab.getRange('O1').setValue(`=IMPORTRANGE("${CONFIG.old_sheet_link}", "${CONFIG.hidden_tabs[1]} - ${bank}!$P$1")`);
  });
};

function mns_create_cc_statement() {
  utils_duplicate_sheets_by_names(
    CONFIG.hidden_tabs[2],
    ['CCStatement']
  );
};

function mns_create_month_templates(tab_formulas) {
  // place all formulas & clear to right dates
  Object.keys(tab_formulas).forEach(mon => {
    utils_duplicate_sheets_by_names(
      CONFIG.hidden_tabs[0],
      [mon]
    );
    var tab = CONFIG.sheet_cache.getSheetByName(mon);
    Object.keys(tab_formulas[mon]).forEach(formula => {
      tab.getRange(tab_formulas[mon][formula]['range']).setValues(tab_formulas[mon][formula]['formula']);
    });

    if (CONFIG.month_days[mon] + 2 <= 32) tab.getRange(`A${CONFIG.month_days[mon] + 2}:A32`).clearContent();
  });

  utils_hide_sheets_by_name(CONFIG.hidden_tabs);

  mns_move_required_sheets()

};

function mns_move_required_sheets() {
  // move tabs to the end of sheet
  CONFIG.move_tabs.forEach(name => {
    var ss = CONFIG.sheet_cache;
    var sht_name = ss.getSheetByName(name);
    ss.setActiveSheet(sht_name);
    ss.moveActiveSheet(ss.getNumSheets());
  });
};
