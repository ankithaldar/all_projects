// Helper Function


function leap_year_check_and_manipulate(year_num, days_arr) {
  if (((year_num % 4 == 0) && (year_num % 100 != 0)) || (year_num % 400 == 0)) {
    if (days_arr[1] == 28) {
      days_arr[1] = 29;
    }
  }
  return days_arr
};

// -----------------------------------------------------------------------------

function duplicate_individual_sheet(template_sheet_name, new_sheet_name) {
  // console.log('Duplicating ' + template_sheet_name + ' Sheet to ' + new_sheet_name + '.');
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  ss.setActiveSheet(ss.getSheetByName(template_sheet_name));
  var itt = ss.getSheetByName(new_sheet_name);
  if (!itt) {
    ss.duplicateActiveSheet();
    ss.getActiveSheet().setName(new_sheet_name);
    ss.moveActiveSheet(ss.getNumSheets());
    if (if_debug) {
      console.log('Duplicated ' + template_sheet_name + ' Sheet to ' + new_sheet_name + '.');
    }

  }
  else {
    if (if_debug) {
      console.log(new_sheet_name + ' sheet already exists.');
    }
  }
};

// -----------------------------------------------------------------------------

function hide_sheets(sheet_name) {
  if (typeof sheet_name === 'string') {
    hide_individual_sheet(sheet_name);
  }
  else {
    for (var i = 0; i < sheet_name.length; i++) {
      hide_individual_sheet(sheet_name[i]);
    }
  }
};

// -----------------------------------------------------------------------------

function hide_individual_sheet(sheet_name) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  ss = ss.setActiveSheet(ss.getSheetByName(sheet_name)).hideSheet();

  if (if_debug) {
    console.log(sheet_name + ' sheet is hidden.');
  }
};

// -----------------------------------------------------------------------------


function delete_individual_sheet(sheet_name) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var itt = ss.getSheetByName(sheet_name);
  if (itt) {
    ss.deleteSheet(ss.getSheetByName(sheet_name));
  }
};


function get_last_row(sheet) {
  return sheet.getLastRow();
};


function add_days(date, days) {
  var result = new Date(date);
  result.setDate(result.getDate() + days);
  return result;
};

function get_sheet(sheet) {
  return SpreadsheetApp.getActiveSpreadsheet().getSheetByName(sheet);
};


function get_date_today() {
  return new Date();
};


function last_date_of_month(mon_3_char) {
  return new Date(years, mons.indexOf(mon_3_char) + 1, 0).getDate()
};


function get_mon_3_char(key_name) {
  return key_name.split(' - ')[1].substring(0, 3)
};
