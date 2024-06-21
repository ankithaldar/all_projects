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
    console.log('Duplicated ' + template_sheet_name + ' Sheet to ' + new_sheet_name + '.');
  }
  else {
    console.log(new_sheet_name + ' sheet already exists.');
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

  console.log(sheet_name + ' sheet is hidden.');

};

// -----------------------------------------------------------------------------

function create_date_array() {
  var
    step = 1,
    start_date = new Date(years + "-01-01"),
    end_date = new Date(years + "-12-31"),
    arr = new Array(),
    dt = new Date(start_date);

  while (dt <= new Date(end_date)) {
    arr.push(new Date(dt));
    dt.setDate(dt.getDate() + step);
  }

  return arr;
};


function delete_individual_sheet(sheet_name) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var itt = ss.getSheetByName(sheet_name);
  if (itt) {
    ss.deleteSheet(ss.getSheetByName(sheet_name));
  }
};
