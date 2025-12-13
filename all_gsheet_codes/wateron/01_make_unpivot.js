function get_sheet(sheet) {
  return SpreadsheetApp.getActiveSpreadsheet().getSheetByName(sheet);
};


function read_and_validate_inputs() {
  try {
    var sheet = get_sheet("Parameters");
    var input_ranges = {
      "start_date": "B4",
      "per_litre_amount": "B5"
    }

    var inputs = {};
    inputs.start_date = sheet.getRange(input_ranges.start_date).getValue();
    inputs.per_litre_amount = sheet.getRange(input_ranges.per_litre_amount).getValue();


    // Month Start Date
    if (!(inputs.start_date instanceof Date) || isNaN(inputs.start_date)) {
      Logger.log("Error: Month Start Date must be a valid date.");
      SpreadsheetApp.getUi().alert("Error: Month Start Date must be a valid date.");
      return null;
    }

    if (typeof inputs.per_litre_amount !== 'number' || inputs.per_litre_amount <= 0) {
      Logger.log("Error: per_litre_amount must be a positive number.");
      SpreadsheetApp.getUi().alert("Error: per_litre_amount must be a positive number.");
      return null;
    }


    return inputs;

  } catch (e) {
    Logger.log("Error in read_and_validate_inputs: " + e);
    SpreadsheetApp.getUi().alert("An error occurred while reading inputs. Check script logs for details.");
    return null;
  }
};


function generate_date_array(month_start_date) {
  if (!(month_start_date instanceof Date) || isNaN(month_start_date)) {
    Logger.log("Error: Month Start Date must be a valid date.");
    SpreadsheetApp.getUi().alert("Error: Month Start Date must be a valid date.");
    return null;
  }

  month_start_date = new Date(month_start_date);
  const year = month_start_date.getFullYear();
  const month = month_start_date.getMonth();

  const days_in_month = new Date(year, month, 0).getDate();
  return Array.from({ length: days_in_month }, (_, i) => {
    const day = i + 1;
    return new Date(year, month - 1, day); // month - 1 because JS Date months are 0-based
  });

};


function generate_apartment_array(sheet_name, ranges) {
  var sheet = get_sheet(sheet_name);
  if (!sheet) throw new Error(`Sheet ${sheet_name} not found.`);

  const values = sheet.getRange(ranges).getValues(); // 100×1 2-D array
  const last_idx = values.findLastIndex(row => row[0] !== '');

  return last_idx === -1 ? [] : values.slice(0, last_idx + 1).map(row => row[0]);

};

function format_date(dates) {
  dates = new Date(dates)
  var yyyy = dates.getFullYear();
  var mm = String(dates.getMonth() + 1).padStart(2, '0');
  var dd = String(dates.getDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
};

function crossjoin_array1_array2(arr_1, arr_2) {
  return arr_1.flatMap(apartment =>
    arr_2.map(date => [apartment, format_date(date)])
  );
};


function add_formulaes_to_crossjoin_array(cj_arr) {
  var i = 2;
  var result = [];
  var row = [];
  for (var j = 0; j < cj_arr.length; j++) {
    row = [];
    k = j + 2;
    row.push(
      cj_arr[j][0],
      cj_arr[j][1],
      /* formulae for raw_data_pre_bill reading */
      `=IFERROR(INDEX(raw_data_pre_bill!$E$2:$AJ$407, MATCH($A${k}, raw_data_pre_bill!$A$2:$A$407, 0), MATCH($B${k}, raw_data_pre_bill!$E$1:$AJ$1, 0)), 0)`

      /* formulae for raw_data_post_bill reading */
      `=IFERROR(INDEX(raw_data_post_bill!$E$2:$AJ$407, MATCH($A${k}, raw_data_post_bill!$A$2:$A$407, 0), MATCH($B${k}, raw_data_post_bill!$E$1:$AJ$1, 0)), 0)`
    )
    result.push(row)
  }
};


function main() {

  var inputs = read_and_validate_inputs();
  var month_dates = generate_date_array(inputs.start_date);
  var apartments = generate_apartment_array('Total_consumption', 'A3:A102')
  var ap_date_combo = add_formulaes_to_crossjoin_array(crossjoin_array1_array2(apartments, month_dates))

  var paste_sheet = get_sheet("Unpivot");
  paste_sheet.getRange(`A2:D${(apartments.length * month_dates.length) + 1}`).setValues(ap_date_combo);

};
