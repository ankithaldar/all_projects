/**
 * Function to get a dynamic range in a Google Sheet.
 * @param {Sheet} sheet The Google Sheet Sheet object.
 * @param {number} header_row_index The row index of the header row (0-based).
 * @param {number} start_column_index The column index where the data starts (0-based).
 * @param {number} value_columns_count The number of columns that contain data for each entry (e.g., 2 for date and amount).
 * @param {number} start_data_row_offset Optional offset for the starting data row from the header (default is 1, meaning data starts on the row after the header).
 * @return {string} The dynamic range notation (e.g., "D18:E"). Returns null if no data is found below the header.
 */
function get_dynamic_range(sheet, header_row_index, start_column_index, value_columns_count, start_data_row_offset = 1) {
  if (value_columns_count <= 0) {
    Logger.log("Error: value_columns_count must be greater than 0.");
    return null; // Or throw an error
  }
  if (start_data_row_offset < 0 || !Number.isInteger(start_data_row_offset)) {
    Logger.log("Error: start_data_row_offset must be a non-negative integer.");
    return null;
  }

  var start_row = header_row_index + 1 + start_data_row_offset; // Data starts after the header
  var last_row = sheet.getLastRow();
  var start_col_letter = columnToLetter(start_column_index + 1); // Convert 0-based index to 1-based and then to letter
  var end_col_letter = columnToLetter(start_column_index + value_columns_count);

  if (last_row <= header_row_index + start_data_row_offset) { // No data rows below the header
    return null; // Or return an empty range like start_col_letter + start_row + ":" + end_col_letter + start_row if you want headers to be read even without data
  }

  return start_col_letter + start_row + ":" + end_col_letter;
}


/**
 * Helper function to convert column number to column letter (e.g., 1 to A, 2 to B, 27 to AA).
 * @param {number} columnNumber
 * @return {string}
 */
function columnToLetter(columnNumber) {
  var temp, letter = '';
  while (columnNumber > 0) {
    temp = (columnNumber - 1) % 26;
    letter = String.fromCharCode(65 + temp) + letter;
    columnNumber = (columnNumber - temp - 1) / 26;
  }
  return letter;
}


/**
 * Helper function to check if a row in a 2D array is effectively empty.
 * Considers null, undefined, and whitespace-only strings as empty.
 * @param {Array} row The row array from getValues().
 * @return {boolean} True if the row is empty, false otherwise.
 */
function isEmptyRow(row) {
  for (var i = 0; i < row.length; i++) {
    var cellValue = row[i];
    if (cellValue != null && String(cellValue).trim() !== "") {
      return false;
    }
  }
  return true;
}


/**
 * Reads and validates input values for the loan amortization calculator from the Google Sheet.
 * Uses snake_case for variable names and dynamic ranges for extra payments and interest rate changes.
 * @return {object|null} An object containing validated input values if successful, null otherwise.
 */
function read_and_validate_inputs() {
  try {
    var sheet = get_sheet("emi_calculator"); // Assuming inputs are on the active sheet, adjust if needed

    // --- Input Cell/Range Definitions ---
    // **IMPORTANT:**  You will need to replace the cell references below
    // to match your Google Sheet layout for the *single-cell* inputs.
    var input_ranges = {
      loan_amount: "Loan_Amount",         // Example: Cell B2 for Loan Amount
      loan_start_date: "Loan_Start_Date",      // Example: Cell B3 for Loan Start Date
      first_payment_date: "First_Payment_Date",   // Example: Cell B4 for First Payment Date
      compound_period: "Compound_Period", // Example: Cell B5 for Monthly Compound Period (Checkbox or TRUE/FALSE)
      payment_type: "Payment_Type",        // Example: Cell B6 for Payment Type (Dropdown)
      rounding: "Rounding",           // Example: Cell B7 for Rounding (Number of decimal places)
      emi: "EMI",                // Example: Cell B8 for EMI
      payment_date: "Monthly_Payment_Date",        // Example: Cell B9 for Payment Date (Day of month)
      end_of_financial_year: "End_Of_Financial_Year", // Example: Cell B10 for End of Financial Year
    };

    // --- Dynamic Range Settings ---
    var extra_payments_header_row = 17; // Row 17 is header for Extra Payments (0-based index is 16)
    var extra_payments_start_column = 1; // Column B is index 1 (0-based)
    var extra_payments_columns_count = 2; // Date and Amount

    var interest_rate_changes_header_row = 17; // Row 17 is header for Interest Rate Changes (0-based index is 16)
    var interest_rate_changes_start_column = 4; // Column E is index 4 (0-based)
    var interest_rate_changes_columns_count = 2; // Date and Rate


    // --- Get Dynamic Ranges ---
    var extra_payments_range_str = get_dynamic_range(sheet, extra_payments_header_row - 1, extra_payments_start_column, extra_payments_columns_count); // Header row is 17, so index is 16
    var interest_rate_changes_range_str = get_dynamic_range(sheet, interest_rate_changes_header_row - 1, interest_rate_changes_start_column, interest_rate_changes_columns_count); // Header row is 17, so index is 16


    // --- Data Retrieval ---
    var inputs = {};
    inputs.loan_amount = sheet.getRange(input_ranges.loan_amount).getValue();
    inputs.loan_start_date = sheet.getRange(input_ranges.loan_start_date).getValue();
    inputs.first_payment_date = sheet.getRange(input_ranges.first_payment_date).getValue();
    inputs.compound_period = sheet.getRange(input_ranges.compound_period).getValue();
    inputs.payment_type = sheet.getRange(input_ranges.payment_type).getValue();
    inputs.rounding = sheet.getRange(input_ranges.rounding).getValue();
    inputs.emi = sheet.getRange(input_ranges.emi).getValue();
    inputs.payment_date = sheet.getRange(input_ranges.payment_date).getValue();
    inputs.end_of_financial_year = sheet.getRange(input_ranges.end_of_financial_year).getValue();

    extra_payments_data = (extra_payments_range_str) ? sheet.getRange(extra_payments_range_str).getValues() : []; // Get data if range is valid, else empty array
    interest_rate_changes_data = (interest_rate_changes_range_str) ? sheet.getRange(interest_rate_changes_range_str).getValues() : []; // Get data if range is valid, else empty array


    // --- Input Validation ---
    // (The validation logic remains the same as in the previous version, just using snake_case input names)
    // ... (rest of the validation code from the previous response remains here, using `inputs.loan_amount`, `inputs.loan_start_date`, etc.) ...

    // Loan Amount
    if (typeof inputs.loan_amount !== 'number' || inputs.loan_amount <= 0) {
      Logger.log("Error: Loan Amount must be a positive number.");
      SpreadsheetApp.getUi().alert("Error: Loan Amount must be a positive number.");
      return null;
    }

    // Loan Start Date
    if (!(inputs.loan_start_date instanceof Date) || isNaN(inputs.loan_start_date)) {
      Logger.log("Error: Loan Start Date must be a valid date.");
      SpreadsheetApp.getUi().alert("Error: Loan Start Date must be a valid date.");
      return null;
    }

    // First Payment Date
    if (!(inputs.first_payment_date instanceof Date) || isNaN(inputs.first_payment_date)) {
      Logger.log("Error: First Payment Date must be a valid date.");
      SpreadsheetApp.getUi().alert("Error: First Payment Date must be a valid date.");
      return null;
    }
    if (inputs.first_payment_date < inputs.loan_start_date) {
      Logger.log("Error: First Payment Date cannot be before Loan Start Date.");
      SpreadsheetApp.getUi().alert("Error: First Payment Date cannot be before Loan Start Date.");
      return null;
    }

    // Monthly Compound Period (Boolean)
    if (typeof inputs.compound_period !== 'string') {
      Logger.log("Error: Compound Period must be 'Monthly', 'Quarterly', or 'Yearly'.");
      SpreadsheetApp.getUi().alert("Error: Compound Period must be 'Monthly', 'Quarterly', or 'Yearly'.");
      return null;
    } else {
      var period_upper = inputs.compound_period.toUpperCase();
      if (period_upper !== "MONTHLY" && period_upper !== "QUARTERLY" && period_upper !== "YEARLY") {
        Logger.log("Error: Compound Period must be 'Monthly', 'Quarterly', or 'Yearly'.");
        SpreadsheetApp.getUi().alert("Error: Compound Period must be 'Monthly', 'Quarterly', or 'Yearly'.");
        return null;
      }
    }

    // Payment Type (For now, just check for "EMI")
    if (!(inputs.payment_type.toUpperCase() === "START OF PERIOD" || inputs.payment_type.toUpperCase() === "END OF PERIOD")) {
      Logger.log("Error: Payment Type must be 'either of Start of Period or End of Period' (for now).");
      SpreadsheetApp.getUi().alert("Error: Payment Type must be 'EMI' (for now).");
      return null;
    }

    // Rounding (ON/OFF Validation)
    if (typeof inputs.rounding === 'string') {
      var rounding_upper = inputs.rounding.toUpperCase();
      if (rounding_upper === "ON" || rounding_upper === "TRUE") {
        inputs.rounding = true; // Set to boolean true for ON
      } else if (rounding_upper === "OFF" || rounding_upper === "FALSE") {
        inputs.rounding = false; // Set to boolean false for OFF
      } else {
        Logger.log("Error: Rounding must be 'ON' or 'OFF' (or TRUE/FALSE or a checkbox).");
        SpreadsheetApp.getUi().alert("Error: Rounding must be 'ON' or 'OFF' (or TRUE/FALSE or a checkbox).");
        return null;
      }
    } else if (typeof inputs.rounding === 'boolean') {
      // If it's already a boolean, use it directly (e.g., from a checkbox)
    } else {
      Logger.log("Error: Rounding must be 'ON' or 'OFF' (or TRUE/FALSE or a checkbox).");
      SpreadsheetApp.getUi().alert("Error: Rounding must be 'ON' or 'OFF' (or TRUE/FALSE or a checkbox).");
      return null;
    }

    // EMI
    if (typeof inputs.emi !== 'number' || inputs.emi <= 0) {
      Logger.log("Error: EMI must be a positive number.");
      SpreadsheetApp.getUi().alert("Error: EMI must be a positive number.");
      return null;
    }

    // Payment Date (Day of month)
    if (typeof inputs.payment_date !== 'number' || !Number.isInteger(inputs.payment_date) || inputs.payment_date < 1 || inputs.payment_date > 31) {
      Logger.log("Error: Payment Date must be an integer between 1 and 31.");
      SpreadsheetApp.getUi().alert("Error: Payment Date must be an integer between 1 and 31.");
      return null;
    }
    inputs.payment_date = parseInt(inputs.payment_date); // Ensure it's an integer

    // End of Financial Year
    if (!(inputs.end_of_financial_year instanceof Date) || isNaN(inputs.end_of_financial_year)) {
      Logger.log("Error: End of Financial Year must be a valid date.");
      SpreadsheetApp.getUi().alert("Error: End of Financial Year must be a valid date.");
      return null;
    }

    // Extra Payments Data Validation
    inputs.extra_payments = [];
    if (extra_payments_data && extra_payments_data.length > 0) {
      for (var i = 0; i < extra_payments_data.length; i++) {
        if (extra_payments_data[i][0] instanceof Date && !isNaN(extra_payments_data[i][0]) &&
          typeof extra_payments_data[i][1] === 'number' && extra_payments_data[i][1] >= 0) {
          inputs.extra_payments.push({
            date: extra_payments_data[i][0],
            amount: extra_payments_data[i][1]
          });
        } else if (!(extra_payments_data[i][0] == null && extra_payments_data[i][1] == null)) {
          if (!isEmptyRow(extra_payments_data[i])) {
            Logger.log("Error: Invalid Extra Payment entry in row " + (i + extra_payments_header_row + 1) + ". Must be Date and Amount.");
            SpreadsheetApp.getUi().alert("Error: Invalid Extra Payment entry in row " + (i + extra_payments_header_row + 1) + ". Must be Date and Amount.");
            return null;
          }
        }
      }
    }

    // Interest Rate Changes Data Validation
    inputs.interest_rate_changes = [];
    if (interest_rate_changes_data && interest_rate_changes_data.length > 0) {
      for (var j = 0; j < interest_rate_changes_data.length; j++) {
        if (interest_rate_changes_data[j][0] instanceof Date && !isNaN(interest_rate_changes_data[j][0]) &&
          typeof interest_rate_changes_data[j][1] === 'number' && interest_rate_changes_data[j][1] <= 1 && interest_rate_changes_data[j][1] >= 0) {
          inputs.interest_rate_changes.push({
            date: interest_rate_changes_data[j][0],
            rate: interest_rate_changes_data[j][1] // Store as decimal percentage
          });
        } else if (!(interest_rate_changes_data[j][0] == null && interest_rate_changes_data[j][1] == null)) {
          if (!isEmptyRow(interest_rate_changes_data[j])) {
            Logger.log("Error: Invalid Interest Rate Change entry in row " + (j + interest_rate_changes_header_row + 1) + ". Must be Date and Interest Rate Percentage.");
            SpreadsheetApp.getUi().alert("Error: Invalid Interest Rate Change entry in row " + (j + interest_rate_changes_header_row + 1) + ". Must be Date and Interest Rate Percentage.");
            return null;
          }
        }
      }
    }

    return inputs; // Return validated inputs as an object

  } catch (e) {
    Logger.log("Error in read_and_validate_inputs: " + e);
    SpreadsheetApp.getUi().alert("An error occurred while reading inputs. Check script logs for details.");
    return null;
  }
}
