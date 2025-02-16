/**
 * Generates the loan amortization schedule as a 2D array.
 * @param {object} inputs Validated input object from read_and_validate_inputs().
 * @return {Array<Array<any>>|null} 2D array representing the amortization schedule, or null if error.
 */
function generate_amortization_schedule(inputs) {
  try {
    var schedule = [];
    schedule.push([
      "Inst #", "Due Date", "Opening Principal", "Principal",
      "Interest", "EMI", "Closing Principal", "Rate %", "Additional Payments"
    ]); // Modified Header row

    var loan_amount = inputs.loan_amount;
    var payment_date_day = inputs.payment_date;
    var emi = inputs.emi;
    var rounding_on = inputs.rounding;
    var compound_period = inputs.compound_period.toUpperCase();
    var end_of_financial_year = inputs.end_of_financial_year;
    var extra_payments = inputs.extra_payments;
    var interest_rate_changes = inputs.interest_rate_changes;

    var current_balance = loan_amount;
    var installment_number = 0; // Changed from period to installment_number for clarity
    var cumulative_interest = 0;
    var fin_year_interest = 0;
    var last_fin_year_end = new Date(end_of_financial_year);
    last_fin_year_end.setDate(last_fin_year_end.getDate() - 1);

    var last_payment_date = new Date(inputs.loan_start_date); // Initialize for first period opening balance display

    for (let i = 0; i < inputs.all_events.length; i++) {
      var event_date = inputs.all_events[i];
      var period_interest_rate_annual = get_interest_rate_for_period(event_date, interest_rate_changes, interest_rate_changes[0].rate);
      var period_interest_rate_monthly;

      if (compound_period === 'MONTHLY') {
        period_interest_rate_monthly = period_interest_rate_annual / 12;
      } else if (compound_period === 'QUARTERLY') {
        period_interest_rate_monthly = Math.pow(1 + period_interest_rate_annual / 4, 1 / 3) - 1;
      } else if (compound_period === 'YEARLY') {
        period_interest_rate_monthly = Math.pow(1 + period_interest_rate_annual, 1 / 12) - 1;
      }

      var extra_payment_amount = get_extra_payment_for_date(event_date, extra_payments);
      var interest_rate_change_amount = get_interest_rate_change_percent_for_date(event_date, interest_rate_changes); // Function to get rate change amount

      var opening_principal = current_balance;
      var principal_payment_installment = 0;
      var interest_payment_installment = 0;
      var installment_amount = 0;

      var is_emi_date = (event_date.getDate() === payment_date_day); // Check if it's an EMI date

      if (is_emi_date) {
        installment_number++; // Increment only on EMI dates
        interest_payment_installment = current_balance * period_interest_rate_monthly;
        principal_payment_installment = emi - interest_payment_installment;

        principal_payment_installment += extra_payment_amount; // Apply extra payment on EMI date
        if (principal_payment_installment < 0) principal_payment_installment = 0;
        if (principal_payment_installment > current_balance) {
          principal_payment_installment = current_balance;
        }
        interest_payment_installment = emi - principal_payment_installment;
        if (interest_payment_installment < 0) interest_payment_installment = 0;

        installment_amount = principal_payment_installment + interest_payment_installment; // Calculate Installment Amount
      } else if (extra_payment_amount > 0 || interest_rate_change_amount !== null) {
        // For non-EMI dates with events, installment components are 0, extra payment handled below
        principal_payment_installment = 0;
        interest_payment_installment = 0;
        installment_amount = 0;
      }


      current_balance -= principal_payment_installment;

      cumulative_interest += interest_payment_installment;

      var period_fin_year_interest = 0;
      if (event_date <= end_of_financial_year) {
        period_fin_year_interest = interest_payment_installment;
      }
      fin_year_interest += period_fin_year_interest;


      if (rounding_on) {
        interest_payment_installment = round_currency(interest_payment_installment, 2);
        principal_payment_installment = round_currency(principal_payment_installment, 2);
        installment_amount = round_currency(installment_amount, 2);
        current_balance = round_currency(current_balance, 2);
        cumulative_interest = round_currency(cumulative_interest, 2);
        fin_year_interest = round_currency(fin_year_interest, 2);
        extra_payment_amount = round_currency(extra_payment_amount, 2);
      }


      schedule.push([
        installment_number, // Installment Number
        Utilities.formatDate(event_date, Session.getTimeZone(), "yyyy-MM-dd"), // Due Date
        opening_principal, // Opening Principal
        principal_payment_installment, // Principal paid in installment
        interest_payment_installment, // Interest paid in installment
        installment_amount, // Installment Amount
        current_balance, // Closing Principal
        period_interest_rate_annual * 100, // Interest Rate for Period
        // cumulative_interest, // Cumulative Interest
        // fin_year_interest, // Financial Year Interest
        extra_payment_amount,     // Display Extra Payment
        // interest_rate_change_amount !== null ? (interest_rate_change_amount * 100) : null // Display Interest Rate Change
      ]);

      if (extra_payment_amount > 0 && !is_emi_date) { // Apply extra payment if not already applied in EMI
        current_balance -= extra_payment_amount;
        if (current_balance < 0) current_balance = 0; // Balance cannot be negative
      }


      last_payment_date = new Date(event_date); // Update last payment date for next period's opening balance


      if (current_balance < 0.01) current_balance = 0;

      if (event_date > last_fin_year_end) {
        fin_year_interest = 0;
        last_fin_year_end = get_next_financial_year_end(end_of_financial_year, event_date);
        last_fin_year_end.setDate(last_fin_year_end.getDate() - 1);
      }

      if (current_balance <= 0) break; // Exit loop if loan is paid off

    }

    return schedule;

  } catch (e) {
    Logger.log("Error in generate_amortization_schedule: " + e);
    SpreadsheetApp.getUi().alert("Error generating amortization schedule. Check script logs.");
    return null;
  }
}


/**
 * Retrieves the interest rate applicable for a given payment date, considering interest rate changes.
 * (No changes in this function)
 */
function get_interest_rate_for_period(paymentDate, interestRateChanges, defaultRate) {
  var applicableRate = defaultRate || 0.05;

  if (interestRateChanges && interestRateChanges.length > 0) {
    for (var i = 0; i < interestRateChanges.length; i++) {
      var change = interestRateChanges[i];
      if (paymentDate >= change.date) {
        applicableRate = change.rate;
      } else {
        break;
      }
    }
  }
  return applicableRate;
}


/**
 * Retrieves the extra payment amount for a given payment date.
 * (No changes in this function)
 */
function get_extra_payment_for_date(paymentDate, extraPayments) {
  var extraPaymentAmount = 0;
  if (extraPayments && extraPayments.length > 0) {
    for (var i = 0; i < extraPayments.length; i++) {
      var extraPayment = extraPayments[i];
      if (paymentDate.toDateString() === extraPayment.date.toDateString()) {
        extraPaymentAmount = extraPayment.amount;
        break;
      }
    }
  }
  return extraPaymentAmount;
}

/**
 * Retrieves the interest rate change percentage for a given date.
 * @param {Date} paymentDate The date to check for interest rate changes.
 * @param {Array<object>} interestRateChanges Array of interest rate change objects.
 * @return {number|null} The interest rate percentage if changed on this date, otherwise null.
 */
function get_interest_rate_change_percent_for_date(paymentDate, interestRateChanges) {
  if (interestRateChanges && interestRateChanges.length > 0) {
    for (var i = 0; i < interestRateChanges.length; i++) {
      var change = interestRateChanges[i];
      if (paymentDate.toDateString() === change.date.toDateString()) {
        return change.rate; // Return the new rate for this date
      }
    }
  }
  return null; // No rate change on this date
}


/**
 * Rounds a currency value to a specified number of decimal places.
 * (No changes in this function)
 */
function round_currency(value, decimals = 2) {
  var multiplier = Math.pow(10, decimals);
  return Math.round(value * multiplier) / multiplier;
}


/**
 * Adds months to a date object, handling cases where the target day of month doesn't exist.
 * (No changes in this function)
 */
function add_months(date, monthsToAdd, targetDayOfMonth) {
  var newDate = new Date(date);
  var currentMonth = newDate.getMonth();
  newDate.setMonth(currentMonth + monthsToAdd);

  if (targetDayOfMonth) {
    var daysInMonth = new Date(newDate.getFullYear(), newDate.getMonth() + 1, 0).getDate();
    if (targetDayOfMonth > daysInMonth) {
      newDate.setDate(daysInMonth);
    } else {
      newDate.setDate(targetDayOfMonth);
    }
  }
  return newDate;
}


/**
 * Calculates the next financial year end date.
 * (No changes in this function)
 */
function get_next_financial_year_end(financialYearEnd, currentDate) {
  var currentYear = currentDate.getFullYear();
  var fyEndThisYear = new Date(financialYearEnd);
  fyEndThisYear.setFullYear(currentYear);

  if (currentDate > fyEndThisYear) {
    return new Date(financialYearEnd.getFullYear() + 1, financialYearEnd.getMonth(), financialYearEnd.getDate());
  } else {
    return fyEndThisYear;
  }
}


/**
 * Outputs the amortization schedule to a Google Sheet starting from the specified range.
 * @param {Array<Array<any>>} schedule The 2D array representing the amortization schedule.
 * @param {string} outputRange Notation of the top-left cell for output (e.g., "Sheet1!I17").
 */
function output_schedule_to_sheet(schedule, outputRange) {
  if (!schedule || schedule.length === 0) {
    Logger.log("No schedule to output.");
    SpreadsheetApp.getUi().alert("No amortization schedule to output.");
    return;
  }

  if (!outputRange) {
    Logger.log("Error: Output range is not defined.");
    SpreadsheetApp.getUi().alert("Error: Output range is not defined. Check script.");
    return;
  }

  try {
    var sheetName = outputRange.split("!")[0];
    var startCell = outputRange.split("!")[1];
    var sheet = get_sheet(sheetName);

    if (!sheet) {
      Logger.log("Error: Sheet '" + sheetName + "' not found.");
      SpreadsheetApp.getUi().alert("Error: Sheet '" + sheetName + "' not found.");
      return;
    }

    var startRow = parseInt(startCell.match(/\d+/)[0]);
    var startColumnLetter = startCell.match(/[A-Z]+/)[0];
    var startColumnIndex = columnLetterToNumber(startColumnLetter); // Helper function below

    var numRows = schedule.length;
    var numCols = schedule[0].length;

    // Clear any existing content in the output range before writing
    var clearRange = sheet.getRange(startRow, startColumnIndex, sheet.getMaxRows() - startRow + 1, numCols);
    clearRange.clearContent();
    // clearRange.clearFormat(); // Clear formatting as well

    var outputSheetRange = sheet.getRange(startRow, startColumnIndex, numRows, numCols);
    outputSheetRange.setValues(schedule);

    // Format headers to be bold
    var headerRange = sheet.getRange(startRow, startColumnIndex, 1, numCols);
    headerRange.setFontWeight("bold");

    // Auto-resize columns for better readability (optional)
    // sheet.autoResizeColumns(startColumnIndex, numCols);

    // Logger.log("Amortization schedule output to " + outputRange);
    // SpreadsheetApp.getUi().alert("Amortization schedule generated and output to " + outputRange);


  } catch (e) {
    Logger.log("Error in output_schedule_to_sheet: " + e);
    SpreadsheetApp.getUi().alert("Error outputting schedule to sheet. Check script logs.");
  }
}


/**
 * Helper function to convert column letter to column number (e.g., A to 1, B to 2, AA to 27).
 * @param {string} columnLetter
 * @return {number}
 */
function columnLetterToNumber(columnLetter) {
  var columnNumber = 0;
  var letters = columnLetter.split('');
  for (var i = 0; i < letters.length; i++) {
    columnNumber += (letters[i].charCodeAt(0) - 64) * Math.pow(26, (letters.length - 1 - i));
  }
  return columnNumber;
}
