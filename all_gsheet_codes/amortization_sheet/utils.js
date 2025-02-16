// function getDynamicRange(sheet, column, startRow) {
//   var range = sheet.getRange(startRow, column, sheet.getLastRow() - startRow + 1);
//   var values = range.getValues().flat().filter(String);
//   return sheet.getRange(startRow, column, values.length, 2); // Expanding to 2 columns
// };


function get_sheet(sheet) {
  return SpreadsheetApp.getActiveSpreadsheet().getSheetByName(sheet);
};


function round_currency(value) {
  return Math.round(value);
}


/**
 * Extracts the maximum interest rate from the interest_rate_changes array.
 * @param {Array<object>} interest_rate_changes An array of objects, each with 'date' and 'rate' properties.
 * @return {number|null} The maximum interest rate found in the array, or null if the array is empty or invalid.
 */
function get_max_interest_rate(interest_rate_changes) {
  if (!interest_rate_changes || interest_rate_changes.length === 0) {
    return null; // Return null if no interest rate changes are provided
  }

  let maxRate = 0; // Initialize with 0, assuming interest rates are non-negative
  let rateFound = false; // Flag to check if at least one valid rate is found

  for (let i = 0; i < interest_rate_changes.length; i++) {
    const rateChange = interest_rate_changes[i];
    if (rateChange && typeof rateChange.rate === 'number') {
      maxRate = Math.max(maxRate, rateChange.rate);
      rateFound = true; // Set flag as a valid rate is found
    }
  }

  if (rateFound) {
    return maxRate;
  } else {
    return null; // Return null if no valid interest rates were found in the array
  }
};


function get_tenure(inputs) {
  var max_interest_rate = get_max_interest_rate(inputs.interest_rate_changes);
  var monthly_rate = max_interest_rate / 12;
  var tenure = Math.ceil(Math.log(inputs.emi / (inputs.emi - inputs.loan_amount * monthly_rate)) / Math.log(1 + monthly_rate));

  var loan_end_date = new Date(inputs.first_payment_date);
  loan_end_date.setMonth(loan_end_date.getMonth() + tenure);

  var row_count = tenure + 1 + inputs.extra_payments.length + inputs.interest_rate_changes.length

  inputs.max_interest_rate = max_interest_rate;
  inputs.loan_tenure = tenure;
  inputs.loan_end_date = loan_end_date;
  inputs.row_count = row_count
  return inputs
}


/**
 * Adds months to a date object, handling cases where the target day of month doesn't exist.
 * @param {Date} date The starting Date object.
 * @param {number} monthsToAdd Number of months to add.
 * @param {number} targetDayOfMonth The desired day of the month (e.g., 31).
 * @return {Date} A new Date object with the months added.
 */
function add_months(date, monthsToAdd, targetDayOfMonth) {
  var newDate = new Date(date);
  var currentMonth = newDate.getMonth();
  newDate.setMonth(currentMonth + monthsToAdd);

  if (targetDayOfMonth) {
    var daysInMonth = new Date(newDate.getFullYear(), newDate.getMonth() + 1, 0).getDate(); // Get days in the new month
    if (targetDayOfMonth > daysInMonth) {
      newDate.setDate(daysInMonth); // Set to last day of the month if target day is invalid
    } else {
      newDate.setDate(targetDayOfMonth); // Otherwise, set to target day
    }
  }
  return newDate;
}


/**
 * Captures all event dates for the loan amortization schedule: EMI dates, extra payment dates, and interest rate change dates.
 * @param {object} inputs Validated input object from read_and_validate_inputs().
 * @return {Array<Date>} An array of unique, sorted event dates.
 */
function capture_all_event_dates(inputs) {
  var event_dates = [];
  var loan_start_date = new Date(inputs.loan_start_date);
  var first_payment_date = new Date(inputs.first_payment_date);
  var payment_date_day = inputs.payment_date;
  var extra_payments = inputs.extra_payments;
  var interest_rate_changes = inputs.interest_rate_changes;

  var current_payment_date = new Date(first_payment_date);

  // 1. Capture EMI dates for the entire tenure
  for (let i = 0; i < inputs.loan_tenure; i++) {
    event_dates.push(new Date(current_payment_date)); // Add a copy of the date
    current_payment_date = add_months(current_payment_date, 1, payment_date_day);
  }

  // 2. Capture Extra Payment dates
  if (extra_payments && extra_payments.length > 0) {
    extra_payments.forEach(function (payment) {
      event_dates.push(payment.date);
    });
  }

  // 3. Capture Interest Rate Change dates
  if (interest_rate_changes && interest_rate_changes.length > 0) {
    interest_rate_changes.forEach(function (change) {
      event_dates.push(change.date);
    });
  }

  // 4. Remove duplicate dates and sort
  var unique_event_dates = event_dates.filter(function (date, index, self) {
    return self.findIndex(d => d.toDateString() === date.toDateString()) === index;
  });

  unique_event_dates.sort(function (a, b) {
    return a - b; // Sort dates in ascending order
  });

  inputs.all_events = unique_event_dates

  return inputs;
}
