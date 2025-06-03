/** @OnlyCurrentDoc */

// ------------------ Leap Year Logic ------------------
const is_leap_year = (year) =>
  (year % 4 === 0 && year % 100 !== 0) || (year % 400 === 0);

// ------------------ Core Configs ------------------

const YEAR = 2026;  // Define year separately for leap year check

const CONFIG = {
  years: YEAR,

  // Banks
  banks: ['HDFC', 'SBI_Own', 'SBI_Joint'],
  salary_bank: 'HDFC',

  // Amounts
  salary_amount: 000000,
  meal_card_amount: 000000,
  home_emi: 000000,
  a_a_emi: 000000,

  month_end_salary: {
    'Maid Monthly Salary': 000000,
    'Car Washing Salary': 000000
  },

  // Migration
  old_sheet_link: '',

  // Debug options
  if_print: false,
  if_debug: false,

  // Month and date settings
  month_days: {
    'Jan': 31,
    'Feb': is_leap_year(YEAR) ? 29 : 28,
    'Mar': 31,
    'Apr': 30,
    'May': 31,
    'Jun': 30,
    'Jul': 31,
    'Aug': 31,
    'Sep': 30,
    'Oct': 31,
    'Nov': 30,
    'Dec': 31,
  },

  // tab templates to hide
  hidden_tabs: ['Template', 'BankStatement', 'CCStatement-Template'],

  // move tabs to end of sheet
  move_tabs: ['ImpEvents', 'Analytics'],

  // Credit Card Configuration
  card_start_row: 25,
  card_map: {
    'CC Cred Flash':    {'bill_date':  1, 'sheet_cell': '$T$26', 'repayment_days':  4, 'last_digit_round': false },
    'CC IndusInd 0596': {'bill_date':  3, 'sheet_cell': '$T$27', 'repayment_days': 20, 'last_digit_round': true  },
    'CC Kotak 5752':    {'bill_date':  5, 'sheet_cell': '$T$28', 'repayment_days': 20, 'last_digit_round': false },
    'CC SC 8148':       {'bill_date':  8, 'sheet_cell': '$T$29', 'repayment_days': 22, 'last_digit_round': false },
    'CC Axis 6599':     {'bill_date': 11, 'sheet_cell': '$T$30', 'repayment_days': 19, 'last_digit_round': false },
    'CC HDFC 6454':     {'bill_date': 12, 'sheet_cell': '$T$31', 'repayment_days': 20, 'last_digit_round': true  },
    'CC ICICI 7007':    {'bill_date': 14, 'sheet_cell': '$T$32', 'repayment_days': 18, 'last_digit_round': false },
    'CC One 0531':      {'bill_date': 14, 'sheet_cell': '$T$33', 'repayment_days': 17, 'last_digit_round': false },
    'CC Yes Bank 3580': {'bill_date': 21, 'sheet_cell': '$T$34', 'repayment_days': 19, 'last_digit_round': false },
    'CC Axis 7878':     {'bill_date': 21, 'sheet_cell': '$T$35', 'repayment_days': 20, 'last_digit_round': false },
    'CC Axis 1879':     {'bill_date': 21, 'sheet_cell': '$T$36', 'repayment_days': 20, 'last_digit_round': false },
    'CC HSBC 0494':     {'bill_date': 22, 'sheet_cell': '$T$37', 'repayment_days': 20, 'last_digit_round': false },
    'CC AU 3806':       {'bill_date': 23, 'sheet_cell': '$T$38', 'repayment_days': 20, 'last_digit_round': false },
    'CC Scapia 0700':   {'bill_date': 25, 'sheet_cell': '$T$39', 'repayment_days': 18, 'last_digit_round': false },
    'CC ICICI 8019':    {'bill_date': 28, 'sheet_cell': '$T$40', 'repayment_days': 18, 'last_digit_round': false }
  },
  meal_card: 'CC Pluxee 6314',

  sheet_cache: SpreadsheetApp.getActiveSpreadsheet(),
};

Object.freeze(CONFIG);
Object.freeze(CONFIG.card_map);
