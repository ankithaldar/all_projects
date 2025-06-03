function GET_LAST_WORKING_DAY(passed_date) {
  var lwd = new Date(Date.parse(passed_date));
  var lwd = new Date(lwd.getFullYear(), lwd.getMonth() + 1, 0);

  if (lwd.getDay() === 0) {
    lwd = new Date(lwd.getFullYear(), lwd.getMonth(), lwd.getDate() - 2);
  } else if (lwd.getDay() === 6) {
    lwd = new Date(lwd.getFullYear(), lwd.getMonth(), lwd.getDate() - 1);
  }

  return lwd.getDate();
};
