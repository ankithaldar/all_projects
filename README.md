# Create Expenditure File for yearly expenditure

This project contains of 4 files

1. __main.js<br/>__
  This file is the controller and high level abstraction for different executions that are needed for creating the expenditure sheet once every year.<br/>

2. __make_new_sheets.js<br/>__
  This creates required sheets and formulaes to track the budget and expenditures.<br/>
  Creates Bank statements, monthly tracker and adds formulaes.<br/>
  contains mostly ulitity functions.<br/>
  *Note: Not to be run individually. Fully controlled from main.js*<br/>

3. __mark_various_dates.js<br/>__
  This marks the different dates for payments in bank statement to be made in credit card bills, salaries (incoming and outgoing).<br/>
  contains mostly ulitity functions.<br/>
  *Note: Not to be run individually. Fully controlled from main.js*<br/>

4. __utis.js<br/>__
  contains only ulitity functions.<br/>
  *Note: Not to be run individually. Fully controlled from main.js*<br/>


<br/><br/><br/>
```
Note:
If new files are added please add the usage of the files here
```

<br/><br/>
---

Issues:<br/>

1. CC statement for some months is not alligning properly<br/>

<br/><br/>
---


```
Note:
To make please create a new branch in format gsheet_expenditure_file/<change_date> and merge to gsheet_expenditure_file/main using PR
```
