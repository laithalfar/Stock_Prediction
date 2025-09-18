import yfinance as yf
import pandas as pd

#get apple data
dat = yf.Ticker("AAPL")

#place apple data in a dataframe variables
OHCL = dat.history(period="2y", interval = "1d") # get OHCL data
General_info = pd.DataFrame([dat.info]) # get general info data
analyst_price_targets = pd.DataFrame([dat.analyst_price_targets]) # get the predictions of analysts for OHCL in 12-18months
quarterly_income_stmt =  dat.quarterly_income_stmt # get the quarterly income statement
quarterly_balance_sheet = dat.quarterly_balance_sheet # get the quarterly balance sheet
quarterly_cashflow = dat.quarterly_cashflow # get the quarterly cashflow


# excel does not support timezones so timezobes are removed prior
OHCL.index = OHCL.index.tz_localize(None)


#save the data in a excel file in different sheets for better viewing and analyses
with pd.ExcelWriter("data/raw.xlsx") as writer:
    OHCL.to_excel(writer, sheet_name="AAPL_OHCL")
    General_info.to_excel(writer, sheet_name="AAPL_General_info")
    analyst_price_targets.to_excel(writer, sheet_name="AAPL_analyst_price_targets")
    quarterly_income_stmt.to_excel(writer, sheet_name="AAPL_quarterly_income_stmt")
    quarterly_balance_sheet.to_excel(writer, sheet_name="AAPL_quarterly_balance_sheet")
    quarterly_cashflow.to_excel(writer, sheet_name="AAPL_quarterly_cashflow")

# print("test: ", dat.info.keys())
# print("\nanalyst price targets: ", dat.analyst_price_targets)
# print("\nquarterly income statement: ", dat.quarterly_income_stmt)
# print("\noption chain: ", dat.option_chain().calls)