import yfinance as yf
from datetime import datetime, timedelta, timezone
import pandas as pd

def get_trading(ticker: str, days: int, interval: str):
    period = f"{min(days, 7)}d" if interval != "1d" and interval != "1mo" else f"{days * 2}d"
    data = yf.download(ticker, period=period, interval=interval)
    if interval != "1d":
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
        data = data[data.index >= pd.to_datetime(cutoff)]

    return data

def get_option_chain(ticker: str, expiry: str):
    stock = yf.Ticker(ticker)
    option_chain = stock.option_chain(expiry)
    return {
        'calls': option_chain.calls.to_dict(orient='records'),
        'puts': option_chain.puts.to_dict(orient='records')
    }

def get_fundamentals(ticker: str, statement_type: str, period: str):
    stock = yf.Ticker(ticker)
    if statement_type == 'income_statement':
        if period == 'annual':
            return stock.income_stmt.to_dict()
        else:
            return stock.quarterly_income_stmt.to_dict()
    elif statement_type == 'balance_sheet':
        if period == 'annual':
            return stock.balance_sheet.to_dict()
        else:
            return stock.quarterly_balance_sheet.to_dict()