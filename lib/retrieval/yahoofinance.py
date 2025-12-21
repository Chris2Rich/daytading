import yfinance as yf
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import os
os.environ['YFINANCE_QUIET'] = '1'

def get_trading(ticker: str, days: int, interval: str):
    """
    Fetch trading data for a ticker.
    
    Args:
        ticker: Stock symbol
        days: Number of days of data to fetch
        interval: Data interval - 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
    
    Returns:
        DataFrame with OHLCV data
    """
    end_date = datetime.now(tz=timezone.utc)
    
    # Yahoo Finance has limits on intraday data availability
    max_days_for_interval = {
        "1m": 7,
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "1h": 730,
        "1d": 10000,
        "1wk": 10000,
        "1mo": 10000,
    }
    
    max_days = max_days_for_interval.get(interval, days)
    actual_days = min(days, max_days)
    
    # For intraday data (1 day), fetch extra days to ensure we get the last trading day
    # This handles weekends and holidays
    if days == 1 and interval in ("1m", "5m", "15m", "30m", "1h"):
        # Fetch up to 5 days to cover weekends + potential holidays
        fetch_days = 5
        start_date = end_date - timedelta(days=fetch_days)
        
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False
        )
        
        if data.empty:
            return data
        
        # Get the last trading day in the data
        last_trading_day = data.index[-1].date()
        
        # Filter to only include data from the last trading day
        data = data[data.index.date == last_trading_day]
        
        return data
    
    start_date = end_date - timedelta(days=actual_days)
    
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False
    )
    
    return data
   
def get_option(ticker: str, expiry: str, direction: str, strike: float):
    optionstring = ticker + expiry + direction + f"{int(strike * 1000):08d}"
    asset = yf.Ticker(optionstring)
    return asset.history()

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

def get_price_targets(ticker: str):
    ticker = yf.Ticker(ticker)

    recommendations = ticker.recommendations
    target_price = ticker.info.get('targetMeanPrice')
    rating = ticker.info.get('recommendationKey')
    return {
        'ticker': ticker,
        'target_mean_price': target_price,
        'recommendation': rating,
        'recommendation_history': recommendations.tail(10)  # last 10
    }

def get_key_metrics_timeseries(ticker: str, period: str = "quarterly"):
    """
    Get common financial metrics over time.
    
    Returns dict with multiple timeseries.
    """
    stock = yf.Ticker(ticker)
    
    # Get statements
    if period == "quarterly":
        income = stock.quarterly_income_stmt
        balance = stock.quarterly_balance_sheet
        cashflow = stock.quarterly_cashflow
    else:
        income = stock.income_stmt
        balance = stock.balance_sheet
        cashflow = stock.cashflow
    
    def extract_metric(df, metric_name):
        """Helper to extract a metric from dataframe"""
        # Try different variations of the metric name
        if df is None:
            return {}

        variations = [
            metric_name,
            metric_name.title(),
            metric_name.replace('_', ' ').title(),
            ' '.join(word.capitalize() for word in metric_name.split('_'))
        ]
        
        for var in variations:
            if var in df.index:
                return df.loc[var].to_dict()
        
        # Try fuzzy match
        for idx in df.index:
            if metric_name.lower() in idx.lower():
                return df.loc[idx].to_dict()
        
        return {}

    # Extract key metrics
    metrics = {
        "revenue": extract_metric(income, "Total Revenue"),
        "net_income": extract_metric(income, "Net Income"),
        "operating_income": extract_metric(income, "Operating Income"),
        "gross_profit": extract_metric(income, "Gross Profit"),
        "total_assets": extract_metric(balance, "Total Assets"),
        "total_debt": extract_metric(balance, "Total Debt"),
        "free_cash_flow": extract_metric(cashflow, "Free Cash Flow"),
    }
    
    # Convert to consistent format
    result = {}
    for metric_name, data in metrics.items():
        result[metric_name] = [
            {
                "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                "value": float(value) if pd.notna(value) else None
            }
            for date, value in data.items()
        ]
        # Sort by date
        result[metric_name].sort(key=lambda x: x["date"])
    
    return result