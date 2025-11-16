import yfinance as yf
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

def get_trading(ticker: str, days: int, interval: str):
    period = f"{min(days, 7)}d" if interval != "1d" and interval != "1mo" else f"{days * 2}d"
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    if interval != "1d":
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
        data = data[data.index >= pd.to_datetime(cutoff)]

    data = data.reset_index()
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
        earnings = stock.quarterly_earnings
    else:
        income = stock.income_stmt
        balance = stock.balance_sheet
        cashflow = stock.cashflow
        earnings = stock.earnings
    
    def extract_metric(df, metric_name):
        """Helper to extract a metric from dataframe"""
        # Try different variations of the metric name
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
        "eps": earnings["Earnings"].to_dict() if "Earnings" in earnings.columns else {},
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