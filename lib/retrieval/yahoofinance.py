import yfinance as yf
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

def get_ticker_news(ticker: str):
    news_items = yf.Search(ticker, news_count=5).news
    cleaned_articles = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for item in news_items:
        link = item.get("link")
        title = item.get("title")
        if not link or not title:
            continue

        try:
            response = requests.get(link, headers=headers, timeout=3)
            response.raise_for_status()
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            article_content = ""
            for s in soup(['script', 'style', 'header', 'footer', 'nav']):
                s.decompose()
            
            body_html = soup.body
            if body_html:
                article_content = md(str(body_html), heading_style="ATX").strip()

            if article_content:
                cleaned_articles.append({"title": title, "content": article_content})

        except Exception as e:
            continue
            
    return cleaned_articles

def get_trading(ticker: str, days: int, interval: str):
    period = f"{min(days, 7)}d" if interval != "1d" and interval != "1mo" else f"{days * 2}d"
    data = yf.download(ticker, period=period, interval=interval)
    if interval != "1d":
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
        data = data[data.index >= pd.to_datetime(cutoff)]

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