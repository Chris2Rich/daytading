import re
import asyncio
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd
import json

os.environ['YFINANCE_QUIET'] = '1'

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from lib.retrieval.yahoofinance import get_price_targets

f = open("./daytading/data/cik/cikmap.json")
tickers = list((json.load(f).keys()))

class MarketDataTool:
    """Tool for fetching and formatting market data and fundamentals"""

    def __init__(self):
        pass

    def detect_tickers(self, user_message: str) -> List[str]:
        """
        Extract stock ticker symbols from user message.
        Looks for uppercase 1-5 letter ticker symbols.
        """
        matches = list(filter(lambda s: f" {s} " in user_message, tickers))
        
        return matches[:10]

    async def fetch_market_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch comprehensive market data for a ticker.
        Returns trading data, fundamentals, and price targets.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Fetch trading data (30 days, daily interval)
            end_date = datetime.now(tz=timezone.utc)
            start_date = end_date - timedelta(days=30)
            
            trading_data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                progress=False
            )
            
            # Get current price info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            previous_close = info.get('previousClose')
            change = None
            change_percent = None
            if current_price and previous_close:
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100 if previous_close else None
            
            # Get fundamentals
            try:
                income_stmt = stock.income_stmt
                balance_sheet = stock.balance_sheet
            except:
                income_stmt = None
                balance_sheet = None
            
            # Get price targets
            try:
                price_targets = get_price_targets(ticker)
            except:
                price_targets = None
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'previous_close': previous_close,
                'change': change,
                'change_percent': change_percent,
                'volume': info.get('volume') or info.get('averageVolume'),
                'market_cap': info.get('marketCap'),
                'trading_data': trading_data,
                'income_stmt': income_stmt,
                'balance_sheet': balance_sheet,
                'price_targets': price_targets,
                'info': info
            }
        except Exception as e:
            print(f"Error fetching market data for {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e)
            }

    def format_context(self, data: Dict[str, Any]) -> str:
        """
        Format market data into a readable context string for the LLM.
        """
        if 'error' in data:
            return f"Error fetching data for {data['ticker']}: {data['error']}"
        
        ticker = data['ticker']
        context = f"# Market Data for {ticker}\n\n"
        
        # Current price information
        if data.get('current_price'):
            context += f"## Current Price Information\n"
            context += f"- **Current Price**: ${data['current_price']:.2f}\n"
            if data.get('previous_close'):
                context += f"- **Previous Close**: ${data['previous_close']:.2f}\n"
            if data.get('change') is not None:
                change_str = f"+${data['change']:.2f}" if data['change'] >= 0 else f"${data['change']:.2f}"
                context += f"- **Change**: {change_str}"
                if data.get('change_percent') is not None:
                    pct_str = f"+{data['change_percent']:.2f}%" if data['change_percent'] >= 0 else f"{data['change_percent']:.2f}%"
                    context += f" ({pct_str})\n"
                else:
                    context += "\n"
            if data.get('volume'):
                vol = data['volume']
                if vol >= 1_000_000_000:
                    vol_str = f"{vol / 1_000_000_000:.2f}B"
                elif vol >= 1_000_000:
                    vol_str = f"{vol / 1_000_000:.2f}M"
                else:
                    vol_str = f"{vol:,.0f}"
                context += f"- **Volume**: {vol_str}\n"
            if data.get('market_cap'):
                cap = data['market_cap']
                if cap >= 1_000_000_000_000:
                    cap_str = f"${cap / 1_000_000_000_000:.2f}T"
                elif cap >= 1_000_000_000:
                    cap_str = f"${cap / 1_000_000_000:.2f}B"
                elif cap >= 1_000_000:
                    cap_str = f"${cap / 1_000_000:.2f}M"
                else:
                    cap_str = f"${cap:,.0f}"
                context += f"- **Market Cap**: {cap_str}\n"
            context += "\n"
        
        # Trading data summary
        if data.get('trading_data') is not None and not data['trading_data'].empty:
            trading = data['trading_data']
            context += f"## Recent Trading Data (Last 30 Days)\n"
            if len(trading) > 0:
                latest = trading.iloc[-1]
                context += f"- **Latest Close**: ${latest['Close']:.2f}\n"
                context += f"- **High**: ${latest['High']:.2f}\n"
                context += f"- **Low**: ${latest['Low']:.2f}\n"
                if len(trading) > 1:
                    first_close = trading.iloc[0]['Close']
                    last_close = trading.iloc[-1]['Close']
                    period_change = ((last_close - first_close) / first_close) * 100 if first_close else None
                    if period_change is not None:
                        context += f"- **30-Day Change**: {period_change:+.2f}%\n"
            context += "\n"
        
        # Key fundamentals
        info = data.get('info', {})
        if info:
            context += f"## Key Fundamentals\n"
            
            # Revenue
            if 'totalRevenue' in info:
                rev = info['totalRevenue']
                if rev:
                    rev_str = f"${rev / 1_000_000_000:.2f}B" if rev >= 1_000_000_000 else f"${rev / 1_000_000:.2f}M"
                    context += f"- **Total Revenue**: {rev_str}\n"
            
            # Net Income
            if 'netIncomeToCommon' in info:
                ni = info['netIncomeToCommon']
                if ni:
                    ni_str = f"${ni / 1_000_000_000:.2f}B" if abs(ni) >= 1_000_000_000 else f"${ni / 1_000_000:.2f}M"
                    context += f"- **Net Income**: {ni_str}\n"
            
            # P/E Ratio
            if 'trailingPE' in info and info['trailingPE']:
                context += f"- **P/E Ratio**: {info['trailingPE']:.2f}\n"
            
            # EPS
            if 'trailingEps' in info and info['trailingEps']:
                context += f"- **EPS (TTM)**: ${info['trailingEps']:.2f}\n"
            
            # Dividend Yield
            if 'dividendYield' in info and info['dividendYield']:
                context += f"- **Dividend Yield**: {info['dividendYield'] * 100:.2f}%\n"
            
            # 52 Week Range
            if 'fiftyTwoWeekLow' in info and 'fiftyTwoWeekHigh' in info:
                context += f"- **52 Week Range**: ${info['fiftyTwoWeekLow']:.2f} - ${info['fiftyTwoWeekHigh']:.2f}\n"
            
            context += "\n"
        
        # Price targets and recommendations
        if data.get('price_targets'):
            pt = data['price_targets']
            context += f"## Analyst Recommendations\n"
            if pt.get('target_mean_price'):
                context += f"- **Price Target (Mean)**: ${pt['target_mean_price']:.2f}\n"
            if pt.get('recommendation'):
                context += f"- **Recommendation**: {pt['recommendation'].upper()}\n"
            context += "\n"
        
        context += "Use this market data to provide accurate, up-to-date information about the stock.\n"
        return context

    async def fetch_data(self, user_message: str) -> tuple[str, List[str]]:
        """
        Main method: detect tickers, fetch data, and format context.
        Returns (formatted_context, list_of_tickers)
        """
        tickers = self.detect_tickers(user_message)
        
        if not tickers:
            return "", []
        
        # Fetch data for all detected tickers
        tasks = [self.fetch_market_data(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Format context for each ticker
        contexts = []
        valid_tickers = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error fetching data for {tickers[i]}: {result}")
                continue
            
            formatted = self.format_context(result)
            if formatted and 'error' not in result:
                contexts.append(formatted)
                valid_tickers.append(tickers[i])
        
        if not contexts:
            return "", []
        
        # Combine all contexts
        combined_context = "\n\n".join(contexts)
        return combined_context, valid_tickers

