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

from lib.retrieval.yahoofinance import get_price_targets, get_key_metrics_timeseries, get_trading

f = open("./data/cik/cikmap.json")
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
        Fetch exactly two historical time series:
        1) Weekly OHLC price data over trailing 1 year
        2) Quarterly fundamentals over trailing 2 years
        """

        # ============================================================
        # 1) WEEKLY PRICE TIME SERIES — 1 YEAR
        # ============================================================
        weekly_df = get_trading(
            ticker=ticker,
            days=365,
            interval="1wk"
        )

        weekly_price_timeseries = []
        if weekly_df is not None and not weekly_df.empty:
            for idx, row in weekly_df.iterrows():
                weekly_price_timeseries.append({
                    "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]) if "Volume" in row else None,
                })

        # ============================================================
        # 2) QUARTERLY FUNDAMENTALS — 2 YEARS (8 QUARTERS)
        # ============================================================
        fundamentals_q = get_key_metrics_timeseries(
            ticker=ticker,
            period="quarterly"
        )

        quarterly_fundamentals_2y = {}
        for metric, series in (fundamentals_q or {}).items():
            # keep last 8 quarters only
            quarterly_fundamentals_2y[metric] = series[-8:] if len(series) > 8 else series

        # ============================================================
        # RETURN STRUCTURE
        # ============================================================

        price_targets_info = get_price_targets(ticker)
        targets_df = price_targets_info.get("recommendation_history")
        if targets_df is not None and not targets_df.empty:
            targets = targets_df.to_dict(orient='records')
        else:
            targets = []

        return {
            "ticker": ticker,
            "weekly_price_timeseries": weekly_price_timeseries,
            "quarterly_fundamentals": quarterly_fundamentals_2y,
            "price_targets": targets,
        }

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
            
            formatted = json.dumps(result)
            if formatted and 'error' not in result:
                contexts.append(formatted)
                valid_tickers.append(tickers[i])
        
        if not contexts:
            return "", []
        
        # Combine all contexts
        combined_context = "\n\n".join(contexts)
        return combined_context, valid_tickers

