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
        Fetch rich, time-aware market data for a ticker.
        Emphasizes historical context across multiple horizons.
        """

        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            # ---- PRICE HISTORY (MULTI-HORIZON) ----
            now = datetime.now(tz=timezone.utc)

            price_history = {}

            def fetch_history(days: int, interval: str):
                start = now - timedelta(days=days)
                df = yf.download(
                    ticker,
                    start=start,
                    end=now,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
                return df if not df.empty else None

            price_history["1d_intraday"] = fetch_history(5, "5m")
            price_history["7d"] = fetch_history(7, "1d")
            price_history["30d"] = fetch_history(30, "1d")
            price_history["6mo"] = fetch_history(180, "1wk")
            price_history["1y"] = fetch_history(365, "1wk")
            price_history["5y"] = fetch_history(1825, "1mo")

            # ---- PRICE DERIVED METRICS ----
            def summarize(df: pd.DataFrame):
                if df is None or len(df) < 2:
                    return None

                first = float(df.iloc[0]["Close"])
                last = float(df.iloc[-1]["Close"])

                return {
                    "start_price": first,
                    "end_price": last,
                    "return_pct": ((last - first) / first) * 100 if first else None,
                    "high": float(df["High"].max()),
                    "low": float(df["Low"].min()),
                    "volatility_pct": float(df["Close"].pct_change().std() * 100),
                }

            price_summaries = {
                horizon: summarize(df)
                for horizon, df in price_history.items()
            }

            # ---- CURRENT PRICE SNAPSHOT ----
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            previous_close = info.get("previousClose")

            change = None
            change_percent = None
            if current_price and previous_close:
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100 if previous_close else None

            # ---- FUNDAMENTALS (TIME-SERIES) ----
            try:
                fundamentals_quarterly = get_key_metrics_timeseries(ticker, "quarterly")
                fundamentals_annual = get_key_metrics_timeseries(ticker, "annual")
            except Exception:
                fundamentals_quarterly = None
                fundamentals_annual = None

            # ---- FINANCIAL STATEMENTS (RAW) ----
            try:
                income_stmt = stock.income_stmt
                balance_sheet = stock.balance_sheet
                cashflow = stock.cashflow
            except Exception:
                income_stmt = balance_sheet = cashflow = None

            # ---- ANALYST DATA ----
            try:
                price_targets = get_price_targets(ticker)
            except Exception:
                price_targets = None

            return {
                "ticker": ticker,

                # snapshot
                "current_price": current_price,
                "previous_close": previous_close,
                "change": change,
                "change_percent": change_percent,
                "volume": info.get("volume") or info.get("averageVolume"),
                "market_cap": info.get("marketCap"),

                # time-aware pricing
                "price_history": price_history,
                "price_summaries": price_summaries,

                # fundamentals over time
                "fundamentals_quarterly": fundamentals_quarterly,
                "fundamentals_annual": fundamentals_annual,

                # raw statements
                "income_stmt": income_stmt,
                "balance_sheet": balance_sheet,
                "cashflow": cashflow,

                # analyst sentiment
                "price_targets": price_targets,

                # raw info passthrough
                "info": info,
            }

        except Exception as e:
            print(f"Error fetching market data for {ticker}: {e}")
            return {
                "ticker": ticker,
                "error": str(e),
            }

    def format_context(self, data: Dict[str, Any]) -> str:
        """
        Format rich, time-aware market data into an LLM-optimized context.
        Emphasizes comparison across time horizons.
        """

        if "error" in data:
            return f"Error fetching data for {data['ticker']}: {data['error']}"

        ticker = data["ticker"]
        info = data.get("info", {})

        ctx = []
        ctx.append(f"# Market Data for {ticker}\n")

        # ============================================================
        # CURRENT SNAPSHOT
        # ============================================================
        ctx.append("## Current Market Snapshot")

        if data.get("current_price") is not None:
            ctx.append(f"- **Price**: ${data['current_price']:.2f}")

        if data.get("change") is not None and data.get("change_percent") is not None:
            sign = "+" if data["change"] >= 0 else ""
            ctx.append(
                f"- **Day Change**: {sign}${data['change']:.2f} "
                f"({sign}{data['change_percent']:.2f}%)"
            )

        if data.get("volume"):
            vol = data["volume"]
            vol_str = (
                f"{vol/1e9:.2f}B" if vol >= 1e9
                else f"{vol/1e6:.2f}M" if vol >= 1e6
                else f"{vol:,.0f}"
            )
            ctx.append(f"- **Volume**: {vol_str}")

        if data.get("market_cap"):
            cap = data["market_cap"]
            cap_str = (
                f"${cap/1e12:.2f}T" if cap >= 1e12
                else f"${cap/1e9:.2f}B" if cap >= 1e9
                else f"${cap/1e6:.2f}M"
            )
            ctx.append(f"- **Market Cap**: {cap_str}")

        ctx.append("")

        # ============================================================
        # PRICE PERFORMANCE OVER TIME
        # ============================================================
        ctx.append("## Price Performance by Time Horizon")

        summaries = data.get("price_summaries", {})

        def fmt_return(summary):
            if not summary or summary.get("return_pct") is None:
                return "N/A"
            return f"{summary['return_pct']:+.2f}%"

        horizon_labels = {
            "7d": "Last 7 Days",
            "30d": "Last 30 Days",
            "6mo": "Last 6 Months",
            "1y": "Last 1 Year",
            "5y": "Last 5 Years",
        }

        for key, label in horizon_labels.items():
            s = summaries.get(key)
            if not s:
                continue

            ctx.append(f"### {label}")
            ctx.append(f"- **Return**: {fmt_return(s)}")
            ctx.append(
                f"- **High / Low**: ${s['high']:.2f} / ${s['low']:.2f}"
                if s.get("high") is not None else "- **High / Low**: N/A"
            )
            if s.get("volatility_pct") is not None:
                ctx.append(f"- **Volatility**: {s['volatility_pct']:.2f}%")
            ctx.append("")

        # ============================================================
        # FUNDAMENTAL TRENDS
        # ============================================================
        ctx.append("## Fundamental Trends")

        def summarize_timeseries(series, name):
            if not series or len(series) < 2:
                return None

            start = series[0]["value"]
            end = series[-1]["value"]

            if start is None or end is None:
                return None

            change_pct = ((end - start) / abs(start)) * 100 if start else None

            return {
                "name": name,
                "start": start,
                "end": end,
                "change_pct": change_pct,
            }

        quarterly = data.get("fundamentals_quarterly") or {}
        annual = data.get("fundamentals_annual") or {}

        key_metrics = [
            ("revenue", "Revenue"),
            ("net_income", "Net Income"),
            ("eps", "EPS"),
            ("free_cash_flow", "Free Cash Flow"),
        ]

        for metric, label in key_metrics:
            q_summary = summarize_timeseries(quarterly.get(metric), label)
            a_summary = summarize_timeseries(annual.get(metric), label)

            if q_summary:
                ctx.append(
                    f"- **{label} (Quarterly Trend)**: "
                    f"{q_summary['change_pct']:+.1f}% over period"
                )

            if a_summary:
                ctx.append(
                    f"- **{label} (Annual Trend)**: "
                    f"{a_summary['change_pct']:+.1f}% over period"
                )

        ctx.append("")

        # ============================================================
        # VALUATION & MULTIPLES
        # ============================================================
        ctx.append("## Valuation Metrics")

        if info.get("trailingPE"):
            ctx.append(f"- **P/E (TTM)**: {info['trailingPE']:.2f}")

        if info.get("trailingEps"):
            ctx.append(f"- **EPS (TTM)**: ${info['trailingEps']:.2f}")

        if info.get("fiftyTwoWeekLow") and info.get("fiftyTwoWeekHigh"):
            ctx.append(
                f"- **52-Week Range**: "
                f"${info['fiftyTwoWeekLow']:.2f} – ${info['fiftyTwoWeekHigh']:.2f}"
            )

        ctx.append("")

        # ============================================================
        # ANALYST SENTIMENT
        # ============================================================
        pt = data.get("price_targets")
        if pt:
            ctx.append("## Analyst Sentiment")

            if pt.get("target_mean_price"):
                upside = None
                if data.get("current_price"):
                    upside = (
                        (pt["target_mean_price"] - data["current_price"])
                        / data["current_price"] * 100
                    )

                ctx.append(
                    f"- **Mean Price Target**: ${pt['target_mean_price']:.2f}"
                    + (f" ({upside:+.1f}% upside)" if upside else "")
                )

            if pt.get("recommendation"):
                ctx.append(f"- **Consensus Rating**: {pt['recommendation'].upper()}")

            ctx.append("")

        # ============================================================
        # GUIDANCE TO MODEL
        # ============================================================
        ctx.append(
            "Use the multi-horizon price performance and fundamental trends above "
            "to reason about recent performance versus historical context."
        )

        return "\n".join(ctx)

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

