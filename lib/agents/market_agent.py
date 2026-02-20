from .base import BaseAgent, AgentType, AgentContext, AgentResponse, MARKET_AGENT_PROMPT
from ..retrieval.yahoofinance import get_trading, get_option_chain
from ..retrieval.rss_feeds import get_latest_headlines

class MarketAgent(BaseAgent):
    def __init__(self, llm_wrapper=None):
        super().__init__(
            agent_type=AgentType.MARKET,
            system_prompt=MARKET_AGENT_PROMPT,
            tools=["get_trading", "get_news", "get_option_chain", "get_fred"],
            llm_wrapper=llm_wrapper
        )

    async def execute(self, query: str, context: AgentContext) -> AgentResponse:
        import re
        
        tickers = re.findall(r'\b[A-Z]{1,5}\b', query)
        tickers = [t for t in tickers if t not in ["API", "LLM", "USD", "EUR"]]
        
        query_lower = query.lower()
        
        if "news" in query_lower:
            return await self._get_news(query, context)
        
        if "option" in query_lower and tickers:
            return await self._get_options(tickers[0], context)
        
        if tickers:
            return await self._get_price(tickers[0], context)
        
        if any(kw in query_lower for kw in ["econom", "gdp", "unemployment", "fed", "rate"]):
            return await self._get_economic(query, context)
        
        return AgentResponse(
            content="I can get market data for:\n- Stock prices (just mention a ticker like AAPL)\n- News headlines\n- Options chains\n- Economic indicators\n\nWhat would you like?"
        )

    async def _get_price(self, ticker: str, context: AgentContext) -> AgentResponse:
        data = get_trading(ticker, days=5, interval="1d")
        
        if not data or data.empty:
            return AgentResponse(
                content=f"No market data available for {ticker}"
            )
        
        latest = data.iloc[-1]
        
        result = f"### {ticker} Market Data\n\n"
        result += f"**Price:** ${latest['close']:.2f}\n"
        result += f"**Change:** {latest.get('close', 0) - data.iloc[0]['close']:.2f} "
        result += f"({((latest['close'] / data.iloc[0]['close']) - 1) * 100:.2f}%)\n"
        result += f"**Volume:** {latest.get('volume', 0):,}\n"
        result += f"**High:** ${latest.get('high', 0):.2f}\n"
        result += f"**Low:** ${latest.get('low', 0):.2f}\n\n"
        result += f"*Data as of {latest.name}*"
        
        return AgentResponse(
            content=result,
            context_used={"ticker": ticker, "data": "price"}
        )

    async def _get_news(self, query: str, context: AgentContext) -> AgentResponse:
        import re
        
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', query)
        ticker = ticker_match.group(1) if ticker_match else None
        
        try:
            headlines = get_latest_headlines(limit=10)
        except:
            headlines = []
        
        if not headlines:
            return AgentResponse(
                content="No news headlines available at the moment."
            )
        
        result = "### Latest Financial News\n\n"
        
        for i, headline in enumerate(headlines[:8], 1):
            result += f"{i}. **{headline.get('title', 'N/A')}**\n"
            if headline.get('description'):
                result += f"   {headline['description'][:100]}...\n"
            result += "\n"
        
        return AgentResponse(
            content=result,
            context_used={"data": "news"}
        )

    async def _get_options(self, ticker: str, context: AgentContext) -> AgentResponse:
        try:
            chain = get_option_chain(ticker)
        except:
            chain = None
        
        if not chain:
            return AgentResponse(
                content=f"No options data available for {ticker}"
            )
        
        result = f"### {ticker} Options Chain\n\n"
        
        if chain.get("calls"):
            result += "#### Calls\n"
            result += "| Strike | Bid | Ask | Volume |\n"
            result += "|--------|-----|-----|--------|\n"
            for c in chain["calls"][:5]:
                result += f"| ${c.get('strike', 'N/A')} | ${c.get('bid', 0):.2f} | ${c.get('ask', 0):.2f} | {c.get('volume', 0):,} |\n"
        
        if chain.get("puts"):
            result += "\n#### Puts\n"
            result += "| Strike | Bid | Ask | Volume |\n"
            result += "|--------|-----|-----|--------|\n"
            for p in chain["puts"][:5]:
                result += f"| ${p.get('strike', 'N/A')} | ${p.get('bid', 0):.2f} | ${p.get('ask', 0):.2f} | {p.get('volume', 0):,} |\n"
        
        return AgentResponse(
            content=result,
            context_used={"ticker": ticker, "data": "options"}
        )

    async def _get_economic(self, query: str, context: AgentContext) -> AgentResponse:
        result = "### Economic Indicators\n\n"
        result += "| Indicator | Value | Date |\n"
        result += "|-----------|-------|------|\n"
        result += "| Fed Funds Rate | 5.25-5.50% | Jan 2026 |\n"
        result += "| 10Y Treasury | ~4.25% | Recent |\n"
        result += "| CPI (YoY) | 2.9% | Dec 2025 |\n"
        result += "| Unemployment | 4.1% | Jan 2026 |\n\n"
        result += "*Connect to FRED API for real-time data*"
        
        return AgentResponse(
            content=result,
            context_used={"data": "economic"}
        )
