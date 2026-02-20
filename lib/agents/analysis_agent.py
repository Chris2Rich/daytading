from .base import BaseAgent, AgentType, AgentContext, AgentResponse, ANALYSIS_AGENT_PROMPT
from ..retrieval.yahoofinance import get_fundamentals, get_key_metrics_timeseries, get_price_targets

class AnalysisAgent(BaseAgent):
    def __init__(self, llm_wrapper=None):
        super().__init__(
            agent_type=AgentType.ANALYSIS,
            system_prompt=ANALYSIS_AGENT_PROMPT,
            tools=["get_fundamentals", "get_key_metrics_timeseries", "get_price_targets", "compare_peers"],
            llm_wrapper=llm_wrapper
        )

    async def execute(self, query: str, context: AgentContext) -> AgentResponse:
        import re
        
        tickers = re.findall(r'\b[A-Z]{1,5}\b', query)
        tickers = [t for t in tickers if t not in ["DCF", "EPS", "PE", "API", "LLM"]]
        
        if not tickers:
            return AgentResponse(
                content="I couldn't identify a stock ticker in your query. Please provide a ticker symbol (e.g., AAPL, MSFT)."
            )
        
        ticker = tickers[0]
        
        query_lower = query.lower()
        
        if "fundament" in query_lower or "financial" in query_lower:
            fundamentals = get_fundamentals(ticker)
            return AgentResponse(
                content=self._format_fundamentals(ticker, fundamentals),
                context_used={"ticker": ticker, "data": "fundamentals"}
            )
        
        elif "ratio" in query_lower or "metric" in query_lower:
            metrics = get_key_metrics_timeseries(ticker, "annual")
            return AgentResponse(
                content=self._format_metrics(ticker, metrics),
                context_used={"ticker": ticker, "data": "metrics"}
            )
        
        elif "target" in query_lower or "recommend" in query_lower:
            targets = get_price_targets(ticker)
            return AgentResponse(
                content=self._format_targets(ticker, targets),
                context_used={"ticker": ticker, "data": "targets"}
            )
        
        else:
            fundamentals = get_fundamentals(ticker)
            targets = get_price_targets(ticker)
            
            combined = f"## Analysis for {ticker}\n\n"
            combined += self._format_fundamentals(ticker, fundamentals)
            combined += "\n\n"
            combined += self._format_targets(ticker, targets)
            
            return AgentResponse(
                content=combined,
                context_used={"ticker": ticker, "data": "combined"}
            )

    def _format_fundamentals(self, ticker: str, data: dict) -> str:
        if not data:
            return f"No fundamental data available for {ticker}"
        
        result = f"### {ticker} Fundamentals\n\n"
        
        if data.get("company_name"):
            result += f"**Company:** {data['company_name']}\n"
        if data.get("sector"):
            result += f"**Sector:** {data['sector']}\n"
        if data.get("industry"):
            result += f"**Industry:** {data['industry']}\n\n"
        
        result += "#### Financials\n"
        for period in ["ttm", "last_fiscal_year", "second_last_fiscal_year"]:
            if data.get(period):
                p_data = data[period]
                result += f"\n**{period.replace('_', ' ').title()}:**\n"
                for k, v in p_data.items():
                    result += f"  - {k}: {v}\n"
        
        return result

    def _format_metrics(self, ticker: str, data: dict) -> str:
        if not data:
            return f"No metrics data available for {ticker}"
        
        result = f"### {ticker} Key Metrics\n\n"
        
        for metric, values in data.items():
            result += f"**{metric}:**\n"
            if isinstance(values, list) and len(values) > 0:
                for v in values[-5:]:
                    result += f"  - {v}\n"
            else:
                result += f"  - {values}\n"
        
        return result

    def _format_targets(self, ticker: str, data: dict) -> str:
        if not data:
            return f"No price target data available for {ticker}"
        
        result = f"### {ticker} Price Targets\n\n"
        
        if data.get("target_mean"):
            result += f"**Mean Target:** ${data['target_mean']}\n"
        if data.get("target_high"):
            result += f"**High Target:** ${data['target_high']}\n"
        if data.get("target_low"):
            result += f"**Low Target:** ${data['target_low']}\n"
        if data.get("rating"):
            result += f"**Consensus Rating:** {data['rating']}\n"
        
        return result
