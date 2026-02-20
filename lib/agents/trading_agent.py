from .base import BaseAgent, AgentType, AgentContext, AgentResponse, TRADING_AGENT_PROMPT

class TradingAgent(BaseAgent):
    def __init__(self, llm_wrapper=None, database=None):
        super().__init__(
            agent_type=AgentType.TRADING,
            system_prompt=TRADING_AGENT_PROMPT,
            tools=["create_strategy", "run_backtest", "get_technical_indicators"],
            llm_wrapper=llm_wrapper
        )
        self.database = database

    async def execute(self, query: str, context: AgentContext) -> AgentResponse:
        query_lower = query.lower()
        
        if "strategy" in query_lower and "create" in query_lower:
            return await self._create_strategy(query, context)
        elif "backtest" in query_lower:
            return await self._run_backtest(query, context)
        elif any(kw in query_lower for kw in ["indicator", "technical", "rsi", "macd"]):
            return await self._get_indicators(query, context)
        else:
            return AgentResponse(
                content="I can help you with:\n- Creating trading strategies\n- Running backtests\n- Getting technical indicators\n\nWhat would you like to do?"
            )

    async def _create_strategy(self, query: str, context: AgentContext) -> AgentResponse:
        import re
        
        name_match = re.search(r'name[:\s]+["\']?([^"\']+)["\']?', query, re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else "New Strategy"
        
        indicators = []
        if "rsi" in query.lower():
            indicators.append("RSI")
        if "macd" in query.lower():
            indicators.append("MACD")
        if "ma" in query.lower() or "moving average" in query.lower():
            indicators.append("SMA")
        if "bollinger" in query.lower():
            indicators.append("BB")
        
        config = {
            "name": name,
            "indicators": indicators,
            "timeframe": "1D",
            "initial_capital": 100000
        }
        
        result = f"### Strategy Created: {name}\n\n"
        result += f"**Configuration:**\n"
        result += f"- Indicators: {', '.join(indicators) if indicators else 'None specified'}\n"
        result += f"- Timeframe: {config['timeframe']}\n"
        result += f"- Initial Capital: ${config['initial_capital']:,}\n\n"
        result += "To run a backtest, use the backtest tool or say 'run backtest'."
        
        return AgentResponse(
            content=result,
            context_used={"action": "create_strategy", "config": config}
        )

    async def _run_backtest(self, query: str, context: AgentContext) -> AgentResponse:
        import re
        
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', query)
        ticker = ticker_match.group(1) if ticker_match else "SPY"
        
        result = f"### Backtest: {ticker}\n\n"
        result += "This is a placeholder for the backtest interface.\n\n"
        result += "**Note:** To run an actual backtest:\n"
        result += "1. Create a strategy first\n"
        result += "2. Use the Strategies page in the UI\n"
        result += "3. Configure parameters and run\n\n"
        result += "Example parameters:\n"
        result += f"- Symbol: {ticker}\n"
        result += "- Period: 1 year\n"
        result += "- Initial Capital: $100,000"
        
        return AgentResponse(
            content=result,
            context_used={"action": "backtest", "ticker": ticker}
        )

    async def _get_indicators(self, query: str, context: AgentContext) -> AgentResponse:
        import re
        
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', query)
        ticker = ticker_match.group(1) if ticker_match else "SPY"
        
        result = f"### Technical Indicators for {ticker}\n\n"
        result += "| Indicator | Value | Signal |\n"
        result += "|-----------|-------|--------|\n"
        result += "| RSI(14) | ~55 | Neutral |\n"
        result += "| MACD | ~0.05 | Bullish |\n"
        result += "| SMA(20) | $450.00 | Price Above |\n"
        result += "| SMA(50) | $445.00 | Price Above |\n"
        result += "| BB(20) | Upper Band | Overbought |\n\n"
        result += "*Note: This is sample data. Connect to market data for real values.*"
        
        return AgentResponse(
            content=result,
            context_used={"action": "indicators", "ticker": ticker}
        )
