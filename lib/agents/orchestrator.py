from typing import Dict, Any, List, Optional
from .base import BaseAgent, AgentType, AgentContext, AgentResponse
from .research_agent import ResearchAgent
from .analysis_agent import AnalysisAgent
from .trading_agent import TradingAgent
from .market_agent import MarketAgent
import re

INTENT_PATTERNS = {
    AgentType.RESEARCH: [
        r'\b(paper|arxiv|academic|research|study)\b',
        r'\b(explain|what is|how does|concept)\b',
        r'\b(wiki|wikipedia)\b',
        r'\b(theory|model|framework)\b',
    ],
    AgentType.ANALYSIS: [
        r'\b(analyze|analysis|fundament)\b',
        r'\b(valuation|fair value|dcf)\b',
        r'\b(ratio|pe |eps|earnings)\b',
        r'\b(metric|financials)\b',
    ],
    AgentType.TRADING: [
        r'\b(strategy|backtest|trading)\b',
        r'\b(indicator|signal|entry|exit)\b',
        r'\b(position|stop loss|take profit)\b',
    ],
    AgentType.MARKET: [
        r'\b(price|current|latest|today)\b',
        r'\b(news|headline|market)\b',
        r'\b(option|chain|strike)\b',
        r'\b(econom|gdp|unemployment|fed|rate)\b',
    ],
}

class OrchestratorAgent(BaseAgent):
    def __init__(
        self,
        research_agent: ResearchAgent = None,
        analysis_agent: AnalysisAgent = None,
        trading_agent: TradingAgent = None,
        market_agent: MarketAgent = None,
        llm_wrapper = None
    ):
        super().__init__(
            agent_type=AgentType.ORCHESTRATOR,
            system_prompt="""You are the main financial AI assistant. Route user queries to the appropriate specialized agent.

Available agents:
- Research Agent: Academic papers, concepts, explanations
- Analysis Agent: Fundamental analysis, valuations, ratios
- Trading Agent: Strategies, backtests, technical indicators
- Market Agent: Prices, news, options, economic data

For complex queries needing multiple agents, coordinate between them.""",
            llm_wrapper=llm_wrapper
        )
        self.research_agent = research_agent
        self.analysis_agent = analysis_agent
        self.trading_agent = trading_agent
        self.market_agent = market_agent
        
        self._agent_map = {
            AgentType.RESEARCH: research_agent,
            AgentType.ANALYSIS: analysis_agent,
            AgentType.TRADING: trading_agent,
            AgentType.MARKET: market_agent,
        }

    def classify_intent(self, query: str) -> List[AgentType]:
        query_lower = query.lower()
        scores = {agent_type: 0 for agent_type in AgentType if agent_type != AgentType.ORCHESTRATOR}
        
        for agent_type, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[agent_type] += 1
        
        max_score = max(scores.values())
        if max_score == 0:
            return [AgentType.MARKET]
        
        return [at for at, score in scores.items() if score == max_score]

    async def execute(self, query: str, context: AgentContext) -> AgentResponse:
        agent_types = self.classify_intent(query)
        
        if len(agent_types) == 1:
            agent = self._agent_map.get(agent_types[0])
            if agent:
                return await agent.execute(query, context)
        
        responses = []
        for agent_type in agent_types:
            agent = self._agent_map.get(agent_type)
            if agent:
                try:
                    response = await agent.execute(query, context)
                    responses.append((agent_type, response))
                except Exception as e:
                    responses.append((agent_type, AgentResponse(content=f"Error: {str(e)}")))
        
        if not responses:
            return AgentResponse(
                content="I'm not sure how to help with that. Try asking about stocks, trading strategies, or financial research."
            )
        
        return self._merge_responses(responses, query)

    def _merge_responses(self, responses: List[tuple], query: str) -> AgentResponse:
        if len(responses) == 1:
            return responses[0][1]
        
        merged = f"## Response to: {query}\n\n"
        
        agent_names = {
            AgentType.RESEARCH: "Research",
            AgentType.ANALYSIS: "Analysis",
            AgentType.TRADING: "Trading",
            AgentType.MARKET: "Market",
        }
        
        for agent_type, response in responses:
            name = agent_names.get(agent_type, "Unknown")
            merged += f"### {name} View\n\n{response.content}\n\n---\n\n"
        
        return AgentResponse(
            content=merged,
            metadata={"agents_used": [str(at) for at, _ in responses]}
        )

    def get_agent(self, agent_type: AgentType):
        return self._agent_map.get(agent_type)
