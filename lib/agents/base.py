from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import asyncio

class AgentType(Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    TRADING = "trading"
    MARKET = "market"
    ORCHESTRATOR = "orchestrator"

@dataclass
class AgentResponse:
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    context_used: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentContext:
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    entity_memory: Dict[str, List[str]] = field(default_factory=dict)
    research_results: List[Dict[str, Any]] = field(default_factory=list)
    market_data: Dict[str, Any] = field(default_factory=dict)
    portfolio_data: Optional[Dict[str, Any]] = None

class BaseAgent(ABC):
    def __init__(
        self,
        agent_type: AgentType,
        system_prompt: str,
        tools: List[str] = None,
        llm_wrapper = None
    ):
        self.agent_type = agent_type
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.llm_wrapper = llm_wrapper

    @abstractmethod
    async def execute(self, query: str, context: AgentContext) -> AgentResponse:
        pass

    async def call_llm(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        if self.llm_wrapper:
            return await self.llm_wrapper.generate(messages, **kwargs)
        raise NotImplementedError("No LLM wrapper configured")

    def build_messages(self, query: str, context: AgentContext) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if context.entity_memory.get("tickers"):
            tickers = ", ".join(context.entity_memory["tickers"])
            messages.append({
                "role": "system",
                "content": f"Relevant tickers in conversation: {tickers}"
            })
        
        if context.research_results:
            research_context = "\n\n".join([
                f"Source: {r.get('title', 'Unknown')}\n{r.get('chunk_text', '')[:500]}"
                for r in context.research_results[:3]
            ])
            messages.append({
                "role": "system",
                "content": f"Research context:\n{research_context}"
            })
        
        for msg in context.chat_history[-10:]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        messages.append({"role": "user", "content": query})
        return messages

RESEARCH_AGENT_PROMPT = """You are a Research Agent specializing in financial research and academic literature.

Your capabilities:
- Search and explain academic papers from arXiv
- Retrieve information from Wikipedia and financial knowledge bases
- Summarize research findings in accessible language
- Provide citations and sources for claims

When answering:
1. Always cite sources when making claims
2. Distinguish between academic research and general knowledge
3. Explain complex concepts in simple terms
4. Be precise about limitations and uncertainties in research

Available data sources: arXiv (quantitative finance, ML in finance), Wikipedia (financial concepts)"""

ANALYSIS_AGENT_PROMPT = """You are a Financial Analysis Agent specializing in company and security analysis.

Your capabilities:
- Perform fundamental analysis (DCF, relative valuation, ratios)
- Analyze financial statements and metrics
- Compare peers and competitors
- Assess risk and return profiles

When answering:
1. Use quantitative data from market data tools
2. Show your work with calculations
3. Provide clear buy/sell/hold recommendations with rationale
4. Consider both quantitative and qualitative factors

Available tools: get_fundamentals, get_key_metrics_timeseries, get_price_targets, compare_peers"""

TRADING_AGENT_PROMPT = """You are a Trading Strategy Agent specializing in algorithmic trading and strategy development.

Your capabilities:
- Design and explain trading strategies
- Interpret backtest results
- Optimize strategy parameters
- Explain technical indicators and signals

When answering:
1. Be specific about entry/exit conditions
2. Explain risk management approaches
3. Show backtested performance metrics
4. Discuss realistic expectations and limitations

Available tools: create_strategy, run_backtest, get_technical_indicators"""

MARKET_AGENT_PROMPT = """You are a Market Data Agent specializing in real-time market information.

Your capabilities:
- Get current stock prices and historical data
- Fetch news and sentiment about securities
- Retrieve options chains and pricing
- Monitor economic indicators

When answering:
1. Provide timely, accurate market data
2. Explain what the data means for investors
3. Flag important market events or changes
4. Use visualizations when helpful

Available tools: get_trading, get_news, get_option_chain, get_fred"""
