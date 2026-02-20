from .base import BaseAgent, AgentType, AgentContext, AgentResponse, RESEARCH_AGENT_PROMPT
from ..retrieval.research_ingestion import ResearchAgent as ResearchIngestionAgent

class ResearchAgent(BaseAgent):
    def __init__(self, research_ingestion, llm_wrapper=None):
        super().__init__(
            agent_type=AgentType.RESEARCH,
            system_prompt=RESEARCH_AGENT_PROMPT,
            tools=["search_papers", "explain_concept", "get_wikipedia"],
            llm_wrapper=llm_wrapper
        )
        self.research_ingestion = research_ingestion
        self.research_client = ResearchIngestionAgent(research_ingestion)

    async def execute(self, query: str, context: AgentContext) -> AgentResponse:
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["paper", "arxiv", "academic", "research", "study"]):
            results = await self.research_client.find_papers(query, limit=5)
            return AgentResponse(
                content=results,
                context_used={"source": "arxiv"}
            )
        
        elif any(kw in query_lower for kw in ["explain", "what is", "how does", "concept", "wiki"]):
            concept = query
            for prefix in ["explain ", "what is ", "how does ", "wiki "]:
                concept = concept.replace(prefix, "")
            results = await self.research_client.explain_concept(concept.strip())
            return AgentResponse(
                content=results,
                context_used={"source": "wikipedia"}
            )
        
        else:
            results = await self.research_client.search(query, limit=5)
            return AgentResponse(
                content=results,
                context_used={"source": "research"}
            )
