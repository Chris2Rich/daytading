import logging
from typing import List, Dict, Any, Optional

import torch
import tiktoken
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ResearchIngestion:
    """Research ingestion for searching cached embeddings only.
    
    Note: Sync functionality has been moved to gpusync/client.py
    for GPU-based embedding and server upload.
    """
    
    def __init__(self, db=None):
        self.db = db
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=self.device
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = 512

    def _chunk_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        return chunks

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False
        )
        return embeddings.tolist()

    async def search_research(
        self,
        query: str,
        limit: int = 10,
        source: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search research papers using vector similarity"""
        if not self.db:
            return []
        
        query_embedding = await self._get_embeddings([query])
        return await self.db.search_research_chunks(
            query_embedding=query_embedding[0],
            limit=limit,
            source_filter=source,
            category_filter=category
        )


class ResearchAgent:
    """Agent for searching research papers and explaining concepts"""
    
    def __init__(self, ingestion: ResearchIngestion):
        self.ingestion = ingestion

    async def search(self, query: str, limit: int = 5) -> str:
        results = await self.ingestion.search_research(query, limit=limit)
        
        if not results:
            return "No research papers found matching your query."
        
        response = f"Found {len(results)} relevant sources:\n\n"
        
        for i, result in enumerate(results, 1):
            source_emoji = "📄" if result['source'] == 'arxiv' else "📖"
            response += f"{source_emoji} **{result['title']}** ({result['source']})\n"
            response += f"   {result['chunk_text'][:300]}...\n"
            if result.get('url'):
                response += f"   🔗 {result['url']}\n"
            response += "\n"
        
        return response

    async def explain_concept(self, concept: str) -> str:
        results = await self.ingestion.search_research(
            concept,
            limit=3,
            source="wikipedia"
        )
        
        if not results:
            return f"No Wikipedia article found for '{concept}'. Try a different search term."
        
        result = results[0]
        return f"**{result['title']}** (from Wikipedia)\n\n{result['chunk_text']}\n\n🔗 {result.get('url', 'N/A')}"

    async def find_papers(self, topic: str, limit: int = 5) -> str:
        results = await self.ingestion.search_research(
            topic,
            limit=limit,
            source="arxiv"
        )
        
        if not results:
            return f"No academic papers found for '{topic}'."
        
        response = f"Found {len(results)} papers on '{topic}':\n\n"
        
        for result in results:
            date_str = ""
            if result.get('published_date'):
                date_str = f" ({result['published_date'][:10]})"
            response += f"📄 **{result['title']}**{date_str}\n"
            response += f"   {result['chunk_text'][:200]}...\n"
            response += f"   🔗 {result.get('url', 'N/A')}\n\n"
        
        return response
