import os
import aiohttp
from typing import Dict, Any, List, Optional
import json
import re
from urllib.parse import urlencode

class SearchTool:
    """Web search tool using Brave Search API + multi-search support"""

    def __init__(self, lm=None, api_key: Optional[str] = None):
        """
        Args:
            lm: Your LM wrapper so we can use stream_async to generate queries
            api_key: Brave API key
        """
        self.lm = lm
        self.api_key = api_key or os.environ.get("BRAVE_TOKEN")
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
    
    # ----------------------------------------------------------------------
    # ORIGINAL SINGLE SEARCH
    # ----------------------------------------------------------------------
    async def search(self, query: str, count: int = 5) -> Dict[str, Any]:
        """Perform simple web search using Brave Search API"""

        if not self.api_key:
            raise ValueError("Brave Search API key not found in BRAVE_TOKEN environment variable")
        
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }

        params = {
            "q": query,
            "count": min(count, 20),
            "search_lang": "en"
        }
        
        query_string = urlencode(params)
        full_url = f"{self.base_url}?{query_string}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    full_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        err = await response.text()
                        raise Exception(f"Brave Search failed ({response.status}): {err}")
                    
                    data = await response.json()
                    return self._format_results(data)
        
        except Exception as e:
            raise Exception(f"Search error: {e}")

    def _format_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format Brave search results into a clean structure"""
        results = []
        
        if "web" in data and "results" in data["web"]:
            for item in data["web"]["results"]:
                results.append({
                    "title": item.get("title") or "",
                    "url": item.get("url") or "",
                    "description": item.get("description") or "",
                    "age": item.get("age") or ""
                })
        
        return {
            "query": data.get("query", {}).get("original", ""),
            "results": results,
            "total": len(results)
        }

    # ----------------------------------------------------------------------
    # MULTI SEARCH EXTENSION
    # ----------------------------------------------------------------------
    async def multi_search(self, user_message: str) -> str:
        """
        High-level multi-search:
        1. Generate multiple queries
        2. Run all searches in parallel
        3. Build a merged context block to feed the LLM
        """

        # Step 1: Ask the model for multiple search queries
        queries = await self.generate_queries(user_message)

        # Step 2: Run all queries concurrently
        async def run(q):
            try:
                return q, await self.search(q, count=5)
            except:
                return q, None

        results = await asyncio.gather(*[run(q) for q in queries])

        # Step 3: Build a merged context block
        ctx = self.build_context(results)

        return ctx

    # ----------------------------------------------------------------------
    # MULTI SEARCH HELPERS
    # ----------------------------------------------------------------------
    async def generate_queries(self, user_message: str, max_queries: int = 5) -> List[str]:
        """
        Use the LLM to propose 3–6 relevant search queries.
        """

        if self.lm is None:
            # fallback to single query mode
            return [user_message]

        prompt = [
            {
                "role": "system",
                "content": (
                    "Generate 3–6 highly relevant, diverse search queries "
                    "that would help answer the user's question.\n"
                    "Return ONLY a JSON list of strings. No commentary."
                )
            },
            {"role": "user", "content": user_message}
        ]

        buffer = ""
        async for tok in self.lm.stream_async(prompt, max_tokens=200):
            buffer += tok

        # Extract the JSON list
        try:
            json_text = re.search(r"\[.*\]", buffer, re.S).group(0)
            queries = json.loads(json_text)
            return [q.strip() for q in queries if isinstance(q, str)]
        except:
            return [user_message]

    def build_context(self, results: List[Any]) -> str:
        """
        Convert multiple search results into one large context block.
        """

        ctx = "# Multi-Search Results\n\n"

        for q, r in results:
            ctx += f"## Query: {q}\n"
            if not r or not r.get("results"):
                ctx += "No results.\n\n"
                continue

            for item in r["results"]:
                ctx += (
                    f"- **{item['title']}**\n"
                    f"  {item['description']}\n"
                    f"  {item['url']}\n"
                )
            ctx += "\n"

        ctx += "\nUse this context to answer the user's question accurately.\n"
        return ctx
