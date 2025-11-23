import os
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
import json
import re
from urllib.parse import urlencode


class SearchTool:
    """Web + News search tool using Brave Search API with multi-search support"""

    def __init__(self, lm=None, api_key: Optional[str] = None, concurrency: int = 3):
        self.lm = lm
        self.api_key = api_key or os.environ.get("BRAVE_TOKEN")
        self.base_web = "https://api.search.brave.com/res/v1/web/search"
        self.base_news = "https://api.search.brave.com/res/v1/news/search"

        self.session: Optional[aiohttp.ClientSession] = None
        self.sem = asyncio.Semaphore(concurrency)

    # -------------------------------------------------------------
    # SESSION HANDLING
    # -------------------------------------------------------------
    async def _get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session

    # -------------------------------------------------------------
    # LOW-LEVEL REQUEST
    # -------------------------------------------------------------
    async def _request(self, url: str) -> Dict[str, Any]:
        """Perform a GET request to Brave."""
        if not self.api_key:
            raise ValueError("Missing BRAVE_TOKEN")

        sess = await self._get_session()

        async with self.sem:
            async with sess.get(
                url,
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self.api_key,
                }
            ) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    raise Exception(f"Brave API error {resp.status}: {txt}")
                return await resp.json()

    # -------------------------------------------------------------
    # UNIVERSAL SEARCH WRAPPER (web or news)
    # -------------------------------------------------------------
    async def search(self, query: str, endpoint: str = "web", count: int = 8) -> Dict[str, Any]:
        """Search either web or news with recency boost."""

        if endpoint == "news":
            base = self.base_news
        else:
            base = self.base_web

        params = {
            "q": query,
            "count": min(count, 20),
            "search_lang": "en",
        }

        url = f"{base}?{urlencode(params)}"
        raw = await self._request(url)
        return self._format_results(raw, endpoint)

    # -------------------------------------------------------------
    # FORMAT RESULTS
    # -------------------------------------------------------------
    def _format_results(self, data: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Normalize Brave web/news results."""
        results = []

        key = "news" if endpoint == "news" else "web"
        bucket = data.get(key, {}).get("results", [])

        for item in bucket:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
                "age": item.get("age", ""),
                "source": endpoint
            })

        return {
            "results": results,
            "total": len(results)
        }

    # -------------------------------------------------------------
    # MULTI SEARCH
    # -------------------------------------------------------------
    async def multi_search(self, user_message: str) -> str:
        """Full multi-search pipeline with summarization."""

        pairs = await self.generate_queries(user_message)

        async def run(pair):
            q = pair["q"]
            ep = pair["endpoint"]
            try:
                r = await self.search(q, ep)
                summary = await self.summarize_results(q, r["results"])
                return q, ep, summary
            except:
                return q, ep, "Search error or no results."

        results = await asyncio.gather(*[run(p) for p in pairs])
        return self.build_context(results)

    # -------------------------------------------------------------
    # QUERY GENERATION
    # -------------------------------------------------------------
    async def generate_queries(self, user_message: str) -> List[Dict[str, str]]:
        """
        Model returns JSON list of:
        [
            {"q": "...", "endpoint": "news"},
            {"q": "...", "endpoint": "web"}
        ]
        """

        if self.lm is None:
            return [{"q": user_message, "endpoint": "web"}]

        prompt = [
            {
                "role": "system",
                "content": (
                    "Generate 3–6 search tasks as JSON.\n"
                    "Each item must be an object: {\"q\": string, \"endpoint\": \"web\" | \"news\"}.\n"
                    "Choose endpoint = 'news' when the question involves recent events, finance, geopolitics, or anything time-sensitive.\n"
                    "Return ONLY the JSON array. No commentary."
                )
            },
            {"role": "user", "content": user_message}
        ]

        buffer = ""
        async for t in self.lm.stream_async(prompt, max_tokens=200):
            buffer += t

        # Extract JSON safely
        try:
            jtxt = re.search(r"\[.*\]", buffer, re.S).group(0)
            arr = json.loads(jtxt)
            clean = []
            for item in arr:
                if not isinstance(item, dict):
                    continue
                q = item.get("q")
                ep = (item.get("endpoint") or "web").lower()
                if q and ep in ("web", "news"):
                    clean.append({"q": q.strip(), "endpoint": ep})
            return clean or [{"q": user_message, "endpoint": "web"}]
        except:
            return [{"q": user_message, "endpoint": "web"}]

    # -------------------------------------------------------------
    # SUMMARIZATION OF SEARCH RESULTS
    # -------------------------------------------------------------
    async def summarize_results(self, original_query: str, items: List[Dict[str, Any]]) -> str:
        """Compress search results to a small, relevant summary."""

        if not items:
            return "No results."

        text = "\n".join(
            f"Title: {i['title']}\nDesc: {i['description']}\nURL: {i['url']}"
            for i in items
        )

        prompt = [
            {
                "role": "system",
                "content": (
                    "Summarize the following search results into 3–7 bullet points.\n"
                    "Focus on relevance to the original query.\n"
                    "Be concise but preserve factual accuracy."
                )
            },
            {
                "role": "user",
                "content": (
                    f"ORIGINAL QUERY:\n{original_query}\n\n"
                    f"SEARCH RESULTS:\n{text}"
                )
            }
        ]

        summary = ""
        async for t in self.lm.stream_async(prompt, max_tokens=250):
            summary += t

        return summary.strip()

    # -------------------------------------------------------------
    # MERGED CONTEXT
    # -------------------------------------------------------------
    def build_context(self, results: List[Any]) -> str:
        """Inject summarized results into a clean context block."""

        ctx = "# Multi-Search Context\n\n"

        for q, ep, summary in results:
            ctx += f"## Query: {q}  (endpoint={ep})\n"
            ctx += summary + "\n\n"

        ctx += "Use this context to answer the user's question accurately.\n"
        return ctx
