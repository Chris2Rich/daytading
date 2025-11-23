import os
import aiohttp
import asyncio
import json
import re
from typing import List, Dict, Any, Optional

class SearchTool:
    """Web + News search tool using Tavily API with multi-search and recency support"""


    def __init__(self, lm=None, concurrency: int = 3):  
        self.lm = lm  
        self.api_key = os.environ.get("TAVILY_KEY")  
        self.base_url = "https://api.tavily.com/search"  
        self.session: Optional[aiohttp.ClientSession] = None  
        self.sem = asyncio.Semaphore(concurrency)  

    # -------------------------------------------------------------  
    # SESSION HANDLING  
    # -------------------------------------------------------------  
    async def _get_session(self):  
        if self.session is None or self.session.closed:  
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))  
        return self.session  

    # -------------------------------------------------------------  
    # LOW-LEVEL REQUEST  
    # -------------------------------------------------------------  
    async def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:  
        if not self.api_key:  
            raise ValueError("Missing TAVILY_KEY")  

        sess = await self._get_session()  

        async with self.sem:  
            async with sess.post(  
                self.base_url,  
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},  
                json=payload  
            ) as resp:  
                if resp.status != 200:  
                    txt = await resp.text()  
                    raise Exception(f"Tavily API error {resp.status}: {txt}")  
                return await resp.json()  

    # -------------------------------------------------------------  
    # UNIVERSAL SEARCH  
    # -------------------------------------------------------------  
    async def search(self, query: str, endpoint: str = "web", start_date: str = None, end_date: str = None, max_results: int = 8) -> Dict[str, Any]:  
        """Search Tavily with optional date filtering."""  

        payload = {  
            "query": query,  
            "endpoint": endpoint,  # "web" or "news"  
            "max_results": max_results  
        }  

        if start_date:  
            payload["start_date"] = start_date  
        if end_date:  
            payload["end_date"] = end_date  

        raw = await self._request(payload)  
        return self._format_results(raw, endpoint)  

    # -------------------------------------------------------------  
    # FORMAT RESULTS  
    # -------------------------------------------------------------  
    def _format_results(self, data: Dict[str, Any], endpoint: str) -> Dict[str, Any]:  
        """Normalize Tavily search results."""  
        results = []  
        bucket = data.get("results", [])  

        for item in bucket:  
            results.append({  
                "title": item.get("title", ""),  
                "url": item.get("url", ""),  
                "description": item.get("description", item.get("content", "")),  
                "source": endpoint,  
                "date": item.get("date", "")  
            })  

        return {"results": results, "total": len(results)}  

    # -------------------------------------------------------------  
    # MULTI SEARCH  
    # -------------------------------------------------------------  
    async def multi_search(self, user_message: str) -> str:  
        """Full multi-search pipeline with summarization."""  

        pairs = await self.generate_queries(user_message)  

        async def run(pair):  
            q = pair.get("q")  
            ep = pair.get("endpoint", "web")  
            start = pair.get("start_date")  
            end = pair.get("end_date")  
            try:  
                r = await self.search(q, ep, start_date=start, end_date=end)    
                return q, ep, r  
            except:  
                return q, ep, "Search error or no results."  

        results = await asyncio.gather(*[run(p) for p in pairs])  
        return self.build_context(results)  

    # -------------------------------------------------------------  
    # QUERY GENERATION  
    # -------------------------------------------------------------  
    async def generate_queries(self, user_message: str) -> List[Dict[str, Any]]:  
        """Model returns JSON list of {q, endpoint, start_date?, end_date?}."""  

        if self.lm is None:  
            return [{"q": user_message, "endpoint": "web"}]  

        prompt = [  
            {"role": "system", "content": (  
                "Generate 3-6 search tasks as JSON array.\n"  
                "Each item must be an object: {\"q\": string, \"endpoint\": \"web\" | \"news\", \"start_date\": \"YYYY-MM-DD\"?, \"end_date\": \"YYYY-MM-DD\"?}.\n"  
                "Choose endpoint='news' for time-sensitive topics, finance, or recent events.\n"  
                "Return ONLY the JSON array. No commentary."  
            )},  
            {"role": "user", "content": user_message}  
        ]  

        buffer = ""  
        async for t in self.lm.stream_async(prompt, max_tokens=1000):  
            buffer += t  

        try:  
            jtxt = re.search(r"\[.*\]", buffer, re.S).group(0)  
            arr = json.loads(jtxt)  
            clean = []  
            for item in arr:  
                if not isinstance(item, dict):  
                    continue  
                q = item.get("q")  
                ep = (item.get("endpoint") or "web").lower()  
                start = item.get("start_date")  
                end = item.get("end_date")  
                if q and ep in ("web", "news"):  
                    clean.append({"q": q.strip(), "endpoint": ep, "start_date": start, "end_date": end})  
            return clean or [{"q": user_message, "endpoint": "web"}]  
        except:  
            return [{"q": user_message, "endpoint": "web"}]  

    # -------------------------------------------------------------  
    # BUILD CONTEXT  
    # -------------------------------------------------------------  
    def build_context(self, results: List[Any]) -> str:  
        ctx = "# Multi-Search Context\n\n"  
        for q, ep, summary in results:  
            ctx += f"## Query: {q}  (endpoint={ep})\n"  
            ctx += summary + "\n\n"  
        ctx += "Use this context to answer the user's question accurately.\n"  
        return ctx  
