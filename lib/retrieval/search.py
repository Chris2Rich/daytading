import os
import aiohttp
from typing import Dict, Any, List, Optional
import json

class SearchTool:
    """Web search tool using Brave Search API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("BRAVE_TOKEN")
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
    
    async def search(self, query: str, count: int = 5) -> Dict[str, Any]:
        """
        Perform web search using Brave Search API
        
        Args:
            query: Search query string
            count: Number of results to return (max 20)
            
        Returns:
            Dict with search results
        """
        if not self.api_key:
            raise ValueError("Brave Search API key not found")
        
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        
        params = {
            "q": query,
            "count": min(count, 20),
            "text_decorations": False,
            "search_lang": "en"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Search API failed: {response.status} - {error_text}")
                    
                    data = await response.json()
                    return self._format_results(data)
        
        except aiohttp.ClientError as e:
            raise Exception(f"Search request failed: {e}")
    
    def _format_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format Brave search results into a clean structure"""
        results = []
        
        if "web" in data and "results" in data["web"]:
            for item in data["web"]["results"]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                    "age": item.get("age", "")
                })
        
        return {
            "query": data.get("query", {}).get("original", ""),
            "results": results,
            "total": len(results)
        }