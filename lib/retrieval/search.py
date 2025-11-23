import os
import aiohttp
from typing import Dict, Any, List, Optional
import json
from urllib.parse import urlencode


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
            raise ValueError("Brave Search API key not found in BRAVE_TOKEN environment variable")
        
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        
        # Build query parameters with proper URL encoding
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
                        error_text = await response.text()
                        raise Exception(f"Brave Search API failed with status {response.status}: {error_text}")
                    
                    data = await response.json()
                    return self._format_results(data)
        
        except aiohttp.ClientError as e:
            raise Exception(f"Search request failed: {e}")
        except Exception as e:
            raise Exception(f"Search error: {e}")
    
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