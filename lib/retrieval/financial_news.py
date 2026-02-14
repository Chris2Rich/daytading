"""
Financial News Retrieval Module

Fetches financial news from various sources using Tavily API.
Provides structured news data for the frontend.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from .search import SearchTool


class FinancialNewsFetcher:
    """Fetches financial news using Tavily API"""
    
    # Default categories for financial news
    CATEGORIES = {
        "general": "stock market finance",
        "technology": "technology stocks tech sector",
        "retail": "retail sector earnings consumer stocks",
        "life sciences": "pharmaceutical biotech healthcare stocks",
        "politics": "government policy economic impact markets",
        "developing markets": "emerging markets economy developing countries",
    }
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("TAVILY_KEY")
        self.search_tool = None
    
    async def _get_search_tool(self):
        if self.search_tool is None:
            self.search_tool = SearchTool()
            self.search_tool.api_key = self.api_key
        return self.search_tool
    
    async def get_news(self, category: str = "general", max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch financial news for a specific category.
        
        Args:
            category: News category (general, technology, retail, life sciences, politics, developing markets)
            max_results: Maximum number of articles to return
            
        Returns:
            List of news articles with metadata
        """
        search_tool = await self._get_search_tool()
        
        # Build query based on category
        base_query = self.CATEGORIES.get(category, self.CATEGORIES["general"])
        
        # Add date constraint for recent news
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        try:
            results = await search_tool.search(
                query=f"{base_query} latest news",
                endpoint="news",
                start_date=start_date,
                end_date=end_date,
                max_results=max_results
            )
            
            articles = self._format_articles(results.get("results", []), category)
            return articles
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    async def get_all_news(self, max_per_category: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch news for all categories.
        
        Args:
            max_per_category: Maximum articles per category
            
        Returns:
            Dictionary mapping categories to article lists
        """
        all_news = {}
        
        async def fetch_category(cat):
            articles = await self.get_news(cat, max_per_category)
            return cat, articles
        
        tasks = [fetch_category(cat) for cat in self.CATEGORIES.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, tuple):
                category, articles = result
                all_news[category] = articles
            else:
                print(f"Error fetching category: {result}")
        
        return all_news
    
    async def get_latest_headlines(self, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Get the latest financial headlines regardless of category.
        
        Args:
            max_results: Total number of headlines to return
            
        Returns:
            List of latest financial news articles
        """
        search_tool = await self._get_search_tool()
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        
        try:
            results = await search_tool.search(
                query="stock market breaking news financial economy",
                endpoint="news",
                start_date=start_date,
                end_date=end_date,
                max_results=max_results
            )
            
            articles = self._format_articles(results.get("results", []), "general")
            return articles
            
        except Exception as e:
            print(f"Error fetching headlines: {e}")
            return []
    
    def _format_articles(self, raw_results: List[Dict], category: str) -> List[Dict[str, Any]]:
        """
        Format raw search results into article objects.
        
        Args:
            raw_results: Raw results from Tavily API
            category: Category these articles belong to
            
        Returns:
            Formatted article objects
        """
        formatted = []
        
        for item in raw_results:
            # Extract image URL if available
            url_to_image = item.get("image_url") or item.get("thumbnail") or ""
            
            # Parse date
            published_at = item.get("date", "")
            if not published_at:
                published_at = datetime.now().isoformat()
            
            article = {
                "title": item.get("title", ""),
                "description": item.get("description", item.get("content", "")),
                "url": item.get("url", ""),
                "urlToImage": url_to_image,
                "publishedAt": published_at,
                "source": {
                    "name": item.get("source", "Unknown")
                },
                "category": [category]
            }
            
            # Only add if has title
            if article["title"]:
                formatted.append(article)
        
        return formatted


# Singleton instance
_news_fetcher: Optional[FinancialNewsFetcher] = None


def get_news_fetcher() -> FinancialNewsFetcher:
    """Get or create the news fetcher singleton"""
    global _news_fetcher
    if _news_fetcher is None:
        _news_fetcher = FinancialNewsFetcher()
    return _news_fetcher
