"""
RSS Feed Parser for Financial News
Fetches news from multiple RSS feeds and normalizes the data.
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import re

# RSS Feed sources
RSS_FEEDS = {
    # CNBC feeds
    "cnbc_top": {
        "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "category": "general",
        "source_name": "CNBC",
    },
    # Bloomberg feeds
    "bloomberg_markets": {
        "url": "https://feeds.bloomberg.com/markets/news.rss",
        "category": "general",
        "source_name": "Bloomberg",
    },
    "bloomberg_technology": {
        "url": "https://feeds.bloomberg.com/technology/news.rss",
        "category": "technology",
        "source_name": "Bloomberg",
    },
    # Guardian feeds
    "guardian_business": {
        "url": "https://www.theguardian.com/business/rss",
        "category": "general",
        "source_name": "The Guardian",
    },
    "guardian_technology": {
        "url": "https://www.theguardian.com/technology/rss",
        "category": "technology",
        "source_name": "The Guardian",
    },
    "guardian_politics": {
        "url": "https://www.theguardian.com/politics/rss",
        "category": "politics",
        "source_name": "The Guardian",
    },
    # NYT feeds
    "nyt_business": {
        "url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "category": "general",
        "source_name": "New York Times",
    },
    "nyt_technology": {
        "url": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
        "category": "technology",
        "source_name": "New York Times",
    },
    "nyt_politics": {
        "url": "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
        "category": "politics",
        "source_name": "New York Times",
    },
    # Reuters
    "reuters_business": {
        "url": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        "category": "general",
        "source_name": "Reuters",
    },
    "reuters_technology": {
        "url": "https://www.reutersagency.com/feed/?best-tech&post_type=best",
        "category": "technology",
        "source_name": "Reuters",
    },
}


@dataclass
class ParsedArticle:
    title: str
    description: Optional[str]
    url: str
    url_to_image: Optional[str]
    source_name: str
    source_url: Optional[str]
    category: str
    published_at: datetime


async def fetch_feed(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """Fetch RSS feed content."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                return await response.text()
            else:
                print(f"Failed to fetch {url}: status {response.status}")
                return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def parse_rss_date(date_str: str) -> Optional[datetime]:
    """Parse various date formats from RSS feeds."""
    if not date_str:
        return None
    
    # Common date formats
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",      # RFC 822: "Mon, 01 Jan 2024 12:00:00 +0000"
        "%a, %d %b %Y %H:%M:%S %Z",       # RFC 822 with timezone name
        "%Y-%m-%dT%H:%M:%S%z",           # ISO 8601: "2024-01-01T12:00:00+00:00"
        "%Y-%m-%dT%H:%M:%SZ",            # ISO 8601 UTC: "2024-01-01T12:00:00Z"
        "%Y-%m-%d %H:%M:%S",             # Simple: "2024-01-01 12:00:00"
        "%Y-%m-%d",                       # Date only: "2024-01-01"
    ]
    
    date_str = date_str.strip()
    
    for fmt in formats:
        try:
            # Handle timezone offset manually for some formats
            if '%z' in fmt:
                # Add colon to timezone offset if missing (e.g., +0000 -> +00:00)
                if re.match(r'.*[+-]\d{4}$', date_str):
                    date_str = date_str[:-2] + ':' + date_str[-2:]
            return datetime.strptime(date_str, fmt).replace(tzinfo=None)
        except ValueError:
            continue
    
    # Try parsing with dateutil if available
    try:
        from dateutil import parser
        return parser.parse(date_str).replace(tzinfo=None)
    except ImportError:
        pass
    
    return None


def extract_image_from_content(content: str) -> Optional[str]:
    """Extract image URL from HTML content."""
    if not content:
        return None
    
    # Try to find img tag
    img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content, re.IGNORECASE)
    if img_match:
        return img_match.group(1)
    
    # Try to find og:image meta tag
    og_match = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', content, re.IGNORECASE)
    if og_match:
        return og_match.group(1)
    
    return None


def parse_cnbc_feed(content: str, feed_info: Dict) -> List[ParsedArticle]:
    """Parse CNBC RSS feed."""
    articles = []
    try:
        root = ET.fromstring(content)
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            description = item.findtext("description", "")
            link = item.findtext("link", "").strip()
            
            # Extract image from media:content or media:thumbnail
            url_to_image = None
            for media in item.findall("media:content") + item.findall("media:thumbnail"):
                url = media.get("url")
                if url:
                    url_to_image = url
                    break
            
            # Also check description for image
            if not url_to_image:
                url_to_image = extract_image_from_content(description)
            
            # Clean HTML from description
            if description:
                description = re.sub(r'<[^>]+>', '', description).strip()
                description = description[:500] if len(description) > 500 else description
            
            pub_date = item.findtext("pubDate", "")
            published_at = parse_rss_date(pub_date) or datetime.now()
            
            if title and link:
                articles.append(ParsedArticle(
                    title=title,
                    description=description,
                    url=link,
                    url_to_image=url_to_image,
                    source_name=feed_info["source_name"],
                    source_url=feed_info["url"],
                    category=feed_info["category"],
                    published_at=published_at
                ))
    except Exception as e:
        print(f"Error parsing CNBC feed: {e}")
    
    return articles


def parse_bloomberg_feed(content: str, feed_info: Dict) -> List[ParsedArticle]:
    """Parse Bloomberg RSS feed."""
    articles = []
    try:
        root = ET.fromstring(content)
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            description = item.findtext("description", "")
            link = item.findtext("link", "").strip()
            
            # Extract image
            url_to_image = None
            for media in item.findall("media:content") + item.findall("media:thumbnail"):
                url = media.get("url")
                if url:
                    url_to_image = url
                    break
            
            if not url_to_image:
                url_to_image = extract_image_from_content(description)
            
            # Clean HTML
            if description:
                description = re.sub(r'<[^>]+>', '', description).strip()
                description = description[:500] if len(description) > 500 else description
            
            pub_date = item.findtext("pubDate", "")
            published_at = parse_rss_date(pub_date) or datetime.now()
            
            if title and link:
                articles.append(ParsedArticle(
                    title=title,
                    description=description,
                    url=link,
                    url_to_image=url_to_image,
                    source_name=feed_info["source_name"],
                    source_url=feed_info["url"],
                    category=feed_info["category"],
                    published_at=published_at
                ))
    except Exception as e:
        print(f"Error parsing Bloomberg feed: {e}")
    
    return articles


def parse_guardian_feed(content: str, feed_info: Dict) -> List[ParsedArticle]:
    """Parse Guardian RSS feed."""
    articles = []
    try:
        root = ET.fromstring(content)
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            description = item.findtext("description", "")
            link = item.findtext("link", "").strip()
            
            # Extract image from media content
            url_to_image = None
            for media in item.findall("media:content") + item.findall("media:thumbnail"):
                url = media.get("url")
                if url and ('image' in url.lower() or 'media' in url.lower()):
                    url_to_image = url
                    break
            
            if not url_to_image:
                url_to_image = extract_image_from_content(description)
            
            # Clean HTML
            if description:
                description = re.sub(r'<[^>]+>', '', description).strip()
                description = description[:500] if len(description) > 500 else description
            
            pub_date = item.findtext("pubDate", "")
            published_at = parse_rss_date(pub_date) or datetime.now()
            
            if title and link:
                articles.append(ParsedArticle(
                    title=title,
                    description=description,
                    url=link,
                    url_to_image=url_to_image,
                    source_name=feed_info["source_name"],
                    source_url=feed_info["url"],
                    category=feed_info["category"],
                    published_at=published_at
                ))
    except Exception as e:
        print(f"Error parsing Guardian feed: {e}")
    
    return articles


def parse_nyt_feed(content: str, feed_info: Dict) -> List[ParsedArticle]:
    """Parse NYT RSS feed."""
    articles = []
    try:
        root = ET.fromstring(content)
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            description = item.findtext("description", "")
            link = item.findtext("link", "").strip()
            
            # Extract image from media:content
            url_to_image = None
            for media in item.findall("media:content") + item.findall("media:thumbnail"):
                url = media.get("url")
                if url:
                    url_to_image = url
                    break
            
            if not url_to_image:
                url_to_image = extract_image_from_content(description)
            
            # Clean HTML
            if description:
                description = re.sub(r'<[^>]+>', '', description).strip()
                description = description[:500] if len(description) > 500 else description
            
            pub_date = item.findtext("pubDate", "")
            published_at = parse_rss_date(pub_date) or datetime.now()
            
            if title and link:
                articles.append(ParsedArticle(
                    title=title,
                    description=description,
                    url=link,
                    url_to_image=url_to_image,
                    source_name=feed_info["source_name"],
                    source_url=feed_info["url"],
                    category=feed_info["category"],
                    published_at=published_at
                ))
    except Exception as e:
        print(f"Error parsing NYT feed: {e}")
    
    return articles


def parse_reuters_feed(content: str, feed_info: Dict) -> List[ParsedArticle]:
    """Parse Reuters RSS feed."""
    articles = []
    try:
        root = ET.fromstring(content)
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            description = item.findtext("description", "")
            link = item.findtext("link", "").strip()
            
            # Extract image from media content
            url_to_image = None
            for media in item.findall("media:content") + item.findall("media:thumbnail"):
                url = media.get("url")
                if url:
                    url_to_image = url
                    break
            
            if not url_to_image:
                url_to_image = extract_image_from_content(description)
            
            # Clean HTML
            if description:
                description = re.sub(r'<[^>]+>', '', description).strip()
                description = description[:500] if len(description) > 500 else description
            
            pub_date = item.findtext("pubDate", "")
            published_at = parse_rss_date(pub_date) or datetime.now()
            
            if title and link:
                articles.append(ParsedArticle(
                    title=title,
                    description=description,
                    url=link,
                    url_to_image=url_to_image,
                    source_name=feed_info["source_name"],
                    source_url=feed_info["url"],
                    category=feed_info["category"],
                    published_at=published_at
                ))
    except Exception as e:
        print(f"Error parsing Reuters feed: {e}")
    
    return articles


def parse_feed(content: str, feed_key: str, feed_info: Dict) -> List[ParsedArticle]:
    """Parse RSS feed based on source type."""
    # Route to appropriate parser based on source
    source_name = feed_info["source_name"].lower()
    
    if "cnbc" in source_name:
        return parse_cnbc_feed(content, feed_info)
    elif "bloomberg" in source_name:
        return parse_bloomberg_feed(content, feed_info)
    elif "guardian" in source_name:
        return parse_guardian_feed(content, feed_info)
    elif "york times" in source_name or "nyt" in source_name:
        return parse_nyt_feed(content, feed_info)
    elif "reuters" in source_name:
        return parse_reuters_feed(content, feed_info)
    else:
        # Default: try generic RSS parsing
        return parse_bloomberg_feed(content, feed_info)


async def fetch_all_feeds() -> List[Dict[str, Any]]:
    """
    Fetch all RSS feeds and return normalized articles.
    """
    all_articles = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for feed_key, feed_info in RSS_FEEDS.items():
            tasks.append(fetch_and_parse_feed(session, feed_key, feed_info))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                print(f"Error fetching feed: {result}")
    
    # Sort by published date (newest first)
    all_articles.sort(key=lambda x: x.published_at, reverse=True)
    
    return [article_to_dict(a) for a in all_articles]


async def fetch_and_parse_feed(
    session: aiohttp.ClientSession, 
    feed_key: str, 
    feed_info: Dict
) -> List[ParsedArticle]:
    """Fetch and parse a single feed."""
    content = await fetch_feed(session, feed_info["url"])
    if content:
        return parse_feed(content, feed_key, feed_info)
    return []


async def fetch_feed_for_category(category: str) -> List[Dict[str, Any]]:
    """
    Fetch feeds for a specific category only.
    """
    all_articles = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for feed_key, feed_info in RSS_FEEDS.items():
            if feed_info["category"] == category:
                tasks.append(fetch_and_parse_feed(session, feed_key, feed_info))
        
        if not tasks:
            # If no specific feeds for category, fetch all and filter
            tasks = [
                fetch_and_parse_feed(session, feed_key, feed_info)
                for feed_key, feed_info in RSS_FEEDS.items()
            ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
    
    # Filter by category
    filtered = [a for a in all_articles if a.category == category]
    
    # Sort by published date
    filtered.sort(key=lambda x: x.published_at, reverse=True)
    
    return [article_to_dict(a) for a in filtered]


def article_to_dict(article: ParsedArticle) -> Dict[str, Any]:
    """Convert ParsedArticle to dictionary for database storage."""
    return {
        "title": article.title,
        "description": article.description,
        "url": article.url,
        "url_to_image": article.url_to_image,
        "source_name": article.source_name,
        "source_url": article.source_url,
        "category": article.category,
        "published_at": article.published_at
    }


# Module-level singleton for the news fetcher
_news_fetcher_instance = None


class RSSNewsFetcher:
    """RSS News Fetcher - fetches and caches news from RSS feeds."""
    
    def __init__(self, db=None):
        self.db = db
        self.cache_age_seconds = 0
        self._cache_max_age = 3600  # 1 hour cache
    
    async def get_news(
        self, 
        category: Optional[str] = None, 
        limit: int = 50,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get news articles, using cache if available.
        
        Args:
            category: Filter by category (general, technology, politics, etc.)
            limit: Maximum number of articles to return
            force_refresh: If True, always fetch fresh data
        """
        # Check cache age
        if self.db:
            cache_age = await self.db.get_news_cache_age()
            if cache_age is not None:
                self.cache_age_seconds = cache_age
        
        # Decide whether to refresh
        should_refresh = (
            force_refresh or 
            self.cache_age_seconds > self._cache_max_age or 
            self.cache_age_seconds == 0
        )
        
        if should_refresh:
            await self._refresh_cache(category)
        
        # Get from database
        articles = await self.db.get_news_articles(category=category, limit=limit)
        
        return articles
    
    async def _refresh_cache(self, category: Optional[str] = None):
        """Fetch fresh articles and store in database."""
        print(f"Refreshing news cache (category: {category})...")
        
        try:
            if category:
                articles = await fetch_feed_for_category(category)
            else:
                articles = await fetch_all_feeds()
            
            if articles and self.db:
                # Bulk insert articles
                await self.db.bulk_insert_news_articles(articles)
                
                # Cleanup old articles (older than 30 days)
                deleted = await self.db.cleanup_old_news_articles(30)
                if deleted > 0:
                    print(f"Cleaned up {deleted} old news articles")
                
                print(f"Refreshed {len(articles)} news articles")
                self.cache_age_seconds = 0
            else:
                print("No articles fetched")
                
        except Exception as e:
            print(f"Error refreshing news cache: {e}")
    
    async def get_latest_headlines(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get latest headlines across all categories."""
        return await self.get_news(category=None, limit=limit)


def get_news_fetcher(db=None) -> RSSNewsFetcher:
    """Get or create the RSS news fetcher instance."""
    global _news_fetcher_instance
    
    if _news_fetcher_instance is None:
        _news_fetcher_instance = RSSNewsFetcher(db=db)
    elif db is not None:
        _news_fetcher_instance.db = db
    
    return _news_fetcher_instance
