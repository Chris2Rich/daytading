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

# Image fetcher configuration
IMAGE_FETCH_CONCURRENCY = 20
IMAGE_FETCH_TIMEOUT = 20  # seconds
PLACEHOLDER_IMAGE_URL = "https://placehold.co/341x192?text=NEWS"
MAX_HEADER_SIZE = 65536  # bytes - increased from default 8190 for sites with large cookies

# RSS Feed sources
RSS_FEEDS = {
    # Bloomberg feeds - returns .rss file with Content-Disposition: attachment
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
    # CNBC feeds
    "cnbc_top": {
        "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "category": "general",
        "source_name": "CNBC",
    },
    # Yahoo Finance
    "yahoo_finance": {
        "url": "https://finance.yahoo.com/news/rssindex",
        "category": "general",
        "source_name": "Yahoo Finance",
    },
    # MarketWatch
    "marketwatch_top": {
        "url": "https://feeds.marketwatch.com/marketwatch/topstories",
        "category": "general",
        "source_name": "MarketWatch",
    },
    # Wall Street Journal - returns .xml file
    "wsj_markets": {
        "url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "category": "general",
        "source_name": "Wall Street Journal",
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
    "investing_com": {
        "url": "https://www.investing.com/rss/news.rss",
        "category": "general",
        "source_name": "Investing.com",
    },
    "seeking_alpha": {
        "url": "https://seekingalpha.com/market_currents.xml",
        "category": "general",
        "source_name": "Seeking Alpha",
    },
    "washington_post_business": {
        "url": "https://feeds.washingtonpost.com/rss/business",
        "category": "general",
        "source_name": "Washington Post",
    },
    "business_insider": {
        "url": "http://feeds2.feedburner.com/businessinsider",
        "category": "general",
        "source_name": "Business Insider",
    },
    "financial_times": {
        "url": "https://www.ft.com/?format=rss",
        "category": "general",
        "source_name": "Financial Times",
    },
    "fortune": {
        "url": "http://fortune.com/feed/",
        "category": "general",
        "source_name": "Fortune",
    },
    "economic_times_india": {
        "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "category": "general",
        "source_name": "Economic Times",
    },
    "financial_post_canada": {
        "url": "http://business.financialpost.com/feed",
        "category": "general",
        "source_name": "Financial Post",
    },
    "sfgate_business": {
        "url": "https://www.sfgate.com/rss/feed/Business-and-Technology-News-448.php",
        "category": "general",
        "source_name": "SFGate",
    },
    "entrepreneur": {
        "url": "https://www.entrepreneur.com/latest.rss",
        "category": "general",
        "source_name": "Entrepreneur",
    },
    "express_personal_finance": {
        "url": "http://feeds.feedburner.com/daily-express-personal-finance-news",
        "category": "general",
        "source_name": "Daily Express",
    },
    "nikkei_asia": {
        "url": "https://asia.nikkei.com/rss/feed/nar",
        "category": "general",
        "source_name": "Nikkei Asia",
    },
    "economist_economics": {
        "url": "http://www.economist.com/sections/economics/rss.xml",
        "category": "general",
        "source_name": "The Economist",
    },
    "fast_company": {
        "url": "https://feeds.feedburner.com/fastcompany/headlines",
        "category": "general",
        "source_name": "Fast Company",
    },
    "coindesk": {
        "url": "https://feeds.feedburner.com/CoinDesk",
        "category": "crypto",
        "source_name": "CoinDesk",
    },
    "bitcoin_com": {
        "url": "https://news.bitcoin.com/feed/",
        "category": "crypto",
        "source_name": "Bitcoin.com",
    },
    "cointelegraph": {
        "url": "https://cointelegraph.com/feed",
        "category": "crypto",
        "source_name": "Cointelegraph",
    },
    "bitcoin_magazine": {
        "url": "https://bitcoinmagazine.com/feed",
        "category": "crypto",
        "source_name": "Bitcoin Magazine",
    },
    "cryptoslate": {
        "url": "https://cryptoslate.com/feed/",
        "category": "crypto",
        "source_name": "CryptoSlate",
    },
    "newsbtc": {
        "url": "http://www.newsbtc.com/feed/",
        "category": "crypto",
        "source_name": "NewsBTC",
    },
    "coinspeaker": {
        "url": "https://www.coinspeaker.com/feed/",
        "category": "crypto",
        "source_name": "Coinspeaker",
    },
    "99bitcoins": {
        "url": "https://99bitcoins.com/feed/",
        "category": "crypto",
        "source_name": "99Bitcoins",
    },
    "the_merkle": {
        "url": "https://themerkle.com/feed/",
        "category": "crypto",
        "source_name": "The Merkle",
    },
    "coincentral": {
        "url": "https://coincentral.com/feed/",
        "category": "crypto",
        "source_name": "CoinCentral",
    },
    "live_bitcoin_news": {
        "url": "http://www.livebitcoinnews.com/feed",
        "category": "crypto",
        "source_name": "Live Bitcoin News",
    },
    "bankless_times": {
        "url": "https://www.banklesstimes.com/feed/",
        "category": "crypto",
        "source_name": "BanklessTimes",
    },
    "forex_news_now": {
        "url": "http://www.binaryoptionsnow.com/feed/",
        "category": "forex",
        "source_name": "Forex News Now",
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


# HTTP headers to mimic a browser request (helps avoid 403 from some feeds)
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, application/atom+xml, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


async def fetch_feed(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """Fetch RSS feed content."""
    try:
        async with session.get(url, headers=HTTP_HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as response:
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


async def fetch_article_thumbnail(
    session: aiohttp.ClientSession,
    article_url: str,
    semaphore: asyncio.Semaphore
) -> Optional[str]:
    """
    Fetch article page and extract og:image or twitter:image.
    
    Returns image URL or None if not found.
    """
    async with semaphore:
        try:
            async with session.get(
                article_url,
                headers=HTTP_HEADERS,
                timeout=aiohttp.ClientTimeout(total=IMAGE_FETCH_TIMEOUT),
                allow_redirects=True
            ) as response:
                if response.status != 200:
                    return None
                    
                html = await response.text()
                
                og_match = re.search(
                    r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                    html, re.IGNORECASE
                )
                if og_match:
                    return og_match.group(1)
                
                og_match2 = re.search(
                    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
                    html, re.IGNORECASE
                )
                if og_match2:
                    return og_match2.group(1)
                
                tw_match = re.search(
                    r'<meta[^>]+(?:name|property)=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
                    html, re.IGNORECASE
                )
                if tw_match:
                    return tw_match.group(1)
                
                tw_match2 = re.search(
                    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:name|property)=["\']twitter:image["\']',
                    html, re.IGNORECASE
                )
                if tw_match2:
                    return tw_match2.group(1)
                
                link_match = re.search(
                    r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)["\']',
                    html, re.IGNORECASE
                )
                if link_match:
                    return link_match.group(1)
                    
                return None
                
        except Exception as e:
            print(f"Error fetching thumbnail for {article_url}: {e}")
            return None


async def fetch_missing_thumbnails(
    session: aiohttp.ClientSession,
    articles: List[ParsedArticle]
) -> List[ParsedArticle]:
    """
    Fetch thumbnails for articles that don't have images.
    Updates articles in-place with found image URLs or placeholder.
    """
    articles_needing_images = [
        (i, article) for i, article in enumerate(articles)
        if not article.url_to_image
    ]
    
    if not articles_needing_images:
        return articles
    
    print(f"Fetching thumbnails for {len(articles_needing_images)} articles...")
    
    semaphore = asyncio.Semaphore(IMAGE_FETCH_CONCURRENCY)
    
    async def fetch_and_update(idx: int, article: ParsedArticle) -> ParsedArticle:
        image_url = await fetch_article_thumbnail(session, article.url, semaphore)
        return ParsedArticle(
            title=article.title,
            description=article.description,
            url=article.url,
            url_to_image=image_url if image_url else PLACEHOLDER_IMAGE_URL,
            source_name=article.source_name,
            source_url=article.source_url,
            category=article.category,
            published_at=article.published_at
        )
    
    tasks = [
        fetch_and_update(idx, article)
        for idx, article in articles_needing_images
    ]
    
    updated_articles = await asyncio.gather(*tasks, return_exceptions=True)
    
    for (orig_idx, _), updated in zip(articles_needing_images, updated_articles):
        if isinstance(updated, ParsedArticle):
            articles[orig_idx] = updated
    
    return articles


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
    
    connector = aiohttp.TCPConnector(limit=IMAGE_FETCH_CONCURRENCY)
    async with aiohttp.ClientSession(
        connector=connector,
        max_line_size=MAX_HEADER_SIZE,
        max_field_size=MAX_HEADER_SIZE
    ) as session:
        tasks = []
        for feed_key, feed_info in RSS_FEEDS.items():
            tasks.append(fetch_and_parse_feed(session, feed_key, feed_info))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                print(f"Error fetching feed: {result}")
        
        all_articles = await fetch_missing_thumbnails(session, all_articles)
    
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
    
    connector = aiohttp.TCPConnector(limit=IMAGE_FETCH_CONCURRENCY)
    async with aiohttp.ClientSession(
        connector=connector,
        max_line_size=MAX_HEADER_SIZE,
        max_field_size=MAX_HEADER_SIZE
    ) as session:
        tasks = []
        for feed_key, feed_info in RSS_FEEDS.items():
            if feed_info["category"] == category:
                tasks.append(fetch_and_parse_feed(session, feed_key, feed_info))
        
        if not tasks:
            tasks = [
                fetch_and_parse_feed(session, feed_key, feed_info)
                for feed_key, feed_info in RSS_FEEDS.items()
            ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
        
        all_articles = await fetch_missing_thumbnails(session, all_articles)
    
    filtered = [a for a in all_articles if a.category == category]
    
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
