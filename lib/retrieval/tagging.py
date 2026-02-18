"""
Automatic Tag Extraction for News Articles
Extracts relevant financial tags from article content using keyword matching.
"""

import re
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass


FINANCIAL_TAGS: Dict[str, Set[str]] = {
    "stocks": {"stock", "stocks", "share", "shares", "equity", "equities", "nasdaq", "dow", "sp500", "s&p", "nyse"},
    "crypto": {"crypto", "cryptocurrency", "bitcoin", "btc", "ethereum", "eth", "blockchain", "defi", "nft", "nfts", "altcoin", "altcoins", "web3", "coinbase"},
    "forex": {"forex", "currency", "currencies", "exchange rate", "fx", "dollar", "euro", "yen", "yuan", "pound", "gbp", "usd", "eur"},
    "bonds": {"bond", "bonds", "treasury", "treasuries", "yield", "debt", "fixed income"},
    "etf": {"etf", "etfs", "index fund", "mutual fund", "vanguard", "blackrock", "ishares"},
    "commodities": {"commodity", "commodities", "oil", "crude", "gold", "silver", "natural gas", "copper", "wheat", "corn"},
    "earnings": {"earnings", "quarterly", "revenue", "profit", "loss", "eps", "guidance", "forecast", "beat", "miss"},
    "ipo": {"ipo", "initial public offering", "go public", "debut", "listing"},
    "mergers": {"merger", "acquisition", "m&a", "buyout", "takeover", "deal", "acquire"},
    "fed": {"fed", "federal reserve", "interest rate", "rates", "monetary policy", "powell", "fomc", "basis point", "rate cut", "rate hike"},
    "inflation": {"inflation", "cpi", "consumer price", "pce", "deflation", "price increase"},
    "recession": {"recession", "downturn", "contraction", "economic slowdown", "bear market"},
    "tech": {"tech", "technology", "software", "ai", "artificial intelligence", "machine learning", "cloud", "saas", "semiconductor", "chip", "nvidia", "apple", "microsoft", "google", "amazon", "meta"},
    "banking": {"bank", "banking", "lender", "loan", "credit", "mortgage", "jpmorgan", "goldman", "morgan stanley", "citi"},
    "healthcare": {"healthcare", "health", "pharma", "pharmaceutical", "drug", "fda", "biotech", "medical", "hospital", "insurance"},
    "energy": {"energy", "oil", "gas", "renewable", "solar", "wind", "electric", "ev", "electric vehicle", "tesla", "exxon", "chevron"},
    "retail": {"retail", "consumer", "shopping", "e-commerce", "amazon", "walmart", "target", "store"},
    "automotive": {"automotive", "auto", "car", "vehicle", "ev", "electric vehicle", "tesla", "ford", "gm", "toyota"},
    "china": {"china", "chinese", "beijing", "shanghai", "xi", "yuan"},
    "europe": {"europe", "european", "eu", "eurozone", "germany", "france", "uk", "britain", "brexit"},
    "asia": {"asia", "asian", "japan", "korea", "india", "singapore", "hong kong"},
    "emerging": {"emerging market", "developing", "brics", "brazil", "russia", "india", "china"},
    "real_estate": {"real estate", "property", "housing", "home", "mortgage", "reit", "commercial property"},
    "regulation": {"regulation", "regulatory", "sec", "lawsuit", "fine", "antitrust", "investigation", "compliance"},
    "layoffs": {"layoff", "layoffs", "job cut", "firing", "termination", "workforce reduction", "restructuring"},
    "dividend": {"dividend", "dividends", "payout", "distribution", "yield"},
    "buyback": {"buyback", "share repurchase", "repurchase", "buy back"},
    "unemployment": {"unemployment", "jobs", "employment", "jobless", "payroll", "labor market"},
    "gdp": {"gdp", "gross domestic", "economic growth", "economic output"},
    "trade": {"trade", "tariff", "import", "export", "trade war", "commerce"},
}

SOURCE_TAG_MAPPING: Dict[str, List[str]] = {
    "coindesk": ["crypto", "bitcoin"],
    "cointelegraph": ["crypto", "bitcoin"],
    "bitcoin.com": ["crypto", "bitcoin"],
    "bitcoin magazine": ["crypto", "bitcoin"],
    "cryptoslate": ["crypto"],
    "newsbtc": ["crypto", "bitcoin"],
    "coinspeaker": ["crypto"],
    "99bitcoins": ["crypto", "bitcoin"],
    "the merkle": ["crypto"],
    "coincentral": ["crypto"],
    "live bitcoin news": ["crypto", "bitcoin"],
    "banklesstimes": ["crypto"],
    "forex news now": ["forex"],
    "dailyfx": ["forex"],
    "bloomberg": ["markets"],
    "cnbc": ["markets"],
    "reuters": ["markets"],
    "wsj": ["markets"],
    "financial times": ["markets"],
    "marketwatch": ["markets"],
    "yahoo finance": ["markets"],
    "seeking alpha": ["stocks"],
    "investing.com": ["markets"],
    "the economist": ["economics"],
    "ft": ["markets"],
    "fortune": ["business"],
    "forbes": ["business"],
    "business insider": ["business"],
    "entrepreneur": ["business"],
}

MAX_TAGS_PER_ARTICLE = 5
TAG_STOP_WORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}


@dataclass
class ArticleContent:
    title: str
    description: Optional[str]
    source_name: str
    category: str


def normalize_text(text: str) -> str:
    """Normalize text for tag matching."""
    if not text:
        return ""
    return text.lower().strip()


def extract_tags(article: ArticleContent) -> List[str]:
    """
    Extract relevant tags from an article using keyword matching and source mapping.
    
    Args:
        article: ArticleContent with title, description, source_name, and category
        
    Returns:
        List of tags (max MAX_TAGS_PER_ARTICLE)
    """
    tags: Set[str] = set()
    
    source_name = normalize_text(article.source_name)
    category = normalize_text(article.category)
    
    for source_pattern, source_tags in SOURCE_TAG_MAPPING.items():
        if source_pattern in source_name:
            tags.update(source_tags)
    
    if category in ["crypto", "forex"]:
        tags.add(category)
    elif category == "technology":
        tags.add("tech")
    elif category == "general":
        tags.add("markets")
    
    combined_text = normalize_text(f"{article.title} {article.description or ''}")
    
    tag_scores: Dict[str, int] = {}
    
    for tag_name, keywords in FINANCIAL_TAGS.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, combined_text))
            if matches > 0:
                tag_scores[tag_name] = tag_scores.get(tag_name, 0) + matches
    
    sorted_tags = sorted(tag_scores.keys(), key=lambda t: tag_scores[t], reverse=True)
    
    for tag in sorted_tags:
        if len(tags) >= MAX_TAGS_PER_ARTICLE:
            break
        tags.add(tag)
    
    return list(tags)[:MAX_TAGS_PER_ARTICLE]


def extract_tags_from_dict(article_dict: Dict[str, Any]) -> List[str]:
    """
    Convenience function to extract tags from a dictionary.
    
    Args:
        article_dict: Dictionary with 'title', 'description', 'source_name', 'category' keys
        
    Returns:
        List of tags
    """
    article = ArticleContent(
        title=article_dict.get('title', ''),
        description=article_dict.get('description'),
        source_name=article_dict.get('source_name', ''),
        category=article_dict.get('category', 'general')
    )
    return extract_tags(article)
