import os
import re
import asyncio
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

import aiohttp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARXIV_CATEGORIES = [
    "q-fin.GN", "q-fin.MF", "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.TR",
    "cs.LG", "stat.ML"
]

FINANCIAL_WIKIPEDIA_TOPICS = [
    "Financial_analysis", "Financial_modeling", "Quantitative_analysis_(finance)",
    "Algorithmic_trading", "High-frequency_trading", "Statistical_arbitrage",
    "Pairs_trading", "Momentum_investing", "Mean_reversion", "Value_investing",
    "Portfolio_theory", "Modern_portfolio_theory", "Capital_asset_pricing_model",
    "Arbitrage_pricing_theory", "Black%E2%80%93Scholes_model", "Monte_Carlo_methods_in_finance",
    "Risk_management", "Value_at_risk", "Expected_shortfall", "Sharpe_ratio",
    "Technical_analysis", "Fundamental_analysis", "Efficient-market_hypothesis",
    "Behavioral_finance", "Derivative_(finance)", "Option_(finance)",
    "Futures_contract", "Credit_default_swap", "Collateralized_debt_obligation",
    "Hedge_fund", "Mutual_fund", "Exchange-traded_fund", "Investment_banking",
    "Corporate_finance", "Mergers_and_acquisitions", "Leveraged_buyout",
    "Private_equity", "Venture_capital", "Decentralized_finance",
    "Machine_learning", "Deep_learning", "Reinforcement_learning",
    "Sentiment_analysis", "Time_series", "GARCH",
]

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
WIKIPEDIA_API_BASE = "https://en.wikipedia.org/api/rest_v1"


@dataclass
class CachedArticle:
    title: str
    abstract: str
    source: str
    url: str
    authors: List[str]
    categories: List[str]
    published_date: Optional[datetime]
    fetched_at: datetime = field(default_factory=datetime.now)


class SimpleTextSearch:
    """Simple in-memory text search index"""
    
    def __init__(self):
        self.articles: Dict[str, CachedArticle] = {}
    
    def add(self, article: CachedArticle):
        article_id = hashlib.md5(article.url.encode()).hexdigest()[:16]
        self.articles[article_id] = article
    
    def search(self, query: str, limit: int = 10, source: Optional[str] = None) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for article_id, article in self.articles.items():
            if source and article.source != source:
                continue
            
            title_lower = article.title.lower()
            abstract_lower = article.abstract.lower()
            
            title_words = set(title_lower.split())
            abstract_words = set(abstract_lower.split())
            
            title_match = len(query_words & title_words)
            abstract_match = len(query_words & abstract_words)
            
            if title_match > 0 or abstract_match > 0:
                score = title_match * 3 + abstract_match
                results.append({
                    "id": article_id,
                    "title": article.title,
                    "abstract": article.abstract,
                    "source": article.source,
                    "url": article.url,
                    "authors": article.authors,
                    "categories": article.categories,
                    "published_date": article.published_date.isoformat() if article.published_date else None,
                    "score": score
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


class ResearchIngestion:
    """Lightweight research ingestion with text search and caching.
    
    Fetches from arXiv/Wikipedia on demand with simple text matching.
    No ML/embeddings - uses basic keyword matching.
    """
    
    def __init__(self, db=None):
        self.db = db
        self.search_index = SimpleTextSearch()
        self._initialized = False
    
    async def _ensure_initialized(self):
        if self._initialized:
            return
        
        logger.info("Initializing research index...")
        
        await self._load_from_database()
        
        if not self.search_index.articles:
            logger.info("No cached articles found, fetching from APIs...")
            await self._fetch_all_sources()
        
        self._initialized = True
        logger.info(f"Research index loaded: {len(self.search_index.articles)} articles")
    
    async def _load_from_database(self):
        if not self.db:
            return
        
        try:
            urls = await self.db.get_all_research_urls()
            for url in urls:
                paper = await self.db.get_research_paper_by_url(url)
                if paper:
                    self.search_index.add(CachedArticle(
                        title=paper.get("title", ""),
                        abstract=paper.get("abstract", ""),
                        source=paper.get("source", ""),
                        url=paper.get("url", ""),
                        authors=paper.get("authors", []),
                        categories=paper.get("categories", []),
                        published_date=paper.get("published_date")
                    ))
        except Exception as e:
            logger.warning(f"Could not load from database: {e}")
    
    async def _fetch_all_sources(self):
        await self._fetch_arxiv()
        await self._fetch_wikipedia()
    
    async def _fetch_arxiv(self):
        logger.info("Fetching arXiv papers...")
        
        for cat in ARXIV_CATEGORIES:
            params = {
                "search_query": f"cat:{cat}",
                "start": 0,
                "max_results": 30,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(ARXIV_API_BASE, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status != 200:
                            continue
                        
                        text = await resp.text()
                        entry_pattern = r"<entry>(.*?)</entry>"
                        
                        for match in re.finditer(entry_pattern, text, re.DOTALL):
                            entry = match.group(1)
                            
                            title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
                            abstract_match = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
                            url_match = re.search(r"<id>(.*?)</id>", entry)
                            published_match = re.search(r"<published>(.*?)</published>", entry)
                            author_matches = re.findall(r"<name>(.*?)</name>", entry)
                            cat_matches = re.findall(r"<category term=\"([^\"]+)\"", entry)
                            
                            if not title_match or not abstract_match:
                                continue
                            
                            title = title_match.group(1).strip()
                            abstract = abstract_match.group(1).strip()
                            url = url_match.group(1).strip() if url_match else ""
                            published = None
                            if published_match:
                                try:
                                    published = datetime.fromisoformat(published_match.group(1).replace("Z", "+00:00"))
                                except:
                                    pass
                            
                            article = CachedArticle(
                                title=title,
                                abstract=abstract,
                                source="arxiv",
                                url=url,
                                authors=author_matches,
                                categories=cat_matches,
                                published_date=published
                            )
                            self.search_index.add(article)
                            
                            if self.db:
                                try:
                                    await self.db.insert_research_paper(
                                        title=title,
                                        abstract=abstract,
                                        source="arxiv",
                                        url=url,
                                        authors=author_matches,
                                        categories=cat_matches,
                                        published_date=published
                                    )
                                except:
                                    pass
                            
                            await asyncio.sleep(0.1)
                            
            except Exception as e:
                logger.warning(f"Error fetching arXiv {cat}: {e}")
        
        logger.info("arXiv fetch complete")
    
    async def _fetch_wikipedia(self):
        logger.info("Fetching Wikipedia articles...")
        
        for topic in FINANCIAL_WIKIPEDIA_TOPICS:
            url = f"{WIKIPEDIA_API_BASE}/page/summary/{topic}"
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status != 200:
                            continue
                        
                        data = await resp.json()
                        
                        article = CachedArticle(
                            title=data.get("title", topic),
                            abstract=data.get("extract", ""),
                            source="wikipedia",
                            url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                            authors=[],
                            categories=["finance", "knowledge"],
                            published_date=None
                        )
                        self.search_index.add(article)
                        
                        if self.db:
                            try:
                                await self.db.insert_research_paper(
                                    title=article.title,
                                    abstract=article.abstract,
                                    source="wikipedia",
                                    url=article.url,
                                    authors=[],
                                    categories=["finance", "knowledge"],
                                    published_date=None
                                )
                            except:
                                pass
                        
                        await asyncio.sleep(0.1)
                        
            except Exception as e:
                logger.warning(f"Error fetching Wikipedia {topic}: {e}")
        
        logger.info("Wikipedia fetch complete")
    
    async def search_research(
        self,
        query: str,
        limit: int = 10,
        source: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        await self._ensure_initialized()
        
        results = self.search_index.search(query, limit=limit, source=source)
        
        if category:
            results = [r for r in results if category in r.get("categories", [])]
        
        return results


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
            response += f"   {result['abstract'][:300]}...\n"
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
        return f"**{result['title']}** (from Wikipedia)\n\n{result['abstract']}\n\n🔗 {result.get('url', 'N/A')}"

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
            response += f"   {result['abstract'][:200]}...\n"
            response += f"   🔗 {result.get('url', 'N/A')}\n\n"
        
        return response
