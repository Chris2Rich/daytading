import os
import re
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import urllib.parse

import aiohttp
import torch
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np

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
    "Private_equity", "Venture_capital", "Initial_coin_offering",
    "Decentralized_finance", "Machine_learning", "Deep_learning",
    "Reinforcement_learning", "Natural_language_processing",
    "Sentiment_analysis", "Time_series", "GARCH", "Autoregressive_model",
]

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
WIKIPEDIA_API_BASE = "https://en.wikipedia.org/api/rest_v1"

SERVER_URL = os.environ.get("GPU_SYNC_SERVER", "https://api.zecant.com")
BATCH_SIZE = int(os.environ.get("GPU_SYNC_BATCH", "10"))


@dataclass
class ResearchPaper:
    title: str
    abstract: str
    source: str
    url: str
    authors: List[str]
    categories: List[str]
    published_date: Optional[datetime] = None


class GPUSyncClient:
    def __init__(self, server_url: str = SERVER_URL):
        self.server_url = server_url.rstrip("/")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=self.device
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = 512
        
        self.synced_urls: set = set()
        self.stats = {"arxiv": 0, "wikipedia": 0, "errors": 0}
    
    async def get_synced_urls(self) -> set:
        """Get all URLs already synced from server"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.server_url}/dogchasedcat33/synced") as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to get synced URLs: {resp.status}")
                data = await resp.json()
                urls = data.get("urls", [])
                logger.info(f"Already synced: {len(urls)} URLs")
                return set(urls)
    
    async def upload_article(self, article_data: dict) -> bool:
        """Upload article package to server"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_url}/dogchasedcat33/sync",
                json=article_data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Upload failed: {resp.status} - {error}")
                    self.stats["errors"] += 1
                    return False
                result = await resp.json()
                logger.info(f"Uploaded: {article_data.get('title', 'Unknown')[:50]}...")
                return True
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks by tokens"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        return chunks
    
    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """Embed text chunks using all-MiniLM-L6-v2"""
        embeddings = self.embedding_model.encode(
            chunks,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    async def fetch_arxiv_paper(self, url: str) -> Optional[ResearchPaper]:
        """Fetch a single arXiv paper by URL"""
        paper_id = url.split("/abs/")[-1] if "/abs/" in url else None
        if not paper_id:
            return None
        
        params = {"id_list": paper_id}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(ARXIV_API_BASE, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        return None
                    
                    text = await resp.text()
                    entry_pattern = r"<entry>(.*?)</entry>"
                    match = re.search(entry_pattern, text, re.DOTALL)
                    if not match:
                        return None
                    
                    entry = match.group(1)
                    
                    title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
                    abstract_match = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
                    url_match = re.search(r"<id>(.*?)</id>", entry)
                    published_match = re.search(r"<published>(.*?)</published>", entry)
                    author_matches = re.findall(r"<name>(.*?)</name>", entry)
                    cat_matches = re.findall(r"<category term=\"([^\"]+)\"", entry)
                    
                    if not title_match or not abstract_match:
                        return None
                    
                    title = title_match.group(1).strip()
                    abstract = abstract_match.group(1).strip()
                    url = url_match.group(1).strip() if url_match else url
                    published = None
                    if published_match:
                        try:
                            published = datetime.fromisoformat(published_match.group(1).replace("Z", "+00:00"))
                        except:
                            pass
                    
                    return ResearchPaper(
                        title=title,
                        abstract=abstract,
                        source="arxiv",
                        url=url,
                        authors=author_matches,
                        categories=cat_matches,
                        published_date=published
                    )
            except Exception as e:
                logger.error(f"Error fetching arXiv {url}: {e}")
                return None
    
    async def fetch_wikipedia_article(self, topic: str) -> Optional[Dict]:
        """Fetch a Wikipedia article"""
        url = f"{WIKIPEDIA_API_BASE}/page/summary/{topic}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        return None
                    
                    data = await resp.json()
                    
                    return {
                        "title": data.get("title", topic),
                        "abstract": data.get("extract", ""),
                        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    }
            except Exception as e:
                logger.error(f"Error fetching Wikipedia {topic}: {e}")
                return None
    
    async def sync_arxiv(self, max_results: int = 100) -> int:
        """Sync new arXiv papers"""
        logger.info(f"Fetching arXiv papers (max {max_results})...")
        
        count = 0
        for cat in ARXIV_CATEGORIES:
            params = {
                "search_query": f"cat:{cat}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(ARXIV_API_BASE, params=params, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        if resp.status != 200:
                            logger.warning(f"arXiv API error for {cat}: {resp.status}")
                            continue
                        
                        text = await resp.text()
                        
                        entry_pattern = r"<entry>(.*?)</entry>"
                        for match in re.finditer(entry_pattern, text, re.DOTALL):
                            entry = match.group(1)
                            
                            url_match = re.search(r"<id>(.*?)</id>", entry)
                            if not url_match:
                                continue
                            
                            url = url_match.group(1).strip()
                            
                            if url in self.synced_urls:
                                continue
                            
                            paper = await self.fetch_arxiv_paper(url)
                            if not paper:
                                continue
                            
                            chunks = self.chunk_text(f"{paper.title}\n\n{paper.abstract}")
                            embeddings = self.embed_chunks(chunks)
                            
                            article_data = {
                                "title": paper.title,
                                "abstract": paper.abstract,
                                "source": "arxiv",
                                "url": paper.url,
                                "authors": paper.authors,
                                "categories": paper.categories,
                                "published_date": paper.published_date.isoformat() if paper.published_date else None,
                                "chunks": [
                                    {"text": chunk, "embedding": emb}
                                    for chunk, emb in zip(chunks, embeddings)
                                ]
                            }
                            
                            success = await self.upload_article(article_data)
                            if success:
                                count += 1
                                self.stats["arxiv"] += 1
                                self.synced_urls.add(url)
                            
                            await asyncio.sleep(0.5)
                            
                except Exception as e:
                    logger.error(f"Error fetching arXiv category {cat}: {e}")
        
        return count
    
    async def sync_wikipedia(self) -> int:
        """Sync new Wikipedia articles"""
        logger.info(f"Fetching Wikipedia articles...")
        
        count = 0
        for topic in FINANCIAL_WIKIPEDIA_TOPICS:
            article = await self.fetch_wikipedia_article(topic)
            if not article:
                continue
            
            url = article.get("url", "")
            if url in self.synced_urls or not url:
                continue
            
            chunks = self.chunk_text(article["abstract"])
            embeddings = self.embed_chunks(chunks)
            
            article_data = {
                "title": article["title"],
                "abstract": article["abstract"],
                "source": "wikipedia",
                "url": url,
                "authors": [],
                "categories": ["finance", "knowledge"],
                "published_date": None,
                "chunks": [
                    {"text": chunk, "embedding": emb}
                    for chunk, emb in zip(chunks, embeddings)
                ]
            }
            
            success = await self.upload_article(article_data)
            if success:
                count += 1
                self.stats["wikipedia"] += 1
                self.synced_urls.add(url)
            
            await asyncio.sleep(0.5)
        
        return count
    
    async def run_full_sync(self):
        """Run full sync"""
        logger.info("=" * 50)
        logger.info("Starting GPU Sync")
        logger.info("=" * 50)
        
        logger.info("Fetching synced URLs from server...")
        self.synced_urls = await self.get_synced_urls()
        
        logger.info(f"Starting arXiv sync...")
        arxiv_count = await self.sync_arxiv()
        logger.info(f"Synced {arxiv_count} arXiv papers")
        
        logger.info(f"Starting Wikipedia sync...")
        wiki_count = await self.sync_wikipedia()
        logger.info(f"Synced {wiki_count} Wikipedia articles")
        
        logger.info("=" * 50)
        logger.info("Sync Complete!")
        logger.info(f"  arXiv: {self.stats['arxiv']}")
        logger.info(f"  Wikipedia: {self.stats['wikipedia']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        logger.info("=" * 50)


async def main():
    client = GPUSyncClient()
    await client.run_full_sync()


if __name__ == "__main__":
    asyncio.run(main())
