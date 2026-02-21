import asyncpg
import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json

class Database:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.db_url = os.environ.get("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable is required")

    async def initialize(self):
        """Create connection pool and initialize schema"""
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        await self._create_tables()

    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()

    async def _create_tables(self):
        """Create database tables if they don't exist"""
        async with self.pool.acquire() as conn:
            # Users table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    clerk_user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            # Chats table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id UUID PRIMARY KEY,
                    clerk_user_id TEXT NOT NULL REFERENCES users(clerk_user_id) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    title TEXT
                )
            """)

            # Create index on clerk_user_id for faster lookups
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chats_clerk_user_id 
                ON chats(clerk_user_id)
            """)

            # Messages table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id UUID PRIMARY KEY,
                    chat_id UUID NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
                    clerk_user_id TEXT NOT NULL REFERENCES users(clerk_user_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
                    content TEXT NOT NULL,
                    tools JSONB,
                    tool_parameters JSONB,
                    tool_results JSONB,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    sequence_number INTEGER NOT NULL
                )
            """)

            # Create indexes for faster queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_chat_id 
                ON messages(chat_id, sequence_number)
            """)

            # Portfolio cache table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_cache (
                    clerk_user_id TEXT PRIMARY KEY REFERENCES users(clerk_user_id) ON DELETE CASCADE,
                    portfolio_data JSONB NOT NULL,
                    last_fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    snaptrade_connection_status TEXT NOT NULL DEFAULT 'connected' 
                        CHECK (snaptrade_connection_status IN ('connected', 'disconnected', 'error'))
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_portfolio_cache_last_fetched 
                ON portfolio_cache(clerk_user_id, last_fetched_at DESC)
            """)

            # Strategies table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id UUID PRIMARY KEY,
                    clerk_user_id TEXT NOT NULL REFERENCES users(clerk_user_id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    description TEXT,
                    config JSONB NOT NULL,
                    status TEXT NOT NULL DEFAULT 'draft' 
                        CHECK (status IN ('draft', 'backtesting', 'completed', 'failed')),
                    tags TEXT[] DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_strategies_clerk_user_id 
                ON strategies(clerk_user_id, updated_at DESC)
            """)

            # Backtest results table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    backtest_id UUID PRIMARY KEY,
                    strategy_id UUID NOT NULL REFERENCES strategies(strategy_id) ON DELETE CASCADE,
                    clerk_user_id TEXT NOT NULL REFERENCES users(clerk_user_id) ON DELETE CASCADE,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'running', 'completed', 'failed')),
                    error_message TEXT,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    duration_seconds INT,
                    starting_capital NUMERIC,
                    ending_capital NUMERIC,
                    portfolio JSONB,
                    risk JSONB,
                    equity_curve JSONB,
                    drawdown_curve JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy_id 
                ON backtest_results(strategy_id, created_at DESC)
            """)

            # Trades table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id UUID PRIMARY KEY,
                    backtest_id UUID NOT NULL REFERENCES backtest_results(backtest_id) ON DELETE CASCADE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
                    quantity NUMERIC NOT NULL,
                    price NUMERIC NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    pnl NUMERIC,
                    pnl_percent NUMERIC
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_backtest_id 
                ON trades(backtest_id, timestamp DESC)
            """)

            # News articles cache table (RSS feeds)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id UUID PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    url TEXT NOT NULL UNIQUE,
                    url_to_image TEXT,
                    source_name TEXT NOT NULL,
                    source_url TEXT,
                    category TEXT NOT NULL,
                    published_at TIMESTAMPTZ NOT NULL,
                    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_articles_category 
                ON news_articles(category, published_at DESC)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_articles_fetched 
                ON news_articles(fetched_at DESC)
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS article_tags (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    article_id UUID NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
                    tag TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(article_id, tag)
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_article_tags_article 
                ON article_tags(article_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_article_tags_tag 
                ON article_tags(tag)
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS research_papers (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title TEXT NOT NULL,
                    abstract TEXT,
                    source TEXT NOT NULL CHECK (source IN ('arxiv', 'wikipedia')),
                    url TEXT UNIQUE,
                    authors TEXT[],
                    categories TEXT[],
                    published_date DATE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS research_chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    paper_id UUID REFERENCES research_papers(id) ON DELETE CASCADE,
                    chunk_text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding vector(1024),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_research_chunks_embedding 
                ON research_chunks 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS context_summaries (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    chat_id UUID NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
                    turn_start INTEGER NOT NULL,
                    turn_end INTEGER NOT NULL,
                    summary_text TEXT NOT NULL,
                    key_entities JSONB DEFAULT '{}',
                    token_count INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_context_summaries_chat_id 
                ON context_summaries(chat_id, turn_start)
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_memory (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    chat_id UUID NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
                    entity_type TEXT NOT NULL CHECK (entity_type IN ('ticker', 'strategy', 'paper', 'concept', 'decision')),
                    entity_value TEXT NOT NULL,
                    context JSONB DEFAULT '{}',
                    first_mentioned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_mentioned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    mention_count INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(chat_id, entity_type, entity_value)
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entity_memory_chat_id 
                ON entity_memory(chat_id)
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS research_papers (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title TEXT NOT NULL,
                    abstract TEXT,
                    source TEXT NOT NULL CHECK (source IN ('arxiv', 'wikipedia')),
                    url TEXT UNIQUE,
                    authors TEXT[],
                    categories TEXT[],
                    published_date DATE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS research_chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    paper_id UUID REFERENCES research_papers(id) ON DELETE CASCADE,
                    chunk_text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding vector(1024),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_research_chunks_embedding 
                ON research_chunks 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS context_summaries (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    chat_id UUID NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
                    turn_start INTEGER NOT NULL,
                    turn_end INTEGER NOT NULL,
                    summary_text TEXT NOT NULL,
                    key_entities JSONB DEFAULT '{}',
                    token_count INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_context_summaries_chat_id 
                ON context_summaries(chat_id, turn_start)
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_memory (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    chat_id UUID NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
                    entity_type TEXT NOT NULL CHECK (entity_type IN ('ticker', 'strategy', 'paper', 'concept', 'decision')),
                    entity_value TEXT NOT NULL,
                    context JSONB DEFAULT '{}',
                    first_mentioned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_mentioned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    mention_count INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(chat_id, entity_type, entity_value)
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entity_memory_chat_id 
                ON entity_memory(chat_id)
            """)

    async def ensure_user(self, clerk_user_id: str):
        """Create user if doesn't exist"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO users (clerk_user_id)
                VALUES ($1)
                ON CONFLICT (clerk_user_id) DO NOTHING
            """, clerk_user_id)

    async def create_chat(self, clerk_user_id: str, title: Optional[str] = None) -> str:
        """Create a new chat session"""
        chat_id = str(uuid.uuid4())
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO chats (chat_id, clerk_user_id, title)
                VALUES ($1, $2, $3)
            """, chat_id, clerk_user_id, title)
        return chat_id

    async def add_message(
        self,
        chat_id: str,
        clerk_user_id: str,
        role: str,
        content: str,
        tools: Optional[List[str]] = None,
        tool_parameters: Optional[Dict[str, Any]] = None,
        tool_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a message to a chat"""
        message_id = str(uuid.uuid4())
        
        async with self.pool.acquire() as conn:
            # Get next sequence number
            seq_num = await conn.fetchval("""
                SELECT COALESCE(MAX(sequence_number), 0) + 1
                FROM messages
                WHERE chat_id = $1
            """, chat_id)

            # Convert lists/dicts to JSON strings
            tools_json = json.dumps(tools) if tools else None
            tool_parameters_json = json.dumps(tool_parameters) if tool_parameters else None
            tool_results_json = json.dumps(tool_results) if tool_results else None

            # Insert message
            await conn.execute("""
                INSERT INTO messages (
                    message_id, chat_id, clerk_user_id, role, content,
                    tools, tool_parameters, tool_results, sequence_number
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, message_id, chat_id, clerk_user_id, role, content,
                tools_json, tool_parameters_json, tool_results_json, seq_num)

            # Update chat updated_at timestamp
            await conn.execute("""
                UPDATE chats
                SET updated_at = NOW()
                WHERE chat_id = $1
            """, chat_id)

        return message_id

    async def get_chat_history(
        self, 
        chat_id: str, 
        clerk_user_id: str
    ) -> List[Dict[str, Any]]:
        """Retrieve all messages for a chat (ordered by sequence_number)"""
        async with self.pool.acquire() as conn:
            # Verify ownership
            owner = await conn.fetchval("""
                SELECT clerk_user_id FROM chats WHERE chat_id = $1
            """, chat_id)

            if not owner:
                raise ValueError(f"Chat {chat_id} not found")
            
            if owner != clerk_user_id:
                raise PermissionError(f"User {clerk_user_id} does not own chat {chat_id}")

            # Fetch messages
            rows = await conn.fetch("""
                SELECT 
                    message_id, chat_id, clerk_user_id, role, content,
                    tools, tool_parameters, tool_results, timestamp, sequence_number
                FROM messages
                WHERE chat_id = $1
                ORDER BY sequence_number ASC
            """, chat_id)

            return [dict(row) for row in rows]

    # ============================================================================
    # News Articles Methods
    # ============================================================================

    async def get_news_articles(
        self, 
        category: Optional[str] = None, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get cached news articles, optionally filtered by category.
        """
        async with self.pool.acquire() as conn:
            if category:
                rows = await conn.fetch("""
                    SELECT id, title, description, url, url_to_image, source_name, 
                           source_url, category, published_at, fetched_at
                    FROM news_articles
                    WHERE category = $1
                    ORDER BY published_at DESC
                    LIMIT $2
                """, category, limit)
            else:
                rows = await conn.fetch("""
                    SELECT id, title, description, url, url_to_image, source_name, 
                           source_url, category, published_at, fetched_at
                    FROM news_articles
                    ORDER BY published_at DESC
                    LIMIT $1
                """, limit)
            
            return [dict(row) for row in rows]

    async def upsert_news_article(
        self,
        title: str,
        description: Optional[str],
        url: str,
        url_to_image: Optional[str],
        source_name: str,
        source_url: Optional[str],
        category: str,
        published_at: datetime
    ) -> None:
        """
        Insert or update a news article. Uses URL as unique constraint.
        """
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO news_articles (
                    id, title, description, url, url_to_image, source_name, 
                    source_url, category, published_at, fetched_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                ON CONFLICT (url) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    url_to_image = EXCLUDED.url_to_image,
                    fetched_at = NOW()
            """, str(uuid.uuid4()), title, description, url, url_to_image, 
                source_name, source_url, category, published_at)

    async def bulk_insert_news_articles(
        self, 
        articles: List[Dict[str, Any]]
    ) -> int:
        """
        Bulk insert news articles. Uses ON CONFLICT to handle duplicates.
        Returns the number of articles processed.
        """
        if not articles:
            return 0
        
        async with self.pool.acquire() as conn:
            for article in articles:
                await conn.execute("""
                    INSERT INTO news_articles (
                        id, title, description, url, url_to_image, source_name, 
                        source_url, category, published_at, fetched_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                    ON CONFLICT (url) DO UPDATE SET
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        url_to_image = EXCLUDED.url_to_image,
                        fetched_at = NOW()
                """, str(uuid.uuid4()), article.get('title'), article.get('description'),
                    article.get('url'), article.get('url_to_image'), article.get('source_name'),
                    article.get('source_url'), article.get('category'), article.get('published_at'))
        
        return len(articles)

    async def cleanup_old_news_articles(self, days: int = 30) -> int:
        """
        Delete news articles older than specified days.
        Returns the number of articles deleted.
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM news_articles
                WHERE fetched_at < NOW() - INTERVAL '1 day' * $1
            """, days)
            # Result is like "DELETE 10"
            if "DELETE" in result:
                return int(result.split()[-1])
            return 0

    async def get_news_cache_age(self) -> Optional[float]:
        """
        Get the age of the news cache in seconds.
        Returns None if no articles exist.
        """
        async with self.pool.acquire() as conn:
            last_fetched = await conn.fetchval("""
                SELECT MAX(fetched_at) FROM news_articles
            """)
            
            if not last_fetched:
                return None
            
            now = datetime.now(timezone.utc)
            age = now - last_fetched
            return age.total_seconds()

    async def get_news_articles_paginated(
        self,
        category: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        Get paginated news articles with tags.
        Returns articles, pagination metadata, and tags for each article.
        """
        offset = (page - 1) * page_size
        
        async with self.pool.acquire() as conn:
            count_query = "SELECT COUNT(*) FROM news_articles"
            count_params = []
            
            if category:
                count_query = "SELECT COUNT(*) FROM news_articles WHERE category = $1"
                count_params = [category]
            
            total_count = await conn.fetchval(count_query, *count_params)
            total_pages = (total_count + page_size - 1) // page_size
            
            if category:
                rows = await conn.fetch("""
                    SELECT id, title, description, url, url_to_image, source_name, 
                           source_url, category, published_at, fetched_at
                    FROM news_articles
                    WHERE category = $1
                    ORDER BY published_at DESC
                    LIMIT $2 OFFSET $3
                """, category, page_size, offset)
            else:
                rows = await conn.fetch("""
                    SELECT id, title, description, url, url_to_image, source_name, 
                           source_url, category, published_at, fetched_at
                    FROM news_articles
                    ORDER BY published_at DESC
                    LIMIT $1 OFFSET $2
                """, page_size, offset)
            
            articles = [dict(row) for row in rows]
            
            if articles:
                article_ids = [article['id'] for article in articles]
                tag_rows = await conn.fetch("""
                    SELECT article_id, tag
                    FROM article_tags
                    WHERE article_id = ANY($1)
                """, article_ids)
                
                tags_by_article = {}
                for tag_row in tag_rows:
                    article_id = tag_row['article_id']
                    if article_id not in tags_by_article:
                        tags_by_article[article_id] = []
                    tags_by_article[article_id].append(tag_row['tag'])
                
                for article in articles:
                    article['tags'] = tags_by_article.get(article['id'], [])
            else:
                for article in articles:
                    article['tags'] = []
            
            return {
                'articles': articles,
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total_count': total_count,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_prev': page > 1
                }
            }

    async def get_popular_tags(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get most popular tags with article counts.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT tag, COUNT(*) as count
                FROM article_tags
                GROUP BY tag
                ORDER BY count DESC
                LIMIT $1
            """, limit)
            
            return [{'name': row['tag'], 'count': row['count']} for row in rows]

    async def bulk_insert_article_tags(
        self,
        article_url: str,
        tags: List[str]
    ) -> int:
        """
        Insert tags for an article by URL.
        Returns number of tags inserted.
        """
        if not tags:
            return 0
        
        async with self.pool.acquire() as conn:
            article_id = await conn.fetchval("""
                SELECT id FROM news_articles WHERE url = $1
            """, article_url)
            
            if not article_id:
                return 0
            
            inserted = 0
            for tag in tags:
                try:
                    await conn.execute("""
                        INSERT INTO article_tags (article_id, tag)
                        VALUES ($1, $2)
                        ON CONFLICT (article_id, tag) DO NOTHING
                    """, article_id, tag)
                    inserted += 1
                except Exception:
                    pass
            
            return inserted

    async def delete_article_tags(self, article_url: str) -> int:
        """
        Delete all tags for an article by URL.
        Returns number of tags deleted.
        """
        async with self.pool.acquire() as conn:
            article_id = await conn.fetchval("""
                SELECT id FROM news_articles WHERE url = $1
            """, article_url)
            
            if not article_id:
                return 0
            
            result = await conn.execute("""
                DELETE FROM article_tags WHERE article_id = $1
            """, article_id)
            
            if "DELETE" in result:
                return int(result.split()[-1])
            return 0

    async def get_user_chats(self, clerk_user_id: str) -> List[Dict[str, Any]]:
        """List all chats for a user"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT chat_id, clerk_user_id, created_at, updated_at, title
                FROM chats
                WHERE clerk_user_id = $1
                ORDER BY updated_at DESC
            """, clerk_user_id)

            return [dict(row) for row in rows]

    async def verify_chat_ownership(self, chat_id: str, clerk_user_id: str) -> bool:
        """Verify that a user owns a chat"""
        async with self.pool.acquire() as conn:
            owner = await conn.fetchval("""
                SELECT clerk_user_id FROM chats WHERE chat_id = $1
            """, chat_id)

            return owner == clerk_user_id if owner else False

    async def chat_exists(self, chat_id: str) -> bool:
        """Check if a chat exists"""
        async with self.pool.acquire() as conn:
            exists = await conn.fetchval("""
                SELECT EXISTS(SELECT 1 FROM chats WHERE chat_id = $1)
            """, chat_id)
            return exists

    async def delete_chat(self, chat_id: str):
        """Delete a chat (messages will cascade delete due to foreign key)"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                DELETE FROM chats
                WHERE chat_id = $1
            """, chat_id)

    async def update_chat_title(self, chat_id: str, title: str):
        """Update the title of a chat"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE chats
                SET title = $1
                WHERE chat_id = $2
            """, title, chat_id)

    async def get_portfolio_cache(
        self, 
        clerk_user_id: str,
        max_age_minutes: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached portfolio data if it exists and is fresh enough.
        Returns None if cache miss or too old.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT portfolio_data, last_fetched_at, snaptrade_connection_status
                FROM portfolio_cache
                WHERE clerk_user_id = $1
            """, clerk_user_id)
            
            if not row:
                return None
            
            # Check if cache is fresh enough
            now = datetime.now(timezone.utc)
            age = now - row['last_fetched_at']
            
            if age.total_seconds() > (max_age_minutes * 60):
                return None  # Too old, needs refresh
            
            return {
                'portfolio_data': row['portfolio_data'],
                'last_fetched_at': row['last_fetched_at'],
                'connection_status': row['snaptrade_connection_status']
            }

    async def update_portfolio_cache(
        self,
        clerk_user_id: str,
        portfolio_data: Dict[str, Any],
        connection_status: str = 'connected'
    ):
        """
        Update or insert portfolio cache data.
        """
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO portfolio_cache (
                    clerk_user_id, 
                    portfolio_data, 
                    last_fetched_at,
                    snaptrade_connection_status
                )
                VALUES ($1, $2, NOW(), $3)
                ON CONFLICT (clerk_user_id) 
                DO UPDATE SET
                    portfolio_data = EXCLUDED.portfolio_data,
                    last_fetched_at = NOW(),
                    snaptrade_connection_status = EXCLUDED.snaptrade_connection_status
            """, clerk_user_id, json.dumps(portfolio_data), connection_status)

    async def invalidate_portfolio_cache(self, clerk_user_id: str):
        """
        Delete portfolio cache for a user (forces refresh on next request).
        """
        async with self.pool.acquire() as conn:
            await conn.execute("""
                DELETE FROM portfolio_cache
                WHERE clerk_user_id = $1
            """, clerk_user_id)

    async def get_portfolio_cache_age(self, clerk_user_id: str) -> Optional[float]:
        """
        Get the age of the portfolio cache in seconds.
        Returns None if no cache exists.
        """
        async with self.pool.acquire() as conn:
            last_fetched = await conn.fetchval("""
                SELECT last_fetched_at
                FROM portfolio_cache
                WHERE clerk_user_id = $1
            """, clerk_user_id)
            
            if not last_fetched:
                return None
            
            now = datetime.now(timezone.utc)
            age = now - last_fetched
            return age.total_seconds()

    # ============================================================================
    # Strategy Methods
    # ============================================================================

    async def create_strategy(self, clerk_user_id: str, name: str, description: str, 
                              config: Dict[str, Any], tags: List[str] = None) -> str:
        """Create a new strategy"""
        strategy_id = str(uuid.uuid4())
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO strategies (strategy_id, clerk_user_id, name, description, config, tags)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, strategy_id, clerk_user_id, name, description, 
                json.dumps(config), tags or [])
        return strategy_id

    async def get_strategy(self, strategy_id: str, clerk_user_id: str) -> Optional[Dict[str, Any]]:
        """Get a strategy by ID, verifying ownership"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT strategy_id, clerk_user_id, name, description, config, status, tags,
                       created_at, updated_at
                FROM strategies
                WHERE strategy_id = $1 AND clerk_user_id = $2
            """, strategy_id, clerk_user_id)
            
            if not row:
                return None
            
            result = dict(row)
            # Parse JSON config
            if isinstance(result['config'], str):
                result['config'] = json.loads(result['config'])
            return result

    async def get_user_strategies(self, clerk_user_id: str) -> List[Dict[str, Any]]:
        """List all strategies for a user"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT strategy_id, clerk_user_id, name, description, config, status, tags,
                       created_at, updated_at
                FROM strategies
                WHERE clerk_user_id = $1
                ORDER BY updated_at DESC
            """, clerk_user_id)
            
            results = []
            for row in rows:
                result = dict(row)
                if isinstance(result['config'], str):
                    result['config'] = json.loads(result['config'])
                results.append(result)
            return results

    async def update_strategy(self, strategy_id: str, clerk_user_id: str,
                              name: str = None, description: str = None,
                              config: Dict[str, Any] = None, 
                              status: str = None,
                              tags: List[str] = None) -> bool:
        """Update a strategy, verifying ownership"""
        async with self.pool.acquire() as conn:
            # Build dynamic update query
            updates = []
            params = [strategy_id, clerk_user_id]
            param_idx = 3
            
            if name is not None:
                updates.append(f"name = ${param_idx}")
                params.append(name)
                param_idx += 1
            
            if description is not None:
                updates.append(f"description = ${param_idx}")
                params.append(description)
                param_idx += 1
            
            if config is not None:
                updates.append(f"config = ${param_idx}")
                params.append(json.dumps(config))
                param_idx += 1
            
            if status is not None:
                updates.append(f"status = ${param_idx}")
                params.append(status)
                param_idx += 1
            
            if tags is not None:
                updates.append(f"tags = ${param_idx}")
                params.append(tags)
                param_idx += 1
            
            if not updates:
                return True  # Nothing to update
            
            updates.append("updated_at = NOW()")
            
            query = f"""
                UPDATE strategies
                SET {', '.join(updates)}
                WHERE strategy_id = $1 AND clerk_user_id = $2
                RETURNING strategy_id
            """
            
            result = await conn.fetchval(query, *params)
            return result is not None

    async def delete_strategy(self, strategy_id: str, clerk_user_id: str) -> bool:
        """Delete a strategy, verifying ownership"""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM strategies
                WHERE strategy_id = $1 AND clerk_user_id = $2
            """, strategy_id, clerk_user_id)
            # result is a string like "DELETE 1" or "DELETE 0"
            return "DELETE 1" in result

    async def verify_strategy_ownership(self, strategy_id: str, clerk_user_id: str) -> bool:
        """Verify that a user owns a strategy"""
        async with self.pool.acquire() as conn:
            owner = await conn.fetchval("""
                SELECT clerk_user_id FROM strategies WHERE strategy_id = $1
            """, strategy_id)
            return owner == clerk_user_id if owner else False

    # ============================================================================
    # Backtest Results Methods
    # ============================================================================

    async def create_backtest(self, strategy_id: str, clerk_user_id: str) -> str:
        """Create a new backtest record"""
        backtest_id = str(uuid.uuid4())
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO backtest_results (backtest_id, strategy_id, clerk_user_id, status, started_at)
                VALUES ($1, $2, $3, 'pending', NOW())
            """, backtest_id, strategy_id, clerk_user_id)
        return backtest_id

    async def update_backtest_status(self, backtest_id: str, clerk_user_id: str,
                                     status: str, error_message: str = None) -> bool:
        """Update backtest status"""
        async with self.pool.acquire() as conn:
            if status == 'completed' or status == 'failed':
                result = await conn.execute("""
                    UPDATE backtest_results
                    SET status = $3, error_message = $4, completed_at = NOW()
                    WHERE backtest_id = $1 AND clerk_user_id = $2
                """, backtest_id, clerk_user_id, status, error_message)
            else:
                result = await conn.execute("""
                    UPDATE backtest_results
                    SET status = $3, error_message = $4
                    WHERE backtest_id = $1 AND clerk_user_id = $2
                """, backtest_id, clerk_user_id, status, error_message)
            return "UPDATE 1" in result

    async def update_backtest_results(self, backtest_id: str, clerk_user_id: str,
                                      results: Dict[str, Any]) -> bool:
        """Update backtest with full results"""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE backtest_results
                SET status = 'completed',
                    completed_at = NOW(),
                    duration_seconds = $3,
                    starting_capital = $4,
                    ending_capital = $5,
                    portfolio = $6,
                    risk = $7,
                    equity_curve = $8,
                    drawdown_curve = $9
                WHERE backtest_id = $1 AND clerk_user_id = $2
            """, backtest_id, clerk_user_id,
                results.get('duration_seconds'),
                results.get('starting_capital'),
                results.get('ending_capital'),
                json.dumps(results.get('portfolio')) if results.get('portfolio') else None,
                json.dumps(results.get('risk')) if results.get('risk') else None,
                json.dumps(results.get('equity_curve')) if results.get('equity_curve') else None,
                json.dumps(results.get('drawdown_curve')) if results.get('drawdown_curve') else None
            )
            return "UPDATE 1" in result

    async def get_backtest(self, backtest_id: str, clerk_user_id: str) -> Optional[Dict[str, Any]]:
        """Get a backtest by ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT backtest_id, strategy_id, clerk_user_id, status, error_message,
                       started_at, completed_at, duration_seconds,
                       starting_capital, ending_capital,
                       portfolio, risk, equity_curve, drawdown_curve, created_at
                FROM backtest_results
                WHERE backtest_id = $1 AND clerk_user_id = $2
            """, backtest_id, clerk_user_id)
            
            if not row:
                return None
            
            result = dict(row)
            # Parse JSON fields
            for field in ['portfolio', 'risk', 'equity_curve', 'drawdown_curve']:
                if result.get(field) and isinstance(result[field], str):
                    result[field] = json.loads(result[field])
            return result

    async def get_strategy_backtests(self, strategy_id: str, clerk_user_id: str) -> List[Dict[str, Any]]:
        """Get all backtests for a strategy"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT backtest_id, strategy_id, clerk_user_id, status, error_message,
                       started_at, completed_at, duration_seconds,
                       starting_capital, ending_capital,
                       portfolio, risk, equity_curve, drawdown_curve, created_at
                FROM backtest_results
                WHERE strategy_id = $1 AND clerk_user_id = $2
                ORDER BY created_at DESC
            """, strategy_id, clerk_user_id)
            
            results = []
            for row in rows:
                result = dict(row)
                for field in ['portfolio', 'risk', 'equity_curve', 'drawdown_curve']:
                    if result.get(field) and isinstance(result[field], str):
                        result[field] = json.loads(result[field])
                results.append(result)
            return results

    async def get_latest_backtest(self, strategy_id: str, clerk_user_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent completed backtest for a strategy"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT backtest_id, strategy_id, clerk_user_id, status, error_message,
                       started_at, completed_at, duration_seconds,
                       starting_capital, ending_capital,
                       portfolio, risk, equity_curve, drawdown_curve, created_at
                FROM backtest_results
                WHERE strategy_id = $1 AND clerk_user_id = $2 AND status = 'completed'
                ORDER BY created_at DESC
                LIMIT 1
            """, strategy_id, clerk_user_id)
            
            if not row:
                return None
            
            result = dict(row)
            for field in ['portfolio', 'risk', 'equity_curve', 'drawdown_curve']:
                if result.get(field) and isinstance(result[field], str):
                    result[field] = json.loads(result[field])
            return result

    # ============================================================================
    # Trades Methods
    # ============================================================================

    async def add_trades(self, backtest_id: str, trades: List[Dict[str, Any]]) -> int:
        """Add multiple trades for a backtest"""
        if not trades:
            return 0
        
        async with self.pool.acquire() as conn:
            # Use executemany for batch insert
            await conn.executemany("""
                INSERT INTO trades (trade_id, backtest_id, symbol, side, quantity, price, timestamp, pnl, pnl_percent)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, [
                (str(uuid.uuid4()), backtest_id, t['symbol'], t['side'], 
                 t['quantity'], t['price'], t['timestamp'], 
                 t.get('pnl'), t.get('pnl_percent'))
                for t in trades
            ])
        return len(trades)

    async def get_trades(self, backtest_id: str, clerk_user_id: str, 
                        limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """Get trades for a backtest"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT t.trade_id, t.backtest_id, t.symbol, t.side, t.quantity, 
                       t.price, t.timestamp, t.pnl, t.pnl_percent
                FROM trades t
                JOIN backtest_results b ON t.backtest_id = b.backtest_id
                WHERE t.backtest_id = $1 AND b.clerk_user_id = $2
                ORDER BY t.timestamp DESC
                LIMIT $3 OFFSET $4
            """, backtest_id, clerk_user_id, limit, offset)
            
            return [dict(row) for row in rows]

    # ============================================================================
    # Research Papers & Vector Search Methods
    # ============================================================================

    async def insert_research_paper(
        self,
        title: str,
        source: str,
        abstract: Optional[str] = None,
        url: Optional[str] = None,
        authors: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        published_date: Optional[datetime] = None
    ) -> str:
        """Insert a research paper and return its ID"""
        paper_id = str(uuid.uuid4())
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO research_papers (id, title, abstract, source, url, authors, categories, published_date)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (url) DO UPDATE SET title = EXCLUDED.title
            """, paper_id, title, abstract, source, url, authors, categories, published_date)
        return paper_id

    async def insert_research_chunk(
        self,
        paper_id: str,
        chunk_text: str,
        chunk_index: int,
        embedding: List[float]
    ) -> str:
        """Insert a research chunk with embedding"""
        chunk_id = str(uuid.uuid4())
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO research_chunks (id, paper_id, chunk_text, chunk_index, embedding)
                VALUES ($1, $2, $3, $4, $5)
            """, chunk_id, paper_id, chunk_text, chunk_index, embedding)
        return chunk_id

    async def search_research_chunks(
        self,
        query_embedding: List[float],
        limit: int = 10,
        source_filter: Optional[str] = None,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search research chunks by vector similarity"""
        async with self.pool.acquire() as conn:
            if source_filter and category_filter:
                rows = await conn.fetch("""
                    SELECT rc.chunk_text, rc.chunk_index, rc.id as chunk_id,
                           rp.title, rp.abstract, rp.source, rp.url, rp.authors, rp.categories, rp.published_date,
                           (rc.embedding <=> $1) as distance
                    FROM research_chunks rc
                    JOIN research_papers rp ON rc.paper_id = rp.id
                    WHERE rp.source = $2 AND $3 = ANY(rp.categories)
                    ORDER BY rc.embedding <=> $1
                    LIMIT $4
                """, query_embedding, source_filter, category_filter, limit)
            elif source_filter:
                rows = await conn.fetch("""
                    SELECT rc.chunk_text, rc.chunk_index, rc.id as chunk_id,
                           rp.title, rp.abstract, rp.source, rp.url, rp.authors, rp.categories, rp.published_date,
                           (rc.embedding <=> $1) as distance
                    FROM research_chunks rc
                    JOIN research_papers rp ON rc.paper_id = rp.id
                    WHERE rp.source = $2
                    ORDER BY rc.embedding <=> $1
                    LIMIT $3
                """, query_embedding, source_filter, limit)
            elif category_filter:
                rows = await conn.fetch("""
                    SELECT rc.chunk_text, rc.chunk_index, rc.id as chunk_id,
                           rp.title, rp.abstract, rp.source, rp.url, rp.authors, rp.categories, rp.published_date,
                           (rc.embedding <=> $1) as distance
                    FROM research_chunks rc
                    JOIN research_papers rp ON rc.paper_id = rp.id
                    WHERE $2 = ANY(rp.categories)
                    ORDER BY rc.embedding <=> $1
                    LIMIT $3
                """, query_embedding, category_filter, limit)
            else:
                rows = await conn.fetch("""
                    SELECT rc.chunk_text, rc.chunk_index, rc.id as chunk_id,
                           rp.title, rp.abstract, rp.source, rp.url, rp.authors, rp.categories, rp.published_date,
                           (rc.embedding <=> $1) as distance
                    FROM research_chunks rc
                    JOIN research_papers rp ON rc.paper_id = rp.id
                    ORDER BY rc.embedding <=> $1
                    LIMIT $2
                """, query_embedding, limit)
            
            return [dict(row) for row in rows]

    async def get_research_paper_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Get a research paper by URL"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, title, abstract, source, url, authors, categories, published_date
                FROM research_papers WHERE url = $1
            """, url)
            return dict(row) if row else None

    async def get_all_research_urls(self) -> List[str]:
        """Get all URLs in research_papers table"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT url FROM research_papers WHERE url IS NOT NULL
            """)
            return [row['url'] for row in rows]

    async def insert_research_article(
        self,
        title: str,
        abstract: str,
        source: str,
        url: str,
        authors: List[str],
        categories: List[str],
        published_date: Optional[datetime],
        chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Insert article and its chunks in a single transaction.
        chunks: [{"text": str, "embedding": List[float]}]
        """
        paper_id = str(uuid.uuid4())
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("""
                    INSERT INTO research_papers (id, title, abstract, source, url, authors, categories, published_date)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (url) DO UPDATE SET title = EXCLUDED.title
                """, paper_id, title, abstract, source, url, authors, categories, published_date)
                
                for i, chunk in enumerate(chunks):
                    await conn.execute("""
                        INSERT INTO research_chunks (id, paper_id, chunk_text, chunk_index, embedding)
                        VALUES ($1, $2, $3, $4, $5)
                    """, str(uuid.uuid4()), paper_id, chunk['text'], i, chunk['embedding'])
        
        return paper_id

    # ============================================================================
    # Context Summary Methods
    # ============================================================================

    async def save_context_summary(
        self,
        chat_id: str,
        turn_start: int,
        turn_end: int,
        summary_text: str,
        key_entities: Dict[str, Any],
        token_count: int
    ) -> str:
        """Save a context summary for a chat turn range"""
        summary_id = str(uuid.uuid4())
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO context_summaries (id, chat_id, turn_start, turn_end, summary_text, key_entities, token_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, summary_id, chat_id, turn_start, turn_end, summary_text, json.dumps(key_entities), token_count)
        return summary_id

    async def get_context_summaries(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get all context summaries for a chat"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, chat_id, turn_start, turn_end, summary_text, key_entities, token_count, created_at
                FROM context_summaries
                WHERE chat_id = $1
                ORDER BY turn_start ASC
            """, chat_id)
            results = []
            for row in rows:
                result = dict(row)
                result['key_entities'] = json.loads(result['key_entities']) if result['key_entities'] else {}
                results.append(result)
            return results

    async def delete_old_context_summaries(self, chat_id: str, keep_from_turn: int) -> int:
        """Delete context summaries older than a certain turn"""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM context_summaries
                WHERE chat_id = $1 AND turn_end < $2
            """, chat_id, keep_from_turn)
            if "DELETE" in result:
                return int(result.split()[-1])
            return 0

    # ============================================================================
    # Entity Memory Methods
    # ============================================================================

    async def upsert_entity_memory(
        self,
        chat_id: str,
        entity_type: str,
        entity_value: str,
        context: Dict[str, Any] = None
    ) -> None:
        """Insert or update an entity in memory"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO entity_memory (chat_id, entity_type, entity_value, context)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (chat_id, entity_type, entity_value) DO UPDATE SET
                    last_mentioned_at = NOW(),
                    mention_count = entity_memory.mention_count + 1,
                    context = EXCLUDED.context
            """, chat_id, entity_type, entity_value, json.dumps(context) if context else json.dumps({}))

    async def get_entity_memory(self, chat_id: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get entity memories for a chat"""
        async with self.pool.acquire() as conn:
            if entity_type:
                rows = await conn.fetch("""
                    SELECT id, chat_id, entity_type, entity_value, context, 
                           first_mentioned_at, last_mentioned_at, mention_count
                    FROM entity_memory
                    WHERE chat_id = $1 AND entity_type = $2
                    ORDER BY mention_count DESC
                """, chat_id, entity_type)
            else:
                rows = await conn.fetch("""
                    SELECT id, chat_id, entity_type, entity_value, context,
                           first_mentioned_at, last_mentioned_at, mention_count
                    FROM entity_memory
                    WHERE chat_id = $1
                    ORDER BY last_mentioned_at DESC
                """, chat_id)
            
            results = []
            for row in rows:
                result = dict(row)
                result['context'] = json.loads(result['context']) if result['context'] else {}
                results.append(result)
            return results

    async def delete_entity_memory(self, chat_id: str, entity_type: Optional[str] = None) -> int:
        """Delete entity memories"""
        async with self.pool.acquire() as conn:
            if entity_type:
                result = await conn.execute("""
                    DELETE FROM entity_memory WHERE chat_id = $1 AND entity_type = $2
                """, chat_id, entity_type)
            else:
                result = await conn.execute("""
                    DELETE FROM entity_memory WHERE chat_id = $1
                """, chat_id)
            if "DELETE" in result:
                return int(result.split()[-1])
            return 0
