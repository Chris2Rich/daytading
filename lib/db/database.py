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
