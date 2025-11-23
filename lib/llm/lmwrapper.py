import dspy
import os
import requests
import json
import asyncio
import aiohttp
from typing import AsyncGenerator, List, Optional, Dict, Any
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib.retrieval.search import SearchTool

routers = {
    "OPENROUTER": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": os.environ["OPENROUTER_TOKEN_DAYTADING"],
        "models": {
            "kimik2": "moonshotai/kimi-k2:free",
            "horizonbeta": "openrouter/horizon-beta",
            "qwen3coder480b": "qwen/qwen3-coder:free",
            "glm4.5air": "z-ai/glm-4.5-air:free",
            "hunyuan13b": "tencent/hunyuan-a13b-instruct:free",
            "r1t2chimera": "tngtech/deepseek-r1t2-chimera:free",
            "r10528": "deepseek/deepseek-r1-0528:free",
            "deepseekv3.1": "deepseek/deepseek-chat-v3.1:free",
        },
    },
    "GROQ": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key": os.environ["GROQ_TOKEN_DAYTADING"],
        "models": {
            "llama8b": "llama-3.1-8b-instant",
            "gptoss120": "openai/gpt-oss-120b",
            "gptoss20": "openai/gpt-oss-20b",
        },
    },
}


class LM_WRAPPER:
    def __init__(
        self,
        model: str = routers["OPENROUTER"]["models"]["deepseekv3.1"],
        temperature: float = 0.0,
        provider: str = "OPENROUTER",
    ):

        self.lm = dspy.LM(
            model,
            api_key=routers[provider]["api_key"],
            api_base=routers[provider]["base_url"],
            temperature=temperature,
            max_tokens=4000,
            stop=None,
            cache=True,
            cache_in_memory=False,
            extra_headers=(
                None
                if provider != "OPENROUTER"
                else {
                    "HTTP-Referer": "https://api.zecant.com",
                    "X-Title": "Zecant",
                }
            ),
        )
        dspy.configure()

        # Store provider configuration for streaming
        self.provider = provider
        self.model_name = model
        self.temperature = temperature
        self.api_key = routers[provider]["api_key"]
        self.base_url = routers[provider]["base_url"]
        self.extra_headers = (
            None
            if provider != "OPENROUTER"
            else {"HTTP-Referer": "https://zecant.vercel.app", "X-Title": "Zecant"}
        )

    async def stream_async(
        self, messages: list, max_tokens: int = 4000
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronous token streaming from the provider.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate

        Yields:
            str: Individual tokens as they are received
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.extra_headers:
            headers.update(self.extra_headers)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(
                        total=300,
                        connect=30,
                        sock_read=30
                    ),
                ) as response:
                    if response.status != 200:
                        try:
                            error_text = await response.text()
                        except Exception as err:
                            error_text = f"(failed to read error body: {err})"
                        raise Exception(
                            f"API request failed with status {response.status}: {error_text or '(no response body)'}"
                        )

                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue

        except aiohttp.ClientError as e:
            raise Exception(f"Async streaming request failed: {e}")

    async def generate_title(self, first_message: str, max_tokens: int = 400) -> str:
        """Generate a short title for a chat based on the first message."""
        messages = [
            {
                "role": "user",
                "content": f"Create a short title (maximum 8 words) that describes this message:\n\n{first_message}\n\nRespond with ONLY the title, nothing else."
            }
        ]
        
        title = ""
        async for token in self.stream_async(messages, max_tokens):
            title += token

        title = title.strip().rstrip()
        
        for prefix in ['"', "'", 'Title:', 'title:', '**', '*', 'Title: ', 'title: ']:
            title = title.strip(prefix).strip()
        
        title = title.replace('\n', ' ').replace('\r', ' ')
        
        if not title or len(title) < 3:
            title = first_message[:50].strip()
            if len(first_message) > 50:
                title += "..."
        
        return title


class LM_WRAPPER_WITH_TOOLS(LM_WRAPPER):
    """Extended LM wrapper with tool support"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_tool = SearchTool(self)

    def _should_search(self, user_message: str) -> bool:
        """
        Determine if the user message requires web search.
        
        Uses simple heuristics to detect search intent.
        """
        search_indicators = [
            'search', 'find', 'look up', 'latest', 'recent', 'news',
            'current', 'today', 'price', 'stock', "what", "where", "who", "why", "how", "?"
        ]
        
        message_lower = user_message.lower()
        return any(indicator in message_lower for indicator in search_indicators)
    
    def _extract_search_query(self, user_message: str) -> str:
        """
        Extract a clean search query from the user message.
        
        Simple extraction - can be enhanced with LLM in future.
        """
        # Remove common question words and clean up
        query = user_message.strip()
        
        # Remove question marks and extra whitespace
        query = query.replace('?', '').strip()
        
        # Limit length for search API
        if len(query) > 100:
            query = query[:100]
        
        return query

    async def stream_async_with_context(
        self,
        user_message: str,
        chat_history: List[Dict[str, Any]],
        requested_tools: Optional[List[str]] = None,
        max_tokens: int = 4000,
        auto_search: bool = True,
        search_count: int = 5
    ) -> AsyncGenerator[str, None]:
        """
        Stream response with full chat history context and automatic search.
        
        Args:
            user_message: Current user message
            chat_history: Full chat history from database
            requested_tools: List of tool names user wants available
            max_tokens: Max tokens to generate
            auto_search: If True, automatically search when query seems search-related
            search_count: Number of search results to fetch
            
        Yields:
            Token chunks from the LLM
        """
        formatted_history = []
        
        # Step 1: Check if we should perform a search
        search_results = None
        if "search" in requested_tools:
            try:
                search_query = self._extract_search_query(user_message)
                search_results = await self.search_tool.multi_search(search_query)
                print(search_results)
            except Exception as e:
                print(f"Search failed: {e}")
                # Continue without search results
        
        # Step 2: Build system instructions with tool specs
        if requested_tools:
            tool_instructions = self._get_tool_instructions(requested_tools)
            formatted_history.append({
                "role": "system",
                "content": tool_instructions
            })
        
        # Step 3: Add search results to context if available
        if search_results:
            formatted_history.append({
                "role": "system",
                "content": search_results
            })
        
        # Step 4: Add user message
        formatted_history.append({
            "role": "user",
            "content": user_message
        })

        # Step 5: Stream from LLM with enriched context
        async for token in self.stream_async(formatted_history, max_tokens):
            yield token

    def _get_tool_instructions(self, tools: List[str]) -> str:
        """Generate system instructions for available tools"""
        
        tool_specs = {
            "stock_chart": """
            - Tool Name: `stock_chart`
            - Description: Displays an interactive stock price chart for a given ticker symbol over a specified number of days.
            - When to use: Use this when the user asks for a stock chart, price history, or to visualize a stock's performance over time. Even if the user does not explicitly request this, if it would be useful, include a graph.
            - If user defines a specific duration, make sure to obey that duration.
            - Format: :::stock_chart\n{"ticker": "TICKER_SYMBOL", "days": NUMBER_OF_DAYS}\n:::
            - Example: To show a 90-day chart for Apple Inc., you would output:
              :::stock_chart
              {"ticker": "AAPL", "days": 90}
              :::
            - Important: Always use a valid JSON object for the parameters. The "ticker" is a string and "days" is a number. Default to 30 days if not specified.
            """,
        }
        
        base_instructions = """
        You are a helpful AI assistant with access to visualization tools.

        IMPORTANT FORMATTING RULES:
        - For mathematical expressions:
        * Display math: Use $$...$$ or \\[...\\]
        * Inline math: Use \\(...\\) ONLY
        * DO NOT use single dollar signs $...$ for math
        * Example inline: The angle \\(\\theta\\) is \\(90^\\circ\\)
        * Example display: $$x^2 + y^2 = r^2$$

        """
        
        if tools:
            base_instructions += "Available visualization tools:\n\n"
            for tool in tools:
                if tool in tool_specs:
                    base_instructions += tool_specs[tool] + "\n"
            
            base_instructions += "\nWhen a tool is appropriate, generate its required markdown block directly in your response. The user's interface will render it."
        
        return base_instructions

    async def stream_with_tools_execution(
        self,
        user_message: str,
        chat_history: List[Dict[str, Any]],
        requested_tools: Optional[List[str]] = None,
        max_tokens: int = 4000,
        max_tool_iterations: int = 5,
        auto_search: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response and automatically execute tools if requested by LLM.
        
        This streams tokens and can signal when search was performed.
        
        Yields:
            Dicts with:
                - {"type": "token", "content": str} for text tokens
                - {"type": "search_performed", "query": str, "results": dict} when search happens
        """
        # Check if search should be performed
        search_results = None
        if auto_search and self._should_search(user_message):
            try:
                search_query = self._extract_search_query(user_message)
                search_results = await self.search_tool.search(search_query, max_results=8)
                
                # Yield search notification
                yield {
                    "type": "search_performed",
                    "query": search_query,
                    "results": search_results
                }
            except Exception as e:
                print(f"Search failed: {e}")
        
        # Stream response with search results in context
        async for token in self.stream_async_with_context(
            user_message,
            chat_history,
            requested_tools,
            max_tokens,
            auto_search=False  # Already searched above
        ):
            yield {
                "type": "token",
                "content": token
            }