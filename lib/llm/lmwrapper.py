import dspy
import os
import requests
import json
import asyncio
import aiohttp
from typing import AsyncGenerator, List, Optional, Dict, Any
import re
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib.retrieval.search import SearchTool
from lib.retrieval.marketdata import MarketDataTool

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
            "llama4scout": "meta-llama/llama-4-scout-17b-16e-instruct",
            "gptoss120": "openai/gpt-oss-120b",
            "gptoss20": "openai/gpt-oss-20b",
        },
    },
}


class LM_WRAPPER:
    def __init__(
        self,
        model: str = routers["GROQ"]["models"]["llama4scout"],
        temperature: float = 0.35,
        provider: str = "GROQ",
    ):

        self.lm = dspy.LM(
            model,
            api_key=routers[provider]["api_key"],
            api_base=routers[provider]["base_url"],
            temperature=temperature,
            max_tokens=8000,
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
        self, messages: list, max_tokens: int = 8000
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
        self.market_data_tool = MarketDataTool()
        # Initialize light model for quick checks (same as title generation)
        self.light_model = LM_WRAPPER(
            model=routers["GROQ"]["models"]["llama8b"],
            temperature=0.1,
            provider="GROQ"
        )
    
    async def should_perform_search(self, user_message: str) -> bool:
        """
        Use light model to determine if query requires web search.
        Returns True if search is needed, False otherwise.
        """
        prompt = [
            {
                "role": "system",
                "content": "You are a classifier. Determine if a user query requires searching the internet for current, recent, or real-time information. Answer with ONLY 'YES' or 'NO'."
            },
            {
                "role": "user",
                "content": f"Does this query require searching the internet for current/recent information? Answer YES or NO only:\n\n{user_message}"
            }
        ]
        
        response = ""
        async for token in self.light_model.stream_async(prompt, max_tokens=10):
            response += token
        
        response = response.strip().upper()
        return "YES" in response or response.startswith("Y")

    async def should_fetch_market_data(self, user_message: str) -> bool:
        """
        Use light model to determine if query mentions stocks and needs market data.
        Returns True if market data is needed, False otherwise.
        """
        prompt = [
            {
                "role": "system",
                "content": "You are a classifier. Determine if a user query would be better answered with market data about stock prices, financial data, company fundamentals, or market information. Answer with ONLY 'YES' or 'NO'."
            },
            {
                "role": "user",
                "content": f"Would this query be better answered with market data about stock prices, financial data, or company fundamentals? Answer YES or NO only:\n\n{user_message}"
            }
        ]
        
        response = ""
        async for token in self.light_model.stream_async(prompt, max_tokens=10):
            response += token
        
        response = response.strip().upper()
        return "YES" in response or response.startswith("Y")

    async def stream_async_with_context(
        self,
        user_message: str,
        chat_history: List[Dict[str, Any]],
        requested_tools: Optional[List[str]] = None,
        max_tokens: int = 4000,
        temperature: Optional[float] = None,
        # FIX #1: Add parameters to accept pre-fetched tool results
        search_context: Optional[str] = None,
        market_data_context: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response with improved context handling.
        
        Args:
            user_message: The user's message
            chat_history: Previous conversation history
            requested_tools: List of tool names that are enabled
            max_tokens: Maximum tokens to generate
            temperature: Override temperature (optional)
            search_context: Pre-fetched search results context (optional)
            market_data_context: Pre-fetched market data context (optional)
        """

        if temperature is None:
            # Lower temperature for factual/search queries
            message_lower = user_message.lower()
            if any(word in message_lower for word in ["what", "when", "who", "where", "why", "define", "explain", "how"]):
                temperature = 0.1
            else:
                temperature = self.temperature
        
        # FIX #3: Store original and use try/finally to ensure reset
        original_temp = self.temperature
        
        try:
            self.temperature = temperature
            
            formatted_history = []
            
            # Step 1: System instructions FIRST
            if requested_tools:
                tool_instructions = self._get_tool_instructions(requested_tools)
                formatted_history.append({
                    "role": "system",
                    "content": tool_instructions
                })
            
            # Step 2: Add chat history (critical for context!)
            for msg in chat_history:
                formatted_history.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            # FIX #1: Remove duplicate tool execution - use passed contexts
            # No longer fetching tools here, they're passed as parameters
            
            # Step 3: Build user message with provided contexts
            final_user_message = user_message
            contexts = []
            if search_context:
                contexts.append(search_context)
            if market_data_context:
                contexts.append(market_data_context)
            
            if contexts:
                final_user_message = "\n\n".join(contexts) + f"\n\nUser Question: {user_message}"

            formatted_history.append({
                "role": "user",
                "content": final_user_message
            })

            # Step 4: Stream response
            async for token in self.stream_async(formatted_history, max_tokens):
                yield token
        
        finally:
            # FIX #3: Always reset temperature, even if error occurs
            self.temperature = original_temp

    def _get_tool_instructions(self, tools: List[str]) -> str:
        """Generate system instructions for available tools"""
        
        tool_specs = {
            "stock_chart": """
            - Tool Name: `stock_chart`
            - Description: Displays an interactive stock price chart for a given ticker symbol.
            - When to use: Use this when the user asks for a stock chart, price history, or to visualize a stock's performance over time. Even if the user does not explicitly request this, if it would be useful, include a graph.
            - Time period is irrelevant, the user can see all data, all you need to provide is the correct ticker.
            - Format: :::stock_chart\n{"ticker": "TICKER_SYMBOL"}\n:::
            - Example: To show a chart for Apple Inc., you would output:
              :::stock_chart
              {"ticker": "AAPL"}
              :::
            - Important: Always use a valid JSON object for the parameters. The "ticker" is a string.
            """,
            "market_data": """
            - Tool Name: `market_data`
            - Description: Market data and fundamentals are automatically fetched and provided in your context when users ask about stocks.
            - When to use: This tool runs automatically - you don't need to call it. When market data is provided in the context, use it to answer questions about stock prices, financial metrics, company fundamentals, and analyst recommendations.
            - Important: Always reference the provided market data when answering questions about stocks. The data includes current prices, recent trading history, key financial metrics, and analyst recommendations.
            - Do not mention the tool name in your response or mention that you have been "provided" or "given" data. Don't say things like "with the provided data..."
            """,
            "graph": """
            - Tool Name: `graph`
            - Description: Creates interactive data visualizations including line charts, bar charts, area charts, scatter plots, pie charts, and combination charts. Perfect for displaying financial data, trends, comparisons, and distributions.
            - When to use: Use when the user needs to visualize data trends, compare metrics, show distributions, or analyze relationships between variables. Include graphs proactively when they would enhance understanding of numerical data or trends being discussed.
            - Format: :::graph
            {JSON_CONFIG}
            :::

            - Parameters (JSON object):
            - "type" (required): Chart type - one of: "line", "bar", "area", "scatter", "pie", "combo"
            - "title" (optional): Chart title string
            - "data" (required): Array of data objects
            - "xAxis" (required): Configuration object with:
                - "key": The data field for x-axis
                - "label" (optional): Axis label
            - "yAxis" (required for non-pie charts): Configuration object with:
                - "key": The data field for y-axis (or array of keys for multiple series)
                - "label" (optional): Axis label
            - "series" (optional): Array of series objects for multi-line/bar charts, each with:
                - "key": Data field to plot
                - "name": Display name for legend
                - "color" (optional): Hex color code
                - "type" (optional, for combo charts): "line" or "bar"
            - "colors" (optional): Array of hex color codes for styling
            - "height" (optional): Chart height in pixels (default: 300)

            - Examples:

            1. Simple line chart:
            :::graph
            {
                "type": "line",
                "title": "Revenue Growth (2020-2024)",
                "data": [
                {"year": "2020", "revenue": 274.5},
                {"year": "2021", "revenue": 365.8},
                {"year": "2022", "revenue": 394.3},
                {"year": "2023", "revenue": 383.3},
                {"year": "2024", "revenue": 391.0}
                ],
                "xAxis": {"key": "year", "label": "Year"},
                "yAxis": {"key": "revenue", "label": "Revenue (Billions USD)"}
            }
            :::

            2. Multi-series comparison:
            :::graph
            {
                "type": "line",
                "title": "Revenue vs Operating Income",
                "data": [
                {"quarter": "Q1", "revenue": 97.3, "income": 28.6},
                {"quarter": "Q2", "revenue": 94.8, "income": 26.7},
                {"quarter": "Q3", "revenue": 89.5, "income": 22.9},
                {"quarter": "Q4", "revenue": 109.4, "income": 33.9}
                ],
                "xAxis": {"key": "quarter"},
                "series": [
                {"key": "revenue", "name": "Revenue", "color": "#10b981"},
                {"key": "income", "name": "Operating Income", "color": "#3b82f6"}
                ],
                "yAxis": {"label": "USD (Billions)"}
            }
            :::

            3. Bar chart:
            :::graph
            {
                "type": "bar",
                "title": "Market Share by Company",
                "data": [
                {"company": "Apple", "share": 28.5},
                {"company": "Samsung", "share": 24.3},
                {"company": "Xiaomi", "share": 12.8},
                {"company": "Others", "share": 34.4}
                ],
                "xAxis": {"key": "company"},
                "yAxis": {"key": "share", "label": "Market Share (%)"}
            }
            :::

            4. Pie chart:
            :::graph
            {
                "type": "pie",
                "title": "Portfolio Allocation",
                "data": [
                {"category": "Stocks", "value": 60},
                {"category": "Bonds", "value": 25},
                {"category": "Cash", "value": 10},
                {"category": "Real Estate", "value": 5}
                ],
                "xAxis": {"key": "category"},
                "yAxis": {"key": "value"}
            }
            :::

            5. Combo chart (bars + line):
            :::graph
            {
                "type": "combo",
                "title": "Sales Volume and Average Price",
                "data": [
                {"month": "Jan", "volume": 1200, "price": 45.3},
                {"month": "Feb", "volume": 1350, "price": 46.8},
                {"month": "Mar", "volume": 1100, "price": 44.2}
                ],
                "xAxis": {"key": "month"},
                "series": [
                {"key": "volume", "name": "Volume", "type": "bar", "color": "#3b82f6"},
                {"key": "price", "name": "Avg Price", "type": "line", "color": "#10b981"}
                ]
            }
            :::

            - Important: 
            - Always use valid JSON with proper escaping
            - Ensure data arrays contain consistent objects with matching keys
            - For financial data, use appropriate precision (typically 2 decimal places)
            - Choose chart types that best represent the data relationship
            - Use colors that match the existing design system (greens for positive, reds for negative)
            - Keep titles concise and descriptive
            """
        }
        
        base_instructions = f"""You are ZECA, an advanced AI assistant developed by Zecant.
        The current date is {datetime.now().strftime("%Y-%m-%d")}.
        CORE PRINCIPLES:
        - Accuracy First: Only make claims you can support with evidence from search results or reliable knowledge
        - Cite Sources: When using search results, cite them as [Source: domain.com]
        - Be Concise: Get to the point quickly, avoid unnecessary preamble
        - Think Step-by-Step: For complex queries, break down your reasoning
        - Admit Uncertainty: If you don't know, say so - don't fabricate information
        - Do not halucinate information

        SEARCH RESULTS USAGE:
        - When search results are provided, prioritize them over your training data
        - Synthesize information from multiple sources when available
        - If sources conflict, note the discrepancy
        - Always attribute specific facts to their sources

        MATH FORMATTING:
        - Display math: $$...$$
        - Inline math: \\(...\\) ONLY (never use $...$)
        - Example: The equation \\(E = mc^2\\) shows that $$E = mc^2$$

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
        max_tokens: int = 8000,
        max_tool_iterations: int = 5,
        auto_search: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response and automatically execute tools if requested by LLM.
        
        This streams tokens and can signal when search was performed.
        
        Yields:
            Dicts with:
                - {"type": "token", "content": str} for text tokens
                - {"type": "search_performed", "query": str, "results": str} when search happens
                - {"type": "market_data_fetched", "tickers": List[str]} when market data is fetched
        """
        
        # FIX #1: Execute tools ONCE and store contexts
        search_context = None
        search_query = None
        
        # Check if search should be performed
        if auto_search and "search" in (requested_tools or []):
            try:
                if await self.should_perform_search(user_message):
                    search_results, query = await self.search_tool.multi_search(user_message)
                    
                    # Format search context for LLM
                    search_context = f"""
SEARCH RESULTS for query: "{query}"

{search_results}

IMPORTANT: Base your response on these search results. Cite sources with [Source: URL].
"""
                    
                    # Yield search notification to frontend
                    yield {
                        "type": "search_performed",
                        "query": query,
                        "results": search_results[:500]  # Send summary to frontend
                    }
            except Exception as e:
                print(f"Search failed: {e}")
                # Yield error event
                yield {
                    "type": "tool_error",
                    "tool": "search",
                    "error": str(e)
                }
        
        # Check if market data should be fetched
        market_data_context = None
        market_data_tickers = []
        if "market_data" in (requested_tools):
            try:
                if await self.should_fetch_market_data(user_message):
                    market_data_context, market_data_tickers = await self.market_data_tool.fetch_data(user_message)
                    
                    if market_data_tickers:
                        # Yield market data notification to frontend
                        yield {
                            "type": "market_data_fetched",
                            "tickers": market_data_tickers
                        }
            except Exception as e:
                print(f"Market data fetch failed: {e}")
                # Yield error event
                yield {
                    "type": "tool_error",
                    "tool": "market_data",
                    "error": str(e)
                }
        
        # FIX #1: Pass pre-fetched contexts to streaming function
        # This ensures tools are executed only ONCE and contexts are properly injected
        async for token in self.stream_async_with_context(
            user_message=user_message,
            chat_history=chat_history,
            requested_tools=requested_tools,
            max_tokens=max_tokens,
            search_context=search_context,
            market_data_context=market_data_context,
        ):
            yield {
                "type": "token",
                "content": token
            }