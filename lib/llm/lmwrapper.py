import dspy
import os
import requests
import json
import asyncio
import aiohttp
from typing import AsyncGenerator, List, Optional, Dict, Any
import sys
from datetime import datetime

from lib.retrieval.search import SearchTool
from lib.retrieval.marketdata import MarketDataTool
from lib.retrieval.snaptrade import get_portfolio_data
import re

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

AGENTIC_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, news, or real-time data. Use this when you need up-to-date information that may not be in your training data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute"
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["web", "news"],
                        "description": "Use 'news' for recent events, 'web' for general information"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_data",
            "description": "Fetch market data, stock prices, and financial fundamentals for stock tickers. Use this when the user asks about specific stocks, their prices, or financial metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock ticker symbols (e.g., ['AAPL', 'GOOGL'])"
                    }
                },
                "required": ["tickers"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": "Fetch the user's linked portfolio data including positions, account balances, and recent trades. Use this when the user asks about their investments or holdings.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


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
        portfolio_context: Optional[str] = None,
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
            
            final_user_message = user_message
            contexts = []
            if search_context:
                contexts.append(search_context)
            if market_data_context:
                contexts.append(market_data_context)
            if portfolio_context:
                contexts.append(portfolio_context)
            
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
                - "type" (optional, for combo charts): "line" or "bar"
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
                {"key": "revenue", "name": "Revenue"},
                {"key": "income", "name": "Operating Income"}
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
                {"key": "volume", "name": "Volume", "type": "bar"},
                {"key": "price", "name": "Avg Price", "type": "line"}
                ]
            }
            :::

            - Important: 
            - Always use valid JSON with proper escaping
            - Ensure data arrays contain consistent objects with matching keys
            - For financial data, use appropriate precision (typically 2 decimal places)
            - Choose chart types that best represent the data relationship
            - Keep titles concise and descriptive
            """,
            "portfolio": """
            - Tool Name: `portfolio`
            - Description: Portfolio data is automatically fetched and provided in your context when enabled.
            - When to use: This tool runs automatically - you don't need to call it. When portfolio data is provided in the context, use it to answer questions about the user's holdings, positions, account balances, and trading history.
            - Important: Always reference the provided portfolio data when answering questions about the user's investments. The data includes current positions, account information, recent trades, and total net asset value (NAV).
            - You can mention things like: "According to the linked portfolios..."
            - If there is no portfolio context, simply say that no portfolio is linked if the request directly relates to the user's specific portfolio data.
            """,
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

    def _get_agentic_system_prompt(self, tools: List[str]) -> str:
        """Generate system prompt for agentic mode with thinking and tool calling"""
        
        base = f"""You are ZECA, an advanced AI assistant developed by Zecant.
The current date is {datetime.now().strftime("%Y-%m-%d")}.

CORE PRINCIPLES:
- Accuracy First: Only make claims you can support with evidence from search results or reliable knowledge
- Cite Sources: When using search results, cite them as [Source: domain.com]
- Be Concise: Get to the point quickly, avoid unnecessary preamble
- Admit Uncertainty: If you don't know, say so - don't fabricate information

AGENTIC BEHAVIOR:
You have access to tools that can fetch real-time information. Use them strategically:

1. THINK FIRST: Before using tools, think through what information you need
   Output your reasoning in a thinking block: :::thinking\\nyour thoughts here\\n:::
   
2. USE TOOLS: When you need current/real-time information, call the appropriate tool:
   - web_search: For current events, news, recent data
   - get_market_data: For stock prices, financial data
   - get_portfolio: For user's investment holdings
   
3. ITERATE: After receiving tool results, analyze them and decide if you need more information

4. RESPOND: Once you have sufficient information, provide your final answer

THINKING FORMAT:
:::thinking
What do I need to know? Which tool would help? What should I search for?
:::

VISUALIZATION TOOLS (embed in response):
"""
        
        tool_specs = {
            "stock_chart": """
- stock_chart: :::stock_chart\\n{"ticker": "AAPL"}\\n:::
""",
            "graph": """
- graph: :::graph\\n{"type": "line", "data": [...], "xAxis": {"key": "..."}, "yAxis": {"key": "..."}}\\n:::
""",
        }
        
        for tool in tools:
            if tool in tool_specs:
                base += tool_specs[tool]
        
        base += """

MATH FORMATTING:
- Display math: $$...$$
- Inline math: \\(...\\) ONLY (never use $...$)
"""
        
        return base

    async def _execute_tool(self, tool_name: str, args: Dict[str, Any], clerk_user_id: str = None, db = None) -> str:
        """Execute a tool and return the result as a string"""
        
        if tool_name == "web_search":
            query = args.get("query", "")
            search_type = args.get("search_type", "web")
            try:
                results, _ = await self.search_tool.multi_search(query)
                return results
            except Exception as e:
                return f"Search failed: {str(e)}"
        
        elif tool_name == "get_market_data":
            tickers = args.get("tickers", [])
            if not tickers:
                return "No tickers provided"
            try:
                results = []
                for ticker in tickers[:5]:
                    data = await self.market_data_tool.fetch_market_data(ticker)
                    results.append(json.dumps(data))
                return "\n\n".join(results)
            except Exception as e:
                return f"Market data fetch failed: {str(e)}"
        
        elif tool_name == "get_portfolio":
            if not clerk_user_id or not db:
                return "Portfolio data not available - user not authenticated or database not connected"
            try:
                portfolio_data = await get_portfolio_data(clerk_user_id, db, force_refresh=False)
                if portfolio_data.get("connected") and not portfolio_data.get("error"):
                    total_nav = portfolio_data.get("totalNav", 0)
                    positions = portfolio_data.get("positions", [])[:20]
                    return json.dumps({
                        "total_nav": total_nav,
                        "positions": positions,
                        "accounts": portfolio_data.get("accounts", [])
                    })
                else:
                    return "No portfolio linked"
            except Exception as e:
                return f"Portfolio fetch failed: {str(e)}"
        
        return f"Unknown tool: {tool_name}"

    async def stream_agentic(
        self,
        user_message: str,
        chat_history: List[Dict[str, Any]],
        requested_tools: Optional[List[str]] = None,
        max_iterations: int = 5,
        max_tokens: int = 8000,
        clerk_user_id: Optional[str] = None,
        db = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Agentic loop that allows the LLM to think, use tools, and iterate.
        
        Yields:
            Dicts with types:
                - {"type": "thinking", "content": str} - thinking content
                - {"type": "tool_call", "tool": str, "args": dict} - tool being called
                - {"type": "tool_result", "tool": str, "result": str} - tool execution result
                - {"type": "token", "content": str} - text tokens
                - {"type": "done"} - generation complete
        """
        
        available_tools = ["web_search", "get_market_data", "get_portfolio"]
        enabled_tools = [t for t in available_tools if t in (requested_tools or []) or t == "web_search"]
        
        messages = []
        messages.append({
            "role": "system",
            "content": self._get_agentic_system_prompt(requested_tools or [])
        })
        
        for msg in chat_history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        messages.append({"role": "user", "content": user_message})
        
        all_tool_calls = []
        
        for iteration in range(max_iterations):
            response_content = ""
            tool_calls_made = []
            
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
            
            if enabled_tools:
                payload["tools"] = [t for t in AGENTIC_TOOLS_SCHEMA if t["function"]["name"] in enabled_tools]
                payload["tool_choice"] = "auto"
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300, connect=30, sock_read=30),
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            yield {"type": "error", "error": f"API error {response.status}: {error_text}"}
                            return
                        
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
                                        
                                        if "tool_calls" in delta:
                                            for tc in delta["tool_calls"]:
                                                if tc.get("function", {}).get("name"):
                                                    tool_name = tc["function"]["name"]
                                                    tool_args = json.loads(tc["function"].get("arguments", "{}"))
                                                    tool_calls_made.append({
                                                        "id": tc.get("id", ""),
                                                        "name": tool_name,
                                                        "args": tool_args
                                                    })
                                        
                                        if "content" in delta and delta["content"]:
                                            content = delta["content"]
                                            response_content += content
                                            yield {"type": "token", "content": content}
                                
                                except json.JSONDecodeError:
                                    continue
            
            except Exception as e:
                yield {"type": "error", "error": str(e)}
                return
            
            if tool_calls_made:
                messages.append({"role": "assistant", "content": response_content, "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                    for tc in tool_calls_made
                ]})
                
                for tc in tool_calls_made:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    
                    yield {"type": "tool_call", "tool": tool_name, "args": tool_args}
                    
                    result = await self._execute_tool(tool_name, tool_args, clerk_user_id, db)
                    
                    all_tool_calls.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result[:1000] if len(result) > 1000 else result
                    })
                    
                    yield {"type": "tool_result", "tool": tool_name, "result": result[:500]}
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result
                    })
                
                continue
            
            break
        
        yield {"type": "tool_log", "calls": all_tool_calls}
        yield {"type": "done"}

    async def stream_with_tools_execution(
        self,
        user_message: str,
        chat_history: List[Dict[str, Any]],
        requested_tools: Optional[List[str]] = None,
        max_tokens: int = 8000,
        max_tool_iterations: int = 5,
        auto_search: bool = True,
        clerk_user_id: Optional[str] = None,
        db = None,
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
        
        # Check if portfolio data should be fetched
        portfolio_context = None
        if "portfolio" in (requested_tools or []) and clerk_user_id and db:
            try:
                
                portfolio_data = await get_portfolio_data(clerk_user_id, db, force_refresh=False)
                
                if portfolio_data.get("connected") and not portfolio_data.get("error"):
                    # Format portfolio context for LLM
                    total_nav = portfolio_data.get("totalNav", 0)
                    accounts = portfolio_data.get("accounts", [])
                    positions = portfolio_data.get("positions", [])
                    trades = portfolio_data.get("trades", [])[:10]  # Last 10 trades
                    
                    portfolio_context = f"""
        USER'S PORTFOLIO DATA:

        Total Portfolio Value (NAV): ${total_nav:,.2f}

        Accounts ({len(accounts)} total):
        """
                    for acc in accounts:
                        portfolio_context += f"- {acc.get('name')} ({acc.get('brokerage')}): {acc.get('type')}\n"
                    
                    portfolio_context += f"\nPositions ({len(positions)} total):\n"
                    for pos in positions[:20]:  # Top 20 positions
                        symbol = pos.get("symbol", {})
                        ticker = symbol.get("symbol") if isinstance(symbol, dict) else symbol
                        quantity = pos.get("fractional_units", 0)
                        price = pos.get("price", 0)
                        value = price * abs(quantity)
                        portfolio_context += f"- {ticker}: {quantity:.4f} shares @ ${price:.2f} = ${value:,.2f}\n"
                    
                    if trades:
                        portfolio_context += f"\nRecent Trades ({len(trades)} shown):\n"
                        for trade in trades:
                            portfolio_context += f"- {trade.get('date')}: {trade.get('side')} {trade.get('qty')} {trade.get('ticker')} @ ${trade.get('price', 0):.2f}\n"
                    
                    portfolio_context += "\nIMPORTANT: Use this portfolio data to answer questions about the user's holdings, positions, and trades."
                    
                    # Yield portfolio notification to frontend
                    yield {
                        "type": "portfolio_fetched",
                        "total_nav": total_nav,
                        "num_positions": len(positions),
                        "num_accounts": len(accounts)
                    }
            except Exception as e:
                print(f"Portfolio fetch failed: {e}")
                yield {
                    "type": "tool_error",
                    "tool": "portfolio",
                    "error": str(e)
                }

        # Pass pre-fetched contexts to streaming function
        async for token in self.stream_async_with_context(
            user_message=user_message,
            chat_history=chat_history,
            requested_tools=requested_tools,
            max_tokens=max_tokens,
            search_context=search_context,
            market_data_context=market_data_context,
            portfolio_context=portfolio_context,  # ADD THIS
        ):
            yield {
                "type": "token",
                "content": token
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