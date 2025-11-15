import dspy
import os
import requests
import json
import asyncio
import aiohttp
from typing import AsyncGenerator, List, Optional, Dict, Any

routers = {
    "OPENROUTER": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": os.environ["OPENROUTER_TOKEN_DAYTADING"],
        "models": {
            "kimik2": "moonshotai/kimi-k2:free",  # unstable with dspy
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
                        total=300,         # 5 minute absolute maximum
                        connect=30,        # 10s to connect
                        sock_read=30       # 30s between chunks
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
                            data = line[6:]  # Remove 'data: ' prefix
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

        # Clean up the title
        title = title.strip().rstrip()
        
        # Remove common prefixes/wrappers
        for prefix in ['"', "'", 'Title:', 'title:', '**', '*', 'Title: ', 'title: ']:
            title = title.strip(prefix).strip()
        
        # Remove newlines
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
        self.tool_registry = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tool implementations (can be extended later)"""
        # Tools will be implemented later
        # For now, just set up the registry structure
        pass

    def register_tool(self, name: str, func: callable):
        """Register a custom tool"""
        self.tool_registry[name] = func

    async def execute_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool and return results"""
        if tool_name not in self.tool_registry:
            return {
                "error": f"Tool '{tool_name}' not found",
                "tool_name": tool_name
            }

        try:
            result = await self.tool_registry[tool_name](parameters)
            return {
                "success": True,
                "tool_name": tool_name,
                "result": result
            }
        except Exception as e:
            return {
                "error": str(e),
                "tool_name": tool_name
            }

    def _format_messages_with_tools(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Format messages from database format to LLM format.
        Includes tool results inline.
        """
        formatted = []
        
        for msg in messages:
            # Basic message
            formatted.append({
                "role": msg["role"],
                "content": msg["content"]
            })

            # If there are tool results, add them as system messages
            if msg.get("tool_results"):
                tool_results_text = self._format_tool_results(msg["tool_results"])
                formatted.append({
                    "role": "system",
                    "content": f"Tool results: {tool_results_text}"
                })

        return formatted

    def _format_tool_results(self, tool_results: Dict[str, Any]) -> str:
        """Format tool results for inclusion in context"""
        try:
            return json.dumps(tool_results, indent=2)
        except:
            return str(tool_results)

    async def stream_async_with_context(
        self,
        user_message: str,
        chat_history: List[Dict[str, Any]],
        requested_tools: Optional[List[str]] = None,
        max_tokens: int = 4000
    ) -> AsyncGenerator[str, None]:
        """
        Stream response with full chat history context.
        
        Args:
            user_message: Current user message
            chat_history: Full chat history from database
            requested_tools: List of tool names user wants available
            max_tokens: Max tokens to generate
            
        Yields:
            Token chunks from the LLM
        """
        # Start with system message
        formatted_history = []
        
        # Add tool instructions at the beginning if tools are enabled
        if requested_tools:
            tool_instructions = self._get_tool_instructions(requested_tools)
            formatted_history.append({
                "role": "system",
                "content": tool_instructions
            })
        
        # Format and add chat history
        formatted_history.extend(self._format_messages_with_tools(chat_history))
        
        # Add current user message
        formatted_history.append({
            "role": "user",
            "content": user_message
        })

        # Stream from LLM
        async for token in self.stream_async(formatted_history, max_tokens):
            yield token

    def _get_tool_instructions(self, tools: List[str]) -> str:
        """Generate system instructions for available tools"""
        
        tool_specs = {
            "stock_chart": """
            To display an interactive stock price chart, use this format:
                ":::stock_chart {"ticker": "AAPL", "days": 30}:::"
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
            
            base_instructions += "\nUse these tools when they would help visualize or present data clearly."
        
        return base_instructions

    async def stream_with_tools_execution(
        self,
        user_message: str,
        chat_history: List[Dict[str, Any]],
        requested_tools: Optional[List[str]] = None,
        max_tokens: int = 4000,
        max_tool_iterations: int = 5
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response and automatically execute tools if requested by LLM.
        
        This is a more advanced version that can handle tool calls.
        For now, it just streams normally - tool execution can be added later.
        
        Yields:
            Dicts with:
                - {"type": "token", "content": str} for text tokens
                - {"type": "tool_call", "tool": str, "parameters": dict} for tool calls
                - {"type": "tool_result", "result": dict} for tool results
        """
        response_text = ""
        
        async for token in self.stream_async_with_context(
            user_message,
            chat_history,
            requested_tools,
            max_tokens
        ):
            response_text += token
            yield {
                "type": "token",
                "content": token
            }

        # Future: Parse response_text for tool calls, execute them,
        # and potentially re-stream with tool results
        # For now, we just stream the text

    def get_full_response_for_storage(
        self,
        streamed_tokens: List[str]
    ) -> str:
        """Combine streamed tokens into full response for database storage"""
        return "".join(streamed_tokens)