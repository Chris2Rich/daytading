import dspy
import os
import requests
import json
import asyncio
import aiohttp
from typing import AsyncGenerator, Generator, Optional, Dict, Any


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
            "kimik2": "moonshotai/kimi-k2-instruct-0905",
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

        async with dspy.context():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30),
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

    async def stream_async_memory(self, messages: list, memory: list, max_tokens: int = 4000):
        """
        Stream tokens asynchronously and automatically append final response to memory.

        Args:
            messages: List of messages (user + assistant) to send to LM
            memory: List[dict] that represents the conversation history
            max_tokens: Maximum tokens to generate

        Yields:
            str: individual token chunks from LM as they are generated

        Side effect:
            Appends {"role": "assistant", "content": response_text} to memory after streaming completes.
        """
        response_text = ""

        async for token in self.stream_async(messages, max_tokens):
            response_text += token
            yield token

        # After streaming completes, append to memory
        memory.append({"role": "assistant", "content": response_text})