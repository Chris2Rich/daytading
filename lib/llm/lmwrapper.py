import dspy
import os


routers = {
    "OPENROUTER" : {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": os.environ["OPENROUTER_TOKEN_DAYTADING"],
        "models": {
            "kimik2": "openrouter/moonshotai/kimi-k2:free", # unstable with dspy
            "horizonbeta": "openrouter/openrouter/horizon-beta",
            "qwen3coder480b": "openrouter/qwen/qwen3-coder:free", 
            "glm4.5air": "openrouter/z-ai/glm-4.5-air:free", 
            "hunyuan13b": "openrouter/tencent/hunyuan-a13b-instruct:free", 
            "r1t2chimera" : "openrouter/tngtech/deepseek-r1t2-chimera:free",
            "r10528" : "openrouter/deepseek/deepseek-r1-0528:free",
            "deepseekv3.1" : "openrouter/deepseek/deepseek-chat-v3.1:free"
        }
    },
    "GROQ" : {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key": os.environ["GROQ_TOKEN_DAYTADING"],
        "models": {
            "llama70bdsr1": "deepseek-r1-distill-llama-70b"
        }
    }
}

class LM_WRAPPER():
    def __init__(self, model: str = routers["OPENROUTER"]["models"]["deepseekv3.1"], temperature: float = 0.0, provider: str = "OPENROUTER"):

        self.lm = dspy.LM(model, api_key=routers[provider]["api_key"], api_base=routers[provider]["base_url"], temperature=temperature, max_tokens=4000, stop=None, cache=True, cache_in_memory=False, extra_headers=(None if provider != "OPENROUTER" else {"HTTP-Referer": "https://nodomainyet.com","X-Title": "notitleyet"}))
        dspy.configure(lm=self.lm)