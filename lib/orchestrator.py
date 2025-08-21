import argparse
import traceback
import os
import sys
import requests
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from pathlib import Path
import json
import dspy
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict
import datetime
import pandas as pd
import numpy as np
import re
import time

# from retrieval.filerag import create_context, Chunk, metadata
# context = create_context(datadir=Path("data/filings"), embeddings_dir=Path("data/embeddings"), should_embed=False)
from executionsteps import ExecutionStep, ExecutionResult

routers = {
    "OPENROUTER" : {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": os.environ["OPENROUTER_TOKEN_DAYTADING"],
        "models": {
            "kimik2": "openrouter/moonshotai/kimi-k2:free", # unstable with dspy
            "horizonbeta": "openrouter/openrouter/horizon-beta",
            "quen3coder30b": "openrouter/qwen/qwen3-coder:free", 
            "glm4.5air": "openrouter/z-ai/glm-4.5-air:free", 
            "hunyuan13b": "openrouter/tencent/hunyuan-a13b-instruct:free", 
            "r1t2chimera" : "openrouter/tngtech/deepseek-r1t2-chimera:free",
            "r10528" : "openrouter/deepseek/deepseek-r1-0528:free",
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

@dataclass
class RetrievalSteps:
    SEARCH_INTERNET: bool = False
    SEARCH_NEWS: bool = False

    RETRIEVE_MACROECONOMIC_DATA: bool = False
    RETRIEVE_MARKET_TRADING_DATA: bool = False
    RETRIEVE_COMPANY_FUNDAMENTAL_DATA: bool = False
    RETRIEVE_SEC_FILING_DATA: bool = False

@dataclass
class StrategySteps:
    DEVELOP_TRADING_STRATEGY: bool = False
    BACKTEST_TRADING_STRATEGY: bool = False

@dataclass
class AnalysisSteps:
    CALCULATE_FUNDAMENTAL_RATIOS: bool = False
    CALCULATE_TIMESERIES_STATISTICS: bool = False

@dataclass
class ReportingSteps:
    SUMMARIZE_FINDINGS: bool = False
    CREATE_SHORT_REPORT: bool = False
    CREATE_DETAILED_REPORT: bool = False

@dataclass
class SystemSteps:
    EVALUATION_CHECKPOINT: bool = False

def steps_to_list(steptype):
    return list(steptype.__annotations__.keys())

@dataclass
class StepTypeUnion:
    retrieval: RetrievalSteps
    strategy: StrategySteps
    analysis: AnalysisSteps
    reporting: ReportingSteps
    system: SystemSteps
    
    @classmethod
    def get_all_step_names(cls, ignore: set = {}) -> List[str]:
        """Get all possible step names from all group classes"""
        all_steps = []
        for _, field_type in cls.__annotations__.items():
            if field_type not in ignore:
                all_steps.extend(steps_to_list(field_type))
        return all_steps

class LM_WRAPPER():
    def __init__(self, model: str = routers["OPENROUTER"]["models"]["r10528"], temperature: float = 0.6, provider: str = "OPENROUTER"):

        self.lm = dspy.LM(model, api_key=routers[provider]["api_key"], api_base=routers[provider]["base_url"], temperature=temperature, max_tokens=4000, stop=None, cache=True, cache_in_memory=False, extra_headers=(None if provider != "OPENROUTER" else {"HTTP-Referer": "https://nodomainyet.com","X-Title": "notitleyet"}))
        dspy.configure(lm=self.lm)

class Orchestrator(LM_WRAPPER):
    def query_to_steps(self, query: str, debug: bool = False) -> Dict:
        return {"steps" : []}

    def parameterize_query(self, query: str, steps: List[str], debug: bool = False) -> Dict:
        class FinancialQueryParameterizer(dspy.Signature):
            """
            Determine the correct parameters for each step.
            Eg. the correct search strings for searches or most relevant ratios when calculating ratios.
            Return in Dspy format.
            Be terse.
            """

            query = dspy.InputField()
            named_entities = dspy.OutputField(
                desc=f"financial query -> (any amount) [named financial entities]"
                "For internet and news searches (corporations, government agencies)"
                "Eg CERN, Columbia University, Lockheed Martin"
                "If empty then return ''"
            )
            named_assets = dspy.OutputField(
                desc=f"financial query -> (any amount) [named assets]"
                "For database searches"
                "Eg SPY, APPL, CPIAUCSL"
                "If empty then return ''"
            )
            named_concepts = dspy.OutputField(
                desc=f"financial query -> (any amount) [named concepts]"
                "For internet and news searches"
                "Specialist terms in industry and finance, eg. backwardation, semaglutide, quantum entanglement"
                "If empty then return ''"
            )
            if "SEARCH_INTERNET" in steps: search_internet_query = dspy.OutputField(desc=f"named_entities, named_assets, named_concepts, financial query -> (any amount) [internet search queries]""Comma seperated values")
            if "SEARCH_NEWS" in steps: search_news_query = dspy.OutputField(desc=f"named_entities, named_assets, financial query -> (any amount) [news search queries]""Comma seperated values")
            # MACRO - TODO
            if "RETRIEVE_MARKET_TRADING_DATA" in steps: market_trading_data_query = dspy.OutputField(desc=f"named_entities, named_assets, financial query -> (any amount) [trading data queries]""Comma seperated values")
            if "RETRIEVE_COMPANY_FUNDAMENTAL_DATA" in steps: company_fundamental_data_query = dspy.OutputField(desc=f"named_entities, named_assets, financial query -> (any amount) [company fundamental data queries]""Comma seperated values")
            if "RETRIEVE_SEC_FILING_DATA" in steps: sec_filings_query = dspy.OutputField(desc=f"named_entities, named_assets, financial query -> (any amount) [sec filing queries]""Comma seperated values")
        parameterizer = dspy.ChainOfThought(FinancialQueryParameterizer)
        result = parameterizer(query=query)

        if debug:
            print(result)
        return dict(result)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="LLM")
        parser.add_argument("query", help="Query")
        parser.add_argument("-db", "--debug", help="Debug Mode", default="False")
        args = parser.parse_args()

        start = time.time_ns() // 1e6
        orchestrator = Orchestrator(temperature=0, model=routers["OPENROUTER"]["models"]["hunyuan13b"])
        print(f"Initialized orchestrator - {time.time_ns() // 1e6 - start}ms")
        res_analyze = orchestrator.query_to_steps(args.query, args.debug.lower() == "true")
        print(f"Analyzed Query - {time.time_ns() // 1e6 - start}ms")
        res_parameterize = orchestrator.parameterize_query(args.query, res_analyze["steps"], args.debug.lower() == "true")
        print(f"Parameterized Query - {time.time_ns() // 1e6 - start}ms")
        print(f"Steps: {res_analyze['steps']}\nParameters: {dict((k, v) for k, v in res_parameterize.items() if k != 'reasoning' and v != str(chr(39)) * 2 )}")
    except:
        traceback.print_exc()
    finally:
        sys.exit()