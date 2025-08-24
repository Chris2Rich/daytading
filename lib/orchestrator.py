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
    BEGIN: bool = False
    END: bool = False
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
    def analyze_query(self, query: str, debug: bool = False) -> Dict:
        class FinancialQueryAnalysis(dspy.Signature):
            """
            Determine minimal workflow steps for financial analysis tasks.
            Return in Dspy format.
            """
            
            query = dspy.InputField(desc="Original input query")
            query_type = dspy.OutputField(
                desc=f"financial query -> analyze, strategize, explain"
            )
            report_format = dspy.OutputField(
                desc=f"financial query -> [report format in : {steps_to_list(ReportingSteps)}]"
            )
            data_sources_needed = dspy.OutputField(
                desc=f"financial query, query type -> (any amount) [data source in : {steps_to_list(RetrievalSteps)}]"
            )
            analysis_type = dspy.OutputField(
                desc=f"financial query, query type -> (any amount) [analysis type in : {steps_to_list(AnalysisSteps)}]"
            )
            strategy_requirements = dspy.OutputField(
                desc=f"financial query, query type -> (any amount) [strategy step in : {steps_to_list(StrategySteps)}."
                "if backtest and develop not precedent -> [develop_strategy, backtest_strategy]"
            )
            workflow_reasoning = dspy.OutputField(
                desc=f"financial query, query type, processes -> financial workflow"
            )
            workflow_simplifier = dspy.OutputField(
                desc=f"financial query, query type, superfluous financial workflow -> neccessary financial workflow"
            )
            suggested_workflow = dspy.OutputField(
                desc=f"Final comma-separated workflow from: {StepTypeUnion.get_all_step_names({SystemSteps})}"
                "data retrieval -> analysis -> strategy -> reporting"
            )
        
        analyzer = dspy.ChainOfThought(FinancialQueryAnalysis)
        result = analyzer(query=query)
        steps_list = [s.strip() for s in str(getattr(result, 'suggested_workflow', '')).upper().split(',') if s.strip()]
        reasoning = str(getattr(result, "reasoning", ""))

        if debug:
            print(result)
        return {"steps": steps_list, "reasoning": reasoning}


    def parameterize_query(self, query: str, steps_v: List[str], reasoning: str, debug: bool = False) -> Dict:
        class FinancialQueryParameterizer(dspy.Signature):
            """
            Determine the correct parameters for each step.
            Eg. the correct search strings for searches or most relevant ratios when calculating ratios.
            Return in Dspy format.
            Be terse.
            """

            query = dspy.InputField(desc="Original input query")
            reasoning = dspy.InputField(desc="Reasoning behind workflow steps")
            steps = dspy.InputField(desc="List of required workflow steps")
            named_organizations = dspy.OutputField(
                desc=f"financial query -> (any amount) [named organizations]"
                "For internet and news searches (corporations, government agencies)"
                "Eg CERN, Columbia University, Lockheed Martin"
                "If empty then return ''"
            )
            named_stocks = dspy.OutputField(
                desc=f"financial query -> (any amount) [named stocks]"
                "For database searches"
                "Eg SPY, APPL, GOOG"
                "If empty then return ''"
            )
            named_concepts = dspy.OutputField(
                desc=f"financial query -> (any amount) [named concepts]"
                "For internet and news searches"
                "Specialist terms in industry and finance, eg. backwardation, semaglutide, quantum entanglement"
                "If empty then return ''"
            )
            if "SEARCH_INTERNET" in steps_v: search_internet_query = dspy.OutputField(desc=f"named_entities, named_assets, named_concepts, financial query -> (any amount) [internet search queries]""Comma seperated values")
            if "SEARCH_NEWS" in steps_v: search_news_query = dspy.OutputField(desc=f"named_entities, named_assets, financial query -> (any amount) [news search queries]""Comma seperated values")
            # MACRO - TODO
            if "RETRIEVE_MARKET_TRADING_DATA" in steps_v: market_trading_data_query = dspy.OutputField(desc=f"named_entities, named_assets, financial query -> (any amount) [trading data queries]""Comma seperated values")
            if "RETRIEVE_COMPANY_FUNDAMENTAL_DATA" in steps_v: company_fundamental_data_query = dspy.OutputField(desc=f"named_entities, named_assets, financial query -> (any amount) [company fundamental data queries]""Comma seperated values")
            if "RETRIEVE_SEC_FILING_DATA" in steps_v: sec_filings_query = dspy.OutputField(desc=f"named_entities, named_assets, financial query -> (any amount) [sec filing queries]""Comma seperated values")
        parameterizer = dspy.ChainOfThought(FinancialQueryParameterizer)
        result = dict(parameterizer(query=query, steps=", ".join(steps_v), reasoning=reasoning))

        if debug:
            print(result)
        return {k: [v.strip().rstrip() for v in result[k].split(",")] for k in result}

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="LLM")
        parser.add_argument("query", help="Query")
        parser.add_argument("-db", "--debug", help="Debug Mode", default="False")
        args = parser.parse_args()

        start = time.time_ns() // 1e6
        orchestrator = Orchestrator(temperature=0, model=routers["OPENROUTER"]["models"]["hunyuan13b"])
        print(f"Initialized orchestrator - {time.time_ns() // 1e6 - start}ms")
        res_analyze = orchestrator.analyze_query(args.query, args.debug.lower() == "true")
        print(f"Found steps - {time.time_ns() // 1e6 - start}ms")
        print(f"Steps: {res_analyze['steps']}")
        res_parameterize = orchestrator.parameterize_query(args.query + str(orchestrator.parameterize_query(args.query, res_analyze["steps"], res_analyze["reasoning"], args.debug.lower() == "true")), res_analyze["steps"], res_analyze["reasoning"], args.debug.lower() == "true")
        print(f"Parameterized steps - {time.time_ns() // 1e6 - start}ms")
        print(f"Parameters: {dict((k, v) for k, v in res_parameterize.items() if k != 'reasoning' and v != str(chr(39)) * 2 )}")
    except:
        traceback.print_exc()
    finally:
        sys.exit()