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

# from retrieval.filerag import create_context, Chunk, metadata
# context = create_context(datadir=Path("data/filings"), embeddings_dir=Path("data/embeddings"), should_embed=False)

routers = {
    "OPENROUTER" : {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": os.environ["OPENROUTER_TOKEN_DAYTADING"],
        "models": {
            "kimik2": "openrouter/moonshotai/kimi-k2:free", 
            "horizonbeta": "openrouter/openrouter/horizon-beta", 
            "quen3coder30b": "openrouter/qwen/qwen3-coder:free", 
            "glm4.5air": "openrouter/z-ai/glm-4.5-air:free", 
            "hunyuana13b": "openrouter/tencent/hunyuan-a13b-instruct:free", 
            "r1t2chimera" : "openrouter/tngtech/deepseek-r1t2-chimera:free",
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
class StepUnion:
    """Union of booleans representing the next steps to take based on query analysis"""
    retrieval: RetrievalSteps
    strategy: StrategySteps
    analysis: AnalysisSteps
    reporting: ReportingSteps
    system: SystemSteps
    
    @classmethod
    def get_all_step_names(cls, ignore: set = {}) -> List[str]:
        """Get all possible step names from all group classes"""
        all_steps = []
        for field_name, field_type in cls.__annotations__.items():
            if hasattr(field_type, '__annotations__') and field_type not in ignore:
                all_steps.extend(field_type.__annotations__.keys())
        return all_steps
    
    @classmethod
    def from_step_list(cls, steps_list: List[str]) -> 'StepUnion':
        """Create StepUnion from a list of step names"""
        retrieval = RetrievalSteps()
        strategy = StrategySteps() 
        analysis = AnalysisSteps()
        reporting = ReportingSteps()
        system = SystemSteps()
        
        step_groups = {
            'retrieval': retrieval,
            'strategy': strategy,
            'analysis': analysis,
            'reporting': reporting,
            'system': system
        }
        
        for step in steps_list:
            step = step.strip().upper()
            for group_name, group_obj in step_groups.items():
                if hasattr(group_obj, step):
                    setattr(group_obj, step, True)
                    break
        
        return cls(retrieval=retrieval, strategy=strategy, analysis=analysis, reporting=reporting, system=system)
    
    def get_active_steps(self) -> List[str]:
        """Get list of all active steps"""
        active_steps = []
        for field_name, field_type in self.__annotations__.items():
            group_obj = getattr(self, field_name)
            if hasattr(field_type, '__annotations__'):
                for step_name in field_type.__annotations__.keys():
                    if getattr(group_obj, step_name, False):
                        active_steps.append(step_name)
        return active_steps

class Orchestrator():
    def __init__(self, model: str = routers["OPENROUTER"]["models"]["kimik2"], temperature: float = 0.6, provider: str = "OPENROUTER"):

        self.lm = dspy.LM(model, api_key=routers[provider]["api_key"], api_base=routers[provider]["base_url"], temperature=temperature, max_tokens=4000, stop=None, cache=False, cache_in_memory=False)
        dspy.configure(lm=self.lm)
    
    def analyze_query(self, query: str, debug: bool = False) -> List:
        class FinancialQueryAnalysis(dspy.Signature):
            """
            Determine minimal workflow steps for financial analysis tasks.
            """
            
            query = dspy.InputField()
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
                desc=f"Final comma-separated workflow from: {StepUnion.get_all_step_names({SystemSteps})}"
                "data retrieval -> analysis -> strategy -> reporting"
            )
        
        analyzer = dspy.ChainOfThought(FinancialQueryAnalysis)
        result = analyzer(query=query)
        steps_list = [s.strip() for s in str(getattr(result, 'suggested_workflow', '')).upper().split(',') if s.strip()]
        
        if debug:
            print(result)
        return steps_list

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="LLM")
        parser.add_argument("query", help="Query")
        parser.add_argument("-db", "--debug", help="Debug Mode", default="False")
        args = parser.parse_args()

        orchestrator = Orchestrator(temperature=0)
        steps = orchestrator.analyze_query(args.query, args.debug.lower() == "true")
        print(f"Next Steps: {steps}")
    except:
        traceback.print_exc()
    finally:
        sys.exit()