import os
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

@dataclass
class RetrievalSteps:
    RETRIEVE_MACROECONOMIC_DATA: bool = False
    RETRIEVE_MARKET_DATA: bool = False
    RETRIEVE_COMPANY_FUNDAMENTAL_DATA: bool = False
    RETRIEVE_SEC_FILING_DATA: bool = False
    SEARCH_NEWS: bool = False
    SEARCH_INTERNET: bool = False

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
class NextStepType:
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
    def from_step_list(cls, steps_list: List[str]) -> 'NextStepType':
        """Create NextStepType from a list of step names"""
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
    def __init__(self, model: str = "deepseek-r1-distill-llama-70b"):
        self.lm = dspy.LM(model, api_key=os.environ.get("GROQ_TOKEN_DAYTADING"), api_base="https://api.groq.com/openai/v1/chat/completions", temperature=0.6, max_tokens=4000, stop=None, cache=False, cache_in_memory=False)
        dspy.configure(lm=self.lm)
    
    def analyze_query(self, query: str) -> List[NextStepType]:
        class QueryAnalysis(dspy.Signature):
            """Analyze a query step-by-step to determine the complete workflow needed."""
            
            query = dspy.InputField()
            end_goal = dspy.OutputField(
                desc=f"What is the ultimate goal? Consider what type of report is needed from: {steps_to_list(ReportingSteps)} and if any type of modelling should be done."
                "Max 100 words"
            )
            data_sources_needed = dspy.OutputField(
                desc=f"What data sources are required? Consider: {steps_to_list(RetrievalSteps)}"
                "Max 200 words"
            )
            analysis_type = dspy.OutputField(
                desc=f"What analysis steps are needed? Consider: {steps_to_list(AnalysisSteps)}"
                "Max 200 words"
            )
            strategy_requirements = dspy.OutputField(
                desc=f"What strategy steps are needed? Consider: {steps_to_list(StrategySteps)}"
                "Max 400 words"
            )
            report_complexity = dspy.OutputField(
                desc=f"What report steps are needed: Consider {steps_to_list(ReportingSteps)}"
                "Max 100 words"
            )
            workflow_reasoning = dspy.OutputField(
                desc=f"What is the logical order of steps"
                "Max 100 words"
            )
            next_steps = dspy.OutputField(
                desc=f"Final comma-separated workflow from: {NextStepType.get_all_step_names()}. "
                "Order: data retrieval -> analysis -> strategy (if needed) -> reporting"
            )
        
        analyzer = dspy.Predict(QueryAnalysis)
        result = analyzer(query=query)
        steps_list = [s.strip() for s in str(getattr(result, 'next_steps', '')).upper().split(',') if s.strip()]
        
        return NextStepType.from_step_list(steps_list)

if __name__ == "__main__":
    # orchestrator = Orchestrator("meta-llama/llama-4-maverick-17b-128e-instruct")
    orchestrator = Orchestrator()

    test_queries = [
        "What are the key financial metrics for AAPL in Q4?",
        "Create a trading strategy for tech stocks",
        "Search for recent news about inflation",
        "Generate a report on overall market volatility and it's impact on options prices",
        "Investigate the market dynamics of IV crush around earnings and create a strategy that implements this tune the hyperparameters and then create a report.",
        "Average inflation rate of TTM",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        steps = orchestrator.analyze_query(query)
        print(f"Next Steps: {steps.get_active_steps()}")