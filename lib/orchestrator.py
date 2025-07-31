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

class NextStepType(TypedDict):
    """Union of booleans representing the next steps to take based on query analysis"""
    STRATEGIZE: bool
    RETRIEVE_MACRO_DATA: bool
    RETRIEVE_TRADING_DATA: bool
    RETRIEVE_FILING_DATA: bool
    BACKTEST_STRATEGY: bool
    SEARCH_INTERNET: bool
    CREATE_REPORT: bool

class Orchestrator():
    def __init__(self):
        self.lm = dspy.LM("deepseek-r1-distill-llama-70b",  api_key=os.environ.get("GROQ_TOKEN_DAYTADING"),  api_base="https://api.groq.com/openai/v1/chat/completions", temperature=0.5, max_tokens=4000, stop=None)
        dspy.configure(lm=self.lm)
        
    def analyze_query(self, query: str) -> List[NextStepType]:
        """Break down query and figure out what is needed"""
        
        class QueryAnalysis(dspy.Signature):
            """Analyze a query and determine what steps are needed from the available options."""
            query = dspy.InputField()
            next_steps = dspy.OutputField(
            desc="Provide a comma-separated list of next steps. "
                "Each item must be one of the following exactly: "
                f"{NextStepType.__annotations__.keys()}. "
                "Do not include anything else."
                "Order each element in the order that the task should be done."
                "You are allowed to use items multiple times if neccessary."
            )
        
        analyzer = dspy.Predict(QueryAnalysis)
        result = analyzer(query=query)

        return [s.strip() for s in str(getattr(result, 'next_steps', '')).upper().split(',') if s.strip() in NextStepType.__annotations__.keys()]

# Example usage
if __name__ == "__main__":
    orchestrator = Orchestrator()
    
    # Test query analysis
    test_queries = [
        "What are the key financial metrics for AAPL in Q4?",
        "Create a trading strategy for tech stocks", 
        "Backtest my momentum strategy on SPY",
        "Search for recent news about inflation",
        "Generate a report on market volatility",
        "Investigate the market dynamics of IV crush around earnings and create a strategy that implements this tune the hyperparameters and then create a report."
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        steps = orchestrator.analyze_query(query)
        print(f"Next Steps: {[s for s in steps]}")