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
    # Retrieval
    RETRIEVE_MACROECONOMIC_DATA: bool
    RETRIEVE_TRADING_DATA: bool
    RETRIEVE_COMPANY_FUNDAMENTAL_DATA: bool
    RETRIEVE_SEC_FILING_DATA: bool
    SEARCH_INTERNET: bool

    # Strategy
    DEVELOP_STRATEGY: bool
    BACKTEST_STRATEGY: bool

    # Reporting
    SUMMARIZE_FINDINGS: bool
    CREATE_SHORT_REPORT: bool
    CREATE_DETAILED_REPORT: bool

    # System
    EVALUATION_CHECKPOINT: bool

class Orchestrator():
    def __init__(self, model: str = "deepseek-r1-distill-llama-70b"):
        self.lm = dspy.LM(model,  api_key=os.environ.get("GROQ_TOKEN_DAYTADING"),  api_base="https://api.groq.com/openai/v1/chat/completions", temperature=1.0, max_tokens=4000, stop=None, cache=False, cache_in_memory=False)
        dspy.configure(lm=self.lm)
        
    def analyze_query(self, query: str) -> List[NextStepType]:
        """Break down query and figure out what is needed"""
        
        class QueryAnalysis(dspy.Signature):
            """Analyze a query and determine what steps are needed from the available options."""
            query = dspy.InputField()
            next_steps = dspy.OutputField(
            desc="Provide a comma-separated list of next steps. "
                "Each item must be one of the following exactly: "
                f"{list(filter(None, [i if i != 'EVALUATION_CHECKPOINT' else None for i in NextStepType.__annotations__.keys()]))}. "
                "Do not include anything else."
                "Order each element in the order that the task should be done."
                "Think out why each step is needed and the consider the entire workflow before responding."
                "Ensure you don't forget steps."
                "Always end with some type of report. Note that CREATE_DETAILED_REPORT is time consuming and could be excessive for simple questions or processes."
                "Consider any potentially retrieved data in the context of the question, don't fetch uneccessary types of data (eg. fetching trading/filing data when answering a simple macro question)."
                "Create the simplest list of steps possible, the list will be modified or added to during checkpoints if neccessary."
                "Make sure to follow all rules specified"
            )
        
        analyzer = dspy.Predict(QueryAnalysis)
        result = analyzer(query=query)

        return [s.strip() for s in str(getattr(result, 'next_steps', '')).upper().split(',') if s.strip() in NextStepType.__annotations__.keys()]

# Example usage
if __name__ == "__main__":
    orchestrator = Orchestrator("meta-llama/llama-4-maverick-17b-128e-instruct")
    
    # Test query analysis
    test_queries = [
        "What are the key financial metrics for AAPL in Q4?",
        "Create a trading strategy for tech stocks", 
        "Search for recent news about inflation",
        "Generate a report on overall market volatility and it's impact on options prices",
        "Investigate the market dynamics of IV crush around earnings and create a strategy that implements this tune the hyperparameters and then create a report."
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        for i in range(3):
            steps = orchestrator.analyze_query(query)
            print(f"Next Steps: {[s for s in steps]}")