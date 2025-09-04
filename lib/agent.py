from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from enum import Enum
import dspy
import time

from llm.lmwrapper import LM_WRAPPER
from executionsteps import ExecutionStep, ExecutionResult
from reporting.summarize import summarize

class AgentType(Enum):
    RETRIEVAL = "RETRIEVAL"
    STRATEGY = "STRATEGY"
    ANALYSIS = "ANALYSIS"
    REPORTING = "REPORTING"
    SYSTEM = "SYSTEM"

class BaseAgent(LM_WRAPPER):
    def __init__(self, category: AgentType = None, instructions: Dict[str, List[str]] = None):
        # Agent takes in type and a list of instruction names and generic parameters as given by parameterizer
        super().__init__()
        self.category = category
        self.instructions = instructions
    
    def _get_tools(self):
        pass

    def _parameterize(self, instruction: Tuple[str, list[str]]):
        pass

    def add(self, t : Union[Tuple[str, List[str]], List[Tuple[str, List[str]]]]):
        if type(t) == list:
            for i in t:
                self.instructions.update(i)
        elif type(t) == tuple:
            self.instructions.update(t)

    def execute(self):
        return None

class ReportingAgent(BaseAgent):
    def __init__(self, instructions: List[Tuple[str, List[str]]] = None):
        super().__init__(category=AgentType.REPORTING, instructions=instructions)
    
    def execute(self, instruction: Optional[Tuple[str, List[str]]], use_md: bool = False) -> ExecutionResult:
        try:
            if instruction == None:
                instruction = self.instructions[0]
                self.instructions = self.instructions[1:]
            match instruction[0]:
                case "SUMMARIZE_FINDINGS":
                    return ExecutionResult(True, summarize(context=instruction[1][0], data=instruction[1][1], use_md=use_md))
                case "CREATE_SHORT_REPORT: 
                    raise Exception("Method not implemented yet")
                case "CREATE_DETAILED_REPORT:
                    raise Exception("Method not implemented yet")
        except Exception as e:
            return ExecutionResult(False, None, e)

class RetrievalAgent(BaseAgent):
    def __init__(self, instructions: List[Tuple[str, List[str]]] = None):
        super().__init__(category=AgentType.RETRIEVAL, instructions=instructions)

    def execute(self, instruction: Optional[Tuple[str, List[str]]], use_md: bool = False) -> ExecutionResult:
        try:
            if instruction == None:
                instruction = self.instructions[0]
                self.instructions = self.instructions[1:]
            match instruction[0]:
                case "RETRIEVE_MARKET_TRADING_DATA":
                    return ExecutionResult(True)
        except Exception as e:
            return ExecutionResult(False, None, e)

class StrategyAgent(BaseAgent):
    def __init(self, instructions:List[Tuple[str, List[str]]] = None):
        super.__init__(category=AgentType.STRATEGY, instructions=instructions)
    def execute(self, instruction: Optional[Tuple[str, List[str]]]) -> ExecutionResult:
        try:
            if instruction == None:
                instruction = self.instructions[0]
                self.instructions = self.instructions[1:]
            match instruction[0]:
                case "DEVELOP_TRADING_STRATEGY":
                    raise Exception("Method not implemented yet")
                case "BACKTEST_TRADING_STRATEGY":
                    raise Exception("Method not implemented yet")
        except Exception as e:
            return ExecutionResult(False, None, e)
