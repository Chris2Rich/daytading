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
        super().__init__(category=AgentType.RETRIEVAL, instructions=instructions)
    
    def execute(self, instruction: Optional[Tuple[str, List[str]]], use_md: bool = False) -> ExecutionResult:
        try:
            if instruction == None:
                instruction = self.instructions[0]
                self.instructions = self.instructions[1:]
            match instruction[0]:
                case "SUMMARIZE_FINDINGS":
                    return ExecutionResult(True, summarize(context=instruction[1][0], data=instruction[1][1], use_md=use_md))
        except Exception as e:
            return ExecutionResult(False, None, e)

def _test():
    context = "Implications of increased interest rates"
    data = [
        "Over the past fiscal year, central banks across major economies have increased interest rates in response to rising inflation. The U.S. Federal Reserve raised the federal funds rate from 3.5% to 5.25%, marking one of the steepest rate hikes in over a decade. Similarly, the European Central Bank followed suit, increasing rates by 175 basis points within six months. These actions aim to tighten monetary policy, curb inflation, and stabilize the economy.",
        "Higher interest rates typically result in increased borrowing costs for both consumers and businesses. As a consequence, consumer spending on big-ticket items such as homes, vehicles, and electronics tends to decline. On the business side, companies delay capital expenditures, freeze hiring, or even reduce workforce to control costs.This contraction in spending slows economic growth, potentially leading to a mild recession.",
        "Firm XYZ, a mid-sized technology and manufacturing firm based in Chicago, has already begun to feel the pressure of higher interest rates. The company reported that its average cost of capital rose from 6.8% in Q1 2024 to 8.3% in Q2 2025.Financing for new equipment and facility upgrades was deferred indefinitely due to projected increases in loan servicing costs.Operating cash flow dropped 12.4% year-over-year, from $43.1 million to $37.7 million.",
        "In terms of revenue, Firm XYZ posted a modest 3.1% increase, rising from $128 million in 2024 to $132 million in 2025. However, this was largely due to a price adjustment strategy implemented to offset rising input costs. Net profit margins narrowed, falling from 14.5% to 11.2% over the same period, largely driven by higher interest payments on outstanding debt, which increased from $6.4 million to $9.8 million.",
        "Executives at Firm XYZ have revised their forward guidance for 2026, anticipating flat revenue growth and lower operating margins if interest rates remain elevated.They have also announced a strategic shift toward asset-light operations, focusing on software services and subscription-based offerings to reduce capital-intensive investment.The CFO noted that while the balance sheet remains healthy, prolonged high rates may force renegotiation of credit facilities in early 2026.",
        "Macroeconomic forecasts indicate that rates may begin to decline gradually in late 2026 or early 2027, depending on inflationary pressures and GDP growth stabilization.Analysts believe that companies like Firm XYZ, with agile cost structures and a shift toward digital offerings, are better positioned to weather prolonged monetary tightening.Nonetheless, access to affordable capital remains a critical constraint for innovation and competitiveness in capital-intensive sectors."
    ]
    
    instruction = ("SUMMARIZE_FINDINGS", [context, data])

    agent = ReportingAgent([instruction])
    
    result = agent.execute(instruction).data
    print("Summary:\n", result)

if __name__ == "__main__":
    _test()