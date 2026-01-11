# config.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List

@dataclass(frozen=True)
class SimulationConfig:
    start: datetime
    end: datetime
    cash: float

    symbols: List[str]
    resolution: str
    warmup: int

    agent_class: str
    agent_params: Dict[str, Any]

    training_frequency: str | None
    execution_constraints: Dict[str, Any]
