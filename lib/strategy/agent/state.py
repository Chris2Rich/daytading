# agent/state.py
from dataclasses import dataclass

@dataclass
class OrderAction:
    type: str
    symbol: str
    quantity: int
