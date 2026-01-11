# agent/base.py
from abc import ABC, abstractmethod
from AlgorithmImports import Slice

class Agent(ABC):

    @abstractmethod
    def observe(self, slice: Slice) -> None:
        pass

    @abstractmethod
    def act(self):
        """Return a list of actions"""
        pass

    def on_finish(self) -> None:
        pass
