# algorithm.py
from AlgorithmImports import *
from config import SimulationConfig
from agent.base import Agent

class TradingAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.config: SimulationConfig = self._load_config()

        self.SetStartDate(self.config.start)
        self.SetEndDate(self.config.end)
        self.SetCash(self.config.cash)

        for symbol in self.config.symbols:
            self.AddEquity(symbol, self._resolution())

        if self.config.warmup > 0:
            self.SetWarmup(self.config.warmup)

        self.agent: Agent = self._build_agent()

    def OnData(self, slice: Slice):
        if self.IsWarmingUp:
            self.agent.observe(slice)
            return

        self.agent.observe(slice)
        actions = self.agent.act()

        for action in actions:
            self._execute(action)

    def OnEndOfAlgorithm(self):
        self.agent.on_finish()

    # --- helpers ---
    def _execute(self, action):
        # Thin translation layer only
        if action.type == "order":
            self.MarketOrder(action.symbol, action.quantity)