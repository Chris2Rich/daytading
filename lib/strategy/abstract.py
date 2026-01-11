# ============================================================
# Quant Research Backbone — LEAN-Native, Configurable, Agent-Based
# ============================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import timedelta
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from AlgorithmImports import QCAlgorithm, BaseData, SubscriptionDataSource, SubscriptionTransportMedium

from datasources import YahooFinanceOHLCV

# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DataConfig:
    symbols: List[str]
    resolution: str = "Minute"  # LEAN Resolution: Minute, Hour, Daily
    start_date: Tuple[int, int, int] = (2018, 1, 1)
    end_date: Tuple[int, int, int] = (2024, 1, 1)
    warmup_bars: int = 500
    ohlcv_days: int = 365  # number of historical days to fetch from Yahoo

@dataclass(frozen=True)
class FeatureConfig:
    lookback_window: int = 50
    feature_set: List[str] = field(default_factory=list)
    normalize: bool = True

@dataclass(frozen=True)
class RegimeConfig:
    enabled: bool = False
    model_type: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class LabelConfig:
    method: str = "triple_barrier"
    horizon: int = 20
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class BeliefConfig:
    type: str = "identity"
    state_dim: Optional[int] = None

@dataclass(frozen=True)
class PolicyConfig:
    type: str = "heuristic"
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AgentConfig:
    agent_type: str = "learning"
    discount_factor: float = 0.99
    memory_size: int = 100_000

@dataclass(frozen=True)
class TrainingConfig:
    enabled: bool = True
    retrain_frequency: timedelta = timedelta(days=30)
    min_samples: int = 5_000
    rolling_window: Optional[int] = None

@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 1_000_000
    max_position_size: float = 0.10

@dataclass(frozen=True)
class BacktestConfig:
    seed: int = 42

@dataclass(frozen=True)
class Config:
    data: DataConfig
    features: FeatureConfig
    regime: RegimeConfig
    labeling: LabelConfig
    belief: BeliefConfig
    policy: PolicyConfig
    agent: AgentConfig
    training: TrainingConfig
    execution: ExecutionConfig
    backtest: BacktestConfig

# ============================================================
# 2. LEAN CUSTOM DATA TYPE (Yahoo OHLCV)
# ============================================================

# ============================================================
# 3. CORE INTERFACES
# ============================================================

class FeaturePipeline(ABC):
    @abstractmethod
    def update(self, observation: Dict[str, Any]) -> np.ndarray:
        pass

class RegimeModel(ABC):
    @abstractmethod
    def infer(self, features: np.ndarray) -> Any:
        pass

class BeliefState(ABC):
    @abstractmethod
    def update(self, features: np.ndarray, regime: Optional[Any]) -> Any:
        pass

class Policy(ABC):
    @abstractmethod
    def act(self, belief: Any) -> Any:
        pass
    @abstractmethod
    def clone(self) -> "Policy":
        pass

class OutcomeModel(ABC):
    @abstractmethod
    def compute(self, history: Any) -> Any:
        pass

class Agent(ABC):
    @abstractmethod
    def act(self, belief: Any) -> Any:
        pass
    @abstractmethod
    def observe_outcome(self, outcome: Any) -> None:
        pass
    @abstractmethod
    def freeze_memory(self) -> Any:
        pass
    @abstractmethod
    def train_on(self, snapshot: Any) -> Policy:
        pass
    @abstractmethod
    def swap_policy(self, policy: Policy) -> None:
        pass

class TrainingScheduler(ABC):
    @abstractmethod
    def should_train(self, time) -> bool:
        pass

class ExecutionEngine(ABC):
    @abstractmethod
    def execute(self, action: Any) -> None:
        pass

# ============================================================
# 4. MINIMAL SAFE IMPLEMENTATIONS
# ============================================================

class PeriodicTrainingScheduler(TrainingScheduler):
    def __init__(self, frequency: timedelta):
        self.frequency = frequency
        self.last_train_time = None
    def should_train(self, time) -> bool:
        if self.last_train_time is None:
            self.last_train_time = time
            return False
        if time - self.last_train_time >= self.frequency:
            self.last_train_time = time
            return True
        return False

class IdentityBeliefState(BeliefState):
    def update(self, features: np.ndarray, regime: Optional[Any]) -> Any:
        return features

# ============================================================
# 5. LEAN ALGORITHM
# ============================================================

class ResearchAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.config = self._load_config()
        self.SetStartDate(*self.config.data.start_date)
        self.SetEndDate(*self.config.data.end_date)
        self.SetCash(self.config.execution.initial_cash)
        self.SetWarmUp(self.config.data.warmup_bars)
        np.random.seed(self.config.backtest.seed)

        # Subscribe Yahoo OHLCV for each symbol
        self.yahoo_bars = []
        for sym in self.config.data.symbols:
            bar = self.AddData(YahooFinanceOHLCV, sym, getattr(self.config.data, "resolution"))
            self.yahoo_bars.append(bar)

        # Build agent components (user-defined)
        self.features = self._build_features()
        self.regime = self._build_regime()
        self.belief = IdentityBeliefState()
        self.agent = self._build_agent()
        self.scheduler = PeriodicTrainingScheduler(self.config.training.retrain_frequency)
        self.execution = self._build_execution()

        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Midnight, self._scheduled_training)

    def OnData(self, slice):
        if self.IsWarmingUp:
            return

        # Extract observation per symbol from slice
        observation = {}
        for sym in self.config.data.symbols:
            bar = slice.Bars.get(sym)
            if bar:
                observation[sym] = {
                    "open": bar.Open,
                    "high": bar.High,
                    "low": bar.Low,
                    "close": bar.Close,
                    "volume": bar.Volume,
                    "timestamp": bar.EndTime,
                }

        features = self.features.update(observation)
        regime = self.regime.infer(features) if self.regime else None
        belief = self.belief.update(features, regime)
        action = self.agent.act(belief)
        self.execution.execute(action)

    def _scheduled_training(self):
        if not self.scheduler.should_train(self.Time):
            return
        snapshot = self.agent.freeze_memory()
        new_policy = self.agent.train_on(snapshot)
        self.agent.swap_policy(new_policy)

    # --------------------------------------------------------
    # User must implement these
    # --------------------------------------------------------
    def _load_config(self) -> Config:
        raise NotImplementedError
    def _build_features(self) -> FeaturePipeline:
        raise NotImplementedError
    def _build_regime(self) -> Optional[RegimeModel]:
        return None
    def _build_agent(self) -> Agent:
        raise NotImplementedError
    def _build_execution(self) -> ExecutionEngine:
        raise NotImplementedError
