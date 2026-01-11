# ============================================================
# Quant Research Backbone — Single File
# LEAN-compatible, walk-forward safe, agent-based architecture
# ============================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import timedelta
from abc import ABC, abstractmethod
import numpy as np

from AlgorithmImports import QCAlgorithm

# ============================================================
# 1. CONFIGURATION (Single Immutable Contract)
# ============================================================

@dataclass(frozen=True)
class DataConfig:
    symbols: List[str]
    resolution: str = "Minute"
    start_date: Tuple[int, int, int] = (2018, 1, 1)
    end_date: Tuple[int, int, int] = (2024, 1, 1)
    warmup_bars: int = 500

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
# 2. CORE INTERFACES (STRICT CONTRACTS)
# ============================================================

class DataSource(ABC):
    @abstractmethod
    def observe(self, slice) -> Dict[str, Any]:
        """Extract causal observations from LEAN Slice"""
        pass

class FeaturePipeline(ABC):
    @abstractmethod
    def update(self, observation: Dict[str, Any]) -> np.ndarray:
        """Update internal buffers and emit features"""
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
        """Return frozen copy for training"""
        pass

class OutcomeModel(ABC):
    @abstractmethod
    def compute(self, history: Any) -> Any:
        """Compute reward or label causally"""
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
        """Snapshot experience for training"""
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
# 3. MINIMAL SAFE IMPLEMENTATIONS (NON-STRATEGIC)
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
# 4. FACTORIES (CONFIG-DRIVEN SUBSTITUTION POINTS)
# ============================================================

class Factory:

    @staticmethod
    def belief(config: Config) -> BeliefState:
        if config.belief.type == "identity":
            return IdentityBeliefState()
        raise NotImplementedError("Unknown belief type")

    @staticmethod
    def scheduler(config: Config) -> TrainingScheduler:
        return PeriodicTrainingScheduler(config.training.retrain_frequency)

# ============================================================
# 5. LEAN ORCHESTRATION (NO STRATEGY LOGIC)
# ============================================================

class ResearchAlgorithm(QCAlgorithm):

    def Initialize(self):

        self.config = self._load_config()

        self.SetStartDate(*self.config.data.start_date)
        self.SetEndDate(*self.config.data.end_date)
        self.SetCash(self.config.execution.initial_cash)
        self.SetWarmUp(self.config.data.warmup_bars)

        np.random.seed(self.config.backtest.seed)

        # Build components (user-provided implementations)
        self.data_source = self._build_data_source()
        self.features = self._build_features()
        self.regime = self._build_regime()
        self.belief = Factory.belief(self.config)
        self.agent = self._build_agent()
        self.scheduler = Factory.scheduler(self.config)
        self.execution = self._build_execution()

        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Midnight,
            self._scheduled_training
        )

    def OnData(self, slice):
        if self.IsWarmingUp:
            return

        observation = self.data_source.observe(slice)
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

    # ========================================================
    # USER MUST IMPLEMENT THESE BUILD METHODS
    # ========================================================

    def _load_config(self) -> Config:
        raise NotImplementedError

    def _build_data_source(self) -> DataSource:
        raise NotImplementedError

    def _build_features(self) -> FeaturePipeline:
        raise NotImplementedError

    def _build_regime(self) -> Optional[RegimeModel]:
        return None

    def _build_agent(self) -> Agent:
        raise NotImplementedError

    def _build_execution(self) -> ExecutionEngine:
        raise NotImplementedError
