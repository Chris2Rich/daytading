from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime


# ============================================================================
# CONFIGURATION SCHEMA
# ============================================================================

class FeatureType(Enum):
    """Feature classification."""
    WEAK = "weak"        # Stateless transformations
    STRONG = "strong"    # Stateful/model-driven


class TrainingMode(Enum):
    """Training paradigm."""
    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement"
    ONLINE = "online"
    BANDIT = "bandit"
    HEURISTIC = "heuristic"


class WindowType(Enum):
    """Training window strategy."""
    EXPANDING = "expanding"
    ROLLING = "rolling"
    ANCHORED = "anchored"


@dataclass
class DataSourceConfig:
    """Defines a data source."""
    name: str
    source_type: str  # 'market', 'fundamental', 'alternative', 'synthetic'
    symbols: List[str]
    fields: List[str]  # ['Open', 'High', 'Low', 'Close', 'Volume']
    interval: str  # '1d', '1h', '5m'
    lookback_days: int
    provider: str  # 'yahoo', 'polygon', 'synthetic'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeakFeatureConfig:
    """Configuration for stateless feature."""
    name: str
    feature_type: str  # 'returns', 'volatility', 'technical', 'transform'
    dependencies: List[str]  # Data sources or other features
    params: Dict[str, Any] = field(default_factory=dict)
    window_size: Optional[int] = None
    symbols: Optional[List[str]] = None  # None = all symbols


@dataclass
class StrongFeatureConfig:
    """Configuration for stateful/model-driven feature."""
    name: str
    feature_type: str  # 'regime', 'sentiment', 'volatility_model', 'embedding'
    dependencies: List[str]  # Required features/data
    model_type: str  # 'lgbm', 'nn', 'hmm', 'custom'
    training_schedule: 'TrainingScheduleConfig'
    params: Dict[str, Any] = field(default_factory=dict)
    symbols: Optional[List[str]] = None


@dataclass
class TrainingScheduleConfig:
    """Defines when and how to train."""
    mode: TrainingMode
    frequency: int  # Retrain every N bars
    min_samples: int
    window_type: WindowType
    window_size: Optional[int] = None  # For rolling windows
    validation_split: float = 0.2
    warm_up_bars: int = 0
    triggers: List[str] = field(default_factory=list)  # ['regime_change', 'vol_spike']


@dataclass
class ModelConfig:
    """Model specification."""
    name: str
    model_type: str  # 'lgbm', 'nn', 'linear', 'rl_dqn', 'rl_ppo'
    input_features: List[str]  # Feature names
    output_schema: Dict[str, str]  # {'action': 'categorical', 'value': 'continuous'}
    training_config: TrainingScheduleConfig
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    ensemble_config: Optional[Dict[str, Any]] = None


@dataclass
class PolicyConfig:
    """Decision-making policy."""
    name: str
    policy_type: str  # 'greedy', 'epsilon_greedy', 'ucb', 'expected_value', 'rl_policy'
    models: List[str]  # Model names to use
    aggregation: str  # 'vote', 'weighted_avg', 'max_confidence'
    risk_scaling: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioConfig:
    """Portfolio management rules."""
    initial_capital: float
    max_leverage: float
    position_limits: Dict[str, float]  # Per-symbol limits
    diversification_constraints: Dict[str, Any]
    cost_model: Dict[str, float]  # {'transaction': 0.001, 'slippage': 0.0005}
    risk_limits: Dict[str, float]  # {'max_drawdown': 0.2, 'max_var': 0.05}


@dataclass
class ExecutorConfig:
    """Execution configuration."""
    executor_type: str  # 'backtest', 'paper', 'live'
    venue: Optional[str] = None  # For live trading
    order_type: str = 'market'
    execution_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Backtest engine configuration."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    benchmark: Optional[str] = None
    metrics: List[str] = field(default_factory=lambda: ['sharpe', 'max_dd', 'total_return'])
    save_states: bool = True
    introspection_level: str = 'full'  # 'minimal', 'standard', 'full'


@dataclass
class SystemConfig:
    """Complete system specification - Single Source of Truth."""
    
    # System Identity
    name: str
    version: str
    description: str
    random_seed: int = 42
    
    # Data Pipeline
    data_sources: List[DataSourceConfig] = field(default_factory=list)
    
    # Feature Pipeline (DAG)
    weak_features: List[WeakFeatureConfig] = field(default_factory=list)
    strong_features: List[StrongFeatureConfig] = field(default_factory=list)
    
    # Model Pipeline
    models: List[ModelConfig] = field(default_factory=list)
    
    # Decision Pipeline
    policies: List[PolicyConfig] = field(default_factory=list)
    
    # Execution Pipeline
    portfolio: PortfolioConfig = None
    executor: ExecutorConfig = None
    
    # Orchestration
    backtest: BacktestConfig = None
    
    # Runtime
    parallel: bool = False
    cache_features: bool = True
    cache_dir: Optional[str] = None
    
    def validate(self) -> List[str]:
        """Validate configuration integrity."""
        errors = []
        
        # Check feature DAG is acyclic
        # Check all dependencies exist
        # Check training schedules are compatible
        # Check model inputs match feature outputs
        
        return errors


# ============================================================================
# DATA LAYER
# ============================================================================

@dataclass
class TimeSeriesData:
    """Unified time-indexed data container."""
    data: pd.DataFrame  # Columns: symbol_field (e.g., 'SPY_Close')
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        assert isinstance(self.data.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
        assert self.data.index.is_monotonic_increasing, "Index must be sorted"


class DataSource(ABC):
    """Abstract data source interface."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.name = config.name
        
    @abstractmethod
    def load(self, start: Optional[datetime] = None, 
             end: Optional[datetime] = None) -> TimeSeriesData:
        """Load data for the configured time range."""
        pass
    
    @abstractmethod
    def stream(self) -> TimeSeriesData:
        """Stream the next data point (for live trading)."""
        pass
    
    def validate_causality(self, data: TimeSeriesData, current_time: datetime) -> bool:
        """Ensure no future data leakage."""
        return data.data.index[-1] <= current_time


class DataRegistry:
    """Central registry for all data sources."""
    
    def __init__(self, configs: List[DataSourceConfig]):
        self.sources: Dict[str, DataSource] = {}
        self._build_sources(configs)
        
    def _build_sources(self, configs: List[DataSourceConfig]):
        """Instantiate data sources from configs."""
        for config in configs:
            source_class = self._get_source_class(config.source_type)
            self.sources[config.name] = source_class(config)
    
    def _get_source_class(self, source_type: str) -> type:
        """Factory for data sources."""
        # Would map to concrete implementations
        raise NotImplementedError
    
    def get(self, name: str) -> DataSource:
        return self.sources[name]
    
    def load_all(self, start: datetime, end: datetime) -> Dict[str, TimeSeriesData]:
        """Load all data sources."""
        return {name: source.load(start, end) 
                for name, source in self.sources.items()}


# ============================================================================
# FEATURE LAYER
# ============================================================================

@dataclass
class FeatureOutput:
    """Output from a feature computation."""
    data: pd.DataFrame  # Columns: feature_name
    dependencies_used: List[str]
    computation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Feature(ABC):
    """Base feature interface."""
    
    def __init__(self, config: Union[WeakFeatureConfig, StrongFeatureConfig]):
        self.config = config
        self.name = config.name
        self.dependencies = config.dependencies
        self.feature_type = FeatureType.WEAK if isinstance(config, WeakFeatureConfig) else FeatureType.STRONG
        
    @abstractmethod
    def compute(self, inputs: Dict[str, pd.DataFrame], 
                current_time: datetime) -> FeatureOutput:
        """Compute feature values. Must be causal."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, str]:
        """Return output schema: {feature_name: dtype}."""
        pass
    
    def validate_inputs(self, inputs: Dict[str, pd.DataFrame]) -> bool:
        """Ensure all dependencies are present."""
        return all(dep in inputs for dep in self.dependencies)


class WeakFeature(Feature):
    """Stateless feature transformation."""
    
    def __init__(self, config: WeakFeatureConfig):
        super().__init__(config)
        self.params = config.params
        self.window_size = config.window_size
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Pure transformation function."""
        pass
    
    def compute(self, inputs: Dict[str, pd.DataFrame], 
                current_time: datetime) -> FeatureOutput:
        """Compute with guaranteed causality."""
        assert self.validate_inputs(inputs)
        
        # Combine inputs
        combined = pd.concat([inputs[dep] for dep in self.dependencies], axis=1)
        
        # Filter to current_time (no lookahead)
        causal_data = combined[combined.index <= current_time]
        
        # Transform
        result = self.transform(causal_data)
        
        return FeatureOutput(
            data=result,
            dependencies_used=self.dependencies,
            computation_time=0.0,  # Would track actual time
            metadata={'window_size': self.window_size}
        )


class StrongFeature(Feature):
    """Stateful/model-driven feature."""
    
    def __init__(self, config: StrongFeatureConfig):
        super().__init__(config)
        self.model_type = config.model_type
        self.training_schedule = config.training_schedule
        self.params = config.params
        self.model: Optional[Any] = None
        self.last_trained: Optional[datetime] = None
        
    @abstractmethod
    def fit(self, training_data: pd.DataFrame, labels: Optional[pd.Series] = None):
        """Train the internal model."""
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate feature values from model."""
        pass
    
    def compute(self, inputs: Dict[str, pd.DataFrame], 
                current_time: datetime) -> FeatureOutput:
        """Compute using internal model."""
        assert self.validate_inputs(inputs)
        
        combined = pd.concat([inputs[dep] for dep in self.dependencies], axis=1)
        causal_data = combined[combined.index <= current_time]
        
        if self.model is None:
            # Return default/null feature
            return FeatureOutput(
                data=pd.DataFrame(index=causal_data.index),
                dependencies_used=self.dependencies,
                computation_time=0.0,
                metadata={'trained': False}
            )
        
        result = self.predict(causal_data)
        
        return FeatureOutput(
            data=result,
            dependencies_used=self.dependencies,
            computation_time=0.0,
            metadata={'trained': True, 'last_trained': self.last_trained}
        )


class FeatureDAG:
    """Directed Acyclic Graph of features."""
    
    def __init__(self, weak_configs: List[WeakFeatureConfig],
                 strong_configs: List[StrongFeatureConfig]):
        self.features: Dict[str, Feature] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.execution_order: List[str] = []
        
        self._build_features(weak_configs, strong_configs)
        self._validate_dag()
        self._compute_execution_order()
    
    def _build_features(self, weak_configs, strong_configs):
        """Instantiate all features."""
        for config in weak_configs:
            feature_class = self._get_weak_feature_class(config.feature_type)
            self.features[config.name] = feature_class(config)
            self.dependency_graph[config.name] = config.dependencies
        
        for config in strong_configs:
            feature_class = self._get_strong_feature_class(config.feature_type)
            self.features[config.name] = feature_class(config)
            self.dependency_graph[config.name] = config.dependencies
    
    def _get_weak_feature_class(self, feature_type: str) -> type:
        """Factory for weak features."""
        # Would map to concrete implementations
        raise NotImplementedError
    
    def _get_strong_feature_class(self, feature_type: str) -> type:
        """Factory for strong features."""
        raise NotImplementedError
    
    def _validate_dag(self):
        """Ensure no cycles exist."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.dependency_graph:
            if node not in visited:
                if has_cycle(node):
                    raise ValueError(f"Cycle detected in feature DAG at {node}")
    
    def _compute_execution_order(self):
        """Topological sort for execution order."""
        in_degree = {node: 0 for node in self.dependency_graph}
        
        for deps in self.dependency_graph.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        queue = [node for node, degree in in_degree.items() if degree == 0]
        
        while queue:
            node = queue.pop(0)
            self.execution_order.append(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
    
    def compute_all(self, data_sources: Dict[str, pd.DataFrame],
                    current_time: datetime) -> Dict[str, FeatureOutput]:
        """Compute all features in dependency order."""
        results = {}
        available = {**data_sources}  # Start with raw data
        
        for feature_name in self.execution_order:
            feature = self.features[feature_name]
            output = feature.compute(available, current_time)
            results[feature_name] = output
            available[feature_name] = output.data
        
        return results
    
    def get_strong_features(self) -> List[StrongFeature]:
        """Get all strong features for training."""
        return [f for f in self.features.values() 
                if isinstance(f, StrongFeature)]


# ============================================================================
# TRAINING LAYER
# ============================================================================

@dataclass
class TrainingContext:
    """Context for a training run."""
    feature_data: pd.DataFrame
    labels: Optional[pd.Series]
    validation_data: Optional[pd.DataFrame]
    metadata: Dict[str, Any]
    current_time: datetime


class Trainer(ABC):
    """Abstract trainer interface."""
    
    def __init__(self, config: TrainingScheduleConfig):
        self.config = config
        self.mode = config.mode
        self.frequency = config.frequency
        self.min_samples = config.min_samples
        
    @abstractmethod
    def should_train(self, current_bar: int, context: TrainingContext) -> bool:
        """Determine if training should occur."""
        pass
    
    @abstractmethod
    def prepare_data(self, raw_data: pd.DataFrame, 
                     current_time: datetime) -> TrainingContext:
        """Prepare training data maintaining causality."""
        pass
    
    @abstractmethod
    def train(self, model: Any, context: TrainingContext) -> Any:
        """Execute training."""
        pass
    
    @abstractmethod
    def validate(self, model: Any, context: TrainingContext) -> Dict[str, float]:
        """Validate model performance."""
        pass


class SupervisedTrainer(Trainer):
    """Trainer for supervised learning tasks."""
    
    def should_train(self, current_bar: int, context: TrainingContext) -> bool:
        if len(context.feature_data) < self.min_samples:
            return False
        
        if current_bar < self.config.warm_up_bars:
            return False
        
        return current_bar % self.frequency == 0
    
    def prepare_data(self, raw_data: pd.DataFrame, 
                     current_time: datetime) -> TrainingContext:
        """Prepare supervised learning data."""
        # Would implement windowing logic
        raise NotImplementedError
    
    def train(self, model: Any, context: TrainingContext) -> Any:
        """Train supervised model."""
        raise NotImplementedError
    
    def validate(self, model: Any, context: TrainingContext) -> Dict[str, float]:
        """Compute validation metrics."""
        raise NotImplementedError


class RLTrainer(Trainer):
    """Trainer for reinforcement learning agents."""
    
    def __init__(self, config: TrainingScheduleConfig):
        super().__init__(config)
        self.experience_buffer = []
        self.episode_count = 0
    
    def should_train(self, current_bar: int, context: TrainingContext) -> bool:
        # RL trains more frequently, based on experience buffer
        return len(self.experience_buffer) >= self.config.min_samples
    
    def prepare_data(self, raw_data: pd.DataFrame, 
                     current_time: datetime) -> TrainingContext:
        """Prepare RL training batch."""
        raise NotImplementedError
    
    def train(self, model: Any, context: TrainingContext) -> Any:
        """Train RL agent."""
        raise NotImplementedError
    
    def validate(self, model: Any, context: TrainingContext) -> Dict[str, float]:
        """Compute RL metrics (returns, etc)."""
        raise NotImplementedError
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience tuple to buffer."""
        self.experience_buffer.append((state, action, reward, next_state, done))
        
        if len(self.experience_buffer) > 10000:  # Max buffer size
            self.experience_buffer.pop(0)


class TrainingOrchestrator:
    """Manages all training schedules."""
    
    def __init__(self, feature_dag: FeatureDAG, models: Dict[str, 'Model']):
        self.feature_dag = feature_dag
        self.models = models
        self.trainers: Dict[str, Trainer] = {}
        self.training_history: List[Dict[str, Any]] = []
        
        self._build_trainers()
    
    def _build_trainers(self):
        """Instantiate trainers for all trainable components."""
        # Build trainers for strong features
        for feature in self.feature_dag.get_strong_features():
            trainer_class = self._get_trainer_class(feature.training_schedule.mode)
            self.trainers[f"feature_{feature.name}"] = trainer_class(feature.training_schedule)
        
        # Build trainers for models
        for model_name, model in self.models.items():
            trainer_class = self._get_trainer_class(model.config.training_config.mode)
            self.trainers[f"model_{model_name}"] = trainer_class(model.config.training_config)
    
    def _get_trainer_class(self, mode: TrainingMode) -> type:
        """Factory for trainers."""
        mapping = {
            TrainingMode.SUPERVISED: SupervisedTrainer,
            TrainingMode.REINFORCEMENT: RLTrainer,
            # ... other modes
        }
        return mapping.get(mode, SupervisedTrainer)
    
    def step(self, current_bar: int, feature_data: pd.DataFrame, 
             current_time: datetime):
        """Check and execute training for current timestep."""
        for name, trainer in self.trainers.items():
            context = trainer.prepare_data(feature_data, current_time)
            
            if trainer.should_train(current_bar, context):
                component_type, component_name = name.split('_', 1)
                
                if component_type == 'feature':
                    component = self.feature_dag.features[component_name]
                elif component_type == 'model':
                    component = self.models[component_name]
                
                # Train
                updated_component = trainer.train(component, context)
                
                # Validate
                metrics = trainer.validate(updated_component, context)
                
                # Log
                self.training_history.append({
                    'time': current_time,
                    'bar': current_bar,
                    'component': name,
                    'metrics': metrics
                })
                
                print(f"Trained {name} at bar {current_bar}")


# ============================================================================
# MODEL LAYER
# ============================================================================

@dataclass
class ModelOutput:
    """Output from model inference."""
    predictions: pd.DataFrame  # Columns match output_schema
    confidence: Optional[pd.Series]
    metadata: Dict[str, Any]


class Model(ABC):
    """Abstract model interface."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name
        self.input_features = config.input_features
        self.output_schema = config.output_schema
        self.trained = False
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> ModelOutput:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def get_input_schema(self) -> List[str]:
        """Required input features."""
        return self.input_features
    
    def validate_inputs(self, X: pd.DataFrame) -> bool:
        """Ensure inputs match schema."""
        return all(feat in X.columns for feat in self.input_features)


class ModelRegistry:
    """Central registry for all models."""
    
    def __init__(self, configs: List[ModelConfig]):
        self.models: Dict[str, Model] = {}
        self._build_models(configs)
    
    def _build_models(self, configs: List[ModelConfig]):
        """Instantiate models from configs."""
        for config in configs:
            model_class = self._get_model_class(config.model_type)
            self.models[config.name] = model_class(config)
    
    def _get_model_class(self, model_type: str) -> type:
        """Factory for models."""
        # Would map to concrete implementations
        raise NotImplementedError
    
    def get(self, name: str) -> Model:
        return self.models[name]
    
    def predict_all(self, features: pd.DataFrame) -> Dict[str, ModelOutput]:
        """Run inference on all models."""
        return {name: model.predict(features[model.input_features])
                for name, model in self.models.items()
                if model.trained and model.validate_inputs(features)}


# ============================================================================
# POLICY LAYER
# ============================================================================

@dataclass
class PolicyDecision:
    """Output from policy."""
    actions: Dict[str, float]  # {symbol: position_size}
    confidence: float
    reasoning: Dict[str, Any]


class Policy(ABC):
    """Abstract policy interface."""
    
    def __init__(self, config: PolicyConfig):
        self.config = config
        self.name = config.name
        self.models = config.models
        
    @abstractmethod
    def decide(self, model_outputs: Dict[str, ModelOutput],
               current_state: Dict[str, Any]) -> PolicyDecision:
        """Make trading decision from model outputs."""
        pass
    
    @abstractmethod
    def scale_risk(self, raw_actions: Dict[str, float],
                   volatility: Dict[str, float]) -> Dict[str, float]:
        """Apply risk scaling."""
        pass


class PolicyRegistry:
    """Central registry for policies."""
    
    def __init__(self, configs: List[PolicyConfig], model_registry: ModelRegistry):
        self.policies: Dict[str, Policy] = {}
        self.model_registry = model_registry
        self._build_policies(configs)
    
    def _build_policies(self, configs: List[PolicyConfig]):
        """Instantiate policies."""
        for config in configs:
            policy_class = self._get_policy_class(config.policy_type)
            self.policies[config.name] = policy_class(config)
    
    def _get_policy_class(self, policy_type: str) -> type:
        """Factory for policies."""
        raise NotImplementedError
    
    def execute(self, policy_name: str, model_outputs: Dict[str, ModelOutput],
                current_state: Dict[str, Any]) -> PolicyDecision:
        """Execute a policy."""
        return self.policies[policy_name].decide(model_outputs, current_state)


# ============================================================================
# PORTFOLIO LAYER
# ============================================================================

@dataclass
class PortfolioState:
    """Current portfolio state."""
    cash: float
    positions: Dict[str, float]  # {symbol: shares}
    equity: float
    leverage: float
    margin_used: float


class Portfolio:
    """Portfolio manager."""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.state = PortfolioState(
            cash=config.initial_capital,
            positions={},
            equity=config.initial_capital,
            leverage=0.0,
            margin_used=0.0
        )
        self.history: List[PortfolioState] = []
        
    def validate_decision(self, decision: PolicyDecision,
                          prices: Dict[str, float]) -> bool:
        """Check if decision violates constraints."""
        # Check leverage limits
        # Check position limits
        # Check risk limits
        raise NotImplementedError
    
    def compute_costs(self, old_positions: Dict[str, float],
                      new_positions: Dict[str, float],
                      prices: Dict[str, float]) -> float:
        """Calculate transaction costs."""
        total_cost = 0.0
        
        for symbol in set(list(old_positions.keys()) + list(new_positions.keys())):
            old_pos = old_positions.get(symbol, 0.0)
            new_pos = new_positions.get(symbol, 0.0)
            
            if abs(new_pos - old_pos) > 1e-6:
                notional = abs(new_pos - old_pos) * prices[symbol]
                total_cost += notional * self.config.cost_model['transaction']
        
        return total_cost
    
    def update(self, decision: PolicyDecision, prices: Dict[str, float]):
        """Update portfolio based on decision."""
        if not self.validate_decision(decision, prices):
            return False
        
        costs = self.compute_costs(self.state.positions, decision.actions, prices)
        
        # Update positions and cash
        # Apply costs
        # Recompute equity, leverage, etc
        
        self.history.append(self.state)
        return True


# ============================================================================
# EXECUTION LAYER
# ============================================================================

class Executor(ABC):
    """Abstract executor interface."""
    
    def __init__(self, config: ExecutorConfig):
        self.config = config
        
    @abstractmethod
    def execute(self, decision: PolicyDecision, 
                current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Execute trading decision."""
        pass


class BacktestExecutor(Executor):
    """Backtest execution (immediate fills)."""
    
    def execute(self, decision: PolicyDecision, 
                current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Perfect execution for backtesting."""
        fills = {}
        
        for symbol, size in decision.actions.items():
            fills[symbol] = {
                'size': size,
                'price': current_prices[symbol],
                'timestamp': datetime.now()
            }
        
        return fills


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """
    The orchestrator. Controls the time loop and all state transitions.
    Enforces causality by construction.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # Build all components from config
        self.data_registry = DataRegistry(config.data_sources)
        self.feature_dag = FeatureDAG(config.weak_features, config.strong_features)
        self.model_registry = ModelRegistry(config.models)
        self.policy_registry = PolicyRegistry(config.policies, self.model_registry)
        self.portfolio = Portfolio(config.portfolio)
        self.executor = self._build_executor(config.executor)
        
        # Training orchestration
        self.training_orchestrator = TrainingOrchestrator(
            self.feature_dag, 
            self.model_registry.models
        )
        
        # State tracking
        self.current_bar = 0
        self.current_time: Optional[datetime] = None
        self.history: Dict[str, List] = {
            'equity': [],
            'positions': [],
            'decisions': [],
            'features': [],
            'training_events': []
        }
        
    def _build_executor(self, config: ExecutorConfig) -> Executor:
        """Factory for executors."""
        if config.executor_type == 'backtest':
            return BacktestExecutor(config)
        # Would handle 'paper', 'live', etc
        raise NotImplementedError(f"Executor type {config.executor_type} not implemented")
    
    def run(self) -> Dict[str, Any]:
        """Execute complete backtest."""
        print(f"Starting backtest: {self.config.name}")
        
        # Load all data
        start = pd.Timestamp(self.config.backtest.start_date) if self.config.backtest.start_date else None
        end = pd.Timestamp(self.config.backtest.end_date) if self.config.backtest.end_date else None
        
        data = self.data_registry.load_all(start, end)
        
        # Get time index
        time_index = next(iter(data.values())).data.index
        
        # Main time loop - THIS IS THE ONLY LOOP
        for t in time_index:
            self.current_time = t
            
            # Get data up to current time (causal)
            current_data = {name: ts.data[ts.data.index <= t] 
                           for name, ts in data.items()}
            
            # 1. FEATURE COMPUTATION
            features = self.feature_dag.compute_all(current_data, t)
            
            # Store for training
            feature_df = pd.concat([f.data for f in features.values()], axis=1)
            
            # 2. TRAINING CHECK & EXECUTION
            self.training_orchestrator.step(self.current_bar, feature_df, t)
            
            # 3. MODEL INFERENCE
            model_outputs = self.model_registry.predict_all(feature_df.loc[[t]])
            
            # 4. POLICY DECISION
            policy_name = self.config.policies[0].name  # Primary policy
            current_state = {
                'portfolio': self.portfolio.state,
                'time': t,
                'bar': self.current_bar
            }
            decision = self.policy_registry.execute(policy_name, model_outputs, current_state)
            
            # 5. PORTFOLIO UPDATE & EXECUTION
            current_prices = {symbol: current_data[self.config.data_sources[0].name].loc[t, f'{symbol}_Close']
                            for symbol in self.config.data_sources[0].symbols}
            
            self.portfolio.update(decision, current_prices)
            fills = self.executor.execute(decision, current_prices)
            
            # 6. RECORD HISTORY
            self.history['equity'].append(self.portfolio.state.equity)
            self.history['positions'].append(self.portfolio.state.positions.copy())
            self.history['decisions'].append(decision)
            
            if self.config.backtest.save_states:
                self.history['features'].append(feature_df.loc[[t]])
            
            self.current_bar += 1
        
        # Compute metrics
        results = self._compute_metrics()
        
        print(f"Backtest complete: {len(time_index)} bars")
        return results
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute performance metrics."""
        equity = pd.Series(self.history['equity'])
        returns = equity.pct_change().dropna()
        
        metrics = {
            'final_equity': equity.iloc[-1],
            'total_return': (equity.iloc[-1] / equity.iloc[0]) - 1,
            'sharpe': np.sqrt(252) * returns.mean() / (returns.std() + 1e-9),
            'max_drawdown': (equity / equity.expanding().max() - 1).min(),
            'win_rate': (returns > 0).mean(),
            'history': self.history,
            'training_history': self.training_orchestrator.training_history
        }
        
        return metrics


# ============================================================================
# SYSTEM BUILDER
# ============================================================================

class SystemBuilder:
    """
    Build complete trading system from config.
    This is the entry point for all systems.
    """
    
    @staticmethod
    def from_config(config: SystemConfig) -> BacktestEngine:
        """Construct entire system from config."""
        
        # Validate config
        errors = config.validate()
        if errors:
            raise ValueError(f"Config validation failed:\n" + "\n".join(errors))
        
        # Set random seed
        np.random.seed(config.random_seed)
        
        # Build engine
        engine = BacktestEngine(config)
        
        return engine
    
    @staticmethod
    def from_yaml(path: str) -> BacktestEngine:
        """Load config from YAML and build system."""
        # Would parse YAML into SystemConfig
        raise NotImplementedError
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> BacktestEngine:
        """Build from dictionary."""
        # Would convert dict to SystemConfig
        raise NotImplementedError


# ============================================================================
# INTROSPECTION & DEBUGGING
# ============================================================================

class SystemIntrospector:
    """Introspection and debugging tools."""
    
    def __init__(self, engine: BacktestEngine):
        self.engine = engine
    
    def visualize_dag(self) -> str:
        """Generate DOT graph of feature DAG."""
        lines = ["digraph FeatureDAG {"]
        
        for feature_name, deps in self.engine.feature_dag.dependency_graph.items():
            feature = self.engine.feature_dag.features[feature_name]
            color = "lightblue" if feature.feature_type == FeatureType.WEAK else "orange"
            lines.append(f'  "{feature_name}" [fillcolor={color}, style=filled];')
            
            for dep in deps:
                lines.append(f'  "{dep}" -> "{feature_name}";')
        
        lines.append("}")
        return "\n".join(lines)
    
    def get_training_timeline(self) -> pd.DataFrame:
        """Return timeline of all training events."""
        return pd.DataFrame(self.engine.training_orchestrator.training_history)
    
    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance for a model."""
        model = self.engine.model_registry.get(model_name)
        # Would extract feature importance
        raise NotImplementedError
    
    def explain_decision(self, bar: int) -> Dict[str, Any]:
        """Explain decision at specific bar."""
        decision = self.engine.history['decisions'][bar]
        features = self.engine.history['features'][bar] if self.engine.history['features'] else None
        
        return {
            'bar': bar,
            'time': self.engine.current_time,
            'decision': decision,
            'features': features,
            'reasoning': decision.reasoning
        }
    
    def replay(self, start_bar: int, end_bar: int):
        """Replay and debug specific section."""
        # Would re-run specific bars with detailed logging
        raise NotImplementedError


# ============================================================================
# EXAMPLE USAGE PATTERN
# ============================================================================

def main():
    # Define complete system via config
    config = SystemConfig(
        name="POMDP_Strategy_v1",
        version="1.0.0",
        description="Regime-aware multi-horizon trading",
        random_seed=42,
        
        data_sources=[
            DataSourceConfig(
                name="market_data",
                source_type="market",
                symbols=["SPY", "QQQ"],
                fields=["Open", "High", "Low", "Close", "Volume"],
                interval="1d",
                lookback_days=730,
                provider="yahoo"
            )
        ],
        
        weak_features=[
            WeakFeatureConfig(
                name="returns",
                feature_type="returns",
                dependencies=["market_data"],
                params={"lags": [1, 5, 21]}
            ),
            WeakFeatureConfig(
                name="volatility",
                feature_type="volatility",
                dependencies=["returns"],
                params={"windows": [10, 21, 63]}
            )
        ],
        
        strong_features=[
            StrongFeatureConfig(
                name="regime",
                feature_type="regime",
                dependencies=["returns", "volatility"],
                model_type="lgbm",
                training_schedule=TrainingScheduleConfig(
                    mode=TrainingMode.SUPERVISED,
                    frequency=21,
                    min_samples=150,
                    window_type=WindowType.EXPANDING,
                    validation_split=0.2
                )
            )
        ],
        
        models=[
            ModelConfig(
                name="probability_model",
                model_type="lgbm",
                input_features=["returns", "volatility", "regime"],
                output_schema={"prob_tp": "continuous", "prob_sl": "continuous"},
                training_config=TrainingScheduleConfig(
                    mode=TrainingMode.SUPERVISED,
                    frequency=21,
                    min_samples=200,
                    window_type=WindowType.ROLLING,
                    window_size=504
                )
            )
        ],
        
        policies=[
            PolicyConfig(
                name="expected_value",
                policy_type="expected_value",
                models=["probability_model"],
                aggregation="weighted_avg",
                risk_scaling={"method": "vol_targeting", "target": 0.02}
            )
        ],
        
        portfolio=PortfolioConfig(
            initial_capital=100000,
            max_leverage=1.0,
            position_limits={"SPY": 0.5, "QQQ": 0.5},
            diversification_constraints={},
            cost_model={"transaction": 0.001, "slippage": 0.0005},
            risk_limits={"max_drawdown": 0.2}
        ),
        
        executor=ExecutorConfig(
            executor_type="backtest"
        ),
        
        backtest=BacktestConfig(
            start_date="2022-01-01",
            end_date="2024-12-31",
            benchmark="SPY",
            metrics=["sharpe", "max_dd", "total_return"],
            save_states=True,
            introspection_level="full"
        )
    )

    # Build and run system
    engine = SystemBuilder.from_config(config)
    results = engine.run()

    # Introspect
    inspector = SystemIntrospector(engine)
    print(inspector.visualize_dag())
    print(inspector.get_training_timeline())

if __name__ == "__main__":
    main()