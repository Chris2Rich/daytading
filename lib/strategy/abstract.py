from AlgorithmImports import *
from typing import Dict, List, Callable, Any, Optional, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# ENUMS AND TYPE DEFINITIONS
# ============================================================================

class ModelOutputType(Enum):
    """Types of outputs a model can produce"""
    ACTION = "action"  # Buy, Sell, Hold, Flatten
    TARGET_PRICE = "target_price"
    THRESHOLD = "threshold"  # Entry/exit thresholds

class TradeAction(Enum):
    """Trading actions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    FLATTEN = "flatten"

class SizingMethod(Enum):
    """Position sizing methods"""
    FIXED = "fixed"
    KELLY = "kelly"
    FRACTIONAL_KELLY = "fractional_kelly"
    RISK_BASED = "risk_based"

class WindowType(Enum):
    """Training window types"""
    EXPANDING = "expanding"
    ROLLING = "rolling"

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ModelPrediction:
    """Output from a model prediction"""
    symbol: str
    action: Optional[TradeAction] = None
    target_price: Optional[float] = None
    entry_threshold: Optional[float] = None
    exit_threshold: Optional[float] = None
    risk_factor: float = 1.0  # Used for position sizing
    confidence: float = 1.0

@dataclass
class BacktestConfig:
    """
    Single source of truth configuration for the entire backtesting system
    """
    # Trading symbols
    traded_symbols: List[str]
    
    # Non-traded timeseries for features (e.g., VIX, economic indicators)
    non_traded_timeseries: List[str]
    
    # Feature definitions: List of (feature_function, parameters)
    # feature_function should be imported from features.py
    # e.g., [(correlation_feature, {'symbols': ['SPY', 'VIX'], 'window': 20})]
    feature_definitions: List[tuple[Callable, Dict[str, Any]]]
    
    # Model configuration
    model_class: Type['BaseModel']  # The model class to use
    model_output_type: ModelOutputType
    model_parameters: Dict[str, Any]  # Parameters passed to model constructor
    
    # Backtest timeframe
    start_date: datetime
    end_date: datetime
    
    # LEAN simulator parameters
    starting_capital: float = 100000.0
    max_leverage: float = 1.0
    commission_model: str = "fixed"  # or "percentage"
    commission_value: float = 1.0
    slippage_model: str = "constant"
    slippage_value: float = 0.0
    
    # Position sizing
    sizing_method: SizingMethod = SizingMethod.FIXED
    sizing_parameters: Dict[str, Any] = None  # e.g., {'fraction': 0.25} for kelly
    
    # Training configuration
    window_type: WindowType = WindowType.EXPANDING
    rolling_window_size: Optional[int] = None  # In bars, required if ROLLING
    initial_training_bars: int = 252  # Minimum bars before first prediction
    
    # Retraining configuration
    retrain_interval_bars: Optional[int] = None  # Retrain every N bars
    retrain_interval_days: Optional[int] = None  # Retrain every N days
    regime_detector: Optional[Callable] = None  # Callback for regime detection
    
    # Data adapter configuration
    data_adapters: Dict[str, Type['BaseLeanDataAdapter']] = None

# ============================================================================
# ABSTRACT BASE CLASSES
# ============================================================================

class BaseModel(ABC):
    """
    Abstract base class for all trading models.
    All models must implement this interface for consistent API.
    """
    
    def __init__(self, output_type: ModelOutputType, **parameters):
        """
        Initialize the model with its output type and parameters.
        
        Args:
            output_type: The type of output this model produces
            **parameters: Model-specific parameters (layers, estimators, etc.)
        """
        self.output_type = output_type
        self.parameters = parameters
        self.is_trained = False
        self._state = {}
    
    @abstractmethod
    def train(self, features: pd.DataFrame, targets: pd.Series, 
              portfolio_state: Dict[str, Any]) -> None:
        """
        Train the model on historical data.
        
        Args:
            features: DataFrame with computed features
            targets: Series with target values (returns, prices, etc.)
            portfolio_state: Current portfolio allocation and state
        """
        pass
    
    @abstractmethod
    def predict(self, features: pd.DataFrame, 
                portfolio_state: Dict[str, Any]) -> ModelPrediction:
        """
        Generate prediction for a single symbol.
        
        Args:
            features: DataFrame with computed features for prediction
            portfolio_state: Current portfolio allocation and state
            
        Returns:
            ModelPrediction object with action, risk_factor, etc.
        """
        pass
    
    @abstractmethod
    def save_state(self, path: str) -> None:
        """Save model weights and state to disk"""
        pass
    
    @abstractmethod
    def load_state(self, path: str) -> None:
        """Load model weights and state from disk"""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get internal model state"""
        return self._state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set internal model state"""
        self._state = state

class BaseLeanDataAdapter(PythonData):
    """
    Abstract base class for LEAN data adapters.
    Adapters must inherit from LEAN's PythonData and implement these methods.
    
    Each adapter represents a discrete data stream that will be automatically
    synchronized by LEAN based on timestamps.
    """
    
    @abstractmethod
    def GetSource(self, config, date, isLiveMode):
        """
        Return the source URL or data for LEAN to fetch.
        This method is called by LEAN to get data source.
        
        Args:
            config: SubscriptionDataConfig object
            date: Date for which to get data
            isLiveMode: Whether running in live or backtest mode
            
        Returns:
            SubscriptionDataSource object
        """
        pass
    
    @abstractmethod
    def Reader(self, config, line, date, isLiveMode):
        """
        Parse a line of data and return a BaseData object.
        
        Args:
            config: SubscriptionDataConfig object
            line: String line of data to parse
            date: Date of the data
            isLiveMode: Whether running in live or backtest mode
            
        Returns:
            Instance of this adapter class with populated data
        """
        pass

# ============================================================================
# POSITION SIZING CLASSES
# ============================================================================

class BasePositionSizer(ABC):
    """Abstract base class for position sizing strategies"""
    
    def __init__(self, **parameters):
        self.parameters = parameters
    
    @abstractmethod
    def calculate_size(self, symbol: str, risk_factor: float, 
                      portfolio_value: float, current_price: float,
                      portfolio_state: Dict[str, Any]) -> float:
        """
        Calculate position size for a symbol.
        
        Args:
            symbol: Trading symbol
            risk_factor: Risk factor from model (0-1+ scale)
            portfolio_value: Current total portfolio value
            current_price: Current price of the symbol
            portfolio_state: Current portfolio holdings
            
        Returns:
            Number of shares to trade (positive for buy, negative for sell)
        """
        pass

class FixedPositionSizer(BasePositionSizer):
    """Fixed fraction position sizing"""
    
    def calculate_size(self, symbol: str, risk_factor: float, 
                      portfolio_value: float, current_price: float,
                      portfolio_state: Dict[str, Any]) -> float:
        fraction = self.parameters.get('fraction', 0.1)
        target_value = portfolio_value * fraction * risk_factor
        return int(target_value / current_price)

class KellyPositionSizer(BasePositionSizer):
    """Kelly Criterion position sizing"""
    
    def calculate_size(self, symbol: str, risk_factor: float, 
                      portfolio_value: float, current_price: float,
                      portfolio_state: Dict[str, Any]) -> float:
        fraction = self.parameters.get('fraction', 1.0)  # Can be fractional
        # Kelly formula: f = (bp - q) / b
        # Simplified here - model's risk_factor encodes edge
        kelly_fraction = risk_factor * fraction
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        target_value = portfolio_value * kelly_fraction
        return int(target_value / current_price)

class RiskBasedPositionSizer(BasePositionSizer):
    """Risk-based position sizing (e.g., volatility-adjusted)"""
    
    def calculate_size(self, symbol: str, risk_factor: float, 
                      portfolio_value: float, current_price: float,
                      portfolio_state: Dict[str, Any]) -> float:
        risk_per_trade = self.parameters.get('risk_per_trade', 0.02)
        # Simplified - in practice would use ATR or volatility
        target_risk = portfolio_value * risk_per_trade * risk_factor
        # Assume 2% stop loss for sizing
        stop_loss_pct = 0.02
        position_value = target_risk / stop_loss_pct
        return int(position_value / current_price)

# ============================================================================
# FEATURE COMPUTATION
# ============================================================================

class FeatureEngine:
    """Handles feature computation from raw data"""
    
    def __init__(self, feature_definitions: List[tuple[Callable, Dict[str, Any]]]):
        """
        Initialize feature engine with feature definitions.
        
        Args:
            feature_definitions: List of (function, parameters) tuples
        """
        self.feature_definitions = feature_definitions
    
    def compute_features(self, data: Dict[str, pd.DataFrame],
                        regime_state: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Compute all features from raw data.
        
        Args:
            data: Dictionary mapping symbol -> DataFrame of OHLCV data
            regime_state: Optional regime detection state to inform features
            
        Returns:
            DataFrame with computed features, indexed by timestamp
        """
        features = {}
        
        for feature_func, params in self.feature_definitions:
            # Pass regime state to feature if it accepts it
            try:
                if regime_state is not None:
                    feature_result = feature_func(data, regime_state=regime_state, **params)
                else:
                    feature_result = feature_func(data, **params)
                
                # Handle both Series and DataFrame returns
                if isinstance(feature_result, pd.Series):
                    features[feature_func.__name__] = feature_result
                elif isinstance(feature_result, pd.DataFrame):
                    for col in feature_result.columns:
                        features[f"{feature_func.__name__}_{col}"] = feature_result[col]
            except Exception as e:
                raise RuntimeError(f"Error computing feature {feature_func.__name__}: {str(e)}")
        
        return pd.DataFrame(features)

# ============================================================================
# MAIN LEAN ALGORITHM
# ============================================================================

class ModularBacktestAlgorithm(QCAlgorithm):
    """
    Main LEAN algorithm that orchestrates the entire backtesting system.
    Driven entirely by BacktestConfig.
    """
    
    def __init__(self):
        super().__init__()
        self.config: Optional[BacktestConfig] = None
        self.model: Optional[BaseModel] = None
        self.position_sizer: Optional[BasePositionSizer] = None
        self.feature_engine: Optional[FeatureEngine] = None
        
        # Data storage
        self.historical_data: Dict[str, List] = {}
        self.bar_count = 0
        self.last_retrain_bar = 0
        self.last_retrain_date = None
        
        # Regime detection
        self.regime_state: Dict[str, Any] = {}
        self.regime_tick_counter = 0
        
    def set_config(self, config: BacktestConfig):
        """Set the configuration for this backtest"""
        self.config = config
        
    def Initialize(self):
        """Initialize the LEAN algorithm"""
        if self.config is None:
            raise ValueError("Config must be set before Initialize()")
        
        # Set backtest parameters
        self.SetStartDate(self.config.start_date)
        self.SetEndDate(self.config.end_date)
        self.SetCash(self.config.starting_capital)
        
        # Set account settings
        self.Settings.FreePortfolioValuePercentage = 1.0 / self.config.max_leverage
        
        # Initialize data storage for all symbols
        all_symbols = self.config.traded_symbols + self.config.non_traded_timeseries
        
        for symbol in all_symbols:
            # Add equity data with custom adapter if specified
            if self.config.data_adapters and symbol in self.config.data_adapters:
                adapter_class = self.config.data_adapters[symbol]
                self.AddData(adapter_class, symbol)
            else:
                # Default to equity data
                self.AddEquity(symbol, Resolution.Daily)
            
            self.historical_data[symbol] = []
        
        # Set commission and slippage models
        if self.config.commission_model == "fixed":
            self.SetSecurityInitializer(lambda x: x.SetFeeModel(
                ConstantFeeModel(self.config.commission_value)))
        elif self.config.commission_model == "percentage":
            self.SetSecurityInitializer(lambda x: x.SetFeeModel(
                ConstantFeeModel(self.config.commission_value, "percentage")))
        
        # Initialize model
        self.model = self.config.model_class(
            output_type=self.config.model_output_type,
            **self.config.model_parameters
        )
        
        # Initialize position sizer
        sizer_map = {
            SizingMethod.FIXED: FixedPositionSizer,
            SizingMethod.KELLY: KellyPositionSizer,
            SizingMethod.FRACTIONAL_KELLY: KellyPositionSizer,
            SizingMethod.RISK_BASED: RiskBasedPositionSizer,
        }
        sizer_class = sizer_map[self.config.sizing_method]
        self.position_sizer = sizer_class(**(self.config.sizing_parameters or {}))
        
        # Initialize feature engine
        self.feature_engine = FeatureEngine(self.config.feature_definitions)
        
        self.Debug(f"Algorithm initialized with {len(self.config.traded_symbols)} symbols")
    
    def OnData(self, data: Slice):
        """Called on each new data slice"""
        # Store historical data
        for symbol_str in self.historical_data.keys():
            symbol = Symbol.Create(symbol_str, SecurityType.Equity, Market.USA)
            if data.ContainsKey(symbol):
                bar = data[symbol]
                self.historical_data[symbol_str].append({
                    'time': bar.Time,
                    'open': bar.Open,
                    'high': bar.High,
                    'low': bar.Low,
                    'close': bar.Close,
                    'volume': bar.Volume
                })
        
        self.bar_count += 1
        self.regime_tick_counter += 1
        
        # Check if we need to detect regime change
        if self.config.regime_detector is not None and self.regime_tick_counter >= 1:
            self.regime_state = self.config.regime_detector(
                self._get_dataframes(), 
                self.regime_state
            )
            self.regime_tick_counter = 0
        
        # Check if we have enough data for initial training
        if self.bar_count < self.config.initial_training_bars:
            return
        
        # Check if we need to retrain
        should_retrain = False
        
        if self.config.retrain_interval_bars is not None:
            if (self.bar_count - self.last_retrain_bar) >= self.config.retrain_interval_bars:
                should_retrain = True
        
        if self.config.retrain_interval_days is not None:
            if self.last_retrain_date is None:
                should_retrain = True
            else:
                days_since = (self.Time - self.last_retrain_date).days
                if days_since >= self.config.retrain_interval_days:
                    should_retrain = True
        
        # Also retrain if regime changed (detected by regime_detector updating state)
        if self.config.regime_detector is not None and 'regime_changed' in self.regime_state:
            if self.regime_state['regime_changed']:
                should_retrain = True
                self.regime_state['regime_changed'] = False
        
        # Perform training if needed
        if should_retrain or not self.model.is_trained:
            self._train_model()
            self.last_retrain_bar = self.bar_count
            self.last_retrain_date = self.Time
        
        # Generate predictions and execute trades
        if self.model.is_trained:
            self._execute_trading_logic()
    
    def _get_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Convert stored historical data to DataFrames"""
        dfs = {}
        for symbol, data_list in self.historical_data.items():
            if len(data_list) > 0:
                dfs[symbol] = pd.DataFrame(data_list).set_index('time')
        return dfs
    
    def _get_training_data(self) -> Dict[str, pd.DataFrame]:
        """Get training data based on window type"""
        all_data = self._get_dataframes()
        
        if self.config.window_type == WindowType.EXPANDING:
            return all_data
        
        elif self.config.window_type == WindowType.ROLLING:
            if self.config.rolling_window_size is None:
                raise ValueError("rolling_window_size must be specified for ROLLING window")
            
            # Return only last N bars
            rolling_data = {}
            for symbol, df in all_data.items():
                rolling_data[symbol] = df.iloc[-self.config.rolling_window_size:]
            return rolling_data
        
        return all_data
    
    def _train_model(self):
        """Train the model on historical data"""
        self.Debug(f"Training model at bar {self.bar_count}")
        
        # Get training window data
        training_data = self._get_training_data()
        
        # Compute features
        features = self.feature_engine.compute_features(training_data, self.regime_state)
        
        # Create targets (simplified - returns for next period)
        # In practice, this should be more sophisticated
        targets = {}
        for symbol in self.config.traded_symbols:
            if symbol in training_data:
                df = training_data[symbol]
                targets[symbol] = df['close'].pct_change().shift(-1)
        
        # Combine targets into single series (simplified)
        target_series = pd.concat(targets.values()).dropna()
        
        # Align features and targets
        common_index = features.index.intersection(target_series.index)
        features = features.loc[common_index]
        target_series = target_series.loc[common_index]
        
        # Get portfolio state
        portfolio_state = self._get_portfolio_state()
        
        # Train model
        try:
            self.model.train(features, target_series, portfolio_state)
            self.Debug(f"Model trained successfully")
        except Exception as e:
            self.Error(f"Error training model: {str(e)}")
    
    def _execute_trading_logic(self):
        """Generate predictions and execute trades"""
        # Get current data
        current_data = self._get_dataframes()
        
        # Compute current features
        features = self.feature_engine.compute_features(current_data, self.regime_state)
        
        # Get portfolio state
        portfolio_state = self._get_portfolio_state()
        
        # Generate predictions for each traded symbol
        predictions: Dict[str, ModelPrediction] = {}
        
        for symbol_str in self.config.traded_symbols:
            try:
                # Get features for this symbol (last row)
                symbol_features = features.iloc[[-1]]
                
                # Generate prediction
                prediction = self.model.predict(symbol_features, portfolio_state)
                prediction.symbol = symbol_str
                predictions[symbol_str] = prediction
                
            except Exception as e:
                self.Error(f"Error predicting for {symbol_str}: {str(e)}")
                continue
        
        # Execute trades based on predictions
        self._execute_trades(predictions)
    
    def _execute_trades(self, predictions: Dict[str, ModelPrediction]):
        """Execute trades based on model predictions"""
        portfolio_value = self.Portfolio.TotalPortfolioValue
        
        for symbol_str, prediction in predictions.items():
            symbol = Symbol.Create(symbol_str, SecurityType.Equity, Market.USA)
            
            if not self.Securities.ContainsKey(symbol):
                continue
            
            current_price = self.Securities[symbol].Price
            current_holdings = self.Portfolio[symbol].Quantity
            
            # Determine action based on model output type
            if self.config.model_output_type == ModelOutputType.ACTION:
                if prediction.action == TradeAction.BUY:
                    target_quantity = self.position_sizer.calculate_size(
                        symbol_str, prediction.risk_factor, portfolio_value,
                        current_price, self._get_portfolio_state()
                    )
                    if target_quantity > current_holdings:
                        self.SetHoldings(symbol, (target_quantity - current_holdings) * current_price / portfolio_value)
                
                elif prediction.action == TradeAction.SELL:
                    target_quantity = -self.position_sizer.calculate_size(
                        symbol_str, prediction.risk_factor, portfolio_value,
                        current_price, self._get_portfolio_state()
                    )
                    if target_quantity < current_holdings:
                        self.SetHoldings(symbol, (target_quantity - current_holdings) * current_price / portfolio_value)
                
                elif prediction.action == TradeAction.FLATTEN:
                    self.Liquidate(symbol)
                
                # HOLD does nothing
            
            elif self.config.model_output_type == ModelOutputType.TARGET_PRICE:
                # Implement target price logic
                if prediction.target_price > current_price * 1.01:  # 1% threshold
                    target_quantity = self.position_sizer.calculate_size(
                        symbol_str, prediction.risk_factor, portfolio_value,
                        current_price, self._get_portfolio_state()
                    )
                    self.SetHoldings(symbol, target_quantity * current_price / portfolio_value)
                elif prediction.target_price < current_price * 0.99:
                    self.Liquidate(symbol)
    
    def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state as features for the model"""
        holdings = {}
        for kvp in self.Portfolio:
            symbol = kvp.Key
            holding = kvp.Value
            holdings[str(symbol)] = {
                'quantity': holding.Quantity,
                'value': holding.HoldingsValue,
                'price': holding.Price,
                'unrealized_pnl': holding.UnrealizedProfit
            }
        
        return {
            'cash': self.Portfolio.Cash,
            'total_value': self.Portfolio.TotalPortfolioValue,
            'holdings': holdings,
            'leverage': self.Portfolio.TotalAbsoluteHoldingsCost / self.Portfolio.TotalPortfolioValue
        }
    
    def OnEndOfAlgorithm(self):
        """Called at the end of the backtest"""
        self.Debug(f"Backtest completed. Final portfolio value: {self.Portfolio.TotalPortfolioValue}")
        
        # Save model state if needed
        if self.model is not None:
            try:
                self.model.save_state(f"model_state_{self.Time.strftime('%Y%m%d')}.pkl")
            except Exception as e:
                self.Error(f"Error saving model state: {str(e)}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def create_algorithm(config: BacktestConfig) -> ModularBacktestAlgorithm:
    """
    Factory function to create a configured algorithm instance.
    
    Args:
        config: BacktestConfig containing all parameters
        
    Returns:
        Configured ModularBacktestAlgorithm instance
    """
    algo = ModularBacktestAlgorithm()
    algo.set_config(config)
    return algo