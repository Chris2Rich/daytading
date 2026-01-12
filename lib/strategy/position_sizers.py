"""
Position sizing strategies for portfolio allocation.
Determines how much capital to allocate to each trade based on signals.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from decimal import Decimal

from models import SignalOutput, TradingAction, ModelOutputType
from nautilus_trader.model.identifiers import InstrumentId


class BasePositionSizer(ABC):
    """
    Abstract base class for position sizing strategies.
    
    Position sizers determine the actual capital allocation for trades
    based on model signals and risk factors.
    """
    
    @abstractmethod
    def calculate_size(self, signal: SignalOutput, current_position: float,
                      portfolio_value: float, instrument_id: InstrumentId) -> float:
        """
        Calculate target position size.
        
        Args:
            signal: Model's trading signal with risk factor
            current_position: Current position size (+ for long, - for short, 0 for flat)
            portfolio_value: Total portfolio value
            instrument_id: Instrument being traded
            
        Returns:
            Target position size in units of the instrument
        """
        pass


class FixedSizer(BasePositionSizer):
    """
    Fixed position size for all trades.
    Simplest sizing strategy - always use the same position size.
    """
    
    def __init__(self, fixed_size: float = 100.0):
        """
        Initialize fixed sizer.
        
        Args:
            fixed_size: Fixed number of units to trade
        """
        self.fixed_size = fixed_size
    
    def calculate_size(self, signal: SignalOutput, current_position: float,
                      portfolio_value: float, instrument_id: InstrumentId) -> float:
        """Calculate fixed position size."""
        if signal.output_type == ModelOutputType.ACTION:
            if signal.action == TradingAction.BUY:
                return self.fixed_size
            elif signal.action == TradingAction.SELL:
                return -self.fixed_size
            elif signal.action == TradingAction.FLATTEN:
                return 0.0
            else:  # HOLD
                return current_position
        
        # For other output types, maintain current position
        return current_position


class FixedFractionalSizer(BasePositionSizer):
    """
    Size positions as a fixed fraction of portfolio value.
    Classic position sizing approach used in many systematic strategies.
    """
    
    def __init__(self, fraction: float = 0.02, use_risk_factor: bool = True):
        """
        Initialize fixed fractional sizer.
        
        Args:
            fraction: Fraction of portfolio to allocate (e.g., 0.02 = 2%)
            use_risk_factor: Whether to scale by signal's risk factor
        """
        self.fraction = fraction
        self.use_risk_factor = use_risk_factor
    
    def calculate_size(self, signal: SignalOutput, current_position: float,
                      portfolio_value: float, instrument_id: InstrumentId) -> float:
        """Calculate position size as fraction of portfolio."""
        # Calculate base allocation
        base_allocation = portfolio_value * self.fraction
        
        # Scale by risk factor if enabled
        if self.use_risk_factor:
            base_allocation *= signal.risk_factor
        
        # Determine direction based on signal type
        if signal.output_type == ModelOutputType.ACTION:
            if signal.action == TradingAction.BUY:
                # Convert dollar allocation to units (simplified - would need price)
                # For now, return dollar allocation (caller handles conversion)
                return base_allocation
            elif signal.action == TradingAction.SELL:
                return -base_allocation
            elif signal.action == TradingAction.FLATTEN:
                return 0.0
            else:  # HOLD
                return current_position
        
        elif signal.output_type == ModelOutputType.TARGET_PRICE:
            # Calculate position based on current vs target price
            # Implementation depends on current market price
            # Placeholder: maintain current position
            return current_position
        
        else:  # THRESHOLDS
            # Would implement threshold-based sizing logic
            return current_position


class KellyCriterionSizer(BasePositionSizer):
    """
    Kelly Criterion position sizing.
    Optimal sizing based on win probability and payoff ratio.
    
    Kelly Fraction = (p * b - q) / b
    where:
    - p = probability of winning
    - q = probability of losing (1 - p)
    - b = ratio of amount won to amount bet
    """
    
    def __init__(self, max_kelly_fraction: float = 0.25,
                 win_rate: float = 0.55, win_loss_ratio: float = 1.5):
        """
        Initialize Kelly Criterion sizer.
        
        Args:
            max_kelly_fraction: Maximum Kelly fraction to use (for safety)
            win_rate: Historical win rate (or use signal confidence)
            win_loss_ratio: Average win / average loss ratio
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.win_rate = win_rate
        self.win_loss_ratio = win_loss_ratio
    
    def calculate_size(self, signal: SignalOutput, current_position: float,
                      portfolio_value: float, instrument_id: InstrumentId) -> float:
        """Calculate Kelly optimal position size."""
        # Use signal's confidence as win probability if available
        p = signal.confidence if signal.confidence else self.win_rate
        q = 1 - p
        b = self.win_loss_ratio
        
        # Calculate Kelly fraction
        kelly_fraction = (p * b - q) / b
        
        # Apply maximum limit
        kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
        
        # Ensure non-negative
        kelly_fraction = max(kelly_fraction, 0.0)
        
        # Scale by risk factor
        kelly_fraction *= signal.risk_factor
        
        # Calculate position size
        position_value = portfolio_value * kelly_fraction
        
        # Determine direction
        if signal.output_type == ModelOutputType.ACTION:
            if signal.action == TradingAction.BUY:
                return position_value
            elif signal.action == TradingAction.SELL:
                return -position_value
            elif signal.action == TradingAction.FLATTEN:
                return 0.0
            else:  # HOLD
                return current_position
        
        return current_position


class RiskParitySizer(BasePositionSizer):
    """
    Risk parity position sizing.
    Allocates capital such that each position contributes equal risk.
    """
    
    def __init__(self, target_volatility: float = 0.15,
                 lookback_period: int = 20):
        """
        Initialize risk parity sizer.
        
        Args:
            target_volatility: Target portfolio volatility (annualized)
            lookback_period: Period for volatility calculation
        """
        self.target_volatility = target_volatility
        self.lookback_period = lookback_period
        self.volatility_cache: Dict[str, float] = {}
    
    def calculate_size(self, signal: SignalOutput, current_position: float,
                      portfolio_value: float, instrument_id: InstrumentId) -> float:
        """Calculate risk-adjusted position size."""
        # Get instrument volatility (would be calculated from historical data)
        # For now, use cached value or default
        instrument_vol = self.volatility_cache.get(str(instrument_id), 0.20)
        
        # Calculate position size to achieve target volatility
        # position_size = (target_vol / instrument_vol) * portfolio_value
        position_value = (self.target_volatility / instrument_vol) * portfolio_value
        
        # Scale by risk factor
        position_value *= signal.risk_factor
        
        # Determine direction
        if signal.output_type == ModelOutputType.ACTION:
            if signal.action == TradingAction.BUY:
                return position_value
            elif signal.action == TradingAction.SELL:
                return -position_value
            elif signal.action == TradingAction.FLATTEN:
                return 0.0
            else:  # HOLD
                return current_position
        
        return current_position
    
    def update_volatility(self, instrument_id: InstrumentId, volatility: float):
        """Update volatility estimate for an instrument."""
        self.volatility_cache[str(instrument_id)] = volatility


class VolatilityTargetingSizer(BasePositionSizer):
    """
    Volatility targeting position sizing.
    Scales positions inversely with realized volatility.
    """
    
    def __init__(self, target_volatility: float = 0.15,
                 max_leverage: float = 2.0):
        """
        Initialize volatility targeting sizer.
        
        Args:
            target_volatility: Target portfolio volatility
            max_leverage: Maximum leverage allowed
        """
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.realized_vol_cache: Dict[str, float] = {}
    
    def calculate_size(self, signal: SignalOutput, current_position: float,
                      portfolio_value: float, instrument_id: InstrumentId) -> float:
        """Calculate volatility-scaled position size."""
        # Get realized volatility
        realized_vol = self.realized_vol_cache.get(str(instrument_id), 0.20)
        
        # Calculate volatility scalar
        vol_scalar = self.target_volatility / realized_vol
        
        # Apply leverage limit
        vol_scalar = min(vol_scalar, self.max_leverage)
        
        # Base position (e.g., 100% of portfolio)
        base_position = portfolio_value
        
        # Scale position
        position_value = base_position * vol_scalar * signal.risk_factor
        
        # Determine direction
        if signal.output_type == ModelOutputType.ACTION:
            if signal.action == TradingAction.BUY:
                return position_value
            elif signal.action == TradingAction.SELL:
                return -position_value
            elif signal.action == TradingAction.FLATTEN:
                return 0.0
            else:  # HOLD
                return current_position
        
        return current_position
    
    def update_realized_volatility(self, instrument_id: InstrumentId, vol: float):
        """Update realized volatility estimate."""
        self.realized_vol_cache[str(instrument_id)] = vol


class MaxDrawdownSizer(BasePositionSizer):
    """
    Position sizing based on maximum drawdown constraint.
    Reduces position size as drawdown increases.
    """
    
    def __init__(self, max_drawdown: float = 0.20,
                 base_fraction: float = 0.10):
        """
        Initialize max drawdown sizer.
        
        Args:
            max_drawdown: Maximum acceptable drawdown (e.g., 0.20 = 20%)
            base_fraction: Base fraction of portfolio to allocate
        """
        self.max_drawdown = max_drawdown
        self.base_fraction = base_fraction
        self.peak_value: float = 0.0
    
    def calculate_size(self, signal: SignalOutput, current_position: float,
                      portfolio_value: float, instrument_id: InstrumentId) -> float:
        """Calculate drawdown-adjusted position size."""
        # Update peak value
        self.peak_value = max(self.peak_value, portfolio_value)
        
        # Calculate current drawdown
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        
        # Calculate drawdown scalar (reduce size as drawdown increases)
        if current_drawdown >= self.max_drawdown:
            drawdown_scalar = 0.0  # Stop trading at max drawdown
        else:
            drawdown_scalar = 1.0 - (current_drawdown / self.max_drawdown)
        
        # Calculate position size
        position_value = portfolio_value * self.base_fraction * drawdown_scalar
        position_value *= signal.risk_factor
        
        # Determine direction
        if signal.output_type == ModelOutputType.ACTION:
            if signal.action == TradingAction.BUY:
                return position_value
            elif signal.action == TradingAction.SELL:
                return -position_value
            elif signal.action == TradingAction.FLATTEN:
                return 0.0
            else:  # HOLD
                return current_position
        
        return current_position


class EnsembleSizer(BasePositionSizer):
    """
    Ensemble of multiple sizing strategies.
    Combines signals from different sizers using weighted average.
    """
    
    def __init__(self, sizers: list[tuple[BasePositionSizer, float]]):
        """
        Initialize ensemble sizer.
        
        Args:
            sizers: List of (sizer, weight) tuples
        """
        self.sizers = sizers
        
        # Normalize weights
        total_weight = sum(weight for _, weight in sizers)
        self.sizers = [(sizer, weight / total_weight) 
                       for sizer, weight in sizers]
    
    def calculate_size(self, signal: SignalOutput, current_position: float,
                      portfolio_value: float, instrument_id: InstrumentId) -> float:
        """Calculate ensemble position size."""
        weighted_size = 0.0
        
        for sizer, weight in self.sizers:
            size = sizer.calculate_size(signal, current_position, 
                                       portfolio_value, instrument_id)
            weighted_size += size * weight
        
        return weighted_size


# ============================================================================
# Utility Functions
# ============================================================================


def create_position_sizer(sizer_type: str, **kwargs) -> BasePositionSizer:
    """
    Factory function to create position sizers by name.
    
    Args:
        sizer_type: Type of sizer ("fixed", "kelly", "risk_parity", etc.)
        **kwargs: Sizer-specific initialization parameters
        
    Returns:
        Instantiated position sizer
    """
    sizer_map = {
        'fixed': FixedSizer,
        'fixed_fractional': FixedFractionalSizer,
        'kelly': KellyCriterionSizer,
        'risk_parity': RiskParitySizer,
        'volatility_targeting': VolatilityTargetingSizer,
        'max_drawdown': MaxDrawdownSizer,
    }
    
    if sizer_type not in sizer_map:
        raise ValueError(
            f"Unknown sizer type: {sizer_type}. "
            f"Available: {list(sizer_map.keys())}"
        )
    
    return sizer_map[sizer_type](**kwargs)