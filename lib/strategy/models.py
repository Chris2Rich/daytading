"""
Abstract base classes and type definitions for ML models.
All models must implement this standardized API for pluggability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union
import numpy as np


class ModelOutputType(Enum):
    """
    Defines the type of output the model produces.
    This determines how predictions are interpreted by the trading system.
    """
    ACTION = "action"  # Outputs: Buy, Sell, Hold, Flatten
    TARGET_PRICE = "target_price"  # Outputs: target price for the asset
    THRESHOLDS = "thresholds"  # Outputs: entry/exit threshold levels


class TradingAction(Enum):
    """Discrete trading actions a model can recommend."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    FLATTEN = "flatten"  # Close all positions


@dataclass
class SignalOutput:
    """
    Standardized output from model prediction.
    Contains both the primary signal and risk assessment.
    """
    output_type: ModelOutputType
    
    # Primary signal (interpretation depends on output_type)
    action: Optional[TradingAction] = None  # For ACTION type
    target_price: Optional[float] = None  # For TARGET_PRICE type
    entry_threshold: Optional[float] = None  # For THRESHOLDS type
    exit_threshold: Optional[float] = None  # For THRESHOLDS type
    
    # Risk factor: [0, 1] where higher = more confident/aggressive
    # Used by position sizer to determine allocation
    risk_factor: float = 0.5
    
    # Optional metadata
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    Enforces standardized API for training and prediction.
    
    All concrete models must:
    1. Accept model-specific parameters in __init__
    2. Implement train() for model fitting
    3. Implement predict() for inference
    4. Handle their own internal state management
    """
    
    def __init__(self, output_type: ModelOutputType, **kwargs):
        """
        Initialize model with output type and model-specific parameters.
        
        Args:
            output_type: Type of output this model produces
            **kwargs: Model-specific parameters (e.g., n_estimators, layers)
        """
        self.output_type = output_type
        self.is_trained = False
        self.model_params = kwargs
        self._internal_state: Optional[Any] = None
    
    @abstractmethod
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
              metadata: Optional[Dict[str, Any]] = None):
        """
        Train the model on provided features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
               Includes raw data, computed features, and portfolio state
            y: Optional target labels/values. If None, model should extract
               targets from X or use unsupervised learning
            metadata: Optional metadata (e.g., current regime, timestamp)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, symbol: Optional[str] = None) -> SignalOutput:
        """
        Generate trading signal from current features.
        
        Args:
            X: Feature vector (1, n_features) for current state
            symbol: Optional symbol identifier for multi-asset models
            
        Returns:
            SignalOutput containing action/target and risk factor
        """
        pass
    
    @abstractmethod
    def save_state(self, path: str):
        """
        Save model state to disk for persistence.
        
        Args:
            path: File path to save model
        """
        pass
    
    @abstractmethod
    def load_state(self, path: str):
        """
        Load model state from disk.
        
        Args:
            path: File path to load model from
        """
        pass
    
    def update_internal_state(self, state: Any):
        """
        Update internal state (for recurrent/stateful models).
        
        Args:
            state: State object (model-specific format)
        """
        self._internal_state = state
    
    def get_internal_state(self) -> Optional[Any]:
        """Get current internal state."""
        return self._internal_state


# ============================================================================
# Example Concrete Implementations
# ============================================================================


class RandomForestModel(BaseModel):
    """
    Example: Random Forest classifier for action prediction.
    
    Model-specific parameters:
    - n_estimators: Number of trees
    - max_depth: Maximum tree depth
    - min_samples_split: Minimum samples to split node
    """
    
    def __init__(self, output_type: ModelOutputType = ModelOutputType.ACTION,
                 n_estimators: int = 100, max_depth: int = 10, 
                 min_samples_split: int = 2, **kwargs):
        super().__init__(output_type=output_type, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        
        # Lazy import to avoid hard dependency
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
    
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None,
              metadata: Optional[Dict[str, Any]] = None):
        """Train Random Forest on features."""
        if y is None:
            # Extract targets from features (e.g., future returns)
            # Simplified: use next-period return as target
            y = self._extract_targets(X)
        
        self.model.fit(X[:-1], y)  # Exclude last sample (no future target)
        self.is_trained = True
    
    def predict(self, X: np.ndarray, symbol: Optional[str] = None) -> SignalOutput:
        """Predict trading action using Random Forest."""
        if not self.is_trained:
            return SignalOutput(
                output_type=self.output_type,
                action=TradingAction.HOLD,
                risk_factor=0.0
            )
        
        # Get prediction and probability
        pred_class = self.model.predict(X)[0]
        pred_proba = self.model.predict_proba(X)[0]
        
        # Map prediction to action
        action_map = {0: TradingAction.SELL, 1: TradingAction.HOLD, 2: TradingAction.BUY}
        action = action_map.get(pred_class, TradingAction.HOLD)
        
        # Use prediction confidence as risk factor
        confidence = np.max(pred_proba)
        risk_factor = confidence
        
        return SignalOutput(
            output_type=self.output_type,
            action=action,
            risk_factor=risk_factor,
            confidence=confidence
        )
    
    def save_state(self, path: str):
        """Save model using pickle."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_state(self, path: str):
        """Load model from pickle."""
        import pickle
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
    
    def _extract_targets(self, X: np.ndarray) -> np.ndarray:
        """
        Extract target labels from features.
        Simplified: classify future returns as -1 (sell), 0 (hold), 1 (buy)
        """
        # Assume last column is the price/return feature
        returns = np.diff(X[:, -1])
        
        # Classify into 3 classes based on threshold
        threshold = np.std(returns) * 0.5
        targets = np.where(returns > threshold, 2,
                          np.where(returns < -threshold, 0, 1))
        
        return targets