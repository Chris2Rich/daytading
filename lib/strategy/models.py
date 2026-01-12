"""
Trading Models Implementation

This file contains concrete implementations of trading models that inherit
from BaseModel defined in abstract.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import pickle
from abstract import BaseModel, ModelOutputType, ModelPrediction, TradeAction


class TrashModel(BaseModel):
    """
    Ultra-simple trading model that makes decisions based on lagged price comparison.
    
    Strategy:
        - BUY if lagged_open < current_open (price is rising)
        - SELL if lagged_open > current_open (price is falling)
        - HOLD otherwise (no change)
    
    This model requires the 'lagged_open' feature to be computed.
    
    Parameters:
        lag_periods: Number of periods to lag (default: 1)
        threshold: Minimum difference to trigger trade (default: 0.0)
    """
    
    def __init__(self, output_type: ModelOutputType, **parameters):
        """
        Initialize the trash model.
        
        Args:
            output_type: Must be ModelOutputType.ACTION
            **parameters: Model parameters
                - lag_periods: Number of periods to lag
                - threshold: Minimum price difference to trigger action
        """
        super().__init__(output_type, **parameters)
        
        if output_type != ModelOutputType.ACTION:
            raise ValueError("TrashModel only supports ACTION output type")
        
        self.lag_periods = parameters.get('lag_periods', 1)
        self.threshold = parameters.get('threshold', 0.0)
        self.risk_factor = parameters.get('risk_factor', 1.0)
        
    def train(self, features: pd.DataFrame, targets: pd.Series, 
              portfolio_state: Dict[str, Any]) -> None:
        """
        Train the model. For TrashModel, this is a no-op since it's rule-based.
        
        Args:
            features: DataFrame with computed features
            targets: Series with target values
            portfolio_state: Current portfolio allocation and state
        """
        # Validate that required features exist
        required_features = ['lagged_open', 'open']
        
        for feature in required_features:
            if feature not in features.columns:
                raise ValueError(f"Required feature '{feature}' not found in features DataFrame")
        
        # For this simple model, we don't actually train anything
        # Just mark as trained
        self.is_trained = True
        
        # Store some basic statistics for reference
        self._state = {
            'trained_on_samples': len(features),
            'feature_columns': list(features.columns),
            'lag_periods': self.lag_periods,
            'threshold': self.threshold
        }
    
    def predict(self, features: pd.DataFrame, 
                portfolio_state: Dict[str, Any]) -> ModelPrediction:
        """
        Generate prediction based on lagged open price comparison.
        
        Args:
            features: DataFrame with computed features (should have lagged_open and open)
            portfolio_state: Current portfolio allocation and state
            
        Returns:
            ModelPrediction with action and risk_factor
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Get the latest row of features
        if len(features) == 0:
            return ModelPrediction(
                symbol="",
                action=TradeAction.HOLD,
                risk_factor=0.0
            )
        
        latest = features.iloc[-1]
        
        # Extract required features
        lagged_open = latest.get('lagged_open', None)
        current_open = latest.get('open', None)
        
        if lagged_open is None or current_open is None:
            return ModelPrediction(
                symbol="",
                action=TradeAction.HOLD,
                risk_factor=0.0
            )
        
        # Handle NaN values
        if pd.isna(lagged_open) or pd.isna(current_open):
            return ModelPrediction(
                symbol="",
                action=TradeAction.HOLD,
                risk_factor=0.0
            )
        
        # Calculate price difference
        price_diff = current_open - lagged_open
        
        # Make decision based on price movement
        if abs(price_diff) < self.threshold:
            action = TradeAction.HOLD
        elif price_diff > 0:
            # Price is rising -> BUY
            action = TradeAction.BUY
        else:
            # Price is falling -> SELL
            action = TradeAction.SELL
        
        # Calculate confidence/risk factor based on magnitude of price change
        # Normalize by current price to get percentage change
        pct_change = abs(price_diff / lagged_open) if lagged_open != 0 else 0
        
        # Scale risk factor: larger moves = higher confidence
        # Cap at reasonable levels
        risk_factor = min(self.risk_factor * (1 + pct_change * 10), 2.0)
        
        return ModelPrediction(
            symbol="",
            action=action,
            risk_factor=risk_factor,
            confidence=pct_change
        )
    
    def save_state(self, path: str) -> None:
        """
        Save model state to disk.
        
        Args:
            path: File path to save state
        """
        state = {
            'is_trained': self.is_trained,
            'parameters': self.parameters,
            'lag_periods': self.lag_periods,
            'threshold': self.threshold,
            'risk_factor': self.risk_factor,
            'model_state': self._state
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, path: str) -> None:
        """
        Load model state from disk.
        
        Args:
            path: File path to load state from
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.is_trained = state['is_trained']
        self.parameters = state['parameters']
        self.lag_periods = state['lag_periods']
        self.threshold = state['threshold']
        self.risk_factor = state['risk_factor']
        self._state = state['model_state']


# Example of a slightly more complex model for reference
class MomentumModel(BaseModel):
    """
    Simple momentum-based model that uses multiple lagged features.
    
    This is a template for more sophisticated models.
    """
    
    def __init__(self, output_type: ModelOutputType, **parameters):
        super().__init__(output_type, **parameters)
        self.lookback_periods = parameters.get('lookback_periods', [1, 5, 10])
        self.weights = None
        
    def train(self, features: pd.DataFrame, targets: pd.Series, 
              portfolio_state: Dict[str, Any]) -> None:
        """Train using simple linear regression on momentum features"""
        # This is a placeholder - would implement actual training logic
        self.weights = np.ones(len(self.lookback_periods))
        self.is_trained = True
        
    def predict(self, features: pd.DataFrame, 
                portfolio_state: Dict[str, Any]) -> ModelPrediction:
        """Generate prediction based on weighted momentum"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Placeholder prediction logic
        return ModelPrediction(
            symbol="",
            action=TradeAction.HOLD,
            risk_factor=1.0
        )
    
    def save_state(self, path: str) -> None:
        """Save model weights"""
        state = {
            'weights': self.weights,
            'is_trained': self.is_trained,
            'parameters': self.parameters
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, path: str) -> None:
        """Load model weights"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.weights = state['weights']
        self.is_trained = state['is_trained']
        self.parameters = state['parameters']