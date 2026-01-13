"""
Complete example configuration for the NautilusTrader backtesting system.
This demonstrates how to set up a full trading strategy with all components.
"""

from datetime import datetime
from pathlib import Path

# Import all necessary components
from models import RandomForestModel, ModelOutputType
from adapters import YFinanceAdapter
from position_sizers import (
    FixedFractionalSizer, KellyCriterionSizer, 
    RiskParitySizer, EnsembleSizer
)
import features


# ============================================================================
# Regime Detection Callback
# ============================================================================

def volatility_regime_detector(market_state: dict) -> str:
    """
    Detect market regime based on volatility levels.
    
    Args:
        market_state: Dict containing 'raw_data', 'features', 'bar_count'
        
    Returns:
        Regime identifier ('high_vol', 'normal', 'low_vol')
    """
    import numpy as np
    
    # Get SPY volatility from features
    feature_df = market_state['features']
    
    if 'volatility_SPY_20' not in feature_df.columns:
        return 'normal'
    
    # Get recent volatility
    recent_vol = feature_df['volatility_SPY_20'].iloc[-20:].mean()
    
    # Define regime thresholds
    if recent_vol > 0.25:
        return 'high_vol'
    elif recent_vol < 0.15:
        return 'low_vol'
    else:
        return 'normal'


# ============================================================================
# Configuration 1: Simple Random Forest Strategy
# ============================================================================

SIMPLE_RF_CONFIG = {
    # Symbols to trade
    'traded_symbols': ['AAPL', 'MSFT', 'GOOGL'],
    
    # Additional data for features (not traded)
    'non_traded_timeseries': ['SPY', '^VIX'],
    
    # Feature engineering pipeline
    'features': [
        # Returns features
        (features.returns_feature, {'symbol': 'AAPL', 'period': 1}),
        (features.returns_feature, {'symbol': 'AAPL', 'period': 5}),
        (features.returns_feature, {'symbol': 'MSFT', 'period': 1}),
        (features.returns_feature, {'symbol': 'GOOGL', 'period': 1}),
        
        # Volatility
        (features.volatility_feature, {'symbol': 'SPY', 'window': 20}),
        
        # Momentum
        (features.momentum_feature, {'symbol': 'AAPL', 'period': 20}),
        (features.momentum_feature, {'symbol': 'MSFT', 'period': 20}),
        
        # Technical indicators
        (features.rsi_feature, {'symbol': 'AAPL', 'period': 14}),
        (features.macd_feature, {'symbol': 'AAPL'}),
        
        # Cross-sectional
        (features.correlation_feature, {'symbols': ['AAPL', 'SPY']}),
    ],
    
    # Model configuration
    'model_type': RandomForestModel,
    'model_output_type': ModelOutputType.ACTION,
    'model_params': {
        'output_type': ModelOutputType.ACTION,
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
    },
    
    # Backtest period
    'backtest_start': datetime(2020, 1, 1),
    'backtest_end': datetime(2023, 12, 31),
    
    # Portfolio parameters
    'starting_capital': 100000,
    'commission_rate': 0.001,  # 0.1%
    'max_leverage': 1.0,  # No leverage
    
    # Position sizing
    'position_sizer_type': FixedFractionalSizer,
    'position_sizer_params': {
        'fraction': 0.10,  # 10% per position
        'use_risk_factor': True,
    },
    
    # Retraining schedule
    'retrain_frequency': 252,  # Retrain yearly (252 trading days)
    'training_window_type': 'expanding',
    
    # Regime detection
    'regime_detector': volatility_regime_detector,
    'regime_check_frequency': 20,  # Check every 20 bars
    
    # Data sources
    'data_adapters': {
        'AAPL': YFinanceAdapter(interval='1d'),
        'MSFT': YFinanceAdapter(interval='1d'),
        'GOOGL': YFinanceAdapter(interval='1d'),
        'SPY': YFinanceAdapter(interval='1d'),
        '^VIX': YFinanceAdapter(interval='1d'),
    },
    
    # Model persistence
    'model_save_path': './models/simple_rf',
}

# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    from main import run_backtest
    
    # Choose which configuration to run
    config = SIMPLE_RF_CONFIG
    
    print("Starting backtest with configuration...")
    print(f"Model: {config['model_type'].__name__}")
    print(f"Symbols: {config['traded_symbols']}")
    print(f"Period: {config['backtest_start']} to {config['backtest_end']}")
    print(f"Starting Capital: ${config['starting_capital']:,.2f}")
    
    # Run backtest
    engine = run_backtest(config)
    
    print("\n" + "="*80)
    print("Backtest Complete!")
    print("="*80)
    
    # Results would be extracted from engine
    # engine.get_result() or similar depending on NautilusTrader API