"""
Complete example configuration for the NautilusTrader backtesting system.
This demonstrates how to set up a full trading strategy with all components.
"""

from datetime import datetime
from pathlib import Path
from nautilus_trader.model.identifiers import Venue
from uuid import uuid4

# Import all necessary components
from main import run_backtest, set_global_seed
from models import RandomForestModel, ModelOutputType
from adapters import YFinanceAdapter
from position_sizers import (
    FixedFractionalSizer, KellyCriterionSizer, 
    RiskParitySizer, EnsembleSizer
)
import features

# ============================================================================
# Configuration 1: Simple Random Forest Strategy
# ============================================================================

SIMPLE_RF_CONFIG = {
    'seed': 42,
    "id": uuid4(),

    # Symbols to trade
    'traded_symbols': ['SPY'],
    
    # Additional data for features (not traded)
    'non_traded_timeseries': ['^VIX'],
    
    # Feature engineering pipeline
    'features': [
        # Returns features
        (features.returns_feature, {'symbol': 'SPY', 'period': 1}),
        
        # Volatility
        (features.volatility_feature, {'symbol': 'SPY', 'window': 20}),
        
        # Momentum
        (features.momentum_feature, {'symbol': 'SPY', 'period': 10}),

        (features.regime_feature, {'symbol': 'SPY', 'window': 10}),
        (features.regime_feature, {'symbol': 'SPY', 'window': 180}),

        (features.regime_feature, {'symbol': '^VIX', 'window': 30}),
    ],
    
    # Model configuration
    'model_type': RandomForestModel,
    'model_output_type': ModelOutputType.ACTION,
    'model_params': {
        'output_type': ModelOutputType.ACTION,
        'n_estimators': 100,
        'max_depth': 7,
        'min_samples_split': 12,
    },
    
    # Backtest period
    'backtest_start': datetime(2020, 1, 1),
    'backtest_end': datetime(2025, 12, 31),
    
    # Portfolio parameters
    'starting_capital': 100000,
    'commission_rate': 0.001,  # 0.1%
    'max_leverage': 1.0,  # No leverage
    
    # Position sizing
    'position_sizer_type': FixedFractionalSizer,
    'position_sizer_params': {
        'fraction': 0.15,  # 10% per position
        'use_risk_factor': True,
    },
    
    # Retraining schedule
    'retrain_frequency': 21,  # Retrain 2x a year
    'training_window_type': 'rolling',
    'training_window_size': 252,
    
    # Data sources
    'data_adapters': {
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
    # Get performance statistics

    # Get the account and positions
    account = engine.trader.generate_account_report(Venue("BACKTESTING"))
    positions = engine.trader.generate_positions_report()
    orders = engine.trader.generate_order_fills_report()