"""
Complete example configuration for the NautilusTrader backtesting system.
This demonstrates how to set up a full trading strategy with all components.
"""

from datetime import datetime
from pathlib import Path

# Import all necessary components
from models import RandomForestModel, LSTMModel, XGBoostModel, ModelOutputType
from adapters import YFinanceAdapter, CSVAdapter, CCXTAdapter
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
# Configuration 2: Advanced LSTM with Kelly Sizing
# ============================================================================

ADVANCED_LSTM_CONFIG = {
    'traded_symbols': ['BTC/USD', 'ETH/USD'],
    'non_traded_timeseries': ['BTC/USDT', 'ETH/USDT'],  # For features
    
    'features': [
        # Price features
        (features.returns_feature, {'symbol': 'BTC/USD', 'period': 1}),
        (features.log_returns_feature, {'symbol': 'BTC/USD', 'period': 1}),
        (features.returns_feature, {'symbol': 'ETH/USD', 'period': 1}),
        
        # Volatility clustering
        (features.volatility_feature, {'symbol': 'BTC/USD', 'window': 24}),
        (features.volatility_feature, {'symbol': 'BTC/USD', 'window': 168}),
        
        # Technical indicators
        (features.rsi_feature, {'symbol': 'BTC/USD', 'period': 14}),
        (features.bollinger_bands_feature, {'symbol': 'BTC/USD', 'window': 20}),
        (features.atr_feature, {'symbol': 'BTC/USD', 'period': 14}),
        
        # Volume
        (features.volume_profile_feature, {'symbol': 'BTC/USD', 'window': 24}),
        
        # Lagged features (for LSTM)
        (features.lagged_features, {
            'symbol': 'BTC/USD',
            'feature': 'close',
            'lags': [1, 6, 12, 24]
        }),
        
        # Correlation with ETH
        (features.correlation_feature, {'symbols': ['BTC/USD', 'ETH/USD'], 'window': 24}),
    ],
    
    'model_type': LSTMModel,
    'model_output_type': ModelOutputType.TARGET_PRICE,
    'model_params': {
        'output_type': ModelOutputType.TARGET_PRICE,
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.3,
        'sequence_length': 24,  # 24-hour lookback
    },
    
    'backtest_start': datetime(2022, 1, 1),
    'backtest_end': datetime(2023, 12, 31),
    
    'starting_capital': 50000,
    'commission_rate': 0.002,  # 0.2% for crypto
    'max_leverage': 2.0,
    
    # Kelly Criterion sizing
    'position_sizer_type': KellyCriterionSizer,
    'position_sizer_params': {
        'max_kelly_fraction': 0.20,
        'win_rate': 0.55,
        'win_loss_ratio': 1.8,
    },
    
    # More frequent retraining for crypto
    'retrain_frequency': 168,  # Weekly (168 hours)
    'training_window_type': 'rolling',
    'training_window_size': 2016,  # 12 weeks
    
    'regime_detector': None,  # No regime detection
    
    # Crypto exchange data
    'data_adapters': {
        'BTC/USD': CCXTAdapter(exchange_id='binance', timeframe='1h'),
        'ETH/USD': CCXTAdapter(exchange_id='binance', timeframe='1h'),
        'BTC/USDT': CCXTAdapter(exchange_id='binance', timeframe='1h'),
        'ETH/USDT': CCXTAdapter(exchange_id='binance', timeframe='1h'),
    },
    
    'model_save_path': './models/lstm_crypto',
}


# ============================================================================
# Configuration 3: Multi-Asset with Ensemble Sizing
# ============================================================================

MULTI_ASSET_CONFIG = {
    'traded_symbols': [
        'SPY',   # S&P 500 ETF
        'QQQ',   # NASDAQ ETF
        'IWM',   # Russell 2000 ETF
        'GLD',   # Gold ETF
        'TLT',   # Treasury Bond ETF
    ],
    
    'non_traded_timeseries': ['^VIX', 'DXY'],  # VIX and Dollar Index
    
    'features': [
        # Returns for all assets
        *[(features.returns_feature, {'symbol': sym, 'period': 1}) 
          for sym in ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']],
        
        # Volatility
        *[(features.volatility_feature, {'symbol': sym, 'window': 20})
          for sym in ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']],
        
        # Momentum
        *[(features.momentum_feature, {'symbol': sym, 'period': 60})
          for sym in ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']],
        
        # Cross-sectional rank (relative strength)
        (features.cross_sectional_rank_feature, {
            'symbols': ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT'],
            'metric': 'returns',
            'period': 20
        }),
        
        # Technical indicators
        (features.macd_feature, {'symbol': 'SPY'}),
        (features.rsi_feature, {'symbol': 'SPY', 'period': 14}),
        (features.bollinger_bands_feature, {'symbol': 'SPY', 'window': 20}),
        
        # Market regime
        (features.regime_feature, {'symbol': 'SPY', 'window': 60}),
        
        # VIX (fear gauge)
        (features.returns_feature, {'symbol': '^VIX', 'period': 1}),
        (features.moving_average_feature, {'symbol': '^VIX', 'window': 20}),
    ],
    
    'model_type': XGBoostModel,
    'model_output_type': ModelOutputType.ACTION,
    'model_params': {
        'output_type': ModelOutputType.ACTION,
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
    },
    
    'backtest_start': datetime(2015, 1, 1),
    'backtest_end': datetime(2023, 12, 31),
    
    'starting_capital': 500000,
    'commission_rate': 0.0005,  # 0.05%
    'max_leverage': 1.5,
    
    # Ensemble of sizing strategies
    'position_sizer_type': EnsembleSizer,
    'position_sizer_params': {
        'sizers': [
            (FixedFractionalSizer(fraction=0.15, use_risk_factor=True), 0.4),
            (RiskParitySizer(target_volatility=0.12), 0.4),
            (KellyCriterionSizer(max_kelly_fraction=0.15), 0.2),
        ]
    },
    
    'retrain_frequency': 63,  # Quarterly
    'training_window_type': 'expanding',
    
    'regime_detector': volatility_regime_detector,
    'regime_check_frequency': 10,
    
    'data_adapters': {
        'SPY': YFinanceAdapter(interval='1d'),
        'QQQ': YFinanceAdapter(interval='1d'),
        'IWM': YFinanceAdapter(interval='1d'),
        'GLD': YFinanceAdapter(interval='1d'),
        'TLT': YFinanceAdapter(interval='1d'),
        '^VIX': YFinanceAdapter(interval='1d'),
        'DXY': YFinanceAdapter(interval='1d'),
    },
    
    'model_save_path': './models/multi_asset_xgb',
}


# ============================================================================
# Configuration 4: CSV Data with Custom Features
# ============================================================================

CSV_CUSTOM_CONFIG = {
    'traded_symbols': ['CUSTOM_STOCK_1', 'CUSTOM_STOCK_2'],
    'non_traded_timeseries': [],
    
    'features': [
        # Basic features
        (features.returns_feature, {'symbol': 'CUSTOM_STOCK_1', 'period': 1}),
        (features.volatility_feature, {'symbol': 'CUSTOM_STOCK_1', 'window': 20}),
        (features.momentum_feature, {'symbol': 'CUSTOM_STOCK_1', 'period': 20}),
        
        # Seasonality
        (features.seasonality_feature, {'symbol': 'CUSTOM_STOCK_1'}),
        
        # Volume
        (features.volume_profile_feature, {'symbol': 'CUSTOM_STOCK_1', 'window': 20}),
    ],
    
    'model_type': RandomForestModel,
    'model_output_type': ModelOutputType.ACTION,
    'model_params': {
        'output_type': ModelOutputType.ACTION,
        'n_estimators': 150,
        'max_depth': 12,
    },
    
    'backtest_start': datetime(2020, 1, 1),
    'backtest_end': datetime(2023, 12, 31),
    
    'starting_capital': 100000,
    'commission_rate': 0.001,
    'max_leverage': 1.0,
    
    'position_sizer_type': FixedFractionalSizer,
    'position_sizer_params': {
        'fraction': 0.20,
        'use_risk_factor': True,
    },
    
    'retrain_frequency': 126,  # Semi-annually
    'training_window_type': 'rolling',
    'training_window_size': 756,  # 3 years
    
    'regime_detector': None,
    
    # Load from CSV files
    'data_adapters': {
        'CUSTOM_STOCK_1': CSVAdapter(
            data_dir='./data/custom',
            timestamp_col='date',
            date_format='%Y-%m-%d'
        ),
        'CUSTOM_STOCK_2': CSVAdapter(
            data_dir='./data/custom',
            timestamp_col='date',
            date_format='%Y-%m-%d'
        ),
    },
    
    'model_save_path': './models/csv_custom',
}


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    from main import run_backtest
    
    # Choose which configuration to run
    config = SIMPLE_RF_CONFIG
    # config = ADVANCED_LSTM_CONFIG
    # config = MULTI_ASSET_CONFIG
    # config = CSV_CUSTOM_CONFIG
    
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