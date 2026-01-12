import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


def lagged_open(data: Dict[str, pd.DataFrame], 
                symbol: str,
                lag_periods: int = 1,
                **kwargs) -> pd.Series:
    """
    Calculate lagged open price for a given symbol.
    
    Args:
        data: Dictionary mapping symbol -> DataFrame with OHLCV data
        symbol: Symbol to calculate lagged open for
        lag_periods: Number of periods to lag (default: 1)
        **kwargs: Additional keyword arguments (e.g., regime_state)
        
    Returns:
        Series with lagged open prices, indexed by timestamp
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    if 'open' not in df.columns:
        raise ValueError(f"'open' column not found in data for {symbol}")
    
    # Calculate lagged values
    lagged = df['open'].shift(lag_periods)
    
    return lagged


def current_open(data: Dict[str, pd.DataFrame],
                 symbol: str,
                 **kwargs) -> pd.Series:
    """
    Get current open price for a given symbol.
    
    Args:
        data: Dictionary mapping symbol -> DataFrame with OHLCV data
        symbol: Symbol to get current open for
        **kwargs: Additional keyword arguments
        
    Returns:
        Series with current open prices, indexed by timestamp
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    if 'open' not in df.columns:
        raise ValueError(f"'open' column not found in data for {symbol}")
    
    return df['open']


def price_momentum(data: Dict[str, pd.DataFrame],
                   symbol: str,
                   periods: int = 10,
                   **kwargs) -> pd.Series:
    """
    Calculate price momentum (rate of change) over specified periods.
    
    Args:
        data: Dictionary mapping symbol -> DataFrame with OHLCV data
        symbol: Symbol to calculate momentum for
        periods: Number of periods for momentum calculation
        **kwargs: Additional keyword arguments
        
    Returns:
        Series with momentum values (percentage change)
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    if 'close' not in df.columns:
        raise ValueError(f"'close' column not found in data for {symbol}")
    
    # Calculate percentage change over periods
    momentum = df['close'].pct_change(periods=periods) * 100
    
    return momentum


def simple_moving_average(data: Dict[str, pd.DataFrame],
                          symbol: str,
                          window: int = 20,
                          price_type: str = 'close',
                          **kwargs) -> pd.Series:
    """
    Calculate simple moving average.
    
    Args:
        data: Dictionary mapping symbol -> DataFrame with OHLCV data
        symbol: Symbol to calculate SMA for
        window: Window size for moving average
        price_type: Price type to use ('open', 'high', 'low', 'close')
        **kwargs: Additional keyword arguments
        
    Returns:
        Series with SMA values
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    if price_type not in df.columns:
        raise ValueError(f"'{price_type}' column not found in data for {symbol}")
    
    sma = df[price_type].rolling(window=window).mean()
    
    return sma


def exponential_moving_average(data: Dict[str, pd.DataFrame],
                               symbol: str,
                               span: int = 20,
                               price_type: str = 'close',
                               **kwargs) -> pd.Series:
    """
    Calculate exponential moving average.
    
    Args:
        data: Dictionary mapping symbol -> DataFrame with OHLCV data
        symbol: Symbol to calculate EMA for
        span: Span for EMA calculation
        price_type: Price type to use ('open', 'high', 'low', 'close')
        **kwargs: Additional keyword arguments
        
    Returns:
        Series with EMA values
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    if price_type not in df.columns:
        raise ValueError(f"'{price_type}' column not found in data for {symbol}")
    
    ema = df[price_type].ewm(span=span, adjust=False).mean()
    
    return ema


def volatility(data: Dict[str, pd.DataFrame],
               symbol: str,
               window: int = 20,
               **kwargs) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        data: Dictionary mapping symbol -> DataFrame with OHLCV data
        symbol: Symbol to calculate volatility for
        window: Window size for volatility calculation
        **kwargs: Additional keyword arguments
        
    Returns:
        Series with volatility values
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    if 'close' not in df.columns:
        raise ValueError(f"'close' column not found in data for {symbol}")
    
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Calculate rolling standard deviation
    vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    return vol


def correlation(data: Dict[str, pd.DataFrame],
                symbols: List[str],
                window: int = 20,
                **kwargs) -> pd.DataFrame:
    """
    Calculate rolling correlation between multiple symbols.
    
    Args:
        data: Dictionary mapping symbol -> DataFrame with OHLCV data
        symbols: List of symbols to calculate correlation for (must be length 2)
        window: Window size for correlation calculation
        **kwargs: Additional keyword arguments
        
    Returns:
        DataFrame with correlation values
    """
    if len(symbols) != 2:
        raise ValueError("Correlation requires exactly 2 symbols")
    
    symbol1, symbol2 = symbols
    
    if symbol1 not in data or symbol2 not in data:
        raise ValueError(f"One or both symbols not found in data")
    
    # Get close prices for both symbols
    prices1 = data[symbol1]['close']
    prices2 = data[symbol2]['close']
    
    # Align the series on common dates
    combined = pd.DataFrame({
        symbol1: prices1,
        symbol2: prices2
    })
    
    # Calculate rolling correlation
    corr = combined[symbol1].rolling(window=window).corr(combined[symbol2])
    
    return corr


def relative_strength_index(data: Dict[str, pd.DataFrame],
                            symbol: str,
                            periods: int = 14,
                            **kwargs) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data: Dictionary mapping symbol -> DataFrame with OHLCV data
        symbol: Symbol to calculate RSI for
        periods: Number of periods for RSI calculation
        **kwargs: Additional keyword arguments
        
    Returns:
        Series with RSI values (0-100)
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    if 'close' not in df.columns:
        raise ValueError(f"'close' column not found in data for {symbol}")
    
    # Calculate price changes
    delta = df['close'].diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=periods).mean()
    avg_losses = losses.rolling(window=periods).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def bollinger_bands(data: Dict[str, pd.DataFrame],
                   symbol: str,
                   window: int = 20,
                   num_std: float = 2.0,
                   **kwargs) -> pd.DataFrame:
    """
    Calculate Bollinger Bands (upper, middle, lower).
    
    Args:
        data: Dictionary mapping symbol -> DataFrame with OHLCV data
        symbol: Symbol to calculate Bollinger Bands for
        window: Window size for moving average
        num_std: Number of standard deviations for bands
        **kwargs: Additional keyword arguments
        
    Returns:
        DataFrame with columns: bb_upper, bb_middle, bb_lower
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    if 'close' not in df.columns:
        raise ValueError(f"'close' column not found in data for {symbol}")
    
    # Calculate middle band (SMA)
    middle = df['close'].rolling(window=window).mean()
    
    # Calculate standard deviation
    std = df['close'].rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return pd.DataFrame({
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower
    })


def volume_ratio(data: Dict[str, pd.DataFrame],
                 symbol: str,
                 window: int = 20,
                 **kwargs) -> pd.Series:
    """
    Calculate volume ratio (current volume / average volume).
    
    Args:
        data: Dictionary mapping symbol -> DataFrame with OHLCV data
        symbol: Symbol to calculate volume ratio for
        window: Window size for average volume
        **kwargs: Additional keyword arguments
        
    Returns:
        Series with volume ratio values
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    if 'volume' not in df.columns:
        raise ValueError(f"'volume' column not found in data for {symbol}")
    
    # Calculate average volume
    avg_volume = df['volume'].rolling(window=window).mean()
    
    # Calculate ratio
    ratio = df['volume'] / avg_volume
    
    return ratio


def regime_aware_feature(data: Dict[str, pd.DataFrame],
                        symbol: str,
                        regime_state: Optional[Dict[str, Any]] = None,
                        **kwargs) -> pd.Series:
    """
    Example of a regime-aware feature that adjusts based on market regime.
    
    Args:
        data: Dictionary mapping symbol -> DataFrame with OHLCV data
        symbol: Symbol to calculate feature for
        regime_state: Optional regime state from regime detector
        **kwargs: Additional keyword arguments
        
    Returns:
        Series with regime-adjusted values
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    returns = df['close'].pct_change()
    
    # If regime state is provided, adjust the feature
    if regime_state and 'current_regime' in regime_state:
        regime = regime_state['current_regime']
        
        if regime == 'high_volatility':
            # In high vol regime, emphasize recent data more
            feature = returns.ewm(span=5).mean()
        elif regime == 'low_volatility':
            # In low vol regime, use longer averages
            feature = returns.ewm(span=20).mean()
        else:
            # Default behavior
            feature = returns.ewm(span=10).mean()
    else:
        # No regime state, use default
        feature = returns.ewm(span=10).mean()
    
    return feature