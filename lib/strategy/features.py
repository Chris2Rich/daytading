"""
Feature engineering functions for ML trading models.
Each function takes raw timeseries data and returns computed features.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def returns_feature(data: Dict[str, pd.DataFrame], symbol: str, 
                   period: int = 1) -> pd.Series:
    """
    Calculate returns for a given period.
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to calculate returns for
        period: Number of periods for return calculation
        
    Returns:
        Series of returns with timestamp index
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    returns = df['close'].pct_change(periods=period)
    return returns.rename(f'returns_{symbol}_{period}')


def log_returns_feature(data: Dict[str, pd.DataFrame], symbol: str,
                        period: int = 1) -> pd.Series:
    """
    Calculate log returns.
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to calculate log returns for
        period: Number of periods
        
    Returns:
        Series of log returns
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    log_returns = np.log(df['close'] / df['close'].shift(period))
    return log_returns.rename(f'log_returns_{symbol}_{period}')


def volatility_feature(data: Dict[str, pd.DataFrame], symbol: str,
                      window: int = 20, annualize: bool = True) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to calculate volatility for
        window: Rolling window size
        annualize: Whether to annualize volatility (assumes daily data)
        
    Returns:
        Series of volatility values
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=window).std()
    
    if annualize:
        volatility = volatility * np.sqrt(252)  # Assume daily data
    
    return volatility.rename(f'volatility_{symbol}_{window}')


def correlation_feature(data: Dict[str, pd.DataFrame], symbols: List[str],
                       window: int = 20) -> pd.DataFrame:
    """
    Calculate rolling correlation between symbols.
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbols: List of symbols to compute pairwise correlations
        window: Rolling window size
        
    Returns:
        DataFrame with correlation features
    """
    if len(symbols) < 2:
        raise ValueError("Need at least 2 symbols for correlation")
    
    # Get returns for all symbols
    returns_dict = {}
    for symbol in symbols:
        if symbol not in data:
            continue
        returns_dict[symbol] = data[symbol]['close'].pct_change()
    
    # Combine into single dataframe
    returns_df = pd.DataFrame(returns_dict)
    
    # Calculate rolling correlations
    corr_features = pd.DataFrame(index=returns_df.index)
    
    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i+1:]:
            if sym1 in returns_dict and sym2 in returns_dict:
                corr = returns_df[sym1].rolling(window=window).corr(returns_df[sym2])
                corr_features[f'corr_{sym1}_{sym2}_{window}'] = corr
    
    return corr_features


def momentum_feature(data: Dict[str, pd.DataFrame], symbol: str,
                    period: int = 20) -> pd.Series:
    """
    Calculate momentum (cumulative return over period).
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to calculate momentum for
        period: Lookback period
        
    Returns:
        Series of momentum values
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    momentum = (df['close'] / df['close'].shift(period)) - 1
    return momentum.rename(f'momentum_{symbol}_{period}')


def moving_average_feature(data: Dict[str, pd.DataFrame], symbol: str,
                          window: int = 20, ma_type: str = 'sma') -> pd.Series:
    """
    Calculate moving average.
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to calculate MA for
        window: Window size
        ma_type: Type of MA ('sma' or 'ema')
        
    Returns:
        Series of moving average values
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    if ma_type == 'sma':
        ma = df['close'].rolling(window=window).mean()
    elif ma_type == 'ema':
        ma = df['close'].ewm(span=window, adjust=False).mean()
    else:
        raise ValueError(f"Unknown MA type: {ma_type}")
    
    return ma.rename(f'{ma_type}_{symbol}_{window}')


def rsi_feature(data: Dict[str, pd.DataFrame], symbol: str,
               period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to calculate RSI for
        period: RSI period
        
    Returns:
        Series of RSI values (0-100)
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    delta = df['close'].diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.rename(f'rsi_{symbol}_{period}')


def bollinger_bands_feature(data: Dict[str, pd.DataFrame], symbol: str,
                           window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands features.
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to calculate bands for
        window: Window size for moving average
        num_std: Number of standard deviations for bands
        
    Returns:
        DataFrame with upper band, lower band, and bandwidth
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    # Calculate middle band (SMA)
    middle = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    
    # Calculate bands
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    # Calculate bandwidth
    bandwidth = (upper - lower) / middle
    
    # Calculate %B (position within bands)
    percent_b = (df['close'] - lower) / (upper - lower)
    
    result = pd.DataFrame({
        f'bb_upper_{symbol}_{window}': upper,
        f'bb_lower_{symbol}_{window}': lower,
        f'bb_bandwidth_{symbol}_{window}': bandwidth,
        f'bb_percent_{symbol}_{window}': percent_b
    })
    
    return result


def macd_feature(data: Dict[str, pd.DataFrame], symbol: str,
                fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to calculate MACD for
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        DataFrame with MACD, signal line, and histogram
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    # Calculate EMAs
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    result = pd.DataFrame({
        f'macd_{symbol}': macd_line,
        f'macd_signal_{symbol}': signal_line,
        f'macd_hist_{symbol}': histogram
    })
    
    return result


def atr_feature(data: Dict[str, pd.DataFrame], symbol: str,
               period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to calculate ATR for
        period: ATR period
        
    Returns:
        Series of ATR values
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR (EMA of True Range)
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    return atr.rename(f'atr_{symbol}_{period}')


def volume_profile_feature(data: Dict[str, pd.DataFrame], symbol: str,
                          window: int = 20) -> pd.DataFrame:
    """
    Calculate volume-based features.
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to calculate volume features for
        window: Rolling window size
        
    Returns:
        DataFrame with volume features
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    # Volume moving average
    vol_ma = df['volume'].rolling(window=window).mean()
    
    # Relative volume
    rel_vol = df['volume'] / vol_ma
    
    # On-balance volume
    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    
    # Volume-weighted average price
    vwap = (df['close'] * df['volume']).rolling(window=window).sum() / \
           df['volume'].rolling(window=window).sum()
    
    result = pd.DataFrame({
        f'vol_ma_{symbol}_{window}': vol_ma,
        f'rel_vol_{symbol}_{window}': rel_vol,
        f'obv_{symbol}': obv,
        f'vwap_{symbol}_{window}': vwap
    })
    
    return result


def cross_sectional_rank_feature(data: Dict[str, pd.DataFrame],
                                 symbols: List[str],
                                 metric: str = 'returns',
                                 period: int = 20) -> pd.DataFrame:
    """
    Calculate cross-sectional rank of symbols.
    Useful for relative strength/momentum strategies.
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbols: List of symbols to rank
        metric: Metric to rank on ('returns', 'volatility', etc.)
        period: Period for metric calculation
        
    Returns:
        DataFrame with rank features for each symbol
    """
    # Calculate metric for each symbol
    metric_dict = {}
    
    for symbol in symbols:
        if symbol not in data:
            continue
        
        df = data[symbol]
        
        if metric == 'returns':
            metric_dict[symbol] = df['close'].pct_change(periods=period)
        elif metric == 'volatility':
            metric_dict[symbol] = df['close'].pct_change().rolling(window=period).std()
        elif metric == 'volume':
            metric_dict[symbol] = df['volume'].rolling(window=period).mean()
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # Combine into dataframe and rank
    metric_df = pd.DataFrame(metric_dict)
    rank_df = metric_df.rank(axis=1, pct=True)  # Percentile rank
    
    # Rename columns
    rank_df.columns = [f'rank_{metric}_{sym}_{period}' for sym in rank_df.columns]
    
    return rank_df


def regime_feature(data: Dict[str, pd.DataFrame], symbol: str,
                  window: int = 60) -> pd.Series:
    """
    Detect market regime (trending vs mean-reverting).
    Uses Hurst exponent approximation.
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to detect regime for
        window: Window for regime detection
        
    Returns:
        Series indicating regime (>0.5 = trending, <0.5 = mean-reverting)
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    prices = df['close']
    
    def hurst_approximation(ts):
        """Approximate Hurst exponent."""
        if len(ts) < 2:
            return 0.5
        
        # Calculate range
        mean_centered = ts - ts.mean()
        cumsum = mean_centered.cumsum()
        R = cumsum.max() - cumsum.min()
        
        # Calculate standard deviation
        S = ts.std()
        
        if S == 0:
            return 0.5
        
        # R/S ratio
        rs = R / S
        
        # Approximate Hurst
        if rs > 0:
            return np.log(rs) / np.log(len(ts))
        return 0.5
    
    # Calculate rolling Hurst
    hurst = prices.rolling(window=window).apply(hurst_approximation, raw=True)
    
    return hurst.rename(f'regime_{symbol}_{window}')


def seasonality_feature(data: Dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
    """
    Extract seasonality features (day of week, month, etc.).
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to extract seasonality for
        
    Returns:
        DataFrame with seasonality features
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    result = pd.DataFrame(index=df.index)
    
    # Day of week (0 = Monday, 6 = Sunday)
    result[f'dow_{symbol}'] = df.index.dayofweek
    
    # Month (1-12)
    result[f'month_{symbol}'] = df.index.month
    
    # Day of month
    result[f'dom_{symbol}'] = df.index.day
    
    # Quarter
    result[f'quarter_{symbol}'] = df.index.quarter
    
    # Sine/cosine encoding for cyclical features
    result[f'dow_sin_{symbol}'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    result[f'dow_cos_{symbol}'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    result[f'month_sin_{symbol}'] = np.sin(2 * np.pi * df.index.month / 12)
    result[f'month_cos_{symbol}'] = np.cos(2 * np.pi * df.index.month / 12)
    
    return result


def lagged_features(data: Dict[str, pd.DataFrame], symbol: str,
                   feature: str = 'close', lags: List[int] = [1, 2, 5, 10]) -> pd.DataFrame:
    """
    Create lagged versions of a feature.
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to create lags for
        feature: Feature to lag ('close', 'volume', etc.)
        lags: List of lag periods
        
    Returns:
        DataFrame with lagged features
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    if feature not in df.columns:
        raise ValueError(f"Feature {feature} not found in data")
    
    result = pd.DataFrame(index=df.index)
    
    for lag in lags:
        result[f'{feature}_lag{lag}_{symbol}'] = df[feature].shift(lag)
    
    return result


def adx_feature(data: Dict[str, pd.DataFrame], symbol: str, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX).
    Measures trend strength regardless of direction.
    ADX > 25 indicates a strong trend.
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to calculate ADX for
        period: ADX period
        
    Returns:
        DataFrame with ADX, +DI, and -DI values
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using EMA
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return pd.DataFrame({
        f'adx_{symbol}_{period}': adx,
        f'plus_di_{symbol}_{period}': plus_di,
        f'minus_di_{symbol}_{period}': minus_di
    })


def ema_feature(data: Dict[str, pd.DataFrame], symbol: str, period: int = 9) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to calculate EMA for
        period: EMA period
        
    Returns:
        Series of EMA values
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    ema = df['close'].ewm(span=period, adjust=False).mean()
    
    return ema.rename(f'ema_{symbol}_{period}')


def donchian_feature(data: Dict[str, pd.DataFrame], symbol: str, period: int = 20) -> pd.DataFrame:
    """
    Calculate Donchian Channel (20-day high/low breakout).
    
    Args:
        data: Dictionary of symbol -> OHLCV dataframe
        symbol: Symbol to calculate for
        period: Lookback period
        
    Returns:
        DataFrame with upper band (high), lower band (low), and middle
    """
    if symbol not in data:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    df = data[symbol]
    
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    
    return pd.DataFrame({
        f'donchian_upper_{symbol}_{period}': upper,
        f'donchian_lower_{symbol}_{period}': lower,
        f'donchian_middle_{symbol}_{period}': middle
    })