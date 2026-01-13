"""
Abstract base classes and protocols for data adapters.
Adapters integrate various data sources into NautilusTrader's ecosystem.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List
import pandas as pd
import yfinance as yf
from pathlib import Path


class BaseAdapter(ABC):
    """
    Abstract base class for data adapters.
    
    Adapters serve as the bridge between external data sources and 
    NautilusTrader's internal data structures. Each adapter handles:
    
    1. Historical data loading for backtesting
    2. (Optional) Real-time streaming for live trading
    3. Data format normalization to NautilusTrader standards
    
    The adapter pattern allows the system to work with any data source
    (APIs, files, databases, live feeds) without changing core logic.
    """
    
    @abstractmethod
    def load_historical(self, symbol: str, start: datetime, 
                       end: datetime) -> pd.DataFrame:
        """
        Load historical data for backtesting.
        
        Args:
            symbol: Instrument identifier (e.g., "AAPL", "BTCUSD")
            start: Start datetime for historical data
            end: End datetime for historical data
            
        Returns:
            DataFrame with columns: [open, high, low, close, volume]
            Index: DatetimeIndex representing bar timestamps
            
        Note:
            - Must return data in chronological order
            - Must handle missing data appropriately (forward-fill, drop, etc.)
            - Timestamps should be timezone-aware (UTC preferred)
        """
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """
        Indicate if adapter supports real-time streaming.
        
        Returns:
            True if adapter can stream live data, False otherwise
        """
        pass
    
    def stream_data(self, symbol: str):
        """
        Stream real-time data (optional, for live trading).
        
        Args:
            symbol: Instrument identifier
            
        Yields:
            Data updates as they arrive (format: adapter-specific)
            
        Note:
            Only needs implementation if supports_streaming() returns True
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming"
        )
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol is available from this data source.
        
        Args:
            symbol: Instrument identifier
            
        Returns:
            True if symbol is valid and available
        """
        return True  # Default: assume all symbols valid


# ============================================================================
# Concrete Adapter Implementations
# ============================================================================


class YFinanceAdapter(BaseAdapter):
    """
    Adapter for Yahoo Finance data via yfinance library.
    Supports historical data only (no streaming).
    """
    
    def __init__(self, interval: str = "1d"):
        """
        Initialize Yahoo Finance adapter.
        
        Args:
            interval: Data interval (1m, 5m, 1h, 1d, etc.)
        """
        self.interval = interval
        
    def load_historical(self, symbol: str, start: datetime, 
                       end: datetime) -> pd.DataFrame:
        """Load historical data from Yahoo Finance."""
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=self.interval)
        
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns for {symbol}")
        
        # Select only required columns
        df = df[required_cols]
        
        return df
    
    def supports_streaming(self) -> bool:
        """Yahoo Finance does not support streaming."""
        return False


class CSVAdapter(BaseAdapter):
    """
    Adapter for CSV files with OHLCV data.
    Useful for custom datasets and research.
    """
    
    def __init__(self, data_dir: str, timestamp_col: str = "timestamp",
                 date_format: Optional[str] = None):
        """
        Initialize CSV adapter.
        
        Args:
            data_dir: Directory containing CSV files
            timestamp_col: Name of timestamp column in CSV
            date_format: Optional datetime format string for parsing
        """
        self.data_dir = data_dir
        self.timestamp_col = timestamp_col
        self.date_format = date_format
    
    def load_historical(self, symbol: str, start: datetime,
                       end: datetime) -> pd.DataFrame:
        """Load historical data from CSV file."""
        
        # Construct file path (assumes files named like "AAPL.csv")
        file_path = Path(self.data_dir) / f"{symbol}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Parse timestamps
        if self.date_format:
            df[self.timestamp_col] = pd.to_datetime(
                df[self.timestamp_col], format=self.date_format
            )
        else:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        
        # Set as index
        df.set_index(self.timestamp_col, inplace=True)
        
        # Filter by date range
        df = df[(df.index >= start) & (df.index <= end)]
        
        # Normalize column names
        df.columns = df.columns.str.lower()
        
        # Ensure timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        return df
    
    def supports_streaming(self) -> bool:
        """CSV files do not support streaming."""
        return False