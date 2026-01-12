from AlgorithmImports import *
from abstract import BaseLeanDataAdapter
from lib.retrieval.yahoofinance import get_trading  # Import the wrapper function
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta

class YFinanceDataAdapter(BaseLeanDataAdapter):
    """
    Adapter for Yahoo Finance data using the get_trading wrapper function.
    
    This adapter fetches OHLCV data using get_trading() and writes it to a temporary
    CSV file that LEAN can read. Supports both backtest and live modes.
    
    Configuration:
        - interval: Data interval (1d, 1h, 5m, etc.)
        - days_history: Number of days of historical data to fetch
    """
    
    # Class-level cache for temporary files to avoid re-downloading
    _cache = {}
    _temp_dir = None
    
    def __init__(self):
        """Initialize the adapter"""
        super().__init__()
        self.interval = "1d"  # Default interval
        self.days_history = 365  # Default history
        
        # Create temp directory if it doesn't exist
        if YFinanceDataAdapter._temp_dir is None:
            YFinanceDataAdapter._temp_dir = tempfile.mkdtemp(prefix="lean_yfinance_")
    
    def GetSource(self, config, date, isLiveMode):
        """
        Return the source for LEAN to fetch data from.
        
        Uses get_trading() to fetch data and saves to CSV.
        
        Args:
            config: SubscriptionDataConfig object from LEAN
            date: Date for which to get data
            isLiveMode: Whether running in live or backtest mode
            
        Returns:
            SubscriptionDataSource pointing to the CSV file
        """
        symbol = config.Symbol.Value
        
        # Create cache key
        cache_key = f"{symbol}_{self.interval}_{self.days_history}_{date.strftime('%Y%m%d')}"
        
        # Check if we already have this data cached
        if cache_key in YFinanceDataAdapter._cache:
            csv_path = YFinanceDataAdapter._cache[cache_key]
            if os.path.exists(csv_path):
                return SubscriptionDataSource(
                    csv_path,
                    SubscriptionTransportMedium.LocalFile
                )
        
        # Fetch data using get_trading wrapper
        try:
            # Use get_trading to fetch the data
            data = get_trading(
                ticker=symbol,
                days=self.days_history,
                interval=self.interval
            )
            
            if data.empty:
                return SubscriptionDataSource(
                    "",
                    SubscriptionTransportMedium.LocalFile
                )
            
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Reset index to make datetime a column
            data = data.reset_index()
            
            # Standardize column names for LEAN
            # get_trading returns: Open, High, Low, Close, Volume
            column_mapping = {
                'Date': 'datetime',
                'Datetime': 'datetime',
                'index': 'datetime',  # In case index is named 'index'
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            data = data.rename(columns=column_mapping)
            
            # Handle case where index might not have a name
            if 'datetime' not in data.columns:
                # First column should be the datetime index
                data.columns = ['datetime'] + list(data.columns[1:])
            
            # Select only the columns we need
            required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            data = data[required_columns]
            
            # Format datetime for LEAN (YYYYMMDD HH:MM:SS)
            data['datetime'] = pd.to_datetime(data['datetime']).dt.strftime('%Y%m%d %H:%M:%S')
            
            # Write to temporary CSV file
            csv_filename = f"{symbol}_{self.interval}_{date.strftime('%Y%m%d')}.csv"
            csv_path = os.path.join(YFinanceDataAdapter._temp_dir, csv_filename)
            
            # Write without header and index
            data.to_csv(csv_path, index=False, header=False)
            
            # Cache the file path
            YFinanceDataAdapter._cache[cache_key] = csv_path
            
            return SubscriptionDataSource(
                csv_path,
                SubscriptionTransportMedium.LocalFile
            )
            
        except Exception as e:
            # Log error and return empty source
            print(f"Error fetching data for {symbol}: {str(e)}")
            return SubscriptionDataSource(
                "",
                SubscriptionTransportMedium.LocalFile
            )
    
    def Reader(self, config, line, date, isLiveMode):
        """
        Parse a line of CSV data and return a BaseData object.
        
        Expected CSV format: datetime,open,high,low,close,volume
        
        Args:
            config: SubscriptionDataConfig object from LEAN
            line: String line from CSV to parse
            date: Date of the data
            isLiveMode: Whether running in live or backtest mode
            
        Returns:
            YFinanceDataAdapter instance with populated data
        """
        if not line or line.strip() == "":
            return None
        
        try:
            # Parse CSV line
            parts = line.split(',')
            
            if len(parts) < 6:
                return None
            
            # Create new instance
            data = YFinanceDataAdapter()
            data.Symbol = config.Symbol
            
            # Parse datetime (format: YYYYMMDD HH:MM:SS)
            datetime_str = parts[0].strip()
            data.Time = datetime.strptime(datetime_str, '%Y%m%d %H:%M:%S')
            
            # Parse OHLCV data
            data.Open = float(parts[1])
            data.High = float(parts[2])
            data.Low = float(parts[3])
            data.Close = float(parts[4])
            data.Volume = int(float(parts[5]))
            
            # Set the value to close price (LEAN convention)
            data.Value = data.Close
            
            return data
            
        except Exception as e:
            print(f"Error parsing line '{line}': {str(e)}")
            return None
    
    @classmethod
    def set_interval(cls, interval: str):
        """
        Set the data interval for all instances.
        
        Args:
            interval: One of 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
        """
        valid_intervals = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of {valid_intervals}")
        return interval
    
    @classmethod
    def set_history_days(cls, days: int):
        """
        Set the number of days of historical data to fetch.
        
        Args:
            days: Number of days of history
        """
        if days <= 0:
            raise ValueError("days must be positive")
        return days
    
    @classmethod
    def cleanup_cache(cls):
        """Clean up temporary CSV files"""
        if cls._temp_dir and os.path.exists(cls._temp_dir):
            import shutil
            shutil.rmtree(cls._temp_dir)
            cls._temp_dir = None
            cls._cache.clear()

class ConfigurableYFinanceAdapter(YFinanceDataAdapter):
    """
    YFinance adapter with configurable parameters.
    
    Usage:
        adapter = ConfigurableYFinanceAdapter.create(interval="1h", days_history=90)
        algo.AddData(adapter, "SPY")
    """
    
    def __init__(self, interval: str = "1d", days_history: int = 365):
        """
        Initialize with specific parameters.
        
        Args:
            interval: Data interval (1d, 1h, 5m, etc.)
            days_history: Number of days of historical data
        """
        super().__init__()
        self.interval = interval
        self.days_history = days_history
    
    @classmethod
    def create(cls, interval: str = "1d", days_history: int = 365):
        """
        Factory method to create a configured adapter class.
        
        Args:
            interval: Data interval (1d, 1h, 5m, etc.)
            days_history: Number of days of historical data
            
        Returns:
            Adapter class configured with specified parameters
        """
        # Create a new class with the specified parameters as defaults
        class _ConfiguredAdapter(ConfigurableYFinanceAdapter):
            def __init__(self):
                super().__init__(interval=interval, days_history=days_history)
        
        return _ConfiguredAdapter