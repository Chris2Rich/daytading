"""
Abstract base classes and protocols for data adapters.
Adapters integrate various data sources into NautilusTrader's ecosystem.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List
import pandas as pd


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
        import yfinance as yf
        
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
        
        # Ensure timezone-aware index
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
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
        from pathlib import Path
        
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


class AlpacaAdapter(BaseAdapter):
    """
    Adapter for Alpaca Markets API.
    Supports both historical and streaming data.
    """
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = "paper"):
        """
        Initialize Alpaca adapter.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: "paper" for paper trading or "live" for live trading
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        
        # Lazy import to avoid hard dependency
        from alpaca_trade_api.rest import REST
        self.api = REST(api_key, api_secret, base_url)
    
    def load_historical(self, symbol: str, start: datetime,
                       end: datetime) -> pd.DataFrame:
        """Load historical data from Alpaca."""
        # Get bars from Alpaca
        bars = self.api.get_bars(
            symbol,
            '1Day',  # Could be configurable
            start=start.isoformat(),
            end=end.isoformat()
        ).df
        
        # Normalize columns
        bars.columns = bars.columns.str.lower()
        
        # Ensure timezone-aware
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize('UTC')
        
        return bars
    
    def supports_streaming(self) -> bool:
        """Alpaca supports streaming."""
        return True
    
    def stream_data(self, symbol: str):
        """Stream real-time data from Alpaca."""
        from alpaca_trade_api.stream import Stream
        
        stream = Stream(self.api_key, self.api_secret, base_url=self.base_url)
        
        @stream.on_bar(symbol)
        async def on_bar(bar):
            yield {
                'symbol': bar.symbol,
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            }
        
        stream.run()


class PolygonAdapter(BaseAdapter):
    """
    Adapter for Polygon.io market data.
    Provides high-quality tick and aggregate data.
    """
    
    def __init__(self, api_key: str, timespan: str = "day",
                 multiplier: int = 1):
        """
        Initialize Polygon adapter.
        
        Args:
            api_key: Polygon.io API key
            timespan: Timespan (minute, hour, day, etc.)
            multiplier: Multiplier for timespan (e.g., 5 for 5-minute bars)
        """
        self.api_key = api_key
        self.timespan = timespan
        self.multiplier = multiplier
    
    def load_historical(self, symbol: str, start: datetime,
                       end: datetime) -> pd.DataFrame:
        """Load historical data from Polygon.io."""
        import requests
        
        # Format dates
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        
        # Build URL
        url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
               f"{self.multiplier}/{self.timespan}/{start_str}/{end_str}")
        
        params = {'apiKey': self.api_key, 'adjusted': 'true', 'sort': 'asc'}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'results' not in data:
            raise ValueError(f"No data returned for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(data['results'])
        
        # Rename columns to match standard format
        df.rename(columns={
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }, inplace=True)
        
        # Convert timestamp (milliseconds) to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def supports_streaming(self) -> bool:
        """Polygon supports streaming via WebSocket."""
        return True
    
    def stream_data(self, symbol: str):
        """Stream real-time data from Polygon.io."""
        # Implementation would use Polygon's WebSocket API
        raise NotImplementedError("Polygon streaming not yet implemented")


class DatabaseAdapter(BaseAdapter):
    """
    Adapter for loading data from SQL databases.
    Generic adapter that can work with any SQL database.
    """
    
    def __init__(self, connection_string: str, table_name: str,
                 symbol_column: str = "symbol",
                 timestamp_column: str = "timestamp"):
        """
        Initialize database adapter.
        
        Args:
            connection_string: SQLAlchemy connection string
            table_name: Name of table containing OHLCV data
            symbol_column: Name of symbol/ticker column
            timestamp_column: Name of timestamp column
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.symbol_column = symbol_column
        self.timestamp_column = timestamp_column
        
        from sqlalchemy import create_engine
        self.engine = create_engine(connection_string)
    
    def load_historical(self, symbol: str, start: datetime,
                       end: datetime) -> pd.DataFrame:
        """Load historical data from database."""
        query = f"""
        SELECT {self.timestamp_column}, open, high, low, close, volume
        FROM {self.table_name}
        WHERE {self.symbol_column} = '{symbol}'
          AND {self.timestamp_column} >= '{start}'
          AND {self.timestamp_column} <= '{end}'
        ORDER BY {self.timestamp_column} ASC
        """
        
        df = pd.read_sql(query, self.engine)
        df.set_index(self.timestamp_column, inplace=True)
        
        # Ensure timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        return df
    
    def supports_streaming(self) -> bool:
        """Database adapter does not support streaming by default."""
        return False


class CCXTAdapter(BaseAdapter):
    """
    Adapter for cryptocurrency exchanges via CCXT library.
    Supports 100+ cryptocurrency exchanges.
    """
    
    def __init__(self, exchange_id: str, timeframe: str = "1d",
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None):
        """
        Initialize CCXT adapter.
        
        Args:
            exchange_id: Exchange identifier (e.g., "binance", "coinbase")
            timeframe: Candle timeframe (1m, 5m, 1h, 1d, etc.)
            api_key: Optional API key for private endpoints
            api_secret: Optional API secret
        """
        import ccxt
        
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        config = {}
        if api_key and api_secret:
            config['apiKey'] = api_key
            config['secret'] = api_secret
        
        self.exchange = exchange_class(config)
    
    def load_historical(self, symbol: str, start: datetime,
                       end: datetime) -> pd.DataFrame:
        """Load historical data from crypto exchange."""
        # Convert to milliseconds timestamp
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        
        all_ohlcv = []
        
        while since < end_ms:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, self.timeframe, since=since, limit=1000
            )
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Move to next candle
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        # Filter by end date
        df = df[df.index <= end]
        
        return df
    
    def supports_streaming(self) -> bool:
        """CCXT supports streaming via WebSocket (exchange-dependent)."""
        return hasattr(self.exchange, 'watch_ohlcv')
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol against exchange's market list."""
        self.exchange.load_markets()
        return symbol in self.exchange.markets


# ============================================================================
# Utility Functions
# ============================================================================


def create_adapter(adapter_type: str, **kwargs) -> BaseAdapter:
    """
    Factory function to create adapters by name.
    
    Args:
        adapter_type: Type of adapter ("yfinance", "csv", "alpaca", etc.)
        **kwargs: Adapter-specific initialization parameters
        
    Returns:
        Instantiated adapter
        
    Example:
        >>> adapter = create_adapter("yfinance", interval="1d")
        >>> adapter = create_adapter("csv", data_dir="./data")
    """
    adapter_map = {
        'yfinance': YFinanceAdapter,
        'csv': CSVAdapter,
        'alpaca': AlpacaAdapter,
        'polygon': PolygonAdapter,
        'database': DatabaseAdapter,
        'ccxt': CCXTAdapter,
    }
    
    if adapter_type not in adapter_map:
        raise ValueError(
            f"Unknown adapter type: {adapter_type}. "
            f"Available: {list(adapter_map.keys())}"
        )
    
    return adapter_map[adapter_type](**kwargs)