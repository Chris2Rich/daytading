from typing import Dict, Any
import pandas as pd

from abstract import DataSource
from lib.retrieval.yahoofinance import get_trading


class YahooFinanceOHLCVDataSource(DataSource):
    """
    LEAN-compatible DataSource that replays Yahoo Finance OHLCV data
    causally, one bar per OnData call.

    - Inherits from DataSource(ABC)
    - Ignores LEAN Slice (LEAN is only the clock)
    - Prevents lookahead via internal cursor
    """

    def __init__(
        self,
        ticker: str,
        days: int,
        interval: str,
        include_volume: bool = True,
    ):
        self.ticker = ticker
        self.reference = self.ticker + "OHLCV"
        self.days = days
        self.interval = interval
        self.include_volume = include_volume

        self._data = self._load_data()
        if self._data.empty:
            raise ValueError(f"No Yahoo Finance data for {ticker}")

        self._cursor = 0
        self._timestamps = self._data.index.tolist()

    # ------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------

    def observe(self, slice) -> Dict[str, Any]:
        """
        Emit the next causal market observation.

        LEAN Slice is ignored intentionally.
        """
        if self._cursor >= len(self._data):
            raise StopIteration("YahooFinanceDataSource exhausted")

        row = self._data.iloc[self._cursor]
        ts = self._timestamps[self._cursor]

        observation = {self.reference : {
            "timestamp": ts,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
        }}

        if self.include_volume and "Volume" in row:
            observation[self.reference]["volume"] = float(row["Volume"])

        self._cursor += 1
        return observation

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        """
        Fetch and normalize Yahoo Finance trading data.
        """
        df = get_trading(
            ticker=self.ticker,
            days=self.days,
            interval=self.interval,
        )

        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        # Standardize column names
        df = df.rename(columns={"Adj Close": "Close"})

        required = {"Open", "High", "Low", "Close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required OHLC columns: {missing}")

        # Sort strictly by time to enforce causality
        df = df.sort_index()

        # Defensive cleaning
        df = df.dropna(subset=list(required))

        return df
