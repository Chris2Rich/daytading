from AlgorithmImports import QCAlgorithm, BaseData, SubscriptionDataSource, SubscriptionTransportMedium
from lib.retrieval.yahoofinance import get_trading

class YahooFinanceOHLCV(BaseData):
    """LEAN BaseData for Yahoo OHLCV bars."""

    def __init__(self):
        super().__init__()
        self.Open = 0.0
        self.High = 0.0
        self.Low = 0.0
        self.Close = 0.0
        self.Volume = 0.0

    @staticmethod
    def GetSource(config, date, isLiveMode):
        ticker = config.Symbol.Value
        resolution = str(config.Resolution)

        # Map LEAN resolution to Yahoo interval
        interval_map = {"Minute": "1m", "Hour": "60m", "Daily": "1d"}
        interval = interval_map.get(resolution, "1d")

        # Days to fetch from config
        days = getattr(config, "ohlcv_days", 365)

        df = get_trading(ticker, days=days, interval=interval)
        if df.empty:
            raise ValueError(f"No Yahoo Finance data for {ticker}")

        # Ensure columns: Open, High, Low, Close, Volume
        df = df.rename(columns={"Adj Close": "Close"})
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing OHLCV columns for {ticker}: {missing}")
        df = df.sort_index()
        df.to_csv(f"data/{ticker}_{interval}.csv")

        return SubscriptionDataSource(f"data/{ticker}_{interval}.csv", SubscriptionTransportMedium.LocalFile)

    def Reader(self, config, line, date, isLiveMode):
        if not line or line.startswith("Date"):
            return None
        parts = line.strip().split(",")
        bar = YahooFinanceOHLCV()
        bar.Symbol = config.Symbol
        bar.Time = pd.to_datetime(parts[0]).to_pydatetime()
        bar.Open = float(parts[1])
        bar.High = float(parts[2])
        bar.Low = float(parts[3])
        bar.Close = float(parts[4])
        bar.Volume = float(parts[5]) if len(parts) > 5 else 0.0
        return bar