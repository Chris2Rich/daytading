"""
Main orchestration file for NautilusTrader-based algorithmic trading system.
This file implements a modular backtesting framework driven by a single config.
"""

from typing import Dict, Any, List, Optional, Callable
from decimal import Decimal
from pathlib import Path
import pickle
from datetime import datetime
import math
import pandas as pd
import numpy as np
from uuid import uuid4 as py_uuid4

from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.position import Position
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.enums import *
from nautilus_trader.model.data import *
from nautilus_trader.model.identifiers import *
from nautilus_trader.model.currencies import *
from nautilus_trader.model.objects import *
from nautilus_trader.model.instruments import *

from models import BaseModel, ModelOutputType, SignalOutput
from adapters import BaseAdapter
from position_sizers import BasePositionSizer


class TradingSystemConfig:
    """
    Single source of truth configuration for the entire trading system.
    All parameters are defined here to ensure modularity and reproducibility.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.seed: int = config_dict['seed']
        self.id: py_uuid4 = config_dict['id']

        # Traded instruments
        self.traded_symbols: List[str] = config_dict['traded_symbols']
        
        # Non-traded timeseries for feature engineering (e.g., VIX, interest rates)
        self.non_traded_timeseries: List[str] = config_dict.get('non_traded_timeseries', [])
        
        # Feature transformations: list of (function, params) tuples
        # Example: [(correlation_feature, {'symbols': ['SPY', 'VIX']}), ...]
        self.features: List[tuple[Callable, Dict[str, Any]]] = config_dict['features']
        
        # Model configuration
        self.model_type: type[BaseModel] = config_dict['model_type']
        self.model_output_type: ModelOutputType = config_dict['model_output_type']
        self.model_params: Dict[str, Any] = config_dict['model_params']
        
        # Backtest timeframe
        self.backtest_start: datetime = config_dict['backtest_start']
        self.backtest_end: datetime = config_dict['backtest_end']
        
        # NautilusTrader simulator parameters
        self.starting_capital: Decimal = Decimal(str(config_dict['starting_capital']))
        self.commission_rate: Decimal = Decimal(str(config_dict.get('commission_rate', 0.001)))
        self.max_leverage: Decimal = Decimal(str(config_dict.get('max_leverage', 1.0)))
        
        # Position sizing strategy
        self.position_sizer_type: type[BasePositionSizer] = config_dict['position_sizer_type']
        self.position_sizer_params: Dict[str, Any] = config_dict.get('position_sizer_params', {})
        
        # Retraining parameters
        self.retrain_frequency: int = config_dict.get('retrain_frequency', 252)  # ticks/bars
        self.training_window_type: str = config_dict.get('training_window_type', 'expanding')  # or 'rolling'
        self.training_window_size: Optional[int] = config_dict.get('training_window_size', None)
        
        # Data adapter configuration
        self.data_adapters: Dict[str, BaseAdapter] = config_dict['data_adapters']
        
        # Model persistence
        self.model_save_path: Path = Path(config_dict.get('model_save_path', './models'))
        self.model_save_path.mkdir(parents=True, exist_ok=True)


class MLTradingStrategy(Strategy):
    """
    Core NautilusTrader strategy that orchestrates:
    - Feature computation
    - Model training and inference
    - Signal generation
    - Position sizing and execution
    """
    
    def __init__(self, config: TradingSystemConfig):
        strategy_config = StrategyConfig(
            strategy_id=f"ML_Strategy_{config.model_type.__name__}",
            order_id_tag="ML"
        )
        super().__init__(config=strategy_config)
        
        self.sys_config = config
        
        # Model instance
        self.models: Dict[str, BaseModel] = {}
        self.models_trained: bool = False

        # Position sizer
        self.position_sizer: BasePositionSizer = config.position_sizer_type(
            **config.position_sizer_params
        )
        
        # Feature storage
        self.feature_df: Optional[pd.DataFrame] = None
        self.raw_data: Dict[str, pd.DataFrame] = {}
        
        # State tracking
        self.bar_count: int = 0
        self.last_retrain_bar: int = 0
        
        # Instrument mapping
        self.instruments: Dict[str, InstrumentId] = {}

        self.account = None
        self.account_id = None
        
    def on_start(self):
        """
        Called when strategy starts. Performs initial setup:
        1. Load historical data from adapters
        2. Compute features in batch mode
        3. Train initial model
        """

        accounts = list(self.cache.accounts())
        if not accounts:
            raise RuntimeError("No accounts found in cache.")
        
        self.account = accounts[0]  # Store the account object
        self.account_id = self.account.id
        self.log.info(f"Using account: {self.account_id}")
        
        for instrument in self.cache.instruments():
            self.instruments[instrument.symbol.value] = instrument.id
            self.log.info(f"Mapped {instrument.symbol.value} to {instrument.id}")

        for symbol in self.sys_config.traded_symbols:
            instrument_id = self.instruments.get(symbol)
            if instrument_id:
                # You must construct the EXACT BarType that was added to the engine.
                # Based on your logs and run_backtest code, it is 1-MINUTE-LAST-EXTERNAL.
                bar_type = BarType(
                    instrument_id=instrument_id,
                    bar_spec=BarSpecification(
                        step=1,
                        aggregation=BarAggregation.MINUTE,
                        price_type=PriceType.LAST
                    ),
                    aggregation_source=AggregationSource.EXTERNAL
                )
                
                self.subscribe_bars(bar_type) # <--- Passing BarType object now
                self.log.info(f"Subscribed to {bar_type}")
            else:
                self.log.error(f"Could not subscribe: No instrument ID for {symbol}")

        self.log.info("Starting ML Trading Strategy")
        
        # Step 1: Load historical data from all adapters
        # --- FIX STARTS HERE ---
        self._load_historical_data()  # <--- THIS LINE WAS MISSING
        # --- FIX ENDS HERE ---
        
        # Step 2: Compute all features in batch mode
        self._compute_features_batch()
        
        # Step 3: Train initial model
        self._train_model(initial=True) 

    def on_bar(self, bar: Bar):
        self.bar_count += 1

        # ----------------------------------------------------
        # 1. Update raw market data
        # ----------------------------------------------------
        symbol = bar.bar_type.instrument_id.symbol.value
        if symbol not in self.raw_data:
            self.raw_data[symbol] = pd.DataFrame()

        timestamp = pd.to_datetime(bar.ts_event, unit='ns', utc=True)

        if timestamp not in self.raw_data[symbol].index:
            bar_data = pd.DataFrame(
                [{
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }],
                index=[timestamp],
            )
            self.raw_data[symbol] = pd.concat([self.raw_data[symbol], bar_data])
            
            # Optional: Trim to keep memory usage low
            if len(self.raw_data[symbol]) > 2000:
                self.raw_data[symbol] = self.raw_data[symbol].iloc[-2000:]

        # ----------------------------------------------------
        # 2. Compute market features (LATEST ONLY)
        # ----------------------------------------------------
        market_features = {}

        for feature_func, params in self.sys_config.features:
            # We pass the current raw_data (which includes the new bar)
            feature_value = feature_func(self.raw_data, **params)

            if isinstance(feature_value, pd.Series):
                # We only want the specific value for THIS timestamp
                # Note: feature functions return a Series with datetime index.
                # We grab the very last value (-1).
                val = feature_value.iloc[-1]
                # Handle cases where feature name might differ from func name
                # (Your feature functions return Series with specific names, 
                # e.g. 'rsi_AAPL_14', so we extract that name)
                name = feature_value.name if feature_value.name else feature_func.__name__
                market_features[name] = float(val)
                
            elif isinstance(feature_value, pd.DataFrame):
                # For functions returning DataFrames (like MACD, Bollinger Bands)
                last_row = feature_value.iloc[-1]
                market_features.update(last_row.to_dict())

        # ----------------------------------------------------
        # 3. Snapshot portfolio state (AFTER previous fills)
        # ----------------------------------------------------
        if not self.models_trained:
            portfolio_features = {}
        else:
            portfolio_features = self._get_portfolio_snapshot()

        # ----------------------------------------------------
        # 4. Append ONE causal feature row
        # ----------------------------------------------------
        feature_row = {**market_features, **portfolio_features}

        row_df = pd.DataFrame([feature_row], index=[timestamp])
        row_df = row_df.reindex(columns=self.feature_columns, fill_value=0.0)

        self.feature_df = pd.concat([self.feature_df, row_df])

        # ----------------------------------------------------
        # 6. Generate signals
        # ----------------------------------------------------
        signals = self._generate_signals()

        # ----------------------------------------------------
        # 7. Execute trades
        # ----------------------------------------------------
        self._execute_trades(signals)

        # ----------------------------------------------------
        # 8. Retraining check
        # ----------------------------------------------------
        if self._should_retrain():
            self.log.info(f"Retraining model at bar {self.bar_count}")
            self._train_model(initial=False)
            self.last_retrain_bar = self.bar_count


    def on_stop(self):
        """Called when strategy stops. Save model state."""
        self._save_model()
        self.log.info("Strategy stopped. Model saved.")
    
    def _load_historical_data(self):
        """Load all historical data from configured adapters."""
        self.log.info("Loading historical data from adapters...")
        
        all_symbols = self.sys_config.traded_symbols + self.sys_config.non_traded_timeseries
        
        for symbol in all_symbols:
            if symbol in self.sys_config.data_adapters:
                adapter = self.sys_config.data_adapters[symbol]
                df = adapter.load_historical(
                    symbol=symbol,
                    start=self.sys_config.backtest_start,
                    end=self.sys_config.backtest_end
                )
                self.raw_data[symbol] = df
                self.log.info(f"Loaded {len(df)} bars for {symbol}")
            else:
                self.log.warning(f"No adapter configured for {symbol}")
    
    def _feature_required_symbols(self, params: dict) -> set[str]:
        symbols = set()

        if "symbol" in params:
            symbols.add(params["symbol"])

        if "symbols" in params:
            symbols.update(params["symbols"])

        return symbols


    def _compute_features_batch(self):
        self.log.info("Computing batch-safe features...")

        feature_dfs = []
        self.online_features = []

        for feature_func, params in self.sys_config.features:
            required_symbols = self._feature_required_symbols(params)

            # Batch-safe only if ALL symbols already loaded
            if required_symbols.issubset(self.raw_data.keys()):
                self.log.info(f"Batch feature: {feature_func.__name__}")

                feature_values = feature_func(self.raw_data, **params)

                if isinstance(feature_values, pd.Series):
                    feature_values = feature_values.to_frame()

                feature_dfs.append(feature_values)
            else:
                self.log.info(f"Online feature: {feature_func.__name__}")
                self.online_features.append((feature_func, params))

        if feature_dfs:
            self.feature_df = pd.concat(feature_dfs, axis=1)
            self.feature_df = self.feature_df.fillna(method="ffill").fillna(0)
        else:
            self.feature_df = pd.DataFrame()

        self.feature_columns = list(self.feature_df.columns)

        self.log.info(
            f"Batch features: {len(feature_dfs)}, Online features: {len(self.online_features)}"
        )

    def _compute_triple_barrier_targets(self, symbol: str, df: pd.DataFrame, horizon: int = 10, profit_taking: float = 0.02, stop_loss: float = 0.01) -> pd.Series:
        if symbol not in self.raw_data:
            return pd.Series(index=df.index, data=np.nan)
            
        # Get raw prices aligned to feature index
        prices = self.raw_data[symbol].reindex(df.index).fillna(method='ffill')
        
        # KEY FIX: Shift prices back by 1.
        # The 'future' for Bar T starts at Bar T+1.
        next_bars_high = prices['high'].shift(-1)
        next_bars_low = prices['low'].shift(-1)
        
        # Setup forward looking window
        # We look at the NEXT 'horizon' bars.
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
        
        # 1. Get Max High and Min Low of the FUTURE window
        # Because we shifted by -1, the window at index T now covers [T+1, ..., T+horizon]
        future_highs = next_bars_high.rolling(window=indexer).max()
        future_lows = next_bars_low.rolling(window=indexer).min()
        
        # 2. Define Barriers based on Close at T
        current_close = prices['close']
        upper_barrier = current_close * (1 + profit_taking)
        lower_barrier = current_close * (1 - stop_loss)
        
        # 3. Determine touches
        # Note: In a real rigorous TBM, we'd check which barrier was hit FIRST.
        # This vectorized approximation assumes if both are hit in the window, it's a Stop Loss (conservative).
        
        touch_upper = future_highs > upper_barrier
        touch_lower = future_lows < lower_barrier
        
        # 4. Generate Labels
        # Default = 1 (Time Limit / Hold)
        y = pd.Series(1, index=df.index)
        
        # If lower barrier hit -> 0 (Sell/Stop)
        # We prioritize Stop Loss: if volatility triggers both, assume we got stopped out.
        y[touch_lower] = 0
        
        # If upper barrier hit AND lower NOT hit -> 2 (Buy/Profit)
        y[touch_upper & ~touch_lower] = 2
        
        # 5. Clean up NaNs
        # The last 'horizon' rows will be NaN because we don't have future data for them yet.
        # We must drop them during training.
        y = y.dropna()
        
        return y

    def _get_portfolio_snapshot(self) -> dict[str, float]:
        """
        Return a portfolio snapshot for ML features using documented NautilusTrader APIs.

        Portfolio value (equity) is computed as:
            equity = cash + sum(position market value + total_pnl)
        """

        snapshot: dict[str, float] = {}

        # Edge case: no instruments mapped
        if not self.instruments:
            base = float(self.sys_config.starting_capital)
            snapshot["cash"] = base
            snapshot["equity"] = base
            for s in self.sys_config.traded_symbols:
                snapshot[f"position_{s}"] = 0.0
            return snapshot

        # Assume single venue
        venue = next(iter(self.instruments.values())).venue

        # ----------------------------------
        # 1) CASH
        # ----------------------------------
        account = self.portfolio.account(venue)
        if account is None:
            raise RuntimeError(f"No account found for venue {venue}")

        balance_obj = account.balance(account.base_currency)
        cash = float(balance_obj.total) if balance_obj else 0.0
        snapshot["cash"] = cash

        # ----------------------------------
        # 2) POSITION QUANTITIES
        # ----------------------------------
        for symbol in self.sys_config.traded_symbols:
            instr_id = self.instruments[symbol]
            qty = float(self.portfolio.net_position(instr_id))
            snapshot[f"position_{symbol}"] = qty

        # ----------------------------------
        # 3) MARKET VALUE + TOTAL PNL → Equity
        # ----------------------------------
        total_equity = cash

        for symbol in self.sys_config.traded_symbols:
            instr_id = self.instruments[symbol]

            # Market value of position
            # Use actual close price from raw data if available
            price = 0.0
            if symbol in self.raw_data and not self.raw_data[symbol].empty:
                price = float(self.raw_data[symbol].iloc[-1]["close"])

            position_value = snapshot[f"position_{symbol}"] * price

            # Total PnL (realized + unrealized)
            # If position is flat, this returns Money(0)
            pnl_money = self.portfolio.total_pnl(instr_id)
            pnl_value = float(pnl_money) if pnl_money is not None else 0.0

            total_equity += position_value + pnl_value

        snapshot["equity"] = total_equity

        return snapshot


    def _train_model(self, initial: bool = False):
        if self.feature_df is None or len(self.feature_df) < 200:
            return

        # 1. Select Training Data Window
        if initial or self.sys_config.training_window_type == 'expanding':
            train_df = self.feature_df
        else:
            ws = self.sys_config.training_window_size or 500
            train_df = self.feature_df.iloc[-ws:]

        # 2. Iterate over each traded symbol
        for symbol in self.sys_config.traded_symbols:
            
            # --- A. Filter Features for THIS symbol ---
            # Grab columns like 'rsi_AAPL' and generic columns like 'VIX'
            # Exclude columns belonging to OTHER traded assets (e.g. 'rsi_MSFT')
            other_symbols = set(self.sys_config.traded_symbols) - {symbol}
            
            # Create boolean mask for columns to keep
            cols_to_keep = []
            for col in train_df.columns:
                # If column has another symbol name in it, skip it
                if any(os in col for os in other_symbols):
                    continue
                cols_to_keep.append(col)
                
            X_df = train_df[cols_to_keep]
            
            # --- B. Compute Targets (Ground Truth) ---
            # This looks at future data (T+1 onwards) to create the label
            y_series = self._compute_triple_barrier_targets(
                symbol, 
                train_df, 
                horizon=10, 
                profit_taking=0.015, 
                stop_loss=0.01
            )
            
            # --- C. Align and Clean ---
            # y_series is shorter than X_df by 'horizon' rows (at the end)
            common_idx = X_df.index.intersection(y_series.index)
            
            if len(common_idx) < 100:
                continue

            X_final = X_df.loc[common_idx].values
            y_final = y_series.loc[common_idx].values
            
            # --- D. Train Symbol-Specific Model ---
            self.log.info(f"Training model for {symbol} | Samples: {len(X_final)}")
            
            # Ensure you initialized self.models = {} in __init__
            if symbol not in self.models:
                # Create new instance if missing
                self.models[symbol] = self.sys_config.model_type(**self.sys_config.model_params)
                
            self.models[symbol].train(X_final, y=y_final)
            
        self.models_trained = True

    def _generate_signals(self) -> Dict[str, SignalOutput]:
        signals = {}
        if not hasattr(self, 'models_trained') or not self.models_trained:
            return signals

        # Latest feature row
        last_idx = self.feature_df.index[-1]
        last_row = self.feature_df.iloc[-1:]

        for symbol in self.sys_config.traded_symbols:
            if symbol not in self.models:
                continue

            # 1. Filter features exactly like in training
            other_symbols = set(self.sys_config.traded_symbols) - {symbol}
            cols_to_keep = [c for c in last_row.columns if not any(os in c for os in other_symbols)]
            
            X_live = last_row[cols_to_keep].values

            # 2. Predict
            # Returns SignalOutput with Action and Risk Factor (Confidence)
            signal = self.models[symbol].predict(X_live)
            
            signals[symbol] = signal
            
        return signals
    
    def _execute_trades(self, signals: Dict[str, SignalOutput]):
        """Execute trades based on signals and position sizing."""
        for symbol, signal in signals.items():
            instrument_id = self.instruments.get(symbol)
            if not instrument_id:
                continue
            
            # 1. Get current price (Last known close)
            # We need this to convert $ amount to # of shares
            last_bar = self.raw_data[symbol].iloc[-1]
            current_price = float(last_bar['close'])
            
            if current_price <= 0:
                continue

            # 2. Get current position size (in Units, e.g., 50 shares)
            current_quantity = self._get_portfolio_snapshot().get(
                f"position_{symbol}", 0.0
            )

            # 3. Calculate target value (in Dollars)
            # The FixedFractionalSizer returns DOLLARS (e.g. 10,000.00)

            target_value = self.position_sizer.calculate_size(
                signal=signal,
                current_position=current_quantity, # Sizer expects units? No, let's look at logic below.
                portfolio_value=self._get_portfolio_snapshot()["equity"],
                instrument_id=instrument_id
            )
            
            # 4. Convert Target Value ($) to Target Quantity (Units)
            # Check sizer type logic: FixedFractional returns Value.
            # FixedSizer returns Units. 
            # We need to handle this based on your sizer implementation.
            # Assuming FixedFractionalSizer is used (which returns value):
            target_quantity = target_value / current_price
            
            # 5. Calculate delta and submit
            quantity_delta = target_quantity - current_quantity
            
            if abs(quantity_delta) < 1.0:  # Minimum 1 share change
                continue
            
            order_side = OrderSide.BUY if quantity_delta > 0 else OrderSide.SELL
            order_quantity = abs(quantity_delta)
            
            # Round to nearest integer (assuming lot_size=1)
            order_quantity = int(order_quantity)
            
            order = MarketOrder(
                self.trader_id,                         # TraderId
                self.id,                                # StrategyId
                instrument_id,                          # InstrumentId
                ClientOrderId(str(py_uuid4())),             # client_order_id
                order_side,                             # OrderSide.BUY / SELL
                Quantity.from_int(order_quantity),       # Quantity
                UUID4.from_str(str(py_uuid4())),                                # init_id
                self.clock.timestamp_ns(),               # ts_init (ENGINE TIME)
                TimeInForce.GTC,                        # optional
            )
            
            self.submit_order(order)
            self.log.info(f"Submitted {order_side} order for {symbol}: {order_quantity} shares @ ~{current_price}")

    def _should_retrain(self) -> bool:
        """Determine if model should be retrained."""
        bars_since_retrain = self.bar_count - self.last_retrain_bar
        return bars_since_retrain >= self.sys_config.retrain_frequency
    
    def _save_model(self):
        """Save trained model to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.sys_config.model_save_path / f"model_{timestamp}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(
                {
                    "models": self.models,
                    "config": self.sys_config,
                },
                f,
            )
        
        self.log.info(f"Model saved to {model_path}")


def run_backtest(config_dict: Dict[str, Any]) -> BacktestEngine:
    """
    Main entry point to run backtest with given configuration.
    
    Args:
        config_dict: Configuration dictionary defining all system parameters
        
    Returns:
        BacktestEngine with completed backtest results
    """
    # Create configuration object
    config = TradingSystemConfig(config_dict)
    
    # Initialize backtest engine
    engine = BacktestEngine(config=BacktestEngineConfig())

    engine.add_venue(
    venue=Venue("BACKTESTING"),
    oms_type=OmsType.NETTING,
    account_type=AccountType.MARGIN,
    base_currency=USD,
    starting_balances=[Money(config.starting_capital, USD)],
    )
    
    # Add venues and instruments (simplified - extend as needed)
    # This would typically load from adapters
    for symbol in config.traded_symbols:
        instrument = Equity(
            instrument_id=InstrumentId(Symbol(symbol), Venue("BACKTESTING")),
            raw_symbol=Symbol(symbol),
            currency=USD,
            price_precision=2,
            price_increment=Price.from_str("0.01"),
            lot_size=Quantity.from_int(1),
            ts_event=0,
            ts_init=0,
        )
        engine.add_instrument(instrument)

    for symbol in config.traded_symbols:
        if symbol in config.data_adapters:
            adapter = config.data_adapters[symbol]
            
            # Load historical data from adapter
            df = adapter.load_historical(
                symbol=symbol,
                start=config.backtest_start,
                end=config.backtest_end
            )
            
            # Convert DataFrame to NautilusTrader Bar objects
            bars = []
            instrument_id = InstrumentId(Symbol(symbol), Venue("BACKTESTING"))
            bar_type = BarType(
                instrument_id=instrument_id,
                bar_spec=BarSpecification(
                    step=1,
                    aggregation=BarAggregation.MINUTE,
                    price_type=PriceType.LAST
                ),
                aggregation_source=AggregationSource.EXTERNAL
            )
            
            def to_price(value: float, precision: int = 2) -> Price:
                q = Decimal(str(value)).quantize(Decimal("1." + "0" * precision))
                return Price.from_str(format(q, "f"))

            def to_quantity(value: float, precision: int = 0) -> Price:
                q = Decimal(str(value)).quantize(Decimal("1." + "0" * precision))
                return Quantity.from_str(format(q, "f"))

            for ts, row in df.iterrows():
                bar = Bar(
                    bar_type=bar_type,
                    open=to_price(row['open']),
                    high=to_price(row['high']),
                    low=to_price(row['low']),
                    close=to_price(row['close']),
                    volume=to_quantity(row['volume']),
                    ts_event=int(ts.timestamp() * 1e9),  # Convert to nanoseconds
                    ts_init=int(ts.timestamp() * 1e9),
                )
                bars.append(bar)
            
            # Add bars to engine
            engine.add_data(bars)

    # Create and add strategy
    strategy = MLTradingStrategy(config)
    engine.add_strategy(strategy)
    
    # Run backtest
    engine.run()
    
    return engine

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)