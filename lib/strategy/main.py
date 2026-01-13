"""
Main orchestration file for NautilusTrader-based algorithmic trading system.
This file implements a modular backtesting framework driven by a single config.
"""

from typing import Dict, Any, List, Optional, Callable
from decimal import Decimal
from pathlib import Path
import pickle
from datetime import datetime

import pandas as pd
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.position import Position
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.config import StrategyConfig
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
        
        # Regime detection callback (optional)
        self.regime_detector: Optional[Callable] = config_dict.get('regime_detector', None)
        self.regime_check_frequency: int = config_dict.get('regime_check_frequency', 20)
        
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
        self.model: BaseModel = config.model_type(**config.model_params)
        
        # Position sizer
        self.position_sizer: BasePositionSizer = config.position_sizer_type(
            **config.position_sizer_params
        )
        
        # Feature storage
        self.feature_df: Optional[pd.DataFrame] = None
        self.raw_data: Dict[str, pd.DataFrame] = {}
        
        # State tracking
        self.bar_count: int = 0
        self.current_regime: Optional[Any] = None
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

        bar_data = pd.DataFrame(
            [{
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }],
            index=[bar.ts_event],
        )

        self.raw_data[symbol] = pd.concat([self.raw_data[symbol], bar_data])
        self.raw_data[symbol] = self.raw_data[symbol].iloc[-2000:]

        # ----------------------------------------------------
        # 2. Compute market features (LATEST ONLY)
        # ----------------------------------------------------
        market_features = {}

        for feature_func, params in self.online_features:
            feature_value = feature_func(self.raw_data, **params)

            if isinstance(feature_value, pd.Series):
                feature_value = feature_value.to_frame(name=feature_func.__name__)

            market_features.update(feature_value.iloc[-1].to_dict())

        # ----------------------------------------------------
        # 3. Snapshot portfolio state (AFTER previous fills)
        # ----------------------------------------------------
        if not self.model.is_trained:
            portfolio_features = {}
        else:
            portfolio_features = self._get_portfolio_snapshot()

        # ----------------------------------------------------
        # 4. Append ONE causal feature row
        # ----------------------------------------------------
        feature_row = {**market_features, **portfolio_features}

        row_df = pd.DataFrame([feature_row], index=[bar.ts_event])
        row_df = row_df.reindex(columns=self.feature_columns, fill_value=0.0)

        self.feature_df = pd.concat([self.feature_df, row_df])

        # ----------------------------------------------------
        # 5. Regime detection (optional)
        # ----------------------------------------------------
        if (
            self.sys_config.regime_detector
            and self.bar_count % self.sys_config.regime_check_frequency == 0
        ):
            new_regime = self._check_regime()
            if new_regime != self.current_regime:
                self.log.info(
                    f"Regime change detected: {self.current_regime} -> {new_regime}"
                )
                self.current_regime = new_regime
                self._handle_regime_change()

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


    def _get_portfolio_snapshot(self) -> dict:
        if not self.account:
            equity = float(self.sys_config.starting_capital)
            cash = equity
            positions = {}
        else:
            equity = float(self.portfolio.account(self.account_id).equity())
            cash = float(self.portfolio.account(self.account_id).cash())

            positions = {
                p.instrument_id.symbol.value: float(p.quantity)
                for p in self.account.positions()
            }

        snapshot = {
            "cash": cash,
            "equity": equity,
        }

        for symbol in self.sys_config.traded_symbols:
            snapshot[f"position_{symbol}"] = positions.get(symbol, 0.0)

        return snapshot


    def _train_model(self, initial: bool = False):
        """Train or retrain the model on available data."""
        if self.feature_df is None or len(self.feature_df) < 100:
            self.log.warning("Insufficient data for training")
            return
        
        # Determine training window
        if initial or self.sys_config.training_window_type == 'expanding':
            train_data = self.feature_df
        else:
            window_size = self.sys_config.training_window_size or len(self.feature_df)
            train_data = self.feature_df.iloc[-window_size:]
        
        self.log.info(f"Training model on {len(train_data)} samples")
        
        # --- IMPROVED TARGET GENERATION ---
        # 1. Get Close prices for the assets (assuming AAPL for this example, 
        #    ideally you loop through symbols or train one model per symbol)
        #    For this specific config, let's grab the Returns of AAPL (period 1) 
        #    which allows us to know if price went up/down.
        
        target_col = 'returns_AAPL_1' # From your config
        
        if target_col not in train_data.columns:
            self.log.error(f"Target column {target_col} not found for training.")
            return

        # Shift returns backwards by 1 to predict *Next* bar's return using *Current* bar's features
        # y = 1 (Buy) if next return > 0, else 0 (Sell/Hold)
        future_returns = train_data[target_col].shift(-1)
        
        # Drop the last row (NaN target) from both X and y
        X = train_data.iloc[:-1].values
        y_raw = future_returns.iloc[:-1].values
        
        # Simple binary classification: 1 if Up, 0 if Down
        y = np.where(y_raw > 0, 2, 0) # 2=Buy, 0=Sell (matching your Random Forest logic)
        
        self.model.train(X, y=y, metadata={'regime': self.current_regime})
        
        self.log.info("Model training complete")
    
    def _generate_signals(self) -> Dict[str, SignalOutput]:
        """Generate trading signals for all symbols using trained model."""
        signals = {}
        
        # Get current features (latest row)
        current_features = self.feature_df.iloc[-1:].values
        
        for symbol in self.sys_config.traded_symbols:
            # Model predicts signal for each symbol
            signal = self.model.predict(current_features, symbol=symbol)
            signals[symbol] = signal
            
        return signals
    
    def _execute_trades(self, signals: Dict[str, SignalOutput]):
        """Execute trades based on signals and position sizing."""
        for symbol, signal in signals.items():
            instrument_id = self.instruments.get(symbol)
            if not instrument_id:
                continue
            
            # Calculate position size using position sizer
            position = next(
                (p for p in self.account.positions() if p.instrument_id == instrument_id),
                None,
            )
            current_quantity = float(position.quantity) if position else 0.0

            
            target_quantity = self.position_sizer.calculate_size(
                signal=signal,
                current_position=current_quantity,
                portfolio_value=float(self.portfolio.account(self.account_id).equity()),
                instrument_id=instrument_id
            )
            
            # Determine order side and quantity
            quantity_delta = target_quantity - current_quantity
            
            if abs(quantity_delta) < 1e-6:  # No change needed
                continue
            
            order_side = OrderSide.BUY if quantity_delta > 0 else OrderSide.SELL
            order_quantity = abs(quantity_delta)
            
            # Submit market order
            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.id,
                instrument_id=instrument_id,
                order_side=order_side,
                quantity=Decimal(str(order_quantity)),
                time_in_force=TimeInForce.GTC,
                init_id=self.generate_order_id()
            )
            
            self.submit_order(order)
            self.log.info(f"Submitted {order_side} order for {symbol}: {order_quantity}")
    
    def _should_retrain(self) -> bool:
        """Determine if model should be retrained."""
        bars_since_retrain = self.bar_count - self.last_retrain_bar
        return bars_since_retrain >= self.sys_config.retrain_frequency
    
    def _check_regime(self) -> Any:
        """Check current market regime using regime detector."""
        if self.sys_config.regime_detector is None:
            return None
        
        market_state = {
            'raw_data': self.raw_data,
            'features': self.feature_df,
            'bar_count': self.bar_count
        }
        
        return self.sys_config.regime_detector(market_state)
    
    def _handle_regime_change(self):
        """Handle regime change event (e.g., trigger retraining)."""
        self.log.info("Handling regime change - triggering model retraining")
        self._train_model(initial=False)
        self.last_retrain_bar = self.bar_count
    
    def _save_model(self):
        """Save trained model to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.sys_config.model_save_path / f"model_{timestamp}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
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
    venue=Venue("BINANCE"),
    oms_type=OmsType.NETTING,
    account_type=AccountType.CASH,
    base_currency=USD,
    starting_balances=[Money(config.starting_capital, USD)],
    )
    
    # Add venues and instruments (simplified - extend as needed)
    # This would typically load from adapters
    for symbol in config.traded_symbols:
        instrument = Equity(
            instrument_id=InstrumentId(Symbol(symbol), Venue("BINANCE")),
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
            instrument_id = InstrumentId(Symbol(symbol), Venue("BINANCE"))
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