import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import os
import sys
from datetime import datetime, timedelta, timezone
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lib.retrieval.yahoofinance import get_trading, get_option, get_option_chain, get_fundamentals

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    SYMBOLS: List[str] = None
    INTERVAL: str = "1d"
    LOOKBACK_DAYS: int = 730

    # Portfolio constraints
    PORTFOLIO_LEVERAGE: float = 1.0  # Sum of |positions| = 1.0
    
    # Costs
    FEE_FIXED: float = 1.0
    FEE_PCT: float = 0.001
    SLIPPAGE_BPS: float = 5.0
    SHORT_COST_ANNUAL: float = 0.02
    TRADE_PENALTY: float = 0.002

    # Training
    MIN_TRAIN_DAYS: int = 252
    RETRAIN_FREQ: int = 21
    
    # Model ensemble
    NUM_MODELS: int = 5        # Reduced for faster training
    MAX_DEPTH: int = 3
    NUM_LEAVES: int = 4
    LEARNING_RATE: float = 0.01
    NUM_ROUNDS: int = 100
    
    # Feature selection
    FEATURE_SELECTION: bool = True
    MAX_FEATURES: int = 50


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(symbols: list, days: int, interval: str) -> pd.DataFrame:
    """Load OHLCV for multiple symbols."""
    print(f"📊 Loading data: {symbols}")
    dfs = []
    for sym in symbols:
        data = get_trading(sym, days=days, interval=interval)
        if data.empty:
            continue
            
        # Normalize columns
        clean_columns = []
        for c in data.columns:
            if isinstance(c, tuple):
                clean_columns.append(c[0])
            else:
                clean_columns.append(c)
        
        # Create clean (Feature, Symbol) MultiIndex
        data.columns = pd.MultiIndex.from_tuples([(c, sym) for c in clean_columns])
        dfs.append(data)
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, axis=1).sort_index().ffill()
    print(f"✓ Loaded {len(combined)} rows")
    return combined

# ============================================================================
# COMPREHENSIVE FEATURE ENGINEERING
# ============================================================================

def engineer_features(data: pd.DataFrame, symbol: str, all_symbols: List[str]) -> pd.DataFrame:
    """Generate extensive feature set for a single symbol."""
    df = pd.DataFrame(index=data.index)
    
    close = data[("Close", symbol)]
    high = data[("High", symbol)]
    low = data[("Low", symbol)]
    volume = data[("Volume", symbol)]
    
    # === MOMENTUM FEATURES ===
    for lag in [1, 2, 3, 5, 10, 21, 42, 63]:
        df[f"ret_{lag}"] = close.pct_change(lag)
    
    # Acceleration (change in momentum)
    df["ret_accel_5"] = df["ret_5"] - df["ret_5"].shift(5)
    df["ret_accel_21"] = df["ret_21"] - df["ret_21"].shift(21)
    
    # === VOLATILITY FEATURES ===
    returns = close.pct_change()
    for w in [5, 10, 21, 42, 63]:
        df[f"vol_{w}"] = returns.rolling(w).std()
        df[f"vol_rank_{w}"] = df[f"vol_{w}"].rolling(min(252, len(close))).rank(pct=True)
    
    # Parkinson volatility (uses high-low)
    for w in [5, 21]:
        hl = np.log(high / low)
        df[f"parkinson_vol_{w}"] = np.sqrt(hl.pow(2).rolling(w).mean() / (4 * np.log(2)))
    
    # === TREND FEATURES ===
    for w in [10, 21, 50, 100, 200]:
        sma = close.rolling(w).mean()
        df[f"sma_{w}"] = sma
        df[f"price_sma_{w}"] = close / sma - 1
        df[f"sma_slope_{w}"] = (sma - sma.shift(5)) / sma.shift(5)
    
    # EMA
    for w in [12, 26, 50]:
        ema = close.ewm(span=w, adjust=False).mean()
        df[f"price_ema_{w}"] = close / ema - 1
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # === MEAN REVERSION FEATURES ===
    for w in [10, 21, 42]:
        mean = close.rolling(w).mean()
        std = close.rolling(w).std()
        df[f"zscore_{w}"] = (close - mean) / (std + 1e-9)
        df[f"rsi_{w}"] = compute_rsi(close, w)
    
    # Bollinger Bands
    for w in [20, 50]:
        sma = close.rolling(w).mean()
        std = close.rolling(w).std()
        df[f"bb_upper_{w}"] = (close - (sma + 2*std)) / close
        df[f"bb_lower_{w}"] = (close - (sma - 2*std)) / close
    
    # === VOLUME FEATURES ===
    df["rel_vol"] = volume / volume.rolling(21).mean()
    df["vol_trend"] = volume.rolling(21).mean() / volume.rolling(63).mean() - 1
    
    # Volume-weighted returns
    for w in [5, 21]:
        vol_ret = (returns * volume).rolling(w).sum() / volume.rolling(w).sum()
        df[f"vwap_ret_{w}"] = vol_ret
    
    # Money flow
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    for w in [10, 21]:
        pos_flow = (money_flow * (typical_price > typical_price.shift(1))).rolling(w).sum()
        neg_flow = (money_flow * (typical_price <= typical_price.shift(1))).rolling(w).sum()
        df[f"mfi_{w}"] = 100 - 100 / (1 + pos_flow / (neg_flow + 1e-9))
    
    # === RANGE & MICROSTRUCTURE ===
    df["range"] = (high - low) / close
    df["range_rank"] = df["range"].rolling(min(252, len(close))).rank(pct=True)
    
    # Gap
    df["gap"] = (close - close.shift(1).shift(-1)) / close.shift(1).shift(-1)
    
    # High/Low position
    df["hl_position"] = (close - low) / (high - low + 1e-9)
    
    # === REGIME FEATURES ===
    # Trend strength
    for w in [10, 21]:
        df[f"trend_strength_{w}"] = abs(df[f"price_sma_{w}"])
    
    # Volatility regime
    vol_21 = returns.rolling(21).std()
    df["vol_regime"] = vol_21.rolling(min(252, len(close))).rank(pct=True)
    
    # Drawdown
    rolling_max = close.rolling(min(252, len(close)), min_periods=1).max()
    df["drawdown"] = (close / rolling_max - 1)
    
    # === CROSS-ASSET FEATURES ===
    if len(all_symbols) > 1:
        other_symbols = [s for s in all_symbols if s != symbol]
        for other in other_symbols[:3]:
            if ("Close", other) in data.columns:
                other_close = data[("Close", other)]
                other_ret = other_close.pct_change()
                
                # Correlation
                for w in [21, 63]:
                    corr = returns.rolling(w).corr(other_ret)
                    df[f"corr_{other}_{w}"] = corr
                
                # Relative strength
                df[f"rel_strength_{other}"] = close / other_close
    
    # === CALENDAR FEATURES ===
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    
    # Add symbol prefix
    df.columns = [f"{symbol}_{c}" for c in df.columns]
    
    return df


def compute_rsi(close: pd.Series, window: int) -> pd.Series:
    """Compute RSI indicator."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def build_feature_matrix(data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """Build comprehensive feature matrix for all symbols."""
    print("🔧 Engineering features...")
    feats = [engineer_features(data, s, symbols) for s in symbols]
    combined = pd.concat(feats, axis=1)
    
    # CRITICAL FIX: Use forward fill instead of dropna
    combined = combined.ffill().bfill()
    
    # Only drop rows where ALL values are NaN
    combined = combined.dropna(how='all')
    
    print(f"✓ Feature matrix: {combined.shape[0]} rows × {combined.shape[1]} columns")
    print(f"  NaN count: {combined.isna().sum().sum()}")
    
    # Fill any remaining NaNs with 0
    combined = combined.fillna(0)
    
    return combined


# ============================================================================
# AI-POWERED FEATURE SELECTION
# ============================================================================

class FeatureSelector:
    """Use LightGBM feature importance for automated feature selection."""
    
    def __init__(self, max_features: int = 50):
        self.max_features = max_features
        self.selected_features = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train a model and extract top features by importance."""
        
        # Filter out features with zero variance
        var = X.var()
        valid_features = var[var > 1e-10].index
        X = X[valid_features]
        
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )
        
        model.fit(X, y)
        
        # Get feature importances
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        self.selected_features = importance.head(self.max_features)['feature'].tolist()
        
        print(f"  ✓ Selected {len(self.selected_features)} features")
        print(f"    Top 5: {self.selected_features[:5]}")
        
        return self.selected_features
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter to selected features."""
        if self.selected_features is None:
            raise ValueError("Must call fit() first")
        return X[self.selected_features]


# ============================================================================
# NO FORWARD-LOOKING BIAS: REWARD FROM PAST EXPERIENCE
# ============================================================================

class ExperienceBuffer:
    """Store past (state, action, reward, next_state) transitions."""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = []
        self.max_size = max_size
    
    def add(self, state: np.ndarray, action: str, position: float, 
            reward: float, next_state: np.ndarray, feature_names: List[str] = None):
        """Add experience tuple."""
        self.buffer.append({
            'state': state,
            'action': action,
            'position': position,
            'reward': reward,
            'next_state': next_state,
            'feature_names': feature_names
        })
        
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def sample(self, n: int) -> List[Dict]:
        """Sample random experiences."""
        indices = np.random.choice(len(self.buffer), min(n, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_all(self) -> List[Dict]:
        """Get all experiences."""
        return self.buffer


# ============================================================================
# ACTION SIMULATOR WITH CONTINUOUS POSITIONS
# ============================================================================

class ActionSimulator:
    """Execute trades with realistic costs and continuous position sizing."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
    
    def execute_action(
        self, 
        equity: float,
        current_position: float,
        action: str,
        volatility: float,
        confidence: float = 0.5
    ) -> Tuple[float, float]:
        """Execute an action and return new equity and position."""
        
        if action == "HOLD":
            return equity, current_position
        
        elif action == "FLATTEN":
            target_position = 0.0
        
        elif action == "BUY":
            vol_scale = np.clip(0.02 / (volatility + 1e-9), 0.1, 2.0)
            target_position = confidence * vol_scale
            target_position = np.clip(target_position, 0.0, 0.99)
        
        elif action == "SELL":
            vol_scale = np.clip(0.02 / (volatility + 1e-9), 0.1, 2.0)
            target_position = -confidence * vol_scale
            target_position = np.clip(target_position, -0.99, 0.0)
        
        else:
            raise ValueError(f"Unknown action: {action}")
        
        # Calculate trade cost
        trade_size = abs(target_position - current_position)
        if trade_size > 0.01:
            slippage = self.cfg.SLIPPAGE_BPS / 10000 * trade_size
            fees = self.cfg.FEE_PCT * trade_size + self.cfg.FEE_FIXED / equity
            penalty = self.cfg.TRADE_PENALTY * trade_size
            total_cost = slippage + fees + penalty
            equity *= (1 - total_cost)
        
        return equity, target_position


# ============================================================================
# PORTFOLIO ALLOCATOR
# ============================================================================

class PortfolioAllocator:
    """Normalize positions so sum of |positions| = leverage target."""
    
    def __init__(self, leverage: float = 1.0):
        self.leverage = leverage
    
    def allocate(self, raw_positions: Dict[str, float]) -> Dict[str, float]:
        """Normalize positions to meet leverage constraint."""
        total_abs = sum(abs(p) for p in raw_positions.values())
        
        if total_abs < 1e-6:
            return {s: 0.0 for s in raw_positions}
        
        scale = self.leverage / total_abs
        return {s: p * scale for s, p in raw_positions.items()}


# ============================================================================
# ENSEMBLE MODEL WITH UNCERTAINTY QUANTIFICATION
# ============================================================================

class EnsembleQModel:
    """Ensemble of LightGBM models for Q-value estimation with uncertainty."""
    
    def __init__(self, cfg: Config, n_models: int = 5):
        self.cfg = cfg
        self.n_models = n_models
        self.models = {action: [] for action in ["BUY", "SELL", "HOLD", "FLATTEN"]}
        self.features = None
    
    def train(self, experiences: List[Dict], feature_selector: Optional[FeatureSelector] = None):
        """Train ensemble models on experience replay buffer."""
        
        # Convert experiences to training data
        states = []
        actions_taken = []
        rewards = []
        feature_names = None
        
        for exp in experiences:
            states.append(exp['state'])
            actions_taken.append(exp['action'])
            rewards.append(exp['reward'])
            if feature_names is None and 'feature_names' in exp:
                feature_names = exp['feature_names']
        
        # Create DataFrame with proper column names
        X = pd.DataFrame(states, columns=feature_names)
        
        # Feature selection
        if feature_selector and self.cfg.FEATURE_SELECTION and len(X) > 100:
            feature_selector.fit(X, pd.Series(rewards))
            X = feature_selector.transform(X)
        
        self.features = X.columns.tolist()
        
        # Train separate ensemble for each action
        for action in self.models.keys():
            action_mask = np.array(actions_taken) == action
            if action_mask.sum() < 10:
                print(f"    ⚠️  {action}: Only {action_mask.sum()} samples, skipping")
                continue
            
            X_action = X[action_mask]
            y_action = np.array(rewards)[action_mask]
            
            # Train ensemble
            for i in range(self.n_models):
                # Bootstrap sample
                indices = np.random.choice(len(X_action), len(X_action), replace=True)
                X_boot = X_action.iloc[indices]
                y_boot = y_action[indices]
                
                # Time-based split
                split_idx = int(len(X_boot) * 0.9)
                X_train, X_val = X_boot.iloc[:split_idx], X_boot.iloc[split_idx:]
                y_train, y_val = y_boot[:split_idx], y_boot[split_idx:]
                
                if len(X_val) < 2:
                    X_train, X_val = X_boot, X_boot
                    y_train, y_val = y_boot, y_boot
                
                ds_train = lgb.Dataset(X_train, y_train)
                ds_val = lgb.Dataset(X_val, y_val, reference=ds_train)
                
                params = {
                    "objective": "regression",
                    "metric": "mse",
                    "max_depth": self.cfg.MAX_DEPTH,
                    "num_leaves": self.cfg.NUM_LEAVES,
                    "learning_rate": self.cfg.LEARNING_RATE,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "seed": 42 + i,
                    "verbosity": -1,
                }
                
                model = lgb.train(
                    params,
                    ds_train,
                    valid_sets=[ds_val],
                    num_boost_round=self.cfg.NUM_ROUNDS,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )
                
                self.models[action].append(model)
            
            print(f"    ✓ {action}: Trained {len(self.models[action])} models on {len(X_action)} samples")
    
    def predict(self, state: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Predict Q-values and uncertainties for all actions."""
        # Ensure state has the same features used during training
        if not all(f in state.columns for f in self.features):
            missing = [f for f in self.features if f not in state.columns]
            raise ValueError(f"Missing features: {missing[:5]}...")
        
        X = state[self.features]
        
        q_values = {}
        uncertainties = {}
        
        for action, models in self.models.items():
            if len(models) == 0:
                q_values[action] = 0.0
                uncertainties[action] = 1.0
                continue
            
            predictions = [m.predict(X)[0] for m in models]
            q_values[action] = np.mean(predictions)
            uncertainties[action] = np.std(predictions)
        
        return q_values, uncertainties
    
    def act(self, state: pd.DataFrame, exploration: float = 0.0) -> Tuple[str, float]:
        """Select action using Upper Confidence Bound (UCB)."""
        q_values, uncertainties = self.predict(state)
        
        # UCB: Q(s,a) + c * uncertainty(s,a)
        ucb_values = {
            a: q_values[a] + exploration * uncertainties[a]
            for a in q_values
        }
        
        best_action = max(ucb_values, key=ucb_values.get)
        
        # Confidence based on Q-value
        q_range = max(q_values.values()) - min(q_values.values())
        confidence = np.clip(
            (q_values[best_action] - min(q_values.values())) / (q_range + 1e-9),
            0.0, 1.0
        )
        
        return best_action, confidence


# ============================================================================
# WALK-FORWARD BACKTEST WITH EXPERIENCE REPLAY
# ============================================================================

def walk_forward_backtest(
    features: pd.DataFrame,
    prices: Dict[str, pd.Series],
    cfg: Config,
) -> Dict:
    """Walk-forward backtest with NO forward-looking bias."""
    
    symbols = list(prices.keys())
    equity = 100.0
    positions = {s: 0.0 for s in symbols}
    
    # History tracking
    equity_history = [equity]
    position_history = [{s: 0.0 for s in symbols}]
    
    # Experience buffers
    buffers = {s: ExperienceBuffer(max_size=5000) for s in symbols}
    
    # Models
    models = {s: None for s in symbols}
    
    # Simulators and allocators
    simulators = {s: ActionSimulator(cfg) for s in symbols}
    allocator = PortfolioAllocator(cfg.PORTFOLIO_LEVERAGE)
    
    # Feature selector
    feature_selector = FeatureSelector(cfg.MAX_FEATURES) if cfg.FEATURE_SELECTION else None
    
    min_t = cfg.MIN_TRAIN_DAYS
    
    print(f"\n🚀 Starting backtest from t={min_t} to t={len(features)-1}")
    print(f"   Initial equity: ${equity:.2f}\n")
    
    for t in range(min_t, len(features) - 1):
        
        # === RETRAIN MODELS ===
        if (t - min_t) % cfg.RETRAIN_FREQ == 0:
            print(f"🔄 Retraining at t={t} ({features.index[t].date()})")
            
            for symbol in symbols:
                experiences = buffers[symbol].get_all()
                
                if len(experiences) < 50:
                    print(f"  ⚠️  {symbol}: Not enough experiences ({len(experiences)})")
                    continue
                
                print(f"  📊 {symbol}: Training on {len(experiences)} experiences")
                model = EnsembleQModel(cfg, n_models=cfg.NUM_MODELS)
                model.train(experiences, feature_selector)
                models[symbol] = model
        
        # === PREDICT ACTIONS FOR EACH SYMBOL ===
        raw_positions = {}
        
        for symbol in symbols:
            # Get state features
            state_cols = [c for c in features.columns if c.startswith(f"{symbol}_")]
            state = features.iloc[[t]][state_cols].copy()
            state[f"{symbol}_position"] = positions[symbol]
            
            # Calculate recent volatility
            recent_returns = prices[symbol].iloc[max(0, t-21):t].pct_change()
            volatility = recent_returns.std() if len(recent_returns) > 0 else 0.02
            
            if models[symbol] is None:
                # Initial exploration
                target_position = np.random.uniform(-0.5, 0.5)
                raw_positions[symbol] = target_position
                continue
            
            # Predict action
            exploration = max(0.1, 1.0 - (t - min_t) / (len(features) - min_t))
            action, confidence = models[symbol].act(state, exploration=exploration)
            
            # Execute action
            equity, new_position = simulators[symbol].execute_action(
                equity, positions[symbol], action, volatility, confidence
            )
            
            raw_positions[symbol] = new_position
        
        # === PORTFOLIO ALLOCATION ===
        positions = allocator.allocate(raw_positions)
        
        # === COMPUTE REALIZED RETURNS AND STORE EXPERIENCES ===
        next_t = t + 1
        total_pnl = 0.0
        
        for symbol in symbols:
            # Compute realized PnL
            price_return = prices[symbol].iloc[next_t] / prices[symbol].iloc[t] - 1
            position_pnl = price_return * positions[symbol]
            
            # Short cost
            if positions[symbol] < 0:
                position_pnl -= cfg.SHORT_COST_ANNUAL / 252 * abs(positions[symbol])
            
            total_pnl += position_pnl
            
            # Store experience
            state_cols = [c for c in features.columns if c.startswith(f"{symbol}_")]
            state = features.iloc[t][state_cols].values
            next_state = features.iloc[next_t][state_cols].values
            
            pos = positions[symbol]
            if abs(pos) < 0.01:
                action_taken = "FLATTEN"
            elif pos > 0:
                action_taken = "BUY"
            else:
                action_taken = "SELL"
            
            buffers[symbol].add(
                state=state,
                action=action_taken,
                position=pos,
                reward=position_pnl,
                next_state=next_state,
                feature_names=state_cols
            )
        
        # Update equity
        equity *= (1 + total_pnl)
        
        equity_history.append(equity)
        position_history.append(positions.copy())
        
        # Progress
        if t % 50 == 0 or t == len(features) - 2:
            print(f"  t={t:4d} | equity=${equity:7.2f} | positions={positions}")
    
    return {
        "equity": equity_history,
        "positions": position_history,
        "final_equity": equity_history[-1]
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = Config(
        SYMBOLS=["SPY", "QQQ"],
        FEATURE_SELECTION=True,
        MAX_FEATURES=50,
        MIN_TRAIN_DAYS=100,  # Reduced for faster startup
        RETRAIN_FREQ=21
    )
    
    # Load data
    data = load_data(cfg.SYMBOLS, cfg.LOOKBACK_DAYS, cfg.INTERVAL)
    
    if data.empty:
        print("❌ No data loaded!")
        return
    
    # Build features
    features = build_feature_matrix(data, cfg.SYMBOLS)
    
    if len(features) < cfg.MIN_TRAIN_DAYS + 10:
        print(f"❌ Not enough data! Need {cfg.MIN_TRAIN_DAYS + 10}, got {len(features)}")
        return
    
    # Extract prices
    prices = {sym: data[("Close", sym)].loc[features.index] for sym in cfg.SYMBOLS}
    
    # Run backtest
    results = walk_forward_backtest(features, prices, cfg)
    
    print(f"\n{'='*60}")
    print(f"🎯 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Final equity:  ${results['final_equity']:.2f}")
    print(f"Total return:  {(results['final_equity'] - 100) / 100 * 100:+.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()