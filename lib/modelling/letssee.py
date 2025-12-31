import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
import os
import sys

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

# Assuming these imports exist in your environment. 
# If running standalone, replace with standard yfinance
from lib.retrieval.yahoofinance import get_trading

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    SYMBOLS: List[str] = None
    INTERVAL: str = "1d"
    LOOKBACK_DAYS: int = 1000

    # Costs
    FEE_PCT: float = 0.001     # 0.1% per trade
    SLIPPAGE_BPS: float = 5.0  # 5 bps slippage
    SHORT_COST_ANNUAL: float = 0.02
    
    # Risk / Sizing
    TARGET_VOL: float = 0.15   # 15% Annualized Vol target
    MAX_UNITS: int = 5         # Max leverage/units (allows scaling)

    # RL / Simulation
    REWARD_HORIZON: int = 5    # Days to look forward
    GAMMA: float = 0.90        # Discount factor
    NUM_PATHS: int = 10        # Number of Monte Carlo paths for reward estimation

    # Model
    MAX_DEPTH: int = 4
    NUM_LEAVES: int = 15
    LEARNING_RATE: float = 0.05
    NUM_ROUNDS: int = 200

    # Backtest
    MIN_TRAIN_DAYS: int = 252
    RETRAIN_FREQ: int = 63     # Retrain quarterly-ish

# ============================================================================
# DATA
# ============================================================================

def load_data(symbols: list, days: int, interval: str) -> pd.DataFrame:
    """Load OHLCV for multiple symbols."""
    print(f"📊 Loading data: {symbols}")
    dfs = []
    for sym in symbols:
        try:
            # Using the imported library wrapper
            data = get_trading(sym, days=days, interval=interval)
            if data.empty:
                continue
            data.columns = pd.MultiIndex.from_product([[sym], data.columns])
            dfs.append(data)
        except Exception as e:
            print(f"Error loading {sym}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, axis=1).sort_index().fillna(method='ffill')
    print(f"✓ Loaded {len(combined)} rows")
    return combined

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = pd.DataFrame(index=data.index)
    close = data[(symbol, "Close")]
    high = data[(symbol, "High")]
    low = data[(symbol, "Low")]
    volume = data[(symbol, "Volume")]

    # 1. Returns
    for lag in [1, 3, 5, 10, 21]:
        df[f"ret_{lag}"] = close.pct_change(lag)

    # 2. Volatility (Annualized)
    for w in [10, 21, 63]:
        df[f"vol_{w}"] = close.pct_change().rolling(w).std() * np.sqrt(252)

    # 3. Trends (Distance from SMA)
    for w in [21, 50, 200]:
        sma = close.rolling(w).mean()
        df[f"dist_sma_{w}"] = (close / sma) - 1

    # 4. Relative Volume
    df["rel_vol"] = volume / (volume.rolling(21).mean() + 1e-9)

    # 5. Range
    df["range"] = (high - low) / close

    # Prefix columns
    df.columns = [f"{symbol}_{c}" for c in df.columns]
    return df

def build_feature_matrix(data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    feats = [engineer_features(data, s) for s in symbols]
    return pd.concat(feats, axis=1).dropna()

# ============================================================================
# PRE-CALCULATION HELPERS
# ============================================================================

def calculate_rolling_stats(prices: pd.Series, window: int = 252) -> pd.DataFrame:
    """
    Pre-calculate Log Return Mean and Std Dev to avoid lookahead bias 
    and redundant computation during simulation.
    """
    # Use Log Returns for geometric Brownian motion
    log_rets = np.log(prices / prices.shift(1))
    
    stats = pd.DataFrame(index=prices.index)
    stats['mu'] = log_rets.rolling(window).mean()
    stats['sigma'] = log_rets.rolling(window).std()
    return stats

# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class ActionSimulator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def execute(self, equity: float, current_pos_units: int, target_pos_units: int, price: float) -> Tuple[float, int]:
        """
        Execute trade with REALISTIC cost model.
        Returns: (new_equity, new_position_units)
        """
        if current_pos_units == target_pos_units:
            return equity, current_pos_units

        # Turnover in units
        units_traded = abs(target_pos_units - current_pos_units)
        
        # Dollar value traded
        traded_value = units_traded * price
        
        # Costs
        slippage_cost = traded_value * (self.cfg.SLIPPAGE_BPS / 10000)
        fee_cost = traded_value * self.cfg.FEE_PCT
        
        total_cost = slippage_cost + fee_cost
        
        new_equity = equity - total_cost
        return new_equity, target_pos_units

    def monte_carlo_reward(
        self,
        price_now: float,
        mu: float,
        sigma: float,
        pos_units: int,
        horizon: int,
        paths: int
    ) -> float:
        """
        Correct Geometric Brownian Motion (GBM) simulation.
        Formula: S_t = S_0 * exp( (mu - 0.5*sigma^2)*t + sigma*W_t )
        """
        if pos_units == 0:
            return 0.0

        dt = 1/252
        
        # Drift adjustment for variance drag (Jensen's Inequality)
        drift = (mu - 0.5 * sigma**2) * dt
        
        # Simulation
        # shape: (paths, horizon)
        shocks = np.random.normal(0, 1, (paths, horizon))
        
        # path evolution (cumulative sum of log returns)
        log_ret_paths = np.cumsum(drift + sigma * np.sqrt(dt) * shocks, axis=1)
        
        # Price paths
        price_paths = price_now * np.exp(log_ret_paths)
        
        # Final prices at horizon
        final_prices = price_paths[:, -1]
        
        # PnL Calculation
        # PnL = Units * (Price_Final - Price_Start)
        pnl = pos_units * (final_prices - price_now)
        
        # Subtract Short Borrow Costs if short
        if pos_units < 0:
            borrow_cost = abs(pos_units * price_now) * (self.cfg.SHORT_COST_ANNUAL * horizon / 252)
            pnl -= borrow_cost

        # Return expected PnL
        return np.mean(pnl)

# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_training_data(
    features: pd.DataFrame,
    prices: pd.Series,
    rolling_stats: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    
    sim = ActionSimulator(cfg)
    
    # Target Actions: 1 (Long), 0 (Neutral), -1 (Short)
    # These represent the TARGET STATE, not the order type.
    target_states = [1, 0, -1] 
    
    rows = []
    # Ensure we don't go off the end
    valid_indices = features.index[:-cfg.REWARD_HORIZON]
    
    print("  > Generating Monte Carlo rewards...")
    
    for t in range(0, len(valid_indices), 1): # Step 1 for full data
        timestamp = valid_indices[t]
        
        # Current Market State
        current_price = prices.loc[timestamp]
        mu = rolling_stats.loc[timestamp, 'mu']
        sigma = rolling_stats.loc[timestamp, 'sigma']
        
        if pd.isna(mu) or pd.isna(sigma):
            continue

        # Feature vector
        state_features = features.loc[timestamp].copy()
        
        # For each possible TARGET state (Long, Neutral, Short)
        q_values = {}
        
        # We assume a fixed hypothetical position size for training labels 
        # (e.g., 1 unit) to normalize the regression target.
        unit_size = 1000 / current_price # Standardize to $1000 exposure

        for target_dir in target_states:
            # Calculate Expected Value (Q) of being in this state
            # Note: We are doing "Horizon Regression" here. 
            # We estimate the expected reward of holding this state for H days.
            
            ev = sim.monte_carlo_reward(
                price_now=current_price,
                mu=mu,
                sigma=sigma,
                pos_units=target_dir * unit_size,
                horizon=cfg.REWARD_HORIZON,
                paths=cfg.NUM_PATHS
            )
            
            # Discounted Reward
            q_values[f"Q_{target_dir}"] = ev * cfg.GAMMA

        # Combine features + targets
        row = pd.concat([state_features, pd.Series(q_values)])
        rows.append(row)

    return pd.DataFrame(rows).dropna()

# ============================================================================
# MODEL
# ============================================================================

class PolicyModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.models = {}
        self.features = None
        self.targets = ["Q_1", "Q_0", "Q_-1"]

    def train(self, data: pd.DataFrame):
        self.features = [c for c in data.columns if c not in self.targets]
        X = data[self.features]

        params = {
            "objective": "regression_l2", # L2 regression is robust for noisy financial data
            "max_depth": self.cfg.MAX_DEPTH,
            "num_leaves": self.cfg.NUM_LEAVES,
            "learning_rate": self.cfg.LEARNING_RATE,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": 42
        }

        for tgt in self.targets:
            y = data[tgt]
            ds = lgb.Dataset(X, y)
            self.models[tgt] = lgb.train(params, ds, num_boost_round=self.cfg.NUM_ROUNDS)

    def predict_action(self, state: pd.DataFrame) -> int:
        """Returns the optimal target direction (1, 0, -1)"""
        X = state[self.features]
        preds = {int(k.split("_")[1]): self.models[k].predict(X)[0] for k in self.models}
        
        # Select key with max value
        best_action = max(preds, key=preds.get)
        return best_action

# ============================================================================
# WALK-FORWARD BACKTEST
# ============================================================================

def walk_forward_backtest(
    features: pd.DataFrame,
    prices: pd.Series,
    cfg: Config,
) -> Dict:
    
    # Pre-calculate rolling GBM stats to avoid loop overhead
    rolling_stats = calculate_rolling_stats(prices, window=252)

    equity = 10000.0
    current_units = 0
    
    history_eq = [equity]
    history_pos = [current_units]
    
    model = None
    sim = ActionSimulator(cfg)

    # Convert index to integer location for slicing
    timestamps = features.index
    
    print(f"🚀 Starting Walk-Forward Backtest ({len(timestamps)} steps)...")

    for t in range(cfg.MIN_TRAIN_DAYS, len(timestamps) - 1):
        
        curr_time = timestamps[t]
        next_time = timestamps[t+1]
        
        # 1. Retrain Logic
        if (t - cfg.MIN_TRAIN_DAYS) % cfg.RETRAIN_FREQ == 0:
            print(f"  [Date: {curr_time.date()}] Retraining model...")
            
            # Slice history for training
            train_feats = features.iloc[:t]
            train_prices = prices.iloc[:t]
            train_stats = rolling_stats.iloc[:t]
            
            train_data = generate_training_data(train_feats, train_prices, train_stats, cfg)
            
            model = PolicyModel(cfg)
            model.train(train_data)

        # 2. Get State & Action
        state = features.iloc[[t]]
        
        # If model hasn't trained yet (shouldn't happen with logic above), hold
        if model is None:
            target_dir = 0
        else:
            target_dir = model.predict_action(state)

        # 3. Dynamic Position Sizing (Volatility Targeting)
        # Volatility of the asset
        current_vol = rolling_stats.loc[curr_time, 'sigma'] * np.sqrt(252)
        
        # Avoid division by zero
        current_vol = max(current_vol, 0.05) 
        
        # Target Exposure ($) = Equity * (Target Vol / Asset Vol)
        # "Kelly-lite" approach
        target_exposure = equity * (cfg.TARGET_VOL / current_vol)
        
        # Cap leverage
        max_exposure = equity * cfg.MAX_UNITS
        target_exposure = min(target_exposure, max_exposure)
        
        # Calculate Target Units
        price = prices.loc[curr_time]
        ideal_units = int(target_exposure / price) * target_dir
        
        # 4. Execute Trade
        equity, current_units = sim.execute(equity, current_units, ideal_units, price)

        # 5. Mark to Market (Step forward)
        next_price = prices.loc[next_time]
        pnl = current_units * (next_price - price)
        equity += pnl
        
        history_eq.append(equity)
        history_pos.append(current_units)

    return {
        "equity": history_eq,
        "positions": history_pos,
        "dates": timestamps[cfg.MIN_TRAIN_DAYS:]
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = Config(SYMBOLS=["SPY"])
    
    # 1. Load Data
    data = load_data(cfg.SYMBOLS, cfg.LOOKBACK_DAYS, cfg.INTERVAL)
    if data.empty:
        print("No data loaded.")
        return

    features = build_feature_matrix(data, cfg.SYMBOLS)
    
    # Align prices to features (drop NAs created by lags)
    features = features.dropna()
    prices = data[(cfg.SYMBOLS[0], "Close")].loc[features.index]

    # 2. Run Backtest
    results = walk_forward_backtest(features, prices, cfg)

    # 3. Report
    final_eq = results['equity'][-1]
    start_eq = results['equity'][0]
    ret = (final_eq / start_eq) - 1
    
    print("-" * 50)
    print(f"Final Equity: ${final_eq:,.2f}")
    print(f"Total Return: {ret*100:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    main()