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
from lib.retrieval.yfinance import get_trading

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    SYMBOLS: List[str] = None
    INTERVAL: str = "1d"
    LOOKBACK_DAYS: int = 1000

    # Costs
    FEE_PCT: float = 0.001      # 0.1% per trade
    SLIPPAGE_BPS: float = 5.0   # 5 bps slippage
    SHORT_COST_ANNUAL: float = 0.02
    
    # Risk / Sizing
    TARGET_VOL: float = 0.15
    MAX_UNITS: int = 5

    # RL / Simulation
    REWARD_HORIZON: int = 5     # Look forward 5 days
    GAMMA: float = 0.90         # Discount future rewards
    NUM_PATHS: int = 20         # Robust Monte Carlo

    # Model
    MAX_DEPTH: int = 6
    NUM_LEAVES: int = 31
    LEARNING_RATE: float = 0.03
    NUM_ROUNDS: int = 300

    # Backtest
    MIN_TRAIN_DAYS: int = 252
    RETRAIN_FREQ: int = 63

# ============================================================================
# DATA & FEATURES
# ============================================================================

def load_data(symbols: list, days: int, interval: str) -> pd.DataFrame:
    print(f"📊 Loading data: {symbols}")
    dfs = []
    for sym in symbols:
        try:
            data = get_trading(sym, days=days, interval=interval)
            if data.empty: continue
            data.columns = pd.MultiIndex.from_product([[sym], data.columns])
            dfs.append(data)
        except: continue
    
    if not dfs: return pd.DataFrame()
    return pd.concat(dfs, axis=1).sort_index().fillna(method='ffill')

def engineer_features(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = pd.DataFrame(index=data.index)
    close = data[(symbol, "Close")]
    high = data[(symbol, "High")]
    low = data[(symbol, "Low")]
    volume = data[(symbol, "Volume")]

    # Returns & Volatility
    for lag in [1, 3, 5, 10, 21]:
        df[f"ret_{lag}"] = close.pct_change(lag)
    
    for w in [10, 21, 63]:
        df[f"vol_{w}"] = close.pct_change().rolling(w).std() * np.sqrt(252)

    # Technicals
    for w in [21, 50, 200]:
        sma = close.rolling(w).mean()
        df[f"dist_sma_{w}"] = (close / sma) - 1

    df["rsi_14"] = 100 - (100 / (1 + close.diff().clip(lower=0).rolling(14).mean() / close.diff().clip(upper=0).abs().rolling(14).mean()))
    df["range"] = (high - low) / close
    df["rel_vol"] = volume / (volume.rolling(21).mean() + 1e-9)

    df.columns = [f"{symbol}_{c}" for c in df.columns]
    return df

def build_feature_matrix(data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    feats = [engineer_features(data, s) for s in symbols]
    return pd.concat(feats, axis=1).dropna()

def calculate_rolling_stats(prices: pd.Series, window: int = 252) -> pd.DataFrame:
    # Use Log Returns for correct GBM math
    log_rets = np.log(prices / prices.shift(1))
    stats = pd.DataFrame(index=prices.index)
    stats['mu'] = log_rets.rolling(window).mean()
    stats['sigma'] = log_rets.rolling(window).std()
    return stats

# ============================================================================
# ACTION & REWARD LOGIC
# ============================================================================

class ActionSimulator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def calculate_cost(self, current_units: float, target_units: float, price: float) -> float:
        """Calculate exact transaction cost (Fee + Slippage) to move from A to B."""
        if current_units == target_units:
            return 0.0
        
        # Calculate Turnover
        units_traded = abs(target_units - current_units)
        traded_value = units_traded * price
        
        # Slippage + Fee
        slippage = traded_value * (self.cfg.SLIPPAGE_BPS / 10000)
        fee = traded_value * self.cfg.FEE_PCT
        
        return slippage + fee

    def simulate_future_price(self, start_price, mu, sigma, horizon, paths):
        """Vectorized GBM Simulation."""
        dt = 1/252
        # Variance Drag Adjustment: mu - 0.5*sigma^2
        drift = (mu - 0.5 * sigma**2) * dt
        
        # Shocks: (Paths, Horizon)
        shocks = np.random.normal(0, 1, (paths, horizon))
        log_path = np.cumsum(drift + sigma * np.sqrt(dt) * shocks, axis=1)
        
        # Price paths
        prices = start_price * np.exp(log_path)
        return prices[:, -1] # Return only the final prices at horizon

# ============================================================================
# Q-LEARNING DATASET GENERATOR
# ============================================================================

def generate_q_data(features: pd.DataFrame, prices: pd.Series, stats: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sim = ActionSimulator(cfg)
    rows = []
    
    # We define explicit actions and their resulting target state (Direction)
    # HOLD is dynamic (depends on current state)
    ACTIONS = ['GO_LONG', 'GO_SHORT', 'FLATTEN', 'HOLD']
    
    # We simulate 3 distinct starting states for the model to learn context
    START_POSITIONS = [-1, 0, 1] # Short, Neutral, Long
    
    # Standardize unit size for reward calculation ($10,000 equivalent)
    BASE_EQUITY = 10_000
    
    valid_idx = features.index[:-cfg.REWARD_HORIZON]
    
    print(f"  > Generating Q-values for {len(valid_idx)} steps x 3 states...")

    for t in range(0, len(valid_idx)):
        timestamp = valid_idx[t]
        price = prices.loc[timestamp]
        mu = stats.loc[timestamp, 'mu']
        sigma = stats.loc[timestamp, 'sigma']
        
        if pd.isna(mu) or pd.isna(sigma): continue

        # Simulate Future Prices Once
        final_prices = sim.simulate_future_price(price, mu, sigma, cfg.REWARD_HORIZON, cfg.NUM_PATHS)
        
        # Base Feature Vector
        base_feats = features.loc[timestamp].copy()
        
        # For each possible EXISTING portfolio state
        for start_pos_dir in START_POSITIONS:
            
            current_units = start_pos_dir * (BASE_EQUITY / price)
            
            # Create a training row
            row_feats = base_feats.copy()
            # CRITICAL: Input the current position so model knows if it needs to trade
            row_feats['current_pos'] = start_pos_dir 
            
            q_vals = {}
            
            for action in ACTIONS:
                # Determine Target
                if action == 'GO_LONG': target_dir = 1
                elif action == 'GO_SHORT': target_dir = -1
                elif action == 'FLATTEN': target_dir = 0
                elif action == 'HOLD': target_dir = start_pos_dir # Target is Current
                
                target_units = target_dir * (BASE_EQUITY / price)
                
                # 1. Calculate Transaction Cost (Immediate Negative Reward)
                cost = sim.calculate_cost(current_units, target_units, price)
                
                # 2. Calculate Expected PnL (Future Reward)
                # PnL = Units * (Price_T - Price_0)
                gross_pnl_paths = target_units * (final_prices - price)
                
                # Apply Short Borrow Cost
                if target_dir < 0:
                    borrow = abs(target_units * price) * (cfg.SHORT_COST_ANNUAL * cfg.REWARD_HORIZON / 252)
                    gross_pnl_paths -= borrow
                
                exp_pnl = np.mean(gross_pnl_paths)
                
                # 3. Total Q-Value = (Expected PnL * Gamma) - Transaction Cost
                # We subtract cost immediately.
                q_value = (exp_pnl * cfg.GAMMA) - cost
                
                q_vals[f"Q_{action}"] = q_value
            
            # Append training row
            full_row = pd.concat([row_feats, pd.Series(q_vals)])
            rows.append(full_row)
            
    return pd.DataFrame(rows).dropna()

# ============================================================================
# MODEL WRAPPER
# ============================================================================

class ActionModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.models = {}
        self.features = None
        self.actions = ['Q_GO_LONG', 'Q_GO_SHORT', 'Q_FLATTEN', 'Q_HOLD']

    def train(self, data: pd.DataFrame):
        self.features = [c for c in data.columns if c not in self.actions]
        X = data[self.features]
        
        params = {
            "objective": "regression",
            "max_depth": self.cfg.MAX_DEPTH,
            "num_leaves": self.cfg.NUM_LEAVES,
            "learning_rate": self.cfg.LEARNING_RATE,
            "bagging_fraction": 0.8,
            "feature_fraction": 0.8,
            "verbosity": -1,
            "seed": 42
        }

        for act in self.actions:
            y = data[act]
            ds = lgb.Dataset(X, y)
            self.models[act] = lgb.train(params, ds, num_boost_round=self.cfg.NUM_ROUNDS)

    def select_action(self, state_features: pd.Series, current_pos_dir: int) -> str:
        # Inject current position into features
        input_data = state_features.copy()
        input_data['current_pos'] = current_pos_dir
        
        # Predict Q for all actions
        # Reshape for single prediction
        X = pd.DataFrame([input_data])[self.features]
        
        preds = {k: self.models[k].predict(X)[0] for k in self.models}
        
        # Return action with highest Q-value
        best_q = max(preds, key=preds.get)
        return best_q.replace("Q_", "")

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_backtest(features: pd.DataFrame, prices: pd.Series, cfg: Config) -> Dict:
    stats = calculate_rolling_stats(prices)
    sim = ActionSimulator(cfg)
    
    equity = 10_000.0
    units = 0.0
    
    history = {'equity': [], 'units': [], 'actions': []}
    
    model = None
    
    print(f"🚀 Starting Backtest...")
    
    # Indices
    times = features.index
    
    for t in range(cfg.MIN_TRAIN_DAYS, len(times) - 1):
        now = times[t]
        next_dt = times[t+1]
        price = prices.loc[now]
        
        # 1. RETRAIN
        if (t - cfg.MIN_TRAIN_DAYS) % cfg.RETRAIN_FREQ == 0:
            print(f"  [Date: {now.date()}] Retraining Model...")
            train_feats = features.iloc[:t]
            train_prices = prices.iloc[:t]
            train_stats = stats.iloc[:t]
            
            train_df = generate_q_data(train_feats, train_prices, train_stats, cfg)
            model = ActionModel(cfg)
            model.train(train_df)

        # 2. DETERMINE CURRENT STATE
        # Normalize current units to Direction (-1, 0, 1) for the model input
        if units == 0: curr_dir = 0
        else: curr_dir = 1 if units > 0 else -1
        
        # 3. SELECT ACTION
        if model is None:
            action = "HOLD"
        else:
            action = model.select_action(features.loc[now], curr_dir)
            
        history['actions'].append(action)
        
        # 4. MAP ACTION TO TARGET UNITS
        # We need to calculate target size based on volatility
        vol = stats.loc[now, 'sigma'] * np.sqrt(252)
        vol = max(vol, 0.05) # Floor vol
        
        target_exposure = equity * (cfg.TARGET_VOL / vol)
        target_exposure = min(target_exposure, equity * cfg.MAX_UNITS)
        
        # Interpret Action
        if action == "GO_LONG":
            target_units = target_exposure / price
        elif action == "GO_SHORT":
            target_units = -1 * (target_exposure / price)
        elif action == "FLATTEN":
            target_units = 0
        elif action == "HOLD":
            # Keep exact current units
            target_units = units 

        # 5. EXECUTE
        # Calculate costs
        cost_val = sim.calculate_cost(units, target_units, price)
        equity -= cost_val
        units = target_units
        
        # 6. MARK TO MARKET
        next_price = prices.loc[next_dt]
        pnl = units * (next_price - price)
        equity += pnl
        
        history['equity'].append(equity)
        history['units'].append(units)

    return history

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    cfg = Config(SYMBOLS=["SPY"])
    
    # Load
    raw = load_data(cfg.SYMBOLS, cfg.LOOKBACK_DAYS, cfg.INTERVAL)
    if raw.empty: return
    
    # Feature Engineering
    X = build_feature_matrix(raw, cfg.SYMBOLS)
    X = X.dropna()
    
    # Align Prices
    P = raw[(cfg.SYMBOLS[0], "Close")].loc[X.index]
    
    # Run
    res = run_backtest(X, P, cfg)
    
    # Stats
    final_eq = res['equity'][-1]
    ret = (final_eq / 10000) - 1
    trades = len([x for x in res['actions'] if x != 'HOLD'])
    
    print("\n" + "="*30)
    print(f"Initial Equity: $10,000.00")
    print(f"Final Equity:   ${final_eq:,.2f}")
    print(f"Return:         {ret*100:.2f}%")
    print(f"Total Trades:   {trades}")
    print("="*30)

if __name__ == "__main__":
    main()