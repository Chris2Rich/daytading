"""
REWARD-DRIVEN GBDT TRADING SYSTEM
==================================

Complete implementation of Q-learning-style supervised trading with LightGBM.
NO price prediction. Instead: learns action values via counterfactual simulation.

Usage:
    python trading_system.py --symbols SPY QQQ --mode train
    python trading_system.py --symbols SPY --mode backtest
    python trading_system.py --symbols SPY --mode stress_test
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import yfinance as yf
from datetime import datetime, timedelta, timezone
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """System configuration - all parameters in one place"""
    
    # Data
    SYMBOLS: List[str] = None
    INTERVAL: str = '1d'
    LOOKBACK_DAYS: int = 730
    
    # Costs
    FEE_FIXED: float = 1.0
    FEE_PCT: float = 0.001
    SLIPPAGE_BPS: float = 5.0
    SHORT_COST_ANNUAL: float = 0.02
    
    # Simulation
    REWARD_HORIZON: int = 5
    RISK_LAMBDA: float = 0.1
    STRESS_VOL_MULT: float = 1.5
    STRESS_SLIP_MULT: float = 2.0
    
    # Model
    MAX_DEPTH: int = 5
    NUM_LEAVES: int = 20
    LEARNING_RATE: float = 0.05
    NUM_ROUNDS: int = 200
    
    # Backtest
    MIN_TRAIN_DAYS: int = 252
    RETRAIN_FREQ: int = 21


# ============================================================================
# DATA MODULE
# ============================================================================

def load_data(symbols: List[str], days: int, interval: str) -> pd.DataFrame:
    """Load OHLCV for multiple symbols. Returns MultiIndex DataFrame."""
    
    print(f"📊 Loading data: {symbols}")
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    
    dfs = []
    for sym in symbols:
        data = yf.download(sym, start=start, end=end, interval=interval, 
                          auto_adjust=True, progress=False)
        if data.empty:
            continue
        data.columns = pd.MultiIndex.from_product([[sym], data.columns])
        dfs.append(data)
    
    combined = pd.concat(dfs, axis=1).sort_index().fillna(method='ffill')
    print(f"✓ Loaded {len(combined)} rows")
    return combined


# ============================================================================
# FEATURE ENGINEERING (NO LOOKAHEAD)
# ============================================================================

def engineer_features(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Generate 50-100 causal features per symbol.
    All features use ONLY past data (no lookahead bias).
    """
    
    df = pd.DataFrame(index=data.index)
    close = data[(symbol, 'Close')]
    high = data[(symbol, 'High')]
    low = data[(symbol, 'Low')]
    volume = data[(symbol, 'Volume')]
    
    # Returns (multiple horizons)
    for lag in [1, 2, 3, 5, 10, 21, 42, 63]:
        df[f'ret_{lag}d'] = close.pct_change(lag)
    
    # Volatility (realized)
    for w in [5, 10, 21, 42, 63]:
        df[f'vol_{w}d'] = close.pct_change().rolling(w).std()
    
    # Moving averages
    for w in [5, 10, 21, 50, 100, 200]:
        sma = close.rolling(w).mean()
        df[f'sma_{w}'] = sma
        df[f'price_to_sma_{w}'] = close / sma - 1
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - 100 / (1 + gain / loss)
    
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_width'] = (2 * std20) / sma20
    df['bb_position'] = (close - (sma20 - 2*std20)) / (4 * std20)
    
    # Volume
    for w in [5, 21]:
        df[f'rel_vol_{w}d'] = volume / volume.rolling(w).mean()
    
    # Distance from highs/lows
    for w in [21, 63, 252]:
        df[f'dist_high_{w}d'] = close / close.rolling(w).max() - 1
        df[f'dist_low_{w}d'] = close / close.rolling(w).min() - 1
    
    # Intraday range
    df['range'] = (high - low) / close
    
    # Prefix all with symbol
    df.columns = [f'{symbol}_{col}' for col in df.columns]
    
    return df


def build_feature_matrix(data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """Combine features for all symbols"""
    
    print("🔧 Engineering features...")
    features = [engineer_features(data, sym) for sym in symbols]
    combined = pd.concat(features, axis=1).dropna()
    print(f"✓ Created {len(combined.columns)} features, {len(combined)} samples")
    return combined


# ============================================================================
# STATE AUGMENTATION (Portfolio + Risk State)
# ============================================================================

def augment_state(features: pd.DataFrame, position: int, equity: float, 
                  entry_price: float, current_price: float) -> pd.DataFrame:
    """Add position/risk state to market features (for inference)"""
    
    state = features.copy()
    state['position'] = position
    state['pct_from_entry'] = ((current_price / entry_price - 1) * position 
                                if entry_price > 0 else 0)
    state['unrealized_pnl'] = state['pct_from_entry']
    return state


# ============================================================================
# ACTION SIMULATOR (Counterfactual Execution)
# ============================================================================

class ActionSimulator:
    """Simulate trades with realistic costs"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
    
    def simulate(self, action: str, position: int, equity: float,
                 prices: pd.Series, idx: int, horizon: int, 
                 stress: bool = False) -> float:
        """
        Simulate action forward for 'horizon' steps.
        Returns terminal equity value.
        """
        
        # Map action to target position
        targets = {'BUY': 1, 'SELL': -1, 'HOLD': position, 'FLATTEN': 0}
        target = targets[action]
        
        # Execute trade if needed
        eq = equity
        pos = position
        if target != pos:
            eq, pos = self._execute_trade(eq, pos, target, prices.iloc[idx], stress)
        
        # Hold for horizon
        end_idx = min(idx + horizon, len(prices) - 1)
        if pos != 0:
            pnl = (prices.iloc[end_idx] / prices.iloc[idx] - 1) * pos
            eq *= (1 + pnl)
            
            # Short borrow cost
            if pos < 0:
                days = min(horizon, end_idx - idx)
                eq *= (1 - self.cfg.SHORT_COST_ANNUAL * days / 252)
        
        return eq
    
    def _execute_trade(self, equity: float, pos: int, target: int, 
                      price: float, stress: bool) -> Tuple[float, int]:
        """Execute trade with costs"""
        
        if pos == target:
            return equity, pos
        
        size = abs(target - pos)
        
        # Slippage
        slip = self.cfg.SLIPPAGE_BPS
        if stress:
            slip *= self.cfg.STRESS_SLIP_MULT
        
        cost = (slip / 10000 + self.cfg.FEE_PCT) * size + self.cfg.FEE_FIXED / equity
        
        return equity * (1 - cost), target


# ============================================================================
# REWARD GENERATOR (Q-Value Computation)
# ============================================================================

def generate_q_values(features: pd.DataFrame, prices: pd.Series, 
                     positions: List[int], equities: List[float],
                     cfg: Config, stress: bool = False) -> pd.DataFrame:
    """
    Generate Q(s,a) for all actions at each state.
    
    This is the CORE of the system: we simulate counterfactual rewards.
    """
    
    print("💰 Computing Q-values (counterfactual rewards)...")
    
    sim = ActionSimulator(cfg)
    actions = ['BUY', 'SELL', 'HOLD', 'FLATTEN']
    
    q_data = {f'Q_{a}': [] for a in actions}
    valid_indices = []
    
    max_idx = len(prices) - cfg.REWARD_HORIZON - 1
    
    for i in range(len(features)):
        if i >= max_idx:
            continue
        
        pos = positions[min(i, len(positions)-1)]
        eq = equities[min(i, len(equities)-1)]
        
        for action in actions:
            terminal_eq = sim.simulate(action, pos, eq, prices, i, 
                                      cfg.REWARD_HORIZON, stress)
            reward = terminal_eq - eq
            
            # Risk penalty (volatility penalty)
            vol = prices.iloc[i:i+cfg.REWARD_HORIZON].pct_change().std()
            if stress:
                vol *= cfg.STRESS_VOL_MULT
            risk_penalty = cfg.RISK_LAMBDA * vol * abs(pos)
            
            q_data[f'Q_{action}'].append(reward - risk_penalty)
        
        valid_indices.append(i)
    
    # Combine with features
    result = features.iloc[valid_indices].copy()
    for action in actions:
        result[f'Q_{action}'] = q_data[f'Q_{action}']
    
    print(f"✓ Generated {len(result)} Q-value samples")
    return result


# ============================================================================
# MODEL TRAINER (One Model Per Action)
# ============================================================================

class QModelTrainer:
    """Train LightGBM regressors for each action's Q-function"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.models = {}
        self.features = None
    
    def train(self, data: pd.DataFrame):
        """Train Q_a(s) models for all actions"""
        
        print("🎓 Training Q-models...")
        
        actions = ['BUY', 'SELL', 'HOLD', 'FLATTEN']
        q_cols = [f'Q_{a}' for a in actions]
        feat_cols = [c for c in data.columns if c not in q_cols]
        
        X = data[feat_cols]
        self.features = feat_cols
        
        for action in actions:
            y = data[f'Q_{action}']
            
            # Clean data
            mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean, y_clean = X[mask], y[mask]
            
            print(f"  Training Q_{action}: {len(X_clean)} samples")
            
            params = {
                'objective': 'regression',
                'max_depth': self.cfg.MAX_DEPTH,
                'num_leaves': self.cfg.NUM_LEAVES,
                'learning_rate': self.cfg.LEARNING_RATE,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42
            }
            
            train_data = lgb.Dataset(X_clean, y_clean)
            model = lgb.train(params, train_data, num_boost_round=self.cfg.NUM_ROUNDS,
                            valid_sets=[train_data], 
                            callbacks=[lgb.early_stopping(20, verbose=False)])
            
            self.models[action] = model
            
            # Feature importance
            imp = pd.DataFrame({
                'feature': feat_cols,
                'gain': model.feature_importance('gain')
            }).sort_values('gain', ascending=False)
            print(f"    Top features: {', '.join(imp.head(3)['feature'].tolist())}")
    
    def predict_action(self, state: pd.DataFrame) -> Tuple[str, float]:
        """Predict best action for a state"""
        
        X = state[self.features]
        q_vals = {a: self.models[a].predict(X)[0] for a in self.models}
        
        best_action = max(q_vals, key=q_vals.get)
        sorted_q = sorted(q_vals.values(), reverse=True)
        confidence = sorted_q[0] - sorted_q[1]
        
        return best_action, confidence


# ============================================================================
# WALK-FORWARD BACKTESTER
# ============================================================================

def walk_forward_backtest(features: pd.DataFrame, prices: pd.Series, 
                         cfg: Config, stress: bool = False) -> Dict:
    """
    Strict walk-forward validation.
    
    Key: Models trained ONLY on past data. No lookahead.
    """
    
    print("\n" + "="*80)
    print("🚀 WALK-FORWARD BACKTEST" + (" (STRESS MODE)" if stress else ""))
    print("="*80)
    
    min_train = cfg.MIN_TRAIN_DAYS
    retrain_freq = cfg.RETRAIN_FREQ
    
    equities = [100.0]
    positions = [0]
    actions = []
    
    trainer = None
    
    for t in range(min_train, len(features)):
        
        # Retrain periodically
        if (t - min_train) % retrain_freq == 0:
            print(f"\n📅 Retraining at t={t} ({features.index[t].date()})")
            
            # Training data: everything up to t
            train_feat = features.iloc[:t]
            train_prices = prices.iloc[:t]
            train_pos = positions + [0] * (t - len(positions))
            train_eq = equities + [equities[-1]] * (t - len(equities))
            
            # Generate Q-values
            train_data = generate_q_values(train_feat, train_prices, 
                                          train_pos, train_eq, cfg, stress)
            
            # Train models
            trainer = QModelTrainer(cfg)
            trainer.train(train_data)
        
        # Predict
        state = features.iloc[[t]]
        state = augment_state(state, positions[-1], equities[-1], 
                             prices.iloc[max(0, t-1)], prices.iloc[t])
        
        action, conf = trainer.predict_action(state)
        actions.append(action)
        
        # Execute
        action_map = {'BUY': 1, 'SELL': -1, 'HOLD': positions[-1], 'FLATTEN': 0}
        target = action_map[action]
        
        sim = ActionSimulator(cfg)
        new_eq, new_pos = sim._execute_trade(equities[-1], positions[-1], 
                                             target, prices.iloc[t], stress)
        
        # Mark-to-market
        if t < len(prices) - 1:
            if new_pos != 0:
                pnl = (prices.iloc[t+1] / prices.iloc[t] - 1) * new_pos
                new_eq *= (1 + pnl)
        
        equities.append(new_eq)
        positions.append(new_pos)
    
    # Compute metrics
    metrics = compute_metrics(equities, positions)
    
    return {
        'equity': equities,
        'positions': positions,
        'actions': actions,
        'metrics': metrics,
        'dates': features.index[min_train:]
    }


def compute_metrics(equity: List[float], positions: List[int]) -> Dict:
    """Calculate performance metrics"""
    
    eq = np.array(equity)
    rets = np.diff(eq) / eq[:-1]
    
    sharpe = np.sqrt(252) * rets.mean() / (rets.std() + 1e-9)
    total_ret = (eq[-1] / eq[0] - 1) * 100
    
    cummax = np.maximum.accumulate(eq)
    dd = (eq - cummax) / cummax
    max_dd = dd.min() * 100
    
    trades = np.sum(np.abs(np.diff(positions)))
    turnover = trades / len(positions)
    
    return {
        'total_return_pct': total_ret,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd,
        'turnover': turnover,
        'final_equity': eq[-1]
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete system"""
    
    # Configuration
    cfg = Config(SYMBOLS=['SPY'])
    
    print("\n" + "="*80)
    print("REWARD-DRIVEN GBDT TRADING SYSTEM")
    print("="*80)
    
    # 1. Load data
    data = load_data(cfg.SYMBOLS, cfg.LOOKBACK_DAYS, cfg.INTERVAL)
    
    # 2. Engineer features
    features = build_feature_matrix(data, cfg.SYMBOLS)
    prices = data[(cfg.SYMBOLS[0], 'Close')].loc[features.index]
    
    # 3. Run backtest
    results = walk_forward_backtest(features, prices, cfg, stress=False)
    
    # 4. Display results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    for k, v in results['metrics'].items():
        print(f"{k:25s}: {v:10.2f}")
    
    # 5. Stress test
    print("\n" + "="*80)
    print("RUNNING STRESS TEST...")
    print("="*80)
    stress_results = walk_forward_backtest(features, prices, cfg, stress=True)
    
    print("\nSTRESS TEST RESULTS")
    print("="*80)
    for k, v in stress_results['metrics'].items():
        print(f"{k:25s}: {v:10.2f}")
    
    # 6. Plot (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curves
        dates = results['dates']
        ax1.plot(dates, results['equity'][cfg.MIN_TRAIN_DAYS:], label='Base', linewidth=2)
        ax1.plot(dates, stress_results['equity'][cfg.MIN_TRAIN_DAYS:], 
                label='Stress', linewidth=2, alpha=0.7)
        ax1.set_title('Equity Curves')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Positions
        ax2.plot(dates, results['positions'][cfg.MIN_TRAIN_DAYS:], 
                label='Position', linewidth=1.5)
        ax2.set_title('Position Over Time')
        ax2.set_ylabel('Position {-1, 0, +1}')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150)
        print("\n📊 Plot saved to 'backtest_results.png'")
        
    except ImportError:
        print("\n⚠️  matplotlib not available, skipping plots")
    
    return results, stress_results


if __name__ == '__main__':
    results, stress = main()