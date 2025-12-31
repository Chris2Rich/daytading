import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
import os
import sys
from datetime import datetime, timedelta, timezone
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib.retrieval.yfinance import get_trading, get_option, get_option_chain, get_fundamentals

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    SYMBOLS: List[str] = None
    INTERVAL: str = "1d"
    LOOKBACK_DAYS: int = 730

    # Costs
    FEE_FIXED: float = 1.0
    FEE_PCT: float = 0.001
    SLIPPAGE_BPS: float = 5.0
    SHORT_COST_ANNUAL: float = 0.02
    TRADE_PENALTY: float = 0.002

    # RL
    REWARD_HORIZON: int = 5
    GAMMA: float = 0.97

    # Model
    MAX_DEPTH: int = 5
    NUM_LEAVES: int = 20
    LEARNING_RATE: float = 0.05
    NUM_ROUNDS: int = 300

    # Backtest
    MIN_TRAIN_DAYS: int = 252
    RETRAIN_FREQ: int = 21


# ============================================================================
# DATA
# ============================================================================

def load_data(symbols: list, days: int, interval: str) -> pd.DataFrame:
    """Load OHLCV for multiple symbols using custom yfinance wrapper."""
    
    print(f"📊 Loading data: {symbols}")
    dfs = []
    for sym in symbols:
        data = get_trading(sym, days=days, interval=interval)
        if data.empty:
            continue
        
        # Make MultiIndex for symbol
        data.columns = pd.MultiIndex.from_product([[sym], data.columns])
        dfs.append(data)
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, axis=1).sort_index().fillna(method='ffill')
    print(f"✓ Loaded {len(combined)} rows")
    return combined


# ============================================================================
# FEATURE ENGINEERING (UNCHANGED, CAUSAL)
# ============================================================================

def engineer_features(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = pd.DataFrame(index=data.index)
    close = data[(symbol, "Close")]
    high = data[(symbol, "High")]
    low = data[(symbol, "Low")]
    volume = data[(symbol, "Volume")]

    for lag in [1, 2, 3, 5, 10, 21, 42]:
        df[f"ret_{lag}"] = close.pct_change(lag)

    for w in [5, 10, 21, 42]:
        df[f"vol_{w}"] = close.pct_change().rolling(w).std()

    for w in [10, 21, 50, 100]:
        sma = close.rolling(w).mean()
        df[f"sma_{w}"] = sma
        df[f"price_sma_{w}"] = close / sma - 1

    df["range"] = (high - low) / close
    df["rel_vol"] = volume / volume.rolling(21).mean()

    df.columns = [f"{symbol}_{c}" for c in df.columns]
    return df


def build_feature_matrix(data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    feats = [engineer_features(data, s) for s in symbols]
    return pd.concat(feats, axis=1).dropna()


# ============================================================================
# ACTION SIMULATOR (EXOGENOUS, POLICY-INDEPENDENT)
# ============================================================================

class ActionSimulator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def execute(self, equity: float, pos: int, target: int) -> Tuple[float, int]:
        if pos == target:
            return equity, pos

        size = abs(target - pos)
        cost = (
            self.cfg.FEE_PCT
            + self.cfg.SLIPPAGE_BPS / 10000
        ) * size + self.cfg.FEE_FIXED / equity

        return equity * (1 - cost - self.cfg.TRADE_PENALTY), target

    def forward_reward(
        self,
        prices: pd.Series,
        t: int,
        pos: int,
        horizon: int,
    ) -> float:
        end = min(t + horizon, len(prices) - 1)
        if pos == 0:
            return 0.0

        pnl = (prices.iloc[end] / prices.iloc[t] - 1) * pos
        reward = pnl

        if pos < 0:
            reward -= self.cfg.SHORT_COST_ANNUAL * horizon / 252

        return reward


# ============================================================================
# Q-VALUE GENERATION (LEAKAGE-FREE)
# ============================================================================

def generate_q_dataset(
    features: pd.DataFrame,
    prices: pd.Series,
    cfg: Config,
) -> pd.DataFrame:
    sim = ActionSimulator(cfg)
    actions = {"BUY": 1, "SELL": -1, "HOLD": None, "FLATTEN": 0}
    base_positions = [-1, 0, 1]

    rows = []

    max_t = len(features) - cfg.REWARD_HORIZON - 1

    for t in range(max_t):
        for pos in base_positions:
            state = features.iloc[t].copy()
            state["position"] = pos

            equity = 1.0
            q_vals = {}

            for a, target in actions.items():
                tgt = pos if target is None else target
                eq2, new_pos = sim.execute(equity, pos, tgt)

                r = sim.forward_reward(prices, t, new_pos, cfg.REWARD_HORIZON)
                q_vals[f"Q_{a}"] = r * cfg.GAMMA

            row = pd.concat([state, pd.Series(q_vals)])
            rows.append(row)

    return pd.DataFrame(rows).dropna()


# ============================================================================
# MODEL TRAINER
# ============================================================================

class QModelTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.models = {}
        self.features = None

    def train(self, data: pd.DataFrame):
        actions = ["BUY", "SELL", "HOLD", "FLATTEN"]
        q_cols = [f"Q_{a}" for a in actions]
        self.features = [c for c in data.columns if c not in q_cols]

        X = data[self.features]

        for a in actions:
            y = data[f"Q_{a}"]
            ds = lgb.Dataset(X, y)

            params = {
                "objective": "regression",
                "max_depth": self.cfg.MAX_DEPTH,
                "num_leaves": self.cfg.NUM_LEAVES,
                "learning_rate": self.cfg.LEARNING_RATE,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "seed": 42,
                "verbosity": -1,
            }

            self.models[a] = lgb.train(
                params,
                ds,
                num_boost_round=self.cfg.NUM_ROUNDS,
            )

    def act(self, state: pd.DataFrame) -> str:
        X = state[self.features]
        q = {a: self.models[a].predict(X)[0] for a in self.models}
        return max(q, key=q.get)


# ============================================================================
# WALK-FORWARD BACKTEST
# ============================================================================

def walk_forward_backtest(
    features: pd.DataFrame,
    prices: pd.Series,
    cfg: Config,
) -> Dict:

    equity = 100.0
    pos = 0
    equities = [equity]
    positions = [pos]

    trainer = None
    sim = ActionSimulator(cfg)

    for t in range(cfg.MIN_TRAIN_DAYS, len(features) - 1):

        if (t - cfg.MIN_TRAIN_DAYS) % cfg.RETRAIN_FREQ == 0:
            train_data = generate_q_dataset(
                features.iloc[:t],
                prices.iloc[:t],
                cfg,
            )
            trainer = QModelTrainer(cfg)
            trainer.train(train_data)

        state = features.iloc[[t]].copy()
        state["position"] = pos

        action = trainer.act(state)
        target = {"BUY": 1, "SELL": -1, "HOLD": pos, "FLATTEN": 0}[action]

        equity, pos = sim.execute(equity, pos, target)

        pnl = (prices.iloc[t + 1] / prices.iloc[t] - 1) * pos
        equity *= 1 + pnl

        equities.append(equity)
        positions.append(pos)

    return {
        "equity": equities,
        "positions": positions,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = Config(SYMBOLS=["SPY"])

    data = load_data(cfg.SYMBOLS, cfg.LOOKBACK_DAYS, cfg.INTERVAL)
    features = build_feature_matrix(data, cfg.SYMBOLS)
    prices = data[(cfg.SYMBOLS[0], "Close")].loc[features.index]

    results = walk_forward_backtest(features, prices, cfg)
    print(f"Final equity: {results['equity'][-1]:.2f}")


if __name__ == "__main__":
    main()
