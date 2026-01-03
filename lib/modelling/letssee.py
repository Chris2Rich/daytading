import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
import os
import sys
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# Import data loading utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from lib.retrieval.yahoofinance import get_trading
except ImportError:
    print("⚠️  yahoofinance module not found. Will use synthetic data.")
    get_trading = None


# ============================================================================
# MARKET REGIME DEFINITIONS
# ============================================================================

class MarketRegime(Enum):
    """Hidden market states in the POMDP."""
    LOW_VOL_TREND = 0
    HIGH_VOL_TREND = 1
    MEAN_REVERTING = 2
    CRISIS = 3
    NEUTRAL = 4


class BarrierOutcome(Enum):
    """Triple barrier outcomes."""
    TAKE_PROFIT = 1
    STOP_LOSS = -1
    TIMEOUT = 0


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class POMDPConfig:
    """Configuration for POMDP trading system."""
    
    # Data
    SYMBOLS: List[str] = None
    LOOKBACK_DAYS: int = 730
    INTERVAL: str = "1d"
    
    # Regime inference
    REGIME_LOOKBACK: int = 63
    
    # Triple barrier
    HORIZONS: List[int] = None
    TP_MULTIPLIER: float = 2.0
    SL_MULTIPLIER: float = 1.0
    VOL_WINDOW: int = 21
    
    # Probability models
    N_ESTIMATORS: int = 200
    MAX_DEPTH: int = 4
    MIN_SAMPLES_PER_MODEL: int = 100
    
    # Policy
    HORIZON_WEIGHTS: Dict[int, float] = None
    REGIME_WEIGHT_DECAY: float = 0.3
    COST_PER_TRADE: float = 0.001
    UCB_EXPLORATION: float = 1.0
    
    # Training
    MIN_TRAIN_BARS: int = 150
    RETRAIN_FREQUENCY: int = 21
    MAX_BUFFER_SIZE: int = 5000
    
    # Portfolio
    PORTFOLIO_LEVERAGE: float = 1.0
    
    def __post_init__(self):
        if self.SYMBOLS is None:
            self.SYMBOLS = ["SPY", "QQQ", "TLT"]
        if self.HORIZONS is None:
            self.HORIZONS = [5, 10, 21]
        if self.HORIZON_WEIGHTS is None:
            total = sum(self.HORIZONS)
            self.HORIZON_WEIGHTS = {h: h/total for h in self.HORIZONS}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(symbols: List[str], days: int, interval: str) -> Dict[str, pd.DataFrame]:
    """Load OHLCV data for multiple symbols."""
    print(f"Loading data: {symbols}")
    
    data_dict = {}
    
    if get_trading is not None:
        for symbol in symbols:
            try:
                data = get_trading(symbol, days=days, interval=interval)
                if not data.empty:
                    data_dict[symbol] = data
            except Exception as e:
                print(f"Failed to load {symbol}: {e}")
    
    # Fallback to synthetic data if needed
    if not data_dict:
        print("  Using synthetic data")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='D')
        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)
            prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.01))
            data_dict[symbol] = pd.DataFrame({
                'Close': prices,
                'Open': prices * (1 + np.random.randn(500) * 0.002),
                'High': prices * (1 + np.abs(np.random.randn(500)) * 0.005),
                'Low': prices * (1 - np.abs(np.random.randn(500)) * 0.005),
                'Volume': np.random.randint(1e6, 1e7, 500)
            }, index=dates)
    
    print(f"Loaded {len(data_dict)} symbols")
    return data_dict


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Generate features for a single symbol."""
    df = pd.DataFrame(index=data.index)
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    returns = close.pct_change()
    
    # Returns at multiple lags
    for lag in [1, 2, 3, 5, 10, 21, 42, 63]:
        df[f"ret_{lag}"] = close.pct_change(lag)
    
    # Acceleration
    df["ret_accel_5"] = df["ret_5"] - df["ret_5"].shift(5)
    df["ret_accel_21"] = df["ret_21"] - df["ret_21"].shift(21)
    
    # Volatility
    for w in [5, 10, 21, 42, 63]:
        df[f"vol_{w}"] = returns.rolling(w).std()
        df[f"vol_rank_{w}"] = df[f"vol_{w}"].rolling(min(252, len(close))).rank(pct=True)
    
    # Parkinson volatility
    for w in [5, 21]:
        hl = np.log(high / low)
        df[f"parkinson_vol_{w}"] = np.sqrt(hl.pow(2).rolling(w).mean() / (4 * np.log(2)))
    
    # Trend
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
    
    # Mean reversion
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
    
    # Volume
    df["rel_vol"] = volume / volume.rolling(21).mean()
    df["vol_trend"] = volume.rolling(21).mean() / volume.rolling(63).mean() - 1
    
    # Range & microstructure
    df["range"] = (high - low) / close
    df["range_rank"] = df["range"].rolling(min(252, len(close))).rank(pct=True)
    df["gap"] = (close - close.shift(1)) / close.shift(1)
    df["hl_position"] = (close - low) / (high - low + 1e-9)
    
    # Regime features
    for w in [10, 21]:
        df[f"trend_strength_{w}"] = abs(df[f"price_sma_{w}"])
    
    vol_21 = returns.rolling(21).std()
    df["vol_regime"] = vol_21.rolling(min(252, len(close))).rank(pct=True)
    
    rolling_max = close.rolling(min(252, len(close)), min_periods=1).max()
    df["drawdown"] = (close / rolling_max - 1)
    
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


def build_feature_matrix(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build comprehensive feature matrix for all symbols."""
    print("Engineering features...")
    
    feats = []
    for symbol, data in data_dict.items():
        feat = engineer_features(data, symbol)
        feats.append(feat)
    
    combined = pd.concat(feats, axis=1)
    combined = combined.ffill().fillna(0)
    
    return combined


# ============================================================================
# REGIME INFERENCE MODULE
# ============================================================================

class RegimeInferenceModule:
    """Infer latent market regime from observable features."""
    
    def __init__(self, config: POMDPConfig):
        self.config = config
        self.model: Optional[lgb.Booster] = None
        self.feature_cols: List[str] = []
    
    def extract_regime_features(self, features_df: pd.DataFrame, symbol: str, t: int) -> pd.Series:
        """Extract regime-relevant features at time t."""
        regime_feats = {}
        
        # Volatility regime
        vol_col = f"{symbol}_vol_21"
        if vol_col in features_df.columns:
            regime_feats['volatility'] = features_df[vol_col].iloc[t]
            regime_feats['vol_rank'] = features_df[f"{symbol}_vol_rank_21"].iloc[t]
        
        # Drawdown
        dd_col = f"{symbol}_drawdown"
        if dd_col in features_df.columns:
            regime_feats['drawdown'] = features_df[dd_col].iloc[t]
        
        # Trend strength
        for w in [10, 21]:
            ret_col = f"{symbol}_ret_{w}"
            if ret_col in features_df.columns:
                regime_feats[f'trend_strength_{w}'] = abs(features_df[ret_col].iloc[t])
        
        # Mean reversion signal
        for w in [10, 21]:
            z_col = f"{symbol}_zscore_{w}"
            if z_col in features_df.columns:
                regime_feats[f'zscore_{w}'] = abs(features_df[z_col].iloc[t])
        
        # Volume regime
        vol_rel_col = f"{symbol}_rel_vol"
        if vol_rel_col in features_df.columns:
            regime_feats['volume_regime'] = features_df[vol_rel_col].iloc[t]
        
        # Range rank
        range_col = f"{symbol}_range_rank"
        if range_col in features_df.columns:
            regime_feats['range_rank'] = features_df[range_col].iloc[t]
        
        return pd.Series(regime_feats)
    
    def label_regimes(self, features_df: pd.DataFrame, symbol: str) -> pd.Series:
        """Create regime labels from observable features using rules."""
        labels = []
        
        for t in range(len(features_df)):
            rf = self.extract_regime_features(features_df, symbol, t)
            
            # 1. CRISIS (Crash Dynamics)
            if rf.get('drawdown', 0) < -0.15 or rf.get('vol_rank', 0) > 0.90:
                labels.append(MarketRegime.CRISIS.value)
                
            # 2. LOW VOLATILITY UPTREND (The "Grind Up")
            # If Z-Score is high BUT volatility is low, this is a Bull Trend, not a top.
            elif rf.get('zscore_21', 0) > 0.5 and rf.get('vol_rank', 0) < 0.4:
                labels.append(MarketRegime.LOW_VOL_TREND.value)
                
            # 3. HIGH VOLATILITY TREND (Strong Momentum)
            elif rf.get('trend_strength_21', 0) > 0.02 and rf.get('vol_rank', 0) > 0.6:
                labels.append(MarketRegime.HIGH_VOL_TREND.value)
                
            # 4. MEAN REVERTING (Choppy / Overextended)
            # Only label "Reversion" if we are EXTREMELY overextended (Z > 2.5) 
            # OR if we are in neutral chop.
            elif abs(rf.get('zscore_21', 0)) > 2.5:
                 labels.append(MarketRegime.MEAN_REVERTING.value)
                 
            # 5. NEUTRAL
            else:
                labels.append(MarketRegime.NEUTRAL.value)
        
    def train(self, features_df: pd.DataFrame, symbol: str):
        """Train regime classifier."""
        X_list = []
        for t in range(len(features_df)):
            rf = self.extract_regime_features(features_df, symbol, t)
            X_list.append(rf)
        
        X = pd.DataFrame(X_list).fillna(0)
        y = self.label_regimes(features_df, symbol)
        
        self.feature_cols = X.columns.tolist()
        
        params = {
            'objective': 'multiclass',
            'num_class': len(MarketRegime),
            'metric': 'multi_logloss',
            'max_depth': 4,
            'num_leaves': 16,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'verbosity': -1,
            'seed': 42
        }
        
        ds = lgb.Dataset(X, label=y)
        self.model = lgb.train(params, ds, num_boost_round=200)
            
    def predict_regime_probabilities(self, features_df: pd.DataFrame, symbol: str, t: int) -> np.ndarray:
        """Predict regime probability vector at time t."""
        if self.model is None:
            return np.ones(len(MarketRegime)) / len(MarketRegime)
        
        rf = self.extract_regime_features(features_df, symbol, t)
        X = pd.DataFrame([rf])[self.feature_cols].fillna(0)
        
        probs = self.model.predict(X)[0]
        return probs


# ============================================================================
# TRIPLE BARRIER LABELING
# ============================================================================

@dataclass
class BarrierLabel:
    """Triple barrier label for a single trade."""
    outcome: BarrierOutcome
    bars_to_hit: int
    return_at_hit: float
    horizon: int


class TripleBarrierLabeler:
    """Label trades using triple-barrier method."""
    
    def __init__(self, config: POMDPConfig):
        self.config = config
        
    def compute_barriers(
        self,
        prices: pd.Series,
        t: int,
        horizon: int,
        action: str
    ) -> Tuple[float, float]:
        """Compute TP and SL barriers in return space."""

        # --- FORCE prices to be Series ---
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]

        returns = prices.pct_change()

        vol_slice = returns.iloc[max(0, t - self.config.VOL_WINDOW):t]

        # --- FORCE scalar volatility ---
        vol = float(vol_slice.std()) if len(vol_slice) > 0 else np.nan

        if not np.isfinite(vol) or vol < 1e-6:
            vol = 0.02  # conservative fallback

        tp_threshold = self.config.TP_MULTIPLIER * vol
        sl_threshold = -self.config.SL_MULTIPLIER * vol

        if action == "SELL":
            tp_threshold = -tp_threshold
            sl_threshold = -sl_threshold

        return tp_threshold, sl_threshold

    def label_single_trade(
    self,
    prices: pd.Series | pd.DataFrame,
    t: int,
    action: str,
    horizon: int
) -> BarrierLabel:
        """Label a single trade with triple barrier."""

        # --- FORCE prices to Series ---
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
            
        if action not in ["BUY", "SELL"]:
            return BarrierLabel(BarrierOutcome.TIMEOUT, horizon, 0.0, horizon)
        
        tp_threshold, sl_threshold = self.compute_barriers(prices, t, horizon, action)
        entry_price = prices.iloc[t]
        
        # Scan forward to find first barrier hit
        for i in range(1, horizon + 1):
            if t + i >= len(prices):
                final_return = (prices.iloc[-1] / entry_price - 1)
                if action == "SELL":
                    final_return = -final_return
                return BarrierLabel(BarrierOutcome.TIMEOUT, i, final_return, horizon)
            
            current_price = prices.iloc[t + i]
            ret = (current_price / entry_price - 1)
            
            if action == "SELL":
                ret = -ret

            ret = float(ret)
            
            if ret >= tp_threshold:
                return BarrierLabel(BarrierOutcome.TAKE_PROFIT, i, ret, horizon)
            elif ret <= sl_threshold:
                return BarrierLabel(BarrierOutcome.STOP_LOSS, i, ret, horizon)
        
        # Timeout
        final_return = (prices.iloc[t + horizon] / entry_price - 1)
        if action == "SELL":
            final_return = -final_return
        return BarrierLabel(BarrierOutcome.TIMEOUT, horizon, final_return, horizon)
    
    def label_dataset(self, prices: pd.Series, features_df: pd.DataFrame, 
                      symbol: str, action: str, min_t: int = 0) -> pd.DataFrame:
        """Create labeled dataset for training."""
        labels_list = []
        max_horizon = max(self.config.HORIZONS)
        
        for t in range(min_t, len(prices) - max_horizon):
            row = {'t': t}
            
            # Add features
            state_cols = [c for c in features_df.columns if c.startswith(f"{symbol}_")]
            for col in state_cols:
                row[col] = features_df[col].iloc[t]
            
            # Label for each horizon
            for h in self.config.HORIZONS:
                label = self.label_single_trade(prices, t, action, h)
                row[f'outcome_h{h}'] = label.outcome.value
                row[f'return_h{h}'] = label.return_at_hit
                row[f'bars_to_hit_h{h}'] = label.bars_to_hit
            
            labels_list.append(row)
        
        return pd.DataFrame(labels_list)


# ============================================================================
# BARRIER PROBABILITY MODELS
# ============================================================================

class BarrierProbabilityModels:
    """Train calibrated probability models for barrier outcomes."""
    
    def __init__(self, config: POMDPConfig):
        self.config = config
        self.models: Dict[Tuple[str, int, str], lgb.Booster] = {}
        self.feature_cols: List[str] = []
    
    def train_horizon_model(self, labeled_data: pd.DataFrame, horizon: int, action: str):
        """Train models for a specific (action, horizon) pair."""
        outcome_col = f'outcome_h{horizon}'
        
        feature_cols = [c for c in labeled_data.columns 
                        if not c.startswith('outcome_') 
                        and not c.startswith('return_')
                        and not c.startswith('bars_to_hit_')
                        and c != 't']
        
        X = labeled_data[feature_cols].fillna(0)
        y = labeled_data[outcome_col]
        
        if len(X) < self.config.MIN_SAMPLES_PER_MODEL:
            print(f"Insufficient data for {action} h={horizon}: {len(X)} samples")
            return
        
        if not self.feature_cols:
            self.feature_cols = feature_cols
        
        # Train TP classifier
        y_tp = (y == BarrierOutcome.TAKE_PROFIT.value).astype(int)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'max_depth': self.config.MAX_DEPTH,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1,
            'seed': 42
        }
        
        ds_tp = lgb.Dataset(X, label=y_tp)
        model_tp = lgb.train(params, ds_tp, num_boost_round=self.config.N_ESTIMATORS)
        
        # Train SL classifier
        y_sl = (y == BarrierOutcome.STOP_LOSS.value).astype(int)
        ds_sl = lgb.Dataset(X, label=y_sl)
        model_sl = lgb.train(params, ds_sl, num_boost_round=self.config.N_ESTIMATORS)
        
        self.models[(action, horizon, 'TP')] = model_tp
        self.models[(action, horizon, 'SL')] = model_sl
        
        pred_tp = model_tp.predict(X) > 0.5
        pred_sl = model_sl.predict(X) > 0.5
        acc_tp = (pred_tp == y_tp).mean()
        acc_sl = (pred_sl == y_sl).mean()
    
    def predict_probabilities(self, state: pd.DataFrame, action: str, horizon: int) -> Tuple[float, float]:
        """Predict P(TP) and P(SL) for given state, action, horizon."""
        key_tp = (action, horizon, 'TP')
        key_sl = (action, horizon, 'SL')
        
        if key_tp not in self.models or key_sl not in self.models:
            return 0.33, 0.33
        
        X = state[self.feature_cols].fillna(0)
        
        p_tp = self.models[key_tp].predict(X)[0]
        p_sl = self.models[key_sl].predict(X)[0]
        
        # Ensure probabilities sum <= 1
        total = p_tp + p_sl
        if total > 1.0:
            p_tp /= total
            p_sl /= total
        
        return p_tp, p_sl


# ============================================================================
# BELIEF STATE CONSTRUCTION
# ============================================================================

@dataclass
class BeliefState:
    """Complete belief state representation."""
    features: np.ndarray
    regime_probs: np.ndarray
    position: float
    volatility: float
    regime_entropy: float
    t: int


class BeliefStateConstructor:
    """Build belief states from observations."""
    
    def __init__(self, regime_module: RegimeInferenceModule, config: POMDPConfig):
        self.regime_module = regime_module
        self.config = config
    
    def construct(self, features_df: pd.DataFrame, symbol: str, t: int, 
                  position: float, prices: pd.Series) -> BeliefState:
        """Construct belief state at time t (no forward-looking)."""
        state_cols = [c for c in features_df.columns if c.startswith(f"{symbol}_")]
        feature_vec = features_df[state_cols].iloc[t].fillna(0).values
        
        regime_probs = self.regime_module.predict_regime_probabilities(features_df, symbol, t)
        entropy = -np.sum(regime_probs * np.log(regime_probs + 1e-9))
        
        # --- FIX: Ensure prices is a simple Series, not a DataFrame ---
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()  # Converts single-column DF to Series

        returns = prices.pct_change()
        vol_calc = returns.iloc[max(0, t - self.config.VOL_WINDOW):t].std()
        
        # --- FIX: Handle if vol_calc is still a generic Series/Object ---
        if isinstance(vol_calc, (pd.Series, np.ndarray)):
            # If it's a series, extract the single value
            vol_calc = vol_calc.item() if not vol_calc.empty else np.nan
            
        # Safe float conversion
        try:
            vol = float(vol_calc)
        except (ValueError, TypeError):
            vol = np.nan
            
        if np.isnan(vol) or vol < 1e-9:
            vol = 0.02
        
        return BeliefState(
            features=feature_vec,
            regime_probs=regime_probs,
            position=position,
            volatility=vol,
            regime_entropy=entropy,
            t=t
        )


# ============================================================================
# EXPECTED VALUE POLICY
# ============================================================================

class ExpectedValuePolicy:
    """Select actions via expected payoff with uncertainty bonuses."""
    
    def __init__(self, prob_models: BarrierProbabilityModels, config: POMDPConfig):
        self.prob_models = prob_models
        self.config = config
    
    def compute_expected_value(self, belief: BeliefState, action: str, 
                                features_df: pd.DataFrame, symbol: str) -> Tuple[float, float]:
        """Compute expected value and confidence for an action."""
        if action not in ["BUY", "SELL"]:
            return 0.0, 1.0
        
        state_cols = [c for c in features_df.columns if c.startswith(f"{symbol}_")]
        state_df = features_df[state_cols].iloc[[belief.t]]
        
        ev_total = 0.0
        
        for horizon in self.config.HORIZONS:
            p_tp, p_sl = self.prob_models.predict_probabilities(state_df, action, horizon)
            
            reward_tp = belief.volatility * self.config.TP_MULTIPLIER
            reward_sl = belief.volatility * self.config.SL_MULTIPLIER
            
            ev_horizon = p_tp * reward_tp - p_sl * reward_sl
            weight = self.config.HORIZON_WEIGHTS.get(horizon, 0.0)
            ev_total += weight * ev_horizon
        
        ev_total -= self.config.COST_PER_TRADE
        entropy_penalty = self.config.REGIME_WEIGHT_DECAY * belief.regime_entropy
        ev_total -= entropy_penalty
        
        confidence = 1.0 / (1.0 + belief.regime_entropy)
        
        return ev_total, confidence
    
    def select_action(self, belief: BeliefState, features_df: pd.DataFrame, 
                      symbol: str, exploration: float = 0.0) -> Tuple[str, float]:
        """Select action using UCB."""
        actions = ["BUY", "SELL", "HOLD", "FLATTEN"]
        
        ev_scores = {}
        confidence_scores = {}
        
        for action in actions:
            ev, conf = self.compute_expected_value(belief, action, features_df, symbol)
            ucb_bonus = exploration * (1.0 - conf)
            ev_scores[action] = ev + ucb_bonus
            confidence_scores[action] = conf
        
        best_action = max(ev_scores, key=ev_scores.get)
        return best_action, confidence_scores[best_action]


# ============================================================================
# POMDP EXPERIENCE BUFFER
# ============================================================================

@dataclass
class POMDPExperience:
    """Experience tuple for POMDP learning."""
    features: np.ndarray
    regime_probs: np.ndarray
    action: str
    position: float
    volatility: float
    barrier_outcomes: Dict[int, BarrierLabel]
    costs: float
    t: int


class POMDPExperienceBuffer:
    """Store POMDP experiences for training."""
    
    def __init__(self, max_size: int = 10000):
        self.buffer: List[POMDPExperience] = []
        self.max_size = max_size
    
    def add(self, experience: POMDPExperience):
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def get_all(self) -> List[POMDPExperience]:
        return self.buffer
    
    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# WALK-FORWARD BACKTEST
# ============================================================================

def pomdp_walk_forward_backtest(
    features_df: pd.DataFrame,
    prices_dict: Dict[str, pd.Series],
    symbols: List[str],
    config: POMDPConfig
) -> Dict:
    """Complete POMDP-based walk-forward backtest."""
    
    print("POMDP WALK-FORWARD BACKTEST")
    
    # Initialize
    equity = 100000.0
    positions = {s: 0.0 for s in symbols}
    equity_history = [equity]
    position_history = [{s: 0.0 for s in symbols}]
    
    # Initialize modules per symbol
    regime_modules = {s: RegimeInferenceModule(config) for s in symbols}
    barrier_labelers = {s: TripleBarrierLabeler(config) for s in symbols}
    prob_models = {s: BarrierProbabilityModels(config) for s in symbols}
    belief_constructors = {s: BeliefStateConstructor(regime_modules[s], config) for s in symbols}
    policies = {s: ExpectedValuePolicy(prob_models[s], config) for s in symbols}
    experience_buffers = {s: POMDPExperienceBuffer(config.MAX_BUFFER_SIZE) for s in symbols}
    
    max_horizon = max(config.HORIZONS)
    min_t = config.MIN_TRAIN_BARS
    max_t = len(features_df) - max_horizon - 1
    
    models_trained = False
    
    # Walk-forward loop
    for t in range(min_t, max_t):
        
        if equity <= 0:
            print(f"\nBANKRUPT at t={t}")
            break
        
        # Retrain phase
        if (t - min_t) % config.RETRAIN_FREQUENCY == 0:
            
            for symbol in symbols:
                # Train regime model
                regime_modules[symbol].train(features_df.iloc[:t], symbol)
                
                # Create labeled datasets and train probability models
                for action in ["BUY", "SELL"]:
                    labeled_data = barrier_labelers[symbol].label_dataset(
                        prices_dict[symbol],
                        features_df,
                        symbol,
                        action,
                        min_t=max(0, t - 504)
                    )
                    
                    if len(labeled_data) < config.MIN_SAMPLES_PER_MODEL:
                        print(f" Only {len(labeled_data)} samples, skipping")
                        continue
                    
                    for horizon in config.HORIZONS:
                        prob_models[symbol].train_horizon_model(labeled_data, horizon, action)
            
            models_trained = True
            
            if not models_trained and t == min_t + config.RETRAIN_FREQUENCY:
                print(f"\n Warm-up complete. Resetting equity to $100,000")
                equity = 100000.0
                positions = {s: 0.0 for s in symbols}
                equity_history = [equity]
                position_history = [{s: 0.0 for s in symbols}]
        
        # Prediction & action phase
        actions_taken = {}
        new_positions = {}
        confidences = {}
        beliefs = {}
        
        for symbol in symbols:
            belief = belief_constructors[symbol].construct(
                features_df, symbol, t, positions[symbol], prices_dict[symbol]
            )
            beliefs[symbol] = belief
            
            if not models_trained:
                action = np.random.choice(["BUY", "SELL", "HOLD", "FLATTEN"])
                confidence = 0.5
            else:
                exploration = max(0.1, 1.0 - (t - min_t) / (max_t - min_t))
                action, confidence = policies[symbol].select_action(
                    belief, features_df, symbol, exploration
                )
            
            actions_taken[symbol] = action
            confidences[symbol] = confidence
            
            # Convert action to position
            if action == "HOLD":
                new_positions[symbol] = positions[symbol]
            elif action == "FLATTEN":
                new_positions[symbol] = 0.0
            elif action == "BUY":
                vol_scale = np.clip(0.02 / (belief.volatility + 1e-9), 0.1, 2.0)
                new_positions[symbol] = confidence * vol_scale * 0.5
                new_positions[symbol] = np.clip(new_positions[symbol], 0, 1)
            elif action == "SELL":
                vol_scale = np.clip(0.02 / (belief.volatility + 1e-9), 0.1, 2.0)
                new_positions[symbol] = -confidence * vol_scale * 0.5
                new_positions[symbol] = np.clip(new_positions[symbol], -1, 0)
        
        # Normalize positions
        total_abs = sum(abs(p) for p in new_positions.values())
        if total_abs > config.PORTFOLIO_LEVERAGE:
            scale = config.PORTFOLIO_LEVERAGE / total_abs
            new_positions = {s: p * scale for s, p in new_positions.items()}
        
        # Compute transaction costs
        total_costs = 0.0
        for symbol in symbols:
            trade_size = abs(new_positions[symbol] - positions[symbol])
            if trade_size > 0.001:
                cost_pct = config.COST_PER_TRADE * trade_size
                total_costs += cost_pct
        
        equity *= (1.0 - total_costs)
        
        # Compute realized PnL
        total_pnl = 0.0
        for symbol in symbols:
            price_t = float(prices_dict[symbol].iloc[t])
            price_t1 = float(prices_dict[symbol].iloc[t + 1])
            
            ret = (price_t1 / price_t - 1)
            pnl = ret * positions[symbol]
            total_pnl += pnl
        
        equity *= (1.0 + total_pnl)
        positions = new_positions.copy()
        
        equity_history.append(equity)
        position_history.append(positions.copy())
        
        # Store experiences
        if models_trained:
            for symbol in symbols:
                barrier_outcomes = {}
                for horizon in config.HORIZONS:
                    label = barrier_labelers[symbol].label_single_trade(
                        prices_dict[symbol], t, actions_taken[symbol], horizon
                    )
                    barrier_outcomes[horizon] = label
                
                exp = POMDPExperience(
                    features=beliefs[symbol].features,
                    regime_probs=beliefs[symbol].regime_probs,
                    action=actions_taken[symbol],
                    position=positions[symbol],
                    volatility=beliefs[symbol].volatility,
                    barrier_outcomes=barrier_outcomes,
                    costs=total_costs / len(symbols),
                    t=t
                )
                experience_buffers[symbol].add(exp)
        
        print(f"\r{round((t - min_t) / (max_t - min_t) * 100, 2)}%", end="\r", flush=True)
    
    # Final results
    print("BACKTEST COMPLETE")
    
    final_equity = equity_history[-1]
    total_return = (final_equity - 100000) / 100000
    
    equity_series = pd.Series(equity_history)
    returns = equity_series.pct_change().dropna()
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)
    
    running_max = equity_series.expanding().max()
    drawdown = (equity_series / running_max - 1)
    max_dd = drawdown.min()
    
    print(f"\nFinal Equity:    ${final_equity:>12,.2f}")
    print(f"Total Return:    {total_return:>12.2%}")
    print(f"Sharpe Ratio:    {sharpe:>12.2f}")
    print(f"Max Drawdown:    {max_dd:>12.2%}")
    print(f"\nExperience buffer sizes:")
    for symbol in symbols:
        print(f"  {symbol}: {len(experience_buffers[symbol])} experiences")
    
    return {
        'equity': equity_history,
        'positions': position_history,
        'final_equity': final_equity,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'experiences': experience_buffers
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_pomdp_performance(results: Dict, feature_index: pd.DatetimeIndex, min_t: int):
    """
    Plot equity curve, drawdown, and position history.
    
    Args:
        results: Dictionary returned from backtest
        feature_index: The full DatetimeIndex from features_df
        min_t: The starting index of the backtest
    """
    print("\n📊 Generating performance plots...")
    
    # 1. Align Data with Dates
    # The simulation runs from min_t to the end. 
    # results['equity'] has one initial value, so we map it to min_t-1 to end
    
    # Slice the index to match the backtest duration
    # We add 1 because equity history includes the starting state
    sim_dates = feature_index[min_t : min_t + len(results['equity'])]
    
    # Handle potential length mismatch by trimming the longer one
    min_len = min(len(sim_dates), len(results['equity']))
    sim_dates = sim_dates[:min_len]
    equity_curve = pd.Series(results['equity'][:min_len], index=sim_dates)
    
    # Positions alignment
    pos_df = pd.DataFrame(results['positions'])
    pos_df = pos_df.iloc[:min_len] # Ensure length matches
    pos_df.index = sim_dates
    
    # 2. Calculate Drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve / running_max - 1)
    
    # 3. Create Plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 2]})
    
    # -- Panel 1: Equity --
    ax1.plot(equity_curve, color='green', linewidth=1.5, label='Portfolio Equity')
    ax1.set_title('POMDP Strategy Performance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Equity ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # -- Panel 2: Drawdown --
    ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax2.plot(drawdown, color='red', linewidth=1)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Risk Profile', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # -- Panel 3: Positions --
    for col in pos_df.columns:
        ax3.plot(pos_df.index, pos_df[col], label=col, linewidth=1.5, alpha=0.8)
    
    ax3.set_ylabel('Position Size (Leverage)')
    ax3.set_title('Asset Allocation', fontsize=10)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='black', linewidth=1, linestyle='--')
    
    # Formatting
    plt.xlabel('Date')
    
    # Format x-axis dates
    if len(sim_dates) > 0:
        fig.autofmt_xdate()
        
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(config: POMDPConfig):
    """Main execution function."""
    
    print("POMDP TRADING SYSTEM - PRODUCTION IMPLEMENTATION")
    
    # Load data
    print("\n[1/4] Loading market data...")
    data_dict = load_data(config.SYMBOLS, config.LOOKBACK_DAYS, config.INTERVAL)
    
    # Feature engineering
    print("\n[2/4] Engineering features...")
    features_df = build_feature_matrix(data_dict)
    
    # Extract prices
    prices_dict = {}
    for symbol in config.SYMBOLS:
        p = data_dict[symbol]['Close']
        if isinstance(p, pd.DataFrame):
            p = p.squeeze()
        prices_dict[symbol] = p
    
    # Align indices
    common_index = features_df.index
    prices_dict = {s: prices_dict[s].loc[common_index] for s in config.SYMBOLS}
    
    # Run backtest
    print("\n[3/4] Running POMDP backtest...")
    results = pomdp_walk_forward_backtest(features_df, prices_dict, config.SYMBOLS, config)
    
    print("\n[4/4] Analysis complete!")

    plot_pomdp_performance(
        results, 
        features_df.index, 
        min_t=config.MIN_TRAIN_BARS
    )
    
    return results


if __name__ == "__main__":
    results = main(config=POMDPConfig(
        SYMBOLS=["SPY"],

        LOOKBACK_DAYS=365,
        INTERVAL="1d",
        HORIZONS=[3, 5, 25],

        TP_MULTIPLIER=2.0,
        SL_MULTIPLIER=1.5,
        VOL_WINDOW=10,

        N_ESTIMATORS=10,
        MAX_DEPTH=6,
        MIN_TRAIN_BARS=45,
        RETRAIN_FREQUENCY=20,

        COST_PER_TRADE=0.001,
        PORTFOLIO_LEVERAGE=1.0
    ))