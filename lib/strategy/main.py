from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import timedelta

# LEAN imports (QuantConnect)
from AlgorithmImports import *


# ============================================================
# ---------------------- CONFIG TYPES ------------------------
# ============================================================

@dataclass(frozen=True)
class BacktestConfig:
    # --- Universe ---
    traded_symbols: List[str]
    auxiliary_symbols: List[str]

    # --- Time ---
    start_date: datetime
    end_date: datetime
    resolution: Resolution

    # --- Features ---
    feature_functions: Dict[str, Callable[[Any], float]]

    # --- Model ---
    model_name: str
    model_params: Dict[str, Any]

    # --- Training ---
    retrain: Dict[str, Any]  # eg {"mode": "ticks", "interval": 10000}

    # --- LEAN Simulator ---
    initial_cash: float
    fee_model: FeeModel
    max_leverage: float


# ============================================================
# ---------------------- MODEL API ----------------------------
# ============================================================

class AbstractTradingModel(ABC):
    """
    All models must conform to this API.
    Implementations live in models.py
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @abstractmethod
    def fit(self, X, y) -> None:
        """Train or retrain the model"""

    @abstractmethod
    def predict(self, X) -> Dict[str, float]:
        """
        Returns target portfolio weights or signal strengths
        Example: {"AAPL": 0.25, "MSFT": -0.10}
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (used on regime change)"""


# ============================================================
# ------------------- DATA ADAPTER API ------------------------
# ============================================================

class AbstractDataAdapter(ABC):
    """
    Adapters translate LEAN data into model-ready structures.
    Concrete implementations live in adapters.py
    """

    @abstractmethod
    def register(self, algorithm: QCAlgorithm) -> None:
        """Subscribe data to LEAN"""

    @abstractmethod
    def get_features(self, algorithm: QCAlgorithm) -> Dict[str, float]:
        """Return latest feature vector"""

    @abstractmethod
    def get_labels(self, algorithm: QCAlgorithm) -> Optional[Any]:
        """Return supervised labels if available"""


# ============================================================
# --------------------- MAIN ALGORITHM ------------------------
# ============================================================

class ModularLeanAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.config: BacktestConfig = self.load_config()

        # --- LEAN Setup ---
        self.SetStartDate(self.config.start_date)
        self.SetEndDate(self.config.end_date)
        self.SetCash(self.config.initial_cash)
        self.Settings.MaxPortfolioLeverage = self.config.max_leverage

        self.SetSecurityInitializer(
            lambda sec: sec.SetFeeModel(self.config.fee_model)
        )

        # --- Data Adapters ---
        self.adapters: List[AbstractDataAdapter] = self.load_adapters()
        for adapter in self.adapters:
            adapter.register(self)

        # --- Model ---
        self.model: AbstractTradingModel = self.load_model()

        # --- Training State ---
        self.feature_buffer = []
        self.label_buffer = []
        self.ticks_since_retrain = 0

        # --- Scheduling ---
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.BeforeMarketClose(self.config.traded_symbols[0], 1),
            self.rebalance
        )

    # ========================================================
    # ---------------------- ON DATA -------------------------
    # ========================================================

    def OnData(self, data: Slice):
        features = {}
        for adapter in self.adapters:
            features.update(adapter.get_features(self))

        label = None
        for adapter in self.adapters:
            label = adapter.get_labels(self) or label

        if features:
            self.feature_buffer.append(features)

        if label is not None:
            self.label_buffer.append(label)

        self.ticks_since_retrain += 1
        self.maybe_retrain()

    # ========================================================
    # -------------------- REBALANCE -------------------------
    # ========================================================

    def rebalance(self):
        if not self.feature_buffer:
            return

        X = self.feature_buffer[-1]
        weights = self.model.predict(X)

        for symbol, weight in weights.items():
            self.SetHoldings(symbol, weight)

    # ========================================================
    # -------------------- RETRAIN ---------------------------
    # ========================================================

    def maybe_retrain(self):
        retrain_cfg = self.config.retrain

        if retrain_cfg["mode"] == "ticks":
            if self.ticks_since_retrain >= retrain_cfg["interval"]:
                self.retrain()

        elif retrain_cfg["mode"] == "time":
            if self.Time - self.LastTrainTime >= retrain_cfg["interval"]:
                self.retrain()

        elif retrain_cfg["mode"] == "regime":
            if self.detect_regime_change():
                self.model.reset()
                self.retrain()

    def retrain(self):
        if not self.label_buffer:
            return

        self.model.fit(self.feature_buffer, self.label_buffer)
        self.feature_buffer.clear()
        self.label_buffer.clear()
        self.ticks_since_retrain = 0
        self.LastTrainTime = self.Time

    def detect_regime_change(self) -> bool:
        # Stub: plug in volatility, HMM, or structural break detection
        return False

    # ========================================================
    # ------------------ LOADING LOGIC -----------------------
    # ========================================================

    def load_model(self) -> AbstractTradingModel:
        from models import MODEL_REGISTRY

        model_cls = MODEL_REGISTRY[self.config.model_name]
        return model_cls(self.config.model_params)

    def load_adapters(self) -> List[AbstractDataAdapter]:
        from adapters import ADAPTERS
        return [adapter(self.config) for adapter in ADAPTERS]

    def load_config(self) -> BacktestConfig:
        """
        Single source of truth.
        In practice this may load from YAML/JSON.
        """
        return BacktestConfig(
            traded_symbols=["SPY"],
            auxiliary_symbols=["VIX"],
            start_date=datetime(2018, 1, 1),
            end_date=datetime(2024, 1, 1),
            resolution=Resolution.Minute,
            feature_functions={},
            model_name="RandomForestModel",
            model_params={"n_estimators": 200},
            retrain={"mode": "ticks", "interval": 5000},
            initial_cash=1_000_000,
            fee_model=ConstantFeeModel(0),
            max_leverage=2.0
        )
