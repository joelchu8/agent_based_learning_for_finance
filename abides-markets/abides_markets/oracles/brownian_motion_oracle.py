import datetime
import logging
from math import sqrt
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from abides_core import NanosecondTime

from .oracle import Oracle


logger = logging.getLogger(__name__)

class GeometricBrownianMotionOracle(Oracle):
    def __init__(self, symbol: str, mkt_open: NanosecondTime, S0: int = 1e5, mu: float=3e-5, sigma: float = 0.0335) -> None:
        self.symbol: str = symbol
        self.mkt_open: NanosecondTime = mkt_open
        self.open_price: int = S0
        self.mu: float = mu
        self.sigma: float = sigma

        self.last_price: float = S0
        self.last_time: NanosecondTime = mkt_open

        self.scaling = 8 * 3600 * 1e9 # 8 hours in nanoseconds
        self.f_log: Dict[str, List[Dict[str, Any]]] = {}
        self.f_log[symbol] = [{"FundamentalTime": mkt_open, "FundamentalValue": S0}]

        logger.debug("GeometricBrownianMotionOracle initialized")
        
    def get_daily_open_price(self, symbol: str) -> int:
        assert symbol == self.symbol, "Oracle get_daily_open_price: symbol mismatch"
        logger.debug("Oracle: market open price was {}", self.open_price)
        return self.open_price
    
    def observe_price(self, symbol: str, current_time: NanosecondTime, random_state: np.random.RandomState) -> int:
        # This function use the Geometric Brownian Motion model to generate the fundamental value at time t
        assert symbol == self.symbol, "Oracle observe_price: symbol mismatch"
        if current_time <= self.mkt_open:
            return self.open_price

        if current_time <= self.last_time:
            return int(self.last_price)

        dt = current_time - self.last_time
        dt = dt / self.scaling
        self.last_price = self.last_price * np.exp((self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * random_state.normal(0,1))
        self.last_time = current_time

        # Append the change to the permanent log of fundamental values for this symbol.
        self.f_log[symbol].append({"FundamentalTime": current_time, "FundamentalValue": self.last_price})
        logger.debug("Oracle: observed price {} at time {}", self.last_price, current_time)
        # print("Oracle: observed price {} at time {}".format(self.last_price, current_time))
        return int(self.last_price)