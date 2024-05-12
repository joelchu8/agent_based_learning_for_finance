from typing import List, Optional

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from ..messages.marketdata import MarketDataMsg, L2SubReqMsg, L2DataMsg
from ..messages.query import QuerySpreadResponseMsg
from ..orders import Side, LimitOrder
from .trading_agent import TradingAgent


class MomentumAgent(TradingAgent):
    """
    Simple Trading Agent that places orders based on momentum signal.
    """

    def __init__(
        self,
        id: int,
        symbol: str = "IBM",
        starting_cash: int = 100_000,
        name: Optional[str] = None,
        type: Optional[str] = None,
        order_size_model=None,
        limit_price_model=None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders=False,
        min_size=20,
        max_size=50,
        alpha=0.1,
        beta_limit=10,
        beta_mkt=2,
        delta_c=0.1,
        gamma=100,
        wake_up_freq: NanosecondTime = str_to_ns("0.1s"),
        data_freq: NanosecondTime = str_to_ns("0.1s"),
        poisson_arrival=True,
        subscribe=True,
    ) -> None:

        super().__init__(id, name, type, random_state, starting_cash, log_orders)
        self.symbol = symbol
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        self.size = (
            self.random_state.randint(self.min_size, self.max_size)
            if order_size_model is None
            else None
        )
        self.order_size_model = order_size_model  # Probabilistic model for order size
        self.limit_price_model = limit_price_model # Probabilistic model for order price
        self.wake_up_freq = wake_up_freq
        self.poisson_arrival = poisson_arrival  # Whether to arrive as a Poisson process
        if self.poisson_arrival:
            self.arrival_rate = self.wake_up_freq

        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscription_requested = False
        self.mid_list: List[float] = []
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"
        self.alpha: float = alpha
        self.beta_limit: float = beta_limit
        self.delta_c: float = delta_c
        self.beta_mkt: float = beta_mkt
        self.gamma: int = gamma
        self.data_freq: NanosecondTime = data_freq
        self.momentum_signal: float = 0.
        self.momentum_signal_list: List = []

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

    def wakeup(self, current_time: NanosecondTime) -> None:
        """Agent wakeup is determined by self.wake_up_freq"""
        can_trade = super().wakeup(current_time)
        if self.subscribe and not self.subscription_requested:
            # print("Momentum trader requesting market data")
            super().request_data_subscription(
                L2SubReqMsg(
                    symbol=self.symbol,
                    freq=int(self.data_freq),
                    depth=1,
                )
            )
            self.subscription_requested = True
            self.state = "AWAITING_MARKET_DATA"
        elif can_trade and not self.subscribe:
            self.cancelOrder_workflow()
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """Momentum agent actions are determined after obtaining the best bid and ask in the LOB"""
        super().receive_message(current_time, sender_id, message)
        if (
            not self.subscribe
            and self.state == "AWAITING_SPREAD"
            and isinstance(message, QuerySpreadResponseMsg)
        ):
            bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
            # print("Momentum trader received spread")
            # print('bid: ', message.bids, 'ask: ', message.asks)
            if bid and ask:
                self.mid_list.append((bid + ask) / 2)
                self.update_momentum_signal()
                # print("Momentum trader submitting orders")
                self.place_orders(self.mid_list[-1])
            self.set_wakeup(current_time + self.get_wake_frequency())
            self.state = "AWAITING_WAKEUP"
        elif (
            self.subscribe
            and self.state == "AWAITING_MARKET_DATA"
            and isinstance(message, L2DataMsg)
        ):
            # print(message.bids, message.asks)
            bids, asks = self.known_bids[self.symbol], self.known_asks[self.symbol]
            if bids and asks:
                self.mid_list.append((bids[0][0] + asks[0][0]) / 2)
                self.update_momentum_signal()
                self.cancelOrder_workflow()
                # print("Momentum trader submitting orders")  
                self.place_orders(self.mid_list[-1])
            self.state = "AWAITING_MARKET_DATA"

    def update_momentum_signal(self) -> None:
        if len(self.mid_list) < 2:
            return
        p_now, p_pre = self.mid_list[-1], self.mid_list[-2]
        self.momentum_signal = self.alpha * np.log(p_now / p_pre) + (1 - self.alpha) * self.momentum_signal
        self.momentum_signal_list.append(self.momentum_signal)

    def get_limit_prob(self) -> float:
        return self.beta_limit * np.tanh(self.gamma * self.momentum_signal)
    
    def get_mkt_prob(self) -> float:
        return self.beta_mkt * np.tanh(self.gamma * self.momentum_signal)
    

    def place_orders(self, ref_prc: float) -> None:
        """Momentum Agent actions logic"""
        if np.abs(self.momentum_signal) < 1e-6:
            return
        s_ = Side.BID if self.momentum_signal > 0 else Side.ASK

        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)
        prc_ = self.limit_price_model.sample(ref_prc, s_, random_state=self.random_state)
        if self.size > 0:
            if self.random_state.rand() < self.get_limit_prob():
                self.place_limit_order(self.symbol, quantity=self.size, side=s_, limit_price=prc_)
            
            if self.random_state.rand() < self.get_mkt_prob():
                self.place_market_order(self.symbol, quantity=self.size, side=s_)

    def cancelOrder_workflow(self) -> None:
        # Cancel a proportion of the orders
        for order in self.orders.values():
            if self.random_state.rand() < self.delta_c:
                if isinstance(order, LimitOrder):
                    self.cancel_order(order)


    def get_wake_frequency(self) -> NanosecondTime:
        if not self.poisson_arrival:
            return self.wake_up_freq
        else:
            delta_time = self.random_state.exponential(scale=self.arrival_rate)
            return int(round(delta_time))

    @staticmethod
    def ma(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n
