import logging
from typing import Optional

import numpy as np

from abides_core import Message, NanosecondTime

from ..generators import OrderSizeGenerator
from ..models.limit_price_model import LimitPriceModel
from ..messages.query import QuerySpreadResponseMsg
from ..orders import Side, LimitOrder
from .trading_agent import TradingAgent


logger = logging.getLogger(__name__)


class NoiseAgent(TradingAgent):
    """
    Noise agent implement simple strategy. The agent wakes up once and places 1 order.
    """

    def __init__(
        self,
        id: int,
        symbol: str = "IBM",
        starting_cash: int = 100000,
        name: Optional[str] = None,
        type: Optional[str] = None,
        order_size_model: Optional[OrderSizeGenerator] = None,
        limit_price_model: Optional[LimitPriceModel] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders: bool = False,
        sigma_limit: float = 0.2,
        sigma_mkt: float = 0.05,
        delta_c: float = 0.1,
        mean_wakeup_gap: int = 1e9,
    ) -> None:

        # Base class init.
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.symbol: str = symbol  # symbol to trade

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading: bool = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state: str = "AWAITING_WAKEUP"

        self.size: Optional[int] = (
            self.random_state.randint(20, 50) if order_size_model is None else None
        )
        self.limit_price_model = limit_price_model
        self.sigma_limit = sigma_limit
        self.sigma_mkt = sigma_mkt
        self.delta_c = delta_c
        self.mean_wakeup_gap = mean_wakeup_gap

        self.order_size_model = order_size_model  # Probabilistic model for order size

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # self.kernel is set in Agent.kernel_initializing()
        # self.exchange_id is set in TradingAgent.kernel_starting()

        super().kernel_starting(start_time)

    def kernel_stopping(self) -> None:
        # Always call parent method to be safe.
        super().kernel_stopping()

        # Fix the problem of logging an agent that has not waken up
        try:
            # noise trader surplus is marked to EOD
            bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)
        except KeyError:
            self.logEvent("FINAL_VALUATION", self.starting_cash, True)
        else:
            # Print end of day valuation.
            H = int(round(self.get_holdings(self.symbol), -2) / 100)

            if bid and ask:
                rT = int(bid + ask) / 2
            else:
                rT = self.last_trade[self.symbol]

            # final (real) fundamental value times shares held.
            surplus = rT * H

            logger.debug("Surplus after holdings: {}", surplus)

            # Add ending cash value and subtract starting cash value.
            surplus += self.holdings["CASH"] - self.starting_cash
            surplus = float(surplus) / self.starting_cash

            self.logEvent("FINAL_VALUATION", surplus, True)

            logger.debug(
                "{} final report.  Holdings: {}, end cash: {}, start cash: {}, final fundamental: {}, surplus: {}",
                self.name,
                H,
                self.holdings["CASH"],
                self.starting_cash,
                rT,
                surplus,
            )

    def wakeup(self, current_time: NanosecondTime) -> None:
        # Parent class handles discovery of exchange times and market_open wakeup call.
        super().wakeup(current_time)

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return
        else:
            if not self.trading:
                self.trading = True

                # Time to start trading!
                logger.debug("{} is ready to start trading now.", self.name)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return
        
        self.get_current_spread(self.symbol)
        self.state = "AWAITING_SPREAD"
        self.cancelOrder_workflow()

        if current_time > self.mkt_open:
            self.set_wakeup(current_time + self.get_wake_frequency())

    def placeOrder(self) -> None:
        # Probabilistically place order in random direction

        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)

        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)

        if self.size > 0 and bid and ask:
            # place limit order and market order
            buy_indicator = self.random_state.randint(0, 1 + 1) # Order direction
            # Limit order
            if self.random_state.rand() < self.sigma_limit:
                ref_prc = (bid + ask) / 2.
                s_ = Side.BID if buy_indicator else Side.ASK
                prc_ = self.limit_price_model.sample(ref_prc, s_, random_state=self.random_state)
                self.place_limit_order(self.symbol, self.size, s_, prc_)
            
            # Market order
            if self.random_state.rand() < self.sigma_mkt:
                s_ = Side.BID if buy_indicator else Side.ASK
                self.place_market_order(self.symbol, quantity=self.size, side=s_)

    def cancelOrder_workflow(self) -> None:
        # Cancel a proportion of the orders
        for order in self.orders.values():
            if self.random_state.rand() < self.delta_c:
                if isinstance(order, LimitOrder):
                    self.cancel_order(order)

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receive_message(current_time, sender_id, message)

        # We have been awakened by something other than our scheduled wakeup.
        # If our internal state indicates we were waiting for a particular event,
        # check if we can transition to a new state.

        if self.state == "AWAITING_SPREAD":
            # We were waiting to receive the current spread/book.  Since we don't currently
            # track timestamps on retained information, we rely on actually seeing a
            # QUERY_SPREAD response message.

            if isinstance(message, QuerySpreadResponseMsg):
                # This is what we were waiting for.
                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed:
                    return
                # We now have the information needed to place a limit order with the eta
                # strategic threshold parameter.
                self.placeOrder()
                self.state = "AWAITING_WAKEUP"

    # Internal state and logic specific to this agent subclass.

    def get_wake_frequency(self) -> NanosecondTime:
        delta_time = self.random_state.exponential(scale=self.mean_wakeup_gap)
        return int(round(delta_time))
