import logging
from typing import Optional

import math
import numpy as np

from abides_core import Message, NanosecondTime

from ..messages.query import QuerySpreadResponseMsg
from ..orders import Side, LimitOrder
from ..models.limit_price_model import LimitPriceModel
from .trading_agent import TradingAgent


logger = logging.getLogger(__name__)


class ValueAgent(TradingAgent):
    def __init__(
        self,
        id: int,
        symbol: str = "IBM",
        starting_cash: int = 100_000,
        name: Optional[str] = None,
        type: Optional[str] = None,
        order_size_model=None,
        limit_price_model: Optional[LimitPriceModel] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders: float = False,
        r_bar: int = 100_000,
        kappa_limit: float = 0.3,
        kappa_mkt: float = 0.05,
        delta_c: float = 0.1,
        mean_wakeup_gap: int = 1e9,
    ) -> None:
        # Base class init.
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        # Store important parameters particular to the ZI agent.
        self.symbol: str = symbol  # symbol to trade
        self.r_bar: int = r_bar  # true mean fundamental value
        self.mean_wakeup_gap: int = mean_wakeup_gap  # mean wake up gap
        self.kappa_limit: float = kappa_limit  # probability of limit order
        self.kappa_mkt: float = kappa_mkt  # probability of market order
        self.delta_c: float = delta_c  # cancel order probability

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading: bool = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state: str = "AWAITING_WAKEUP"

        # The agent maintains two priors: r_t and sigma_t (value and error estimates).
        self.r_t: int = r_bar

        self.size: Optional[int] = (
            self.random_state.randint(20, 50) if order_size_model is None else None
        )
        self.order_size_model = order_size_model  # Probabilistic model for order size
        self.limit_price_model = limit_price_model

        self.depth_spread: int = 2

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # self.kernel is set in Agent.kernel_initializing()
        # self.exchange_id is set in TradingAgent.kernel_starting()

        super().kernel_starting(start_time)

        self.oracle = self.kernel.oracle

    def kernel_stopping(self) -> None:
        # Always call parent method to be safe.
        super().kernel_stopping()

        # Print end of day valuation.
        H = int(round(self.get_holdings(self.symbol), -2) / 100)
        # May request real fundamental value from oracle as part of final cleanup/stats.

        # marked to fundamental
        rT = self.oracle.observe_price(
            self.symbol, self.current_time, random_state=self.random_state
        )

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

        delta_time = self.random_state.exponential(scale=self.mean_wakeup_gap)
        self.set_wakeup(current_time + int(round(delta_time)))

        # If we get here, we are in the middle of the trading day and we need to
        self.cancel_all_orders()
        self.get_current_spread(self.symbol)
        self.state = "AWAITING_SPREAD"

    def updateEstimates(self) -> int:
        # Called by a background agent that wishes to obtain a new fundamental observation,
        # update its internal estimation parameters, and compute a new total valuation for the
        # action it is considering.

        # The agent obtains a new noisy observation of the current fundamental value
        # and uses this to update its internal estimates in a Bayesian manner.

        obs_t = self.oracle.observe_price(
            self.symbol,
            self.current_time,
            random_state=self.random_state,
        )

        logger.debug("{} observed {} at {}", self.name, obs_t, self.current_time)

        # Update internal estimates of the current fundamental value and our error of same

        return obs_t

    def placeOrder(self) -> None:
        # estimate final value of the fundamental price
        # used for surplus calculation
        r_T = self.updateEstimates()

        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)

        if bid and ask:
            mid = int((ask + bid) / 2)

            if r_T < mid:
                # fundamental belief that price will go down, place a sell order
                side = Side.ASK
                ref_prc = mid
                distortion = (mid / r_T - 1) * 100.
            elif r_T >= mid:
                # fundamental belief that price will go up, buy order
                side = Side.BID
                ref_prc = mid
                distortion = (r_T/ mid - 1) * 100.
        else:
            # initialize randomly
            buy = self.random_state.randint(0, 1 + 1)
            side = Side.BID if buy == 1 else Side.ASK
            ref_prc = r_T
            distortion = None

        # Place the order
        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)

        if self.size > 0:
            if distortion is None: # When there is no bid and ask in the beginning. Submit order to initial the market
                prc_ = self.limit_price_model.sample(ref_prc, side, random_state=self.random_state)
                self.place_limit_order(self.symbol, self.size, side, prc_)
            else:
                if self.random_state.rand() < self.kappa_limit * distortion:
                    prc_ = self.limit_price_model.sample(ref_prc, side, random_state=self.random_state)
                    self.place_limit_order(self.symbol, math.ceil(self.size * distortion * 10), side, prc_)
                if self.random_state.rand() < self.kappa_mkt * distortion:
                    self.place_market_order(self.symbol, quantity=math.ceil(self.size * distortion * 10), side=side)

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

        # Cancel all open orders.
        # Return value: did we issue any cancellation requests?

    def get_wake_frequency(self) -> NanosecondTime:
        delta_time = self.random_state.exponential(scale=self.mean_wakeup_gap)
        return int(round(delta_time))
