import logging
from typing import Optional

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from ..generators import OrderSizeGenerator
from ..models.limit_price_model import LimitPriceModel
from ..messages.query import QueryTransactedVolResponseMsg
from ..orders import Side
from .trading_agent import TradingAgent


logger = logging.getLogger(__name__)


class InstitutionalTraderAgent(TradingAgent):
    """
    Institutional Trader wakes up at specified time and begins to sell its inventory.
    """

    def __init__(
        self,
        id: int,
        trigger_time: NanosecondTime,
        symbol: str = "IBM",
        starting_cash: int = 100000,
        name: Optional[str] = None,
        type: Optional[str] = None,
        order_size_model: Optional[OrderSizeGenerator] = None,
        limit_price_model: Optional[LimitPriceModel] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders: bool = False,
        inventory: int = 1e9,
        sell_frequency: str = "00:00:02",
        sell_volume_factor: float = 0.1
    ) -> None:

        # Base class init.
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.trigger_time: NanosecondTime = trigger_time  # time at which agent should start selling off inventory

        self.symbol: str = symbol  # symbol to trade
        self.inventory: int = inventory  # how much inventory agent holds to sell off
        self.holdings[self.symbol] = self.inventory

        self.sell_frequency: str = sell_frequency  # length of time between each sell order in crash
        self.sell_volume_factor = sell_volume_factor  # factor to multiply by for volume of sell order

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading: bool = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state: str = "AWAITING_WAKEUP"

        # The agent must track its previous wake time, so it knows how many time
        # units have passed.
        self.prev_wake_time: Optional[NanosecondTime] = None

        self.size: Optional[int] = (
            self.random_state.randint(20, 50) if order_size_model is None else None
        )

        self.order_size_model = order_size_model  # Probabilistic model for order size
        self.limit_price_model = limit_price_model

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # self.kernel is set in Agent.kernel_initializing()
        # self.exchange_id is set in TradingAgent.kernel_starting()

        super().kernel_starting(start_time)

        self.oracle = self.kernel.oracle

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

        self.state = "INACTIVE"

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

        if self.trigger_time > current_time:
            self.set_wakeup(self.trigger_time)
            return
        
        # if we are triggered to begin selling, we wake up in 2 seconds and sell again
        if current_time >= self.trigger_time and self.symbol in self.holdings and self.holdings[self.symbol] > 0:
            self.set_wakeup(current_time + str_to_ns(self.sell_frequency))

        # if self.mkt_closed and self.symbol not in self.daily_close_price:
        #     self.get_current_spread(self.symbol)
        #     self.state = "AWAITING_SPREAD"
        #     return

        if type(self) == InstitutionalTraderAgent:
            # get recent transacted volume
            self.get_transacted_volume(self.symbol, lookback_period='1min')
            self.state = "AWAITING_TRANSACTED_VOLUME"
        else:
            self.state = "ACTIVE"

    def placeOrder(self, current_time: NanosecondTime) -> None:
        if current_time >= self.trigger_time and self.symbol in self.holdings and self.holdings[self.symbol] > 0:
            
            buy_transacted_volume = self.transacted_volume[self.symbol][0]
            sell_transacted_volume = self.transacted_volume[self.symbol][1]
            total_transacted_volume = buy_transacted_volume + sell_transacted_volume

            # sell order size is dependent on recent transacted volume
            order_size = min(self.holdings[self.symbol], int(total_transacted_volume * self.sell_volume_factor * self.inventory))

            # place market sell order
            self.place_market_order(self.symbol, order_size, Side.ASK)

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receive_message(current_time, sender_id, message)

        # We have been awakened by something other than our scheduled wakeup.
        # If our internal state indicates we were waiting for a particular event,
        # check if we can transition to a new state.

        if self.state == "AWAITING_TRANSACTED_VOLUME":
            # We were waiting to receive recentr transacted volume.

            if isinstance(message, QueryTransactedVolResponseMsg):
                # This is what we were waiting for.

                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed:
                    return

                # We now have the information needed to place a limit order with the eta
                # strategic threshold parameter.
                self.placeOrder(current_time)
                self.state = "AWAITING_WAKEUP"

    # Internal state and logic specific to this agent subclass.

    def get_wake_frequency(self) -> NanosecondTime:
        return self.random_state.randint(low=0, high=100)
