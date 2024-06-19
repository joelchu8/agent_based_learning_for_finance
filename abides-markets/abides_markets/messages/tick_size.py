from dataclasses import dataclass
from typing import Dict, Optional

from abides_core import Message, NanosecondTime


@dataclass
class TickSizeChange(Message):
    """
    This message is sent by an ``ExchangeAgent`` to all agents if a circuit breaker triggers
    """

    new_tick_size: int  # new minimum tick size for the stock