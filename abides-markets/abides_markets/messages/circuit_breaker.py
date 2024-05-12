from dataclasses import dataclass
from typing import Dict, Optional

from abides_core import Message, NanosecondTime


@dataclass
class CircuitBreakerStart(Message):
    """
    This message is sent by an ``ExchangeAgent`` to all agents if a circuit breaker triggers
    """

    start_time: NanosecondTime  # when the circuit breaker started
    cooldown: NanosecondTime  # how long circuit breaker lasts for

@dataclass
class CircuitBreakerEnd(Message):
    """
    This message is sent by an ``ExchangeAgent`` to all agents when circuit breaker ends
    and trading resumes
    """