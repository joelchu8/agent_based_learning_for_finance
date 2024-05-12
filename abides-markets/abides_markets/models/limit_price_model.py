import numpy as np
from ..orders import Side


class LimitPriceModel:
    def __init__(self, scale: float = 2.) -> None:
        self.scale = scale

    def sample(self, ref_prc: float, s_: Side, random_state: np.random.RandomState) -> int:
        if s_ == Side.BID:
            return round(ref_prc - np.ceil(random_state.exponential(scale=self.scale)))
        else:
            return round(ref_prc + np.ceil(random_state.exponential(scale=self.scale)))