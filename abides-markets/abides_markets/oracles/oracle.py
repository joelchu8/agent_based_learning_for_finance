from abides_core import NanosecondTime


class Oracle:
    def get_daily_open_price(
        self, symbol: str) -> int:
        raise NotImplementedError
