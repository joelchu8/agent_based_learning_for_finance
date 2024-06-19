from abides_core.utils import get_wake_time, str_to_ns
########################################################################################################################
############################################### GENERAL CONFIG #########################################################

class ExchangeConfig:
    def __init__(self, book_logging=True, book_log_depth=10, log_orders=True, pipeline_delay=0, computation_delay=0, stream_history=500, use_metric_tracker=True):
        self.book_logging = book_logging
        self.book_log_depth = book_log_depth
        self.log_orders = log_orders
        self.pipeline_delay = pipeline_delay
        self.computation_delay = computation_delay
        self.stream_history = stream_history
        self.use_metric_tracker = use_metric_tracker



class NoiseAgentConfig:
    def __init__(self, sigma_limit=0.1, sigma_mkt=0.05, delta_c=0.1, mean_wakeup_gap=1e9):
        self.sigma_limit = sigma_limit  # NoiseAgent sigma_limit
        self.sigma_mkt = sigma_mkt  # NoiseAgent sigma_mkt
        self.delta_c = delta_c  # NoiseAgent delta_c
        self.mean_wakeup_gap = mean_wakeup_gap  # NoiseAgent mean wakeup gap


class ValueAgentConfig:
    def __init__(self, r_bar=100_000, kappa_limit=1.3, kappa_mkt=0.5, delta_c=0.1, mean_wakeup_gap=1e9):
        self.r_bar = r_bar # true mean fundamental value
        self.kappa_limit = kappa_limit # probability of limit order
        self.kappa_mkt = kappa_mkt # probability of market order
        self.delta_c = delta_c # cancel order probability
        self.mean_wakeup_gap = mean_wakeup_gap # ValueAgent arrival rate


class GBMOracleConfig:
    def __init__(self, mu=3e-5, sigma=0.0335):
        self.mu = mu
        self.sigma = sigma

class FlashCrashOracleConfig:
    def __init__(self, mu=3e-5, sigma=0.0335):
        self.mu = mu
        self.sigma = sigma


class MarketMakerAgentConfig:
    def __init__(self, anchor="middle", window_size="adaptive", pov=0.025, num_ticks=3, wake_up_freq="1S", min_order_size=20, level_spacing=5,
                 cancel_limit_delay=50, skew_beta=0, price_skew_param=None, spread_alpha=0.75, backstop_quantity=0, min_imbalance=0.9,
                 delta_c=0.5, poisson_arrival=True, subscribe=False, subscribe_freq='1S', subscribe_num_levels=1):
        self.anchor = anchor
        self.window_size = window_size  # MarketMakerAgent window size
        self.pov = pov  # MarketMakerAgent pov
        self.num_ticks = num_ticks  # MarketMakerAgent num_ticks
        self.wake_up_freq = wake_up_freq  # MarketMakerAgent wake up frequency
        self.min_order_size = min_order_size  # MarketMakerAgent min order size
        self.level_spacing = level_spacing  # MarketMakerAgent level spacing
        self.cancel_limit_delay = cancel_limit_delay  # MarketMakerAgent cancel limit delay
        self.skew_beta = skew_beta  # MarketMakerAgent skew beta
        self.price_skew_param = price_skew_param  # MarketMakerAgent price skew param
        self.spread_alpha = spread_alpha  # MarketMakerAgent spread alpha
        self.backstop_quantity = backstop_quantity  # MarketMakerAgent backstop quantity
        self.min_imbalance = min_imbalance  # MarketMakerAgent min imbalance
        self.delta_c = delta_c  # MarketMakerAgent delta_c
        self.poisson_arrival = poisson_arrival  # Whether to arrive as a Poisson process
        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscribe_freq = subscribe_freq  # MarketMakerAgent subscribe frequency
        self.subscribe_num_levels = subscribe_num_levels  # MarketMakerAgent subscribe num levels



class MomentumAgentConfig:
    def __init__(self, min_size=20, max_size=50, alpha=0.1, beta_limit=10, beta_mkt=2, delta_c=0.1, gamma=100, wake_up_freq="1s", 
                 data_freq="0.1s", poisson_arrival=True, subscribe=True):
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        self.alpha = alpha  # MomentumAgent alpha
        self.beta_limit = beta_limit  # MomentumAgent beta for limit orders
        self.beta_mkt = beta_mkt  # MomentumAgent beta for market orders
        self.delta_c = delta_c  # MomentumAgent delta_c
        self.gamma = gamma  # MomentumAgent gamma
        self.wake_up_freq = str_to_ns(wake_up_freq)  # MomentumAgent wake up frequency
        self.data_freq = str_to_ns(data_freq)  # MomentumAgent data frequency
        self.poisson_arrival = poisson_arrival  # Whether to arrive as a Poisson process
        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism

class InstitutionalTraderAgentConfig:
    def __init__(self, inventory=1e13, sell_frequency="00:00:02", sell_volume_factor=1000):
        self.inventory = inventory  # how much inventory agent holds to sell off
        self.sell_frequency = sell_frequency  # length of time between each sell order in crash
        self.sell_volume_factor = sell_volume_factor  # factor to multiply by for volume of sell order


def generate_background_config_extra_kvargs(log_dir='simulation'):
    oracle_config = GBMOracleConfig(mu=1e-9, sigma=0.0135)
    mm_config = MarketMakerAgentConfig(price_skew_param=4, wake_up_freq='1s', subscribe=False, subscribe_freq='1s', subscribe_num_levels=10)
    value_agent_config = ValueAgentConfig(kappa_limit=0.3, kappa_mkt=0.1, mean_wakeup_gap=1e8)
    momentum_agent_config = MomentumAgentConfig(beta_limit=50, beta_mkt=20, wake_up_freq='1s', subscribe=False)
    exchange_config = ExchangeConfig(log_orders=True)

    bg_kvargs = {'seed': 50,
                 'end_time': "10:30:00",
                 'log_dir': log_dir,
                 'num_noise_agents': 5,
                 'num_value_agents': 15,
                 'num_mm_agents': 5,
                 'num_momentum_agents': 15,
                 'oracle_params': oracle_config,
                 'mm_agent_params': mm_config,
                 'value_agent_params': value_agent_config,
                 'momentum_agent_params': momentum_agent_config,
                 'exchange_params': exchange_config}
    return bg_kvargs