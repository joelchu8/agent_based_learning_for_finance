# RMSC-5 (Reference Market Simulation Configuration):

import os
from datetime import datetime

import numpy as np
import pandas as pd

from abides_core.utils import get_wake_time, str_to_ns
from abides_markets.agents import (
    ExchangeAgent,
    NoiseAgent,
    ValueAgent,
    AdaptiveMarketMakerAgent,
    MomentumAgent,
)
from abides_markets.models import OrderSizeModel, LimitPriceModel
from abides_markets.oracles import SparseMeanRevertingOracle, GeometricBrownianMotionOracle
from abides_markets.utils import generate_latency_model
from abides_markets.configs.agent_params import ExchangeConfig, NoiseAgentConfig, ValueAgentConfig, MarketMakerAgentConfig, MomentumAgentConfig, GBMOracleConfig


def build_config(
    seed=int(datetime.now().timestamp() * 1_000_000) % (2**32 - 1),
    date="20210205",
    end_time="10:00:00",
    stdout_log_level="INFO",
    ticker="ABM",
    starting_cash=10_000_000,  # Cash in this simulator is always in CENTS.
    trader_log_orders=False,  # if True log messages in the traders
    log_dir=None,
    # 1) Exchange Agent
    exchange_params=ExchangeConfig(),
    # 2) Noise Agent
    num_noise_agents=10,
    noise_agent_params=NoiseAgentConfig(),
    # 3) Value Agents
    num_value_agents=10,
    value_agent_params=ValueAgentConfig(),
    # oracle
    oracle_params=GBMOracleConfig(),
    # 4) Market Maker Agents
    num_mm_agents=2,
    mm_agent_params=MarketMakerAgentConfig(),
    # 5) Momentum Agents
    num_momentum_agents=20,
    momentum_agent_params=MomentumAgentConfig(),
):
    """
    create the background configuration for rmsc04
    These are all the non-learning agent that will run in the simulation
    :param seed: seed of the experiment
    :type seed: int
    :param log_orders: debug mode to print more
    :return: all agents of the config
    :rtype: list
    """
    r_bar = value_agent_params.r_bar
    # fix seed
    np.random.seed(seed)

    # order size model
    ORDER_SIZE_MODEL = OrderSizeModel()  # Order size model
    # limit order price model
    LIMIT_PRICE_MODEL = LimitPriceModel()

    # date&time
    DATE = int(pd.to_datetime(date).to_datetime64())
    MKT_OPEN = DATE + str_to_ns("09:30:00")
    MKT_CLOSE = DATE + str_to_ns(end_time)

    # oracle
    oracle = GeometricBrownianMotionOracle(ticker, MKT_OPEN, S0=r_bar, mu=oracle_params.mu, sigma=oracle_params.sigma)

    # Agent configuration
    agent_count, agents, agent_types = 0, [], []

    agents.extend(
        [
            ExchangeAgent(
                id=0,
                mkt_open=MKT_OPEN,
                mkt_close=MKT_CLOSE,
                symbols=[ticker],
                name="ExchangeAgent",
                type="ExchangeAgent",
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")),
                book_logging=exchange_params.book_logging,
                book_log_depth=exchange_params.book_log_depth,
                log_orders=exchange_params.log_orders,
                pipeline_delay=exchange_params.pipeline_delay,
                computation_delay=exchange_params.computation_delay,
                stream_history=exchange_params.stream_history,
                use_metric_tracker=exchange_params.use_metric_tracker,
            )
        ]
    )
    agent_types.extend(["ExchangeAgent"])
    agent_count += 1

    agents.extend(
        [
            NoiseAgent(
                id=j,
                symbol=ticker,
                starting_cash=starting_cash,
                name="NoiseAgent_{}".format(j),
                type="NoiseAgent",
                order_size_model=ORDER_SIZE_MODEL,
                limit_price_model=LIMIT_PRICE_MODEL,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")),
                sigma_limit=noise_agent_params.sigma_limit,
                sigma_mkt=noise_agent_params.sigma_mkt,
                delta_c=noise_agent_params.delta_c,
                mean_wakeup_gap=noise_agent_params.mean_wakeup_gap,
                log_orders=trader_log_orders,
            )
            for j in range(agent_count, agent_count + num_noise_agents)
        ]
    )
    agent_count += num_noise_agents
    agent_types.extend(["NoiseAgent"] * num_noise_agents)
    print('Noise Agents: ', num_noise_agents)

    agents.extend(
        [
            ValueAgent(
                id=j,
                symbol=ticker,
                starting_cash=starting_cash,
                name="ValueAgent_{}".format(j),
                type="ValueAgent",
                order_size_model=ORDER_SIZE_MODEL,
                limit_price_model=LIMIT_PRICE_MODEL,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")),
                r_bar=value_agent_params.r_bar,
                mean_wakeup_gap=value_agent_params.mean_wakeup_gap,
                log_orders=trader_log_orders,
            )
            for j in range(agent_count, agent_count + num_value_agents)
        ]
    )
    agent_count += num_value_agents
    agent_types.extend(["ValueAgent"] * num_value_agents)
    print('Value Agents: ', num_value_agents)

    # market marker derived parameters
    agents.extend(
        [
            AdaptiveMarketMakerAgent(
                id=j,
                symbol=ticker,
                starting_cash=starting_cash,
                name="AdaptiveMarketMaker_{}".format(j),
                type="AdaptivePOVMarketMakerAgent",
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
                log_orders=trader_log_orders,
                anchor=mm_agent_params.anchor,
                window_size=mm_agent_params.window_size,
                pov=mm_agent_params.pov,
                num_ticks=mm_agent_params.num_ticks,
                wake_up_freq=mm_agent_params.wake_up_freq,
                min_order_size=mm_agent_params.min_order_size,
                level_spacing=mm_agent_params.level_spacing,
                cancel_limit_delay=mm_agent_params.cancel_limit_delay,
                skew_beta=mm_agent_params.skew_beta,
                price_skew_param=mm_agent_params.price_skew_param,
                spread_alpha=mm_agent_params.spread_alpha,
                backstop_quantity=mm_agent_params.backstop_quantity,
                min_imbalance=mm_agent_params.min_imbalance,
                delta_c=mm_agent_params.delta_c,
                poisson_arrival=mm_agent_params.poisson_arrival,
                subscribe=mm_agent_params.subscribe,
                subscribe_freq=mm_agent_params.subscribe_freq,
                subscribe_num_levels=mm_agent_params.subscribe_num_levels,
            )
            for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))
        ]
    )
    agent_count += num_mm_agents
    agent_types.extend(["POVMarketMakerAgent"] * num_mm_agents)
    print('Market Maker Agents: ', num_mm_agents)

    agents.extend(
        [
            MomentumAgent(
                id=j,
                name="MomentumAgent_{}".format(j),
                type="MomentumAgent",
                symbol=ticker,
                starting_cash=starting_cash,
                order_size_model=ORDER_SIZE_MODEL,
                limit_price_model=LIMIT_PRICE_MODEL,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")),
                min_size=momentum_agent_params.min_size,
                max_size=momentum_agent_params.max_size,
                alpha=momentum_agent_params.alpha,
                gamma=momentum_agent_params.gamma,
                beta_limit=momentum_agent_params.beta_limit,
                beta_mkt=momentum_agent_params.beta_mkt,
                delta_c=momentum_agent_params.delta_c,
                wake_up_freq=momentum_agent_params.wake_up_freq,
                data_freq=momentum_agent_params.data_freq,
                poisson_arrival=momentum_agent_params.poisson_arrival,
                subscribe=momentum_agent_params.subscribe,
                log_orders=trader_log_orders,
            )
            for j in range(agent_count, agent_count + num_momentum_agents)
        ]
    )
    agent_count += num_momentum_agents
    agent_types.extend(["MomentumAgent"] * num_momentum_agents)
    print('Momentum Agents: ', num_momentum_agents)

    # extract kernel seed here to reproduce the state of random generator in old version
    random_state_kernel = np.random.RandomState(
        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
    )
    # LATENCY
    latency_model = generate_latency_model(agent_count)

    default_computation_delay = 50  # 50 nanoseconds

    ##kernel args
    kernelStartTime = DATE
    kernelStopTime = MKT_CLOSE + str_to_ns("1s")
    
    return {
        "seed": seed,
        "start_time": kernelStartTime,
        "stop_time": kernelStopTime,
        "agents": agents,
        "agent_types": agent_types,
        "agent_latency_model": latency_model,
        "default_computation_delay": default_computation_delay,
        "custom_properties": {"oracle": oracle},
        "random_state_kernel": random_state_kernel,
        "stdout_log_level": stdout_log_level,
        "log_dir": log_dir,
    }
