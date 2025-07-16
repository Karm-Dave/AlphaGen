# strategy/strategy_generator.py

"""
Handles generation and transformation of strategy vectors.
"""

import uuid
import numpy as np
from copy import deepcopy

from config.constants import (
    ASSETS, TIMEFRAMES, INDICATORS, OPERATORS, PRICE_TYPES,
    PERIOD_RANGE, STOP_LOSS_RANGE, RISK_PER_TRADE_RANGE,
    PROFIT_TARGET_RANGE, STRATEGY_TEMPLATE
)

class StrategyGenerator:
    """
    Class for creating, normalizing, and converting strategy vectors.
    """

    def __init__(self):
        self.asset_count = len(ASSETS)

    def generate_random_vector(self, asset_idx: int) -> np.ndarray:
        timeframe_idx = np.random.randint(len(TIMEFRAMES))
        ind_left_idx = np.random.randint(len(INDICATORS))
        ind_right_idx = np.random.randint(len(INDICATORS))
        op_idx = np.random.randint(len(OPERATORS))
        source_idx = np.random.randint(len(PRICE_TYPES))

        period_left = np.random.randint(PERIOD_RANGE[0], PERIOD_RANGE[1]+1)
        period_right = np.random.randint(PERIOD_RANGE[0], PERIOD_RANGE[1]+1)
        stop_loss = np.random.uniform(STOP_LOSS_RANGE[0], STOP_LOSS_RANGE[1])
        risk_per_trade = np.random.uniform(RISK_PER_TRADE_RANGE[0], RISK_PER_TRADE_RANGE[1])
        profit_target = np.random.uniform(PROFIT_TARGET_RANGE[0], PROFIT_TARGET_RANGE[1])

        vector = np.array([
            asset_idx, timeframe_idx, ind_left_idx, ind_right_idx,
            op_idx, source_idx, period_left, period_right,
            stop_loss, risk_per_trade, profit_target
        ], dtype=np.float32)

        return vector

    def vector_to_strategy(self, vector: np.ndarray) -> dict:
        strategy = deepcopy(STRATEGY_TEMPLATE)
        strategy["strategy"]["metadata"]["name"] = f"Strategy_{uuid.uuid4().hex[:8]}"
        asset_idx = int(np.clip(vector[0], 0, len(ASSETS)-1))
        strategy["strategy"]["metadata"]["asset"] = ASSETS[asset_idx]
        strategy["strategy"]["metadata"]["timeframe"] = TIMEFRAMES[int(vector[1])]
        strategy["strategy"]["entry_rules"]["indicator_left"]["name"] = INDICATORS[int(vector[2])]
        strategy["strategy"]["entry_rules"]["indicator_right"]["name"] = INDICATORS[int(vector[3])]
        strategy["strategy"]["entry_rules"]["operator"] = OPERATORS[int(vector[4])]
        source = PRICE_TYPES[int(vector[5])]
        strategy["strategy"]["entry_rules"]["indicator_left"]["parameters"]["source"] = source
        strategy["strategy"]["entry_rules"]["indicator_right"]["parameters"]["source"] = source
        strategy["strategy"]["entry_rules"]["indicator_left"]["parameters"]["period"] = int(vector[6])
        strategy["strategy"]["entry_rules"]["indicator_right"]["parameters"]["period"] = int(vector[7])
        strategy["strategy"]["stop_loss_rules"]["percentage"] = vector[8]
        strategy["strategy"]["position_sizing"]["risk_per_trade_percentage"] = vector[9]
        strategy["strategy"]["exit_rules"]["condition"]["profit_percentage"] = vector[10]
        return strategy

    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm_vector = vector.copy()
        ranges = [
            (0, len(ASSETS) - 1), (0, len(TIMEFRAMES) - 1), (0, len(INDICATORS) - 1),
            (0, len(INDICATORS) - 1), (0, len(OPERATORS) - 1), (0, len(PRICE_TYPES) - 1),
            PERIOD_RANGE, PERIOD_RANGE, STOP_LOSS_RANGE, RISK_PER_TRADE_RANGE, PROFIT_TARGET_RANGE
        ]
        for i, (min_val, max_val) in enumerate(ranges):
            if max_val != min_val:
                norm_vector[i] = (vector[i] - min_val) / (max_val - min_val)
            else:
                norm_vector[i] = 0
        norm_vector = np.clip(norm_vector, 0, 1)
        return norm_vector

    def denormalize_vector(self, norm_vector: np.ndarray) -> np.ndarray:
        vector = norm_vector.copy()
        ranges = [
            (0, len(ASSETS) - 1), (0, len(TIMEFRAMES) - 1), (0, len(INDICATORS) - 1),
            (0, len(INDICATORS) - 1), (0, len(OPERATORS) - 1), (0, len(PRICE_TYPES) - 1),
            PERIOD_RANGE, PERIOD_RANGE, STOP_LOSS_RANGE, RISK_PER_TRADE_RANGE, PROFIT_TARGET_RANGE
        ]
        for i, (min_val, max_val) in enumerate(ranges):
            if max_val != min_val:
                vector[i] = np.clip(vector[i] * (max_val - min_val) + min_val, min_val, max_val)
            else:
                vector[i] = min_val
            if i < 6:  # Categorical indices
                vector[i] = np.round(vector[i])
        return vector