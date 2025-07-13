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

        period_left = np.random.randint(*PERIOD_RANGE)
        period_right = np.random.randint(*PERIOD_RANGE)
        stop_loss = np.random.uniform(*STOP_LOSS_RANGE)
        risk_per_trade = np.random.uniform(*RISK_PER_TRADE_RANGE)
        profit_target = np.random.uniform(*PROFIT_TARGET_RANGE)

        vector = np.array([
            asset_idx, timeframe_idx, ind_left_idx, ind_right_idx,
            op_idx, source_idx, period_left, period_right,
            stop_loss, risk_per_trade, profit_target
        ], dtype=np.float32)

        return vector

    def vector_to_strategy(self, vector: np.ndarray) -> dict:
        strategy = deepcopy(STRATEGY_TEMPLATE)
        strategy["strategy"]["metadata"]["name"] = f"Strategy_{uuid.uuid4().hex[:8]}"
        strategy["strategy"]["metadata"]["asset"] = ASSETS[int(vector[0])]
        strategy["strategy"]["metadata"]["timeframe"] = TIMEFRAMES[int(vector[1])]

        entry = strategy["strategy"]["entry_rules"]
        entry["indicator_left"]["name"] = INDICATORS[int(vector[2])]
        entry["indicator_right"]["name"] = INDICATORS[int(vector[3])]
        entry["operator"] = OPERATORS[int(vector[4])]

        source = PRICE_TYPES[int(vector[5])]
        entry["indicator_left"]["parameters"]["source"] = source
        entry["indicator_right"]["parameters"]["source"] = source

        entry["indicator_left"]["parameters"]["period"] = int(vector[6])
        entry["indicator_right"]["parameters"]["period"] = int(vector[7])

        strategy["strategy"]["stop_loss_rules"]["percentage"] = vector[8]
        strategy["strategy"]["position_sizing"]["risk_per_trade_percentage"] = vector[9]
        strategy["strategy"]["exit_rules"]["condition"]["profit_percentage"] = vector[10]

        return strategy

    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm_vector = vector.copy()
        ranges = self._get_ranges()

        for i, (min_val, max_val) in enumerate(ranges):
            if max_val != min_val:
                norm_vector[i] = (vector[i] - min_val) / (max_val - min_val)
            else:
                norm_vector[i] = 0.0

        return np.clip(norm_vector, 0, 1)

    def denormalize_vector(self, norm_vector: np.ndarray) -> np.ndarray:
        vector = norm_vector.copy()
        ranges = self._get_ranges()

        for i, (min_val, max_val) in enumerate(ranges):
            if max_val != min_val:
                vector[i] = np.clip(vector[i] * (max_val - min_val) + min_val, min_val, max_val)
            else:
                vector[i] = min_val

            # Round categorical indices
            if i < 6:
                vector[i] = round(vector[i])

        return vector

    def _get_ranges(self):
        return [
            (0, len(ASSETS) - 1),
            (0, len(TIMEFRAMES) - 1),
            (0, len(INDICATORS) - 1),
            (0, len(INDICATORS) - 1),
            (0, len(OPERATORS) - 1),
            (0, len(PRICE_TYPES) - 1),
            PERIOD_RANGE,
            PERIOD_RANGE,
            STOP_LOSS_RANGE,
            RISK_PER_TRADE_RANGE,
            PROFIT_TARGET_RANGE
        ]
