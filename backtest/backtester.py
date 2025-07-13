# backtest/backtester.py

"""
Handles backtesting of trading strategies.
"""

import numpy as np
import logging

from config.constants import TRANSACTION_COST
from data.data_fetcher import DataFetcher


class Backtester:
    """
    Backtests a given strategy against historical market data.
    """

    def __init__(self):
        self.fetcher = DataFetcher()

    def run(self, strategy_json: dict, data: np.ndarray) -> tuple:
        """
        Backtests a strategy and returns PnL, Sharpe ratio, and Drawdown.
        """
        asset = strategy_json["strategy"]["metadata"]["asset"]
        timeframe = strategy_json["strategy"]["metadata"]["timeframe"]
        entry = strategy_json["strategy"]["entry_rules"]
        exit = strategy_json["strategy"]["exit_rules"]
        stop_loss = strategy_json["strategy"]["stop_loss_rules"]["percentage"] / 100
        risk_per_trade = strategy_json["strategy"]["position_sizing"]["risk_per_trade_percentage"] / 100
        profit_target = exit["condition"]["profit_percentage"] / 100

        if data.size == 0 or len(data) < 50:
            logging.warning(f"Insufficient data for {asset} at {timeframe}")
            return 0, 0, 1.0

        ind_left = self.fetcher.compute_indicator(data, entry["indicator_left"]["name"],
                                                  int(entry["indicator_left"]["parameters"]["period"]),
                                                  entry["indicator_left"]["parameters"]["source"])

        ind_right = self.fetcher.compute_indicator(data, entry["indicator_right"]["name"],
                                                   int(entry["indicator_right"]["parameters"]["period"]),
                                                   entry["indicator_right"]["parameters"]["source"])

        position = 0
        entry_price = 0
        trades = []
        equity = 100_000
        volatility = self.fetcher.compute_volatility(data)
        position_size = (equity * risk_per_trade) / (volatility + 1e-6)

        for i in range(1, len(data)):
            op = entry["operator"]
            signal = self._generate_signal(op, ind_left, ind_right, i)

            if signal and position == 0:
                entry_price = data[i, 3]
                position = position_size / entry_price
                cost = position * entry_price * TRANSACTION_COST
                equity -= cost
                trades.append({'entry_time': i, 'entry_price': entry_price, 'type': 'buy', 'cost': cost})

            if position > 0:
                current_price = data[i, 3]
                profit = (current_price - entry_price) / entry_price

                if profit >= profit_target or profit <= -stop_loss:
                    cost = position * current_price * TRANSACTION_COST
                    equity -= cost
                    trades.append({
                        'exit_time': i,
                        'exit_price': current_price,
                        'type': 'sell',
                        'profit': profit,
                        'cost': cost
                    })
                    position = 0

        if not trades:
            return 0, 0, 1.0

        profits = [t['profit'] for t in trades if 'profit' in t]
        if not profits:
            return 0, 0, 1.0

        total_costs = sum(t['cost'] for t in trades if 'cost' in t)
        pnl = np.sum(profits) * equity - total_costs
        returns = np.array(profits)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252) if len(returns) > 1 else 0
        cum_profits = np.cumsum(profits)
        drawdown = np.min(cum_profits) if len(cum_profits) > 0 else 0

        return pnl, sharpe, drawdown

    def _generate_signal(self, operator: str, left: np.ndarray, right: np.ndarray, i: int) -> bool:
        if operator == "crosses_above" and i > 0 and left[i - 1] < right[i - 1] and left[i] >= right[i]:
            return True
        elif operator == "crosses_below" and i > 0 and left[i - 1] > right[i - 1] and left[i] <= right[i]:
            return True
        elif operator == "greater_than" and left[i] > right[i]:
            return True
        elif operator == "less_than" and left[i] < right[i]:
            return True
        return False
