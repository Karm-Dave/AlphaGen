# backtest/backtester.py

"""
Handles backtesting of trading strategies, matching test3.py's backtest_strategy logic.
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

    def run(self, strategy: dict, data: np.ndarray) -> tuple:
        """
        Backtests a strategy and returns PnL, Sharpe ratio, and Drawdown.
        Matches test3.py's backtest_strategy exactly.
        """
        # Extract strategy parameters
        asset = strategy["strategy"]["metadata"]["asset"]
        timeframe = strategy["strategy"]["metadata"]["timeframe"]
        entry = strategy["strategy"]["entry_rules"]
        exit = strategy["strategy"]["exit_rules"]
        stop_loss = strategy["strategy"]["stop_loss_rules"]["percentage"] / 100
        risk_per_trade = strategy["strategy"]["position_sizing"]["risk_per_trade_percentage"] / 100
        profit_target = exit["condition"]["profit_percentage"] / 100

        # Check for insufficient data
        if data.size == 0 or len(data) < 50:
            logging.warning(f"Insufficient data for {asset} at {timeframe}")
            return 0, 0, 1.0

        # Compute indicators
        ind_left = self.fetcher.compute_indicator(
            data,
            entry["indicator_left"]["name"],
            int(entry["indicator_left"]["parameters"]["period"]),
            entry["indicator_left"]["parameters"]["source"]
        )
        ind_right = self.fetcher.compute_indicator(
            data,
            entry["indicator_right"]["name"],
            int(entry["indicator_right"]["parameters"]["period"]),
            entry["indicator_right"]["parameters"]["source"]
        )

        # Initialize trading variables
        position = 0
        entry_price = 0
        trades = []
        equity = 100000
        volatility = self.fetcher.compute_volatility(data)
        position_size = (equity * risk_per_trade) / (volatility + 1e-6)

        # Backtest loop
        for i in range(1, len(data)):
            op = entry["operator"]
            # Generate signal based on operator
            signal = False
            if op == "crosses_above" and i > 0 and ind_left[i-1] < ind_right[i-1] and ind_left[i] >= ind_right[i]:
                signal = True
            elif op == "crosses_below" and i > 0 and ind_left[i-1] > ind_right[i-1] and ind_left[i] <= ind_right[i]:
                signal = True
            elif op == "greater_than" and ind_left[i] > ind_right[i]:
                signal = True
            elif op == "less_than" and ind_left[i] < ind_right[i]:
                signal = True

            # Enter position on signal
            if signal and position == 0:
                entry_price = data[i, 3]  # Close price
                position = position_size / entry_price
                cost = position * entry_price * TRANSACTION_COST
                equity -= cost
                trades.append({'entry_time': i, 'entry_price': entry_price, 'type': 'buy', 'cost': cost})

            # Check for exit conditions
            if position > 0:
                current_price = data[i, 3]  # Close price
                profit = (current_price - entry_price) / entry_price
                # Exit on profit target
                if profit >= profit_target:
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
                # Exit on stop-loss
                elif profit <= -stop_loss:
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

        # Compute performance metrics
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