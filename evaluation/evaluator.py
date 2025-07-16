# evaluation/evaluator.py

"""
Evaluates strategy performance during a test period.
"""

import logging
import numpy as np

from data.data_fetcher import DataFetcher
from backtest.backtester import Backtester
from config.constants import ASSETS

class StrategyEvaluator:
    """
    Evaluates approved strategies during a separate test period.
    """

    def __init__(self):
        self.backtester = Backtester()
        self.fetcher = DataFetcher()

    def evaluate(self, approved_strategies: list, data_dict: dict) -> float:
        """
        Evaluates strategies on test data and logs detailed PnL and allocations.
        """
        if not approved_strategies:
            logging.warning("No approved strategies for test period evaluation")
            return 0

        total_pnl = 0
        asset_counts = {asset: 0 for asset in ASSETS}
        for strategy, allocation in approved_strategies:
            asset = strategy["strategy"]["metadata"]["asset"]
            timeframe = strategy["strategy"]["metadata"]["timeframe"]
            asset_counts[asset] += 1
            data_key = (asset, timeframe + '_test')
            data = data_dict.get(data_key)
            if data is None:
                data = self.fetcher.fetch(asset, '2023-12-31', '1d')
                data_dict[data_key] = data
            pnl, sharpe, drawdown = self.backtester.run(strategy, data)
            total_pnl += pnl * allocation
            logging.info(f"Test Strategy for {asset} at {timeframe}: PnL=${pnl:.2f}, Sharpe={sharpe:.2f}, Drawdown={drawdown:.2f}, Allocation={allocation:.4f}")
        logging.info(f"Asset distribution in test strategies: {asset_counts}")
        logging.info(f"Total Test Period PnL: ${total_pnl:.2f}")
        return total_pnl

    def summarize_returns(self, total_pnl: float) -> dict:
        """
        Computes total, annual, and monthly returns.
        """
        initial_equity = 100000
        total_return = total_pnl / initial_equity
        years = 3
        months = years * 12
        annual_return = ((1 + total_return) ** (1 / years) - 1) * 100
        monthly_return = ((1 + total_return) ** (1 / months) - 1) * 100
        total_return_pct = total_return * 100
        summary = {
            "total_return_pct": total_return_pct,
            "annual_return_pct": annual_return,
            "monthly_return_pct": monthly_return
        }
        return summary