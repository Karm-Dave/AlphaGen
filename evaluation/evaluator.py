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

    def evaluate(self, approved_strategies: list, data_dict: dict, test_end_date: str = '2023-12-31') -> float:
        """
        Evaluates strategies on test data and logs detailed PnL and allocations.
        """
        if not approved_strategies:
            logging.warning("No approved strategies to evaluate.")
            return 0.0

        total_pnl = 0
        asset_counts = {asset: 0 for asset in ASSETS}

        for strategy, alloc in approved_strategies:
            asset = strategy["strategy"]["metadata"]["asset"]
            timeframe = strategy["strategy"]["metadata"]["timeframe"]
            asset_counts[asset] += 1
            key = (asset, f"{timeframe}_test")

            if key not in data_dict:
                data_dict[key] = self.fetcher.fetch(asset, test_end_date, interval='1d')

            data = data_dict[key]
            pnl, sharpe, drawdown = self.backtester.run(strategy, data)
            weighted_pnl = pnl * alloc
            total_pnl += weighted_pnl

            logging.info(
                f"[TEST] {asset} | PnL=${pnl:.2f}, Sharpe={sharpe:.2f}, Drawdown={drawdown:.2f}, Allocation={alloc:.4f}"
            )

        logging.info(f"[TEST] Asset distribution in test strategies: {asset_counts}")
        logging.info(f"[TEST] Total Test Period PnL: ${total_pnl:.2f}")

        return total_pnl

    def summarize_returns(self, total_pnl: float, initial_equity: float = 100000.0, years: int = 3) -> dict:
        """
        Computes total, annual, and monthly returns.
        """
        total_return = total_pnl / initial_equity
        months = years * 12
        annual_return = ((1 + total_return) ** (1 / years) - 1) * 100
        monthly_return = ((1 + total_return) ** (1 / months) - 1) * 100
        total_return_pct = total_return * 100

        summary = {
            "total_return_pct": round(total_return_pct, 2),
            "annual_return_pct": round(annual_return, 2),
            "monthly_return_pct": round(monthly_return, 2)
        }

        logging.info(f"[REPORT] Profit Summary: {summary}")
        return summary
