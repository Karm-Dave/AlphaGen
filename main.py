# main.py

"""
Main orchestrator for training the Fitness-Guided Evolutionary Generator (FGEG),
generating top 5 trading strategies, and evaluating them on a 5-year test period
after 10 years of training data.
"""

import logging
import torch
import numpy as np
from datetime import datetime

from config.constants import ASSETS
from models.fgeg import FGEG
from evaluation.evaluator import StrategyEvaluator
from data.data_fetcher import DataFetcher

# -------------------- SETUP LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="quantdaddy_fgeg.log",
    filemode="w"
)

def main():
    # Initialize required components
    logging.info("Initializing FGEG training pipeline (10Y train / 5Y test)...")
    data_dict = {}
    fetcher = DataFetcher()
    evaluator = StrategyEvaluator()
    
    # -------------------- TRAIN FGEG --------------------
    logging.info("Starting FGEG training...")
    fgeg = FGEG(input_dim=100, output_dim=11, lr=0.0002, top_k=5)
    
    # Training: 2010–2019
    train_start = "2010-01-01"
    train_end = "2019-12-31"
    # Testing: 2020–2024
    test_start = "2020-01-01"
    test_end = "2024-12-31"

    # Pre-fetch minimal data for both train and test periods
    for asset in ASSETS:
        try:
            train_data = fetcher.fetch(asset, train_end, "1d")
            test_data = fetcher.fetch(asset, test_end, "1d")
            data_dict[(asset, "train")] = train_data
            data_dict[(asset, "test")] = test_data
            logging.info(f"Fetched data for {asset} (train + test).")
        except Exception as e:
            logging.error(f"Failed to fetch data for {asset}: {e}")
    
    # Train FGEG using 10 years of data
    top_strategies = fgeg.train(data_dict, epochs=100, batch_size=20)
    logging.info(f"FGEG training completed. Selected top {len(top_strategies)} strategies.")
    
    # -------------------- EVALUATE TEST PERIOD --------------------
    logging.info("Evaluating selected strategies on 5-year test period...")
    equal_alloc = 1.0 / len(top_strategies)
    approved_strategies = [(s, equal_alloc) for s in top_strategies]
    test_pnl = evaluator.evaluate(approved_strategies, data_dict)

    print(f"Test Period PnL (2020–2024): ${test_pnl:.2f}")
    logging.info(f"Test Period PnL (2020–2024): ${test_pnl:.2f}")
    
    # -------------------- PROFIT REPORT --------------------
    initial_equity = 100000
    total_return = test_pnl / initial_equity
    years = 5  # test period duration
    months = years * 12
    annual_return = ((1 + total_return) ** (1 / years) - 1) * 100
    monthly_return = ((1 + total_return) ** (1 / months) - 1) * 100
    total_return_pct = total_return * 100
    
    print("\n📈 Profit Report (5-Year Test Period):")
    print(f"Annual Return:  {annual_return:.2f}%")
    print(f"Monthly Return: {monthly_return:.2f}%")
    print(f"Total Return:   {total_return_pct:.2f}%")

    logging.info(f"Profit Report - Annual: {annual_return:.2f}%, Monthly: {monthly_return:.2f}%, Total: {total_return_pct:.2f}%")
    logging.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
