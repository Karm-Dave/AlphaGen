# main.py

"""
Main orchestrator for training GAN, generating strategies, allocating with PPO,
and evaluating on a test period, matching test3.py logic.
"""

import logging
import torch
import numpy as np

from config.constants import ASSETS
from models.gan import StrategyGAN
from models.ppo import PPOAllocator
from strategy.strategy_generator import StrategyGenerator
from evaluation.evaluator import StrategyEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='quantdaddy1.log',
    filemode='w'
)

def main():
    data_dict = {}
    
    logging.info("Starting GAN training...")
    gan = StrategyGAN()
    generator = gan.train(data_dict, epochs=100)
    
    logging.info("Generating strategies...")
    strategies = []
    strategies_per_asset = 20 // len(ASSETS)  # Match test3.py: 5 strategies per asset
    for asset_idx in range(len(ASSETS)):
        for _ in range(strategies_per_asset):
            noise = torch.randn(1, 100)
            fake_vector = generator(noise).detach().numpy()[0]
            fake_vector[0] = asset_idx  # Force asset index
            strategy_gen = StrategyGenerator()
            strategy_vec = strategy_gen.denormalize_vector(fake_vector)
            strategy = strategy_gen.vector_to_strategy(strategy_vec)
            strategies.append(strategy)
    asset_counts = {asset: sum(1 for s in strategies if s["strategy"]["metadata"]["asset"] == asset) for asset in ASSETS}
    logging.info(f"Generated strategy asset distribution: {asset_counts}")
    
    logging.info("QUANTDADDY allocating with PPO...")
    allocator = PPOAllocator()
    approved_strategies = allocator.allocate(strategies, data_dict, episodes=200)
    
    logging.info("Evaluating test period...")
    evaluator = StrategyEvaluator()
    test_pnl = evaluator.evaluate(approved_strategies, data_dict)
    print(f"Test Period PnL: ${test_pnl:.2f}")
    
    initial_equity = 100000
    total_return = test_pnl / initial_equity
    years = 3
    months = years * 12
    annual_return = ((1 + total_return) ** (1 / years) - 1) * 100
    monthly_return = ((1 + total_return) ** (1 / months) - 1) * 100
    total_return_pct = total_return * 100
    print(f"Profit Report:")
    print(f"Annual Return: {annual_return:.2f}%")
    print(f"Monthly Return: {monthly_return:.2f}%")
    print(f"Total Return: {total_return_pct:.2f}%")

if __name__ == "__main__":
    main()