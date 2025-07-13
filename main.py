# main.py

"""
Main orchestrator for training GAN, generating strategies, allocating with PPO,
and evaluating on a test period.
"""

import logging
import torch

from config.constants import ASSETS
from models.gan import StrategyGAN
from models.ppo import PPOAllocator
from strategy.strategy_generator import StrategyGenerator
from evaluation.evaluator import StrategyEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='quantdaddy.log',
    filemode='w'
)


def main():
    data_dict = {}
    strategy_gen = StrategyGenerator()
    evaluator = StrategyEvaluator()

    logging.info("=== Training GAN ===")
    gan = StrategyGAN()
    generator = gan.train(data_dict, num_assets=len(ASSETS), strategies_per_asset=20, epochs=100)

    logging.info("=== Generating Strategies ===")
    strategies = []
    for asset_idx in range(len(ASSETS)):
        for _ in range(20):
            noise = torch.randn(1, 100)
            fake_vec = generator(noise).detach().numpy()[0]
            fake_vec[0] = asset_idx  # enforce asset diversity
            denorm_vec = strategy_gen.denormalize_vector(fake_vec)
            strat = strategy_gen.vector_to_strategy(denorm_vec)
            strategies.append(strat)

    asset_counts = {asset: sum(1 for s in strategies if s["strategy"]["metadata"]["asset"] == asset) for asset in ASSETS}
    logging.info(f"Generated strategy asset distribution: {asset_counts}")

    logging.info("=== Allocating with PPO ===")
    allocator = PPOAllocator()
    approved_allocations = allocator.allocate(strategies, data_dict, episodes=200)

    if not approved_allocations:
        logging.warning("No strategies approved. Exiting.")
        return

    logging.info("=== Evaluating on Test Period ===")
    test_pnl = evaluator.evaluate(approved_allocations, data_dict)

    print(f"\nTest Period PnL: ${test_pnl:.2f}")

    summary = evaluator.summarize_returns(test_pnl)
    print("\n=== Profit Report ===")
    print(f"Annual Return: {summary['annual_return_pct']}%")
    print(f"Monthly Return: {summary['monthly_return_pct']}%")
    print(f"Total Return: {summary['total_return_pct']}%")


if __name__ == "__main__":
    main()
