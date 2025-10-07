# models/fgeg.py


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from copy import deepcopy

from strategy.strategy_generator import StrategyGenerator
from backtest.backtester import Backtester
from data.data_fetcher import DataFetcher
from config.constants import ASSETS

class Generator(nn.Module):
    """
    Simple feedforward generator network:
    - Input: random latent vector (noise)
    - Output: strategy vector (normalized between 0 and 1)
    """
    def __init__(self, input_dim=100, output_dim=11):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # ensures output is between 0 and 1
        )

    def forward(self, x):
        return self.model(x)

class FGEG:
    """
    Main FGEG class:
    - Generates candidate strategies using the generator
    - Evaluates each strategy using backtester
    - Selects top 5 strategies based on a fitness function
    """
    def __init__(self, input_dim=100, output_dim=11, lr=0.0002, top_k=5):
        self.generator = Generator(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.criterion = nn.MSELoss()  # For supervised training on elite strategies
        self.strategy_gen = StrategyGenerator()
        self.backtester = Backtester()
        self.fetcher = DataFetcher()
        self.top_k = top_k  # number of final strategies to select

    def compute_fitness(self, pnl, sharpe, drawdown, alpha=1.0, beta=0.5, gamma=0.5):
        """
        Composite fitness score combining:
        - PnL (profit)
        - Sharpe ratio
        - Drawdown
        """
        normalized_pnl = pnl / 100000  # scale PnL relative to initial equity
        fitness = alpha * normalized_pnl + beta * sharpe - gamma * max(0, drawdown)
        return fitness

    def train(self, data_dict, epochs=100, batch_size=20):
        """
        FGEG training loop:
        1. Sample random latent vectors
        2. Generate candidate strategies
        3. Evaluate strategies
        4. Select top elite strategies
        5. Train generator to imitate elites
        """
        top_k_ratio = 0.2  # top 20% strategies per generation
        top_k_gen = max(1, int(batch_size * top_k_ratio))

        for epoch in range(epochs):
            # Step 1: Sample random noise
            noise = torch.randn(batch_size, self.input_dim)
            
            # Step 2: Generate candidate strategy vectors
            fake_vectors = self.generator(noise)
            fake_vectors_denorm = [self.strategy_gen.denormalize_vector(fake_vectors[i].detach().numpy()) for i in range(batch_size)]

            # Step 2b: Assign assets cyclically to ensure diversity
            for i in range(len(fake_vectors_denorm)):
                fake_vectors_denorm[i][0] = i % len(ASSETS)

            # Step 3: Convert vectors to strategy dicts
            strategies = [self.strategy_gen.vector_to_strategy(vec) for vec in fake_vectors_denorm]

            # Step 3b: Evaluate fitness for each strategy
            fitness_scores = []
            for strategy in strategies:
                asset = strategy["strategy"]["metadata"]["asset"]
                timeframe = strategy["strategy"]["metadata"]["timeframe"]
                
                # Fetch data if not already present
                if (asset, timeframe) not in data_dict:
                    data_dict[(asset, timeframe)] = self.fetcher.fetch(asset, '2020-12-31', timeframe)
                data = data_dict[(asset, timeframe)]

                pnl, sharpe, drawdown = self.backtester.run(strategy, data)
                volatility = self.fetcher.compute_volatility(data)
                normalized_pnl = pnl / (volatility + 1e-6)
                fitness = self.compute_fitness(normalized_pnl, sharpe, drawdown)
                fitness_scores.append(fitness)

            # Step 4: Select elite strategies
            fitness_scores = np.array(fitness_scores)
            elite_indices = np.argsort(fitness_scores)[-top_k_gen:]
            elite_noise = noise[elite_indices]
            elite_vectors = [fake_vectors_denorm[i] for i in elite_indices]
            elite_normals = torch.tensor(np.array([self.strategy_gen.normalize_vector(vec) for vec in elite_vectors]), dtype=torch.float32)

            # Step 5: Train generator to imitate elites
            self.g_optimizer.zero_grad()
            generated_vectors = self.generator(elite_noise)
            g_loss = self.criterion(generated_vectors, elite_normals)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.g_optimizer.step()

            if epoch % 10 == 0:
                avg_fitness = np.mean(fitness_scores)
                best_fitness = np.max(fitness_scores)
                logging.info(f"Epoch {epoch}, G Loss: {g_loss.item():.4f}, Avg Fitness: {avg_fitness:.4f}, Best Fitness: {best_fitness:.4f}")

        # After training, pick final top-K strategies
        final_noise = torch.randn(batch_size, self.input_dim)
        final_vectors = self.generator(final_noise)
        final_vectors_denorm = [self.strategy_gen.denormalize_vector(final_vectors[i].detach().numpy()) for i in range(batch_size)]
        final_strategies = [self.strategy_gen.vector_to_strategy(vec) for vec in final_vectors_denorm]

        final_fitness = []
        for strategy in final_strategies:
            asset = strategy["strategy"]["metadata"]["asset"]
            timeframe = strategy["strategy"]["metadata"]["timeframe"]
            if (asset, timeframe) not in data_dict:
                data_dict[(asset, timeframe)] = self.fetcher.fetch(asset, '2020-12-31', timeframe)
            data = data_dict[(asset, timeframe)]
            pnl, sharpe, drawdown = self.backtester.run(strategy, data)
            volatility = self.fetcher.compute_volatility(data)
            normalized_pnl = pnl / (volatility + 1e-6)
            fitness = self.compute_fitness(normalized_pnl, sharpe, drawdown)
            final_fitness.append(fitness)

        top_indices = np.argsort(final_fitness)[-self.top_k:]
        top_strategies = [final_strategies[i] for i in top_indices]

        logging.info(f"Selected top {self.top_k} strategies after FGEG training.")
        return top_strategies
