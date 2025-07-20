# models/gan.py

"""
Contains the FGEG model classes and training logic for strategy generation.
Implements Fitness-Guided Evolutionary Generator (FGEG) without a discriminator.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from strategy.strategy_generator import StrategyGenerator
from backtest.backtester import Backtester
from data.data_fetcher import DataFetcher
from config.constants import ASSETS

class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=11):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class StrategyGAN:
    def __init__(self, input_dim=100, output_dim=11, lr=0.0002):
        self.generator = Generator(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.criterion = nn.MSELoss()  # For supervised learning on elite strategies
        self.strategy_gen = StrategyGenerator()
        self.backtester = Backtester()
        self.fetcher = DataFetcher()

    def compute_fitness(self, pnl, sharpe, drawdown, alpha=1.0, beta=0.5, gamma=0.5):
        """
        Compute composite fitness score based on PnL, Sharpe, and Drawdown.
        """
        normalized_pnl = pnl / 100000  # Normalize PnL
        fitness = alpha * normalized_pnl + beta * sharpe - gamma * max(0, drawdown)
        return fitness

    def train(self, data_dict, epochs=100):
        """
        Trains the FGEG model to generate profitable trading strategies using fitness-guided evolution.
        Matches test3.py logic for compatibility.
        """
        batch_size = 20  # Number of strategies to sample per generation
        top_k_ratio = 0.2  # Select top 20% of strategies as elites
        top_k = max(1, int(batch_size * top_k_ratio))

        for epoch in range(epochs):
            # Sample latent noise vectors
            noise = torch.randn(batch_size, self.input_dim)
            fake_vectors = self.generator(noise)
            fake_vectors_denorm = [self.strategy_gen.denormalize_vector(fake_vectors[i].detach().numpy()) for i in range(batch_size)]
            
            # Assign assets cyclically to ensure diversity
            for i in range(len(fake_vectors_denorm)):
                fake_vectors_denorm[i][0] = i % len(ASSETS)

            # Evaluate strategies
            strategies = [self.strategy_gen.vector_to_strategy(vec) for vec in fake_vectors_denorm]
            fitness_scores = []
            for strategy in strategies:
                asset = strategy["strategy"]["metadata"]["asset"]
                timeframe = strategy["strategy"]["metadata"]["timeframe"]
                if (asset, timeframe) not in data_dict:
                    data_dict[(asset, timeframe)] = self.fetcher.fetch(asset, '2020-12-31', timeframe)
                data = data_dict[(asset, timeframe)]
                pnl, sharpe, drawdown = self.backtester.run(strategy, data)
                volatility = self.fetcher.compute_volatility(data)
                normalized_pnl = pnl / (volatility + 1e-6)
                fitness = self.compute_fitness(normalized_pnl, sharpe, drawdown)
                fitness_scores.append(fitness)

            # Select elite strategies
            fitness_scores = np.array(fitness_scores)
            elite_indices = np.argsort(fitness_scores)[-top_k:]  # Top-K indices
            elite_noise = noise[elite_indices]
            elite_vectors = [fake_vectors_denorm[i] for i in elite_indices]
            elite_normals = torch.tensor(np.array([self.strategy_gen.normalize_vector(vec) for vec in elite_vectors]), dtype=torch.float32)

            # Train generator on elite strategies
            self.g_optimizer.zero_grad()
            generated_vectors = self.generator(elite_noise)
            g_loss = self.criterion(generated_vectors, elite_normals)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.g_optimizer.step()

            # Log metrics
            if epoch % 10 == 0:
                avg_fitness = np.mean(fitness_scores)
                best_fitness = np.max(fitness_scores)
                logging.info(f"Epoch {epoch}, G Loss: {g_loss.item():.4f}, Avg Fitness: {avg_fitness:.4f}, Best Fitness: {best_fitness:.4f}")

        return self.generator