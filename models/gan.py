# models/gan.py

"""
Contains the GAN model classes and training logic for strategy generation.
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

class Discriminator(nn.Module):
    def __init__(self, input_dim=11):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class StrategyGAN:
    def __init__(self, input_dim=100, output_dim=11, lr=0.0002):
        self.generator = Generator(input_dim, output_dim)
        self.discriminator = Discriminator(output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self.strategy_gen = StrategyGenerator()
        self.backtester = Backtester()
        self.fetcher = DataFetcher()

    def train(self, data_dict, epochs=100):
        """
        Trains the GAN model to generate realistic trading strategy vectors, matching test3.py logic.
        """
        # Generate equal number of strategies per asset
        strategies_per_asset = 20 // len(ASSETS)  # Ensure at least 4 strategies per asset
        real_strategies = []
        for asset_idx in range(len(ASSETS)):
            for _ in range(strategies_per_asset):
                vec = self.strategy_gen.generate_random_vector(asset_idx)
                real_strategies.append(vec)
        
        # Evaluate real strategies
        real_metrics = []
        for vec in real_strategies:
            strategy = self.strategy_gen.vector_to_strategy(vec)
            asset = strategy["strategy"]["metadata"]["asset"]
            timeframe = strategy["strategy"]["metadata"]["timeframe"]
            if (asset, timeframe) not in data_dict:
                data_dict[(asset, timeframe)] = self.fetcher.fetch(asset, '2020-12-31', timeframe)
            data = data_dict[(asset, timeframe)]
            pnl, sharpe, drawdown = self.backtester.run(strategy, data)
            volatility = self.fetcher.compute_volatility(data)
            normalized_pnl = pnl / (volatility + 1e-6)
            real_metrics.append((normalized_pnl, sharpe, drawdown))
        
        # Select top strategies per asset to ensure diversity
        real_indices = []
        for asset_idx in range(len(ASSETS)):
            asset_metrics = [(i, m[0]) for i, m in enumerate(real_metrics) if int(real_strategies[i][0]) == asset_idx]
            asset_indices = [i for i, _ in sorted(asset_metrics, key=lambda x: x[1], reverse=True)[:strategies_per_asset // 2]]
            real_indices.extend(asset_indices)
        
        real_strategies = [real_strategies[i] for i in real_indices]
        
        for epoch in range(epochs):
            self.d_optimizer.zero_grad()
            real_normals = torch.tensor(np.array([self.strategy_gen.normalize_vector(vec) for vec in real_strategies]), dtype=torch.float32)
            real_labels = torch.ones(len(real_strategies), 1)
            d_real = self.discriminator(real_normals)
            d_real_loss = self.criterion(d_real, real_labels)
            
            noise = torch.randn(len(real_strategies), self.input_dim)
            fake_vectors = self.generator(noise)
            fake_vectors_denorm = [self.strategy_gen.denormalize_vector(fake_vectors[i].detach().numpy()) for i in range(len(real_strategies))]
            for i in range(len(fake_vectors_denorm)):
                fake_vectors_denorm[i][0] = i % len(ASSETS)  # Assign each asset cyclically
            fake_normals = torch.tensor(np.array([self.strategy_gen.normalize_vector(vec) for vec in fake_vectors_denorm]), dtype=torch.float32)
            fake_labels = torch.zeros(len(real_strategies), 1)
            d_fake = self.discriminator(fake_normals)
            d_fake_loss = self.criterion(d_fake, fake_labels)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.d_optimizer.step()
            
            self.g_optimizer.zero_grad()
            noise = torch.randn(len(real_strategies), self.input_dim)
            fake_vectors = self.generator(noise)
            fake_vectors_denorm = [self.strategy_gen.denormalize_vector(fake_vectors[i].detach().numpy()) for i in range(len(real_strategies))]
            for i in range(len(fake_vectors_denorm)):
                fake_vectors_denorm[i][0] = i % len(ASSETS)  # Assign each asset cyclically
            fake_normals = torch.tensor(np.array([self.strategy_gen.normalize_vector(vec) for vec in fake_vectors_denorm]), dtype=torch.float32)
            g_output = self.discriminator(fake_normals)
            g_loss = self.criterion(g_output, real_labels)
            
            pnl_loss = 0
            for vec in fake_vectors_denorm:
                strategy = self.strategy_gen.vector_to_strategy(vec)
                asset = strategy["strategy"]["metadata"]["asset"]
                timeframe = strategy["strategy"]["metadata"]["timeframe"]
                if (asset, timeframe) not in data_dict:
                    data_dict[(asset, timeframe)] = self.fetcher.fetch(asset, '2020-12-31', timeframe)
                data = data_dict[(asset, timeframe)]
                pnl, _, _ = self.backtester.run(strategy, data)
                volatility = self.fetcher.compute_volatility(data)
                pnl_loss += max(0, -pnl / (100000 * (volatility + 1e-6)))
            pnl_loss = torch.tensor(pnl_loss / len(real_strategies), requires_grad=True)
            total_g_loss = g_loss + 0.5 * pnl_loss
            total_g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.g_optimizer.step()
            
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {total_g_loss.item():.4f}, PnL Loss: {pnl_loss.item():.4f}")
        
        return self.generator