# models/ppo.py

"""
Implements PPO-based allocation optimization using actor-critic reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Normal

from backtest.backtester import Backtester
from data.data_fetcher import DataFetcher
from config.constants import ASSETS

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        mean = self.model(state)
        dist = Normal(mean, self.log_std.exp())
        return dist

class PPOCritic(nn.Module):
    def __init__(self, state_dim):
        super(PPOCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        return self.model(state)

class PPOAllocator:
    def __init__(self):
        self.backtester = Backtester()
        self.fetcher = DataFetcher()

    def adjust_allocations(self, allocations, strategies, min_alloc=0.15, max_asset_alloc=0.4, min_asset_alloc=0.1):
        allocations = np.array(allocations, dtype=np.float32)
        asset_allocs = {asset: 0 for asset in ASSETS}
        for alloc, strat in zip(allocations, strategies):
            asset = strat["strategy"]["metadata"]["asset"]
            asset_allocs[asset] += alloc
        
        # Enforce max asset allocation (40%) and min asset allocation (10%)
        for asset, total in asset_allocs.items():
            if total > max_asset_alloc:
                scale = max_asset_alloc / total
                for i, strat in enumerate(strategies):
                    if strat["strategy"]["metadata"]["asset"] == asset:
                        allocations[i] *= scale
            elif total < min_asset_alloc and total > 0:
                scale = min_asset_alloc / total
                for i, strat in enumerate(strategies):
                    if strat["strategy"]["metadata"]["asset"] == asset:
                        allocations[i] *= scale
        
        # Enforce min allocation per strategy (15%)
        for i in range(len(allocations)):
            if allocations[i] < min_alloc:
                allocations[i] = min_alloc
        
        # Normalize to sum to 1
        total = allocations.sum()
        if total > 0:
            allocations /= total
        
        return allocations

    def allocate(self, strategies, data_dict, episodes=200):
        if not strategies:
            logging.warning("No strategies provided for allocation")
            return []
        
        approved = []
        metrics = []
        asset_counts = {asset: 0 for asset in ASSETS}
        # Collect metrics per asset
        asset_strategies = {asset: [] for asset in ASSETS}
        for strategy in strategies:
            asset = strategy["strategy"]["metadata"]["asset"]
            timeframe = strategy["strategy"]["metadata"]["timeframe"]
            if (asset, timeframe) not in data_dict:
                data_dict[(asset, timeframe)] = self.fetcher.fetch(asset, '2020-12-31', timeframe)
            data = data_dict[(asset, timeframe)]
            pnl, sharpe, drawdown = self.backtester.run(strategy, data)
            volatility = self.fetcher.compute_volatility(data)
            normalized_pnl = pnl / (volatility + 1e-6)
            asset_strategies[asset].append((strategy, normalized_pnl, sharpe, drawdown))
        
        # Approve top strategies per asset to ensure diversity
        min_strategies_per_asset = 2
        for asset in ASSETS:
            asset_metrics = asset_strategies[asset]
            if asset_metrics:
                sorted_metrics = sorted(asset_metrics, key=lambda x: x[1], reverse=True)
                for strategy, normalized_pnl, sharpe, drawdown in sorted_metrics[:min_strategies_per_asset]:
                    initial_equity = 100000
                    test_period_days = 1095
                    annual_pnl_threshold = initial_equity * 0.1 * (test_period_days / 365)
                    if normalized_pnl * volatility > annual_pnl_threshold and sharpe > 0.8 and drawdown > -0.25:
                        approved.append(strategy)
                        metrics.append([normalized_pnl, sharpe, drawdown])
                        asset_counts[asset] += 1
        
        if not approved:
            logging.info("No strategies approved by QUANTDADDY initially")
            return []
        
        state_dim = len(approved) * 3
        action_dim = len(approved)
        actor = PPOActor(state_dim, action_dim)
        critic = PPOCritic(state_dim)
        actor_optimizer = optim.Adam(actor.parameters(), lr=0.0003, weight_decay=1e-5)
        critic_optimizer = optim.Adam(critic.parameters(), lr=0.0003, weight_decay=1e-5)
        scheduler = StepLR(actor_optimizer, step_size=50, gamma=0.9)
        
        total_pnl = 0
        for episode in range(episodes):
            state = torch.tensor(np.array(metrics).flatten(), dtype=torch.float32).unsqueeze(0)
            if state.shape[1] != state_dim:
                state = state[:, :state_dim]
                if state.shape[1] < state_dim:
                    state = torch.cat((state, torch.zeros(1, state_dim - state.shape[1])), dim=1)
            
            dist = actor(state)
            allocation = dist.sample()
            allocation = torch.clamp(allocation, 0.0, 1.0)
            allocation_sum = torch.sum(allocation)
            if allocation_sum > 0:
                allocation = allocation / allocation_sum
            
            portfolio_pnl = 0
            portfolio_returns = []
            for i, (strategy, alloc) in enumerate(zip(approved, allocation[0])):
                asset = strategy["strategy"]["metadata"]["asset"]
                timeframe = strategy["strategy"]["metadata"]["timeframe"]
                data = data_dict[(asset, timeframe)]
                pnl, _, _ = self.backtester.run(strategy, data)
                portfolio_pnl += pnl * alloc.item()
                portfolio_returns.append(pnl / 100000 * alloc.item())
            
            asset_diversity = len(set(s["strategy"]["metadata"]["asset"] for s in approved))
            diversity_penalty = max(0, (len(ASSETS) - asset_diversity) / len(ASSETS)) * 2.0
            reward = portfolio_pnl / 100000 - 0.5 * np.std(portfolio_returns) - diversity_penalty
            
            constraint_penalty = 0
            for alloc in allocation[0]:
                if alloc < 0.15:
                    constraint_penalty += (0.15 - alloc)**2
            asset_allocs = {asset: sum(alloc.item() for s, alloc in zip(approved, allocation[0]) if s["strategy"]["metadata"]["asset"] == asset) for asset in ASSETS}
            for asset_alloc in asset_allocs.values():
                if asset_alloc > 0.4:
                    constraint_penalty += (asset_alloc - 0.4)**2
                if asset_alloc < 0.1 and asset_alloc > 0:
                    constraint_penalty += (0.1 - asset_alloc)**2
            reward -= 2000 * constraint_penalty
            
            total_pnl += portfolio_pnl
            
            critic_optimizer.zero_grad()
            value = critic(state)
            value_loss = (reward - value) ** 2
            value_loss.backward()
            critic_optimizer.step()
            
            actor_optimizer.zero_grad()
            new_dist = actor(state)
            log_prob = new_dist.log_prob(allocation).sum()
            advantage = reward - value.detach()
            actor_loss = -log_prob * advantage
            actor_loss.backward()
            actor_optimizer.step()
            scheduler.step()
            
            if episode % 10 == 0:
                logging.info(f"PPO Episode {episode}, Reward: {reward:.4f}, Portfolio PnL: ${portfolio_pnl:.2f}")
        
        state = torch.tensor(np.array(metrics).flatten(), dtype=torch.float32).unsqueeze(0)
        if state.shape[1] != state_dim:
            state = state[:, :state_dim]
            if state.shape[1] < state_dim:
                state = torch.cat((state, torch.zeros(1, state_dim - state.shape[1])), dim=1)
        
        with torch.no_grad():
            allocation = actor(state).mean
            allocation = torch.clamp(allocation, 0.0, 1.0)
            allocation_sum = torch.sum(allocation)
            if allocation_sum > 0:
                allocation = allocation / allocation_sum
        
        allocations = self.adjust_allocations(allocation[0].numpy(), approved)
        logging.info(f"Approved {len(approved)} strategies with PPO allocations: {allocations}")
        logging.info(f"Asset distribution in training strategies: {asset_counts}")
        logging.info(f"Total Training Period PnL: ${total_pnl:.2f}")
        return list(zip(approved, allocations))