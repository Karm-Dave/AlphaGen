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
        super().__init__()
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
        super().__init__()
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
        allocations = np.array(
            [float(np.array(a).flatten()[0]) if isinstance(a, (list, np.ndarray)) else float(a)
             for a in allocations], dtype=np.float32
        )

        asset_allocs = {asset: 0.0 for asset in ASSETS}

        for alloc, strat in zip(allocations, strategies):
            asset = strat["strategy"]["metadata"]["asset"]
            asset_allocs[asset] += alloc

        for asset, total in asset_allocs.items():
            if total > max_asset_alloc:
                scale = max_asset_alloc / total
                for i, strat in enumerate(strategies):
                    if strat["strategy"]["metadata"]["asset"] == asset:
                        allocations[i] *= scale
            elif 0 < total < min_asset_alloc:
                scale = min_asset_alloc / total
                for i, strat in enumerate(strategies):
                    if strat["strategy"]["metadata"]["asset"] == asset:
                        allocations[i] *= scale

        for i in range(len(allocations)):
            if allocations[i] < min_alloc:
                allocations[i] = min_alloc

        total = allocations.sum()
        if total > 0:
            allocations /= total

        return allocations

    def allocate(self, strategies, data_dict, episodes=200):
        if not strategies:
            logging.warning("No strategies provided for allocation")
            return []

        metrics = []
        approved = []

        for strat in strategies:
            asset = strat["strategy"]["metadata"]["asset"]
            timeframe = strat["strategy"]["metadata"]["timeframe"]
            key = (asset, timeframe)

            if key not in data_dict:
                data_dict[key] = self.fetcher.fetch(asset, '2020-12-31', timeframe)

            data = data_dict[key]
            pnl, sharpe, dd = self.backtester.run(strat, data)
            vol = self.fetcher.compute_volatility(data)

            normalized_pnl = pnl / (vol + 1e-6)
            if normalized_pnl > 1500 and sharpe > 0.6:
                approved.append(strat)
                metrics.append([normalized_pnl, sharpe, dd])

        if not approved:
            logging.info("No strategies approved by PPO.")
            return []

        state_dim = len(approved) * 3
        action_dim = len(approved)

        actor = PPOActor(state_dim, action_dim)
        critic = PPOCritic(state_dim)
        actor_opt = optim.Adam(actor.parameters(), lr=3e-4, weight_decay=1e-5)
        critic_opt = optim.Adam(critic.parameters(), lr=3e-4, weight_decay=1e-5)
        scheduler = StepLR(actor_opt, step_size=50, gamma=0.9)

        for episode in range(episodes):
            state = torch.tensor(np.array(metrics).flatten(), dtype=torch.float32).unsqueeze(0)
            if state.shape[1] < state_dim:
                pad = torch.zeros(1, state_dim - state.shape[1])
                state = torch.cat((state, pad), dim=1)
            elif state.shape[1] > state_dim:
                state = state[:, :state_dim]

            dist = actor(state)
            allocation = dist.sample()
            allocation = torch.clamp(allocation, 0.0, 1.0)
            allocation_sum = allocation.sum()
            if allocation_sum > 0:
                allocation /= allocation_sum

            port_pnl = 0
            port_returns = []

            for i, (strat, alloc) in enumerate(zip(approved, allocation[0])):
                asset = strat["strategy"]["metadata"]["asset"]
                timeframe = strat["strategy"]["metadata"]["timeframe"]
                key = (asset, timeframe)
                data = data_dict[key]
                pnl, _, _ = self.backtester.run(strat, data)
                port_pnl += pnl * alloc.item()
                port_returns.append(pnl / 100000 * alloc.item())

            asset_set = set(s["strategy"]["metadata"]["asset"] for s in approved)
            diversity_penalty = max(0, (len(ASSETS) - len(asset_set)) / len(ASSETS)) * 2.0
            reward = port_pnl / 100000 - 0.2 * np.std(port_returns) - diversity_penalty


            constraint_penalty = 0
            for alloc in allocation[0]:
                if alloc < 0.15:
                    constraint_penalty += (0.15 - alloc.item()) ** 2

            asset_allocs = {
                asset: sum(float(alloc.item()) for s, alloc in zip(approved, allocation[0])
                           if s["strategy"]["metadata"]["asset"] == asset)
                for asset in ASSETS
            }

            for val in asset_allocs.values():
                if val > 0.4:
                    constraint_penalty += (val - 0.4) ** 2
                if 0 < val < 0.1:
                    constraint_penalty += (0.1 - val) ** 2

            reward -= 2000 * constraint_penalty

            critic_opt.zero_grad()
            value = critic(state)
            v_loss = (reward - value) ** 2
            v_loss.backward()
            critic_opt.step()

            actor_opt.zero_grad()
            new_dist = actor(state)
            log_prob = new_dist.log_prob(allocation).sum()
            advantage = reward - value.detach()
            actor_loss = -log_prob * advantage
            actor_loss.backward()
            actor_opt.step()
            scheduler.step()

            if episode % 10 == 0:
                logging.info(f"[PPO] Episode {episode}, Reward: {reward:.4f}, PnL: ${port_pnl:.2f}")

        final_state = torch.tensor(np.array(metrics).flatten(), dtype=torch.float32).unsqueeze(0)
        if final_state.shape[1] < state_dim:
            final_state = torch.cat((final_state, torch.zeros(1, state_dim - final_state.shape[1])), dim=1)

        with torch.no_grad():
            final_alloc = actor(final_state).mean
            final_alloc = torch.clamp(final_alloc, 0.0, 1.0)
            if final_alloc.sum() > 0:
                final_alloc /= final_alloc.sum()

        final_alloc = self.adjust_allocations(final_alloc.numpy(), approved)
        return list(zip(approved, final_alloc))
