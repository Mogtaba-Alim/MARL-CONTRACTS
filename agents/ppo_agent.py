"""
PPO agent implementation for the MARL environment.

This module implements a Proximal Policy Optimization (PPO) agent for the
multi-agent reinforcement learning environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOPolicy(nn.Module):
    """
    Neural network policy for PPO.
    
    This network takes observations as input and outputs action probabilities
    and value estimates.
    """
    
    def __init__(self, observation_space, action_space):
        """
        Initialize the policy network.
        
        Args:
            observation_space: Observation space of the environment.
            action_space: Action space of the environment.
        """
        super(PPOPolicy, self).__init__()
        
        # Determine input size based on observation space
        # For Dict observation space, we flatten and concatenate all components
        if isinstance(observation_space, dict):
            # For the Cleanup environment, we have position, grid, and waste_level
            # We'll use a CNN for the grid and concatenate with other features
            self.has_grid = "grid" in observation_space
            
            # Calculate total input size for non-grid features
            self.input_size = 0
            for key, space in observation_space.items():
                if key != "grid":
                    self.input_size += np.prod(space.shape)
            
            # CNN for grid observations
            if self.has_grid:
                grid_shape = observation_space["grid"].shape
                self.conv = nn.Sequential(
                    nn.Conv2d(grid_shape[2], 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Flatten()
                )
                
                # Calculate CNN output size
                with torch.no_grad():
                    dummy_input = torch.zeros(1, grid_shape[2], grid_shape[0], grid_shape[1])
                    conv_output_size = self.conv(dummy_input).shape[1]
                
                # Add CNN output size to total input size
                self.input_size += conv_output_size
        else:
            self.has_grid = False
            self.input_size = np.prod(observation_space.shape)
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)
        )
        
        # Value network
        self.value = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, obs):
        """
        Forward pass through the network.
        
        Args:
            obs: Observation from the environment.
            
        Returns:
            action_probs: Action probabilities.
            value: Value estimate.
        """
        # Process observation
        if self.has_grid:
            # Extract grid and other features
            grid = obs["grid"].permute(0, 3, 1, 2)  # (B, C, H, W)
            grid_features = self.conv(grid)
            
            # Extract and flatten other features
            other_features = []
            for key, value in obs.items():
                if key != "grid":
                    other_features.append(value.view(value.size(0), -1))
            
            # Concatenate all features
            if other_features:
                other_features = torch.cat(other_features, dim=1)
                features = torch.cat([grid_features, other_features], dim=1)
            else:
                features = grid_features
        else:
            # Flatten observation
            features = obs.view(obs.size(0), -1)
        
        # Forward pass through policy and value networks
        action_logits = self.policy(features)
        value = self.value(features)
        
        return action_logits, value
    
    def get_action(self, obs, deterministic=False):
        """
        Get an action from the policy.
        
        Args:
            obs: Observation from the environment.
            deterministic (bool): Whether to use deterministic action selection.
            
        Returns:
            action: Selected action.
            action_log_prob: Log probability of the selected action.
            value: Value estimate.
        """
        # Forward pass
        action_logits, value = self.forward(obs)
        
        # Create categorical distribution
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        # Sample action
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()
        
        # Get log probability
        action_log_prob = dist.log_prob(action)
        
        return action, action_log_prob, value


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    
    This agent uses the PPO algorithm to learn a policy for the environment.
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        learning_rate=3e-4,
        gamma=0.99,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cpu"
    ):
        """
        Initialize the PPO agent.
        
        Args:
            observation_space: Observation space of the environment.
            action_space: Action space of the environment.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor.
            clip_ratio (float): PPO clipping parameter.
            value_coef (float): Value loss coefficient.
            entropy_coef (float): Entropy coefficient.
            max_grad_norm (float): Maximum gradient norm.
            device (str): Device to use for computation.
        """
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Create policy
        self.policy = PPOPolicy(observation_space, action_space).to(device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
    
    def select_action(self, obs, deterministic=False):
        """
        Select an action from the policy.
        
        Args:
            obs: Observation from the environment.
            deterministic (bool): Whether to use deterministic action selection.
            
        Returns:
            action: Selected action.
            action_log_prob: Log probability of the selected action.
            value: Value estimate.
        """
        # Convert observation to tensor
        if isinstance(obs, dict):
            obs_tensor = {
                k: torch.FloatTensor(v).unsqueeze(0).to(self.device)
                for k, v in obs.items()
            }
        else:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Get action from policy
        with torch.no_grad():
            action, action_log_prob, value = self.policy.get_action(
                obs_tensor, deterministic=deterministic
            )
        
        return action.item(), action_log_prob.item(), value.item()
    
    def update(self, obs, actions, old_log_probs, returns, advantages, epochs=10, batch_size=64):
        """
        Update the policy using PPO.
        
        Args:
            obs: Observations from the environment.
            actions: Actions taken.
            old_log_probs: Log probabilities of actions under the old policy.
            returns: Discounted returns.
            advantages: Advantages.
            epochs (int): Number of epochs to update for.
            batch_size (int): Batch size for updates.
            
        Returns:
            metrics (dict): Dictionary of training metrics.
        """
        # Convert to tensors
        if isinstance(obs, dict):
            obs_tensor = {
                k: torch.FloatTensor(v).to(self.device)
                for k, v in obs.items()
            }
        else:
            obs_tensor = torch.FloatTensor(obs).to(self.device)
        
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Update policy for multiple epochs
        metrics = {
            "policy_loss": 0,
            "value_loss": 0,
            "entropy": 0,
            "total_loss": 0,
            "approx_kl": 0,
            "clip_fraction": 0
        }
        
        for _ in range(epochs):
            # Generate random indices
            indices = np.random.permutation(len(actions))
            
            # Update in batches
            for start_idx in range(0, len(actions), batch_size):
                # Get batch indices
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Get batch data
                if isinstance(obs_tensor, dict):
                    batch_obs = {
                        k: v[batch_indices]
                        for k, v in obs_tensor.items()
                    }
                else:
                    batch_obs = obs_tensor[batch_indices]
                
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                # Forward pass
                action_logits, values = self.policy(batch_obs)
                
                # Calculate action probabilities
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                
                # Get log probabilities and entropy
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calculate ratio and clipped ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                )
                
                # Calculate policy loss
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()
                
                # Calculate value loss
                value_loss = 0.5 * ((values.squeeze() - batch_returns) ** 2).mean()
                
                # Calculate total loss
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
                
                # Calculate approximate KL divergence
                approx_kl = 0.5 * ((new_log_probs - batch_old_log_probs) ** 2).mean().item()
                
                # Calculate clip fraction
                clip_fraction = (
                    (ratio < 1 - self.clip_ratio).float().mean()
                    + (ratio > 1 + self.clip_ratio).float().mean()
                ).item()
                
                # Update metrics
                metrics["policy_loss"] += policy_loss.item() / epochs
                metrics["value_loss"] += value_loss.item() / epochs
                metrics["entropy"] += entropy.item() / epochs
                metrics["total_loss"] += total_loss.item() / epochs
                metrics["approx_kl"] += approx_kl / epochs
                metrics["clip_fraction"] += clip_fraction / epochs
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        return metrics
    
    def save(self, path):
        """
        Save the agent's policy to a file.
        
        Args:
            path (str): Path to save the policy to.
        """
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        """
        Load the agent's policy from a file.
        
        Args:
            path (str): Path to load the policy from.
        """
        self.policy.load_state_dict(torch.load(path, map_location=self.device))


def create_ppo_agents(env, device="cpu"):
    """
    Create PPO agents for a multi-agent environment.
    
    Args:
        env: Multi-agent environment.
        device (str): Device to use for computation.
        
    Returns:
        agents (dict): Dictionary mapping agent IDs to PPO agents.
    """
    agents = {}
    
    for agent_id in env.possible_agents:
        agents[agent_id] = PPOAgent(
            observation_space=env.observation_space[agent_id],
            action_space=env.action_space[agent_id],
            device=device
        )
    
    return agents 