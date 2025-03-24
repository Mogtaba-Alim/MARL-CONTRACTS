"""
Training module for PPO agents in the MARL environment.

This module implements functions for training PPO agents in the multi-agent
reinforcement learning environment.
"""

import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm


def collect_rollouts(env, agents, num_steps):
    """
    Collect rollouts from the environment using the current policies.
    
    Args:
        env: Environment to collect rollouts from.
        agents (dict): Dictionary mapping agent IDs to PPO agents.
        num_steps (int): Number of steps to collect.
        
    Returns:
        rollouts (dict): Dictionary of rollout data for each agent.
    """
    # Initialize rollout buffers for each agent
    rollouts = {
        agent_id: {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": []
        }
        for agent_id in env.possible_agents
    }
    
    # Reset the environment
    observations, _ = env.reset()
    
    # Collect rollouts
    for _ in range(num_steps):
        # Select actions
        actions = {}
        log_probs = {}
        values = {}
        
        for agent_id, agent in agents.items():
            if agent_id in env.agents:
                action, log_prob, value = agent.select_action(observations[agent_id])
                actions[agent_id] = action
                log_probs[agent_id] = log_prob
                values[agent_id] = value
        
        # Step the environment
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Store rollout data
        for agent_id in env.agents:
            rollouts[agent_id]["observations"].append(observations[agent_id])
            rollouts[agent_id]["actions"].append(actions[agent_id])
            rollouts[agent_id]["rewards"].append(rewards[agent_id])
            rollouts[agent_id]["values"].append(values[agent_id])
            rollouts[agent_id]["log_probs"].append(log_probs[agent_id])
            rollouts[agent_id]["dones"].append(
                terminations[agent_id] or truncations[agent_id]
            )
        
        # Update observations
        observations = next_observations
        
        # Check if episode is done
        if all(terminations.values()) or all(truncations.values()):
            observations, _ = env.reset()
    
    return rollouts


def compute_returns_and_advantages(rollouts, agents):
    """
    Compute returns and advantages for the collected rollouts.
    
    Args:
        rollouts (dict): Dictionary of rollout data for each agent.
        agents (dict): Dictionary mapping agent IDs to PPO agents.
        
    Returns:
        processed_rollouts (dict): Dictionary of processed rollout data for each agent.
    """
    processed_rollouts = {
        agent_id: {
            "observations": [],
            "actions": [],
            "log_probs": [],
            "returns": [],
            "advantages": []
        }
        for agent_id in agents.keys()
    }
    
    # Process each agent's rollouts
    for agent_id, agent_rollouts in rollouts.items():
        # Get agent's discount factor
        gamma = agents[agent_id].gamma
        
        # Convert to numpy arrays
        observations = agent_rollouts["observations"]
        actions = np.array(agent_rollouts["actions"])
        rewards = np.array(agent_rollouts["rewards"])
        values = np.array(agent_rollouts["values"])
        log_probs = np.array(agent_rollouts["log_probs"])
        dones = np.array(agent_rollouts["dones"])
        
        # Compute returns and advantages
        returns = []
        advantages = []
        
        # Initialize with zeros
        next_return = 0
        next_value = 0
        next_advantage = 0
        
        # Iterate backwards through the rollout
        for t in reversed(range(len(rewards))):
            # Compute return
            if dones[t]:
                next_return = 0
                next_value = 0
                next_advantage = 0
            
            # Compute return (discounted sum of rewards)
            current_return = rewards[t] + gamma * next_return * (1 - dones[t])
            returns.insert(0, current_return)
            
            # Compute advantage (using TD error)
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            current_advantage = delta + gamma * 0.95 * next_advantage * (1 - dones[t])
            advantages.insert(0, current_advantage)
            
            # Update next values
            next_return = current_return
            next_value = values[t]
            next_advantage = current_advantage
        
        # Store processed rollout data
        processed_rollouts[agent_id]["observations"] = observations
        processed_rollouts[agent_id]["actions"] = actions
        processed_rollouts[agent_id]["log_probs"] = log_probs
        processed_rollouts[agent_id]["returns"] = np.array(returns)
        processed_rollouts[agent_id]["advantages"] = np.array(advantages)
    
    return processed_rollouts


def train_agents(env, agents, num_iterations=100, steps_per_iteration=1000, update_epochs=10, batch_size=64, verbose=True):
    """
    Train PPO agents in the environment.
    
    Args:
        env: Environment to train in.
        agents (dict): Dictionary mapping agent IDs to PPO agents.
        num_iterations (int): Number of training iterations.
        steps_per_iteration (int): Number of steps to collect per iteration.
        update_epochs (int): Number of epochs to update for.
        batch_size (int): Batch size for updates.
        verbose (bool): Whether to print progress information.
        
    Returns:
        metrics (dict): Dictionary of training metrics.
    """
    # Initialize metrics
    metrics = {
        "iterations": [],
        "rewards": defaultdict(list),
        "total_reward": [],
        "waste_level": [],
        "apple_count": [],
        "cleaning_counts": defaultdict(list),
        "policy_loss": defaultdict(list),
        "value_loss": defaultdict(list),
        "entropy": defaultdict(list)
    }
    
    # Training loop
    for iteration in tqdm(range(num_iterations), desc="Training", disable=not verbose):
        # Collect rollouts
        rollouts = collect_rollouts(env, agents, steps_per_iteration)
        
        # Compute returns and advantages
        processed_rollouts = compute_returns_and_advantages(rollouts, agents)
        
        # Update agents
        for agent_id, agent in agents.items():
            # Get agent's rollout data
            agent_rollouts = processed_rollouts[agent_id]
            
            # Update agent
            update_metrics = agent.update(
                agent_rollouts["observations"],
                agent_rollouts["actions"],
                agent_rollouts["log_probs"],
                agent_rollouts["returns"],
                agent_rollouts["advantages"],
                epochs=update_epochs,
                batch_size=batch_size
            )
            
            # Store metrics
            metrics["policy_loss"][agent_id].append(update_metrics["policy_loss"])
            metrics["value_loss"][agent_id].append(update_metrics["value_loss"])
            metrics["entropy"][agent_id].append(update_metrics["entropy"])
        
        # Evaluate agents
        eval_metrics = evaluate_agents(env, agents, num_episodes=5)
        
        # Store evaluation metrics
        metrics["iterations"].append(iteration)
        for agent_id, reward in eval_metrics["rewards"].items():
            metrics["rewards"][agent_id].append(reward)
        metrics["total_reward"].append(eval_metrics["total_reward"])
        metrics["waste_level"].append(eval_metrics["waste_level"])
        metrics["apple_count"].append(eval_metrics["apple_count"])
        for agent_id, count in eval_metrics["cleaning_counts"].items():
            metrics["cleaning_counts"][agent_id].append(count)
        
        # Print progress
        if verbose and (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"Total reward: {eval_metrics['total_reward']:.2f}")
            print(f"Waste level: {eval_metrics['waste_level']:.2f}")
            print(f"Apple count: {eval_metrics['apple_count']:.2f}")
            print(f"Rewards: {', '.join([f'{agent_id}: {reward:.2f}' for agent_id, reward in eval_metrics['rewards'].items()])}")
            print(f"Cleaning counts: {', '.join([f'{agent_id}: {count}' for agent_id, count in eval_metrics['cleaning_counts'].items()])}")
            print()
    
    return metrics


def evaluate_agents(env, agents, num_episodes=10):
    """
    Evaluate agents in the environment.
    
    Args:
        env: Environment to evaluate in.
        agents (dict): Dictionary mapping agent IDs to PPO agents.
        num_episodes (int): Number of episodes to evaluate for.
        
    Returns:
        metrics (dict): Dictionary of evaluation metrics.
    """
    # Initialize metrics
    total_rewards = defaultdict(float)
    episode_lengths = []
    waste_levels = []
    apple_counts = []
    cleaning_counts = defaultdict(int)
    
    # Evaluate for multiple episodes
    for _ in range(num_episodes):
        # Reset the environment
        observations, _ = env.reset()
        done = False
        episode_step = 0
        
        # Run episode
        while not done:
            # Select actions (deterministic)
            actions = {}
            for agent_id, agent in agents.items():
                if agent_id in env.agents:
                    action, _, _ = agent.select_action(observations[agent_id], deterministic=True)
                    actions[agent_id] = action
            
            # Step the environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Update metrics
            for agent_id, reward in rewards.items():
                total_rewards[agent_id] += reward
            
            # Count cleaning actions
            for agent_id, info in infos.items():
                if info.get("cleaned", False):
                    cleaning_counts[agent_id] += 1
            
            # Record waste level and apple count
            waste_levels.append(env.waste_level)
            apple_counts.append(len(env.apple_positions))
            
            # Check if episode is done
            done = all(terminations.values()) or all(truncations.values())
            episode_step += 1
        
        # Record episode length
        episode_lengths.append(episode_step)
    
    # Calculate average metrics
    avg_rewards = {agent_id: reward / num_episodes for agent_id, reward in total_rewards.items()}
    avg_total_reward = sum(avg_rewards.values())
    avg_waste_level = sum(waste_levels) / len(waste_levels)
    avg_apple_count = sum(apple_counts) / len(apple_counts)
    avg_episode_length = sum(episode_lengths) / len(episode_lengths)
    avg_cleaning_counts = {agent_id: count / num_episodes for agent_id, count in cleaning_counts.items()}
    
    # Compile metrics
    metrics = {
        "rewards": avg_rewards,
        "total_reward": avg_total_reward,
        "waste_level": avg_waste_level,
        "apple_count": avg_apple_count,
        "episode_length": avg_episode_length,
        "cleaning_counts": avg_cleaning_counts
    }
    
    return metrics


def train_with_contract(env_factory, contract_factory, num_iterations=100, steps_per_iteration=1000, device="cpu", verbose=True):
    """
    Train agents with a contract in the environment.
    
    Args:
        env_factory: Function that creates a new environment instance.
        contract_factory: Function that creates a contract.
        num_iterations (int): Number of training iterations.
        steps_per_iteration (int): Number of steps to collect per iteration.
        device (str): Device to use for computation.
        verbose (bool): Whether to print progress information.
        
    Returns:
        agents (dict): Trained agents.
        metrics (dict): Training metrics.
    """
    # Create environment
    env = env_factory()
    
    # Create contract
    contract = contract_factory()
    
    # Add contract to environment
    env.add_contract(contract)
    
    # Create agents
    from agents.ppo_agent import create_ppo_agents
    agents = create_ppo_agents(env, device=device)
    
    # Train agents
    metrics = train_agents(
        env,
        agents,
        num_iterations=num_iterations,
        steps_per_iteration=steps_per_iteration,
        verbose=verbose
    )
    
    return agents, metrics 