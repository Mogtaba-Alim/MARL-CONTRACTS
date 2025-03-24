"""
Analysis module for the MARL environment.

This module provides functions for analyzing the results of experiments.
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def calculate_summary_statistics(metrics):
    """
    Calculate summary statistics for the metrics.
    
    Args:
        metrics (dict): Dictionary of metrics.
        
    Returns:
        summary (dict): Dictionary of summary statistics.
    """
    summary = {}
    
    # Calculate average rewards
    summary["avg_total_reward"] = np.mean(metrics["total_reward"])
    summary["std_total_reward"] = np.std(metrics["total_reward"])
    
    # Calculate average waste level
    summary["avg_waste_level"] = np.mean(metrics["waste_level"])
    summary["std_waste_level"] = np.std(metrics["waste_level"])
    
    # Calculate average apple count
    summary["avg_apple_count"] = np.mean(metrics["apple_count"])
    summary["std_apple_count"] = np.std(metrics["apple_count"])
    
    # Calculate average rewards per agent
    summary["avg_rewards_per_agent"] = {
        agent_id: np.mean(rewards)
        for agent_id, rewards in metrics["rewards"].items()
    }
    
    # Calculate average cleaning counts per agent
    summary["avg_cleaning_counts_per_agent"] = {
        agent_id: np.mean(counts)
        for agent_id, counts in metrics["cleaning_counts"].items()
    }
    
    return summary


def compare_scenarios(no_contract_metrics, contract_metrics):
    """
    Compare metrics between no contract and contract scenarios.
    
    Args:
        no_contract_metrics (dict): Dictionary of metrics without contract.
        contract_metrics (dict): Dictionary of metrics with contract.
        
    Returns:
        comparison (dict): Dictionary of comparison metrics.
    """
    comparison = {}
    
    # Calculate summary statistics for each scenario
    no_contract_summary = calculate_summary_statistics(no_contract_metrics)
    contract_summary = calculate_summary_statistics(contract_metrics)
    
    # Calculate percentage improvements
    comparison["total_reward_improvement"] = (
        (contract_summary["avg_total_reward"] - no_contract_summary["avg_total_reward"])
        / no_contract_summary["avg_total_reward"] * 100
    )
    
    comparison["waste_level_improvement"] = (
        (no_contract_summary["avg_waste_level"] - contract_summary["avg_waste_level"])
        / no_contract_summary["avg_waste_level"] * 100
    )
    
    comparison["apple_count_improvement"] = (
        (contract_summary["avg_apple_count"] - no_contract_summary["avg_apple_count"])
        / no_contract_summary["avg_apple_count"] * 100
    )
    
    # Calculate reward improvements per agent
    comparison["reward_improvements_per_agent"] = {}
    for agent_id in no_contract_summary["avg_rewards_per_agent"].keys():
        no_contract_reward = no_contract_summary["avg_rewards_per_agent"][agent_id]
        contract_reward = contract_summary["avg_rewards_per_agent"][agent_id]
        
        improvement = (
            (contract_reward - no_contract_reward)
            / no_contract_reward * 100 if no_contract_reward != 0 else float('inf')
        )
        
        comparison["reward_improvements_per_agent"][agent_id] = improvement
    
    # Calculate cleaning count improvements per agent
    comparison["cleaning_count_improvements_per_agent"] = {}
    for agent_id in no_contract_summary["avg_cleaning_counts_per_agent"].keys():
        no_contract_count = no_contract_summary["avg_cleaning_counts_per_agent"][agent_id]
        contract_count = contract_summary["avg_cleaning_counts_per_agent"][agent_id]
        
        improvement = (
            (contract_count - no_contract_count)
            / no_contract_count * 100 if no_contract_count != 0 else float('inf')
        )
        
        comparison["cleaning_count_improvements_per_agent"][agent_id] = improvement
    
    return comparison


def analyze_contract_stability(env_factory, contracts, num_episodes=10):
    """
    Analyze the stability of contracts.
    
    Args:
        env_factory: Function that creates a new environment instance.
        contracts (list): List of Contract objects to analyze.
        num_episodes (int): Number of episodes to run for each contract.
        
    Returns:
        stability_metrics (dict): Dictionary of stability metrics.
    """
    stability_metrics = {}
    
    # Create environment
    env = env_factory()
    
    # Run episodes with random actions
    for contract in contracts:
        # Add contract to environment
        env.add_contract(contract)
        
        # Initialize metrics
        total_rewards = defaultdict(float)
        contract_transfers = defaultdict(float)
        
        # Run episodes
        for _ in range(num_episodes):
            observations, _ = env.reset()
            done = False
            
            while not done:
                # Take random actions
                actions = {agent: env.action_space[agent].sample() for agent in env.agents}
                
                # Step the environment
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # Update metrics
                for agent, reward in rewards.items():
                    total_rewards[agent] += reward
                
                # Update contract transfers
                for agent, info in infos.items():
                    if "contract_transfers" in info:
                        contract_transfers[agent] += info["contract_transfers"]
                
                # Check if episode is done
                done = all(terminations.values()) or all(truncations.values())
        
        # Calculate average metrics
        avg_rewards = {agent: reward / num_episodes for agent, reward in total_rewards.items()}
        avg_transfers = {agent: transfer / num_episodes for agent, transfer in contract_transfers.items()}
        
        # Store metrics
        stability_metrics[contract.name] = {
            "rewards": avg_rewards,
            "transfers": avg_transfers,
            "total_reward": sum(avg_rewards.values())
        }
        
        # Remove contract from environment
        env.remove_contract(contract.name)
    
    return stability_metrics


def create_contract_comparison_table(stability_metrics, baseline_rewards=None):
    """
    Create a table comparing different contracts.
    
    Args:
        stability_metrics (dict): Dictionary of stability metrics.
        baseline_rewards (dict): Dictionary of baseline rewards (no contract).
        
    Returns:
        df: Pandas DataFrame with contract comparison.
    """
    # Initialize data
    data = []
    
    # Add baseline if provided
    if baseline_rewards:
        row = {
            "Contract": "No Contract",
            "Total Reward": sum(baseline_rewards.values())
        }
        
        for agent, reward in baseline_rewards.items():
            row[f"{agent} Reward"] = reward
            row[f"{agent} Transfer"] = 0
        
        data.append(row)
    
    # Add contract data
    for contract_name, metrics in stability_metrics.items():
        row = {
            "Contract": contract_name,
            "Total Reward": metrics["total_reward"]
        }
        
        for agent, reward in metrics["rewards"].items():
            row[f"{agent} Reward"] = reward
            row[f"{agent} Transfer"] = metrics["transfers"].get(agent, 0)
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate improvements if baseline is provided
    if baseline_rewards:
        baseline_total = sum(baseline_rewards.values())
        df["Improvement (%)"] = (df["Total Reward"] - baseline_total) / baseline_total * 100
        
        for agent in baseline_rewards.keys():
            baseline_reward = baseline_rewards[agent]
            if baseline_reward != 0:
                df[f"{agent} Improvement (%)"] = (df[f"{agent} Reward"] - baseline_reward) / baseline_reward * 100
    
    return df


def analyze_contract_parameters(env_factory, contract_factory, parameter_ranges, num_episodes=10):
    """
    Analyze the effect of contract parameters on performance.
    
    Args:
        env_factory: Function that creates a new environment instance.
        contract_factory: Function that creates a contract with parameters.
        parameter_ranges (dict): Dictionary mapping parameter names to lists of values.
        num_episodes (int): Number of episodes to run for each parameter combination.
        
    Returns:
        parameter_metrics (dict): Dictionary of metrics for each parameter combination.
    """
    from itertools import product
    
    # Generate all parameter combinations
    param_names = list(parameter_ranges.keys())
    param_values = list(parameter_ranges.values())
    param_combinations = list(product(*param_values))
    
    # Initialize metrics
    parameter_metrics = {}
    
    # Create environment
    env = env_factory()
    
    # Evaluate each parameter combination
    for params in param_combinations:
        # Create parameter dictionary
        param_dict = {name: value for name, value in zip(param_names, params)}
        
        # Create contract with these parameters
        contract = contract_factory(**param_dict)
        
        # Add contract to environment
        env.add_contract(contract)
        
        # Initialize metrics
        total_rewards = defaultdict(float)
        waste_levels = []
        apple_counts = []
        cleaning_counts = defaultdict(int)
        
        # Run episodes
        for _ in range(num_episodes):
            observations, _ = env.reset()
            done = False
            
            while not done:
                # Take random actions
                actions = {agent: env.action_space[agent].sample() for agent in env.agents}
                
                # Step the environment
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # Update metrics
                for agent, reward in rewards.items():
                    total_rewards[agent] += reward
                
                # Count cleaning actions
                for agent, info in infos.items():
                    if info.get("cleaned", False):
                        cleaning_counts[agent] += 1
                
                # Record waste level and apple count
                waste_levels.append(env.waste_level)
                apple_counts.append(len(env.apple_positions))
                
                # Check if episode is done
                done = all(terminations.values()) or all(truncations.values())
        
        # Calculate average metrics
        avg_rewards = {agent: reward / num_episodes for agent, reward in total_rewards.items()}
        avg_total_reward = sum(avg_rewards.values())
        avg_waste_level = sum(waste_levels) / len(waste_levels)
        avg_apple_count = sum(apple_counts) / len(apple_counts)
        avg_cleaning_counts = {agent: count / num_episodes for agent, count in cleaning_counts.items()}
        
        # Store metrics
        param_key = tuple(param_dict.items())
        parameter_metrics[param_key] = {
            "params": param_dict,
            "rewards": avg_rewards,
            "total_reward": avg_total_reward,
            "waste_level": avg_waste_level,
            "apple_count": avg_apple_count,
            "cleaning_counts": avg_cleaning_counts
        }
        
        # Remove contract from environment
        env.remove_contract(contract.name)
    
    return parameter_metrics


def create_parameter_analysis_table(parameter_metrics):
    """
    Create a table analyzing the effect of contract parameters.
    
    Args:
        parameter_metrics (dict): Dictionary of metrics for each parameter combination.
        
    Returns:
        df: Pandas DataFrame with parameter analysis.
    """
    # Initialize data
    data = []
    
    # Add data for each parameter combination
    for param_key, metrics in parameter_metrics.items():
        row = {}
        
        # Add parameters
        for param_name, param_value in metrics["params"].items():
            row[param_name] = param_value
        
        # Add metrics
        row["Total Reward"] = metrics["total_reward"]
        row["Waste Level"] = metrics["waste_level"]
        row["Apple Count"] = metrics["apple_count"]
        
        for agent, reward in metrics["rewards"].items():
            row[f"{agent} Reward"] = reward
        
        for agent, count in metrics["cleaning_counts"].items():
            row[f"{agent} Cleaning Count"] = count
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df 