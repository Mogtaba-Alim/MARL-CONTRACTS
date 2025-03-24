"""
Visualization module for the MARL environment.

This module provides functions for visualizing the environment and training results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_grid(grid, agent_positions=None, title=None, figsize=(8, 8)):
    """
    Plot the environment grid.
    
    Args:
        grid (np.ndarray): Grid to plot (H, W, 3).
        agent_positions (dict): Dictionary mapping agent IDs to positions.
        title (str): Title for the plot.
        figsize (tuple): Figure size.
        
    Returns:
        fig: Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the grid
    ax.imshow(grid)
    
    # Add agent positions
    if agent_positions:
        for agent_id, pos in agent_positions.items():
            row, col = pos
            ax.plot(col, row, 'ro', markersize=10, label=agent_id)
    
    # Add title
    if title:
        ax.set_title(title)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig


def plot_training_metrics(metrics, figsize=(12, 8)):
    """
    Plot training metrics.
    
    Args:
        metrics (dict): Dictionary of training metrics.
        figsize (tuple): Figure size.
        
    Returns:
        fig: Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot total reward
    axes[0, 0].plot(metrics["iterations"], metrics["total_reward"])
    axes[0, 0].set_title("Total Reward")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True)
    
    # Plot waste level
    axes[0, 1].plot(metrics["iterations"], metrics["waste_level"])
    axes[0, 1].set_title("Waste Level")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Waste Level")
    axes[0, 1].grid(True)
    
    # Plot apple count
    axes[1, 0].plot(metrics["iterations"], metrics["apple_count"])
    axes[1, 0].set_title("Apple Count")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Apple Count")
    axes[1, 0].grid(True)
    
    # Plot rewards per agent
    for agent_id, rewards in metrics["rewards"].items():
        axes[1, 1].plot(metrics["iterations"], rewards, label=agent_id)
    axes[1, 1].set_title("Rewards per Agent")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    return fig


def plot_cleaning_counts(metrics, figsize=(10, 6)):
    """
    Plot cleaning counts per agent.
    
    Args:
        metrics (dict): Dictionary of training metrics.
        figsize (tuple): Figure size.
        
    Returns:
        fig: Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot cleaning counts per agent
    for agent_id, counts in metrics["cleaning_counts"].items():
        ax.plot(metrics["iterations"], counts, label=agent_id)
    
    ax.set_title("Cleaning Counts per Agent")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cleaning Count")
    ax.legend()
    ax.grid(True)
    
    return fig


def plot_contract_comparison(no_contract_metrics, contract_metrics, figsize=(12, 8)):
    """
    Plot comparison between no contract and contract scenarios.
    
    Args:
        no_contract_metrics (dict): Dictionary of metrics without contract.
        contract_metrics (dict): Dictionary of metrics with contract.
        figsize (tuple): Figure size.
        
    Returns:
        fig: Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot total reward
    axes[0, 0].plot(no_contract_metrics["iterations"], no_contract_metrics["total_reward"], label="No Contract")
    axes[0, 0].plot(contract_metrics["iterations"], contract_metrics["total_reward"], label="With Contract")
    axes[0, 0].set_title("Total Reward")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot waste level
    axes[0, 1].plot(no_contract_metrics["iterations"], no_contract_metrics["waste_level"], label="No Contract")
    axes[0, 1].plot(contract_metrics["iterations"], contract_metrics["waste_level"], label="With Contract")
    axes[0, 1].set_title("Waste Level")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Waste Level")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot apple count
    axes[1, 0].plot(no_contract_metrics["iterations"], no_contract_metrics["apple_count"], label="No Contract")
    axes[1, 0].plot(contract_metrics["iterations"], contract_metrics["apple_count"], label="With Contract")
    axes[1, 0].set_title("Apple Count")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Apple Count")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot cleaning counts (sum across agents)
    no_contract_cleaning = np.zeros_like(no_contract_metrics["iterations"], dtype=float)
    contract_cleaning = np.zeros_like(contract_metrics["iterations"], dtype=float)
    
    for agent_id, counts in no_contract_metrics["cleaning_counts"].items():
        no_contract_cleaning += np.array(counts)
    
    for agent_id, counts in contract_metrics["cleaning_counts"].items():
        contract_cleaning += np.array(counts)
    
    axes[1, 1].plot(no_contract_metrics["iterations"], no_contract_cleaning, label="No Contract")
    axes[1, 1].plot(contract_metrics["iterations"], contract_cleaning, label="With Contract")
    axes[1, 1].set_title("Total Cleaning Actions")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Cleaning Count")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    return fig


def plot_reward_distribution(metrics, figsize=(10, 6)):
    """
    Plot the distribution of rewards among agents.
    
    Args:
        metrics (dict): Dictionary of training metrics.
        figsize (tuple): Figure size.
        
    Returns:
        fig: Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get final rewards for each agent
    final_rewards = {agent_id: rewards[-1] for agent_id, rewards in metrics["rewards"].items()}
    
    # Plot bar chart
    agent_ids = list(final_rewards.keys())
    rewards = list(final_rewards.values())
    
    ax.bar(agent_ids, rewards)
    ax.set_title("Final Reward Distribution")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Reward")
    ax.grid(True, axis='y')
    
    return fig


def create_episode_animation(env, agents, max_steps=100, interval=200):
    """
    Create an animation of an episode.
    
    Args:
        env: Environment to animate.
        agents (dict): Dictionary mapping agent IDs to agents.
        max_steps (int): Maximum number of steps to animate.
        interval (int): Interval between frames in milliseconds.
        
    Returns:
        anim: Matplotlib animation.
    """
    import matplotlib.animation as animation
    
    # Reset the environment
    observations, _ = env.reset()
    
    # Initialize figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Initialize plot
    im = ax.imshow(env.grid)
    title = ax.set_title("Step 0")
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Animation function
    def animate(i):
        if i > 0:
            # Select actions
            actions = {}
            for agent_id, agent in agents.items():
                if agent_id in env.agents:
                    action, _, _ = agent.select_action(observations[agent_id], deterministic=True)
                    actions[agent_id] = action
            
            # Step the environment
            nonlocal observations
            observations, _, _, _, _ = env.step(actions)
        
        # Update plot
        im.set_array(env.grid)
        title.set_text(f"Step {i}")
        
        return [im, title]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=max_steps, interval=interval, blit=True
    )
    
    return anim 