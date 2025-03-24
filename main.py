"""
Main script for running experiments with the MARL-CONTRACTS project.

This script provides functions for running experiments with the Cleanup environment
and cooperative contracts.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from environment.cleanup import make_cleanup_env
from environment.contract_wrapper import make_contract_env, CleaningContract, ThresholdContract
from contracts.search import find_optimal_contracts
from agents.ppo_agent import create_ppo_agents
from agents.training import train_agents, train_with_contract, evaluate_agents
from utils.visualization import (
    plot_training_metrics, plot_cleaning_counts, plot_contract_comparison,
    plot_reward_distribution, create_episode_animation
)
from utils.analysis import (
    calculate_summary_statistics, compare_scenarios, analyze_contract_stability,
    create_contract_comparison_table, analyze_contract_parameters,
    create_parameter_analysis_table
)


def create_env_factory(num_agents=2, grid_size=(20, 20), episode_length=500, **kwargs):
    """
    Create a factory function for creating Cleanup environments.
    
    Args:
        num_agents (int): Number of agents in the environment.
        grid_size (tuple): Size of the grid (height, width).
        episode_length (int): Maximum number of steps per episode.
        **kwargs: Additional parameters for the environment.
        
    Returns:
        env_factory: Function that creates a new environment instance.
    """
    def env_factory():
        return make_cleanup_env(
            num_agents=num_agents,
            grid_size=grid_size,
            episode_length=episode_length,
            **kwargs
        )
    
    return env_factory


def create_contract_factory(contract_type="cleaning", **kwargs):
    """
    Create a factory function for creating contracts.
    
    Args:
        contract_type (str): Type of contract to create.
        **kwargs: Additional parameters for the contract.
        
    Returns:
        contract_factory: Function that creates a new contract.
    """
    def contract_factory():
        if contract_type == "cleaning":
            return CleaningContract(
                name="cleaning_contract",
                payer=kwargs.get("payer", "agent_0"),
                cleaner=kwargs.get("cleaner", "agent_1"),
                payment_per_cleaning=kwargs.get("payment", 1.0)
            )
        elif contract_type == "threshold":
            return ThresholdContract(
                name="threshold_contract",
                payer=kwargs.get("payer", "agent_0"),
                cleaner=kwargs.get("cleaner", "agent_1"),
                payment=kwargs.get("payment", 0.2),
                threshold=kwargs.get("threshold", 0.4)
            )
        else:
            raise ValueError(f"Unknown contract type: {contract_type}")
    
    return contract_factory


def run_baseline_experiment(env_factory, num_iterations=100, steps_per_iteration=1000, device="cpu"):
    """
    Run a baseline experiment without contracts.
    
    Args:
        env_factory: Function that creates a new environment instance.
        num_iterations (int): Number of training iterations.
        steps_per_iteration (int): Number of steps to collect per iteration.
        device (str): Device to use for computation.
        
    Returns:
        agents (dict): Trained agents.
        metrics (dict): Training metrics.
    """
    print("Running baseline experiment (no contracts)...")
    
    # Create environment
    env = env_factory()
    
    # Create agents
    agents = create_ppo_agents(env, device=device)
    
    # Train agents
    metrics = train_agents(
        env,
        agents,
        num_iterations=num_iterations,
        steps_per_iteration=steps_per_iteration,
        verbose=True
    )
    
    return agents, metrics


def run_contract_experiment(env_factory, contract_factory, num_iterations=100, steps_per_iteration=1000, device="cpu"):
    """
    Run an experiment with a contract.
    
    Args:
        env_factory: Function that creates a new environment instance.
        contract_factory: Function that creates a contract.
        num_iterations (int): Number of training iterations.
        steps_per_iteration (int): Number of steps to collect per iteration.
        device (str): Device to use for computation.
        
    Returns:
        agents (dict): Trained agents.
        metrics (dict): Training metrics.
    """
    print("Running contract experiment...")
    
    # Create environment
    env = env_factory()
    
    # Create contract
    contract = contract_factory()
    print(f"Using contract: {contract.name}")
    
    # Add contract to environment
    env.add_contract(contract)
    
    # Create agents
    agents = create_ppo_agents(env, device=device)
    
    # Train agents
    metrics = train_agents(
        env,
        agents,
        num_iterations=num_iterations,
        steps_per_iteration=steps_per_iteration,
        verbose=True
    )
    
    return agents, metrics


def run_contract_search_experiment(env_factory, num_episodes=10, max_depth=2):
    """
    Run an experiment to search for optimal contracts.
    
    Args:
        env_factory: Function that creates a new environment instance.
        num_episodes (int): Number of episodes to run for evaluation.
        max_depth (int): Maximum depth of the search tree.
        
    Returns:
        best_contracts (list): List of the best contracts found.
        best_value (float): Value of the best contracts.
        metrics (dict): Dictionary of evaluation metrics.
    """
    print("Running contract search experiment...")
    
    # Find optimal contracts
    best_contracts, best_value, metrics = find_optimal_contracts(
        env_factory,
        num_episodes=num_episodes,
        max_depth=max_depth,
        verbose=True
    )
    
    # Print results
    print(f"Best contracts found with value {best_value}:")
    for contract in best_contracts:
        print(f"  {contract.name}")
    
    return best_contracts, best_value, metrics


def run_parameter_analysis_experiment(env_factory, num_episodes=10):
    """
    Run an experiment to analyze the effect of contract parameters.
    
    Args:
        env_factory: Function that creates a new environment instance.
        num_episodes (int): Number of episodes to run for each parameter combination.
        
    Returns:
        parameter_metrics (dict): Dictionary of metrics for each parameter combination.
        df (pd.DataFrame): DataFrame with parameter analysis.
    """
    print("Running parameter analysis experiment...")
    
    # Define parameter ranges
    parameter_ranges = {
        "payment": [0.5, 1.0, 1.5, 2.0]
    }
    
    # Create contract factory
    def contract_factory(payment):
        return CleaningContract(
            name=f"cleaning_contract_{payment}",
            payer="agent_0",
            cleaner="agent_1",
            payment_per_cleaning=payment
        )
    
    # Analyze contract parameters
    parameter_metrics = analyze_contract_parameters(
        env_factory,
        contract_factory,
        parameter_ranges,
        num_episodes=num_episodes
    )
    
    # Create parameter analysis table
    df = create_parameter_analysis_table(parameter_metrics)
    
    # Print results
    print("Parameter analysis results:")
    print(df)
    
    return parameter_metrics, df


def save_results(results, experiment_name, output_dir="results"):
    """
    Save experiment results to disk.
    
    Args:
        results: Results to save.
        experiment_name (str): Name of the experiment.
        output_dir (str): Directory to save results to.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save results
    if isinstance(results, tuple) and len(results) == 2 and isinstance(results[1], dict):
        # Save agents and metrics
        agents, metrics = results
        
        # Save metrics
        np.save(os.path.join(experiment_dir, "metrics.npy"), metrics)
        
        # Create and save plots
        fig = plot_training_metrics(metrics)
        fig.savefig(os.path.join(experiment_dir, "training_metrics.png"))
        plt.close(fig)
        
        fig = plot_cleaning_counts(metrics)
        fig.savefig(os.path.join(experiment_dir, "cleaning_counts.png"))
        plt.close(fig)
        
        fig = plot_reward_distribution(metrics)
        fig.savefig(os.path.join(experiment_dir, "reward_distribution.png"))
        plt.close(fig)
        
        # Save summary statistics
        summary = calculate_summary_statistics(metrics)
        with open(os.path.join(experiment_dir, "summary.txt"), "w") as f:
            f.write("Summary Statistics:\n")
            f.write(f"Average Total Reward: {summary['avg_total_reward']:.2f} ± {summary['std_total_reward']:.2f}\n")
            f.write(f"Average Waste Level: {summary['avg_waste_level']:.2f} ± {summary['std_waste_level']:.2f}\n")
            f.write(f"Average Apple Count: {summary['avg_apple_count']:.2f} ± {summary['std_apple_count']:.2f}\n")
            f.write("\nAverage Rewards per Agent:\n")
            for agent_id, reward in summary["avg_rewards_per_agent"].items():
                f.write(f"  {agent_id}: {reward:.2f}\n")
            f.write("\nAverage Cleaning Counts per Agent:\n")
            for agent_id, count in summary["avg_cleaning_counts_per_agent"].items():
                f.write(f"  {agent_id}: {count:.2f}\n")
    
    elif isinstance(results, tuple) and len(results) == 3 and isinstance(results[0], list):
        # Save contract search results
        best_contracts, best_value, metrics = results
        
        # Save metrics
        np.save(os.path.join(experiment_dir, "metrics.npy"), metrics)
        
        # Save best contracts
        with open(os.path.join(experiment_dir, "best_contracts.txt"), "w") as f:
            f.write(f"Best contracts found with value {best_value}:\n")
            for contract in best_contracts:
                f.write(f"  {contract.name}\n")
        
        # Save summary statistics
        summary = calculate_summary_statistics(metrics)
        with open(os.path.join(experiment_dir, "summary.txt"), "w") as f:
            f.write("Summary Statistics:\n")
            f.write(f"Average Total Reward: {summary['avg_total_reward']:.2f} ± {summary['std_total_reward']:.2f}\n")
            f.write(f"Average Waste Level: {summary['avg_waste_level']:.2f} ± {summary['std_waste_level']:.2f}\n")
            f.write(f"Average Apple Count: {summary['avg_apple_count']:.2f} ± {summary['std_apple_count']:.2f}\n")
            f.write("\nAverage Rewards per Agent:\n")
            for agent_id, reward in summary["avg_rewards_per_agent"].items():
                f.write(f"  {agent_id}: {reward:.2f}\n")
            f.write("\nAverage Cleaning Counts per Agent:\n")
            for agent_id, count in summary["avg_cleaning_counts_per_agent"].items():
                f.write(f"  {agent_id}: {count:.2f}\n")
    
    elif isinstance(results, tuple) and len(results) == 2 and not isinstance(results[1], dict):
        # Save parameter analysis results
        parameter_metrics, df = results
        
        # Save DataFrame
        df.to_csv(os.path.join(experiment_dir, "parameter_analysis.csv"), index=False)
        
        # Create and save plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for param_key, metrics in parameter_metrics.items():
            payment = metrics["params"]["payment"]
            ax.bar(str(payment), metrics["total_reward"])
        ax.set_title("Total Reward by Payment Amount")
        ax.set_xlabel("Payment Amount")
        ax.set_ylabel("Total Reward")
        ax.grid(True, axis='y')
        fig.savefig(os.path.join(experiment_dir, "parameter_analysis.png"))
        plt.close(fig)
    
    print(f"Results saved to {experiment_dir}")


def compare_experiments(baseline_results, contract_results, output_dir="results"):
    """
    Compare baseline and contract experiments.
    
    Args:
        baseline_results: Results from baseline experiment.
        contract_results: Results from contract experiment.
        output_dir (str): Directory to save results to.
    """
    # Extract metrics
    baseline_agents, baseline_metrics = baseline_results
    contract_agents, contract_metrics = contract_results
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = os.path.join(output_dir, f"comparison_{timestamp}")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Create and save comparison plots
    fig = plot_contract_comparison(baseline_metrics, contract_metrics)
    fig.savefig(os.path.join(comparison_dir, "contract_comparison.png"))
    plt.close(fig)
    
    # Calculate and save comparison metrics
    comparison = compare_scenarios(baseline_metrics, contract_metrics)
    with open(os.path.join(comparison_dir, "comparison.txt"), "w") as f:
        f.write("Comparison Metrics:\n")
        f.write(f"Total Reward Improvement: {comparison['total_reward_improvement']:.2f}%\n")
        f.write(f"Waste Level Improvement: {comparison['waste_level_improvement']:.2f}%\n")
        f.write(f"Apple Count Improvement: {comparison['apple_count_improvement']:.2f}%\n")
        f.write("\nReward Improvements per Agent:\n")
        for agent_id, improvement in comparison["reward_improvements_per_agent"].items():
            f.write(f"  {agent_id}: {improvement:.2f}%\n")
        f.write("\nCleaning Count Improvements per Agent:\n")
        for agent_id, improvement in comparison["cleaning_count_improvements_per_agent"].items():
            f.write(f"  {agent_id}: {improvement:.2f}%\n")
    
    print(f"Comparison results saved to {comparison_dir}")


def main():
    """Main function for running experiments."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MARL-CONTRACTS experiments")
    parser.add_argument("--experiment", type=str, default="baseline",
                        choices=["baseline", "contract", "search", "parameter", "compare"],
                        help="Type of experiment to run")
    parser.add_argument("--num-agents", type=int, default=2,
                        help="Number of agents in the environment")
    parser.add_argument("--grid-size", type=int, nargs=2, default=[20, 20],
                        help="Size of the grid (height, width)")
    parser.add_argument("--episode-length", type=int, default=500,
                        help="Maximum number of steps per episode")
    parser.add_argument("--num-iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--steps-per-iteration", type=int, default=1000,
                        help="Number of steps to collect per iteration")
    parser.add_argument("--contract-type", type=str, default="cleaning",
                        choices=["cleaning", "threshold"],
                        help="Type of contract to use")
    parser.add_argument("--payment", type=float, default=1.0,
                        help="Payment amount for the contract")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Threshold for the threshold contract")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for computation")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results to")
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create environment factory
    env_factory = create_env_factory(
        num_agents=args.num_agents,
        grid_size=tuple(args.grid_size),
        episode_length=args.episode_length
    )
    
    # Run experiment
    if args.experiment == "baseline":
        # Run baseline experiment
        results = run_baseline_experiment(
            env_factory,
            num_iterations=args.num_iterations,
            steps_per_iteration=args.steps_per_iteration,
            device=args.device
        )
        save_results(results, "baseline", args.output_dir)
    
    elif args.experiment == "contract":
        # Create contract factory
        contract_factory = create_contract_factory(
            contract_type=args.contract_type,
            payment=args.payment,
            threshold=args.threshold
        )
        
        # Run contract experiment
        results = run_contract_experiment(
            env_factory,
            contract_factory,
            num_iterations=args.num_iterations,
            steps_per_iteration=args.steps_per_iteration,
            device=args.device
        )
        save_results(results, f"contract_{args.contract_type}", args.output_dir)
    
    elif args.experiment == "search":
        # Run contract search experiment
        results = run_contract_search_experiment(
            env_factory,
            num_episodes=10,
            max_depth=2
        )
        save_results(results, "contract_search", args.output_dir)
    
    elif args.experiment == "parameter":
        # Run parameter analysis experiment
        results = run_parameter_analysis_experiment(
            env_factory,
            num_episodes=10
        )
        save_results(results, "parameter_analysis", args.output_dir)
    
    elif args.experiment == "compare":
        # Run baseline experiment
        baseline_results = run_baseline_experiment(
            env_factory,
            num_iterations=args.num_iterations,
            steps_per_iteration=args.steps_per_iteration,
            device=args.device
        )
        save_results(baseline_results, "baseline", args.output_dir)
        
        # Create contract factory
        contract_factory = create_contract_factory(
            contract_type=args.contract_type,
            payment=args.payment,
            threshold=args.threshold
        )
        
        # Run contract experiment
        contract_results = run_contract_experiment(
            env_factory,
            contract_factory,
            num_iterations=args.num_iterations,
            steps_per_iteration=args.steps_per_iteration,
            device=args.device
        )
        save_results(contract_results, f"contract_{args.contract_type}", args.output_dir)
        
        # Compare experiments
        compare_experiments(baseline_results, contract_results, args.output_dir)


if __name__ == "__main__":
    main()
