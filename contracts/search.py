"""
Contract search algorithm implementation.

This module implements a tree-based search algorithm for finding optimal contracts
in a multi-agent environment. The search algorithm explores the space of possible
contracts and evaluates them based on their performance in the environment.
"""

import numpy as np
import itertools
from collections import defaultdict
from tqdm import tqdm

from environment.contract_wrapper import CleaningContract, ThresholdContract


class ContractSearchNode:
    """
    Node in the contract search tree.
    
    Each node represents a specific contract configuration.
    """
    
    def __init__(self, contracts=None, parent=None, depth=0):
        """
        Initialize a contract search node.
        
        Args:
            contracts (list): List of Contract objects.
            parent: Parent node in the search tree.
            depth (int): Depth of the node in the search tree.
        """
        self.contracts = contracts or []
        self.parent = parent
        self.depth = depth
        self.children = []
        self.value = None
        self.metrics = None
    
    def add_child(self, child):
        """
        Add a child node to this node.
        
        Args:
            child: Child node to add.
        """
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1
    
    def __str__(self):
        """String representation of the node."""
        return f"ContractNode(depth={self.depth}, contracts={len(self.contracts)}, value={self.value})"


class ContractSearchAlgorithm:
    """
    Tree-based search algorithm for finding optimal contracts.
    
    This algorithm explores the space of possible contracts and evaluates them
    based on their performance in the environment.
    """
    
    def __init__(self, env_factory, evaluator, max_depth=3):
        """
        Initialize the contract search algorithm.
        
        Args:
            env_factory: Function that creates a new environment instance.
            evaluator: Function that evaluates a contract in the environment.
            max_depth (int): Maximum depth of the search tree.
        """
        self.env_factory = env_factory
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.root = ContractSearchNode()
        self.best_node = self.root
    
    def search(self, verbose=True):
        """
        Perform the contract search.
        
        Args:
            verbose (bool): Whether to print progress information.
            
        Returns:
            best_contracts (list): List of the best contracts found.
            best_value (float): Value of the best contracts.
        """
        # Evaluate the root node (no contracts)
        env = self.env_factory()
        self.root.value, self.root.metrics = self.evaluator(env, [])
        self.best_node = self.root
        
        # Initialize the search queue with the root node
        queue = [self.root]
        
        # Perform iterative deepening search
        while queue:
            node = queue.pop(0)
            
            # Skip if we've reached the maximum depth
            if node.depth >= self.max_depth:
                continue
            
            # Generate child nodes
            children = self._generate_children(node)
            
            # Evaluate and prune children
            promising_children = []
            for child in tqdm(children, desc=f"Evaluating depth {node.depth + 1} contracts", disable=not verbose):
                # Create a new environment for evaluation
                env = self.env_factory()
                
                # Evaluate the child node
                child.value, child.metrics = self.evaluator(env, child.contracts)
                
                # Check if this is the best node so far
                if child.value > self.best_node.value:
                    self.best_node = child
                
                # Check if this child is promising
                if self._is_promising(child):
                    promising_children.append(child)
                    node.add_child(child)
            
            # Add promising children to the queue
            queue.extend(promising_children)
            
            if verbose:
                print(f"Depth {node.depth + 1}: Evaluated {len(children)} contracts, {len(promising_children)} promising")
                if self.best_node.depth == node.depth + 1:
                    print(f"New best contract found at depth {self.best_node.depth} with value {self.best_node.value}")
        
        return self.best_node.contracts, self.best_node.value
    
    def _generate_children(self, node):
        """
        Generate child nodes for a given node.
        
        Args:
            node: Parent node.
            
        Returns:
            children (list): List of child nodes.
        """
        children = []
        
        # Get the environment to determine agent IDs
        env = self.env_factory()
        agent_ids = env.possible_agents
        
        # Generate cleaning contracts
        for payer, cleaner in itertools.permutations(agent_ids, 2):
            for payment in [0.5, 1.0, 1.5, 2.0]:
                # Create a new cleaning contract
                contract = CleaningContract(
                    name=f"cleaning_{payer}_to_{cleaner}_{payment}",
                    payer=payer,
                    cleaner=cleaner,
                    payment_per_cleaning=payment
                )
                
                # Create a new node with this contract added
                new_contracts = node.contracts.copy()
                new_contracts.append(contract)
                child = ContractSearchNode(contracts=new_contracts)
                children.append(child)
        
        # Generate threshold contracts
        for payer, cleaner in itertools.permutations(agent_ids, 2):
            for payment in [0.1, 0.2, 0.3]:
                for threshold in [0.3, 0.4, 0.5]:
                    # Create a new threshold contract
                    contract = ThresholdContract(
                        name=f"threshold_{payer}_to_{cleaner}_{payment}_{threshold}",
                        payer=payer,
                        cleaner=cleaner,
                        payment=payment,
                        threshold=threshold
                    )
                    
                    # Create a new node with this contract added
                    new_contracts = node.contracts.copy()
                    new_contracts.append(contract)
                    child = ContractSearchNode(contracts=new_contracts)
                    children.append(child)
        
        return children
    
    def _is_promising(self, node):
        """
        Check if a node is promising and should be explored further.
        
        Args:
            node: Node to check.
            
        Returns:
            promising (bool): Whether the node is promising.
        """
        # Check if the node improves over its parent
        if node.parent and node.value <= node.parent.value:
            return False
        
        # Check if the contract is stable (all agents benefit)
        if not self._is_stable(node):
            return False
        
        # Check if the node is within some percentage of the best value
        if node.value < 0.8 * self.best_node.value:
            return False
        
        return True
    
    def _is_stable(self, node):
        """
        Check if a contract is stable (all agents benefit).
        
        Args:
            node: Node to check.
            
        Returns:
            stable (bool): Whether the contract is stable.
        """
        # Get the baseline rewards (no contract)
        baseline_rewards = self.root.metrics["rewards"]
        
        # Get the rewards with this contract
        contract_rewards = node.metrics["rewards"]
        
        # Check if all agents are at least as well off as without the contract
        for agent, reward in contract_rewards.items():
            if reward < baseline_rewards[agent]:
                return False
        
        return True


def evaluate_contracts(env, contracts, num_episodes=10, max_steps=None):
    """
    Evaluate a set of contracts in the environment.
    
    This function runs episodes in the environment with the given contracts and
    returns metrics about the performance.
    
    Args:
        env: Environment to evaluate in.
        contracts (list): List of Contract objects to enforce.
        num_episodes (int): Number of episodes to run.
        max_steps (int): Maximum number of steps per episode.
        
    Returns:
        value (float): Value of the contracts (average total reward).
        metrics (dict): Dictionary of evaluation metrics.
    """
    # Add contracts to the environment
    for contract in contracts:
        env.add_contract(contract)
    
    # Initialize metrics
    total_rewards = defaultdict(float)
    episode_lengths = []
    waste_levels = []
    apple_counts = []
    cleaning_counts = defaultdict(int)
    
    # Run episodes
    for episode in range(num_episodes):
        observations, _ = env.reset()
        done = False
        step = 0
        
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
            
            # Record waste level
            waste_levels.append(env.waste_level)
            
            # Record apple count
            apple_counts.append(len(env.apple_positions))
            
            # Check if episode is done
            done = all(terminations.values()) or all(truncations.values())
            step += 1
            
            if max_steps and step >= max_steps:
                break
        
        episode_lengths.append(step)
    
    # Calculate average metrics
    avg_rewards = {agent: reward / num_episodes for agent, reward in total_rewards.items()}
    avg_total_reward = sum(avg_rewards.values())
    avg_waste_level = sum(waste_levels) / len(waste_levels)
    avg_apple_count = sum(apple_counts) / len(apple_counts)
    avg_episode_length = sum(episode_lengths) / len(episode_lengths)
    
    # Compile metrics
    metrics = {
        "rewards": avg_rewards,
        "waste_level": avg_waste_level,
        "apple_count": avg_apple_count,
        "episode_length": avg_episode_length,
        "cleaning_counts": cleaning_counts,
    }
    
    # Remove contracts from the environment
    for contract in contracts:
        env.remove_contract(contract.name)
    
    return avg_total_reward, metrics


def find_optimal_contracts(env_factory, num_episodes=10, max_depth=2, verbose=True):
    """
    Find the optimal contracts for a given environment.
    
    Args:
        env_factory: Function that creates a new environment instance.
        num_episodes (int): Number of episodes to run for evaluation.
        max_depth (int): Maximum depth of the search tree.
        verbose (bool): Whether to print progress information.
        
    Returns:
        best_contracts (list): List of the best contracts found.
        best_value (float): Value of the best contracts.
        metrics (dict): Dictionary of evaluation metrics.
    """
    # Create an evaluator function
    def evaluator(env, contracts):
        return evaluate_contracts(env, contracts, num_episodes=num_episodes)
    
    # Create the search algorithm
    search_algorithm = ContractSearchAlgorithm(
        env_factory=env_factory,
        evaluator=evaluator,
        max_depth=max_depth
    )
    
    # Perform the search
    best_contracts, best_value = search_algorithm.search(verbose=verbose)
    
    # Get metrics for the best contracts
    env = env_factory()
    _, metrics = evaluator(env, best_contracts)
    
    return best_contracts, best_value, metrics 