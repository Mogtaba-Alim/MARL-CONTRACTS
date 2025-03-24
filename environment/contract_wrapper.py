"""
Contract wrapper for the Cleanup environment.

This wrapper modifies the reward function of the environment to enforce contracts
between agents. Contracts are agreements between agents to transfer rewards based
on certain conditions.
"""

import numpy as np
from gymnasium import Wrapper


class ContractWrapper(Wrapper):
    """
    Wrapper that enforces contracts between agents in a multi-agent environment.
    
    A contract is an agreement between agents to transfer rewards based on certain
    conditions. This wrapper modifies the reward function of the environment to
    enforce these contracts.
    """
    
    def __init__(self, env, contracts=None):
        """
        Initialize the contract wrapper.
        
        Args:
            env: The environment to wrap.
            contracts (list): List of Contract objects to enforce.
        """
        super().__init__(env)
        self.env = env
        self.contracts = contracts or []
        
        # Keep track of contract-related metrics
        self.contract_transfers = {agent: 0 for agent in self.env.possible_agents}
        self.contract_events = []
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment and contract state.
        
        Args:
            seed (int): Random seed.
            options (dict): Additional options.
            
        Returns:
            observations (dict): Initial observations for each agent.
            info (dict): Additional information.
        """
        observations, info = self.env.reset(seed=seed, options=options)
        
        # Reset contract metrics
        self.contract_transfers = {agent: 0 for agent in self.env.possible_agents}
        self.contract_events = []
        
        return observations, info
    
    def step(self, actions):
        """
        Take a step in the environment and apply contract transfers.
        
        Args:
            actions (dict): Actions for each agent.
            
        Returns:
            observations (dict): Observations for each agent.
            rewards (dict): Rewards for each agent after contract transfers.
            terminations (dict): Whether each agent is done.
            truncations (dict): Whether each agent is truncated.
            infos (dict): Additional information.
        """
        # Take a step in the environment
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # Apply contract transfers
        rewards = self._apply_contracts(rewards, infos)
        
        # Update infos with contract information
        for agent in self.env.agents:
            infos[agent]["contract_transfers"] = self.contract_transfers[agent]
        
        return observations, rewards, terminations, truncations, infos
    
    def _apply_contracts(self, rewards, infos):
        """
        Apply contract transfers to the rewards.
        
        Args:
            rewards (dict): Original rewards for each agent.
            infos (dict): Additional information from the environment step.
            
        Returns:
            modified_rewards (dict): Rewards after contract transfers.
        """
        # Make a copy of the original rewards
        modified_rewards = rewards.copy()
        
        # Apply each contract
        for contract in self.contracts:
            transfers = contract.calculate_transfers(self.env, infos)
            
            # Apply transfers
            for agent, transfer in transfers.items():
                modified_rewards[agent] += transfer
                self.contract_transfers[agent] += transfer
                
                # Record contract event if transfer is non-zero
                if transfer != 0:
                    self.contract_events.append({
                        "step": self.env.current_step,
                        "contract": contract.name,
                        "agent": agent,
                        "transfer": transfer
                    })
        
        return modified_rewards
    
    def add_contract(self, contract):
        """
        Add a contract to the environment.
        
        Args:
            contract: Contract object to add.
        """
        self.contracts.append(contract)
    
    def remove_contract(self, contract_name):
        """
        Remove a contract from the environment.
        
        Args:
            contract_name (str): Name of the contract to remove.
        """
        self.contracts = [c for c in self.contracts if c.name != contract_name]
    
    def get_contract_metrics(self):
        """
        Get metrics about contract enforcement.
        
        Returns:
            metrics (dict): Dictionary of contract metrics.
        """
        return {
            "transfers": self.contract_transfers,
            "events": self.contract_events
        }


class Contract:
    """
    Base class for contracts between agents.
    
    A contract defines conditions under which agents transfer rewards to each other.
    Subclasses should implement the calculate_transfers method.
    """
    
    def __init__(self, name, participants):
        """
        Initialize a contract.
        
        Args:
            name (str): Name of the contract.
            participants (list): List of agent IDs participating in the contract.
        """
        self.name = name
        self.participants = participants
    
    def calculate_transfers(self, env, infos):
        """
        Calculate reward transfers based on the contract terms.
        
        Args:
            env: The environment.
            infos (dict): Additional information from the environment step.
            
        Returns:
            transfers (dict): Dictionary mapping agent IDs to reward transfers.
        """
        raise NotImplementedError("Subclasses must implement calculate_transfers")


class CleaningContract(Contract):
    """
    Contract that rewards agents for cleaning waste.
    
    This contract transfers rewards from one agent to another when the latter
    cleans waste from the river.
    """
    
    def __init__(self, name, payer, cleaner, payment_per_cleaning):
        """
        Initialize a cleaning contract.
        
        Args:
            name (str): Name of the contract.
            payer (str): ID of the agent who pays for cleaning.
            cleaner (str): ID of the agent who cleans.
            payment_per_cleaning (float): Amount to transfer per cleaning action.
        """
        super().__init__(name, [payer, cleaner])
        self.payer = payer
        self.cleaner = cleaner
        self.payment_per_cleaning = payment_per_cleaning
    
    def calculate_transfers(self, env, infos):
        """
        Calculate reward transfers based on cleaning actions.
        
        Args:
            env: The environment.
            infos (dict): Additional information from the environment step.
            
        Returns:
            transfers (dict): Dictionary mapping agent IDs to reward transfers.
        """
        transfers = {agent: 0 for agent in self.participants}
        
        # Check if the cleaner cleaned waste
        if self.cleaner in infos and infos[self.cleaner].get("cleaned", False):
            # Transfer payment from payer to cleaner
            transfers[self.payer] -= self.payment_per_cleaning
            transfers[self.cleaner] += self.payment_per_cleaning
        
        return transfers


class ThresholdContract(Contract):
    """
    Contract that rewards agents for maintaining waste below a threshold.
    
    This contract transfers rewards from one agent to another when the waste level
    is kept below a specified threshold.
    """
    
    def __init__(self, name, payer, cleaner, payment, threshold):
        """
        Initialize a threshold contract.
        
        Args:
            name (str): Name of the contract.
            payer (str): ID of the agent who pays for cleaning.
            cleaner (str): ID of the agent who cleans.
            payment (float): Amount to transfer when threshold is met.
            threshold (float): Waste threshold below which payment is made.
        """
        super().__init__(name, [payer, cleaner])
        self.payer = payer
        self.cleaner = cleaner
        self.payment = payment
        self.threshold = threshold
    
    def calculate_transfers(self, env, infos):
        """
        Calculate reward transfers based on waste threshold.
        
        Args:
            env: The environment.
            infos (dict): Additional information from the environment step.
            
        Returns:
            transfers (dict): Dictionary mapping agent IDs to reward transfers.
        """
        transfers = {agent: 0 for agent in self.participants}
        
        # Check if waste is below threshold
        if env.waste_level < self.threshold:
            # Transfer payment from payer to cleaner
            transfers[self.payer] -= self.payment
            transfers[self.cleaner] += self.payment
        
        return transfers


def make_contract_env(env, contracts=None):
    """
    Create a contract-enforcing environment.
    
    Args:
        env: The environment to wrap.
        contracts (list): List of Contract objects to enforce.
        
    Returns:
        wrapped_env: The wrapped environment.
    """
    return ContractWrapper(env, contracts) 