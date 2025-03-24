"""
Implementation of the Cleanup domain, a social dilemma environment.

In this environment, agents can either clean a polluted river or harvest apples.
Apples only grow if the river is clean, creating a social dilemma where
cleaning has no direct reward but is necessary for the collective good.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers


class CleanupEnv(ParallelEnv):
    """
    Cleanup environment implementation based on the PettingZoo ParallelEnv interface.
    
    This environment simulates a social dilemma where agents can either clean a polluted
    river or harvest apples. Apples only grow if the river is clean, creating a tension
    between individual and collective interests.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "cleanup_v0",
    }
    
    def __init__(
        self,
        num_agents=2,
        grid_size=(20, 20),
        episode_length=1000,
        waste_threshold=0.5,
        waste_start_amount=0.75,
        apple_spawn_rate=0.05,
        waste_decay_rate=0.0,
        render_mode=None,
    ):
        """
        Initialize the Cleanup environment.
        
        Args:
            num_agents (int): Number of agents in the environment.
            grid_size (tuple): Size of the grid (height, width).
            episode_length (int): Maximum number of steps per episode.
            waste_threshold (float): Threshold above which apples stop spawning.
            waste_start_amount (float): Initial amount of waste in the river.
            apple_spawn_rate (float): Base rate at which apples spawn.
            waste_decay_rate (float): Rate at which waste naturally decays.
            render_mode (str): Rendering mode.
        """
        self.num_agents = num_agents
        self.grid_height, self.grid_width = grid_size
        self.episode_length = episode_length
        self.waste_threshold = waste_threshold
        self.waste_start_amount = waste_start_amount
        self.apple_spawn_rate = apple_spawn_rate
        self.waste_decay_rate = waste_decay_rate
        self.render_mode = render_mode
        
        # Define agent IDs
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        
        # Define action and observation spaces
        self.action_space = {
            agent: spaces.Discrete(5)  # 0: NOOP, 1: UP, 2: RIGHT, 3: DOWN, 4: LEFT, 5: CLEAN/HARVEST
            for agent in self.possible_agents
        }
        
        # Observation: agent position, other agents' positions, waste level, apple positions
        self.observation_space = {
            agent: spaces.Dict({
                "position": spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.int32),
                "grid": spaces.Box(low=0, high=1, shape=(grid_size[0], grid_size[1], 3), dtype=np.float32),
                "waste_level": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            })
            for agent in self.possible_agents
        }
        
        # Initialize environment state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        
        Args:
            seed (int): Random seed.
            options (dict): Additional options.
            
        Returns:
            observations (dict): Initial observations for each agent.
            info (dict): Additional information.
        """
        self.agents = self.possible_agents[:]
        self.current_step = 0
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize grid
        self.grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.float32)
        
        # Set up river area (bottom 25% of grid)
        river_height = self.grid_height // 4
        self.river_area = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        self.river_area[-river_height:, :] = True
        
        # Set up orchard area (top 75% of grid)
        self.orchard_area = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        self.orchard_area[:-river_height, :] = True
        
        # Initialize waste in river
        self.waste_level = self.waste_start_amount
        self._update_waste_visualization()
        
        # Initialize apples (none at start if waste is above threshold)
        self.apple_positions = set()
        if self.waste_level < self.waste_threshold:
            self._spawn_apples(initial=True)
        
        # Initialize agent positions (randomly)
        self.agent_positions = {}
        for agent in self.agents:
            pos = (np.random.randint(0, self.grid_height), np.random.randint(0, self.grid_width))
            self.agent_positions[agent] = pos
        
        # Update grid visualization
        self._update_grid()
        
        # Get initial observations
        observations = self._get_observations()
        
        # Initialize rewards and done flags
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        return observations, self.infos
    
    def step(self, actions):
        """
        Take a step in the environment.
        
        Args:
            actions (dict): Actions for each agent.
            
        Returns:
            observations (dict): Observations for each agent.
            rewards (dict): Rewards for each agent.
            terminations (dict): Whether each agent is done.
            truncations (dict): Whether each agent is truncated.
            infos (dict): Additional information.
        """
        self.current_step += 1
        
        # Process actions
        for agent in self.agents:
            action = actions[agent]
            self._process_agent_action(agent, action)
        
        # Update environment state
        self._update_environment()
        
        # Get observations
        observations = self._get_observations()
        
        # Check if episode is done
        if self.current_step >= self.episode_length:
            self.truncations = {agent: True for agent in self.agents}
        
        return observations, self.rewards, self.terminations, self.truncations, self.infos
    
    def _process_agent_action(self, agent, action):
        """
        Process an agent's action.
        
        Args:
            agent (str): Agent ID.
            action (int): Action to take.
        """
        # Get current position
        pos = self.agent_positions[agent]
        row, col = pos
        
        # Initialize reward
        self.rewards[agent] = 0
        
        # Process movement actions
        if action == 1:  # UP
            new_row = max(0, row - 1)
            self.agent_positions[agent] = (new_row, col)
        elif action == 2:  # RIGHT
            new_col = min(self.grid_width - 1, col + 1)
            self.agent_positions[agent] = (row, new_col)
        elif action == 3:  # DOWN
            new_row = min(self.grid_height - 1, row + 1)
            self.agent_positions[agent] = (new_row, col)
        elif action == 4:  # LEFT
            new_col = max(0, col - 1)
            self.agent_positions[agent] = (row, new_col)
        elif action == 5:  # CLEAN/HARVEST
            # If in river area, clean
            if self.river_area[row, col]:
                self._clean_waste(agent)
            # If in orchard area, try to harvest
            elif self.orchard_area[row, col]:
                self._harvest_apple(agent)
    
    def _clean_waste(self, agent):
        """
        Clean waste from the river.
        
        Args:
            agent (str): Agent ID.
        """
        # Reduce waste level
        if self.waste_level > 0:
            self.waste_level = max(0, self.waste_level - 0.01)
            # No direct reward for cleaning in the base environment
            self.infos[agent]["cleaned"] = True
        else:
            self.infos[agent]["cleaned"] = False
    
    def _harvest_apple(self, agent):
        """
        Harvest an apple if present at the agent's position.
        
        Args:
            agent (str): Agent ID.
        """
        pos = self.agent_positions[agent]
        # Check if there's an apple at this position
        if pos in self.apple_positions:
            # Remove the apple
            self.apple_positions.remove(pos)
            # Give reward
            self.rewards[agent] += 1
            self.infos[agent]["harvested"] = True
        else:
            self.infos[agent]["harvested"] = False
    
    def _update_environment(self):
        """Update the environment state after all agents have acted."""
        # Natural waste decay
        if self.waste_decay_rate > 0:
            self.waste_level = max(0, self.waste_level - self.waste_decay_rate)
        
        # Spawn apples if waste is below threshold
        if self.waste_level < self.waste_threshold:
            self._spawn_apples()
        
        # Update grid visualization
        self._update_grid()
        self._update_waste_visualization()
    
    def _spawn_apples(self, initial=False):
        """
        Spawn apples in the orchard area.
        
        Args:
            initial (bool): Whether this is the initial spawn.
        """
        # Calculate spawn probability based on waste level
        spawn_prob = self.apple_spawn_rate * (1 - self.waste_level / self.waste_threshold)
        
        # Spawn apples with probability proportional to cleanliness
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if self.orchard_area[row, col] and (row, col) not in self.apple_positions:
                    if np.random.random() < spawn_prob:
                        self.apple_positions.add((row, col))
    
    def _update_grid(self):
        """Update the grid visualization."""
        # Reset grid
        self.grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.float32)
        
        # Add river area (blue)
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if self.river_area[row, col]:
                    self.grid[row, col, 2] = 0.5  # Blue channel
        
        # Add waste visualization
        self._update_waste_visualization()
        
        # Add apples (green)
        for pos in self.apple_positions:
            row, col = pos
            self.grid[row, col, 1] = 1.0  # Green channel
        
        # Add agents (red)
        for agent, pos in self.agent_positions.items():
            row, col = pos
            self.grid[row, col, 0] = 1.0  # Red channel
    
    def _update_waste_visualization(self):
        """Update the waste visualization in the river area."""
        # Visualize waste as darker blue in the river area
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if self.river_area[row, col]:
                    # Darker blue means more waste
                    cleanliness = 1 - min(1, self.waste_level / self.waste_threshold)
                    self.grid[row, col, 2] = 0.2 + 0.8 * cleanliness
    
    def _get_observations(self):
        """
        Get observations for all agents.
        
        Returns:
            observations (dict): Observations for each agent.
        """
        observations = {}
        for agent in self.agents:
            observations[agent] = {
                "position": np.array(self.agent_positions[agent]),
                "grid": self.grid.copy(),
                "waste_level": np.array([self.waste_level]),
            }
        return observations
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            rgb_array (np.ndarray): RGB array of the rendered environment.
        """
        if self.render_mode == "rgb_array":
            return self.grid
        elif self.render_mode == "human":
            # Implement human rendering if needed
            pass
    
    def close(self):
        """Close the environment."""
        pass


def make_cleanup_env(num_agents=2, **kwargs):
    """
    Create a Cleanup environment with the given parameters.
    
    Args:
        num_agents (int): Number of agents in the environment.
        **kwargs: Additional parameters for the environment.
        
    Returns:
        env (CleanupEnv): The Cleanup environment.
    """
    env = CleanupEnv(num_agents=num_agents, **kwargs)
    # Apply PettingZoo wrappers
    env = wrappers.OrderEnforcingWrapper(env)
    return env 