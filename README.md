# MARL-CONTRACTS

## Cooperative Contracts in Multi-Agent Reinforcement Learning

This project implements a framework for designing and testing cooperative contracts in multi-agent reinforcement learning (MARL) environments. It focuses on resolving social dilemmas where individual rationality leads to poor collective outcomes.

### Key Features

- Implementation of the Cleanup domain, a classic social dilemma environment
- Contract mechanism for incentivizing cooperation among self-interested agents
- Tree-based search algorithm for finding optimal contracts
- Evaluation framework for assessing contract performance and stability

### Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the simulation:
```bash
python main.py
```

### Project Structure

- `environment/`: Contains the implementation of the Cleanup domain
- `contracts/`: Defines the contract mechanism and search algorithm
- `agents/`: Implements the reinforcement learning agents
- `utils/`: Utility functions for visualization and analysis
- `main.py`: Entry point for running simulations

### References

This project is based on research in cooperative game theory, contract theory, and multi-agent reinforcement learning. Key references include:

- Christoffersen et al. (2023) - "Get It in Writing: Formal Contracts Mitigate Social Dilemmas in MARL"
- Hughes et al. (2018) - "Inequity Aversion Improves Cooperation in Intertemporal Social Dilemmas"
- Yang et al. (2020) - "Learning to Incentivize Other Learning Agents"