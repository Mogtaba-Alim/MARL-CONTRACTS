# Introduction

- **Introduce myself**
- **Talk about the name being temporary** (so don’t think of it too much)

This is some work I’ve been on for the past month and a half (two months). We’re in the experimenting/code phase, but before we get too deep into it, I wanted to present it to you with the hope that you can catch any blind spots I might have missed and give me genuinely scathing feedback. This includes the idea itself, the direction we’re taking, and the implementation—feel free to criticize anything. In fact, the harsher, the better.

# Motivation

This section discusses how this paper is relevant from a game theory perspective.

- **Contracting in Economics and Other Domains:**  
  How contracting is relevant in economics and other fields.

- **Multi-Agent Reinforcement Learning (MARL):**  
  There are papers showing that contracting can solve social dilemmas in multi-agent RL. For example, see [this paper](https://arxiv.org/abs/2208.10469).

- **Dynamically Sized Contracts:**  
  Recent work creates dynamically sized contracts by extending the number of agents in the environment via “recruitment” (i.e., introducing more agents into the environment). For more details, refer to [this paper](https://arxiv.org/abs/2408.09686).  
  *(Later, this work actually quotes Sriram’s paper "Multi Type Mean Field Reinforcement Learning": [link](https://arxiv.org/abs/2408.09686).)*

- **Real-World Considerations:**  
  In both papers, the contracts are designed to include every single agent in the environment. However, in the real world—and from a game theoretic perspective—it is almost impossible to get every participant or agent to agree to a certain set of conditions. A recent example is the US and Liechtenstein being the only two nations not part of the World Health Organization.

- **Real-Life Contracts:**  
  The goal is to design contracts that are applicable in real-life scenarios.

# Background

- Introduce the **Contracting Paper**
- Describe their setup for the MARL games
- Present the **formulas** used
- Explain how **contracts are formed**
- Outline **how agreements are reached** and how the game is played
- Discuss their **theoretical assumptions**
- Detail **Theorem 3.1** and its implications
- Review their **experimental setup**
- Summarize their **results**

# What We Want To Do

- **Objective:**  
  Find a set of contracts **S** that include at most **K** agents (where **K < N**, and **N** is the total number of agents in the environment).

- **Theoretical Guarantees:**  
  Explore what theoretical guarantees we can have about the quality of these contracts:
  - Do the same assumptions from the original agents paper apply when only a subset (K out of N) of agents is involved?
  - Which assumptions hold and which do not?
  - Given the assumptions that do hold, can we ensure optimality as in the original paper, but with smaller contracts?
  - If we cannot guarantee the same optimality under the old conditions, can we modify the contract conditions to guarantee optimality?
  - If not, can we prove a weaker guarantee that still shows an improvement in the total welfare for agents within the contract?
  - Investigate if it’s possible to find contracts that don’t require all agents.

This section outlines our plan to modify the original scope of the paper and determine which theoretical assumptions persist, which do not, and whether we can prove a similar theory for a smaller contract—or at least a weaker guarantee.

# Theoretical Assumptions

*(Content to be added)*

# Experimental Setup

- **K Agent Agreements:**  
  Start with agreements involving **K** agents, where **K** is a hyperparameter representing the maximum contract size we want.

- **Modified Agreement Stage:**  
  Adjust the agreement stage so that only at most **K** (or exactly **K**) agents need to agree to a contract.

- **External Agents:**  
  By default, all remaining agents (i.e., the **N-K** agents that did not agree) are considered external to the contract.

- **Simulation:**  
  Simulate the contract under these conditions.
