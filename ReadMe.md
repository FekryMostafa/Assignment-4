# 1. Introduction
In this report, we discuss the findings from the implementation of a value iteration algorithm applied to the FrozenLake environment from OpenAI's Gym. The algorithm's objective is to find the optimal policy for navigating a grid world to reach a goal while avoiding holes.

# 2. Environment Setup
- Environment Name: FrozenLake-v1
- γ: 0.9
- Number of Test Episodes: 20
- Random Seed: 42
- The environment is slippery, meaning that the agent's actions do not always result in the intended movement.

# 3. Final Value Table
| State | Value               |
|-------|---------------------|
| 0     | 0.061971877642296   |
| 1     | 0.054183381771724   |
| 2     | 0.064357872984165   |
| 3     | 0.049702728410445   |
| 4     | 0.082037691170851   |
| 5     | 0.000000000000000   |
| 6     | 0.102403555120940   |
| 7     | 0.000000000000000   |
| 8     | 0.127699404228965   |
| 9     | 0.220440228176505   |
| 10    | 0.271403453561735   |
| 11    | 0.000000000000000   |
| 12    | 0.000000000000000   |
| 13    | 0.343998260960187   |
| 14    | 0.579103452850066   |
| 15    | 0.000000000000000   |
# 4. Extracted Policy
The optimal policy extracted from the value function is as follows:
LURU
LLRL
UDLL
LRR-
L: Move Left
R: Move Right
U: Move Up
D: Move Down
-: Terminal/Blocked State
# 5. Evaluation Results
The evaluation of the policy was conducted over 20 episodes, resulting in an average reward of 0.85. This indicates that the agent has learned to navigate to the goal effectively while avoiding holes.

# 6. Number of Iterations until Convergence
The algorithm converged to the optimal policy and value function in 38 iterations, demonstrating its efficiency in solving the problem at hand.