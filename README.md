# Racetrack learning

Reinforcement learning allows a policy with the optimal expected reward to be learned. Value iteration can estimate the value of all states, if the models of resulting rewards and states from each state-action pair is known. These values can be used find the optimal policy through policy iteration. If the models of the resulting rewards and states is not known, the optimal policy can still be reached through Q-learning and SARSA. While these methods converge to the optimal policy, they can be computationally expensive. We investigated the performance of value iteration, Q-learning, and SARSA on the racetrack problem, with four tracks. For Q-learning and SARSA, the current action was chosen based on random sampling based on the softmax of the Q-values of each action, with a decaying temperature scaling factor. We found that using a high discount factor is important to reinforcement learning performance on the racetrack problem.

- [Video](Project_demo.mp4) demonstrating the functionality of the code
- [Report](Racetrack%20performance%20with%20machine%20learning.pdf) of racecar performance using value iteration (given the model of results from state-action pairs), Q-learning, and SARSA.
- [Python implementation](RaceTrack.py) of reinforcement learning algorithms
