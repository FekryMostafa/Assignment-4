import gym
import numpy as np

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20
EXPLORATION_STEPS = 100

class Agent:
    def __init__(self, is_slippery=True):
        self.env = gym.make(ENV_NAME, is_slippery=is_slippery)
        self.state_space = self.env.observation_space.n
        self.action_space = self.env.action_space.n
        self.value_table = np.zeros(self.state_space)
        self.best_reward = -np.inf

    def explore_environment(self):
        state = self.env.reset()
        for _ in range(EXPLORATION_STEPS):
            action = self.env.action_space.sample()
        step_result = self.env.step(action)
        print("Step result:", step_result)
        next_state, reward, done, _, _ = step_result  # Corrected this line



    def calc_action_value(self, state):
        action_values = np.zeros(self.action_space)
        for action in range(self.action_space):
            for prob, next_state, reward, _ in self.env.P[state][action]:
                action_values[action] += prob * (reward + GAMMA * self.value_table[next_state])
        return action_values

    def value_iteration(self):
        iteration_count = 0
        while True:
            iteration_count += 1
            new_value_table = np.copy(self.value_table)
            for state in range(self.state_space):
                action_values = self.calc_action_value(state)
                new_value_table[state] = np.max(action_values)
            if np.all(np.isclose(new_value_table, self.value_table, atol=1e-5)):
                break
            self.value_table = new_value_table
        print(f"Converged after {iteration_count} iterations.")


    def extract_policy(self):
        policy = np.zeros(self.state_space, dtype=int)
        for state in range(self.state_space):
            action_values = self.calc_action_value(state)
            policy[state] = np.argmax(action_values)
        return policy

    def play_episode(self, policy):
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Assuming the first element is the integer state
        state = int(state)  # Make sure state is an integer
        total_reward = 0
        done = False
        while not done:
            action = policy[state]
            next_state, reward, done, _, _ = self.env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]  # Assuming the first element is the integer state
            state = int(next_state)  # Make sure next_state is an integer
            total_reward += reward
        return total_reward



if __name__ == "__main__":
    slippery_agent = Agent(is_slippery=True)
    while True:
        slippery_agent.explore_environment()
        slippery_agent.value_iteration()
        optimal_policy = slippery_agent.extract_policy()
        
        total_rewards = 0
        for _ in range(TEST_EPISODES):
            reward = slippery_agent.play_episode(optimal_policy)
            total_rewards += reward
        
        average_reward = total_rewards / TEST_EPISODES
        if average_reward > slippery_agent.best_reward:
            slippery_agent.best_reward = average_reward
        
        print(f"Average rewards: {average_reward}, Best reward: {slippery_agent.best_reward}")
        
        if average_reward > 0.80:
            print("Stopping as average reward exceeds 0.80.")
            break
    
    not_slippery_agent = Agent(is_slippery=False)

    while True:
            not_slippery_agent.explore_environment()
            not_slippery_agent.value_iteration()
            optimal_policy = not_slippery_agent.extract_policy()
            
            total_rewards = 0
            for _ in range(TEST_EPISODES):
                reward = not_slippery_agent.play_episode(optimal_policy)
                total_rewards += reward
            
            average_reward = total_rewards / TEST_EPISODES
            if average_reward > not_slippery_agent.best_reward:
                not_slippery_agent.best_reward = average_reward
            
            print(f"Average rewards: {average_reward}, Best reward: {not_slippery_agent.best_reward}")
            
            if average_reward > 0.80:
                print("Stopping as average reward exceeds 0.80.")
                break
