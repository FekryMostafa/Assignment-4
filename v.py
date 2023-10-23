import gym
import collections

# Initialize constants
ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20
SEED = 42

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME, is_slippery=True)
        self.state = self.env.reset()
        self.state = self.state[0] if isinstance(self.state, tuple) else self.state
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = {i: 0.0 for i in range(self.env.observation_space.n)}

    @staticmethod
    def create_env():
        return gym.make(ENV_NAME, is_slippery=True)
    
    def get_state(self, state):
        return state[0] if isinstance(state, tuple) else state

    def update_transits_rewards(self, state, action, new_state, reward):
        self.transits[(state, action)][new_state] += 1
        self.rewards[(state, action, new_state)] = reward
    
    def play_n_random_steps(self, count):
        state = self.get_state(self.env.reset())
        for i in range(count):
            action = self.env.action_space.sample()
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            new_state = self.get_state(new_state)
            self.update_transits_rewards(state, action, new_state, reward)
            if terminated or truncated:
                state = self.get_state(self.env.reset())
            else:
                state = new_state

    def print_value_table(self):
        max_key_length = max(len(str(key)) for key in self.values.keys())
        max_value_length = max(len(str(value)) for value in self.values.values())

        # Print the table header
        max_key_length = max(len(str(key)) for key in self.values.keys())
        max_value_length = max(len("{:.15f}".format(value)) for value in self.values.values())

        # Print the table header
        print(f"{'Key':<{max_key_length}}{'Value':<{max_value_length + 2}}")
        print('-' * (max_key_length + max_value_length + 3))

        # Print the table data
        for key, value in self.values.items():
            print(f"{key:<{max_key_length}}: {value:.15f}")
    
    def extract_policy(self):
        policy = [self.select_action(state) for state in range(self.env.observation_space.n)]
        return policy

    def print_policy(self, policy):
        action_names = ['L', 'D', 'R', 'U']
        for i in range(0, len(policy), 4):
            print("".join(action_names[policy[i+j]] for j in range(4)))

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for target_state, count in target_counts.items():
            reward = self.rewards[(state, action, target_state)]
            action_value += (count / total) * (reward + GAMMA * self.values[target_state])
        return action_value
    
    def select_action(self, state): 
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = self.get_state(env.reset())
        while True:
            action = self.select_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state = self.get_state(new_state)
            self.update_transits_rewards(state, action, new_state, reward)
            total_reward += reward
            state = new_state 
            if terminated or truncated:
                break
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)

if __name__ == "__main__":
    test_env = Agent.create_env()
    agent = Agent()
    iter_no = 0
    best_reward = 0.0
    
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES

        if reward > best_reward:
            print(f"Best reward updated {best_reward:.3f} -> {reward:.3f}")
            best_reward = reward
  
        if reward > 0.80:
            print(f"Solved in {iter_no} iterations!")
            agent.print_value_table()
            policy = agent.extract_policy()
            agent.print_policy(policy)
            break