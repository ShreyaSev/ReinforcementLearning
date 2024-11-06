##Importing Packages
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

class GridWorld:
    def __init__(self, size=100, obstacle_density=0.3):
        self.size = size
        self.grid = np.zeros((size, size))
        self.generate_environment(obstacle_density)
        
    def generate_environment(self, obstacle_density):
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < obstacle_density:
                    self.grid[i][j] = 1  # 1 -> obstacle
                    
       
        while True:
            self.start = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            self.goal = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            
            if (self.grid[self.start[0]][self.start[1]] == 0 and 
                self.grid[self.goal[0]][self.goal[1]] == 0 and
                self.start != self.goal):
                break

        self.grid[self.start[0]][self.start[1]] = 2  # 2 -> start
        self.grid[self.goal[0]][self.goal[1]] = 3   # 3 -> goal

class MDPAgent:
    def __init__(self, env):
        self.env = env
        self.size = env.size
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        self.action_names = ['→', '↓', '←', '↑']
        self.gamma = 0.99  #df 
        self.theta = 1e-6  #Convergence Thresh
        self.values = np.zeros((self.size, self.size))
        self.policy = np.zeros((self.size, self.size), dtype=int)
        
        self.policy_history = []
        self.value_history = []
        self.policy_changes = defaultdict(int)
        
    def is_valid_state(self, state):
        x, y = state
        return (0 <= x < self.size and 
                0 <= y < self.size and 
                self.env.grid[x][y] != 1)
    
    def get_reward(self, state):
        if state == self.env.goal:
            return 100
        return -1
    
    def get_next_state(self, state, action):
        next_state = (state[0] + action[0], state[1] + action[1])
        if self.is_valid_state(next_state):
            return next_state
        return state
    
    def value_iteration(self):
        iteration = 0
        while True:
            delta = 0
            old_policy = self.policy.copy()
            
            for i in range(self.size):
                for j in range(self.size):
                    if self.env.grid[i][j] == 1:  #SkipObstacles
                        continue
                        
                    current_state = (i, j)
                    if current_state == self.env.goal:
                        continue
                        
                    v = self.values[i][j]
                    action_values = []
                    
                    for action in self.actions:
                        next_state = self.get_next_state(current_state, action)
                        reward = self.get_reward(next_state)
                        action_values.append(reward + self.gamma * self.values[next_state[0]][next_state[1]])
                    
                    self.values[i][j] = max(action_values)
                    new_policy = np.argmax(action_values)
                    
                    if new_policy != self.policy[i][j]:
                        self.policy_changes[(i, j)] += 1
                    
                    self.policy[i][j] = new_policy
                    delta = max(delta, abs(v - self.values[i][j]))
            
            if iteration % 10 == 0:  
                self.policy_history.append(self.policy.copy())
                self.value_history.append(self.values.copy())
            
            iteration += 1
            
            ##Calculate the policy change percentage
            policy_diff = np.sum(old_policy != self.policy) / (self.size * self.size)
            print(f"Iteration {iteration}: Policy changes: {policy_diff:.2%}")
            
            if delta < self.theta:
                break
        
        return iteration
    
    def get_optimal_path(self):
        path = []
        current_state = self.env.start
        
        while current_state != self.env.goal:
            path.append(current_state)
            action = self.actions[self.policy[current_state[0]][current_state[1]]]
            current_state = self.get_next_state(current_state, action)
            
            if len(path) > self.size * self.size:
                return None
        
        path.append(self.env.goal)
        return path

    def get_policy_string(self, position):
        return self.action_names[self.policy[position[0]][position[1]]]

##Helper Functions
    
def visualize_grid_with_policy(env, agent, path=None, save_prefix=None):
    plt.figure(figsize=(15, 15))
    grid_masked = np.ma.masked_where(env.grid == 1, env.grid)

    plt.imshow(grid_masked, cmap='binary')
    
    for i in range(env.size):
        for j in range(env.size):
            if env.grid[i][j] != 1:  
                arrow = agent.get_policy_string((i, j))
                color = 'red' if (i, j) in agent.policy_changes else 'black'
                plt.text(j, i, arrow, ha='center', va='center', color=color)
   
    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], 'g-', linewidth=2, label='Optimal Path')
    
   
    changes = np.zeros((env.size, env.size))
    for (i, j), count in agent.policy_changes.items():
        changes[i][j] = count
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(changes, cmap='YlOrRd', annot=False)
    plt.title('Policy Change Frequency Heatmap')
    
    if save_prefix:
        plt.savefig(f'{save_prefix}_changes.png')
    plt.show()

def analyze_policy_evolution(agent):
    print("\nPolicy Evolution Analysis:")
    print(f"Total iterations: {len(agent.policy_history)}")
    total_states = agent.size * agent.size
    changes_over_time = []
    
    for i in range(1, len(agent.policy_history)):
        diff = np.sum(agent.policy_history[i] != agent.policy_history[i-1])
        changes_over_time.append(diff / total_states)
    
    plt.figure(figsize=(10, 5))
    plt.plot(changes_over_time)
    plt.title('Policy Changes Over Time')
    plt.xlabel('Iteration (x10)')
    plt.ylabel('Fraction of States Changed')
    plt.show()
    print("\nMost frequently changed states:")
    volatile_states = sorted(agent.policy_changes.items(), key=lambda x: x[1], reverse=True)[:10]
    for state, changes in volatile_states:
        print(f"State {state}: Changed {changes} times")

env = GridWorld(size=100, obstacle_density=0.3)
agent = MDPAgent(env)
iterations = agent.value_iteration()
print(f"\nConverged after {iterations} iterations")
path = agent.get_optimal_path()
print(f"Found path of length: {len(path)}")
visualize_grid_with_policy(env, agent, path)
analyze_policy_evolution(agent)
with open('final_policy.txt', 'w', encoding='utf-8') as f:
    f.write("Final Policy (→:right, ↓:down, ←:left, ↑:up):\n\n")
    for i in range(env.size):
        for j in range(env.size):
            if env.grid[i][j] == 1:
                f.write('█ ')  
            else:
                f.write(f"{agent.get_policy_string((i,j))} ")
            f.write('\n')