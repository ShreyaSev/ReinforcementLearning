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
        self.grid = (np.random.random((self.size, self.size)) < obstacle_density).astype(int)
        while True:
            self.start = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            self.goal = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            
            if (self.grid[self.start[0]][self.start[1]] == 0 and 
                self.grid[self.goal[0]][self.goal[1]] == 0 and
                self.start != self.goal):
                break

        self.grid[self.start[0]][self.start[1]] = 2  # 2 -> start
        self.grid[self.goal[0]][self.goal[1]] = 3    # 3 -> goal

class MDPAgent:
    def __init__(self, env):
        self.env = env
        self.size = env.size
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        self.action_names = ['→', '↓', '←', '↑']
        self.gamma = 0.99  #df
        self.theta = 1e-6  #ct
        
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
    
    def value_iteration(self):
        iteration = 0
        convergence_data = []
        
        while True:
            delta = 0
            old_policy = self.policy.copy()
            for i in range(self.size):
                for j in range(self.size):
                    if self.env.grid[i][j] == 1:  # Skip obstacles
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
            
            policy_diff = np.sum(old_policy != self.policy) / (self.size * self.size)
            convergence_data.append((iteration, delta, policy_diff))
            print(f"Iteration {iteration}: Policy changes: {policy_diff:.2%}, Delta: {delta:.6f}")
            
            if delta < self.theta:
                break
                
        self._analyze_convergence(convergence_data)
        return iteration
    
    def get_next_state(self, state, action):
        next_state = (state[0] + action[0], state[1] + action[1])
        if self.is_valid_state(next_state):
            return next_state
        return state
    
    def _analyze_convergence(self, convergence_data):
        iterations, deltas, changes = zip(*convergence_data)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(iterations, deltas)
        plt.title('Value Function Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Delta')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        plt.plot(iterations, changes)
        plt.title('Policy Stability')
        plt.xlabel('Iteration')
        plt.ylabel('Fraction of Policy Changes')
        plt.show()
        
        print("\nConvergence Analysis:")
        print(f"Total iterations: {len(iterations)}")
        print(f"Final delta: {deltas[-1]:.8f}")
        print(f"Final policy change rate: {changes[-1]:.4%}")
        
    def get_optimal_path(self):
        path = []
        current_state = self.env.start
        visited = set()
        
        while current_state != self.env.goal:
            if current_state in visited:
                print("Warning: Cycle detected in policy path")
                return None
                
            path.append(current_state)
            visited.add(current_state)
            action = self.actions[self.policy[current_state[0]][current_state[1]]]
            current_state = self.get_next_state(current_state, action)
            
            if len(path) > self.size * self.size:
                print("Warning: Path exceeds maximum possible length")
                return None
        
        path.append(self.env.goal)
        return path

def visualize_grid_with_policy(env, agent, path=None, save_prefix=None):
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 3, 1)
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
    plt.title('Optimal Policy with Path')
    
    plt.subplot(1, 3, 2)
    changes = np.zeros((env.size, env.size))
    for (i, j), count in agent.policy_changes.items():
        changes[i][j] = count
    sns.heatmap(changes, cmap='YlOrRd', annot=False)
    plt.title('Policy Change Frequency')
  
    plt.subplot(1, 3, 3)
    sns.heatmap(agent.values, cmap='viridis')
    plt.title('Final Value Function')
    
    if save_prefix:
        plt.savefig(f'{save_prefix}_analysis.png')
    plt.show()

def write_analysis_to_file(env, agent, iterations, path, filename='mdp_analysis_output_v2.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        
        f.write("=" * 80 + "\n")
        f.write("MARKOV DECISION PROCESS (MDP) ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. ENVIRONMENT CONFIGURATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Grid Size: {env.size}x{env.size} ({env.size * env.size} total states)\n")
        f.write(f"Start Position: {env.start}\n")
        f.write(f"Goal Position: {env.goal}\n")
        obstacle_count = np.sum(env.grid == 1)
        f.write(f"Obstacle Count: {obstacle_count}\n")
        f.write(f"Obstacle Density: {obstacle_count/(env.size * env.size):.2%}\n\n")

        
        f.write("2. ALGORITHM PARAMETERS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Discount Factor (gamma): {agent.gamma}\n")
        f.write(f"Convergence Threshold (theta): {agent.theta}\n")
        f.write("Available Actions: Right (→), Down (↓), Left (←), Up (↑)\n\n")

        
        f.write("3. CONVERGENCE ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Iterations until Convergence: {iterations}\n")
        
        
        final_changes = sum(1 for i in range(env.size) for j in range(env.size) 
                          if (i,j) in agent.policy_changes)
        f.write(f"States with Policy Changes: {final_changes}\n")
        f.write(f"Average Changes per State: {sum(agent.policy_changes.values())/len(agent.policy_changes):.2f}\n\n")

        
        f.write("4. OPTIMAL PATH ANALYSIS\n")
        f.write("-" * 30 + "\n")
        if path:
            f.write(f"Path Found: Yes\n")
            f.write(f"Path Length: {len(path)}\n")
            manhattan_distance = abs(env.start[0] - env.goal[0]) + abs(env.start[1] - env.goal[1])
            f.write(f"Manhattan Distance (Start to Goal): {manhattan_distance}\n")
            f.write(f"Path Efficiency Ratio: {manhattan_distance/len(path):.2%}\n")
            
            f.write("\nPath Coordinates:\n")
            for idx, pos in enumerate(path):
                f.write(f"Step {idx}: {pos}\n")
        else:
            f.write("No valid path found - possible disconnected environment\n\n")

        
        f.write("\n5. VALUE FUNCTION ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Maximum Value: {np.max(agent.values):.2f}\n")
        f.write(f"Minimum Value: {np.min(agent.values):.2f}\n")
        f.write(f"Average Value: {np.mean(agent.values):.2f}\n")
        f.write(f"Value Standard Deviation: {np.std(agent.values):.2f}\n\n")

      
        f.write("6. POLICY STATISTICS\n")
        f.write("-" * 30 + "\n")
        action_counts = defaultdict(int)
        for i in range(env.size):
            for j in range(env.size):
                if env.grid[i][j] != 1:  # Skip obstacles
                    action_counts[agent.action_names[agent.policy[i][j]]] += 1
        
        total_valid_states = sum(action_counts.values())
        f.write("Action Distribution in Final Policy:\n")
        for action, count in action_counts.items():
            f.write(f"{action}: {count} states ({count/total_valid_states:.2%})\n")

       
        f.write("\n7. MOST VOLATILE STATES\n")
        f.write("-" * 30 + "\n")
        f.write("States with Most Policy Changes:\n")
        volatile_states = sorted(agent.policy_changes.items(), key=lambda x: x[1], reverse=True)[:10]
        for state, changes in volatile_states:
            f.write(f"State {state}: Changed {changes} times\n")

     
        f.write("\n8. FINAL POLICY GRID\n")
        f.write("-" * 30 + "\n")
        f.write("Legend: → (Right), ↓ (Down), ← (Left), ↑ (Up), █ (Obstacle)\n\n")
        
        for i in range(env.size):
            for j in range(env.size):
                if env.grid[i][j] == 1:
                    f.write('█ ')
            f.write('\n')

env = GridWorld(size=100, obstacle_density=0.3)
agent = MDPAgent(env)
iterations = agent.value_iteration()
path = agent.get_optimal_path()
write_analysis_to_file(env, agent, iterations, path)
print("done")