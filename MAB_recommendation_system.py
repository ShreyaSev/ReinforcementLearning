##Import Packages
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from typing import List, Tuple
import seaborn as sns

class Arm:
    def __init__(self, true_mean: float, true_std: float):
        self.true_mean = true_mean
        self.true_std = true_std
        self.pulls = 0
        self.total_reward = 0
        self.mean_reward = 0
        
    def pull(self) -> float:
        reward = np.random.normal(self.true_mean, self.true_std)
        self.pulls += 1
        self.total_reward += reward
        self.mean_reward = self.total_reward / self.pulls
        return reward

class MultiArmedBandit:
    def __init__(self, n_arms: int, method: str, epsilon: float = 0.1, temperature: float = 1.0):
        self.arms = [Arm(np.random.normal(0, 1), 0.5) for _ in range(n_arms)]
        self.n_arms = n_arms
        self.method = method
        self.epsilon = epsilon
        self.temperature = temperature
        self.total_reward = 0
        self.actions_history = []
        self.rewards_history = []
        self.cumulative_rewards = []
        
    def select_arm(self) -> int:
        if self.method == "epsilon-greedy":
            return self._epsilon_greedy()
        elif self.method == "ucb":
            return self._ucb()
        elif self.method == "thompson":
            return self._thompson_sampling()
        elif self.method == "softmax":
            return self._softmax()
        else:
            raise ValueError("Unknown method")
    
    def _epsilon_greedy(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax([arm.mean_reward for arm in self.arms])
    
    def _ucb(self) -> int:
        total_pulls = sum(arm.pulls for arm in self.arms)
        if total_pulls < self.n_arms:
            return total_pulls
        
        ucb_values = []
        for arm in self.arms:
            if arm.pulls == 0:
                return self.arms.index(arm)
            
            exploration = np.sqrt(2 * np.log(total_pulls) / arm.pulls)
            ucb = arm.mean_reward + exploration
            ucb_values.append(ucb)
            
        return np.argmax(ucb_values)
    
    def _thompson_sampling(self) -> int:
        samples = []
        for arm in self.arms:
            if arm.pulls == 0:
                samples.append(np.random.normal(0, 1))
            else:
                posterior_std = 1 / np.sqrt(arm.pulls)
                samples.append(np.random.normal(arm.mean_reward, posterior_std))
        return np.argmax(samples)
    
    def _softmax(self) -> int:
        probabilities = np.array([arm.mean_reward for arm in self.arms])
        probabilities = np.exp(probabilities / self.temperature)
        probabilities = probabilities / np.sum(probabilities)
        return np.random.choice(self.n_arms, p=probabilities)
    
    def run(self, n_iterations: int) -> None:
        for _ in range(n_iterations):
            chosen_arm = self.select_arm()
            reward = self.arms[chosen_arm].pull()
            
            self.total_reward += reward
            self.actions_history.append(chosen_arm)
            self.rewards_history.append(reward)
            self.cumulative_rewards.append(self.total_reward)
    
    def save_results(self, filename: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{filename}_{timestamp}.txt", "w") as f:
            f.write(f"Multi-Armed Bandit Results - {self.method}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Parameters:\n")
            f.write(f"Number of arms: {self.n_arms}\n")
            f.write(f"Method: {self.method}\n")
            if self.method == "epsilon-greedy":
                f.write(f"Epsilon: {self.epsilon}\n")
            elif self.method == "softmax":
                f.write(f"Temperature: {self.temperature}\n")
            
            f.write("\nArm Statistics:\n")
            for i, arm in enumerate(self.arms):
                f.write(f"\nArm {i}:\n")
                f.write(f"True mean: {arm.true_mean:.4f}\n")
                f.write(f"Estimated mean: {arm.mean_reward:.4f}\n")
                f.write(f"Number of pulls: {arm.pulls}\n")
            
            f.write(f"\nTotal reward: {self.total_reward:.4f}\n")
            f.write(f"Average reward: {self.total_reward/len(self.rewards_history):.4f}\n")
    
    def plot_results(self, save_path: str = None):
       
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        ax1.plot(self.cumulative_rewards)
        ax1.set_title(f'Cumulative Rewards Over Time ({self.method})')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cumulative Reward')
      
        arm_counts = np.bincount(self.actions_history)
        ax2.bar(range(self.n_arms), arm_counts)
        ax2.set_title('Arm Selection Frequency')
        ax2.set_xlabel('Arm')
        ax2.set_ylabel('Number of Pulls')
        
        running_avg = pd.Series(self.rewards_history).rolling(window=100).mean()
        ax3.plot(running_avg)
        ax3.set_title('Running Average Reward (window=100)')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Average Reward')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

def benchmark_methods(n_arms: int, n_iterations: int) -> None:
    methods = {
        "epsilon-greedy": {"epsilon": 0.1},
        "ucb": {},
        "thompson": {},
        "softmax": {"temperature": 0.1}
    }
    
    results = []
    
    for method, params in methods.items():
        bandit = MultiArmedBandit(n_arms=n_arms, method=method, **params)
        bandit.run(n_iterations)
       
        bandit.save_results(f"results_{method}")
        bandit.plot_results(f"plots_{method}.png")
        
        results.append({
            'method': method,
            'total_reward': bandit.total_reward,
            'average_reward': bandit.total_reward/n_iterations
        })
    
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='method', y='average_reward', data=results_df)
    plt.title('Comparison of Different Methods')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('method_comparison.png')
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)  # for the reproducibility
    benchmark_methods(n_arms=10, n_iterations=10000)