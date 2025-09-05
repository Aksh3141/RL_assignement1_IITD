import numpy as np
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv
import matplotlib.pyplot as plt
from scipy.stats import mode
import random as random
import os

def precompute_transitions(env):
    transitions = np.zeros((env.max_weight+1, env.N, 2, 3), dtype=int)
    # shape: [cur_wt, item, action, (mdp_state, done, reward)]
    for w in range(env.max_weight+1):
        for itm_idx in range(env.N):
            for act in [0, 1]:
                if act == 1 and w + env.item_weights[itm_idx] <= env.max_weight:
                    transitions[w, itm_idx, act] = (w + env.item_weights[itm_idx], 0, env.item_values[itm_idx])
                elif act == 1:
                    transitions[w, itm_idx, act] = (w, 1, 0)  # done = 1
                else:
                    transitions[w, itm_idx, act] = (w, 0, 0)
    return transitions

class ValueIterationOnlineKnapsack:
    def __init__(self, env, time_step,transitions, gamma=0.95, epsilon=1e-4):
        self.env = env
        self.time_step = time_step
        self.gamma = gamma
        self.epsilon = epsilon
        self.transitions = transitions

    def value_iteration(self):
        V = np.zeros((self.env.max_weight+1, self.env.N,self.time_step+1))
        policy = np.zeros_like(V, dtype=int)
        max_iterations = 1000
        for iteration in range(max_iterations):
            delta = 0
            print("Iteration:",iteration)
            for i in range(self.env.max_weight):
                for itm_idx in range(self.env.N):
                    for time in range(self.time_step): 
                        max_val = float('-inf')
                        best_action = 0
                        for act in [0,1]:
                            mdp_state, done,reward= self.transitions[i, itm_idx, act]
                            if done or time == self.time_step:
                                val_est = 0
                            else:
                                val_est = (reward + self.gamma * np.mean(V[mdp_state,:,time+1]))
                            if val_est > max_val:
                                max_val = val_est
                                best_action = act
                        delta = max(delta, abs(V[i, itm_idx,time] - max_val))
                        V[i, itm_idx, time] = max_val
                        policy[i, itm_idx,time] = best_action
            if delta < self.epsilon:
                break

        return V, policy


def plot_heatmap(V, env, sorted_idx, xlabel, title, filename):
    weights = range(env.max_weight + 1)
    values = np.zeros((len(weights), env.N))
    
    for w in weights:
        for j, i in enumerate(sorted_idx):
            state = (w, i, 0)  # time step = 0
            cur_w, item_idx, t = state   # unpack the tuple
            values[w, j] = V[cur_w, item_idx, t]
    
    plt.figure(figsize=(8,6))
    plt.imshow(values, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="Value")
    plt.xlabel(xlabel)
    plt.ylabel("Current Knapsack Weight")
    plt.title(title)

    # Ensure folder exists
    os.makedirs("plots2", exist_ok=True)
    plt.savefig(os.path.join("plots2", filename), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    steps_list = [10,50,500]   # different VI steps
    seed = 42                 
    env = OnlineKnapsackEnv(seed=seed)
    transitions = precompute_transitions(env)

    for steps in steps_list:
        print(f"\nRunning Value Iteration with {steps} steps...")
        solver_VI = ValueIterationOnlineKnapsack(env, steps, transitions)
        V, policy = solver_VI.value_iteration()

        item_weights = np.array(env.item_weights)
        item_values = np.array(env.item_values)
        ratios = item_values / item_weights

        # Prepare sorted indices
        idx_w = np.argsort(item_weights)
        idx_v = np.argsort(item_values)
        idx_r = np.argsort(ratios)

        # Save heatmaps
        plot_heatmap(V, env, idx_w, "Item Weights (sorted)",
                     f"Value Function Heatmap (steps={steps})",
                     f"heatmap_weights_steps{steps}.png")

        plot_heatmap(V, env, idx_v, "Item Values (sorted)",
                     f"Value Function Heatmap (steps={steps})",
                     f"heatmap_values_steps{steps}.png")

        plot_heatmap(V, env, idx_r, "Weight-to-Value Ratio (sorted)",
                     f"Value Function Heatmap (steps={steps})",
                     f"heatmap_ratio_steps{steps}.png")