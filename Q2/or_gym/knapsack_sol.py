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

# def transition_and_reward(env,cur_wt,itm_wt,itm_val,action):
#     max_wt = env.max_weight
#     current_wt_knapsack = cur_wt
#     current_item_wt = itm_wt
#     current_item_val = itm_val
#     reward = 0
#     done = False
#     if(action==1):
#         if(current_wt_knapsack+current_item_wt<=max_wt):
#             current_wt_knapsack+=current_item_wt
#             reward = current_item_val
#         else:
#             done = True
#     return current_wt_knapsack,done,reward


class PolicyIterationOnlineKnapsack:
    def __init__(self, env, time_step, transitions, gamma=0.95, epsilon=1e-4):
        self.env = env
        self.time_step = time_step
        self.gamma = gamma
        self.epsilon = epsilon
        self.transitions = transitions

    def policy_evaluation(self, policy, max_time_steps=50, max_eval_iterations=10):
        V = np.zeros((self.env.max_weight+1, self.env.N, max_time_steps+1))

        for _ in range(max_eval_iterations):
            delta = 0

            for w in range(self.env.max_weight+1):
                for itm_idx in range(self.env.N):
                    for t in range(max_time_steps+1):
                        act = policy[w, itm_idx, t]
                        mdp_state, done, reward = self.transitions[w, itm_idx, act]
                        if done or t == max_time_steps:
                            val_est = 0
                        else:
                            val_est = reward + self.gamma * np.mean(V[mdp_state,:, t+1])

                        diff = abs(V[w, itm_idx, t] - val_est)
                        if diff > delta:
                            delta = diff
                        V[w, itm_idx, t] = val_est

            if delta < self.epsilon:
                break
        return V

    def policy_improvement(self, V, max_time_steps=50, old_policy=None):
        policy_stable = True
        W, N = self.env.max_weight, self.env.N

        if old_policy is None:
            policy = np.zeros((W, N, max_time_steps+1), dtype=int)
        else:
            policy = old_policy.copy()

        for w in range(W+1):
            for itm_idx in range(N):
                for t in range(max_time_steps+1):
                    old_action = policy[w, itm_idx, t]
                    best_val, best_action = float("-inf"), old_action

                    for act in (0, 1):
                        mdp_state, done, reward = self.transitions[w, itm_idx, act]
                        if done or t == max_time_steps:
                            val_est = 0
                        else:
                            val_est = reward + self.gamma * np.mean(V[mdp_state,:, t+1])

                        if val_est > best_val:
                            best_val, best_action = val_est, act

                    policy[w, itm_idx, t] = best_action
                    if best_action != old_action:
                        policy_stable = False
        return policy, policy_stable

    def policy_iteration(self, max_time_steps=50, eval_iterations=1000):
        W, N = self.env.max_weight + 1, self.env.N
        policy = np.zeros((W, N, max_time_steps+1), dtype=int)

        iteration = 0
        while True:
            print(f"Iteration {iteration}")
            V = self.policy_evaluation(policy, max_time_steps, max_eval_iterations=eval_iterations)
            policy, stable = self.policy_improvement(V, max_time_steps, policy)

            if stable:
                print("Policy converged.")
                break
            iteration += 1

        return V, policy

class ValueIterationOnlineKnapsack:
    def __init__(self, env, time_step,transitions, gamma=0.95, epsilon=1e-4):
        self.env = env
        self.time_step = time_step
        self.gamma = gamma
        self.epsilon = epsilon
        self.transitions = transitions

    def value_iteration(self, max_time_steps=50):
        V = np.zeros((self.env.max_weight+1, self.env.N,max_time_steps+1))
        policy = np.zeros_like(V, dtype=int)
        max_iterations = 1000
        for iteration in range(max_iterations):
            delta = 0
            print("Iteration:",iteration)
            for i in range(self.env.max_weight+1):
                for itm_idx in range(self.env.N):
                    for time in range(max_time_steps+1): 
                        max_val = float('-inf')
                        best_action = 0
                        for act in [0,1]:
                            mdp_state, done,reward= self.transitions[i, itm_idx, act]
                            if done or time == max_time_steps:
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

def run_policy(env, policy, time_steps=50):
    obs = env._RESET()
    state = obs['state']
    rewards_trace = []
    total_reward = 0
    
    for t in range(time_steps):
        current_weight = state[0]
        current_item_idx = state[1]
        
        action = policy[current_weight, current_item_idx, t]
        obs, reward, done, info = env._STEP(action)
        
        state = obs['state']
        total_reward += reward
        rewards_trace.append(total_reward)
        
        if done:
            break
    
    return rewards_trace, total_reward

# --- Updated plotting: compare VI vs PI ---
def plot_results(all_traces_VI, all_traces_PI, V_VI, V_PI, env, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Line plot VI ---
    plt.figure(figsize=(10,6))
    for seed, trace in all_traces_VI.items():
        plt.plot(range(1, len(trace)+1), trace, label=f"VI Seed {seed}")
    plt.xlabel("Item index (time step)")
    plt.ylabel("Cumulative knapsack value")
    plt.title("Value Iteration policy evaluation (5 seeds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "knapsack_value_traces_VI.png"))
    plt.close()

    # --- 2. Line plot PI ---
    plt.figure(figsize=(10,6))
    for seed, trace in all_traces_PI.items():
        plt.plot(range(1, len(trace)+1), trace, label=f"PI Seed {seed}")
    plt.xlabel("Item index (time step)")
    plt.ylabel("Cumulative knapsack value")
    plt.title("Policy Iteration policy evaluation (5 seeds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "knapsack_value_traces_PI.png"))
    plt.close()

    # --- 3. Heatmaps VI + PI ---
    def make_heatmaps(V, label):
        item_weights = env.item_weights
        item_values = env.item_values
        item_ratios = item_weights / (item_values + 1e-6)

        sortings = {
            "Sorted by weight": np.argsort(item_weights),
            "Sorted by value": np.argsort(item_values),
            "Sorted by ratio": np.argsort(item_ratios),
        }

        for title, idx in sortings.items():
            heatmap = np.zeros((env.max_weight+1, len(idx)))
            for j, item_idx in enumerate(idx):
                heatmap[:, j] = V[:, item_idx, 0]

            plt.figure(figsize=(12,6))
            plt.imshow(heatmap, aspect="auto", cmap="viridis", origin="lower")
            plt.colorbar(label="Value function")
            plt.xlabel("Items (" + title + ")")
            plt.ylabel("Current knapsack weight")
            plt.title(f"{label} - Value Function Heatmap ({title})")
            filename = f"value_function_heatmap_{label}_{title.replace(' ', '_').lower()}.png"
            plt.savefig(os.path.join(save_dir, filename))
            plt.close()

    make_heatmaps(V_VI, "VI")
    make_heatmaps(V_PI, "PI")

# --- Main ---
if __name__=="__main__":
    seeds = [0, 1, 2, 3, 4]

    # Train VI once
    env = OnlineKnapsackEnv(seed=42)
    transitions = precompute_transitions(env)
    solver_VI = ValueIterationOnlineKnapsack(env, 50, transitions)
    print("Bachao")
    V_VI, policy_VI = solver_VI.value_iteration()

    # Train PI once
    solver_PI = PolicyIterationOnlineKnapsack(env, 50, transitions)
    V_PI, policy_PI = solver_PI.policy_iteration()

    # Evaluate both policies on 5 seeds
    all_traces_VI, all_traces_PI = {}, {}
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        trace_VI, total_VI = run_policy(env, policy_VI, 50)
        trace_PI, total_PI = run_policy(env, policy_PI, 50)

        all_traces_VI[seed] = trace_VI
        all_traces_PI[seed] = trace_PI

        print(f"Seed {seed} -> VI reward: {total_VI}, PI reward: {total_PI}")

    # Plots
    plot_results(all_traces_VI, all_traces_PI, V_VI, V_PI, env)



