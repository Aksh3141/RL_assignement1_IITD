import numpy as np
import itertools
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import mode
import sys
import copy
import time
import random
import os
random.seed(42)     
np.random.seed(42) 
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv

def build_state_space(env):
    steps = env.step_limit
    prices = env.asset_price_means.flatten()
    max_price = prices.max()
    max_cash = int(env.initial_cash + steps * (max_price + 2))
    states = []
    for t in range(steps+1):
        for cash in range(0, max_cash+1):
            for hold in range(0, env.holding_limit[0]+1):
                states.append((t, cash, hold))
    return states

def transition(env, state, action):
    t, cash, hold = state
    price = env.asset_price_means[0, t]
    if action < 0:  # sell
        a = min(abs(action), hold)
        cash_new = cash + (price - 1) * a
        hold_new = hold - a
    elif action > 0:  # buy
        a = action
        cost = (price + 1) * a
        if cash >= cost and hold + a <= env.holding_limit[0]:
            cash_new = cash - cost
            hold_new = hold + a
        else:
            cash_new, hold_new = cash, hold
    else:
        cash_new, hold_new = cash, hold

    t_new = t + 1
    done = (t_new == env.step_limit)
    reward = 0
    if done:
        reward = cash_new + hold_new * env.asset_price_means[0, t_new-1]
    return (t_new, cash_new, hold_new), reward, done

def value_iteration(env, gamma=0.999, tol=1e-4, max_iter=500):
    states = build_state_space(env)
    V = {s: 0 for s in states}
    policy = {s: 0 for s in states}
    start_time = time.time()
    rewards_trace = []

    for it in range(max_iter):
        delta = 0
        new_V = V.copy()
        for s in states:
            t, _, _ = s
            if t == env.step_limit: continue
            best_val, best_a = -np.inf, 0
            for a in [-2, -1, 0, 1, 2]:
                s2, r, done = transition(env, s, a)
                if s2 not in V:
                    V[s2] = 0
                    new_V[s2] = 0
                    policy[s2] = 0
                val = r + (0 if done else gamma * V[s2])
                if val > best_val:
                    best_val, best_a = val, a
            new_V[s] = best_val
            policy[s] = best_a
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        rewards_trace.append(V[(0, env.initial_cash, 0)])
        if delta < tol:
            break
    exec_time = time.time() - start_time
    return V, policy, rewards_trace, exec_time

def policy_iteration(env, gamma=0.999, tol=1e-4, max_iter=100):
    states = build_state_space(env)
    policy = {s: np.random.choice([-2,-1,0,1,2]) for s in states if s[0] < env.step_limit}
    V = {s: 0 for s in states}
    start_time = time.time()
    rewards_trace = []
    it = 0
    while it < max_iter:
        # policy evaluation
        while True:
            delta = 0
            new_V = V.copy()
            for s in states:
                t, _, _ = s
                if t == env.step_limit: continue
                a = policy[s]
                s2, r, done = transition(env, s, a)
                if s2 not in V:
                    V[s2] = 0
                    new_V[s2] = 0
                    policy[s2] = 0
                new_V[s] = r + (0 if done else gamma * V[s2])
                delta = max(delta, abs(new_V[s] - V[s]))
            V = new_V
            if delta < tol:
                break
        # policy improvement
        stable = True
        for s in states:
            t, _, _ = s
            if t == env.step_limit: continue
            old_a = policy[s]
            best_val, best_a = -np.inf, old_a
            for a in [-2,-1,0,1,2]:
                s2, r, done = transition(env, s, a)
                if s2 not in V:
                    V[s2] = 0
                    new_V[s2] = 0
                    policy[s2] = 0
                val = r + (0 if done else gamma * V[s2])
                if val > best_val:
                    best_val, best_a = val, a
            policy[s] = best_a
            if best_a != old_a:
                stable = False
        rewards_trace.append(V[(0, env.initial_cash, 0)])
        if stable:
            break
        it += 1
    exec_time = time.time() - start_time
    return V, policy, rewards_trace, exec_time

# ================= Episode Runner =================

def run_episode(env, policy):
    state = (0, env.initial_cash, 0)
    wealth_hist, cash_hist, hold_hist = [], [], []
    for t in range(env.step_limit):
        cash, price, hold = state[1], env.asset_price_means[0, t], state[2]
        wealth = cash + hold*price
        wealth_hist.append(wealth)
        cash_hist.append(cash)
        hold_hist.append(hold)
        a = policy[state]
        state, _, done = transition(env, state, a)
        if done: break
    return wealth_hist, cash_hist, hold_hist

# ================= Main Script =================


configs = {
    "config1": [1, 3, 5, 5 , 4, 3, 2, 3, 5, 8],
    "config2": [2, 2, 2, 4 ,2, 2, 4, 2, 2, 2],
    "config3": [4, 1, 4, 1 ,4, 4, 4, 1, 1, 4]
}

# make output folder
os.makedirs("plots_finance", exist_ok=True)
gamma = 0.999
for name, prices in configs.items():
    print(f"\n===== Running {name} (gamma={gamma}) =====")
    env = DiscretePortfolioOptEnv(prices=prices)

    # ----- Value Iteration -----
    V, pi, trace, t_exec = value_iteration(env, gamma=gamma)
    print(f"{name} VI time (gamma={gamma}): {t_exec:.4f}s")
    plt.figure()
    plt.plot(trace)
    plt.title(f"{name} - VI Convergence (γ={gamma})")
    plt.xlabel("Iteration"); plt.ylabel("Value (start state)")
    plt.savefig(f"plots_finance/{name}_vi_convergence_gamma{gamma}.png")
    plt.close()

    wealth, cash, hold = run_episode(env, pi)
    plt.figure()
    plt.plot(wealth, label="Wealth")
    plt.plot(cash, label="Cash")
    plt.plot(hold, label="Holdings")
    plt.title(f"{name} - Episode Evolution (VI, γ={gamma})")
    plt.xlabel("Step"); plt.legend()
    plt.savefig(f"plots_finance/{name}_vi_episode_gamma{gamma}.png")
    plt.close()

    # ----- Policy Iteration -----
    Vpi, pipi, trace_pi, t_exec_pi = policy_iteration(env, gamma=gamma)
    print(f"{name} PI time (gamma={gamma}): {t_exec_pi:.4f}s")
    plt.figure()
    plt.plot(trace_pi)
    plt.title(f"{name} - PI Convergence (γ={gamma})")
    plt.xlabel("Iteration"); plt.ylabel("Value (start state)")
    plt.savefig(f"plots_finance/{name}_pi_convergence_gamma{gamma}.png")
    plt.close()

    wealth, cash, hold = run_episode(env, pipi)
    plt.figure()
    plt.plot(wealth, label="Wealth")
    plt.plot(cash, label="Cash")
    plt.plot(hold, label="Holdings")
    plt.title(f"{name} - Episode Evolution (PI, γ={gamma})")
    plt.xlabel("Step"); plt.legend()
    plt.savefig(f"plots_finance/{name}_pi_episode_gamma{gamma}.png")
    plt.close()

# if __name__=="__main__":
#     start_time=time.time()


#     ###Part 1 and Part 2
#     ####Please train the value and policy iteration training algo for the given three sequences of prices
#     ####Config1
#     env = DiscretePortfolioOptEnv(prices=[1, 3, 5, 5 , 4, 3, 2, 3, 5, 8])

#     ####Config2
#     env = DiscretePortfolioOptEnv(prices=[2, 2, 2, 4 ,2, 2, 4, 2, 2, 2])

#     ####Config3
#     env = DiscretePortfolioOptEnv(prices=[4, 1, 4, 1 ,4, 4, 4, 1, 1, 4])



#     ####Run the evaluation on the following prices and save the plots.



#     ###Part 3. (Portfolio Optimizaton)
#     env = DiscretePortfolioOptEnv(variance=1)
