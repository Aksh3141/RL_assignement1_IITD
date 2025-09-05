import numpy as np
import matplotlib.pyplot as plt
import time
import os
from math import sqrt, pi, exp
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv

# ---------------- Gaussian support builder -----------------
def gaussian_pmf(mu, sigma, grid_width=4):
    lo = max(0, int(np.floor(mu - grid_width * sigma)))
    hi = int(np.ceil(mu + grid_width * sigma))
    values = np.arange(lo, hi + 1, dtype=int)
    probs = np.exp(-0.5 * ((values - mu) / sigma) ** 2) / (sigma * sqrt(2 * pi))
    if probs.sum() == 0:
        probs = np.ones_like(probs, dtype=float)
    return values, probs / probs.sum()

# ---------------- Build state space ------------------------
def build_state_space_with_price(env, sigma=1.0, grid_width=100, cash_cap_mult=3):
    T = env.step_limit
    means = env.asset_price_means.flatten()
    price_supports = []
    max_price_val = 0
    for t in range(T):
        vals, probs = gaussian_pmf(means[t], sigma, grid_width=grid_width)
        price_supports.append((vals.astype(int), probs.astype(float)))
        max_price_val = max(max_price_val, int(vals.max()))
    max_cash = int(env.initial_cash + env.step_limit * (max_price_val + env.buy_cost[0] + 2))
    max_cash = min(max_cash, int(env.initial_cash * cash_cap_mult + env.step_limit * (max_price_val + 10)))
    cash_values = np.arange(0, max_cash + 1, dtype=int)
    hold_values = np.arange(0, env.holding_limit[0] + 1, dtype=int)

    states = []
    for t in range(T):
        vals, _ = price_supports[t]
        for cash in cash_values:
            for hold in hold_values:
                for p in vals:
                    states.append((t, int(cash), int(hold), int(p)))
    return states, price_supports, cash_values, hold_values, max_cash

# ---------------- Env-like action --------------------------
def apply_action_env_like(cash, price, hold, action, env):
    a = int(action)
    cash = float(cash); hold = int(hold); price = int(price)
    if a == 0:
        return int(cash), int(hold), True
    if a < 0:  # sell
        units = min(abs(a), hold)
        cash_new = cash + (price - env.sell_cost[0]) * units
        hold_new = hold - units
        if cash_new < 0: return None, None, False
        return int(cash_new), int(hold_new), True
    # buy
    units = a
    if hold + units > env.holding_limit[0]:
        return int(cash), int(hold), True
    purchase_cost = (price + env.buy_cost[0]) * units
    if cash < purchase_cost:
        return int(cash), int(hold), True
    cash_new = cash - purchase_cost
    hold_new = hold + units
    if cash_new < 0: return None, None, False
    return int(cash_new), int(hold_new), True

# ---------------- Expectation helper -----------------------
def expected_value_next(V, t_next, cash1, hold1, price_support_next, max_cash, env):
    vals_next, probs_next = price_support_next
    exp = 0.0
    c1_clamped = min(int(cash1), int(max_cash))
    for p_next, pprob in zip(vals_next, probs_next):
        exp += pprob * V.get((t_next, c1_clamped, hold1, int(p_next)), 0.0)
    return exp

# ---------------- Policy Iteration -------------------------
def policy_iteration_with_price(env, sigma=1.0, grid_width=10000, gamma=1, tol=1e-4, max_iter=1000):
    states, price_supports, _, _, max_cash = build_state_space_with_price(env, sigma, grid_width)
    np.random.seed(0)
    policy = {s: np.random.choice([-2, -1, 0, 1, 2]) for s in states}
    V = {s: 0.0 for s in states}
    actions = [-2, -1, 0, 1, 2]
    T = env.step_limit
    deltas = []
    it = 0
    start_time = time.time()

    while it < max_iter:
        while True:
            delta = 0.0
            newV = {}
            for s in states:
                t, cash, hold, price = s
                a = policy[s]
                c1, h1, valid = apply_action_env_like(cash, price, hold, a, env)
                if not valid or c1 is None:
                    v_new = -1e9
                else:
                    t_next = t + 1
                    if t_next == T:
                        v_new = c1 + h1 * price
                    else:
                        v_new = gamma * expected_value_next(V, t_next, c1, h1, price_supports[t_next], max_cash, env)
                newV[s] = v_new
                delta = max(delta, abs(v_new - V.get(s, 0.0)))
            V = newV
            deltas.append(delta)
            if delta < tol:  # eval converged
                break
        print("Iteration:", it)
          # log max value diff this outer iteration

        # Policy improvement
        stable = True
        for s in states:
            t, cash, hold, price = s
            old_a = policy[s]
            best_val = -1e18; best_a = old_a
            for a in actions:
                c1, h1, valid = apply_action_env_like(cash, price, hold, a, env)
                if not valid or c1 is None:
                    val = -1e9
                else:
                    t_next = t + 1
                    if t_next == T:
                        val = c1 + h1 * price
                    else:
                        val = gamma * expected_value_next(V, t_next, c1, h1, price_supports[t_next], max_cash, env)
                if val > best_val:
                    best_val, best_a = val, a
            policy[s] = best_a
            if best_a != old_a:
                stable = False

        it += 1
        if stable: break

    exec_time = time.time() - start_time
    return V, policy, deltas, exec_time, price_supports, max_cash

def run_policy(env, policy, price_supports, max_cash, sigma=1.0):
    """
    Rollout of learned policy from (t=0, initial_cash, hold=0, price = median of support).
    Returns: history dict with lists for cash, holdings, wealth.
    """
    T = env.step_limit
    cash = env.initial_cash
    hold = 0
    vals0, probs0 = price_supports[0]
    price = int(vals0[len(vals0)//2])  # pick median of support at t=0

    cash_hist, hold_hist, wealth_hist, price_hist = [], [], [], []

    for t in range(T):
        state = (t, int(cash), int(hold), int(price))
        action = policy.get(state, 0)

        # apply action
        c1, h1, valid = apply_action_env_like(cash, price, hold, action, env)
        if not valid or c1 is None:
            c1, h1 = cash, hold  # no-op

        # sample next price from Gaussian support (to simulate stochasticity)
        vals, probs = price_supports[t]
        price = np.random.choice(vals, p=probs)

        cash, hold = c1, h1
        wealth = cash + hold * price

        cash_hist.append(cash)
        hold_hist.append(hold)
        wealth_hist.append(wealth)
        price_hist.append(price)

    return {
        "cash": cash_hist,
        "hold": hold_hist,
        "wealth": wealth_hist,
        "price": price_hist
    }
# ---------------- Run -------------------------
if __name__ == "__main__":
    os.makedirs("plots_finance2", exist_ok=True)
    gamma = 0.999; sigma = 1.0; grid_width = 4
    env = DiscretePortfolioOptEnv(variance=sigma**2)

    V_pi, pi_pi, deltas, t_pi, price_supports, max_cash = policy_iteration_with_price(
    env, sigma=sigma, grid_width=grid_width, gamma=gamma, tol=1e-4, max_iter=1000
)

    print(f"PI time: {t_pi:.4f}s, iterations: {len(deltas)}")
    plt.figure()
    plt.plot(range(1, len(deltas)+1), deltas, marker="o")
    #plt.yscale("log")  # log-scale for clarity
    plt.title(f"Policy Iteration Convergence (γ={gamma})")
    plt.xlabel("Iteration")
    plt.ylabel("Max Value Difference")
    plt.grid(True)
    plt.savefig(f"plots_finance2/default_pi_delta_gamma{gamma}_sigma{sigma}.png")
    plt.close()

    print("Done. Plot saved in ./plots_finance2")

    history = run_policy(env, pi_pi, price_supports, max_cash, sigma=sigma)

    plt.figure(figsize=(8,5))
    plt.plot(history["wealth"], label="Wealth", linewidth=2)
    plt.plot(history["cash"], label="Cash", linestyle="--")
    plt.plot(history["hold"], label="Holdings", linestyle="-.")
    plt.title("Learned Policy Rollout")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots_finance2/policy_rollout.png", dpi=300)
    plt.close()
