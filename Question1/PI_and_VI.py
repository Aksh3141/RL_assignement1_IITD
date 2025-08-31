from env import FootballSkillsEnv
import random
import numpy as np


def policy_evaluation(policy,V,env,gamma):
    states = env.grid_size * env.grid_size * 2
    call_count = 0
    while(True):
        delta =0
        for s in range(states):
            old_s = env.index_to_state(s)
            transitions = env.get_transitions_at_time(old_s, policy[s])
            call_count += 1  
            new_value = sum(probs*(env._get_reward((real_s[0],real_s[1]),policy[s],(old_s[0],old_s[1])) 
                                  + gamma*V[env.state_to_index(real_s)])
                                    for (probs,real_s) in transitions)
            delta = max(delta, abs(V[s] - new_value))
            V[s] = new_value
        if delta < 1e-6:
            break
    return V, call_count

def value_iteration(env,gamma):
    states = env.grid_size * env.grid_size * 2
    action = len(env.action_mapping)
    V = {}
    policy = {}
    for i in range(states):
        V[i] = 0
    total_calls = 0
    while(True):
        delta =0
        for s in range(states):
            old_s = env.index_to_state(s)
            if env._is_terminal(old_s):
                V[s] = 0
                continue
            max_val = float('-inf')
            for act in range(action):
                transitions =  env.get_transitions_at_time(old_s,act)
                total_calls += 1 
                value_action = sum(probs*(env._get_reward((real_s[0],real_s[1]),act,(old_s[0],old_s[1])) 
                                  + gamma*V[env.state_to_index(real_s)]) for (probs,real_s) in transitions)
                max_val = max(max_val, value_action)
            delta = max(delta, abs(max_val - V[s]))
            V[s] = max_val

        if delta < 1e-6:
            break

    for s in range(states):
        best_action = None
        old_s= env.index_to_state(s)
        if env._is_terminal(old_s):
            continue
        best_val = float('-inf')   
        for act in range(action):
            transitions = env.get_transitions_at_time(old_s,act)
            total_calls +=1 
            exp_val = sum(probs*(env._get_reward((real_s[0],real_s[1]),act,(old_s[0],old_s[1])) 
                                  + gamma*V[env.state_to_index(real_s)]) for (probs,real_s) in transitions)
            if exp_val > best_val or (exp_val == best_val and act < best_action):
                best_val = exp_val
                best_action = act
        policy[s] = best_action
    return policy,V,total_calls

def policy_iteration(env,gamma):
    '''
    Implements the Policy Iteration algorithm to find the optimal policy for the 
    Football Skills Environment.
    
    Args:
        envr (class, optional): Environment class to instantiate. Defaults to FootballSkillsEnv.
    
    Returns:
        tuple: (optimal_policy, value_function, num_iterations)
            - optimal_policy (dict): Maps state indices to optimal actions
            - value_function (numpy.ndarray): Value of each state under optimal policy  
            - num_iterations (int): Number of iterations until convergence
    
    Algorithm:
    1. Initialize arbitrary policy and value function
    2. Policy Evaluation: Iteratively update value function until convergence
    3. Policy Improvement: Update policy greedily based on current values  
    4. Repeat steps 2-3 until policy converges
    
    Key Environment Methods to Use:
    - env.state_to_index(state_tuple): Converts (x, y, has_shot) tuple to integer index
    - env.index_to_state(index): Converts integer index back to (x, y, has_shot) tuple
    - env.get_transitions_at_time(state, action, time_step=None): Default method for accessing transitions.
    - env._is_terminal(state): Check if state is terminal (has_shot=True)
    - env._get_reward(ball_pos, action, player_pos): Get reward for transition
    - env.reset(seed=None): Reset environment to initial state, returns (observation, info)
    - env.step(action): Execute action, returns (obs, reward, done, truncated, info)
    - env.get_gif(policy, seed=20, filename="output.gif"): Generate GIF visualization 
      of policy execution from given seed
    
    Key Env Variables Notes:
    - env.observation_space.n: Total number of states (use env.grid_size^2 * 2)
    - env.action_space.n: Total number of actions (7 actions: 4 movement + 3 shooting)
    - env.grid_size: Total number of rows in the grid
    '''
    states = env.grid_size * env.grid_size * 2
    action = len(env.action_mapping)
    V = {}
    policy = {}
    for i in range(states):
        #policy[i] = random.randrange(0,action)
        policy[i] = 0
        V[i] = 0
    total_calls = 0
    #policy improvement
    while(True):
        policy_stable = True
        V,call_by_eval = policy_evaluation(policy,V,env,gamma)
        total_calls += call_by_eval
        for s in range(states):
            old_action = policy[s]
            old_s= env.index_to_state(s)
            if env._is_terminal(old_s):
                V[s] = 0
                continue
            action_value = {}
            for act in range(action):
                transitions = env.get_transitions_at_time(old_s,act)
                total_calls += 1
                action_value[act] = sum(probs*(env._get_reward((real_s[0],real_s[1]),act,(old_s[0],old_s[1])) 
                                  + gamma*V[env.state_to_index(real_s)]) for (probs,real_s) in transitions)
            max_value = max(action_value.values())
            best_actions = [act for act, val in action_value.items() if val == max_value]
            new_action = min(best_actions)
            if new_action != old_action:
                policy_stable = False
            policy[s] = new_action
        if policy_stable:
            break
    return policy,V, total_calls     



normal_env = FootballSkillsEnv(render_mode='gif')
policy_by_PI,V_PI,call_by_PI = policy_iteration(normal_env,gamma = 0.95)
policy_by_VI,V_VI,call_by_VI = value_iteration(normal_env, gamma = 0.95)


# Checking if the policies are same at non terminal states
different_states = []
for state_index in policy_by_PI:
    state = normal_env.index_to_state(state_index)
    if normal_env._is_terminal(state):
        continue
    if policy_by_PI[state_index] != policy_by_VI.get(state_index, None):
        different_states.append(state)

if different_states:
    print("States where policy differs (non-terminal states):")
    for state in different_states:
        print(state)
else:
    print("Policies match for all non-terminal states.")


# Total number of calls made by Policy iteration and Value iteration

print("Number of calls made by PI:", call_by_PI)
print("Number of calls made by VI:", call_by_VI)

# checking for value function obtained by PI and VI
different_keys = []
for key in V_PI.keys():
    if abs(V_PI[key] - V_VI.get(key, float('inf'))) > 1e-9:
        different_keys.append(key)

if different_keys:
    print(f"Value functions differ at states: {different_keys}")
else:
    print("Value functions are the same for all states.")

def run_policy(env, policy, seeds, max_steps=100, render_mode='gif'):
    env.render_mode = render_mode
    results = {}
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < max_steps:
            state_index = env.state_to_index(obs)
            action = policy.get(state_index, None)
            if action is None or env._is_terminal(obs):
                break
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                break

        results[seed] = (steps, total_reward)
    return results



seeds = list(range(1,21))  
results_by_PI = run_policy(normal_env, policy_by_PI, seeds)
resutls_by_VI = run_policy(normal_env, policy_by_VI, seeds)

def get_mean_std(results):
    rewards = [total_reward for (steps, total_reward) in results.values()]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards, ddof=1) 
    return mean_reward, std_reward

mean_PI, std_PI = get_mean_std(results_by_PI)
mean_VI, std_VI = get_mean_std(resutls_by_VI)

print(f"Policy Iteration: Mean Reward = {mean_PI:.2f}, Std Dev = {std_PI:.2f}")
print(f"Value Iteration: Mean Reward = {mean_VI:.2f}, Std Dev = {std_VI:.2f}")


#saving GIF for one run
steps_PI, total_reward_PI = normal_env.get_gif(policy_by_PI, seed=20, filename="policy_iteration_output.gif")
steps_VI, total_reward_VI = normal_env.get_gif(policy_by_VI, seed=20, filename="value_iteration_output.gif")


