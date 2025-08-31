from env import FootballSkillsEnv
import random
import numpy as np
def value_iteration_with_degrading_pitch(env,gamma):
    states = env.grid_size * env.grid_size * 2
    action = len(env.action_mapping)
    max_time = 40
    V = {(s, t): 0 for s in range(states) for t in range(max_time)}
    policy = {(s, t): random.randrange(action) for s in range(states) for t in range(max_time)}
    count =0 
    while(True):
        delta =0
        for s in range(states):
            for time in range(max_time):
                old_s = env.index_to_state(s)
                if env._is_terminal(old_s):
                    V[s,time] = 0
                    continue
                max_val = float('-inf')  
                for act in range(action):
                    transitions = env.get_transitions_at_time(old_s,act,time)
                    count +=1
                    value_action = sum(probs*(env._get_reward((real_s[0],real_s[1]),act,(old_s[0],old_s[1])) 
                                    + (gamma * V[env.state_to_index(real_s), time+1] if time+1 < max_time else 0)) 
                                    for (probs,real_s) in transitions)
                    max_val = max(max_val, value_action)
                delta = max(delta, abs(max_val - V[s,time]))
                V[s,time] = max_val

        if delta < 1e-6:
            break

    for s in range(states):
        best_action = None
        old_s= env.index_to_state(s)
        if env._is_terminal(old_s):
            continue
        best_val = float('-inf')
        for time in range(max_time):    
            for act in range(action):
                count +=1
                transitions = env.get_transitions_at_time(old_s,act,time)
                exp_val = sum(probs*(env._get_reward((real_s[0],real_s[1]),act,(old_s[0],old_s[1])) 
                                    + (gamma * V[env.state_to_index(real_s), time+1] if time+1 < max_time else 0)) for (probs,real_s) in transitions)
                if exp_val > best_val:
                    best_val = exp_val
                    best_action = act
            policy[s,time] = best_action
    return policy,count

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
    return policy,total_calls
env= FootballSkillsEnv(render_mode='gif',degrade_pitch=True)    

policy_deg_time,count = value_iteration_with_degrading_pitch(env,gamma = 0.95)
print()
print("Number of transition function calls in VI with time in state: ", count)
policy_normal,total_count = value_iteration(env,gamma=0.95)
print()
print("Number of transition function calls in VI without time in state: ",total_count)

def run_policy(env, policy, seeds, max_steps=100, render_mode='gif',time_dep=True):
    env.render_mode = render_mode
    results = {}
    
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            env.render()
            s_index = env.state_to_index(obs)
            
            if time_dep:
                time_step = min(steps, len(policy) - 1)
                action = policy.get((s_index, time_step), None)
            else:
                action = policy.get(s_index, None)
            
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
results_by_deg_time = run_policy(env, policy_deg_time, seeds)
resutls_by_normal = run_policy(env, policy_normal, seeds,time_dep=False)

def get_mean_std(results):
    rewards = [total_reward for (steps, total_reward) in results.values()]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards, ddof=1) 
    return mean_reward, std_reward

mean_deg_time, std_deg_time= get_mean_std(results_by_deg_time)
mean_normal, std_normal = get_mean_std(resutls_by_normal)

print(f"Value Iteration with degrading pitch: Mean Reward = {mean_deg_time:.2f}, Std Dev = {std_deg_time:.2f}")
print(f"Value Iteration normal: Mean Reward = {mean_normal:.2f}, Std Dev = {std_normal:.2f}")


_, _ = env.get_gif(policy_deg_time, seed=20, filename="output1.gif",)
_, _ = env.get_gif_special(policy_normal, seed=20, filename="output2.gif")





