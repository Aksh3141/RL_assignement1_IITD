
from env import FootballSkillsEnv
import random
import numpy as np

def modified_value_iteration(env, gamma):
    states = env.grid_size * env.grid_size * 2
    action = len(env.action_mapping)
    V = {i: 0 for i in range(states)}
    total_calls = 0
    policy = {}
    # Initialize states_to_update before the loop
    states_to_update = set(range(states))
    
    while True:
        delta = 0
        next_states_to_update = set()
        
        for s in states_to_update:
            old_s = env.index_to_state(s)
            if env._is_terminal(old_s):
                V[s] = 0
                continue
            
            max_val = float('-inf')
            successors_values = []
            for act in range(action):
                transitions = env.get_transitions_at_time(old_s, act)
                total_calls += 1
                value_action = sum(probs * (env._get_reward((real_s[0], real_s[1]), act, (old_s[0], old_s[1]))
                                           + gamma * V[env.state_to_index(real_s)]) for (probs, real_s) in transitions)
                max_val = max(max_val, value_action)
                successors_values.extend([V[env.state_to_index(real_s)] for _, real_s in transitions])
            
            prev_val = V[s]
            V[s] = max_val
            delta = max(delta, abs(max_val - prev_val))
            
            # Check if successors values have changed => if yes, add predecessors (approximately) for the next update
            # Since predecessors are not available, add current state to update next time if its value changed
            if abs(max_val - prev_val) > 1e-6:
                # Add all states that could be successors of this state to next iteration
                # Here approximated by adding all states where this state might be successor
                # Since we do not have predecessors, we add the current state itself
                next_states_to_update.add(s)
        
        if delta < 1e-6:
            break
        
        states_to_update = next_states_to_update

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


normal_env = FootballSkillsEnv(render_mode='gif')
policy_by_MVI,V_MVI,call_by_MVI = modified_value_iteration(normal_env,gamma = 0.95)
print("Number of calls made by VI:", call_by_MVI)
steps_PI, total_reward_PI = normal_env.get_gif(policy_by_MVI, seed=20, filename="MVI_output.gif")