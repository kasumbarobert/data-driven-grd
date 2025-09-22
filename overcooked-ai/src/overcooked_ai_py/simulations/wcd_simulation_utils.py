"""
WCD Simulation Utilities for Overcooked-AI Goal Recognition Design

Author: Robert Kasumba (rkasumba@wustl.edu)

This module provides utility functions for computing Worst-Case Distance (WCD) values
in Overcooked-AI environments using oracle simulation. It implements the core algorithms
for goal recognition design evaluation and environment analysis.

WHY THIS IS NEEDED:
- WCD computation is the core metric for goal recognition design evaluation
- Oracle simulation provides ground truth WCD values for training and validation
- Environment encoding/decoding enables tensor-based processing
- Goal recognition algorithms need efficient WCD computation for optimization

KEY FUNCTIONALITIES:
1. **WCD Computation**: Oracle simulation to compute exact WCD values
2. **Environment Processing**: Encoding/decoding between Overcooked-AI formats and tensors
3. **Goal Recognition**: Implementation of goal recognition algorithms
4. **Data Management**: Utilities for dataset creation and management
5. **Visualization**: Tools for environment and result visualization

CORE ALGORITHMS:
- compute_true_wcd(): Oracle simulation for exact WCD computation
- encode_env()/decode_env(): Environment format conversion
- goal_recognition_algorithm(): Core goal recognition implementation
- generate_grids(): Random environment generation
- update_or_create_dataset(): Dataset management utilities

USAGE:
    from wcd_simulation_utils import compute_true_wcd, encode_env, decode_env
    wcd = compute_true_wcd(environment_tensor, grid_size=6)
"""

import sys
sys.path.insert(0, "../../")
import overcooked_ai_py.mdp.overcooked_env as Env
import overcooked_ai_py.mdp.overcooked_mdp as Mdp
from overcooked_ai_py.mdp.actions import Action, Direction
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pandas as pd
from os.path import exists
import pickle
from utils import *

def set_seed(seed=42):
    """Set random seeds for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def randomly_choose_gamma(ranges=[(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)], probabilities=[0.1, 0.3, 0.6]):
    """
    Randomly sample gamma values for suboptimal agent behavior.
    
    Args:
        ranges: List of (low, high) tuples defining gamma ranges
        probabilities: Probability weights for each range
    
    Returns:
        float: Randomly sampled gamma value for agent discount factor
    """
    local_random = np.random.default_rng()
    selected_range = local_random.choice(len(ranges), p=probabilities)
    low, high = ranges[selected_range]
    # Randomly sample a value within the selected range
    gamma_value = local_random.uniform(low, high)
    return gamma_value


def transition(state, action, oc_env):
    """
    Compute state transition and reward for a given action in Overcooked environment.
    
    Args:
        state: Current environment state
        action: Action to take
        oc_env: Overcooked environment instance
    
    Returns:
        tuple: (next_state, reward) after taking the action
    """
    results = oc_env.mdp.get_state_transition(state, [action])
    reward = results[1]["sparse_reward_by_agent"][0] + results[1]["shaped_reward_by_agent"][0]
    return results[0], reward

def init_value_iteration(oc_env):
    """
    Initialize value iteration algorithm for computing optimal policies.
    
    Args:
        oc_env: Overcooked environment instance
    
    Returns:
        tuple: (state_info_list, transition_matrix, reward_table, state_hash)
    """
    oc_states = oc_env.mdp.get_all_possible_states()
    oc_states_hashed = {}
    reward_table = []
    Transition = np.zeros((len(oc_states), len(Action.ALL_ACTIONS)))
    
    for count, state in enumerate(oc_states):
        state_info = {"state": state, "value": 0.0, "best_action": 0, "action_reward": [], "index": count}
        oc_states_hashed[state.__hash__()] = state_info
        max_reward = -1
        action_reward = []
        
        for action in Action.ALL_ACTIONS:
            results = oc_env.mdp.get_state_transition(state, [action])
            reward = results[1]["sparse_reward_by_agent"][0]
            
            action_reward.append(reward)
            
            if reward > max_reward:
                max_reward = reward
                s_prime = results[0]
        
        oc_states_hashed[state.__hash__()]["value"] = max_reward
        reward_table.append(action_reward)
    
    for s in oc_states_hashed.keys():
        s_idx = oc_states_hashed[s]["index"]
        state_0 = oc_states_hashed[s]["state"]
        
        for act_idx, action in enumerate(Action.ALL_ACTIONS):
            s_prime, rw = transition(state_0, action,oc_env)
            oc_states_hashed[s]["action_reward"] += [(s_prime, rw)]
            Transition[s_idx][act_idx]=oc_states_hashed[s_prime.__hash__()]["index"]
            
    return oc_states_hashed, oc_states, reward_table, Transition


def reconfigure_agent(recipe = None, oc_env=None):
    num_items = 3
    max_ingredients = 3
    recipe_values = 5.2
    recipe_times = 5.0

    recipes = {
        1: [{
            'num_items_for_soup': num_items,
            'all_orders': [{'ingredients': ["tomato",'tomato', 'tomato']}],
            'recipe_values': [recipe_values],
            'recipe_times': [recipe_times],
            'max_num_ingredients': max_ingredients
                }, {
            "PLACEMENT_IN_POT_REW": 0.0,
            "DISH_PICKUP_REWARD": 0.0,
            "SOUP_PICKUP_REWARD": 0.0,
            "DISH_DISP_DISTANCE_REW": 0.0,
            "POT_DISTANCE_REW": 0.0,
            "SOUP_DISTANCE_REW": 0.0,
            "BEGIN_COOKING_REWARD":0.0,
            "USELESS_ACTION_PENALTY":0,
            "ONION_POT_PLACEMENT":0,
            "TOMATO_POT_PLACEMENT":0.3

        }],
                2: [{
                    'num_items_for_soup': num_items,
                    'all_orders': [{'ingredients': [ 'onion','tomato', 'tomato']}],
                    'recipe_values': [recipe_values],
                    'recipe_times': [recipe_times],
                    'max_num_ingredients': max_ingredients
                }, {
            "PLACEMENT_IN_POT_REW": 0.0,
            "DISH_PICKUP_REWARD": 0.0,
            "SOUP_PICKUP_REWARD": 0.0,
            "DISH_DISP_DISTANCE_REW": 0.0,
            "POT_DISTANCE_REW": 0.0,
            "SOUP_DISTANCE_REW": 0.0,
            "BEGIN_COOKING_REWARD":0.0,
            "USELESS_ACTION_PENALTY":0,
            "ONION_POT_PLACEMENT":0.30,
            "TOMATO_POT_PLACEMENT":0.22

        }],
                3: [{
                    'num_items_for_soup': num_items,
                    'all_orders': [{'ingredients': [ 'onion', 'onion','tomato']}],
                    'recipe_values': [recipe_values],
                    'recipe_times': [recipe_times],
                    'max_num_ingredients': max_ingredients
                },{
            "PLACEMENT_IN_POT_REW": 0.0,
            "DISH_PICKUP_REWARD": 0.0,
            "SOUP_PICKUP_REWARD": 0.0,
            "DISH_DISP_DISTANCE_REW": 0.0,
            "POT_DISTANCE_REW": 0.0,
            "SOUP_DISTANCE_REW": 0.0,
            "BEGIN_COOKING_REWARD":0.0,
            "USELESS_ACTION_PENALTY":0,
            "ONION_POT_PLACEMENT":0.22,
            "TOMATO_POT_PLACEMENT":0.30

        }],
                4: [{
                    'num_items_for_soup': num_items,
                    'all_orders': [{'ingredients': ['onion', 'onion', 'onion']}],
                    'recipe_values': [recipe_values],
                    'recipe_times': [recipe_times],
                    'max_num_ingredients': max_ingredients
                },{
            "PLACEMENT_IN_POT_REW": 0.0,
            "DISH_PICKUP_REWARD": 0.0,
            "SOUP_PICKUP_REWARD": 0.0,
            "DISH_DISP_DISTANCE_REW": 0.0,
            "POT_DISTANCE_REW": 0.0,
            "SOUP_DISTANCE_REW": 0.0,
            "BEGIN_COOKING_REWARD":0.0,
            "USELESS_ACTION_PENALTY":0,
            "ONION_POT_PLACEMENT":0.3,
            "TOMATO_POT_PLACEMENT":0

        }]
    }
    
    
    conf = Mdp.Recipe.configuration # get current configuration
    
    if recipe != None:
        # conf["recipe_values"] = [recipe_values]*5 # change the configuration
        # conf["recipe_values"][recipe-1] = 1000
        conf = recipes[recipe][0]
        shaping_params = recipes[recipe][1]
    else:
        recipes = {
            'num_items_for_soup': num_items,
            'all_orders': [
                {'ingredients': ['tomato', 'tomato', 'tomato']},
                {'ingredients': ['tomato', 'tomato', 'onion']},
                {'ingredients': ['tomato', 'onion', 'onion']},
                {'ingredients': ['onion', 'onion', 'onion']}
            ],
            'recipe_values': [recipe_values, recipe_values, recipe_values, recipe_values],
            'recipe_times': [recipe_times, recipe_times, recipe_times, recipe_times],
            'max_num_ingredients': max_ingredients
        }
        conf= recipes # change the configuration
    
    oc_env.mdp._configure_recipes(conf["all_orders"],conf["num_items_for_soup"],max_num_ingredients=max_ingredients)
    oc_env.mdp.start_all_orders = conf["all_orders"]
    oc_env.mdp.reward_shaping_params = shaping_params
    Mdp.Recipe._computed = False
    Mdp.Recipe.ALL_RECIPES_CACHE ={}


    Mdp.Recipe.configure(conf) # reconfigure


def find_wcd(paths):
    max_prefix_length = 0
    for path1 in paths:
        for path2 in paths:
            if path1 == path2: continue
            prefix_length = -1
            i = 0
            while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
                prefix_length += 1
                i += 1
            if prefix_length > max_prefix_length:
                max_prefix_length = prefix_length
    return max_prefix_length

def computeQFunction(Transition, reward_table, gamma):
    Nstate = Transition.shape[0]
    Naction = Transition.shape[1]
    T = 60 if Nstate<20_000 else 120
    R = torch.tensor(reward_table, device ="cuda")
    P = torch.tensor(Transition, device ="cuda") # NxA

    policy = torch.zeros([T, Nstate], device="cuda")
    Rs = torch.zeros([T, Nstate], device="cuda")
    Q = torch.zeros([T, Nstate, Naction], device="cuda")

    # Initialize sum of v across all states
    sum_v = 0.0

    for step in range(T):
        pos = T - step - 1

        if pos == T - 1: # greedy in last step
            v = R
            policy[pos,:] = torch.argmax(v, axis=1)
            Q[pos, :, : ] = v
            Rs[pos,:] = torch.max(v, axis=1)[0]

        else: # optimal in expectation
            R_primes = Rs[pos + 1].index_select(0, P.view(-1).long()).view(P.shape)
            v = R[:,:] + gamma * R_primes
            Q[pos, :, :] = v
            policy[pos,:] = torch.argmax(v, axis=1)
            Rs[pos,:] = torch.max(v, axis=1)[0]
    return policy.cpu(), Q[0].cpu()

def get_features(oc_env=None):
    env_features =  oc_env.featurize_environment()
    return env_features

def recompute_reward_function(oc_states,oc_states_hashed, recipe=1,oc_env=None):
    # reconfigure_agent(recipe ) # first reconfigure the recipe to the specific agent type
    reward_table = []
    
    for state in oc_states:
        max_reward = -1
        action_reward = []
        state_0 = state
        for action in Action.ALL_ACTIONS:
            s_prime,rw = transition(state_0, action,oc_env)
            action_reward.append(rw)
        reward_table.append(action_reward)
    return reward_table

def simulate(gamma, env_layout, label, base_layout_params=None, subset_goals=None, verbose=True):
    """
    Main oracle simulation function for computing WCD and goal recognition metrics.
    
    This function runs the complete oracle simulation pipeline:
    1. Creates Overcooked environment from layout
    2. Computes optimal policies for each goal
    3. Simulates agent behavior and tracks paths
    4. Computes WCD and other goal recognition metrics
    
    Args:
        gamma: Discount factor for agent behavior
        env_layout: Environment layout specification
        label: Simulation label for identification
        base_layout_params: Environment configuration parameters
        subset_goals: List of goals to consider (if None, randomly sampled)
        verbose: Whether to print simulation details
    
    Returns:
        dict: Simulation results including WCD, paths, and other metrics
    """
    display_all = False
    paths = []
    T = 80  # Simulation horizon
    
    # Set default environment parameters if not provided
    if base_layout_params is None:
        base_layout_params = {
            'num_items_for_soup': 3,
            'all_orders': [{'ingredients': ["tomato", 'tomato', 'tomato']}],
            'recipe_values': [1],
            'recipe_times': [5.0],
            'max_num_ingredients': 3
        }
    
    import random

    # Randomly sample goals if not specified
    lst = [1, 2, 3, 4]  # Available goal types
    n = random.randint(2, 4)  # Number of goals to consider
    if subset_goals is None:
        subset_goals = random.sample(lst, n)

    # Process each goal in the subset
    for goal in subset_goals:
        # Create Overcooked environment from layout
        over_cookedgridwrld = Mdp.OvercookedGridworld.from_grid(env_layout, base_layout_params)
        oc_env = Env.OvercookedEnv.from_mdp(over_cookedgridwrld, horizon=120)
        
        # Configure agent for this specific goal
        reconfigure_agent(recipe=goal, oc_env=oc_env)
        
        # Initialize value iteration and compute optimal policy
        oc_states_hashed, oc_states, reward_table, Transition = init_value_iteration(oc_env)
        rwd_tab = recompute_reward_function(oc_states, oc_states_hashed, recipe=goal, oc_env=oc_env)
        policy, Q = computeQFunction(Transition, rwd_tab, gamma)
        policy = policy[0].int()

        # Simulate agent behavior to find optimal path for this goal
        path = []  # Initialize path tracking for this agent

        # Inner loop to run the simulation for 50 steps
        oc_env.reset()
        cumulative_score=0
        for i in range(0,T):
            idx = oc_states_hashed[oc_env.state.__hash__()]["index"]  # Get the index of the current state
            action = Action.ALL_ACTIONS[policy[idx]]  # Select the action based on the policy and the current state

            step_result = oc_env.step([action,action])  # Take the action in the environment
            rewards = step_result[3]["sparse_r_by_agent"][0]+step_result[3]["shaped_r_by_agent"][0]  # Get the rewards for this step
            cumulative_score+=rewards  # Add the rewards to the cumulative score

            # If display_all is True, display the environment and agent information
            if display_all:
                print("Next action: ",Action.ACTION_TO_CHAR[action])  # Display the action taken
                print(oc_env)  # Display the environment
                print("The agent received ",rewards," points: Total score is ",cumulative_score)  # Display the rewards and cumulative score

            path.append((idx,Action.ACTION_TO_INDEX[action]))  # Append the current state and action to the path
        # Display the final environment and agent information
        if verbose:
            print("The agent received ",rewards," points: Total score is ",cumulative_score)
        paths.append(path)  # Append the path for this agent to the overall list of paths
    
    env_features = get_features(oc_env)
    env_features["paths"]= paths
    env_features["env_goals"]= subset_goals
    env_features["gamma"]= gamma
    env_features["env_layout_file"]= label
    env_features["wcd"]= find_wcd(paths)

    
    
    return env_features


def compute_true_wcd(x_i, grid_size=6):
    """
    Compute the true Worst-Case Distance (WCD) using oracle simulation.
    
    This is the core function for computing exact WCD values through oracle simulation,
    which provides ground truth for training the CNN oracle and validating results.
    
    Args:
        x_i: Environment tensor containing layout, gamma, and goal information
        grid_size: Size of the gridworld environment
    
    Returns:
        float: True WCD value computed via oracle simulation
    
    Process:
        1. Extract environment layout, gamma value, and goal subset from tensor
        2. Run oracle simulation to compute exact WCD
        3. Return the computed WCD value
    """
    env_layout, gamma, subset_goals = extract_env_details(x_i, grid_size=grid_size)
    true_wcd = simulate(gamma, env_layout, "true_wcd", subset_goals=subset_goals, verbose=False)["wcd"]
    return true_wcd