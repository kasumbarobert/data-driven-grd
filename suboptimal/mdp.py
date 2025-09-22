import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import torch
import pandas as pd
import itertools
from itertools import product,combinations
import math

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


class GridworldMDP:
    def __init__(self, n, goal_state_pos=None, goal_state_rewards=None, blocked_pos=[], start_pos=None, special_reward_pos=[], special_rewards=None):
        """
        Initializes the Gridworld MDP (Markov Decision Process).

        Parameters:
        - n (int): The size of the gridworld (n x n).
        - goal_states (list of tuples): Positions of the goal states in the grid.
        - goal_state_rewards (list of floats): Rewards for reaching each goal state.
        - blocked_pos (list of tuples): Positions of the blocked states in the grid.
        - start_pos (tuple): The starting position of the agent in the grid.
        - special_reward_states (list of tuples): Positions of states with special rewards.
        - special_rewards (list of floats): Rewards for reaching each special reward state.
        """

        # Initialize class variables
        self.n = n  # Grid size
        self.init_pos = start_pos  # Initial position of the agent
        self.goal_state_pos= goal_state_pos if goal_state_pos is not None else []  # Goal states
        # self.goal_states = [(pos, True) for pos in self.goal_state_pos] # the goal state is only after a subgoal is reached
        self.goal_state_rewards = goal_state_rewards if goal_state_rewards is not None else []  # Rewards for goal states
        self.agent_pos = start_pos  # Current position of the agent
        self.special_reward_pos = special_reward_pos if special_reward_pos is not None else []  # States with special rewards
        # self.special_reward_states = [(pos, True) for pos in self.special_reward_pos] # the goal state is only after a subgoal is reached
        self.special_rewards = special_rewards if special_rewards is not None else []  # Special rewards
        self.blocked_pos = blocked_pos  # Blocked states
        self.states = self.generate_states()  # Generate all possible states
        
                
        self.actions = ['left', "stay", 'up', 'down', 'right']  # Possible actions
        self.transitions = self._generate_transitions()  # Generate state transitions
        self.rewards = self._generate_rewards()  # Generate rewards for state transitions
        self.visited_sub_goals = [] # no state is visited
        self.cumulative_reward = 0.0
        

    def _generate_transitions(self):
        """
        Generates the transition probabilities for each state-action pair.

        Returns:
        - P (list): A 3D list containing transition probabilities.
        """
        P = [None]*len(self.states)
        for s in range(len(self.states)):
            P[s] = [None]*len(self.actions)
            for a in range(len(self.actions)):
                P[s][a] = [0]*len(self.states)
                next_state=self._next_state(self.states[s], self.actions[a], p=1.0)
                next_state_idx = self.states.index(next_state)
                
                if self.states[s][0] in self.goal_state_pos and next_state[0] not in self.goal_state_pos:  # prevent leaving goal state
                    P[s][a][next_state_idx] =  0.0
                elif self.states[s][0] in self.blocked_pos:
                    P[s][a][next_state_idx] = 0.0
                elif self.states[next_state_idx][0] in self.blocked_pos:
                    P[s][a][next_state_idx] = 0.0
                else:
                    P[s][a][next_state_idx] = 1.0
        return P

    def get_transitions(self):
        """
        Returns the transition probabilities.

        Returns:
        - self.transitions (list): The transition probabilities.
        """
        return self.transitions

    def _generate_rewards(self):
        """
        Generates the rewards for each state-action pair in a Markov Decision Process.
        Rewards are assigned based on the state and action taken, with special consideration for goal states, special reward states, and subgoals.

        Returns:
        - R (list): A 3D list containing rewards for each state-action-next state triplet.
        """
        # Initialize the rewards list
        R = [None]*len(self.states)
        for s in range(len(self.states)):
            R[s] = [None]*len(self.actions)
            for a in range(len(self.actions)):
                R[s][a] = [0]*len(self.states)

                # Calculate the next state based on the current state and action
                s_prime = self._next_state(self.states[s], self.actions[a], p=1)
                # print(s_prime)
                # if s_prime[0] in self.special_reward_pos:
                #     print(s_prime[0] in s_prime[1], s_prime[0], s_prime[1])
                if s_prime[0] in s_prime[1]: # already visited
                    R[s][a][self.states.index(s_prime)] = 0.0 # Zero reward for going to an already visited state
                elif self.states[s][0] in self.goal_state_pos:  # If in a goal state
                    # Penalize any action other than 'stay'
                    R[s][a][self.states.index(s_prime)] = 0 # penalize leaving the goal state

                elif self.states[s][0] in self.special_reward_pos:  # If in a goal state
                    # Penalize any action other than 'stay'
                    if self.actions[a] == "stay":
                        R[s][a][self.states.index(s_prime)] = 0
                        
                elif s_prime[0] in self.goal_state_pos:  # If the next state is a goal state
                    # Assign reward based on reaching the goal state
                    R[s][a][self.states.index(s_prime)] = self.goal_state_rewards[self.goal_state_pos.index(s_prime[0])]

                elif s_prime[0] in self.special_reward_pos:  # If the next state has a special reward
                    # Assign a special reward for reaching this state
                    R[s][a][self.states.index(s_prime)] = self.special_rewards[self.special_reward_pos.index(s_prime[0])]

                elif not s_prime[1]:  # If the subgoal has not been reached in the next state
                    # Assign a negative reward to encourage reaching subgoals
                    R[s][a][self.states.index(s_prime)] = 0.0 #-0.01

                else:  # For all other cases
                    # Assign -0.1 for staying and  no reqard for moving
                    if self.actions[a] == "stay":
                        R[s][a][self.states.index(s_prime)] = -0.01 # staying in a non-goal state should be punished
        
        return R


    def _next_state(self, state, action, p=1):
        """
        Computes the next state given a current state and an action.

        Parameters:
        - state (tuple): The current state.
        - action (str): The action to be taken.
        - p (float): Probability of the action (default is 1).

        Returns:
        - next_state (tuple): The next state after performing the action.
        """
        (i, j), visited_subgoals = state
        sub_goal_status = state[1]
        
        if action == 'up':
            next_i = max(i-1, 0)
            next_j = j
        elif action == 'down':
            next_i = min(i+1, self.n-1)
            next_j = j
            
        elif action == 'left':
            next_i = i
            next_j = max(j-1, 0)
        elif action == 'right':
            next_i = i
            next_j = min(j+1, self.n-1)
        elif action == 'stay':
            next_i = i
            next_j = j
        else:
            raise ValueError("Invalid action")
        
        
        next_pos = (next_i, next_j)
        visited_subgoals = visited_subgoals.copy()
        # print(visited_subgoals)
        if (i,j) in self.special_reward_pos and (i,j) not in visited_subgoals:
            visited_subgoals.append((i,j))
            visited_subgoals = sorted(visited_subgoals)
        
        if (i,j) in self.goal_state_pos: #once a goal state is reached all rewards are 0
            next_pos = (i,j)
            visited_subgoals = sorted(self.special_reward_pos + self.goal_state_pos )
            
        if next_pos in self.blocked_pos:
            return (i, j), visited_subgoals # update this state to show that (i,j) is visited if it's a subgoal
        
        next_state = (next_pos,visited_subgoals)
        
        return next_state

    def reset(self):
        """
        Resets the agent to the initial position.

        Returns:
        - self.agent_pos (tuple): The reset position of the agent.
        """
        self.agent_pos = self.init_pos
        self.visited_sub_goals = [] # NO state visited so far
        return (self.agent_pos,self.visited_sub_goals)
    

    def move(self, action):
        """
        Moves the agent according to the given action.

        Parameters:
        - action (str): The action to be taken.

        Returns:
        - self.agent_pos (tuple): The new position of the agent.
        """
        curr_state = (self.agent_pos,self.visited_sub_goals)
        next_state = self._next_state(curr_state, action, p=1)
        self.agent_pos,self.visited_sub_goals= next_state
        reward = self.get_rewards()[self.states.index(curr_state)][self.actions.index(action)][self.states.index(next_state)] # extract the reward from the reward table
        
        self.cumulative_reward+=reward
        # print("to",next_state[0],"reward",reward)
        return self.agent_pos,self.visited_sub_goals, reward
    
    def get_cumulative_reward(self):
        return self.cumulative_reward
    
    def generate_all_subgoal_visit_combinations(self):
        """
        Generates all possible state matrices with varying combinations of True and False.

        Returns:
        - subgoals_visited_combinations : List of subgoals_visited_combinations.
        """
    
        
        subgoals_visited_combinations =  [
                sorted(list(combination))
                for r in range(0, len(self.special_reward_pos+self.goal_state_pos)+1)
                for combination in combinations(self.special_reward_pos+self.goal_state_pos, r)
            ]
        return subgoals_visited_combinations

    def generate_states(self):
        """
        Generates all possible states in the gridworld.

        Returns:
        - all_states (list of tuples): All possible states in the grid.
        """
        state_visited = [True,False]
        
        all_states = []
        subgoal_visit_combinations = self.generate_all_subgoal_visit_combinations()

        
        
        for i in range(self.n):
            for j in range(self.n):
                for is_visited in subgoal_visit_combinations:
                    if (i,j) not in self.blocked_pos : # these will always be unreachable
                        all_states.append(((i, j),is_visited))
        
        # print("There are ", len(all_states)," states")
        return all_states
                                

    def get_rewards(self):
        """
        Returns the rewards.

        Returns:
        - self.rewards (list): The rewards.
        """
        return self.rewards

    def get_states(self):
        """
        Returns the states.

        Returns:
        - self.states (list of tuples): The states.
        """
        return self.states

    def get_curr_state_index(self):
        """
        Returns the index of the current state of the agent.

        Returns:
        - Index (int): The index of the current state.
        """
       
        
        return self.states.index((self.agent_pos,self.visited_sub_goals))

    def get_state_index(self, state):
        return self.states.index(state)
        
    
    def action_index(self, action):
        if action in self.actions:
            return self.actions[index]
        else:
            raise ValueError('Invalid action')
    
    def index_action(self, index):
        index = int (index)
        if index < len(self.actions): 
            return self.actions[index]
        else:
            raise ValueError('Invalid action index')
    def get_mdp_representation(self):
        return {
        "grid_size":self.n,
        "blocked_positions": self.blocked_pos,
        "start_pos": self.init_pos,
        "sub_goal_positions": self.special_reward_pos,
        "sub_goal_rewards": self.special_rewards,
        "goal_positions": self.goal_state_pos,
        "goal_rewards": self.goal_state_rewards,
        "gamma": 1
    }
        
    def visualize(self, path=None):
        
        grid = construct_grid(self.get_mdp_representation())
        plot_grid(grid,path)
        
        
    def computeQFunction(self, gamma, T = 50):
        Nstate = len(self.get_states())
        Naction = len(self.actions)
        T = T
        R = torch.tensor(self.get_rewards())
        P = torch.tensor(self.get_transitions())
        policy = torch.zeros([ T, Nstate])
        Vs = torch.zeros([T, Nstate])

        Q = torch.zeros([T, Nstate, Naction])
        R = torch.sum((R*P), dim=2)
        # print(R)
        for step in range (T):
            pos = T  - step - 1
            if pos == T -1: # greedy in last step
                q_f = R
                policy[pos,:] = torch.argmax(q_f, dim=1)
                Q[pos, :, : ] = q_f
                Vs[pos,:] = torch.max(q_f,axis=1)[0]
            else: # optimal in expectation
                q_f = R  + gamma * torch.sum(P[:, :, :] * (  Vs[pos + 1, :]), axis=2) 
                # + torch.randn(R.shape) * (1e-4)
                # + torch.randn(R.shape) * (1e-5)
                Q[pos, :, :] = q_f
                policy[pos,:] = torch.argmax(q_f,dim=1)
                Vs[pos,:] = torch.max(q_f,axis=1)[0]

        df =pd.DataFrame(zip(self.get_states(),torch.argmax(Q[0],dim=1).numpy(),Q[0].numpy()))
        pd.set_option('display.max_rows', None)


        return policy.cpu()[0],Q[0].cpu()
    
    def computeHyperbolicQ(self, gamma=  0.1, T = 20):
        ## hyperbolic discounting factor is 1/(1 + gamma * t), gamma is usually range of [0,10]
        ### T is a predifined parameter, should be large enough to cover optimal path
        R = np.array(self.rewards) ## s*a*s
        P = np.array(self.transitions) ## s*a*s 
        Naction = len(self.actions)
        Nstate = P.shape[0]
        gt = np.array([1.0 / (1 + gamma * t) for t in range(T)]) ## general discounting factor

        policy = np.zeros([T, Nstate],dtype=int)
        Rs = np.zeros([T, Nstate]) # records of cumulative reward
        Q = np.zeros([T, Nstate, Naction])
        trans = np.zeros([T, Nstate, Nstate]) # state transform probability by current policy
        Rst = np.zeros([T, Nstate]) # records of reward when following policy
        transRst = np.zeros([T, Nstate]) # records of transed reward (  trans [1] * trans[2] *Rst[3])

        def naive_solver(tau):  # solve a problem in belief
            subpolicy = np.zeros([tau, Nstate], dtype=int)
            subRs = np.zeros([tau, Nstate])  # records of cumulative reward
            subQ = np.zeros([tau, Nstate, Naction])
            for step in range(tau):
                pos = tau - step - 1
                if pos == tau - 1:  # greedy in last step
                    for s in range(Nstate):
                        v = gt[pos] * np.sum(P[s,:,:] * R[s,:,:], axis=1)
                        subpolicy[pos, s] = np.argmax(v)
                        subQ[pos, s, :] = v
                        subRs[pos, s] = np.max(v)
                else:  # optimal in expectation
                    for s in range(Nstate):
                        v = gt[pos] * np.sum(P[s,:,:] * R[s,:,:], axis=1) + np.sum(P[s, :, :] * subRs[pos + 1, :], axis=1)
                        subQ[pos, s, :] = v
                        subpolicy[pos, s] = np.argmax(v)
                        subRs[pos, s] = np.max(v)
            return subpolicy.astype(int), subQ

        for step in range(T):
            subans = naive_solver(T-step)
            Q[step, :, :] = subans[1][0,:,:] 
            policy[step,:] = subans[0][0,:]
            for j in range(Nstate):
                Rs[step,j] = Q[step, j, policy[step,j]]

        return policy[0,:]



def construct_grid(decoded_info):
    # Initialize the grid
    grid_size = decoded_info["grid_size"]
    grid = np.full((grid_size, grid_size)," ",dtype='U10')

    # Mark blocked positions
    blocked_positions = decoded_info["blocked_positions"]
    for pos in blocked_positions:
        grid[pos[0], pos[1]] = "X"

    # Mark start position
    start_pos = decoded_info["start_pos"]
    grid[start_pos[0], start_pos[1]] = "S"

    # Mark sub-goal positions and rewards
    sub_goal_positions = decoded_info["sub_goal_positions"]
    sub_goal_rewards = decoded_info["sub_goal_rewards"]
    for pos, reward in zip(sub_goal_positions, sub_goal_rewards):
        grid[pos[0], pos[1]] = f"{reward}"

    # Mark goal positions and rewards
    goal_positions = decoded_info["goal_positions"]
    goal_rewards = decoded_info["goal_rewards"]
    for pos, reward in zip(goal_positions, goal_rewards):
        grid[pos[0], pos[1]] = f"{reward}"

    return grid
    
def plot_grid(grid, path=None):
    # Map each character to a color
    color_map = {
        ' ': 'white',   # Empty space
        'X': 'white',   # Blocked
        'S': 'orange'   # Start
    }

    # Define colors for values below and above 1
    color_below_1 = 'lightblue'
    color_above_1 = 'lightgreen'

    # Plotting
    fig, ax = plt.subplots()

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            c = grid[i][j]

            # Assign colors based on the value
            if c in ['X', ' ', 'S']:
                color = color_map[c]
            elif float(c) < 1:
                color = color_below_1
            else:
                color = color_above_1

            # Display the cell with the assigned color
            ax.add_patch(plt.Rectangle((j, len(grid) - i - 1), 1, 1, fill=True, edgecolor='black', facecolor=color))
            ax.text(j + 0.5, len(grid) - i - 0.5, str(c), va='center', ha='center', color='red' if c in ['X', ' '] else 'black')

    if path:
        for pos in path:
            ax.add_patch(plt.Rectangle((pos[1], len(grid) - pos[0] - 1), 1, 1, fill=True, edgecolor='black', facecolor='orange'))

    # Setting axis limits
    ax.set_xlim(0, len(grid[0]))
    ax.set_ylim(0, len(grid))

    # Set axis labels
    ax.set_xticks(np.arange(0.5, len(grid[0]) + 0.5, 1))
    ax.set_yticks(np.arange(0.5, len(grid) + 0.5, 1))
    ax.set_xticklabels(np.arange(0, len(grid[0]), 1))
    # ax.set_yticklabels(np.arange(0, len(grid), 1))
    ax.set_yticklabels(np.arange(len(grid) - 1, -1, -1))
    plt.show()
    
