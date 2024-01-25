import sys
import os
from pathlib import Path

from utils import *
from collections import deque
import time
import seaborn as sns
import multiprocessing
import pdb
import numpy as np

class GRDModel:
    def __init__(self, grid_size = 6, start_pos = (0,0), goal_positions = [],blocked_positions = [], unblocked_positions = [],
                 init_goal_costs = [], compute_wcd = True, n_changes_so_far = [0,0], n_max_changes =[3,5]):
        self.grid_size =grid_size
        self.start_pos = start_pos
        self.unblocked_positions = unblocked_positions
        self.blocked_positions = blocked_positions
        self.goal_positions = goal_positions
        self.init_goal_costs = init_goal_costs
        self. wcd = None # default None
        self.n_max_changes = n_max_changes
        if compute_wcd:
            self. wcd = self.compute_true_wcd()
        else:
            
            self.init_encoding= encode_grid_design(grid_size, goal_positions, blocked_positions, start_pos).float().to(DEVICE) # will be useful in computing predicted
        self.n_changes_so_far = n_changes_so_far #[n_blockings,n_unblockings]
            
    
        
    def reduce_exhaustive(self):
        children = []
        
        if self.get_wcd()==0: # no need to expand this model
            return self.get_wcd(), []
        
        if self.n_changes_so_far[0] < self.n_max_changes[0]: # still more room to block
            for action in self.unblocked_positions:
                blocked = self.blocked_positions+[action]
                unblocked = self.unblocked_positions.copy()
                unblocked.remove(action)
                n_changes_so_far = [self.n_changes_so_far[0]+1,self.n_changes_so_far[1]] #unblocking increases by 1

                child = GRDModel(grid_size = self.grid_size, start_pos = self.start_pos,goal_positions = self.goal_positions,
                                 blocked_positions = blocked, unblocked_positions = unblocked,init_goal_costs=self.init_goal_costs, 
                                 n_changes_so_far = n_changes_so_far, n_max_changes = self.n_max_changes )

                is_valid, cost_to_goals = is_design_valid(self.grid_size, self.goal_positions, blocked, self.start_pos)

                if (np.array(cost_to_goals)<=np.array(self.init_goal_costs)).all() and is_valid:  # All goals must be reachable and the cost must not increase
                    if child.get_wcd() is not None:  #the modifications should not lead to an invalid design where the goal is blocked
                        if child.get_wcd()<= self.get_wcd(): # WCD does not get worse
                            children.append(child)
                            
        if self.n_changes_so_far[1] < self.n_max_changes[1]: # still more room to block   
            for action in self.blocked_positions:
                unblocked = self.unblocked_positions+[action]
                blocked = self.blocked_positions.copy()
                blocked.remove(action)
                n_changes_so_far = [self.n_changes_so_far[0],self.n_changes_so_far[1]+1] #blocking increases by 1
                child = GRDModel(grid_size = self.grid_size, start_pos = self.start_pos,goal_positions = self.goal_positions,
                                 blocked_positions = blocked, unblocked_positions = unblocked,init_goal_costs=self.init_goal_costs,n_changes_so_far = n_changes_so_far,
                                 n_max_changes = self.n_max_changes)
                is_valid, cost_to_goals = is_design_valid(self.grid_size, self.goal_positions, blocked, self.start_pos)

                if (np.array(cost_to_goals)<=np.array(self.init_goal_costs)).all() and is_valid:  # All goals must be reachable and the cost must not increase
                    if child.get_wcd() is not None:  #the modifications should not lead to an invalid design where the goal is blocked
                        if child.get_wcd()<= self.get_wcd(): # WCD does not get worse
                            children.append(child)
        
        return self.get_wcd(),children
    
    def greedy_next_true_wcd(self,all_modifications_uniform):
        children = []
        
        if self.get_wcd()==0: # no need to expand this model
            return self
        
        best_wcd = self.get_wcd()
        next_child = self
        
        if self.n_changes_so_far[0] < self.n_max_changes[0] or all_modifications_uniform: # still more room to block
            for action in self.unblocked_positions:
                blocked = self.blocked_positions+[action]
                unblocked = self.unblocked_positions.copy()
                unblocked.remove(action)
                n_changes_so_far = [self.n_changes_so_far[0]+1,self.n_changes_so_far[1]] #blocking increases by 1
                child = GRDModel(grid_size = self.grid_size, start_pos = self.start_pos,goal_positions = self.goal_positions,
                                 blocked_positions = blocked, unblocked_positions = unblocked,init_goal_costs=self.init_goal_costs, 
                                 n_changes_so_far = n_changes_so_far,
                                 n_max_changes = self.n_max_changes)
                is_valid, cost_to_goals = is_design_valid(self.grid_size, self.goal_positions, blocked, self.start_pos)

                if (np.array(cost_to_goals)<=np.array(self.init_goal_costs)).all() and is_valid:  # All goals must be reachable and the cost must not increase

                    if child.get_wcd()<= best_wcd: # WCD does not get worse
                        next_child = child
                        best_wcd = child.get_wcd()
                    
        if self.n_changes_so_far[1] < self.n_max_changes[1] or all_modifications_uniform: # still more room to unblock         
            for action in self.blocked_positions:
                unblocked = self.unblocked_positions+[action]
                blocked = self.blocked_positions.copy()
                blocked.remove(action)
                n_changes_so_far = [self.n_changes_so_far[0],self.n_changes_so_far[1]+1] #unblocking increases by 1
                child = GRDModel(grid_size = self.grid_size, start_pos = self.start_pos,goal_positions = self.goal_positions,
                                 blocked_positions = blocked, unblocked_positions = unblocked,
                                 init_goal_costs=self.init_goal_costs, n_changes_so_far = n_changes_so_far,
                                 n_max_changes = self.n_max_changes)
                is_valid, cost_to_goals = is_design_valid(self.grid_size, self.goal_positions, blocked, self.start_pos)

                if (np.array(cost_to_goals)<=np.array(self.init_goal_costs)).all() and is_valid:  # All goals must be reachable and the cost must not increase
                    if child.get_wcd()<= best_wcd: # WCD does not get worse
                        next_child = child
                        best_wcd = child.get_wcd()
        
        return next_child
    
        
    
    def greedy_next_predicted_wcd(self,model, all_mods_uniform=False):
        children = []
        # pdb.set_trace()
        best_wcd = self.compute_predicted_wcd(model)
        next_child = self
        
        if best_wcd<0.05: # no need to expand this model since the predicted WCD is close to 0
            return self
        if self.n_changes_so_far[0] < self.n_max_changes[0] or all_mods_uniform: # still more room to block
            for action in self.unblocked_positions:
                blocked = self.blocked_positions+[action]
                unblocked = self.unblocked_positions.copy()
                unblocked.remove(action)
                n_changes_so_far = [self.n_changes_so_far[0]+1,self.n_changes_so_far[1]] #blocking increases by 1
                child = GRDModel(grid_size = self.grid_size, start_pos = self.start_pos,goal_positions = self.goal_positions,
                                 blocked_positions = blocked, unblocked_positions = unblocked,init_goal_costs=self.init_goal_costs,
                                 compute_wcd= False,n_changes_so_far = n_changes_so_far,
                                 n_max_changes = self.n_max_changes)
                is_valid, cost_to_goals = is_design_valid(self.grid_size, self.goal_positions, blocked, self.start_pos)
                if (np.array(cost_to_goals)<=np.array(self.init_goal_costs)).all() and is_valid:  # All goals must be reachable and the cost must not increase
                    child_wcd = child.compute_predicted_wcd(model)
                    if child_wcd<= best_wcd: # WCD does not get worse
                        next_child = child
                        best_wcd = child_wcd
        
        if self.n_changes_so_far[1] < self.n_max_changes[1] or all_mods_uniform: # still more room to unblock
            for action in self.blocked_positions:
                unblocked = self.unblocked_positions+[action]
                blocked = self.blocked_positions.copy()
                blocked.remove(action)
                n_changes_so_far = [self.n_changes_so_far[0],self.n_changes_so_far[1]+1] #unblocking increases by 1
                child = GRDModel(grid_size = self.grid_size, start_pos = self.start_pos,goal_positions = self.goal_positions,
                                 blocked_positions = blocked, unblocked_positions = unblocked,init_goal_costs=self.init_goal_costs,
                                 compute_wcd= False,n_changes_so_far = n_changes_so_far,
                                 n_max_changes = self.n_max_changes)
                is_valid, cost_to_goals = is_design_valid(self.grid_size, self.goal_positions, blocked, self.start_pos)

                if (np.array(cost_to_goals)<=np.array(self.init_goal_costs)).all() and is_valid:  # All goals must be reachable and the cost must not increase
                    child_wcd = child.compute_predicted_wcd(model)
                    if child_wcd<= best_wcd: # WCD does not get worse
                        next_child = child
                        best_wcd = child_wcd
        
        return next_child
        
    
    def compute_predicted_wcd(self,model):
        return model(self.init_encoding)
    
    def compute_true_wcd(self):
        return compute_wcd_single_env(self.grid_size, self.goal_positions, self.blocked_positions, self.start_pos, vis_paths = False, return_paths = False)
    
    def get_grid(self):
        return 
    
    def get_wcd(self):
        if self.wcd is None:
            return self.compute_true_wcd()
        else:
            return self.wcd
    
    def get_blocked(self):
        return set(self.blocked_positions)
    
    def get_num_changes_so_far(self):
        return self.n_changes_so_far
    
    def __eq__(self, other):
        if not isinstance(other, GRDModel):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.get_blocked() == other.get_blocked()

    def __hash__(self):
        # The hash is a unique representation of the object. 
        # For this case, we can use the hash of the frozenset of the blocked items.
        return hash(frozenset(self.blocked_positions))


class GRDModelBlockingOnly:
    def __init__(self, grid, paths,blocked = [],compute_wcd = True, fixed_positions = None, n_changes_so_far = 0):
        # assert wcd_paths[0] in paths[0] and wcd_paths[1] in paths[1]
        self.grid = grid
        self.paths = paths
        self.blocked = blocked
        if fixed_positions is not None:
                self.grid_size, self.goal_positions, self.start_pos = fixed_positions
                self.fixed_positions = fixed_positions
        else:
            self.grid_size, self.goal_positions, _ , self.start_pos = extract_positions(self.grid)
                
        if compute_wcd: # compute True WCD
            self. wcd,self.wcd_paths = self.compute_wcd(paths)
        else:   
            self.init_encoding= encode_grid_design(self.grid_size, self.goal_positions, self.blocked, self.start_pos).float().to(DEVICE) # will be useful in computing predicted
        self.n_changes_so_far = n_changes_so_far
        
    def get_actions(self):
        return self.get_unique_actions(self.paths) # extract uinque actions
    
    def reduce_prune_reduce(self):
        local_paths = self.paths.copy()
        for i in range(len(self.paths)): # for paths to each goal
            new_paths = []
            for j in range(len(self.paths[i])):
                if not (set(self.paths[i][j]) & set(self.blocked)): # no intersections
                    new_paths.append(self.paths[i][j])
            local_paths[i] = new_paths
            
            if len(local_paths[i]) ==0: # ignore this - one goal is not reachable
                return 100, []
        
        wcd, local_wcd_paths =   self.compute_wcd(local_paths)
        
        if wcd <= self.get_wcd():
            children = []
            # get unique actions in wcd_paths and create GRD models for each of them
            for action in self.get_unique_actions(local_wcd_paths):
                if not ("G" in self.grid[action[0],action[1]] or "S" in self.grid[action[0],action[1]] ): # start or goal state cant be blocked
                    
                    children.append(GRDModelBlockingOnly(self.grid,local_paths, self.blocked+[action] ,n_changes_so_far = self.n_changes_so_far+1,fixed_positions = self.fixed_positions))
            return wcd,children
        else:
            return wcd, [] # do not explore this further
        
    def reduce_exhaustive(self):
        children = []
        # get unique actions in wcd_paths and create GRD models for each of them
        for action in self.get_unique_actions(self.paths):
            if not ("G" in self.grid[action[0],action[1]]or "S" in self.grid[action[0],action[1]] ): # start or goal state cant be blocked
                local_paths = self.paths.copy()
                for i in range(len(self.paths)): # for paths to each goal
                    new_paths = []
                    for j in range(len(self.paths[i])):
                        if not (set(self.paths[i][j]) & set([action])): # no intersections
                            new_paths.append(self.paths[i][j])
                    local_paths[i] = new_paths

                    if len(local_paths[i]) ==0: # ignore this - one goal is not reachable
                        local_paths = None
                        break
                if not local_paths is None: # there are paths to each goal
                    children.append(GRDModelBlockingOnly(self.grid,local_paths, self.blocked+[action], n_changes_so_far = self.n_changes_so_far+1, fixed_positions = self.fixed_positions))
        
        return self.get_wcd(),children
    
    def greedy_next_true_wcd(self):
        
        next_child = self
        best_wcd = self.get_wcd()
        
        children = []
        # get unique actions in wcd_paths and create GRD models for each of them
        for action in self.get_unique_actions(self.paths):
            if not ("G" in self.grid[action[0],action[1]]): # goal state cant be blocked
                local_paths = self.paths.copy()
                for i in range(len(self.paths)): # for paths to each goal
                    new_paths = []
                    for j in range(len(self.paths[i])):
                        if not (set(self.paths[i][j]) & set([action])): # no intersections
                            new_paths.append(self.paths[i][j])
                    local_paths[i] = new_paths

                    if len(local_paths[i]) ==0: # ignore this - one goal is not reachable
                        local_paths = None
                        break
                if not local_paths is None: # there are paths to each goal
                    child= GRDModelBlockingOnly(self.grid,local_paths, self.blocked+[action],n_changes_so_far = self.n_changes_so_far+1,fixed_positions = self.fixed_positions )
                    if child.get_wcd()<=best_wcd:
                        next_child = child
                        best_wcd = child.get_wcd()
        
        return next_child
    
    def greedy_next_pred_wcd(self,model):
        
        next_child = self
        best_wcd = self.compute_predicted_wcd(model)
        children = []
        # get unique actions in wcd_paths and create GRD models for each of them
        for action in self.get_unique_actions(self.paths):
            if not ("G" in self.grid[action[0],action[1]]): # goal state cant be blocked
                local_paths = self.paths.copy()
                for i in range(len(self.paths)): # for paths to each goal
                    new_paths = []
                    for j in range(len(self.paths[i])):
                        if not (set(self.paths[i][j]) & set([action])): # no intersections
                            new_paths.append(self.paths[i][j])
                    local_paths[i] = new_paths

                    if len(local_paths[i]) ==0: # ignore this - one goal is not reachable
                        local_paths = None
                        break
                if not local_paths is None: # there are paths to each goal
                    child= GRDModelBlockingOnly(self.grid,local_paths, self.blocked+[action],n_changes_so_far = self.n_changes_so_far+1,compute_wcd= False)
                    child_wcd = child.compute_predicted_wcd(model)
                    
                    if child_wcd<=best_wcd:
                        next_child = child
                        best_wcd = child_wcd
        
        return next_child
        
        
    def compute_wcd(self,paths):
        return compute_wcd_from_paths(paths[0], paths[1], return_wcd_paths = True)
    
    
    def get_unique_actions(self,list_of_sublists):
        flattened_list = [item for sublist in list_of_sublists for list_sublist in sublist for item in list_sublist]
        # Extract unique items from the flattened list
        unique_items = set(flattened_list)
        # Convert the set back to a list if needed
        unique_items_list = list(unique_items)
        return unique_items_list
    
    
    def compute_predicted_wcd(self,model):
        return model(self.init_encoding)
    
    def get_grid(self):
        return self.grid
    
    def get_wcd(self):
        return self.wcd
    
    def get_blocked(self):
        return set(self.blocked)
    
    def get_num_changes_so_far(self):
        return self.n_changes_so_far
    
    def __eq__(self, other):
        if not isinstance(other, GRDModelBlockingOnly):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.get_blocked() == other.get_blocked()

    def __hash__(self):
        # The hash is a unique representation of the object. 
        # For this case, we can use the hash of the frozenset of the blocked items.
        return hash(frozenset(self.blocked))

def breadth_first_search_blocking(root, max_budget,use_exhaustive_search = False):
    # Initialize the BFS queue with the root node
    queue = deque([root])
    visited = set()  # To keep track of visited nodes if needed
    lowest_wcd = root.get_wcd()
    lowest_wcd_node = root

    while queue:
        # Get the next node from the queue
        current_node = queue.popleft()

        # Here, we assume that we have a method to check if the node is a goal state
        # For instance, we could check if current_node.wcd is a desirable value
        if current_node.get_wcd() == 0 or current_node.get_num_changes_so_far() == max_budget:
            # If the goal state is reached, return the current node or path
            return current_node

        # If the current node is not the goal state, expand the search
        if use_exhaustive_search:
            wcd, children = current_node.reduce_exhaustive()
        else:
            wcd, children = current_node.reduce_prune_reduce()
            
        if current_node.get_wcd() < lowest_wcd:
            lowest_wcd = wcd
            lowest_wcd_node = current_node
        for child in children:
            # We may want to add more sophisticated visited checks
            # For now, we'll just check if the wcd is in visited
            if not (child in visited):
                queue.append(child)
                visited.add(child)  # Add the child's wcd to the visited set
        # print(len(queue))
    # If no solution is found, return None or an appropriate indication
    return lowest_wcd_node

def greedy_search_true_wcd_blocking_only(root, max_budget):
    current_node = root
    parent = root

    while True:
        # Get the next node from the queue
        parent = current_node
        current_node = current_node.greedy_next_true_wcd()
        if current_node.__eq__(parent):
            return current_node
        elif current_node.get_num_changes_so_far() == max_budget:
            return current_node
    return parent

def greedy_search_pred_wcd_blocking_only(root, max_budget,model):
    current_node = root
    parent = root

    while True:
        # Get the next node from the queue
        parent = current_node
        current_node = current_node.greedy_next_pred_wcd(model)
        if current_node.__eq__(parent):
            return current_node
        elif current_node.get_num_changes_so_far() == max_budget:
            return current_node
    return parent


def greedy_search_true_wcd_all_mods(root, max_budget, all_mods_uniform = False): # all actions
    current_node = root
    parent = root
    while True:
        # Get the next node from the queue
        parent = current_node
        current_node = current_node.greedy_next_true_wcd(all_mods_uniform)
        if current_node.__eq__(parent):
            return current_node
        
        if all_mods_uniform: 
            if np.sum(current_node.get_num_changes_so_far()) >= np.sum(max_budget):
                return current_node
        else:
            num_changes = current_node.get_num_changes_so_far()
            if  num_changes[0]>= max_budget[0] and num_changes[1]>= max_budget[1] :
                return current_node
    return parent

def greedy_search_predicted_wcd_all_mods(root, model, max_budget, all_mods_uniform = False):
    current_node = root
    parent = root
    while True:
        # Get the next node from the queue
        parent = current_node
        current_node = current_node.greedy_next_predicted_wcd(model,all_mods_uniform)
        if current_node.__eq__(parent):
            return current_node
        
        if all_mods_uniform: 
            if np.sum(current_node.get_num_changes_so_far()) >= np.sum(max_budget):
                return current_node
        else:
            num_changes = current_node.get_num_changes_so_far()
            if  num_changes[0]>= max_budget[0] and num_changes[1]>= max_budget[1] :
                return current_node
            
    return parent

def breadth_first_search_all_actions(root, max_budget,all_mods_uniform = False):
    # Initialize the BFS queue with the root node
    queue = deque([root])
    visited = set()  # To keep track of visited nodes if needed
    lowest_wcd = root.get_wcd()
    lowest_wcd_node = root

    while queue:
        # Get the next node from the queue
        current_node = queue.popleft()

        # Here, we assume that we have a method to check if the node is a goal state
        
        if all_mods_uniform:
            if current_node.get_wcd() == 0 or np.sum(current_node.get_num_changes_so_far()) >= np.sum(max_budget):
                return current_node
        else:
            # For instance, we could check if current_node.wcd is a desirable value
            num_changes = current_node.get_num_changes_so_far()
            
            if current_node.get_wcd() == 0 or (num_changes[0]>= max_budget[0] and num_changes[1]>= max_budget[1]):
                # If the goal state is reached, return the current node or path
                return current_node

        # If the current node is not the goal state, expand the search
        wcd, children = current_node.reduce_exhaustive()
        if wcd < lowest_wcd:
            lowest_wcd = wcd
            lowest_wcd_node = current_node
        
        for child in children:
            # We may want to add more sophisticated visited checks
            # For now, we'll just check if the wcd is in visited
            if not (child in visited):
                queue.append(child)
                visited.add(child)  # Add the child's wcd to the visited set
        # print(len(queue))
    # If no solution is found, return None or an appropriate indication
    return lowest_wcd_node