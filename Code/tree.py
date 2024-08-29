import numpy as np
import random as rd
import time
from param import RRTParameters, AgentParameters
from collections import deque
import copy
import statistics

from functions.a_star_v2 import *
from functions.entropy_functions import get_stdev_error
from functions.intersection import get_prob_intersecting, bird_dist_check, man_dist_check
from functions.get_meet_pos_distrib import get_agent_distrib, get_meet_pos_distrib
from functions.get_meet_pos_distrib_plans import get_agent_distrib_plans
from functions.get_meet_pos_distrib_mc_separate import get_agent_distrib_mc_presence, get_agent_distrib_mc_supress, get_agent_distrib_mc_compil, build_agent_distrib_mc_presence_memo
from functions.meeting import get_prob_meeting
from functions.merge_segments import merge_segments
from virtual_agent2 import VirtualAgent

np.random.seed(1)
rd.seed(1)


class RRT:
    def __init__(self, id, submaps) -> None:
        self.id = id

        #external vaiables
        self.submaps = copy.deepcopy(submaps)

        #variables
        self.ideal_n_nodes = None
        self.bias_map = None
        self.penalty_points = []
        
        #rrt parameters
        self.max_extend_tries = 50
        self.max_n_nodes = RRTParameters.MAX_N_NODES
        self.min_n_nodes = RRTParameters.MIN_N_NODES
        self.nodes_ratio = RRTParameters.NODES_RATIO
        self.min_new = RRTParameters.MIN_NEW
        self.ideal_nearby_nodes = RRTParameters.IDEAL_NEARBY_NODES
        self.sampling_max_range = RRTParameters.SAMPLING_MAX_RANGE
        self.sampling_method = RRTParameters.SAMPLING_METHOD
        self.bias_rate = RRTParameters.BIAS_RATE

        self.rrt_method = RRTParameters.RRT_METHOD
        self.min_dist_b_nodes = RRTParameters.MIN_DIST_B_NODES
        self.max_edge_length = RRTParameters.MAX_EDGE_LENGHT
        self.min_edge_length = RRTParameters.MIN_EDGE_LENGHT
        self.neighbourhood_distance = RRTParameters.NEIGHBOURHOOD_DISTANCE

        #calculate multiverse
        self.calculate_mv_loop = RRTParameters.CALCULATE_LOOP
        self.calculate_mv_meet = RRTParameters.CALCULATE_MEET

        #meet impact
        self.distrib_method = RRTParameters.DISTRIB_METHOD
        self.mc_memo_method = RRTParameters.MC_MEMO
        self.lost_thres = RRTParameters.LOST_STEPS
        self.impact_thres = RRTParameters.IMPACT_STEPS
        self.trust_plan_factor = RRTParameters.TRUST_PLAN_FACTOR
        self.scan_range = AgentParameters.RANGE

        #get loop prob function calculation
        self.odom_error_rate = AgentParameters.ODOM_ERROR_RATE
        self.segment_size = 4
        self.mc_init_max_try = 10
        self.mc_n_simulations = 100
        self.dl_memo = {}
        self.prob_loop_threshold = 0.1
        self.qc_threshold = 0.99
        self.prob_meet_threshold = 0.01

        self.start = self.submaps.ag_pos
        self.reset_tree(self.start)
        self.reset_metrics(0)
        self.reset_rrt_exec_time_dic()

    def reset_tree(self, start):
        self.tree = {start : []}
        self.reset_tree_lists()
        self.reset_tree_paths()
        self.reset_tree_plans()
        self.reset_action_variables()
    
    def reset_tree_lists(self):
        self.unaccessible_nodes = []
        self.obsolete_nodes = []
        self.last_nodes_added = []
        self.nodes_to_update = []
        self.no_path_points = []
        self.need_to_rebuild_paths = False
        self.tree_trajectory_points = []
        self.nodes_surroundings_points = []

    def reset_tree_paths(self):
        self.node_parent = {}
        self.path_from_parent = {}
        self.node_depth = {}
        self.node_path = {}
        self.path_to_node = {}
        self.leaves = []
    
    def reset_tree_plans(self):
        self.multiverse = {}
        self.information_gain = {}
        
        #external variables
        self.team_plans = None

        #mc_memo
        self.team_distrib_presence_memo = None
        self.team_distrib_presence_null = None

    def reset_action_variables(self):
        self.has_set_action = None
        self.new_target = None
        self.new_node_path = None
        self.expected_gain = None
        self.new_goal_set = None
        self.new_path_set = None

    def reset_metrics(self, curr_step):
        #metrics
        self.rrt_metrics = {
            'step' : curr_step,
            'tree' : {},
            'start' : None,
            'unaccessible_nodes' : [],
            'obsolete_nodes' : [],
            'penalty_points' : [],
            'last_nodes_added' : [],
            'nodes_to_update' : [],
            'temporary_banned_points' : [],
            'no_path_points' : [],
            'need_to_rebuild_paths' : False,

            'n_nodes' : len(self.tree),
            'ideal n_nodes' : self.ideal_n_nodes,

            'node_parent' : {},
            'path_from_parent' : {},
            'node_depth' : {},
            'node_path' : {},
            'path_to_node' : {},
            'leaves' : [],

            'multiverse' : {},
            'information gain' : {},

            'has_set_action' : None,
            'new_target' : None,
            'new_node_path' : None,
            'expected_gain' : None,
            'new_goal_set' : None,
            'new_path_set' : None,
        }

    def reset_rrt_exec_time_dic(self):
        self.rrt_exec_time = {
            'update external variables' : [],
            'update rrt' : [],
            'unaccessible nodes' : [],
            'add start' : [],
            'new tree' : [],
            'extend tree' : [],
            'leaves' : [],
            'update information' : [],
            'update information - build memo' : [],
            'update information-n_iter' : [],
            'get multiverse' : [],
            'get multiverse-1' : [],
            'get multiverse-2' : [],
            'get multiverse-3' : [],
            'get multiverse-n_va' : [],
            'create_va' : [],
            'virtual_move' : [],
            'virtual_correction' : [],
            'virtual_meeting' : [],
            'update_entropy' : [],
            'get loop prob' : [],
            'get loop prob-n_iter' : [],
            'get loop prob1' : [],
            'get loop prob2' : [],
            'get loop prob3dev' : [],
            'get loop prob3f' : [],
            'get loop prob3p' : [],
            'get meet prob' : [],

            'get weighted_sit' : [],
            'get IG' : [],
            'get future' : [],
        }

        self.rrt_exec_stats = {}


    def is_node(self, point):
        return point in self.tree.keys()
    
    def manhattan_dist(self, pos1, pos2):
        return abs(pos2[0] - pos1[0]) + abs(pos2[1] - pos1[1])

    def get_nearest_node(self, point, banned = []):
        min_dist = float('inf')
        nearest_node = None
        for node in self.tree.keys():
            if node not in banned:
                dist = self.manhattan_dist(point, node)
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node
        return nearest_node
    
    def get_neighbourhood(self, point, neighbourhood_distance, banned = []):
        neighbourhood = []
        for node in self.tree.keys():
            if node not in banned:
                dist = self.manhattan_dist(point, node)
                if dist <= neighbourhood_distance:
                    neighbourhood.append(node)
        return neighbourhood
    
    def get_path(self, from_point, to_point, disp = False):
        if disp : print('Getting path from point :', from_point, 'to point :', to_point, '...')
        maze = self.submaps.get_maze1()
        path, n_operations = a_star(maze, from_point, to_point)

        if path:
            if disp : print('Path calculated :', path)
            return path
        else:
            #no path can be found
            print(" error - path to point", to_point,  "cannot be found")
            self.no_path_points.append(to_point)
            return False
        
    def add_child_node(self, parent, new_point):
        if not self.is_node(parent):
            print('error - parent not in tree')
        self.tree[parent].append(new_point)
        self.tree[new_point] = []

    def add_parent_node(self, new_point, child):
        if not self.is_node(child):
            print('error - child not in tree')
        self.tree[new_point] = [child]

    def del_node(self, node):
        if self.is_node(node):
            del self.tree[node]
            
            if node in self.node_parent : del self.node_parent[node]
            if node in self.path_from_parent : del self.path_from_parent[node]
            if node in self.node_depth : del self.node_depth[node]
            if node in self.node_path : del self.node_path[node]
            if node in self.path_to_node : del self.path_to_node[node]
            if node in self.leaves : self.leaves.remove(node)
            if node in self.multiverse : del self.multiverse[node]
            if node in self.information_gain : del self.information_gain[node]
    
    def clear_nodes(self, disp = False):
        if disp : print('Clearing nodes ...')
        if disp : print('tree :', self.tree)

        #DFS
        visited = []
        stack = deque()
        stack.append(self.start)

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.append(node)
                successors = self.get_successors(node)
                if successors : 
                    unvisited = [n for n in successors if n not in visited]
                    stack.extend(unvisited)
        
        #clear
        curr_nodes = list(self.tree.keys())
        for node in curr_nodes:
            if node not in visited:
                self.del_node(node)


    def clear_branches(self, disp = False):
        if disp : print('Clearing branches ...')
        if disp : print('tree :', self.tree)

        #node list
        nodes_list = self.tree.keys()
        if disp : print('nodes list :', nodes_list)

        #DFS
        visited = []
        stack = deque()
        stack.append(self.start)

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.append(node)
                if disp : print('visited node :', node)

                for node_to_remove in self.tree[node][:]:
                    if node_to_remove not in nodes_list or node_to_remove in visited:
                        self.tree[node].remove(node_to_remove)

                unvisited = [n for n in self.get_successors(node) if n not in visited]
                stack.extend(unvisited)


    def build_tree(self, parent_node, new_node, path, reverse_edge = False, disp = False):
        if disp : print("Building tree with new_node", new_node, ':')
        #check if node already exists
        if self.is_node(new_node):
            print('error - node selected', new_node, 'already exists')
            return False
        
        #add node
        if not reverse_edge:
            self.add_child_node(parent_node, new_node)
            self.node_parent[new_node] = parent_node
            self.path_from_parent[new_node] = path

            self.last_nodes_added.append(new_node)
            self.extend_tree_trajectory_points(path)
            self.extend_nodes_surroundings_points(new_node)
        else:
            path.reverse()
            self.add_parent_node(new_node, parent_node)
            self.node_parent[parent_node] = new_node
            self.path_from_parent[parent_node] = path
            self.node_parent[new_node] = None
            self.path_from_parent[new_node] = None

            self.last_nodes_added.append(new_node)
            self.nodes_to_update.append(parent_node)
            self.extend_tree_trajectory_points(path)
            self.extend_nodes_surroundings_points(new_node)
        if disp : print('point added :', new_node,'/ nearest node :', parent_node, '/ path :', path)
        return True
    
    def rewire(self, node, past_parent, new_parent, path, disp = False):
        if disp : print("Rewire node", node, ' from past parent', past_parent, 'to new parent', new_parent)
        self.tree[past_parent].remove(node)
        self.tree[new_parent].append(node)
        self.node_parent[node] = new_parent
        self.path_from_parent[node] = path

        children = self.get_children(node)
        self.nodes_to_update.extend(children)
        self.extend_tree_trajectory_points(path)

    def get_successors(self, node):
        if self.is_node(node):
            return self.tree[node]
        else:
            return False
    
    def get_children(self, starting_node):
        #DFS
        visited = []
        stack = deque()
        stack.append(starting_node)

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.append(node)
                successors = self.get_successors(node)
                unvisited = [n for n in successors if n not in visited]
                stack.extend(unvisited)
        return visited
    
    def get_parent(self, node):
        for n in self.tree:
            if node in self.get_successors(n):
                return n
        print('error - no parent')
        return False
    
    def get_lineage(self, node):
        no = node
        lineage = []
        k = 0
        while k<100:
            k+=1
            no = self.node_parent[no]
            if not no:
                print('error - lineage not found')
                return False
            
            lineage.append(no)
            if no == self.start:
                return lineage
            

        print('error - lineage not found')
        return False
    
    def get_node_path(self, node):
        lineage = self.get_lineage(node)
        if lineage:
            lineage.reverse()
            return lineage + [node]
        else:
            return False
        
    def get_path_to_node(self, node):
        path_to_node = []
        node_path = self.node_path[node]      
        for n in range(1, len(node_path)):
            path_to_node += self.path_from_parent[node_path[n]][:-1]
        path_to_node += [node]
        return path_to_node
    

    #1. RRT functions
    #a.
    def check_node_unreachable_o_update(self, node, disp = False):
        if node not in self.tree:
            if disp : print('node not in tree')
            return None
        if node == self.start:
            if disp : print('node is start')
            return False
        
        for pos in self.path_from_parent[node][1:-1]:
            if self.submaps.is_obstacle(pos) or self.submaps.is_out(pos):
                if disp : print(' pos', pos, 'on obstacle or out')
                possible_new_path = self.get_path(self.node_parent[node], node)
                if possible_new_path and len(possible_new_path)-1 <= self.max_edge_length +2 and len(possible_new_path)-1 >= self.min_edge_length: #+2 corresponds the the length of the path to bypass a single obstacle
                    if disp : print('possible new path :', possible_new_path, 'accepted') 
                    #update
                    self.path_from_parent[node] = possible_new_path
                    children = self.get_children(node)
                    self.nodes_to_update.extend(children)
                    return False
                else:
                    if disp : print('possible new path :', possible_new_path, 'rejected') 
                    return True
        return False

    def chack_n_get_unaccessible_nodes(self, disp):
        if disp : print('Getting nodes on obstacles ...')

        nodes_on_obstacles = []
        nodes_unreachable = []

        #DFS
        visited = []
        stack = deque()
        stack.append(self.start)

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.append(node)
                if self.submaps.is_obstacle(node) or self.submaps.is_out(node):
                    nodes_on_obstacles.append(node)
                    if disp : print('node', node, 'on obstacle or out')
                elif self.check_node_unreachable_o_update(node, disp):
                    nodes_unreachable.append(node)
                    if disp : print('node', node, 'unreachable')
                else:
                    successors = self.get_successors(node)
                    unvisited = [n for n in successors if n not in visited]
                    stack.extend(unvisited)
        
        return nodes_on_obstacles+nodes_unreachable

    def get_weak_leaves(self, n_leaves):
        leaves_dict = {}
        weak_leaves_list = []
        for leaf in self.leaves:
            leaves_dict[leaf] = self.information_gain[leaf]['from_parent']
        
        for _ in range(n_leaves):
            if leaves_dict != {}:
                min_leaf = min(leaves_dict, key=leaves_dict.get)
                weak_leaves_list.append(min_leaf)
                leaves_dict.pop(min_leaf)
        return weak_leaves_list
    
    def del_nodes_list(self, remove_list, disp = False):
        if disp : print('Removing nodes :', remove_list , '...')
        for remove_node in remove_list:
            if self.is_node(remove_node):
                self.del_node(remove_node)
        
    #b.
    def add_root_node(self, new_start, disp = False):
        if disp : print('Adding starting node', new_start, '...')
        new_set = self.RRT_new_node(new_start, forced = True, disp = disp)
        if new_set:
            child_node, new_node, path = new_set
            has_built = self.build_tree(child_node, new_node, path, reverse_edge = True, disp = disp)
            return has_built
        else:
            return False

    #c1.
    def get_tree_trajectory_points(self):
        trajectory_points = []
        #DFS
        visited = []
        stack = deque()
        stack.append(self.start)

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.append(node)
                if node in self.path_from_parent and self.path_from_parent[node]:
                    trajectory_points += list(self.path_from_parent[node])
                successors = self.get_successors(node)
                unvisited = [n for n in successors if n not in visited]
                stack.extend(unvisited)
        return trajectory_points
    
    def extend_tree_trajectory_points(self, trajectory):
        self.tree_trajectory_points.extend(trajectory)

    def get_1node_surroundings_points(self, node, surr_range):
        node_surr = []
        for i in range(-surr_range, surr_range+1):
            for j in range(-surr_range, surr_range+1):
                point = (node[0]+i,node[1]+j)
                if self.submaps.on_submap(point):
                    if self.manhattan_dist(node, point) <= surr_range:
                        node_surr.append(point)
        return node_surr
            
    def get_nodes_surroundings_points(self):
        nodes_surroundings = []
        surr_range = max(self.min_dist_b_nodes, 0)
        for node in self.tree:
            node_surr = self.get_1node_surroundings_points(node, surr_range)
            nodes_surroundings.extend(node_surr)
        return nodes_surroundings

    def extend_nodes_surroundings_points(self, node):
        surr_range = max(self.min_dist_b_nodes -1, 0)
        node_surr = self.get_1node_surroundings_points(node, surr_range)
        self.nodes_surroundings_points.extend(node_surr)

    #c2.
    def get_rd_point(self, max_range = False, banned_points = []):
        list_e = self.submaps.get_ext_map_points_list(max_range = max_range)
        for b_w in banned_points:
            if b_w in list_e:
                list_e.remove(b_w)
        if list_e == []:
            return None
        else:
            return rd.choice(list_e)
        
    def get_rd_point_in_known_space(self, max_range = False, banned_points = []):
        list_k = self.submaps.get_known_points_list(max_range = max_range)
        for b_w in banned_points:
            if b_w in list_k:
                list_k.remove(b_w)
        if list_k == []:
            return None
        else:
            s=0
            while True:
                pos = rd.choice(list_k)
                if self.submaps.is_free(pos):
                    return pos
                s+=1
                if s>10:
                    print("error - random goal in known space failed")
                    return None
            
    def get_rd_point_in_frontier_space(self, max_range = False, banned_points = []):
        list_f = self.submaps.get_frontier_points_list(max_range = max_range)
        for b_w in banned_points:
            if b_w in list_f:
                list_f.remove(b_w)
        if list_f == []:
            return None
        else:
            return rd.choice(list_f)
        
    def get_point_w_bias(self, bias_map = None, max_range = False, banned_points = []):
        list_e = self.submaps.get_ext_map_points_list(max_range = max_range)
        for b_w in banned_points:
            if b_w in list_e:
                list_e.remove(b_w)
        if list_e == []:
            return None
        else:
            #get weights
            pos_weights = []
            for elem in list_e:
                pos_weights.append(max(bias_map[elem[0], elem[1]], 0))
            if max(pos_weights) > 0:
                return rd.choices(list_e, weights = pos_weights, k=1)[0]
            else:
                return rd.choice(list_e)

    #c3
    def RRT_new_node(self, point, forced = False, disp = False):
        if disp : print("RRT algorithm : finding new node with point", point, ':')
        #get neighbour and path to a new node
        k=0
        banned_neighbours = []
        while True:
            nearest_node = self.get_nearest_node(point, banned_neighbours)
            if nearest_node:
                path = self.get_path(nearest_node, point)
                if path:
                    #shortcut path/steer node
                    if forced or (len(path)-1 <= self.max_edge_length and len(path)-1 >= self.min_edge_length):
                        new_node = point
                        break
                    elif len(path)-1 < self.min_edge_length:
                        print('error - point', point,' too close to nearest node', nearest_node)
                        return False
                    else: #len(path) > self.max_edge_length
                        #chose a new point and check that the nearest node hasn't changed
                        new_node = path[self.max_edge_length]
                        path = path[:self.max_edge_length+1]
                        if self.get_nearest_node(new_node, banned_neighbours) == nearest_node:
                            break
                        else:
                            banned_neighbours.append(nearest_node)
                            print('error - neighbour', nearest_node, 'is not the nearest anymore')
                else:
                    banned_neighbours.append(nearest_node)
                    print('error - neighbour', nearest_node, 'is not accessible')
                    if k == 2:
                        print('error - no neighbours accessible')
                        return False
                
                k+=1
                if k == 5:
                    print('error - no good neighbour found')
                    return False
                
            else:
                print('error - no neighbour found')
                return False

        return nearest_node, new_node, path
        
    def RRT_star_new_node(self, point, disp = False):
        if disp : print("RRT star algorithm : finding new node with point", point, ':')
        if disp : print('tree :', self.tree)
        #get neighbourhood
        neighbourhood = self.get_neighbourhood(point, self.neighbourhood_distance)
        if disp : print('neighbourhood :', neighbourhood)
        
        #get list of parent node and path candidates
        if neighbourhood != []:
            
            #init candidates list
            candidates_list = []
            for neighbour in neighbourhood:
                if disp : print('neighbour :' , neighbour)
                path = self.get_path(neighbour, point)
                if path:
                    #shortcut path/steer node
                    if len(path)-1 < self.min_dist_b_nodes:
                        print('error - point', point,' too close to nearest neighbour', neighbour)
                        return False
                    
                    elif len(path)-1 <= self.max_edge_length and len(path)-1 >= self.min_dist_b_nodes:
                        candidates_list.append((neighbour, point, path))
                        if disp : print('neighbour', neighbour, 'appent to candidates list')
                    
                    else: #len(path) > self.max_edge_length
                        new_point = path[self.max_edge_length]
                        path = path[:self.max_edge_length+1]
                        if disp : print('shortcut : new point :', new_point)
                        nearest_node = self.get_nearest_node(new_point)
                        if self.manhattan_dist(new_point, nearest_node) >= self.min_dist_b_nodes:
                            candidates_list.append((neighbour, new_point, path))
                            if disp : print('neighbour', neighbour, 'appent to candidates list after shortcut')
                        else:
                            if disp : print('new point was too close to', nearest_node)

            #find the best candidate
            if candidates_list != []:
                if disp : print('candidates_list :', candidates_list)
                min_dist = float('inf')
                best_candidate = None
                for candidate in candidates_list:
                    parent = candidate[0]
                    new_node = candidate[1]
                    path = candidate[2]
                    if parent in self.path_to_node:
                        if disp : print(self.path_to_node[parent])
                        dist = len(self.path_to_node[parent]) -1 + self.manhattan_dist(parent, new_node)
                        if disp : print('parent :', parent, '; h dist :', dist, min_dist)
                    else:
                        dist = self.manhattan_dist(self.start, new_node)
                        if disp : print('parent :', parent, '; man dist :', dist, min_dist)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_candidate = candidate
                
                if len(best_candidate[2]) -1 >= self.min_edge_length:
                    if disp : print('best candidate found :', best_candidate)
                    return best_candidate
                else:
                    print('error - best candidate has too small edge lenght')
                    return False
            else:
                print('error - no candidate found')
                return False
        else:
            print('error - no neighbourhood found')
            return self.RRT_new_node(point, disp = disp)
    
    #c4.
    def rewire_around_point(self, point, disp = False):
        if disp : print('Rewire around point', point, '(new node) ...')
        if disp : self.display_tree()
        
        #get neighbourhood
        neighbourhood = self.get_neighbourhood(point, self.neighbourhood_distance)
        if neighbourhood != []:
            if disp : print('neighbourhood :', neighbourhood)

            #init rewire list
            rewire_list = []
            for oth_node in [elem for elem in neighbourhood if elem != point and elem != self.start]: #only considering nodes with parent and inside the neighbourhood
                current_parent = self.node_parent[oth_node]
                if disp : print('oth_node :', oth_node)
                if disp : print('current_parent :', current_parent)
                if oth_node in self.path_to_node:
                    if disp : print(self.path_to_node[oth_node])
                    current_dist = len(self.path_to_node[oth_node])-1
                    if disp : print('current dist to node', oth_node, '():', current_dist)
                elif current_parent in self.path_to_node:
                    if disp : print(self.path_to_node[current_parent])
                    current_dist = len(self.path_to_node[current_parent])-1 + self.manhattan_dist(current_parent, oth_node)
                    if disp : print('current dist to node', oth_node, '()+h:', current_dist)
                else:
                    current_dist = self.manhattan_dist(self.start, point)
                    if disp : print('current dist to node', oth_node, 'man:', current_dist)

                #init best parent and dist
                best_parent = current_parent
                best_dist = current_dist
                new_path = None
                for possible_new_parent in [elem for elem in neighbourhood if elem not in [oth_node, current_parent]]:
                    if self.manhattan_dist(oth_node, possible_new_parent) <= self.max_edge_length or possible_new_parent == self.start: #pre test #exception in max edge length if the rewire is with the starting node
                        if disp : print('possible new parent :', possible_new_parent)
                        if possible_new_parent in self.path_to_node:
                            if disp : print('path to possible_new_parent', self.path_to_node[possible_new_parent])
                            parent_dist = len(self.path_to_node[possible_new_parent])-1
                            if disp : print('possible new dist :', parent_dist)
                        else:
                            parent_dist = self.manhattan_dist(self.start, possible_new_parent)
                            if disp : print('possible new dist (man):', parent_dist)
                        if parent_dist < best_dist: #pre test
                            path = self.get_path(possible_new_parent, oth_node)
                            if disp : print('possible path from new_parent:', path)
                            if path and (len(path)-1 <= self.max_edge_length or possible_new_parent == self.start): #exception in max edge length if the rewire is with the starting node
                                new_dist = parent_dist + len(path)-1
                                if disp : print('new dist to node :', oth_node, 'from parent', possible_new_parent,':', new_dist)
                                if new_dist < best_dist: #update best parent and dist
                                    best_dist = new_dist
                                    best_parent = possible_new_parent
                                    new_path = path
                                    if disp : print('parent', possible_new_parent, 'is better')
                
                #append to list if a better parent has been found
                if best_parent != current_parent:
                    #rewire
                    self.rewire(oth_node, current_parent, best_parent, new_path, disp)
                    succession = self.get_children(oth_node)
                    for node in [oth_node]+succession:
                        self.calculate_node_path(node, new_calculation = True)

                    rewire_list.append({'node':oth_node, 'curr':current_parent, 'best':best_parent, 'path': new_path})
                    if disp : print('better parent ', best_parent, 'has been found for node', oth_node, 'with dist', best_dist)
            
            if rewire_list != []:
                return rewire_list
            else:
                if disp : print('No need for rewire')
                return False
        else:
            if disp : print('No neighbourhood')
            return False

    def RRT_star_cascade_rewire(self, init_node, disp = False):
        rew_visited = []
        rew_queue = deque()
        rew_queue.append(init_node)
        k=0
        while rew_queue:
            k+=1
            if k>100: 
                print('error - infinite rewire')
                break
            rew_node = rew_queue.pop()
            if rew_node not in rew_visited:
                rew_visited.append(rew_node)
                rewire_set = self.rewire_around_point(rew_node, disp)
                if rewire_set:
                    rew_queue.extend([rn['node'] for rn in rewire_set])
    #cf.
    def is_point_valid(self, point):
        return not self.submaps.is_obstacle(point) and not self.submaps.is_out(point)
    
    def calculate_node_path(self, node, new_calculation = True):
        if node != self.start:
            if new_calculation:
                self.node_path[node] = self.get_node_path(node)
                self.path_to_node[node] = self.get_path_to_node(node)

            else:
                try:
                    start_index = int(self.node_path[node].index(self.start))
                    self.node_path[node] = self.node_path[node][start_index:]
                except ValueError:
                    self.node_path[node] = self.get_node_path(node)
                
                try:
                    start_index = int(self.path_to_node[node].index(self.start))
                    self.path_to_node[node] = self.path_to_node[node][start_index:]
                except ValueError:
                    self.path_to_node[node] = self.get_path_to_node(node)

            self.node_depth[node] = len(self.node_path[node]) - 1
            #self.node_depth[node] = self.node_depth[parent] + 1

    def extend_RRT(self, disp = False):
        if disp : print('Extending RRT with sampling method', self.sampling_method, 'and rrt method', self.rrt_method, '...')
        k=0
        banned_points = []
        while len(self.tree) < self.ideal_n_nodes and k < max(self.max_extend_tries, 3*self.ideal_n_nodes):
            k+=1

            #set the banned points
            banned_points = list(set(self.tree_trajectory_points + self.nodes_surroundings_points + banned_points + self.penalty_points))
            if disp : print('banned points :', banned_points)

            #set nearby nodes and not banned points
            nearby_points = self.submaps.get_points_list_in_range(self.start, self.max_edge_length)
            nearby_not_banned = [elem for elem in nearby_points if elem not in banned_points]

            nearby_nodes = [elem for elem in self.tree if elem in nearby_points]
            
            #set max_range
            if nearby_not_banned != [] and len(nearby_nodes) < self.ideal_nearby_nodes:
                max_range = self.max_edge_length
            else:
                max_range = self.sampling_max_range
            
            if self.sampling_method == 'known': point = self.get_rd_point_in_known_space(max_range = max_range, banned_points = banned_points)
            elif self.sampling_method == 'explore': point = self.get_rd_point_in_frontier_space(max_range = max_range, banned_points = banned_points)
            elif self.sampling_method == 'bias': point = self.get_point_w_bias(self.bias_map, max_range = max_range, banned_points = banned_points)
            else: point = self.get_rd_point(max_range = max_range, banned_points = banned_points)
            
            if disp : print('point chosen :', point)
            if not point: #no more points
                break

            #init boolean
            has_built = None
            if self.is_node(point): 
                print('ERROR THAT SHOULD NEVER HAPPEN - A NODE HAS BEEN TAKEN AS POINT')
            
            #build
            if self.is_point_valid(point) and not self.is_node(point) and point not in self.penalty_points:
                #add this point to the tree if possible
                if self.rrt_method == 'rrt':
                    new_set = self.RRT_new_node(point, disp = disp)
                    if new_set:
                        parent_node, new_node, path = new_set
                        if new_node not in self.penalty_points:
                            has_built = self.build_tree(parent_node, new_node, path, disp = disp)                         

                elif 'rrt_star' in self.rrt_method: 
                    new_set = self.RRT_star_new_node(point, disp = disp)
                    if new_set:
                        parent_node, new_node, path = new_set
                        if new_node not in self.penalty_points:
                            has_built = self.build_tree(parent_node, new_node, path, disp = disp)
            else:
                if disp : print('point', point, 'is not valid')

            #calculate, rewire or ban
            if has_built:
                self.calculate_node_path(new_node, new_calculation = True)

                if 'rewire' in self.rrt_method:
                    self.RRT_star_cascade_rewire(new_node, disp = disp)

            else: #ban the point if not valid
                surr_range = max(self.min_dist_b_nodes -1, 0)
                banned_points.extend(self.get_1node_surroundings_points(point, surr_range))
                if disp : print('point', point, 'added to temporary banned points')

        if disp : print('RRT extended. Nodes added :', self.last_nodes_added)
    
    #d.
    def get_leaves(self):
        leaves = []

        #DFS
        visited = []
        stack = deque()
        stack.append(self.start)

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.append(node)
                successors = self.get_successors(node)
                if successors == []:
                    leaves.append(node)
                unvisited = [n for n in successors if n not in visited]
                stack.extend(unvisited)
        return leaves

    #e.
    def calculate_tree_paths(self, reset_paths = False, recalculate = False, obsolete_o_new_nodes = [], disp = False):
        if disp : print('Calculating tree paths ...')
        
        if reset_paths:
            self.reset_tree_paths()

        #init
        self.node_parent[self.start] = None
        self.path_from_parent[self.start] = None
        self.node_depth[self.start] = 0
        self.node_path[self.start] = [self.start]
        self.path_to_node[self.start] = [self.start]

        #DFS
        visited = []
        stack = deque()
        stack.append(self.start)
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.append(node)

                #
                if disp : print('visited node :', node)
                #

                if node != self.start:
                    if reset_paths:
                        self.path_from_parent[node] = self.get_path(self.node_parent[node], node)
                    if recalculate or node in obsolete_o_new_nodes:
                        self.node_parent[node] = self.get_parent(node)
                        self.node_path[node] = self.get_node_path(node)
                        self.path_to_node[node] = self.get_path_to_node(node)

                    else:
                        if self.node_parent[node] != self.get_parent(node): print('error - NEED TO CHANGE PARENT!!')
                        try:
                            start_index = int(self.node_path[node].index(self.start))
                            self.node_path[node] = self.node_path[node][start_index:]
                        except ValueError:
                            self.node_path[node] = self.get_node_path(node)
                        
                        try:
                            start_index = int(self.path_to_node[node].index(self.start))
                            self.path_to_node[node] = self.path_to_node[node][start_index:]
                        except ValueError:
                            self.path_to_node[node] = self.get_path_to_node(node)

                    self.node_depth[node] = len(self.node_path[node]) - 1
                    #self.node_depth[node] = self.node_depth[parent] + 1

                successors = self.get_successors(node)
                unvisited = [n for n in successors if n not in visited]
                stack.extend(unvisited)
    
    ###
    def update_rrt(self, disp = False, measure_time = True):
        print('Updating tree ...')
        if measure_time : trc = time.time()

        #reset lists
        self.reset_tree_lists()
        
        #remove unaccessible nodes
        self.unaccessible_nodes = self.chack_n_get_unaccessible_nodes(disp = disp)
        self.del_nodes_list(list(set(self.unaccessible_nodes+self.penalty_points)), disp)

        #set new tree
        if measure_time : trc_n = time.time()
        new_start = self.submaps.ag_pos
        if disp : print('Setting new tree with start', new_start, '...')

        #check and add agent's pos as node to the tree if needed
        if not self.is_node(new_start):
            has_added_start = self.add_root_node(new_start, disp)
            if not has_added_start:
                self.reset_tree(new_start)
                if disp : print("Tree reset")
        else:
            has_added_start = False

        self.start = new_start

        #clear and recalculate
        self.clear_nodes()
        self.clear_branches()
        self.calculate_tree_paths(recalculate = has_added_start, obsolete_o_new_nodes = self.nodes_to_update, disp = disp)

        #rewire (optional)
        if 'rewire' in self.rrt_method:
            self.RRT_star_cascade_rewire(self.start, disp)

        if disp : print('New tree set')
        if disp : self.display_tree()
        if measure_time : self.rrt_exec_time['new tree'].append(round(time.time() - trc_n, 3))

        #remove weak leafs
        if self.ideal_n_nodes - len(self.tree) < self.min_new:
            self.weak_leafs = self.get_weak_leaves(min(max(0, self.ideal_n_nodes - len(self.tree)), self.min_new))
            self.del_nodes_list(list(set(self.weak_leafs)), disp)
            self.clear_nodes()
            self.clear_branches()
            self.calculate_tree_paths(recalculate = has_added_start, obsolete_o_new_nodes = self.nodes_to_update, disp = disp)
        
        #add new nodes
        if measure_time : trc_e = time.time()
        self.tree_trajectory_points = self.get_tree_trajectory_points()
        self.nodes_surroundings_points = self.get_nodes_surroundings_points()
        self.extend_RRT(disp = disp)
        if measure_time : self.rrt_exec_time['extend tree'].append(round(time.time() - trc_e, 3))

        #set leaves
        #if measure_time : trc_l = time.time()
        self.leaves = self.get_leaves()
        #if measure_time : self.rrt_exec_time['leaves'].append(round(time.time() - trc_l, 3))

        if disp : self.display_tree()
        if measure_time : self.rrt_exec_time['update rrt'].append(round(time.time() - trc, 3))



    #2. information functions    
    def get_loop_prob(self, future_poses, traces, last_blind_v, disp = False, measure_time = True):
        if disp : print('Getting loop prob ...')
        if measure_time : trc_lo = time.time()
        if measure_time : n_iterations_seg = 0
        if measure_time : n_iterations_ch = 0
        if measure_time : n_iterations_try = 0
        if measure_time : n_iterations_true = 0
        if disp : print('future_poses :', future_poses)

        loop = {}
        for i_tr in traces:
            if disp : print('i_tr :', i_tr)
            if disp : print('trace :', traces[i_tr].ag_pos)
            loop[i_tr] = {}

            #first step : cut the track into segments of length segment_size
            sequence = list(range(len(traces[i_tr].ag_pos)))
            if disp : print('sequence :', sequence)
            num_segments = len(sequence) // self.segment_size
            partition = [sequence[i * self.segment_size: (i+1) * self.segment_size + 1] for i in range(num_segments)]
            if len(sequence) % self.segment_size != 0 and len(sequence[num_segments * self.segment_size:]) >= 2:
                partition.append(sequence[num_segments * self.segment_size:])
                    
            #second step 
            if disp : print('partition :', partition)
            for segment_indexes in partition:
                if disp : print('segment_indexes :', segment_indexes)
                if measure_time : n_iterations_seg += 1

                #get poses, blind dist and stdev
                segment_poses = traces[i_tr].ag_pos[segment_indexes[0]:segment_indexes[-1]]
                segment_steps = traces[i_tr].time_steps[segment_indexes[0]:segment_indexes[-1]]
                if disp : print('segment_poses :', segment_poses)

                blind_dist_list = []
                for idx in segment_indexes:
                    blind_dist_options = [last_blind_v - traces[i_tr].blind_list[idx][idx2] for idx2 in range(len(traces[i_tr].blind_list[idx]))]
                    blind_dist = max(min(blind_dist_options), 0)
                    blind_dist_list.append(blind_dist)
                if disp : print('blind_dist_list :', blind_dist_list)

                blind_dist_mean = statistics.mean(blind_dist_list)
                segment_stdev = get_stdev_error(blind_dist_mean, self.odom_error_rate)
                if disp : print('segment_stdev :', segment_stdev)

                #get prob
                check = man_dist_check(future_poses, segment_poses, segment_stdev)
                if check:
                    if disp : print('man dist check passed')
                    if measure_time : n_iterations_ch += 1

                    #create and memorize devations list for the mc process
                    if segment_stdev in self.dl_memo:
                        deviations = self.dl_memo[segment_stdev]
                    else:
                        if measure_time : trc_lo3dev = time.time()
                        deviations = np.random.multivariate_normal(mean=(0,0), cov=segment_stdev * np.identity(n=2), size = self.mc_n_simulations)
                        self.dl_memo[segment_stdev] = deviations
                        #if measure_time : self.rrt_exec_time['get loop prob3dev'].append(round(time.time() - trc_lo3dev, 5))

                    if measure_time : trc_lo3f = time.time()
                    first_prob = get_prob_intersecting(future_poses, segment_poses, deviations = deviations, num_simulations = self.mc_init_max_try)
                    #if measure_time : self.rrt_exec_time['get loop prob3f'].append(round(time.time() - trc_lo3f, 5))
                    
                    if first_prob >= self.prob_loop_threshold/2:
                        if disp : print('first prob check passed')
                        if measure_time : n_iterations_try += 1
                        
                        #calculate more precise probability
                        if measure_time : trc_lo3p = time.time()
                        prob_loop_w_segment = get_prob_intersecting(future_poses, segment_poses, deviations = deviations, num_simulations = self.mc_n_simulations)
                        #if measure_time : self.rrt_exec_time['get loop prob3p'].append(round(time.time() - trc_lo3p, 4))
                        
                        if prob_loop_w_segment >= self.qc_threshold:
                            if measure_time : n_iterations_true += 1
                            
                            #set loop dictionary
                            loop[i_tr][(segment_steps[0], segment_steps[-1])] = prob_loop_w_segment
                            if disp : print('qc_loop_w_segment :', prob_loop_w_segment)
                            break #we can break as the earliest segment we loop with is the best

                        elif prob_loop_w_segment >= self.prob_loop_threshold:
                            if measure_time : n_iterations_true += 1
                            #add to loop dictionary
                            loop[i_tr][(segment_steps[0], segment_steps[-1])] = prob_loop_w_segment
                            if disp : print('prob_loop_w_segment :', prob_loop_w_segment)
                            continue

        #if measure_time : self.rrt_exec_time['get loop prob-n_iter'].append((n_iterations_seg, n_iterations_ch, n_iterations_try, n_iterations_true))
        if measure_time : self.rrt_exec_time['get loop prob'].append(round(time.time() - trc_lo, 4))
        return loop
    
    def fuse_loops(self, first_loop):
        loop ={}
        for tr_id in first_loop:
            new_loop_tr = {}
            init_partition = list(first_loop[tr_id].keys())
            new_partition = merge_segments(init_partition)
            for seg_n in new_partition:
                if seg_n in init_partition:
                    new_prob = first_loop[tr_id][seg_n]
                else:
                    new_prob = max(first_loop[tr_id][seg_p] for seg_p in init_partition if seg_n[0] <= seg_p[0] and seg_n[1] >= seg_p[1])
                new_loop_tr[seg_n] = new_prob
            loop[tr_id] = new_loop_tr
        return loop

    def get_meet_prob(self, future_poses, future_vstep, past_tracks, team_pos, team_plans, distrib_method = 'mc', disp = False, measure_time = True):
        if True : print('Getting meet prob ...')
        if measure_time : trc_me = time.time()

        meet = {}
        #presence_distrib_dict = {} 
        tracks_dict = past_tracks['root'] if 'root' in past_tracks else past_tracks #past tracks can be under tracks or dict format
        for oth_id in tracks_dict:
            if oth_id != self.id:
                if self.team_distrib_presence_null and oth_id in self.team_distrib_presence_null and future_vstep >= self.team_distrib_presence_null[oth_id]:
                    if True : print(' -> presence distrib should be null because vstep is', future_vstep)
                    continue

                oth_pos = team_pos[oth_id]
                if disp : print(" future_vstep : ", future_vstep)
                if disp : print(" oth_pos : ", oth_pos)

                #define last step contact and last agent contact
                last_step_contact = max(oth_pos[observer]['time_step'] for observer in oth_pos)
                candidates = []
                for observer in oth_pos:
                    if oth_pos[observer]['time_step'] == last_step_contact:
                        candidates.append(observer)
                last_agent_contact = max(candidates)
                last_pos_contact = oth_pos[last_agent_contact]['seen_pos']
                if disp : print(" Agent", oth_id, 'has been lastly seen at step', last_step_contact, 'by agent', last_agent_contact, 'at pos', last_pos_contact)

                #presence distrib
                tracks = past_tracks['root'] if 'root' in past_tracks else past_tracks
                tracks_added = past_tracks['added'] if 'added' in past_tracks else None
                if distrib_method == 'mc':
                    presence_map, exec_time_dic = get_agent_distrib_mc_presence(
                        self.submaps, self.id, oth_id, future_vstep, last_step_contact, last_pos_contact, last_agent_contact, team_plans[oth_id], mc_memo=self.team_distrib_presence_memo,
                        trust_plan_factor=self.trust_plan_factor, lost_steps=self.lost_thres,
                        disp=disp, measure_time=measure_time)
                    
                    if presence_map is not False:
                        no_presence_map, exec_time_dic = get_agent_distrib_mc_supress(
                            self.submaps, tracks, future_vstep, last_step_contact, last_agent_contact, tracks_added,
                            impact_steps=self.impact_thres, scan_range=self.scan_range,
                            disp=disp, measure_time=measure_time)

                        presence_distrib, exec_time_dic = get_agent_distrib_mc_compil(
                            self.submaps, presence_map, no_presence_map, disp=disp, rnd=None, measure_time=measure_time)
                    else:
                        presence_distrib = False
                        self.team_distrib_presence_null[oth_id] = future_vstep #record that distribution is null if False
                        print(' ... null presence recorded at step', future_vstep)

                else:
                    if team_plans[oth_id]:
                        presence_distrib = get_agent_distrib_plans(
                            self.submaps, tracks, self.id, 
                            oth_id, future_vstep, last_step_contact, last_pos_contact, last_agent_contact, team_plans[oth_id],
                            tracks_added,
                            trust_plan_factor = self.trust_plan_factor, lost_steps = self.lost_thres, impact_steps = self.impact_thres, scan_range = self.scan_range,
                            disp = disp)
                    else:
                        presence_distrib = get_agent_distrib(
                            self.submaps, tracks, self.id,
                            oth_id, future_vstep, last_step_contact, last_pos_contact, last_agent_contact,
                            tracks_added,
                            self.lost_thres, self.impact_thres, self.scan_range, 
                            disp = disp)

                #presence_distrib_dict[oth_id] = presence_distrib
                #if disp : print(" presence distrib : \n", (presence_distrib*100).astype(int))

                if presence_distrib is not False:
                    prob_meeting_oth = get_prob_meeting(self.submaps, future_poses, presence_distrib, self.scan_range)
            
                    if prob_meeting_oth > self.prob_meet_threshold:
                        meet[oth_id] = prob_meeting_oth

                if disp : print(' The probability of meeting agent', oth_id, 'is', prob_meeting_oth)

        if measure_time : self.rrt_exec_time['get meet prob'].append(round(time.time() - trc_me, 3))
        return meet
    
    def get_multiverse(self, parent_VA, node, disp = False, measure_time = True):
        if True : print('Getting multiverse from node', node, '...')
        if measure_time : trc_mv = time.time()
        multiverse = {}
        n_va1, n_va2, n_va3 = 0, 0, 0

        #1
        if measure_time : trc_mv1 = time.time()
        #init a new virtual agent
        if measure_time : trc_mvva = time.time()
        move_VA = VirtualAgent(self.id, parent_VA.v_step, parent_VA.submaps, parent_VA.tracks, parent_VA.bv, parent_VA.h)
        n_va1+=1
        if measure_time : self.rrt_exec_time['create_va'].append(round(time.time() - trc_mvva, 3))
        
        #move through future steps without loop nor meet
        if disp : print('Virtual move of self ...')
        future_poses = self.path_from_parent[node][1:]
        move_VA.moving_self(future_poses)

        #update_entropy
        if disp : print('Getting va situation ...')
        if measure_time : trc_vs = time.time()
        move_VA.update_entropy(disp = disp)
        if measure_time : self.rrt_exec_time['update_entropy'].append(round(time.time() - trc_vs, 3))

        #store VA after movings
        multiverse[('move')] = {
            'future_poses' : future_poses,
            'init_VA' : parent_VA,
            'VA' : move_VA,

            'init_VA.bv' : parent_VA.bv,
            'VA.bv' : move_VA.bv,
            'init_bv_traces' : {ag_id : parent_VA.tracks['traces'][ag_id].blind_v for ag_id in parent_VA.tracks['traces']},
            'bv_traces' : {ag_id : move_VA.tracks['traces'][ag_id].blind_v for ag_id in move_VA.tracks['traces']},
            'init_VA.h' : parent_VA.h,
            'VA.h' : move_VA.h,

            'w' : None,
            'will' : None
        }
            
        if measure_time : self.rrt_exec_time['get multiverse-1'].append(round(time.time() - trc_mv1, 3))

        #2
        if self.calculate_mv_loop:
            if measure_time : trc_mv2 = time.time()
            #get possibility to loop with any segment of any trace
            parent_traces = {tr_id : parent_VA.tracks[tr_id].trace for tr_id in parent_VA.tracks} if 'traces' not in parent_VA.tracks else parent_VA.tracks['traces']
            first_loop = self.get_loop_prob(future_poses, parent_traces, parent_VA.bv, disp = disp) #loop = {tr1 : {prob:0.1, rh : rh, mh : mh}, ...} (rh and mh calculated)
            loop = self.fuse_loops(first_loop)
            if disp: 
                if first_loop != loop : print('first loop :', first_loop, '\nloop :', loop)
                else : print('loop :', loop)
            
            #get and store future situation after every correction
            for tr_id in loop:
                for ts_segment in loop[tr_id]:
                    loop_step = (ts_segment[0]+ts_segment[1])//2
                    if disp : print(' tr_id :', tr_id)
                    if disp : print(' ts_segment :', ts_segment)
                    if disp : print(' loop_step :', loop_step)

                    #create VA
                    if measure_time : trc_mvva = time.time()
                    loop_VA = VirtualAgent(self.id, move_VA.v_step, move_VA.submaps, move_VA.tracks, move_VA.bv, move_VA.h)
                    n_va2+=1
                    if measure_time : self.rrt_exec_time['create_va'].append(round(time.time() - trc_mvva, 3))
                    
                    #correct
                    if disp : print('Virtual correction with trace', tr_id, 'at step', loop_step, '...')
                    if measure_time : trc_vc = time.time()
                    loop_VA.update_after_looping(tr_id, loop_step, disp)
                    if measure_time : self.rrt_exec_time['virtual_correction'].append(round(time.time() - trc_vc, 3))
                    
                    #update_entropy
                    if disp : print('Getting va situation ...')
                    if measure_time : trc_vs = time.time()
                    loop_VA.update_entropy(disp = disp)
                    if measure_time : self.rrt_exec_time['update_entropy'].append(round(time.time() - trc_vs, 3))
                    
                    #store VA after correcting
                    multiverse[('loop', tr_id, ts_segment)] = {
                        'prob' : loop[tr_id][ts_segment],
                        'segment_pos' : [parent_traces[tr_id].ag_pos[idx] for idx in range(len(parent_traces[tr_id].ag_pos)) if parent_traces[tr_id].time_steps[idx] <= ts_segment[1] and parent_traces[tr_id].time_steps[idx] >= ts_segment[0]],

                        'init_VA' : move_VA,
                        'VA' : loop_VA,
                        'VA.bv' : loop_VA.bv,
                        'bv_traces' : {ag_id : loop_VA.tracks['traces'][ag_id].blind_v for ag_id in loop_VA.tracks['traces']},
                        'VA.h' : loop_VA.h,

                        'w' : None,
                        'will' : None,
                    }
            if measure_time : self.rrt_exec_time['get multiverse-2'].append(round(time.time() - trc_mv2, 3))
        
        #3
        if self.calculate_mv_meet:
            if measure_time : trc_mv3 = time.time()
            #get possibility to meet each agent and the potential benefit
            meet = self.get_meet_prob(future_poses, move_VA.v_step, parent_VA.tracks, parent_VA.submaps.ag_team_pos, self.team_plans, distrib_method = self.distrib_method, disp = disp) #meet = {ag1 : {prob:0.1, rh : rh, mh : mh}, ...} (rh and mh estimated)
            if disp : print('meet :', meet)
            
            for m_id in meet:
                #create VA
                if measure_time : trc_mvva = time.time()
                meet_VA = VirtualAgent(self.id, move_VA.v_step, move_VA.submaps, move_VA.tracks, move_VA.bv, move_VA.h)
                n_va3+=1
                if measure_time : self.rrt_exec_time['create_va'].append(round(time.time() - trc_mvva, 3))

                #meet
                if disp : print('Virtual meeting with agent', m_id, '...')
                if measure_time : trc_mt = time.time()
                meet_VA.update_after_meeting(m_id, disp = disp)
                if measure_time : self.rrt_exec_time['virtual_meeting'].append(round(time.time() - trc_mt, 3))
    
                #update_entropy
                if disp : print('Getting va situation ...')
                if measure_time : trc_vs = time.time()
                meet_VA.update_entropy(disp = disp)
                if measure_time : self.rrt_exec_time['update_entropy'].append(round(time.time() - trc_vs, 3))
                
                multiverse[('meet', m_id)] = {
                    'prob' : meet[m_id],

                    'init_VA' : move_VA,
                    'VA' : meet_VA,
                    'VA.bv' : meet_VA.bv,
                    'bv_traces' : {ag_id : meet_VA.tracks['traces'][ag_id].blind_v for ag_id in meet_VA.tracks['traces']},
                    'VA.h' : meet_VA.h,

                    'w' : None,
                }
            if measure_time : self.rrt_exec_time['get multiverse-3'].append(round(time.time() - trc_mv3, 3))
        
        if measure_time : self.rrt_exec_time['get multiverse'].append(round(time.time() - trc_mv, 3))
        if measure_time : self.rrt_exec_time['get multiverse-n_va'].append((n_va1, n_va2, n_va3))
        return multiverse
    
    def calculate_weights(self, node, disp = False, measure_time = True):
        if disp : print('Getting weighted situations ...')
        
        #calculate the none event (nor loop nor meet) prob
        none_event_prob = 1
        sum_events_prob = 0
        for event in self.multiverse[node]:
            if type(event) == tuple:
                none_event_prob = none_event_prob * (1-self.multiverse[node][event]['prob'])
                sum_events_prob += self.multiverse[node][event]['prob']

        #set the weights
        for event in self.multiverse[node]:
            if event == 'move':
                self.multiverse[node]['move']['w'] = round(none_event_prob, 3)
            elif none_event_prob < 1:
                self.multiverse[node][event]['w'] = round(self.multiverse[node][event]['prob']/sum_events_prob*(1-none_event_prob) ,3)
        
    def get_IG(self, parent_VA, node, disp = False, measure_time = True):
        if disp : print('Getting IG for reaching node', node, '...')
        
        #get the weighted mh
        wmh = sum([(self.multiverse[node][event]['VA'].h['mhuk']+self.multiverse[node][event]['VA'].h['mhuc']) * self.multiverse[node][event]['w'] for event in self.multiverse[node]])

        delta_mh = parent_VA.h['mhuk']+parent_VA.h['mhuc'] - wmh

        if disp : print(delta_mh)
        return delta_mh
    
    def set_future_sit(self, node, disp = False, measure_time = True):
        if disp : print('Getting future situation ...')
        events_list = []
        prob_list = []

        for event in list(self.multiverse[node].keys()):
            if disp : print('event :', event)
            if event[0] == 'loop':
                prob = self.multiverse[node][event]['prob']
                events_list.append(event)
                prob_list.append(prob)
            
        if disp : print('events_list :', events_list)
        if disp : print('prob_list :', prob_list)

        #add the none event
        none_event_prob = 1
        for i_event in range(len(events_list)):
            none_event_prob = none_event_prob * (1-prob_list[i_event])
        
        if disp : print('none_event_prob :', none_event_prob)

        #choose the event
        rand_num = rd.random()
        is_event = rand_num < (1-none_event_prob)
        if is_event:
            choice = rd.choices(events_list, prob_list)[0]
            self.multiverse[node][choice]['will'] = True #mark somewhere that event will occcur
        else:
            self.multiverse[node][('move')]['will'] = True

        if disp : print('rand_num :', rand_num)
        if disp : print('is_event :', is_event)
        if disp and is_event : print('choice :', choice)

    def build_team_distrib_presence_memo(self, disp=False, measure_time=True):       
        if True : print('Building team distrib mc memo ...')
        if measure_time : trc_dbm = time.time()

        self.team_distrib_presence_memo = {}
        for m_id in self.tracks:
            if m_id != self.id:
                if True : print(' Building Agent', m_id, 'distrib memo ...')
                m_pos = self.submaps.ag_team_pos[m_id]

                #find last step contact and last agent contact
                last_step_contact = max(m_pos[observer]['time_step'] for observer in m_pos)
                last_agent_contact = max([observer for observer in m_pos if m_pos[observer]['time_step'] == last_step_contact])
                last_pos_contact = m_pos[last_agent_contact]['seen_pos']
                self.team_distrib_presence_memo[m_id] = build_agent_distrib_mc_presence_memo(
                    self.submaps, last_pos_contact,
                    self.team_plans[m_id], lost_steps=self.lost_thres,
                    disp=disp, measure_time=measure_time)
        
        if measure_time : self.rrt_exec_time['update information - build memo'].append(round(time.time() - trc_dbm, 3))

    def update_information(self, init_state, team_plans, reset_plans = False, disp = False, measure_time = True):
        print('Updating information ...')
        if measure_time : trc = time.time()
        if measure_time : n_iterations_IG = 0
        if disp : self.display_tree()

        #reset if needed
        if reset_plans : 
            self.reset_tree_plans()
            if disp : print(' Plans reset')

        #set ext var
        self.team_plans = team_plans
        
        #mc_memo
        if self.distrib_method == 'mc' and self.mc_memo_method:
            self.build_team_distrib_presence_memo(disp=disp, measure_time=measure_time)
        
        self.team_distrib_presence_null = {}

        #set obsolete or new nodes
        obsolete_o_new_sit = self.last_nodes_added + self.nodes_to_update
        if disp : print(' obsolete or new nodes :', obsolete_o_new_sit)
        
        #init start VA
        init_VA = VirtualAgent(self.id, self.tracks[self.id].obs_list[-1].time_step, self.submaps, self.tracks, init_state.blind_v, {'ph':round(init_state.path_entropy, 3), 'mhuk':round(self.submaps.map_entropy_unknown, 3), 'mhuc':round(self.submaps.map_entropy_uncertain, 3)})
        self.multiverse[self.start] = {'init' : {'VA' : init_VA, 'VA.h' : init_VA.h,}}
        self.information_gain[self.start] = {'from_parents' : None, 'cumulative' : 0, 'per_step' : 0}

        #DFS
        visited = []
        stack = deque()
        stack.append(self.start)

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.append(node)
                if disp : print(' visited node :', node)

                if node != self.start:
                    parent = self.node_parent[node]
                    if reset_plans or node in obsolete_o_new_sit:
                       
                        #evaluate the expected situation and the IG of reaching this node from the parent
                        for event in self.multiverse[parent]:
                            if event == 'init' or self.multiverse[parent][event]['will']:
                                parent_VA = self.multiverse[parent][event]['VA']
                                break
                        
                        self.multiverse[node] = self.get_multiverse(parent_VA, node, disp = disp)
                        #print(rd.random())
                        self.calculate_weights(node, disp, measure_time)
                        self.information_gain[node] = {'from_parent' : self.get_IG(parent_VA, node, disp, measure_time)}                        
                        self.set_future_sit(node, disp, measure_time)
                        
                        if disp : print('mutual info (entropy loss) between', parent, 'and', node, ':', self.information_gain[node]['from_parent'])
                        n_iterations_IG += 1

                    self.information_gain[node]['cumulative'] = round(self.information_gain[parent]['cumulative'] + self.information_gain[node]['from_parent'], 3)
                    self.information_gain[node]['per_step'] = round(self.information_gain[node]['cumulative'] / (len(self.path_to_node[node]) -1), 3)

                unvisited = [n for n in self.get_successors(node) if n not in visited]
                stack.extend(unvisited)
        if measure_time : self.rrt_exec_time['update information-n_iter'].append(n_iterations_IG)
        if measure_time : self.rrt_exec_time['update information'].append(round(time.time() - trc, 3))

    
    #
    def update_rrt_metrics(self):
        self.rrt_metrics['tree'] = self.tree
        self.rrt_metrics['start'] = self.start
        self.rrt_metrics['unaccessible_nodes'] = self.unaccessible_nodes
        self.rrt_metrics['obsolete_nodes'] = self.obsolete_nodes
        self.rrt_metrics['penalty_points'] = self.penalty_points
        self.rrt_metrics['last_nodes_added'] = self.last_nodes_added
        self.rrt_metrics['nodes_to_update'] = self.nodes_to_update
        self.rrt_metrics['no_path_points'] = self.no_path_points
        self.rrt_metrics['need_to_rebuild_paths'] = self.need_to_rebuild_paths

        self.rrt_metrics['n_nodes'] = len(self.tree)
        self.rrt_metrics['ideal n_nodes'] = self.ideal_n_nodes

        self.rrt_metrics['node_parent'] = self.node_parent
        self.rrt_metrics['path_from_parent'] = self.path_from_parent
        self.rrt_metrics['node_depth'] = self.node_depth
        self.rrt_metrics['node_path'] = self.node_path
        self.rrt_metrics['path_to_node'] = self.path_to_node
        self.rrt_metrics['leaves'] = self.leaves

        self.rrt_metrics['multiverse'] = {node : {event : {k : v for k, v in self.multiverse[node][event].items() if k not in ['VA','init_VA']} for event in self.multiverse[node]} for node in self.multiverse}
        self.rrt_metrics['information gain'] = self.information_gain
    
    
    
    def update_tree(self, state, submaps, costmaps, tracks, metrics, team_plans, penalty_points, curr_step, reset_information, disp = False, measure_time = True):
        #update external variables
        if measure_time : tac = time.time()
        self.submaps = copy.deepcopy(submaps)
        self.tracks = copy.deepcopy(tracks)
        if measure_time : self.rrt_exec_time['update external variables'].append(round(time.time()-tac, 3))

        #update n_nodes
        self.ideal_n_nodes = min(max(int(len(self.submaps.get_ext_map_points_list()) / self.nodes_ratio), self.min_n_nodes), self.max_n_nodes)
        self.dl_memo = {}

        #update penalty points
        self.penalty_points = penalty_points

        #set bias map
        self.reset_metrics(curr_step)
        if 'bias' in self.sampling_method and costmaps != None:
            self.bias_map = costmaps.norm_global_cost_map * self.bias_rate + np.ones_like(self.submaps) * (1-self.bias_rate)
        else:
            self.bias_map = None
        
        self.update_rrt(disp = disp)

        self.update_information(state, team_plans, reset_information, disp, measure_time)
        
        self.update_rrt_metrics()





    #3. plan function
    def choose_action(self, method = 'rd_leaf', disp = False):
        if disp : print('Choosing action ...')
        if disp : self.display_tree()

        if len(list(self.tree.keys())) == 0:
            return False, False, False
        
        max_IG = 0
        action = None

        if method == 'rd':
            action = rd.choice(list(self.tree.keys()))
            max_IG = None

        elif method == 'rd_leaf':
            action = rd.choice(list(self.leaves))
            max_IG = None

        elif method == 'max_IG':
            cum_IG = {node : self.information_gain[node]['cumulative'] for node in self.information_gain}
            action, max_IG = max(cum_IG.items(), key=lambda x: x[1])

        elif method == 'max_step_IG':
            step_IG = {node : self.information_gain[node]['per_step'] for node in self.information_gain}
            action, max_IG = max(step_IG.items(), key=lambda x: x[1])
        
        if action == self.start:
            return [], True, []
        else:
            node_path = self.node_path[action]
            goal = node_path[1]
            path = self.path_from_parent[goal]

        #record
        self.has_set_action = method
        self.new_target = action
        self.expected_gain = max_IG
        self.new_node_path = node_path
        self.new_goal_set = goal
        self.new_path_set = path

        if disp : print(node_path, goal, path)
        return node_path[1:], goal, path[1:]
    
    def update_plan_metrics(self):
        self.rrt_metrics['has_set_action'] = self.has_set_action
        self.rrt_metrics['new_target'] = self.new_target
        self.rrt_metrics['new_node_path'] = self.new_node_path
        self.rrt_metrics['expected_gain'] = self.expected_gain
        self.rrt_metrics['new_goal_set'] = self.new_goal_set
        self.rrt_metrics['new_path_set'] = self.new_path_set
    
    def display_tree(self):
        print('---Desplaying tree', self.id)
        print('tree :\n', self.tree)
        print('start :', self.start)
        print('node parent :\n', self.node_parent)
        print('path from parent :\n', self.path_from_parent)
        print('node depth :\n', self.node_depth)
        print('node path :\n', self.node_path)
        print('path to node :\n', self.path_to_node)
        print('leaves :', self.leaves)
        print('need_to_rebuild_paths :', self.need_to_rebuild_paths)
        print('last_nodes_added :', self.last_nodes_added)
        print('nodes_to_update :', self.nodes_to_update)
        print('---end disp')