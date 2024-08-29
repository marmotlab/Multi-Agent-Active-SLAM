#This file defined the 4 components of the Agent : Sensor, Submaps, Planner and Move Base
import math
import numpy as np
import time
import random as rd

import statistics

from functions.path_exists import *
from functions.connected_points import *
from functions.a_star_v2 import *
from functions.entropy_functions import *

from param import AgentParameters, RRTParameters

class Sensor:
    def __init__(self):
        #parameter
        self.range = AgentParameters.RANGE
        self.dim = self.range*2 + 1

        #own variables
        self.local_map = None
        self.borders = None
        self.local_map_completed = None
        self.team_in_range = {}
        self.agents_scan = None

    def is_within_range(self, pos, move_base):
        return abs(move_base.pos[0] - pos[0]) <= self.range and abs(move_base.pos[1] - pos[1]) <= self.range

    def get_local_map(self, map, move_base):
        dim = self.dim
        local_map = np.zeros((dim, dim))

        for i in range(dim):
            for j in range(dim):
                rel_pos = (move_base.pos[0]-dim//2+i, move_base.pos[1]-dim//2+j)
                local_map[i,j] = map.get_square(rel_pos)
        return local_map

    def get_borders(self):
        borders = []
        dim = self.dim
        if self.local_map[0,0] == -1:
            if self.local_map[0,dim-1] == -1:
                borders.append('top')
            if self.local_map[dim-1,0] == -1:
                borders.append('left')

        if self.local_map[dim-1,dim-1] == -1:
            if self.local_map[dim-1,0] == -1:
                borders.append('bottom')
            if self.local_map[0,dim-1] == -1:
                borders.append('right')
        return borders

    def scan_for_agents(self, team, move_base):

        #update team_in_range
        self.team_in_range = {}
        #method 1
        for _id, agent in team.items():
            agent_pos = agent.move_base.pos
            if self.is_within_range(agent_pos, move_base):
                self.team_in_range[agent.id] = agent_pos
        
        '''
        #method 2
        for i in range(dim):
            for j in range(dim):
                rel_pos = (move_base.pos[0]-dim//2+i, move_base.pos[1]-dim//2+j)
                for _id, agent in team.items():
                    if agent.move_base.pos == rel_pos:
                        ag_team_pos["(" + str(i) + "," + str(j) + ")"] = agent.id #obsolete
        '''

        #
        pos = move_base.pos
        dim = self.dim
        agents_scan = [[[] for j in range(dim)] for i in range(dim)]

        for ag_id in self.team_in_range:
            i = self.team_in_range[ag_id][0]-pos[0]+dim//2
            j = self.team_in_range[ag_id][1]-pos[1]+dim//2
            agents_scan[i][j].append(ag_id)

        return agents_scan

    def get_local_map_completed(self, map, move_base):
        height = map.height
        width = map.width
        local_map_completed = -1* np.ones((height, width))

        for i in range(height):
            for j in range(width):
                if self.is_within_range((i,j), move_base):
                    local_map_completed[i,j] = map.get_square((i,j))
        return local_map_completed

    def update_sensor(self, env, move_base, disp = False):
        self.local_map = self.get_local_map(env.map, move_base)
        self.borders = self.get_borders()
        self.local_map_completed = self.get_local_map_completed(env.map, move_base)
        self.agents_scan = self.scan_for_agents(env.team, move_base)
        if disp : print("Scan updated")


    def display_local_map(self, disp = False):
        print("Scan :")
        print("Local Map :")
        print(self.local_map)
        #print("Local Map Completed :")
        #print(self.local_map_completed)
        print("Scan fo agents :")
        print(self.agents_scan)

    def render_sensor(self):
        pass










class SubMaps:
    def __init__(self, init_pos = (0,0)):
        
        #parameters
        self.sub_height = AgentParameters.SUBMAP_MAX_HEIGHT
        self.sub_width = AgentParameters.SUBMAP_MAX_WIDTH
        self.off_set = AgentParameters.OFF_SET
        self.frontier_depth = AgentParameters.FRONTIER_DEPTH

        #entropy
        self.init_blind = AgentParameters.INIT_BLIND
        self.unknown_penalty = AgentParameters.MH_UNKNOWN_PENALTY
        self.thres1 = int(1/AgentParameters.ODOM_ERROR_RATE) if AgentParameters.ODOM_ERROR_RATE else 1e3
        self.thres2 = AgentParameters.MH_THRES_2

        #init own variables
        self.ag_pos = (init_pos[0] + self.off_set[0], init_pos[1] + self.off_set[1]) #agent's pose belief
        self.init_submaps()

        #meta data
        self.frontier_map = self.get_frontier_map() #map representing the frontier : unknown area near the known area (distance below frontier depth)
        self.ext_map = self.get_ext_map() #agent's map belief with the frontier added
        self.blind_v_map, self.blind_d_map = self.get_blind_maps(0) #map representing the blind value over each square seen ; related to the map entropy
        self.probabilistic_occupancy_grid = self.get_prob_occ_grid()
        self.map_entropy_uncertain, self.map_entropy_unknown = self.get_map_entropy()

        #others
        self.suspected_holes = []
        self.treated_holes = []

        #metrics
        self.metrics = {
            #map metrics
            'n_squares_known' : None,
            'n_obstacles' : None,

            #scans metrics
            'n_squares_scanned' : None,
            'n_scans_mean' : None,
            'n_scans_med' : None,
            'n_scans_q1' : None,
            'n_scans_d1' : None,
            'n_scans_min' : None,
            
            #blind metrics
            'bv_mean' : None,
            'bv_med' : None,
            'bv_q3' : None,
            'bv_d9' : None,
            'bv_max' : None,

            'bd_mean' : None,
            'bd_med' : None,
            'bd_q3' : None,
            'bd_d9' : None,
            'bd_max' : None,

            #team metrics
            'team_n_agents_known' : None,
            'team_self_viewed_n' : None,
            'team_self_viewed_perc' : None,

            'team_lastly_seen_mean_ts' : None,
            'team_lastly_seen_n' : None,
            'team_lastly_seen_perc' : None
        }

    def init_submaps(self):
        self.ag_map = self.init_ag_map() #agent's map belief, occupancy grid
        self.n_scans_map = self.init_zeros_map() #grid representing the number of times each square has been seen
        self.blind_table = self.init_blind_table() #map representing the blind value over each square seen ; related to the map entropy
        self.ag_team_pos = {}

    #submap
    def init_zeros_map(self):
        return np.zeros((self.sub_height, self.sub_width))
    
    def on_submap(self, pos):
        return pos[0] >= 0 and pos[0] < self.sub_height and pos[1] >= 0 and pos[1] < self.sub_width

    def bird_dist(self, pos1, pos2):
        return math.sqrt(((pos1[0] - pos2[0]) ** 2) + ((pos1[1] - pos2[1]) ** 2))
    
    def manhattan_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def is_in_range(self, pos, range):
        return self.manhattan_dist(self.ag_pos, pos) <= range
    
    #agent's map
    def init_ag_map(self):
        ag_map = -1* np.ones((self.sub_height, self.sub_width))
        return ag_map

    def is_known(self, pos):
        return self.ag_map[pos[0], pos[1]] >= 0
    
    def is_free(self, pos):
        return self.ag_map[pos[0], pos[1]] == 0

    def is_obstacle(self, pos):
        return self.ag_map[pos[0], pos[1]] == 1

    def is_out(self, pos):
        return self.ag_map[pos[0], pos[1]] == 10
    
    def get_out_n_obs_list(self):
        out_n_obs_list = []
        for i in range(self.sub_height):
            for j in range(self.sub_width):
                pos = (i,j)
                if self.is_out(pos) or self.is_obstacle(pos):
                    out_n_obs_list.append(pos)
        return out_n_obs_list
        
    def get_points_list_in_range(self, center, sq_range):
        list = []
        for i in range(self.sub_height):
            for j in range(self.sub_width):
                pos = (i,j)
                if self.manhattan_dist(center, pos) <= sq_range:
                    list.append(pos)
        return list
    

    def get_points_list(self, max_range = False): #unused
        list = []
        for i in range(self.sub_height):
            for j in range(self.sub_width):
                pos = (i,j)
                if not max_range or self.is_in_range(pos, max_range):
                    list.append(pos)
        return list
    
    def get_known_points_list(self, max_range = False):
        list = []
        for i in range(self.sub_height):
            for j in range(self.sub_width):
                pos = (i,j)
                if self.is_known(pos) and (not max_range or self.is_in_range(pos, max_range)):
                    list.append(pos)
        return list
        
    #frontier map
    def get_frontier_map(self):
        frontier_map = np.zeros((self.sub_height, self.sub_width))
        for i in range(self.sub_height):
            for j in range(self.sub_width):
                pos1 = (i,j)
                #consider the unknown position
                if not self.is_known(pos1):
                    #is the square is at distance below frontier depth from a known square
                    
                    for k in range(i-self.frontier_depth, i+self.frontier_depth+1):
                        for l in range(j-self.frontier_depth, j+self.frontier_depth+1):
                            pos2 = (k,l)
                            if self.on_submap(pos2) and self.manhattan_dist(pos1, pos2) <= self.frontier_depth:
                                if self.is_known(pos2):
                                    frontier_map[i,j] = 3
        return frontier_map

    def is_on_frontier(self, pos):
        return self.frontier_map[pos[0], pos[1]] == 3
    
    def get_frontier_points_list(self, max_range = False):
        list = []
        for i in range(self.sub_height):
            for j in range(self.sub_width):
                pos = (i,j)
                if self.is_on_frontier(pos) and (not max_range or self.is_in_range(pos, max_range)):
                    list.append(pos)
        return list
    
    def is_frontier(self):
        return bool(np.amax(self.frontier_map))
    
    def are_there_unknown(self):
        for i in range(self.sub_height):
            for j in range(self.sub_width):
                pos = (i,j)
                if not self.is_known(pos):
                    return True
        return False 
       
    #extended map
    def get_ext_map(self):
        ext_map = -1* np.ones((self.sub_height, self.sub_width))
        for i in range(self.sub_height):
            for j in range(self.sub_width):
                pos = (i,j)
                if self.ag_map[i,j] == -1 and self.is_on_frontier(pos):
                    ext_map[i,j] = -0.1
                else:
                    ext_map[i,j] = self.ag_map[i,j]
        return ext_map
    
    def is_known_or_on_frontier(self, pos):
        return self.ext_map[pos[0], pos[1]] >= -0.1
    
    def get_ext_map_points_list(self, max_range = False):
        list = []
        for i in range(self.sub_height):
            for j in range(self.sub_width):
                pos = (i,j)
                if self.is_known_or_on_frontier(pos) and (not max_range or self.is_in_range(pos, max_range)):
                    list.append(pos)
        return list
    
    def get_diameter(self):
        ext_map_points_list = self.get_ext_map_points_list()
        height_list = [pos[0] for pos in ext_map_points_list]
        height = max(height_list) - min(height_list)
        width_list = [pos[1] for pos in ext_map_points_list]
        width = max(width_list) - min(width_list)
        return height + width

    #maze
    def get_maze1(self):
        maze = np.copy(self.ag_map)
        for i in range(self.sub_height):
            for j in range(self.sub_width):
                if maze[i,j] == -1:
                    maze[i,j] = 0
                elif maze[i,j] == 10:
                    maze[i,j] = 1
        return maze

    def get_maze2(self):
        maze = np.zeros((self.sub_height,self.sub_width))
        for i in range(self.sub_height):
            for j in range(self.sub_width):
                if self.ag_map[i,j] == 10:
                    maze[i,j] = 1
                else:
                    maze[i,j] = 0
        return maze

    def is_point_connected_to_map(self, pos):
        maze = self.get_maze2()
        start = pos
        end = self.ag_pos
        #print("Maze :\n", maze)
        #print("Start :", start, "End :", end)
        return path_exists(maze, start, end)
    
    def get_points_connected_to_agent_list(self, max_range = None):
        maze = self.get_maze1()
        start = self.ag_pos
        #print("Maze :\n", maze)
        #print("Start :", start, "End :", end)    
        return connected_points(maze, start, max_range)
    
    #neighbours
    def get_neighbours(self, pos):
        neighbours = []
        for shift in [(-1,0), (1,0), (0,1), (0,-1)]:
            pot_neighbour = (pos[0]+shift[0], pos[1]+shift[1])
            if self.on_submap(pot_neighbour):
                if not self.is_out(pot_neighbour):
                    neighbours.append(pot_neighbour)
        return neighbours


    #updating the submaps : agent map, agent belief pos, agent team pos, other submaps
    #agent belief pos
    def update_ag_pos(self, odom):
        if odom != None and odom != False:
            new_pos = (self.ag_pos[0]+odom[0], self.ag_pos[1]+odom[1])
            self.ag_pos = new_pos
    
    #update agent map
    def update_ag_map(self, observation, disp = False):
        local_map = observation.local_map
        ag_pos = observation.pos_belief[-1]['pos_belief']
        blind_v = observation.pos_belief[-1]['blind_v']
        dim = len(local_map)
        for i in range(dim):
            for j in range(dim):
                obs_point = (ag_pos[0]-dim//2+i, ag_pos[1]-dim//2+j)
                if self.on_submap(obs_point):

                    #update ag map
                    if local_map[i,j] != -1: #only consider points that belong to the real map (not outide map borders)
                        self.ag_map[obs_point[0], obs_point[1]] = local_map[i,j]
                        
                        #add meta data
                        self.n_scans_map[obs_point[0], obs_point[1]] += 1
                        self.blind_table[obs_point[0]][obs_point[1]].append(blind_v)

                    else:
                        self.ag_map[obs_point[0], obs_point[1]] = 10
                        self.blind_table[obs_point[0]][obs_point[1]].append(-1)

        if disp : print("Agent map updated")

    def fake_update_ag_map(self, observation, dim, updated_bv = None, disp = False): #for a virtual agent
        ag_pos = observation.pos_belief[-1]['pos_belief']
        blind_v = observation.pos_belief[-1]['blind_v'] if updated_bv == None else updated_bv
        for i in range(dim):
            for j in range(dim):
                obs_point = (ag_pos[0]-dim//2+i, ag_pos[1]-dim//2+j)
                if self.on_submap(obs_point):
                    self.ag_map[obs_point[0], obs_point[1]] = 0
                    self.n_scans_map[obs_point[0], obs_point[1]] += 1
                    self.blind_table[obs_point[0]][obs_point[1]].append(blind_v)

    #treat borders and holes
    def treat_borders(self, observation, disp = False):
        new_submap = self.ag_map
        borders = observation.borders
        ag_pos = observation.pos_belief[-1]['pos_belief']
        dim = len(observation.local_map)

        top = max(ag_pos[0]-dim//2, 0)
        bottom = min(ag_pos[0]+dim//2, self.sub_height-1)
        left = max(ag_pos[1]-dim//2, 0)
        right = min(ag_pos[1]+dim//2, self.sub_width-1)

        #treat borders
        if 'top' in borders:
            if top != 0:
                for i in range(0,top):
                    for j in range(left,right +1):
                        new_submap[i,j] = 10
        elif 'bottom' in borders:
            if bottom != self.sub_height -1:
                for i in range(bottom +1,self.sub_height):
                    for j in range(left,right +1):
                        new_submap[i,j] = 10 
        if 'left' in borders:
            if left != 0:
                for i in range(top,bottom +1):
                    for j in range(0,left):
                        new_submap[i,j] = 10
        elif 'right' in borders:
            if right != self.sub_width -1:
                for i in range(top,bottom +1):
                    for j in range(right +1,self.sub_width):
                        new_submap[i,j] = 10

        self.ag_map = new_submap
        if disp : print("Borders treated")

        #add corners as suspected holes
        if 'top' in borders and 'left' in borders:
            top_left_corner = (0,0)
            if top_left_corner not in self.treated_holes:
                self.suspected_holes.append(top_left_corner)
                if disp : print("top_left_corner suspected ; suspected :", self.suspected_holes)

        if 'top' in borders and 'right' in borders:
            top_right_corner = (0, self.sub_width-1)
            if top_right_corner not in self.treated_holes: 
                self.suspected_holes.append(top_right_corner)
                if disp : print("top_right_corner suspected ; suspected :", self.suspected_holes)

        if 'bottom' in borders and 'left' in borders:
            bottom_left_corner = (self.sub_height-1, 0)
            if bottom_left_corner not in self.treated_holes: 
                self.suspected_holes.append(bottom_left_corner)
                if disp : print("bottom_left_corner suspected ; suspected :", self.suspected_holes)

        if 'bottom' in borders and 'right' in borders:
            bottom_right_corner = (self.sub_height-1, self.sub_width-1)
            if bottom_right_corner not in self.treated_holes: 
                self.suspected_holes.append(bottom_right_corner)
                if disp : print("bottom_right_corner suspected ; suspected :", self.suspected_holes)


    def treat_holes(self, holes = 'suspected', disp = False):
        if disp : print('Treating', holes, 'holes ...')
        if disp : print("Suspeted holes :", self.suspected_holes)
        if disp : print("Treated holes :", self.treated_holes)
        has_treated = False
        if holes == 'suspected' : queue = list(set(self.suspected_holes))
        elif holes == 'all' : queue = list(set(self.treated_holes + self.suspected_holes))
        if disp : print('queue :', queue)
        
        k = 0
        while queue != [] and k < 30 :
            #print("queue :", queue)
            k += 1
            pos = queue.pop(0)
            if disp : print('hole to treat:', pos)

            if self.ag_map[pos[0],pos[1]] == -1:
                if not self.is_point_connected_to_map(pos):
                    self.ag_map[pos[0],pos[1]] = 10
                    queue = list(set(queue + self.get_neighbours(pos)))
                    self.treated_holes.append(pos)
                    if not has_treated : has_treated = [pos]
                    else : has_treated.append(pos)
                    if disp : print('hole', pos, 'has been treated')
        #set lists
        self.treated_holes = list(set(self.treated_holes + self.suspected_holes))
        self.suspected_holes = []
        if disp : print("Holes treated and Agent Map updated")
        if disp : print("Treated holes :", self.treated_holes)
        return has_treated

    #update other agents' pos
    def reset_ag_team_pos(self):
        self.ag_team_pos = {}

    def update_ag_team_pos(self, observation, disp = False):
        #add new pos
        agents_scan = observation.agents_scan
        ag_pos = observation.pos_belief[-1]['pos_belief']
        dim = len(agents_scan)
        for i in range(dim):
            for j in range(dim):
                if agents_scan[i][j] != []:
                    for ag_id in agents_scan[i][j]:
                        rel_pos = (ag_pos[0]-dim//2+i, ag_pos[1]-dim//2+j)
                        
                        #create dic for agent if needed
                        if ag_id not in self.ag_team_pos:
                            self.ag_team_pos[ag_id] = {}
                        
                        #set observer and add the pos to the dic
                        obs_id = observation.ag_id
                        time_step = observation.time_step

                        self.ag_team_pos[ag_id][obs_id] = {
                            'seen_pos' : rel_pos, 
                            'time_step' : time_step}

        #remove obsolete pos
        for ag_id in self.ag_team_pos:
            if ag_id in self.ag_team_pos[ag_id]:
                last_self_step = self.ag_team_pos[ag_id][ag_id]['time_step']
                
                del_list = []
                for obs_id in self.ag_team_pos[ag_id]:
                    if obs_id == ag_id:
                        pass
                    elif obs_id < ag_id:
                        if self.ag_team_pos[ag_id][obs_id]['time_step'] <= last_self_step:
                            del_list.append(obs_id)
                    elif obs_id > ag_id:
                        if self.ag_team_pos[ag_id][obs_id]['time_step'] < last_self_step:
                            del_list.append(obs_id)
                
                for obs_id in del_list:
                    del self.ag_team_pos[ag_id][obs_id]

        if disp : print("Other agents' pos updated")


    #get other submaps
    def update_submaps_ext(self, disp = False, measure_time = True):
        if measure_time : tsc = time.time()
        self.frontier_map = self.get_frontier_map()
        if measure_time : exec_1 = time.time() - tsc
        if measure_time : tsc = time.time()
        self.ext_map = self.get_ext_map()
        if measure_time : exec_2 = time.time() - tsc
        if disp : print("Map extentions updated")
        return (exec_1, exec_2)


    #other variables/submaps
    #n_scans map
    def n_times_scanned(self, pos):
        return self.n_scans_map[pos[0], pos[1]]

    #blind map
    def init_blind_table(self):
        return [[[] for j in range(self.sub_width)] for i in range(self.sub_height)]

    def get_blind_maps(self, agent_bv):
        blind_v_map = np.ones_like(self.ag_map)
        blind_d_map = np.ones_like(self.ag_map)
        for i in range(self.sub_height):
            for j in range(self.sub_width):
                blind_list = self.blind_table[i][j]
                if blind_list == []:
                    blind_v_map[i,j] = self.init_blind
                    blind_d_map[i,j] = self.init_blind
                else:
                    blind_list_positive = [elem for elem in blind_list if elem != -1]
                    blind_dist = [(agent_bv - elem) for elem in blind_list if elem != -1]
                    if blind_list_positive == []:
                        blind_v_map[i,j] = -1
                        blind_d_map[i,j] = -1
                    else:
                        blind_v_map[i,j] = min(blind_list_positive)
                        blind_d_map[i,j] = max(min(blind_dist), 0)
        return blind_v_map, blind_d_map
    
    def update_blind_maps(self, agent_bv, disp = False):
        self.blind_v_map, self.blind_d_map = self.get_blind_maps(agent_bv)
        if disp : print('Blind d map updated :\n ', self.blind_d_map)

    def get_bvalue(self, pos):
        return self.blind_v_map[pos[0], pos[1]]
    def get_bdistance(self, pos):
        return self.blind_d_map[pos[0], pos[1]]
    
    #probabilistic occupancy grid
    def get_prob_cell(self, state, blind_d):
        if state == -1: #unknown
            return 0.5
        elif state == 10 or blind_d == -1: #wall
            return -1
        else:
            s = state
            x = get_correct_cell_prob(blind_d, self.thres1, self.thres2)
            pc = 0.5 * (1 - x + 2*x*s)
            return pc
    
    def get_prob_occ_grid(self):
        prob_occ_grid = np.zeros_like(self.ag_map)
        for i in range(self.sub_height):
            for j in range(self.sub_width):
                state = self.ag_map[i,j]
                blind_d = self.blind_d_map[i,j]
                blind_v = self.blind_v_map[i,j]
                prob_occ_grid[i,j] = self.get_prob_cell(state, blind_v) #blind_v or blind_d
        return prob_occ_grid

    def update_prob_occ_grid(self, pos, blind_v):
        self.probabilistic_occupancy_grid[pos[0], pos[1]] = self.get_prob_cell(blind_v)

    def prob_cell(self, pos):
        return self.probabilistic_occupancy_grid[pos[0], pos[1]]
    
    #entropy
    def get_map_entropy(self):
        map_entropy_uncertain = 0
        map_entropy_unknown = 0
        for i in range(self.sub_height):
            for j in range(self.sub_width):
                pos = (i,j)
                pc = self.prob_cell(pos)
                if pc in [0,1,-1]:
                    pass
                else:
                    cell_entropy = - math.log(pc) * pc /2 - math.log(1-pc) * (1-pc) /2 #maximum 0.35
                    map_entropy_uncertain += cell_entropy
                if pc == 0.5:
                    map_entropy_unknown += self.unknown_penalty
        #map_entropy = map_entropy/(self.sub_height*self.sub_width)
        return map_entropy_uncertain, map_entropy_unknown

    #update entropy
    def update_map_entropy(self, disp = False):
        self.probabilistic_occupancy_grid = self.get_prob_occ_grid()
        if disp : print('POG updated :\n ', self.probabilistic_occupancy_grid)
        self.map_entropy_uncertain, self.map_entropy_unknown = self.get_map_entropy()
        if disp : print('map entropy :', self.map_entropy_uncertain, self.map_entropy_unknown)

    #update metrics
    def update_metrics(self, self_id, time_step, recent_threshold, disp = False):
        if disp : print(" Getting submaps metrics ...")

        #updating map metrics
        known_points_list = self.get_known_points_list(max_range = False)
        
        #init
        n_obstacles = 0
        scans_list = []
        blind_v_list = []
        blind_d_list = []

        for pos in known_points_list:
            if self.is_obstacle(pos):
                n_obstacles += 1

            if not self.is_out(pos):
                scans_list.append(self.n_times_scanned(pos))
                blind_v_list.append(self.get_bvalue(pos))
                blind_d_list.append(self.get_bdistance(pos))

        self.metrics['n_squares_known'] = len(known_points_list)
        self.metrics['n_obstacles'] = n_obstacles

        if len(scans_list) >= 5:

            self.metrics['n_squares_scanned'] = len(scans_list)
            self.metrics['n_scans_mean'] = round(statistics.mean(scans_list), 1)
            self.metrics['n_scans_med'] = int(statistics.median(scans_list))
            self.metrics['n_scans_q1'] = int(statistics.quantiles(scans_list, n=4)[0])
            self.metrics['n_scans_d1'] = int(statistics.quantiles(scans_list, n=10)[0])
            self.metrics['n_scans_min'] = min(scans_list)
        
        if len(blind_v_list) >= 5:
            self.metrics['bv_mean'] = round(statistics.mean(blind_v_list), 1)
            self.metrics['bv_med'] = int(statistics.median(blind_v_list))
            self.metrics['bv_q3'] = int(statistics.quantiles(blind_v_list, n=4)[-1])
            self.metrics['bv_d9'] = int(statistics.quantiles(blind_v_list, n=10)[-1])
            self.metrics['bv_max'] = max(blind_v_list)

        if len(blind_v_list) >= 5:
            self.metrics['bd_mean'] = round(statistics.mean(blind_d_list), 1)
            self.metrics['bd_med'] = int(statistics.median(blind_d_list))
            self.metrics['bd_q3'] = int(statistics.quantiles(blind_d_list, n=4)[-1])
            self.metrics['bd_d9'] = int(statistics.quantiles(blind_d_list, n=10)[-1])
            self.metrics['bd_max'] = max(blind_d_list)
        
        #update team metrics
        self.metrics['team_n_agents_known'] = len(self.ag_team_pos) -1

        #init
        n_agent_self_viewed = 0
        n_step_back_seen = {}
        n_agents_recently_seen = 0

        for ag_id in self.ag_team_pos:
            if ag_id != self_id:
                seen_step = []

                for viewer_id in self.ag_team_pos[ag_id]:
                    if viewer_id == self_id:
                        n_agent_self_viewed += 1
                    seen_step.append(self.ag_team_pos[ag_id][viewer_id]['time_step'])
                last_step_seen = max(seen_step)
                n_step_back_seen[ag_id] = time_step - last_step_seen
                
                if n_step_back_seen[ag_id] <= recent_threshold:
                    n_agents_recently_seen += 1

        if self.metrics['team_n_agents_known'] >= 1:
            self.metrics['team_self_viewed_n'] = n_agent_self_viewed
            self.metrics['team_self_viewed_perc'] = int(self.metrics['team_self_viewed_n'] / self.metrics['team_n_agents_known'] *100)

            self.metrics['team_lastly_seen_mean_ts'] = int(statistics.mean(list(n_step_back_seen.values())))
            self.metrics['team_lastly_seen_n'] = n_agents_recently_seen
            self.metrics['team_lastly_seen_perc'] = int(n_agents_recently_seen / self.metrics['team_n_agents_known'] *100)


    #display and render
    def render_submaps(self):
        pass

    def display_submaps(self):
        print("Submaps :")
        print("Agent's Map :\n", self.ag_map)
        print("Agent's Position Belief :", self.ag_pos)
        print("Scans Map :\n", self.n_scans_map)
        #print("Agent's team Pos :", self.ag_team_pos)
        #print("Frontier Map :\n", self.frontier_map)
        #print("Frontier Distance Map :\n", self.frontier_dist_map)
        #print("Extended Agent Map :\n"(self.ext_map)









class PathPlanner:
    def __init__(self):
        #parameters
        self.pp_mode = AgentParameters.PLANNER_MODE
        self.pp_param = AgentParameters.PLANNER_PARAM
        self.pp_range = self.pp_param["PLANNER_RANGE"]
        self.costmaps = self.pp_param["COSTMAPS"]
        self.vpp_mode = self.pp_param["VPP_MODE"]
        self.tree_mode = self.pp_param["TREE_MODE"]
        self.multi_goals_mode = self.pp_param["MULTI_GOALS"]
        self.ma_penalty_mode = AgentParameters.MA_PENALTY_MODE
        self.ma_penalty_range = AgentParameters.MA_PENALTY_RANGE
        self.ma_penalty_max_time = AgentParameters.MA_PENALTY_MAX_TIME

        if self.tree_mode:
            self.action_method = self.pp_param["ACTION_METHOD"]

        #action variables
        self.do_update_path = None
        self.do_reset_path = None
        self.do_reset_plans = None
        self.do_reset_bans = None
        
        self.max_ts_non_replanning = AgentParameters.MAX_TS_NON_REPLANNING
        self.last_replanning = None

        #init own variables
        self.reset_pp_variables(None)

        #other variables
        self.temporary_banned_waypoints = []
        self.ma_penalty_points = []

        #path planner history (that includes exec_time)
        self.pp_history = {}
    
    def reset_pp_variables(self, value, disp = False):
        self.goal = value
        self.multi_goals = value
        self.expected_gain = value
        self.path = value
        if disp : print(" Variables reset")

    def is_goal(self):
        return self.goal != None and self.goal != False
    
    def is_multi_goals(self):
        return self.multi_goals != None and self.multi_goals != False and self.multi_goals != []

    def set_plan(self, costmaps, viewpoint_planner, job_done = False, disp = False):
        #reset goal and path before
        self.reset_pp_variables(None)

        #check is the job is already done
        if job_done:
            pass
        else: #set goal are multi goals
            if self.pp_mode == 'random':
                self.goal = viewpoint_planner.get_random_goal(self.pp_range)
            elif self.pp_mode == 'rd explore':
                exploration_rate = self.pp_param['EXPLO_RATE']
                rd_rate = self.pp_param['RANDOM_RATE']
                self.goal = viewpoint_planner.get_rd_explore_goal(exploration_rate, self.pp_range, rd_rate, self.temporary_banned_waypoints, self.ma_penalty_points)
            
            elif self.pp_mode == 'frontier':
                rd_f_rate = self.pp_param['RD_FRONTIER_RATE']
                self.goal = viewpoint_planner.get_frontier_goal(self.pp_range, rd_f_rate, self.temporary_banned_waypoints, self.ma_penalty_points)

            elif self.pp_mode == 'max cm':
                self.goal = viewpoint_planner.get_max_potential_goal(costmaps, self.pp_param, self.temporary_banned_waypoints, self.ma_penalty_points)
            elif self.pp_mode == 'fast cm':
                self.goal = viewpoint_planner.get_opt_potential_goal(costmaps, 
                    self.pp_param, 
                    self.temporary_banned_waypoints, 
                    self.ma_penalty_points)
            elif self.pp_mode == 'long term cm':
                self.multi_goals = viewpoint_planner.get_long_term_multi_goals(costmaps,
                    multi_goals_param = self.pp_param,
                    banned_waypoints = self.temporary_banned_waypoints, 
                    penalty_points = self.ma_penalty_points,
                    disp = disp)
                if self.multi_goals == True:
                    self.goal = True
                elif self.multi_goals:
                    self.goal = self.multi_goals[0]
                else:
                    self.goal = False
                if disp : print(" New multi-goals set :", self.multi_goals, "\n Expected gain :", viewpoint_planner.expected_gain, "\n New goal set :", self.goal)
            
            #update goal metrics
            viewpoint_planner.update_vpp_metrics()

    def goal_dist_heuristic(self, submaps):
        return submaps.bird_dist(submaps.ag_pos, self.goal)
    
    def is_goal_out_of_range(self, submaps):
        return self.goal_dist_heuristic(submaps) > self.pp_range

    def is_goal_unreachable(self, submaps):
        return submaps.is_obstacle(self.goal) or submaps.is_out(self.goal)

    def is_goal_on_agent(self, submaps):
        return submaps.ag_pos == self.goal
    
    def is_goal_and_valid(self, submaps):
        return self.is_goal() and not self.is_goal_on_agent(submaps) and not self.is_goal_unreachable(submaps) and not self.is_goal_out_of_range(submaps)
    
    def check_or_set_goal(self, submaps, costmaps, viewpoint_planner, job_done = False, disp = False):
        #update goal if necessary
        if disp : print(" Updating goal if necessary ... ( Current goal :", self.goal,")")
        if self.goal == True:
            if disp : print('Goal is True')
        else:
            if not self.is_goal(): #check if there is a goal
                self.set_plan(costmaps, viewpoint_planner, job_done)
                if disp : print(" No current goal. New goal is set :", self.goal)
            elif self.is_goal_on_agent(submaps): #check if the goal is not reached yet
                self.set_plan(costmaps, viewpoint_planner, job_done)
                if disp : print(" Error - Current goal has been reached. \n New goal is set :", self.goal)
            elif self.is_goal_unreachable(submaps): #check if the goal is reachable
                self.set_plan(costmaps, viewpoint_planner, job_done)
                if disp : print(" Error - Current goal is unreachable. \n New goal is set :", self.goal)
            elif self.is_goal_out_of_range(submaps) and not self.is_path():
                self.set_plan(costmaps, viewpoint_planner, job_done)
                if disp : print(" Error - Current goal is out of range. \n New goal is set :", self.goal)
            else :
                if disp : print(" All checks are passed. Goal should be valid.")
        
        '''
        #other method
        if not self.is_goal_and_valid(submaps):
            self.set_plan(costmaps, viewpoint_planner, job_done)
        '''

    #path functions
    def is_path(self):
        return self.path != None and self.path != [] and self.path != False
    
    def is_path_and_valid(self, submaps):
        if self.is_path():
            path_to_check = [submaps.ag_pos] + self.path

            for i in range(len(path_to_check)-1):
                pos1 = path_to_check[i]
                pos2 = path_to_check[i+1]
                potential_move = (pos2[0] - pos1[0], pos2[1] - pos1[1])
                if potential_move not in [(-1,0), (1,0), (0,1), (0,-1)]:
                    return False
                if submaps.is_obstacle(pos2) or submaps.is_out(pos2) : return False
            return True
        
        else: #no path
            return None

    def is_path_leads_to_goal(self):
        if self.is_path():
            return self.path[-1] == self.goal
        else: #no path
            return None
        
    def reset_path(self, disp = False):
        self.path = None
        if disp : print(" Path reset")

    def set_path(self, submaps, disp = False):
        if disp : print(" Finding path. Running A* ...")
        start = submaps.ag_pos
        end = self.goal
        maze = submaps.get_maze1()
        path, n_operation = a_star(maze, start, end)

        if path:
            path.pop(0)
            if disp : print(" Success. Path is found")
            self.path = path
        else:
            #no path can be found
            print(" error - path to goal", self.goal,"cannot be found")
            self.path = False

    def set_path_w_multi_goals(self, disp = False):
        pass

    def check_or_set_path(self, submaps, disp = False):
        #update path if necessary
        if disp : print(" Updating path if necessary ... ( Current path :", self.path,")")
        if not self.is_path(): #check if there is a path
            if disp : print(" error - no path set")
            self.set_path(submaps, disp)
            if disp : print(" New path is set :", self.path)
        elif not self.is_path_and_valid(submaps): #check if the path is valid
            if disp : print(" error - path is not valid")
            self.set_path(submaps, disp)
            if disp : print(" Correction - New path is set :", self.path)
        else:
            if disp : print(" Path should be valid")

    #get action function
    def get_next_pos(self):
        if self.is_path():
            return self.path[0]
        else :
            print(" error - no path is set ; path =", self.path)
            return None
    

    #update devices funcions
    def set_goal_n_path(self, submaps, costmaps, viewpoint_planner, max_try = 10, disp = False, measure_time = True):
        k_try = 0
        while True:
            #incremente
            k_try += 1
            if k_try == max_try+1:
                return False

            #init history over the tries
            self.pp_history[k_try] = {
                'init multi goals' : self.multi_goals,
                'init goal' : self.goal,
                'init path' : self.path,
                'current goal' : None,
                'current path' : None,
                'is goal valid' : None,
                'is path valid' : None,
            }

            if measure_time : tppc_t = time.time()

            #1st step : goal
            self.check_or_set_goal(submaps, costmaps, viewpoint_planner, disp = disp)
            if disp : print(" Goal is", self.goal)
            self.pp_history[k_try]['current multi goals'] = self.multi_goals
            self.pp_history[k_try]['current goal'] = self.goal

            #stop if True
            if self.goal == True:
                self.path = []
                if disp : print(" Path is set as", self.path)
                return True

            #check goal before updating the path
            if not self.is_goal_and_valid(submaps):
                if disp : print(" Goal not valid!")
                self.pp_history[k_try]['is goal valid'] = False
                if self.goal : self.temporary_banned_waypoints.append(self.goal)
                self.reset_pp_variables(False)
                continue

            if disp : print(" Goal valid!")
            self.pp_history[k_try]['is goal valid'] = True
                
            #2nd step : set path
            if measure_time : tppc_up = time.time()
            self.check_or_set_path(submaps, disp)
            if disp : print(" Path is", self.path)
            self.pp_history[k_try]['current path'] = self.path
            if measure_time and k_try > 1 : self.pp_history[k_try]['cnu path exec time'] = (round(time.time()-tppc_up,2))

            #check the path before validating both
            if not self.is_path_and_valid(submaps) or not self.is_path_leads_to_goal():
                if disp : print(" Path not valid")
                self.pp_history[k_try]['is path valid'] = False
                if self.goal : self.temporary_banned_waypoints.append(self.goal)
                self.reset_pp_variables(False)
                continue
            
            #all checks passed
            if disp : print(" Path valid and leads to goal!")
            self.pp_history[k_try]['is path valid'] = True
            if measure_time : self.pp_history[k_try]['try exec time'] = (round(time.time()-tppc_t,2))    
            return True


    def set_tree_plan(self, tree, method = 'rd', disp = False):
        #reset goal and path before
        self.reset_pp_variables(None)
        
        #set goal and path
        self.multi_goals, self.goal, self.path = tree.choose_action(method, disp)
        
        #update goal metrics
        tree.update_plan_metrics()


    def update_planning_variables(self, job_done, curr_ts, disp = False):
        if self.last_replanning == None or curr_ts - self.last_replanning > self.max_ts_non_replanning:
            self.do_reset_plans = True

        #reset if required
        if job_done: #reset if job is done
            self.reset_pp_variables(None)
            if disp : print(" Job is already done! Path Planner set to None")
        elif self.do_reset_plans:
            self.reset_pp_variables(None)
        elif self.do_reset_path:
            self.reset_path()
        elif self.do_update_path:
            self.update_path_after_moving()

        if self.do_reset_bans:
            self.temporary_banned_waypoints = []

    def update_penalty_points(self, submaps, team_plans, curr_ts, disp = False):
        penalty_points = []
        if disp : print('team plans :', team_plans)
        for ag_id in team_plans:
            if curr_ts - team_plans[ag_id]['time_step'] <= self.ma_penalty_max_time:
                if team_plans[ag_id]['multi_goals'] and team_plans[ag_id]['multi_goals'] != True:
                    add_penalty = submaps.get_points_list_in_range(team_plans[ag_id]['multi_goals'][0], self.ma_penalty_range)
                    penalty_points.extend(add_penalty)
                elif team_plans[ag_id]['goal'] and team_plans[ag_id]['goal'] != True:
                    add_penalty = submaps.get_points_list_in_range(team_plans[ag_id]['goal'], self.ma_penalty_range)
                    penalty_points.extend(add_penalty)
        self.ma_penalty_points = list(set(penalty_points))
        if disp : print('Penatly points :', self.ma_penalty_points)
    
    def is_goal_banned(self):
        return self.goal in self.ma_penalty_points
        
    def update_planner(self, state, submaps, tracks, metrics, team_plans, curr_step, costmaps, viewpoint_planner, tree, max_try, disp = False, measure_time = True):
        if disp : print("Updating planner ...")
        has_planned = None

        #updating pp history
        self.pp_history = {}
        self.pp_history['current'] = {
                'reset_plans' : self.do_reset_plans,
                'reset_path' : self.do_reset_path,
                'update_path' : self.do_update_path,

                'last replanning' : self.last_replanning,
                'multi_goals/node_path' : self.multi_goals,
                'goal' : self.goal,
                'path' : self.path,

                'reset_bans' : self.do_reset_bans,
                'banned waypoints' : self.temporary_banned_waypoints[:],
                'penalty points' : self.ma_penalty_points[:],
            }

        if not (self.is_goal_and_valid(submaps) and not self.is_goal_banned() and self.is_path_and_valid(submaps) and self.is_path_leads_to_goal()): #need to update
            #differentiate 4 cases
            #0. need to reset plans
            #1. no goal
            #2a. incorrect goal
            #2b. goal banned
            #3. robot has reached goal
            #4. path is not correct (but goal is)

            #things to do in cases #0, #1, #2 and #3:
            #1. update costmaps (if costmaps)
            #2. update tree (if tree)
                #a. updating starting node
                #b. removing obsolete branches
                #c. adding new nodes
                #d. calculating/updating paths
            #3. update information (if tree)
                #calculating or updating path information
            #4. set goal and path
                #if goal is valid : set new path
                #if goal is not valid : set new goal and new path 
            
            if not self.is_goal_and_valid(submaps) or self.is_goal_banned(): #case 0. reset case 1. no goal or #case 2. goal incorrect or #case 3. robot on goal

                #1. update costmaps
                if self.costmaps:
                    tcc = time.time()
                    costmaps.update_costmaps(submaps, tracks, metrics, team_plans, curr_step)
                    costmaps_exec_time = round(time.time()-tcc,2)
                else :
                    costmaps = None

                if self.tree_mode: #method 3
                    ttc = time.time()

                    #update tree #2. and information #3.
                    tree.update_tree(state, submaps, costmaps, tracks, metrics, team_plans, self.ma_penalty_points, curr_step, self.do_reset_plans, disp = disp)

                    #update goal and path
                    self.set_tree_plan(tree, self.action_method, disp)

                    #extend banned waypoints
                    self.temporary_banned_waypoints.extend(list(set(tree.no_path_points)))
                    
                    tree_exec_time = round(time.time()-ttc,2)
                    has_planned = 'rrt'

                elif self.vpp_mode: #method 2                
                    #update viewpoint_planner
                    tnc = time.time()
                    viewpoint_planner.update_vpp_var(submaps, curr_step)
                    vpp_exec_time = (round(time.time()-tnc,2))

                    #update goal and path
                    tppc = time.time()
                    has_set = self.set_goal_n_path(submaps, costmaps, viewpoint_planner, max_try, disp = disp)
                    if not has_set : self.reset_pp_variables(False)
                    goal_path_exec_time = round(time.time()-tppc,2)

                    if disp:
                        if has_set : print(" Path planner set. Multi Goals : ", self.multi_goals, "Goal : ", self.goal, " ; Path : ", self.path)
                        else: print(" Operation failed! Path Planner set to False.")
                    
                    has_planned = 'nav'

                else: #update goal and path #method 1
                    tppc = time.time()
                    has_set = self.set_goal_n_path(submaps, costmaps, viewpoint_planner, max_try, disp, measure_time)
                    if not has_set : self.reset_pp_variables(False)
                    goal_path_exec_time = (round(time.time()-tppc,2))

                    if disp:
                        if has_set : print(" Path planner set. Goal : ", self.goal, " ; Path : ", self.path)
                        else: print(" Operation failed! Path Planner set to False.")

                    has_planned = 'goal'

                if self.costmaps:
                    has_planned += ' cm'
                if self.path == False:
                    has_planned += ' False'
                if self.path == [] or self.goal == True:
                    has_planned += ' True'


            else: #in case 4. goal is valid but path is not -> we adjust the path
                if disp : print(" Case 4. : goal is valid but path is not")
                if disp : print(" goal :", self.goal)
                if disp : print(" path :", self.path)

                tpp = time.time()
                self.set_path(submaps, disp)
                path_exec_time = round(time.time()-tpp,2)

                #check the path before validation
                if self.is_path_and_valid(submaps) and self.is_path_leads_to_goal():
                    if disp : print(" Path valid and leads to goal!")
                else:
                    if disp : print(" Path not valid")
                    self.temporary_banned_waypoints.append(self.goal)
                    self.reset_pp_variables(False)

                #return
                has_planned = 'path'
                if self.path == False:
                    has_planned += ' False'

            if self.do_reset_plans :
                has_planned += ' reset'

            #update pph history
            self.pp_history['updated'] = {
                'has planned' : has_planned,
                'multi goals' : self.multi_goals,
                'goal' : self.goal,
                'path' : self.path,
                'banned waypoints' : [_ for _ in self.temporary_banned_waypoints],
            }

            if 'cm' in has_planned:
                self.pp_history['updated']['costmaps exec time'] = costmaps_exec_time

            if 'rrt' in has_planned:
                self.pp_history['updated']['tree exec time'] = tree_exec_time
            if 'nav' in has_planned:
                self.pp_history['updated']['nav exec time'] = vpp_exec_time
            if 'nav' in has_planned or 'goal' in has_planned:
                self.pp_history['updated']['goal and path exec time'] = goal_path_exec_time
            if 'path' in has_planned:
                self.pp_history['updated']['path exec time'] = path_exec_time

        return has_planned

    def update_path_after_moving(self, disp = False):
        self.path = self.path[1:]
        if disp : print(" Path updated after moving")


    
    #render and display functions
    def display_path_planner(self):
        print("Path Planner :")
        print("Multi Goals", self.multi_goals,
            "\nGoal :", self.goal,
            "\nPath :", self.path)

    def render_path_planner(self):
        pass










class MoveBase:
    def __init__(self, init_pos):
        #own variables
        self.init_pos = init_pos
        self.pos = self.init_pos
        self.move_base_pos_map = None
        
        #param
        self.odom_error_rate = AgentParameters.ODOM_ERROR_RATE

    def get_move_base_pos_map(self, map):
        move_base_pos_map = np.zeros((map.height, map.width))
        move_base_pos_map[self.pos[0], self.pos[1]] = 2
        return move_base_pos_map

    def reset(self):
        self.pos = self.init_pos

    def move(self, move, map): #move is a tuple (+1/0/-1, +1/0/-1) (update pos)
        if move != None:
            pos = self.pos
            new_pos = (pos[0]+move[0], pos[1]+move[1])
            if map.is_out(new_pos):
                return False
            elif map.get_square(new_pos) == 1:
                return False
            else :
                self.pos = new_pos
                self.move_base_pos_map = self.get_move_base_pos_map(map)
                return True
            
    def get_odom(self, action, disp = False):
        rand = rd.uniform(0,1)
        #set the correct odometry
        if rand > self.odom_error_rate:
            odom = action

        #set a random odometry
        elif rand < self.odom_error_rate/2:
            rd_odom = rd.choice([(0,1), (1,0), (0,-1), (-1,0), (0,0)])
            odom = (action[0] + rd_odom[0], action[1] + rd_odom[1])

        #set a null odometry
        else:
            if disp : print(" Ouch! Odometry broken")
            odom = False

        if disp : print(" Odometry mesured :", odom)
        return odom

    def display_move_base(self):
        print("Position of the agent :", self.pos)
        #print(self.move_base_pos_map)