import random as rd
import numpy as np
import time
import math

from param import AgentParameters, CostmapsParameters, RewardParameters
from functions.distance_to_line import *
from functions.diffuse_array import *
from functions.clustering import *
from functions.multi_obj_tsp import gain_optimal
from functions.multi_obj_tsp_2 import gain_optimal_2

from functions.entropy_functions import get_stdev_error

from functions.get_loop_distrib import get_loop_trace_u_x_distrib
from functions.get_meet_pos_distrib import get_agent_distrib, get_meet_pos_distrib
from functions.get_meet_pos_distrib_plans import get_agent_distrib_plans
from functions.get_meet_pos_distrib_mc import get_agent_distrib_mc, get_meet_pos_distrib


#np.random.seed(2)
#rd.seed(2)

def distance_manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class CostMaps:
    def __init__(self, self_id, submaps):
        #external variables
        self.self_id = self_id
        self.submaps = submaps

        #parameters
        self.height = AgentParameters.SUBMAP_MAX_HEIGHT
        self.width = AgentParameters.SUBMAP_MAX_WIDTH
        self.scan_range = AgentParameters.RANGE

        self.distrib_method = CostmapsParameters.DISTRIB_METHOD
        self.lost_steps = CostmapsParameters.LOST_STEPS
        self.impact_steps = CostmapsParameters.IMPACT_STEPS
        self.trust_factor = CostmapsParameters.TRUST_PLAN_FACTOR

        #cost maps variables
        self.team_visited_pos = None
        self.explore_potential = None

        self.loop_prob = {}
        self.add_n_loops = {}
        self.add_n_corr = {}
        self.loop_u_trace = {}
        self.loop_trace_u_eff = {}
        self.loop_delta_blind_v = {}
        self.loop_pos_stdev_trace = {}
        self.loop_trace_u_x_distrib = {}
        self.loop_cost_maps = {}

        self.meet_cost_maps = {}
        self.meet_u = {}
        self.meet_add_n_meeting_step = {}
        self.meet_add_n_new_neighbours = {}
        self.meet_add_n_meeting_meta_loops = {}
        self.meet_add_n_meeting_semi_loops = {}
        self.meet_add_n_meeting_corr = {}
        self.meet_add_n_agents = {}
        self.meet_add_n_known_square = {}
        self.meet_add_n_obs = {}

        self.meet_m_pos = {}
        self.meet_curr_step = {}
        self.meet_last_step_contact = {}
        
        self.presence_distrib = {}
        self.meet_distrib = {}

        self.global_cost_map = None
        self.norm_global_cost_map = None
        self.max_explore = None #arbitraly, will change at the first round
        self.max_gain = None

        #metrics
        self.reset_cm_metrics(0)

        #exec time
        self.reset_cm_exec_time_dic()

    def reset_cm_metrics(self, curr_step):
        #metrics
        self.cm_metrics = {
            'step' : curr_step,
            'explore_potential' : None,
            'max_explore_potential' : None,

            'loop_prob' : {},
            'add_n_loops' : {},
            'add_n_corr' : {},
            'loop_u_trace' : {},
            'loop_trace_u_eff' : {},
            'loop_delta_blind_v' : {},
            'loop_pos_stdev_trace' : {},
            'loop_cost_maps' : {},
            'max_loop_cost_maps' : {},
            'max_loop_pos' : {},

            'meet_u' : {},
            'meet_add_n_meeting_step' : {},
            'meet_add_n_new_neighbours' : {},
            'meet_add_n_meeting_meta_loops' : {},
            'meet_add_n_meeting_semi_loops' : {},
            'meet_add_n_meeting_corr' : {},
            'meet_add_n_agents' : {},
            'meet_add_n_known_square' : {},
            'meet_add_n_obs' : {},

            'meet_m_pos' : {},
            'meet_curr_step' : {},
            'meet_last_step_contact' : {},

            'presence_distrib' : {},
            'meet_distrib' : {},

            'meet_cost_maps' : {},
            'max_meet_cost_maps' : {},

            'final_cost_map' : {},
            'max_gain' : {},

            'penalty_points' : {},
        }

    def reset_cm_exec_time_dic(self):
        self.cm_exec_time = {
            'explore_cost_map' : [],

            'loop_cost_map' : [],
            'loop_cost_map - get_loop_trace_u_x_distrib' : [],
            'loop_cost_map - i_operations' : [],
            'loop_cost_map - n_operations' : [],
            'loop_cost_map - max_j_operations' : [],
            'loop_cost_map - f_operations' : [],

            'loop_cost_map - max_stdev' : [],
            'loop_cost_map - max_blind_v' : [],
            'loop_cost_map - sum_blind_v' : [],

            'loop_cost_map - get loop trace u x distrib - part1' : [],
            'loop_cost_map - get loop trace u x distrib - part1toc' : [],
            'loop_cost_map - get loop trace u x distrib - part2' : [],
            'loop_cost_map - get loop trace u x distrib - part2toc1' : [],
            'loop_cost_map - get loop trace u x distrib - part2toc1toc1a' : [],
            'loop_cost_map - get loop trace u x distrib - part2toc1toc1b' : [],
            'loop_cost_map - get loop trace u x distrib - part2toc1toc1c' : [],
            'loop_cost_map - get loop trace u x distrib - part2toc1toc2' : [],
            'loop_cost_map - get loop trace u x distrib - part2toc1toc3' : [],
            'loop_cost_map - get loop trace u x distrib - part2toc2' : [],

            'presence_distrib - init' : [],
            'presence_distrib - mc' : [],
            'presence_distrib - mc - part1' : [],
            'presence_distrib - mc - part2a' : [],
            'presence_distrib - mc - part2b' : [],
            'presence_distrib - mc - part3' : [],
            'presence_distrib - mc - part4' : [],
            'presence_distrib - mc - part5' : [],
            'presence_distrib - mc - ftime' : [],
            'presence_distrib - mc - fn_calls' : [],
            'presence_distrib - mc - n_steps' : [],
            'presence_distrib - mc - n_steps_list' : [],
            'presence_distrib - mc - n_parts' : [],
            'presence_distrib - mc - selection' : [],
            'presence_distrib - plans' : [],
            'presence_distrib - simple' : [],
            'presence_distrib - global' : [],
            'meet_cost_maps' : [],
        }

        self.cm_exec_stats = {}
    
    def get_potential_composition(self, pos):
        return {
            'explore' : self.explore_potential[pos],
            'loop' : {l_id : self.loop_cost_maps[l_id][pos] for l_id in self.loop_cost_maps},
            'meet' : {m_id : self.meet_cost_maps[m_id][pos] for m_id in self.meet_cost_maps if type(self.meet_cost_maps[m_id]) is np.ndarray},
        }

    
    #---explore potential---
    def get_n_unknown_squares_in_range(self):
        n_unknown_squares = np.zeros_like(self.submaps.ag_map)
        for i in range(self.height):
            for j in range(self.width):
                if self.submaps.is_known_or_on_frontier((i,j)): #only consider a square belonging to the extented map
                    for k in range(max(i-self.scan_range, 0), min(i+self.scan_range +1, self.height)):
                        for l in range(max(j-self.scan_range, 0), min(j+self.scan_range +1, self.width)):
                            rel_pos = (k,l)
                            if not self.submaps.is_known(rel_pos):
                                n_unknown_squares[i,j] += 1
        return n_unknown_squares

    def get_team_visited_pos(self, tracks):
        team_visited_pos = np.zeros_like(self.submaps.ag_map)
        for ag_id in tracks:
            for pos in tracks[ag_id].trace.ag_pos:
                if self.submaps.on_submap(pos):
                    team_visited_pos[pos[0], pos[1]] = 1
        return team_visited_pos
    
    def get_explore_potential(self, tracks, disp = False, measure_time = True):
        if disp : print(" Getting explore potential map ...")
        if measure_time : tnc_e = time.time()

        explore_prob = 1

        add_known_squares = self.get_n_unknown_squares_in_range()
        add_team_visited_pos = np.ones(self.submaps.ag_map.shape) - self.get_team_visited_pos(tracks)
        add_obs = np.ones(self.submaps.ag_map.shape)
        explore_u = add_known_squares * RewardParameters.N_SQUARE_KNOWN + add_team_visited_pos * RewardParameters.N_VISITED_POS + add_obs * RewardParameters.N_OBS
        
        explore_distrib = np.ones(self.submaps.ag_map.shape)
        
        explore_potential = explore_prob * explore_u * explore_distrib
        explore_potential = explore_potential.astype(int)

        if disp : print(" explore_potential :\n", explore_potential)
        if measure_time : self.cm_exec_time['explore_cost_map'].append(round(time.time()-tnc_e,2))
        return explore_potential    





    #---loop cost map--
    def get_loop_cost_map(self, tr_id, tracks, disp = False, measure_time = True):
        if disp : print(" Getting loop potential map ...")
        if measure_time : tnc_l = time.time()

        self_track = tracks[self.self_id]
        ag_track = tracks[tr_id]

        curr_step = self_track.obs_list[-1].time_step
        curr_ag_pos = self_track.obs_list[-1].pos_belief[-1]['pos_belief']
        curr_blind_v = self_track.obs_list[-1].pos_belief[-1]['blind_v']

        #display
        if disp : print("Trace caracteristics :")
        if disp : print("self id :", self.self_id)
        if disp : print("tr id :", tr_id)

        if disp : print("track ag trace :", ag_track.trace.ag_pos)
        if disp : print(" len track ag trace :", len(ag_track.trace.ag_pos))
        if disp : print(" curr_ag_pos :", curr_ag_pos)

        if disp : print("trace steps :", ag_track.trace.time_steps)
        if disp : print(" curr_step :", curr_step)

        if disp : print("trace blind_v :", ag_track.trace.blind_v)
        if disp : print(" len trace blind_v :", len(ag_track.trace.blind_v))
        if disp : print(" curr_blind_v :", curr_blind_v)
        if disp : print("trace n_corr :", ag_track.trace.n_corr)

        #-probability to loop- (always equal to 1)
        self.loop_prob[tr_id] = 1

        #-incremente each metrics-
        #a loop here corresponds to a self loop or a ma meta loop
        #the n_loops is incremented by 1 iif the current pos is different from the trace point
        #add_n_loops = [1 * (curr_step != ag_track.trace.time_steps[i]) for i in range(len(ag_track.trace.ag_pos))]
        self.add_n_loops[tr_id] = [1 * (curr_ag_pos != ag_track.trace.ag_pos[i]) for i in range(len(ag_track.trace.ag_pos))]
        #add_semi_loops = len(ag_track.meta_ext) #unused
        
        #the n_corr expected is equal to the blind_v delta
        if tr_id == self.self_id: #self loop
            self.add_n_corr[tr_id] = [abs(curr_blind_v - self_track.trace.blind_v[i]) for i in range(len(self_track.trace.ag_pos))]
        else: #ma loop with other agent : n_corr expected is sum of n_corr on self trace and on agent trace (intermediate traces not considered)
        #in case of meeting another agent, set the last meeting step and the last meeting blind v
            meeting_step = ag_track.owners[-1]['from step']
            meeting_self_blind_v = self_track.obs_list[meeting_step-1].pos_belief[-1]['blind_v']
            if disp : print(" meeting_step :", meeting_step)
            if disp : print(" meeting_self_blind_v :", meeting_self_blind_v)            
            self.add_n_corr[tr_id] = [abs(curr_blind_v - meeting_self_blind_v) + abs(ag_track.trace.blind_v[-1] - ag_track.trace.blind_v[i]) for i in range(len(ag_track.trace.ag_pos))]

        #the add_n_corr_trace list must be decreasing
        for i in range(1, len(ag_track.trace.ag_pos) +1):
            self.add_n_corr[tr_id][-i] = max(self.add_n_corr[tr_id][-i:])
        
        #display
        if disp : print("Trace utility :")
        if disp : print("add_n_loop :", [round(elem, 2) for elem in self.add_n_loops[tr_id]])
        if disp : print("add_n_corr :", [round(elem, 2) for elem in self.add_n_corr[tr_id]])

        #sum the utilities
        self.loop_u_trace[tr_id] = [self.add_n_loops[tr_id][i] * RewardParameters.N_MA_META_LOOPS + self.add_n_corr[tr_id][i] * RewardParameters.N_CORRECTIONS for i in range(len(ag_track.trace.ag_pos))]
        if disp : print("loop_u_trace :", [round(elem, 2) for elem in self.loop_u_trace[tr_id]])

        #traces distribution
        #get the agent's stdev_trace with the blind_v trace
        if tr_id == self.self_id: #self loop
            #self.loop_pos_stdev_trace[tr_id] = [1/(track.trace.n_corr[i]+1) for i in range(len(track.trace.ag_pos))]
            self.loop_delta_blind_v[tr_id] = [abs(curr_blind_v - self_track.trace.blind_v[i_move]) for i_move in range(len(ag_track.trace.ag_pos))] #should be always positive
        else: #ma loop with other agent : distribution variance expected is based on the blind_v of self trace and of agent trace (intermediate traces not considered)
            meeting_step = ag_track.owners[-1]['from step']
            meeting_self_blind_v = self_track.obs_list[meeting_step-1].pos_belief[-1]['blind_v']
            if disp : print(" meeting_step :", meeting_step)
            if disp : print(" meeting_self_blind_v :", meeting_self_blind_v) 
            self.loop_delta_blind_v[tr_id] = [abs(curr_blind_v - meeting_self_blind_v) + abs(ag_track.trace.blind_v[-1] - ag_track.trace.blind_v[i_move]) for i_move in range(len(ag_track.trace.ag_pos))]
        
        #round stdev trace
        self.loop_pos_stdev_trace[tr_id] = [round(get_stdev_error(elem, AgentParameters.ODOM_ERROR_RATE), 3) for elem in self.loop_delta_blind_v[tr_id]]
        if disp : print("loop_pos_stdev_trace :", self.loop_pos_stdev_trace[tr_id])
        
        #get the cost map
        if disp : print(" enter")
        if measure_time : tnc = time.time()
        self.loop_trace_u_x_distrib[tr_id], self.loop_trace_u_eff[tr_id], (i_operations, n_operations, j_operations_list, f_operations, exec_time_dic) = get_loop_trace_u_x_distrib(
            self.submaps, ag_track.trace, self.loop_pos_stdev_trace[tr_id], self_track.trace, self.loop_pos_stdev_trace[self.self_id], self.loop_u_trace[tr_id], disp)
        if measure_time : self.cm_exec_time['loop_cost_map - get_loop_trace_u_x_distrib'].append(round(time.time()-tnc,2))
        if measure_time : self.cm_exec_time['loop_cost_map - i_operations'].append(i_operations)
        if measure_time : self.cm_exec_time['loop_cost_map - n_operations'].append(n_operations)
        if measure_time and j_operations_list != [] : self.cm_exec_time['loop_cost_map - max_j_operations'].append(max(j_operations_list))
        if measure_time : self.cm_exec_time['loop_cost_map - f_operations'].append(f_operations)
        
        if measure_time and self.loop_pos_stdev_trace[tr_id] != [] : self.cm_exec_time['loop_cost_map - max_stdev'].append(max(self.loop_pos_stdev_trace[tr_id]))
        if measure_time and self.loop_delta_blind_v[tr_id] != [] : self.cm_exec_time['loop_cost_map - max_blind_v'].append(max(self.loop_delta_blind_v[tr_id]))
        if measure_time : self.cm_exec_time['loop_cost_map - sum_blind_v'].append(sum(self.loop_delta_blind_v[tr_id]))
        
        if measure_time : self.cm_exec_time['loop_cost_map - get loop trace u x distrib - part1'].append(round(sum(exec_time_dic['part1']), 4))
        if measure_time : self.cm_exec_time['loop_cost_map - get loop trace u x distrib - part1toc'].append(round(sum(exec_time_dic['part1toc']), 4))
        if measure_time : self.cm_exec_time['loop_cost_map - get loop trace u x distrib - part2'].append(round(sum(exec_time_dic['part2']), 4))
        if measure_time : self.cm_exec_time['loop_cost_map - get loop trace u x distrib - part2toc1'].append(round(sum(exec_time_dic['part2toc1']), 4))
        if measure_time : self.cm_exec_time['loop_cost_map - get loop trace u x distrib - part2toc1toc1a'].append(round(sum(exec_time_dic['part2toc1toc1a']), 4))
        if measure_time : self.cm_exec_time['loop_cost_map - get loop trace u x distrib - part2toc1toc1b'].append(round(sum(exec_time_dic['part2toc1toc1b']), 4))
        if measure_time : self.cm_exec_time['loop_cost_map - get loop trace u x distrib - part2toc1toc1c'].append(round(sum(exec_time_dic['part2toc1toc1c']), 4))
        if measure_time : self.cm_exec_time['loop_cost_map - get loop trace u x distrib - part2toc1toc2'].append(round(sum(exec_time_dic['part2toc1toc2']), 4))
        if measure_time : self.cm_exec_time['loop_cost_map - get loop trace u x distrib - part2toc1toc3'].append(round(sum(exec_time_dic['part2toc1toc3']), 4))
        if measure_time : self.cm_exec_time['loop_cost_map - get loop trace u x distrib - part2toc2'].append(round(sum(exec_time_dic['part2toc2']), 4))

        #multiply by prob
        loop_cost_map = self.loop_prob[tr_id] * self.loop_trace_u_x_distrib[tr_id]
        
        #cut to int
        loop_cost_map = loop_cost_map.astype(int)
        if disp : print("loop_cost_map :\n", loop_cost_map)
        if measure_time : self.cm_exec_time['loop_cost_map'].append(round(time.time()-tnc_l,2))
        return loop_cost_map





    #---meet potential---
    def get_meet_cost_maps(self, m_id, tracks, metrics, curr_step, disp = False, measure_time = True):
        if disp : print(" Getting meet potential map ...")
        if measure_time : tnc_m = time.time()
        if disp : print(" m_id :", m_id)

        m_pos = self.submaps.ag_team_pos[m_id]
        if disp : print(" last time agent has been seen : m_pos :", m_pos)

        #prob
        meet_prob = 1

        #define last step contact and last agent contact
        last_step_contact = max(m_pos[observer]['time_step'] for observer in m_pos)
        candidates = []
        for observer in m_pos:
            if m_pos[observer]['time_step'] == last_step_contact:
                candidates.append(observer)
        last_agent_contact = max(candidates)
        last_pos_contact = m_pos[last_agent_contact]['seen_pos']

        #utility
        add_n_meeting_step = 1
        add_n_new_neighbours = 1 * (m_id not in tracks[self.self_id].obs_list[-1].neighbours)
        add_n_meeting_meta_loops = round(metrics['agent']['n_meeting_meta_loops']/(metrics['agent']['n_new_neighbours'] +1e-5) * add_n_new_neighbours, 2) #empirical estimations
        add_n_meeting_semi_loops = round(metrics['agent']['n_meeting_semi_loops']/(metrics['agent']['n_new_neighbours'] +1e-5) * add_n_new_neighbours, 2)
        
        add_n_meeting_corr = round(metrics['agent']['n_meeting_corrections']/(metrics['agent']['n_new_neighbours'] +1e-5) * (1 - last_step_contact/curr_step) * 0.5 * add_n_new_neighbours, 2)
        add_n_agents = round((metrics['tracks']['n_tracks'] -1)/(metrics['agent']['n_new_neighbours'] +1e-5) * (1 - last_step_contact/curr_step) * add_n_new_neighbours, 2)

        n_tot = self.height * self.width
        n_known = metrics['submaps']['n_squares_known']

        add_n_known_square = round((n_tot - n_known) * n_known/n_tot * (1 - last_step_contact/curr_step) * 0.5, 2)
        add_n_obs = round(metrics['tracks']['n_obs'] * (1 - last_step_contact/curr_step), 2)

        if disp : print(" add_n_meeting_step : ", add_n_meeting_step)
        if disp : print(" add_n_new_neighbours : ", add_n_new_neighbours)
        if disp : print(" add_n_meeting_meta_loops : ", add_n_meeting_meta_loops)
        if disp : print(" add_n_meeting_semi_loops : ", add_n_meeting_semi_loops)
        if disp : print(" add_n_meeting_corr : ", add_n_meeting_corr)
        if disp : print(" add_n_known_square : ", add_n_known_square)
        if disp : print(" add_n_obs : ", add_n_obs)

        meet_u = sum([
            add_n_meeting_step * RewardParameters.N_MEETINGS,
            add_n_new_neighbours * RewardParameters.N_NEW_NEIGHBOURS,
            add_n_meeting_meta_loops * RewardParameters.N_MEETING_META_LOOPS,
            add_n_meeting_corr * RewardParameters.N_CORRECTIONS,
            add_n_agents * RewardParameters.N_AGENTS,
            add_n_known_square * RewardParameters.N_SQUARE_KNOWN,
            add_n_obs * RewardParameters.N_OBS,
        ])

        meet_u = round(meet_u, 2)
        if disp : print(" utility : ", meet_u)

        #update variables
        self.meet_u[m_id] = meet_u

        self.meet_add_n_meeting_step[m_id] = add_n_meeting_step
        self.meet_add_n_new_neighbours[m_id] = add_n_new_neighbours
        self.meet_add_n_meeting_meta_loops[m_id] = add_n_meeting_meta_loops
        self.meet_add_n_meeting_semi_loops[m_id] = add_n_meeting_semi_loops
        self.meet_add_n_meeting_corr[m_id] = add_n_meeting_corr
        self.meet_add_n_agents[m_id] = add_n_agents
        self.meet_add_n_known_square[m_id] = add_n_known_square
        self.meet_add_n_obs[m_id] = add_n_obs

        self.meet_m_pos[m_id] = m_pos
        self.meet_curr_step[m_id] = curr_step
        self.meet_last_step_contact[m_id] = last_step_contact

        if disp : print(" m_id : ", m_id)
        if disp : print(" curr_step : ", curr_step)
        if disp : print(" last_step_contact : ", last_step_contact)
        if disp : print(" last_agent_contact : ", last_agent_contact)
        if disp : print(" last_pos_contact : ", last_pos_contact)

        #meet distrib
        if type(self.presence_distrib[m_id]) is np.ndarray:
            self.meet_distrib[m_id] = get_meet_pos_distrib(self.submaps, self.presence_distrib[m_id], self.scan_range)
            if disp : print(" meet distrib : \n", (self.meet_distrib[m_id]*100).astype(int))

            #cost map
            meet_cost_map = meet_prob * (meet_u + CostmapsParameters.MEET_OFFSET) * self.meet_distrib[m_id]
            meet_cost_map = meet_cost_map.astype(int)
            if disp : print(" meet_cost_map : \n", meet_cost_map.astype(int))
            if measure_time : self.cm_exec_time['meet_cost_maps'].append(round(time.time()-tnc_m,2))
            return meet_cost_map
        else:
            self.meet_distrib[m_id] = None
            return None


    def update_presence_distrib(self, tracks, ag_plans, curr_step, distrib_method = 'mc', disp = False, measure_time = True):
        if True : print(" Updating presence distrib ...")
        if measure_time : tdc = time.time()
        for m_id in tracks:
            if m_id == self.self_id:
                continue
            if measure_time : tdc_i = time.time()
            if disp : print(" m_id :", m_id)

            m_pos = self.submaps.ag_team_pos[m_id]
            if disp : print(" last time agent ", m_id, "has been seen : m_pos :", m_pos)

            #define last step contact and last agent contact
            last_step_contact = max(m_pos[observer]['time_step'] for observer in m_pos)
            candidates = []
            for observer in m_pos:
                if m_pos[observer]['time_step'] == last_step_contact:
                    candidates.append(observer)
            last_agent_contact = max(candidates)
            last_pos_contact = m_pos[last_agent_contact]['seen_pos']

            if curr_step - last_step_contact + (-1 * (m_id >= self.self_id and last_agent_contact >= m_id)) > self.lost_steps:
                if disp : print('agent', m_id, 'lost')
                self.presence_distrib[m_id] = None
                continue
            if measure_time : self.cm_exec_time['presence_distrib - init'].append(round(time.time()-tdc_i,2))

            #distrib
            if measure_time : tdc_d = time.time()
            if distrib_method == 'mc':
                self.presence_distrib[m_id], exec_time_dic = get_agent_distrib_mc(
                    self.submaps, tracks, self.self_id, m_id, curr_step, 
                    last_step_contact, last_pos_contact, last_agent_contact, ag_plans[m_id],
                    tracks_added = None,
                    trust_plan_factor = self.trust_factor, lost_steps = self.lost_steps, impact_steps = self.impact_steps, scan_range = self.scan_range,
                    disp = disp, measure_time = True)
                
                if measure_time : self.cm_exec_time['presence_distrib - mc'].append(round(time.time()-tdc_d,2))
                if measure_time : self.cm_exec_time['presence_distrib - mc - part1'].append(round(sum(exec_time_dic['part1']), 2))
                if measure_time : self.cm_exec_time['presence_distrib - mc - part2a'].append(round(sum(exec_time_dic['part2a']), 2))
                if measure_time : self.cm_exec_time['presence_distrib - mc - part2b'].append(round(sum(exec_time_dic['part2b']), 2))
                if measure_time : self.cm_exec_time['presence_distrib - mc - part3'].append(round(sum(exec_time_dic['part3']), 2))
                if measure_time : self.cm_exec_time['presence_distrib - mc - part4'].append(round(sum(exec_time_dic['part4']), 2))
                if measure_time : self.cm_exec_time['presence_distrib - mc - part5'].append(round(sum(exec_time_dic['part5']), 2))
                if measure_time : self.cm_exec_time['presence_distrib - mc - ftime'].append(round(sum(exec_time_dic['ftime']), 2))
                if measure_time : self.cm_exec_time['presence_distrib - mc - fn_calls'].append(len(exec_time_dic['ftime']))
                if measure_time : self.cm_exec_time['presence_distrib - mc - n_steps'].append(round(sum(exec_time_dic['n_steps']), 2))
                if measure_time : self.cm_exec_time['presence_distrib - mc - n_steps_list'].append(exec_time_dic['n_steps'])
                if measure_time : self.cm_exec_time['presence_distrib - mc - n_parts'].append(round(sum(exec_time_dic['n_parts']), 2))
                if measure_time : self.cm_exec_time['presence_distrib - mc - selection'].append(exec_time_dic['selection'])

            elif ag_plans[m_id] != None:
                self.presence_distrib[m_id] = get_agent_distrib_plans(
                    self.submaps, tracks, self.self_id, m_id, curr_step, 
                    last_step_contact, last_pos_contact, last_agent_contact, ag_plans[m_id],
                    None,
                    self.trust_factor, self.lost_steps, self.impact_steps, self.scan_range,
                    disp = disp)
                if measure_time : self.cm_exec_time['presence_distrib - plans'].append(round(time.time()-tdc_d,2))
            else:
                self.presence_distrib[m_id] = get_agent_distrib(
                    self.submaps, tracks, self.self_id, m_id, curr_step, 
                    last_step_contact, last_pos_contact, last_agent_contact,
                    None,
                    self.lost_steps, self.impact_steps, self.scan_range,
                    disp = disp, rnd = 3)
                if measure_time : self.cm_exec_time['presence_distrib - simple'].append(round(time.time()-tdc_d,2))

            if disp and self.presence_distrib[m_id] is not False : print(" presence distrib of ", m_id, ": \n", (self.presence_distrib[m_id]*1000).astype(int))
        
        if measure_time : self.cm_exec_time['presence_distrib - global'].append(round(time.time()-tdc,2))


    #---main function---
    def update_potential_maps(self, tracks, metrics, curr_step, disp = False):
        #explore map
        if CostmapsParameters.EXPLORE_WEIGHT > 0:
            self.explore_potential = self.get_explore_potential(tracks, disp)
            if self.max_explore == None: #first round
                self.max_explore = np.max(self.explore_potential) * CostmapsParameters.EXPLORE_WEIGHT

        #loop map
        if CostmapsParameters.LOOP_WEIGHT > 0:
            for ag_id in tracks:
                self.loop_cost_maps[ag_id] = self.get_loop_cost_map(ag_id, tracks, disp) #can be True

        #meet map
        if CostmapsParameters.MEET_WEIGHT > 0:
            for m_id in tracks:
                if m_id != self.self_id:
                    self.meet_cost_maps[m_id] = self.get_meet_cost_maps(m_id, tracks, metrics, curr_step, disp)

    def remove_obstacles_and_out(self, potential_map):
        new_cost_map = potential_map
        for i in range(self.height):
            for j in range(self.width):
                pos = (i,j)
                if self.submaps.is_out(pos) or self.submaps.is_obstacle(pos):
                    new_cost_map[i,j] = 0
        return new_cost_map
    
    def remove_points(self, potential_map, points_list):
        new_cost_map = potential_map
        for i in range(self.height):
            for j in range(self.width):
                if (i,j) in points_list:
                    new_cost_map[i,j] = 0
        return new_cost_map
    
    def update_global_cost_map(self, disp = False):
        #init (reset)
        global_cost_map = np.zeros(self.submaps.ag_map.shape)

        #explore
        if CostmapsParameters.EXPLORE_WEIGHT > 0:
            global_cost_map += self.explore_potential * CostmapsParameters.EXPLORE_WEIGHT

        #loop
        if CostmapsParameters.LOOP_WEIGHT > 0:
            global_cost_map += np.maximum.reduce([self.loop_cost_maps[ag_id] for ag_id in self.loop_cost_maps]) * CostmapsParameters.LOOP_WEIGHT #model max
            #global_cost_map += np.sum([self.loop_cost_maps[ag_id] for ag_id in self.loop_cost_maps], axis=0) * CostmapsParameters.LOOP_WEIGHT #model sum

        #meet
        if CostmapsParameters.MEET_WEIGHT > 0 and len(self.meet_cost_maps) > 0 :
            #global_cost_map += np.maximum.reduce([self.meet_cost_maps[m_id] for m_id in self.meet_cost_maps]) * CostmapsParameters.MEET_WEIGHT #model max
            global_cost_map += np.sum([self.meet_cost_maps[m_id] for m_id in self.meet_cost_maps if type(self.meet_cost_maps[m_id]) is np.ndarray], axis=0) * CostmapsParameters.MEET_WEIGHT #model sum
        
        #int
        self.global_cost_map = global_cost_map.astype(int)

        #get max_gain
        self.max_gain = np.max(self.global_cost_map)

        #normalise such that max = 1
        if np.max(global_cost_map) > 0:
            #self.norm_global_cost_map = self.global_cost_map / max(np.max(global_cost_map), 30) #30 is arbitraly fixed
            self.norm_global_cost_map = self.global_cost_map / max(np.max(self.explore_potential), self.max_explore)

        #print
        if disp : print(self.global_cost_map)
        if disp : print(self.norm_global_cost_map)

    def is_exploring_interesting(self, value):
        return self.max_gain > value        

    def update_cm_metrics(self):
        self.cm_metrics['explore_potential'] = self.explore_potential
        self.cm_metrics['max_explore_potential'] = np.max(self.explore_potential)
        
        self.cm_metrics['loop_prob'] = {ag_id : self.loop_prob[ag_id] for ag_id in self.loop_prob}
        self.cm_metrics['add_n_loops'] = {ag_id : self.add_n_loops[ag_id] for ag_id in self.add_n_loops}
        self.cm_metrics['add_n_corr'] = {ag_id : self.add_n_corr[ag_id] for ag_id in self.add_n_corr}
        self.cm_metrics['loop_u_trace'] = {ag_id : self.loop_u_trace[ag_id] for ag_id in self.loop_u_trace}
        self.cm_metrics['loop_trace_u_eff'] = {ag_id : self.loop_trace_u_eff[ag_id] for ag_id in self.loop_trace_u_eff}
        self.cm_metrics['loop_delta_blind_v'] = {ag_id : self.loop_delta_blind_v[ag_id] for ag_id in self.loop_delta_blind_v}
        self.cm_metrics['loop_pos_stdev_trace'] = {ag_id : self.loop_pos_stdev_trace[ag_id] for ag_id in self.loop_pos_stdev_trace}
        self.cm_metrics['loop_cost_maps'] = {ag_id : self.loop_cost_maps[ag_id] for ag_id in self.loop_cost_maps}
        self.cm_metrics['max_loop_cost_maps'] = {ag_id : np.max(self.loop_cost_maps[ag_id]) for ag_id in self.loop_cost_maps}
        self.cm_metrics['max_loop_pos'] = {ag_id : np.unravel_index(self.loop_cost_maps[ag_id].argmax(), (self.height, self.width)) for ag_id in self.loop_cost_maps}

        self.cm_metrics['meet_u'] = {m_id : self.meet_u[m_id] for m_id in self.meet_u}
        self.cm_metrics['meet_add_n_meeting_step'] = {m_id : self.meet_add_n_meeting_step[m_id] for m_id in self.meet_cost_maps}
        self.cm_metrics['meet_add_n_new_neighbours'] = {m_id : self.meet_add_n_new_neighbours[m_id] for m_id in self.meet_cost_maps}
        self.cm_metrics['meet_add_n_meeting_meta_loops'] = {m_id : self.meet_add_n_meeting_meta_loops[m_id] for m_id in self.meet_cost_maps}
        #self.cm_metrics['meet_add_n_meeting_semi_loops'] = {m_id : self.meet_add_n_meeting_semi_loops[m_id] for m_id in self.meet_cost_maps}
        self.cm_metrics['meet_add_n_meeting_corr'] = {m_id : self.meet_add_n_meeting_corr[m_id] for m_id in self.meet_cost_maps}
        self.cm_metrics['meet_add_n_agents'] = {m_id : self.meet_add_n_agents[m_id] for m_id in self.meet_cost_maps}
        self.cm_metrics['meet_add_n_known_square'] = {m_id : self.meet_add_n_known_square[m_id] for m_id in self.meet_cost_maps}
        #self.cm_metrics['meet_add_n_obs'] = {m_id : self.meet_add_n_obs[m_id] for m_id in self.meet_cost_maps}
        
        self.cm_metrics['meet_m_pos'] = {m_id : self.meet_m_pos[m_id] for m_id in self.meet_cost_maps}
        #self.cm_metrics['meet_curr_step'] = {m_id : self.meet_curr_step[m_id] for m_id in self.meet_cost_maps}
        self.cm_metrics['meet_last_step_contact'] = {m_id : self.meet_last_step_contact[m_id] for m_id in self.meet_cost_maps}
        
        self.cm_metrics['presence_distrib'] = {m_id : self.presence_distrib[m_id] for m_id in self.meet_cost_maps}
        #self.cm_metrics['meet_distrib'] = {m_id : self.meet_distrib[m_id] for m_id in self.meet_cost_maps}

        self.cm_metrics['meet_cost_maps'] = {m_id : self.meet_cost_maps[m_id] for m_id in self.meet_cost_maps}
        self.cm_metrics['max_meet_cost_maps'] = {m_id : np.max(self.meet_cost_maps[m_id]) for m_id in self.meet_cost_maps}
        
        self.cm_metrics['final_cost_map'] = self.global_cost_map
        self.cm_metrics['max_gain'] = self.max_gain
        
    def update_costmaps(self, submaps, tracks, metrics, team_plans, curr_step, disp = False):
        print('Updating costmaps ...')
        self.submaps = submaps
        self.reset_cm_metrics(curr_step)
        self.update_presence_distrib(tracks, team_plans, curr_step, distrib_method = self.distrib_method, disp = disp)
        self.update_potential_maps(tracks, metrics, curr_step, disp)
        self.update_global_cost_map(disp)
        self.update_cm_metrics()









class ViewpointPlanner:
    def __init__(self, submaps):
        #external variables
        self.submaps = submaps

        #goal variables
        self.reset_vpp_var()

        #metrics
        self.reset_vpp_metrics(0)

        #exec time
        self.reset_vpp_exec_time_dic()

    def reset_vpp_var(self):
        self.has_set_goal = None
        self.critical_value = None
        self.n_potential_targets = None
        self.potential_targets = None
        self.cluster_range = None
        self.n_targets = None
        self.targets = None
        self.targets_cost_function = None
        self.path_dist = None
        self.path_gains = None
        self.path_steps = None
        self.expected_gain = None
        self.expected_dist = None
        self.gain_composition = None
        self.new_multi_goal_set = None
        self.new_goal_set = None
        self.n_dfs = None

    def reset_vpp_metrics(self, curr_step):
        #metrics
        self.vpp_metrics = {
            'step' : curr_step,
            'has_set_goal': None,
            'critical_value' : None,
            'n_potential_targets' : None,
            'potential_targets' : None,
            'cluster_range' : None,
            'n_targets' : None,
            'targets_cost_function' : None,
            'targets' : None,
            'path_gains' : None,
            'expected_gain' : None,
            'gain_composition' : None,
            'new_multi_goal_set' : None,
            'new_goal_set' : None,
            'n_dfs' : None,
        }

    def reset_vpp_exec_time_dic(self):
        self.vpp_exec_time = {
            'max cm goal' : [],
            'fast cm goal' : [],
            'multi_goals - global' : [],
            'multi_goals - cluster' : [],
            'multi_goals - motsp time' : [],
            'multi_goals - motsp n_dfs' : [],
        }

        self.vpp_exec_stats = {}

    #---goal functions---
    def get_rd_pos(self, max_range = None, accessible = True):
        list_p = self.submaps.get_ext_map_points_list(max_range = max_range)

        if accessible:
            list_c = self.submaps.get_points_connected_to_agent_list(max_range = max_range)
            list_r = list(set(list_p) & set(list_c))
        else:
            list_r = list_p

        if list_r == []:
            return None
        else :
            return rd.choice(list_r)
        
    def get_rd_pos_in_known_space(self, max_range = None, accessible = True, banned_waypoints = []):
        list_p = self.submaps.get_known_points_list(max_range = max_range)

        if accessible:
            list_c = self.submaps.get_points_connected_to_agent_list(max_range = max_range)
            list_k = list(set(list_p) & set(list_c))
        else:
            list_k = list_p

        for b_w in banned_waypoints:
            if b_w in list_k:
                list_k.remove(b_w)
        if list_k == []:
            return None
        else :
            s=0
            while True:
                pos = rd.choice(list_k)
                if self.submaps.is_free(pos):
                    return pos
                s+=1
                if s>10:
                    print("Error - random goal in known space failed")
                    return None
            
    def get_rd_pos_in_frontier_space(self, max_range = None, accessible = True, banned_waypoints = []):
        list_p = self.submaps.get_frontier_points_list(max_range = max_range)

        if accessible:
            list_c = self.submaps.get_points_connected_to_agent_list(max_range = max_range)
            list_f = list(set(list_p) & set(list_c))
        else:
            list_f = list_p

        for b_w in banned_waypoints:
            if b_w in list_f:
                list_f.remove(b_w)
        if list_f == []:
            return None
        else :
            return rd.choice(list_f)
        
    def get_max_potential_pos(self, costmaps, max_range = None, accessible = True, banned_waypoints = [], disp = False, measure_time = True):
        if disp : print(" Setting goals wrt max potential map ...")
        if measure_time : tic_g = time.time()

        list_p = self.submaps.get_ext_map_points_list(max_range = max_range)

        if accessible:
            list_c = self.submaps.get_points_connected_to_agent_list(max_range = max_range)
            list_1 = list(set(list_p) & set(list_c))
        else:
            list_1 = list_p

        val = {pos : costmaps.global_cost_map[pos[0],pos[1]] for pos in list_1 if pos not in banned_waypoints}
        inverse = [(value, key) for key, value in val.items()]
        if inverse != []:
            max_val = max(inverse)[0]
            potential_val = [key for key, value in val.items() if value == max_val]
            new_goal = rd.choice(potential_val)
            self.expected_gain = val[new_goal] #record expected gain
            self.gain_composition = costmaps.get_potential_composition(new_goal)
        else:
            new_goal = False
            self.expected_gain = None
            self.gain_composition = None
        if measure_time : self.vpp_exec_time['max cm goal'].append(round(time.time()-tic_g,2))
        return new_goal
        
    def get_goal_wrt_potential_map(self, costmaps, max_range = None, accessible = True, min_range = 0, discount_rate = 1, banned_waypoints = [], disp = False, measure_time = True):
        if disp : print(" Setting goals wrt potential map ...")
        if measure_time : tic_g = time.time()
        
        list_p = self.submaps.get_ext_map_points_list(max_range = max_range)

        if accessible:
            list_c = self.submaps.get_points_connected_to_agent_list(max_range = max_range)
            list_1 = list(set(list_p) & set(list_c))
        else:
            list_1 = list_p

        list_2 = [pos for pos in list_1 if pos not in banned_waypoints and distance_manhattan(pos, self.submaps.ag_pos) >= min_range] #add min range

        cm = {pos : costmaps.global_cost_map[pos[0],pos[1]] for pos in list_2}
        dist = {pos : distance_manhattan(pos, self.submaps.ag_pos) for pos in list_2}
        val = {pos : cm[pos]*(discount_rate**dist[pos])/dist[pos] for pos in list_2}
        inverse = [(value, key) for key, value in val.items()]
        if inverse != []:
            max_val = max(inverse)[0]
            potential_val = [key for key, value in val.items() if value == max_val]
            new_goal = rd.choice(potential_val)
            self.targets_cost_function = cm[new_goal]
            self.path_dist = dist[new_goal]
            self.expected_gain = val[new_goal] #record expected gain
            self.gain_composition = costmaps.get_potential_composition(new_goal)
        else:
            new_goal = False
            self.expected_gain = None
            self.gain_composition = None
        if measure_time : self.vpp_exec_time['fast cm goal'].append(round(time.time()-tic_g,2))
        return new_goal

    def get_mutli_goals_wrt_potential_map(
            self, costmaps, max_n_goals = 20, min_n_goals = 8, max_goals_ratio = 15, max_value_tolerance_in = 0.5, max_value_tolerance_out = 0.1, tolerance_out = 1,
            max_range = None, accessible = True, min_range = 0, banned_waypoints = [], max_distance = 50, max_distance_btw_targets = 10, discount_rate = 0.95, discount_ratio = 0.7,
            cluster_method = 'v2', per_step = True,
            disp = False, measure_time = True):        
        if disp : print(" Setting multi goals wrt potential map ...")
        if measure_time : tic_mg = time.time()

        list_p = self.submaps.get_ext_map_points_list(max_range = max_range)

        if accessible:
            list_c = self.submaps.get_points_connected_to_agent_list(max_range = max_range)
            list_1 = list(set(list_p) & set(list_c))
        else:
            list_1 = list_p

        list_2 = [pos for pos in list_1 if pos not in banned_waypoints and distance_manhattan(pos, self.submaps.ag_pos) >= min_range] #add min range
        
        cm_values_dict = {pos : costmaps.global_cost_map[pos[0],pos[1]] for pos in list_2}
        cm_discounted_dict = {pos : costmaps.global_cost_map[pos[0],pos[1]]*discount_rate**distance_manhattan(pos, self.submaps.ag_pos) for pos in list_2}

        if cm_values_dict == {}:
            return False
        
        #choose a critical value
        n_goals = min(max(int(len(cm_values_dict)/max_goals_ratio), min_n_goals), min(max_n_goals, len(cm_values_dict)))
        values = [value for key, value in cm_discounted_dict.items()]
        values.sort(reverse = True)
        
        critical_val = max([min(values[n_goals-1], max(values)*max_value_tolerance_in), max(values)*max_value_tolerance_out, tolerance_out])

        #set possible targets with respect to the criteria
        potential_targets_dict = {pos : cm_discounted_dict[pos] for pos in cm_discounted_dict if cm_discounted_dict[pos] >= critical_val}
        
        #set the start
        start = self.submaps.ag_pos
        
        #main function
        max_try = 10
        k_try = 0

        #init
        cluster_range = 2
        while k_try < max_try:
            k_try += 1

            if cluster_method == 'v1':
                #set targets
                if measure_time : toc_mg = time.time()
                grid, targets, cost_function = cluster_v1(potential_targets_dict, cluster_range, self.submaps.ag_map.shape)
                if measure_time : self.vpp_exec_time['multi_goals - cluster'].append(round(time.time()-toc_mg,2))

            elif cluster_method == 'v2':
                if measure_time : toc_mg = time.time()
                targets = cluster_v2_targets(potential_targets_dict, cluster_range)
                clusters, cost_function = cluster_v2_assign(targets, cm_discounted_dict, cluster_range)
                grid = cluster_v2_grid(clusters, cost_function, self.submaps.ag_map.shape)
                if measure_time : self.vpp_exec_time['multi_goals - cluster'].append(round(time.time()-toc_mg,2))

            #increase cluster range if too many targets
            if len(targets) > n_goals:
                cluster_range += 1
                print(" error -", len(targets), "targets. Cluster range extent to", cluster_range)
                continue

            #else
            self.critical_value = critical_val
            self.n_potential_targets = len(potential_targets_dict)
            self.potential_targets = potential_targets_dict
            self.cluster_range = cluster_range
            self.n_targets = len(targets)
            self.targets = targets
            self.targets_cost_function = cost_function

            #call the function
            if disp : print(' ', len(targets), 'targets :', targets, 'start :', start)
            if disp : print('  max distance :', max_distance, ' ; max distance btw :', max_distance_btw_targets, ' ; discount rate :', discount_rate, ' ; discount ratio :', discount_ratio,)
            #if disp : print(' grid :\n', grid)
           
            if measure_time : toc_mg = time.time()
            if not per_step:
                multi_goals, path_gains, expected_gain, n_dfs = gain_optimal(grid, targets, start, max_distance, max_distance_btw_targets, discount_rate, discount_ratio)
            else:
                if disp : print(grid, targets, start, max_distance, max_distance_btw_targets, discount_rate, discount_ratio)
                multi_goals, path_steps, _, path_gains, _, expected_dist, expected_gain, n_dfs = gain_optimal_2(grid, targets, start, max_distance, max_distance_btw_targets, discount_rate, discount_ratio, disp = False)
                if disp : print(multi_goals, path_steps, _, path_gains, _, expected_dist, expected_gain, n_dfs)
            if measure_time : self.vpp_exec_time['multi_goals - motsp time'].append(round(time.time()-toc_mg,2))
            if measure_time : self.vpp_exec_time['multi_goals - motsp n_dfs'].append(n_dfs)
            
            #...has worked or not?
            if multi_goals == False:
                n_goals = int(len(targets)/1.2)
                cluster_range += 1
                print(" error - Multi Objective TSP failed with", len(targets), "targets. Max n_targets reduced to", n_goals, "; cluster range extent to", cluster_range)
                continue
            
            #else ...
            if disp : print(" Multi Objective TSP worked.")
            
            #update records
            self.path_gains = path_gains
            self.expected_gain = expected_gain
            self.n_dfs = n_dfs

            if per_step:
                self.path_steps = path_steps
                self.expected_dist = expected_dist
            break

        if measure_time : self.vpp_exec_time['multi_goals - global'].append(round(time.time()-tic_mg,2))
        return multi_goals
    
    def get_closest_frontier_goal(self, max_range, accessible = True, banned_waypoints = []):
        list_p = self.submaps.get_frontier_points_list(max_range = max_range)
        
        if accessible:
            list_c = self.submaps.get_points_connected_to_agent_list(max_range = max_range)
            list_f = list(set(list_p) & set(list_c))
        else:
            list_f = list_p

        for b_w in banned_waypoints:
            if b_w in list_f:
                list_f.remove(b_w)

        if list_f == []:
            print('error - frontier candidates list emply')
            return None
        else:
            #set the closest frontier
            closest_f_list = []
            ag_pos = self.submaps.ag_pos
            frontier_dist = 1e3
            for pot_w in list_f:
                w_dist = self.submaps.manhattan_dist(ag_pos, pot_w)
                if  w_dist < frontier_dist:
                    closest_f_list = [pot_w]
                    frontier_dist = w_dist
                elif w_dist == frontier_dist:
                    closest_f_list.append(pot_w)
                    frontier_dist = w_dist
            if closest_f_list == []:
                print('error - closest frontier list empty')
                return None
            else:
                return rd.choice(closest_f_list)



    #get goal (depending on vpp mode)
    def get_random_goal(self, max_range = None):
        self.reset_vpp_var()

        new_goal = self.get_rd_pos(max_range = max_range, accessible = True)
        self.has_set_goal = 'random'
        self.new_goal_set = new_goal
        return new_goal

    def get_rd_explore_goal(self, exploration_rate, max_range = None, rd_rate = 0, banned_waypoints = [], penalty_points = []):
        self.reset_vpp_var()

        rand = rd.uniform(0,1)
        if rand > rd_rate:
            explore = rd.uniform(0,1)
            if explore < exploration_rate:
                new_goal = self.get_rd_pos_in_frontier_space(max_range = max_range, accessible = True, banned_waypoints = banned_waypoints+penalty_points)
                self.has_set_goal = 'rd frontier'
            else:
                new_goal = self.get_rd_pos_in_known_space(max_range = max_range, accessible = True, banned_waypoints = banned_waypoints+penalty_points)
                self.has_set_goal = 'known'
        else:
            new_goal = self.get_rd_pos(max_range = max_range, accessible = True)
            self.has_set_goal = 'random'

        self.new_goal_set = new_goal        
        return new_goal
    
    def get_max_potential_goal(self, costmaps, pp_param, banned_waypoints = [], penalty_points = []):
        self.reset_vpp_var()

        #stop if no more interesting exploration
        if not costmaps.is_exploring_interesting(pp_param['INTEREST_THRES']):
            self.has_set_goal = 'max stopped'
            self.new_goal_set = True
            return True
        
        rand = rd.uniform(0,1)
        if rand > pp_param['RANDOM_RATE']:
            new_goal = self.get_max_potential_pos(costmaps, max_range = pp_param['PLANNER_RANGE'], accessible = True, banned_waypoints = banned_waypoints+penalty_points)
            self.has_set_goal = 'max'
        if not rand > pp_param['RANDOM_RATE'] or not new_goal:
            new_goal = self.get_rd_pos(max_range = pp_param['ST_PLANNING_RANGE'], accessible = True)
            self.has_set_goal = 'random'
        self.new_goal_set = new_goal        
        return new_goal
    
    def get_opt_potential_goal(self, costmaps, pp_param, banned_waypoints = [], penalty_points = []):
        self.reset_vpp_var()

        #stop if no more interesting exploration
        if not costmaps.is_exploring_interesting(pp_param['INTEREST_THRES']):
            self.has_set_goal = 'opt stopped'
            self.new_goal_set = True
            print('warning - explo not interesting')
            return True
        
        #fast short term
        rand = rd.uniform(0,1)
        if rand > pp_param['RANDOM_RATE']:
            new_goal = self.get_goal_wrt_potential_map(costmaps, 
                max_range = pp_param['PLANNER_RANGE'],
                accessible = True, 
                min_range = pp_param['MIN_RANGE'],
                discount_rate = pp_param['DISCOUNT_RATE'],
                banned_waypoints = banned_waypoints+penalty_points,
                )
            self.has_set_goal = 'short term'
        
        #random
        if not rand > pp_param['RANDOM_RATE'] or not new_goal:
            new_goal = self.get_rd_pos(max_range = pp_param['ST_PLANNING_RANGE'], accessible = True)
            self.has_set_goal = 'random'
        self.new_goal_set = new_goal
        return new_goal
    
    def get_long_term_multi_goals(self, costmaps, multi_goals_param, banned_waypoints = [], penalty_points = [], disp = False):
        self.reset_vpp_var()

        #stop if no more interesting exploration
        if not costmaps.is_exploring_interesting(multi_goals_param['INTEREST_THRES']):
            self.has_set_goal = 'long term stopped'
            self.new_multi_goal_set = []
            return True
        
        rand = rd.uniform(0,1)
        if rand > multi_goals_param['RANDOM_RATE']:
            planning_horizon = max(self.submaps.get_diameter() * 1.5, multi_goals_param['MAX_HORIZON_PLANNING'])
            multi_goals = self.get_mutli_goals_wrt_potential_map(costmaps,
                max_n_goals = multi_goals_param['MAX_N_GOALS'],
                min_n_goals = multi_goals_param['MIN_N_GOALS'],
                max_goals_ratio = multi_goals_param['MAX_GOALS_RATIO'],
                max_value_tolerance_in = multi_goals_param['MAX_VALUE_TOL_IN'],
                max_value_tolerance_out = multi_goals_param['MAX_VALUE_TOL_OUT'],
                tolerance_out = multi_goals_param['INTEREST_THRES'],
                max_range = multi_goals_param['PLANNER_RANGE'],
                accessible = True, 
                min_range = multi_goals_param['MIN_RANGE'],
                banned_waypoints = banned_waypoints+penalty_points,
                max_distance = planning_horizon,
                max_distance_btw_targets = multi_goals_param['MAX_DIST_BTW_TARGETS'],
                discount_rate = multi_goals_param['DISCOUNT_RATE'],
                discount_ratio = multi_goals_param['DISCOUNT_RATIO'],
                cluster_method = multi_goals_param['CLUSTERING_METHOD'],
                per_step = multi_goals_param['PER_STEP'],
                disp = disp
            )
            self.has_set_goal = 'long term multi goals'
            self.new_multi_goal_set = multi_goals
            if multi_goals:
                return multi_goals

        if not rand > multi_goals_param['RANDOM_RATE'] or not multi_goals:
            new_goal_set = self.get_rd_pos(max_range = multi_goals_param['ST_PLANNING_RANGE'], accessible = True)
            self.has_set_goal = 'random'
            self.new_goal_set = new_goal_set
            return [new_goal_set]
    

    def get_frontier_goal(self, max_range, rd_f_rate = 0, banned_waypoints = [], penalty_points = [], disp = False):
        self.reset_vpp_var()

        #stop if no more frontier
        if not self.submaps.are_there_unknown():
            self.has_set_goal = 'frontier stopped'
            self.new_goal_set = True
            return True
        
        rand = rd.uniform(0,1)
        if rand > rd_f_rate:
            if disp and banned_waypoints != [] : print('some banned waypoints', banned_waypoints)
            new_goal = self.get_closest_frontier_goal(max_range = max_range, accessible = True, banned_waypoints = banned_waypoints+penalty_points)
            if new_goal is not None:
                self.has_set_goal = 'closest frontier'
        else:
            new_goal = self.get_rd_pos_in_frontier_space(max_range = max_range, accessible = True, banned_waypoints = banned_waypoints+penalty_points)
            if new_goal is not None:
                self.has_set_goal = 'rd frontier'
        
        if disp : print('new_goal from frontier :', new_goal)

        if new_goal is None:
            new_goal = self.get_rd_pos(max_range = max_range, accessible = True)
            self.has_set_goal = 'random'

        self.new_goal_set = new_goal
        return new_goal
    
    def update_vpp_var(self, submaps, curr_step):
        self.submaps = submaps
        self.reset_vpp_metrics(curr_step)

    def update_vpp_metrics(self):
        self.vpp_metrics['has_set_goal'] = self.has_set_goal
        self.vpp_metrics['critical_value'] = self.critical_value
        self.vpp_metrics['n_potential_targets'] = self.n_potential_targets
        self.vpp_metrics['potential_targets'] = self.potential_targets
        self.vpp_metrics['cluster_range'] = self.cluster_range
        self.vpp_metrics['n_targets'] = self.n_targets
        self.vpp_metrics['targets_cost_function'] = self.targets_cost_function
        self.vpp_metrics['path_dist'] = self.path_dist
        #self.vpp_metrics['targets'] = self.targets
        self.vpp_metrics['path_gains'] = self.path_gains
        self.vpp_metrics['path_steps'] = self.path_steps
        self.vpp_metrics['expected_gain'] = self.expected_gain
        self.vpp_metrics['n_dfs'] = self.n_dfs
        self.vpp_metrics['expected_dist'] = self.expected_dist
        self.vpp_metrics['gain_composition'] = self.gain_composition
        self.vpp_metrics['new_multi_goal_set'] = self.new_multi_goal_set
        self.vpp_metrics['new_goal_set'] = self.new_goal_set