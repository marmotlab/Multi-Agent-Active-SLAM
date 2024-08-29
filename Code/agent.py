import math
import copy
import time
import random as rd
import numpy as np
import statistics


from param import MetaParameters, AgentParameters, RenderingParameters, RewardParameters

from matplotlib.colors import hsv_to_rgb
if MetaParameters.RENDER:
    from gym.envs.classic_control import rendering #works with gym==0.21.0 installed
    from functions.render import *

from functions.union_intervals import *
from functions.entropy_functions import *

from objects import Observation, State, History, Memory, Meeting_Record
from components import Sensor, SubMaps, PathPlanner, MoveBase
from planner import CostMaps, ViewpointPlanner
from tree import RRT

from param import MetaParameters

class Agent:
    def __init__(self, id, color, init_pos, map):

        #id & parameters
        self.id = id
        self.color = color

        #entropy parameters
        self.phw = AgentParameters.PHW
        self.mhw = AgentParameters.MHW

        #metric parameters
        self.recently_seen_threshold = AgentParameters.RECENTLY_SEEN_THRESHOLD

        #init pose
        self.init_pos = self.set_init_pos(init_pos, map)

        '''
        if init_pos != None:
            if self.check_free_init_pos(init_pos):
                self.init_pos = init_pos
            else:
                self.init_pos = self.rd_init_pos()
        else:
            self.init_pos = self.rd_init_pos()
        '''

        #sharing
        self.sharing_data = AgentParameters.SHARING_DATA
        self.neighbours_period = AgentParameters.NEIGHBOURHOOD_PERIOD
        
        #corrector
        self.self_loop_cor = AgentParameters.SELF_LOOP_COR
        self.ma_loop_cor = AgentParameters.MA_LOOP_COR
        self.meeting_loop_cor = AgentParameters.MEETING_LOOP_COR

        self.reset_agent()

        #perf parameters
        self.completness_threshold = AgentParameters.COMPLETENESS_THRESHOLD
        self.correctness_threshold = AgentParameters.CORRECTNESS_THRESHOLD
        self.mean_error_threshold = AgentParameters.MEAN_ERROR_THRESHOLD

        self.completness_done = AgentParameters.COMPLETENESS_DONE
        self.correctness_done = AgentParameters.CORRECTNESS_DONE

        #render maps
        self.render_ag_map = MetaParameters.RENDER_AG_MAP
        self.render_planner = MetaParameters.RENDER_PL
        self.render_costmaps = MetaParameters.RENDER_COSTMAPS
        self.render_vpp = MetaParameters.RENDER_VPP
        self.render_tree = MetaParameters.RENDER_RRT
        self.render_blind = MetaParameters.RENDER_BLIND
        self.render_pog = MetaParameters.RENDER_POG
        self.render_distrib = MetaParameters.RENDER_PDP

        #rendering variables
        self.screen_dim = RenderingParameters.SCREEN_DIM
        self.agent_viewer = None
        self.planner_viewer = None
        self.blind_viewer = None
        self.pog_viewer = None
        self.distrib_viewer = None

        self.episode_agents_frames = []
        self.episode_planner_frames = []
        self.episode_blind_frames = []
        self.episode_pog_frames = []
        self.episode_distrib_frames = []

        #rendering parameters
        self.render_ground_truth = RenderingParameters.RENDER_GT
        self.render_trace = RenderingParameters.RENDER_TRACE
        self.render_loops = RenderingParameters.RENDER_LOOPS
        self.n_steps_back = RenderingParameters.RENDER_TRACE_LENGTH
        self.fully_scanned = RenderingParameters.FULLY_SCANNED
        self.render_visits = RenderingParameters.RENDER_VISITS
        self.max_blind = RenderingParameters.MAX_BLIND
        self.init_blind = RenderingParameters.INIT_BLIND

        #meta variables
        self.reset_step_exec_time_dic()


    def reset_state_variables(self):
        self.running = False
        self.time_step = None
        self.done = False

        self.has_planned = None
        self.action = None

        self.has_moved = None
        self.odom = None
        self.neighbours = []
        self.new_neighbours = []
        self.data = []
        self.new_data = []
        self.last_data = []
        self.blind_v = 0
        self.robot_entropy = None
        self.path_entropy = None
        self.global_entropy = None
        self.ag_loc_error = (0,0)
        self.ag_loc_err_dist = 0

        self.score = 0
        self.utility = 0

        self.gt_trace = []
        self.pos_trace = []

    def reset_loop_variables(self):
        #reset self loop variables
        self.self_loop_gt = None
        self.self_loop_ag = None
        self.self_loop_ref_observation_step = None
        self.self_loop_correction = None

        #reset ma variables
        self.ma_meta_loops = []
        self.ma_semi_loops = {}
        self.ma_corrections = {}
        self.ma_gt_loops_dic = {}
        self.ma_ag_loops_dic = {}

        #reset meeting variables
        self.meeting_rpr = {}
        self.meeting_batches = {}
        self.meeting_meta_loops = []
        self.meeting_semi_loops = {}
        self.meeting_corrections = {}
        self.meeting_gt_loops_dic = {}
        self.meeting_ag_loops_dic = {}

        self.has_corrected_map = None
        self.has_treated_holes = None
        self.n_corr_inst = None
    
    def reset_components(self):
        #composition of the agent
        self.move_base = MoveBase(self.init_pos)
        self.sensor = Sensor()
        self.submaps = SubMaps(self.init_pos)
        self.costmaps = CostMaps(self.id, self.submaps)
        self.viewpoint_planner = ViewpointPlanner(self.submaps)
        self.RRTtree = RRT(self.id, self.submaps)
        self.path_planner = PathPlanner()

        #memory and history
        self.memory = Memory(self.id)
        self.history = History(self.id)

        self.episode_cm_metrics = []
        self.episode_vpp_metrics = []
        self.episode_rrt_metrics = []
        self.episode_path_planner_history = []

        self.tracks_records = []

    def reset_global_variables(self):
        #observation and state variables
        self.observation = None
        self.state = None

        #others plans
        self.team_plans = {}
        
        #meeting and metrics variables
        self.meeting = None
        self.meetings_records = []

    def reset_metrics(self):
        self.metrics = None

        self.agent_metrics = {
            #moving
            'travelled_dist' : 0,
            'travelled_dist_per_step' : None,

            'n_staying_steps' : 0,

            'n_broken_odom' : 0,
            'n_broken_odom_per_step' : None,

            'n_collisions' : 0,
            'n_collisions_per_step' : None,

            #meeting
            'n_meetings' : 0,
            'n_meetings_per_step' : None,

            'n_new_neighbours' : 0,
            'n_new_neighbours_inst' : 0,
            'n_new_neighbours_per_step' : None,

            'n_new_data' : 0,
            'n_new_data_per_step' : None,

            #loops and corrections
            'n_self_loops' : 0,
            'n_self_corrections' : 0,
            'n_self_corrections_inst' : 0,
            'n_self_loops_correcting' : 0,

            'n_self_loops_per_step' : None,
            'n_self_corrections_per_step' : None,
            'n_self_corrections_per_self_loops' : None,
            'n_self_corrections_per_self_loops_correcting' : None,
            'ratio_self_loops_correcting' : None,


            'n_ma_meta_loops' : 0,
            'n_ma_semi_loops' : 0,
            'n_ma_corrections' : 0,
            'n_ma_corrections_inst' : 0,
            'n_ma_meta_loops_correcting' : 0,

            'n_ma_meta_loops_per_step' : None,
            'n_ma_corrections_per_step' : None,
            'n_ma_corrections_per_ma_loops' : None,
            'n_ma_corrections_per_ma_loops_correcting' : None,
            'ratio_ma_meta_loops_correcting' : None,


            'n_meeting_meta_loops' : 0,
            'n_meeting_semi_loops' : 0,
            'n_meeting_corrections' : 0,
            'n_meeting_corrections_inst' : 0,
            'n_meeting_meta_loops_correcting' : 0,

            'n_meeting_meta_loops_per_step' : None,
            'n_meeting_corrections_per_step' : None,
            'n_meeting_corrections_per_meeting_loop' : None,
            'n_meeting_corrections_per_meeting_loop_correcting' : None,
            'ratio_meeting_meta_loops_correcting' : None,

            'n_meeting_corrections_per_meeting' : None,
            'n_meeting_corrections_per_new_meeting' : None,


            'n_corrections' : 0,
            'n_corrections_inst' : 0,
            'n_corrections_per_step' : None,

            'n_corrections_per_loop' : None,
            'n_corrections_per_loop_correcting' : None,

            'n_corrections_per_meeting' : None,
            'n_corrections_per_new_meeting' : None,

            #entropy
            'entropy_loss_robot_inst' : None,
            'entropy_loss_mapping_uc_inst' : None,
            'entropy_loss_mapping_uk_inst' : None,
            'entropy_loss_path_inst' : None,
            'entropy_loss_global_inst' : None,
        }

    def reset_evaluations(self):
        #evaluating metrics
        self.submaps_eval = {
            #map
            'squares_known_perc' : None,

            #obstacles
            'n_obstacles_corr' : None,
            'n_obstacles_wrong' : None,
            'n_obstacles_missed' : None,
            'obstacles_corr_perc' : None,
            'obstacles_missed_perc' : None,

            #team perf
            'team_agents_known_perc' : None,
            'team_lastly_seen_mean_ts_corr' : None,
            'team_known_perc' : None,
        }

        self.obs_eval = {
            'obs_mean_err' : None,
            #'obs_self_err_list' : None,
            'obs_self_mean_err' : None,
        }

        self.eval_records = []
        
    def reset_performance(self):
        #measure performance of an agent
        self.perf = {
            'success' : None, #"good map" (above or below a threshold)
            'done' : None, #perfect map
            'success_step' : False, #step when the agent has a "good map", False if success is not
            'done_step' : False, #step when the agent has a "good map", False if success is not
        }
    
    def reset_sequences(self):
        self.sequences = {
            'state' : {
                'agent' : {
                    'robot_localization_error' : [],
                    'robot_blind_v' : [],
                    'robot_entropy' : [],
                    'path_entropy' : [],
                    'mapping_entropy_uncertain' : [],
                    'mapping_entropy_unknown' : [],
                    'global_entropy' : [],
                    'score' : [],
                },

            },
            'metrics' : {
                'agent' : {},
                'submaps' : {},
                'tracks' : {},
            },
            'eval' : {
                'submaps' : {},
                'obs' : {},
            },
        }

        for metric in self.agent_metrics:
            self.sequences['metrics']['agent'][metric] = []
        for metric in self.submaps.metrics:
            self.sequences['metrics']['submaps'][metric] = []
        for metric in self.memory.tracks_metrics:
            self.sequences['metrics']['tracks'][metric] = []
        for m_eval in self.submaps_eval:
            self.sequences['eval']['submaps'][m_eval] = []
        for m_eval in self.obs_eval:
            if type(self.obs_eval[m_eval]) not in [list, dict]:
                self.sequences['eval']['obs'][m_eval] = []

    def reset_step_exec_time_dic(self):
        self.step_exec_time = {
            'policy' : [],
            'step' : [],
            'eval' : [],
            'save' : [],
            'render' : [],

            #step
            'do action' : [],
            'update odom' : [],
            'localise' : [],
            'update sensor - global' : [],
            #'update sensor - sensor' : [],
            #'update sensor - observation' : [],
            #'update sensor - neighbours' : [],
            'share data' : [], 
            'build map from tracks - 1 step' : [],
            'build map from tracks - from start' : [],
            'build map from tracks - from start_inf' : [],
            'build map from tracks - others' : [],
            'build map from tracks - others_inf' : [],
            'correct w self - global' : [],
            #'correct w self - get loop' : [],
            #'correct w self - correct ag pos' : [],
            #'correct w self - correct obs' : [],
            'correct w self - correct trace' : [],
            'correct w self - build map' : [],
            #'correct w self - get ag loop' : [],
            'correct w ma - global' : [],
            #'correct w ma - find loops' : [],
            #'correct w ma - get semi loops dic' : [],
            'correct w ma - correct obs' : [],
            #'correct w ma - correct ag pos' : [],
            'correct w ma - correct ag traces' : [],
            'correct w ma - build map' : [],
            #'correct w ma - get loops list' : [],             
            'correct w meeting - global' : [],
            #'correct w meeting - get rpr' : [],
            #'correct w meeting - get batches' : [],
            'correct w meeting - find loops' : [],
            #'correct w meeting - get semi loops dic' : [],
            'correct w meeting - correct obs' : [],
            #'correct w meeting - correct ag pos' : [],
            #'correct w meeting - correct ag traces' : [],
            'correct w meeting - build map' : [],
            #'correct w meeting - get loops list' : [],
            'treat holes - all' : [],
            'treat holes - suspected' : [],
            'treat holes - save progression' : [],
            'update submaps ext' : [],
            'update traces' : [],
            'update blind maps' : [],
            'update presence distrib' : [],
            'update entropy': [],
            'update metrics' : [],
            'get utility' : [],

            #'save - state' : [],
            'save - planner' : [],
            'save - path planner' : [],
            'save - history' : [],
            'save - meeting' : [],
            'save - tracks' : [],
            'save - metrics' : [],

            'render - agent map' : [],
            'render - ag_ext_map' : [],
            'render - planner' : [],
            'render - blind' : [],
            'render - pog' : [],
            'render - pdp' : [],
        }

        self.step_exec_stats = {}

    def reset_agent(self):
        self.reset_state_variables()
        self.reset_loop_variables()
        self.reset_components()
        self.reset_global_variables()
        self.reset_metrics()
        self.reset_evaluations()
        self.reset_performance()
        self.reset_sequences()


    #init functions
    def rd_init_pos(self, map):
        return map.set_rd_free_point()

    def check_free_init_pos(self, pos, map):
        return map.get_square(pos) == 0
    
    def set_init_pos(self, init_pos, map):
        if init_pos == None:
            init_pos = self.rd_init_pos(map)
        k = 0
        while k < 10:
            if self.check_free_init_pos(init_pos, map):
                return init_pos
            else:
                init_pos = self.rd_init_pos(map)
        return None
    
    #get trace functions
    def get_gt_trace(self, n_steps_back):
        start = max(1, self.time_step - n_steps_back)
        return self.history.pos_history[start-1:]+[self.move_base.pos]
    
    def get_pos_trace(self, n_steps_back):
        pos_trace = []
        start = max(1, self.time_step - n_steps_back)
        for i in range(start, self.time_step+1):
            observation = self.memory.tracks[self.id].obs_list[i-1]
            ag_pos = observation.pos_belief[-1]['pos_belief']
            pos_trace.append(ag_pos)
        return pos_trace
    




    #-------------------------------------------
    #------------run loop functions-------------
    #-------------------------------------------


    #---move base functions---
    def do_action(self, env, disp = False, measure = MetaParameters.MEASURE_TIME):
        if measure : tic = time.time()
        if disp : print("Doing action : ", self.action)

        if self.action == None:
            self.has_moved = None

        else:
            #try to move
            self.has_moved = self.move_base.move(self.action, env.map)
            if self.has_moved:
                if disp : print(" Agent has moved successfully!\n New Position :", self.move_base.pos)
            else:
                if disp : print(" Error - agent could not move")

        if disp : print("Action done")
        if measure : self.step_exec_time['do action'].append(round(time.time()-tic,4))
    
    def update_blind_value(self):
        if self.has_moved:
            self.blind_v += 1

    def update_odometry(self, disp = False, measure = MetaParameters.MEASURE_TIME):
        if measure : tic = time.time()
        if disp : print("Using Odometry ...")

        if self.has_moved:
            self.odom = self.move_base.get_odom(self.action)
        else:
            self.odom = None
    
        if measure : self.step_exec_time['update odom'].append(round(time.time()-tic,4))

    #---localise functions---
    def get_loc_error(self, est_pos, true_pos, off_set):
        return(est_pos[0] - true_pos[0] - off_set[0], est_pos[1] - true_pos[1] - off_set[1])

    def localise(self, disp = False, measure = MetaParameters.MEASURE_TIME):
        if measure : tic = time.time()
        if disp : print("Localising ...")

        self.submaps.update_ag_pos(self.odom)
        
        if disp : print(" Agent's Position Belief :", self.submaps.ag_pos)
        
        self.ag_loc_error = self.get_loc_error(self.submaps.ag_pos, self.move_base.pos, self.submaps.off_set)
        self.ag_loc_err_dist = abs(self.ag_loc_error[0]) + abs(self.ag_loc_error[1])

        if disp : print(" Agent's Localisation Error :", self.ag_loc_error)
        if measure : self.step_exec_time['localise'].append(round(time.time()-tic,4))


    #---update sensor--- cf functions above

    def update_neighbours(self, disp = False, measure = MetaParameters.MEASURE_TIME):
        #set last neighbours since last period
        last_neighbours = []
        for i_last in range(1, self.neighbours_period+1):
            if len(self.history.neighbours_history) >= i_last:
                for n_id in self.history.neighbours_history[-i_last]:
                    last_neighbours.append(n_id)
        last_neighbours = list(set(last_neighbours))

        #set current neighbourhood
        self.neighbours = []
        self.new_neighbours = []
        for ag_id in self.sensor.team_in_range:
            if ag_id != self.id:
                self.neighbours.append(ag_id)
                if ag_id not in last_neighbours:
                    self.new_neighbours.append(ag_id)
        if disp : print("Neighbours updated")

    def update_last_data(self, disp = False):
        #last data
        last_data = []
        for i_last in range(1, self.neighbours_period+1):
            if len(self.history.data_history) >= i_last:
                for d_id in self.history.data_history[-i_last]:
                    last_data.append(d_id)
        self.last_data = list(set(last_data))



    #-------while meeting agents-------
    #----------------------------------

    #---share and update tracks sub-functions---
    def update_meeting_dic(self, disp):
        if disp : print("Updating meeting dic ...")
        for oth_id in self.neighbours:
            #add neigbours to meeting dic in self track
            if oth_id not in self.memory.tracks[self.id].meeting_dic:
                self.memory.tracks[self.id].meeting_dic[oth_id] = [self.time_step]
            else:
                self.memory.tracks[self.id].meeting_dic[oth_id].append(self.time_step)
            
            #add neigbours to meeting dic
            if oth_id not in self.memory.meeting_dict:
                self.memory.meeting_dict[oth_id] = [self.time_step]
            else:
                self.memory.meeting_dict[oth_id].append(self.time_step)
        
        if disp : print("Self meeting dic :", self.memory.meeting_dict)

    def share_data(self, team, disp):
        if disp : print("Sharing data ...")

        #init
        self.data, self.new_data = [], []

        #share data
        for m_id in self.neighbours:
            #copy the observations list and the meeting dic of the agent
            self.memory.copy_track(m_id, team[m_id].memory.tracks[m_id])
            self.memory.tracks[m_id].owners.append({'owner' : self.id, 'from step' : self.time_step})
            self.data.append(m_id)
            if m_id not in self.last_data:
                self.new_data.append(m_id)

            if m_id not in self.memory.data_dict:
                self.memory.data_dict[m_id] = [self.time_step]
            else:
                self.memory.data_dict[m_id].append(self.time_step)
            if disp : print("Neighbour agent track (Agent "+str(m_id)+") copied!")
            
            #copy the observations list and the meeting dic of thirds agents from the agent's memory
            for third_id in team[m_id].memory.tracks:
                if third_id != self.id and third_id != m_id:
                    
                    #if self has never met this agent or if self has met the agent but before the other agent
                    if third_id not in self.memory.tracks or len(self.memory.tracks[third_id].obs_list) < len(team[m_id].memory.tracks[third_id].obs_list) or (len(self.memory.tracks[third_id].obs_list) == len(team[m_id].memory.tracks[third_id].obs_list) and self.memory.tracks[third_id].last_update < team[m_id].memory.tracks[third_id].last_update and team[m_id].memory.tracks[third_id].owners[-2]['owner'] != self.id):
                        self.memory.copy_track(third_id, team[m_id].memory.tracks[third_id])
                        self.memory.tracks[third_id].owners.append({'owner' : self.id, 'from step' : self.time_step})
                        self.data.append(third_id)
                        if third_id not in self.last_data:
                            self.new_data.append(third_id)
                        if third_id not in self.memory.data_dict:
                            self.memory.data_dict[third_id] = [self.time_step]
                        else:
                            self.memory.data_dict[third_id].append(self.time_step)
                        if disp : print("Third agent track (Agent "+str(third_id)+") copied!")

        if disp : print("Tracks updated :", self.memory.tracks)
    

    def share_last_plans(self, team, neigbours_only = False, disp = False):
        #share plans
        for m_id in self.neighbours:
            self.team_plans[m_id] = {
                'goal' : team[m_id].path_planner.goal,
                'path' : team[m_id].path_planner.path,
                'multi_goals' : team[m_id].path_planner.multi_goals,
                'time_step' : self.time_step + (-1 * (self.id < m_id)),
                }
            
            if not neigbours_only:
                for third_id in team[m_id].team_plans:
                    if third_id != self.id and third_id != m_id:

                        #if self has never met this agent or if self has met the agent but before the other agent
                        if third_id not in self.team_plans or self.team_plans[third_id]['time_step'] < team[m_id].team_plans[third_id]['time_step']:
                            self.team_plans[third_id] = copy.deepcopy(team[m_id].team_plans[third_id])
        
        if disp : print("Plans shared")


    #---update agent's submaps---
    #---rebuild the map from tracks---
    def build_map_from_tracks(self, start, disp = False, measure = MetaParameters.MEASURE_TIME): #this function aims to build a map with different lines when meeting of finding MA loops
        if measure : tac = time.time()
        if measure : counter = 0
        if disp : print(" Rebuilding the map from step :", start, " (last obs or start)")
        
        if start == 1:
            self.submaps.init_submaps()

        else: #reset from stored progression
            self.submaps.ag_map = self.memory.ag_map_progression[start-2]
            self.submaps.n_scans_map = self.memory.n_scans_map_progression[start-2]
            self.submaps.blind_table = self.memory.blind_table_progression[start-2]
            self.submaps.ag_team_pos = self.memory.ag_team_pos_progression[start-2]

        for step in range(start, self.time_step+1):

            #rebuild
            for ag_id in list(reversed(self.memory.tracks.keys())):
                if ag_id > self.id:
                    obs_step = step -1
                else:
                    obs_step = step

                if obs_step > 0 and len(self.memory.tracks[ag_id].obs_list) > obs_step -1:
                    #if disp : print("  Track", ag_id, "; Observation step", obs_step)
                    if measure : counter += 1

                    observation = self.memory.tracks[ag_id].obs_list[obs_step-1]
                    self.submaps.update_ag_map(observation)
                    self.submaps.treat_borders(observation)
                    self.submaps.update_ag_team_pos(observation)

            #save progression
            self.memory.save_ag_map_progression(step, self.submaps.ag_map)
            self.memory.save_n_scans_map_progression(step, self.submaps.n_scans_map)
            self.memory.save_blind_table_progression(step, self.submaps.blind_table)
            self.memory.save_ag_team_pos_progression(step, self.submaps.ag_team_pos)
            
            if disp : print(" Map and other agents' pos rebuilt from step", step)
        
        if disp : print(" Agent's map has been updated")        
        if measure : 
            if start == self.time_step:
                self.step_exec_time['build map from tracks - 1 step'].append(round(time.time()-tac,4))
            elif start == 1:
                self.step_exec_time['build map from tracks - from start'].append(round(time.time()-tac,4))
                self.step_exec_time['build map from tracks - from start_inf'].append((round(time.time()-tac,4), 'n_op :', counter, 'step :', self.time_step))
            else:
                self.step_exec_time['build map from tracks - others'].append(round(time.time()-tac,4))
                self.step_exec_time['build map from tracks - others_inf'].append((round(time.time()-tac,4), 'n_op :', counter, 'step :', self.time_step, 'start :', start))






    #-------correcting functions - big part-------
    #---------------------------------------------

    #---self loop closure correction : 1. get self loop closure 2. update ag pos, observations and map 3. get the loop in the agent submap
    #correct w self sub-functions
    def get_gt_self_closed_loop(self, disp = False):
        if disp : print(" Looking for a self loop ...")

        for i in range(1,len(self.history.pos_history)+1):
            if self.history.pos_history[-i] == self.move_base.pos:
                loop = self.history.pos_history[-i:]+[self.move_base.pos]
                if len(loop)>2:
                    if disp : print(" A self loop has been closed! Here is the loop :\n  Real Loop :", loop)
                    return loop
        if disp : print(" No self loop found")
        return None
    
    def get_self_ag_closed_loop(self, disp = False):
        if self.self_loop_gt:
            if disp : print(" Updating agent closed loop ...")

            ref_observation = self.memory.tracks[self.id].obs_list[-len(self.self_loop_gt)]
            if ref_observation.real_pos == self.move_base.pos: #should be always True
                ag_loop = []
                for i in range(ref_observation.time_step, self.time_step+1):
                    observation = self.memory.tracks[self.id].obs_list[i-1]
                    ag_pos = observation.pos_belief[-1]['pos_belief']
                    ag_loop.append(ag_pos)
                return ag_loop
        else:
            return None

    def correct_loop_obs(self, ref_observation, disp = False):
        #correct the past observations pos belief and blind values since the last observation (last observation exluded, current observation included)
        ref_blind_v = ref_observation.pos_belief[-1]['blind_v']
        ref_pos_err = ref_observation.pos_belief[-1]['err']
        
        first_correcting_step = False #init first correcting step

        for i_step in range(ref_observation.time_step+1, self.time_step+1):
            observation = self.memory.tracks[self.id].obs_list[i_step-1]

            #set first correcting step the first time the blind_v differs
            if first_correcting_step == False and observation.pos_belief[-1]['blind_v'] != ref_blind_v:
                first_correcting_step = i_step

            #correct if first correcting step is set
            if first_correcting_step != False:
                if observation.pos_belief[-1]['err'] != ref_pos_err:
                    observation.add_new_error(self.time_step, ref_pos_err, ref_blind_v, self.submaps.off_set)
                elif observation.pos_belief[-1]['blind_v'] != ref_blind_v:
                    observation.update_last_blind_v(ref_blind_v)

        if disp : print(" Loop observations have been corrected")
        return first_correcting_step

    def correct_ag_var(self, ref_observation, disp = False):
        if disp : print(" Correcting agent pos after ag loop ...")
        #correct the agent position and blind value
        self.submaps.ag_pos = ref_observation.pos_belief[-1]['pos_belief']
        self.ag_loc_error = ref_observation.pos_belief[-1]['err']
        self.blind_v = ref_observation.pos_belief[-1]['blind_v']

        if disp : print(
                " Localisation has been corrected. ",
                "\n  New Position Belief :", self.submaps.ag_pos,
                "\n  New Localisation Error :", self.ag_loc_error,
                "\n  New Blind Value :", self.blind_v,
            )

    #function
    def correct_w_self(self, disp = False, measure = MetaParameters.MEASURE_TIME):
        if measure : tic = time.time()
        if disp : print("Correction alone ...")

        #reset variables
        self.self_loop_correction = None
        self.self_loop_ref_observation_step = None
        
        #find a self loop
        #if measure : toc = time.time()
        self.self_loop_gt = self.get_gt_self_closed_loop(disp) #the function returns None if no loop is found
        #if measure : self.step_exec_time['correct w self - get loop'].append(round(time.time()-toc,4))

        if self.self_loop_gt != None:
            if disp : print(" Correcting Localisation and Mapping ...")            

            #set reference
            ref_observation = self.memory.tracks[self.id].obs_list[-len(self.self_loop_gt)]
            
            #check
            if ref_observation.real_pos != self.move_base.pos: #should be always False
                print('ERROR - correct')
                return False
            
            #set last observation step
            self.self_loop_ref_observation_step = ref_observation.time_step

            #check if need to correct observations and map
            
            #correct agent's observations
            #if measure : toc = time.time()
            self.self_loop_correction = self.correct_loop_obs(ref_observation, disp)
            #if measure : self.step_exec_time['correct w self - correct obs'].append(round(time.time()-toc,4))

            #correct agent's map
            if measure : toc = time.time()
            self.build_map_from_tracks(self.self_loop_correction, disp)

            self.has_corrected_map = True
            if not self.n_corr_inst : self.n_corr_inst = 0
            self.n_corr_inst += self.time_step - self.self_loop_correction +1
            if measure : self.step_exec_time['correct w self - build map'].append(round(time.time()-toc,4))
            
            #update track's last_update
            self.memory.tracks[self.id].update_last_update(self.time_step)
            
            #rebuild agent trace
            if measure : toc = time.time()
            self.memory.tracks[self.id].rebuild_trace()
            if measure : self.step_exec_time['correct w self - correct trace'].append(round(time.time()-toc,4))

            #correct agent's pos and blind value (after observation corrections)
            #if measure : toc = time.time()
            self.correct_ag_var(ref_observation, disp)
            #if measure : self.step_exec_time['correct w self - correct ag pos'].append(round(time.time()-toc,4))

        #build the loop in the agent's representation
        #if measure : toc = time.time()
        self.self_loop_ag = self.get_self_ag_closed_loop(disp)
        #if measure : self.step_exec_time['correct w self - get ag loop'].append(round(time.time()-toc,4))

        if measure : self.step_exec_time['correct w self - global'].append(round(time.time()-tic,4))



    #---MAG loop closure---

    #---finding part---
    #create the meta loop that compiles all the portions that lead from the last observation to the current one
    #ex : meta loop = [(1st ag, [obs, meeting]), (2nd ag, [meeting, meeting]), ..., (self, [meeting, obs])]
    def get_meta_loop(self, oth_id, last_oth_ag_observation, disp):
        
        #get meta_ext
        meta_ext = self.memory.tracks[oth_id].meta_ext
        if disp : print(" meta ext :", meta_ext)
        
        #init meta_loop
        meta_loop = []

        #function
        for i_portion in range(len(meta_ext)):
            if meta_ext[i_portion]['intv'][1] >= last_oth_ag_observation.time_step:
                meta_loop.append({'ag' : meta_ext[i_portion]['ag'], 'intv' : [last_oth_ag_observation.time_step, meta_ext[i_portion]['intv'][1]]})
                break

        for j_portion in range(i_portion +1, len(meta_ext)):
            meta_loop.append((meta_ext[j_portion]))
        
        return meta_loop

    #---multi agent loop closure correction when agent is alone ; correction based on other agents observations---
    #correct with ma sub-functions          
    def find_and_get_ma_loops(self, disp = False):
        if disp : print(" Finding ma loops ...")

        #init
        meta_loops_list = []

        #function
        for oth_id in self.memory.tracks:
            if oth_id != self.id and self.memory.tracks[oth_id].obs_list != []:

                #look forward the observations of the other agent
                for i in range(len(self.memory.tracks[oth_id].obs_list)):
                    
                    #stop if the other agent has already met the current position >> a loop has been found
                    if self.memory.tracks[oth_id].obs_list[i].real_pos == self.move_base.pos:

                        #the loop starts from the last observation of the other agent and ends at the current observation
                        last_oth_ag_observation = self.memory.tracks[oth_id].obs_list[i]

                        if disp : print(" Agent", oth_id, "has already met this position :", last_oth_ag_observation.real_pos, "at step :", last_oth_ag_observation.time_step)

                        meta_loop = self.get_meta_loop(oth_id, last_oth_ag_observation, disp)
                        meta_loops_list.append(meta_loop)

                        if disp : print(" meta loop :", meta_loop)

                        #to increase the number of meta loops, we consider other path by which the the data could have been carried, with the same number of carrier
                        #to solve this, we look for all the agents that have met the "other agent" after is has observed the current pose the last time
                        if disp : print( " Lets find other loops from this seen position carried by other agents ...")
                        for all_id in self.memory.tracks:
                            if all_id != oth_id:
                                
                                if oth_id in self.memory.tracks[all_id].meeting_dic:
                                    meeting_step_after_obs = [step for step in self.memory.tracks[all_id].meeting_dic[oth_id] if step >= (last_oth_ag_observation.time_step + (1*(all_id < oth_id)))] #we need to add 1 if the all_id has 1 step in advance compared to oth_id
                                    
                                    if meeting_step_after_obs != []:
                                        first_meeting_step_after_obs = min(meeting_step_after_obs)
                                        
                                        #the loop starts from the first time the "all agent" met the other agent after the first observation and ends at the current observation
                                        last_meeting_observation = self.memory.tracks[all_id].obs_list[first_meeting_step_after_obs-1]
                                        
                                        if disp : print(" Agent", all_id, "has met Agent", oth_id,"in position :", last_meeting_observation.real_pos, "at step :", last_meeting_observation.time_step, "check step :", first_meeting_step_after_obs)
                                        
                                        meta_loop = self.get_meta_loop(all_id, last_meeting_observation, disp)
                                        meta_loops_list.append(meta_loop)

                                        if disp : print(" meta loop :", meta_loop)
                        
                        #break once the other agent have seen the position
                        break
                    
        if disp : print(" meta loops :", meta_loops_list)
        
        return meta_loops_list



    #---multi agent retro loop closure correction when meeting other agents, after having share the data---
    #correct w meeting sub-functions
    def get_rpr(self, disp = False):
        if disp : print(" Getting rpr for each agent ...")

        #init
        meeting_rpr = {}

        #function
        for ag_id in self.memory.tracks:

            if disp : print("  Getting rpr for Agent", ag_id, "...")

            #init
            meeting_rpr[ag_id] = {}

            #each track is represented by an agent that takes part of the meeting (the first owner of the track in [self and neighbours])
            meta_ext = self.memory.tracks[ag_id].meta_ext
            for portion in range(len(meta_ext)):
                current_ag = meta_ext[portion]['ag']
                if current_ag in [self.id] + self.neighbours:
                    rpr = current_ag
                    break
            
            #for each track, we look for a loop closure on a branch that contains another representant at the meeting
            other_rpr = copy.copy(self.neighbours + [self.id])
            other_rpr.remove(rpr)

            meeting_rpr[ag_id]["rpr"] = rpr
            meeting_rpr[ag_id]["oth_rpr"] = other_rpr


        if disp : print("  meeting rpr :", meeting_rpr)

        return meeting_rpr


    def get_batches(self, disp = False):
        if disp : print(" Getting batches ...")

        #init
        batches_dic = {}

        #function
        for ag_id in self.memory.tracks:
            if disp : print(" Define a batch for ag id ", ag_id)
            #create batch
            #we define the batch of observations that we are going to look at before starting the loops
            #finally we will compare the ag_id portion with the right oth_id portions
            
            if self.memory.tracks[ag_id].obs_list == []: #skip if ag_id has not made any observation
                continue

            #init
            batch = {} #batch = {oth_id : [(id_1, [step, step]), (next_id, [step, step]), ...] which looks like the meta_ext, another_id : [(), (), ...]}
            
            #function
            for oth_id in self.memory.tracks:
                if disp : print("  oth id :", oth_id)

                if self.memory.tracks[oth_id].obs_list == []:
                    continue

                #special cases
                if oth_id == ag_id:
                    continue
                elif oth_id == self.meeting_rpr[ag_id]['rpr']:
                    continue
                elif self.meeting_rpr[oth_id]['rpr'] in self.meeting_rpr[ag_id]['oth_rpr']:
                    batch[oth_id] = self.memory.tracks[oth_id].meta_ext[0]['intv'] #a list with 1 element
                
                #main case
                else: # --> oth_id is neither ag_id nor his rpr, and is not represented by one of the other_rpr list --> oth_id is represented by ag_id's rpr    
                    #we use the meta extended obs list of the oth_id
                    meta_ext = self.memory.tracks[oth_id].meta_ext
                    
                    for k in range(len(meta_ext)):
                        current_ag = meta_ext[k]['ag']
                        start = meta_ext[k]['intv'][0]

                        if current_ag == self.meeting_rpr[ag_id]['rpr']: #break if the current id is ag_id's rpr
                            break

                        #we look at the meeting dic of the current ag
                        meeting_dic = self.memory.tracks[current_ag].meeting_dic
                    
                        #check if a rpr has been met in the segment
                        step_candidates = {}
                        for met_id in meeting_dic:
                            if met_id in self.meeting_rpr[ag_id]['oth_rpr']:
                                if meeting_dic[met_id][-1] > start:
                                    step_candidates[met_id] = meeting_dic[met_id][-1]
                        
                        if step_candidates != {}:
                            if disp : print("  step candidates :", step_candidates)
                            if k == 0:
                                best_candidate = max([v for k, v in step_candidates.items()])
                                if disp : print("  best candidate :", best_candidate)
                                batch[oth_id] = [start, best_candidate]
                            else:
                                batch[oth_id] = self.memory.tracks[oth_id].meta_ext[0]['intv']
                            break
                        else:
                            if disp : print("  no candidates")
            
            #store the batch inside the dic
            batches_dic[ag_id] = batch  
            if disp : print(" Batch for Agent", ag_id,":",batch) #batch = {first_id : [(oth_id, [step, step]), (next_id, [step, step]), ...] which looks like the meta_ext, ... }
            
        return batches_dic
    
    def find_and_get_retro_loops(self, disp = False):
        if disp : print(" Finding retro loops ...")

        #init
        meta_loops_list = []

        #function
        for ag_id in self.meeting_batches:
            batch = self.meeting_batches[ag_id]

            if batch != {}:

                #forward loop : scan in chronological order the observations of the agent
                if disp : print(" Forward loop ...")
                for i in range(len(self.memory.tracks[ag_id].obs_list)):
                    #if disp : print(" i =", i)
                    
                    #compare these observations with the other agent's ones in the batch
                    for oth_id in batch:
                        #if disp : print(" oth id :", oth_id)
                    
                        #look backward the observations of the other agent in the database
                        #if disp : print(" Backward loop ...")
                        for j in range(batch[oth_id][1] - batch[oth_id][0] +1):

                            #stop when this observation has already been observed by an other agent >> a loop has been found
                            if self.memory.tracks[ag_id].obs_list[i].real_pos == self.memory.tracks[oth_id].obs_list[batch[oth_id][1]-j-1].real_pos:
                                
                                #the portion starts from the last observation of the agent and ends at the current observation of the rpr
                                last_ag_observation = self.memory.tracks[ag_id].obs_list[i]
                                last_oth_ag_observation = self.memory.tracks[oth_id].obs_list[batch[oth_id][1]-j-1]
                                
                                if disp : print("  While meeting, a loop in the past has been found :",
                                                "\n   Agent", ag_id, "has observed position", last_ag_observation.real_pos, "at step :", last_ag_observation.time_step,
                                                "\n   ... while Agent", oth_id, "in batch has observed position", last_oth_ag_observation.real_pos, "at step :", last_oth_ag_observation.time_step)
                                
                                meta_loop = self.get_meta_loop(ag_id, last_ag_observation, disp)
                                meta_loops_list.append(meta_loop)

                                if disp : print(" meta loop :", meta_loop)                                
                                break

                        else:
                            #if disp : print(" no loop found at step", i,"with oth_id :", oth_id)
                            continue
                        if disp : print(" Should break")
                        break

                    else:
                        #if disp : print(" no loop found at step", i)
                        continue
                    if disp : print(" Should break again")
                    break
            
                if disp : print(" We are out!")

        return meta_loops_list
    

    def get_semi_loops_dic(self, meta_loops_list, disp = False):
        if disp : print(" Getting semi loop ...")

        #init
        semi_loops_dic = {}

        #function
        for meta_loop in meta_loops_list:
            for portion in meta_loop:
                cur_id = portion['ag']
                window = portion['intv']
                if cur_id not in semi_loops_dic:
                    semi_loops_dic[cur_id] = [window]
                else:
                    if window not in semi_loops_dic[cur_id]:
                        semi_loops_dic[cur_id].append(window)
        
        if disp : print(" semi loops dic before union :", semi_loops_dic)

        #merge semi_loops dic intervals list
        for ag_id in semi_loops_dic:
            intervals_list = semi_loops_dic[ag_id]
            union_list = union_intervals(intervals_list)
            semi_loops_dic[ag_id] = union_list
            
        if disp : print(" semi loops dic after union :", semi_loops_dic)

        return semi_loops_dic


    #---correcting part---
    def correct_mag_obs(self, semi_loops_dic, disp = False):
        if disp : print(" Correcting observations if needed ...")

        #init
        corrections = {}

        #function
        for ag_id in semi_loops_dic:
            if disp : print("  ag_id :", ag_id)

            #set correction
            corrections[ag_id] = {}

            for window in semi_loops_dic[ag_id]:
                if disp : print("  window :", window)

                #check if need to correct
                if disp : print(" Checking if need to correct ...")
                first_obs = self.memory.tracks[ag_id].obs_list[window[0]-1]
                last_obs = self.memory.tracks[ag_id].obs_list[window[1]-1]
                ref_blind_v = first_obs.pos_belief[-1]['blind_v']
                ref_pos_err = first_obs.pos_belief[-1]['err']

                #correct observations' position belief and blind value inside the portion                
                first_correcting_step = False #init first correcting_step
                
                for k_step in range(first_obs.time_step, last_obs.time_step +1):
                    observation = self.memory.tracks[ag_id].obs_list[k_step-1]
                    
                    #set first correcting step the first time the blind_v differs
                    if first_correcting_step == False and observation.pos_belief[-1]['blind_v'] != ref_blind_v:
                        first_correcting_step = k_step
                    
                    #correct if first correcting step is set
                    if first_correcting_step != False:
                        if observation.pos_belief[-1]['err'] != ref_pos_err:
                            observation.add_new_error(self.time_step, ref_pos_err, ref_blind_v, self.submaps.off_set)
                        elif observation.pos_belief[-1]['blind_v'] != ref_blind_v:
                            observation.update_last_blind_v(ref_blind_v)

                #set correction first step
                corrections[ag_id][tuple(window)] = first_correcting_step

                #update track's last_update
                if first_correcting_step : self.memory.tracks[ag_id].update_last_update(self.time_step)

        return corrections

    def correct_mag_var(self, disp = False):
        if disp : print(" Correcting agent pos after mag loop ...")
        #correct the agent position and blind value
        self.submaps.ag_pos = self.memory.tracks[self.id].obs_list[-1].pos_belief[-1]['pos_belief']
        self.blind_v = self.memory.tracks[self.id].obs_list[-1].pos_belief[-1]['blind_v']
        self.ag_loc_error = self.memory.tracks[self.id].obs_list[-1].pos_belief[-1]['err']
        
        if disp : print(
                " Localisation has been corrected. ",
                "\n  New Position Belief :", self.submaps.ag_pos,
                "\n  New Blind Value :", self.blind_v,
                "\n  New Localisation Error :", self.ag_loc_error,
            )

    def get_loop_list(self, ag_id, window):
        semi_loop = []
        for o in range(window[0], window[1] +1):
            semi_loop.append(self.memory.tracks[ag_id].obs_list[o-1].real_pos)
        return semi_loop
    
    def get_ag_loop_list(self, ag_id, window):
        semi_ag_loop = []
        for o in range(window[0], window[1] +1):
            semi_ag_loop.append(self.memory.tracks[ag_id].obs_list[o-1].pos_belief[-1]['pos_belief'])
        return semi_ag_loop
    
    def get_mag_loops_list(self, semi_loops_dic, disp = False):
        if disp : print(" Getting semi loop (gt and ag) ...")

        #init
        loops_dic = {}
        ag_loops_dic = {}

        #function
        for ag_id in semi_loops_dic:
            if disp : print("  ag_id :", ag_id)

            for window in semi_loops_dic[ag_id]:
                if disp : print("  window :", window)

                semi_loop = self.get_loop_list(ag_id, window)
                semi_ag_loop = self.get_ag_loop_list(ag_id, window)

                if ag_id not in loops_dic:
                    loops_dic[ag_id] = [semi_loop]
                    ag_loops_dic[ag_id] = [semi_ag_loop]
                else:
                    loops_dic[ag_id].append(semi_loop)
                    ag_loops_dic[ag_id].append(semi_ag_loop)
        
        return loops_dic, ag_loops_dic
    


    #---global functions---
    def correct_w_ma(self, disp = False, measure = MetaParameters.MEASURE_TIME):
        if measure : tic = time.time()
        if disp : print("Looking for Multi-Agent loops ...")
        
        #step 1 : looking for loops
        #if measure : toc = time.time()     
        self.ma_meta_loops = self.find_and_get_ma_loops(disp) #this function build the ma_meta_loops
        #if measure : self.step_exec_time['correct w ma - find loops'].append(round(time.time()-toc,4))

        #complile the meta loops into semi loops
        if self.ma_meta_loops != []:
            #if measure : toc = time.time()
            self.ma_semi_loops = self.get_semi_loops_dic(self.ma_meta_loops, disp)
            #if measure : self.step_exec_time['correct w ma - get semi loops dic'].append(round(time.time()-toc,4))

        #step 2 : correcting
        if self.ma_semi_loops != {}:
            
            #correct observations
            if measure : toc = time.time()     
            self.ma_corrections = self.correct_mag_obs(self.ma_semi_loops, disp)
            if measure : self.step_exec_time['correct w ma - correct obs'].append(round(time.time()-toc,4))

            #correct the agent pos
            #if measure : toc = time.time()
            self.correct_mag_var(disp)
            #if measure : self.step_exec_time['correct w ma - correct ag pos'].append(round(time.time()-toc,4))

            #correct traces
            #if measure : toc = time.time()
            for ag_id in self.ma_corrections:
                self.memory.tracks[ag_id].rebuild_trace()
            #if measure : self.step_exec_time['correct w ma - correct ag traces'].append(round(time.time()-toc,4))
                
            #rebuild the map if corrections have been made since the first correction
            first_step_cor_list = []
            for _k, dic in self.ma_corrections.items():
                for _w, v in dic.items():
                    if v : first_step_cor_list.append(v) 

            if first_step_cor_list != []:
                start = min(first_step_cor_list)
                if measure : toc = time.time()
                self.build_map_from_tracks(start, disp)

                self.has_corrected_map = True
                if not self.n_corr_inst : self.n_corr_inst = 0
                for _id in self.ma_corrections:
                    for portion in self.ma_corrections[_id]:
                        if self.ma_corrections[_id][portion]:
                            self.n_corr_inst += portion[1] - portion[0] +1

                if measure : self.step_exec_time['correct w ma - build map'].append(round(time.time()-toc,4))

            #get loops
            #if measure : toc = time.time()
            self.ma_gt_loops_dic, self.ma_ag_loops_dic = self.get_mag_loops_list(self.ma_semi_loops)
            #if measure : self.step_exec_time['correct w ma - get loops list'].append(round(time.time()-toc,4))

        if disp : print("Ma correction done")
        if measure : self.step_exec_time['correct w ma - global'].append(round(time.time()-tic,4))


    #global function
    def correct_w_meeting(self, disp = False, measure = MetaParameters.MEASURE_TIME): #this function aims to correct lines when 2 agents have met
        if measure : tic = time.time()     
        if disp : print("Correction after meeting ...")
        
        #step 1 : looking for loops
        #get rpr of each track
        #if measure : toc = time.time()     
        self.meeting_rpr = self.get_rpr(disp)
        #if measure : self.step_exec_time['correct w meeting - get rpr'].append(round(time.time()-toc,4))

        #get batches
        #if measure : toc = time.time()     
        self.meeting_batches = self.get_batches(disp)
        #if measure : self.step_exec_time['correct w meeting - get batches'].append(round(time.time()-toc,4))

        #find loops
        if measure : toc = time.time()
        self.meeting_meta_loops = self.find_and_get_retro_loops(disp) #this function build the meeting_meta_loops
        if measure : self.step_exec_time['correct w meeting - find loops'].append(round(time.time()-toc,4))

        #complile the meta loops into semi loops
        if self.meeting_meta_loops != []:
            #if measure : toc = time.time()
            self.meeting_semi_loops = self.get_semi_loops_dic(self.meeting_meta_loops, disp)
            #if measure : self.step_exec_time['correct w meeting - get semi loops dic'].append(round(time.time()-toc,4))


        #step 2 : correcting
        if self.meeting_semi_loops != {}:
            
            #correct observations
            if measure : toc = time.time()     
            self.meeting_corrections = self.correct_mag_obs(self.meeting_semi_loops, disp)
            if measure : self.step_exec_time['correct w meeting - correct obs'].append(round(time.time()-toc,4))
            
            #correct the agent pos
            #if measure : toc = time.time()
            self.correct_mag_var(disp)
            #if measure : self.step_exec_time['correct w meeting - correct ag pos'].append(round(time.time()-toc,4))

            #correct traces
            #if measure : toc = time.time()
            for ag_id in self.meeting_corrections:
                self.memory.tracks[ag_id].rebuild_trace()
            #if measure : self.step_exec_time['correct w meeting - correct ag traces'].append(round(time.time()-toc,4))
                
            #rebuild the map if corrections have been made since the first correction
            first_step_cor_list = []
            for _k, dic in self.meeting_corrections.items():
                for _w, v in dic.items():
                    if v : first_step_cor_list.append(v) 

            if first_step_cor_list != []: 
                start = min(first_step_cor_list)
                if measure : toc = time.time()
                self.build_map_from_tracks(start, disp)

                self.has_corrected_map = True
                if not self.n_corr_inst : self.n_corr_inst = 0
                for _id in self.meeting_corrections:
                    for portion in self.meeting_corrections[_id]:
                        if self.meeting_corrections[_id][portion]:
                            self.n_corr_inst += portion[1] - portion[0] +1

                if measure : self.step_exec_time['correct w meeting - build map'].append(round(time.time()-toc,4))

            #get loops
            #if measure : toc = time.time()
            self.meeting_gt_loops_dic, self.meeting_ag_loops_dic = self.get_mag_loops_list(self.meeting_semi_loops)
            #if measure : self.step_exec_time['correct w meeting - get loops list'].append(round(time.time()-toc,4))

        if disp : print("Meeting correction done")
        if measure : self.step_exec_time['correct w meeting - global'].append(round(time.time()-tic,4))

    
    def set_planning_variables_updates(self, disp = False):
        #reset
        self.path_planner.do_update_path = None
        self.path_planner.do_reset_path = None
        self.path_planner.do_reset_plans = None
        self.path_planner.do_reset_bans = None

        #after moving
        if self.has_moved == True and self.odom != False: #agent has moved correctly and odometry has worked
            self.path_planner.do_update_path = True
        elif self.has_moved == False: #check if the agent has been blocked
            self.path_planner.do_reset_path = True
            if disp : print(" Agent has been blocked")
        elif self.odom == False: #check if the agent thinks it has not moved, because of odometry 
            if disp : print(" Agent odometry is broken")
            self.path_planner.do_reset_path = True

        if self.has_moved and self.submaps.ag_pos == self.path_planner.goal: #goal has been reached
            if disp : print(" ... Goal reached!")
            self.path_planner.do_reset_plans = True        
        
        #after correcting
        if self.has_corrected_map == True: #map has been corrected
            if disp : print(" Localisation has been corrected")
            self.path_planner.do_reset_path = True
            self.path_planner.do_reset_bans = True
            if self.n_corr_inst >= 20 and (self.time_step - self.path_planner.last_replanning >= 3 or not self.history.has_corrected_map_history[-1]):
                self.path_planner.do_reset_plans = True

        #after sharing
        if self.new_data != [] or self.new_neighbours != []:
            self.path_planner.do_reset_plans = True


        


    #---update traces---
    def update_traces(self, n_steps_back, disp = False, measure = MetaParameters.MEASURE_TIME):
        if measure : tic = time.time()
        if disp : print("Updating traces ...")
        
        self.gt_trace = self.get_gt_trace(n_steps_back)
        self.pos_trace = self.get_pos_trace(n_steps_back)
        
        if disp : print("Traces updated")
        if measure : self.step_exec_time['update traces'].append(round(time.time()-tic,4))
    
    #---update entropy---
    def get_obs_entropy_list(self):
        obs_entropy_list = []
        for tr_id in self.memory.tracks:
            blind_v_trace = self.memory.tracks[tr_id].trace.blind_v
            for i in range(len(blind_v_trace)):
                oh = get_stdev_error(blind_v_trace[i], self.move_base.odom_error_rate)
                obs_entropy_list.append(oh)
        return obs_entropy_list
    
    def update_robot_entropy(self):
        self.robot_entropy = get_stdev_error(self.blind_v, self.move_base.odom_error_rate)

    def update_path_entropy(self):
        obs_entropy_list = self.get_obs_entropy_list()
        self.path_entropy = sum(obs_entropy_list)

    def get_localization_factor(self, rh):
        if rh < 1:
            return 1
        else:
            return 1/rh

    def updadte_global_entropy(self, disp = False):
        ph = self.path_entropy
        len_obs_list = sum([len(self.memory.tracks[tr_id].obs_list) for tr_id in self.memory.tracks])
        mh_uc = self.submaps.map_entropy_uncertain
        mh_uk = self.submaps.map_entropy_unknown
        #alpha = self.get_localization_factor(self.robot_entropy)
        #if disp : print('rh :', self.robot_entropy)
        #if disp : print('alpha :', alpha)
        add_ph = self.phw * ph / len_obs_list if len_obs_list else 0
        add_mh = self.mhw * (mh_uc+mh_uk)/(self.submaps.sub_height*self.submaps.sub_width)
        self.global_entropy = add_ph+add_mh

        if disp : print('ph :', ph)
        if disp : print('mh_uc :', mh_uc, 'mh_uk :', mh_uk)
        if disp : print('gh :', self.global_entropy)

    #---update metrics---
    def update_agent_metrics(self, disp = False):
        if disp : print(" Updating historical metrics ...")

        #move
        if self.has_moved:
            self.agent_metrics['travelled_dist'] += 1
        elif self.has_moved in [None, False]:
            self.agent_metrics['n_staying_steps'] += 1
        self.agent_metrics['travelled_dist_per_step'] = round(self.agent_metrics['travelled_dist'] / self.time_step, 3)

        #odom
        if self.odom == False:
            self.agent_metrics['n_broken_odom'] += 1
        self.agent_metrics['n_broken_odom_per_step'] = round(self.agent_metrics['n_broken_odom'] / self.time_step, 3)

        #collision
        if self.has_moved == False:
            self.agent_metrics['n_collisions'] += 1
        self.agent_metrics['n_collisions_per_step'] = round(self.agent_metrics['n_collisions'] / self.time_step, 3)

        #meeting
        self.agent_metrics['n_meetings'] += len(self.neighbours)
        self.agent_metrics['n_new_neighbours'] += len(self.new_neighbours)
        self.agent_metrics['n_new_neighbours_inst'] = len(self.new_neighbours)
        self.agent_metrics['n_new_data'] += len(self.new_data)
        self.agent_metrics['n_meetings_per_step'] = round(self.agent_metrics['n_meetings'] / self.time_step, 3)
        self.agent_metrics['n_new_neighbours_per_step'] = round(self.agent_metrics['n_new_neighbours'] / self.time_step, 3)
        self.agent_metrics['n_new_data_per_step'] = round(self.agent_metrics['n_new_data'] / self.time_step, 3)
        
        #loops and corr
        self.agent_metrics['n_corrections_inst'] = 0

        #self loops
        self.agent_metrics['n_self_corrections_inst'] = 0
        if self.self_loop_gt != None:
            self.agent_metrics['n_self_loops'] += 1
            if self.self_loop_correction:
                self.agent_metrics['n_self_loops_correcting'] += 1

                self.agent_metrics['n_self_corrections'] += self.time_step - self.self_loop_correction +1
                self.agent_metrics['n_corrections'] += self.time_step - self.self_loop_correction +1

                self.agent_metrics['n_self_corrections_inst'] += self.time_step - self.self_loop_correction +1
                self.agent_metrics['n_corrections_inst'] += self.time_step - self.self_loop_correction +1

        self.agent_metrics['n_self_loops_per_step'] = round(self.agent_metrics['n_self_loops'] / self.time_step, 3)
        self.agent_metrics['n_self_corrections_per_step'] = round(self.agent_metrics['n_self_corrections'] / self.time_step, 1)
        
        self.agent_metrics['n_self_corrections_per_self_loops'] = round(self.agent_metrics['n_self_corrections'] / self.agent_metrics['n_self_loops'], 2) if self.agent_metrics['n_self_loops'] > 0 else False
        self.agent_metrics['n_self_corrections_per_self_loops_correcting'] = round(self.agent_metrics['n_self_corrections'] / self.agent_metrics['n_self_loops_correcting'], 2) if self.agent_metrics['n_self_loops_correcting'] > 0 else False
        
        self.agent_metrics['ratio_self_loops_correcting'] = round(self.agent_metrics['n_self_loops_correcting'] / self.agent_metrics['n_self_loops'], 2) if self.agent_metrics['n_self_loops'] > 0 else False

        #ma loops
        self.agent_metrics['n_ma_corrections_inst'] = 0
        if self.ma_gt_loops_dic != {}:
            for meta_loop in self.ma_meta_loops:
                self.agent_metrics['n_ma_meta_loops'] += 1
            for _id in self.ma_semi_loops:
                for semi_loop in self.ma_semi_loops[_id]:
                    self.agent_metrics['n_ma_semi_loops'] += 1
            for _id in self.ma_corrections:
                for portion in self.ma_corrections[_id]:
                    if self.ma_corrections[_id][portion]:
                        self.agent_metrics['n_ma_meta_loops_correcting'] += 1

                        self.agent_metrics['n_ma_corrections'] += portion[1] - portion[0] +1
                        self.agent_metrics['n_corrections'] += portion[1] - portion[0] +1

                        self.agent_metrics['n_ma_corrections_inst'] += portion[1] - portion[0] +1
                        self.agent_metrics['n_corrections_inst'] += portion[1] - portion[0] +1
                        
        self.agent_metrics['n_ma_meta_loops_per_step'] = round(self.agent_metrics['n_ma_meta_loops'] / self.time_step, 3)
        self.agent_metrics['n_ma_corrections_per_step'] = round(self.agent_metrics['n_ma_corrections'] / self.time_step, 1)
        
        self.agent_metrics['n_ma_corrections_per_ma_loops'] = round(self.agent_metrics['n_ma_corrections'] / self.agent_metrics['n_ma_meta_loops'], 2) if self.agent_metrics['n_ma_meta_loops_correcting'] > 0 else False
        self.agent_metrics['n_ma_corrections_per_ma_loops_correcting'] = round(self.agent_metrics['n_ma_corrections'] / self.agent_metrics['n_ma_meta_loops_correcting'], 2) if self.agent_metrics['n_ma_meta_loops_correcting'] > 0 else False

        self.agent_metrics['ratio_ma_meta_loops_correcting'] = round(self.agent_metrics['n_ma_meta_loops_correcting'] / self.agent_metrics['n_ma_meta_loops'], 2) if self.agent_metrics['n_ma_meta_loops'] > 0 else False

        #meeting loops
        self.agent_metrics['n_meeting_corrections_inst'] = 0
        if self.meeting_gt_loops_dic != {}:
            for meta_loop in self.meeting_meta_loops:
                self.agent_metrics['n_meeting_meta_loops'] += 1
            for _id in self.meeting_semi_loops:
                for _semi_loop in self.meeting_semi_loops[_id]:
                    self.agent_metrics['n_meeting_semi_loops'] += 1
            for _id in self.meeting_corrections:
                for portion in self.meeting_corrections[_id]:
                    if self.meeting_corrections[_id][portion]:
                        self.agent_metrics['n_meeting_meta_loops_correcting'] += 1

                        self.agent_metrics['n_meeting_corrections'] += portion[1] - portion[0] +1
                        self.agent_metrics['n_corrections'] += portion[1] - portion[0] +1

                        self.agent_metrics['n_meeting_corrections_inst'] += portion[1] - portion[0] +1
                        self.agent_metrics['n_corrections_inst'] += portion[1] - portion[0] +1

        self.agent_metrics['n_meeting_meta_loops_per_step'] = round(self.agent_metrics['n_meeting_meta_loops'] / self.time_step, 3)
        self.agent_metrics['n_meeting_corrections_per_step'] = round(self.agent_metrics['n_meeting_corrections'] / self.time_step, 1)

        self.agent_metrics['n_meeting_corrections_per_meeting'] = round(self.agent_metrics['n_meeting_corrections'] / self.agent_metrics['n_meetings'], 2) if self.agent_metrics['n_meetings'] != 0 else False
        self.agent_metrics['n_meeting_corrections_per_new_meeting'] = round(self.agent_metrics['n_meeting_corrections'] / self.agent_metrics['n_new_neighbours'], 2) if self.agent_metrics['n_new_neighbours'] != 0 else False

        self.agent_metrics['n_meeting_corrections_per_meeting_loop'] = round(self.agent_metrics['n_meeting_corrections'] / self.agent_metrics['n_meeting_meta_loops'], 2) if self.agent_metrics['n_meeting_meta_loops_correcting'] > 0 else False
        self.agent_metrics['n_meeting_corrections_per_meeting_loop_correcting'] = round(self.agent_metrics['n_meeting_corrections'] / self.agent_metrics['n_meeting_meta_loops_correcting'], 2) if self.agent_metrics['n_meeting_meta_loops_correcting'] > 0 else False

        self.agent_metrics['ratio_meeting_meta_loops_correcting'] = round(self.agent_metrics['n_meeting_meta_loops_correcting'] / self.agent_metrics['n_meeting_meta_loops'], 2) if self.agent_metrics['n_meeting_meta_loops'] > 0 else False
        
        #correction
        self.agent_metrics['n_corrections_per_step'] = round(self.agent_metrics['n_corrections'] / self.time_step, 0)
        
        self.agent_metrics['n_corrections_per_meeting'] = round(self.agent_metrics['n_corrections'] / self.agent_metrics['n_meetings'], 2) if self.agent_metrics['n_meetings'] != 0 else False
        self.agent_metrics['n_corrections_per_new_meeting'] = round(self.agent_metrics['n_corrections'] / self.agent_metrics['n_new_neighbours'], 2) if self.agent_metrics['n_new_neighbours'] != 0 else False

        self.agent_metrics['n_corrections_per_loop'] = round(self.agent_metrics['n_corrections'] / (self.agent_metrics['n_self_loops']+self.agent_metrics['n_ma_meta_loops']+self.agent_metrics['n_meeting_meta_loops']), 2) if (self.agent_metrics['n_self_loops']+self.agent_metrics['n_ma_meta_loops']+self.agent_metrics['n_meeting_meta_loops']) > 0 else False
        self.agent_metrics['n_corrections_per_loop_correcting'] = round(self.agent_metrics['n_corrections'] / (self.agent_metrics['n_self_loops_correcting']+self.agent_metrics['n_ma_meta_loops_correcting']+self.agent_metrics['n_meeting_meta_loops_correcting']), 2) if (self.agent_metrics['n_self_loops_correcting']+self.agent_metrics['n_ma_meta_loops_correcting']+self.agent_metrics['n_meeting_meta_loops_correcting']) > 0 else False

        #entropy
        if self.time_step > 1:
            self.agent_metrics['entropy_loss_robot_inst'] = round(self.robot_entropy - self.history.rh_history[-1], 3)
            self.agent_metrics['entropy_loss_mapping_uc_inst'] = round(self.submaps.map_entropy_uncertain - self.history.mhc_history[-1], 1)
            self.agent_metrics['entropy_loss_mapping_uk_inst'] = round(self.submaps.map_entropy_unknown - self.history.mhk_history[-1], 1)
            self.agent_metrics['entropy_loss_path_inst'] = round(self.path_entropy - self.history.ph_history[-1], 3)
            self.agent_metrics['entropy_loss_global_inst'] = round(self.global_entropy - self.history.gh_history[-1], 3)

    #function
    def update_metrics(self, disp = False, measure = MetaParameters.MEASURE_TIME):
        if measure : tic = time.time()     
        if disp : print("Updating metrics ...")
        self.update_agent_metrics(disp)
        self.submaps.update_metrics(self.id, self.time_step, self.recently_seen_threshold, disp)
        self.memory.update_metrics(disp)

        if disp : print("Metrics updated")
        if measure : self.step_exec_time['update metrics'].append(round(time.time()-tic,4))


    def get_utility(self, disp = False, measure = MetaParameters.MEASURE_TIME):
        if disp : print("Updating utility ...")
        if measure : tic = time.time()

        prev_score = self.score

        current_score = 0

        current_score += self.agent_metrics['travelled_dist'] * RewardParameters.TRAVELLED_DIST
        current_score += self.agent_metrics['n_meetings'] * RewardParameters.N_MEETINGS
        current_score += self.agent_metrics['n_new_neighbours'] * RewardParameters.N_NEW_NEIGHBOURS
        current_score += self.agent_metrics['n_new_data'] * RewardParameters.N_NEW_DATA

        current_score += self.agent_metrics['n_self_loops'] * RewardParameters.N_SELF_LOOPS
        current_score += self.agent_metrics['n_ma_meta_loops'] * RewardParameters.N_MA_META_LOOPS
        current_score += self.agent_metrics['n_meeting_meta_loops'] * RewardParameters.N_MEETING_META_LOOPS
        current_score += self.agent_metrics['n_meeting_corrections'] * RewardParameters.N_MEETING_CORR
        current_score += self.agent_metrics['n_corrections'] * RewardParameters.N_CORRECTIONS

        current_score += self.submaps.metrics['n_squares_known'] * RewardParameters.N_SQUARE_KNOWN
        current_score += self.submaps.metrics['team_n_agents_known'] * RewardParameters.N_AGENTS

        current_score += self.memory.tracks_metrics['n_tracks'] * RewardParameters.N_TRACKS
        current_score += self.memory.tracks_metrics['n_obs_self'] * RewardParameters.N_SELF_OBS
        current_score += self.memory.tracks_metrics['n_obs'] * RewardParameters.N_OBS
        current_score += self.memory.tracks_metrics['n_visited_pos'] * RewardParameters.N_VISITED_POS
        
        self.score = current_score
        self.utility = current_score - prev_score

        if disp : print("Utility updated")
        if measure : self.step_exec_time['get utility'].append(round(time.time()-tic,4))





    #----------------------------------
    #------------MAIN loop-------------
    #----------------------------------
    def set_time(self, env):
        self.time_step = env.time_step

    #---agent's planner functions---
    def pre_update(self, env): #optional : to consider the latest neighbour plan
        #pre update the team plans
        if self.sharing_data and self.neighbours != []:
            self.share_last_plans(env.team, neigbours_only = True)


    def policy(self, measure = MetaParameters.MEASURE_TIME):

        #update the planning variables in the path planner with the last step
        self.path_planner.update_planning_variables(self.done, self.time_step)
        if not self.done and self.state != None:
            #update the planner, and deduce the agent's next action        
            print("Agent", self.id, "policy ...")
            if measure : tic_pol = time.time()

            #update penalty points
            if self.path_planner.ma_penalty_mode:
                self.path_planner.update_penalty_points(self.submaps, self.team_plans, self.time_step)

            #update plans
            self.has_planned = self.path_planner.update_planner(self.state, self.submaps, self.memory.tracks, self.metrics, self.team_plans, self.time_step, self.costmaps, self.viewpoint_planner, self.RRTtree, max_try = AgentParameters.PLANNER_MAX_TRY)
            
            if self.has_planned:
                self.submaps.suspected_holes = [elem for elem in self.path_planner.temporary_banned_waypoints]
            
            if self.has_planned and 'False' in self.has_planned: #raise a warning if agent has planned something (the path set might be False)
                print('warning - path is False')
            
            if self.has_planned and 'True' in self.has_planned:
                print('warning - agent stopped moving')

            if self.has_planned and self.has_planned != 'path':
                self.path_planner.last_replanning = self.time_step

            #get next pos and update action
            next_pos = self.path_planner.get_next_pos()
            self.action = (next_pos[0] - self.submaps.ag_pos[0], next_pos[1] - self.submaps.ag_pos[1]) if next_pos != None else None
            
            if measure : self.step_exec_time['policy'].append(round(time.time()-tic_pol,4))


    #---step mian function : move, process SLAM, calculate entropy and utility---
    def step(self, env, disp = False, measure = True):
        print("Agent", self.id, "step ...")
        if measure : tic_step = time.time()

        self.do_action(env, disp, measure) #move
        self.update_odometry(disp, measure)
        self.update_blind_value()
        self.localise(disp, measure)

        self.sensor.update_sensor(env, self.move_base, disp) #giving the env
        self.update_neighbours(disp)
        self.observation = Observation(self.id, self.time_step, self.sensor.local_map, self.sensor.borders, self.sensor.agents_scan, self.neighbours, self.move_base.pos, self.submaps.ag_pos, self.submaps.off_set, self.blind_v)
        self.memory.add_observation(self.observation, env)

        if self.sharing_data:
            self.update_last_data(disp)
            if self.neighbours != []:
                self.update_meeting_dic(disp)
                if measure : toc = time.time()
                self.share_data(env.team, disp)
                self.share_last_plans(env.team, disp = disp)
                if measure : self.step_exec_time['share data'].append(round(time.time()-toc,4))
            else:
                self.data, self.new_data = [], []

        self.memory.set_obs_list_extensions(disp)
        
        start = 1 if self.new_data != [] or self.new_neighbours != [] else self.time_step #rebuild the map from the start if the neighbours have change #else build only the current step either
        self.build_map_from_tracks(start, disp, measure)

        self.reset_loop_variables()
        if self.has_moved and self.self_loop_cor:
            self.correct_w_self(disp, measure) #find self closed loops, correct localisation observations and mapping
        if self.new_data == [] and self.has_moved and self.ma_loop_cor:
            self.correct_w_ma(disp, measure) #find MA closed loops, with other agents's previous observations, correct localisation observations and mapping
        if self.new_data != [] and self.sharing_data and self.meeting_loop_cor:
            self.correct_w_meeting(disp, measure) #when meeting agent, correct localisation observations and mapping

        if self.has_corrected_map or self.new_data != [] or self.new_neighbours != []:
            self.has_treated_holes = self.submaps.treat_holes(holes = 'all', disp = disp)
        elif self.submaps.suspected_holes != []:
            self.has_treated_holes = self.submaps.treat_holes(holes = 'suspected', disp = disp)
        
        if self.has_treated_holes:
            self.memory.save_ag_map_progression(self.time_step, self.submaps.ag_map)

            for pos in self.has_treated_holes:
                if pos in self.path_planner.temporary_banned_waypoints:
                    self.path_planner.temporary_banned_waypoints.remove(pos)
        
        self.set_planning_variables_updates()

        if measure : tic_pdp = time.time()
        self.submaps.update_submaps_ext(disp, measure) #to get the extended map
        if measure : self.step_exec_time['update submaps ext'].append(round(time.time()-tic_pdp, 5))

        if self.render_distrib and not self.path_planner.costmaps:
            if measure : tic_pdp = time.time()
            self.costmaps.update_presence_distrib(self.memory.tracks, self.team_plans, self.time_step, disp)
            if measure : self.step_exec_time['update presence distrib'].append(round(time.time()-tic_pdp, 5))
        
        if self.render_ag_map:
            self.update_traces(self.n_steps_back, disp, measure)

        if measure : tic_bm = time.time()
        self.submaps.update_blind_maps(self.blind_v, disp = disp)
        if measure : self.step_exec_time['update blind maps'].append(round(time.time()-tic_bm, 5))

        if measure : tic_h = time.time()
        self.update_robot_entropy()
        self.update_path_entropy()
        self.submaps.update_map_entropy(disp = disp)
        self.updadte_global_entropy()
        if measure : self.step_exec_time['update entropy'].append(round(time.time()-tic_h, 5))

        self.update_metrics(disp, measure)
        self.get_utility(disp, measure)

        if measure : self.step_exec_time['step'].append(round(time.time()-tic_step, 4))


    #---save function---
    def save(self, disp = False, measure = MetaParameters.MEASURE_TIME):
        print("Saving ...")
        if measure : tic_save = time.time()

        #record planner metrics
        if self.has_planned:
            if measure : toc = time.time()
            if 'cm' in self.has_planned:
                self.episode_cm_metrics.append(copy.deepcopy(self.costmaps.cm_metrics))
            if 'nav' in self.has_planned:
                self.episode_vpp_metrics.append(copy.deepcopy(self.viewpoint_planner.vpp_metrics))
            if 'rrt' in self.has_planned:
                self.episode_rrt_metrics.append(copy.deepcopy(self.RRTtree.rrt_metrics))
            if measure : self.step_exec_time['save - planner'].append(round(time.time()-toc,4))
        
        if measure : toc = time.time()
        self.episode_path_planner_history.append(copy.deepcopy(self.path_planner.pp_history))
        if measure : self.step_exec_time['save - path planner'].append(round(time.time()-toc,4))
        
        #update state
        if measure : toc = time.time()
        self.state = State(self.move_base, self.sensor, self.submaps, self.path_planner, 
            self.done, self.has_planned, self.action, self.has_moved, self.odom, self.blind_v, self.robot_entropy, self.path_entropy, self.submaps.map_entropy_uncertain, self.submaps.map_entropy_unknown, self.global_entropy,
            self.neighbours, self.new_neighbours, self.data, self.new_data, self.last_data, self.has_treated_holes, self.has_corrected_map, self.n_corr_inst, self.ag_loc_error, self.score, self.utility,
            self.self_loop_gt, self.self_loop_ag, self.self_loop_ref_observation_step, self.self_loop_correction, 
            self.ma_meta_loops, self.ma_semi_loops, self.ma_corrections, self.ma_gt_loops_dic, self.ma_ag_loops_dic,
            self.meeting_rpr, self.meeting_batches, self.meeting_meta_loops, self.meeting_semi_loops, self.meeting_corrections, self.meeting_gt_loops_dic, self.meeting_ag_loops_dic,
            self.team_plans
            )
        
        #update history
        self.history.update_history(self.state)
        if disp : print("History updated")
        if measure : self.step_exec_time['save - history'].append(round(time.time()-toc,4))

        #update and record meeting
        if disp : print("Recording meeting ...")
        if self.neighbours != []:
            if measure : toc = time.time()
            self.meeting = Meeting_Record(
                self.id, self.time_step, self.neighbours,
                self.memory.tracks, self.meeting_rpr, self.meeting_batches, 
                self.meeting_meta_loops, self.meeting_semi_loops, self.meeting_corrections,
                self.meeting_gt_loops_dic, self.meeting_ag_loops_dic,
            )
            self.meetings_records.append(self.meeting)
            if measure : self.step_exec_time['save - meeting'].append(round(time.time()-toc,4))
        else:
            self.meeting = None

        #record tracks
        if disp : print("Recording tracks ...")
        if measure : toc = time.time()
        self.tracks_records.append(copy.deepcopy(self.memory.tracks))
        if measure : self.step_exec_time['save - tracks'].append(round(time.time()-toc,4))

        #record metrics
        if disp : print("Recording metrics ...")
        if measure : toc = time.time()
        self.metrics = {
            'agent' : copy.deepcopy(self.agent_metrics),
            'submaps' : copy.deepcopy(self.submaps.metrics),
            'tracks' : copy.deepcopy(self.memory.tracks_metrics),
        }
        if measure : self.step_exec_time['save - metrics'].append(round(time.time()-toc,4))
        if measure : self.step_exec_time['save'].append(round(time.time()-tic_save,4))


    #---evaluate functions---
    def evaluate_submaps(self, map, n_agents, disp = False):
        if disp : print(" Evaluating agent's submaps ...")

        #map performance
        #area visited
        self.submaps_eval['squares_known_perc'] = round(self.submaps.metrics['n_squares_known']/(self.submaps.sub_height*self.submaps.sub_width) *100, 1)

        #obstacles
        #init
        known_points_list = self.submaps.get_known_points_list(max_range = False)
        n_correct_obstacles = 0
        n_wrong_obstacles = 0
        n_missed_obstacles = 0

        off_set = self.submaps.off_set
        for sm_pos in known_points_list:
            cor_pos = (sm_pos[0] - off_set[0], sm_pos[1] - off_set[1])
            if self.submaps.is_obstacle(sm_pos):
                if map.get_square(cor_pos) == 1:
                    n_correct_obstacles += 1
                else:
                    n_wrong_obstacles += 1
            elif self.submaps.is_free(sm_pos) and map.get_square(cor_pos) == 1:
                n_missed_obstacles += 1
        
        self.submaps_eval['n_obstacles_corr'] = n_correct_obstacles
        self.submaps_eval['n_obstacles_wrong'] = n_wrong_obstacles
        self.submaps_eval['n_obstacles_missed'] = n_missed_obstacles
        
        if self.submaps.metrics['n_obstacles'] >= 1:
            self.submaps_eval['obstacles_corr_perc'] = round(n_correct_obstacles / self.submaps.metrics['n_obstacles'] *100, 1)
            self.submaps_eval['obstacles_missed_perc'] = round(n_missed_obstacles / (n_correct_obstacles + n_missed_obstacles) *100, 1) if (n_correct_obstacles + n_missed_obstacles) > 0 else 0

        #team metrics
        if n_agents >=2:
            self.submaps_eval['team_agents_known_perc'] = round(self.submaps.metrics['team_n_agents_known'] / (n_agents-1) *100, 1)
            
            if self.submaps.metrics['team_lastly_seen_mean_ts'] != None: 
                self.submaps_eval['team_lastly_seen_mean_ts_corr'] = round((self.submaps.metrics['team_n_agents_known'] * (self.submaps.metrics['team_lastly_seen_mean_ts']) + (n_agents-1-self.submaps.metrics['team_n_agents_known']) * self.time_step) / (n_agents-1), 1)
                self.submaps_eval['team_known_perc'] = round((1 - self.submaps_eval['team_lastly_seen_mean_ts_corr'] / self.time_step) * 100, 1)
            else: 
                self.submaps_eval['team_lastly_seen_mean_ts_corr'] = self.time_step
                self.submaps_eval['team_known_perc'] = 0

            

    def evaluate_tracks(self, disp = False):
        if disp : print(" Evaluating agent's tracks ...")

        self_obs_err_list = []
        obs_err_list = []

        for ag_id in self.memory.tracks:
            for obs in self.memory.tracks[ag_id].obs_list:
                err = obs.pos_belief[-1]['err']
                #sq_err_dist = round(err[0]**2 + err[1]**2, 2) #bird distance
                err_dist = abs(err[0]) + abs(err[1]) #manhattan distance
                obs_err_list.append(err_dist)
                if ag_id == self.memory.self_id:
                    self_obs_err_list.append(err_dist)

        self.obs_eval['obs_mean_err'] = round(statistics.mean(obs_err_list), 2)
        #self.obs_eval['self_obs_err_list'] = self_obs_err_list
        #self.obs_eval['mean_self_obs_err'] = round(math.sqrt(statistics.mean(self_obs_err_list)), 2) #bird dist
        self.obs_eval['obs_self_mean_err'] = round(statistics.mean(self_obs_err_list), 2) #manhattan dist
    
    def update_perf(self, time_step, disp = False):
        if disp : print(" Evaluating agent's performance ...")
        
        self.perf['success'] = self.submaps_eval['squares_known_perc'] >= self.completness_threshold and self.submaps_eval['obstacles_corr_perc'] >= self.correctness_threshold and self.obs_eval['obs_mean_err'] <= self.mean_error_threshold        
        self.perf['done'] = self.submaps_eval['squares_known_perc'] >= self.completness_done and self.submaps_eval['obstacles_corr_perc'] >= self.correctness_done

        #turn success step to current step when success occurs (occurs only once) (same with done)
        if self.perf['success_step'] == False and self.perf['success']:
            self.perf['success_step'] = time_step
        if self.perf['done_step'] == False and self.perf['done']:
            self.perf['done_step'] = time_step

    def eval(self, env, measure = MetaParameters.MEASURE_TIME, disp = False):
        if disp : print("Agent", self.id, "evaluating step ...")
        if measure : tic_eval = time.time()

        self.evaluate_submaps(env.map, env.n_agents)
        self.evaluate_tracks()

        self.update_perf(env.time_step)
        if self.perf['done']:
            self.done = True
            print("Mission done")

        if self.has_planned and 'True' in self.has_planned:
            self.done = True
            print("Mission seems to be done")

        if measure : self.step_exec_time['eval'].append(round(time.time()-tic_eval,4))


    #---sequences---
    def update_sequences(self, disp = False):
        if disp : print(" Updating sequences ...")
        self.sequences['state']['agent']['robot_localization_error'].append(self.ag_loc_err_dist)
        self.sequences['state']['agent']['robot_blind_v'].append(self.blind_v)
        self.sequences['state']['agent']['robot_entropy'].append(round(self.robot_entropy, 3))
        self.sequences['state']['agent']['path_entropy'].append(round(self.path_entropy, 3))
        self.sequences['state']['agent']['mapping_entropy_uncertain'].append(round(self.submaps.map_entropy_uncertain, 1))
        self.sequences['state']['agent']['mapping_entropy_unknown'].append(round(self.submaps.map_entropy_unknown, 1))
        self.sequences['state']['agent']['global_entropy'].append(round(self.global_entropy, 3))
        self.sequences['state']['agent']['score'].append(self.score)

        for metric in self.agent_metrics:
            self.sequences['metrics']['agent'][metric].append(self.agent_metrics[metric])
        for metric in self.submaps.metrics:
            self.sequences['metrics']['submaps'][metric].append(self.submaps.metrics[metric])
        for metric in self.memory.tracks_metrics:
            self.sequences['metrics']['tracks'][metric].append(self.memory.tracks_metrics[metric])
        for m_eval in self.submaps_eval:
            self.sequences['eval']['submaps'][m_eval].append(self.submaps_eval[m_eval])
        for m_eval in self.obs_eval:
            if type(self.obs_eval[m_eval]) not in [list, dict]:
                self.sequences['eval']['obs'][m_eval].append(self.obs_eval[m_eval])




    #---display function---
    def display_init(self):
        print("A new agent has been deployed!\n ID :", self.id, 
                                            "\n Initial position :", self.move_base.init_pos, 
                                            "\n Range :", self.path_planner.pp_range,
                                            "\n Color :", self.color)
        self.display_agent()

    def display_agent(self):
        print("\n----------Agent Display----------")
        self.sensor.display_local_map()
        self.submaps.display_submaps()
        self.path_planner.display_path_planner()
        self.move_base.display_move_base()
        print("---------------------------------\n")














    #-------render functions-------
    #------------------------------
    #---render main loop---
    def render_agent_map(self, team = {}, mode = 'human', reversed_method = False, disp = False, measure = MetaParameters.MEASURE_TIME):
        if disp : print("Rendering Agent's", self.id, "Map ...")
        if measure : toc_ra = time.time()

        margin = 0

        #init screen
        screen_dim = self.screen_dim
        if self.agent_viewer is None:
            self.agent_viewer = rendering.Viewer(screen_dim[0], screen_dim[1])

            #black background
            create_rectangle(self.agent_viewer, 0, screen_dim[1], screen_dim[0], screen_dim[1], (0,0,0), permanent = True)
        
        #dark gray unknown field
        width = self.submaps.sub_width
        height = self.submaps.sub_height
        square_size = min(screen_dim[0]/width, screen_dim[1]/height)

        #
        if self.render_visits:
            reversed_method = False
        
        if reversed_method == False:
            background_color = (.35,.35,.35)
        else:
            background_color = (.8,.8,.8)
        
        create_rectangle(self.agent_viewer, 0, screen_dim[1], square_size*width, square_size*height, background_color)

        #---writing agent's representation---
        if self.running:
            #1. writing the agent extended map
            if measure : toc = time.time()  
            for i in range(height):
                for j in range(width):
                    write = False
                    square = self.submaps.ext_map[i,j] 
                    #square = self.submaps.ag_map[i,j] #can swich to ag_map if ext_map is not updated (not random explore planner)
                    n_scanned = self.submaps.n_scans_map[i,j]
                    fully_scanned = self.fully_scanned
                    
                    if reversed_method == False:
                        if square == -1:
                            pass
                        elif square == -0.1:
                            write = 0.45
                        elif square == 0: #between 0.65 and 0.85
                            if self.render_visits:
                                write = 0.65 + 0.20*min([1, n_scanned/fully_scanned])
                            else:
                                write = 0.8
                        elif square == 1:
                            write = 0.15
                        elif square == 10:
                            write = 1

                    else:
                        if square == 0:
                            pass
                        elif square == -0.1:
                            write = 0.6
                        elif square == -1:
                            write = 0.4
                        elif square == 1:
                            write = 0.2
                        elif square == 10:
                            write = 1

                    if write is not False:
                        x = j*square_size
                        y = screen_dim[1] - i*square_size
                        create_rectangle(self.agent_viewer, x, y, square_size, square_size, (write, write, write))
            if measure : self.step_exec_time['render - ag_ext_map'].append(round(time.time()-toc,4))

            #2. writing the traces
            if self.pos_trace != [] and self.render_trace:
                #print(" render ag trace :", self.pos_trace)
                resized_pos_trace = []
                for i in range(len(self.pos_trace)):
                    resized_point = ((self.pos_trace[i][1]+0.5)*square_size, screen_dim[1] - (self.pos_trace[i][0]+0.5)*square_size)
                    resized_pos_trace.append(resized_point)
                
                #draw_trace(self.agent_viewer, resized_pos_trace, self.color, self.n_cuts, self.dot_ratio, linewidth = 1)
                draw_path(self.agent_viewer, resized_pos_trace, self.color, linewidth = 1)

            '''
            if self.gt_trace != [] and self.render_trace and self.render_ground_truth:
                #print(" render trace :", self.gt_trace)
                resized_trace = []
                for i in range(len(self.gt_trace)):
                    resized_point = ((self.gt_trace[i][1]+0.5)*square_size, screen_dim[1] - (self.gt_trace[i][0]+0.5)*square_size)
                    resized_trace.append(resized_point)
                
                draw_trace(self.agent_viewer, resized_trace, (0,0,0.5), self.n_cuts, self.dot_ratio, linewidth = 1)
            '''

            #3. ---writing eventual loops---
            if self.render_loops:
                if self.self_loop_gt:

                    #write ag self loop
                    ag_loop_no_corr = self.self_loop_ag[:self.self_loop_correction-self.self_loop_ref_observation_step]
                    ag_loop_corrected = self.self_loop_ag[self.self_loop_correction-self.self_loop_ref_observation_step-1:]
                    #print(" render ag loop not corrected :", ag_loop_no_corr)
                    #print(" render ag loop corrected :", ag_loop_corrected)
                    
                    resized_loop = []
                    for i in range(len(ag_loop_no_corr)):
                        resized_point = ((ag_loop_no_corr[i][1]+0.5)*square_size, screen_dim[1] - (ag_loop_no_corr[i][0]+0.5)*square_size)
                        resized_loop.append(resized_point)
                    color = (1,1,0) #yellow
                    draw_path(self.agent_viewer, resized_loop, color, linewidth = 4)

                    resized_loop = []
                    for i in range(len(ag_loop_corrected)):
                        resized_point = ((ag_loop_corrected[i][1]+0.5)*square_size, screen_dim[1] - (ag_loop_corrected[i][0]+0.5)*square_size)
                        resized_loop.append(resized_point)
                    color = (0,1,0) #green
                    draw_path(self.agent_viewer, resized_loop, color, linewidth = 4)
                    
                    #write self loop
                    if self.render_ground_truth:
                        loop = self.self_loop_gt
                        #print(" render loop :", loop)
                        off_set = self.submaps.off_set
                        resized_loop = []
                        for i in range(len(loop)):
                            resized_point = ((loop[i][1]+off_set[1]+0.5)*square_size, screen_dim[1] - (loop[i][0]+off_set[0]+0.5)*square_size)
                            resized_loop.append(resized_point)
                        color = (0,0,0.5)
                        draw_path(self.agent_viewer, resized_loop, color, linewidth = 2)

                #writing ma loop
                if self.ma_semi_loops != {}:
                    for ag_id in self.ma_gt_loops_dic:
                        for i_loop in range(len(self.ma_semi_loops[ag_id])):
                            
                            #write ag ma loop
                            ag_loop = self.ma_ag_loops_dic[ag_id][i_loop]
                            meta_loop = self.ma_semi_loops[ag_id][i_loop]
                            #print(" render ag loop :", ag_loop)
                            resized_loop = []
                            for i in range(len(ag_loop)):
                                resized_point = ((ag_loop[i][1]+0.5)*square_size, screen_dim[1] - (ag_loop[i][0]+0.5)*square_size)
                                resized_loop.append(resized_point)
                            color = (1,1,0) #yellow
                            draw_path(self.agent_viewer, resized_loop, color, linewidth = 4)
                            
                            if self.ma_corrections[ag_id][tuple(meta_loop)]:
                                ag_loop_corrected = self.ma_ag_loops_dic[ag_id][i_loop][self.ma_corrections[ag_id][tuple(meta_loop)]-meta_loop[0]-1:]
                                #print(" render ag loop corrected :", ag_loop_corrected)
                                resized_loop = []
                                for i in range(len(ag_loop_corrected)):
                                    resized_point = ((ag_loop_corrected[i][1]+0.5)*square_size, screen_dim[1] - (ag_loop_corrected[i][0]+0.5)*square_size)
                                    resized_loop.append(resized_point)
                                color = (1,0.6,0) #orange
                                draw_path(self.agent_viewer, resized_loop, color, linewidth = 4)              
                        
                            #write ma loop
                            if self.render_ground_truth:
                                loop = self.ma_gt_loops_dic[ag_id][i_loop]
                                #print(" render loop :", loop)
                                off_set = self.submaps.off_set
                                resized_loop = []
                                for i in range(len(loop)):
                                    resized_point = ((loop[i][1]+off_set[1]+0.5)*square_size, screen_dim[1] - (loop[i][0]+off_set[0]+0.5)*square_size)
                                    resized_loop.append(resized_point)
                                color = (0,0,0.5)
                                draw_path(self.agent_viewer, resized_loop, color, linewidth = 2)

                #writing meeting loop
                if self.meeting_gt_loops_dic != {}:
                    for ag_id in self.meeting_ag_loops_dic:
                        for i_loop in range(len(self.meeting_semi_loops[ag_id])):
                                                    
                            #write ag ma loop
                            ag_loop = self.meeting_ag_loops_dic[ag_id][i_loop]
                            meta_loop = self.meeting_semi_loops[ag_id][i_loop]
                            #print(" render ag loop :", ag_loop)
                            resized_loop = []
                            for i in range(len(ag_loop)):
                                resized_point = ((ag_loop[i][1]+0.5)*square_size, screen_dim[1] - (ag_loop[i][0]+0.5)*square_size)
                                resized_loop.append(resized_point)
                            color = (1,1,0) #yellow
                            draw_path(self.agent_viewer, resized_loop, color, linewidth = 4)
                            
                            if self.meeting_corrections[ag_id][tuple(meta_loop)]:
                                ag_loop_corrected = self.meeting_ag_loops_dic[ag_id][i_loop][self.meeting_corrections[ag_id][tuple(meta_loop)]-meta_loop[0]-1:]
                                #print(" render ag loop corrected :", ag_loop_corrected)
                                resized_loop = []
                                for i in range(len(ag_loop_corrected)):
                                    resized_point = ((ag_loop_corrected[i][1]+0.5)*square_size, screen_dim[1] - (ag_loop_corrected[i][0]+0.5)*square_size)
                                    resized_loop.append(resized_point)
                                color = (0,1,1) #cyan
                                draw_path(self.agent_viewer, resized_loop, color, linewidth = 4)   

                            #write meeting loop
                            if self.render_ground_truth:
                                loop = self.meeting_gt_loops_dic[ag_id][i_loop]
                                #print(" render loop :", loop)
                                off_set = self.submaps.off_set
                                resized_loop = []
                                for i in range(len(loop)):
                                    resized_point = ((loop[i][1]+off_set[1]+0.5)*square_size, screen_dim[1] - (loop[i][0]+off_set[0]+0.5)*square_size)
                                    resized_loop.append(resized_point)
                                color = (0,0,0.5)
                                draw_path(self.agent_viewer, resized_loop, color, linewidth = 2)


            #4. writing the seen position of the other agents
            dim = self.sensor.dim
            pos = self.submaps.ag_pos
            for ag_id in self.submaps.ag_team_pos :
                last_obs_id = max(self.submaps.ag_team_pos[ag_id], key=lambda x: (self.submaps.ag_team_pos[ag_id][x]['time_step'], x))
                obs_point = self.submaps.ag_team_pos[ag_id][last_obs_id]['seen_pos']
                obs_step = self.submaps.ag_team_pos[ag_id][last_obs_id]['time_step']
                
                #set the delta
                delta = self.time_step - obs_step
                if last_obs_id == ag_id:
                    if ag_id > self.id:
                        delta -= 1
                else:
                    if last_obs_id > self.id:
                        delta -= 1

                if delta < self.recently_seen_threshold: #write the agent's pose
                    margin = square_size/2 * delta/self.recently_seen_threshold
                    x = obs_point[1]*square_size + margin
                    y = screen_dim[1] - obs_point[0]*square_size - margin
                    color = team[ag_id].color
                    create_rectangle(self.agent_viewer, x, y, square_size -2*margin, square_size -2*margin, color)

                    # if delta > 0: #write the area in which the agent might be
                    #     margin = square_size * delta
                    #     x = (obs_point[1]+1/2)*square_size
                    #     y = screen_dim[1] - (obs_point[0]+1/2)*square_size
                    #     color = team[ag_id].color
                    #     create_diamond(self.agent_viewer, x, y, margin, margin, color, empty = True, linewidth = 1, dot = True)

            #4bis. writing the plans of the team
            for ag_id in self.team_plans:
                goal = self.team_plans[ag_id]['goal']
                multi_goals = self.team_plans[ag_id]['multi_goals']
                plan_step = self.team_plans[ag_id]['time_step']
                #print(" render goal:", goal)
                #print(" render multi goals:", multi_goals)

                #set the delta
                delta = self.time_step - plan_step
                if last_obs_id == ag_id:
                    if ag_id > self.id:
                        delta -= 1
                else:
                    if last_obs_id > self.id:
                        delta -= 1

                if delta < self.recently_seen_threshold: #write the agent's plan
                    if multi_goals and multi_goals != True and len(multi_goals) > 1:
                        for i_goal in range(len(multi_goals)):
                            goal = multi_goals[i_goal]
                            x = goal[1]*square_size
                            y = screen_dim[1] - goal[0]*square_size
                            circle_diameter = square_size/6 * (2 - i_goal/(len(multi_goals)-1))
                            color = team[ag_id].color
                            draw_circle(self.agent_viewer, x, y, circle_diameter, square_size, color)

                    elif goal and goal != True:
                        x = goal[1]*square_size
                        y = screen_dim[1] - goal[0]*square_size
                        color = team[ag_id].color
                        draw_circle(self.agent_viewer, x, y, square_size/3, square_size, color)


            #5. writing the believed position of the agent
            pos = self.submaps.ag_pos
            x = pos[1]*square_size
            y = screen_dim[1] - pos[0]*square_size
            create_rectangle(self.agent_viewer, x, y, square_size, square_size, self.color, empty = False, linewidth = None)

            #6. writing the scanner frame and the agent scan
            dim = self.sensor.dim
            pos = self.submaps.ag_pos

            left_pos = pos[1] - dim//2 #right_pos = pos[1] + dim//2
            up_pos = pos[0] - dim//2 #down_pos = pos[0] + dim//2
            x = left_pos*square_size
            y = screen_dim[1] - up_pos*square_size
 
            if self.neighbours == []:
                color = (0,0,0.5) #dark blue
            else:
                color = self.color #purple
    
            create_rectangle(self.agent_viewer, x, y, dim*square_size, dim*square_size, color, empty = True, linewidth = 2)

            for i in range(dim):
                for j in range(dim):
                    ag_list = [elem for elem in self.sensor.agents_scan[i][j] if elem != self.id]
                    if ag_list != []:
                        obs_point = (pos[0]-dim//2+i, pos[1]-dim//2+j)
                        margin = square_size *2/5
                        x = obs_point[1]*square_size + margin
                        y = screen_dim[1] - obs_point[0]*square_size - margin
                        create_rectangle(self.agent_viewer, x, y, square_size -2*margin, square_size -2*margin, self.color)                       

            #7. writing the goal
            goal = self.path_planner.goal
            #print(" render goal:", goal)
            if goal and goal != True:
                x = goal[1]*square_size
                y = screen_dim[1] - goal[0]*square_size
                draw_circle(self.agent_viewer, x, y, square_size*2/3, square_size, self.color)

            #7bis writing the multi-goals
            multi_goals = self.path_planner.multi_goals
            #print(" render multi goals:", multi_goals)
            if multi_goals and multi_goals != True and len(multi_goals) > 1:
                for i_goal in range(len(multi_goals)):
                    goal = multi_goals[i_goal]
                    x = goal[1]*square_size
                    y = screen_dim[1] - goal[0]*square_size
                    circle_diameter = square_size/6 * (2 - i_goal/(len(multi_goals)-1))
                    draw_circle(self.agent_viewer, x, y, circle_diameter, square_size, self.color)
                
            #8. writing the path
            if self.path_planner.path not in [None, False]:
                path = [self.submaps.ag_pos] + self.path_planner.path
                #print(" render path :", path)
                resized_path = []
                for i in range(len(path)):
                    resized_point = ((path[i][1]+0.5)*square_size, screen_dim[1] - (path[i][0]+0.5)*square_size)
                    resized_path.append(resized_point)
                draw_path(self.agent_viewer, resized_path, self.color, linewidth = 3)

            #9. writing the next action
            action = self.action
            if action:
                x = (self.submaps.ag_pos[1]+0.5)*square_size
                y = screen_dim[1]-(self.submaps.ag_pos[0]+0.5)*square_size
                dir_x = action[1]*square_size
                dir_y = action[0]*square_size
                draw_arrow(self. agent_viewer, x, y, dir_x, dir_y, (0,0,0))
        
        #writing the real position of the world (ground truth)
        if self.render_ground_truth:
            margin = square_size/20
            off_set = self.submaps.off_set
            world_size = (self.submaps.sub_height - 2*off_set[0], self.submaps.sub_width - 2*off_set[1])

            x = off_set[1]*square_size - margin
            y = screen_dim[1] - off_set[0]*square_size + margin            
            create_rectangle(self.agent_viewer, x, y, square_size*world_size[1] +2*margin, square_size*world_size[0] +2*margin, (0,0,0), empty = True, linewidth = 2)

        #writing the real position of the team (ground truth)
        if self.render_ground_truth:
            margin = square_size/10
            for _id, agent in team.items():
                pos = agent.move_base.pos
                off_set = agent.submaps.off_set
                x = (pos[1] + off_set[1])*square_size - margin
                y = screen_dim[1] - (pos[0]+off_set[0])*square_size + margin
                create_rectangle(self.agent_viewer, x, y, square_size +2*margin, square_size +2*margin, agent.color, empty = True, linewidth = 3)
        
        if measure : self.step_exec_time['render - agent map'].append(round(time.time()-toc_ra,4))





    def render_planner_map(self, render_costmaps, render_vpp, render_tree, disp = False, measure = MetaParameters.MEASURE_TIME):
        if disp : print("Rendering Planner Map", self.id, "...")
        if measure : toc_rp = time.time()

        #init screen
        screen_dim = self.screen_dim
        if self.planner_viewer is None:
            self.planner_viewer = rendering.Viewer(screen_dim[0], screen_dim[1])

            #black background
            create_rectangle(self.planner_viewer, 0, screen_dim[1], screen_dim[0], screen_dim[1], (0,0,0), permanent = True)
        
        #dark gray unknown field
        height = self.costmaps.height
        width = self.costmaps.width
        square_size = min(screen_dim[0]/width, screen_dim[1]/height)
        background_color = (.9,.9,.5)
        create_rectangle(self.planner_viewer, 0, screen_dim[1], square_size*width, square_size*height, background_color)

        if render_costmaps and self.costmaps.norm_global_cost_map is not None:
            #writing global cost map
            for i in range(height):
                for j in range(width):
                    write = False
                    square = self.costmaps.norm_global_cost_map[i,j]
                    if square > 0:
                        write = .5 * (1 - square) + 0.3
                        x = j*square_size
                        y = screen_dim[1] - i*square_size
                        create_rectangle(self.planner_viewer, x, y, square_size, square_size, (0.9, write, write))

        else:
            #writing the agent extended map
            for i in range(height):
                for j in range(width):
                    write = False
                    square = self.submaps.ext_map[i,j] 
                    #square = self.submaps.ag_map[i,j] #can swich to ag_map if ext_map is not updated (not random explore planner)
                    
                    if square == -1:
                        pass
                    elif square == -0.1:
                        write = 0.45
                    elif square == 0:
                        write = 0.85
                    elif square == 1:
                        write = 0.15
                    elif square == 10:
                        write = 0.05

                    if write is not False:
                        x = j*square_size
                        y = screen_dim[1] - i*square_size
                        create_rectangle(self.planner_viewer, x, y, square_size, square_size, (write, write, write))

        #render nav
        if render_vpp:
            #writing multi goals set        
            if self.viewpoint_planner.new_multi_goal_set and len(self.viewpoint_planner.new_multi_goal_set) > 1:
                for i_goal in range(len(self.viewpoint_planner.new_multi_goal_set)):
                    goal = self.viewpoint_planner.new_multi_goal_set[i_goal]
                    x = goal[1]*square_size
                    y = screen_dim[1] - goal[0]*square_size
                    size = square_size
                    diameter = square_size/6 * (2 - i_goal/(len(self.viewpoint_planner.new_multi_goal_set)-1))
                    color = (1,0,0)
                    draw_circle(self.planner_viewer, x, y, diameter, size, color)

            #writing potential targets
            if self.viewpoint_planner.potential_targets is not None:
                for pot_target in self.viewpoint_planner.potential_targets:
                    x = pot_target[1]*square_size
                    y = screen_dim[1] - pot_target[0]*square_size                
                    size = square_size
                    diameter = square_size *1/5
                    color = (0,1,1)
                    draw_circle(self.planner_viewer, x, y, diameter, size, color)

            #writing targets
            if self.viewpoint_planner.targets is not None:
                for target in self.viewpoint_planner.targets:
                    x = target[1]*square_size
                    y = screen_dim[1] - target[0]*square_size
                    size = square_size
                    diameter = square_size *1/2
                    color = (0,1,1)
                    draw_circle(self.planner_viewer, x, y, diameter, size, color)
        
        #writing tree
        if render_tree:
            #writing nodes and path
            for node in self.RRTtree.tree:

                #draw path
                if node == self.RRTtree.start:
                    k_d = 0.3
                    color = (1,1,0)
                
                else:
                    information_gain = self.RRTtree.information_gain[node]['from_parent']
                    cum_IG = self.RRTtree.information_gain[node]['cumulative']
                    
                    wmax = 0.05
                    wmin = 0.95
                    cum_max = 60 #4
                    val_c = max(min(cum_IG, cum_max), 0)
                    #val_c = max(min(math.cbrt(cum_IG), cum_max), 0)
                    write = wmin + (wmax-wmin)*val_c/cum_max

                    kmin = 0.15
                    kmax = 0.95
                    val_max = 8 #2
                    val = max(min(information_gain, val_max),0)
                    #val = max(min(math.cbrt(information_gain), val_max),0)
                    k_d = (kmin + (kmax-kmin)*val/val_max)
                    
                    if self.RRTtree.new_node_path and node in self.RRTtree.new_node_path:
                        color = (1,write,0)
                    else:
                        color = (0,write,1)

                    #draw path
                    path = self.RRTtree.path_from_parent[node]
                    resized_path = []
                    for i in range(len(path)):
                        resized_point = ((path[i][1]+0.5)*square_size, screen_dim[1] - (path[i][0]+0.5)*square_size)
                        resized_path.append(resized_point)
                    
                    draw_path(self.planner_viewer, resized_path, color, linewidth = 2)
                
                #draw nodes
                x = node[1]*square_size
                y = screen_dim[1] - node[0]*square_size
                size = square_size
                diameter = square_size*k_d
                draw_circle(self.planner_viewer, x, y, diameter, size, color)

        #writing ag pos
        margin = square_size/10
        pos = self.submaps.ag_pos
        x = pos[1]*square_size - margin
        y = screen_dim[1] - pos[0]*square_size + margin
        create_rectangle(self.planner_viewer, x, y, square_size +2*margin, square_size +2*margin, (0.2, 0.2, 0.2), empty = True, linewidth = 7)

        if measure : self.step_exec_time['render - planner'].append(round(time.time()-toc_rp,4))


    def render_blind_map(self, disp = False, measure = MetaParameters.MEASURE_TIME):
        if disp : print("Rendering Blind Map", self.id, "...")
        if measure : toc_rb = time.time()

        #init screen
        screen_dim = self.screen_dim
        if self.blind_viewer is None:
            self.blind_viewer = rendering.Viewer(screen_dim[0], screen_dim[1])

            #black background
            create_rectangle(self.blind_viewer, 0, screen_dim[1], screen_dim[0], screen_dim[1], (0,0,0), permanent = True)

        #dark green init blind
        height = self.submaps.sub_height
        width = self.submaps.sub_width
        square_size = min(screen_dim[0]/width, screen_dim[1]/height)        
        background_color = (.1, .5, .3)
        create_rectangle(self.blind_viewer, 0, screen_dim[1], square_size*width, square_size*height, background_color)

        #writing blind
        if self.submaps.blind_v_map is not None:
            for i in range(height):
                for j in range(width):
                    blind_v = self.submaps.blind_v_map[i,j]
                    blind_d = self.submaps.blind_d_map[i,j]
                    write = 1 - min(blind_v, self.max_blind)/self.max_blind *0.8 #between 0.2 and 1
                    x = j*square_size
                    y = screen_dim[1] - i*square_size

                    if blind_d == -1:
                        create_rectangle(self.blind_viewer, x, y, square_size, square_size, (0.3, 0.3, 0.3))
                    
                    elif blind_v != self.init_blind:
                        create_rectangle(self.blind_viewer, x, y, square_size, square_size, (write, 1, 0.4+write/2))
                        diameter = min(blind_d, self.max_blind)/self.max_blind *square_size *3/4
                        draw_circle(self.blind_viewer, x, y, diameter, square_size, (0, 0, 0.2), empty=True, linewidth=2) #might take some calculation time (to check)



    #writing ag pos
        margin = square_size/10
        pos = self.submaps.ag_pos
        x = pos[1]*square_size - margin
        y = screen_dim[1] - pos[0]*square_size + margin
        create_rectangle(self.blind_viewer, x, y, square_size +2*margin, square_size +2*margin, (0.2, 0.2, 0.2), empty = True, linewidth = 7)

        if measure : self.step_exec_time['render - blind'].append(round(time.time()-toc_rb,4))




    def render_prob_occ_gric(self, disp = False,  measure = MetaParameters.MEASURE_TIME):
        if disp : print("Rendering Probabilistic Occupancy Grid", self.id, "...")
        if measure : toc_rp = time.time()

        #init screen
        screen_dim = self.screen_dim
        if self.pog_viewer is None:
            self.pog_viewer = rendering.Viewer(screen_dim[0], screen_dim[1])

            #black background
            create_rectangle(self.pog_viewer, 0, screen_dim[1], screen_dim[0], screen_dim[1], (0,0,0), permanent = True)

        #unknown grid
        height = self.submaps.sub_height
        width = self.submaps.sub_width
        square_size = min(screen_dim[0]/width, screen_dim[1]/height)        
        background_color = (.5,.5, 1)
        create_rectangle(self.pog_viewer, 0, screen_dim[1], square_size*width, square_size*height, background_color)

        #writing POG
        if self.submaps.probabilistic_occupancy_grid is not None:
            for i in range(height):
                for j in range(width):
                    square = self.submaps.probabilistic_occupancy_grid[i,j]
                    if square == -1:
                        write = .8 * (1-square) + 0.2 #between 0.2 and 1.0
                        x = j*square_size
                        y = screen_dim[1] - i*square_size
                        create_rectangle(self.pog_viewer, x, y, square_size, square_size, (0.3, 0.3, 0.3))

                    elif square != 0.5:
                        write = .6 * (1-square) + 0.2 #between 0.2 and 0.8
                        x = j*square_size
                        y = screen_dim[1] - i*square_size
                        create_rectangle(self.pog_viewer, x, y, square_size, square_size, (write, write, 1))
                    


        #writing ag pos
        margin = square_size/10
        pos = self.submaps.ag_pos
        x = pos[1]*square_size - margin
        y = screen_dim[1] - pos[0]*square_size + margin
        create_rectangle(self.pog_viewer, x, y, square_size +2*margin, square_size +2*margin, (0.2, 0.2, 0.2), empty = True, linewidth = 7)

        if measure : self.step_exec_time['render - pdp'].append(round(time.time()-toc_rp,4))



    def render_prob_distrib(self, team, disp = False,  measure = MetaParameters.MEASURE_TIME):
        if disp : print("Rendering Probabilisty Distribution of Presence", self.id, "...")
        if measure : toc_rp = time.time()

        #init screen
        screen_dim = self.screen_dim
        if self.distrib_viewer is None:
            self.distrib_viewer = rendering.Viewer(screen_dim[0], screen_dim[1])

            #black background
            create_rectangle(self.distrib_viewer, 0, screen_dim[1], screen_dim[0], screen_dim[1], (0,0,0), permanent = True)

        #white grid
        height = self.submaps.sub_height
        width = self.submaps.sub_width
        square_size = min(screen_dim[0]/width, screen_dim[1]/height)        
        background_color = (.95, .9, .85)
        create_rectangle(self.distrib_viewer, 0, screen_dim[1], square_size*width, square_size*height, background_color, permanent = True)


        #1. write the PDP maps for each agent
        if self.costmaps.presence_distrib != {}:
            for i in range(height):
                for j in range(width):
                    color_list = []
                    for m_id in self.costmaps.presence_distrib:
                        if m_id not in self.neighbours and type(self.costmaps.presence_distrib[m_id]) is np.ndarray:
                            if self.costmaps.presence_distrib[m_id][i,j] != 0:
                                prob = self.costmaps.presence_distrib[m_id][i,j]
                                write = prob**(1/3) #between 0 and 1
                                color = team[m_id].color
                                w_color = ((1-write*(1-color[0])) *.8+.15, (1-write*(1-color[1])) *.85+.1, (1-write*(1-color[2])) *.9+.05)
                                color_list.append(w_color)

                    if color_list != []:
                        final_color = (statistics.mean([color[0] for color in color_list]), statistics.mean([color[1] for color in color_list]), statistics.mean([color[2] for color in color_list]))
                        x = j*square_size
                        y = screen_dim[1] - i*square_size
                        create_rectangle(self.distrib_viewer, x, y, square_size, square_size, final_color)

        #2. writing the seen position of the other agents
        dim = self.sensor.dim
        pos = self.submaps.ag_pos
        for ag_id in self.submaps.ag_team_pos :
            last_obs_id = max(self.submaps.ag_team_pos[ag_id], key=lambda x: (self.submaps.ag_team_pos[ag_id][x]['time_step'], x))
            obs_point = self.submaps.ag_team_pos[ag_id][last_obs_id]['seen_pos']
            obs_step = self.submaps.ag_team_pos[ag_id][last_obs_id]['time_step']
            
            #set the delta
            delta = self.time_step - obs_step
            if last_obs_id == ag_id:
                if ag_id > self.id:
                    delta -= 1
            else:
                if last_obs_id > self.id:
                    delta -= 1

            if delta < self.recently_seen_threshold: #write the agent's pose
                margin = square_size/2 * delta/self.recently_seen_threshold
                x = obs_point[1]*square_size + margin
                y = screen_dim[1] - obs_point[0]*square_size - margin
                color = team[ag_id].color
                create_rectangle(self.distrib_viewer, x, y, square_size -2*margin, square_size -2*margin, color)

                if delta > 0: #write the area in which the agent might be
                    margin = square_size * delta
                    x = (obs_point[1]+1/2)*square_size
                    y = screen_dim[1] - (obs_point[0]+1/2)*square_size
                    color = team[ag_id].color
                    create_diamond(self.distrib_viewer, x, y, margin, margin, color, empty = True, linewidth = 1, dot = True)

        #3. writing the plans of the team
        for ag_id in self.team_plans:
            goal = self.team_plans[ag_id]['goal']
            multi_goals = self.team_plans[ag_id]['multi_goals']
            plan_step = self.team_plans[ag_id]['time_step']
            #print(" render goal:", goal)
            #print(" render multi goals:", multi_goals)

            #set the delta
            delta = self.time_step - plan_step
            if last_obs_id == ag_id:
                if ag_id > self.id:
                    delta -= 1
            else:
                if last_obs_id > self.id:
                    delta -= 1

            if delta < self.recently_seen_threshold: #write the agent's plan
                if multi_goals and multi_goals != True and len(multi_goals) > 1:
                    for i_goal in range(len(multi_goals)):
                        goal = multi_goals[i_goal]
                        x = goal[1]*square_size
                        y = screen_dim[1] - goal[0]*square_size
                        circle_diameter = square_size/6 * (2 - i_goal/(len(multi_goals)-1))
                        color = team[ag_id].color
                        draw_circle(self.distrib_viewer, x, y, circle_diameter, square_size, color)

                elif goal and goal != True:
                    x = goal[1]*square_size
                    y = screen_dim[1] - goal[0]*square_size
                    color = team[ag_id].color
                    draw_circle(self.distrib_viewer, x, y, square_size/3, square_size, color)

        #writing self ag pos
        margin = square_size/10
        pos = self.submaps.ag_pos
        x = pos[1]*square_size - margin
        y = screen_dim[1] - pos[0]*square_size + margin
        create_rectangle(self.distrib_viewer, x, y, square_size +2*margin, square_size +2*margin, self.color, empty = True, linewidth = 7)

        #writing the real position of the world (ground truth)
        if self.render_ground_truth:
            margin = square_size/20
            off_set = self.submaps.off_set
            world_size = (self.submaps.sub_height - 2*off_set[0], self.submaps.sub_width - 2*off_set[1])

            x = off_set[1]*square_size - margin
            y = screen_dim[1] - off_set[0]*square_size + margin            
            create_rectangle(self.distrib_viewer, x, y, square_size*world_size[1] +2*margin, square_size*world_size[0] +2*margin, (0,0,0), empty = True, linewidth = 2)

        #writing the real position of the team (ground truth)
        if self.render_ground_truth:
            margin = square_size/10
            for _id, agent in team.items():
                pos = agent.move_base.pos
                off_set = agent.submaps.off_set
                x = (pos[1] + off_set[1])*square_size - margin
                y = screen_dim[1] - (pos[0]+off_set[0])*square_size + margin
                create_rectangle(self.distrib_viewer, x, y, square_size +2*margin, square_size +2*margin, agent.color, empty = True, linewidth = 3)



        if measure : self.step_exec_time['render - pog'].append(round(time.time()-toc_rp,4))




    def render(self, env, disp = MetaParameters.DISPLAY, measure = MetaParameters.MEASURE_TIME):
        print("Rendering ...")
        if measure : tic_r = time.time()

        if self.path_planner.path == False:
            self.episode_agents_frames.append(None)
            self.episode_planner_frames.append(None)
            self.episode_blind_frames.append(None)
            self.episode_pog_frames.append(None)
            self.episode_distrib_frames.append(None)
            print("error - path False render error")
            return
        
        if self.render_ag_map:
            #reversed_method = self.time_step > self.n_steps/3
            reversed_method = False
            self.render_agent_map(env.team, mode = 'human', reversed_method = reversed_method, disp = disp, measure = measure)
            self.episode_agents_frames.append(self.agent_viewer.render(return_rgb_array = True))

        if self.render_planner:
            self.render_planner_map(self.render_costmaps, self.render_vpp, self.render_tree)
            self.episode_planner_frames.append(self.planner_viewer.render(return_rgb_array = True))

        if self.render_blind:
            self.render_blind_map()
            self.episode_blind_frames.append(self.blind_viewer.render(return_rgb_array = True))

        if self.render_pog:
            self.render_prob_occ_gric()
            self.episode_pog_frames.append(self.pog_viewer.render(return_rgb_array = True))
        
        if self.render_distrib:
            self.render_prob_distrib(env.team)
            self.episode_distrib_frames.append(self.distrib_viewer.render(return_rgb_array = True))

        if measure : self.step_exec_time['render'].append(round(time.time()-tic_r,4))





    def close_viewers(self):
        if self.agent_viewer:
            self.agent_viewer.close()
            self.agent_viewer = None
        
        if self.planner_viewer:
            self.planner_viewer.close()
            self.planner_viewer = None

        if self.planner_viewer:
            self.planner_viewer.close()
            self.planner_viewer = None

        if self.blind_viewer:
            self.blind_viewer.close()
            self.blind_viewer = None

        if self.pog_viewer:
            self.pog_viewer.close()
            self.pog_viewer = None

        if self.distrib_viewer:
            self.distrib_viewer.close()
            self.distrib_viewer = None


