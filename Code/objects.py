import copy
import copy
import numpy as np
import statistics

class Observation:
    def __init__(self, ag_id, time_step, local_map, borders, agents_scan, neighbours, real_pos, pos_belief, off_set, blind_v):
        
        self.ag_id = ag_id
        self.time_step = time_step

        self.local_map = local_map
        self.borders = borders
        self.agents_scan = agents_scan
        self.neighbours = neighbours

        self.real_pos = real_pos #cheat code
        self.pos_belief = []
        self.pos_belief.append({
            'time_step' : self.time_step,
            'pos_belief' : pos_belief, #the position belief from which the agent did the scan
            'blind_v' : blind_v, #the blind value is the number of moves without certainty of no error from the start of the episode, it converges to zero, it estimates the certainty level of the pos belief
            'err' : (pos_belief[0] - self.real_pos[0] - off_set[0], pos_belief[1] - self.real_pos[1] - off_set[1])
        })

        self.n_updates = 0

    def add_new_pos_belief(self, time_step, new_pos_belief, new_blind_v, off_set):
        self.pos_belief.append({
            'time_step' : time_step, 
            'pos_belief' : new_pos_belief,
            'blind_v' : new_blind_v,
            'err' : (new_pos_belief[0] - self.real_pos[0] - off_set[0], new_pos_belief[1] - self.real_pos[1] - off_set[1])
        })
        self.n_updates += 1
    
    def add_new_error(self, time_step, new_error, new_blind_v, off_set):
        self.pos_belief.append({
            'time_step' : time_step, 
            'pos_belief' : (self.real_pos[0] + new_error[0] + off_set[0], self.real_pos[1] + new_error[1] + off_set[1]), 
            'blind_v' : new_blind_v,
            'err' : new_error
        })
        self.n_updates += 1

    def update_last_blind_v(self, new_blind_v):
        self.pos_belief[-1]['blind_v'] = new_blind_v
        self.n_updates += 1


class Trace:
    def __init__(self, ag_id):

        self.ag_id = ag_id
        self.reset()
    
    def reset(self):
        self.gt_pos = [] #successive grount truth poses (without repetitions if the agent is static)
        self.ag_pos = [] #successive agent belived poses (without repetition if the agent is static)
        self.blind_list = [] #blind counters associated to every agent believed past poses
        self.blind_v = [] #blind counters associated to every agent believed past poses
        self.n_corr = [] #number of corrections of oberservations associated to every agent believed past poses
        self.time_steps = [] #time_steps associated to every agent believed past poses
    
    def add_obs(self, obs):
        #append a position (ground truth or believed position) to the lists      
        if self.gt_pos == [] or obs.real_pos != self.gt_pos[-1]:
            self.gt_pos.append(obs.real_pos)

        if self.ag_pos == [] or obs.pos_belief[-1]['pos_belief'] != self.ag_pos[-1]:
            self.ag_pos.append(obs.pos_belief[-1]['pos_belief'])
            self.blind_list.append([obs.pos_belief[idx]['blind_v'] for idx in range(len(obs.pos_belief))])
            self.blind_v.append(min([obs.pos_belief[idx]['blind_v'] for idx in range(len(obs.pos_belief))]))
            self.time_steps.append(obs.time_step)
            self.n_corr.append(len(obs.pos_belief) -1)

    def rebuild(self, obs_list):
        if obs_list == []:
            return None
        self.reset()
        for obs in obs_list:
            self.add_obs(obs)


class Track:
    def __init__(self, ag_id):
        self.ag_id = ag_id

        self.obs_list = [] #observation list ; contend of the tracks with every observations
        self.last_update = 0 #last time the obs_list has been updated

        self.visits_counter = {} #count how many times a position has been visited
        self.seen_counter = {} #count how many times a square has been seen

        self.trace = Trace(self.ag_id)

        self.meeting_dic = {}

        self.owners = [] #list of the previous owners of the track (-> where the data comes from)
        
        #extention of the observation list (from time_step 0 to the current time_step) built by the current owner
        #self.ext = [] #unused
        self.meta_ext = [] #description of the extented observations list  (-> which agent at which time_step)

    def update_last_update(self, time_step = None):
        self.last_update = time_step if time_step else self.obs_list[-1].time_step

    def update_visits_counter(self, observation, disp = False):
        if disp : print(" Updating visited pos counter ...")
        if observation.real_pos not in self.visits_counter:
            self.visits_counter[observation.real_pos] = 1
        else:
            self.visits_counter[observation.real_pos] += 1

    def update_seen_counter(self, observation, map, disp = False):
        if disp : print(" Updating seen pos counter ...")
        dim = len(observation.local_map)
        for i in range(dim):
            for j in range(dim):
                seen_pos = (observation.real_pos[0]-dim//2+i, observation.real_pos[1]-dim//2+j)
                if not map.is_out(seen_pos):
                    if seen_pos not in self.seen_counter:
                        self.seen_counter[seen_pos] = 1
                    else:
                        self.seen_counter[seen_pos] += 1

    def rebuild_trace(self):
        if self.obs_list == []:
            return None
        
        self.trace.reset()

        for obs in self.obs_list:
            self.trace.add_obs(obs)
        

class Memory:
    def __init__(self, self_id):

        self.self_id = self_id

        #memory variables
        self.tracks = {} #save past observations of all agents met
        self.tracks[self.self_id] = Track(self.self_id)
        self.tracks[self.self_id].owners.append({'owner' : self.self_id, 'from step' : 0})

        self.ag_map_progression = [] #save submap visits steps through episode
        self.n_scans_map_progression = []
        self.blind_table_progression = []
        self.ag_team_pos_progression = [] #save team position through episode

        #dict
        self.meeting_dict = {}
        self.data_dict = {}

        #metrics
        self.tracks_metrics = { #set metrics on the memory
            #obs
            'n_tracks' : None,
            'n_obs' : None,
            'n_obs_self' : None,
            'n_obs_self_perc' : None,

            #visited pos
            'n_visited_pos' : None,
            'visits_new_perc' : None,
            'visits_re_perc' : None,

            'visits_counter_mean' : None,
            'visits_counter_max' : None,

            'n_seen_pos' : None,
            'seen_counter_mean' : None,
            'seen_counter_med' : None,
            'seen_counter_q3' : None,
            'seen_counter_d9' : None,
            'seen_counter_max' : None,

            #blind v
            'obs_self_blind_v_mean' : None,
            'obs_self_blind_v_med' : None,
            'obs_self_blind_v_q3' : None,
            'obs_self_blind_v_d9' : None,
            'obs_self_blind_v_max' : None,

            'obs_blind_v_mean' : None,
            'obs_blind_v_med' : None,
            'obs_blind_v_q3' : None,
            'obs_blind_v_d9' : None,
            'obs_blind_v_max' : None,

            #n corr
            'obs_self_n_corr_mean' : None,
            'obs_self_n_corr_med' : None,
            'obs_n_corr_mean' : None,
            'obs_n_corr_med' : None,

            #meeting_dic
            'ts_between_meetings' : None, #unsused
            'ts_between_data' : None, #unsused
        }

    def add_observation(self, observation, env):
        self.tracks[self.self_id].obs_list.append(observation)
        self.tracks[self.self_id].trace.add_obs(observation)
        self.tracks[self.self_id].update_last_update()
        self.tracks[self.self_id].update_visits_counter(observation)
        self.tracks[self.self_id].update_seen_counter(observation, env.map)

    def copy_track(self, oth_id, track):
        if oth_id not in self.tracks:
            self.tracks[oth_id] = Track(oth_id)

        self.tracks[oth_id].obs_list = copy.deepcopy(track.obs_list)
        self.tracks[oth_id].last_update = track.last_update
        self.tracks[oth_id].trace = copy.deepcopy(track.trace)
        self.tracks[oth_id].meeting_dic = copy.deepcopy(track.meeting_dic)
        self.tracks[oth_id].owners = track.owners[:]
    
    def set_obs_list_extensions(self, disp = False):
        if disp : print("Extending all tracks ...")

        for oth_id in self.tracks:
            
            #reset ext variables
            self.tracks[oth_id].meta_ext = []

            for i_ow in range(len(self.tracks[oth_id].owners)):
                current_owner = self.tracks[oth_id].owners[i_ow]['owner']
                if self.tracks[current_owner].obs_list != []:
                    start = max(self.tracks[oth_id].owners[i_ow]['from step'], 1)
                    end = self.tracks[current_owner].obs_list[-1].time_step

                    #add the current id obs list to the oth id ext list
                    self.tracks[oth_id].meta_ext.append({'ag' : current_owner, 'intv' : [start, end]})
                    
            if disp : print(" meta ext :", self.tracks[oth_id].meta_ext)

    def save_ag_map_progression(self, time_step, ag_map):
        if len(self.ag_map_progression) > time_step-1:
            self.ag_map_progression[time_step-1] = np.copy(ag_map)
        elif len(self.ag_map_progression) == time_step-1:
            self.ag_map_progression.append(np.copy(ag_map))

    def save_n_scans_map_progression(self, time_step, n_scans_map):
        if len(self.n_scans_map_progression) > time_step-1:
            self.n_scans_map_progression[time_step-1] = np.copy(n_scans_map)
        elif len(self.n_scans_map_progression) == time_step-1:
            self.n_scans_map_progression.append(np.copy(n_scans_map))

    def save_blind_table_progression(self, time_step, blind_table):
        if len(self.blind_table_progression) > time_step-1:
            self.blind_table_progression[time_step-1] = copy.deepcopy(blind_table)
        elif len(self.blind_table_progression) == time_step-1:
            self.blind_table_progression.append(copy.deepcopy(blind_table))

    def save_ag_team_pos_progression(self, time_step, ag_team_pos):
        if len(self.ag_team_pos_progression) > time_step-1:
            self.ag_team_pos_progression[time_step-1] = copy.deepcopy(ag_team_pos)
        elif len(self.ag_team_pos_progression) == time_step-1:
            self.ag_team_pos_progression.append(copy.deepcopy(ag_team_pos))

    def update_metrics(self, disp = False):
        if disp : print(" Getting tracks metrics ...")

        #observations
        self.tracks_metrics['n_tracks'] = len(self.tracks)
        self.tracks_metrics['n_obs'] = sum(len(self.tracks[ag_id].obs_list) for ag_id in self.tracks)
        self.tracks_metrics['n_obs_self'] = len(self.tracks[self.self_id].obs_list)
        self.tracks_metrics['n_obs_self_perc'] = int(self.tracks_metrics['n_obs_self'] / self.tracks_metrics['n_obs'] *100)

        #visited and and seen positions counters
        total_visits_counter = {} #count how many times a position has been visited
        total_seen_counter = {} #count how many times a square has been seen
        for ag_id in self.tracks:
            for real_pos in self.tracks[ag_id].visits_counter:
                if real_pos in total_visits_counter:
                    total_visits_counter[real_pos] += self.tracks[ag_id].visits_counter[real_pos]
                else:
                    total_visits_counter[real_pos] = self.tracks[ag_id].visits_counter[real_pos]
            
            for seen_pos in self.tracks[ag_id].seen_counter:
                if seen_pos in total_seen_counter:
                    total_seen_counter[seen_pos] += self.tracks[ag_id].seen_counter[seen_pos]
                else:
                    total_seen_counter[seen_pos] = self.tracks[ag_id].seen_counter[seen_pos]

        self.tracks_metrics['n_visited_pos'] = len(total_visits_counter)
        self.tracks_metrics['n_seen_pos'] = len(total_seen_counter)
        self.tracks_metrics['visits_new_perc'] = int(self.tracks_metrics['n_visited_pos'] / self.tracks_metrics['n_obs'] *100)
        self.tracks_metrics['visits_re_perc'] = int((self.tracks_metrics['n_obs'] - self.tracks_metrics['n_visited_pos']) / self.tracks_metrics['n_obs'] *100)

        if len(total_visits_counter) >= 1:
            self.tracks_metrics['visits_counter_mean'] = round(statistics.mean(list(total_visits_counter.values())), 2)
            self.tracks_metrics['visits_counter_max'] = max(list(total_visits_counter.values()))

        if len(total_seen_counter) >= 1:
            self.tracks_metrics['seen_counter_mean'] = round(statistics.mean(list(total_seen_counter.values())), 2)
            self.tracks_metrics['seen_counter_med'] = int(round(statistics.median(list(total_seen_counter.values())), 2))
            self.tracks_metrics['seen_counter_max'] = max(list(total_seen_counter.values()))
        if len(total_seen_counter) >= 3:
            self.tracks_metrics['seen_counter_q3'] = int(statistics.quantiles(list(total_seen_counter.values()), n=4)[-1])
            self.tracks_metrics['seen_counter_d9'] = int(statistics.quantiles(list(total_seen_counter.values()), n=10)[-1])

        #seen squares

        #blind v
        if len(self.tracks[self.self_id].trace.blind_list) >= 1:
            #blind v
            self_obs_blind_v = self.tracks[self.self_id].trace.blind_v
            obs_blind_v = []
            for ag_id in self.tracks:
                obs_blind_v += self.tracks[ag_id].trace.blind_v

            self.tracks_metrics['obs_self_blind_v_mean'] = round(statistics.mean(self_obs_blind_v), 2)
            self.tracks_metrics['obs_self_blind_v_max'] = max(self_obs_blind_v)
            self.tracks_metrics['obs_blind_v_mean'] = round(statistics.mean(obs_blind_v), 2)
            self.tracks_metrics['obs_blind_v_max'] = max(obs_blind_v)

            if len(self.tracks[self.self_id].trace.blind_list) >= 3:
                self.tracks_metrics['obs_self_blind_v_med'] = int(statistics.median(self_obs_blind_v))
                self.tracks_metrics['obs_self_blind_v_q3'] = int(statistics.quantiles(self_obs_blind_v, n=4)[-1])
                self.tracks_metrics['obs_self_blind_v_d9'] = int(statistics.quantiles(self_obs_blind_v, n=10)[-1])
                self.tracks_metrics['obs_blind_v_med'] = int(statistics.median(obs_blind_v))
                self.tracks_metrics['obs_blind_v_q3'] = int(statistics.quantiles(obs_blind_v, n=4)[-1])
                self.tracks_metrics['obs_blind_v_d9'] = int(statistics.quantiles(obs_blind_v, n=10)[-1])

            #trace n corr
            self_obs_n_corr = self.tracks[self.self_id].trace.n_corr
            obs_n_corr = []
            for ag_id in self.tracks:
                obs_n_corr += self.tracks[ag_id].trace.n_corr

            self.tracks_metrics['obs_self_n_corr_mean'] = round(statistics.mean(self_obs_n_corr), 2)
            self.tracks_metrics['obs_n_corr_mean'] = round(statistics.mean(obs_n_corr), 2)

            if len(self.tracks[self.self_id].trace.n_corr) >= 3:
                self.tracks_metrics['obs_self_n_corr_med'] = int(statistics.median(self_obs_n_corr))
                self.tracks_metrics['obs_n_corr_med'] = int(statistics.median(obs_n_corr))


        #ts between meetings n data
        all_agents_ts_btw_meetings = []
        for ag_id in self.meeting_dict:
            ts_btw_meetings = [0]
            for i_ts in range(len(self.meeting_dict[ag_id])-1):
                ts_btw_meetings.append(self.meeting_dict[ag_id][i_ts+1] - self.meeting_dict[ag_id][i_ts])
            ts_btw_meetings_limited = [elem for elem in ts_btw_meetings if elem < 3]
            all_agents_ts_btw_meetings.extend(ts_btw_meetings_limited)
        if all_agents_ts_btw_meetings != []:
            self.tracks_metrics['ts_between_meetings'] = round(statistics.mean(all_agents_ts_btw_meetings), 2)

        all_agents_ts_btw_data = []
        for ag_id in self.meeting_dict:
            ts_btw_data = [0]
            for i_ts in range(len(self.meeting_dict[ag_id])-1):
                ts_btw_data.append(self.meeting_dict[ag_id][i_ts+1] - self.meeting_dict[ag_id][i_ts])
            ts_btw_data_limited = [elem for elem in ts_btw_data if elem < 3]
            all_agents_ts_btw_data.extend(ts_btw_data_limited)
        if all_agents_ts_btw_data != []:
            self.tracks_metrics['ts_between_data'] = round(statistics.mean(all_agents_ts_btw_data), 2)





class State:
    def __init__(
        self, move_base, sensor, submaps, path_planner, 
        done, has_planned, action, has_moved, odom, blind_v, robot_entropy, path_entropy, map_entropy_uc, map_entropy_uk, global_entropy,
        neighbours, new_neighbours, data, new_data, last_data, has_treated_holes, has_corrected_map, n_corr_inst, ag_loc_error, score, utility,
        self_loop_gt, self_loop_ag, self_loop_ref_observation_step, self_loop_correction,
        ma_meta_loops, ma_semi_loops, ma_corrections, ma_gt_loops_dic, ma_ag_loops_dic, 
        meeting_rpr, meeting_batches, meeting_meta_loops, meeting_semi_loops, meeting_corrections, meeting_gt_loops_dic, meeting_ag_loops_dic,
        team_plans
        ):

        self.pos = move_base.pos

        self.local_map = np.copy(sensor.local_map)
        self.borders = sensor.borders
        self.team_in_range = copy.copy(sensor.team_in_range)
        self.ag_pos = submaps.ag_pos
        self.ag_map = np.copy(submaps.ag_map)
        self.n_scans_map = np.copy(submaps.n_scans_map)
        self.blind_v_map = np.copy(submaps.blind_v_map)
        self.blind_d_map = np.copy(submaps.blind_d_map)
        self.pog = np.copy(submaps.probabilistic_occupancy_grid)
        self.ag_team_pos = copy.deepcopy(submaps.ag_team_pos)
        self.goal = path_planner.goal
        self.path = copy.copy(path_planner.path)
        self.multi_goals = path_planner.multi_goals
        self.expected_gain = path_planner.expected_gain
        self.team_plans = copy.deepcopy(team_plans)

        self.done = done
        self.has_planned = has_planned
        self.action = action
        self.has_moved = has_moved
        self.odom = odom
        self.neighbours = neighbours
        self.new_neighbours = new_neighbours
        self.data = data
        self.new_data = new_data
        self.last_data = last_data
        self.blind_v = blind_v
        self.robot_entropy = robot_entropy
        self.path_entropy = path_entropy
        self.map_entropy_uncertain = map_entropy_uc
        self.map_entropy_unknown = map_entropy_uk
        self.global_entropy = global_entropy

        self.self_loop_gt = self_loop_gt
        self.self_loop_ag = self_loop_ag
        self.self_loop_ref_observation_step = self_loop_ref_observation_step
        self.self_loop_correction = self_loop_correction

        self.ma_meta_loops = copy.deepcopy(ma_meta_loops)
        self.ma_semi_loops = copy.deepcopy(ma_semi_loops)
        self.ma_gt_loops_dic = copy.deepcopy(ma_gt_loops_dic)
        self.ma_ag_loops_dic = copy.deepcopy(ma_ag_loops_dic)
        self.ma_corrections = copy.deepcopy(ma_corrections)

        self.meeting_rpr = copy.deepcopy(meeting_rpr)
        self.meeting_batches = copy.deepcopy(meeting_batches)
        self.meeting_meta_loops = copy.deepcopy(meeting_meta_loops)
        self.meeting_semi_loops = copy.deepcopy(meeting_semi_loops)
        self.meeting_corrections = copy.deepcopy(meeting_corrections)
        self.meeting_gt_loops_dic = copy.deepcopy(meeting_gt_loops_dic)
        self.meeting_ag_loops_dic = copy.deepcopy(meeting_ag_loops_dic)

        self.has_treated_holes = has_treated_holes
        self.has_corrected_map = has_corrected_map
        self.n_corr_inst = n_corr_inst

        self.ag_loc_error = ag_loc_error
        self.score = score
        self.utility = utility


class History:
    def __init__(self, self_id):
        self.self_id = self_id

        #history variables
        self.state_history = []

        self.pos_history = []
        self.local_map_history = []
        self.borders_history = []
        self.team_in_range_history = []
        self.ag_pos_history = []
        self.ag_map_history = []
        self.n_scans_map_history = []
        self.blind_v_map_history = []
        self.blind_d_map_history = []
        self.pog_history = []
        self.ag_team_pos_history = []
        self.goal_history = []
        self.multi_goals_history = []
        self.goal_history_not_repeted = []
        self.expected_gain_history = []
        self.path_planner_history = []
        self.team_plans_history = []

        self.done_history = []
        self.has_planned_history = []
        self.action_history = []
        self.has_moved_history = []
        self.odom_history = []
        self.blind_v_history = []
        self.rh_history = []
        self.ph_history = []
        self.mhc_history = []
        self.mhk_history = []
        self.gh_history = []
        self.neighbours_history = []
        self.data_history = []

        self.self_loop_gt_history = []
        self.self_loop_ag_history = []
        self.self_loop_ref_observation_step_history = []
        self.self_loop_correction_history = []

        self.ma_meta_loops_history = []
        self.ma_semi_loops_history = []
        self.ma_gt_loops_history = []
        self.ma_ag_loops_history = []
        self.ma_corrections_history = []

        self.meeting_meta_loops_history = []
        self.metting_semi_loops_history = []
        self.meeting_rpr_history = []
        self.meeting_batches_history = []
        self.meeting_corrections_history = []
        self.meeting_gt_loop_history = []
        self.meeting_ag_loop_history = []

        self.has_corrected_map_history = []
        self.has_treated_holes_history = []
        self.n_corr_inst_history = []

        self.ag_loc_error_history = []
        self.score_history = []
        self.utility_history = []

    def update_history(self, state):
        self.state_history.append(state)

        self.pos_history.append(state.pos)
        self.local_map_history.append(np.copy(state.local_map))
        self.borders_history.append(state.borders)
        self.team_in_range_history.append(copy.copy(state.team_in_range))
        self.ag_pos_history.append(state.ag_pos)
        self.ag_map_history.append(np.copy(state.ag_map))
        #self.n_scans_map_history.append(np.copy(state.n_scans_map))
        self.blind_v_map_history.append(np.copy(state.blind_v_map))
        self.blind_d_map_history.append(np.copy(state.blind_d_map))
        self.pog_history.append(np.copy(state.pog))
        self.ag_team_pos_history.append(copy.deepcopy(state.ag_team_pos))
        self.goal_history.append(state.goal)
        self.multi_goals_history.append(state.multi_goals)
        self.path_planner_history.append(state.path)
        self.expected_gain_history.append(state.expected_gain)
        if self.goal_history_not_repeted == [] or state.goal != self.goal_history_not_repeted[-1]: self.goal_history.append(state.goal)
        self.team_plans_history.append(state.team_plans)

        self.done_history.append(state.done)
        self.has_planned_history.append(state.has_planned)
        self.action_history.append(state.action)
        self.has_moved_history.append(state.has_moved)
        self.odom_history.append(state.odom)
        self.blind_v_history.append(state.blind_v)
        self.rh_history.append(state.robot_entropy)
        self.ph_history.append(state.path_entropy)
        self.mhc_history.append(state.map_entropy_uncertain)
        self.mhk_history.append(state.map_entropy_unknown)
        self.gh_history.append(state.global_entropy)
        self.neighbours_history.append(state.neighbours)
        self.data_history.append(state.data)

        self.self_loop_gt_history.append(state.self_loop_gt)
        self.self_loop_ag_history.append(state.self_loop_ag)
        self.self_loop_ref_observation_step_history.append(state.self_loop_ref_observation_step)
        self.self_loop_correction_history.append(state.self_loop_correction)

        self.ma_meta_loops_history.append(copy.deepcopy(state.ma_meta_loops))
        self.ma_semi_loops_history.append(copy.deepcopy(state.ma_semi_loops))
        self.ma_gt_loops_history.append(copy.deepcopy(state.ma_gt_loops_dic))
        self.ma_ag_loops_history.append(copy.deepcopy(state.ma_ag_loops_dic))
        self.ma_corrections_history.append(copy.deepcopy(state.ma_corrections))

        self.meeting_rpr_history.append(copy.deepcopy(state.meeting_rpr))
        self.meeting_batches_history.append(copy.deepcopy(state.meeting_batches))
        self.meeting_meta_loops_history.append(copy.deepcopy(state.meeting_meta_loops))
        self.metting_semi_loops_history.append(copy.deepcopy(state.meeting_semi_loops))
        self.meeting_corrections_history.append(copy.deepcopy(state.meeting_corrections))
        self.meeting_gt_loop_history.append(copy.deepcopy(state.meeting_gt_loops_dic))
        self.meeting_ag_loop_history.append(copy.deepcopy(state.meeting_ag_loops_dic))

        self.has_corrected_map_history.append(state.has_corrected_map)
        self.has_treated_holes_history.append(state.has_treated_holes)
        self.n_corr_inst_history.append(state.n_corr_inst)

        self.ag_loc_error_history.append(state.ag_loc_error)
        self.utility_history.append(state.utility)
        self.score_history.append(state.score)






class Meeting_Record:
    def __init__(self, self_id, time_step, neighbours, tracks, 
        meeting_rpr, meeting_batches, 
        meeting_meta_loops, meeting_semi_loops, meeting_corrections, 
        meeting_gt_loops_dic, meeting_ag_loops_dic
        ):
        
        self.self_id = self_id
        self.meeting_time_step = time_step
        self.neighbours = neighbours

        #self.tracks = copy.deepcopy(tracks)
        self.tracks = None
        self.meeting_rpr = copy.deepcopy(meeting_rpr)
        self.meeting_batches = copy.deepcopy(meeting_batches)
        self.meeting_corrections = copy.deepcopy(meeting_corrections)
        self.meeting_meta_loops = copy.deepcopy(meeting_meta_loops)
        self.meeting_semi_loops = copy.deepcopy(meeting_semi_loops)
        self.meeting_gt_loops_dic = copy.deepcopy(meeting_gt_loops_dic)
        self.meeting_ag_loops_dic = copy.deepcopy(meeting_ag_loops_dic)

class Evaluation_Record:
    def __init__(self, time_step, ag_id, submaps_eval, obs_eval):
        self.time_step = time_step
        self.ag_id = ag_id
        self.submaps_eval = copy.deepcopy(submaps_eval)
        self.obs_eval = copy.deepcopy(obs_eval)