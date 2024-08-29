from param import AgentParameters
from objects import Observation, Track, Trace
from components import SubMaps
import numpy as np
import random as rd
import statistics
import copy

from functions.generate_path import generate_path
from functions.a_star_v2 import a_star
from functions.entropy_functions import get_stdev_error

class VirtualAgent:
    def __init__(self, self_id, v_step, init_submaps, init_tracks, init_bv, init_h, disp = False) -> None:
        #print('New virtual agent : submaps :', init_submaps.ag_pos, '; tracks :',init_tracks, '; bv :', init_bv, '; h :', init_h)
        self.self_id = self_id

        #parameters
        self.range = AgentParameters.RANGE
        self.odom_error_rate = AgentParameters.ODOM_ERROR_RATE

        #variables
        self.v_step = v_step
        self.bv = init_bv
        self.h = copy.deepcopy(init_h)

        #data
        if 'root' not in init_tracks:
            #init data converting init_tracks to tracks_dic
            self.tracks = {
                'root' : init_tracks,
                'added' : {ag_id : Track(ag_id) for ag_id in init_tracks},
                'updates' : {ag_id : {} for ag_id in init_tracks},
                'traces' : {ag_id : Trace(ag_id) for ag_id in init_tracks},
                'meta_ext' : {},
            }
        else:
            #copy and deepcopy init_tracks
            self.tracks = {
                'root' : init_tracks['root'], #cannot be updated
                'added' : copy.deepcopy(init_tracks['added']), #can be updated
                'updates' : copy.deepcopy(init_tracks['updates']), #can be updated
                'traces' : {ag_id : Trace(ag_id) for ag_id in init_tracks['root']},
                'meta_ext' : {},
            }
        
        #rebuil init_traces
        for tr_id in self.tracks['traces']:
            self.rebuild_trace(tr_id)
        
        # self.submaps = copy.deepcopy(init_submaps)
        self.submaps = copy.copy(init_submaps)
        self.submaps.ag_map = copy.copy(init_submaps.ag_map)
        self.submaps.blind_table = copy.deepcopy(init_submaps.blind_table)


    def add_trace(self, ag_id, obs):
        if self.tracks['traces'][ag_id].ag_pos == [] or obs.pos_belief[-1]['pos_belief'] != self.tracks['traces'][ag_id].ag_pos[-1]:
            self.tracks['traces'][ag_id].time_steps.append(obs.time_step)
            self.tracks['traces'][ag_id].ag_pos.append(obs.pos_belief[-1]['pos_belief'])
            self.tracks['traces'][ag_id].blind_list.append([obs.pos_belief[idx]['blind_v'] for idx in range(len(obs.pos_belief))])
            if obs.time_step in self.tracks['updates'][ag_id]:
                self.tracks['traces'][ag_id].blind_list[-1][-1] = self.tracks['updates'][ag_id][obs.time_step]
            self.tracks['traces'][ag_id].blind_v.append(min(self.tracks['traces'][ag_id].blind_list[-1]))
    
    def rebuild_trace(self, ag_id):
        if self.tracks['root'][ag_id].obs_list == [] and self.tracks['added'][ag_id].obs_list == []:
            return None
        self.tracks['traces'][ag_id].reset()
        for obs in self.tracks['root'][ag_id].obs_list + self.tracks['added'][ag_id].obs_list:
            self.add_trace(ag_id, obs)

    def observe(self, ag_id, step, pos, bv):
        #no observation
        observation = Observation(ag_id, step, None, None, None, None, (0,0), pos, self.submaps.off_set, bv)
        self.tracks['added'][ag_id].obs_list.append(observation)
        self.tracks['added'][ag_id].trace.add_obs(observation)
        self.add_trace(ag_id, observation)
        self.submaps.fake_update_ag_map(observation, self.range*2 + 1)
    
    def moving_self(self, future_poses, disp = False):
        for i_f in range(len(future_poses)):
            self.v_step += 1
            self.submaps.ag_pos = future_poses[i_f]
            self.bv += 1
            self.observe(self.self_id, self.v_step, self.submaps.ag_pos, self.bv)
        if disp :
            print(' After moving self to', future_poses)
            print('tracks_root', self.tracks['root'][self.self_id].obs_list)
            print('tracks_added', self.tracks['added'][self.self_id].obs_list)
            print('tracks_updates', self.tracks['updates'][self.self_id])

    def moving_oth(self, oth_id, oth_step, oth_v_poses, fixed_bv, disp = False):
        for i_v in range(len(oth_v_poses)):
            oth_step += 1
            oth_pos = oth_v_poses[i_v]
            self.observe(oth_id, oth_step, oth_pos, fixed_bv)
        if disp :
            print(' After moving', oth_id, 'to', oth_v_poses)
            print('tracks_root', self.tracks['root'][oth_id].obs_list)
            print('tracks_added', self.tracks['added'][oth_id].obs_list)
            print('tracks_updates', self.tracks['updates'][oth_id])

    def set_obs_list_extensions(self, disp = False):
        for oth_id in self.tracks['root']:
            
            #reset ext variables
            self.tracks['meta_ext'][oth_id] = []

            for i_ow in range(len(self.tracks['root'][oth_id].owners)):
                current_owner = self.tracks['root'][oth_id].owners[i_ow]['owner']
                if self.tracks['root'][current_owner].obs_list != []:
                    start = max(self.tracks['root'][oth_id].owners[i_ow]['from step'], 1)
                    end1 = self.tracks['root'][current_owner].obs_list[-1].time_step
                    end2 = self.tracks['added'][current_owner].obs_list[-1].time_step if self.tracks['added'][current_owner].obs_list != [] else 0
                    end = max(end1, end2)

                    #add the current id obs list to the oth id ext list
                    self.tracks['meta_ext'][oth_id].append({'ag' : current_owner, 'intv' : [start, end]})

            if disp : print(" meta ext :", self.tracks['meta_ext'][oth_id])
            
    def get_semi_loops(self, tr_id, starting_step, disp = False):
        if disp : print("Getting semi loops of trace", tr_id, "since step", starting_step, "...")
        if disp : print(self.tracks['root'][tr_id].obs_list)
        
        
        #the loop starts from the last observation of the other agent (tr_id) and ends at the current observation
        meta_ext = self.tracks['meta_ext'][tr_id]
        if disp : print('meta_ext', meta_ext)

        #get the meta loop
        meta_loop = []
        for i_portion in range(len(meta_ext)):
            if meta_ext[i_portion]['intv'][1] >= starting_step:
                meta_loop.append({'ag' : meta_ext[i_portion]['ag'], 'intv' : [starting_step, meta_ext[i_portion]['intv'][1]]})
                break
        for j_portion in range(i_portion +1, len(meta_ext)):
            meta_loop.append((meta_ext[j_portion]))
        if disp : print('meta_loop', meta_loop)

        #complile the meta loop into semi loops
        semi_loops_dic = {}
        for portion in meta_loop:
            cur_id = portion['ag']
            window = portion['intv']
            semi_loops_dic[cur_id] = [window]
        if disp : print('semi_loops_dic', semi_loops_dic)
        if disp : print('end')
        return semi_loops_dic

    def correct_semi_loops(self, semi_loops_dic, disp = False):
        if disp : print("Correcting semi loops ...")

        #step 2 : correcting
        corrections = {}

        #correct blind_v in observations
        for ag_id in semi_loops_dic:
            corrections[ag_id] = {}
            for window in semi_loops_dic[ag_id]:                
                corrections[ag_id][tuple(window)] = True
                first_vstep = window[0]
                last_vstep = window[1]

                if first_vstep in self.tracks['updates'][ag_id]:
                    first_blind_v = self.tracks['updates'][ag_id][first_vstep]
                elif first_vstep <= self.tracks['root'][ag_id].obs_list[-1].time_step:
                    first_blind_v = self.tracks['root'][ag_id].obs_list[first_vstep-1].pos_belief[-1]['blind_v']
                else:
                    first_blind_v = self.tracks['added'][ag_id].obs_list[first_vstep-1-len(self.tracks['root'][ag_id].obs_list)].pos_belief[-1]['blind_v']

                for k_step in range(first_vstep, last_vstep +1):
                    self.tracks['updates'][ag_id][k_step] = first_blind_v

        if disp : print('corrections :', corrections)

        #correct the virtual agent blind_v
        if self.v_step in self.tracks['updates'][ag_id]:
            self.bv = self.tracks['updates'][ag_id][first_vstep]
        elif self.tracks['added'][ag_id].obs_list != []:
            self.bv = self.tracks['added'][ag_id].obs_list[-1].pos_belief[-1]['blind_v']
        else:
            self.bv = self.tracks['root'][ag_id].obs_list[-1].pos_belief[-1]['blind_v']

        #rebuild trace
        for ag_id in corrections:
            #self.tracks['root'][ag_id].rebuild_trace()
            self.rebuild_trace(ag_id)

        #rebuild map
        first_step_cor_list = []
        for _k, dic in corrections.items():
            for w, v in dic.items():
                if v : first_step_cor_list.append(w[0])

        start = min(first_step_cor_list)
        last_step = max(self.tracks['root'][self.self_id].obs_list[-1].time_step, self.tracks['added'][self.self_id].obs_list[-1].time_step)
        for step in range(start, last_step+1):
            for ag_id in list(reversed(self.tracks['root'].keys())):
                obs_step = step -1 if ag_id > self.self_id else step
                if obs_step <= 0:
                    continue
                if self.tracks['root'][ag_id].obs_list != [] and self.tracks['root'][ag_id].obs_list[-1].time_step > obs_step -1:
                    observation = self.tracks['root'][ag_id].obs_list[obs_step-1]
                    updated_blind_v = self.tracks['updates'][ag_id][step] if step in self.tracks['updates'][ag_id] else None
                    self.submaps.fake_update_ag_map(observation, 2*self.range +1, updated_blind_v)
                elif self.tracks['added'][ag_id].obs_list != [] and self.tracks['added'][ag_id].obs_list[-1].time_step > obs_step -1:
                    observation = self.tracks['added'][ag_id].obs_list[obs_step-1-len(self.tracks['root'][ag_id].obs_list)]
                    updated_blind_v = self.tracks['updates'][ag_id][step] if step in self.tracks['updates'][ag_id] else None
                    self.submaps.fake_update_ag_map(observation, 2*self.range +1, updated_blind_v)

    def update_after_looping(self, tr_id, loop_step, disp = False):
        self.set_obs_list_extensions(disp)
        semi_loops_dic = self.get_semi_loops(tr_id, loop_step, disp)
        self.correct_semi_loops(semi_loops_dic, disp)

    def get_ellipse(self, start_pos, end_pos, path_dist):
        #print('Ellipse function')
        if path_dist < 0:
            return None
        
        if self.submaps.manhattan_dist(start_pos, end_pos) > path_dist:
            maze = self.submaps.get_maze1()
            path, n_operation = a_star(maze, start_pos, end_pos)
            if path:
                return path[1:]
            else:
                return None
        
        ellipse = []
        for i in range(self.submaps.sub_height):
            for j in range(self.submaps.sub_width):
                point = (i,j)
                #print('point', point)
                dist = self.submaps.manhattan_dist(point, start_pos) + self.submaps.manhattan_dist(point, end_pos)
                #print('dist', dist, path_dist)
                if dist <= path_dist:
                   ellipse.append(point)
                   #print('append point', point)
        return ellipse
    
    def get_rd_points_in_ellipse(self, ellipse_list, n_points):
        if ellipse_list in [None, []] or n_points < 0:
            return None
        
        rd_list = []
        for _ in range(n_points):
            rd_point = rd.choice(ellipse_list)
            #print(rd_point)
            rd_list.append(rd_point)
        return rd_list
    
    def update_after_meeting(self, m_id, disp = False):
        if disp : print('VA Updating after potentially meeting agent', m_id, '...')
        
        #define last step contact and last pos contact, and get the last blind value hold by the other agent met (m_id)
        oth_pos = self.submaps.ag_team_pos[m_id]
        
        last_step_contact = max(oth_pos[observer]['time_step'] for observer in oth_pos)

        candidates = []
        for observer in oth_pos:
            if oth_pos[observer]['time_step'] == last_step_contact:
                candidates.append(observer)
        last_agent_contact = max(candidates)
        last_pos_contact = oth_pos[last_agent_contact]['seen_pos']

        #last_step_contact = self.tracks['root'][m_id].obs_list[-1].time_step
        #last_pos_contact = self.tracks['root'][m_id].obs_list[-1].pos_belief[-1]['pos_belief']
        
        last_bv_contact = self.tracks['root'][m_id].obs_list[-1].pos_belief[-1]['blind_v'] if self.tracks['root'][m_id].obs_list != [] else 0
        
        if disp : print('last_step_contact', last_step_contact)
        if disp : print('last_pos_contact', last_pos_contact)
        if disp : print('last_bv_contact', last_bv_contact)

        self.set_obs_list_extensions(disp) #get the meta_ext
        semi_loops_dic = self.get_semi_loops(m_id, last_step_contact, disp) #define past pos of the virtual agent (self, not m_id) using the meta_ext from last contact step

        #method 1
        #generate a path that m_id might have followed between last observation of meeting point
        #limits : this method does not work if the path if too long or if the meeting point is too far from the last step contact
        '''
        _multi_steps, m_v_path, _m_v_steps = generate_path(last_step_contact, last_pos_contact, self.v_step, self.submaps.ag_pos)
        if m_v_path : m_v_poses = m_v_path[1:]
        #'''

        #method 2
        #calculate the potential area covered by this path with an ellipse and take some random observation inside the ellipse (we don't need the virtual meeting agent to move correctly)
        if disp : print('current v agent pos', self.submaps.ag_pos)
        if disp : print('current v step', self.v_step)
        ellipse_list = self.get_ellipse(last_pos_contact, self.submaps.ag_pos, self.v_step - last_step_contact)       
        if disp : print('ellipse_list :', ellipse_list)
        m_v_poses = self.get_rd_points_in_ellipse(ellipse_list, self.v_step - last_step_contact)
        if disp : print('m_v_poses :', m_v_poses)

        if m_v_poses:
            #let the other met agent moving through these positions and add its observations to the tracks
            self.moving_oth(m_id, last_step_contact-(1*(last_agent_contact < m_id)), m_v_poses, last_bv_contact, disp = disp)

            #corrrecting bv, map and tracks after meeting
            self.correct_semi_loops(semi_loops_dic, disp = disp)
        else:
            print('error - v_path/v_poses could not be generated')
    
    def update_entropy(self, major_bv = None, disp = False):
        if disp : print('VA Updating entropy ...')

        #update blind_maps
        self.submaps.update_blind_maps(self.bv)

        #path entropy
        if disp : print('Updating path entropy ...')
        obs_entropy_list = []
        for tr_id in self.tracks['root']:
            blind_v_trace = self.tracks['traces'][tr_id].blind_v
            for i in range(len(blind_v_trace)):
                bv = blind_v_trace[i] if not major_bv else min(blind_v_trace[i], major_bv)
                oh = get_stdev_error(bv, self.odom_error_rate)
                obs_entropy_list.append(oh)
        self.h['ph'] = sum(obs_entropy_list)
        
        #map entropy
        if disp : print('Updating map entropy ...')
        self.submaps.update_map_entropy(disp)
        self.h['mhuk'] = round(self.submaps.map_entropy_unknown, 3)
        self.h['mhuc'] = round(self.submaps.map_entropy_uncertain, 3)
        
