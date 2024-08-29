import numpy as np
import sys
import time
np.set_printoptions(threshold=sys.maxsize)

from functions.get_pos_distrib_mc import *
from functions.get_pos_distrib_mc_build_memo import *
from param import AgentParameters, CostmapsParameters

def manhattan_dist(pos1, pos2):
    return abs(pos2[0] - pos1[0]) + abs(pos2[1] - pos1[1])


#build memo for pos presence distrib
def build_agent_distrib_mc_presence_memo(
        submaps, last_pos_contact,
        ag_plans, lost_steps = CostmapsParameters.LOST_STEPS,
        disp = False, measure_time = True):
    if disp : print(" Building agent's distribution presence memo (method 3 Monte Carlo) ...")

    #init memo
    mc_memo = {}

    #set center_list
    center_list = []
    center_list.append(last_pos_contact)
    if True : print(' pos', last_pos_contact)

    if ag_plans and ag_plans['path']:
        if True : print(' path', ag_plans['path'])
        #select i_path
        #randomly select k steps in path
        max_len_selection = max(min(10, int(len(ag_plans['path'])/3)), min(int(len(ag_plans['path'])/2), 5)) #10 and 3 arbitrarily selected
        selection_i_path = rd.sample(range(len(ag_plans['path'])), k=max_len_selection) #calculate k path steps maximum (randomly selected)
        if disp : print(' selection_i_path', selection_i_path)
        center_list.extend([ag_plans['path'][i_path] for i_path in selection_i_path])

    if ag_plans and ag_plans['multi_goals'] and ag_plans['multi_goals'] != True:
        if True : print('mg', ag_plans['multi_goals'])
        center_list.extend(ag_plans['multi_goals'])
    
    center_list = list(set(center_list))
    if disp : print(' center_list', center_list)
    
    #build memo
    for center in center_list:
        if True : print("  building pos distrib memo for center", center,"...")
        center_memo = get_pos_distrib_mc_build_memo(submaps.get_maze1(), center, lost_steps)
        mc_memo[center] = center_memo

    return mc_memo


#get presence pos distrib 2 takes in input ag plans to better estimate its probability distribution of presence
def get_agent_distrib_mc_presence(
        submaps, self_id, m_id, curr_step,
        last_step_contact, last_pos_contact, last_agent_contact, ag_plans, 
        trust_plan_factor = CostmapsParameters.TRUST_PLAN_FACTOR, 
        lost_steps = CostmapsParameters.LOST_STEPS,
        mc_memo = None, disp = False, measure_time = True):
    if True : print("Getting Agent", m_id, "distribution presence (method 3 Monte Carlo) ...")

    exec_time_dic = {
        'part1': [],
        'part2a': [],
        'part2b': [],
        'part3': [],
        'ftime': [],
        'n_steps':[],
        'n_parts':[],
        'selection':[],
    }

    if mc_memo:
        m_distrib_memo = mc_memo[m_id]

    #init
    distrib_dict = {}
    init_step = last_step_contact + (-1 * (m_id > last_agent_contact))
    last_moving_step = curr_step + (-1 * (m_id > self_id))


    #1st : agent distribution based on the last time seen
    if measure_time : tacp1 = time.time()
    distrib_step = init_step
    n_steps_ago = last_moving_step - distrib_step
    
    if n_steps_ago <= lost_steps:
        center = last_pos_contact
        if mc_memo and center in m_distrib_memo and n_steps_ago in m_distrib_memo[center]:
            distrib = m_distrib_memo[center][n_steps_ago]
        else:
            if measure_time : tacf = time.time()
            distrib, n_steps, n_parts = get_pos_distrib_mc(submaps.get_maze1(), center, n_steps_ago)
            if measure_time : exec_time_dic['ftime'].append(time.time() - tacf)
            if measure_time : exec_time_dic['n_steps'].append(n_steps)
            if measure_time : exec_time_dic['n_parts'].append(n_parts)

        if disp : print(' Agent', m_id, 'has been seen by Agent', last_agent_contact, 'at step', last_step_contact, '- ( init-last_move =', init_step, last_moving_step, ') -', n_steps_ago, 'steps ago, at pos', last_pos_contact)
    
        distrib_dict[distrib_step] = {
            'center' : center,
            'n_steps_ago' : n_steps_ago,
            'distrib' : distrib,
        }
        if measure_time : exec_time_dic['part1'].append(time.time() - tacp1)
        if disp : print('The init distrib centered in ', center,' is :\n', (distrib_dict[distrib_step]['distrib']*100).astype(int), 'max =', np.max(distrib_dict[distrib_step]['distrib']))
    else:
        distrib = np.zeros_like(submaps.ag_map)
        if disp : print(' Agent', m_id, 'lost')


    #2nd : agent's  distribution based on its plans
    if ag_plans and ag_plans['path']:
        if measure_time : tacp2a = time.time()
        if disp : print(' agent has planned path :', ag_plans['path'])
        
        #select i_path
        #pre select
        pre_selection_i_path = []
        for i_path in range(len((ag_plans['path']))):
            distrib_step = init_step + i_path + 1
            n_steps_ago = last_moving_step - distrib_step

            if distrib_step <= last_moving_step and n_steps_ago <= lost_steps:
                pre_selection_i_path.append(i_path)
            else:
                break
        if disp : print('pre_selection_i_path :', pre_selection_i_path)
        
        #set min and max len (arbitrary)
        max_len_selection = max(min(10, int(len(pre_selection_i_path)/3)), min(int(len(pre_selection_i_path)/2), 5)) #10 and 3 arbitrarily selected
        min_len_selection = min(int(len(pre_selection_i_path)/3), 5)

        #set centers in memo
        if mc_memo:
            pre_selection_in_memo = []
            pre_selection_not_in_memo = []
            for i_path in pre_selection_i_path:
                center = ag_plans['path'][i_path]
                if center in m_distrib_memo:
                    pre_selection_in_memo.append(i_path)
                else:
                    pre_selection_not_in_memo.append(i_path)
            
            if len(pre_selection_in_memo) >= min_len_selection:
                selection_i_path = pre_selection_in_memo
            else:
                selection_i_path = pre_selection_in_memo+rd.sample(pre_selection_not_in_memo, k=min_len_selection - len(pre_selection_in_memo))
        
        else:
            #set n_selected
            selection_i_path = rd.sample(pre_selection_i_path, k=max_len_selection) #calculate 5 path steps maximum (randomly selected)

        #add the last one or at least the current one
        last_i_path = len((ag_plans['path'])) -1
        current_i_path = last_moving_step - init_step -1 #such that n_step ago = 0 <=> i_path +1 = last_moving_step - init_step

        if init_step + last_i_path +1 <= last_moving_step and last_moving_step - init_step + last_i_path +1 <= lost_steps:
            if len((ag_plans['path'])) -1 not in selection_i_path:
                selection_i_path.append(last_i_path)
                if disp : print('last added')
        elif current_i_path in range(len(ag_plans['path'])):
            if current_i_path not in selection_i_path:
                selection_i_path.append(current_i_path)
                if disp : print('current added')
        selection_i_path.sort()
        if disp : print('selection_i_path :', selection_i_path)
        if measure_time : exec_time_dic['selection'].append(selection_i_path)


        #set distributions
        for i_path in selection_i_path:
            distrib_step = init_step + i_path + 1
            n_steps_ago = last_moving_step - distrib_step
            center = ag_plans['path'][i_path]
            if mc_memo and center in m_distrib_memo and n_steps_ago in m_distrib_memo[center]:
                distrib = m_distrib_memo[center][n_steps_ago]
            else:
                tacf = time.time()
                if True : print(' ... recalculating pos distrib for center', center, '...')
                distrib, n_steps, n_parts = get_pos_distrib_mc(submaps.get_maze1(), center, n_steps_ago)
                if measure_time : exec_time_dic['ftime'].append(time.time() - tacf)
                if measure_time : exec_time_dic['n_steps'].append(n_steps)
                if measure_time : exec_time_dic['n_parts'].append(n_parts)

            if disp : print(' According to its plans, Agent', m_id, 'should be at step', distrib_step, '-', n_steps_ago, 'steps ago, at pos', center)

            distrib_dict[distrib_step] = {
                'n_steps_ago' : n_steps_ago,
                'center' : center,
                'distrib' : distrib,
            }
            if disp : print('The init distrib centered in ', center,' is :\n', (distrib_dict[distrib_step]['distrib']*100).astype(int), 'max =', np.max(distrib_dict[distrib_step]['distrib']))
            
        if measure_time : exec_time_dic['part2a'].append(time.time() - tacp2a)

    
    #2bis : multi_goal planning
    if ag_plans and ag_plans['multi_goals'] and ag_plans['multi_goals'] != True:
        if measure_time : tacp2b = time.time()
        if disp : print(' agent has planned multi_goals :', ag_plans['multi_goals'])
        distrib_step = init_step + len(ag_plans['path'])
        if disp : print(init_step + len(ag_plans['path']))

        for i_goal in range(1, len(ag_plans['multi_goals'])):
            distrib_step += manhattan_dist(ag_plans['multi_goals'][i_goal], ag_plans['multi_goals'][i_goal-1])
            n_steps_ago = last_moving_step - distrib_step
            
            if distrib_step <= last_moving_step and n_steps_ago <= lost_steps:
                center = ag_plans['multi_goals'][i_goal]
                if mc_memo and center in m_distrib_memo and n_steps_ago in m_distrib_memo[center]:
                    distrib = m_distrib_memo[center][n_steps_ago]
                else:
                    tacf = time.time()
                    distrib, n_steps, n_parts = get_pos_distrib_mc(submaps.get_maze1(), center, n_steps_ago)
                    if measure_time : exec_time_dic['ftime'].append(time.time() - tacf)
                    if measure_time : exec_time_dic['n_steps'].append(n_steps)
                    if measure_time : exec_time_dic['n_parts'].append(n_parts)
                
                if disp : print(' According to its plans, Agent', m_id, 'should be at step', distrib_step, '-', n_steps_ago, 'steps ago at pos', last_pos_contact)
                
                distrib_dict[distrib_step] = {
                    'n_steps_ago' : n_steps_ago,
                    'center' : center,
                    'distrib' : distrib,
                }
            else:
                break
        if measure_time : exec_time_dic['part2b'].append(time.time() - tacp2b)

    
    #final calculation
    if disp : print('Final Calculation :')
    if measure_time : tacp3 = time.time()
    if ag_plans == None:
        presence_map = distrib_dict[init_step]['distrib']
    else:
        presence_map = np.zeros_like(submaps.ag_map)
        for d_step in distrib_dict:
            distrib = distrib_dict[d_step]['distrib']
            k = int(d_step - init_step)
            if disp : print('   d_step :', d_step, ' ; k =', k)
            if d_step == max(distrib_dict):
                presence_map += distrib * trust_plan_factor**k
            else:
                next_d_step = min([key for key in distrib_dict if key > d_step])
                l = int(next_d_step - init_step)
                #presence_map += distrib * ((1-trust_plan_factor) * trust_plan_factor**k)
                presence_map += distrib * (trust_plan_factor**k-trust_plan_factor**l)
        
    if disp : print('Finally the presence_map is :\n', (presence_map*100).astype(int), 'max =', np.max(presence_map))
    if measure_time : exec_time_dic['part3'].append(time.time() - tacp3)


    #check if presence map is null
    if np.max(presence_map) < 1e-5:
        print('warning - presence map nearly null - max =', np.max(presence_map))
        return False, exec_time_dic

    return presence_map, exec_time_dic



#2nd : supress 
#get information where the other agent might not be : suppress score where the horde have not seen the other agent
def get_agent_distrib_mc_supress(
        submaps, tracks, curr_step, 
        last_step_contact, last_agent_contact, 
        tracks_added = None,
        impact_steps = CostmapsParameters.IMPACT_STEPS, 
        scan_range = AgentParameters.RANGE, 
        disp = False, measure_time = True):
    if disp : print("Getting other agent's pos distribution, and meeting probability with method 3 Monte Carlo ...")

    exec_time_dic = {
        'part4': [],
        'part5': [],
        'ftime': [],
        'n_steps':[],
        'n_parts':[],
        'selection':[],
    }

    no_presence_map = np.ones_like(submaps.ag_map)
    if disp : print(" No presence ...")
    if measure_time : tacp4 = time.time()

    for ag_id in tracks:
        if tracks[ag_id].obs_list != []:
            if disp : print(" horde agent : ag_id :", ag_id)
            if disp : print(' last_step_contact with the m_id :', last_step_contact)
            if disp : print(' last_agent_contact with the m_id :', last_agent_contact)

            first_data_after_m_id_disparition = last_step_contact + 1 * (ag_id <= last_agent_contact)
            #last_data_from_ag_id = tracks[ag_id].obs_list[-1].time_step
            last_data_from_ag_id = tracks_added[ag_id].obs_list[-1].time_step if tracks_added and tracks_added[ag_id].obs_list != [] else tracks[ag_id].obs_list[-1].time_step #new version
            if disp : print(' first_data_after_m_id_disparition :', first_data_after_m_id_disparition)
            if disp : print(' last_data_from_ag_id :', last_data_from_ag_id)

            if last_data_from_ag_id >= first_data_after_m_id_disparition: #check if ag_id has data after m_id disparition
                if disp : print(' we can suppress')
                #check each observation
                for i_obs_step in range(first_data_after_m_id_disparition, last_data_from_ag_id +1):
                    #obs = tracks[ag_id].obs_list[i_obs_step-1]
                    obs = tracks[ag_id].obs_list[i_obs_step-1] if i_obs_step <= tracks[ag_id].obs_list[-1].time_step else tracks_added[ag_id].obs_list[i_obs_step-1-len(tracks[ag_id].obs_list)] #new version
                    ag_pos = obs.pos_belief[-1]['pos_belief']
                    impact = min(curr_step - i_obs_step, impact_steps) / impact_steps
                    #the impact represents the probability to trust the absence of m_id in an area after a certain time 
                    if impact < 1:
                        i, j = ag_pos[0], ag_pos[1]
                        for k in range(max(0, i - scan_range), min(i +  scan_range +1, submaps.sub_height)):
                            for l in range(max(0, j - scan_range), min(j +  scan_range +1, submaps.sub_width)):
                                no_presence_map[k, l] = min(no_presence_map[k, l], impact)
                        
                        #if disp : print('ag_id :', ag_id)
                        #if disp : print('i_obs_step :', i_obs_step)
                        #if disp : print('ag_pos :', ag_pos)
                        #if disp : print('impact :', impact)
                        #if disp : print('intermediate no presence_map :\n', (no_presence_map*100).astype(int))
    if disp : print('no presence_map :\n', (no_presence_map*100).astype(int))
    if measure_time : exec_time_dic['part4'].append(time.time() - tacp4)
    return no_presence_map, exec_time_dic



#3rd compil
#multiply both information : presence by no presence
def get_agent_distrib_mc_compil(
        submaps, presence_map, no_presence_map,
        disp = False, rnd = None, measure_time = True):
    if disp : print("Getting other agent's pos distribution, and meeting probability with method 3 Monte Carlo ...")

    exec_time_dic = {
        'part1': [],
        'part2a': [],
        'part2b': [],
        'part3': [],
        'part4': [],
        'part5': [],
        'ftime': [],
        'n_steps':[],
        'n_parts':[],
        'selection':[],
    }
 
    #multiply both information : presence by no presence
    presence_distribution_map = presence_map * no_presence_map
    if disp : print('distribution_map :\n', (presence_distribution_map*100).astype(int))

    #retrieve out and obstacles
    if measure_time : tacp5 = time.time()
    for i in range(submaps.sub_height):
        for j in range(submaps.sub_width):
            pos = (i,j)
            if submaps.is_out(pos) or submaps.is_obstacle(pos):
                presence_distribution_map[i,j] = 0

    #normalise the distribution
    if np.max(presence_distribution_map) < 1e-5:
        print("warning : distribution null after calculation")
        return False, exec_time_dic
    else:
        presence_distribution_map = presence_distribution_map /(np.sum(presence_distribution_map))

    #round
    if rnd:
        presence_distribution_map = np.round(presence_distribution_map, rnd)
    if disp : print('distribution_map after normalisation and round :\n', (presence_distribution_map*100).astype(int))
    
    if measure_time : exec_time_dic['part5'].append(time.time() - tacp5)
    return presence_distribution_map, exec_time_dic



def get_meet_distrib(submaps, presence_distribution_map, scan_range = AgentParameters.RANGE, disp = False):
    #sum with kernel to get the distribution (space probability to meet the other agent)
    meet_distribution = np.zeros_like(submaps.ag_map)
    for i in range(submaps.sub_height):
        for j in range(submaps.sub_width):
            for k in range(max(i-scan_range, 0), min(i+scan_range +1, submaps.sub_height)):
                for l in range(max(j-scan_range, 0), min(j+scan_range +1, submaps.sub_width)):
                    meet_distribution[i,j] += presence_distribution_map[k,l]
    if disp : print('meet distribution :\n', (meet_distribution*100).astype(int))
    return meet_distribution