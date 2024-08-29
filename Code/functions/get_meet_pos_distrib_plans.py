import numpy as np

from param import AgentParameters, CostmapsParameters
from functions.get_pos_distrib import *
def get_stdev_agent_diffusion(n_steps_ago):
    return n_steps_ago/3

def manhattan_dist(pos1, pos2):
    return abs(pos2[0] - pos1[0]) + abs(pos2[1] - pos1[1])

#get presence pos distrib 2 takes in input ag plans to better estimate its probability distribution of presence
def get_agent_distrib_plans(
        submaps, tracks, self_id, m_id, curr_step, 
        last_step_contact, last_pos_contact, last_agent_contact, ag_plans, 
        tracks_added = None,
        trust_plan_factor = CostmapsParameters.TRUST_PLAN_FACTOR, 
        lost_steps = CostmapsParameters.LOST_STEPS, impact_steps = CostmapsParameters.IMPACT_STEPS, 
        scan_range = AgentParameters.RANGE, 
        disp = False, rnd = None):
    if disp : print("Getting other agent's pos distribution, and meeting probability with method 2 ...")

    #init
    distrib_dict = {}
    init_step = last_step_contact + (-1 * (m_id > last_agent_contact))
    last_moving_step = curr_step + (-1 * (m_id > self_id))

    #1st : agent distribution based on the last time seen
    distrib_step = init_step
    n_steps_ago = last_moving_step - distrib_step

    if n_steps_ago <= lost_steps:
        center = last_pos_contact
        stdev = get_stdev_agent_diffusion(n_steps_ago) #depends on the model
        distrib = get_pos_distrib(submaps, center, stdev)
        if disp : print(' Agent', m_id, 'has been seen by Agent', last_agent_contact, 'at step', last_step_contact, '- ( init-last_move =', init_step, last_moving_step, ') -', n_steps_ago, 'steps ago ( stdev =', stdev,'), at pos', last_pos_contact)
    
        distrib_dict[distrib_step] = {
            'n_steps_ago' : n_steps_ago,
            'stdev' : stdev,
            'center' : center,
            'distrib' : distrib,
        }
        if disp : print('The init distrib centered in ', center,' is :\n', (distrib_dict[distrib_step]['distrib']*100).astype(int))
    else:
        distrib = np.zeros_like(submaps.ag_map)
        if disp : print(' Agent', m_id, 'lost')

    #2nd : agent's  distribution based on its plans
    if ag_plans['path']:
        if disp : print(' agent has planned path :', ag_plans['path'])
        for i_path in range(len(ag_plans['path'])):
            distrib_step = init_step + i_path + 1
            n_steps_ago = last_moving_step - distrib_step
            
            if distrib_step <= last_moving_step and n_steps_ago <= lost_steps:
                center = ag_plans['path'][i_path]
                stdev = get_stdev_agent_diffusion(n_steps_ago) #depends on the model
                distrib = get_pos_distrib(submaps, center, stdev)    

                if disp : print(' According to its plans, agent', m_id, 'should be at step', distrib_step, '-', n_steps_ago, 'steps ago ( stdev =', stdev,'), at pos', center)

                distrib_dict[distrib_step] = {
                    'n_steps_ago' : n_steps_ago,
                    'stdev' : stdev,
                    'center' : center,
                    'distrib' : distrib,
                }

                if disp : print('The init distrib centered in ', center,' is :\n', (distrib_dict[distrib_step]['distrib']*100).astype(int))
            
            else:
                break
    
    #2bis : multi_goal planning
    if ag_plans['multi_goals'] and ag_plans['multi_goals'] != True:
        if disp : print(' agent has planned multi_goals :', ag_plans['multi_goals'])
        distrib_step = init_step + len(ag_plans['path'])
        if disp : print(init_step + len(ag_plans['path']))

        for i_goal in range(1, len(ag_plans['multi_goals'])):
            distrib_step += manhattan_dist(ag_plans['multi_goals'][i_goal], ag_plans['multi_goals'][i_goal-1])
            n_steps_ago = last_moving_step - distrib_step
            
            if distrib_step <= last_moving_step and n_steps_ago <= lost_steps:
                center = ag_plans['multi_goals'][i_goal]
                stdev = get_stdev_agent_diffusion(n_steps_ago) #depends on the model
                distrib = get_pos_distrib(submaps, center, stdev)

                if disp : print(' According to its plans, agent', m_id, 'should be at step', distrib_step, '-', n_steps_ago, 'steps ago ( stdev =', stdev,'), at pos', last_pos_contact)
                
                distrib_dict[distrib_step] = {
                    'n_steps_ago' : n_steps_ago,
                    'stdev' : stdev,
                    'center' : center,
                    'distrib' : distrib,
                }
            else:
                break
    
    #final calculation
    if disp : print('Final Calculation :')
    presence_map = np.zeros_like(submaps.ag_map)
    for d_step in distrib_dict:
        distrib = distrib_dict[d_step]['distrib']
        k = int(d_step - init_step)
        if disp : print('     d_step :', d_step, ' ; k =', k)
        if d_step == max(distrib_dict):
            presence_map += distrib * trust_plan_factor**k
        else:
            next_d_step = min([key for key in distrib_dict if key > d_step])
            l = int(next_d_step - init_step)
            #presence_map += distrib * ((1-trust_plan_factor) * trust_plan_factor**k)
            presence_map += distrib * (trust_plan_factor**k-trust_plan_factor**l)
    
    if disp : print('Finally the presence_map is :\n', (presence_map*100).astype(int))


    #check if presence map is null
    if np.max(presence_map) < 1e-5:
        print('warning - presence map nearly null')
        return False


    #2nd : supress 
    #init
    no_presence_map = np.ones_like(submaps.ag_map)
    if disp : print(" No presence ...")

    #get information where the other agent might not be : suppress score where the horde have not seen the other agent
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
    
    #multiply both information : presence by no presence
    presence_distribution_map = presence_map * no_presence_map
    if disp : print('distribution_map :\n', (presence_distribution_map*100).astype(int))

    #retrieve out and obstacles
    for i in range(submaps.sub_height):
        for j in range(submaps.sub_width):
            pos = (i,j)
            if submaps.is_out(pos) or submaps.is_obstacle(pos):
                presence_distribution_map[i,j] = 0

    #normalise the distribution
    if np.max(presence_distribution_map) < 1e-5:
        print("warning : distribution null after calculation")
        return False
    else:
        presence_distribution_map = presence_distribution_map /(np.sum(presence_distribution_map))

    #round
    if rnd:
        presence_distribution_map = np.round(presence_distribution_map, rnd)
    if disp : print('distribution_map after normalisation and round :\n', (presence_distribution_map*100).astype(int))

    return presence_distribution_map



def get_meet_pos_distrib(submaps, presence_distribution_map, scan_range = AgentParameters.RANGE, disp = False):
    #sum with kernel to get the distribution (space probability to meet the other agent)
    meet_distribution = np.zeros_like(submaps.ag_map)
    for i in range(submaps.sub_height):
        for j in range(submaps.sub_width):
            for k in range(max(i-scan_range, 0), min(i+scan_range +1, submaps.sub_height)):
                for l in range(max(j-scan_range, 0), min(j+scan_range +1, submaps.sub_width)):
                    meet_distribution[i,j] += presence_distribution_map[k,l]
    if disp : print('meet distribution :\n', (meet_distribution*100).astype(int))
    return meet_distribution