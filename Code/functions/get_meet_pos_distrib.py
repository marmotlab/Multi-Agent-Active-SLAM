import numpy as np

from param import AgentParameters, CostmapsParameters
from functions.get_pos_distrib import *
def get_stdev_agent_diffusion(n_steps_ago):
    return n_steps_ago/3

def get_agent_distrib(
        submaps, tracks, self_id, oth_id, curr_step, 
        last_step_contact, last_pos_contact, last_agent_contact, 
        tracks_added = None, 
        lost_steps = CostmapsParameters.LOST_STEPS, impact_steps = CostmapsParameters.IMPACT_STEPS, 
        scan_range = AgentParameters.RANGE, 
        disp = False, rnd = None):
    if disp : print("Getting other agent's pos distribution, and meeting probability ...")
    if disp : print(" oth id :", oth_id)

    #add
    #init
    presence_map = np.zeros_like(submaps.ag_map)

    #add score where the agent might be according to the last time seen
    if disp : print(" Presence ...")

    init_step = last_step_contact + (-1 * (oth_id > last_agent_contact))
    last_moving_step = curr_step + (-1 * (oth_id > self_id))
    n_steps_ago = last_moving_step - init_step

    if n_steps_ago <= lost_steps:
        center = last_pos_contact
        stdev = get_stdev_agent_diffusion(n_steps_ago) #depends on the model
        distrib = get_pos_distrib(submaps, center, stdev)
        
        presence_map = np.maximum(presence_map, distrib)
        
        if disp : print(' Agent', oth_id, 'has been seen by Agent', last_agent_contact, 'at step', last_step_contact, '- ( init-last_move =', init_step, last_moving_step, ') -', n_steps_ago, 'steps ago ( stdev =', stdev,'), at pos', last_pos_contact)

        #if disp : print(' intermediate distrib :\n', (distrib*100).astype(int))
    if disp : print('presence_map :\n', (presence_map*100).astype(int))

    #check if presence map is null
    if np.max(presence_map) < 1e-5:
        print('warning - presence map nearly null')
        return False

    #supress
    #init
    no_presence_map = np.ones_like(submaps.ag_map)
    if disp : print(" No presence ...")

    #get information where the other agent might not be : suppress score where the horde have not seen the other agent
    for ag_id in tracks:
        if tracks[ag_id].obs_list != []:
            if disp : print(" horde agent : ag_id :", ag_id)
            if disp : print(' last_step_contact with the oth_id :', last_step_contact)
            if disp : print(' last_agent_contact with the oth_id :', last_agent_contact)

            first_data_after_oth_id_disparition = last_step_contact + 1 * (ag_id <= last_agent_contact)
            #last_data_from_ag_id = tracks[ag_id].obs_list[-1].time_step
            last_data_from_ag_id = tracks_added[ag_id].obs_list[-1].time_step if tracks_added and tracks_added[ag_id].obs_list != [] else tracks[ag_id].obs_list[-1].time_step #new version
            if disp : print(' first_data_after_oth_id_disparition :', first_data_after_oth_id_disparition)
            if disp : print(' last_data_from_ag_id :', last_data_from_ag_id)

            if last_data_from_ag_id >= first_data_after_oth_id_disparition: #check if ag_id has data after oth_id disparition
                if disp : print(' we can suppress')
                #check each observation
                for i_obs_step in range(first_data_after_oth_id_disparition, last_data_from_ag_id +1):
                    #obs = tracks[ag_id].obs_list[i_obs_step-1]
                    obs = tracks[ag_id].obs_list[i_obs_step-1] if i_obs_step <= tracks[ag_id].obs_list[-1].time_step else tracks_added[ag_id].obs_list[i_obs_step-1-len(tracks[ag_id].obs_list)] #new version
                    ag_pos = obs.pos_belief[-1]['pos_belief']
                    impact = min(curr_step - i_obs_step, impact_steps) / impact_steps
                    #the impact represents the probability to trust the absence of oth_id in an area after a certain time 
                    if impact < 1:
                        i, j = ag_pos[0], ag_pos[1]
                        for k in range(max(0, i - scan_range), min(i +  scan_range +1, submaps.sub_height)):
                            for l in range(max(0, j - scan_range), min(j +  scan_range +1, submaps.sub_width)):
                                no_presence_map[k, l] = min(no_presence_map[k, l], impact)
                        
                        #if disp : print('ag_id :', ag_id)
                        #if disp : print('i_obs_step :', i_obs_step). 
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