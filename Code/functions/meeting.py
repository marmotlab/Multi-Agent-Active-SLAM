def get_prob_meeting(submaps, path, presence_distrib, scan_range):
    seen_pos_list = []
    for i_step in range(len(path)):
        ag_pos = path[i_step]
        for k in range(max(ag_pos[0]-scan_range, 0), min(ag_pos[0]+scan_range +1, submaps.sub_height)):
            for l in range(max(ag_pos[1]-scan_range, 0), min(ag_pos[1]+scan_range +1, submaps.sub_width)):
                seen_pos = (k,l)
                seen_pos_list.append(seen_pos)
    seen_pos_list = list(set(seen_pos_list))
    
    prob_meeting = 0
    for s_pos in seen_pos_list:
        prob_meeting += presence_distrib[s_pos[0],s_pos[1]]
    return prob_meeting