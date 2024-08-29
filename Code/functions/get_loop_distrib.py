import numpy as np
import random as rd
import time
import gc 
gc.disable()

from functions.get_pos_distrib import *

def get_point_distrib(submaps, out_n_obs_list, i_pos, point_pos, point_stdev, point_step, self_trace_poses_distrib_list, self_trace, loop_w_self = False, disp = False):
    #loop pos, the position of the point
    
    if disp : print(" Getting loop pos distrib :")
    #if disp : print("  point_pos :", point_pos)
    #if disp : print("  point_stdev :", point_stdev)
    #if disp : print("  point_step :", point_step)

    #get the first loop pos distrib
    toctoc = time.time()
    if loop_w_self:
        first_point_distrib = self_trace_poses_distrib_list[i_pos]
    else:
        first_point_distrib = get_pos_distrib(submaps, point_pos, point_stdev) #sum is 1
    if disp : print("  step 1 : first_point_distrib :\n", (first_point_distrib*100).astype(int)/100)
    exec_time_toctoc1a = time.time() - toctoc


    #suppress obstacles and renormalize
    toctoc = time.time()
    # for pos in out_n_obs_list:
    #     first_point_distrib[pos[0],pos[1]] = 0
    exec_time_toctoc1b = time.time() - toctoc

    if np.sum(first_point_distrib) < 1e-2:
        return first_point_distrib, 0, exec_time_toctoc1a, exec_time_toctoc1b, 0, 0, 0
    
    toctoc = time.time()
    first_point_distrib = first_point_distrib / np.sum(first_point_distrib)
    exec_time_toctoc1c = time.time() - toctoc
    if disp : print("  step 2 : obstacles removed :\n", (first_point_distrib*100).astype(int)/100)


    #use the self trace to eliminate squares where the loop pos is not
    toctoc = time.time()

    #if disp : print("  trace distribution ...")
    #init
    self_trace_distribution = np.zeros_like(submaps.ag_map, dtype=float)
    j_operations = 0

    #loop over the self trace, only considering poses visited after the pos step (+3 times the standard deviation)
    for j_pos in range(len(self_trace.time_steps)):
        if disp : print(j_pos)
        #if self_trace.time_steps[j_pos] > point_step: #method 1
        if (self_trace.time_steps[j_pos] > point_step + int(3*point_stdev)) or not loop_w_self: #method upgraded
            
            self_pos = self_trace.ag_pos[j_pos]

            #bug occurs when the self pos is out of bounds
            if not submaps.on_submap(self_pos):
                if disp : print(" error - self pos is out")

            #only consider self poses that interfere with the loop pos distrib
            elif first_point_distrib[self_pos[0], self_pos[1]] > 0:
                #if disp : print("in")
                #add the self pos distrib to the self trace distrib
                self_trace_distribution = np.maximum(self_trace_distribution, self_trace_poses_distrib_list[j_pos])
                j_operations += 1
    exec_time_toctoc2 = time.time() - toctoc
    
    if disp : print("  step 3 : self_trace_distribution :\n", (self_trace_distribution*100).astype(int)/100)

    #decresing the loop pos distrib by the self pos distribution (using sqrt function)
    toctoc = time.time()
    point_distrib = np.maximum(first_point_distrib - self_trace_distribution, np.zeros_like(submaps.ag_map, dtype=float))
    if disp : print("  step 4 : point_distrib after removing :\n", (point_distrib*1000).astype(int)/1000)

    #renormalise or cancel
    if disp : print("  step 5 (renormalize) : sum array =", np.sum(point_distrib)) #sum array should be between 0 and 1
    if np.sum(point_distrib) > 0.3:
        norm_point_distrib = point_distrib / np.sum(point_distrib)
        return norm_point_distrib, j_operations, exec_time_toctoc1a, exec_time_toctoc1b, exec_time_toctoc1c, exec_time_toctoc2, time.time() - toctoc
    elif np.sum(point_distrib) > 0.1:
        if disp : print(" pos distrib not renormalized because some uncertainty")
        return point_distrib, j_operations, exec_time_toctoc1a, exec_time_toctoc1b, exec_time_toctoc1c, exec_time_toctoc2, time.time() - toctoc
    else: #too much uncertainy because more than 90% of the distrib was removed
        if disp : print(" pos distrib refused because too much uncertainty")
        return False, j_operations, exec_time_toctoc1a, exec_time_toctoc1b, exec_time_toctoc1c, exec_time_toctoc2, time.time() - toctoc
    
    



def get_loop_trace_u_x_distrib(submaps, ag_trace, ag_trace_stdev, self_trace, self_trace_stdev, u_trace, disp = False, rnd = None):
    #It takes in input a trace with poses (centers), a trace with the stdev around centers and a trace with utility. It takes also a self trace with stdev.
    #It returns a map with gaussians spread around each center.
    if disp : print("Getting trace distrib and utility :")
    
    #init
    loop_trace_u_x_distrib = np.zeros_like(submaps.ag_map, dtype=float)
    loop_trace_u_eff = []
    i_operations = 0
    n_operations = 0
    j_operations_list = []
    f_operations = 0

    exec_time_dic = {
        'part1':[],
        'part1toc':[],
        'part2':[],
        'part2toc1':[],
        'part2toc1toc1a':[],
        'part2toc1toc1b':[],
        'part2toc1toc1c':[],
        'part2toc1toc2':[],
        'part2toc1toc3':[],
        'part2toc2':[],
        'part3':[],
    }

    loop_w_self = ag_trace == self_trace

    #get self loop poses distrib prior
    self_trace_poses_distrib_list = []
    for i_self_pos in range(len(self_trace.ag_pos)):
        tac = time.time()
        self_trace_pos = self_trace.ag_pos[i_self_pos]
        self_pos_stdev = self_trace_stdev[i_self_pos]
        if disp : print(" Loop over self trace : i_pos =", i_self_pos, "; pos :", self_trace_pos, "; stdev :", self_pos_stdev)
        toc = time.time()
        self_point_distrib = get_pos_distrib(submaps, self_trace_pos, self_pos_stdev)
        exec_time_dic['part1toc'].append(time.time() - toc)

        self_trace_poses_distrib_list.append(self_point_distrib)
        exec_time_dic['part1'].append(time.time() - tac)

    #get out and obstacles list
    out_n_obs_list = submaps.get_out_n_obs_list()

    #iterate : loop over each element of the loop (ag loop)
    for i_pos in range(len(ag_trace.ag_pos)):
        tac = time.time()

        #get the loop pos distribution using center, stdev and self trace
        toc2 = time.time()
        point_pos = ag_trace.ag_pos[i_pos]
        pos_stdev = ag_trace_stdev[i_pos]
        pos_step = ag_trace.time_steps[i_pos]
        
        if disp : print(" Loop over agent trace : i_pos =", i_pos, "; pos :", point_pos, "; stdev :", pos_stdev)

        point_distrib, j_operations, exec_time_toctoc1a, exec_time_toctoc1b, exec_time_toctoc1c, exec_time_toctoc2, exec_time_toctoc3 = get_point_distrib(submaps, out_n_obs_list, i_pos, point_pos, pos_stdev, pos_step, self_trace_poses_distrib_list, self_trace, loop_w_self, disp)
        
        n_operations += j_operations
        i_operations += 1
        j_operations_list.append(j_operations)
        
        exec_time_dic['part2toc1toc1a'].append(exec_time_toctoc1a)
        exec_time_dic['part2toc1toc1b'].append(exec_time_toctoc1b)
        exec_time_dic['part2toc1toc1c'].append(exec_time_toctoc1c)
        exec_time_dic['part2toc1toc2'].append(exec_time_toctoc2)
        exec_time_dic['part2toc1toc3'].append(exec_time_toctoc3)
        exec_time_dic['part2toc1'].append(time.time() - toc2)

        if disp :
            if type(point_distrib) == bool:
                print(" point_distrib for pos ", point_pos,":", point_distrib)
            else:
                print("  point_distrib :\n", (point_distrib*1000).astype(int)/1000)

        toc2 = time.time()
        #multiply the distribution by the utility
        utility = u_trace[i_pos]
        if disp : print("  utility :", utility)

        #add the pos x u distribution to the trace one
        if type(point_distrib) == np.ndarray and np.max(point_distrib) > 1e-3:
            pos_u_x_distrib = utility * point_distrib #model not used
            pos_u_x_distrib = utility * point_distrib / np.max(point_distrib) * np.sum(point_distrib)
            loop_trace_u_eff.append(round(utility * np.sum(point_distrib),0))
            #loop_trace_u_eff.append(int(np.sum(pos_u_x_distrib))) #not used

            #add this distribution around the pos to the one concerning the whole trace
            #loop_trace_u_x_distrib = loop_trace_u_x_distrib + pos_u_x_distrib #adding all
            loop_trace_u_x_distrib = np.maximum(loop_trace_u_x_distrib, pos_u_x_distrib) #only keeping the maximum point => maximum utility at the pos with maximum probability over the distribution
        else :
            loop_trace_u_eff.append(0)
            f_operations += 1
            if disp : print(" error - loop pos not found (distribution null)")
        exec_time_dic['part2toc2'].append(time.time() - toc2)

        exec_time_dic['part2'].append(time.time() - tac)

    #round loop trace_distrib
    if rnd != None:
        loop_trace_u_x_distrib = np.round(loop_trace_u_x_distrib, rnd)

    return loop_trace_u_x_distrib, loop_trace_u_eff, (i_operations, n_operations, j_operations_list, f_operations, exec_time_dic)





if __name__ == '__main__':

    from objects import Trace
    from components import SubMaps
    gc.disable()

    print("Start!")
    max_map_height = 30
    max_map_width = 30
    ext_map_extension = 2
    init_pos = (0,0)
    submaps = SubMaps(init_pos)

    ag_trace = Trace(ag_id = 1)

    step = 16
    from objects import Trace
    ag_trace = Trace(ag_id = 1)
    ag_trace.ag_pos = [(29, 16), (29, 15), (28, 15), (27, 15), (27, 14), (27, 13), (27, 12), (27, 13), (27, 14), (27, 15), (27, 16), (27, 17), (27, 18), (27, 19), (27, 20), (28, 20), (29, 20), (29, 19), (30, 19), (30, 18), (30, 17), (30, 16), (31, 16), (31, 15), (31, 14), (31, 13), (31, 12), (30, 12), (29, 12), (29, 11), (28, 11), (27, 11), (26, 11), (25, 11), (24, 11), (23, 11), (22, 11), (21, 11), (21, 12), (20, 12), (20, 13), (20, 14), (20, 15), (19, 15), (18, 15), (18, 14), (17, 14), (17, 13), (16, 13), (16, 12), (17, 12), (17, 11), (18, 11), (18, 10), (18, 9), (18, 8), (18, 9), (18, 10), (18, 11), (18, 12), (17, 12), (17, 13), (16, 13), (16, 14), (17, 14), (18, 14), (18, 15), (19, 15), (19, 16), (19, 17), (19, 18), (18, 18), (17, 18), (16, 18), (16, 19), (16, 20), (15, 20), (14, 20), (13, 20), (14, 20), (15, 20), (16, 20), (17, 20), (17, 21), (18, 21), (19, 21), (20, 21), (21, 21), (22, 22), (21, 22), (22, 22), (23, 22), (24, 22), (25, 22), (26, 22), (26, 23), (27, 23), (26, 23), (27, 23), (27, 24), (27, 25), (26, 25), (25, 25), (25, 26), (25, 27), (25, 28), (24, 28), (24, 27), (23, 27), (22, 27), (21, 27), (20, 27), (19, 27), (19, 26), (18, 26), (18, 28), (17, 28), (17, 29), (16, 29), (15, 29), (14, 29), (14, 28), (13, 28), (11, 28), (10, 28), (10, 27), (10, 26), (9, 26), (9, 27), (8, 27), (8, 28), (8, 29), (8, 28), (9, 28), (9, 27), (9, 26), (9, 25), (9, 24), (8, 24), (8, 23), (8, 22), (8, 21), (9, 21), (9, 20), (8, 20), (8, 19), (9, 19), (10, 19), (11, 19), (11, 18), (11, 17), (11, 16), (11, 15), (11, 14), (11, 13), (12, 13), (12, 12), (12, 11), (12, 10), (13, 10), (13, 9), (13, 8), (13, 7), (14, 7), (15, 7)]
    print(len(ag_trace.ag_pos))

    #ag_trace.ag_pos = [(ag_trace.ag_pos[i][0] +20, ag_trace.ag_pos[i][1] +20) for i in range(len(ag_trace.ag_pos))]
    ag_trace.time_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166, 168, 169, 170, 172, 174, 175, 176, 177, 178]
    print(len(ag_trace.time_steps))

    #u_trace = [1 for i in range(len(ag_trace.ag_pos))]
    ag_trace_stdev = [4.399, 4.382, 4.365, 4.347, 4.33, 4.313, 4.295, 4.313, 4.33, 4.347, 4.33, 4.313, 4.295, 4.278, 4.26, 4.243, 4.207, 4.189, 4.171, 4.153, 4.135, 4.117, 4.099, 4.08, 4.062, 4.044, 4.025, 4.006, 3.987, 3.969, 3.95, 3.912, 3.892, 3.873, 3.854, 3.834, 3.814, 3.775, 3.755, 3.735, 3.715, 3.695, 3.674, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.654, 3.633, 3.612, 3.592, 3.571, 3.55, 3.528, 3.507, 3.486, 3.464, 3.464, 3.464, 3.464, 3.464, 3.486, 3.464, 3.442, 3.421, 3.376, 3.354, 3.332, 3.309, 3.286, 3.309, 3.286, 3.263, 3.217, 3.194, 3.17, 3.17, 3.17, 3.17, 3.146, 3.122, 3.098, 3.074, 3.05, 3.0, 2.975, 2.95, 2.924, 2.898, 2.872, 2.846, 2.793, 2.766, 2.739, 2.711, 2.655, 2.627, 2.598, 2.569, 2.54, 2.51, 2.48, 2.449, 2.419, 2.387, 2.356, 2.324, 2.291, 2.258, 2.225, 2.191, 2.156, 2.191, 2.156, 2.258, 2.291, 2.258, 2.225, 2.191, 2.156, 2.121, 2.086, 2.049, 2.012, 1.975, 1.936, 1.857, 1.817, 1.775, 1.732, 1.688, 1.597, 1.549, 1.5, 1.449, 1.396, 1.285, 1.225, 1.162, 1.025, 0.866, 0.775, 0.671, 0.548, 0.387]

    
    print(len(ag_trace_stdev))
    self_trace = ag_trace
    self_trace_stdev = ag_trace_stdev

    u_trace = [130, 129, 128, 127, 127, 127, 127, 127, 127, 127, 126, 125, 124, 123, 122, 121, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 103, 102, 101, 100, 99, 98, 96, 95, 94, 93, 92, 91, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 89, 88, 87, 86, 85, 84, 83, 82, 82, 82, 82, 82, 82, 82, 81, 80, 79, 77, 76, 75, 74, 74, 74, 73, 72, 70, 69, 68, 68, 68, 68, 67, 66, 65, 64, 63, 61, 60, 59, 58, 57, 56, 55, 53, 52, 51, 50, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 36, 36, 36, 36, 36, 36, 36, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 24, 23, 22, 21, 20, 18, 17, 16, 15, 14, 12, 11, 10, 8, 6, 5, 4, 3, 1]
    
    disp = False
    rnd = 2

    tic = time.time()
    loop_trace_u_x_distrib, loop_trace_u_eff, (i_operations, n_operations, j_operations_list, f_operations, exec_time_dic) = get_loop_trace_u_x_distrib(
        submaps, ag_trace, ag_trace_stdev, self_trace, self_trace_stdev, u_trace, disp, rnd)
    exec_time_g = round(time.time() - tic, 3)

    loop_cost_map = loop_trace_u_x_distrib.astype(int)
    print("loop trace u x distrib : \n", loop_trace_u_x_distrib)
    print("loop_cost_map :\n", loop_cost_map)

    print("u_trace : \n", u_trace)
    print("loop_trace_u_eff : \n", loop_trace_u_eff)

    print(" sum u =", sum(u_trace))
    print(" sum u_x_distrib = ", np.sum(loop_trace_u_x_distrib))
    print(" sum loop_cost_map = ", np.sum(loop_cost_map))

    print('len trace :', len(ag_trace.ag_pos))
    print("n_operations :", n_operations)
    print("i_operations :", i_operations)
    print("j_operations_list :", j_operations_list)
    print("max_j_operations :", max(j_operations_list))
    print("f_operations :", f_operations)
    print("exec time :", exec_time_g)

    for part in exec_time_dic:
        print(part, sum(exec_time_dic[part]))
        print(exec_time_dic[part])
        #print(len(exec_time_dic[part]))