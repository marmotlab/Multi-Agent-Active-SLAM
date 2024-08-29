import numpy as np
import random as rd
import math
from shapely.geometry import LineString
import time

def barycenter_list(pos_list):
    pos_list_0 = []
    pos_list_1 = []
    for pos in pos_list:
        pos_list_0.append(pos[0])
        pos_list_1.append(pos[1])
    return (sum(pos_list_0)/len(pos_list_0), sum(pos_list_1)/len(pos_list_1))

def center(pos1, pos2):
    return ((pos1[0]+pos2[0])/2, (pos1[1]+pos2[1])/2)

def center_list(pos_list):
    return center(pos_list[0], pos_list[-1])

def length(pos_list):
    length = 0
    for i in range(len(pos_list)-1):
        length += man_dist(pos_list[i], pos_list[i+1])
    return length

def radius(pos1, pos2):
    return bird_dist(pos1, pos2)/2

def radius_list(pos_list):
    return radius(pos_list[0], pos_list[-1])

def bird_dist(pos1, pos2):
    return math.sqrt(((pos1[0] - pos2[0]) ** 2) + ((pos1[1] - pos2[1]) ** 2))

def man_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def segments_intersect(p1, q1, p2, q2):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False

def segments_intersect_2(p1, q1, p2, q2):
    segment1 = [p1, p2]
    segment2 = [q1, q2]

    line1 = LineString(segment1)
    line2 = LineString(segment2)

    return line1.intersects(line2) and line1.crosses(line2)

def is_intersecting(path1, path2, n_operations = 0):
    for k in range(len(path1)-1):
        for l in range(len(path2)-1):
            p1 = path1[k]
            p2 = path1[k+1]
            q1 = path2[l]
            q2 = path2[l+1]
            #if bird_dist(center(p1,p2), center(q1,q2)) <= radius(p1,p2)+radius(q1,q2):
            if man_dist(center(p1,p2), center(q1,q2)) <= 2:
                #method1
                n_operations += 1
                if segments_intersect(p1, p2, q1, q2):
                    return True, n_operations
    return False, n_operations


def bird_dist_check(path, segment_pos, segment_stdev): #check if the two paths are close enough to intersect
    return bird_dist(center(path[0], path[-1]), center(segment_pos[0], segment_pos[-1])) <= length(path)/2 + length(segment_pos)/2 + 3*segment_stdev #else less than 5% chances for intersection

def man_dist_check(path, segment_pos, segment_stdev): #check if the two paths are close enough to intersect
    return man_dist(center(path[0], path[-1]), center(segment_pos[0], segment_pos[-1])) <= length(path)/2 + length(segment_pos)/2 + 1*segment_stdev #else less than 5% chances for intersection


def get_prob_intersecting(path, segment_pos, segment_stdev = 0, deviations = None, num_simulations = 1):
    n_operations = 0  
    intersect_count = 0

    tic = time.time()
    if deviations is None or len(deviations) < num_simulations:
        deviations = np.random.multivariate_normal(mean=(0,0), cov=segment_stdev * np.identity(n=2), size = num_simulations)
        #print('deviation', round(time.time() - tic, 4))

    for s in range(num_simulations):
        tuc = time.time()
        segment_deviated = [(round(segment_pos[i][0]+deviations[s][0]), round(segment_pos[i][1]+deviations[s][1])) for i in range(len(segment_pos))]
        
        #print('path :', path)
        #print('segment_deviated :', segment_deviated)
        is_inter, n_operations = is_intersecting(path, segment_deviated, n_operations)
        if is_inter:
            intersect_count += 1
        #print(round(time.time()-tuc, 4))

    probability = intersect_count / num_simulations
    
    #print('prob to intersect found : prob =', probability, '; exec time =',  round(time.time() - tic, 4), '; n_operations =', n_operations)
    return probability

if __name__ == '__main__':
    path = [(24, 17), (25, 17), (26, 17), (27, 17), (28, 17), (29, 17)]
    #path = [(22, 21), (22, 20), (22, 19), (22, 18), (22, 17), (23, 17)]
    segment_pos = [(29, 16), (29, 17), (30, 17), (30, 18)]
    #segment_pos = [(24, 22), (23, 21), (22, 21)]
    
    segment_stdev = 1.8973665961010275

    # blind_dist_list = [26, 25, 24, 23, 22]
    # import statistics
    # import math
    # blind_dist_mean = statistics.mean(blind_dist_list)
    # segment_stdev = math.sqrt(blind_dist_mean*0.15)
    # print('seg stdev :', segment_stdev)

    num_simulations = 100

    import time
    #deviations_list = None
    for i in range(10):
        deviations = np.random.multivariate_normal(mean=(0,0), cov=segment_stdev * np.identity(n=2), size = num_simulations)
        loop_prob = get_prob_intersecting(path, segment_pos, segment_stdev, deviations, num_simulations = num_simulations)
        print('prob to intersect found : prob =', loop_prob)