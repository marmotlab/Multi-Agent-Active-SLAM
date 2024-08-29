import random as rd
import numpy as np

def distance_manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def cluster_v1(potential_target_dict, cluster_range, grid_shape):
    list_k = list(potential_target_dict.keys())
    rd.shuffle(list_k)

    #clusters
    clusters = {}
    for pos in list_k:
        curr_b = list(clusters.keys())
        rd.shuffle(curr_b)
        for posb in curr_b:
            if distance_manhattan(pos, posb) <= cluster_range:
                clusters[posb].append(pos)
                break
        else:
            clusters[pos] = [pos]

    #targets list
    targets = clusters.keys()

    #cost function
    cost_function = {}
    for target in targets:
        sum_t = 0
        for neighbour in clusters[target]:
            sum_t += potential_target_dict[neighbour]
        cost_function[target] = sum_t
    
    #grid
    grid = np.zeros(grid_shape)
    for pos in targets:
        grid[pos[0]][pos[1]] = cost_function[pos]

    return grid, targets, cost_function


#order potential_target_dict
def cluster_v2_targets(potential_target_dict, cluster_range):
    sorted_dict = sorted(potential_target_dict.items(), key=lambda x: x[1])
    sorted_targets = [k for k, v in sorted_dict]

    #clusters
    clusters = {}
    for pos in sorted_targets:
        curr_b = list(clusters.keys())
        rd.shuffle(curr_b)
        for posb in curr_b:
            if distance_manhattan(pos, posb) <= cluster_range:
                clusters[posb].append(pos)
                break
        else:
            clusters[pos] = [pos]

    return list(clusters.keys())

def cluster_v2_assign(targets_list, cm_dict, cluster_range):
    #cost function
    clusters = {target : [] for target in targets_list}
    cost_function = {target : 0 for target in targets_list}
    for point in cm_dict:
        best_target = None
        best_dist = 1e3
        for target in targets_list:
            dist = distance_manhattan(point, target)
            if dist < best_dist:
                best_target = target
                best_dist = dist
        if best_dist < cluster_range:
            clusters[best_target].append(point)
            cost_function[best_target]+=cm_dict[point]

    return clusters, cost_function

def cluster_v2_grid(clusters, cost_function, grid_shape):
    #grid
    grid = np.zeros(grid_shape)
    for target in clusters.keys():
        grid[target[0]][target[1]] = cost_function[target]
    return grid