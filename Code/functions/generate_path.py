#generate path
import random as rd
from collections import deque
from itertools import permutations
import heapq
import math
import itertools

from functions.a_star_v2 import a_star

rd.seed(0)

def get_distance_manhattan(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def get_total_distance(multi_steps, max_edge_length):
    distance = 0
    for i in range(len(multi_steps) - 1):
        edge_length = get_distance_manhattan(multi_steps[i], multi_steps[i+1])
        if edge_length > max_edge_length:
            return False
        distance += edge_length
    return distance

def short_path(start, goal):
    sub_path = []
    moves = [(1,0), (0,1), (-1,0), (0,-1)]

    current_pos = start
    while current_pos != goal:
        current_dist = get_distance_manhattan(current_pos, goal)
        rd.shuffle(moves)
        for move in moves:
            next_pos = (current_pos[0]+move[0], current_pos[1]+move[1])
            if get_distance_manhattan(next_pos, goal) < current_dist:
                sub_path.append(current_pos)
                current_pos = next_pos
                break

    sub_path.append(current_pos)
    return sub_path

def set_path(multi_steps):
    path = []
    for i in range(len(multi_steps)-1):
        sub_path = short_path(multi_steps[i], multi_steps[i+1])
        path = path + sub_path[:-1]
    path.append(multi_steps[-1])
    return path

def generate_rd_points(n_rd_points, sample_range, start_pos, end_pos):
    rd_points = []
    for _ in range(n_rd_points):
        ref_point = rd.choice(rd_points+[start_pos, end_pos])
        rx_var = rd.randint(-sample_range, sample_range)
        ry_var = rd.randint(-sample_range, sample_range)
        new_point = (ref_point[0] + rx_var, ref_point[1] + ry_var)
        rd_points.append(new_point)
    return sorted(list(set(rd_points)))

def generate_multi_steps(start_pos, end_pos, min_steps, max_steps, max_sample_range = 5, max_edge_length = 10, n_rd_points = 20):
    #check if we need intermediate points
    heur_distance = get_distance_manhattan(start_pos, end_pos)
    if heur_distance >= min_steps:
        return [start_pos, end_pos], 0
    
    factor = max_steps/heur_distance
    print('factor', factor)
    print('max_steps', max_steps)
    print('heur_distance', heur_distance)
    if factor < 3:
        sample_range = min(int(heur_distance/3), max_sample_range)
    else:
        sample_range = min(int(heur_distance/2), max_sample_range)
    
    #generate random points
    rd_points = generate_rd_points(n_rd_points, sample_range, start_pos, end_pos)

    k=0
    for n_inter_points in [1, 2, 3]:
        if (n_inter_points+1) * max_edge_length < min_steps:
            continue

        #set permutations
        print('getting permutations ...')
        permutations = list(itertools.permutations(rd_points, n_inter_points))
        print('done')

        print('n_inter_points :', n_inter_points)
        print('rd_points :', rd_points)
        print('number of permutations :', len(permutations))
        rd.shuffle(permutations)
        for perm in permutations:
            k+=1         
            #set path with intermediate points (random order)
            multi_steps = [start_pos] + list(perm) + [end_pos]
            print('multi_steps', multi_steps)

            #calculate total distance
            total_distance = get_total_distance(multi_steps, max_edge_length)
            print('total_distance', total_distance)
            if total_distance != False and total_distance <= max_steps and total_distance >= min_steps:
                return multi_steps, k
            if n_inter_points == 1 and total_distance > max_steps:
                rd_points.remove(list(perm)[0])
            
        n_inter_points += 1

    print("No path has been found")
    return None, k

def generate_path(start_step, start_pos, end_step, end_pos, min_steps_percentage = 0.7):
    print('Generating path from step', start_step, 'to step', end_step, 'from pos', start_pos, 'to pos', end_pos, '...')
    min_steps = int((end_step - start_step) * min_steps_percentage)
    max_steps = end_step - start_step

    #check if path exist
    heur_distance = get_distance_manhattan(start_pos, end_pos)
    if heur_distance > max_steps:
        print("Path impossible to set")
        return None, None, None
    
    #set multi_steps
    multi_steps, k = generate_multi_steps(start_pos, end_pos, min_steps, max_steps, max_sample_range = 5, max_edge_length = 10, n_rd_points = 20)
    print(k, 'operations')
    print(multi_steps)
    if not multi_steps:
        return None, None, None
    
    #set path
    path = set_path(multi_steps)

    n_missing_step = max_steps - (len(path)-1)
    for _ in range(n_missing_step):
        idx = rd.randint(0, len(path) -1)
        path.insert(idx, path[idx])
    
    #set steps
    steps = [idx+start_step for idx in range(len(path))]
    return multi_steps, path, steps


if __name__ == '__main__':
    start_step = 10
    end_step = 50 #difference not above 50
    start_pos = (0,0)
    end_pos = (17,17) #not further than 40

    multi_steps, path, steps = generate_path(start_step, start_pos, end_step, end_pos)
    print('multi_steps :', multi_steps)
    print('path :', path)
    print('steps :', steps)