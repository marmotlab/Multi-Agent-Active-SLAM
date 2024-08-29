import numpy as np
import random as rd
from components import SubMaps


def get_pos_distrib(submaps, center, stdev, k_influence = 3):
    #create discrete normal distribution around the center
    
    #init
    pos_distrib = np.zeros_like(submaps.ag_map, dtype=float)

    if not submaps.on_submap(center): #special case
        return pos_distrib

    if stdev == 0: #special case
        pos_distrib[center[0], center[1]] = 1
        return pos_distrib

    max_influence_dist = int(stdev*k_influence) #this 1.5 factor is crucial and experimentally determined --> cf study : discrete diffusion
    #max_influence_dist = 1
    
    #look over neighboring elements in a square centered on the current element
    for k in range(max(0, center[0]-max_influence_dist), min(submaps.sub_height, center[0]+max_influence_dist+1)):
        for l in range(max(0, center[1]-max_influence_dist), min(submaps.sub_width, center[1]+max_influence_dist+1)):
            if submaps.on_submap((k,l)):
                #calculate the distance between the current element and its neighbor
                #dist = np.sqrt((pos[0]-k)**2 + (pos[1]-l)**2) #spatial distance
                dist = abs(center[0]-k) + abs(center[1]-l) #orthogonal distance
                if dist <= max_influence_dist:
                    pos_distrib[k,l] = np.exp(-0.5*(dist/stdev)**2)
                    #pos_distrib[k,l] = 1/(dist+1) #linear distribution              
    
    #normalise the pos distrib
    if np.sum(pos_distrib) > 1e-3:
        pos_distrib = pos_distrib / np.sum(pos_distrib)
    return pos_distrib


def get_pos_distrib_2(stdev, k_influence = 3):
    max_influence_dist = int(stdev*k_influence)
    pos_distrib = np.zeros((2*max_influence_dist+1, 2*max_influence_dist+1))
    center = (max_influence_dist, max_influence_dist)
    
    #look over neighboring elements in a square centered on the current element
    for k in range(0, 2*max_influence_dist+2):
        for l in range(0, 2*max_influence_dist+2):
            #calculate the distance between the current element and its neighbor
            #dist = np.sqrt((pos[0]-k)**2 + (pos[1]-l)**2) #spatial distance
            dist = abs(center[0]-k) + abs(center[1]-l) #orthogonal distance
            if dist <= max_influence_dist:
                pos_distrib[k,l] = np.exp(-(dist**2)/(2*((stdev + 1e-3)**2)))
    
    #normalise the pos distrib
    if np.sum(pos_distrib) > 1e-3:
        pos_distrib = pos_distrib / np.sum(pos_distrib)
    return pos_distrib


if __name__ == "__main__":
    print("Start!")
    max_map_height = 30
    max_map_width = 30
    ext_map_extension = 3
    init_pos = (rd.randrange(0, max_map_height),rd.randrange(0, max_map_width))

    submaps = SubMaps(init_pos, max_map_height, max_map_width, ext_map_extension)

    center = (5,5)
    stdev = 3

    distrib = get_pos_distrib(submaps, center, stdev)
    print("distrib :\n", (distrib*10000).astype(int)/100)
    print("sum : ", np.sum(distrib))

