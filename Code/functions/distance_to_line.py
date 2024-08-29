#Function that takes in input a map with lines and returned a map with the distances to frontier
import numpy as np

def is_line(line_map):
    return bool(np.amax(line_map))

def get_dist_to_line_map(line_map, max_dist = 10):
    print(line_map.shape)
    map_height, map_width = line_map.shape
    #initialisation
    dist_map = np.zeros((map_height, map_width))

    #create a list of positions to consider
    pos_list = []

    #initialise dist_map
    for i in range(map_height):
        for j in range(map_width):

            #check if the pose is not part of the frontier
            if line_map[i,j] == 0:
                dist_map[i,j] = max_dist
                pos_list.append((i,j))

    r = 0

    #check if there is a frontier
    if not is_line(line_map):
        return dist_map

    #else, iterate
    while pos_list != []:
        pos_to_remove = []
        for pos in pos_list:
            i=pos[0]
            j=pos[1]

            #check if the pose has at least one neighbour part of the previous frontier distance
            for k in [+1, -1]:
                try:
                    if abs(dist_map[i+k,j]) == r and i+k >= 0 and i+k < map_height:
                        dist_map[i,j] = (r+1)
                except IndexError:
                    pass

                try:
                    if abs(dist_map[i,j+k]) == r and j+k >= 0 and j+k < map_width:
                        dist_map[i,j] = (r+1)
                except IndexError:
                    pass
            
            if dist_map[i,j] == r+1:
                pos_to_remove.append(pos)

        for pos in pos_to_remove:
            pos_list.remove(pos)

        r += 1
        if r>=max_dist:
            break

    return dist_map


if __name__ == '__main__':
    line_map = np.array([
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0]
    ])
    max_dist = 10
    dist_map = get_dist_to_line_map(line_map, max_dist = max_dist)
    print(dist_map)
