#Function takes in input maze and start, and returns all the points connected to start
"""
Find all points connected to the starting point in a 2D grid.

Parameters:
grid (list of list of int): The 2D map where 0 represents a free space and 1 represents an obstacle.
start (tuple of int): The starting point (x, y).

Returns:
set of tuple of int: A set of all connected points.
"""

#function used with array or list of list (table)
def is_out(maze, node_position):
    return node_position[0] >= len(maze) or node_position[0] < 0 or node_position[1] >= len(maze[0]) or node_position[1] < 0
    
def is_free(maze, node_position):
    return maze[node_position[0]][node_position[1]] == 0

def is_empty(maze, node_position):
    return maze[node_position[0]][node_position[1]] == 1

def is_unknown(maze, node_position):
    return maze[node_position[0]][node_position[1]] < 0

def heuristic(pos1, pos2): #Manhattan
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


#main function #from Chat GPT
def connected_points(grid, start, max_range = None):
    rows, cols = len(grid), len(grid[0])
    connected_points = []
    to_visit = [start]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    if max_range == None:
        max_range = 1e3

    while to_visit:
        pos = to_visit.pop()
        if pos in connected_points:
            continue
        
        connected_points.append(pos)

        for dx, dy in directions:
            node_position = (pos[0] + dx, pos[1] + dy)
            if node_position not in connected_points and not is_out(grid, node_position) and heuristic(start, node_position) <= max_range and (is_free(grid, node_position) or is_unknown(grid, node_position)):
                to_visit.append(node_position)

    return connected_points

if __name__ == '__main__':
    import numpy as np
    # Example usage
    grid = np.array([
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    start = (0, 0)
    connected_points = connected_points(grid, start,5)
    print(connected_points)