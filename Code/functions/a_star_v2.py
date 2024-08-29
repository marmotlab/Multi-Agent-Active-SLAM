#version edited and used by the program
import heapq
import time
import numpy as np

class Node:    
    def __init__(self, x, y, cost, heuristic, parent=None):
        self.parent = parent
        
        self.x = x
        self.y = y

        self.cost = cost
        self.heuristic = heuristic
        self.total_cost = cost + heuristic

    def __lt__(self, other):
        return self.total_cost < other.total_cost

#function used with array or list of list (table)
def is_out(maze, node_position):
    return node_position[0] >= len(maze) or node_position[0] < 0 or node_position[1] >= len(maze[0]) or node_position[1] < 0
    
def is_free(maze, node_position):
    return maze[node_position[0]][node_position[1]] == 0

def is_empty(maze, node_position):
    return maze[node_position[0]][node_position[1]] == 1

def is_unknown(maze, node_position):
    return maze[node_position[0]][node_position[1]] < 0

def heuristic(node_pos, goal_pos): #Manhattan
    return abs(node_pos[0] - goal_pos[0]) + abs(node_pos[1] - goal_pos[1])

def a_star(grid, start, goal):
    #create start and end node
    start_node = Node(start[0], start[1], 0, heuristic(start, goal))
    goal_node = Node(goal[0], goal[1], 0, 0)

    #init open list and the closed set
    open_list = []
    heapq.heappush(open_list, start_node)
    
    closed_set = set()

    #init number of operation
    n_operation = 0

    #loop until you find the end node
    while open_list and n_operation < 1e4: #1e5 also works
        n_operation += 1
        current_node = heapq.heappop(open_list)

        #... is the current node the goal ?
        if (current_node.x, current_node.y) == (goal_node.x, goal_node.y):
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            #print("a* worked and n_operation is", n_operation)
            return path[::-1], n_operation
        
        #...else
        closed_set.add((current_node.x, current_node.y))

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            node_pos = (current_node.x + dx, current_node.y + dy)
            
            #make sure node is within range and in walkable terrain
            if is_out(grid, node_pos):
                continue
            if is_empty(grid, node_pos):
                continue
            
            #make sure node is not in the closed set
            if node_pos in closed_set:
                continue

            #... then create a new node and add child
            neighbor = Node(node_pos[0], node_pos[1], current_node.cost + 1, heuristic((node_pos[0], node_pos[1]), goal), current_node)
            heapq.heappush(open_list, neighbor)

    #print("a* failed and n_operation is", n_operation)
    return False, n_operation  # No path found