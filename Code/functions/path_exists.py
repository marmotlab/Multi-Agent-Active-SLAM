#from chat GPT
import numpy as np

def path_exists(arr, start, end):
    rows = len(arr)
    cols = len(arr[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    queue = [start]
    visited[start[0]][start[1]] = True
    k = 0
    while queue:
        k += 1
        current_point = queue.pop(0)
        row, col = current_point[0], current_point[1]

        if current_point == end:
            print("number of loops to get the solution with path_exists :", k)
            return True

        for row_offset, col_offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_row, new_col = row + row_offset, col + col_offset
            if (0 <= new_row < rows) and (0 <= new_col < cols) and (not visited[new_row][new_col]) and (arr[new_row][new_col] != 1):
                queue.append((new_row, new_col))
                visited[new_row][new_col] = True
    return False

def test():

    maze = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    maze2 = np.array(
            [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 0]]
    )

    maze3 = np.array(
            [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]
    )

    start = (0, 0)
    end = (5, 5)
    connected = path_exists(maze2, start, end)

    print(maze2)
    print(connected)

if __name__ == '__main__':
    test()