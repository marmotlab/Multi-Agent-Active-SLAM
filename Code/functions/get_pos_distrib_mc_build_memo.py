import numpy as np
import random as rd
import copy
import statistics

def get_pos_distrib_mc_build_memo(maze, center, max_n_steps, dist_char_low = 1, dist_char_high = 10, wall_cost = 2, kernel_size = 2):

    #init memo
    particules_pos_memo = {}

    #init run
    n_particules = int(min(int((max_n_steps-1)/10 +1) * 5e2, 2e3))
    particules_pos = {part : (center[0], center[1]) for part in range(n_particules)}
    particules_dir = {part : ((0,0), 0) for part in range(n_particules)}
    particules_energy = {part : 0 for part in range(n_particules)}
    particules_path = {part : [] for part in range(n_particules)}
    particules_sleep = {part : 0 for part in range(n_particules)}

    #save pos at 0
    particules_pos_memo[0] = copy.copy(particules_pos)

    moves = [(0,1), (0,-1), (1,0), (-1,0), (0,0)]

    size1, size2 = maze.shape

    def is_in(pos):
        return 0<=pos[0]<=size1-1 and 0<=pos[1]<=size2-1

    def is_free(maze, pos):
        return maze[pos[0], pos[1]] == 0
    
    #version1
    def run_move():
        for i_step in range(1, max_n_steps+1):
            for part in particules_pos:
                corr_moves = []
                for move in moves:
                    future_pos = (particules_pos[part][0] + move[0], particules_pos[part][1] + move[1])
                    if is_in(future_pos) and is_free(maze, future_pos):
                        corr_moves.append(move)

                next_move = rd.choice(corr_moves)
                particules_pos[part] = (particules_pos[part][0] + next_move[0], particules_pos[part][1] + next_move[1])

            #save pos at i_step
            particules_pos_memo[i_step] = copy.copy(particules_pos)
            
    #version 2
    def run_dir():
        for i_step in range(1, max_n_steps+1):
            for part in particules_pos:
                
                #record
                particules_path[part].append(particules_pos[part])

                if particules_energy[part] > 0:
                    move = particules_dir[part]
                    future_pos = (particules_pos[part][0] + move[0], particules_pos[part][1] + move[1])
                    if is_in(future_pos) and is_free(maze, future_pos):
                        particules_pos[part] = future_pos
                        particules_energy[part] = particules_energy[part] - 1
                        continue
                    else:
                        particules_energy[part] = 0
                        continue

                #else set a move
                corr_moves = []
                for move in moves:
                    future_pos = (particules_pos[part][0] + move[0], particules_pos[part][1] + move[1])
                    if is_in(future_pos) and is_free(maze, future_pos):
                        corr_moves.append(move)

                next_move = rd.choice(corr_moves)
                particules_pos[part] = (particules_pos[part][0] + next_move[0], particules_pos[part][1] + next_move[1])
                particules_dir[part] = next_move
                particules_energy[part] = rd.randint(dist_char_low, dist_char_high) -1
            
            #save pos at i_step
            particules_pos_memo[i_step] = copy.copy(particules_pos)


    #version 3
    def run_walls():
        for i_step in range(1, max_n_steps+1):
            for part in particules_pos:

                if particules_sleep[part] > 0:
                    particules_sleep[part] = particules_sleep[part] -1
                    particules_energy[part] = max(particules_energy[part] -1, 0)
                    continue

                if particules_energy[part] > 0:
                    move = particules_dir[part]
                    future_pos = (particules_pos[part][0] + move[0], particules_pos[part][1] + move[1])
                    _is_in = is_in(future_pos)
                    if _is_in and is_free(maze, future_pos):
                        particules_energy[part] = particules_energy[part] - 1
                        particules_pos[part] = future_pos
                    elif _is_in and particules_energy[part] > wall_cost:
                        particules_energy[part] = particules_energy[part] - 1
                        particules_pos[part] = future_pos
                        particules_sleep[part] = wall_cost
                    else:
                        particules_energy[part] = 0
                    continue

                #else set a move
                corr_moves = {}
                for move in moves:
                    future_pos = (particules_pos[part][0] + move[0], particules_pos[part][1] + move[1])
                    if is_in(future_pos):
                        corr_moves[move] = future_pos
                next_move = rd.choice(list(corr_moves.keys()))
                energy = rd.randint(dist_char_low, dist_char_high) if next_move != (0,0) else 1
                future_pos = corr_moves[next_move]
                if is_free(maze, future_pos):
                    particules_pos[part] = future_pos
                    particules_dir[part] = next_move
                    particules_energy[part] = energy-1
                elif energy > wall_cost:
                    particules_pos[part] = future_pos
                    particules_dir[part] = next_move
                    particules_energy[part] = energy-1
                    particules_sleep[part] = wall_cost
                else:
                    particules_energy[part] = 0
            
            #save pos at i_step
            particules_pos_memo[i_step] = copy.copy(particules_pos)

    #run
    run_walls()

    #init array
    mat_memo = {}
    
    for i_step in particules_pos_memo:
        particules_pos = particules_pos_memo[i_step]

        #Array
        mat = np.zeros_like(maze)
        mat = mat.astype('int64')

        for part in particules_pos:
            i,j = particules_pos[part]
            mat[i,j] += 1

        #Smooth
        s_mat = np.zeros_like(mat)
        for i in range(size1):
            for j in range(size2):
                if is_free(maze, (i,j)) and mat[i,j]!=0:
                    values = []
                    for k in range(i-kernel_size, i+kernel_size+1):
                        for l in range(j-kernel_size, j+kernel_size+1):
                            if is_in((k,l)) and abs(i-k) + abs(j-l) <= kernel_size:
                                values.append(mat[k,l])
                    s_mat[i,j] = int(statistics.mean(values))

        #Max
        k_mat = np.zeros_like(mat)
        for i in range(size1):
            for j in range(size2):
                if is_free(maze, (i,j)):
                    values = []
                    for k in range(i-1, i+1+1):
                        for l in range(j-1, j+1+1):
                            if is_in((k,l)) and abs(i-k) + abs(j-l) <= 1:
                                values.append(s_mat[k,l])
                    if mat[i,j]!=0 or (mat[i,j]==0 and len([elem for elem in values if elem != 0]) >= 3):
                        k_mat[i,j] = max(values)

        #Normalize
        sum_mat = np.sum(k_mat)
        if sum_mat < 1e-5:
            mat_memo[i_step] = np.zeros_like(mat)
            continue

        norm_mat = np.zeros_like(maze)
        norm_mat = mat.astype('float')
        for i in range(size1):
            for j in range(size2):
                norm_mat[i,j] = k_mat[i,j]/sum_mat

        mat_memo[i_step] = norm_mat
    
    return mat_memo


if __name__ == '__main__':
    rd.seed(0)
    n_blind_steps = 30

    #define map
    size = 20
    maze = np.zeros((size,size))
    maze = maze.astype('int64')

    obs_ratio = 0.15
    n_obs = int(obs_ratio * size**2)
    obstacles_pos = []
    for obs in range(n_obs):
        i = rd.randint(0,size-1)
        j = rd.randint(0,size-1)
        maze[i,j] = 1
        obstacles_pos.append((i,j))
    print(maze)

    #define pos
    for _ in range(10):
        center = (rd.randint(0,size-1), rd.randint(0,size-1))
        if maze[center[0], center[1]] == 0:
            print('center', center)
            break

    mat_memo = get_pos_distrib_mc_build_memo(maze, center, n_blind_steps, dist_char_low = 1, dist_char_high = 10, wall_cost = 2, kernel_size = 2)
    
    for i_step in mat_memo:
        print('i_step', i_step)
        print((mat_memo[i_step]*1000).astype(int))
