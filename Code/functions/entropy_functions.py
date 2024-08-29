import numpy as np
import math

n_d2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
no_err_prob_2d = [1, 0.85, 0.728, 0.628, 0.546, 0.479, 0.422, 0.375, 0.335, 0.301, 0.272, 0.247, 0.225, 0.207, 0.191, 0.176, 0.164, 0.15, 0.143]

n_d2_guess = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
no_err_prob_2d_guess = [0.137, 0.132, 0.1280, 0.1245, 0.1215, 0.1190, 0.1170, 0.1163, 0.1158, 0.1145, 0.1134, 0.1124, 0.1115, 0.1107, 0.1100]

def get_correct_cell_prob(blind_v, thres1 = 10, thres2 = 50): #beta
    if blind_v < thres1:
        return 1
    elif blind_v-thres1 in n_d2:
        return no_err_prob_2d[int(blind_v-thres1)]
    elif blind_v-thres1 in n_d2_guess:
        return no_err_prob_2d_guess[n_d2_guess.index(int(blind_v-thres1))]
    elif blind_v < thres2:
        return 0.11
    else :
        return 0.1
        #return np.exp(-blind_v *0.13)

def get_stdev_error(blind_v, odom_err_rate): #beta
    sigma = (pow(blind_v*odom_err_rate, 1/3) + pow(blind_v*odom_err_rate, 1/2))/2 #model 2
    return sigma


if __name__ == '__main__':
    bd = 12
    pc = get_correct_cell_prob(bd)
    print(pc)