#code taken from Chat GPT and then updated
import time
import numpy as np

# Calcule la distance Manhattan entre deux points
def distance_manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def discount(dist, discount_rate, ratio):
    return discount_rate**dist*ratio + (1-ratio)

# Calcule le gain possible à partir d'une position donnée avec une distance donnée
def dfs(position, cibles_restantes, matrice, distance_parcourue, gain_ramasse, hist_pos, hist_gain_par_etapes, memo, distance_max = 70, distance_max_entre_cibles = 70, discount_rate = 0.99, discount_ratio = 1, max_dfs = 1e5, disp = False):
    global n_dfs
    n_dfs += 1

    if disp:
        print('---dfs start')
        print('from position', position)
        print('distance_parcourue', distance_parcourue)
        print('gain ramasse', gain_ramasse)
        print('historique des pos', hist_pos)
        print('historique des gains', hist_gain_par_etapes)
        print('cibles_restantes', cibles_restantes)

    # Si nous avons atteint le nombre de récursion maximum
    if n_dfs > max_dfs:
        if disp : print('max dfs---')
        return [], [], [], [], [], False, False

    if (position, distance_parcourue, tuple(cibles_restantes)) in memo:
        if disp : print('memo---')
        return memo[(position, distance_parcourue, tuple(cibles_restantes))]

    # Si nous ne pouvons visiter aucun autre point, nous retournons 0
    if not cibles_restantes or distance_parcourue > distance_max:
        if disp : print('no targets or distance_max---')
        return [], [], [], [], [], 0, 0
        
    hist_pos = hist_pos[:] + [position]

    curr_gain_per_step = gain_ramasse/distance_parcourue if distance_parcourue != 0 else 0
    hist_gain_par_etapes = hist_gain_par_etapes[:] + [curr_gain_per_step]

    dist_plus_proche = min([distance_manhattan(position, cible) for cible in cibles_restantes] + [1e2])
    max_gain_remaining = sum([matrice[cible[0]][cible[1]] for cible in cibles_restantes]) * discount(distance_parcourue+dist_plus_proche, discount_rate, discount_ratio)
    if disp:
        print('curr_gain_per_step', curr_gain_per_step)
        print('max_gain_remaining', max_gain_remaining, 'nearest_next_target', dist_plus_proche)
        print('best potential gain', (gain_ramasse+max_gain_remaining)/(distance_parcourue+dist_plus_proche))

    if disp :
        print('historique des pos', hist_pos)
        print('historique des gains', hist_gain_par_etapes)

    if (gain_ramasse+max_gain_remaining)/(distance_parcourue+dist_plus_proche) < max(hist_gain_par_etapes):
        if disp :
            print('not enough gain to continue---')
        return [], [], [], [], [], 0, 0
    
   #init best
    meilleur_gain = 0
    meilleur_dist = 0
    max_gain_per_step = curr_gain_per_step

    meilleur_chemin = []
    distances_entre_cibles = []
    etapes = []
    meilleur_list_gains = []
    meilleur_list_gains_reel = []
    
    for cible in cibles_restantes:
        dist = distance_manhattan(position, cible)
        gain_local = matrice[cible[0]][cible[1]] * discount(distance_parcourue + dist, discount_rate, discount_ratio)
        if disp :
            print('from', position, 'try', cible)
            print('dist', dist)
            print('gain_local', gain_local)

        # Si nous pouvons atteindre la cible
        if distance_parcourue + dist  <= distance_max and (dist <= distance_max_entre_cibles or dist_plus_proche > distance_max_entre_cibles):

            nouvelles_cibles_restantes = [c for c in cibles_restantes if c != cible]
            chemin_recursif, etapes_rec, distances_entre_cibles_rec, list_gains_recursif, list_gain_reels_rec, distance_recursive, gain_recursif = dfs(
                cible, nouvelles_cibles_restantes, matrice, distance_parcourue + dist, gain_ramasse + gain_local, 
                hist_pos, hist_gain_par_etapes, memo, distance_max, distance_max_entre_cibles, discount_rate, discount_ratio, max_dfs, disp)
            
            if gain_recursif is False:
                return [], [], [], [], [], False, False
            
            dist_totale = dist + distance_recursive
            gain_total = gain_local + gain_recursif
            if disp :
                print('then, trying', cible, 'from', hist_pos)
                print('distance_recursive', distance_recursive)
                print('gain_rec', gain_recursif)
                print('distance_future', dist_totale)
                print('gain_future', gain_total)
                print('distance_totale', (distance_parcourue+dist_totale))
                print('gain_total', gain_ramasse+gain_total)
                print('potential gain per step if accepted : '+str((gain_ramasse+gain_total)/(distance_parcourue+dist_totale)))
                print('max_gain_per_step', max_gain_per_step)

            if (gain_ramasse+gain_total)/(distance_parcourue+dist_totale) > max_gain_per_step :
                meilleur_gain = gain_total
                meilleur_dist = dist_totale
                max_gain_per_step = (gain_ramasse+gain_total)/(distance_parcourue+dist_totale)
                hist_gain_par_etapes[-1] = max_gain_per_step

                meilleur_chemin = [cible] + chemin_recursif
                etapes = [distance_parcourue+dist] + etapes_rec
                distances_entre_cibles = [dist] + distances_entre_cibles_rec
                meilleur_list_gains = [round(gain_local,1)] + list_gains_recursif
                meilleur_list_gains_reel = [round(matrice[cible[0]][cible[1]],1)] + list_gain_reels_rec
                
                if disp : 
                    print('best : branch accepted')
                    print('gain', gain_total)
                    print('dist', dist_totale)                
                    print('new max_gain_per_step', max_gain_per_step)

                    print(meilleur_chemin)
                    print('etapes', etapes)
                    print('dist_cibles', distances_entre_cibles)
                    print(meilleur_list_gains)
                    print(meilleur_list_gains_reel)

    #print('n_dfs :', n_dfs)
    memo[(position, distance_parcourue, tuple(cibles_restantes))] = (meilleur_chemin, etapes, distances_entre_cibles, meilleur_list_gains, meilleur_list_gains_reel, meilleur_dist, meilleur_gain)
    if disp : print('return---')
    return meilleur_chemin, etapes, distances_entre_cibles, meilleur_list_gains, meilleur_list_gains_reel, meilleur_dist, meilleur_gain


def gain_optimal_2(matrice, cibles, depart, distance_max = 70, distance_max_entre_cibles = 50, discount_rate = 0.99, discount_ratio = 0.7, disp = False):
    
    #limiter la profondeur de la récursion
    max_dfs = 2e4
    global n_dfs
    n_dfs = 0

    #restriction du nombre de cibles et de la distance max a parcourrir
    if len(cibles) > 30:
        #print(' error - Heavy Calculation ; number of targets :', len(cibles), ' ; distance travelled :', distance_max)
        if disp : print('multi obj tsp error - heavy calculation ; number of targets :', len(cibles))
        return False, [], [], [], [], False, False, n_dfs
    
    #init memo
    memo = {}
    distance_parcourue = 0
    gain_ramasse = 0

    hist_pos = []
    hist_gain_par_etapes = []

    '''
    import sys
    print(sys.getrecursionlimit())
    sys.setrecursionlimit(5000)
    print(sys.getrecursionlimit())
    '''
    
    chemin, etapes, distances_entre_cibles, list_gains, list_gains_reel, distance_totale, gain_total = dfs(depart, cibles, matrice, distance_parcourue, gain_ramasse, hist_pos, hist_gain_par_etapes, memo, distance_max, distance_max_entre_cibles, discount_rate, discount_ratio, max_dfs, disp)  # En supposant que la position de départ est (0,0)
    
    #print('n_dfs :', n_dfs)

    if gain_total is False:
        if True : print("multi obj tsp failed, dfs called", n_dfs, "times")
        return False, [], [], [], [], False, False, n_dfs
    
    if True : print("multi obj tsp worked, dfs called", n_dfs, "times")
    return chemin, etapes, distances_entre_cibles, list_gains, list_gains_reel, distance_totale, round(gain_total,1), n_dfs


if __name__ == '__main__':

    matrice4 = np.zeros((4,4))
    cibles = [(0, 1), (1, 2), (2, 3)]
    gains = {
        (0, 1): 5,
        (1, 2): 10,
        (2, 3): 7
    }
    for cible, gain in gains.items():
        matrice4[cible[0]][cible[1]] = gain

    matrice15 = np.zeros((15,15))
    cibles15 = []
    gains15 = {
        (4, 4): 10,
        (4, 7): 3,
        (6, 6): 10,
    }
    for cible, gain in gains15.items():
        cibles15.append(cible)
        matrice15[cible[0]][cible[1]] = gain

    matrice30 = np.zeros((36,36))
    '''
    gains30 = {
        (4, 7) : 206,
        (4, 12) : 264,
        (5, 3) : 217,
        (9, 3) : 93,
        (10, 15) : 482,
        (14, 24) : 129,
        (15, 21) : 119,
        (16, 26) : 138,
        (20, 26) : 245,
        (25, 27) : 83,
        (29, 25) : 122,
                }
    '''
    gains30 = {
        (32, 19) : 54,
        (33, 21) : 81,
        (28, 21) : 128,
        (26, 16) : 145,
        (30, 12) : 129,
        (33, 16) : 82,
        (25, 14) : 82,
        (25, 18) : 76,
        (31, 14) : 113,
        (30, 22) : 60,
        }
    #'''
    '''
        (3, 1) : 10,
        (3, 2) : 10,
        (4, 1) : 10,
        (4, 2) : 10,
        (5, 1) : 10,
        (5, 2) : 10,


        (24, 16) : 10,
        (24, 9)  : 10,
        (22, 11) : 10,
        (24, 15) : 10,
        (26, 9)  : 10,
        (25, 16) : 10,
        (24, 12) : 10,

        

        (0, 10): 10,
        (10, 2): 10,
        (10, 1): 10,
        (3, 13): 10,
        (26, 6): 10,
        (12, 12): 10,
        (1, 28): 10,
        (10, 5): 10,
        (14, 23): 10,
        (23, 13): 10,
#10 -> 1 14 30 110 200(80)

        (16, 8): 10,
        (12, 2): 10,
        (4, 28): 10,
        (12, 5): 10,
        (27, 20): 10,
        (5, 10): 10,
#16 -> 12 625 1455 4310(70)

        (24, 1): 10,
        (14, 28): 10,
        (10, 5): 10,
        (14, 23): 10,
        (23, 13): 10,
        (17, 9): 10,
        (16, 16): 10,
#23 -> 20 1255 5365(60)


        (5, 18): 10,
        (14, 8): 10,
        (23, 10): 10,
        (4, 1): 10,
        (13, 10): 10,
        (2, 1): 10,
        (14, 10): 10,
#30 -> 130(30) 1700(40) 19000(50)

        (14, 9): 10,
        (6, 1): 10,
        (9, 8): 10,
        (24, 18): 10,
        (28, 12): 10,
        (2, 15): 10,
        (3, 11): 10,
        (12, 1): 10,
        (4, 25): 10,
        (25, 4): 10,

#40 -> 5200(30)
        (1, 22): 10,
        (9, 3): 10,
        (24, 24): 10,
        (28, 5): 10,
        (2, 16): 10,
        (3, 7): 10,
        (12, 18): 10,
        (3, 9): 10,
        (25, 10): 10,
        (14, 20): 10,

#50 220(20)

    }
            #'''

    cibles30 = []
    for cible, gain in gains30.items():
        cibles30.append(cible)
        matrice30[cible[0]][cible[1]] = gain

    print(matrice30)

    '''

    '''
    depart = (30, 18)
    #depart = (27,6)
    distance_max = 70
    distance_max_entre_cibles = 70
    discount_rate = 0.97
    discount_ratio = 0.8

    tic = time.time()
    chemin, etapes, distances_entre_cibles, list_gains, list_gains_reels, dist, gain, n_dfs = gain_optimal_2(matrice30, cibles30, depart, distance_max, distance_max_entre_cibles, discount_rate, discount_ratio, disp = False)
    tac = time.time()
    print("Chemin optimal :", chemin)
    print("Etapes :", etapes)
    print("Distance entre cibles :", distances_entre_cibles)
    print("Liste des gains :", list_gains)
    print("Liste des gains reels :", list_gains_reels)
    print("Distance total :", dist)
    print("Gain total :", gain)
    print("Temps d'execution :", tac - tic)
