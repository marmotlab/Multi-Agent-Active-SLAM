#code taken from Chat GPT and then updated
import time
import numpy as np

# Calcule la distance Manhattan entre deux points
def distance_manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def discount(dist, discount_rate, ratio):
    return discount_rate**dist*ratio + (1-ratio)

# Calcule le gain possible à partir d'une position donnée avec une distance donnée
def dfs(position, distance_restante, cibles_restantes, memo, distance_totale, matrice, distance_max_entre_cibles, discount_rate, discount_ratio, max_dfs):
    global n_dfs
    n_dfs += 1

    # Si nous avons atteint le nombre de récursion maximum
    if n_dfs > max_dfs:
        return False, [], []

    if (position, distance_restante, tuple(cibles_restantes)) in memo:
        return memo[(position, distance_restante, tuple(cibles_restantes))]

    # Si nous ne pouvons visiter aucun autre point, nous retournons 0
    if not cibles_restantes or distance_restante <= 0:
        return 0, [], []

    max_gain = 0
    meilleur_chemin = []
    meilleur_list_gains = []

    dist_plus_proche = min([distance_manhattan(position, cible) for cible in cibles_restantes] + [1e2])

    for cible in cibles_restantes:
        dist = distance_manhattan(position, cible)

        # Si nous pouvons atteindre la cible
        if dist <= distance_restante and (dist <= distance_max_entre_cibles or dist_plus_proche > distance_max_entre_cibles):
            nouveau_cibles_restantes = [c for c in cibles_restantes if c != cible]
            gain_recursif, chemin_recursif, list_gains_recursif = dfs(cible, distance_restante - dist, nouveau_cibles_restantes, memo, distance_totale + dist, matrice, distance_max_entre_cibles, discount_rate, discount_ratio, max_dfs)
            
            if gain_recursif is False:
                return False, [], []
            
            gain_local = matrice[cible[0]][cible[1]] * discount(dist + distance_totale, discount_rate, discount_ratio)
            gain_total = gain_local + gain_recursif
            
            if gain_total > max_gain :
                max_gain = gain_total
                meilleur_chemin = [cible] + chemin_recursif
                meilleur_list_gains = [round(gain_local,0)] + list_gains_recursif

    #print('n_dfs :', n_dfs)
    memo[(position, distance_restante, tuple(cibles_restantes))] = (max_gain, meilleur_chemin, meilleur_list_gains)
    return max_gain, meilleur_chemin, meilleur_list_gains


def gain_optimal(matrice, cibles, depart, distance_max = 50, distance_max_entre_cibles = 50, discount_rate = 0.99, discount_ratio = 0.7):
    
    #limiter la profondeur de la récursion
    max_dfs = 2e4
    global n_dfs
    n_dfs = 0

    #restriction du nombre de cibles et de la distance max a parcourrir
    if len(cibles) > 30:
        #print(' error - Heavy Calculation ; number of targets :', len(cibles), ' ; distance travelled :', distance_max)
        print('multi obj tsp error - heavy calculation ; number of targets :', len(cibles))
        return False, [], [], n_dfs
    
    #init memo
    memo = {}

    distance_totale = 0
    gain_total, chemin, list_gains = dfs(depart, distance_max, cibles, memo, distance_totale, matrice, distance_max_entre_cibles, discount_rate, discount_ratio, max_dfs)  # En supposant que la position de départ est (0,0)
    
    #print('n_dfs :', n_dfs)

    if gain_total is False:
        print("multi obj tsp failed, dfs called", n_dfs, "times")
        return False, [], [], n_dfs
    
    print("multi obj tsp worked, dfs called", n_dfs, "times")
    return chemin, list_gains, round(gain_total,0), n_dfs


if __name__ == '__main__':

    matrice = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

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
    cibles15 = [(0, 0), (1, 2), (3, 3), (6, 6), (12, 12), (1, 8), (10, 5), (14, 3)]
    gains15 = {
        (0, 1): 10,
        (1, 2): 10,
        (3, 3): 10,
        (6, 6): 10,
        (12, 12): 10,
        (1, 8): 10,
        (10, 5): 10,
        (14, 3): 10
    }
    for cible, gain in gains15.items():
        matrice15[cible[0]][cible[1]] = gain

    matrice30 = np.zeros((30,30))
    #'''
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
        (10, 11) : 194,
        (11, 0) : 95,
        (11, 5) : 215,
        (12, 14) : 183,
        (16, 16) : 128,
        (18, 18) : 149,
        (22, 17) : 185,
        (26, 0) : 42,
        (27, 3) : 76,
        (27, 17) : 158,
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

    depart = (7,6)
    #depart = (27,6)
    distance_max = 70
    distance_max_entre_cibles = 15
    discount_rate = 0.9
    discount_ratio = 0.6

    tic = time.time()
    chemin, list_gains, gain, n_dfs = gain_optimal(matrice30, cibles30, depart, distance_max, distance_max_entre_cibles, discount_rate, discount_ratio)
    tac = time.time()
    print("Chemin optimal :", chemin)
    print("Liste des gains :", list_gains)
    print("Gain total :", gain)
    print("Temps d'execution :", tac - tic)
