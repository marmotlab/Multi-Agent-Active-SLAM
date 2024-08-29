import numpy as np

def impact_voisins_gauss(tableau, coefficient, distance):
    # création d'un tableau de même dimension que l'entrée
    sortie = np.zeros_like(tableau, dtype=float)

    # parcours de chaque élément du tableau
    for i in range(tableau.shape[0]):
        for j in range(tableau.shape[1]):
            # parcours des éléments voisins dans un carré centré sur l'élément courant
            for k in range(max(0, i-distance), min(tableau.shape[0], i+distance+1)):
                for l in range(max(0, j-distance), min(tableau.shape[1], j+distance+1)):
                    if k == i and l == j:
                        sortie[i,j] += tableau[k,l]
                    else:
                        # calcul de la distance entre l'élément courant et son voisin
                        distance_voisin = np.sqrt((i-k)**2 + (j-l)**2)
                        # calcul de l'influence du voisin sur l'élément courant
                        influence_voisin = coefficient * np.exp(-(distance_voisin**2)/(2*(distance**2)))
                        # ajout de la contribution du voisin au résultat final
                        sortie[i,j] += influence_voisin * tableau[k,l]
    return sortie


def neighbor_impact_varying(array, stdev_array, max_influence_dist, rnd):
    # Create an output array of the same dimensions as the input
    output = np.zeros_like(array, dtype=float)

    # Loop over each element in the array
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            # Loop over neighboring elements in a square centered on the current element
            for k in range(max(0, i-max_influence_dist), min(array.shape[0], i+max_influence_dist+1)):
                for l in range(max(0, j-max_influence_dist), min(array.shape[1], j+max_influence_dist+1)):
                    # Calculate the distance between the current element and its neighbor
                    neighbor_distance = np.sqrt((i-k)**2 + (j-l)**2)
                    # Calculate the neighbor's influence on the current element
                    neighbor_influence = 1 / ((stdev_array[i,j]+0.55) * np.sqrt(2*np.pi)) * np.exp(-(neighbor_distance**2)/(2*((stdev_array[i,j]+0.55)**2)))
                    # Add the neighbor's contribution to the final result
                    output[i,j] += neighbor_influence * array[k,l]
                    if i == j == 3:
                        print(k, l)
                        print(neighbor_distance, neighbor_influence, 1 / ((stdev_array[i,j]+0.4) * np.sqrt(2*np.pi)), np.exp(-(neighbor_distance**2)/(2*((stdev_array[i,j]+0.3)**2))))

    #round
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            output[i,j] = round(output[i,j], rnd)
    
    return output

if __name__ == '__main__':
    array = np.array([
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,100,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0]
    ])

    distance = 1

    #diffusion = impact_voisins(array, coefficient, distance)
    #diffusion_gauss = impact_voisins_gauss(array, coefficient, distance)

    stdev_array = np.array([
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0]
    ])

    max_influence_dist = 1
    rnd = 1
    
    diffusion_gauss_variant = neighbor_impact_varying(array, stdev_array, max_influence_dist, rnd)

    #print(diffusion)
    #print(diffusion_gauss)
    print(diffusion_gauss_variant)
    