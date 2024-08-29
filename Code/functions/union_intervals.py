#from ChatGPT
def union_intervals(intervals):
    intervals.sort() # trier les intervalles par ordre croissant de leur début
    merged = [intervals[0]] # initialiser la liste fusionnée avec le premier intervalle
    
    for current in intervals: # parcourir chaque intervalle dans la liste d'entrée
        if current[0] <= merged[-1][1]: # si le début de l'intervalle actuel est inclus dans l'intervalle fusionné
            merged[-1][1] = max(current[1], merged[-1][1]) # fusionner les intervalles en étendant la fin de l'intervalle fusionné si nécessaire
        else:
            merged.append(current) # sinon, ajouter l'intervalle actuel à la liste fusionnée
    
    return merged # retourner le dernier intervalle fusionné, qui correspond à la réunion de tous les intervalles

# Exemple d'utilisation:
def test():
    intervals = [[1,3], [2,9], [5,7], [6,8]]
    union = union_intervals(intervals)
    print(union) # Output: [1, 8]

if __name__ == "__main__":
    test()