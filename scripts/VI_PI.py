import numpy as np
import time

#Facteur de dévaluation
gamma = 0.99

#Liste contenant les actions possibles
actions = ['N', 'E', 'S', 'W']

#Tableau représentant les récompenses
R = np.array([[-0.02,   -0.02,  -0.02,  1],
              [-0.02,   0.0,    -0.02,  -1],
              [-0.02,   -0.02,  -0.02,  -0.02]])

#Tableau représentant le monde (1 = case accessible; 0 = case non accessible)
map = np.array([[1, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 1, 1]])

#Liste contenant les états qui terminent le parcours du robot (succès ou échec)
end_state = [(3,0), (3,1)]

#Calcule la probabilité de passer de l'état state1 (x1, y1) à l'état state2 (x2, y2) avec l'action 'action'
def compute_probability(state1, state2, action):
    x1 = state1[0]
    y1 = state1[1]
    
    x2 = state2[0]
    y2 = state2[1]
    
    if action=='N':
        if y2==y1+1 and x2==x1:
            return 0.0
        elif y2==y1-1 and x2==x1:
            return 0.8
        elif (x2==x1-1 or x2==x1+1) and y2==y1:
            return 0.1
        else:
            return 0.0
    elif action=='E':
        if x2==x1+1 and y2==y1:
            return 0.8
        elif x2==x1-1 and y2==y1:
            return 0.0
        elif (y2==y1-1 or y2==y1+1) and x2==x1:
            return 0.1
        else:
            return 0.0
    elif action=='S':
        if y2==y1+1 and x2==x1:
            return 0.8
        elif y2==y1-1 and x2==x1:
            return 0.0
        elif (x2==x1-1 or x2==x1+1) and y2==y1:
            return 0.1
        else:
            return 0.0
    elif action=='W':
        if x2==x1+1 and y2==y1:
            return 0.0
        elif x2==x1-1 and y2==y1:
            return 0.8
        elif (y2==y1-1 or y2==y1+1) and x2==x1:
            return 0.1
        else:
            return 0.0
    
    return 0.0

#Retourne la valeur de l'état state
def getValue(V, actual_state, state):
    value = 0.0
    
    #Si l'état adjacent est inaccessible la valeur de l'état à prendre pour résoudre l'équation de Bellmann est la valeur de l'état actuel
    if state[0]==-1 or state[0]==len(map[0]) or state[1]==-1 or state[1]==len(map):
        value = V[actual_state[1]][actual_state[0]]
    elif map[state[1]][state[0]] == 0:
        value = V[actual_state[1]][actual_state[0]]
    else:
        #Sinon si l'état est accessible, on prend sa valeur
        value = V[state[1]][state[0]]
    
    return value

#Résoud l'équation de Bellmann, retourne la fonction de valeur
def compute_bellman(V, policy, actual_state):
    somme = 0.0
    
    #Récupérer les coordonnées (x,y) des cases adjacentes (sans les cases en diagonale)
    states = [(actual_state[0]-1, actual_state[1]), (actual_state[0]+1, actual_state[1]), (actual_state[0], actual_state[1]-1), (actual_state[0], actual_state[1]+1)]
    
    #Pour chaque état adjacent on somme le produit Psa(s')*V(s')
    for s in states:
        somme += compute_probability(actual_state, s, policy[actual_state[1]][actual_state[0]]) * getValue(V, actual_state, s)
    
    return R[actual_state[1]][actual_state[0]] + gamma*somme

#Calcule la politique optimale, retourne la fonction de politique optimale
def compute_optimal_policy(V, actual_state):
    somme_action = [0.0, 0.0, 0.0, 0.0]
    
    #Récupérer les coordonnées (x,y) des états adjacentes (sans les cases en diagonale)
    states = [(actual_state[0]-1, actual_state[1]), (actual_state[0]+1, actual_state[1]), (actual_state[0], actual_state[1]-1), (actual_state[0], actual_state[1]+1)]
    
    #Pour chaque action, on stocke la somme des produits Psa(s')*V(s') avec s' les états adjacents
    for a in range(len(actions)):
        for s in states:
            somme_action[a] += compute_probability(actual_state, s, actions[a]) * getValue(V, actual_state, s)
    
    #On retourne l'action (N,E,S,W) en récupérant l'indice de somme_action qui correspond à la valeur maximale
    return actions[np.argmax(somme_action)]

#Résoud l'équation de Bellmann optimale, retourne la fonction de valeur optimale et la fonction de politique optimale
def compute_optimal_bellman(V, actual_state):
    somme_action = [0.0, 0.0, 0.0, 0.0]
    
    #Récupérer les coordonnées (x,y) des états adjacentes (sans les cases en diagonale)
    states = [(actual_state[0]-1, actual_state[1]), (actual_state[0]+1, actual_state[1]), (actual_state[0], actual_state[1]-1), (actual_state[0], actual_state[1]+1)]
    
    #Pour chaque action, on stocke la somme des produits Psa(s')*V(s') avec s' les états adjacents
    for a in range(len(actions)):
        for s in states:                
            somme_action[a] += compute_probability(actual_state, s, actions[a]) * getValue(V, actual_state, s)
    
    #On retourne la valeur maximale de somme_action et l'action (N,E,S,W) en récupérant l'indice de somme_action qui correspond à la valeur maximale
    return R[actual_state[1]][actual_state[0]] + gamma*np.max(somme_action), actions[np.argmax(somme_action)]

#QUESTION 1
#Calcule et retourne la fonction de valeur et la politique en appliquant l'algorithme value iteration
def value_iteration(map):
    #Compteur d'itération jusqu'à convergence de V
    it = 0
    
    #Initialisation du tableau de valeurs
    V = np.zeros((3,4))
    V[0][3] = 1.0
    V[1][3] = -1.0
    
    #Initialisation du tableau de valeurs de l'itération précédente
    V_old = np.zeros((3,4))
    
    #Initialisation du tableau de politique (X=Aucune action)
    policy = np.array([['N', 'N', 'N', 'X'],
                       ['N', 'X', 'N', 'X'],
                       ['N', 'N', 'N', 'N']])
    
    #Tant qu'il n'y a pas convergence de V (égalité entre V et V_old)
    while not np.array_equal(V, V_old):
        V_old[:] = V[:] #Copie de V dans V_old
        for row in range(len(map)):
            for col in range(len(map[0])):
                s = (col, row)
                if not s in end_state and not s==(1,1):
                    V[s[1]][s[0]], policy[s[1]][s[0]] = compute_optimal_bellman(V, s)
        it+=1
        
    print("Value iteration: it=" + str(it) + "\n")
                    
    return V, policy

#QUESTION 2
#Calcule et retourne la fonction de valeur et la politique en appliquant l'algorithme policy iteration
def policy_iteration(map):
    #Compteur d'itération jusqu'à convergence de V et stabilité de policy
    it = 0
    
    #Initialisation du tableau de valeurs
    V = np.zeros((3,4))
    V[0][3] = 1.0
    V[1][3] = -1.0

    #Initialisation du tableau de valeurs de l'itération précédente
    V_old = np.zeros((3,4))
    
    #Initialisation du tableau de politique (X=Aucune action)
    policy = np.array([['E', 'N', 'S', 'X'],
                       ['N', 'X', 'S', 'X'],
                       ['N', 'W', 'W', 'S']])
    
    #Initialisation du tableau de politique de l'itération précédente (X=Aucune action)
    policy_old = np.array([['X', 'X', 'X', 'X'],
                           ['X', 'X', 'X', 'X'],
                           ['X', 'X', 'X', 'X']])
    
    #Tant qu'il n'y a pas convergence de V (égalité entre V et V_old) et stabilité de policy
    while not np.array_equal(V, V_old) or not np.array_equal(policy, policy_old):
        V_old[:] = V[:] #Copie de V dans V_old
        policy_old[:] = policy[:] #Copie de policy dans policy_old
        for row in range(len(map)):
            for col in range(len(map[0])):
                s = (col, row)
                if not s in end_state and not s==(1,1):
                    V[s[1]][s[0]] = compute_bellman(V, policy, s)
                    policy[s[1]][s[0]] = compute_optimal_policy(V, s)
        it+=1
        
    print("Policy iteration: it=" + str(it) + "\n")
                    
    return V, policy

if __name__=="__main__":
    
    print("Map :")
    print(map)
    print("\n")
    
    #Initialisation des compteurs de temps
    time1 = time2 = 0.0
    
    time1 = time.time()
    value_fn, policy_fn = value_iteration(map)
    time2 = time.time()
    
    print("----- Value iteration -----")
    print("Duration : " + str(time2-time1) + " s")
    print("Optimal value function and policy :")
    print(value_fn)
    print(policy_fn)
    print("\n")
    
    time1 = time.time()
    value_fn, policy_fn = policy_iteration(map)
    time2 = time.time()
    
    print("----- Policy iteration -----")
    print("Duration : " + str(time2-time1) + " s")
    print("Optimal value function and policy :")
    print(value_fn)
    print(policy_fn)    


#QUESTION 3
#Les deux algorithmes donnent le même résultat
#L'algorithme value iteration s'effectue avec moins d'itérations que l'algorithme policy iteration (pour policy iteration le nombre d'itération
#dépend de l'initialisation du tableau policy) et prend en générale moins de temps que policy iteration (en moyenne 3 ms d'écart)
