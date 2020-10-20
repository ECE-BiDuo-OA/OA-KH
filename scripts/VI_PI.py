import numpy as np
import time

gamma = 0.99

actions = ['N', 'E', 'S', 'W']

R = np.array([[-0.02,   -0.02,  -0.02,  1],
              [-0.02,   0.0,    -0.02,  -1],
              [-0.02,   -0.02,  -0.02,  -0.02]])

map = np.array([[1, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 1, 1]])

end_state = [(3,0), (3,1)]

#Calcule la probabilité de passer de l'état state1 (x1, y1) à l'état state2 (x2, y2) avec l'action 'action'
def compute_probability(state1, state2, action):
    if action=='N':
        if state2[1]==state1[1]+1 and state2[0]==state1[0]:
            return 0.0
        elif state2[1]==state1[1]-1 and state2[0]==state1[0]:
            return 0.8
        elif (state2[0]==state1[0]-1 or state2[0]==state1[0]+1) and state2[1]==state1[1]:
            return 0.1
        else:
            return 0.0
    elif action=='E':
        if state2[0]==state1[0]+1 and state2[1]==state1[1]:
            return 0.8
        elif state2[0]==state1[0]-1 and state2[1]==state1[1]:
            return 0.0
        elif (state2[1]==state1[1]-1 or state2[1]==state1[1]+1) and state2[0]==state1[0]:
            return 0.1
        else:
            return 0.0
    elif action=='S':
        if state2[1]==state1[1]+1 and state2[0]==state1[0]:
            return 0.8
        elif state2[1]==state1[1]-1 and state2[0]==state1[0]:
            return 0.0
        elif (state2[0]==state1[0]-1 or state2[0]==state1[0]+1) and state2[1]==state1[1]:
            return 0.1
        else:
            return 0.0
    elif action=='W':
        if state2[0]==state1[0]+1 and state2[1]==state1[1]:
            return 0.0
        elif state2[0]==state1[0]-1 and state2[1]==state1[1]:
            return 0.8
        elif (state2[1]==state1[1]-1 or state2[1]==state1[1]+1) and state2[0]==state1[0]:
            return 0.1
        else:
            return 0.0
    
    return 0.0

def getValue(V, actual_state, state):
    value = 0.0
    
    #Si l'état adjacent est inaccessible la valeur de l'état à prendre pour résoudre l'équation de Bellmann est la valeur de l'état actuel
    if state[0]==-1 or state[0]==len(map[0]) or state[1]==-1 or state[1]==len(map):
        value = V[actual_state[1]][actual_state[0]]
    elif map[state[1]][state[0]] == 0:
        value = V[actual_state[1]][actual_state[0]]
    else:
        value = V[state[1]][state[0]]
    
    return value

#Résoud l'équation de Bellmann, retourne la fonction de valeur
def compute_bellman(V, policy, actual_state):
    somme = 0.0
    
    #Récupérer les coordonnées (x,y) des cases adjacentes (sans les cases en diagonale)
    states = [(actual_state[0]-1, actual_state[1]), (actual_state[0]+1, actual_state[1]), (actual_state[0], actual_state[1]-1), (actual_state[0], actual_state[1]+1)]
    
    for s in states:
        somme += compute_probability(actual_state, s, policy[actual_state[1]][actual_state[0]]) * getValue(V, actual_state, s)
    
    return R[actual_state[1]][actual_state[0]] + gamma*somme

#Calcule la politique optimale, retourne la fonction de politique optimale
def compute_optimal_policy(V, actual_state):
    somme_action = [0.0, 0.0, 0.0, 0.0]
    
    #Récupérer les coordonnées (x,y) des états adjacentes (sans les cases en diagonale)
    states = [(actual_state[0]-1, actual_state[1]), (actual_state[0]+1, actual_state[1]), (actual_state[0], actual_state[1]-1), (actual_state[0], actual_state[1]+1)]
    
    for a in range(len(actions)):
        for s in states:              
            somme_action[a] += compute_probability(actual_state, s, actions[a]) * getValue(V, actual_state, s)
    
    return actions[np.argmax(somme_action)]

#Résoud l'équation de Bellmann optimale, retourne la fonction de valeur optimale et la fonction de politique optimale
def compute_optimal_bellman(V, actual_state):
    somme_action = [0.0, 0.0, 0.0, 0.0]
    
    #Récupérer les coordonnées (x,y) des états adjacentes (sans les cases en diagonale)
    states = [(actual_state[0]-1, actual_state[1]), (actual_state[0]+1, actual_state[1]), (actual_state[0], actual_state[1]-1), (actual_state[0], actual_state[1]+1)]
    
    for a in range(len(actions)):
        for s in states:                
            somme_action[a] += compute_probability(actual_state, s, actions[a]) * getValue(V, actual_state, s)
    
    return R[actual_state[1]][actual_state[0]] + gamma*np.max(somme_action), actions[np.argmax(somme_action)]
    
def value_iteration(map):
    it = 0
    
    V = np.zeros((3,4))
    V[0][3] = 1.0
    V[1][3] = -1.0
    
    V_old = np.zeros((3,4))
    
    policy = np.array([['N', 'N', 'N', 'X'],
                       ['N', 'X', 'N', 'X'],
                       ['N', 'N', 'N', 'N']])
    
    while not np.array_equal(V, V_old):
        V_old[:] = V[:]
        for row in range(len(map)):
            for col in range(len(map[0])):
                s = (col, row)
                if not s in end_state and not s==(1,1):
                    V[s[1]][s[0]], policy[s[1]][s[0]] = compute_optimal_bellman(V, s)
        it+=1
        
    print("Value iteration: it=" + str(it) + "\n")
                    
    return V, policy

def policy_iteration(map):
    it = 0
    
    V = np.zeros((3,4))
    V[0][3] = 1.0
    V[1][3] = -1.0
    
    V_old = np.zeros((3,4))
    
    policy = np.array([['S', 'W', 'W', 'X'],
                       ['S', 'X', 'N', 'X'],
                       ['E', 'E', 'N', 'W']])
    
    policy_old = np.array([['X', 'X', 'X', 'X'],
                           ['X', 'X', 'X', 'X'],
                           ['X', 'X', 'X', 'X']])
    
    while not np.array_equal(V, V_old) or not np.array_equal(policy, policy_old):
        V_old[:] = V[:]
        policy_old[:] = policy[:]
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

