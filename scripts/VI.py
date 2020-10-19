import numpy as np

gamma = 0.99

iteration = 20

actions = ['N', 'E', 'S', 'W']

R = np.array([[-0.02, -0.02, -0.02, 1],
              [-0.02, 0.0, -0.02, -1],
              [-0.02, -0.02, -0.02, -0.02]])

map = np.array([[1, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 1, 1]])

fixed_cell = [(3,0), (3,1), (1,1)]

#Calcule la probabilité de passer de l'état (x1, y1) à l'état (x2, y2) avec l'action action
def compute_probability(state1, state2, action):
    if action=='N':
        if state2[1]==state1[1]+1 and state2[0]==state1[0]:
            return 0.0
        elif state2[1]==state1[1]-1 and state2[0]==state1[0]:
            return 0.8
        elif (state2[0]==state1[0]-1 and state2[1]==state1[1]) or (state2[0]==state1[0]+1 and state2[1]==state1[1]):
            return 0.1
        else:
            return 0.0
    elif action=='E':
        if state2[0]==state1[0]+1 and state2[1]==state1[1]:
            return 0.8
        elif state2[0]==state1[0]-1 and state2[1]==state1[1]:
            return 0.0
        elif (state2[1]==state1[1]-1 and state2[0]==state1[0]) or (state2[1]==state1[1]+1 and state2[0]==state1[0]):
            return 0.1
        else:
            return 0.0
    elif action=='S':
        if state2[1]==state1[1]+1 and state2[0]==state1[0]:
            return 0.8
        elif state2[1]==state1[1]-1 and state2[0]==state1[0]:
            return 0.0
        elif (state2[0]==state1[0]-1 and state2[1]==state1[1]) or (state2[0]==state1[0]+1 and state2[1]==state1[1]):
            return 0.1
        else:
            return 0.0
    elif action=='W':
        if state2[0]==state1[0]+1 and state2[1]==state1[1]:
            return 0.0
        elif state2[0]==state1[0]-1 and state2[1]==state1[1]:
            return 0.8
        elif (state2[1]==state1[1]-1 and state2[0]==state1[0]) or (state2[1]==state1[1]+1 and state2[0]==state1[0]):
            return 0.1
        else:
            return 0.0
    
    return 0.0

def getStates(world):
    states = []
    
    for i in range(len(world)):
        for j in range(len(world[0])):
            if not world[i][j]==0:
                states.append((j,i))
    
    return states

def compute_optimal_bellman(V, states, init_state):
    somme_action = [0.0, 0.0, 0.0, 0.0]
    for a in range(4):
        somme = 0.0
        for s in states:
                if not s==init_state:
                    somme += compute_probability(init_state, s, actions[a])*V[s[1]][s[0]]
        somme_action[a] = gamma*somme
    
    return R[init_state[1]][init_state[0]] + np.max(somme_action), actions[np.argmax(somme_action)]
    
def value_iteration(states):
    V = np.array([[0., 0., 0., 1],
                  [0., 0., 0., -1],
                  [0., 0., 0., 0.]])
    
    policy = np.array([['N', 'N', 'N', 'N'],
                       ['N', 'X', 'N', 'N'],
                       ['N', 'N', 'N', 'N']])
    
    for it in range(iteration):
        for s in states:
            if not s in fixed_cell:
                V[s[1]][s[0]], policy[s[1]][s[0]] = compute_optimal_bellman(V, states, s)
                    
    return V, policy

if __name__=="__main__":
    
    states = getStates(map)
    print("Map :")
    print(map)
    print("\n")
    
    value_fn, policy_fn = value_iteration(states)
    
    print("Optimal value function and policy :")
    print(value_fn)
    print(policy_fn)    

