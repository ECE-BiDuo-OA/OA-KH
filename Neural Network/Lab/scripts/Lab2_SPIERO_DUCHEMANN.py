# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

n_neurons_hidden = 10
alpha1 = alpha2 = 0.1
iteration_max = 500

def find_label(G, labels):
    I = len(G)
    label = np.zeros((I,1))
    for i in range(I):
        label[i] = labels[np.argmax(G[i])]
    
    return label

def getLabels(output_data):
    I = len(output_data)
    
    labels = []
    for i in range(I):
        if output_data[i] not in labels:
                labels.append(output_data[i])
    
    return labels
    
def normalize_output_data(output_data, labels):
    I = len(output_data)
    M = len(labels)
    
    Y = np.zeros((I,M))
    for i in range(I):
        for j in range(M):
            if output_data[i] == labels[j]:
                Y[i][j] = 1
                break
    
    return Y

def compute_error(G, Y, X, W):
    I = len(X)
    J = len(W[0])
    
    E = 0
    for i in range(I):
        for j in range(J):
            E += (G[i,j] - Y[i,j])**2
    
    E = 1/2*E
    
    return E
    
def ffnn(input_data, V, W):
    X = input_data[:]
    
    I = len(X)
    
    X_ = np.concatenate((np.ones((I,1)), X), axis=1) #I x N+1
    X__ = X_.dot(V) # I x K
    
    F = np.array([1/(1+np.exp(-x__)) for x__ in X__]) #I x K
    F_ = np.concatenate((np.ones((I,1)), F), axis=1) # I x (K+1)  
    F__ = F_.dot(W) # I x J
    
    G = np.array([1/(1+np.exp(-x__)) for x__ in F__]) # I x J
    
    return G, F

def fbnn(X, Y, G, F, V, W):    
    N = len(X[0])+1
    I = len(X)
    K = len(V[0])+1
    J = len(W[0])
    
    X_ = np.concatenate((np.ones((I,1)), X), axis=1)
    F_ = np.concatenate((np.ones((I,1)), F), axis=1)
    
    for k in range(K):
        for j in range(J):
            somme = 0
            for i in range(I):
                somme += (G[i,j] - Y[i,j]) * G[i,j] * (1-G[i,j]) * F_[i,k]
                
            W[k,j] = W[k,j] - alpha1 * somme
            
    for n in range(N):
        for k in range(K-1):
            somme = 0
            for i in range(I):
                for j in range(J):
                    somme += (G[i,j] - Y[i,j]) * G[i,j] * (1-G[i,j]) * W[k,j] * F[i,k] * (1 - F[i,k]) * X_[i,n]
                
            V[n,k] = V[n,k] - alpha2 * somme
    
    return V, W

def train_nn(X, Y, iteration):
    N = len(X[0])
    J = len(Y[0])
    
    V = np.random.rand(N+1,n_neurons_hidden) # N+1 x K
    W = np.random.rand(n_neurons_hidden+1,J) # K+1 x J

    E = []
    for i in range(iteration):
        G, F = ffnn(X, V, W)
        V, W = fbnn(X, Y, G, F, V, W)
        E.append(compute_error(G, Y, X, W))
    
    return V, W, E
    
if __name__ == "__main__" :
    
    file = open("../data_ffnn.txt", "r")
    dataset = np.loadtxt(file, delimiter="\t", skiprows=1)
    
    input_data = dataset[:, 0:3]
    output_data = dataset[:, 3:4]
    labels = getLabels(output_data)
    
    Y = normalize_output_data(output_data, labels)
    
    #Train the ANN
    print("\nTraining with {} iterations and alpha1 = {}, alpha2 = {}".format(iteration_max, alpha1, alpha2))
    V,W,E = train_nn(input_data, Y, iteration_max)
    print("\nV =")
    print(V)
    print("\nW =")
    print(W)
    #print("Error = {}".format(E))
    
    # V = np.array([[ 0.86965597,  0.92171657,  0.57626106],
    #               [ 3.19111108,  2.30370477, -4.58865742],
    #               [ 1.80519308,  0.18978403,  6.96393871],
    #               [-0.05953019, -6.19977945, -0.03836276]])
    
    # W = np.array([[  0.14293218],
    #               [ 10.60326524],
    #               [ -4.9337514 ],
    #               [-10.76768206]])

    #Comparing with data output
    print("\nComparing the predicted outputs with the actual outputs")
    G, F = ffnn(input_data, V, W)
    E = compute_error(G, Y, input_data, W)
    print("Output =")
    #print(find_label(G, labels))
    
    print("\nPredicting outputs with new inputs")
    input_data = np.array([[  2,   2,   -3],
                           [  3,   4,    3],
                           [4.5, 1.5,    0]])
    
    G, F = ffnn(input_data, V, W)
    E = compute_error(G, Y, input_data, W)
    print("Output =")
    print(find_label(G, labels))