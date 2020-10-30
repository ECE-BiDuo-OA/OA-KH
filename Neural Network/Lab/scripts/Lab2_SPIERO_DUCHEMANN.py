# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
    
def ffnn(input_data, V, W):
    X = input_data[:]
    
    I = len(X)
    
    X_ = np.hstack((np.ones((I,1)), X)) #I x N+1
    X__ = X_.dot(V) # I x K
    
    F = np.array([1/(1+np.exp(-x__)) for x__ in X__]) #I x K
    F_ = np.hstack((np.ones((I,1)), F)) # I x (N+1)  
    F__ = F_.dot(W) # I x J
    
    G = np.array([1/(1+np.exp(-x__)) for x__ in F__]) # I x J
    
    return G, X_, F, F_

def compute_error(G, Y, X, W):
    I = len(X)
    J = len(W[0])
    
    E = 0
    for i in range(I):
        for j in range(J):
            E += (G[i,j] - Y[i,j])**2
    
    E = 1/2*E
    
    return E

def fbnn(X, X_, Y, G, F, F_, V, W):
    
    alpha1 = 0.08
    alpha2 = 0.08
    
    N = len(X[0])+1
    I = len(X)
    K = len(V[0])+1
    J = len(W[0])
    
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

def train_nn(input_data, output_data, iteration):
    
    X = input_data[:]
    Y = output_data[:]
    
    N = len(X[0])
    K = 3
    J = 1
    
    V = np.zeros((N+1,K)) # N+1 x K
    W = np.zeros((N+1,J)) # N+1 x J

    E = []
    for i in range(iteration):
        G, X_, F, F_ = ffnn(X, V, W)
        V, W = fbnn(X,X_,Y,G,F,F_,V,W)
        E.append(compute_error(G, Y, X, W))
        print(E[i])
        
    i_optimal = np.argmin(E)
    
    V = np.zeros((N+1,K)) # N+1 x K
    W = np.zeros((N+1,J)) # N+1 x J
    
    E = []
    for i in range(i_optimal+1):
        G, X_, F, F_ = ffnn(X, V, W)
        V, W = fbnn(X,X_,Y,G,F,F_,V,W)
        E.append(compute_error(G, Y, X, W))
    
    return V,W,E[i_optimal]
    
if __name__ == "__main__" :
    file = open("../data_ffnn.txt", "r")
    dataset = np.loadtxt(file, delimiter="\t", skiprows=1)
    
    X = dataset[:, 0:3]
    Y = dataset[:, 3:4]
    
    norm_max = np.max(Y)
    
    Y = Y/norm_max
    
    #Train the ANN
    V,W,E = train_nn(X,Y,1000)
    # print(V)
    # print(W)
    # print(E)
    
    # V = np.array([[ 0.86965597,  0.92171657,  0.57626106],
    #               [ 3.19111108,  2.30370477, -4.58865742],
    #               [ 1.80519308,  0.18978403,  6.96393871],
    #               [-0.05953019, -6.19977945, -0.03836276]])
    
    # W = np.array([[  0.14293218],
    #               [ 10.60326524],
    #               [ -4.9337514 ],
    #               [-10.76768206]])
    
    #Comparing with data output
    G, X_, F, F_ = ffnn(X, V, W)
    E = compute_error(G, Y, X, W)
    G = np.round(G*norm_max)
    print(G)
    
    X = np.array([[  2,   2,   -3],
                  [  3,   4,    3],
                  [4.5, 1.5,    0]])
    
    G, X_, F, F_ = ffnn(X, V, W)
    E = compute_error(G, Y, X, W)
    G = np.round(G*norm_max)
    print(G)