# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
    
def ffnn(input_data, output_data, V, W):
    X[:] = input_data[:]
    Y[:] = output_data[:]
    
    N = len(X[0])
    I = len(X)
    K = len(V[0])
    J = len(W[0])
    
    X_ = np.hstack((np.ones((I,1)), X)) #I x N+1
    X__ = X_.dot(V) # I x K
    
    F = np.array([1/(1+np.exp(-x__)) for x__ in X__]) #I x K
    F_ = np.hstack((np.ones((I,1)), F)) # I x (N+1)  
    F__ = F_.dot(W) # I x J
    
    G = np.array([1/(1+np.exp(-x__)) for x__ in F__]) # I x J
    
    print(G)
    
    E = 0
    for i in range(I):
        for j in range(J):
            E += (G[i,j] - Y[i,j])**2
    
    E = 1/2*E
    
    return E
    
if __name__ == "__main__" :
    file = open("../data_ffnn.txt", "r")
    dataset = np.loadtxt(file, delimiter="\t", skiprows=1)
    
    X = dataset[:, 0:3]
    Y = dataset[:, 3:4]
    print(Y)
    
    N = len(X[0])
    I = len(X)
    K = 3
    J = 1
    
    V = np.random.rand(N+1,K) # N+1 x K
    W = np.random.rand(N+1,J) # N+1 x J
    
    print(ffnn(X, Y, V, W))