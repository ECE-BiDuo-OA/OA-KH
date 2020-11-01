# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

n_neurons_hidden = 10
alpha1 = alpha2 = 0.1
iteration_max = 100

"""
Return a list that contains the label associated with the output of the neurons network
    G: the output array of the ANN
    labels : the list of available labels
"""
def find_label(G, labels):
    I = len(G)
    label = np.zeros((I,1))
    for i in range(I):
        label[i] = labels[np.argmax(G[i])]
    
    return label

"""
Return a list that contains the avalaible labels in the output data
    output_data: the output array from the data file
"""
def getLabels(output_data):
    I = len(output_data)
    
    labels = []
    for i in range(I):
        if output_data[i] not in labels:
                labels.append(output_data[i])
    
    return labels

"""
This function adapts the output from the data file in order to be used in the ANN

Return an array of I rows and M columns, each row represents an data entry and each column represents a label
An element (i,m) of the array is equal to 1 when the data entry i has the label m, otherwise (i,m) is equal to 0
    output_data: the output array from the data file
    labels : the list of available labels
"""
def adapt_output_data(output_data, labels):
    I = len(output_data)
    M = len(labels)
    
    Y = np.zeros((I,M))
    for i in range(I):
        for j in range(M):
            if output_data[i] == labels[j]:
                Y[i][j] = 1
                break
    
    return Y

"""
This function computes the error

Return the quadratic error between the predicted output and the expected output 
    G: the output array of the ANN
    Y : the expected output array    
"""
def compute_error(G, Y):
    I = len(G)
    J = len(G[0])
    
    E = 0
    for i in range(I):
        for j in range(J):
            E += (G[i,j] - Y[i,j])**2
    
    E = 1/2*E
    
    return E

"""
This function computes the forward propagation of the ANN

Return the output array of the ANN and the output array of the hidden layer 
    input_data: the data entry array
    V : the value array of the neurons in the hidden layer
    W : the value array of the neurons in the output layer
"""
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

"""
This function computes the back propagation of the ANN

Return the value array of the neurons in the hidden layer and the value array of the neurons in the output layer
    input_data: the data entry array
    Y : the expected output array
    G : the output array of the ANN
    F : the output array of the hidden layer 
    V : the value array of the neurons in the hidden layer
    W : the value array of the neurons in the output layer
"""
def fbnn(input_data, Y, G, F, V, W):
    X = input_data[:]
    N = len(X[0])+1
    I = len(X)
    K = len(V[0])+1
    J = len(W[0])
    
    X_ = np.concatenate((np.ones((I,1)), X), axis=1)
    F_ = np.concatenate((np.ones((I,1)), F), axis=1)
    
    #Compute W
    for k in range(K):
        for j in range(J):
            somme = 0
            for i in range(I):
                somme += (G[i,j] - Y[i,j]) * G[i,j] * (1-G[i,j]) * F_[i,k]
                
            W[k,j] = W[k,j] - alpha1 * somme
            
    #Compute V
    for n in range(N):
        for k in range(K-1):
            somme = 0
            for i in range(I):
                for j in range(J):
                    somme += (G[i,j] - Y[i,j]) * G[i,j] * (1-G[i,j]) * W[k,j] * F[i,k] * (1 - F[i,k]) * X_[i,n]
                
            V[n,k] = V[n,k] - alpha2 * somme
    
    return V, W

"""
This function trains the ANN model

Return the value array of the neurons in the hidden layer and the value array of the neurons in the output layer 
and the list of error at each iteration
    input_data: the data entry array
    Y : the expected output array
    iteration : the number of iteration for training the model
"""
def train_nn(input_data, Y, iteration):
    X = input_data[:]
    N = len(X[0])
    J = len(Y[0])
    
    V = np.random.rand(N+1,n_neurons_hidden) # N+1 x K
    W = np.random.rand(n_neurons_hidden+1,J) # K+1 x J

    E = []
    optimal_VW = []
    for i in range(iteration):
        G, F = ffnn(X, V, W)
        V, W = fbnn(X, Y, G, F, V, W)
        optimal_VW.append((V,W))
        E.append(compute_error(G, Y))
    
    V = optimal_VW[np.argmin(E)][0]
    W = optimal_VW[np.argmin(E)][1]
    
    return V, W, E
    
if __name__ == "__main__" :
    
    file = open("../data_ffnn.txt", "r")
    dataset = np.loadtxt(file, delimiter="\t", skiprows=1)
    
    input_data = dataset[:, 0:3]
    output_data = dataset[:, 3:4]
    labels = getLabels(output_data)
    
    Y = adapt_output_data(output_data, labels)
    
    #Question 2
    N = len(input_data[0])
    J = len(Y[0])
    
    V = np.random.rand(N+1,n_neurons_hidden) # N+1 x K
    W = np.random.rand(n_neurons_hidden+1,J) # K+1 x J
    
    G, F = ffnn(input_data, V, W)
    E = compute_error(G, Y)
    print("Error with non trained model :")
    print(E)
    
    
    #Question 3 : Train the ANN
    print("\nTraining with {} iterations and alpha1 = {}, alpha2 = {}".format(iteration_max, alpha1, alpha2))
    V,W,E = train_nn(input_data, Y, iteration_max)
    
    #Question 4 : Show that your algorithm converges by illustrating the error reduction at each iteration.
    print("\nError at each iteration :")
    print(E)
    
    #Question 5 : What are the optimal parameter values for the hidden layer (v) and for the output layer (ω)?
    print("\nOptimal V =")
    print(V)
    print("\nOptimal W =")
    print(W)

    #Question 6 : Show that your classifier works properly by comparing the predicted output values to the
    #             actual training output values.
    print("\nComparing the predicted outputs with the actual outputs")
    G, F = ffnn(input_data, V, W)
    print("Output =")
    print(find_label(G, labels))
    print("\nIs the predicted output equals to actual output ?")
    print((find_label(G, labels)==output_data).all())
    
    #Question 7 : Test your optimized model by doing forward propagation over the following test data set:
    #                 (x1, x2, x3)=(2, 2, −3), (x1, x2, x3)=(3, 4, 3), and (x1, x2, x3)=(4.5, 1.5, 0).
    print("\nPredicting outputs with new inputs")
    input_data = np.array([[  2,   2,   -3],
                           [  3,   4,    3],
                           [4.5, 1.5,    0]])
    
    G, F = ffnn(input_data, V, W)
    print("Output =")
    print(find_label(G, labels))