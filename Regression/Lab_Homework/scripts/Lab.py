# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np

def compute_error(Y,Y_train):
    I = len(Y)
    
    somme = 0.0
    for i in range(I):
        somme += np.linalg.norm(Y[i] - Y_train[i])**2
    
    return 1/2*somme
        
def construct_train_data(input_data, input_dimension, output_dimension, tests_number):
    N=input_dimension
    J=output_dimension
    I = len(input_data) - N - tests_number*N
    I_test = tests_number
    
    X_train = np.zeros((I,N))
    Y_train = np.zeros((I,J))
    
    X_test = np.zeros((I_test,N))
    Y_test = np.zeros((I_test,J))
    
    for i in range(I):
        for n in range(N):
            X_train[i,n] = input_data[i+n]
            
        Y_train[i,0] = input_data[(i+1)+N]
        
    for i in range(I_test):
        for n in range(N):
            X_test[i,n] = input_data[n+(i+I)]
            
        Y_test[i,0] = input_data[(i+I+1)+N]
        
    return X_train, Y_train, X_test, Y_test
    
def bgd(X,Y, alpha):
    I = len(X)
    N = len(X[0])
    
    X_ = np.concatenate((np.ones((I,1)), X), 1) # I x (N+1)
    
    theta = np.random.rand(N+1,1) # (N+1) x 1
    E = compute_error(compute(X,theta), Y)
    while E > 0.1 :
        for n in range(N+1):
            h = (X_).dot(theta)
            somme = 0.0
            for i in range(I):
                somme += (h[i,0] - Y[i,0])*X_[i,n]
            theta[n] = theta[n] - alpha*somme
        E = compute_error(compute(X,theta), Y)
        print(E)
    return theta

def sgd(X,Y, alpha):
    I = len(X)
    N = len(X[0])
    
    X_ = np.concatenate((np.ones((I,1)), X), 1) # I x (N+1)
    
    theta = np.random.rand(N+1,1) # (N+1) x 1
    
    E = compute_error(compute(X,theta), Y)
    while E > 0.01 :
        for n in range(N+1):
            h = (X_).dot(theta)
            i = np.random.default_rng().integers(0,I)
            theta[n] = theta[n] - alpha*((h[i,0] - Y[i,0])*X_[i,n])
        E = compute_error(compute(X,theta), Y)
        print(E)

    return theta

def cfs(X,Y):
    I = len(X)
    
    X_ = np.concatenate((np.ones((I,1)), X), 1) # I x (N+1)
    
    theta = (np.linalg.inv((X_.T).dot(X_))).dot(X_.T).dot(Y)
    return theta

def compute(X, theta):
    I = len(X)
    
    X_ = np.concatenate((np.ones((I,1)), X), 1) # I x (N+1)
    Y = np.zeros((I,1))
    
    Y = (X_).dot(theta)
    
    return Y

if __name__ == "__main__":
    
    t, z = np.loadtxt('../data.csv', delimiter=',', unpack=True, skiprows=1)
    
    norme = np.linalg.norm(z)
    z = z/norme
    
    X_train, Y_train, X_test, Y_test = construct_train_data(z, 150, 1, 3)
    theta_bgd = bgd(X_train, Y_train, 0.002)
    theta_sgd = sgd(X_train, Y_train, 0.9)
    theta_cfs = cfs(X_train, Y_train)
    
    y_bgd = compute(X_train, theta_bgd)*norme
    y_sgd = compute(X_train, theta_sgd)*norme
    y_cfs = compute(X_train, theta_cfs)*norme
    
    # print(y_bgd)
    # print(y_sgd)
    # print(y_cfs)
    
    # print(X_train)
    # print(Y_train)
    # print("\n")
    # print(X_test)
    # print(Y_test)
    # print(theta)
    
    plt.plot(t,z*norme)
    plt.plot(t[150:657+150], y_bgd, label="bgd")
    plt.plot(t[150:657+150], y_sgd, label="sgd")
    plt.plot(t[150:657+150], y_cfs, label="cfs")
    plt.xlabel('t')
    plt.ylabel('z')
    plt.legend()
    plt.show()