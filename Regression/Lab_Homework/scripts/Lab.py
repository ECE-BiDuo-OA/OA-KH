# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np

def compute_error(Y_predict,Y_train):
    I = len(Y_predict)
    J = len(Y_predict[0])
    
    somme = 0.0
    for i in range(I):
        for j in range(J):
            somme += (Y_predict[i,j] - Y_train[i,j])**2
    
    return somme/2
        
def construct_train_data(input_data, input_dimension, output_dimension, tests_number):
    N=input_dimension
    J=output_dimension
    I = len(input_data) - N - J - tests_number
    I_test = 3
    
    X_train = np.zeros((I,N))
    Y_train = np.zeros((I,J))
    
    X_test = np.zeros((I_test,N))
    Y_test = np.zeros((I_test,J))
    
    for i in range(I):
        for n in range(N):
            X_train[i,n] = input_data[i+n]
            
        for j in range(J):
            Y_train[i,j] = input_data[N+(i+j)]
        
    for i in range(I_test):
        for n in range(N):
            X_test[i,n] = input_data[n+(i+I)]
        
        for j in range(J):
            Y_test[i,j] = input_data[I+N+(i+j)]
        
    return X_train, Y_train, X_test, Y_test
    
def bgd(X, Y, alpha, epsilon):
    J = len(Y[0])
    I = len(X)
    N = len(X[0])
    
    X_ = np.concatenate((np.ones((I,1)), X), 1) # I x (N+1)
    
    theta = np.random.rand(N+1,J) # (N+1) x J
    theta_old = np.zeros((N+1,J)) # (N+1) x J
    
    while not (np.abs(theta-theta_old) < epsilon).all():
        theta_old[:] = theta[:]
        for n in range(N+1):
            Y_predict = (X_).dot(theta)
            somme = 0
            
            for i in range(I):
                somme += (Y_predict[i] - Y[i])*X_[i,n]
                
            theta[n] = (theta[n] - alpha*somme)
            
        E = compute_error(compute(X,theta), Y)
        
        print(E)
        
    return theta

def sgd(X,Y, alpha, epsilon):
    J = len(Y[0])
    I = len(X)
    N = len(X[0])
    
    X_ = np.concatenate((np.ones((I,1)), X), 1) # I x (N+1)
    
    theta = np.random.rand(N+1,J) # (N+1) x 1 
    theta_old = np.zeros((N+1,J)) # (N+1) x J
    
    while not (np.abs(theta-theta_old) < epsilon).all():
        theta_old[:] = theta[:]
        for n in range(N+1):
            Y_predict = (X_).dot(theta)
            i = np.random.default_rng().integers(0,I)
            
            theta[n] = theta[n] - alpha*((Y_predict[i] - Y[i])*X_[i,n])
            
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
    J = len(theta[0])
    
    X_ = np.concatenate((np.ones((I,1)), X), 1) # I x (N+1)
    Y = np.zeros((I,J))
    
    Y = (X_).dot(theta)
    
    return Y

if __name__ == "__main__":
    
    #Question 1
    t, z = np.loadtxt('../data.csv', delimiter=',', unpack=True, skiprows=1)
    plt.plot(t,z)
    
    #To avoid data overflow
    norme = np.max(z)#np.linalg.norm(z)
    z = z/norme
    
    #Question 2
    X_train, Y_train, X_test, Y_test = construct_train_data(z, 150, 30, 3)
    
    #Question 3
    theta_bgd = bgd(X_train, Y_train, 0.00001, 0.0003)
    print("Optimal values of the parameters compute with BGD :")
    print(theta_bgd)
    
    #Question 4
    theta_sgd = sgd(X_train, Y_train, 0.01, 0.001)
    print("Optimal values of the parameters compute with SGD :")
    print(theta_sgd)
    
    #Question 5
    theta_cfs = cfs(X_train, Y_train)
    print("Optimal values of the parameters compute with CFS :")
    print(theta_cfs)
    
    y_bgd = compute(X_train, theta_bgd)*norme
    
    #y_sgd = compute(X_train, theta_sgd)*norme
    
    y_cfs = compute(X_train, theta_cfs)*norme
    
    y_bgd_test = compute(X_test, theta_bgd)*norme
    #y_sgd_test = compute(X_test, theta_sgd)*norme
    y_cfs_test = compute(X_test, theta_cfs)*norme
    
    t_train = t[len(X_train[0]):len(X_train)+len(X_train[0])]
    t_test = t[len(X_train)+len(X_train[0]):-1]
    
    plt.plot(t_train, y_bgd[:,0], label="bgd")
    #plt.plot(t_train, y_sgd[:,0], label="sgd")
    plt.plot(t_train, y_cfs[:,0], label="cfs")
    
    plt.plot(t_test, y_bgd_test, label="bgd_test")
    #plt.plot(t_test, y_sgd_test, label="sgd_test")
    plt.plot(t_test, y_cfs_test, label="cfs_test")
    plt.xlabel('t')
    plt.ylabel('z')
    plt.legend()
    plt.show()