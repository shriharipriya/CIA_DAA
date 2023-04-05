# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 22:58:22 2023

@author: Hari
"""

import numpy as np
import pandas as pd
import pygad
from sklearn.model_selection import train_test_split

# Defining neural network architecture
inlayer = 2
hidlayer = 5
outlayer = 1
num_weights = (inlayer * hidlayer) + hidlayer + (hidlayer * outlayer) + outlayer

def neural_net(weights, inlayer, hidlayer, outlayer,inputs):
    # Reshaping weights
    W1 = np.reshape(weights[0:hidlayer*inlayer], (inlayer, hidlayer))
    b1 = np.reshape(weights[hidlayer*inlayer:hidlayer*inlayer+hidlayer], (1, hidlayer))
    W2 = np.reshape(weights[hidlayer*inlayer+hidlayer:hidlayer*inlayer+hidlayer+hidlayer*outlayer], (hidlayer, outlayer))
    b2 = np.reshape(weights[hidlayer*inlayer+hidlayer+hidlayer*outlayer:], (1, outlayer))

    # Computing output of the neural network
    z1 = np.dot(inputs, W1) + b1
    a1 = np.tanh(z1)#activation function
    z2 = np.dot(a1, W2) + b2
    output = np.sigmoid(z2)
    return output

def fitness_function(weights,target):
    # Evaluate the fitness of each set of weights using the mean squared error
    output = neural_net(weights, inlayer, hidlayer, outlayer,ytestl)
    acc = (output - target).mean()
    return acc

#input=target=ytest,output=ypred
df=pd.read_csv("D:\\SNU\\Sem4\\MLT\\clsfctn\\classification.csv")
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)
ytestl=list(ytest)
# Create an instance of the pygad.GA class
ga = pygad.GA(num_generations=50, 
              num_parents_mating=4, 
              fitness_func=fitness_function, 
              sol_per_pop=10, 
              num_genes=num_weights)

ga.run()
best_weights = ga.best_solution()

# Evaluate the neural network using the best set of weights
output = neural_net(best_weights, inlayer, hidlayer, outlayer,ytestl)
print("Best set of weights found: ", best_weights)
print("Output of neural network: ", output)
