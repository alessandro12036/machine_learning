#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:28:49 2019

@author: alessandro
"""

import gym
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from RBF import FeatureCreator



class MySGDRegressor:
    
    def __init__(self, input_size, lr=0.001):
        
        self.W = np.random.randn(input_size) / np.sqrt(input_size)
        self.lr = lr
    
    def predict(self, X):
        
        pred = np.dot(X, self.W)
        return pred
    
    def partial_fit(self, X, target, eligibilities):  # eligibilities has shape X
        
        y_pred = self.predict(X)
        diff = y_pred - target
        gradient = diff * eligibilities
        self.W -= self.lr * gradient
        

class Model:
    
    def __init__(self, n_nodes, input_size, ft_creat, lr=0.001):
        
        self.models = []
        self.ft_creat = ft_creat
        
        for i in range(n_nodes):
            m = MySGDRegressor(input_size=input_size, lr=lr)
            self.models.append(m)
        
    
        
    def train(self, env, epochs, gamma = 0.99, epsilon=0.5, lambda_=0.7):
        
        for i in range(epochs):
            
            done = False
            s1_raw = env.reset()
            s1 = self.ft_creat.transform(s1_raw)
            iterations = 0
            eligibilities = np.zeros((env.action_space.n, s1.shape[1]))
            print("Eligibilities shape: {}".format(eligibilities.shape))
            
            while not done and iterations <= 10000:
                iterations += 1
                a1_preds = self.predict_all(s1)
                
                if np.random.rand() <= epsilon:
                    a1 = env.action_space.sample()
                
                else:
                    a1 = np.argmax(a1_preds)
                
                s2_raw, reward, done, _ = env.step(a1)
                done = s2_raw[0] >= env.goal_position
                
                s2 = self.ft_creat.transform(s2_raw)
                s2_preds = self.predict_all(s2)
                
                eligibilities *= gamma * lambda_
                eligibilities[a1] += s1.squeeze()
                target = reward + gamma * np.max(s2_preds)
                self.models[a1].partial_fit(s1, target=target, eligibilities=eligibilities[a1])
                
                s1 = s2
                
            if iterations < 200:
                epsilon *= 0.97
            
            print("Epoch {} took {} iterations".format(i, iterations))
            
            
    def predict_all(self, X):
        
        return [m.predict(X) for m in self.models]
    
            
    def playN(self, env, n=10):
        
        log = []
        
        for i in range(n):
            iterations = 0
            s = env.reset()
            done = False
            
            while not done:
                iterations += 1
                s = self.ft_creat.transform(s)
                preds = self.predict_all(s)
                a = np.argmax(preds)
                s, reward, done, _ = env.step(a)
            
            log.append(iterations)
        
        print(np.mean(log))
                

if __name__ == "__main__":
    
    env = gym.make("MountainCar-v0")
    ft_creat = FeatureCreator(env)
    model = Model(n_nodes=env.action_space.n, input_size=2000, ft_creat=ft_creat)
    model.train(env, 300)
                
                
    