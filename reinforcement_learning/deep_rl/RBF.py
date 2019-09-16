#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:41:16 2019

@author: alessandro
"""

import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor


class MySGDRegressor:
    
    def __init__(self, learning_rate):
        
        self.learning_rate = learning_rate
        self.not_initialized = True
        self.coef_ = None
        self.lr = 0.01
        
    
    def partial_fit(self, X, y):
        
        if self.not_initialized:
            self.not_initialized = False
            self.coef_ = np.random.randn(X.shape[1]).reshape(-1, 1)
            self.coef_ = self.coef_ * np.sqrt(1 / X.shape[1])
        
        z = self.predict(X)[0]
        gradient = ((z-y) * X).T
        assert gradient.shape == self.coef_.shape
        self.coef_ = self.coef_ - (self.lr * gradient)
            
    
    def predict(self, X):
        
        z = np.dot(X, self.coef_)
        return z


class FeatureCreator:
    
    def __init__(self, env, challenge="MountainCar", n_components=500, gamma_values=[0.5, 1.0, 2.0, 5.0]):
        
        self.scaler = StandardScaler()
        
        if challenge == "CartPole":
            sample_observations = np.array(self.collectSamples(env))   # bisognerebbe vedere se un maggior numero di campioni aiuta
            #sample_observations = np.random.random((2000, 4)) * 2 - 2   # Funziona meglio ma con gamma [0.5, 1.0, 0.1, 0.05]
            
        else:
            sample_observations = np.array([env.observation_space.sample() for i in range(10000)])
        
        featurer_list = []
        
        for c, gamma in enumerate(gamma_values, 1):
            featurer_list.append(("rbfs{}".format(c), RBFSampler(gamma=gamma, n_components=n_components)))
        
        self.featurer = FeatureUnion(featurer_list)
    
        self.scaler.fit(sample_observations)
        self.featurer.fit(self.scaler.transform(sample_observations))
        
    
    def transform(self, x):
        
        x = np.array(x).reshape(1, -1)
        assert len(x.shape) == 2
        scaled = self.scaler.transform(x)
        
        return self.featurer.transform(scaled)
    
    
    def collectSamples(self, env):
        
        observations = []
        
        for i in range(1000):
            
            ob = env.reset()
            done = False
            
            while not done:
                observations.append(ob)
                action = env.action_space.sample()
                ob, reward, done, _ = env.step(action)
                
        return observations


class Model:
    
    def __init__(self, ft_creat, env):
        
        self.units = []
        self.ft_creat = ft_creat
        n_units = env.action_space.n
        
        dummy_variables = self.ft_creat.transform(env.reset())
        
        for m in range(n_units):
            reg = MySGDRegressor(learning_rate = "constant")
            reg.partial_fit(dummy_variables, [0])
            self.units.append(reg)
        
    
    def predict_all(self, inputs):
        
        predictions = []
        inputs = self.ft_creat.transform(inputs)
        for unit in self.units:
            predictions.append(unit.predict(inputs)[0])
        return np.array(predictions, ndmin=1)
            
    
    def update(self, X, target, index):
        X = self.ft_creat.transform(X)
        self.units[index].partial_fit(X=X, y=[target])
        
        
    def train(self, env, steps=100, gamma=0.99):
        
        iterations = 0
        reward_means = []
        
        for i in range(steps):
            if steps > 1000:
                epsilon = 1.0 / np.sqrt(i)
            else:
                epsilon = 0.1 * (0.97 ** i)
            print("Epoch {}; previous number of iterations: {}".format(i, iterations))
            s1 = env.reset()
            done = False
            total_reward = 0
            iterations = 0
            
            while not done and iterations < 10000:
                if np.random.rand() < epsilon:
                    a1 = env.action_space.sample()
                else:
                    a1 = np.argmax(self.predict_all(s1))
                iterations += 1
                s2, reward, done, _ = env.step(a1)
                if env.unwrapped.spec.id == "MountainCar-v0":  # We don't want it to cap at 200
                    done = bool(s2[0] >= env.goal_position and s2[1] >= env.goal_velocity)
                if done:
                    break
                predictions = self.predict_all(s2)
                target = reward + gamma * np.max(predictions)
                self.update(X=s1, target=target, index=a1)
                total_reward += reward
                s1 = s2
            
            reward_means.append(total_reward)
            print()
        return reward_means
        

    def playOne(self, env, record=False):
        
        if record:
            env = wrappers.Monitor(env, "./video", force=True)
        done = False
        tot_r = 0
        iterations = 0
        ob = env.reset()
        
        while not done:
            iterations += 1
            if iterations >= 10000:
                print("Failed")
                break
            preds = self.predict_all(ob)
            a = np.argmax(preds)
            ob, reward, done, _ = env.step(a)
            if env.unwrapped.spec.id == "MountainCar-v0":
                done = bool(ob[0] >= env.goal_position and ob[1] >= env.goal_velocity)
            tot_r += 1
        
        return tot_r


def test_mountain_car():
    
    env = gym.make("MountainCar-v0")
    print("Testing {}".format(env.unwrapped.spec.id))
    
    ft_creat = FeatureCreator(env)
    model = Model(ft_creat, env)
    log = model.train(env, 300)
    
    final_means = []
        
    for i in range(10):
        r = model.playOne(env)
        final_means.append(r)
        
    print(np.mean(final_means))
    
    
  
def test_cartPole():
    
    env = gym.make("CartPole-v0")
    print("Testing {}".format(env.unwrapped.spec.id))
    
    ft_creat = FeatureCreator(env, challenge="CartPole", gamma_values=[0.5, 1.0, 0.1, 0.05]) # non è detto che queste gamma vadano bene per la raccolta di campioni
    model = Model(ft_creat, env)
    
    log = model.train(env, 10000, gamma=0.9)
    
    final_means = []
    
    for i in range(10):
        r = model.playOne(env)
        final_means.append(r)
    
    print(np.mean(final_means))
    

if __name__ == "__main__":
   test_mountain_car()
   test_cartPole()
