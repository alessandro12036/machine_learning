#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:09:24 2019

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
from RBF import Model, FeatureCreator, MySGDRegressor



class Model2(Model):  
    
    def train(self, env, steps=100, gamma=0.99, n_steps=5):
        
        print("Using new version of Model")
        multiplier = ([gamma]*n_steps) ** np.arange(n_steps)
        states = []
        actions = []
        rewards = []
        epsilon = 0.2
        
        for i in range(steps):
            
            s1 = env.reset()
            done = False
            iterations = 0
            
            while not done and iterations <= 10000:
                
                iterations += 1
                
                if np.random.rand() <= epsilon:
                    a1 = env.action_space.sample()
                else:
                    a1 = np.argmax(self.predict_all(s1))
                    
                states.append(s1)
                actions.append(a1)
                
                s2, reward, done, _ = env.step(a1)
                rewards.append(reward)
                done = s2[0] >= env.goal_position
                
                if len(rewards) >= n_steps:
                    
                    returns_up_to_n = multiplier * rewards
                    final_term = (gamma ** n_steps) * np.max(self.predict_all(s2))
                    target = np.sum(np.concatenate((returns_up_to_n, [final_term])))
                    self.update(states[0], target, actions[0])
                    states.pop(0)
                    actions.pop(0)
                    rewards.pop(0)
                
                s1 = s2
            
            # the game goes faster than our updating so we have to finish things up
            if done:
                while len(rewards) > 0:
                    final_returns = multiplier[:len(rewards)] * rewards
                    target = np.sum(final_returns)
                    self.update(states[0], target=target, index=actions[0])
                    states.pop(0)
                    actions.pop(0)
                    rewards.pop(0)
            else:
                while len(rewards) > 0:
                    guess_rewards = np.concatenate((rewards, [-1]*(n_steps-len(rewards))))
                    target = np.sum(multiplier * guess_rewards)
                    self.update(states[0], target, actions[0])
                    states.pop(0)
                    actions.pop(0)
                    rewards.pop(0)
            
            assert len(rewards) == len(states) == len(actions) == 0
            print("Epoch {} took {} iterations".format(i, iterations))
            

env = gym.make("MountainCar-v0")
feat_ct = FeatureCreator(env)

model = Model2(feat_ct, env)
model.train(env)