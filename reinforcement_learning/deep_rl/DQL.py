#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:08:16 2019

@author: alessandro
"""

import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import socket
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
import pickle
import pandas as pd



tf.reset_default_graph()

class Layer:
    
    def __init__(self, n_units, input_size, layer_name):
        
        with tf.variable_scope(layer_name):
            
            self.W = tf.get_variable(name="W", shape=(input_size, n_units))
            self.b = tf.get_variable(name="b", shape=(1, n_units))
    
    def forward(self, X):
        
        return tf.matmul(X, self.W) + self.b
    
    
class Model:
    
    def __init__(self, architecture, input_size, session, lr, scaler, featurer):
        
        self.architecture = architecture
        self.session = session
        self.layers = []
        self.activations = {}
        self.linear = {}
        self.n_layers = len(architecture)
        self.X = tf.placeholder(tf.float32, shape=(None, input_size))
        self.y = tf.placeholder(tf.float32, shape=(None, architecture[-1]))
        self.scaler = scaler
        self.featurer = featurer

        for i in range(self.n_layers):
            if i == 0:
                inputs_shape = input_size
            else:
                inputs_shape = self.layers[i-1].b.shape[1]
                
            l = Layer(architecture[i], inputs_shape, "layer{}".format(i))
            self.layers.append(l)
        
        for c, l in enumerate(self.layers):
            if c == 0:
                self.linear[c] = l.forward(self.X)
            else:
                self.linear[c] = l.forward(self.activations[c-1])
            if c != self.n_layers - 1:
                print("self.activations[{}]== tf.nn.tanh(self.linear[{}]".format(c, c))
                self.activations[c] = tf.nn.tanh(self.linear[c])
            else:
                print("self.activations[{}]== self.linear[{}]".format(c, c))
                self.activations[c] = self.linear[c]
        
        self.output = self.activations[self.n_layers-1]

        self.loss = tf.losses.mean_squared_error(self.y, self.output)
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.fit_op = self.optimizer.minimize(self.loss)
        self.session.run(tf.global_variables_initializer())
        
        print("Initialized model with parameters:\n architecture = {}\n learning rate = {}".format(architecture, lr))
            
    
    def get_predictions(self, X):
        
        scaled_X = self.scaler.transform(np.array(X).reshape(1, -1))
        transformed_X = self.featurer.transform(scaled_X)
        
        return self.session.run(self.output, feed_dict={self.X:transformed_X})
        
    
    def train(self, s1, a1, r, predictions_s2):
        
        """" sess: the current tensorflow session
             s1, s2: the states we're working on
             r: the current reward"""
        gamma = 0.99
        scaled_s1 = self.scaler.transform(np.array(s1).reshape(1, -1))
        transformed_s1 = self.featurer.transform(scaled_s1)
        
        ret = r + gamma * np.max(predictions_s2)
        predictions_s1 = self.get_predictions(s1)
        predictions_s1[0, a1] = ret
        _, cost = self.session.run([self.fit_op, self.loss], feed_dict={self.X:transformed_s1, 
                                                                        self.y:predictions_s1})
        
        return cost
    
    
def train(env, model, steps, verbose=False, mod_rewards=False):
    
    print("Mod_rewards: ", mod_rewards)
    iterations = 0
    reward_means = []
    final_positions = []
    epsilon = 0.3
    
    for i in range(steps):

        print("Epsilon is now {}".format(epsilon))
        print("Epoch {} took {} iterations".format(i-1, iterations))
        s1 = env.reset()
        done = False
        iterations = 0
        start = time.time()
        counters = [0, 0, 0]
        
        costs = []
        
        while not done and iterations < 10000:
            
            if iterations % 100 == 0 and iterations != 0 and verbose:
                print("100 iterations took {} seconds".format(time.time() - start))
                start = time.time()
            
            all_predictions_s1 = model.get_predictions(s1)
            
            if np.random.rand() < epsilon:
                a1 = env.action_space.sample()
            else:
                a1 = np.argmax(all_predictions_s1)
                
            iterations += 1
            s2, reward, done, _ = env.step(a1)
            done = bool(s2[0] >= env.goal_position and s2[1] >= env.goal_velocity)
            
#            if mod_rewards:
#                if abs(s2[1]) > 0.05:
#                    reward += 15
#                elif abs(s2[1]) > 0.03:
#                    reward += 10
                                    
            all_predictions_s2 = model.get_predictions(s2)
                        
            cost = model.train(s1, a1, reward, all_predictions_s2)
            counters[a1] += 1
            costs.append(cost)
            
            s1 = s2
        
        if iterations < 200:
            epsilon *= 0.98
        
        print("Costs for last epoch were: {}".format(np.mean(costs)))
        print(counters)
        sample_s = env.observation_space.sample()
        
        final_positions.append(s1)
        print(model.get_predictions(sample_s))
    
    ma = calculate_rn_mean(final_positions)
    plt.plot(final_positions, alpha=0.8, label="Final positions")
    plt.plot(ma, label="Running means")
    plt.legend()
    return reward_means # Currently empty


def calculate_rn_mean(data, alpha=0.1):
    m = data[0]
    rn_list = [m]
    
    for d in data[1:]:
        
        m = (1 - alpha) * m + alpha * d
        rn_list.append(m)
    
    return rn_list
        
        
def playOne(env, model):
    
    ob = env.reset()
    done = False
    iterations = 0
    
    while not done:
        
        if iterations >= 10000:
            print("Failed")
            break
        iterations += 1
        preds = model.get_predictions(ob)
        a = np.argmax(preds)
        ob, reward, done, _ = env.step(a)
        done = bool(ob[0] >= env.goal_position and ob[1] >= env.goal_velocity)
        
    return iterations


def tryHyperparameters(env, scaler, featurer):
    
    final_log = {}
    architectures = [[3]]
    lr_list = [0.001, 0.01]
    best = [None, None, np.float("-inf")]
        
    for architecture in architectures:
        final_log[tuple(architecture)] = {}
        for lr in lr_list:
            model = create_model(architecture=architecture,
                                        lr=lr,
                                        scaler=scaler,
                                        env=env,
                                        featurer=featurer)
            rewards_mean = train(env, model, 2000, verbose=False, mod_rewards=True)
            iterations_on_test = playOne(env, model)
            
            final_log[tuple(architecture)][lr] = [rewards_mean, iterations_on_test]
            
            if rewards_mean > best[2]:
                best = [architecture, lr, rewards_mean]
            
            model.session.close()
    
    with open("./log.pickle", "wb") as file:
        pickle.dump(final_log, file)
    
    return best
            
        
def create_model(architecture, lr, scaler, env, featurer):
    
    s = tf.Session()
    model = Model(architecture=architecture, 
                  input_size=2000,#env.observation_space.shape[0], 
                  session=s,
                  lr=lr,
                  scaler=scaler,
                  featurer=featurer)
    
    return model


if __name__ == "__main__":
    
    env = gym.make("MountainCar-v0")
    
    log = []
    
    gamma_values = [0.5, 1.0, 2.0, 5.0]
    featurer_list = []
    
    for c, gamma in enumerate(gamma_values, 1):
            featurer_list.append(("rbfs{}".format(c), RBFSampler(gamma=gamma, n_components=500)))
    
    scaler = StandardScaler()
    featurer = FeatureUnion(featurer_list)
    sample_observations = np.array([env.observation_space.sample() for i in range(30000)])
    scaler.fit(sample_observations)
    featurer.fit(scaler.transform(sample_observations))
    
    home_machine = socket.gethostname() == "Alessandros-MacBook-Pro.local"
    home_machine=False
    if home_machine == False:
        best = tryHyperparameters(env, scaler, featurer)
        print(best)
        
#        for c, model in enumerate(models_list):
#            savers_list[c].save(model.session, "./tf_sessions/model{}".format(c))
#            model.session.close()
    
#    else:
#        for c, model in enumerate(models_list):
#            savers_list[c].restore(model.session, "./tf_sessions/model{}".format(c))
#            print("model_restored")
#    
#        for i in range(10):
#            iters = playOne(env, models_list)
#            log.append(iters)
#        
#        print(np.mean(log))
        
    