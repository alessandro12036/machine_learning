#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:37:10 2019

@author: alessandro
"""

import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt


tf.reset_default_graph()

# Create HiddenLayer class
# Create Model class with replay buffer and copy method


class HiddenLayer:
    
    def __init__(self, layer_name, input_size, n_nodes, activation=tf.nn.tanh):
        
        self.f = activation
        
        with tf.variable_scope(layer_name):
            self.W = tf.get_variable(name="W", 
                                     shape=(input_size, n_nodes),
                                     dtype=tf.float32)
            
            self.b = tf.get_variable(name="b",
                                     shape=(1, n_nodes),
                                     dtype=tf.float32)
        
        self.params = [self.W, self.b]
            
        
    def forward(self, X):
        
        out = tf.matmul(X, self.W) + self.b
        out = self.f(out)
        
        return out
    
    

class Model:
    
    def __init__(self, name, input_size, n_outputs, architecture, lr=10e-4, max_experience = 10000, min_experience = 100, batch_size=32):
        
        self.session = None
        self.lr = lr
        self.layers = []
        self.experience = {"s1": [], "a": [], "r": [], "s2": [], "done": []}
        self.max_experience = max_experience
        self.min_experience = min_experience
        self.batch_size = batch_size
        self.costs = []
        self.X = tf.placeholder(dtype=tf.float32,
                                shape=(None, input_size))
        self.y = tf.placeholder(dtype=tf.float32,
                                shape=(None,))
        self.actions = tf.placeholder(dtype=tf.int32,
                                      shape=(None,))
        
        for c, l in enumerate(architecture):
            
            layer = HiddenLayer(layer_name="{}/Layer{}".format(name, c),
                                input_size=input_size,
                                n_nodes=l,
                                activation=tf.nn.tanh)
            
            self.layers.append(layer)
            input_size = l
        
        layer = HiddenLayer(layer_name="{}/Layer{}".format(name, len(architecture)),
                            input_size=input_size,
                            n_nodes=n_outputs,
                            activation=lambda x: x)
        
        self.layers.append(layer)
        
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        
        self.output = Z
        
        specific_out = tf.reduce_sum(self.output * tf.one_hot(self.actions, depth=2), 
                                     axis=1) #depth hardcoded, poi cambiala
        squared_diff = tf.square(self.y-specific_out)
        self.cost = tf.reduce_sum(squared_diff) # reduce_mean converge ovviamente più lentamente
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.cost)
    
    
    def copy(self, other):
        
        ops = []
        print("Updating target model...")
        
        for own_layer, other_layer in zip(self.layers, other.layers):
            
            actual = self.session.run(other_layer.params)
            for c, p in enumerate(actual):
                op = tf.assign(own_layer.params[c], p)
                ops.append(op)
            
        self.session.run(ops)
        
    
    def predict(self, X):
        
        X = X.reshape(1, -1)
        preds = self.session.run(self.output, feed_dict={self.X:X})
        
        return preds
    
    
    def set_session(self, session):
        
        self.session = session
    
    
    def add_exp(self, s1, a, r, s2, done):
        
        # Sicuramente snellibile
        
        self.experience["s1"].append(s1)
        self.experience["a"].append(a)
        self.experience["r"].append(r)
        self.experience["s2"].append(s2)
        self.experience["done"].append(done)
        
        if len(self.experience["s1"]) > self.max_experience:
            self.experience["s1"].pop(0)
            self.experience["a"].pop(0)
            self.experience["r"].pop(0)
            self.experience["s2"].pop(0)
            self.experience["done"].pop(0)
    
        
    def train(self, env, target_model, gamma=0.99):
        
        # ci si allena solo sulla base del buffer di exp, non della iterazione in cui viene chiamato
        
        if len(self.experience["s1"]) < self.min_experience:
            return
        
        idxs = np.random.choice(len(self.experience["s1"]), size=self.batch_size, replace=False)
        s1 = []
        a = []
        g = []
        for idx in idxs:
            s1.append(self.experience["s1"][idx])
            a.append(self.experience["a"][idx])
            s2 = self.experience["s2"][idx]
            r = self.experience["r"][idx]
            done = self.experience["done"][idx]
            if done:
                ret = r
            else:
                ret = r + gamma*np.max(target_model.predict(s2))
            g.append(ret)
        
        cost, _ = self.session.run([self.cost, self.train_op], feed_dict={self.X:s1,
                                                                          self.y:g,
                                                                          self.actions:a})
        
        self.costs.append(cost)
        
    

def playOne(model, env, target_model, eps, copy_period=50):
    
    s1 = env.reset()
    done = False
    total_reward = 0
    iterations = 0

    while not done:
        iterations += 1
        if np.random.rand() <= eps:
            a = env.action_space.sample()
        else:
            a = np.argmax(model.predict(s1))
        s2, reward, done, _ = env.step(a)
        total_reward += reward
        model.add_exp(s1, a, reward, s2, done)
        model.train(env, target_model)
        if iterations % copy_period == 0:
            target_model.copy(model)
        s1 = s2
        
    return total_reward


def playMultiple(model, target_model, env, epochs=3000):
    
    rewards = 0
    alpha = 0.1
    improvements = 0
    max_tot_r = 0
    
    for i in range(epochs):
        
        eps = 1.0/np.sqrt(improvements+1)

        tot_r = playOne(model, env, target_model, eps)
        if tot_r > max_tot_r or tot_r >= 200:
            improvements += 1
            max_tot_r = tot_r
        if i == 0:
            rewards = tot_r
        else:
            rewards = (1-alpha) * rewards + alpha * tot_r
        
        if i % 100 == 0:
            print("Current mean: {}, Current epsilon: {}".format(rewards, eps))
            
    return rewards
            


if __name__ == "__main__":
    
    # learning rate migliore: 10e-4
    
    env = gym.make("CartPole-v0")
    model = Model(name="MainModel",
                  input_size=env.observation_space.shape[0], 
                  n_outputs=env.action_space.n,
                  architecture=[200, 200])
    target_model = Model(name="TargetModel",
                         input_size=env.observation_space.shape[0], 
                         n_outputs=env.action_space.n,
                         architecture=[200, 200])
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        model.set_session(sess)
        target_model.set_session(sess)
        
        rewards = playMultiple(model=model,
                               target_model=target_model,
                               env=env)
        
    #plt.title("Model costs")
#    plt.plot(model.costs)
#    plt.show()
