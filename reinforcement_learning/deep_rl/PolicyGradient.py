#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:00:56 2019

@author: alessandro
"""

import numpy as np
import tensorflow as tf
import gym

tf.reset_default_graph()

class Layer:
    
    def __init__(self, input_size, output_size, name, activation=tf.nn.tanh, use_bias=True):
        
        self.use_bias = use_bias
        self.f = activation
        
        with tf.variable_scope(name):
            self.W = tf.get_variable(name="W", 
                                     shape=(input_size, output_size), 
                                     dtype=tf.float32)
            if self.use_bias:
                self.b = tf.get_variable(name="b", 
                                         shape=(1, output_size), 
                                         dtype=tf.float32,
                                         initializer=tf.zeros_initializer)
 
    
    def forward(self, X):
        
        if self.use_bias:
            output = tf.matmul(X, self.W) + self.b
        
        else:
            output = tf.matmul(X, self.W)
        
        return self.f(output)


class PModel:
    
    def __init__(self, input_size, architecture, n_outputs, lr=10e-2, name="PModel"):
        
        self.lr = lr
        self.layers = []
        self.X = tf.placeholder(tf.float32, shape=(None, input_size))
        self.actions = tf.placeholder(tf.int32, shape=(None,))
        self.advantages = tf.placeholder(tf.float32, shape=(None, 1))
        
        for i, l in enumerate(architecture):
            layer = Layer(input_size=input_size, 
                          output_size=l, 
                          name="{}/layer{}".format(name, i+1))
            self.layers.append(layer)
            input_size = l
        
        layer = Layer(input_size=input_size,
                      output_size=n_outputs,
                      name="{}/layer{}".format(name, len(self.layers)+1),
                      activation=tf.nn.softmax)
        self.layers.append(layer)
        
        Z = self.X
        
        for l in self.layers:
            Z = l.forward(Z)
        
        self.pred_op = Z
        
        self.specific_p = tf.reduce_sum(self.pred_op * tf.one_hot(self.actions, n_outputs, axis=1), axis=[1])
        log_specific_p = tf.log(self.specific_p)
        
        self.cost = -tf.reduce_sum(self.advantages * log_specific_p)
        
        self.train_op = tf.train.AdagradOptimizer(self.lr).minimize(self.cost)
    
    
    def set_session(self, session):
        
        self.session = session
    
    
    def predict(self, X):
        
        preds = self.session.run(self.pred_op, feed_dict={self.X:X})
        return preds
     
    
    def partial_fit(self, X, advantages, actions):
        
        cost, _ = self.session.run([self.cost, self.train_op], feed_dict={self.X:X,
                                                                          self.advantages:advantages,
                                                                          self.actions:actions})
        print("Cost: ", cost)
    
    def sample_action(self, s):
        
        preds = self.predict(s).squeeze()
        return np.random.choice(len(preds), p=preds)
        
        
        
class VModel:
    
    def __init__(self, input_size, architecture, lr=10e-5, name="VModel"):
        
        self.lr = lr
        self.layers = []
        self.X = tf.placeholder(tf.float32, shape=(None, input_size))
        self.y = tf.placeholder(tf.float32, shape=(None, 1))
        
        for i, l in enumerate(architecture):
            layer = Layer(input_size=input_size, 
                          output_size=l, 
                          name="{}/layer{}".format(name, i+1))
            self.layers.append(layer)
            input_size = l
        
        layer = Layer(input_size=input_size,
                      output_size=1,
                      name="{}/layer{}".format(name, len(self.layers)+1),
                      activation=lambda x: x)
        self.layers.append(layer)
        
        Z = self.X
        
        for l in self.layers:
            Z = l.forward(Z)
        
        self.pred_op = Z
        
        squared_diff = tf.square(self.pred_op - self.y)
        self.cost = tf.reduce_sum(squared_diff)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.cost)
        
        
    def partial_fit(self, X, target):
        
        cost, _ = self.session.run([self.cost, self.train_op], feed_dict={self.X:X,
                                                                          self.y:target})
        
    
    def predict(self, X):
        
        preds = self.session.run(self.pred_op, feed_dict={self.X:X})
        return preds[0]
    
    
    def set_session(self, session):
        
        self.session = session


def trainWithMC(env, vModel, pModel, epochs):
    
    gamma = 0.99
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        vModel.set_session(sess)
        pModel.set_session(sess)
        
        for epoch in range(epochs): 
            s1 = env.reset()
            done = False
            states = []
            actions = []
            returns = []
            rewards = []
            iterations = 0
            
            while not done:
                
                states.append(s1)
                iterations += 1
                a = pModel.sample_action(s1.reshape(1, -1))
                s2, reward, done, _ = env.step(a)
                rewards.append(reward)
                actions.append(a)
                s1 = s2
            
            assert len(rewards) == len(states)
            g = 0
            for r in reversed(rewards):
                returns.append(g)
                g = r + gamma * g
            returns.reverse()
            returns = np.expand_dims(returns, axis=1)
            vPred = vModel.predict(states)
            vModel.partial_fit(states, returns)
            advantages = returns - vPred
            pModel.partial_fit(states[:-1], advantages[:-1], actions[:-1])
        
            print("Epoch {} took {} iterations".format(epoch, iterations))
            

if __name__ == "__main__":
    
    env = gym.make("CartPole-v0")
    pModel = PModel(input_size=env.observation_space.shape[0], architecture=[], n_outputs=env.action_space.n)
    vModel = VModel(input_size=env.observation_space.shape[0], architecture=[10])
    trainWithMC(env, vModel, pModel, 10000)
                