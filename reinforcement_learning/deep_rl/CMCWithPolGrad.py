#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:03:39 2019

@author: alessandro
"""

import numpy as np
import tensorflow as tf
import gym
from RBF import FeatureCreator
import time

tf.reset_default_graph()

class HiddenLayer:
    
    def __init__(self, layer_name, input_size, n_nodes, use_bias=True, activation=tf.nn.tanh):
        
        self.use_bias = use_bias
        self.f = activation
        self.params = []
        
        with tf.variable_scope(layer_name):
        
            self.W = tf.get_variable(name="W",
                                     shape=(input_size, n_nodes))
            self.params.append(self.W)
            
            if self.use_bias:
                self.b = tf.get_variable(name="b",
                                         shape=(1, n_nodes))
                self.params.append(self.b)
        
    
    def forward(self, X):
        
        if self.use_bias:
            return self.f(tf.matmul(X, self.W) + self.b)
        
        else:
            return self.f(tf.matmul(X, self.W))
    

class BaseModel:
    
    def __init__(self, name, input_size, architecture=[], n_outputs=1, activations=[tf.nn.tanh, lambda x: x]):
        
        self.layers = []
        final_layer_i = len(architecture)
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, input_size))
        self.params = []
        
        assert len(activations) == len(architecture) + 1
        
        for c, l in enumerate(architecture):
            layer = HiddenLayer(layer_name="{}/Layer{}".format(name, c), 
                                input_size=input_size, 
                                n_nodes=l,
                                activation=activations[c])
            self.layers.append(layer)
            input_size = l
        
        final_layer = HiddenLayer(layer_name="{}/Layer{}".format(name, final_layer_i),
                                  input_size = input_size,
                                  n_nodes = n_outputs,
                                  activation=activations[final_layer_i])
        
        self.layers.append(final_layer)
        
        out = self.X
        
        for layer in self.layers:
            self.params += layer.params
            Z = layer.forward(out)
            out = Z
        
        self._output = out
        
    
    def predict(self):
        
        return self._output
    
    
    def set_session(self, session):
        
        self.session = session
    
    
    def set_params(self, o_params):
        
        ops = []
        init_op = tf.variables_initializer(self.params)
        ops.append(init_op)
        
        for p, o in zip(self.params, o_params):
            actual = self.session.run(o)
            op = p.assign(actual)
            ops.append(op)
        
        self.session.run(ops)
        
        
    
class PModel:
    
    _instances = 0
    
    def __init__(self, input_size, session, ft_creat):
        
        PModel._instances += 1
        name = PModel._instances
        self.session = session
        self.input_size = input_size
        self.ft_creat = ft_creat
        self.muModel = BaseModel(name="{}/muModel".format(name),
                                 input_size=input_size,
                                 architecture=[],
                                 activations=[lambda x: x],
                                 n_outputs=1)
        
        self.vModel = BaseModel(name="{}/vModel".format(name),
                                input_size=input_size,
                                architecture=[],
                                n_outputs=1,
                                activations=[tf.nn.softplus])
        
        self.muModel.set_session(self.session)
        self.vModel.set_session(self.session)
        
        mu = self.muModel.predict()
        v = self.vModel.predict()
        
        self.pi = tf.contrib.distributions.Normal(mu, v)
        self.predict_op = tf.clip_by_value(self.pi.sample(), -1, 1)
        
        
    def copy(self):
        
        clone = PModel(input_size=self.input_size, session=self.session, ft_creat=self.ft_creat)
        clone.muModel.set_params(self.muModel.params)
        clone.vModel.set_params(self.vModel.params)
        return clone
    
    
    def predict(self, X):
        
        X = self.ft_creat.transform(X)
        pred = self.session.run(self.predict_op, feed_dict={self.muModel.X:X, self.vModel.X:X})
        return pred
    
    
    def perturb_parameters(self, epsilon=0.1):
        
        ops = []
        
        for p in self.muModel.params:
            actual = self.session.run(p)
            noise = np.random.randn(*actual.shape) / np.sqrt(actual.shape[0]) * 5.0
            if np.random.rand() > 0.1:
                ops.append(p.assign(p + noise))
            else:
                ops.append(p.assign(p))
            
        
        for p in self.vModel.params:
            actual = self.session.run(p)
            noise = np.random.randn(*actual.shape) / np.sqrt(actual.shape[0]) * 5.0
            if np.random.rand() > 0.1:
                ops.append(p.assign(p + noise))
            else:
                ops.append(p.assign(p))
        
        self.session.run(ops)
        
    
def playOne(env, model):
    
    s = env.reset()
    done = False
    total_reward = 0
    iterations = 0
    
    while not done and iterations < 1000:
        
        iterations += 1
        a = model.predict(s)
        s, reward, done, _ = env.step(a)
        done = s[0] >= 0.5
        total_reward += reward
    
    return total_reward


def playMultiple(env, model, n_per_params):
    
    rewards = []
    start = time.time()    
    
    for i in range(n_per_params):
        
        reward = playOne(env, model)
        rewards.append(reward)
    
    print("Time: {})".format(time.time()-start))
    mean = np.mean(rewards)
    return mean


if __name__ == "__main__":
    
    env = gym.make("MountainCarContinuous-v0")
    ft_creat = FeatureCreator(env=env, n_components=500)
    best_avg = np.float("-inf")

    with tf.Session() as sess:
        model = PModel(input_size=2000, session=sess, ft_creat=ft_creat)
        best_model = model
        sess.run(tf.global_variables_initializer())
        
        for i in range(100):
            new_model = best_model.copy()
            new_model.perturb_parameters()
        
            avg_total_rewards = playMultiple(env, new_model, 3)
            print("Mean from epoch number {}: {}".format(i, avg_total_rewards))
            if avg_total_rewards > best_avg:
                best_avg = avg_total_rewards
                best_model = new_model
        
        final_avg = playMultiple(env, best_model, 100)
        print("Final Result: {}".format(final_avg))
        
    
    
    
    
    
    
    
    
    
     
        
        
        
    