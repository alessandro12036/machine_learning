#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:39:15 2019

@author: alessandro
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ConvLayer:
    
    def __init__(self, layer_name, kernel_size, input_channels, output_channels, strides=1, padding="valid", activation="relu"):
        
        self.strides = strides
        self.padding = padding
        
        with tf.variable_scope(layer_name):  
            self.W = tf.get_variable(name="W", 
                                     dtype=tf.float32, 
                                     shape=[kernel_size, kernel_size, input_channels, output_channels],
                                     initializer=tf.initializers.glorot_uniform())
            
            self.b = tf.get_variable(name="b",
                                     dtype=tf.float32,
                                     shape=output_channels,
                                     initializer=tf.initializers.zeros())
            
    
    def forward(self, X):
        
        self.Z = tf.nn.conv2d(X, 
                              self.W,
                              self.strides,
                              self.padding) + self.b
                              
        return tf.nn.relu(self.Z)
    
    
    def set_session(self, session):
        
        self.session = session
        
        
    def copyKerasLayers(self, layer):
        
        assert self.session != None
        
        W, b = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        
        self.session.run([op1, op2])
    

class BNLayer:
    
    def __init__(self, layer_name, output_size):
        
        with tf.variable_scope(layer_name): 
            self.gamma = tf.get_variable(name="gamma",
                                         dtype=tf.float32,
                                         shape=output_size)
            self.beta = tf.get_variable(name="beta",
                                        dtype=tf.float32,
                                        shape=output_size)