# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ConvLayer:
    
    def __init__(self, layerName, shape, stride, padding="VALID"):
        
        self.stride = (1, stride, stride, 1)
        self.padding = padding
        
        
        with tf.variable_scope(layerName):
            self.W = tf.get_variable(name="W",
                                     dtype=tf.float32, 
                                     shape=shape,
                                     initializer=tf.keras.initializers.glorot_normal)
            self.b = tf.get_variable(name="b",
                                     dtype=tf.float32,
                                     shape=(shape[3]))
        
    def forward(self, X):
        
        return tf.nn.conv2d(X, 
                            self.W, 
                            strides=self.stride, 
                            padding=self.padding) + self.b
                            
    
    def copyFromKerasLayers(self, layer):
        
        W, b = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        self.session.run([op1, op2])
        
    
    def get_params(self):
        
        return [self.W, self.b]
        
        

class BatchLayer:
    
    def __init__(self, layerName, shape, activation="relu"):
        
        with tf.variable_scope(layerName):
    
            self.gamma = tf.get_variable(name="gamma",
                                     dtype=tf.float32, 
                                     shape=shape,
                                     initializer=tf.keras.initializers.ones)
            
            self.beta = tf.get_variable(name="beta",
                                        dtype=tf.float32, 
                                        shape=shape,
                                        initializer=tf.keras.initializers.zeros)
            
            self.rn_mean = tf.get_variable(name="rn_mean",
                                           dtype=tf.float32, 
                                           shape=shape,
                                           initializer=tf.keras.initializers.zeros, 
                                           trainable = False)
            
            self.rn_std = tf.get_variable(name="rn_std",
                                          dtype=tf.float32, 
                                          shape=shape,
                                          initializer=tf.keras.initializers.ones, 
                                          trainable = False)
            
        
    def forward(self, X, train):
        
       # mean, std = tf.nn.moments(X, axes=[0, 1, 2])
        
#        update_mean = self.rn_mean.assign(0.99 * self.rn_mean + 0.01 * mean)
#        update_std = self.rn_std.assign(0.99 * self.rn_std + 0.01 * std)
        
        result = tf.nn.batch_normalization(X, 
                                           self.rn_mean, 
                                           self.rn_std,
                                           self.beta,
                                           self.gamma,
                                           1e-3)
#        with tf.control_dependencies([update_mean, update_std]):
#            result = tf.cond(train, lambda:tf.nn.batch_normalization(X, 
#                                                  mean,
#                                                  std,
#                                                  self.beta,
#                                                  self.gamma,
#                                                  1e-3),
#                             lambda:tf.nn.batch_normalization(X, 
#                                                          self.rn_mean,
#                                                          self.rn_std,
#                                                          self.beta,
#                                                          self.gamma,
#                                                          1e-3))

        return result

    
    def copyFromKerasLayers(self, layer):
        
        gamma, beta, rn_mean, rn_std = layer.get_weights()
        op1 = self.rn_mean.assign(rn_mean)
        op2 = self.rn_std.assign(rn_std)
        op3 = self.gamma.assign(gamma)
        op4 = self.beta.assign(beta)
        
        self.session.run([op1, op2, op3, op4])        
    
    
    def get_params(self):
        return [self.gamma, self.beta, self.rn_mean, self.rn_std]
    

class ConvBlock:
    
    def __init__(self, input_shape, architecture, stride):
        assert(len(architecture) == 3)
        self.conv1 = ConvLayer("Layer1", 
                               shape=(1, 1, input_shape[-1], architecture[0]),
                               stride=stride)
        self.batch1 = BatchLayer("Layer1", shape=architecture[0])
        self.conv2 = ConvLayer("Layer2", 
                               shape=(3, 3, architecture[0], architecture[1]),
                               stride=stride, 
                               padding="SAME")
        self.batch2 = BatchLayer("Layer2", shape=architecture[1])
        self.conv3 = ConvLayer("Layer3", 
                               shape=(1, 1, architecture[1], architecture[2]),
                               stride=stride)
        self.batch3 = BatchLayer("Layer3", shape=architecture[2])
        
        self.convs = ConvLayer("LayerS", 
                               shape=(1, 1, input_shape[-1], architecture[2]), 
                               stride=stride)
        self.batchs = BatchLayer("LayerS", shape=architecture[2])
        

    def forward(self, X, is_train):
        FX = self.conv1.forward(X)
        FX = self.batch1.forward(FX, is_train)
        FX = tf.nn.relu(FX)
        FX = self.conv2.forward(FX)
        FX = self.batch2.forward(FX, is_train)
        FX = tf.nn.relu(FX)
        FX = self.conv3.forward(FX)
        FX = self.batch3.forward(FX, is_train)
        rX = self.convs.forward(X)
        rX = self.batchs.forward(rX, is_train)
        FX = FX + rX
  
        return FX
  
    
#    def predict(self, X, is_train=True):
#        assert self.session is not None
#        return self.session.run(self._output, feed_dict={self._input:X, self.is_train:is_train})
 
    
    def get_params(self):
        
        params = [self.conv1.get_params(), 
                  self.batch1.get_params(),
                  self.conv2.get_params(), 
                  self.batch2.get_params(),
                  self.conv3.get_params(), 
                  self.batch3.get_params(),
                  self.convs.get_params(), 
                  self.batchs.get_params()]
        return params
    
    
    def set_session(self, session):
        self.session = session
        self.conv1.session = session
        self.batch1.session = session
        self.conv2.session = session
        self.batch2.session = session
        self.conv3.session = session
        self.batch3.session = session
        self.convs.session = session
        self.batchs.session = session
        
    
    def copyFromKerasLayers(self, layers):
        self.conv1.copyFromKerasLayers(layers[0])
        self.batch1.copyFromKerasLayers(layers[1])
        self.conv2.copyFromKerasLayers(layers[3])
        self.batch2.copyFromKerasLayers(layers[4])
        self.conv3.copyFromKerasLayers(layers[6])
        self.batch3.copyFromKerasLayers(layers[8])
        self.convs.copyFromKerasLayers(layers[7])
        self.batchs.copyFromKerasLayers(layers[9])


class IdentityBlock:
    
    def __init__(self, input_shape, architecture, stride):
        assert(len(architecture) == 3)
        self.conv1 = ConvLayer("Layer1", 
                               shape=(1, 1, input_shape[-1], architecture[0]),
                               stride=stride)
        self.batch1 = BatchLayer("Layer1", shape=architecture[0])
        self.conv2 = ConvLayer("Layer2", 
                               shape=(3, 3, architecture[0], architecture[1]),
                               stride=stride, 
                               padding="SAME")
        self.batch2 = BatchLayer("Layer2", shape=architecture[1])
        self.conv3 = ConvLayer("Layer3", 
                               shape=(1, 1, architecture[1], architecture[2]),
                               stride=stride)
        
        self._input = tf.placeholder(shape=(None, input_shape[0], input_shape[1], input_shape[2]), 
                                     dtype=tf.float32)
        
        self._output = self.forward()
        
        self.is_train = tf.placeholder(dtype=tf.bool)

    
    def forward(self):
        
        FX = self.conv1.forward(self._input)
        FX = self.batch1.forward(FX, self.is_train)
        FX = self.conv2.forward(FX)
        FX = self.batch2.forward(FX, self.is_train)
        FX = self.conv3.forward(FX)
        
        FX = FX + self._input
        
        return tf.nn.relu(FX)
    
    
    def predict(self, X, is_train=True):
        
        assert self.session is not None
        return self.session.run(self._output, feed_dict={self._input: X, self.is_train:is_train})
        
        
    
if __name__ == '__main__':
    tf.reset_default_graph()
    conv_block = ConvBlock([224, 224, 3], [64, 64, 256], 1)
    
    # make a fake image
    X = np.random.random((1, 224, 224, 3))

    init = tf.global_variables_initializer()
    with tf.Session() as session:
      conv_block.set_session(session)
      session.run(init)

      output = conv_block.predict(X)
      print("output.shape:", output.shape)