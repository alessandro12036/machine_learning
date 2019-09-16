#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 00:42:59 2019

@author: Alessandro
"""

from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


# Let's go up to the end of the first conv block
# to make sure everything has been loaded correctly
# compared to keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from resnet import ConvLayer, BatchLayer, ConvBlock



class PartialResNet:

    def __init__(self):
        self.conv1 = ConvLayer("conv1", shape=(7, 7, 3, 64), stride=2, padding="VALID")
        self.batch1 = BatchLayer("batch1", 64)

        self.conv_block1 = ConvBlock([224, 224, 64], [64, 64, 256], 1)
        self.is_train = tf.placeholder(dtype=tf.bool)
        self._input = tf.placeholder(name="input", shape=(None, 224, 224, 3), dtype=tf.float32)
        self._output = self.forward()


    def forward(self):
        
        FX = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(self._input)
        FX = self.conv1.forward(FX)
        print(FX.shape)
        FX = self.batch1.forward(FX, self.is_train)
        print(FX.shape)
        FX = tf.nn.relu(FX)
        FX = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(FX)
        print(FX.shape)
        FX = tf.nn.max_pool(FX, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="VALID")
        print(FX.shape)
        FX = self.conv_block1.forward(FX, self.is_train)
        FX = tf.nn.relu(FX)

        return FX


    def copyFromKerasLayers(self, layers):

        self.conv1.copyFromKerasLayers(layers[2])
        self.batch1.copyFromKerasLayers(layers[3])
        self.conv_block1.copyFromKerasLayers(layers[7:])


    def predict(self, X, is_train=True):

        assert self.session is not None
        result = self.session.run(self._output, feed_dict={self._input:X,
                                                           self.is_train:is_train})

        return result


    def set_session(self, session):
        self.session = session
        self.conv1.session = session
        self.batch1.session = session
        self.conv_block1.set_session(session)


    def get_params(self):
        conv_block1_params = []
        for layer in self.conv_block1.get_params():
            for param in layer:
                conv_block1_params.append(param)
        params = [*self.conv1.get_params(), *self.batch1.get_params(), *conv_block1_params]
        return params


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.keras.backend.clear_session()
    # you can also set weights to None, it doesn't matter
    resnet = ResNet50(weights='imagenet')

    # you can determine the correct layer
    # by looking at resnet.layers in the console
    partial_model = Model(
            inputs=resnet.input,
            outputs=resnet.layers[18].output)
    print(partial_model.summary())
    # for layer in partial_model.layers:
    #   layer.trainable = False

    my_partial_resnet = PartialResNet()

    # make a fake image
    X = np.random.random((1, 224, 224, 3))

    # get keras output
    keras_output = partial_model.predict(X)

    # get my model output
    init = tf.variables_initializer(my_partial_resnet.get_params())

    # note: starting a new session messes up the Keras model
    session = tf.keras.backend.get_session()
    my_partial_resnet.set_session(session)
    session.run(init)

    # first, just make sure we can get any output
    first_output = my_partial_resnet.predict(X)
    print("first_output.shape:", first_output.shape)

    # copy params from Keras model
    my_partial_resnet.copyFromKerasLayers(partial_model.layers)

    # compare the 2 models
    output = my_partial_resnet.predict(X)
    diff = np.abs(output - keras_output).sum()
    if diff < 1e-10:
        print("Everything's great!")
    else:
        print("diff = %s" % diff)