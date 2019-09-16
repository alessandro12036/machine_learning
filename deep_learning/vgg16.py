# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from glob import glob
from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix


def get_confusion_matrix(data_path, N, model, batch_size):
    i = 0
    predictions = []
    ground = []
    for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size):
        i+=1
        if i % 50 == 0:
            print(i)
        p = model.predict(x)
        p = np.argmax(p)
        y = np.argmax(y)
        predictions = np.concatenate((predictions, p))
        ground = labels.concatenate((ground, y))
        if len(ground) >= N:
            break
    cm = confusion_matrix(ground, predictions)
    return cm


IMAGE_SIZE = [100, 100]

epochs = 5
batch_size = 32

train_path = "./Training/fruits_360_small"
test_path = "./Test/fruits_360_small"

train_images = glob(train_path + "/*/*.jp*g")
test_images = glob(test_path + "/*/*.jp*g")
folders = glob(train_path + "/*")
N_classes = len(folders)
print("Total number of train images: {}".format(len(train_images)))
print("Total number of test images: {}".format(len(test_images)))
print("Total number of classes: {}".format(N_classes))

plt.imshow(keras.preprocessing.image.load_img(np.random.choice(train_images)))
plt.show()

#vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False)
#for layer in vgg.layers:
#    layer.trainable = False

resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights = "imagenet", include_top=False)

x = keras.layers.Flatten()(resnet.output)
x = keras.layers.Dense(1000, activation="relu")(x)
prediction = keras.layers.Dense(N_classes, activation="softmax")(x)

model = keras.Model(inputs=resnet.input, outputs=prediction)
model.summary()

model.compile(loss="categorical_crossentropy", 
              optimizer="adam", 
              metrics=["accuracy"])

gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                                             width_shift_range=0.1,
                                             height_shift_range=0.1,
                                             shear_range=0.1,
                                             zoom_range=0.1,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             preprocessing_function=keras.applications.resnet50.preprocess_input)

test_gen = gen.flow_from_directory(test_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * N_classes
for k, v in test_gen.class_indices.items():
    labels[v] = k
print(labels)

for x, y in test_gen:
    print(x[0].min(), " ", x[0].max())
    plt.title(labels[np.argmax(y[0])])
    plt.imshow(x[0])
    plt.show()
    break

plt.clf()
train_gen = gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=batch_size) 

r = model.fit_generator(train_gen,
                        validation_data = test_gen,
                        epochs=epochs, 
                        steps_per_epoch=len(train_images)//batch_size,
                        validation_steps=len(train_images)//batch_size)


cm = get_confusion_matrix(train_path, N_classes, model, batch_size*2)
plot_confusion_matrix(cm, labels)

