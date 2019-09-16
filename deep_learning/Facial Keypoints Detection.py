#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.keras as keras
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import time
import os
import socket
import pickle
from tensorflow.keras.models import Sequential
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
from sklearn.model_selection import train_test_split


def preprocess_data(path, cols=None, test=False):
    df = pd.read_csv(path)
    if cols:
        df = pd.concat(df[cols], df["Images"], axis=1)
    df = df.dropna()
    df["Image"] = df["Image"].apply(lambda x: np.fromstring(x, sep=" "))
    X = np.vstack(df["Image"].values) / 255
    X = X.astype(np.float32)
    print("Number of valid images in the original dataset: {}".format(X.shape[0]))
    y = None
    
    if test == False:
        y = df.iloc[:, :-1]
        y = (y.values - 48) / 48
        y = y.astype(np.float32)
        X, y = shuffle(X, y, random_state=48)
        assert X.shape[0] == y.shape[0], "X and y dimensions don't match"
        
    return X, y


def load2d(path, cols=None, test=False):
    X, y = preprocess_data(path, cols, test)
    X = X.reshape(X.shape[0], 96, 96, 1)
    
    return X, y

    
class Plotter():
    
    def __init__(self):
        self.clean_figure()
    
    def clean_figure(self):
        self.fig = plt.figure(figsize=(6, 6))

    def show_results(self, h, name):
        self.clean_figure()
        rows = len(h.keys()) // 2
        self.fig.suptitle(name)
        if len(h.keys()) % 2 != 0:
            rows += 1
        cols = 2
        for i, par in enumerate(h.keys(), 1):
            ax = plt.subplot(rows, cols, i)
            ax.set_title(par)
            ax.plot(h[par])
        plt.tight_layout()
        plt.show()

    def plot_predictions(self, img, y, ax, n, conv=False):
        ax.set_title("Image " + str(n))
        img = img.reshape(96, 96)
        img = img * 255
        y = y * 48 + 48
        ax.imshow(img, cmap="gray")
        ax.scatter(y[0::2], y[1::2], marker = "x")
        ax.axis("off")
    
    def show_predictions(self, model, path, title="Unitled Figure", conv=False):
        self.clean_figure()
        self.fig.suptitle(title)
        self.fig.subplots_adjust(hspace=0.5, wspace=0.05)
        if conv:
            X_test, _ = load2d(path, test=True)
        else:
            X_test, _ = preprocess_data(path, test=True)
        y_pred = model.predict(X_test)
        
        for i in range(16):
            ax = self.fig.add_subplot(4, 4, i+1)

            if conv:
                img = X_test[i, :, :, :]
            else:
                img = X_test[i]
                
            self.plot_predictions(img, y_pred[i, :], ax, i, conv)
        plt.show()
        
        
def train_model(model, x, y, checkpoints=False, save_history = False, epochs=10, augmentation=False, val_x=None, val_y=None):
    
    model.compile(optimizer="Adam",
                  loss = "mean_squared_error",
                  metrics = ["accuracy"])
        
    start_time = time.time()
    
    callbacks_list = [keras.callbacks.BaseLogger(), keras.callbacks.History()]
    
    if checkpoints:
        if os.path.exists("./checkpoints") == False:
            os.mkdir("checkpoints")
        
        checkpoint = keras.callbacks.ModelCheckpoint(filepath="checkpoints/weights.{epoch:02d}-{val_loss:0.2f}.hdf5", 
                                                     save_weights_only=True,  
                                                     verbose=1,
                                                     period=100)
        callbacks_list = callbacks_list + [checkpoint]
    
    if augmentation:
        
        train_gen = ImageDataGenerator(rotation_range=15,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')
        
        test_gen = ImageDataGenerator()
        
        
        h = model.fit_generator(train_gen.flow(x, y, batch_size=32),
                                steps_per_epoch=x.shape[0]/32,
                                epochs=epochs, 
                                validation_data=test_gen.flow(x, y, batch_size=32),
                                validation_steps=len(val_x) / 32,
                                verbose=1, 
                                callbacks=callbacks_list)
    else:
        
        h = model.fit(x=x, 
                      y=y, 
                      epochs=epochs, 
                      batch_size=32, 
                      validation_split=0.2, 
                      verbose=1, 
                      callbacks=callbacks_list)
    
    total_training_time = time.time() - start_time
    
    if save_history:
        with open(os.path.expanduser("~/hist"), "wb") as file:
            pickle.dump(h.history, file)
    
    print("Training took {} seconds".format(total_training_time))
    
    return h
    

def compare_results(models, test_path, types):
    
    flatten_X, _ = preprocess_data(test_path, test=True)
    bidimensional_X = flatten_X.reshape(flatten_X.shape[0], 96, 96, 1)
    imgs = bidimensional_X.reshape(bidimensional_X.shape[0], 96, 96) * 255
    predictions = {}
    index = 0
    cols = 4
    rows = math.ceil(len(models) / 4)
    
    
    while index >= 0:
        for i, model in enumerate(models):
            if types[i] == "Dense":
                model_preds = model.predict(flatten_X)
            else:
                model_preds = model.predict(bidimensional_X)
            model_preds = model_preds * 48 + 48
            predictions["Model {}".format(i+1)] = model_preds
        
        img = imgs[index]
        
        fig = plt.figure()
        fig.suptitle("Predictions for image {}".format(index))
        
        for i in range (len(models)):
            pred = predictions["Model {}".format(i+1)][index]
            ax = fig.add_subplot(rows, cols, i+1)
            ax.set_title("Model {}".format(i+1))
            ax.imshow(img, cmap="gray")
            ax.scatter(pred[0::2], pred[1::2], marker="x")
            
        plt.show()
        index = int(input("Enter index of image to compare: "))
        
        
def load_weights(models, weight_paths):
    assert len(models) == len(weight_paths)
    
    for i, model in enumerate(models):
        selection = glob(weight_paths[i] + "/*.hdf5")
        model.load_weights(selection[0])
        

def load_hist(h_paths, hists):
    for path in h_paths:
        model_name = path.split("/")[-2]
        with open(path, "rb") as file:  
            hists[model_name] = pickle.load(file)
    return hists


###############################################################################


local = socket.gethostname() == "Alessandros-MBP"


if local:
    training_path = "../datasets/Facial Keypoints Detection/training.csv"
    test_path = "../datasets/Facial Keypoints Detection/test.csv"
    
else:
    training_path = "~/training.csv"
    test_path = "~/test.csv"

histories = {}

# Simple model with one hidden layer
X, y = preprocess_data(training_path)

model1 = Sequential()
model1.add(keras.layers.Dense(units=100, 
                              input_shape=X.shape[1:], 
                              activation="relu"))
model1.add(keras.layers.Dense(units=y.shape[1]))

h = train_model(model1, x=X, y=y, epochs=50)
histories["Model_1"] = h.history

plotter = Plotter()

plotter.show_results(histories["Model_1"], "Model 1")
plotter.show_predictions(model1, test_path, "Model 1 predictions")


# Conv model

imgs, y = load2d(training_path)

X_train, X_test, y_train, y_test = train_test_split(imgs, y, test_size=0.2)

model2 = Sequential()

model2.add(keras.layers.Conv2D(filters=32, 
                        kernel_size=(3, 3), 
                        strides=(1, 1), 
                        padding="same", 
                        activation="relu",
                        input_shape=imgs.shape[1:]))
model2.add(keras.layers.MaxPooling2D())
model2.add(keras.layers.Conv2D(filters=64, 
                        kernel_size=(2, 2), 
                        strides=(1, 1), 
                        padding="same", 
                        activation="relu"))
model2.add(keras.layers.MaxPooling2D())
model2.add(keras.layers.Conv2D(filters=128, 
                        kernel_size=(2, 2), 
                        strides=(1, 1), 
                        padding="same", 
                        activation="relu"))
model2.add(keras.layers.MaxPooling2D())
model2.add(keras.layers.Flatten())
model2.add(keras.layers.Dense(units=500, activation="relu"))
model2.add(keras.layers.Dense(units=500, activation="relu"))
model2.add(keras.layers.Dense(units=y.shape[1]))

model3 = keras.models.clone_model(model2)

if local == False:
    train_model(model3, 
                save_history=True, 
                checkpoints=True, 
                epochs=1000, 
                x=X_train, 
                y=y_train,
                val_x = X_test,
                val_y = y_test,
                augmentation=True)
     
else:
    models_list = [model1, model2, model3]

    w_paths = glob("./Weights/*")
    load_weights(models_list[1:], w_paths) # Esclude il modello più semplice che non ha nessun peso da caricare
    
    h_paths = glob("./Histories/*/hist")
    histories = load_hist(h_paths, histories)
    
    for i, model in enumerate(models_list[1:], 2):
        name = "Model_" + str(i)
        space_separated_name = " ".join(name.split("_"))
        plotter.show_results(histories[name], name=space_separated_name)
        plotter.show_predictions(model, test_path, "Model {} predictions".format(i), conv=True)

    compare_results(models_list, test_path, types=["Dense", "Conv", "Conv"])
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    
    ax_arr = [ax1, ax2, ax3, ax4]
    
    for model in histories.keys():
        for i, metric in enumerate(histories[model].keys()):
            ax_arr[i].set_title(metric)
            ax_arr[i].plot(histories[model][metric], label=model)
    plt.legend()     
    plt.show()


