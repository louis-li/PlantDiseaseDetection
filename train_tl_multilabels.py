import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import date

import tensorflow.keras as keras
from tensorflow.keras import applications
from tensorflow.keras.layers import Flatten, LeakyReLU, Dense
from tensorflow.keras.models import *
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import random
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from os import listdir
from os.path import isfile, join

import importlib
import data_load
import util


# import image data and combine labels
# import image data and combine labels
def load_data(image_size=224, image_dir='data/multipleplants'):
    
    species = ['Apple', 'Bell Pepper', 'Blueberry', 'Cherry (including sour)',
           'Corn (maize)', 'Grape', 'Orange', 'Peach', 'Potato', 'Raspberry',
           'Soybean', 'Squash', 'Strawberry', 'Tomato']

    classes = ['Apple Black rot' 'Apple Cedar apple rust' 'Apple healthy' 'Apple scab'
     'Bell Pepper Bacterial spot' 'Bell Pepper healthy' 'Blueberry healthy'
     'Cherry (including sour) Powdery mildew'
     'Cherry (including sour) healthy'
     'Corn (maize) Cercospora leaf spot Gray leaf spot'
     'Corn (maize) Common rust ' 'Corn (maize) Northern Leaf Blight'
     'Corn (maize) healthy' 'Grape Black rot' 'Grape Esca (Black Measles)'
     'Grape Leaf blight (Isariopsis Leaf Spot)' 'Grape healthy'
     'Orange Haunglongbing (Citrus greening)' 'Peach Bacterial spot'
     'Peach healthy' 'Potato Early blight' 'Potato Late blight'
     'Potato healthy' 'Raspberry healthy' 'Soybean healthy'
     'Squash Powdery mildew' 'Strawberry Leaf scorch' 'Strawberry healthy'
     'Tomato American Serpentine Leafminer' 'Tomato Bacterial spot'
     'Tomato Early blight' 'Tomato Insect Bite' 'Tomato Late blight'
     'Tomato Leaf Mold' 'Tomato Powdery mildew' 'Tomato Septoria leaf spot'
     'Tomato Spider mites Two-spotted spider mite' 'Tomato Stem rot'
     'Tomato Target Spot' 'Tomato Wilt' 'Tomato Yellow Leaf Curl Virus'
     'Tomato healthy' 'Tomato mosaic virus']
    X = []
    y = []
    labels = []
    for root, folder, files in os.walk(image_dir):
        #print( folder)
        for f in files:
            #print(f)
            if f.lower().endswith('.jpg') or f.lower().endswith('.png') or f.lower().endswith('.jpeg'):
                #print(root, folder, f)
                img = load_img(f'{root}/{f}', target_size=(image_size,image_size,3))
                img_array = img_to_array(img, dtype='uint8')
                X.append(img_array)
                
                # get y
                specie, classname = root[20:].split('___')
                specie = specie.replace('_', ' ')
                classname = classname.replace('_', ' ')
                if classname[:len(specie)].lower() != specie.lower():
                    classname=specie +' ' + classname
                #print(species.index(specie), classes.index(classname))
                y.append([species.index(specie), classes.index(classname)])
                labels.append([specie, classname])
                
    return np.array(X, dtype=np.uint8), np.array(y), np.array(labels)

def train_tl(model, batch_size = 64, warm_up_learning_rate = 0.005, 
             warm_up_reduce_lr_patience = 3, warm_up_early_stop_patience = 6, 
             warm_up_min_lr=0.00005, learning_rate = 0.0001, reduce_lr_patience = 5, 
             early_stop_patience = 10, min_lr=0.00001, val_train_epoch=3):
    #main        
    data_x, data_y, labels  = load_data() 
    x_train, x_val, y_train, y_val = train_test_split(data_x, data_y[:,1], test_size=0.05, random_state=42)

    #importlib.import_module("data_load")
    
    # Warm up head
    adam = optimizers.Adam(learning_rate=warm_up_learning_rate)
    #lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                               factor=0.2,  
                               patience=warm_up_reduce_lr_patience, 
                               min_lr=warm_up_min_lr)
    early_stop = EarlyStopping(
        monitor='val_loss', 
        min_delta=0, 
        patience=warm_up_early_stop_patience, 
        verbose=0, 
        mode='auto',
        baseline=None, 
        restore_best_weights=True
    )

    # freeze pretrained weights
    for l in model.layers[:-2]:
        l.trainable = False

    model.compile(optimizer=adam, 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()])


    history = model.fit(data_load.generator_with_label(x_train, y_train, batch_size),  
                        shuffle=True,  
                        validation_data = (x_val, y_val),
                        callbacks = [reduce_lr,early_stop],                        
                        epochs=100,
                        steps_per_epoch=len(x_train)/batch_size ,
                        verbose=True
                       )

    # Train entire network
    adam = optimizers.Adam(learning_rate=learning_rate)
    #lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                               factor=0.2,  
                               patience=reduce_lr_patience , 
                               min_lr=min_lr)
    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        min_delta=0, 
        patience=early_stop_patience, 
        verbose=0, 
        mode='auto',
        baseline=None, 
        restore_best_weights=True
    )

    for l in model.layers:
        l.trainable = True

    model.compile(optimizer=adam, 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history2 = model.fit(data_load.generator_with_label(x_train, y_train, batch_size),  
                        shuffle=True,  
                        validation_data = (x_val, y_val),
                        callbacks = [reduce_lr,early_stop],                        
                        epochs=200,
                        steps_per_epoch=len(x_train)/batch_size ,
                        verbose=True
                       )

    util.showConfusionMatrix(model, x_val, y_val)
    model.fit(data_load.generator_with_label(x_val, y_val, batch_size),            
                        epochs=val_train_epoch,
                        steps_per_epoch=len(x_train)/batch_size ,
                        verbose=True
                       )
    
    return model, [history, history2]