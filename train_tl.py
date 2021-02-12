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
def load_data(dataset='Train',image_size = 224, image_dir= 'data/fgvc7/images', labels = pd.read_csv("data/fgvc7/train.csv")):
    X = []
    y = []
    for f in listdir(image_dir):
        if isfile(join(image_dir, f)) and f[:len(dataset)]==dataset:
            # add filename
            #train_list.append(f)
            # add image to the list
            img = load_img(f'{image_dir}/{f}', target_size=(image_size,image_size,3))
            img_array = img_to_array(img, dtype='uint8')
            # Get id
            id = int(f[len(dataset)+1:-4])
            X.append(img_array)
            if dataset=='Train':
                y.append(labels.iloc[id][['healthy', 'multiple_diseases', 'rust', 'scab']].to_numpy())
            else:
                y.append([f[:-4], id])
    if dataset=='Train':
        y = np.array(y, dtype=np.uint8)   
    return np.array(X, dtype=np.uint8), y

def train_tl(model, batch_size = 64, warm_up_learning_rate = 0.005, 
             warm_up_reduce_lr_patience = 3, warm_up_early_stop_patience = 6, 
             warm_up_min_lr=0.00005, learning_rate = 0.0001, reduce_lr_patience = 5, 
             early_stop_patience = 10, min_lr=0.00001, val_train_epoch=3, submit_to_kaggle = True, submission_filename = "submission/test.csv"):
    #main        
    data_x, data_y  = load_data() 
    x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.30, random_state=42)
    y_train = np.argmax(y_train,axis=1)
    y_val = np.argmax(y_val,axis=1)

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
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
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
    
    if submit_to_kaggle:
        # Load test data and their filename for submission file
        test_x, test_id  = load_data("Test") 
        test_y = model.predict(test_x)
        test_pred = tf.nn.softmax(test_y).numpy()
        test_set = np.hstack((test_id, test_pred))
        test_set = test_set[test_set[:,1].astype('uint16').argsort()]

        test_DF = pd.DataFrame(test_set, index=test_set[:,1], columns=["image_id","id","healthy","multiple_diseases","rust","scab"])
        test_DF[["image_id","healthy","multiple_diseases","rust","scab"]].to_csv(submission_filename, index=False)
            #print(history)                   
    return model, [history, history2]