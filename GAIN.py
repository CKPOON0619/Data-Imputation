#%% Packages
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from datetime import datetime

#%% Tensorboard logging
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = './tf_logs' + current_time + "/"

#%% System Parameters 

# Training Data 
file='measureGenerator'

# 1. Missing rate
p_miss = 0.5
# 3. Hint rate
p_hint = 0.5
# 4. Loss Hyperparameters
alpha = 1
# 5. Train Rate
train_rate = 0.8
# 6. iteration
iteration=4000
# 7. Dropout Rate
drop_rate=0

#%% Data

# Data generation
file='measureGenerator'
rawData = np.genfromtxt("../test inputs/{}.csv".format(file), delimiter=",",skip_header=1)

# Parameters
sample_size = len(rawData)
Dim = len(rawData[0,:])
train_size=int(sample_size*train_rate)

#%% Helpers
# Normalization (0 to 1)
def createNormaliser(dataMin,dataMax):
    '''
    dataMin:tensor of data min
    dataMax:tensor of data  max
    return: range normalised tensor
    '''
    return lambda rawDataTensor:(rawDataTensor-dataMin)/(dataMax-dataMin+1e-6)

# Return real value
def createDenormaliser(dataMin,dataMax):
    '''
    dataMin:tensor of data min
    dataMax:tensor of data  max
    return: de-normalised tensor
    '''
    return lambda dataTensor:(dataTensor)*(dataMax-dataMin+1e-6)+dataMin
# Create Mask
def createMask(data,maskRatio):
    '''
    data: tensor to be masked
    maskRatio: proportion of entries to be marked as 1
    return: 0,1 matrix of the same shape as data
    '''
    return tf.dtypes.cast((tf.random.uniform(tf.shape(data),minval=0,maxval=1)>(1-maskRatio)),dtype=tf.float32)

# Create generator/discriminator counterparts
def compositLayers(layer_sizes,dropout_rate=drop_rate):
    layers = []
    for idx in range(len(layer_sizes)-2):
        layers+=[tf.keras.layers.Dense(layer_sizes[idx], activation=tf.nn.relu, kernel_initializer='glorot_normal')]
        layers+=[tf.keras.layers.Dropout(dropout_rate)]
    
    layers+=[tf.keras.layers.Dense(layer_sizes[-2], activation=tf.nn.relu, kernel_initializer='glorot_normal')]
    layers=layers+[tf.keras.layers.Dense(layer_sizes[-1], activation=tf.nn.sigmoid, kernel_initializer='glorot_normal')]
    return tf.keras.Sequential(layers)
