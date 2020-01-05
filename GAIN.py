#%% Packages
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from datetime import datetime


#%% System Parameters 

# Training Data 
file='measureGenerator'
defaultParams={
    'p_miss': 0.5, 
    'p_hint': 0.5, 
    'alpha': 1, 
    'train_rate': 0.8, 
    'iteration': 4000, 
    'drop_rate': 0, 
    'batch_size':0.2
    }


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

#%% Adversial Opponents

#Generator
class myGenerator(Model):
    def __init__(self,body=compositLayers([Dim*12,Dim*6,Dim*3,Dim*2,Dim*3,Dim*6,Dim*12,Dim],0.2)):
        super(myGenerator, self).__init__()
        self.body = body

    def call(self,x,mask):
        masked_x=mask*x
        mask_sample=(1-mask)*tf.random.uniform(tf.shape(x),minval=0,maxval=1,dtype=tf.float32)
        return self.body(tf.concat(axis = 1, values = [masked_x,mask_sample,mask]))+masked_x

#Discriminator
class myDiscriminator(Model):
    def __init__(self,body=compositLayers([Dim*12,Dim*6,Dim*3,Dim*2,Dim*3,Dim*6,Dim*12,Dim],0.2)):
        super(myDiscriminator, self).__init__()
        self.body = body

    def call(self,x_hat,hints):
        return self.body(tf.concat(axis = 1, values = [x_hat,hints]))

#%% GAN Model
class MyModel(Model):
    def __init__(self,logdir= './tf_logs' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/",generator=myGenerator(),discriminator=myDiscriminator(),optimizer=tf.keras.optimizers.Adam()):
        super(MyModel, self).__init__()
        self.generator = generator
        self.discriminator=discriminator
        self.optimizer=optimizer

        #Internal parameters
        self.iter=0
        self.epoch=tf.Variable(0,dtype=tf.int64)
        os.makedirs(logdir, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(logdir)
        print('tensorboard --logdir {}'.format(logdir))

