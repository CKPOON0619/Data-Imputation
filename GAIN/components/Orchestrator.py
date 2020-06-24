#%% Packages
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from datetime import datetime
from os import getcwd

#%%
class Orchestrator():
    '''
    Orchestrator that orchastrate different model components.
    Args:
        logdir: logging directory for tensorboard. Default to be "./logs/tf_logs(dateTime)"
        hyperParams: hyperparameters for the model.
        optimizer: A tensorflow optimizer class object
    '''
    def __init__(self,summary_writer=False, hyperParams={}, optimizer=tf.keras.optimizers.Adam()):
        self.__dict__.update(hyperParams)
        self.optimizer = optimizer
        self.epoch = tf.Variable(0,dtype=tf.int64)
        if(summary_writer):
            self.summary_writer=summary_writer
        else:
            self.set_logDir()
        
    def setHyperParams(self,hyperParams):
        '''
        A function to update model hyperParams.
        Args: 
            hyperParams: hyperparameters for the GAN model:
                p_miss: missing rate of data
                p_hint: proportion of data entry to be given as known answer to the discriminator. 
                alpha: regulation parameters.
        '''
        self.__dict__.update(hyperParams)

    def set_logDir(self,logdir= getcwd()+'\\logs\\tf_logs' + datetime.now().strftime("%Y%m%d-%H%M%S")):
        '''
        A function to reset logging directory and training epoch.
        Args: 
            logdir: logging directory for tensorboard
        '''
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(logdir)
        print('tensorboard --logdir {}'.format(logdir)+' --host localhost')

# %%


# %%
