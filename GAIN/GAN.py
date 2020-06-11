#%% Packages
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from datetime import datetime
from os import getcwd
from components.Discriminator import myDiscriminator

#%%
class GAN():
    '''
    Generative Adversarial Net(GAN) structure for Generative Adversarial Information Net(GAIN).
    Args:
        logdir: logging directory for tensorboard. Default to be "./logs/tf_logs(dateTime)"
        hyperParams: hyperparameters for the GAN model, default to be {'p_miss': 0.5, 'alpha': 0, episode_num: 5}
                    p_miss: missing rate of data for rebalancing during training for generator
                    alpha: regulation parameters.
                    episode_num: the number of episode the discriminator would be unrolled.
        optimizer: A tensorflow optimizer class object
    '''
    def __init__(self,summary_writer=False, hyperParams={}, optimizer=tf.keras.optimizers.Adam()):
        self.alpha=0
        self.p_miss=0.5
        self.episode_num=5
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

    def tensorboard_log(self,prefix,dataBatch,mask,hints,hintMask,generator,discriminator):
        generated_data=generator.generate(dataBatch,mask)
        adjusted_generated_data=mask*dataBatch+generated_data*(1-mask)
        discriminator.performance_log(self.summary_writer,prefix,adjusted_generated_data,hints,mask,hintMask,self.p_miss,self.epoch)
        generator.performance_log(self.summary_writer,prefix,dataBatch,mask,hints,hintMask,discriminator.discriminate,self.epoch)
        self.epoch.assign_add(1)

    def train(self,dataBatch,mask,hints,generator,discriminator,steps=1):
        '''
        A function that train generator and respective discriminator.
        Args:
            dataBatch: data input.
            mask: mask of the data, 0,1 matrix of the same shape as data.
            hints: hints matrix with 1,0.5,0 values. Same shape as data.
            generator: A generator model for the GAIN structure.
            discriminator: A discriminator model for the GAIN structure.
            steps: The number of steps training the discriminator each time before training the generator.
        '''
        for i in range(0,steps):
            generated_data=generator.generate(dataBatch,mask)
            adjusted_generated_data=mask*dataBatch+generated_data*(1-mask)
            discriminator.train(adjusted_generated_data,mask,hints,self.p_miss,self.optimizer)
        generator.train(dataBatch,mask,hints,discriminator.discriminate,self.optimizer,self.alpha)

    def train_with_unrolling(self,dataBatch,mask,hints,generator,discriminator,unrolling_steps=1):
        '''
        A function that train generator and respective discriminator.
        Args:
            dataBatch: data input.
            mask: mask of the data, 0,1 matrix of the same shape as data.
            hints: hints matrix with 1,0.5,0 values. Same shape as data.
            generator: A generator model for the GAIN structure.
            discriminator: A discriminator model for the GAIN structure.
            steps: The number of steps training the discriminator each time before training the generator.
        '''
        generated_data=generator.generate(dataBatch,mask)
        adjusted_generated_data=mask*dataBatch+generated_data*(1-mask)
        discriminator.train(adjusted_generated_data,mask,hints,self.p_miss,self.optimizer)
        discriminator.unroll(adjusted_generated_data,mask,hints,self.optimizer,self.p_miss,self.alpha,unrolling_steps)
        generator.train(dataBatch,mask,hints,discriminator.discriminate_with_episodes,self.optimizer)
 
        
            
            