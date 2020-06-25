#%% Packages
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from datetime import datetime
from os import getcwd
from components.Ochestrator import Orchestrator
from components.Discriminator import myDiscriminator

#%%
class GAN(Orchestrator):
    '''
    GAN orchestrator for Generative Adversarial Information Net(GAIN).
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
        super().__init__(summary_writer=summary_writer, hyperParams=hyperParams, optimizer=optimizer)

    def tensorboard_log(self,prefix,data_batch,mask,hints,hint_mask,generator,discriminator):
        generated_data=generator.generate(data_batch,mask)
        adjusted_generated_data=mask*data_batch+generated_data*(1-mask)
        discriminator.performance_log(self.summary_writer,prefix,adjusted_generated_data,hints,mask,hint_mask,self.p_miss,self.epoch)
        generator.performance_log_with_discrimination(self.summary_writer,prefix,data_batch,mask,hints,hint_mask,discriminator.discriminate,self.epoch)
        self.epoch.assign_add(1)

    @tf.function
    def train(self,data_batch,mask,hints,generator,discriminator,steps=1):
        '''
        A function that train generator and respective discriminator.
        Args:
            data_batch: data input.
            mask: mask of the data, 0,1 matrix of the same shape as data.
            hints: hints matrix with 1,0.5,0 values. Same shape as data.
            generator: A generator model for the GAIN structure.
            discriminator: A discriminator model for the GAIN structure.
            steps: The number of steps training the discriminator each time before training the generator.
        '''
        for i in range(0,steps):
            generated_data=generator.generate(data_batch,mask)
            adjusted_generated_data=mask*data_batch+generated_data*(1-mask)
            discriminator.train(adjusted_generated_data,mask,hints,self.p_miss,self.optimizer)
        generator.train_with_discrimination(data_batch,mask,hints,discriminator.discriminate,self.optimizer,self.alpha)

    def train_with_unrolling(self,data_batch,mask,hints,generator,discriminator,unrolling_steps=1):
        '''
        A function that train generator and respective discriminator.
        Args:
            data_batch: data input.
            mask: mask of the data, 0,1 matrix of the same shape as data.
            hints: hints matrix with 1,0.5,0 values. Same shape as data.
            generator: A generator model for the GAIN structure.
            discriminator: A discriminator model for the GAIN structure.
            steps: The number of steps training the discriminator each time before training the generator.
        '''
        generated_data=generator.generate(data_batch,mask)
        adjusted_generated_data=mask*data_batch+generated_data*(1-mask)
        discriminator.train(adjusted_generated_data,mask,hints,self.p_miss,self.optimizer)
        discriminator.unroll(adjusted_generated_data,mask,hints,self.optimizer,self.p_miss,self.alpha,unrolling_steps)
        generator.train_with_discrimination(data_batch,mask,hints,discriminator.discriminate_with_episodes,self.optimizer)
 
        
            
            