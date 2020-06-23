#%% Packages
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from datetime import datetime
from os import getcwd
from components.Orchestrator import Orchestrator
from components.Critic import myCritic

#%%
class WGAN(Orchestrator):
    '''
    WGAN orchestrator for Generative Adversarial Information Net(GAIN).
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
        self.episode_num=5
        super().__init__(summary_writer=summary_writer, hyperParams=hyperParams, optimizer=optimizer)


    def train(self,data_batch,mask,hints,generator,critic,steps=1):
        '''
        A function that train generator and respective discriminator.
        Args:
            data_batch: data input.
            mask: mask of the data, 0,1 matrix of the same shape as data.
            hints: hints matrix with 1,0.5,0 values. Same shape as data.
            generator: A generator model for the GAIN structure.
            critic: A critic model for the GAIN structure.
            steps: The number of steps training the discriminator each time before training the generator.
        '''
        for i in range(0,steps):
            generated_data=generator.generate(data_batch,mask)
            adjusted_generated_data=mask*data_batch+generated_data*(1-mask)
            critic.train(data_batch,adjusted_generated_data,mask,hints,self.alpha,self.optimizer)
        # generator.train_with_critic(data_batch,mask,hints,critic.criticise,self.optimizer,self.alpha)
            
            