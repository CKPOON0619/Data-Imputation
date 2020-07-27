#%% Packages
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from datetime import datetime
from os import getcwd
from components.Orchestrator import Orchestrator
from components.Critic import myCritic

def generate_random(data_batch,mask):
    random_generated=tf.random.uniform(tf.shape(data_batch),minval=0,maxval=1,dtype=tf.float32)
    return random_generated*(1-mask)+data_batch*mask
def get_test_mask(x):
    '''
    Produce a test mask for the distribution of the last column.

    Args:
        x: Input data.
    Returns:
        An array of test mask with 0 at the last entries of each row and 1 in the rest of the entries.
    '''
    shape=tf.shape(x)
    testMask=tf.tile(tf.concat([tf.ones([1,shape[1]-1],dtype=tf.float32),tf.zeros([1,1],dtype=tf.float32)],axis=1),[shape[0],1])
    return testMask
def get_last_column(x):
    '''
    Collect the last column of matrx x.

    Args:
        x: Input data.
    Returns:
        All last entries of each row of x.
    '''
    return tf.gather(x,[tf.shape(x)[1]-1],axis=1)
def get_generated_value_errors(mask,hint_mask,x,generated_x):
    '''
    Get the values of the generated values that are unknown to the generator.

    Args:
        mask: a matrix with the same size as discriminated_probs. Entry value 1 indicate a genuine value, value 0 indicate missing(generated) value.
        hint_mask: mask for creating hints with 1,0 values. Same size as discriminated_probs.
        x: Input data.
        generated_x: data generated by the generator.
    Returns:
        values of the generated values that are unknown to the generator.
    '''
    ## Check the difference between generated value and actual value
    return tf.gather_nd((generated_x-x),tf.where((1-mask)*(1-hint_mask)))

def get_total_generator_truth_error(data_batch,generated_data,mask):
    '''
    Get the logLoss of generated true values.

    Args:
        x: Input data.
        generated_x: generated data by the generator.
        mask:a matrix with the same size as x. Entry value 1 indicate a genuine value, value 0 indicate missing(generated) value.
        
    Returns:
        logLoss value contributed by genuine value reconstructed by the generator.
    '''
    ## Regulating term for alteration of known truth
    return tf.reduce_sum(mask*(data_batch-generated_data)**2) / tf.reduce_sum(mask)


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
    def __init__(self,generator,critic,summary_writer=False, hyperParams={}, optimizer=tf.keras.optimizers.Adam()):
        self.critic=critic
        self.generator=generator
        self.p_miss=0.5
        self.alpha=10
        self.episode_num=5
        super().__init__(summary_writer=summary_writer, hyperParams=hyperParams, optimizer=optimizer)

    def tensorboard_log(self,prefix,data_batch,mask,hints,hint_mask):
        tau=tf.random.uniform([tf.shape(data_batch)[0],1], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None)
        
        generated_data=self.generator.generate(data_batch,mask)
        adjusted_generated_data=mask*data_batch+generated_data*(1-mask)
        interpolated_data=tau*adjusted_generated_data+(1-tau)*data_batch             
  
        lastColMask=get_test_mask(data_batch)
        lastColHints=lastColMask+(1-lastColMask)*0.5
        lastColMasked_sample=lastColMask*data_batch+(1-lastColMask)*tf.random.uniform(tf.shape(data_batch),minval=0,maxval=1,dtype=tf.float32)
        generatedLastCol=self.generator.body(tf.concat(axis = 1, values = [lastColMasked_sample,lastColMask]))
        randomLastColCritics=self.critic.criticise(lastColMasked_sample,lastColHints)
        
        generated_critics=self.critic.criticise(adjusted_generated_data,hints)
        blinded_generated_genuine_critics=tf.gather_nd(generated_critics,tf.where(mask*(1-hint_mask)))
        hinted_generated_genuine_critics=tf.gather_nd(generated_critics,tf.where(mask*(hint_mask)))
        generated_genuine_critics=tf.gather_nd(generated_critics,tf.where(mask))
        
        blinded_generated_fake_critics=tf.gather_nd(generated_critics,tf.where((1-mask)*(1-hint_mask)))  
        hinted_generated_fake_critics=tf.gather_nd(generated_critics,tf.where((1-mask)*(hint_mask)))
        generated_fake_critics=tf.gather_nd(generated_critics,tf.where((1-mask)))
        
        mean_critics_diff=self.critic.calc_critic_diff_ind(data_batch,adjusted_generated_data,mask,hints,self.p_miss)
        penalty_regulation=self.critic.calc_critic_penalty(interpolated_data,hints)
        critic_loss=mean_critics_diff+self.alpha*penalty_regulation
        
        with self.summary_writer.as_default():
            tf.summary.histogram(prefix+' generated_fake_critics',generated_fake_critics, step=self.epoch) 
            tf.summary.histogram(prefix+' generated_genuine_critics',generated_genuine_critics, step=self.epoch) 
            tf.summary.histogram(prefix+' blinded_generated_fake_critics',blinded_generated_fake_critics, step=self.epoch) 
            tf.summary.histogram(prefix+' blinded_generated_genuine_critics',blinded_generated_genuine_critics, step=self.epoch) 
            tf.summary.histogram(prefix+' hinted_generated_genuine_critics',hinted_generated_genuine_critics, step=self.epoch) 
            tf.summary.histogram(prefix+' hinted_generated_fake_critics',hinted_generated_fake_critics, step=self.epoch) 
            tf.summary.scalar(prefix+' mean_critics_diff',mean_critics_diff, step=self.epoch) 
            tf.summary.scalar(prefix+' penalty_regulation',penalty_regulation, step=self.epoch) 
            tf.summary.scalar(prefix+' critic_loss',critic_loss, step=self.epoch) 
            
            tf.summary.histogram(prefix+' critic distribution of uniformly random last column',get_last_column(randomLastColCritics), step=self.epoch) 
            tf.summary.histogram(prefix+' generated last column distribution',get_last_column(generatedLastCol), step=self.epoch) 
            tf.summary.histogram(prefix+' actual last column distribution',get_last_column(data_batch), step=self.epoch) 
            self.summary_writer.flush() 
        self.epoch.assign_add(1)
    
    @tf.function
    def tensorboard_log_with_random(self,prefix,data_batch,mask,hints,hint_mask):
        tau=tf.random.uniform([tf.shape(data_batch)[0],1], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None)
        
        generated_data=generate_random(data_batch,mask)
        adjusted_generated_data=mask*data_batch+generated_data*(1-mask)
        interpolated_data=tau*adjusted_generated_data+(1-tau)*data_batch
                
        lastColMask=get_test_mask(data_batch)
        lastColHints=lastColMask+(1-lastColMask)*0.5
        lastColMasked_sample=lastColMask*data_batch+(1-lastColMask)*tf.random.uniform(tf.shape(data_batch),minval=0,maxval=1,dtype=tf.float32)
        randomLastColCritics=self.critic.criticise(lastColMasked_sample,lastColHints)
        
        generated_critics=self.critic.criticise(adjusted_generated_data,hints)
        blinded_generated_genuine_critics=tf.gather_nd(generated_critics,tf.where(mask*(1-hint_mask)))
        hinted_generated_genuine_critics=tf.gather_nd(generated_critics,tf.where(mask*(hint_mask)))
        
        generated_genuine_critics=tf.gather_nd(generated_critics,tf.where(mask))
        blinded_generated_fake_critics=tf.gather_nd(generated_critics,tf.where((1-mask)*(1-hint_mask)))
        hinted_generated_fake_critics=tf.gather_nd(generated_critics,tf.where((1-mask)*(hint_mask)))
        generated_fake_critics=tf.gather_nd(generated_critics,tf.where((1-mask)))
        mean_critics_diff=self.critic.calc_critic_diff_ind(data_batch,adjusted_generated_data,mask,hints,self.p_miss)
        penalty_regulation=self.critic.calc_critic_penalty(interpolated_data,hints)
        critic_loss=mean_critics_diff+self.alpha*penalty_regulation
        
        with self.summary_writer.as_default():
            tf.summary.histogram(prefix+' generated_fake_critics',generated_fake_critics, step=self.epoch) 
            tf.summary.histogram(prefix+' generated_genuine_critics',generated_genuine_critics, step=self.epoch) 
            tf.summary.histogram(prefix+' blinded_generated_fake_critics',blinded_generated_fake_critics, step=self.epoch) 
            tf.summary.histogram(prefix+' blinded_generated_genuine_critics',blinded_generated_genuine_critics, step=self.epoch) 
            tf.summary.histogram(prefix+' hinted_generated_genuine_critics',hinted_generated_genuine_critics, step=self.epoch) 
            tf.summary.histogram(prefix+' hinted_generated_fake_critics',hinted_generated_fake_critics, step=self.epoch) 
            tf.summary.histogram(prefix+' critic distribution of uniformly random last column',get_last_column(randomLastColCritics), step=self.epoch) 
            tf.summary.histogram(prefix+' hidden value generation errors',get_generated_value_errors(mask,hint_mask,data_batch,generated_data), step=self.epoch) 
            tf.summary.scalar(prefix+' mean_critics_diff',mean_critics_diff, step=self.epoch) 
            tf.summary.scalar(prefix+' penalty_regulation',penalty_regulation, step=self.epoch) 
            tf.summary.scalar(prefix+' critic_loss',critic_loss, step=self.epoch) 
            self.summary_writer.flush()
        
        self.epoch.assign_add(1)
        return critic_loss
        

    def train_critic_with_random(self,data_batch,mask,hints):
        '''
        A function that train critic with randomly generated data.
        Args:
            data_batch: data input.
            mask: mask of the data, 0,1 matrix of the same shape as data.
            hints: hints matrix with 1,0.5,0 values. Same shape as data.
            steps: The number of steps training the critic each time before training the generator.
        '''
        generated_data=generate_random(data_batch,mask)
        adjusted_generated_data=mask*data_batch+generated_data*(1-mask)
        self.critic.train(data_batch,adjusted_generated_data,mask,hints,self.alpha,self.p_miss,self.optimizer)

    @tf.function
    def train_critic(self,data_batch,mask,hints,steps=1):
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
            generated_data=self.generator.generate(data_batch,mask)
            adjusted_generated_data=mask*data_batch+generated_data*(1-mask)
            self.critic.train(data_batch,adjusted_generated_data,mask,hints,self.alpha,self.p_miss,self.optimizer)
        
    @tf.function
    def train_generator(self,data_batch,mask,hints):
        with tf.GradientTape(persistent=True) as tape:
            generated_data=self.generator.generate(data_batch,mask)
            adjusted_generated_data=generated_data*(1-mask)+mask*data_batch
            generated_critics=self.critic.criticise(adjusted_generated_data,hints)
            critic_loss=-tf.reduce_mean(generated_critics*(1-mask))
        loss_gradients = tape.gradient(critic_loss,self.critic.body.trainable_variables)
        self.optimizer.apply_gradients(zip(loss_gradients, self.critic.body.trainable_variables))
        return critic_loss
