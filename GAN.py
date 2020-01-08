#%% Packages
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from datetime import datetime
from os import getcwd
#%% Helpers

# Create Mask
def createMask(data,maskRatio):
    '''
    data: tensor to be masked
    maskRatio: proportion of entries to be marked as 1
    return: 0,1 matrix of the same shape as data
    '''
    return tf.dtypes.cast((tf.random.uniform(tf.shape(data),minval=0,maxval=1)>(1-maskRatio)),dtype=tf.float32)

def createInputs(X,missRate,hintRate):
    mask=createMask(X,1-missRate)
    hintMask=createMask(X,hintRate)
    hints=hintMask*mask+(1-hintMask)*0.5
    masked_X=mask*X
    return mask,masked_X,hints,hintMask

def generate(generator,x,mask):
    # Create generated x
    generated_x=generator(x,mask)
    # Create generated probs
    x_hat=mask*x+generated_x*(1-mask)
    return generated_x,x_hat

def discriminate(discriminator,x_hat,hints):
    discriminated_probs=discriminator(x_hat,hints)
    return discriminated_probs
    
def getDiscriminatorLoss(discriminated_probs,mask,missingRate):
    ## log Likelinhood comparison to ground truth + sample rebalancing
    discriminator_loss=-tf.reduce_mean(mask * tf.math.log(discriminated_probs + 1e-8) + (1-missingRate)/missingRate*(1-mask) * tf.math.log(1. - discriminated_probs + 1e-8))
    return discriminator_loss

    ## Generator loss:
def getGeneratorLoss(alpha,discriminated_probs,X,generated_X,mask):
    ## Likelinhood loss caused by discriminable values
    generator_fakeLoss = -tf.reduce_mean((1-mask) * tf.math.log(discriminated_probs + 1e-8))
    ## Regulating term for alteration of known truth
    generator_truthLoss= tf.reduce_mean((mask*X - mask * generated_X)**2) / tf.reduce_mean(mask)
    ## Total generator loss
    generator_loss=generator_fakeLoss+alpha*generator_truthLoss
    return generator_loss

#%% GAN Model

# Model params
defaultParams={
    'p_miss': 0.5, 
    'p_hint': 0.5, 
    'alpha': 1, 
    'iteration': 4000, 
    'drop_rate': 0, 
    'batch_size':0.2
    }
    
class GAN(Model):
    def __init__(self, logdir= getcwd()+'\\logs\\tf_logs' + datetime.now().strftime("%Y%m%d-%H%M%S") + ' --host localhost', hyperParams=defaultParams, optimizer=tf.keras.optimizers.Adam()):
        super(GAN, self).__init__()
        self.iter=0
        self.__dict__.update(hyperParams)
        self.optimizer = optimizer
        self.reset(logdir)

    def reset(self,logdir):
        self.epoch = tf.Variable(0,dtype=tf.int64)
        os.makedirs(logdir, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(logdir)
        print('tensorboard --logdir {}'.format(logdir))

    def calcLoss(self,X,generator,discriminator):
        [mask,masked_X,hints,hintMask]=createInputs(X,self.p_miss,self.p_hint)
        [generated_X,X_hat]=generate(generator,X,mask)
        discriminated_probs=discriminate(discriminator,X_hat,hints)
        generator_loss=getGeneratorLoss(self.alpha,discriminated_probs,X,generated_X,mask)
        discriminator_loss=getDiscriminatorLoss(discriminated_probs,mask,self.p_miss)
        return generator_loss,discriminator_loss

    @tf.function
    def trainWithBatch(self,dataBatch,generator,discriminator):
        with tf.GradientTape(persistent=True) as tape:
            G_loss,D_loss=self.calcLoss(dataBatch,generator,discriminator)
        # Learning and update weights
        G_loss_gradients = tape.gradient(G_loss,generator.trainable_variables)
        D_loss_gradients = tape.gradient(D_loss,discriminator.trainable_variables)
        self.optimizer.apply_gradients(zip(G_loss_gradients, generator.trainable_variables))
        self.optimizer.apply_gradients(zip(D_loss_gradients, discriminator.trainable_variables))
        return G_loss,D_loss
            