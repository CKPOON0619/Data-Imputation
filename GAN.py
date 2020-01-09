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

def createMasknHint(X,missRate,hintRate):
    mask=createMask(X,1-missRate)
    hintMask=createMask(X,hintRate)
    hints=hintMask*mask+(1-hintMask)*0.5
    return mask,hintMask,hints

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
def getGeneratorFakeLoss(mask,discriminated_probs):
    ## Likelinhood loss caused by discriminable values
    return -tf.reduce_mean((1-mask) * tf.math.log(discriminated_probs + 1e-8))
def getGeneratorTruthLoss(mask,X,generated_X):
    ## Regulating term for alteration of known truth
    return tf.reduce_mean((mask*X - mask * generated_X)**2) / tf.reduce_mean(mask)

def getHiddenTruthDiscrimination(mask,hintMask,discriminated_probs):
    ## Check if discriminator correctly predicted real data 
    return tf.gather_nd(discriminated_probs,tf.where(mask*(1-hintMask)))
def getHiddenFakeDiscrimination(mask,hintMask,discriminated_probs):
    ## Check if discriminator correctly predicted generated(fake) data 
    return tf.gather_nd(discriminated_probs,tf.where((1-mask)*(1-hintMask)))
def getHiddenFakeGeneratedError(mask,hintMask,X,generated_X):
    ## Check the difference between generated value and actual value
    return tf.gather_nd((generated_X-X),tf.where((1-mask)*(1-hintMask)))

def getGeneratorLoss(alpha,discriminated_probs,X,generated_X,mask):
    return getGeneratorFakeLoss(mask,discriminated_probs)+alpha*getGeneratorTruthLoss(mask,X,generated_X)


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
    def __init__(self, logdir= getcwd()+'\\logs\\tf_logs' + datetime.now().strftime("%Y%m%d-%H%M%S"), hyperParams=defaultParams, optimizer=tf.keras.optimizers.Adam()):
        super(GAN, self).__init__()
        self.iter=0
        self.__dict__.update(hyperParams)
        self.optimizer = optimizer
        self.reset(logdir)

    def reset(self,logdir):
        self.epoch = tf.Variable(0,dtype=tf.int64)
        os.makedirs(logdir, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(logdir)
        print('tensorboard --logdir {}'.format(logdir)+' --host localhost')

    def calcLoss(self,X,generator,discriminator):
        [mask,_,hints]=createMasknHint(X,self.p_miss,self.p_hint)
        [generated_X,X_hat]=generate(generator,X,mask)
        discriminated_probs=discriminate(discriminator,X_hat,hints)
        generator_loss=getGeneratorLoss(self.alpha,discriminated_probs,X,generated_X,mask)
        discriminator_loss=getDiscriminatorLoss(discriminated_probs,mask,self.p_miss)
        return generator_loss,discriminator_loss
    
    @tf.function
    def performanceLog(self,prefix,X,generator,discriminator):
        [mask,hintMask,hints]=createMasknHint(X,self.p_miss,self.p_hint)
        [generated_X,X_hat]=generate(generator,X,mask)
        discriminated_probs=discriminate(discriminator,X_hat,hints)

        G_fakeLoss=getGeneratorFakeLoss(mask,discriminated_probs)
        G_truthLoss=getGeneratorTruthLoss(mask,X,generated_X)
        with self.summary_writer.as_default():
            tf.summary.scalar(prefix+': generator_fakeLoss', G_fakeLoss, step=self.epoch)
            tf.summary.scalar(prefix+': generator_truthLoss', G_truthLoss, step=self.epoch)
            tf.summary.scalar(prefix+': discriminator_loss', getDiscriminatorLoss(discriminated_probs,mask,self.p_miss), step=self.epoch) 
            tf.summary.histogram(prefix+': hidden truth discrimination',getHiddenTruthDiscrimination(mask,hintMask,discriminated_probs), step=self.epoch) 
            tf.summary.histogram(prefix+': hidden fake discrimination',getHiddenFakeDiscrimination(mask,hintMask,discriminated_probs), step=self.epoch) 
            tf.summary.histogram(prefix+': hidden fake generation error',getHiddenFakeGeneratedError(mask,hintMask,X,generated_X), step=self.epoch) 
            self.summary_writer.flush()

    @tf.function
    def trainWithBatch(self,dataBatch,generator,discriminator):
        with tf.GradientTape(persistent=True) as tape:
            G_loss,D_loss=self.calcLoss(dataBatch,generator,discriminator)
        # Learning and update weights
        G_loss_gradients = tape.gradient(G_loss,generator.trainable_variables)
        D_loss_gradients = tape.gradient(D_loss,discriminator.trainable_variables)
        self.optimizer.apply_gradients(zip(G_loss_gradients, generator.trainable_variables))
        self.optimizer.apply_gradients(zip(D_loss_gradients, discriminator.trainable_variables))
        self.epoch.assign_add(1)
        return G_loss,D_loss
            