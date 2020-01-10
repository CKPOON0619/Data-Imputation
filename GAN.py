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
    Args:
        data: tensor to be masked
        maskRatio: proportion of entries to be marked as 1
    Returns: 
        0,1 matrix of the same shape as data
    '''
    return tf.dtypes.cast((tf.random.uniform(tf.shape(data),minval=0,maxval=1)>(1-maskRatio)),dtype=tf.float32)

def createMasknHint(X,missRate,hintRate):
    '''
    Args:
        X: data input which determine the shape of the outputs
        missRate: the rate of missing data
        hintRate: the rate of reveal the truth
    return 
        mask: mask for creating missing data with 1,0 values. Same size as X.
        hintMask: mask for creating hints with 1,0 values. Same size as X.
        hints: hints matrix with 1,0.5,0 values. Same size as X.
    '''
    mask=createMask(X,1-missRate)
    hintMask=createMask(X,hintRate)
    hints=hintMask*mask+(1-hintMask)*0.5
    return mask,hintMask,hints

def generate(generator,x,mask):
    '''
    A wrapper around generator to create generated data.
    Args:
        generator: A model that generate results based on original data x and mask. 
        x: a matrix of data where each row a record entry.
        mask: a matrix with the same size as x. Entry value 1 indicate a genuine value, value 0 indicate missing value.
    Returns:
        generated_x: value entries estimated by the generator.
        x_hat: value entries with unmasked generated value corrected to be genuine.
    '''
    # Create generated x
    generated_x=generator(x,mask)
    # Create generated probs
    x_hat=mask*x+generated_x*(1-mask)
    return generated_x,x_hat

def discriminate(discriminator,x_hat,hints):
    '''
    A wrapper around discriminator to create discrimination probabilities.
    Args:
        discriminator: A model that discriminate generated results based on generated data x_hat and hints. 
        x_hat: a matrix of generated data where each row a record entry.
        hints: a matrix with the same size as x_hat. Entry value 1 indicate a genuine value, value 0 indicate missing value, value 0.5 indicate unknown.
    Returns:
        discriminated_probs: probability predicted by discriminator that a data entry is real.
    '''
    discriminated_probs=discriminator(x_hat,hints)
    return discriminated_probs
    
def getDiscriminatorLoss(discriminated_probs,mask,missRate):
    '''
    The loss function for discriminator.

    Args:
        discriminated_probs: probability predicted by discriminator that a data entry is real.
        mask:a matrix with the same size as discriminated_probs. Entry value 1 indicate a genuine value, value 0 indicate missing(generated) value.
        missRate:the rate of missing data
    Returns:
        discriminator_loss: loss value for discriminator
    
    '''
    ## log Likelinhood comparison to ground truth + sample rebalancing
    discriminator_loss=-tf.reduce_mean(mask * tf.math.log(discriminated_probs + 1e-8) + (1-missRate)/missRate*(1-mask) * tf.math.log(1. - discriminated_probs + 1e-8))
    return discriminator_loss

    ## Generator loss:
def getGeneratorFakeLoss(mask,discriminated_probs):
    '''
    The loss function for generator for the generated values.

    Args:
        discriminated_probs: probability predicted by discriminator that a data entry is real.
        mask:a matrix with the same size as discriminated_probs. Entry value 1 indicate a genuine value, value 0 indicate missing(generated) value.
    Returns:
        loss value contributed by generated value by the generator.
    
    '''
    ## Likelinhood loss caused by discriminable values
    return -tf.reduce_mean((1-mask) * tf.math.log(discriminated_probs + 1e-8))
def getGeneratorTruthLoss(mask,X,generated_X):
    '''
    The loss function for generator for the generated values.

    Args:
        X: Input data
        generated_X: generated data by the generator
        mask:a matrix with the same size as X. Entry value 1 indicate a genuine value, value 0 indicate missing(generated) value.
    Returns:
        loss value contributed by genuine value reconstructed by the generator.
    '''
    ## Regulating term for alteration of known truth
    return tf.reduce_mean((mask*X - mask * generated_X)**2) / tf.reduce_mean(mask)

def getHiddenTruthDiscrimination(mask,hintMask,discriminated_probs):
    '''
    Get the discrimination probabilities of the genuine values that are unknown to the discriminator.

    Args:
        mask: a matrix with the same size as discriminated_probs. Entry value 1 indicate a genuine value, value 0 indicate missing(generated) value.
        hintMask: mask for creating hints with 1,0 values. Same size as discriminated_probs.
        discriminated_probs: probability predicted by discriminator that a data entry is real.
    Returns:
        discrimination probabilities of the genuine values that are unknown to the discriminator.
    '''
    ## Check if discriminator correctly predicted real data 
    return tf.gather_nd(discriminated_probs,tf.where(mask*(1-hintMask)))
def getHiddenFakeDiscrimination(mask,hintMask,discriminated_probs):
    '''
    Get the discrimination probabilities of the generated values that are unknown to the discriminator.

    Args:
        mask: a matrix with the same size as discriminated_probs. Entry value 1 indicate a genuine value, value 0 indicate missing(generated) value.
        hintMask: mask for creating hints with 1,0 values. Same size as discriminated_probs.
        discriminated_probs: probability predicted by discriminator that a data entry is real.
    Returns:
        discrimination probabilities of the generated values that are unknown to the discriminator.
    '''
    ## Check if discriminator correctly predicted generated(fake) data 
    return tf.gather_nd(discriminated_probs,tf.where((1-mask)*(1-hintMask)))
def getHiddenFakeGeneratedError(mask,hintMask,X,generated_X):
    '''
    Get the values of the generated values that are unknown to the generator.

    Args:
        mask: a matrix with the same size as discriminated_probs. Entry value 1 indicate a genuine value, value 0 indicate missing(generated) value.
        hintMask: mask for creating hints with 1,0 values. Same size as discriminated_probs.
        X: Input data.
        generated_X: data generated by the generator.
    Returns:
        values of the generated values that are unknown to the generator.
    '''
    ## Check the difference between generated value and actual value
    return tf.gather_nd((generated_X-X),tf.where((1-mask)*(1-hintMask)))

def getGeneratorLoss(alpha,discriminated_probs,X,generated_X,mask):
    '''
    Get the loss of the generator.

    Args:
        alpha: a constant for regulation term
        mask: a matrix with the same size as discriminated_probs. Entry value 1 indicate a genuine value, value 0 indicate missing(generated) value.
        X: Input data.
        generated_X: data generated by the generator.
    Returns:
        loss of the generator.
    '''
    return getGeneratorFakeLoss(mask,discriminated_probs)+alpha*getGeneratorTruthLoss(mask,X,generated_X)


#%% GAN Model

# Model params
defaultParams={
    'p_miss': 0.5, 
    'p_hint': 0.5, 
    'alpha': 0, 
    'G_train_step':1
    }
    
class GAN(Model):
    '''
    Generative Adversarial Net(GAN) structure for Generative Adversarial Information Net(GAIN).
    Args:
        logdir: logging directory for tensorboard. Default to be "./logs/tf_logs(dateTime)"
        hyperParams: hyperparameters for the GAN model, default to be {'p_miss': 0.5, 'p_hint': 0.5, 'alpha': 1, 'G_train_step':10}
                    p_miss: missing rate of data
                    p_hint: proportion of data entry to be given as known answer to the discriminator. 
                    alpha: regulation parameters.
                    G_train_step: Number of steps that discriminator is trained before we train generator again.
        optimizer: A tensorflow optimizer class object
    '''
    def __init__(self, logdir= getcwd()+'\\logs\\tf_logs' + datetime.now().strftime("%Y%m%d-%H%M%S"), hyperParams={}, optimizer=tf.keras.optimizers.Adam()):
        super(GAN, self).__init__()
        self.iter=0
        self.__dict__.update(defaultParams)
        self.__dict__.update(hyperParams)
        self.optimizer = optimizer
        self.reset(logdir)

    def setHyperParams(self,hyperParams):
        '''
        A function to update model hyperParams.
        Args: 
            hyperParams: hyperparameters for the GAN model:
                p_miss: missing rate of data
                p_hint: proportion of data entry to be given as known answer to the discriminator. 
                alpha: regulation parameters.
                G_train_step: Number of steps that discriminator is trained before we train generator again.
        '''
        self.__dict__.update(hyperParams)

    def reset(self,logdir= getcwd()+'\\logs\\tf_logs' + datetime.now().strftime("%Y%m%d-%H%M%S")):
        '''
        A function to reset logging directory and training epoch.
        Args: 
            logdir: logging directory for tensorboard
        '''
        self.logdir=logdir
        self.epoch = tf.Variable(0,dtype=tf.int64)
        os.makedirs(logdir, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(logdir)
        print('tensorboard --logdir {}'.format(logdir)+' --host localhost')

    def calcLoss(self,X,generator,discriminator):
        '''
        Calculate the loss of the generator and discriminator.

        Args:
            X: Input data.
            generator: A generator model for the GAIN structure.
            discriminator: A discriminator model for the GAIN structure.
        Returns:
            generator_loss: loss result of the generator.1
            discriminator_loss: loss result of the discriminator.
        '''
        [mask,_,hints]=createMasknHint(X,self.p_miss,self.p_hint)
        [generated_X,X_hat]=generate(generator,X,mask)
        discriminated_probs=discriminate(discriminator,X_hat,hints)
        generator_loss=getGeneratorLoss(self.alpha,discriminated_probs,X,generated_X,mask)
        discriminator_loss=getDiscriminatorLoss(discriminated_probs,mask,self.p_miss)
        return generator_loss,discriminator_loss
    
    @tf.function
    def performanceLog(self,prefix,X,generator,discriminator):
        '''
        A performance logger linked to tensorboard logs in the logdir. 
        Calculations are executed in graph mode.

        Args:
            prefix: prefix for the name for tensorboard logs.
            X: Input data.
            generator: A generator model for the GAIN structure.
            discriminator: A discriminator model for the GAIN structure.
        '''
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
        '''
        A function that train that generator with respective discriminator.
        Args:
            dataBatch: data input.
            generator: A generator model for the GAIN structure.
            discriminator: A discriminator model for the GAIN structure.
        '''
        with tf.GradientTape(persistent=True) as tape:
            G_loss,D_loss=self.calcLoss(dataBatch,generator,discriminator)
        # Learning and update weights
       
        D_loss_gradients = tape.gradient(D_loss,discriminator.trainable_variables)
        self.optimizer.apply_gradients(zip(D_loss_gradients, discriminator.trainable_variables))
        if(self.epoch%self.G_train_step==0):
            G_loss_gradients = tape.gradient(G_loss,generator.trainable_variables)
            self.optimizer.apply_gradients(zip(G_loss_gradients, generator.trainable_variables))
        
        self.epoch.assign_add(1)
        return G_loss,D_loss
            