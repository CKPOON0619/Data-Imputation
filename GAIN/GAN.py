#%% Packages
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from datetime import datetime
from os import getcwd
from components.Discriminator import myDiscriminator
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

def createHint(mask,hintRate):
    '''
    Args:
        mask: mask of the data, 0,1 matrix of the same shape as data.
        hintRate: the rate of reveal the truth
    return 
        hintMask: mask for creating hints with 1,0 values. Same size as X.
        hints: hints matrix with 1,0.5,0 values. Same size as X.
    '''
    hintMask=createMask(mask,1-hintRate)
    hints=hintMask*mask+(1-hintMask)*0.5
    return hintMask,hints

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


def getTestMask(X):
    '''
    Produce a test mask for the distribution of the last column.

    Args:
        X: Input data.
    Returns:
        An array of test mask with 0 at the last entries of each row and 1 in the rest of the entries.
    '''
    shape=tf.shape(X)
    testMask=tf.tile(tf.concat([tf.ones([1,shape[1]-1],dtype=tf.float32),tf.zeros([1,1],dtype=tf.float32)],axis=1),[shape[0],1])
    return testMask

def getLastColumn(x):
    '''
    Collect the last column of matrx X.

    Args:
        X: Input data.
    Returns:
        All last entries of each row of X.
    '''
    return tf.gather(x,[tf.shape(x)[1]-1],axis=1)

def cloneModel(model,MyModel,inputDim):
    '''
    Clone a model created by our defined model class in which `body` is a kera model.
    
    Args:
        model: the instance of model to be cloned.
        MyModel: the class definition of the model to be cloned.
        inputDim: the number of dimension of the input.
        
    Return:
        A copy of the instance.
    '''
    newModel=MyModel()
    newModel.body=tf.keras.models.clone_model(model.body)
    newModel.body.build((None,inputDim)) # Should detect shape or get it inputted from somewhere.
    return newModel

def cloneWeights(model1,model2):
    '''
    Clone a trainable variables of two instance of a keras model class from one to another.
    
    Args:
        model1: the instance of the model to be cloned.
        model2: the intance of the model to be copied to.
    '''
    for var1,var2 in zip(model1.trainable_variables,model2.trainable_variables):
        var2.assign(var1)

def createEpisodes(discriminator,myDiscriminator,inputDim,episode_num):
    '''
    Create mulitple copies of discriminator instance of instantiated by MyDiscriminator class, 
    returning an array of copies of the instance.
    
    Args:
        discriminator: the instance of model to be cloned.
        myDiscriminator: the class definition of the model to be cloned.
        inputDim: the number of dimension of the input.
        episode_num: the number of copies created.
    
    Return:
        Multiple copies of the instance as an array.
    '''
    episodes=[]
    for i in range(0,episode_num):
        episode=cloneModel(discriminator,myDiscriminator,inputDim)
        cloneWeights(discriminator,episode)
        episodes.append(episode)
    return episodes


def trainDiscriminator(dataBatch,X_hat,mask,hints,discriminator,optimizer,p_miss):
    '''
    A function that train the discriminator against given generator.
    Args:
        dataBatch: data input.
        generator: A generator model for the GAIN structure.
        discriminator: A discriminator model for the GAIN structure.
        optimizer: tensorflow optimizer object.
        p_miss: missing data ratio for balancing loss.
    
    Return: The discrminator loss.
    '''
    with tf.GradientTape(persistent=True) as tape:
        discriminated_probs=discriminate(discriminator,X_hat,hints)
        D_loss=getDiscriminatorLoss(discriminated_probs,mask,p_miss)
    D_loss_gradients = tape.gradient(D_loss,discriminator.trainable_variables)
    optimizer.apply_gradients(zip(D_loss_gradients, discriminator.trainable_variables))
    return D_loss

def run(X,mask,hints,generator,discriminator):
    '''
    Calculate the loss of the generator and discriminator.

    Args:
        X: Input data.
        mask: mask of the data, 0,1 matrix of the same shape as X.
        hints: hints matrix with 1,0.5,0 values. Same shape as X.
        generator: A generator model for the GAIN structure.
        discriminator: A discriminator model for the GAIN structure.
    Returns:
        generated_X: data generated by generator.
        discriminated_probs: probability deduced by the discriminator to discriminate generated data.
    '''
    [generated_X,X_hat]=generate(generator,X,mask)
    discriminated_probs=discriminate(discriminator,X_hat,hints)
    return generated_X,discriminated_probs

def trainDiscriminatorWithGenerator(dataBatch,mask,hints,generator,discriminator,optimizer,p_miss):
    '''
    A function that train the discriminator against given generator.
    
    Args:
        dataBatch: data input.
        generator: A generator model for the GAIN structure.
        discriminator: A discriminator model for the GAIN structure.
        optimizer: A tensorflow optimizer object.
        
    Return:
        The loss of the discrminator.
    '''
    with tf.GradientTape(persistent=True) as tape:
        generated_X,discriminated_probs=run(dataBatch,mask,hints,generator,discriminator)
        D_loss=getDiscriminatorLoss(discriminated_probs,mask,p_miss)
    D_loss_gradients = tape.gradient(D_loss,discriminator.trainable_variables)
    optimizer.apply_gradients(zip(D_loss_gradients, discriminator.trainable_variables))
    return D_loss

def calcMultiGeneratorLoss(X,mask,hints,generator,discriminators,alpha):
    '''
    A function that calculate the loss of the generator against multiple discriminators.
    
    Args:
        dataBatch: data input.
        generator: A generator model for the GAIN structure.
        discriminator: A discriminator model for the GAIN structure.
        optimizer: A tensorflow optimizer object.
    
    Return:
        Total loss of hte discriminator.
    
    '''
    [generated_X,X_hat]=generate(generator,X,mask)
    generator_losses=[]
    for i in range(0,len(discriminators)):
        generator_losses.append(getGeneratorLoss(alpha,discriminate(discriminators[i],X_hat,hints),X,generated_X,mask))
    return tf.math.add_n(generator_losses)


def trainGeneratorWithDiscriminator(dataBatch,mask,hints,generator,discriminator,optimizer,alpha):
    '''
    A function that train the generator against multiple discriminators and return the generator loss.
    
    Args:
        dataBatch: data input.
        generator: A generator model for the GAIN structure.
        discriminator: A discriminator model for the GAIN structure.
        optimizer: A tensorflow optimizer object.
    
    Return:
        Total loss of the generator.
    
    '''
    [generated_X,X_hat]=generate(generator,dataBatch,mask)
    with tf.GradientTape(persistent=True) as tape:
        total_G_episodes_loss=getGeneratorLoss(alpha,discriminate(discriminator,X_hat,hints),X,generated_X,mask)
    # Learning and update weights
    G_loss_gradients = tape.gradient(total_G_episodes_loss,generator.trainable_variables)
    optimizer.apply_gradients(zip(G_loss_gradients, generator.trainable_variables))
    return total_G_episodes_loss

def trainGeneratorWithDiscriminators(dataBatch,mask,hints,generator,discriminators,optimizer,alpha):
    '''
    A function that train the generator against multiple discriminators and return the generator loss.
    
    Args:
        dataBatch: data input.
        generator: A generator model for the GAIN structure.
        discriminator: A list of discriminator class model instances for the GAIN structure.
        optimizer: A tensorflow optimizer object.
    
    Return:
        Total loss of the generator.
    
    '''
    with tf.GradientTape(persistent=True) as tape:
        total_G_episodes_loss=calcMultiGeneratorLoss(dataBatch,mask,hints,generator,discriminators,alpha)
    # Learning and update weights
    G_loss_gradients = tape.gradient(total_G_episodes_loss,generator.trainable_variables)
    optimizer.apply_gradients(zip(G_loss_gradients, generator.trainable_variables))
    return total_G_episodes_loss

def unrollDiscriminator(data_batch,mask,hints,X_hat,discriminator,optimizer,episodes,p_miss,alpha,leap=2):
    '''
    A function that unroll discriminators changes on an array of the discrminator class instances.
    
    Args:
        data_Batch: data input.
        mask: The mask of the data, 0,1 matrix of the same shape as X.
        hints: The hints matrix with 1,0.5,0 values. Same shape as X.
        X_hat: A matrix of generated data where each row a record entry.
        discriminator: A discriminator model for the GAIN structure.
        optimizer: A tensorflow optimizer object.
        episodes: An array of discrimintor class instances.
        p_miss: The missing rate of the mask.
        alpha: A regulation parameters.
        leap: The number of walk before making an episode record of the discriminator.
    
    Return:
        Total loss of the generator.
    '''
    cloneWeights(discriminator,episodes[0])
    episode_num=len(episodes)
    for i in range(0,episode_num-1):
        for j in range(0,leap):
            with tf.GradientTape(persistent=True) as tape:
                discriminated_probs=discriminate(episodes[i],X_hat,hints)
                D_loss=getDiscriminatorLoss(discriminated_probs,mask,p_miss)
            D_loss_gradients = tape.gradient(D_loss,episodes[i].trainable_variables)
            optimizer.apply_gradients(zip(D_loss_gradients,episodes[i].trainable_variables))
        cloneWeights(episodes[i],episodes[i+1])
        
    with tf.GradientTape(persistent=True) as tape:
        discriminated_probs=discriminate(episodes[episode_num-1],X_hat,hints)
        D_loss=getDiscriminatorLoss(discriminated_probs,mask,p_miss)
    D_loss_gradients = tape.gradient(D_loss,episodes[episode_num-1].trainable_variables)
    optimizer.apply_gradients(zip(D_loss_gradients,episodes[episode_num-1].trainable_variables))        
            
            
    # trainDiscriminatorWithGenerator(data_batch,mask,hints,generator,episodes[episode_num-1],optimizer,p_miss)

#%% GAN Model
    
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

    def calcLoss(self,X,mask,hints,generator,discriminator):
        '''
        Calculate the loss of the generator and discriminator.

        Args:
            X: Input data.
            mask: mask of the data, 0,1 matrix of the same shape as X.
            hints: hints matrix with 1,0.5,0 values. Same shape as X.
            generator: A generator model for the GAIN structure.
            discriminator: A discriminator model for the GAIN structure.
        Returns:
            generator_loss: loss result of the generator.1
            discriminator_loss: loss result of the discriminator.
        '''
        [generated_X,X_hat]=generate(generator,X,mask)
        discriminated_probs=discriminate(discriminator,X_hat,hints)
        generator_loss=getGeneratorLoss(self.alpha,discriminated_probs,X,generated_X,mask)
        discriminator_loss=getDiscriminatorLoss(discriminated_probs,mask,self.p_miss)
        return generator_loss,discriminator_loss
        
    @tf.function
    def calcMultiGeneratorLoss(self,X,mask,hints,generator,discriminators):
        '''
        A function that calculate the loss of the generator against multiple discriminators.
        
        Args:
            dataBatch: data input.
            generator: A generator model for the GAIN structure.
            discriminator: A discriminator model for the GAIN structure.
            optimizer: A tensorflow optimizer object.
        
        Return:
            Total loss of the discriminator.
        '''
        return calcMultiGeneratorLoss(X,mask,hints,generator,discriminators,self.alpha)
    
    def initialiseUnRolling(self,discriminator,myDiscriminator,dim):
        '''
        Create mulitple copies of discriminator instance of instantiated by MyDiscriminator class, 
        returning an array of copies of the instance.
        
        Args:
            discriminator: the instance of model to be cloned.
            myDiscriminator: the class definition of the model to be cloned.
            inputDim: the number of dimension of the input.
            episode_num: the number of copies created.
        
        Return:
            Multiple copies of the instance as an array.
        '''
        if not hasattr(self, 'episodes'):
            self.episodes=createEpisodes(discriminator,myDiscriminator,dim*2,self.episode_num)
        return self.episodes
    
    @tf.function
    def trainDiscriminator(self,dataBatch,X_hat,mask,hints,discriminator):
        '''
        A function that train the discriminator against given generator.
        Args:
            dataBatch: data input.
            generator: A generator model for the GAIN structure.
            discriminator: A discriminator model for the GAIN structure.
        '''
        return trainDiscriminator(dataBatch,X_hat,mask,hints,discriminator,self.optimizer,self.p_miss)
    
    @tf.function
    def trainDiscriminatorWithMemory(self,dataBatch,X_hat,mask_recalled,hints,discriminator_with_memory):
        '''
        A function that train the discriminator against given generator.
        Args:
            dataBatch: The data input.
            X_hat: A matrix of generated data where each row a record entry.
            mask_recalled: Mask recalled from mask memoriser, sync-ed with discriminator memory.
            discriminator_with_memory: A discriminator model wrapped in memoriser module.
        '''
        return trainDiscriminator(dataBatch,X_hat,mask_recalled,hints,discriminator_with_memory,self.optimizer,self.p_miss)
    
    
    @tf.function
    def trainDiscriminatorWithGenerator(self,dataBatch,mask,hints,generator,discriminator):
        '''
        A function that train the discriminator against given generator.
        Args:
            dataBatch: data input.
            generator: A generator model for the GAIN structure.
            discriminator: A discriminator model for the GAIN structure.
        '''
        return trainDiscriminatorWithGenerator(dataBatch,mask,hints,generator,discriminator,self.optimizer,self.p_miss)
    
    @tf.function
    def trainGeneratorWithDiscriminator(self,dataBatch,mask,hints,generator,discriminator):
        '''
        A function that train that generator against respective discriminator.
        Args:
            dataBatch: data input.
            generator: A generator model for the GAIN structure.
            discriminator: A discriminator model for the GAIN structure.
            customMask: a custom mask to be applied, if not provided, a random mask would be generated.
        '''
        return trainGeneratorWithDiscriminator(dataBatch,mask,hints,generator,discriminator,self.optimizer,self.alpha)
    
    @tf.function
    def performanceLog(self,prefix,X,mask,hintMask,hints,generator,discriminator):
        '''
        A performance logger linked to tensorboard logs in the logdir. 
        Calculations are executed in graph mode.

        Args:
            prefix: prefix for the name for tensorboard logs.
            X: Input data.
            generator: A generator model for the GAIN structure.
            discriminator: A discriminator model for the GAIN structure.
        '''
        [generated_X,X_hat]=generate(generator,X,mask)
        discriminated_probs=discriminate(discriminator,X_hat,hints)

        [generatedLastCol,_]=generate(generator,X,getTestMask(X))
        
        G_fakeLoss=getGeneratorFakeLoss(mask,discriminated_probs)
        G_truthLoss=getGeneratorTruthLoss(mask,X,generated_X)
        with self.summary_writer.as_default():
            tf.summary.scalar(prefix+' generator_fakeLoss', G_fakeLoss, step=self.epoch)
            tf.summary.scalar(prefix+' generator_truthLoss', G_truthLoss, step=self.epoch)
            tf.summary.scalar(prefix+' discriminator_loss', getDiscriminatorLoss(discriminated_probs,mask,self.p_miss), step=self.epoch) 
            tf.summary.histogram(prefix+' hidden truth discrimination',getHiddenTruthDiscrimination(mask,hintMask,discriminated_probs), step=self.epoch) 
            tf.summary.histogram(prefix+' hidden fake discrimination',getHiddenFakeDiscrimination(mask,hintMask,discriminated_probs), step=self.epoch) 
            tf.summary.histogram(prefix+' hidden fake generation error',getHiddenFakeGeneratedError(mask,hintMask,X,generated_X), step=self.epoch) 
            tf.summary.histogram(prefix+' generated last column distribution',getLastColumn(generatedLastCol), step=self.epoch) 
            tf.summary.histogram(prefix+' actual last column distribution',getLastColumn(X), step=self.epoch) 
            self.summary_writer.flush()

    @tf.function
    def trainWithSteps(self,dataBatch,mask,hints,generator,discriminator,steps=1):
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
        with tf.GradientTape(persistent=True) as tape:
            generated_X,discriminated_probs=run(dataBatch,mask,hints,generator,discriminator)
            G_loss=getGeneratorLoss(self.alpha,discriminated_probs,dataBatch,generated_X,mask)
            D_loss=getDiscriminatorLoss(discriminated_probs,mask,self.p_miss)
        D_loss_gradients = tape.gradient(D_loss,discriminator.trainable_variables)
        self.optimizer.apply_gradients(zip(D_loss_gradients, discriminator.trainable_variables))

        if(steps and self.epoch%steps==0):
            G_loss_gradients = tape.gradient(G_loss,generator.trainable_variables)
            self.optimizer.apply_gradients(zip(G_loss_gradients, generator.trainable_variables))
        
        self.epoch.assign_add(1)
        return G_loss,D_loss
 
    @tf.function
    def generate(self,generator,x,mask):
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
        return generate(generator,x,mask)
 
    @tf.function
    def unrollDiscriminator(self,data_batch,mask,hints,generator,discriminator,leap=5):
        '''
        A function that unroll discriminators changes on an array of the discrminator class instances.
        
        Args:
            data_Batch: data input.
            mask: The mask of the data, 0,1 matrix of the same shape as X.
            hints: The hints matrix with 1,0.5,0 values. Same shape as X.
            X_hat: A matrix of generated data where each row a record entry.
            discriminator: A discriminator model for the GAIN structure.
            optimizer: A tensorflow optimizer object.
            episodes: An array of discrimintor class instances.
            p_miss: The missing rate of the mask.
            alpha: A regulation parameters.
            leap: The number of walk before making an episode record of the discriminator.
        
        Return:
            Total loss of the generator.
        '''
        unrollDiscriminator(data_batch,mask,hints,generator,discriminator,self.optimizer,self.episodes,self.p_miss,leap)
    
    @tf.function
    def trainGeneratorWithDiscriminators(self,dataBatch,mask,hints,generator,discriminators):
        '''
        A function that train the generator against multiple discriminators and return the generator loss.
        
        Args:
            dataBatch: data input.
            generator: A generator model for the GAIN structure.
            discriminator: A discriminator model for the GAIN structure.
            optimizer: A tensorflow optimizer object.
        
        Return:
            Total loss of the generator.
        '''
        self.epoch.assign_add(1)
        return trainGeneratorWithDiscriminators(dataBatch,mask,hints,generator,discriminators,self.optimizer,self.alpha)
        

    