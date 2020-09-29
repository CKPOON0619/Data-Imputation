import tensorflow as tf
from tensorflow import Module

def cloneWeights(model1,model2):
    '''
    Clone a trainable variables of two instance of a keras model class from one to another.
    
    Args:
        model1: the instance of the model to be cloned.
        model2: the intance of the model to be copied to.
    '''
    for var1,var2 in zip(model1.trainable_variables,model2.trainable_variables):
        var2.assign(var1)

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

def discrimination_logLoss(discriminations,hints,mask,missRate):
    ## log Likelinhood comparison to ground truth + sample rebalancing
    discriminator_loss=-tf.reduce_mean(mask * tf.math.log(discriminations + 1e-8) + (1-missRate)/missRate*(1-mask) * tf.math.log(1. - discriminations + 1e-8))
    return discriminator_loss
    

# Discriminator
class myDiscriminator(Module):
    """
    A discriminator class for the GAIN model.

    Args:
        Dim: Dimension of data point.
        body: A kera Model that return a matrix of the same shape as data input. 
    """
    def __init__(self,body=False):
        super(myDiscriminator, self).__init__()
        if(body):
            self.body = body
    
    def save(self,path):
        self.body.save(path)
    
    def load(self,path):
        self.body=tf.keras.models.load_model(path)
        
    def performance_log(self,writer,prefix,adjusted_generated_x,hints,mask,hint_mask,missRate,epoch):
        '''
        To be filled.
        '''    
        discriminated_probs=self.discriminate(adjusted_generated_x,hints)
        discriminator_loss=-tf.reduce_mean(mask * tf.math.log(discriminated_probs + 1e-8) + (1-missRate)/missRate*(1-mask) * tf.math.log(1. - discriminated_probs + 1e-8))
        truth_loss=tf.gather_nd(discriminated_probs,tf.where(mask*(1-hint_mask)))
        fake_loss=tf.gather_nd(discriminated_probs,tf.where((1-mask)*(1-hint_mask)))
        with writer.as_default():
            tf.summary.scalar(prefix+' discriminator loss',discriminator_loss, step=epoch) 
            tf.summary.histogram(prefix+' hidden truth discrimination',truth_loss, step=epoch) 
            tf.summary.histogram(prefix+' hidden fake discrimination',fake_loss, step=epoch) 
            writer.flush()
    
    def intiateUnrolling(self,dim,episode_num=5):
        self.episode_num=5
        if not hasattr(self, 'episodes'):
            self.episodes=[]
        for i in range(0,self.episode_num):
            newEpisode=tf.keras.models.clone_model(self.body)
            newEpisode.build(input_shape=[None,dim*2])
            cloneWeights(self.body,newEpisode)
            if i > len(self.episodes)-1:
                self.episodes.append(newEpisode)
            else:
                self.episodes[i]=newEpisode
        return self.episodes
        
    def unroll(self,adjusted_generated_x,mask,hints,optimizer,missRate,alpha,steps=2):
        '''
        A function that unroll discriminators changes on an array of the discriminator class instances.
        
        Args:
            adjusted_generated_x: A matrix of generated data where each row a record entry.
            mask: The mask of the data, 0,1 matrix of the same shape as x.
            hints: The hints matrix with 1, 0.5, 0 values. Same shape as x.
            discriminator: A discriminator model for the GAIN structure.
            optimizer: A tensorflow optimizer object.
            episodes: An array of discriminator class instances.
            p_miss: The missing rate of the mask.
            alpha: A regulation parameters.
            steps: The number of walks before making an episode record of the discriminator.
        
        Return:
            Total loss of the generator.
        '''
        if not hasattr(self, 'episodes'):
            raise Exception("Episodes not initiated.")
        
        cloneWeights(self.body,self.episodes[0])
        for i in range(0,self.episode_num-1):
            for j in range(0,steps):
                with tf.GradientTape(persistent=True) as tape:
                    discriminated_probs=self.call_episode(i,adjusted_generated_x,hints)
                    D_loss=tf.reduce_mean(mask * tf.math.log(discriminated_probs + 1e-8) + (1-missRate)/missRate*(1-mask) * tf.math.log(1. - discriminated_probs + 1e-8))
                D_loss_gradients = tape.gradient(D_loss,self.episodes[i].trainable_variables)
                optimizer.apply_gradients(zip(D_loss_gradients,self.episodes[i].trainable_variables))
            cloneWeights(self.episodes[i],self.episodes[i+1])
            
        with tf.GradientTape(persistent=True) as tape:
            discriminated_probs=self.call_episode(self.episode_num-1,adjusted_generated_x,hints)
            D_loss=tf.reduce_mean(mask * tf.math.log(discriminated_probs + 1e-8) + (1-missRate)/missRate*(1-mask) * tf.math.log(1. - discriminated_probs + 1e-8))
        D_loss_gradients = tape.gradient(D_loss,self.episodes[self.episode_num-1].trainable_variables)
        optimizer.apply_gradients(zip(D_loss_gradients,self.episodes[self.episode_num-1].trainable_variables))
    
    def train(self,adjusted_generated_x,mask,hints,missRate,optimizer):
        '''
        The training the discriminator.

        Args:
            adjusted_generated_x: Data generated by generator, with adjusted truth.
            hints: hints matrix for the discriminator. 1 = genuine, 0 = generated, 0.5 = unknown
            mask:a matrix with the same size as discriminated_probs. Entry value 1 indicate a genuine value, value 0 indicate missing(generated) value.
            missRate:the rate of missing data.
            optimizer: optimizer used for training the discriminator.
        Returns:
            discriminator_loss: loss value for discriminator
        '''
        with tf.GradientTape(persistent=True) as tape:
            discriminations=self.discriminate(adjusted_generated_x,hints)
            D_loss=discrimination_logLoss(discriminations,hints,mask,missRate)
        D_loss_gradients = tape.gradient(D_loss,self.body.trainable_variables)
        optimizer.apply_gradients(zip(D_loss_gradients, self.body.trainable_variables))
        return D_loss   
        
    def call_episode(self,episode_index,adjusted_generated_x,hints):
        return self.episodes[episode_index](tf.concat(axis = 1, values = [adjusted_generated_x,hints]))
        
    def discriminate(self,adjusted_generated_x,hints):
        """
        Discriminator model call for GAIN which is a residual block with a dense sequential body.

        Args: 
            adjusted_generated_x: Data generated by generator, with adjusted truth.
            hints: hints matrix for the discriminator. 1 = genuine, 0 = generated, 0.5 = unknown

        Returns:
            Output of the generated by the generator.
        """
        return self.body([adjusted_generated_x,hints])
    
    def discriminate_with_episodes(self,adjusted_generated_x,hints):
        """
        Discriminator model call for GAIN which is a residual block with a dense sequential body.

        Args: 
            adjusted_generated_x: Data generated by generator, with adjusted truth.
            hints: hints matrix for the discriminator. 1 = genuine, 0 = generated, 0.5 = unknown

        Returns:
            Output of the generated by the generator.
        """
        discriminations=[]
        for episode in self.episodes:
            discriminations.append(episode(tf.concat(axis = 1, values = [adjusted_generated_x,hints])))
        return discriminations
        