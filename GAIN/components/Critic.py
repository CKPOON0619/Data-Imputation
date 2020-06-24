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
class myCritic(Module):
    """
    A discriminator class for the GAIN model.

    Args:
        Dim: Dimension of data point.
        body: A kera Model that return a matrix of the same shape as data input. 
    """
    def __init__(self,body=False):
        super(myCritic, self).__init__()
        if(body):
            self.body = body
    
    def save(self,path):
        self.body.save(path)
    
    def load(self,path):
        self.body=tf.keras.models.load_model(path)
        
    def performance_log(self,writer,prefix,adjusted_generated_x,hints,mask,hintMask,missRate):
        '''
        To be filled.
        '''    

    def intiateUnrolling(self,dim,episode_num=5):
        self.episode_num=5
        if not hasattr(self, 'episodes'):
            self.episodes=[]
        for i in range(0,self.episode_num):
            newEpisode=tf.keras.models.clone_model(self.body)
            newEpisode.build(None,dim*2)
            cloneWeights(self.body,newEpisode)
            if i > len(self.episodes)-1:
                self.episodes.append(newEpisode)
            else:
                self.episodes[i]=newEpisode
        return self.episodes
        
    def unroll(self,data_batch,mask,hints,adjusted_generated_x,optimizer,missRate,alpha,leap=2):
        '''
        A function that unroll discriminators changes on an array of the discriminator class instances.
        
        Args:
            data_Batch: data input.
            mask: The mask of the data, 0,1 matrix of the same shape as x.
            hints: The hints matrix with 1, 0.5, 0 values. Same shape as x.
            adjusted_generated_x: A matrix of generated data where each row a record entry.
            discriminator: A discriminator model for the GAIN structure.
            optimizer: A tensorflow optimizer object.
            episodes: An array of discriminator class instances.
            p_miss: The missing rate of the mask.
            alpha: A regulation parameters.
            leap: The number of walks before making an episode record of the discriminator.
        
        Return:
            Total loss of the generator.
        '''
        cloneWeights(self.body,self.episodes[0])
        if not hasattr(self, 'episodes'):
            raise Exception("Episodes not initiated.")
        
        # for i in range(0,self.episode_num-1):
        #     for j in range(0,leap):
        #         with tf.GradientTape(persistent=True) as tape:
        #             discriminated_probs=self.call_episode(i,adjusted_generated_x,hints)
        #             D_loss=tf.reduce_mean(mask * tf.math.log(discriminated_probs + 1e-8) + (1-missRate)/missRate*(1-mask) * tf.math.log(1. - discriminated_probs + 1e-8))
        #         D_loss_gradients = tape.gradient(D_loss,self.episodes[i].trainable_variables)
        #         optimizer.apply_gradients(zip(D_loss_gradients,self.episodes[i].trainable_variables))
        #     cloneWeights(self.episodes[i],self.episodes[i+1])
            
        # with tf.GradientTape(persistent=True) as tape:
        #     discriminated_probs=self.episodes[self.episode_num-1](adjusted_generated_x,hints)
        #     D_loss=tf.reduce_mean(mask * tf.math.log(discriminated_probs + 1e-8) + (1-missRate)/missRate*(1-mask) * tf.math.log(1. - discriminated_probs + 1e-8))
        # D_loss_gradients = tape.gradient(D_loss,self.episodes[self.episode_num-1].trainable_variables)
        # optimizer.apply_gradients(zip(D_loss_gradients,self.episodes[self.episode_num-1].trainable_variables))        
            
    def calc_critic_diff(self,data_batch,adjusted_generated_data,hints):
        generated_critics=self.criticise(adjusted_generated_data,hints)
        genuine_critics=self.criticise(data_batch,hints)
        mean_critics_diff=-tf.reduce_mean(generated_critics)-tf.reduce_mean(genuine_critics)
        return mean_critics_diff
       
    def calc_critic_penalty(self,data_batch,adjusted_generated_data,hints):
        
        tau=tf.random.uniform([tf.shape(data_batch)[0],1], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None)
        interpolated_data=tau*adjusted_generated_data+(1-tau)*data_batch
        with tf.GradientTape(persistent=True) as tape:
            interpolated_critics=tf.reduce_sum(self.body(tf.concat(axis = 1, values = [interpolated_data,hints])),axis=1)
        interpolated_critic_gradients = tape.jacobian(interpolated_critics,self.body.trainable_variables)
        gradients_square_sum=tf.reduce_sum(tf.zeros(shape=tf.shape(interpolated_critic_gradients[0]),dtype=tf.float32),axis=[-1,1])
       
        for interpolated_critic_gradient in interpolated_critic_gradients:
            gradients_square_sum=gradients_square_sum+tf.reduce_sum(tf.square(interpolated_critic_gradient),axis=[-1,1])
        
        penalty=tf.reduce_mean(tf.square(tf.math.sqrt(gradients_square_sum)-1))
        return penalty

    def train(self,data_batch,adjusted_generated_data,mask,hints,alpha,optimizer):
        '''
        The training the discriminator.

        Args:
            data_batch: Raw data input.
            adjusted_generated_data: Data generated by generator, with adjusted truth.
            hints: hints matrix for the discriminator. 1 = genuine, 0 = generated, 0.5 = unknown
            mask:a matrix with the same size as discriminated_probs. Entry value 1 indicate a genuine value, value 0 indicate missing(generated) value.
            missRate:the rate of missing data.
            optimizer: optimizer used for training the discriminator.
        Returns:
            discriminator_loss: loss value for discriminator
        '''
        
        tau=tf.random.uniform([tf.shape(data_batch)[0],1], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None)
        interpolated_data=tau*adjusted_generated_data+(1-tau)*data_batch
        
        with tf.GradientTape() as tape:
            mean_critics_diff=self.calc_critic_diff(data_batch,adjusted_generated_data,hints)
            with tf.GradientTape() as tape2:           
                interpolated_critics=tf.reduce_sum(self.body(tf.concat(axis = 1, values = [interpolated_data,hints])),axis=1)
            interpolated_critic_gradients = tape2.jacobian(interpolated_critics,self.body.trainable_variables)
            gradients_square_sum=tf.reduce_sum(tf.zeros(shape=tf.shape(interpolated_critic_gradients[0]),dtype=tf.float32),axis=[-1,1])
            for interpolated_critic_gradient in interpolated_critic_gradients:
                gradients_square_sum=gradients_square_sum+tf.reduce_sum(tf.square(interpolated_critic_gradient),axis=[-1,1])
        
            penalty_regulation=tf.reduce_mean(tf.square(tf.math.sqrt(gradients_square_sum)-1))
            critic_loss=-mean_critics_diff-alpha*penalty_regulation
        critic_loss_gradient = tape.gradient(critic_loss,self.body.trainable_variables)
        optimizer.apply_gradients(zip(critic_loss_gradient, self.body.trainable_variables))
        return critic_loss
        
    def call_episode(self,episode_index,adjusted_generated_x,hints):
        return tf.reduce_sum(self.body(self.episodes[episode_index](tf.concat(axis = 1, values = [adjusted_generated_x,hints])),axis=1),axis=1)
        
    def criticise(self,data,hints):
        """
        Discriminator model call for GAIN which is a residual block with a dense sequential body.

        Args: 
            data: data.
            hints: hints matrix. 1 = genuine, 0 = generated, 0.5 = unknown

        Returns:
            Output of the generated by the generator.
        """
        return tf.reduce_sum(self.body(tf.concat(axis = 1, values = [data,hints])),axis=1)