#%%
import tensorflow as tf
import numpy as np


#%% Helpers
def createTestMaskNHints(data):
    '''
    Produce a test mask for the distribution of the last column.

    Args:
        data: Input data.
    Returns:
        An array of test mask with 0 at the last entries of each row and 1 in the rest of the entries.
    '''
    shape=tf.shape(data)
    testMask=tf.tile(tf.concat([tf.ones([1,shape[1]-1],dtype=tf.float32),tf.zeros([1,1],dtype=tf.float32)],axis=1),[shape[0],1])
    return testMask

# Normalization (0 to 1)
def createNormaliser(dataRange):
    '''
    Args:
        dataRange:[min,max]
    Returns: 
        range normalised tensor
    '''
    dataMin,dataMax=dataRange
    return lambda rawDataTensor:(rawDataTensor-dataMin)/(dataMax-dataMin+1e-6)

# Return real value
def createDenormaliser(dataRange):
    '''
    Args:
        dataRange:[min,max]
    Returns: 
        de-normalised tensor
    '''
    dataMin,dataMax=dataRange
    return lambda dataTensor:(dataTensor)*(dataMax-dataMin+1e-6)+dataMin

def createMask(data,maskRatio):
    '''
    Args:
        data: tensor to be masked
        maskRatio: proportion of entries to be marked as 1
    Returns: 
        0,1 matrix of the same shape as data
    '''
    print(tf.shape(data))
    return tf.dtypes.cast((tf.random.uniform(tf.shape(data),minval=0,maxval=1)>(1.-maskRatio)),dtype=tf.float32)

def create_random_mask(data,fixed_mask,maskRatio):
    '''
    Args:
        data: tensor to be masked
        mask: mask showing entries to be revealed. 1: revealed, 0: reveal according to maskRatio Rate.
        maskRatio: proportion of entries to be marked as 1
    Returns: 
        0,1 matrix of the same shape as data
    '''
    [sample_size,dim]=tf.shape(data)
    full_random_mask=tf.dtypes.cast((tf.random.uniform(tf.shape(data),minval=0,maxval=1)>(1.-maskRatio)),dtype=tf.float32)
    return tf.tile([fixed_mask],[sample_size,1])+full_random_mask*(1.-fixed_mask)
    
   


def createHint(mask,hintRate):
    '''
    Args:
        mask: mask of the data, 0,1 matrix of the same shape as data.
        hintRate: the rate of reveal the truth
    return 
        hint_mask: mask for creating hints with 1,0 values. Same size as X.
        hints: hints matrix with 1,0.5,0 values. Same size as X.
    '''
    hint_mask=createMask(mask,1.-hintRate)
    hints=hint_mask*mask+(1.-hint_mask)*0.5
    return hint_mask,hints
    
def createHints(mask,hintRate):
    hint_mask,hints=createHint(mask,hintRate)
    return hint_mask,hints
    

    
#%% Data Model
class DataModel():
    """
    A data model class built to handle data transformation and input for the GAIN model.
    
    Args:
        data_path: A path to a comma delimited csv file with data header.
    """
    def __init__(self,data_path,rangeBoost=0):
        self.data_path=data_path
        self.rawData=tf.convert_to_tensor(np.genfromtxt(self.data_path, delimiter=",",skip_header=1),dtype=tf.float32)
        [self.sample_size,self.Dim]=tf.shape(self.rawData).numpy()
        currentMax=tf.math.reduce_max(self.rawData,axis=0)
        currentMin=tf.math.reduce_min(self.rawData,axis=0)
        
        # Increase data range TODO: review how this should work
        dataMax=currentMax+rangeBoost*(currentMax-currentMin) # max + boost*(max-min)
        dataMin=currentMin-rangeBoost*(currentMax-currentMin) # min - boost*(max-min)
        self.range=[dataMin,dataMax]
        self.normaliser=createNormaliser(self.range)
        self.denormaliser=createDenormaliser(self.range)

        

    # Setting up data pipeline
    def getPipeLine(self,train_rate,batch_ratio,repeat,fix_mask=None,p_miss=0.5,p_hints=0.5):
        """
        This function create and return a tensorflow data object with provided arguments.

        Args:
            train_rate: Ratio of the data to be used for training.
            batch_ratio: Ratio of the data to be used in each batch.
            repeat: Number of times the dataset got repeated in the dataset iterator.

        Returns:
            A tensorflow dataset object zipped with train and test data.
        """
        if fix_mask is not None:
            self.fix_masks=tf.tile([fix_mask],[self.sample_size,1])
        else:
            self.fix_masks=tf.zeros([self.sample_size,self.Dim],dtype=tf.float32)
            
        full_random_masks=tf.dtypes.cast((tf.random.uniform([self.sample_size,self.Dim],minval=0,maxval=1)>(1.-p_miss)),dtype=tf.float32)
        full_random_hints_masks=tf.dtypes.cast((tf.random.uniform([self.sample_size,self.Dim],minval=0,maxval=1)>(1.-p_hints)),dtype=tf.float32)
        
        train_size=int(self.sample_size*train_rate)
        train_batch=int(train_size*batch_ratio)
        self.train_batch_size=int(train_batch)

        masks=(1.-self.fix_masks)*full_random_masks+self.fix_masks
        hint_masks=(1.-self.fix_masks)*full_random_hints_masks+self.fix_masks
        hints=(1.-hint_masks)*0.5+hint_masks*masks
        
        fix_mask_set=tf.data.Dataset.from_tensor_slices(self.fix_masks)
        masks_set=tf.data.Dataset.from_tensor_slices(masks)
        hint_masks_set=tf.data.Dataset.from_tensor_slices(hint_masks)
        hints_set=tf.data.Dataset.from_tensor_slices(hints)
        
        masks_and_hints=tf.data.Dataset.zip((fix_mask_set,masks_set,hint_masks_set,hints_set))
        masks_and_hints=masks_and_hints.shuffle(buffer_size=self.sample_size)
        dataset=tf.data.Dataset.from_tensor_slices(self.normaliser(self.rawData)).shuffle(buffer_size=self.sample_size)
        
        dataset_with_masks_and_hints=tf.data.Dataset.zip((dataset,masks_and_hints))       
        dataset_train=dataset_with_masks_and_hints.take(train_size).batch(train_batch,drop_remainder=True).repeat(repeat)

        if train_rate!=1:
            test_size=self.sample_size-train_size
            test_batch=int(test_size*batch_ratio)
            dataset_test=dataset_with_masks_and_hints.skip(train_size).batch(test_batch,drop_remainder=True).repeat(int(repeat*train_rate/(1.-train_rate)))
            return dataset_train,dataset_test
        
        
        
        return dataset_train,None

    def predict(self,generator,data,mask):
        """
        The predict function that use the generator to create predicted missing values.

        Args:
            generator: A Generator class for the GAIN model.
            data: A piece of data with the same dim as the input data.
            mask: data mask, indicating missing values. genuine = 1, missing = 0.
        
        Returns:
            Missing filled data.
        """
        x=self.normaliser(data)
        return self.denormaliser(generator(x,mask)*(1.-mask)+mask*x)

    def discriminate(self,discriminator,data,mask):
        """
        The discriminate function that use the discriminator to predict generated values.

        Args:
            discriminator: A Discriminator class for the GAIN model.
            data: A piece of data with the same dim as the input data.
            mask: data mask, indicating missing values. genuine = 1, missing = 0.
        
        Returns:
            A probability matrix with entries indicating the predicted probability that 
            the data is genuine but not generated.
            genuine -> 1, generated -> 0.
        """
        hints=mask+(1.-mask)*0.5
        x_hat=self.normaliser(data)
        return discriminator(x_hat,hints) 