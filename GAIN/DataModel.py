import tensorflow as tf
import numpy as np

#%% Helpers
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

#%% Data Model
class DataModel():
    """
    A data model class built to handle data transformation and input for the GAIN model.
    
    Args:
        data_path: A path to a comma delimited csv file with data header.
    """
    def __init__(self,data_path,rangeBoost=2):
        self.data_path=data_path
        self.rawData=tf.convert_to_tensor(np.genfromtxt(self.data_path, delimiter=",",skip_header=1),dtype=tf.float32)
        [self.sample_size,self.Dim]=tf.shape(self.rawData).numpy()
        currentMax=tf.math.reduce_max(self.rawData,axis=0)
        currentMin=tf.math.reduce_min(self.rawData,axis=0)
        
        # Increase data range TODO: review how this should work
        dataMax=currentMax+rangeBoost*(currentMax-currentMin) # max + 2*(max-min)
        dataMin=currentMin-rangeBoost*(currentMax-currentMin) # min - 2*(max-min)
        self.range=[dataMin,dataMax]
        self.normaliser=createNormaliser(self.range)
        self.denormaliser=createDenormaliser(self.range)

    # Setting up data pipeline
    def getPipeLine(self,train_rate,batch_ratio,repeat):
        """
        This function create and return a tensorflow data object with provided arguments.

        Args:
            train_rate: Ratio of the data to be used for training.
            batch_ratio: Ratio of the data to be used in each batch.
            repeat: Number of times the dataset got repeated in the dataset iterator.

        Returns:
            A tensorflow dataset object zipped with train and test data.
        """
        train_size=int(self.sample_size*train_rate)
        test_size=self.sample_size-train_size
        train_batch=int(train_size*batch_ratio)
        test_batch=int(test_size*batch_ratio)
        
        dataset=tf.data.Dataset.from_tensor_slices(self.normaliser(self.rawData)).shuffle(buffer_size=self.sample_size)
        dataset_train=dataset.take(train_size).batch(train_batch,drop_remainder=True).repeat(repeat)
        
        dataset_test=dataset.skip(train_size).batch(test_batch,drop_remainder=True).repeat(int(repeat*train_rate/(1-train_rate)))
        return dataset_train,dataset_test

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
        return self.denormaliser(generator(x,mask)*(1-mask)+mask*x)

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
        hints=mask+(1-mask)*0.5
        x_hat=self.normaliser(data)
        return discriminator(x_hat,hints) 