import tensorflow as tf
import numpy as np

#%% Helpers
# Normalization (0 to 1)
def createNormaliser(dataMin,dataMax):
    '''
    dataMin:tensor of data min
    dataMax:tensor of data  max
    return: range normalised tensor
    '''
    return lambda rawDataTensor:(rawDataTensor-dataMin)/(dataMax-dataMin+1e-6)

# Return real value
def createDenormaliser(dataMin,dataMax):
    '''
    dataMin:tensor of data min
    dataMax:tensor of data  max
    return: de-normalised tensor
    '''
    return lambda dataTensor:(dataTensor)*(dataMax-dataMin+1e-6)+dataMin

#%% Data Model
class DataModel():
    def __init__(self,data_path):
        self.data_path=data_path

    # Setting up data pipeline
    def getPipeLine(self,train_rate,batch_ratio,repeat):
        self.rawData=tf.convert_to_tensor(np.genfromtxt(self.data_path, delimiter=",",skip_header=1),dtype=tf.float32)
        [self.sample_size,self.Dim]=tf.shape(self.rawData).numpy()
        self.train_size=int(self.sample_size*train_rate)
        self.batch_size=int(self.sample_size*batch_ratio)
        currentMax=tf.math.reduce_max(self.rawData,axis=0)
        currentMin=tf.math.reduce_min(self.rawData,axis=0)
        
        # Increase data range TODO: review how this should work
        dataMax=3*currentMax-2*currentMin
        dataMin=3*currentMin-2*currentMax
        self.range=[dataMin,dataMax]
        
        self.normaliser=createNormaliser(dataMax,dataMin)
        self.denormaliser=createDenormaliser(dataMax,dataMin)
        
        dataset=tf.data.Dataset.from_tensor_slices(self.normaliser(self.rawData)).shuffle(buffer_size=self.sample_size)
        dataset_train=dataset.take(self.train_size).batch(self.batch_size,drop_remainder=True).repeat(repeat)
        dataset_test=dataset.skip(self.train_size).batch(self.batch_size,drop_remainder=True).repeat(int(repeat*train_rate/(1-train_rate)))
        return tf.data.Dataset.zip((dataset_train, dataset_test))

    def predict(self,generator,data,mask):
        x=self.normaliser(data)
        return self.denormaliser(generator(x,mask)*(1-mask)+mask*x)

    def discriminate(self,discriminator,data,mask):
        hints=mask+(1-mask)*0.5
        x_hat=self.normaliser(data)
        return discriminator(x_hat,hints) 
