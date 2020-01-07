import tensorflow as tf
from components.NetworkComponents import compositLayers
from tensorflow.keras import Model

#Generator
class myGenerator(Model):
    def __init__(self,Dim,drop_rate):
        super(myGenerator, self).__init__()
        self.body = compositLayers([Dim*12,Dim*6,Dim*3,Dim*2,Dim*3,Dim*6,Dim*12,Dim],drop_rate)

    def call(self,x,mask):
        masked_x=mask*x
        mask_sample=(1-mask)*tf.random.uniform(tf.shape(x),minval=0,maxval=1,dtype=tf.float32)
        return self.body(tf.concat(axis = 1, values = [masked_x,mask_sample,mask]))+masked_x
