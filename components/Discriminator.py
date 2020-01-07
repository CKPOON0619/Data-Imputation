import tensorflow as tf
from components.NetworkComponents import compositLayers
from tensorflow.keras import Model

# Discriminator
class myDiscriminator(Model):
    def __init__(self,Dim,drop_rate):
        super(myDiscriminator, self).__init__()
        self.body = compositLayers([Dim*12,Dim*6,Dim*3,Dim*2,Dim*3,Dim*6,Dim*12,Dim],drop_rate)

    def call(self,x_hat,hints):
        return self.body(tf.concat(axis = 1, values = [x_hat,hints]))